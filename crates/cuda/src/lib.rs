//! The CUDA driver, as far as kernite's kernels need it: memory on a GPU, PTX modules loaded
//! into it, their kernels launched and their memory read and written; and the floats of 16
//! bits, [`f16`](struct@f16) and [`bf16`], that Rust has not.
//!
//! The driver is `libcuda.so.1`, which comes with NVIDIA's driver rather than with the
//! toolkit. It is opened by [`Cuda::new`], not linked: a program that never asks for a GPU
//! builds and runs where there is none, and one that asks is told why it got none.
//!
//! ```no_run
//! let cuda = kernite_cuda::Cuda::new()?;
//! let xs = cuda.upload(&[1.0f32, 2.0, 3.0])?;
//! // A kernel is launched on `xs.ptr()` here.
//! cuda.sync()?;
//! println!("{:?}", xs.read()?);
//! # Ok::<(), kernite_cuda::Error>(())
//! ```
//!
//! A machine may have several GPUs. The GPUs are those the driver shows the program: all of the
//! machine's, unless `CUDA_VISIBLE_DEVICES` chooses which, and the driver numbers them from 0 in
//! its order. [`Cuda::count`] says how many there are, 0 where the driver shows none, and
//! [`Cuda::on`] takes one by its index; [`Cuda::new`] takes GPU 0. An index past the last is an
//! error that says how many there are: `cuDeviceGet: CUDA_ERROR_INVALID_DEVICE (101): invalid
//! device ordinal: the driver shows 1 GPU, and none is numbered 1`. Where the driver shows none,
//! [`Cuda::new`] and [`Cuda::on`] are the error of `cuInit`: `cuInit: CUDA_ERROR_NO_DEVICE
//! (100): no CUDA-capable device is detected`.
//!
//! A [`Cuda`] is for the thread that made it, and so are the buffers and modules it makes: they
//! hold on to it, so none is `Send`. A thread takes the GPU it uses with [`Cuda::on`], and threads
//! may take the same GPU. One thread may take several, and use them in turn: each call goes to
//! the GPU of the `Cuda` it is made on, or of the buffer or the module it is made on, whichever
//! GPU the call before it went to. The driver does the work of a call in the context that is
//! current on the thread. This crate makes a GPU's primary context current when a call goes to
//! another GPU than the call before it, with one `cuCtxSetCurrent`; a call to the same GPU asks
//! the driver for nothing more. A context that the program makes current itself, with the
//! driver's own calls, is not seen: the next call to the GPU of the last call goes to it.
//!
//! Kernels run on a GPU in the order they are launched, and a copy to or from a buffer waits
//! for those launched before it on the buffer's GPU; [`Cuda::sync`] waits for all of its GPU's,
//! and says whether one failed. A buffer is on one GPU, and a kernel is given the buffers of
//! the GPU it runs on: [`Buffer::copy_from`] copies a buffer to a buffer on another.
//!
//! A kernel whose blocks wait for each other is launched cooperatively,
//! [`Module::launch_cooperative`]: every block of the grid runs at once, so there are at most
//! [`Module::resident`] of them, and the driver gives the launch a barrier for the grid.

use std::cell::Cell;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;

mod half;
#[doc(hidden)]
pub mod named;

pub use half::{bf16, f16};

/// What every function of the driver returns: 0, or what went wrong.
type CuResult = c_int;

/// A context, a module, a function or a stream: what the driver hands out, opaque.
type Handle = *mut c_void;

// The dynamic loader, in the C library that every Rust program on Linux links.
unsafe extern "C" {
    fn dlopen(file: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

/// `dlopen` binds the library's own references to others as it opens it.
const RTLD_NOW: c_int = 2;

/// The driver's functions that kernite calls, each by the name `cuda.h` gives it under its own:
/// `cuMemAlloc` is `cuMemAlloc_v2`.
macro_rules! api {
    ($($field:ident = $symbol:literal($($arg:ty),*);)*) => {
        struct Api {
            $($field: unsafe extern "C" fn($($arg),*) -> CuResult,)*
        }

        impl Api {
            /// The functions of the driver that `lib` is a handle to.
            fn load(lib: *mut c_void) -> Result<Api, Error> {
                Ok(Api {
                    $($field: {
                        let f = symbol(lib, $symbol)?;
                        // SAFETY: `f` is not null, and `cuda.h` declares the function of this
                        // name with these parameters.
                        unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> CuResult>(f) }
                    },)*
                })
            }
        }
    };
}

api! {
    init = c"cuInit"(c_uint);
    device_count = c"cuDeviceGetCount"(*mut c_int);
    device_get = c"cuDeviceGet"(*mut c_int, c_int);
    device_name = c"cuDeviceGetName"(*mut c_char, c_int, c_int);
    attribute = c"cuDeviceGetAttribute"(*mut c_int, c_int, c_int);
    retain = c"cuDevicePrimaryCtxRetain"(*mut Handle, c_int);
    release = c"cuDevicePrimaryCtxRelease_v2"(c_int);
    set_current = c"cuCtxSetCurrent"(Handle);
    can_access_peer = c"cuDeviceCanAccessPeer"(*mut c_int, c_int, c_int);
    enable_peer_access = c"cuCtxEnablePeerAccess"(Handle, c_uint);
    synchronize = c"cuCtxSynchronize"();
    alloc = c"cuMemAlloc_v2"(*mut u64, usize);
    free = c"cuMemFree_v2"(u64);
    to_device = c"cuMemcpyHtoD_v2"(u64, *const c_void, usize);
    to_host = c"cuMemcpyDtoH_v2"(*mut c_void, u64, usize);
    copy_peer = c"cuMemcpyPeer"(u64, Handle, u64, Handle, usize);
    memset = c"cuMemsetD8_v2"(u64, u8, usize);
    load_module = c"cuModuleLoadDataEx"(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void);
    unload_module = c"cuModuleUnload"(Handle);
    function = c"cuModuleGetFunction"(*mut Handle, Handle, *const c_char);
    global = c"cuModuleGetGlobal_v2"(*mut u64, *mut usize, Handle, *const c_char);
    function_attribute = c"cuFuncGetAttribute"(*mut c_int, c_int, Handle);
    set_attribute = c"cuFuncSetAttribute"(Handle, c_int, c_int);
    launch = c"cuLaunchKernel"(
        Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void
    );
    launch_cooperative = c"cuLaunchCooperativeKernel"(
        Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void
    );
    occupancy = c"cuOccupancyMaxActiveBlocksPerMultiprocessor"(*mut c_int, Handle, c_int, usize);
    error_name = c"cuGetErrorName"(CuResult, *mut *const c_char);
    error_string = c"cuGetErrorString"(CuResult, *mut *const c_char);
}

impl Api {
    /// What the function `call` returned, as a `Result`: a call that succeeds costs a test.
    #[inline]
    fn check(&self, call: &'static str, result: CuResult) -> Result<(), Error> {
        if result == 0 { Ok(()) } else { Err(self.failed(call, result)) }
    }

    /// What the function `call` said by returning `result`, which is not 0, by name, number and
    /// meaning.
    #[cold]
    fn failed(&self, call: &'static str, result: CuResult) -> Error {
        let (mut name, mut meaning) = (ptr::null(), ptr::null());
        // SAFETY: each sets its pointer to a string that lasts as long as the driver, or
        // leaves it null for a result it does not know.
        let (name, meaning) = unsafe {
            (self.error_name)(result, &mut name);
            (self.error_string)(result, &mut meaning);
            (string(name), string(meaning))
        };
        Error::new(call, format!("{name} ({result}): {meaning}"))
    }

    /// Starts the driver, as `cuInit` does: what it returned, which is 0 once the driver is
    /// started.
    fn start(&self) -> CuResult {
        // SAFETY: it takes flags, of which there are none yet: 0.
        unsafe { (self.init)(0) }
    }

    /// How many GPUs the driver shows the program, once it is started.
    fn gpus(&self) -> Result<usize, Error> {
        let mut count = 0;
        // SAFETY: it writes the count to `count`.
        self.check("cuDeviceGetCount", unsafe { (self.device_count)(&mut count) })?;
        Ok(count as usize)
    }
}

/// The driver's functions, found but not started; or why there is no driver to have.
fn find() -> Result<Api, Error> {
    // SAFETY: opening the driver runs its initialisers, which ask nothing of this
    // program. It is never closed: like a library linked, it lasts as long as the program.
    let lib = unsafe { dlopen(c"libcuda.so.1".as_ptr(), RTLD_NOW) };
    if lib.is_null() {
        return Err(Error::new("dlopen", format!("no CUDA driver: {}", dl_error())));
    }
    Api::load(lib)
}

/// The driver, opened and started: its functions, or why there is no driver to have.
fn open() -> Result<Api, Error> {
    let api = find()?;
    api.check("cuInit", api.start())?;
    Ok(api)
}

/// The function `name` of the library `lib`.
fn symbol(lib: *mut c_void, name: &CStr) -> Result<*mut c_void, Error> {
    // SAFETY: `lib` is a handle `dlopen` gave, and `name` is a C string.
    let f = unsafe { dlsym(lib, name.as_ptr()) };
    if f.is_null() {
        return Err(Error::new("dlsym", format!("the driver has no `{}`: {}", name.to_string_lossy(), dl_error())));
    }
    Ok(f)
}

/// What the loader said went wrong, last.
fn dl_error() -> String {
    // SAFETY: `dlerror` returns null or a C string that lasts until it is called again on
    // this thread, which `string` copies first.
    unsafe { string(dlerror()) }
}

/// The C string at `p`, or `?` if there is none.
///
/// # Safety
///
/// `p` is null, or points to a C string.
unsafe fn string(p: *const c_char) -> String {
    if p.is_null() {
        return "?".into();
    }
    // SAFETY: the caller's.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// What went wrong in a call to the driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// The function that failed, by its name in C: `cuInit`, or `dlopen` if there is no driver.
    pub call: &'static str,
    /// What it said: the `CUresult` it returned, by name, number and meaning, or why there is
    /// no driver.
    pub message: String,
    /// What the driver's compiler wrote when it failed on PTX, which says where in it.
    pub log: String,
}

impl Error {
    fn new(call: &'static str, message: String) -> Error {
        Error { call, message, log: String::new() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.call, self.message)?;
        if !self.log.is_empty() {
            write!(f, "\n{}", self.log.trim_end())?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

thread_local! {
    /// The context this crate made current on this thread last, or null before it made any:
    /// what a call tests, to know whether its own GPU's is current. The last release of a GPU's
    /// primary context destroys it, and the driver leaves it current on the thread; the next
    /// retain makes it again with the handle it had, so what is remembered is right still, as
    /// `tests/again.rs` shows.
    static CURRENT: Cell<Handle> = const { Cell::new(ptr::null_mut()) };
}

/// Makes `context` current on the thread by `set`, unless `current`, which holds the context
/// made current last, says it is. `current` holds `context` only if `set` worked.
#[inline]
fn enter(current: &Cell<Handle>, context: Handle, set: impl FnOnce() -> Result<(), Error>) -> Result<(), Error> {
    if current.get() != context {
        set()?;
        current.set(context);
    }
    Ok(())
}

/// The driver's functions, and the GPU whose primary context they work in, which is let go
/// when the last thing holding it is dropped.
struct Driver {
    api: Api,
    device: c_int,
    /// The GPU's primary context, which the driver keeps while this holds it.
    context: Handle,
}

impl Driver {
    /// The driver's functions, once this GPU's context is current on the thread: the ones that
    /// allocate, copy, wait, load or launch work in the current context, and every call of
    /// them gets its functions here. The thread-local test is all it costs a call to the GPU
    /// the thread's last call went to; one to another GPU asks the driver once to switch.
    #[inline]
    fn current(&self) -> Result<&Api, Error> {
        CURRENT.with(|current| enter(current, self.context, || self.set_current()))?;
        Ok(&self.api)
    }

    /// Makes this GPU's context current on the thread, whichever is now.
    fn set_current(&self) -> Result<(), Error> {
        // SAFETY: the context was retained, and is held until `self` is dropped.
        self.api.check("cuCtxSetCurrent", unsafe { (self.api.set_current)(self.context) })
    }

    /// Copies `data` to the GPU at `ptr`, after the kernels launched before have run.
    fn write<T: Plain>(&self, ptr: u64, data: &[T]) -> Result<(), Error> {
        let api = self.current()?;
        // SAFETY: it reads as many bytes from `data` as it has.
        let copied = unsafe { (api.to_device)(ptr, data.as_ptr().cast(), size_of_val(data)) };
        api.check("cuMemcpyHtoD", copied)
    }

    /// Copies what the GPU holds at `ptr` to `out`, after the kernels launched before have
    /// run.
    fn read_into<T: Plain>(&self, ptr: u64, out: &mut [T]) -> Result<(), Error> {
        let api = self.current()?;
        // SAFETY: it writes as many bytes to `out` as it has, and any bits are a `T`, as
        // `Plain` promises.
        let copied = unsafe { (api.to_host)(out.as_mut_ptr().cast(), ptr, size_of_val(out)) };
        api.check("cuMemcpyDtoH", copied)
    }

    /// Sets the `bytes` bytes at `ptr` on the GPU to zero, after the kernels launched before
    /// have run.
    fn zero(&self, ptr: u64, bytes: usize) -> Result<(), Error> {
        let api = self.current()?;
        // SAFETY: it writes on the GPU alone.
        api.check("cuMemsetD8", unsafe { (api.memset)(ptr, 0, bytes) })
    }

    /// Copies the `bytes` bytes at `from_ptr` on the GPU of `from` to `ptr` on this one, after
    /// the kernels launched before have run on either, and before those launched after run on
    /// either. The copy is made in this GPU's context, which reaches the other's memory by a
    /// link between the two GPUs where they have one, and through the host where they have not.
    fn copy_from(&self, ptr: u64, from: &Driver, from_ptr: u64, bytes: usize) -> Result<(), Error> {
        let api = self.current()?;
        if self.device != from.device {
            self.reach(api, from)?;
        }
        // SAFETY: the caller's addresses are on the GPUs of the two contexts, with `bytes`
        // bytes from each.
        let copied = unsafe { (api.copy_peer)(ptr, self.context, from_ptr, from.context, bytes) };
        api.check("cuMemcpyPeer", copied)
    }

    /// Lets this GPU's context, which is current, reach the memory of `from`'s GPU directly,
    /// if the two have a link for that. The driver keeps it until a context is destroyed.
    fn reach(&self, api: &Api, from: &Driver) -> Result<(), Error> {
        let mut linked = 0;
        // SAFETY: it writes whether this GPU may reach the other's memory to `linked`.
        let asked = unsafe { (api.can_access_peer)(&mut linked, self.device, from.device) };
        api.check("cuDeviceCanAccessPeer", asked)?;
        if linked == 0 {
            return Ok(());
        }
        // SAFETY: `from.context` is held by `from`, which the caller has.
        let enabled = unsafe { (api.enable_peer_access)(from.context, 0) };
        match enabled {
            // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, by an earlier copy; and
            // CUDA_ERROR_TOO_MANY_PEERS, where the driver has no link left to give and stages
            // the copy through the host, as it does for GPUs that have none.
            704 | 711 => Ok(()),
            _ => api.check("cuCtxEnablePeerAccess", enabled),
        }
    }

    /// What the GPU says of itself as the `CUdevice_attribute` `attribute`.
    fn attribute(&self, attribute: c_int) -> Result<u32, Error> {
        let mut v = 0;
        // SAFETY: it writes the attribute's value to `v`.
        let asked = unsafe { (self.api.attribute)(&mut v, attribute, self.device) };
        self.api.check("cuDeviceGetAttribute", asked).map(|()| v as u32)
    }

    /// What the driver says of kernel `f` as the `CUfunction_attribute` `attribute`.
    fn function_attribute(&self, f: Handle, attribute: c_int) -> Result<u32, Error> {
        let api = self.current()?;
        let mut v = 0;
        // SAFETY: `f` is a kernel of a module that is loaded; it writes the attribute's value
        // to `v`.
        let asked = unsafe { (api.function_attribute)(&mut v, attribute, f) };
        api.check("cuFuncGetAttribute", asked).map(|()| v as u32)
    }

    /// `e`, which the driver gave by returning `result` for a launch over `grid` of kernel `k`,
    /// with the first size of the launch that is past what the GPU takes, or the kernel's
    /// `.maxntid` or `.reqntid` after the GPU's, and what it takes. The driver says
    /// `CUDA_ERROR_INVALID_VALUE` of each and no more; any other error says what it is, and is
    /// left as it is. It asks the GPU only now, after the launch has failed.
    fn refused(&self, mut e: Error, result: CuResult, grid: &Grid, k: &Kernel) -> Error {
        // CUDA_ERROR_INVALID_VALUE.
        if result != 1 {
            return e;
        }
        // Wide enough for the product of three sizes of 32 bits.
        let [bx, by, bz] = grid.blocks.map(u128::from);
        let [tx, ty, tz] = grid.threads.map(u128::from);
        let shared = format!("bytes of shared memory, {} of the kernel's and {} of the launch's", k.own, grid.shared);
        // Each size, the least the GPU takes of it, and the `CUdevice_attribute` of the most:
        // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, _Y and _Z; MAX_BLOCK_DIM_X, _Y and _Z;
        // MAX_THREADS_PER_BLOCK; and MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
        let sizes = [
            (5, 1, bx, "the grid has", "blocks along x"),
            (6, 1, by, "the grid has", "blocks along y"),
            (7, 1, bz, "the grid has", "blocks along z"),
            (2, 1, tx, "each block has", "threads along x"),
            (3, 1, ty, "each block has", "threads along y"),
            (4, 1, tz, "each block has", "threads along z"),
            (1, 1, tx * ty * tz, "each block has", "threads"),
            (97, 0, u128::from(k.own) + u128::from(grid.shared), "each block has", shared.as_str()),
        ];
        // Each size with who takes how many of it, from the least to the most: the GPU, and then the
        // kernel, if it says a block.
        let mut rows = vec![];
        for (attribute, least, size, has, what) in sizes {
            if let Ok(most) = self.attribute(attribute) {
                rows.push((size, has, what, "the GPU".to_string(), least, u128::from(most)));
            }
        }
        if let Some(Block { directive, sizes }) = &k.block {
            let written: Vec<String> = sizes.iter().map(u128::to_string).collect();
            let who = format!("the kernel's `.{directive} {}`", written.join(", "));
            if *directive == "maxntid" {
                // The driver weighs the threads of a block in all, not along each axis.
                rows.push((tx * ty * tz, "each block has", "threads", who, 1, sizes.iter().product()));
            } else if grid.threads != [1, 1, 1] {
                // The driver launches a block of one thread, whatever `.reqntid` says.
                let along = [(tx, "threads along x"), (ty, "threads along y"), (tz, "threads along z")];
                for (i, (size, what)) in along.into_iter().enumerate() {
                    let exactly = sizes.get(i).copied().unwrap_or(1);
                    rows.push((size, "each block has", what, who.clone(), exactly, exactly));
                }
            }
        }
        for (size, has, what, who, least, most) in rows {
            // The limit of the size, how the sentence says it, and whether the size is past it: a
            // `.reqntid` takes one size, so its least is its most.
            let (limit, tail) = match size {
                _ if least == most && size != least => (least, "exactly"),
                _ if size < least => (least, "at least"),
                _ if size > most => (most, "at most"),
                _ => continue,
            };
            e.message += &format!(": {has} {size} {what}, and {who} takes {limit} {tail}");
            return e;
        }
        e
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        // SAFETY: `Cuda::on` retained the primary context, once for each `Driver`.
        unsafe { (self.api.release)(self.device) };
    }
}

/// One GPU, through its primary context, which a call makes current on its thread if the call
/// before it went to another GPU.
pub struct Cuda {
    driver: Rc<Driver>,
    name: String,
    sm: u32,
    max_shared: u32,
    multiprocessors: u32,
    cooperative: bool,
}

impl Cuda {
    /// How many GPUs the driver shows the program, through the driver opened now: all of the
    /// machine's, or those `CUDA_VISIBLE_DEVICES` chooses. It is 0 where `cuInit` says
    /// `CUDA_ERROR_NO_DEVICE`, as it does where the machine has no GPU, or the variable chooses
    /// none. It is an error where there is no driver, or `cuInit` fails any other way, as
    /// [`Cuda::new`] says why it has none to take.
    pub fn count() -> Result<usize, Error> {
        let api = find()?;
        match api.start() {
            // CUDA_ERROR_NO_DEVICE.
            100 => Ok(0),
            started => api.check("cuInit", started).and_then(|()| api.gpus()),
        }
    }

    /// GPU 0, through the driver opened now; or why there is none to have. It is `Cuda::on(0)`.
    pub fn new() -> Result<Cuda, Error> {
        Cuda::on(0)
    }

    /// GPU `index` of those the driver shows the program, counted from 0 in its order, through
    /// the driver opened now; or why there is none to have. An `index` past the last GPU is an
    /// error that says how many there are: `cuDeviceGet: CUDA_ERROR_INVALID_DEVICE (101):
    /// invalid device ordinal: the driver shows 1 GPU, and none is numbered 1`. Where the driver
    /// shows none, it is the error of `cuInit`: `cuInit: CUDA_ERROR_NO_DEVICE (100): no
    /// CUDA-capable device is detected`. The GPU's context is made current on this thread.
    pub fn on(index: usize) -> Result<Cuda, Error> {
        let api = open()?;
        let (mut device, mut context) = (0, ptr::null_mut());
        // An index past what an `int` holds is past the GPUs, and the driver says so of the
        // greatest.
        let ordinal = c_int::try_from(index).unwrap_or(c_int::MAX);
        // SAFETY: it writes the device to `device`.
        let got = unsafe { (api.device_get)(&mut device, ordinal) };
        api.check("cuDeviceGet", got).map_err(|mut e| {
            // CUDA_ERROR_INVALID_DEVICE, which says no more.
            if got == 101
                && let Ok(gpus) = api.gpus()
            {
                let gpus = if gpus == 1 { "1 GPU".to_string() } else { format!("{gpus} GPUs") };
                e.message += &format!(": the driver shows {gpus}, and none is numbered {index}");
            }
            e
        })?;
        // SAFETY: it writes the context to `context`.
        api.check("cuDevicePrimaryCtxRetain", unsafe { (api.retain)(&mut context, device) })?;
        let driver = Rc::new(Driver { api, device, context });
        driver.current()?;
        let (api, mut name) = (&driver.api, [0u8; 256]);
        // SAFETY: the name is written as a C string, cut to fit.
        api.check("cuDeviceGetName", unsafe {
            (api.device_name)(name.as_mut_ptr().cast(), name.len() as c_int, device)
        })?;
        let name = CStr::from_bytes_until_nul(&name).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR and _MINOR,
        // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, _MULTIPROCESSOR_COUNT and
        // _COOPERATIVE_LAUNCH.
        let [major, minor, shared, multiprocessors, cooperative] = [75, 76, 97, 16, 95].map(|a| driver.attribute(a));
        Ok(Cuda {
            sm: major? * 10 + minor?,
            max_shared: shared?,
            multiprocessors: multiprocessors?,
            cooperative: cooperative? != 0,
            driver,
            name,
        })
    }

    /// The GPU's name, as its driver gives it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The GPU's compute capability, as PTX names it after `sm_`: 61 for a GTX 10, 86 for an
    /// RTX 30.
    pub fn sm(&self) -> u32 {
        self.sm
    }

    /// The most bytes of shared memory a block may have, what its kernel declares and what
    /// the launch gives together: 48 KB on a GTX 10, 99 KB on an RTX 30.
    pub fn max_shared(&self) -> u32 {
        self.max_shared
    }

    /// How many multiprocessors, SMs, the GPU has: each runs as many blocks at once as its
    /// registers, its shared memory and its threads hold.
    pub fn multiprocessors(&self) -> u32 {
        self.multiprocessors
    }

    /// Whether the GPU launches a kernel cooperatively, every block of the grid at once, as
    /// one whose blocks wait for each other needs: a GPU from Pascal's, `sm_60`, on does.
    pub fn cooperative(&self) -> bool {
        self.cooperative
    }

    /// Waits for every kernel launched on the GPU to finish, and says whether one failed.
    pub fn sync(&self) -> Result<(), Error> {
        let api = self.driver.current()?;
        // SAFETY: it takes nothing.
        api.check("cuCtxSynchronize", unsafe { (api.synchronize)() })
    }

    /// `len` elements on the GPU, all zero.
    pub fn alloc<T: Plain>(&self, len: usize) -> Result<Buffer<T>, Error> {
        let mut b = self.reserve(len)?;
        b.zero()?;
        Ok(b)
    }

    /// A copy of `data` on the GPU.
    pub fn upload<T: Plain>(&self, data: &[T]) -> Result<Buffer<T>, Error> {
        let mut b = self.reserve(data.len())?;
        b.write(data)?;
        Ok(b)
    }

    /// `len` elements on the GPU, as they were left.
    fn reserve<T: Plain>(&self, len: usize) -> Result<Buffer<T>, Error> {
        let bytes = len.checked_mul(size_of::<T>()).expect("a buffer of more bytes than there are addresses");
        let (api, mut ptr) = (self.driver.current()?, 0);
        // SAFETY: it writes the address to `ptr`. It refuses no bytes, so it is asked for one.
        api.check("cuMemAlloc", unsafe { (api.alloc)(&mut ptr, bytes.max(1)) })?;
        Ok(Buffer { driver: self.driver.clone(), ptr, len, of: PhantomData })
    }
}

/// `len` elements of `T` on one GPU, the one that made it, freed when it is dropped.
pub struct Buffer<T: Plain> {
    driver: Rc<Driver>,
    ptr: u64,
    len: usize,
    of: PhantomData<T>,
}

impl<T: Plain> Buffer<T> {
    /// Its address, for a kernel to be given.
    pub fn ptr(&self) -> Ptr<T> {
        self.at(0)
    }

    /// The address of its element `i`, for a kernel to be given the part from there on.
    /// Panics if `i` is past its end.
    pub fn at(&self, i: usize) -> Ptr<T> {
        assert!(i <= self.len, "element {i} of a buffer of {}", self.len);
        Ptr { addr: self.ptr + (i * size_of::<T>()) as u64, of: PhantomData }
    }

    /// How many elements it has.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies `data` to it, after the kernels launched before have run. Panics if `data` has
    /// not as many elements.
    pub fn write(&mut self, data: &[T]) -> Result<(), Error> {
        assert_eq!(data.len(), self.len, "a buffer of {} elements written with {}", self.len, data.len());
        self.driver.write(self.ptr, data)
    }

    /// What it holds, after the kernels launched before have run.
    pub fn read(&self) -> Result<Vec<T>, Error> {
        let mut out = vec![T::default(); self.len];
        self.read_into(&mut out)?;
        Ok(out)
    }

    /// Copies what it holds to `out`, after the kernels launched before have run. Panics if
    /// `out` has not as many elements.
    pub fn read_into(&self, out: &mut [T]) -> Result<(), Error> {
        assert_eq!(out.len(), self.len, "a buffer of {} elements read into {}", self.len, out.len());
        self.driver.read_into(self.ptr, out)
    }

    /// Sets every element to zero, after the kernels launched before have run.
    pub fn zero(&mut self) -> Result<(), Error> {
        self.driver.zero(self.ptr, self.len * size_of::<T>())
    }

    /// Copies what `from` holds to it, after the kernels launched before have run on the GPU
    /// of either, and before those launched after run on either. `from` may be on another
    /// GPU than it: where the two have a link between them, the driver's peer access, the copy
    /// goes over the link, and where they have none it goes through the host. Panics if `from`
    /// has not as many elements.
    pub fn copy_from(&mut self, from: &Buffer<T>) -> Result<(), Error> {
        assert_eq!(from.len, self.len, "a buffer of {} elements copied from one of {}", self.len, from.len);
        self.driver.copy_from(self.ptr, &from.driver, from.ptr, self.len * size_of::<T>())
    }
}

impl<T: Plain> Drop for Buffer<T> {
    fn drop(&mut self) {
        // A buffer whose GPU's context cannot be made current is left, and not freed in another's.
        if let Ok(api) = self.driver.current() {
            // SAFETY: `cuMemAlloc` gave the address, and it is freed once, here.
            unsafe { (api.free)(self.ptr) };
        }
    }
}

impl<T: Plain> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Buffer({} elements at {:#x})", self.len, self.ptr)
    }
}

/// An address on the GPU, of elements of `T`: what a kernel's `*const T` or `*mut T` is
/// given. A [`Buffer`] gives it, and it points at something only while the buffer lasts. A kernel's
/// pointer is to a type with a size, but `T` may have none: a kernel that is wrong in this is one
/// error, at its type, and not a second, here.
#[repr(transparent)]
pub struct Ptr<T: ?Sized> {
    addr: u64,
    of: PhantomData<*const T>,
}

impl<T: ?Sized> Ptr<T> {
    /// The address, as the GPU reckons it.
    pub fn addr(self) -> u64 {
        self.addr
    }
}

impl<T: ?Sized> Clone for Ptr<T> {
    fn clone(&self) -> Ptr<T> {
        *self
    }
}

impl<T: ?Sized> Copy for Ptr<T> {}

impl<T: ?Sized> fmt::Debug for Ptr<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Ptr({:#x})", self.addr)
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A type a [`Buffer`] holds: one whose every pattern of bits is a value of it, so that what
/// a kernel wrote can be read back as one.
pub trait Plain: sealed::Sealed + Copy + Default {}

/// A type a kernel takes as a parameter, passed as the bits its `.param` holds: a pointer as
/// a [`Ptr`], an integer or a float as itself.
pub trait Param: sealed::Sealed + Copy {}

macro_rules! plain {
    ($($t:ty),*) => { $(impl sealed::Sealed for $t {} impl Plain for $t {} impl Param for $t {})* };
}

plain!(u8, u16, u32, u64, i8, i16, i32, i64, f16, bf16, f32, f64);

impl sealed::Sealed for usize {}
impl sealed::Sealed for isize {}
impl<T: ?Sized> sealed::Sealed for Ptr<T> {}

impl Param for usize {}
impl Param for isize {}
impl<T: ?Sized> Param for Ptr<T> {}

/// `x` as [`Module::launch`] takes a parameter: the address of its value, which the driver
/// reads as it launches.
pub fn arg<T: Param>(x: &T) -> *const c_void {
    (x as *const T).cast()
}

/// How many blocks a kernel is launched in, how many threads each block has, what it reads
/// as `nctaid` and `ntid`; and the shared memory the launch gives each block.
///
/// From Kepler's `sm_30` on, a GPU takes 1 to 2^31 - 1 blocks along x, 1 to 65535 along y and 1
/// to 65535 along z, and 1 to 1024 threads in a block, of them 1 to 1024 along x, 1 to 1024
/// along y and 1 to 64 along z. A launch over a grid the GPU does not take, one with more of a
/// size than it takes or with none of it, is an error that says which size is past which limit:
/// `the grid has 65536 blocks along y, and the GPU takes 65535 at most`, or `the grid has 0
/// blocks along x, and the GPU takes 1 at least`. It says so after `CUDA_ERROR_INVALID_VALUE`
/// alone, which is what the driver says of such a grid; any other error is said as it is. The
/// driver starts at most 65535 blocks along x of a module written for an older GPU, `sm_20` or
/// `sm_21`, whatever the GPU it runs on, and says only `CUDA_ERROR_INVALID_VALUE` of a launch
/// past that.
///
/// A block has shared memory of the kernel's own, what it declares, and of the launch's, beside
/// it: together at most [`Cuda::max_shared`] bytes, 99 KB on an RTX 30. A launch of more is an
/// error that says how much the block has: `each block has 102400 bytes of shared memory, 20480
/// of the kernel's and 81920 of the launch's, and the GPU takes 101376 at most`.
///
/// A kernel's `.maxntid` or `.reqntid` limits its blocks as well, and a block past it is said in
/// the same words, after the GPU's limits: ``each block has 512 threads, and the kernel's
/// `.maxntid 256` takes 256 at most``.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Blocks along x, y and z.
    pub blocks: [u32; 3],
    /// Threads in each block along x, y and z.
    pub threads: [u32; 3],
    /// Bytes of shared memory for each block beyond what the kernel declares: what its
    /// `shared<[T]>` holds. With the kernel's own, at most [`Cuda::max_shared`].
    pub shared: u32,
}

impl Grid {
    /// `blocks` blocks of `threads` threads, along x, and no shared memory but what the
    /// kernel declares.
    pub fn new(blocks: u32, threads: u32) -> Grid {
        Grid { blocks: [blocks, 1, 1], threads: [threads, 1, 1], shared: 0 }
    }

    /// The grid with `bytes` of shared memory for each block, for the kernel's `shared<[T]>`.
    pub fn with_shared(self, bytes: u32) -> Grid {
        Grid { shared: bytes, ..self }
    }
}

/// A kernel's `.maxntid` or `.reqntid`: the threads a block has along x, then y and z, as the
/// directive gives them, at most for `.maxntid`, and exactly for `.reqntid`. The driver holds a
/// launch to it, and says no more than `CUDA_ERROR_INVALID_VALUE` of one it refuses.
struct Block {
    /// `maxntid` or `reqntid`.
    directive: &'static str,
    sizes: Vec<u128>,
}

impl Block {
    /// The directive of the `.entry` called `name` in `ptx`, which goes between its parameters
    /// and its body, if it has one: nothing in the driver's API gives it.
    fn of(ptx: &str, name: &str) -> Option<Block> {
        let (_, rest) = ptx.split_once(&format!(".entry {name}("))?;
        let between = rest.split_once(')')?.1.split_once('{')?.0;
        ["maxntid", "reqntid"].into_iter().find_map(|directive| {
            // Its sizes go up to the next directive.
            let sizes = between.split_once(&format!(".{directive}"))?.1.split('.').next()?;
            Some(Block { directive, sizes: sizes.split(',').map(|n| n.trim().parse().ok()).collect::<Option<_>>()? })
        })
    }
}

/// A kernel of a module, its shared memory and its blocks: the shared memory it declares, and
/// what a launch may give it beside that; and the threads of a block its PTX says it takes.
struct Kernel {
    function: Handle,
    /// Bytes of shared memory it declares itself, which its PTX fixes:
    /// CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES.
    own: u32,
    /// The most bytes of shared memory a launch may give it beside its own, as the driver has it:
    /// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES. It starts at what the driver gives unasked,
    /// 48 KB less its own, and a launch that needs more raises it.
    room: Cell<u32>,
    /// Its `.maxntid` or `.reqntid`, if it says one.
    block: Option<Block>,
}

/// A PTX module, compiled by the driver for its GPU, and the kernels and the memory of it that
/// were asked for.
pub struct Module {
    driver: Rc<Driver>,
    module: Handle,
    kernels: Vec<Kernel>,
    /// Where each of its variables asked for is on the GPU, and how many bytes it has.
    globals: Vec<(u64, usize)>,
    /// The GPU's, as [`Cuda`] has them.
    multiprocessors: u32,
    cooperative: bool,
}

impl Module {
    /// `ptx` compiled for `cuda`'s GPU, with its kernels `names` and its memory `globals`, its
    /// variables in global or constant memory, each by its name, in that order. If the driver
    /// will not compile it, the error's `log` is what its compiler said.
    pub fn load(cuda: &Cuda, ptx: &str, names: &[&str], globals: &[&str]) -> Result<Module, Error> {
        let api = cuda.driver.current()?;
        let image =
            CString::new(ptx).map_err(|_| Error::new("cuModuleLoadDataEx", "the PTX has a NUL in it".into()))?;
        let mut log = vec![0u8; 1 << 14];
        // CU_JIT_ERROR_LOG_BUFFER, and CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES.
        let mut options: [c_int; 2] = [5, 6];
        let mut values = [log.as_mut_ptr().cast(), ptr::without_provenance_mut(log.len())];
        let mut module = ptr::null_mut();
        // SAFETY: `image` is a C string, and each option has a value of the kind it takes: the
        // log a buffer, and its size the buffer's.
        let loaded = unsafe {
            (api.load_module)(&mut module, image.as_ptr().cast(), 2, options.as_mut_ptr(), values.as_mut_ptr())
        };
        if let Err(mut e) = api.check("cuModuleLoadDataEx", loaded) {
            e.log = CStr::from_bytes_until_nul(&log).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            return Err(e);
        }
        let mut m = Module {
            driver: cuda.driver.clone(),
            module,
            kernels: vec![],
            globals: vec![],
            multiprocessors: cuda.multiprocessors,
            cooperative: cuda.cooperative,
        };
        for name in names {
            let c =
                CString::new(*name).map_err(|_| Error::new("cuModuleGetFunction", format!("{name:?} has a NUL")))?;
            let mut f = ptr::null_mut();
            // SAFETY: the module is loaded, and the name is a C string.
            let found = unsafe { (api.function)(&mut f, module, c.as_ptr()) };
            api.check("cuModuleGetFunction", found).map_err(|mut e| {
                e.message += &format!(", for `{name}`");
                e
            })?;
            // CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, and _MAX_DYNAMIC_SHARED_SIZE_BYTES.
            let [own, room] = [1, 8].map(|attribute| m.driver.function_attribute(f, attribute));
            m.kernels.push(Kernel { function: f, own: own?, room: Cell::new(room?), block: Block::of(ptx, name) });
        }
        for name in globals {
            let c = CString::new(*name).map_err(|_| Error::new("cuModuleGetGlobal", format!("{name:?} has a NUL")))?;
            let (mut ptr, mut bytes) = (0, 0);
            // SAFETY: the module is loaded, and the name is a C string; it writes the address
            // and the size of the variable to `ptr` and `bytes`.
            let found = unsafe { (api.global)(&mut ptr, &mut bytes, module, c.as_ptr()) };
            api.check("cuModuleGetGlobal", found).map_err(|mut e| {
                e.message += &format!(", for `{name}`");
                e
            })?;
            m.globals.push((ptr, bytes));
        }
        Ok(m)
    }

    /// The module's memory `index`, as [`load`](Module::load) was given its name, as elements
    /// of `T`. Panics if its bytes are no whole number of them.
    pub fn global<T: Plain>(&self, index: usize) -> Global<'_, T> {
        let (ptr, bytes) = self.globals[index];
        let size = size_of::<T>();
        assert!(bytes.is_multiple_of(size), "{bytes} bytes are no whole number of elements of {size}");
        Global { module: self, ptr, len: bytes / size, of: PhantomData }
    }

    /// Launches kernel `index`, as [`load`](Module::load) was given its name, over `grid`,
    /// with `args`, each a parameter by [`arg`]. A block's shared memory, the kernel's own and
    /// the grid's together, may be all [`Cuda::max_shared`] gives. The driver gives 48 KB unasked,
    /// and a GPU from Volta on can give more once the kernel is let have it: the launch that
    /// needs more asks the driver, once, and a launch of no more than that asks it nothing. A
    /// `grid` the GPU does not take, or the kernel's `.maxntid` or `.reqntid` does not, is an
    /// error that says which size of it is past which limit.
    ///
    /// # Safety
    ///
    /// `args` must be as many as the kernel's parameters, each of the type its `.param` is;
    /// and the kernel is trusted with them: each pointer among them must be to a buffer as
    /// long as the kernel reads and writes through it, while it runs.
    pub unsafe fn launch(&self, index: usize, grid: Grid, args: &[*const c_void]) -> Result<(), Error> {
        let ([bx, by, bz], [tx, ty, tz]) = (grid.blocks, grid.threads);
        let (api, k, params) = (self.driver.current()?, &self.kernels[index], args.as_ptr().cast_mut().cast());
        self.let_have(k, &grid)?;
        // SAFETY: the caller's. The driver reads `args`, and through them, as it launches.
        let launched = unsafe {
            (api.launch)(k.function, bx, by, bz, tx, ty, tz, grid.shared, ptr::null_mut(), params, ptr::null_mut())
        };
        api.check("cuLaunchKernel", launched).map_err(|e| self.driver.refused(e, launched, &grid, k))
    }

    /// Launches kernel `index` as [`launch`](Module::launch) does, but cooperatively: every
    /// block of `grid` runs at once, so that each may wait for the others, and the driver gives
    /// the launch a barrier for the grid, which a kernel's `grid.sync()` waits at. A grid of
    /// more blocks than the GPU runs at once, [`resident`](Module::resident), is refused, as is
    /// a GPU that launches none cooperatively, one before Pascal's.
    ///
    /// # Safety
    ///
    /// As for [`launch`](Module::launch).
    pub unsafe fn launch_cooperative(&self, index: usize, grid: Grid, args: &[*const c_void]) -> Result<(), Error> {
        if !self.cooperative {
            let message = "the GPU launches no kernel cooperatively; one from Pascal's on, `sm_60`, does";
            return Err(Error::new("cuLaunchCooperativeKernel", message.into()));
        }
        let ([bx, by, bz], [tx, ty, tz]) = (grid.blocks, grid.threads);
        let (api, k, params) = (self.driver.current()?, &self.kernels[index], args.as_ptr().cast_mut().cast());
        self.let_have(k, &grid)?;
        // SAFETY: the caller's. The driver reads `args`, and through them, as it launches.
        let launched = unsafe {
            (api.launch_cooperative)(k.function, bx, by, bz, tx, ty, tz, grid.shared, ptr::null_mut(), params)
        };
        api.check("cuLaunchCooperativeKernel", launched).map_err(|e| {
            let mut e = self.driver.refused(e, launched, &grid, k);
            // CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE, which says no more.
            let threads = tx.saturating_mul(ty).saturating_mul(tz);
            if launched == 720
                && let Ok(most) = self.resident(index, threads, grid.shared)
            {
                let blocks = u64::from(bx) * u64::from(by) * u64::from(bz);
                e.message += &format!(": the grid has {blocks} blocks, and the GPU runs {most} of these at once");
            }
            e
        })
    }

    /// How many blocks of kernel `index` the GPU runs at once, of `threads` threads each and
    /// `shared` bytes of the launch's shared memory: as many as one of its SMs holds, by the
    /// registers, the shared memory and the threads each takes, times how many SMs it has. A
    /// cooperative launch of the kernel has at most this many. The kernel is let have `shared`
    /// bytes first, as for a launch, and `shared` past what a block has is an error that says so.
    pub fn resident(&self, index: usize, threads: u32, shared: u32) -> Result<u32, Error> {
        let (api, k) = (self.driver.current()?, &self.kernels[index]);
        self.let_have(k, &Grid::new(1, threads).with_shared(shared))?;
        let (threads, mut blocks) = (c_int::try_from(threads).unwrap_or(c_int::MAX), 0);
        // SAFETY: `k.function` is a kernel of the module, which is loaded; it writes the count to
        // `blocks`.
        let found = unsafe { (api.occupancy)(&mut blocks, k.function, threads, shared as usize) };
        api.check("cuOccupancyMaxActiveBlocksPerMultiprocessor", found)?;
        Ok(blocks as u32 * self.multiprocessors)
    }

    /// Lets kernel `k` have the `grid.shared` bytes of shared memory the launch over `grid` gives
    /// it beside its own. The driver gives a kernel 48 KB in all unasked, and a GPU from Volta on
    /// can give more once the kernel is let have it, up to what a block has in all. A launch that
    /// needs no more than the kernel has been let have asks the driver nothing; one that needs
    /// more asks once, and the kernel keeps what it was let. One past what the GPU gives is the
    /// driver's `CUDA_ERROR_INVALID_VALUE` and the size that is past which limit. `grid` is
    /// borrowed and not copied, so that a launch that asks nothing does no more than compare.
    fn let_have(&self, k: &Kernel, grid: &Grid) -> Result<(), Error> {
        if grid.shared <= k.room.get() {
            return Ok(());
        }
        let api = self.driver.current()?;
        // The driver refuses a count of bytes that is no `c_int`, and past what the GPU has.
        let bytes = c_int::try_from(grid.shared).unwrap_or(c_int::MAX);
        // SAFETY: `k.function` is a kernel of the module, which is loaded.
        let set = unsafe { (api.set_attribute)(k.function, 8, bytes) };
        api.check("cuFuncSetAttribute", set).map_err(|e| self.driver.refused(e, set, grid, k))?;
        k.room.set(grid.shared);
        Ok(())
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // A module whose GPU's context cannot be made current is left, and not unloaded in
        // another's.
        if let Ok(api) = self.driver.current() {
            // SAFETY: the module was loaded, and is unloaded once, here.
            unsafe { (api.unload_module)(self.module) };
        }
    }
}

/// A module's memory on the GPU, one of its variables in global or constant memory, as
/// elements of `T`: memory its kernels name, which keeps what they write from launch to launch.
/// The driver allocated it as it loaded the module, holding what the PTX declares, zeros or
/// what it is initialized to, and frees it with the module. It is written through the module
/// shared, as a launch writes it.
pub struct Global<'m, T: Plain> {
    module: &'m Module,
    ptr: u64,
    len: usize,
    of: PhantomData<T>,
}

impl<T: Plain> Global<'_, T> {
    /// Its address, for a kernel to be given; it points at something while the module lasts.
    pub fn ptr(&self) -> Ptr<T> {
        self.at(0)
    }

    /// The address of its element `i`, for a kernel to be given the part from there on.
    /// Panics if `i` is past its end.
    pub fn at(&self, i: usize) -> Ptr<T> {
        assert!(i <= self.len, "element {i} of a module's memory of {}", self.len);
        Ptr { addr: self.ptr + (i * size_of::<T>()) as u64, of: PhantomData }
    }

    /// How many elements it has.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies `data` to it, after the kernels launched before have run. Panics if `data` has
    /// not as many elements.
    pub fn write(&self, data: &[T]) -> Result<(), Error> {
        assert_eq!(data.len(), self.len, "a module's memory of {} elements written with {}", self.len, data.len());
        self.module.driver.write(self.ptr, data)
    }

    /// What it holds, after the kernels launched before have run.
    pub fn read(&self) -> Result<Vec<T>, Error> {
        let mut out = vec![T::default(); self.len];
        self.read_into(&mut out)?;
        Ok(out)
    }

    /// Copies what it holds to `out`, after the kernels launched before have run. Panics if
    /// `out` has not as many elements.
    pub fn read_into(&self, out: &mut [T]) -> Result<(), Error> {
        assert_eq!(out.len(), self.len, "a module's memory of {} elements read into {}", self.len, out.len());
        self.module.driver.read_into(self.ptr, out)
    }

    /// Sets every element to zero, after the kernels launched before have run.
    pub fn zero(&self) -> Result<(), Error> {
        self.module.driver.zero(self.ptr, self.len * size_of::<T>())
    }
}

impl<T: Plain> fmt::Debug for Global<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Global({} elements at {:#x})", self.len, self.ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A call asks the driver to switch only when it goes to another GPU than the last call
    /// did, and a context the driver would not make current is not taken for the current one.
    /// The driver here is a cell that holds the context it was last given, and counts the gives.
    #[test]
    fn a_context_is_made_current_only_when_another_is() {
        let (remembered, driver, gives) = (Cell::new(ptr::null_mut()), Cell::new(ptr::null_mut()), Cell::new(0));
        let (a, b) = (ptr::without_provenance_mut(1), ptr::without_provenance_mut(2));
        let give = |context| {
            driver.set(context);
            gives.set(gives.get() + 1);
            Ok(())
        };
        for (context, gives_so_far) in [(a, 1), (a, 1), (b, 2), (b, 2), (a, 3), (a, 3)] {
            enter(&remembered, context, || give(context)).unwrap();
            assert_eq!((driver.get(), gives.get()), (context, gives_so_far));
        }
        let refused = enter(&remembered, b, || Err(Error::new("cuCtxSetCurrent", "refused".into())));
        assert_eq!(refused.unwrap_err().message, "refused");
        // It tries again, for the context the driver last had is still `a`.
        enter(&remembered, b, || give(b)).unwrap();
        assert_eq!((driver.get(), gives.get()), (b, 4));
    }
}
