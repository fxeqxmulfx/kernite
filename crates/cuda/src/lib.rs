//! The CUDA driver, as far as kgen's kernels need it: memory on a GPU, PTX modules loaded
//! into it, and their kernels launched.
//!
//! The driver is `libcuda.so.1`, which comes with NVIDIA's driver rather than with the
//! toolkit. It is opened by [`Cuda::new`], not linked: a program that never asks for a GPU
//! builds and runs where there is none, and one that asks is told why it got none.
//!
//! ```no_run
//! let cuda = kgen_cuda::Cuda::new()?;
//! let xs = cuda.upload(&[1.0f32, 2.0, 3.0])?;
//! // A kernel is launched on `xs.ptr()` here.
//! cuda.sync()?;
//! println!("{:?}", xs.read()?);
//! # Ok::<(), kgen_cuda::Error>(())
//! ```
//!
//! Everything is for the thread that made the [`Cuda`]: the GPU's primary context is current
//! there, and the buffers and modules hold on to it, so none of them is `Send`. Kernels run
//! in the order they are launched, and a copy to or from a buffer waits for those launched
//! before it; [`Cuda::sync`] waits for all of them, and says whether one failed.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;

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

/// The driver's functions that kgen calls, each by the name `cuda.h` gives it under its own:
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
    device_get = c"cuDeviceGet"(*mut c_int, c_int);
    device_name = c"cuDeviceGetName"(*mut c_char, c_int, c_int);
    retain = c"cuDevicePrimaryCtxRetain"(*mut Handle, c_int);
    release = c"cuDevicePrimaryCtxRelease_v2"(c_int);
    set_current = c"cuCtxSetCurrent"(Handle);
    synchronize = c"cuCtxSynchronize"();
    alloc = c"cuMemAlloc_v2"(*mut u64, usize);
    free = c"cuMemFree_v2"(u64);
    to_device = c"cuMemcpyHtoD_v2"(u64, *const c_void, usize);
    to_host = c"cuMemcpyDtoH_v2"(*mut c_void, u64, usize);
    memset = c"cuMemsetD8_v2"(u64, u8, usize);
    load_module = c"cuModuleLoadDataEx"(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void);
    unload_module = c"cuModuleUnload"(Handle);
    function = c"cuModuleGetFunction"(*mut Handle, Handle, *const c_char);
    launch = c"cuLaunchKernel"(
        Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void
    );
    error_name = c"cuGetErrorName"(CuResult, *mut *const c_char);
    error_string = c"cuGetErrorString"(CuResult, *mut *const c_char);
}

impl Api {
    /// What the function `call` returned, as a `Result`.
    fn check(&self, call: &'static str, result: CuResult) -> Result<(), Error> {
        if result == 0 {
            return Ok(());
        }
        let (mut name, mut meaning) = (ptr::null(), ptr::null());
        // SAFETY: each sets its pointer to a string that lasts as long as the driver, or
        // leaves it null for a result it does not know.
        let (name, meaning) = unsafe {
            (self.error_name)(result, &mut name);
            (self.error_string)(result, &mut meaning);
            (string(name), string(meaning))
        };
        Err(Error::new(call, format!("{name} ({result}): {meaning}")))
    }
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

/// The driver's functions, and the GPU whose primary context they work in, which is let go
/// when the last thing holding it is dropped.
struct Driver {
    api: Api,
    device: c_int,
}

impl Drop for Driver {
    fn drop(&mut self) {
        // SAFETY: `Cuda::new` retained the primary context, once for each `Driver`.
        unsafe { (self.api.release)(self.device) };
    }
}

/// A GPU, its primary context current on this thread.
pub struct Cuda {
    driver: Rc<Driver>,
    name: String,
}

impl Cuda {
    /// The first GPU, through the driver opened now; or why there is none to have.
    pub fn new() -> Result<Cuda, Error> {
        // SAFETY: opening the driver runs its initialisers, which ask nothing of this
        // program. It is never closed: like a library linked, it lasts as long as the program.
        let lib = unsafe { dlopen(c"libcuda.so.1".as_ptr(), RTLD_NOW) };
        if lib.is_null() {
            return Err(Error::new("dlopen", format!("no CUDA driver: {}", dl_error())));
        }
        let api = Api::load(lib)?;
        let (mut device, mut context) = (0, ptr::null_mut());
        // SAFETY: each is called as `cuda.h` declares it, and writes only what it is given
        // the address of.
        unsafe {
            api.check("cuInit", (api.init)(0))?;
            api.check("cuDeviceGet", (api.device_get)(&mut device, 0))?;
            api.check("cuDevicePrimaryCtxRetain", (api.retain)(&mut context, device))?;
        }
        let driver = Rc::new(Driver { api, device });
        let (api, mut name) = (&driver.api, [0u8; 256]);
        // SAFETY: the context was retained; the name is written as a C string, cut to fit.
        unsafe {
            api.check("cuCtxSetCurrent", (api.set_current)(context))?;
            api.check("cuDeviceGetName", (api.device_name)(name.as_mut_ptr().cast(), name.len() as c_int, device))?;
        }
        let name = CStr::from_bytes_until_nul(&name).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        Ok(Cuda { driver, name })
    }

    /// The GPU's name, as its driver gives it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Waits for every kernel launched to finish, and says whether one failed.
    pub fn sync(&self) -> Result<(), Error> {
        let api = &self.driver.api;
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
        let (api, mut ptr) = (&self.driver.api, 0);
        // SAFETY: it writes the address to `ptr`. It refuses no bytes, so it is asked for one.
        api.check("cuMemAlloc", unsafe { (api.alloc)(&mut ptr, bytes.max(1)) })?;
        Ok(Buffer { driver: self.driver.clone(), ptr, len, of: PhantomData })
    }
}

/// `len` elements of `T` on the GPU, freed when it is dropped.
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
        let api = &self.driver.api;
        // SAFETY: it reads as many bytes from `data` as it has, into a buffer as long.
        api.check("cuMemcpyHtoD", unsafe { (api.to_device)(self.ptr, data.as_ptr().cast(), self.bytes()) })
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
        let api = &self.driver.api;
        // SAFETY: it writes as many bytes to `out` as it has, and any bits are a `T`, as
        // `Plain` promises.
        api.check("cuMemcpyDtoH", unsafe { (api.to_host)(out.as_mut_ptr().cast(), self.ptr, self.bytes()) })
    }

    /// Sets every element to zero, after the kernels launched before have run.
    pub fn zero(&mut self) -> Result<(), Error> {
        let api = &self.driver.api;
        // SAFETY: it writes the buffer's bytes, and no others.
        api.check("cuMemsetD8", unsafe { (api.memset)(self.ptr, 0, self.bytes()) })
    }

    fn bytes(&self) -> usize {
        self.len * size_of::<T>()
    }
}

impl<T: Plain> Drop for Buffer<T> {
    fn drop(&mut self) {
        // SAFETY: `cuMemAlloc` gave the address, and it is freed once, here.
        unsafe { (self.driver.api.free)(self.ptr) };
    }
}

impl<T: Plain> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Buffer({} elements at {:#x})", self.len, self.ptr)
    }
}

/// An address on the GPU, of elements of `T`: what a kernel's `*const T` or `*mut T` is
/// given. A [`Buffer`] gives it, and it points at something only while the buffer lasts.
#[repr(transparent)]
pub struct Ptr<T> {
    addr: u64,
    of: PhantomData<*const T>,
}

impl<T> Ptr<T> {
    /// The address, as the GPU reckons it.
    pub fn addr(self) -> u64 {
        self.addr
    }
}

impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Ptr<T> {
        *self
    }
}

impl<T> Copy for Ptr<T> {}

impl<T> fmt::Debug for Ptr<T> {
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
/// a [`Ptr`], an integer of 32 or 64 bits or an `f32` as itself.
pub trait Param: sealed::Sealed + Copy {}

macro_rules! plain {
    ($($t:ty),*) => { $(impl sealed::Sealed for $t {} impl Plain for $t {})* };
}

plain!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl sealed::Sealed for usize {}
impl sealed::Sealed for isize {}
impl<T> sealed::Sealed for Ptr<T> {}

impl Param for u32 {}
impl Param for i32 {}
impl Param for u64 {}
impl Param for i64 {}
impl Param for usize {}
impl Param for isize {}
impl Param for f32 {}
impl<T> Param for Ptr<T> {}

/// `x` as [`Module::launch`] takes a parameter: the address of its value, which the driver
/// reads as it launches.
pub fn arg<T: Param>(x: &T) -> *const c_void {
    (x as *const T).cast()
}

/// How many blocks a kernel is launched in, and how many threads each block has: what it
/// reads as `nctaid` and `ntid`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Blocks along x, y and z.
    pub blocks: [u32; 3],
    /// Threads in each block along x, y and z.
    pub threads: [u32; 3],
}

impl Grid {
    /// `blocks` blocks of `threads` threads, along x.
    pub fn new(blocks: u32, threads: u32) -> Grid {
        Grid { blocks: [blocks, 1, 1], threads: [threads, 1, 1] }
    }
}

/// A PTX module, compiled by the driver for its GPU, and the kernels of it that were asked for.
pub struct Module {
    driver: Rc<Driver>,
    module: Handle,
    kernels: Vec<Handle>,
}

impl Module {
    /// `ptx` compiled for `cuda`'s GPU, with its kernels `names`, in that order. If the driver
    /// will not compile it, the error's `log` is what its compiler said.
    pub fn load(cuda: &Cuda, ptx: &str, names: &[&str]) -> Result<Module, Error> {
        let api = &cuda.driver.api;
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
        let mut m = Module { driver: cuda.driver.clone(), module, kernels: vec![] };
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
            m.kernels.push(f);
        }
        Ok(m)
    }

    /// Launches kernel `index`, as [`load`](Module::load) was given its name, over `grid`,
    /// with `args`, each a parameter by [`arg`].
    ///
    /// # Safety
    ///
    /// `args` must be as many as the kernel's parameters, each of the type its `.param` is;
    /// and the kernel is trusted with them: each pointer among them must be to a buffer as
    /// long as the kernel reads and writes through it, while it runs.
    pub unsafe fn launch(&self, index: usize, grid: Grid, args: &[*const c_void]) -> Result<(), Error> {
        let ([bx, by, bz], [tx, ty, tz]) = (grid.blocks, grid.threads);
        let (api, f, params) = (&self.driver.api, self.kernels[index], args.as_ptr().cast_mut().cast());
        // SAFETY: the caller's. The driver reads `args`, and through them, as it launches.
        let launched = unsafe { (api.launch)(f, bx, by, bz, tx, ty, tz, 0, ptr::null_mut(), params, ptr::null_mut()) };
        api.check("cuLaunchKernel", launched)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: the module was loaded, and is unloaded once, here.
        unsafe { (self.driver.api.unload_module)(self.module) };
    }
}
