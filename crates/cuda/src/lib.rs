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
//! [`Cuda::memory`] says how many bytes of a GPU's memory are free, and how many it has.
//!
//! [`Cuda::reserve`] makes a buffer that grows and shrinks at the same addresses: it reserves addresses for the
//! elements it may have, [`Buffer::grow`] backs them with the GPU's memory, and [`Buffer::shrink`], which is `unsafe`,
//! gives the memory back, a granule of the driver's at a time, 2 MiB on the GPU here. [`Buffer::move_split`], `unsafe`
//! too, puts its granules from a split on in the host's memory, at the same addresses, where the GPU's kernels reach
//! them over the bus.
//!
//! A buffer is the GPU's own memory, and the host reaches it by copies. [`Unified`] memory is
//! reached by the host and by the GPU's kernels at one address, with no copy: [`Cuda::managed`]
//! makes memory the driver moves to whichever touches it, [`Cuda::pinned`] makes page-locked memory
//! in the host, [`Cuda::lend`] lends a slice of the host's own to the GPU, and [`Cuda::register`]
//! page-locks one where it is. The first two are zeros, and [`Cuda::managed_uninit`] and
//! [`Cuda::pinned_uninit`], which are `unsafe`, make them without, for memory that is written whole
//! before it is read. The slices it lends the host wait for the GPU's kernels first, and
//! [`Module::launch`] says what a launch keeps to of the slices the host holds. Managed memory is
//! sent to the GPU before a kernel needs it by [`Unified::prefetch`], and advised how it is used by
//! [`Unified::advise`].
//!
//! Kernels run in the order they are launched on one stream, and a GPU runs the work of two streams at once where it
//! has room. All the work goes to the GPU's null stream, unless a thread makes another stream current for a closure,
//! [`Stream::run`], which [`Cuda::stream`] makes: a launch goes to the current stream of its GPU on the thread, and so
//! do a prefetch, a copy that does not wait, [`Cuda::copy`], and a zeroing: `zero`, and the zeros that [`Cuda::alloc`],
//! [`Cuda::managed`], [`Buffer::grow`] and the load of a module write. So do [`Cuda::copy_at`] and [`Cuda::zero_at`],
//! which copy and zero at addresses that a program works out, and are `unsafe`, as kernite checks none of them. A
//! stream is blocking, so the copies and the drops that wait for the null stream's work wait for its work too.
//! [`Cuda::stream_apart`] makes a stream apart, which is not, at the GPU's greatest priority where it is urgent. An
//! [`Event`], which [`Cuda::event`] makes, is a point in the work of a stream: the host asks whether the work before it
//! is done, or waits for it, and another stream waits for it with no wait of the host. [`Cuda::timed_event`] makes one
//! that times work, and [`Event::duration_since`] says the time between two.
//!
//! A kernel whose blocks wait for each other is launched cooperatively,
//! [`Module::launch_cooperative`]: every block of the grid runs at once, so there are at most
//! [`Module::resident`] of them, and the driver gives the launch a barrier for the grid.

use std::cell::Cell;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::ptr;
use std::rc::Rc;

mod copy;
mod event;
mod graph;
mod half;
#[doc(hidden)]
pub mod named;
mod reserve;
mod safe;
mod stream;
mod unified;

use reserve::{Access, Properties, Reservation};
use safe::bounds;

pub use event::{Event, Timed, Untimed};
pub use graph::Graph;
pub use half::{bf16, f16};
pub use safe::{Memory, MemoryMut, Part, Quantity, Requirement, Why, holds};
pub use stream::Stream;
pub use unified::{Advice, Atomic, Location, Placement, Unified};

/// What every function of the driver returns: 0, or what went wrong.
type CuResult = c_int;

/// A context, a module, a function, a stream or an event: what the driver hands out, opaque.
type Handle = *mut c_void;

/// A place that the driver is told of, as `CUmemLocation` is: a kind of place, and which one.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MemLocation {
    kind: c_int,
    id: c_int,
}

impl MemLocation {
    /// The GPU `device`, which memory is on, or is reached from.
    fn gpu(device: c_int) -> MemLocation {
        MemLocation { kind: 1, id: device }
    }
}

// The dynamic loader, in the C library that every Rust program on Linux links.
unsafe extern "C" {
    fn dlopen(file: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

/// `dlopen` binds the library's own references to others as it opens it.
const RTLD_NOW: c_int = 2;

/// The driver's functions that kernite calls, each by the name `cuda.h` gives it under its own:
/// `cuMemAlloc` is `cuMemAlloc_v2`. A driver lacks the functions after the `?` if it is older than
/// the release that brought them. It is opened all the same, and the table keeps the error of
/// the lookup for the call that needs the function to return.
macro_rules! api {
    (
        $($field:ident = $symbol:literal($($arg:ty),*);)*
        ?
        $($maybe:ident = $maybe_symbol:literal($($maybe_arg:ty),*);)*
    ) => {
        struct Api {
            $($field: unsafe extern "C" fn($($arg),*) -> CuResult,)*
            $($maybe: Result<unsafe extern "C" fn($($maybe_arg),*) -> CuResult, Error>,)*
        }

        impl Api {
            /// The functions of the driver that `lookup` finds, by their names.
            fn load(lookup: impl Fn(&CStr) -> Result<*mut c_void, Error>) -> Result<Api, Error> {
                Ok(Api {
                    $($field: {
                        let f = lookup($symbol)?;
                        // SAFETY: `f` is not null, and `cuda.h` declares the function of this
                        // name with these parameters.
                        unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> CuResult>(f) }
                    },)*
                    $($maybe: lookup($maybe_symbol).map(|f| {
                        // SAFETY: as above.
                        unsafe {
                            std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($maybe_arg),*) -> CuResult>(f)
                        }
                    }),)*
                })
            }
        }

        /// The functions of a driver that refuses every call, for a test to replace the few it
        /// needs.
        #[cfg(test)]
        impl Api {
            fn refusing() -> Api {
                Api {
                    $($field: {
                        unsafe extern "C" fn refuse($(_: $arg),*) -> CuResult {
                            // CUDA_ERROR_UNKNOWN.
                            999
                        }
                        refuse
                    },)*
                    $($maybe: {
                        unsafe extern "C" fn refuse($(_: $maybe_arg),*) -> CuResult {
                            // CUDA_ERROR_UNKNOWN.
                            999
                        }
                        Ok(refuse as unsafe extern "C" fn($($maybe_arg),*) -> CuResult)
                    },)*
                }
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
    create_stream = c"cuStreamCreateWithPriority"(*mut Handle, c_uint, c_int);
    priorities = c"cuCtxGetStreamPriorityRange"(*mut c_int, *mut c_int);
    destroy_stream = c"cuStreamDestroy_v2"(Handle);
    sync_stream = c"cuStreamSynchronize"(Handle);
    create_event = c"cuEventCreate"(*mut Handle, c_uint);
    destroy_event = c"cuEventDestroy_v2"(Handle);
    record_event = c"cuEventRecord"(Handle, Handle);
    query_event = c"cuEventQuery"(Handle);
    sync_event = c"cuEventSynchronize"(Handle);
    wait_event = c"cuStreamWaitEvent"(Handle, Handle, c_uint);
    begin_capture = c"cuStreamBeginCapture_v2"(Handle, c_int);
    end_capture = c"cuStreamEndCapture"(Handle, *mut Handle);
    launch_graph = c"cuGraphLaunch"(Handle, Handle);
    destroy_graph = c"cuGraphDestroy"(Handle);
    destroy_graph_exec = c"cuGraphExecDestroy"(Handle);
    alloc = c"cuMemAlloc_v2"(*mut u64, usize);
    alloc_managed = c"cuMemAllocManaged"(*mut u64, usize, c_uint);
    free = c"cuMemFree_v2"(u64);
    alloc_host = c"cuMemHostAlloc"(*mut *mut c_void, usize, c_uint);
    free_host = c"cuMemFreeHost"(*mut c_void);
    register = c"cuMemHostRegister_v2"(*mut c_void, usize, c_uint);
    unregister = c"cuMemHostUnregister"(*mut c_void);
    to_device = c"cuMemcpyHtoD_v2"(u64, *const c_void, usize);
    to_host = c"cuMemcpyDtoH_v2"(*mut c_void, u64, usize);
    copy_peer = c"cuMemcpyPeer"(u64, Handle, u64, Handle, usize);
    copy_async = c"cuMemcpyAsync"(u64, u64, usize, Handle);
    memset = c"cuMemsetD8Async"(u64, u8, usize, Handle);
    memory_info = c"cuMemGetInfo_v2"(*mut usize, *mut usize);
    granularity = c"cuMemGetAllocationGranularity"(*mut usize, *const Properties, c_int);
    reserve_addresses = c"cuMemAddressReserve"(*mut u64, usize, usize, u64, u64);
    free_addresses = c"cuMemAddressFree"(u64, usize);
    create_memory = c"cuMemCreate"(*mut u64, usize, *const Properties, u64);
    release_memory = c"cuMemRelease"(u64);
    map_memory = c"cuMemMap"(u64, usize, usize, u64, u64);
    set_access = c"cuMemSetAccess"(u64, usize, *const Access, usize);
    unmap_memory = c"cuMemUnmap"(u64, usize);
    retain_memory = c"cuMemRetainAllocationHandle"(*mut u64, *mut c_void);
    range_attribute = c"cuMemRangeGetAttribute"(*mut c_void, usize, c_int, u64, usize);
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
    // These came with CUDA 12.2, which brought the NUMA node as a place.
    ?
    prefetch = c"cuMemPrefetchAsync_v2"(u64, usize, MemLocation, c_uint, Handle);
    advise = c"cuMemAdvise_v2"(u64, usize, c_int, MemLocation);
    // This one came with CUDA 12.8: `cuda.h` of the newest release maps `cuEventElapsedTime` to it, and a driver
    // before it has the first name alone, which kernite-cuda does not call.
    elapsed_time = c"cuEventElapsedTime_v2"(*mut f32, Handle, Handle);
    // This one came with CUDA 11.4: a driver before it makes no graph of a capture.
    instantiate = c"cuGraphInstantiateWithFlags"(*mut Handle, Handle, u64);
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

    /// What the driver says of GPU `device` as the `CUdevice_attribute` `attribute`.
    fn attribute_of(&self, device: c_int, attribute: c_int) -> Result<u32, Error> {
        let mut v = 0;
        // SAFETY: it writes the attribute's value to `v`.
        let asked = unsafe { (self.attribute)(&mut v, attribute, device) };
        self.check("cuDeviceGetAttribute", asked).map(|()| v as u32)
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
    Api::load(|name| symbol(lib, name))
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
        return Err(missing(name, &dl_error()));
    }
    Ok(f)
}

/// The error that says the driver has no function `name`, and why, as the loader says.
fn missing(name: &CStr, why: &str) -> Error {
    Error::new("dlsym", format!("the driver has no `{}`: {why}", name.to_string_lossy()))
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
    /// The function that failed, by its name in C: `cuInit`; `dlopen` if there is no driver, and
    /// `dlsym` for a function the driver does not have.
    /// A call refused before it is made names that call, whatever the reason: `cuMemAllocManaged`
    /// where the GPU cannot allocate managed memory, or a GPU cannot reach it while the host does,
    /// `cuMemPrefetchAsync` where the memory is page-locked, which the driver does not prefetch,
    /// `cuMemAdvise` where a NUMA node is given as an accessor, and `cuMemHostRegister` where the
    /// GPU cannot page-lock memory the host has, or reach it at the host's addresses.
    /// [`Cuda::lend`] refuses no call of the driver's, and names `cuDeviceGetAttribute`, which told
    /// that the GPU cannot.
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

    /// `Ok` if `other` is this GPU. A launch, a copy and the time between two events are each of one GPU, so what is of
    /// another is refused: the error is of `call`, and `message` says what is of another GPU than the call is made on.
    fn require_same_gpu(&self, other: &Driver, call: &'static str, message: &str) -> Result<(), Error> {
        if self.context != other.context {
            return Err(Error::new(call, message.to_string()));
        }
        Ok(())
    }

    /// Makes this GPU's context current on the thread, whichever is now.
    fn set_current(&self) -> Result<(), Error> {
        // SAFETY: the context was retained, and is held until `self` is dropped.
        self.api.check("cuCtxSetCurrent", unsafe { (self.api.set_current)(self.context) })
    }

    /// Waits for every kernel launched on this GPU to finish, and says whether one failed.
    fn sync(&self) -> Result<(), Error> {
        let api = self.current()?;
        // SAFETY: it takes nothing.
        api.check("cuCtxSynchronize", unsafe { (api.synchronize)() })
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

    /// Sets the `bytes` bytes at `ptr` to zero, on the current stream: after the work that stream was given before,
    /// and it does not wait.
    fn zero(&self, ptr: u64, bytes: usize) -> Result<(), Error> {
        let api = self.current()?;
        // SAFETY: it writes the `bytes` bytes at `ptr`, which the caller has, and the stream is the null stream or one
        // that a `run` of this thread holds.
        api.check("cuMemsetD8Async", unsafe { (api.memset)(ptr, 0, bytes, self.stream()) })
    }

    /// How many bytes of this GPU's memory are free, and how many it has in all.
    fn memory(&self) -> Result<(usize, usize), Error> {
        let api = self.current()?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: it writes the free bytes to `free` and the total to `total`.
        api.check("cuMemGetInfo", unsafe { (api.memory_info)(&mut free, &mut total) })?;
        Ok((free, total))
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
        self.api.attribute_of(self.device, attribute)
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
                // The driver weighs the threads of a block in all, not along each axis. A product past `u128` is more
                // than any size is: sizes are read from the PTX's text, comments too, which the driver does not.
                let all = sizes.iter().fold(1, |all: u128, &n| all.saturating_mul(n));
                rows.push((tx * ty * tz, "each block has", "threads", who, 1, all));
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
        Cuda::of(driver, name)
    }

    /// The GPU of `driver`, called `name`, with what the GPU says of itself.
    fn of(driver: Rc<Driver>, name: String) -> Result<Cuda, Error> {
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
        self.driver.sync()
    }

    /// The bytes of the GPU's memory that nothing holds, and all the bytes it has for programs, as `(free, total)`:
    /// what `cuMemGetInfo` says. The total is the same each time. Any program holds memory of the GPU, not this one
    /// alone, so the free bytes are true as they are asked, and a buffer made after may find fewer.
    pub fn memory(&self) -> Result<(usize, usize), Error> {
        self.driver.memory()
    }

    /// `len` elements on the GPU, all zero. The zeros go to the current stream of the GPU, as a launch does, after the
    /// work that stream was given before, and the call does not wait for them: the host's calls that reach the buffer
    /// do, as they wait for a kernel.
    pub fn alloc<T: Plain>(&self, len: usize) -> Result<Buffer<T>, Error> {
        let mut b = self.uninit(len)?;
        b.zero()?;
        Ok(b)
    }

    /// A copy of `data` on the GPU.
    pub fn upload<T: Plain>(&self, data: &[T]) -> Result<Buffer<T>, Error> {
        let mut b = self.uninit(data.len())?;
        b.write(data)?;
        Ok(b)
    }

    /// `len` elements on the GPU, as they were left.
    fn uninit<T: Plain>(&self, len: usize) -> Result<Buffer<T>, Error> {
        let bytes = bytes::<T>("cuMemAlloc", len)?;
        let (api, mut ptr) = (self.driver.current()?, 0);
        // SAFETY: it writes the address to `ptr`. It refuses no bytes, so it is asked for one.
        api.check("cuMemAlloc", unsafe { (api.alloc)(&mut ptr, bytes.max(1)) })?;
        Ok(Buffer { driver: self.driver.clone(), ptr, len, reservation: None, of: PhantomData })
    }
}

/// How many bytes `len` elements of `T` take; or the error that says the `call` is refused, as they are more than there
/// are addresses.
fn bytes<T>(call: &'static str, len: usize) -> Result<usize, Error> {
    len.checked_mul(size_of::<T>()).ok_or_else(|| too_many::<T>(call, len))
}

/// The error that says the `call` is refused, as `len` elements of `T` are more bytes than there are addresses.
fn too_many<T>(call: &'static str, len: usize) -> Error {
    Error::new(call, format!("{len} elements of {} bytes are more bytes than there are addresses", size_of::<T>()))
}

/// `Ok` if `got` is `want`, the elements of the memory `owner`; else the error that says the `call` is refused, as
/// `owner` has `want` elements, and `action` them with `got`.
fn same_len(call: &'static str, owner: &str, want: usize, action: &str, got: usize) -> Result<(), Error> {
    if want != got {
        return Err(Error::new(call, format!("{owner} of {want} elements {action} {got}")));
    }
    Ok(())
}

/// `len` elements of `T` on one GPU, the one that made it, freed when it is dropped.
pub struct Buffer<T: Plain> {
    driver: Rc<Driver>,
    ptr: u64,
    len: usize,
    /// What it holds beside its memory, if [`Cuda::reserve`] made it, and `None` if `alloc` or `upload` did.
    reservation: Option<Reservation>,
    of: PhantomData<T>,
}

impl<T: Plain> Buffer<T> {
    /// Its address, for a kernel to be given.
    pub fn ptr(&self) -> Ptr<T> {
        Ptr::new(self.ptr)
    }

    /// The address of its element `i`, for a kernel to be given the part from there on; `None`
    /// if `i` is past its end.
    pub fn at(&self, i: usize) -> Option<Ptr<T>> {
        Ptr::element(self.ptr, self.len, i)
    }

    /// Its elements in `range`, borrowed, for a kernel that is no `unsafe fn` and a copy to be lent that part alone:
    /// `xs.part(3..10)` is elements 3 to 9, and `xs.part(3..)` is elements 3 to the last. `None` if `range` ends past
    /// the end or before it starts, as [`at`](Buffer::at) gives `None` for an element past it.
    pub fn part(&self, range: impl RangeBounds<usize>) -> Option<Part<&Buffer<T>>> {
        let (from, len) = bounds(range, self.len)?;
        Some(Part::new(self, self.at(from)?, len))
    }

    /// Its elements in `range`, borrowed mutably, for a kernel that is no `unsafe fn` and a copy to write that part
    /// alone: `xs.part_mut(3..10)` is elements 3 to 9. It borrows the buffer mutably, as `&mut xs[3..10]` borrows a
    /// slice, so the host holds nothing of the buffer, and no kernel is lent it, while the part lasts. `None` if
    /// `range` ends past the end or before it starts.
    pub fn part_mut(&mut self, range: impl RangeBounds<usize>) -> Option<Part<&mut Buffer<T>>> {
        let (from, len) = bounds(range, self.len)?;
        let at = self.at(from)?;
        Some(Part::new(self, at, len))
    }

    /// How many elements it has.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies `data` to it, after the kernels launched before have run. It is an error if `data`
    /// has not as many elements: `cuMemcpyHtoD: a buffer of 4 elements written with 2`.
    pub fn write(&mut self, data: &[T]) -> Result<(), Error> {
        same_len("cuMemcpyHtoD", "a buffer", self.len, "written with", data.len())?;
        self.driver.write(self.ptr, data)
    }

    /// What it holds, after the kernels launched before have run.
    pub fn read(&self) -> Result<Vec<T>, Error> {
        let mut out = vec![T::default(); self.len];
        self.read_into(&mut out)?;
        Ok(out)
    }

    /// Copies what it holds to `out`, after the kernels launched before have run. It is an error
    /// if `out` has not as many elements: `cuMemcpyDtoH: a buffer of 4 elements read into 2`.
    pub fn read_into(&self, out: &mut [T]) -> Result<(), Error> {
        same_len("cuMemcpyDtoH", "a buffer", self.len, "read into", out.len())?;
        self.driver.read_into(self.ptr, out)
    }

    /// Sets every element to zero, on the current stream, after the work that stream was given before. It returns
    /// before the GPU has done it: `read`, `write` and [`Cuda::sync`] wait for it.
    pub fn zero(&mut self) -> Result<(), Error> {
        self.zero_bytes(0, self.len * size_of::<T>())
    }

    /// Copies what `from` holds to it, after the kernels launched before have run on the GPU of either, and before
    /// those launched after run on either. It returns before the GPU has done it, as `zero` does: `read`, `write` and
    /// [`Cuda::sync`] wait for it. `from` may be on another GPU than it: where the two have a link between them, the
    /// driver's peer access, the copy goes over the link, and where they have none it goes through the host. It is an
    /// error if `from` has not as many elements: `cuMemcpyPeer: a buffer of 4 elements copied from one of 2`.
    pub fn copy_from(&mut self, from: &Buffer<T>) -> Result<(), Error> {
        same_len("cuMemcpyPeer", "a buffer", self.len, "copied from one of", from.len)?;
        self.driver.copy_from(self.ptr, &from.driver, from.ptr, self.len * size_of::<T>())
    }
}

impl<T: Plain> Drop for Buffer<T> {
    fn drop(&mut self) {
        match self.reservation {
            // SAFETY: `cuMemAlloc` gave the address, and it is freed once, here.
            None => self.driver.give_back(|api| unsafe { (api.free)(self.ptr) }),
            Some(reservation) => self.release(reservation),
        }
    }
}

impl<T: Plain> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Buffer({} elements at {:#x})", self.len, self.ptr)
    }
}

/// An address on the GPU, of elements of `T`: what a kernel's `*const T` or `*mut T` is given. A [`Buffer`] gives it,
/// and it points at something only while the buffer lasts; [`Ptr::new`] makes it from an address a program works out.
/// A kernel's pointer is to a type with a size, but `T` may have none: a kernel that is wrong in this is one error, at
/// its type, and not a second, here.
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

impl<T> Ptr<T> {
    /// The address `addr`, as the GPU reckons it: one that a program works out from those its memory gives, say. It is
    /// safe to make, as only an `unsafe` launch takes it, which says that the memory there is the kernel's.
    pub fn new(addr: u64) -> Ptr<T> {
        Ptr { addr, of: PhantomData }
    }

    /// The address of element `i` of the `len` elements at `addr`, or of the place past them; `None` for `i` beyond
    /// that.
    fn element(addr: u64, len: usize, i: usize) -> Option<Ptr<T>> {
        (i <= len).then(|| Ptr::new(addr + (i * size_of::<T>()) as u64))
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
    /// The memory it allocated for the variables the module has no room for, freed with it.
    outside: Vec<Buffer<u8>>,
    /// The GPU's, as [`Cuda`] has them.
    multiprocessors: u32,
    cooperative: bool,
}

impl Module {
    /// `ptx` compiled for `cuda`'s GPU, with its kernels `names` and its memory `globals`, each by its name, in that
    /// order: a variable in global or constant memory, `None`; or `Some((bytes, align))`, memory the module has no room
    /// for, which `ptxas` cannot lay out in it: the name is that of a `.u64` in the module's constant memory, where its
    /// kernels read the address of the memory this allocates, `bytes` of it at a multiple of `align`, freed with the
    /// module. The memory is zeros, which go to the current stream of the GPU as those of [`alloc`](Cuda::alloc) do,
    /// after the work that stream was given before, and the call does not wait for them. If the driver will not compile
    /// the PTX, the error's `log` is what its compiler said; and a GPU that has not the memory is an error that names
    /// the variable and its bytes. Panics if an alignment is 0.
    pub fn load(
        cuda: &Cuda,
        ptx: &str,
        names: &[&str],
        globals: &[(&str, Option<(usize, usize)>)],
    ) -> Result<Module, Error> {
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
            outside: vec![],
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
        for &(name, outside) in globals {
            let c = CString::new(name).map_err(|_| Error::new("cuModuleGetGlobal", format!("{name:?} has a NUL")))?;
            let (mut ptr, mut bytes) = (0, 0);
            // SAFETY: the module is loaded, and the name is a C string; it writes the address
            // and the size of the variable to `ptr` and `bytes`.
            let found = unsafe { (api.global)(&mut ptr, &mut bytes, module, c.as_ptr()) };
            api.check("cuModuleGetGlobal", found).map_err(|mut e| {
                e.message += &format!(", for `{name}`");
                e
            })?;
            if let Some((size, align)) = outside {
                // As many bytes more as rounding the address up to a multiple of `align` may take.
                let memory = cuda.alloc::<u8>(size.saturating_add(align.saturating_sub(1))).map_err(|mut e| {
                    e.message += &format!(", for `{name}`, {size} bytes");
                    e
                })?;
                // The variable is the module's `.u64` that its kernels read the address from.
                let at = memory.ptr().addr().next_multiple_of(align as u64);
                m.driver.write(ptr, &[at])?;
                m.outside.push(memory);
                (ptr, bytes) = (at, size);
            }
            m.globals.push((ptr, bytes));
        }
        Ok(m)
    }

    /// The module's memory `index`, as [`load`](Module::load) was given its name, as elements
    /// of `T`. It is an error if `load` was given no such name, or if its bytes are no whole
    /// number of `T`: `cuModuleGetGlobal: 6 bytes are no whole number of elements of 4`.
    pub fn global<T: Plain>(&self, index: usize) -> Result<Global<'_, T>, Error> {
        let refused = |message| Error::new("cuModuleGetGlobal", message);
        let Some(&(ptr, bytes)) = self.globals.get(index) else {
            return Err(refused(format!("there is no memory {index} in a module of {}", self.globals.len())));
        };
        let size = size_of::<T>();
        if !bytes.is_multiple_of(size) {
            return Err(refused(format!("{bytes} bytes are no whole number of elements of {size}")));
        }
        Ok(Global { module: self, ptr, len: bytes / size, of: PhantomData })
    }

    /// Launches kernel `index`, as [`load`](Module::load) was given its name, over `grid`,
    /// with `args`, each a parameter by [`arg`], on the current stream of the module's GPU: the
    /// null stream, or the one [`Stream::run`] made current. A block's shared memory, the kernel's own and
    /// the grid's together, may be all [`Cuda::max_shared`] gives. The driver gives 48 KB unasked,
    /// and a GPU from Volta on can give more once the kernel is let have it: the launch that
    /// needs more asks the driver, once, and a launch of no more than that asks it nothing. A
    /// `grid` the GPU does not take, or the kernel's `.maxntid` or `.reqntid` does not, is an
    /// error that says which size of it is past which limit. So is an `index` of no kernel the
    /// module has.
    ///
    /// # Safety
    ///
    /// `args` must be as many as the kernel's parameters, each of the type its `.param` is; and the
    /// kernel is trusted with them. Each pointer among them must be to a buffer or to [`Unified`]
    /// memory of the GPU the module is loaded on, as long as the kernel reads and writes through
    /// it. From the launch until the kernel is done, the host must hold no slice of memory the
    /// kernel writes, and no slice it may write of memory the kernel reads. The host may hold
    /// [`atomics`](Unified::atomics) of any memory the kernel reaches, and the kernel then reaches each of its
    /// elements by the element's size alone, as a race of two sizes is outside PTX's memory model. A slice that `host`
    /// or `host_mut` gives after the launch waits for the kernel, so only a slice taken before it, or one that
    /// `host_now` or `host_now_mut` gives, can break this.
    pub unsafe fn launch(&self, index: usize, grid: Grid, args: &[*const c_void]) -> Result<(), Error> {
        let ([bx, by, bz], [tx, ty, tz]) = (grid.blocks, grid.threads);
        let k = self.kernel("cuLaunchKernel", index)?;
        let (api, params) = (self.driver.current()?, args.as_ptr().cast_mut().cast());
        self.let_have(k, &grid)?;
        // SAFETY: the caller's. The driver reads `args`, and through them, as it launches.
        let launched = unsafe {
            (api.launch)(k.function, bx, by, bz, tx, ty, tz, grid.shared, self.driver.stream(), params, ptr::null_mut())
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
        let k = self.kernel("cuLaunchCooperativeKernel", index)?;
        let (api, params) = (self.driver.current()?, args.as_ptr().cast_mut().cast());
        self.let_have(k, &grid)?;
        // SAFETY: the caller's. The driver reads `args`, and through them, as it launches.
        let launched = unsafe {
            (api.launch_cooperative)(k.function, bx, by, bz, tx, ty, tz, grid.shared, self.driver.stream(), params)
        };
        api.check("cuLaunchCooperativeKernel", launched).map_err(|e| {
            let mut e = self.driver.refused(e, launched, &grid, k);
            // CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE, which says no more.
            let threads = tx.saturating_mul(ty).saturating_mul(tz);
            if launched == 720
                && let Ok(most) = self.resident(index, threads, grid.shared)
            {
                let blocks = u128::from(bx) * u128::from(by) * u128::from(bz);
                e.message += &format!(": the grid has {blocks} blocks, and the GPU runs {most} of these at once");
            }
            e
        })
    }

    /// How many blocks of kernel `index` the GPU runs at once, of `threads` threads each and
    /// `shared` bytes of the launch's shared memory: as many as one of its SMs holds, by the
    /// registers, the shared memory and the threads each takes, times how many SMs it has. A
    /// cooperative launch of the kernel has at most this many. The kernel is let have `shared`
    /// bytes first, as for a launch, and `shared` past what a block has is an error that says
    /// so. So is an `index` of no kernel the module has.
    pub fn resident(&self, index: usize, threads: u32, shared: u32) -> Result<u32, Error> {
        let k = self.kernel("cuOccupancyMaxActiveBlocksPerMultiprocessor", index)?;
        let api = self.driver.current()?;
        self.let_have(k, &Grid::new(1, threads).with_shared(shared))?;
        let (threads, mut blocks) = (c_int::try_from(threads).unwrap_or(c_int::MAX), 0);
        // SAFETY: `k.function` is a kernel of the module, which is loaded; it writes the count to
        // `blocks`.
        let found = unsafe { (api.occupancy)(&mut blocks, k.function, threads, shared as usize) };
        api.check("cuOccupancyMaxActiveBlocksPerMultiprocessor", found)?;
        Ok(blocks as u32 * self.multiprocessors)
    }

    /// `Ok` if `memory` is of this module's GPU: memory of another is no memory of a kernel loaded on this one.
    pub(crate) fn reaches(&self, memory: &Driver) -> Result<(), Error> {
        let message = "the memory is on another GPU than the kernels are loaded on";
        self.driver.require_same_gpu(memory, "cuLaunchKernel", message)
    }

    /// Kernel `index`, as [`load`](Module::load) was given its name; or why the module has none, as the call `call`
    /// that asks for it says.
    fn kernel(&self, call: &'static str, index: usize) -> Result<&Kernel, Error> {
        self.kernels.get(index).ok_or_else(|| {
            Error::new(call, format!("there is no kernel {index} in a module of {}", self.kernels.len()))
        })
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
        // SAFETY: the module was loaded, and is unloaded once, here.
        self.driver.give_back(|api| unsafe { (api.unload_module)(self.module) });
    }
}

/// A module's memory on the GPU, one of its variables in global or constant memory, as
/// elements of `T`: memory its kernels name, which keeps what they write from launch to launch.
/// The driver allocated it as it loaded the module, holding what the PTX declares, zeros or
/// what it is initialized to, and frees it with the module; memory the module has no room for, [`Module::load`]
/// allocated, zeros, and frees with it. It is written through the module shared, as a launch writes it.
pub struct Global<'m, T: Plain> {
    module: &'m Module,
    ptr: u64,
    len: usize,
    of: PhantomData<T>,
}

impl<T: Plain> Global<'_, T> {
    /// Its address, for a kernel to be given; it points at something while the module lasts.
    pub fn ptr(&self) -> Ptr<T> {
        Ptr::new(self.ptr)
    }

    /// The address of its element `i`, for a kernel to be given the part from there on; `None`
    /// if `i` is past its end.
    pub fn at(&self, i: usize) -> Option<Ptr<T>> {
        Ptr::element(self.ptr, self.len, i)
    }

    /// How many elements it has.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies `data` to it, after the kernels launched before have run. It is an error if `data`
    /// has not as many elements: `cuMemcpyHtoD: a module's memory of 4 elements written with 2`.
    pub fn write(&self, data: &[T]) -> Result<(), Error> {
        same_len("cuMemcpyHtoD", "a module's memory", self.len, "written with", data.len())?;
        self.module.driver.write(self.ptr, data)
    }

    /// What it holds, after the kernels launched before have run.
    pub fn read(&self) -> Result<Vec<T>, Error> {
        let mut out = vec![T::default(); self.len];
        self.read_into(&mut out)?;
        Ok(out)
    }

    /// Copies what it holds to `out`, after the kernels launched before have run. It is an error
    /// if `out` has not as many elements: `cuMemcpyDtoH: a module's memory of 4 elements read
    /// into 2`.
    pub fn read_into(&self, out: &mut [T]) -> Result<(), Error> {
        same_len("cuMemcpyDtoH", "a module's memory", self.len, "read into", out.len())?;
        self.module.driver.read_into(self.ptr, out)
    }

    /// Sets every element to zero, on the current stream, after the work that stream was given before, and does not
    /// wait for it. The host's calls that reach it wait for the zeros, as they wait for a kernel.
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
    use std::cell::RefCell;
    use std::panic::{self, AssertUnwindSafe};
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

    /// A function of the fake driver that every lookup finds: it refuses.
    unsafe extern "C" fn refuse() -> CuResult {
        // CUDA_ERROR_UNKNOWN.
        999
    }

    /// The lookup of a fake driver that has every function but those of `lacks`.
    fn finds<'a>(lacks: &'a [&'a CStr]) -> impl Fn(&CStr) -> Result<*mut c_void, Error> + 'a {
        move |name| {
            if lacks.contains(&name) {
                Err(missing(name, "libcuda.so.1: undefined symbol"))
            } else {
                Ok(refuse as unsafe extern "C" fn() -> CuResult as *mut c_void)
            }
        }
    }

    /// A driver of a release before CUDA 11.4 is opened all the same: the table has the functions
    /// it has, and has for the four it lacks the error of the lookup, which a call that needs one
    /// returns. A driver that lacks any other function is not opened.
    #[test]
    fn a_driver_before_cuda_11_4_is_opened_and_one_that_lacks_any_other_function_is_not() -> Result {
        let lacks = |name: &str| format!("dlsym: the driver has no `{name}`: libcuda.so.1: undefined symbol");
        let newer =
            [c"cuMemPrefetchAsync_v2", c"cuMemAdvise_v2", c"cuEventElapsedTime_v2", c"cuGraphInstantiateWithFlags"];
        let api = Api::load(finds(&newer)).ok().ok_or("ok gave None")?;
        assert_eq!(api.prefetch.err().ok_or("err gave None")?.to_string(), lacks("cuMemPrefetchAsync_v2"));
        assert_eq!(api.advise.err().ok_or("err gave None")?.to_string(), lacks("cuMemAdvise_v2"));
        assert_eq!(api.elapsed_time.err().ok_or("err gave None")?.to_string(), lacks("cuEventElapsedTime_v2"));
        assert_eq!(api.instantiate.err().ok_or("err gave None")?.to_string(), lacks("cuGraphInstantiateWithFlags"));
        let api = Api::load(finds(&[])).ok().ok_or("ok gave None")?;
        assert!(
            api.prefetch.is_ok() && api.advise.is_ok() && api.elapsed_time.is_ok() && api.instantiate.is_ok(),
            "a driver that has them has them"
        );
        for name in [
            c"cuInit",
            c"cuMemRangeGetAttribute",
            c"cuMemGetInfo_v2",
            c"cuMemHostRegister_v2",
            c"cuMemHostUnregister",
            c"cuMemGetAllocationGranularity",
            c"cuMemAddressReserve",
            c"cuMemAddressFree",
            c"cuMemCreate",
            c"cuMemRelease",
            c"cuMemMap",
            c"cuMemSetAccess",
            c"cuMemUnmap",
            c"cuStreamDestroy_v2",
            c"cuEventCreate",
            c"cuEventDestroy_v2",
            c"cuEventRecord",
            c"cuEventQuery",
            c"cuEventSynchronize",
            c"cuStreamWaitEvent",
            c"cuMemcpyAsync",
            c"cuMemsetD8Async",
            c"cuStreamBeginCapture_v2",
            c"cuStreamEndCapture",
            c"cuGraphLaunch",
            c"cuGraphDestroy",
            c"cuGraphExecDestroy",
            c"cuGetErrorString",
        ] {
            let e = Api::load(finds(&[name])).err().ok_or("err gave None")?;
            assert_eq!(e.to_string(), lacks(&name.to_string_lossy()), "{name:?}");
        }
        Ok(())
    }

    /// A call asks the driver to switch only when it goes to another GPU than the last call
    /// did, and a context the driver would not make current is not taken for the current one.
    /// The driver here is a cell that holds the context it was last given, and counts the gives.
    #[test]
    fn a_context_is_made_current_only_when_another_is() -> Result {
        let (remembered, driver, gives) = (Cell::new(ptr::null_mut()), Cell::new(ptr::null_mut()), Cell::new(0));
        let (a, b) = (ptr::without_provenance_mut(1), ptr::without_provenance_mut(2));
        let give = |context| {
            driver.set(context);
            gives.set(gives.get() + 1);
            Ok(())
        };
        for (context, gives_so_far) in [(a, 1), (a, 1), (b, 2), (b, 2), (a, 3), (a, 3)] {
            enter(&remembered, context, || give(context))?;
            assert_eq!((driver.get(), gives.get()), (context, gives_so_far));
        }
        let refused = enter(&remembered, b, || Err(Error::new("cuCtxSetCurrent", "refused".into())));
        assert_eq!(refused.err().ok_or("refused gave Ok")?.message, "refused");
        // It tries again, for the context the driver last had is still `a`.
        enter(&remembered, b, || give(b))?;
        assert_eq!((driver.get(), gives.get()), (b, 4));
        Ok(())
    }

    /// What is of another GPU is refused by one check, which the launch, the copy and the time between two events each
    /// give their words: the GPU whose context is the same is this GPU, and any other is not, and the error is of the
    /// call, with the words it was given.
    #[test]
    fn what_is_of_another_gpu_is_refused_by_one_check_that_each_call_gives_its_words() -> Result {
        let on = |context| Driver { api: Api::refusing(), device: 0, context: ptr::without_provenance_mut(context) };
        assert_eq!(on(1).require_same_gpu(&on(1), "cuAnyCall", "any words"), Ok(()));
        let e = on(1).require_same_gpu(&on(2), "cuAnyCall", "any words").err().ok_or("the check gave Ok")?;
        assert_eq!((e.call, e.message.as_str()), ("cuAnyCall", "any words"));
        Ok(())
    }

    /// Memory is lent to a kernel loaded on the GPU the memory is on, a buffer, or a part of one, and to no other:
    /// the error says so, in GUIDE.md's words, before the kernel runs.
    #[test]
    fn memory_is_lent_only_to_a_kernel_loaded_on_its_own_gpu() -> Result {
        let (cuda, other) = (gpu_in(1)?, gpu_in(2)?);
        let module = kernel(&cuda)?;
        let (here, there) = (cuda.alloc::<u32>(4)?, other.alloc::<u32>(4)?);
        let said = "cuLaunchKernel: the memory is on another GPU than the kernels are loaded on";
        let e = there.lend(&module).err().ok_or("lend gave Ok")?;
        assert_eq!(e.to_string(), said, "a buffer of another GPU");
        let e = there.part(1..).ok_or("part gave None")?.lend(&module).err().ok_or("lend gave Ok")?;
        assert_eq!(e.to_string(), said, "a part of a buffer of another GPU");
        assert_eq!(here.lend(&module)?.1, 4, "a buffer of this GPU");
        assert_eq!(here.part(1..).ok_or("part gave None")?.lend(&module)?.1, 3, "a part of a buffer of this GPU");
        Ok(())
    }

    thread_local! {
        /// What the fake driver below was asked since the test last took it, in order, as words.
        static ASKED: RefCell<Vec<String>> = const { RefCell::new(vec![]) };
        /// Where the fake driver puts the next memory it allocates.
        static NEXT: Cell<u64> = const { Cell::new(0x7000_0000_0000) };
        /// The context the fake driver works in: the one it was last made to make current.
        static WORKING_IN: Cell<usize> = const { Cell::new(0) };
        /// How many streams the fake driver has made.
        static STREAMS_MADE: Cell<usize> = const { Cell::new(0) };
        /// How many events the fake driver has made.
        static EVENTS_MADE: Cell<usize> = const { Cell::new(0) };
        /// How many captures of this thread the fake driver has on: begun, and not ended.
        static CAPTURES_ON: Cell<usize> = const { Cell::new(0) };
        /// How many more calls that wait for the whole GPU the fake driver refuses, as a capture of another thread
        /// makes it do, before it takes one.
        pub(super) static ELSEWHERE: Cell<usize> = const { Cell::new(0) };
        /// How many captures the fake driver begins before it refuses to begin one.
        static BEGINS_LEFT: Cell<usize> = const { Cell::new(usize::MAX) };
        /// The context that the fake driver refuses to make current, or 0 where there is none.
        static UNSETTABLE: Cell<usize> = const { Cell::new(0) };
    }

    /// What the fake driver was asked since the last call, which empties it.
    pub(super) fn asked() -> Vec<String> {
        ASKED.take()
    }

    pub(super) fn ask(what: String) {
        ASKED.with_borrow_mut(|asked| asked.push(what));
    }

    /// The fake driver as it was before the test: it has made nothing, been asked nothing, and has no capture on.
    pub(super) fn start_over() {
        NEXT.set(0x7000_0000_0000);
        STREAMS_MADE.set(0);
        EVENTS_MADE.set(0);
        CAPTURES_ON.set(0);
        ELSEWHERE.set(0);
        BEGINS_LEFT.set(usize::MAX);
        UNSETTABLE.set(0);
        asked();
    }

    /// What the fake driver says of a call that gives memory, a module or a graph back, as the driver does: 900, which
    /// is `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, while a capture of this thread is on, and 0 else.
    pub(super) fn gives_back() -> CuResult {
        if CAPTURES_ON.get() > 0 { 900 } else { 0 }
    }

    /// What it says of a call that waits for the whole GPU: the same, and 900 too while `ELSEWHERE` says that a capture
    /// of another thread is on, which takes one of its refusals each time.
    pub(super) fn waits_for_the_gpu() -> CuResult {
        if gives_back() != 0 {
            return 900;
        }
        let elsewhere = ELSEWHERE.get();
        ELSEWHERE.set(elsewhere.saturating_sub(1));
        if elsewhere > 0 { 900 } else { 0 }
    }

    /// The fake driver's `cuDeviceGetAttribute`: every attribute is 1.
    unsafe extern "C" fn attribute(value: *mut c_int, _: c_int, _: c_int) -> CuResult {
        // SAFETY: the caller gives a place for the value.
        unsafe { *value = 1 };
        0
    }

    /// The fake driver's `cuCtxSetCurrent`: it works in the context from now on, but for the one `UNSETTABLE` says,
    /// which it refuses with `CUDA_ERROR_INVALID_CONTEXT`.
    pub(super) unsafe extern "C" fn set_current(context: Handle) -> CuResult {
        if context.addr() == UNSETTABLE.get() {
            return 201;
        }
        WORKING_IN.set(context.addr());
        0
    }

    /// The fake driver's `cuMemGetInfo`: the GPU of context 11 has 3 GiB free of 4, and that of 12 has 5 of 8. In any
    /// other context it is `CUDA_ERROR_INVALID_CONTEXT`.
    unsafe extern "C" fn memory_info(free: *mut usize, total: *mut usize) -> CuResult {
        let (have, all) = match WORKING_IN.get() {
            11 => (3 << 30, 4 << 30),
            12 => (5 << 30, 8 << 30),
            // CUDA_ERROR_INVALID_CONTEXT.
            _ => return 201,
        };
        // SAFETY: the caller gives a place for each.
        unsafe { (*free, *total) = (have, all) };
        0
    }

    /// The fake driver's `cuModuleLoadDataEx`: the module is 7.
    unsafe extern "C" fn load_module(
        module: *mut Handle,
        _: *const c_void,
        _: c_uint,
        _: *mut c_int,
        _: *mut *mut c_void,
    ) -> CuResult {
        // SAFETY: the caller gives a place for the module.
        unsafe { *module = ptr::without_provenance_mut(7) };
        0
    }

    /// The fake driver's `cuModuleUnload`: a capture of this thread refuses it.
    unsafe extern "C" fn unload_module(module: Handle) -> CuResult {
        ask(format!("unload {}", module.addr()));
        gives_back()
    }

    /// The fake driver's `cuModuleGetFunction`: each kernel is the function 5.
    unsafe extern "C" fn function(at: *mut Handle, _: Handle, _: *const c_char) -> CuResult {
        // SAFETY: the caller gives a place for the function.
        unsafe { *at = ptr::without_provenance_mut(5) };
        0
    }

    /// The fake driver's `cuFuncGetAttribute`: every attribute is 0.
    unsafe extern "C" fn function_attribute(value: *mut c_int, _: c_int, _: Handle) -> CuResult {
        // SAFETY: the caller gives a place for the value.
        unsafe { *value = 0 };
        0
    }

    /// The fake driver's `cuLaunchKernel`: it says the stream, and the context it works in.
    unsafe extern "C" fn launch(
        _: Handle,
        _: c_uint,
        _: c_uint,
        _: c_uint,
        _: c_uint,
        _: c_uint,
        _: c_uint,
        _: c_uint,
        stream: Handle,
        _: *mut *mut c_void,
        _: *mut *mut c_void,
    ) -> CuResult {
        ask(format!("launch on {:#x} in {}", stream.addr(), WORKING_IN.get()));
        0
    }

    /// The fake driver's `cuLaunchCooperativeKernel`: it says the same.
    unsafe extern "C" fn launch_cooperative(
        _: Handle,
        _: c_uint,
        _: c_uint,
        _: c_uint,
        _: c_uint,
        _: c_uint,
        _: c_uint,
        _: c_uint,
        stream: Handle,
        _: *mut *mut c_void,
    ) -> CuResult {
        ask(format!("launch cooperatively on {:#x} in {}", stream.addr(), WORKING_IN.get()));
        0
    }

    /// The fake driver's `cuStreamCreateWithPriority`: the streams are 0x100, 0x200 and so on, and it says the flags
    /// and the priority it is given, and the context it works in. In the context 13 it is `CUDA_ERROR_INVALID_CONTEXT`.
    pub(super) unsafe extern "C" fn create_stream(at: *mut Handle, flags: c_uint, priority: c_int) -> CuResult {
        if WORKING_IN.get() == 13 {
            return 201;
        }
        STREAMS_MADE.set(STREAMS_MADE.get() + 1);
        let stream = STREAMS_MADE.get() * 0x100;
        ask(format!("create {stream:#x} with flags {flags} at {priority} in {}", WORKING_IN.get()));
        // SAFETY: the caller gives a place for the stream.
        unsafe { *at = ptr::without_provenance_mut(stream) };
        0
    }

    /// The fake driver's `cuCtxGetStreamPriorityRange`: the priorities of the fake GPU are 0, the least, to -5, the
    /// greatest, and it writes those it is given a place for. It says the context it works in. In the context 13 it is
    /// `CUDA_ERROR_INVALID_CONTEXT`.
    unsafe extern "C" fn priorities(least: *mut c_int, greatest: *mut c_int) -> CuResult {
        if WORKING_IN.get() == 13 {
            return 201;
        }
        ask(format!("ask the priorities in {}", WORKING_IN.get()));
        // SAFETY: the caller gives a place for each value it asks, and a null pointer for the others.
        unsafe {
            if let Some(least) = least.as_mut() {
                *least = 0;
            }
            if let Some(greatest) = greatest.as_mut() {
                *greatest = -5;
            }
        }
        0
    }

    /// The fake driver's `cuStreamDestroy`: it says the stream, and the context it works in.
    pub(super) unsafe extern "C" fn destroy_stream(stream: Handle) -> CuResult {
        ask(format!("destroy {:#x} in {}", stream.addr(), WORKING_IN.get()));
        0
    }

    /// The fake driver's `cuStreamSynchronize`: it says the stream, and the context it works in. In the context 14 the
    /// work of every stream has failed, with `CUDA_ERROR_ILLEGAL_ADDRESS`.
    unsafe extern "C" fn sync_stream(stream: Handle) -> CuResult {
        ask(format!("sync {:#x} in {}", stream.addr(), WORKING_IN.get()));
        if WORKING_IN.get() == 14 { 700 } else { 0 }
    }

    /// The fake driver's `cuEventCreate`: the events are 0x1000, 0x2000 and so on, and it says the flags it is given,
    /// and the context it works in. In the context 13 it is `CUDA_ERROR_INVALID_CONTEXT`.
    unsafe extern "C" fn create_event(at: *mut Handle, flags: c_uint) -> CuResult {
        if WORKING_IN.get() == 13 {
            return 201;
        }
        EVENTS_MADE.set(EVENTS_MADE.get() + 1);
        let event = EVENTS_MADE.get() * 0x1000;
        ask(format!("create event {event:#x} with flags {flags} in {}", WORKING_IN.get()));
        // SAFETY: the caller gives a place for the event.
        unsafe { *at = ptr::without_provenance_mut(event) };
        0
    }

    /// The fake driver's `cuEventDestroy`: it says the event, and the context it works in.
    unsafe extern "C" fn destroy_event(event: Handle) -> CuResult {
        ask(format!("destroy event {:#x} in {}", event.addr(), WORKING_IN.get()));
        0
    }

    /// The fake driver's `cuEventRecord`: it says the event, the stream, and the context it works in. In the context 14
    /// the work has failed, with `CUDA_ERROR_ILLEGAL_ADDRESS`.
    unsafe extern "C" fn record_event(event: Handle, stream: Handle) -> CuResult {
        ask(format!("record {:#x} on {:#x} in {}", event.addr(), stream.addr(), WORKING_IN.get()));
        if WORKING_IN.get() == 14 { 700 } else { 0 }
    }

    /// The fake driver's `cuEventQuery`: it says the event, and the context it works in. In the context 14 the work
    /// has failed, with `CUDA_ERROR_ILLEGAL_ADDRESS`, and in the context 15 it is not done, `CUDA_ERROR_NOT_READY`.
    unsafe extern "C" fn query_event(event: Handle) -> CuResult {
        ask(format!("query {:#x} in {}", event.addr(), WORKING_IN.get()));
        match WORKING_IN.get() {
            14 => 700,
            15 => 600,
            _ => 0,
        }
    }

    /// The fake driver's `cuEventSynchronize`: it says the event, and the context it works in. In the context 14 the
    /// work has failed, with `CUDA_ERROR_ILLEGAL_ADDRESS`.
    unsafe extern "C" fn sync_event(event: Handle) -> CuResult {
        ask(format!("sync event {:#x} in {}", event.addr(), WORKING_IN.get()));
        if WORKING_IN.get() == 14 { 700 } else { 0 }
    }

    /// The fake driver's `cuStreamWaitEvent`: it says the stream, the event, the flags, and the context it works in. In
    /// the context 14 the work has failed, with `CUDA_ERROR_ILLEGAL_ADDRESS`.
    unsafe extern "C" fn wait_event(stream: Handle, event: Handle, flags: c_uint) -> CuResult {
        ask(format!("{:#x} waits for {:#x} with flags {flags} in {}", stream.addr(), event.addr(), WORKING_IN.get()));
        if WORKING_IN.get() == 14 { 700 } else { 0 }
    }

    /// The fake driver's `cuEventElapsedTime_v2`: it says the two events and the context it works in, and the time is
    /// the events' difference over 0x1000, in milliseconds: from the event 0x1000 to 0x3000 is 2 ms, and from 0x3000 to
    /// 0x1000 is below zero. In the context 15 an event is not done, `CUDA_ERROR_NOT_READY`, and in the context 16 one
    /// was never recorded, `CUDA_ERROR_INVALID_HANDLE`.
    unsafe extern "C" fn elapsed_time(millis: *mut f32, start: Handle, end: Handle) -> CuResult {
        ask(format!("time from {:#x} to {:#x} in {}", start.addr(), end.addr(), WORKING_IN.get()));
        match WORKING_IN.get() {
            15 => return 600,
            16 => return 400,
            _ => {}
        }
        // SAFETY: the caller gives a place for the milliseconds.
        unsafe { *millis = (end.addr() as f32 - start.addr() as f32) / 4096.0 };
        0
    }

    /// The fake driver's `cuStreamBeginCapture`: it says the stream, the mode, and the context it works in, and counts
    /// the capture as on. In the context 14 the stream is capturing already, and the call is refused with
    /// `CUDA_ERROR_ILLEGAL_ADDRESS`; and it is refused after as many captures as `BEGINS_LEFT` says.
    pub(super) unsafe extern "C" fn begin_capture(stream: Handle, mode: c_int) -> CuResult {
        ask(format!("begin capture on {:#x} with mode {mode} in {}", stream.addr(), WORKING_IN.get()));
        if WORKING_IN.get() == 14 || BEGINS_LEFT.get() == 0 {
            return 700;
        }
        BEGINS_LEFT.set(BEGINS_LEFT.get() - 1);
        CAPTURES_ON.set(CAPTURES_ON.get() + 1);
        0
    }

    /// The fake driver's `cuStreamEndCapture`: it says the stream and the context it works in, counts the capture as
    /// over, and the graph is 0x5000. In the context 15 the capture was invalidated, and the call gives no graph, and
    /// `CUDA_ERROR_ILLEGAL_ADDRESS`.
    pub(super) unsafe extern "C" fn end_capture(stream: Handle, graph: *mut Handle) -> CuResult {
        ask(format!("end capture on {:#x} in {}", stream.addr(), WORKING_IN.get()));
        CAPTURES_ON.set(CAPTURES_ON.get().saturating_sub(1));
        let invalidated = WORKING_IN.get() == 15;
        // SAFETY: the caller gives a place for the graph.
        unsafe { *graph = if invalidated { ptr::null_mut() } else { ptr::without_provenance_mut(0x5000) } };
        if invalidated { 700 } else { 0 }
    }

    /// The fake driver's `cuGraphInstantiateWithFlags`: it says the graph, the flags and the context it works in, and
    /// the graph made of it is 0x6000. In the context 16 it is refused, with `CUDA_ERROR_ILLEGAL_ADDRESS`.
    unsafe extern "C" fn instantiate(exec: *mut Handle, graph: Handle, flags: u64) -> CuResult {
        ask(format!("instantiate {:#x} with flags {flags} in {}", graph.addr(), WORKING_IN.get()));
        if WORKING_IN.get() == 16 {
            return 700;
        }
        // SAFETY: the caller gives a place for the graph.
        unsafe { *exec = ptr::without_provenance_mut(0x6000) };
        0
    }

    /// The fake driver's `cuGraphLaunch`: it says the graph, the stream, and the context it works in. In the context
    /// 14 the work has failed, with `CUDA_ERROR_ILLEGAL_ADDRESS`.
    unsafe extern "C" fn launch_graph(exec: Handle, stream: Handle) -> CuResult {
        ask(format!("launch graph {:#x} on {:#x} in {}", exec.addr(), stream.addr(), WORKING_IN.get()));
        if WORKING_IN.get() == 14 { 700 } else { 0 }
    }

    /// The fake driver's `cuGraphDestroy`: it says the graph, and the context it works in.
    pub(super) unsafe extern "C" fn destroy_graph(graph: Handle) -> CuResult {
        ask(format!("destroy graph {:#x} in {}", graph.addr(), WORKING_IN.get()));
        0
    }

    /// The fake driver's `cuGraphExecDestroy`: it says the graph, and the context it works in. A capture of this thread
    /// refuses it.
    unsafe extern "C" fn destroy_graph_exec(exec: Handle) -> CuResult {
        ask(format!("destroy executable graph {:#x} in {}", exec.addr(), WORKING_IN.get()));
        gives_back()
    }

    /// The fake driver's `cuCtxSynchronize`, for a test that says whether a call waits for the GPU: it says that it is
    /// asked. A capture refuses it.
    unsafe extern "C" fn synchronize() -> CuResult {
        ask("cuCtxSynchronize".to_string());
        waits_for_the_gpu()
    }

    /// The fake driver's `cuModuleGetGlobal`: the module has `small`, 16 bytes at 0x1000, and two `.u64` of constant
    /// memory, `slot` at 0x2000 and `other` at 0x3000.
    unsafe extern "C" fn module_global(at: *mut u64, bytes: *mut usize, _: Handle, name: *const c_char) -> CuResult {
        // SAFETY: the caller gives a C string, and a place for the address and the bytes.
        let (name, at, bytes) = unsafe { (CStr::from_ptr(name), &mut *at, &mut *bytes) };
        let (addr, size) = match name.to_bytes() {
            b"small" => (0x1000, 16),
            b"slot" => (0x2000, 8),
            b"other" => (0x3000, 8),
            // CUDA_ERROR_NOT_FOUND.
            _ => return 500,
        };
        (*at, *bytes) = (addr, size);
        0
    }

    /// The fake driver's `cuMemAlloc`: a GPU of 8 GiB, which hands out addresses one after the other, and refuses more
    /// than 8 GiB at once, as the driver refuses what it has not the bytes for.
    unsafe extern "C" fn alloc(at: *mut u64, bytes: usize) -> CuResult {
        if bytes > 8 << 30 {
            // CUDA_ERROR_OUT_OF_MEMORY.
            return 2;
        }
        ask(format!("alloc {bytes}"));
        // SAFETY: the caller gives a place for the address.
        unsafe { *at = NEXT.get() };
        NEXT.set(NEXT.get() + bytes as u64);
        0
    }

    /// The fake driver's `cuMemsetD8Async`: it says the stream, when it is not the null stream.
    unsafe extern "C" fn memset(at: u64, value: u8, bytes: usize, stream: Handle) -> CuResult {
        let on = if stream.is_null() { String::new() } else { format!(" on {:#x}", stream.addr()) };
        ask(format!("set {at:#x} to {value}, {bytes} bytes{on}"));
        0
    }

    /// The fake driver's `cuMemAllocManaged`: it hands out addresses as `cuMemAlloc` does.
    unsafe extern "C" fn alloc_managed(at: *mut u64, bytes: usize, _: c_uint) -> CuResult {
        ask(format!("alloc managed {bytes}"));
        // SAFETY: the caller gives a place for the address.
        unsafe { *at = NEXT.get() };
        NEXT.set(NEXT.get() + bytes as u64);
        0
    }

    /// The fake driver's `cuDeviceGetCount`: one GPU.
    unsafe extern "C" fn device_count(count: *mut c_int) -> CuResult {
        // SAFETY: the caller gives a place for the count.
        unsafe { *count = 1 };
        0
    }

    /// The fake driver's `cuDeviceGet`: GPU `ordinal` is the device `ordinal`.
    unsafe extern "C" fn device_get(device: *mut c_int, ordinal: c_int) -> CuResult {
        // SAFETY: the caller gives a place for the device.
        unsafe { *device = ordinal };
        0
    }

    /// The fake driver's `cuMemcpyAsync`: it says the addresses, the bytes, the stream and the context it works in. In
    /// the context 14 the copy is refused, with `CUDA_ERROR_ILLEGAL_ADDRESS`.
    unsafe extern "C" fn copy_async(to: u64, from: u64, bytes: usize, stream: Handle) -> CuResult {
        ask(format!("copy {bytes} bytes from {from:#x} to {to:#x} on {:#x} in {}", stream.addr(), WORKING_IN.get()));
        if WORKING_IN.get() == 14 { 700 } else { 0 }
    }

    /// The fake driver's `cuMemcpyDtoH`: it writes 7 to each byte it is given.
    unsafe extern "C" fn to_host(out: *mut c_void, at: u64, bytes: usize) -> CuResult {
        // SAFETY: the caller gives `bytes` bytes at `out`.
        unsafe { out.cast::<u8>().write_bytes(7, bytes) };
        ask(format!("read {bytes} bytes at {at:#x}"));
        0
    }

    /// The fake driver's `cuMemcpyHtoD`: it says the `u64` it is given.
    unsafe extern "C" fn to_device(at: u64, from: *const c_void, bytes: usize) -> CuResult {
        // SAFETY: the caller gives `bytes` bytes at `from`, which are a `u64`.
        let value = unsafe { from.cast::<u64>().read_unaligned() };
        ask(format!("write {value:#x} at {at:#x}, {bytes} bytes"));
        0
    }

    /// The fake driver's `cuMemFree`: a capture of this thread refuses it.
    unsafe extern "C" fn free(at: u64) -> CuResult {
        ask(format!("free {at:#x}"));
        gives_back()
    }

    /// The fake driver's `cuMemFree` of a driver that fails it with `CUDA_ERROR_ILLEGAL_ADDRESS`, as it does after an
    /// error of the work.
    unsafe extern "C" fn free_fails(at: u64) -> CuResult {
        ask(format!("free {at:#x}"));
        700
    }

    /// The fake driver's `cuGetErrorName`, and `cuGetErrorString`, which know the seven errors it has.
    unsafe extern "C" fn error_name(result: CuResult, name: *mut *const c_char) -> CuResult {
        let known = match result {
            2 => c"CUDA_ERROR_OUT_OF_MEMORY".as_ptr(),
            201 => c"CUDA_ERROR_INVALID_CONTEXT".as_ptr(),
            400 => c"CUDA_ERROR_INVALID_HANDLE".as_ptr(),
            600 => c"CUDA_ERROR_NOT_READY".as_ptr(),
            700 => c"CUDA_ERROR_ILLEGAL_ADDRESS".as_ptr(),
            900 => c"CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED".as_ptr(),
            901 => c"CUDA_ERROR_STREAM_CAPTURE_INVALIDATED".as_ptr(),
            _ => ptr::null(),
        };
        // SAFETY: the caller gives a place for the string.
        unsafe { *name = known };
        0
    }

    unsafe extern "C" fn error_string(result: CuResult, meaning: *mut *const c_char) -> CuResult {
        let known = match result {
            2 => c"out of memory".as_ptr(),
            201 => c"invalid device context".as_ptr(),
            400 => c"invalid resource handle".as_ptr(),
            600 => c"device not ready".as_ptr(),
            700 => c"an illegal memory access was encountered".as_ptr(),
            900 => c"operation not permitted when stream is capturing".as_ptr(),
            901 => c"operation failed due to a previous error during capture".as_ptr(),
            _ => ptr::null(),
        };
        // SAFETY: the caller gives a place for the string.
        unsafe { *meaning = known };
        0
    }

    /// A GPU of the fake driver, whose primary context is `context`, which has the calls a module's memory makes, its
    /// launches', the streams' and the events', and `cuMemGetInfo`, and refuses the rest.
    fn gpu_in(context: usize) -> Result<Cuda> {
        gpu_replacing(context, |_| {})
    }

    /// `gpu_in(context)`, with the functions of its driver that `replace` changes.
    fn gpu_replacing(context: usize, replace: impl FnOnce(&mut Api)) -> Result<Cuda> {
        start_over();
        let mut api = Api::refusing();
        api.device_count = device_count;
        api.device_get = device_get;
        api.attribute = attribute;
        api.set_current = set_current;
        api.load_module = load_module;
        api.unload_module = unload_module;
        api.function = function;
        api.function_attribute = function_attribute;
        api.global = module_global;
        api.launch = launch;
        api.launch_cooperative = launch_cooperative;
        api.create_stream = create_stream;
        api.priorities = priorities;
        api.destroy_stream = destroy_stream;
        api.sync_stream = sync_stream;
        api.create_event = create_event;
        api.destroy_event = destroy_event;
        api.record_event = record_event;
        api.query_event = query_event;
        api.sync_event = sync_event;
        api.wait_event = wait_event;
        api.begin_capture = begin_capture;
        api.end_capture = end_capture;
        api.instantiate = Ok(instantiate);
        api.launch_graph = launch_graph;
        api.destroy_graph = destroy_graph;
        api.destroy_graph_exec = destroy_graph_exec;
        api.elapsed_time = Ok(elapsed_time);
        api.alloc = alloc;
        api.memset = memset;
        api.copy_async = copy_async;
        api.memory_info = memory_info;
        api.to_device = to_device;
        api.to_host = to_host;
        api.free = free;
        api.error_name = error_name;
        api.error_string = error_string;
        replace(&mut api);
        let driver = Rc::new(Driver { api, device: 0, context: ptr::without_provenance_mut(context) });
        Ok(Cuda::of(driver, String::new())?)
    }

    /// A GPU of the fake driver whose context is 1, which has no memory to say.
    fn gpu() -> Result<Cuda> {
        gpu_in(1)
    }

    /// Memory the module has no room for is allocated as it loads, 6 GiB of it, with zeros in it; its address is
    /// written where the kernels read it, the variable the module names; the host reaches the memory at that address,
    /// whole; and the module frees it when it is dropped. The module's other memory is where the module has it.
    #[test]
    fn memory_the_module_has_no_room_for_is_allocated_as_it_loads_and_freed_with_it() -> Result {
        let cuda = gpu()?;
        let module = Module::load(&cuda, "", &[], &[("small", None), ("slot", Some((6 << 30, 1)))])?;
        let at = 0x7000_0000_0000u64;
        let want = [
            format!("alloc {}", 6u64 << 30),
            format!("set {at:#x} to 0, {} bytes", 6u64 << 30),
            format!("write {at:#x} at 0x2000, 8 bytes"),
        ];
        assert_eq!(asked(), want);
        let (small, far) = (module.global::<u8>(0)?, module.global::<u8>(1)?);
        assert_eq!((small.ptr().addr(), small.len()), (0x1000, 16));
        assert_eq!((far.ptr().addr(), far.len()), (at, 6 << 30));
        assert_eq!(module.global::<u64>(1)?.len(), 6 << 27);
        assert!(asked().is_empty(), "nothing is asked while the module lasts");
        drop(module);
        assert_eq!(asked(), ["unload 7".to_string(), format!("free {at:#x}")]);
        Ok(())
    }

    /// A GPU that has not the memory is an error: the driver's, with the variable it was for and its bytes. What was
    /// allocated before is freed with the module that is not made.
    #[test]
    fn a_gpu_that_has_not_the_memory_is_an_error_that_names_the_variable() -> Result {
        let cuda = gpu()?;
        let e = Module::load(&cuda, "", &[], &[("slot", Some((1 << 30, 1))), ("other", Some((16 << 30, 1)))])
            .err()
            .ok_or("load gave Ok")?;
        let said = "cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY (2): out of memory, for `other`, 17179869184 bytes";
        assert_eq!(e.to_string(), said);
        let first = 0x7000_0000_0000u64;
        let want = [
            format!("alloc {}", 1u64 << 30),
            format!("set {first:#x} to 0, {} bytes", 1u64 << 30),
            format!("write {first:#x} at 0x2000, 8 bytes"),
            "unload 7".to_string(),
            format!("free {first:#x}"),
        ];
        assert_eq!(asked(), want);
        Ok(())
    }

    /// Memory the module has no room for is at a multiple of its alignment: for 4096, the driver is asked for 4095
    /// bytes more, and the address it gives is rounded up. The module frees what it was given.
    #[test]
    fn memory_the_module_has_no_room_for_is_at_its_alignment() -> Result {
        let cuda = gpu()?;
        let module = Module::load(&cuda, "", &[], &[("slot", Some((256, 1))), ("other", Some((4096, 4096)))])?;
        let (first, given, at) = (0x7000_0000_0000u64, 0x7000_0000_0100u64, 0x7000_0000_1000u64);
        let want = [
            "alloc 256".to_string(),
            format!("set {first:#x} to 0, 256 bytes"),
            format!("write {first:#x} at 0x2000, 8 bytes"),
            "alloc 8191".to_string(),
            format!("set {given:#x} to 0, 8191 bytes"),
            format!("write {at:#x} at 0x3000, 8 bytes"),
        ];
        assert_eq!(asked(), want);
        let other = module.global::<u8>(1)?;
        assert_eq!((other.ptr().addr(), other.len()), (at, 4096));
        drop(module);
        assert_eq!(asked(), ["unload 7".to_string(), format!("free {first:#x}"), format!("free {given:#x}")]);
        Ok(())
    }

    /// A GPU says its own memory, as the driver says it of the context that is current, and the call is made in the
    /// context of the GPU it is made on, whichever GPU the call before it went to. The free bytes come first, and the
    /// total second, whole: the second GPU has more than a 32-bit count holds.
    #[test]
    fn each_gpu_says_its_own_memory_whichever_gpu_the_call_before_went_to() -> Result {
        let (a, b) = (gpu_in(11)?, gpu_in(12)?);
        let (of_a, of_b) = ((3 << 30, 4 << 30), (5 << 30, 8 << 30));
        for _ in 0..2 {
            assert_eq!(a.memory()?, of_a);
            assert_eq!(b.memory()?, of_b);
            assert_eq!(b.memory()?, of_b);
            assert_eq!(a.memory()?, of_a);
        }
        Ok(())
    }

    /// What the driver refuses is the error it gave, which names the call `cuMemGetInfo`, in the words of the driver.
    #[test]
    fn a_refusal_of_the_driver_is_the_error_of_memory() -> Result {
        let e = gpu_in(13)?.memory().err().ok_or("memory gave Ok")?;
        assert_eq!(e.to_string(), "cuMemGetInfo: CUDA_ERROR_INVALID_CONTEXT (201): invalid device context");
        Ok(())
    }

    /// A module of one kernel, `k`, on `cuda`, with what it asked of the driver forgotten.
    fn kernel(cuda: &Cuda) -> Result<Module> {
        let module = Module::load(cuda, "", &["k"], &[])?;
        asked();
        Ok(module)
    }

    /// Kernel 0 of `module` launched over one thread, and what that asked of the driver.
    fn launched(module: &Module) -> Result<Vec<String>> {
        // SAFETY: the fake driver reads no argument, and runs no kernel.
        unsafe { module.launch(0, Grid::new(1, 1), &[]) }?;
        Ok(asked())
    }

    /// `work`, with `stream` the current stream of its GPU.
    fn running<R>(stream: &Stream, work: impl FnOnce() -> R) -> R {
        // SAFETY: the fake driver runs no work, so no stream's work reaches memory.
        unsafe { stream.run(work) }
    }

    /// A stream is made as the driver is asked for a blocking one, with no flags. It is waited for as the driver is
    /// asked, and destroyed once, when it is dropped, even after the `Cuda` that made it.
    #[test]
    fn a_stream_is_made_blocking_waited_for_and_destroyed_as_the_driver_is_asked() -> Result {
        let cuda = gpu()?;
        let (first, second) = (cuda.stream()?, cuda.stream()?);
        assert_eq!(asked(), ["create 0x100 with flags 0 at 0 in 1", "create 0x200 with flags 0 at 0 in 1"]);
        first.sync()?;
        assert_eq!(asked(), ["sync 0x100 in 1"]);
        drop(cuda);
        drop(first);
        assert_eq!(asked(), ["destroy 0x100 in 1"]);
        drop(second);
        assert_eq!(asked(), ["destroy 0x200 in 1"]);
        Ok(())
    }

    /// A stream apart is made as the driver is asked for a stream with `CU_STREAM_NON_BLOCKING`, 1, at the priority 0,
    /// and an urgent one at the greatest priority that the driver says the GPU has: -5 on the fake GPU. A refusal of
    /// either call is the error of making it.
    #[test]
    fn a_stream_apart_is_made_non_blocking_and_an_urgent_one_at_the_greatest_priority() -> Result {
        let cuda = gpu()?;
        let streams = (cuda.stream_apart(false)?, cuda.stream_apart(true)?);
        let made =
            ["create 0x100 with flags 1 at 0 in 1", "ask the priorities in 1", "create 0x200 with flags 1 at -5 in 1"];
        assert_eq!(asked(), made);
        drop(streams);
        let said = "cuCtxGetStreamPriorityRange: CUDA_ERROR_INVALID_CONTEXT (201): invalid device context";
        assert_eq!(gpu_in(13)?.stream_apart(true).err().ok_or("stream_apart gave Ok")?.to_string(), said);
        let said = "cuStreamCreateWithPriority: CUDA_ERROR_INVALID_CONTEXT (201): invalid device context";
        assert_eq!(gpu_in(13)?.stream_apart(false).err().ok_or("stream_apart gave Ok")?.to_string(), said);
        Ok(())
    }

    /// What the driver refuses is the error of the call it refused, in the driver's words: a stream is not made where
    /// it refuses `cuStreamCreateWithPriority`, and the work that failed on a stream is the error of the `sync` that
    /// waits for it.
    #[test]
    fn a_refusal_of_the_driver_is_the_error_of_making_a_stream_and_of_waiting_for_one() -> Result {
        let e = gpu_in(13)?.stream().err().ok_or("stream gave Ok")?;
        assert_eq!(
            e.to_string(),
            "cuStreamCreateWithPriority: CUDA_ERROR_INVALID_CONTEXT (201): invalid device context"
        );
        gpu()?.stream()?.sync()?;
        let e = gpu_in(14)?.stream()?.sync().err().ok_or("sync gave Ok")?;
        let said = "cuStreamSynchronize: CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered";
        assert_eq!(e.to_string(), said);
        Ok(())
    }

    /// A launch goes to the null stream, the driver's 0, until a `run` makes a stream current, and then to the stream;
    /// and it asks the driver for the launch alone, whichever. Afterwards it goes to the null stream again.
    #[test]
    fn a_launch_goes_to_the_null_stream_and_in_a_run_to_the_stream_and_asks_the_driver_for_the_launch_alone() -> Result
    {
        let cuda = gpu()?;
        let (module, stream) = (kernel(&cuda)?, cuda.stream()?);
        asked();
        assert_eq!(launched(&module)?, ["launch on 0x0 in 1"]);
        assert_eq!(running(&stream, || launched(&module))?, ["launch on 0x100 in 1"]);
        assert_eq!(launched(&module)?, ["launch on 0x0 in 1"]);
        assert_eq!(running(&stream, || 7), 7, "run gives back what the closure gives");
        Ok(())
    }

    /// A cooperative launch goes to the current stream too.
    #[test]
    fn a_cooperative_launch_goes_to_the_current_stream_too() -> Result {
        let cuda = gpu()?;
        let (module, stream) = (kernel(&cuda)?, cuda.stream()?);
        asked();
        let cooperatively = || -> Result<Vec<String>> {
            // SAFETY: as in `launched`.
            unsafe { module.launch_cooperative(0, Grid::new(1, 1), &[]) }?;
            Ok(asked())
        };
        assert_eq!(cooperatively()?, ["launch cooperatively on 0x0 in 1"]);
        assert_eq!(running(&stream, cooperatively)?, ["launch cooperatively on 0x100 in 1"]);
        Ok(())
    }

    /// A `run` in a `run` makes its stream current, and gives the stream before it back when it ends, whether it
    /// returns or panics; the same stream in itself too.
    #[test]
    fn a_run_in_a_run_gives_back_the_stream_before_it_when_it_returns_and_when_it_panics() -> Result {
        let cuda = gpu()?;
        let (module, a, b) = (kernel(&cuda)?, cuda.stream()?, cuda.stream()?);
        asked();
        let mut seen = vec![];
        running(&a, || -> Result {
            seen.push(launched(&module)?);
            running(&b, || -> Result {
                seen.push(launched(&module)?);
                Ok(())
            })?;
            seen.push(launched(&module)?);
            running(&a, || -> Result {
                running(&a, || ());
                seen.push(launched(&module)?);
                Ok(())
            })?;
            seen.push(launched(&module)?);
            let panicked =
                panic::catch_unwind(AssertUnwindSafe(|| running(&b, || panic::resume_unwind(Box::new("a panic")))));
            assert!(panicked.is_err(), "the closure did not panic");
            seen.push(launched(&module)?);
            Ok(())
        })?;
        seen.push(launched(&module)?);
        let to = |stream: usize| vec![format!("launch on {stream:#x} in 1")];
        assert_eq!(seen, [to(0x100), to(0x200), to(0x100), to(0x100), to(0x100), to(0x100), to(0)]);
        Ok(())
    }

    /// A stream is made, waited for and destroyed in the context of its own GPU, whichever GPU the call before it went
    /// to. It is current for its own GPU and its own thread alone: a launch on another GPU goes to the null stream of
    /// that one, a stream of each GPU current is current for each, and a launch of another thread goes to the null
    /// stream.
    #[test]
    fn a_stream_is_of_its_gpu_and_current_for_its_gpu_and_its_thread_alone() -> Result {
        let (first, second) = (gpu_in(11)?, gpu_in(12)?);
        let (on_first, on_second) = (kernel(&first)?, kernel(&second)?);
        let (a, b) = (first.stream()?, second.stream()?);
        assert_eq!(asked(), ["create 0x100 with flags 0 at 0 in 11", "create 0x200 with flags 0 at 0 in 12"]);
        let (launch_a, launch_b) = running(&a, || (launched(&on_first), launched(&on_second)));
        assert_eq!(launch_a?, ["launch on 0x100 in 11"]);
        assert_eq!(launch_b?, ["launch on 0x0 in 12"]);
        let (launch_a, launch_b) = running(&a, || running(&b, || (launched(&on_first), launched(&on_second))));
        assert_eq!(launch_a?, ["launch on 0x100 in 11"]);
        assert_eq!(launch_b?, ["launch on 0x200 in 12"]);
        let elsewhere = running(&a, || {
            std::thread::scope(|scope| {
                scope.spawn(|| -> Result<Vec<String>> { launched(&kernel(&gpu_in(11)?)?) }).join()
            })
        });
        assert_eq!(elsewhere.map_err(|_| "the thread panicked")??, ["launch on 0x0 in 11"]);
        a.sync()?;
        b.sync()?;
        a.sync()?;
        assert_eq!(asked(), ["sync 0x100 in 11", "sync 0x200 in 12", "sync 0x100 in 11"]);
        drop(a);
        drop(b);
        assert_eq!(asked(), ["destroy 0x100 in 11", "destroy 0x200 in 12"]);
        Ok(())
    }

    /// A launch through another `Cuda` of the GPU, the same context, goes to the stream of the GPU: the stream is the
    /// context's, and not that of the `Cuda` that made it.
    #[test]
    fn a_launch_through_another_cuda_of_the_gpu_goes_to_the_stream_of_the_gpu() -> Result {
        let (cuda, other) = (gpu()?, gpu()?);
        let (stream, module) = (cuda.stream()?, kernel(&other)?);
        assert_eq!(running(&stream, || launched(&module))?, ["launch on 0x100 in 1"]);
        Ok(())
    }

    /// An event is made as the driver is asked: one made not to time with the flag that says so, and one made to time
    /// with none. Dropping each destroys it once, even after the `Cuda` that made it. Where the driver refuses to make
    /// one, the error is its own.
    #[test]
    fn an_event_is_made_to_time_or_not_as_the_driver_is_asked_and_destroyed_once_when_it_is_dropped() -> Result {
        let cuda = gpu()?;
        let (plain, timed) = (cuda.event()?, cuda.timed_event()?);
        assert_eq!(asked(), ["create event 0x1000 with flags 2 in 1", "create event 0x2000 with flags 0 in 1"]);
        drop(cuda);
        drop(plain);
        assert_eq!(asked(), ["destroy event 0x1000 in 1"]);
        drop(timed);
        assert_eq!(asked(), ["destroy event 0x2000 in 1"]);
        let refused = "cuEventCreate: CUDA_ERROR_INVALID_CONTEXT (201): invalid device context";
        let cuda = gpu_in(13)?;
        assert_eq!(cuda.event().err().ok_or("event gave Ok")?.to_string(), refused);
        assert_eq!(cuda.timed_event().err().ok_or("timed_event gave Ok")?.to_string(), refused);
        Ok(())
    }

    /// A record and a wait go to the current stream of the event's GPU: the null stream, until a `run` makes another
    /// current for the GPU, and then that stream; a stream that is current for another GPU is not the one.
    #[test]
    fn a_record_and_a_wait_go_to_the_current_stream_of_the_events_gpu() -> Result {
        let (first, second) = (gpu_in(11)?, gpu_in(12)?);
        let (event, a, b) = (first.event()?, first.stream()?, second.stream()?);
        asked();
        event.record()?;
        event.wait()?;
        assert_eq!(asked(), ["record 0x1000 on 0x0 in 11", "0x0 waits for 0x1000 with flags 0 in 11"]);
        running(&a, || -> Result {
            event.record()?;
            event.wait()?;
            // The stream of the other GPU is current for that GPU alone.
            running(&b, || event.record())?;
            Ok(())
        })?;
        let on_a = |what: &str| format!("{what} in 11");
        let want = [
            on_a("record 0x1000 on 0x100"),
            on_a("0x100 waits for 0x1000 with flags 0"),
            on_a("record 0x1000 on 0x100"),
        ];
        assert_eq!(asked(), want);
        running(&b, || event.wait())?;
        assert_eq!(asked(), ["0x0 waits for 0x1000 with flags 0 in 11"]);
        Ok(())
    }

    /// `done` is `cuEventQuery`: true where the driver says the work is done, false where it says `NOT_READY`, and an
    /// error where it says any other, as the work that failed does. `sync`, `record` and `wait` say the same failure,
    /// each by the name of its own call.
    #[test]
    fn an_event_is_done_or_not_as_the_driver_says_and_an_error_where_the_work_failed() -> Result {
        let cuda = gpu()?;
        let event = cuda.event()?;
        asked();
        assert!(event.done()?);
        event.sync()?;
        assert_eq!(asked(), ["query 0x1000 in 1", "sync event 0x1000 in 1"]);
        assert!(!gpu_in(15)?.event()?.done()?, "the driver said the work is not ready");
        let failed = gpu_in(14)?.event()?;
        let said = "CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered";
        let errors = [
            ("cuEventQuery", failed.done().map(drop)),
            ("cuEventSynchronize", failed.sync()),
            ("cuEventRecord", failed.record()),
            ("cuStreamWaitEvent", failed.wait()),
        ];
        for (call, result) in errors {
            assert_eq!(result.err().ok_or("a call gave Ok")?.to_string(), format!("{call}: {said}"));
        }
        Ok(())
    }

    /// The time between two events is what the driver says from the first to the second, as a `Duration`: the first
    /// is the one the GPU reached earlier. An event and itself have none between them. The other way round is an error
    /// that says so, and the driver's own refusals, for an event not done and one never recorded, come as the driver
    /// says them.
    #[test]
    fn the_time_between_two_events_is_the_drivers_from_the_earlier_and_below_zero_is_an_error() -> Result {
        let cuda = gpu()?;
        let (a, _, c) = (cuda.timed_event()?, cuda.timed_event()?, cuda.timed_event()?);
        asked();
        assert_eq!(c.duration_since(&a)?, Duration::from_millis(2));
        assert_eq!(asked(), ["time from 0x1000 to 0x3000 in 1"]);
        assert_eq!(a.duration_since(&a)?, Duration::ZERO);
        let backwards = a.duration_since(&c).err().ok_or("duration_since gave Ok")?;
        let said = "the GPU reached this event before the one it is timed since: ask the later of the two";
        assert_eq!(backwards.to_string(), format!("cuEventElapsedTime: {said}"));
        for (context, said) in [
            (15, "CUDA_ERROR_NOT_READY (600): device not ready"),
            (16, "CUDA_ERROR_INVALID_HANDLE (400): invalid resource handle"),
        ] {
            let cuda = gpu_in(context)?;
            let (a, b) = (cuda.timed_event()?, cuda.timed_event()?);
            let refused = b.duration_since(&a).err().ok_or("duration_since gave Ok")?;
            assert_eq!(refused.to_string(), format!("cuEventElapsedTime: {said}"));
        }
        Ok(())
    }

    /// The time between events of two GPUs is an error that says so, and the driver is not asked: the clocks of two
    /// GPUs are not one clock.
    #[test]
    fn events_of_two_gpus_have_no_time_between_them() -> Result {
        let (first, second) = (gpu_in(11)?, gpu_in(12)?);
        let (a, b) = (first.timed_event()?, second.timed_event()?);
        asked();
        let e = b.duration_since(&a).err().ok_or("duration_since gave Ok")?;
        let said = "cuEventElapsedTime: the events are of two GPUs, and the clocks of two GPUs are not one clock";
        assert_eq!(e.to_string(), said);
        assert!(asked().is_empty(), "the driver was asked");
        Ok(())
    }

    /// A driver that lacks `cuEventElapsedTime_v2` is opened, and the time between two events of it is the error of
    /// the lookup, in the driver's own words.
    #[test]
    fn a_driver_that_lacks_the_newer_function_gives_no_time_between_two_events() -> Result {
        let lacks = |api: &mut Api| api.elapsed_time = Err(missing(c"cuEventElapsedTime_v2", "undefined symbol"));
        let cuda = gpu_replacing(1, lacks)?;
        let (a, b) = (cuda.timed_event()?, cuda.timed_event()?);
        let e = b.duration_since(&a).err().ok_or("duration_since gave Ok")?;
        assert_eq!(e.to_string(), "dlsym: the driver has no `cuEventElapsedTime_v2`: undefined symbol");
        Ok(())
    }

    /// A driver that lacks `cuGraphInstantiateWithFlags`, which came with CUDA 11.4, is opened, and a capture on a
    /// stream of it is the error of the lookup: it asks the driver for nothing, begins no capture, and runs no work.
    #[test]
    fn a_driver_that_lacks_graph_instantiation_gives_no_graph_and_is_asked_nothing() -> Result {
        let lacks = |api: &mut Api| api.instantiate = Err(missing(c"cuGraphInstantiateWithFlags", "undefined symbol"));
        let cuda = gpu_replacing(1, lacks)?;
        let stream = cuda.stream()?;
        asked();
        let mut ran = false;
        let e = stream
            .capture(|| {
                ran = true;
                Ok(())
            })
            .err()
            .ok_or("capture gave Ok")?;
        assert_eq!(e.to_string(), "dlsym: the driver has no `cuGraphInstantiateWithFlags`: undefined symbol");
        assert!(!ran, "the closure ran");
        let asked = asked();
        assert!(asked.is_empty(), "the driver was asked {asked:?}");
        Ok(())
    }

    /// A capture begins on its stream, in the mode that refuses the calls of its own thread alone; runs its work with
    /// the stream current, so that a launch goes to it; ends the capture; makes a graph of what it recorded; and
    /// destroys what the capture made, as the graph holds its own copy. The graph's launch goes to the current stream
    /// of its GPU, the null stream where there is none, and dropping it destroys it, once. Afterwards a launch goes to
    /// the stream it went to before.
    #[test]
    fn a_capture_records_the_work_of_its_stream_and_makes_a_graph_that_launches_on_the_current_stream() -> Result {
        let cuda = gpu()?;
        let (module, stream) = (kernel(&cuda)?, cuda.stream()?);
        asked();
        let graph = stream.capture(|| {
            // SAFETY: as in `launched`.
            unsafe { module.launch(0, Grid::new(1, 1), &[]) }
        })?;
        let capture = [
            "begin capture on 0x100 with mode 1 in 1",
            "launch on 0x100 in 1",
            "end capture on 0x100 in 1",
            "instantiate 0x5000 with flags 0 in 1",
            "destroy graph 0x5000 in 1",
        ];
        assert_eq!(asked(), capture);
        assert_eq!(launched(&module)?, ["launch on 0x0 in 1"], "after the capture, a launch goes where it went");
        // SAFETY: the fake driver runs no work.
        unsafe { graph.launch() }?;
        assert_eq!(asked(), ["launch graph 0x6000 on 0x0 in 1"]);
        // SAFETY: as above.
        running(&stream, || unsafe { graph.launch() })?;
        assert_eq!(asked(), ["launch graph 0x6000 on 0x100 in 1"]);
        drop(graph);
        assert_eq!(asked(), ["destroy executable graph 0x6000 in 1"]);
        Ok(())
    }

    /// A capture ends where its work fails, where it panics, and where the driver cannot record it, and the stream is
    /// not left capturing: the error of the work is given back, or the panic, the graph that the driver made is
    /// destroyed, and none is made of it. The error of the work comes before the driver's. A capture the driver
    /// refuses to begin runs no work, and one it refuses to make a graph of destroys what it recorded.
    #[test]
    fn a_capture_is_ended_where_its_work_fails_or_panics_or_the_driver_cannot_record_it() -> Result {
        let refusal = "cuStreamEndCapture: CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered";
        let in_context = |context: usize| [format!("begin capture on 0x100 with mode 1 in {context}")];
        let (cuda, work) = (gpu()?, Error::new("cuLaunchKernel", "the work failed".to_string()));
        let stream = cuda.stream()?;
        asked();
        let e = stream.capture(|| Err(work.clone())).err().ok_or("capture gave Ok")?;
        assert_eq!(e, work);
        let ended = ["end capture on 0x100 in 1", "destroy graph 0x5000 in 1"];
        assert_eq!(asked(), [&in_context(1)[..], &ended.map(String::from)[..]].concat());
        let panicked =
            panic::catch_unwind(AssertUnwindSafe(|| stream.capture(|| panic::resume_unwind(Box::new("a panic")))));
        assert!(panicked.is_err(), "the closure did not panic");
        assert_eq!(asked(), [&in_context(1)[..], &ended.map(String::from)[..]].concat());
        let module = kernel(&cuda)?;
        assert_eq!(launched(&module)?, ["launch on 0x0 in 1"], "a panic gives the stream before it back");

        // The driver invalidated the capture, and gave no graph.
        let stream = gpu_in(15)?.stream()?;
        asked();
        let e = stream.capture(|| Ok(())).err().ok_or("capture gave Ok")?;
        assert_eq!(e.to_string(), refusal);
        assert_eq!(asked(), [&in_context(15)[..], &["end capture on 0x100 in 15".to_string()][..]].concat());
        // Where the work failed too, its error is the one given back.
        let e = stream.capture(|| Err(work.clone())).err().ok_or("capture gave Ok")?;
        assert_eq!(e, work);
        asked();

        // The driver refuses to begin: the work is not run, and no capture is ended.
        let stream = gpu_in(14)?.stream()?;
        asked();
        let mut ran = false;
        let e = stream
            .capture(|| {
                ran = true;
                Ok(())
            })
            .err()
            .ok_or("capture gave Ok")?;
        assert!(!ran, "the work ran");
        assert_eq!(e.call, "cuStreamBeginCapture");
        assert_eq!(asked(), in_context(14));

        // The driver refuses to make a graph: what it recorded is destroyed.
        let stream = gpu_in(16)?.stream()?;
        asked();
        let e = stream.capture(|| Ok(())).err().ok_or("capture gave Ok")?;
        assert_eq!(e.call, "cuGraphInstantiate");
        let made =
            ["end capture on 0x100 in 16", "instantiate 0x5000 with flags 0 in 16", "destroy graph 0x5000 in 16"];
        assert_eq!(asked(), [&in_context(16)[..], &made.map(String::from)[..]].concat());
        Ok(())
    }

    /// A graph is of the GPU whose stream it was captured on: its launch and its drop make that GPU's context current,
    /// whichever the thread has then. The drop waits for nothing: the driver frees a graph that runs when it ends.
    #[test]
    fn a_graph_is_launched_and_destroyed_in_the_context_of_its_gpu_whichever_is_current() -> Result {
        let graph = gpu_replacing(1, |api| api.synchronize = synchronize)?.stream()?.capture(|| Ok(()))?;
        let second = gpu_in(2)?;
        second.stream()?;
        asked();
        // SAFETY: the fake driver runs no work.
        unsafe { graph.launch() }?;
        assert_eq!(asked(), ["launch graph 0x6000 on 0x0 in 1"]);
        second.stream()?;
        asked();
        drop(graph);
        assert_eq!(asked(), ["destroy executable graph 0x6000 in 1"]);
        Ok(())
    }

    /// What `capture` says where a drop ended the capture: the end of the capture is invalidated.
    const INVALIDATED: &str = "cuStreamEndCapture: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (901): \
                               operation failed due to a previous error during capture";

    /// What the fake driver is asked to begin and end a capture on stream `stream`, in context 1.
    fn begin(stream: usize) -> String {
        format!("begin capture on {stream:#x} with mode 1 in 1")
    }

    fn end(stream: usize) -> String {
        format!("end capture on {stream:#x} in 1")
    }

    /// A drop makes the call that gives its memory back once, where no capture refuses it. An error of the driver that
    /// is no capture's is left, as it was before captures: in a capture too, the call is not made again, and no capture
    /// is ended or begun for it.
    #[test]
    fn a_drop_makes_its_call_once_where_no_capture_refuses_it_and_leaves_any_other_error() -> Result {
        let cuda = gpu()?;
        let buffer = cuda.alloc::<u32>(4)?;
        asked();
        drop(buffer);
        assert_eq!(asked(), ["free 0x700000000000"], "no capture is on");

        let cuda = gpu_replacing(1, |api| api.free = free_fails)?;
        let (buffer, stream) = (cuda.alloc::<u32>(4)?, cuda.stream()?);
        asked();
        stream.capture(|| {
            drop(buffer);
            Ok(())
        })?;
        let capture =
            [begin(0x100), "free 0x700000000000".into(), end(0x100), "instantiate 0x5000 with flags 0 in 1".into()];
        assert_eq!(asked()[..4], capture, "the driver refused the free, and a capture was on, but not for the capture");
        Ok(())
    }

    /// A drop in the closure of a capture finds that the driver refuses its call while the capture of its thread is
    /// on. It ends the capture, makes the call, and begins the capture again on its stream, and in its mode, so that
    /// what the closure enqueues next is recorded, and does not run. The capture is lost: the closure said `Ok`, and
    /// the error of `capture` is the end's, `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (901)`, and no graph is made of what
    /// was recorded after the drop. The graph of the capture that was ended is destroyed, as is the last.
    #[test]
    fn a_drop_in_a_capture_ends_it_makes_its_call_begins_it_again_and_the_capture_is_lost() -> Result {
        let cuda = gpu()?;
        let (buffer, stream, module) = (cuda.alloc::<u32>(4)?, cuda.stream()?, kernel(&cuda)?);
        asked();
        let mut seen = None;
        let e = stream
            .capture(|| {
                drop(buffer);
                seen = Some(launched(&module));
                Ok(())
            })
            .err()
            .ok_or("capture gave Ok")?;
        assert_eq!(e.to_string(), INVALIDATED);
        let seen = seen.ok_or("the closure did not run")??;
        let drop_in_the_capture = [
            begin(0x100),
            "free 0x700000000000".to_string(),
            end(0x100),
            "free 0x700000000000".to_string(),
            begin(0x100),
            "launch on 0x100 in 1".to_string(),
        ];
        assert_eq!(seen, drop_in_the_capture[..], "what the closure enqueued after the drop went to the stream");
        let done = [end(0x100), "destroy graph 0x5000 in 1".into(), "destroy graph 0x5000 in 1".into()];
        assert_eq!(asked(), done, "the capture ended, and what it made was destroyed");
        Ok(())
    }

    /// With more than one capture of the thread on, a drop ends each of them, the newest first, makes its call, and
    /// begins each again, the oldest first, as each had begun. Each is lost: the closure of each said `Ok`.
    #[test]
    fn a_drop_in_nested_captures_ends_each_the_newest_first_and_begins_each_the_oldest_first() -> Result {
        let cuda = gpu()?;
        let (buffer, outer, inner) = (cuda.alloc::<u32>(4)?, cuda.stream()?, cuda.stream()?);
        asked();
        let mut lost = None;
        let e = outer
            .capture(|| {
                lost = Some(inner.capture(|| {
                    drop(buffer);
                    Ok(())
                }));
                Ok(())
            })
            .err()
            .ok_or("capture gave Ok")?;
        let inner_lost = lost.ok_or("the closure did not run")?.err().ok_or("the inner capture gave Ok")?;
        assert_eq!((e.to_string(), inner_lost.to_string()), (INVALIDATED.to_string(), INVALIDATED.to_string()));
        let want = [
            begin(0x100),
            begin(0x200),
            "free 0x700000000000".to_string(),
            end(0x200),
            end(0x100),
            "free 0x700000000000".to_string(),
            begin(0x100),
            begin(0x200),
            // The inner capture is over, and what it made, twice, is destroyed; then the outer's.
            end(0x200),
            "destroy graph 0x5000 in 1".to_string(),
            "destroy graph 0x5000 in 1".to_string(),
            end(0x100),
            "destroy graph 0x5000 in 1".to_string(),
            "destroy graph 0x5000 in 1".to_string(),
        ];
        assert_eq!(asked(), want);
        Ok(())
    }

    /// Where the driver refuses to begin the capture again, the rest of the closure goes to the null stream, which is
    /// safe to give work to, and not to the stream that no longer captures, where the work would run beside the
    /// work that the capture records. The capture is lost, all the same. A drop after that finds no capture on, and
    /// makes its call once, as it does where none is on.
    #[test]
    fn where_a_capture_cannot_be_begun_again_the_rest_of_the_closure_works_on_the_null_stream() -> Result {
        let cuda = gpu()?;
        let (first, second) = (cuda.alloc::<u32>(4)?, cuda.alloc::<u32>(4)?);
        let (stream, module) = (cuda.stream()?, kernel(&cuda)?);
        asked();
        BEGINS_LEFT.set(1);
        let mut seen = None;
        let e = stream
            .capture(|| {
                drop(first);
                seen = Some(launched(&module));
                drop(second);
                Ok(())
            })
            .err()
            .ok_or("capture gave Ok")?;
        assert_eq!(e.to_string(), INVALIDATED);
        let seen = seen.ok_or("the closure did not run")??;
        let want = [
            begin(0x100),
            "free 0x700000000000".to_string(),
            end(0x100),
            "free 0x700000000000".to_string(),
            // Refused, and the launch is not.
            begin(0x100),
            "launch on 0x0 in 1".to_string(),
        ];
        assert_eq!(seen, want);
        // The capture is not on, so the drop after the launch makes its call once, begins no capture, and the capture
        // is not ended again. The graph the end made is destroyed.
        assert_eq!(asked(), ["free 0x700000000010", "destroy graph 0x5000 in 1"]);
        assert_eq!(launched(&module)?, ["launch on 0x0 in 1"], "after the capture, a launch goes where it went");
        Ok(())
    }

    /// A capture that was lost, as the driver refused to begin it again, is on no more, and a drop after that ends and
    /// begins the captures that are on, and asks the driver nothing of the lost one. The first drop begins the outer
    /// capture again, and the driver refuses the inner; the second finds the outer on, and the driver refuses that.
    #[test]
    fn a_capture_that_was_lost_is_left_alone_by_the_drops_after_it() -> Result {
        let cuda = gpu()?;
        let (first, second) = (cuda.alloc::<u32>(4)?, cuda.alloc::<u32>(4)?);
        let (outer, inner) = (cuda.stream()?, cuda.stream()?);
        asked();
        // The outer capture begins, the inner begins, and the first drop begins the outer again.
        BEGINS_LEFT.set(3);
        let mut lost = None;
        let e = outer
            .capture(|| {
                lost = Some(inner.capture(|| {
                    drop(first);
                    drop(second);
                    Ok(())
                }));
                Ok(())
            })
            .err()
            .ok_or("capture gave Ok")?;
        let inner_lost = lost.ok_or("the closure did not run")?.err().ok_or("the inner capture gave Ok")?;
        assert_eq!((e.to_string(), inner_lost.to_string()), (INVALIDATED.to_string(), INVALIDATED.to_string()));
        let want = [
            begin(0x100),
            begin(0x200),
            // The first drop.
            "free 0x700000000000".to_string(),
            end(0x200),
            end(0x100),
            "free 0x700000000000".to_string(),
            begin(0x100),
            // Refused: the inner capture is lost.
            begin(0x200),
            // The second drop: the outer capture alone is on, and the inner one is not asked of.
            "free 0x700000000010".to_string(),
            end(0x100),
            "destroy graph 0x5000 in 1".to_string(),
            "free 0x700000000010".to_string(),
            // Refused: the outer capture is lost too.
            begin(0x100),
            // What the two ends made is destroyed: the inner capture's, and the outer's last.
            "destroy graph 0x5000 in 1".to_string(),
            "destroy graph 0x5000 in 1".to_string(),
        ];
        assert_eq!(asked(), want);
        Ok(())
    }

    /// A drop in a capture whose GPU's context cannot be made current ends the capture all the same, as the driver ends
    /// a capture in whichever context is current, makes its call, and begins the capture again, in that context too.
    /// The capture is lost, as it is by any drop. The buffer is of the GPU of context 1 and the capture of the GPU of
    /// context 2.
    #[test]
    fn a_drop_in_a_capture_whose_context_cannot_be_made_current_ends_it_makes_its_call_and_begins_it_again() -> Result {
        let (first, second) = (gpu_in(1)?, gpu_in(2)?);
        let (buffer, stream) = (first.alloc::<u32>(4)?, second.stream()?);
        asked();
        let e = stream
            .capture(|| {
                UNSETTABLE.set(2);
                drop(buffer);
                Ok(())
            })
            .err()
            .ok_or("capture gave Ok")?;
        UNSETTABLE.set(0);
        // The context of the capture's GPU could not be made current until `capture` returned, and the error is the
        // capture's that is lost, not the context's.
        assert_eq!(e.to_string(), INVALIDATED);
        let want = [
            begin(0x100).replace("in 1", "in 2"),
            "free 0x700000000000".to_string(),
            // The context of the buffer's GPU is current, and the capture is ended and begun again in it.
            end(0x100),
            "free 0x700000000000".to_string(),
            begin(0x100),
            // `capture` ends it again, and what the two ends made is destroyed.
            end(0x100),
            "destroy graph 0x5000 in 1".to_string(),
            "destroy graph 0x5000 in 1".to_string(),
        ];
        assert_eq!(asked(), want);
        Ok(())
    }

    /// A capture whose GPU's context cannot be made current once the closure is done ends the capture all the same, in
    /// the context that is current, destroys what it made, asks for no graph, and gives the error of `cuCtxSetCurrent`,
    /// but for the error of the closure, which comes first. The capture is of the GPU of context 2, and the closure
    /// makes the context 1 current by making a stream of its GPU, and destroying it.
    #[test]
    fn a_capture_whose_context_cannot_be_made_current_once_its_closure_is_done_is_ended_all_the_same() -> Result {
        let (first, second) = (gpu_in(1)?, gpu_in(2)?);
        let stream = second.stream()?;
        let work = Error::new("cuLaunchKernel", "the work failed".to_string());
        let refused = "cuCtxSetCurrent: CUDA_ERROR_INVALID_CONTEXT (201): invalid device context";
        for (made, result, says) in [(0x200, Ok(()), refused.to_string()), (0x300, Err(work.clone()), work.to_string())]
        {
            asked();
            let e = stream
                .capture(|| {
                    drop(first.stream()?);
                    UNSETTABLE.set(2);
                    result.clone()
                })
                .err()
                .ok_or("capture gave Ok")?;
            UNSETTABLE.set(0);
            assert_eq!(e.to_string(), says);
            let want = [
                begin(0x100).replace("in 1", "in 2"),
                format!("create {made:#x} with flags 0 at 0 in 1"),
                format!("destroy {made:#x} in 1"),
                // The capture's context cannot be made current: the capture is ended in the one that is, and the
                // instantiate is not asked for.
                end(0x100),
                "destroy graph 0x5000 in 1".to_string(),
            ];
            assert_eq!(asked(), want, "{says}");
        }
        Ok(())
    }

    /// A drop of a buffer, of a module and of a graph, in a capture, each makes its call as it does where no capture
    /// refuses it, once the capture is ended, and the capture begins again.
    #[test]
    fn the_drops_of_a_buffer_a_module_and_a_graph_make_their_calls_once_the_capture_is_ended() -> Result {
        let cuda = gpu()?;
        let stream = cuda.stream()?;
        let made: [(&str, Box<dyn std::any::Any>, &str); 3] = [
            ("buffer", Box::new(cuda.alloc::<u32>(4)?), "free 0x700000000000"),
            ("module", Box::new(kernel(&cuda)?), "unload 7"),
            ("graph", Box::new(cuda.stream()?.capture(|| Ok(()))?), "destroy executable graph 0x6000 in 1"),
        ];
        for (name, what, call) in made {
            asked();
            let e = stream
                .capture(|| {
                    drop(what);
                    Ok(())
                })
                .err()
                .ok_or("capture gave Ok")?;
            assert_eq!(e.to_string(), INVALIDATED, "{name}");
            let want = [begin(0x100), call.to_string(), end(0x100), call.to_string(), begin(0x100), end(0x100)];
            assert_eq!(asked()[..6], want, "{name}");
        }
        Ok(())
    }

    /// A module that launches its kernel when it is dropped, and sends what that asked of the fake driver.
    struct LaunchesWhenDropped(Module, mpsc::Sender<Result<Vec<String>, String>>);

    impl Drop for LaunchesWhenDropped {
        fn drop(&mut self) {
            // The thread has ended: if no one hears it, no one needs it.
            let _ = self.1.send(launched(&self.0).map_err(|e| e.to_string()));
        }
    }

    /// A launch in the destructor of a thread-local, as the thread ends, goes to the null stream, and no thread-local
    /// of kernite's is read there that may be gone, as a panic in a destructor aborts the process. The thread-local
    /// that launches is touched after the fake driver's record, and before the first `run`, which touches what a
    /// `run` keeps if it is a thread-local with a destructor; Linux ends a thread's thread-locals in the reverse
    /// order of their first touches, so that one is gone when the launch is made.
    #[test]
    fn a_launch_as_a_thread_ends_goes_to_the_null_stream_whichever_thread_locals_are_gone() -> Result {
        thread_local! {
            static LATE: RefCell<Option<LaunchesWhenDropped>> = const { RefCell::new(None) };
        }
        let (send, hear) = mpsc::channel();
        std::thread::spawn(move || -> Result {
            let cuda = gpu()?;
            let module = kernel(&cuda)?;
            LATE.set(Some(LaunchesWhenDropped(module, send)));
            let stream = cuda.stream()?;
            running(&stream, || ());
            // What is asked from here is the launch's alone.
            drop((stream, cuda));
            asked();
            Ok(())
        })
        .join()
        .map_err(|_| "the thread panicked")??;
        assert_eq!(hear.recv()?, Ok(vec!["launch on 0x0 in 1".to_string()]));
        Ok(())
    }

    /// A copy asks the driver for the copy alone, from the address of one memory to the other's with their bytes, on
    /// the null stream until a `run` makes a stream current, and then on that stream; a part of a buffer is copied
    /// from, and to, by its own address.
    #[test]
    fn a_copy_asks_the_driver_for_the_copy_alone_on_the_current_stream() -> Result {
        let cuda = gpu()?;
        let (from, mut to, stream) = (cuda.alloc::<u32>(4)?, cuda.alloc::<u32>(4)?, cuda.stream()?);
        let (first, second) = (0x7000_0000_0000u64, 0x7000_0000_0010u64);
        asked();
        cuda.copy(&mut to, &from)?;
        assert_eq!(asked(), [format!("copy 16 bytes from {first:#x} to {second:#x} on 0x0 in 1")]);
        running(&stream, || cuda.copy(&mut to, &from))?;
        assert_eq!(asked(), [format!("copy 16 bytes from {first:#x} to {second:#x} on 0x100 in 1")]);
        let tail = from.part(1..).ok_or("part gave None")?;
        let mut end = cuda.alloc::<u32>(3)?;
        asked();
        cuda.copy(&mut end, &tail)?;
        let (from_tail, to_end) = (first + 4, second + 16);
        assert_eq!(asked(), [format!("copy 12 bytes from {from_tail:#x} to {to_end:#x} on 0x0 in 1")]);
        // A part is copied to by its own address too, and a copy is of the part's elements and no more.
        let mut middle = cuda.alloc::<u32>(4)?;
        let into = middle.ptr().addr() + 4;
        asked();
        cuda.copy(&mut middle.part_mut(1..3).ok_or("part_mut gave None")?, &from.part(..2).ok_or("part gave None")?)?;
        assert_eq!(asked(), [format!("copy 8 bytes from {first:#x} to {into:#x} on 0x0 in 1")]);
        Ok(())
    }

    /// A copy and a zeroing at addresses go to the current stream, as `copy` and `zero` do, with the bytes of the
    /// elements; and a write and a read at addresses are the driver's that wait, as `write` and `read` are. A length of
    /// more bytes than there are addresses is refused, and the driver is not asked.
    #[test]
    fn copies_and_zeros_at_addresses_go_to_the_current_stream_and_writes_and_reads_wait() -> Result {
        let cuda = gpu()?;
        let stream = cuda.stream()?;
        let (to, from) = (Ptr::<u32>::new(0x7000_0000_1000), Ptr::<u32>::new(0x7000_0000_2000));
        asked();
        // SAFETY: the fake driver reaches no memory.
        unsafe {
            cuda.copy_at(to, from, 3)?;
            running(&stream, || cuda.copy_at(to, from, 3))?;
            cuda.zero_at(to, 4)?;
            running(&stream, || cuda.zero_at(to, 4))?;
            cuda.write_at(Ptr::<u64>::new(0x7000_0000_1000), &[0x1234])?;
        }
        let mut out = [0u16; 3];
        // SAFETY: the fake driver reaches no memory.
        unsafe { cuda.read_at(Ptr::new(0x7000_0000_2000), &mut out) }?;
        let want = [
            "copy 12 bytes from 0x700000002000 to 0x700000001000 on 0x0 in 1",
            "copy 12 bytes from 0x700000002000 to 0x700000001000 on 0x100 in 1",
            "set 0x700000001000 to 0, 16 bytes",
            "set 0x700000001000 to 0, 16 bytes on 0x100",
            "write 0x1234 at 0x700000001000, 8 bytes",
            "read 6 bytes at 0x700000002000",
        ];
        assert_eq!((asked(), out), (want.map(String::from).to_vec(), [0x707; 3]));
        // SAFETY: the fake driver reaches no memory.
        let (copied, zeroed) = unsafe { (cuda.copy_at(to, from, usize::MAX), cuda.zero_at(to, usize::MAX)) };
        let words = format!("{} elements of 4 bytes are more bytes than there are addresses", usize::MAX);
        let said = (copied.err().map(|e| e.to_string()), zeroed.err().map(|e| e.to_string()));
        assert_eq!(said, (Some(format!("cuMemcpyAsync: {words}")), Some(format!("cuMemsetD8Async: {words}"))));
        assert!(asked().is_empty(), "the driver was asked");
        Ok(())
    }

    /// Zeroing a buffer, or a module's memory, goes to the current stream, and not to the null stream where one is
    /// current.
    #[test]
    fn zeroing_memory_goes_to_the_current_stream() -> Result {
        let cuda = gpu()?;
        let (mut buffer, stream) = (cuda.alloc::<u32>(4)?, cuda.stream()?);
        let at = 0x7000_0000_0000u64;
        asked();
        buffer.zero()?;
        assert_eq!(asked(), [format!("set {at:#x} to 0, 16 bytes")]);
        running(&stream, || buffer.zero())?;
        assert_eq!(asked(), [format!("set {at:#x} to 0, 16 bytes on 0x100")]);
        let module = Module::load(&cuda, "", &[], &[("small", None)])?;
        asked();
        running(&stream, || module.global::<u8>(0)?.zero())?;
        assert_eq!(asked(), ["set 0x1000 to 0, 16 bytes on 0x100"]);
        Ok(())
    }

    /// The zeros of `alloc`, of `managed` and of a module's load, for memory the module has no room for, go to the
    /// current stream, as those of `zero` do, and not to the null stream where one is current: each is queued after
    /// the work that stream was given, with no wait.
    #[test]
    fn the_zeros_of_new_memory_go_to_the_current_stream() -> Result {
        let cuda = gpu_replacing(1, |api| api.alloc_managed = alloc_managed)?;
        let stream = cuda.stream()?;
        let made = || -> Result<Vec<String>> {
            asked();
            let _memory = (
                cuda.alloc::<u32>(4)?,
                cuda.managed::<u32>(4)?,
                Module::load(&cuda, "", &[], &[("slot", Some((256, 1)))])?,
            );
            Ok(asked())
        };
        let at = 0x7000_0000_0000u64;
        let (buffer, managed, module) = (at, at + 16, at + 32);
        let on = |stream: &str| {
            [
                "alloc 16".to_string(),
                format!("set {buffer:#x} to 0, 16 bytes{stream}"),
                "alloc managed 16".to_string(),
                format!("set {managed:#x} to 0, 16 bytes{stream}"),
                "alloc 256".to_string(),
                format!("set {module:#x} to 0, 256 bytes{stream}"),
                format!("write {module:#x} at 0x2000, 8 bytes"),
            ]
        };
        assert_eq!(made()?, on(""), "where no stream is current");
        NEXT.set(at);
        assert_eq!(running(&stream, made)?, on(" on 0x100"), "in a run");
        Ok(())
    }

    /// Memory of another number of elements is not copied, nor is memory of another GPU, and the error says which, in
    /// GUIDE.md's words, before the driver is asked for anything.
    #[test]
    fn memory_of_another_length_or_gpu_is_not_copied_and_the_driver_is_asked_for_nothing() -> Result {
        let (cuda, other) = (gpu_in(1)?, gpu_in(2)?);
        let (long, mut short, mut tail) = (cuda.alloc::<u32>(4)?, cuda.alloc::<u32>(2)?, cuda.alloc::<u32>(3)?);
        let (mut there, mut here) = (other.alloc::<u32>(4)?, cuda.alloc::<u32>(4)?);
        asked();
        let e = cuda.copy(&mut short, &long).err().ok_or("copy gave Ok")?;
        assert_eq!(e.to_string(), "cuMemcpyAsync: memory of 2 elements copied from memory of 4");
        let e = cuda.copy(&mut here, &short).err().ok_or("copy gave Ok")?;
        assert_eq!(e.to_string(), "cuMemcpyAsync: memory of 4 elements copied from memory of 2");
        let gpu = "cuMemcpyAsync: the memory is on another GPU than the copy is made on";
        let e = cuda.copy(&mut here, &there).err().ok_or("copy gave Ok")?;
        assert_eq!(e.to_string(), gpu, "from memory of another GPU");
        let e = cuda.copy(&mut there, &here).err().ok_or("copy gave Ok")?;
        assert_eq!(e.to_string(), gpu, "to memory of another GPU");
        let part = there.part(1..).ok_or("part gave None")?;
        let e = cuda.copy(&mut tail, &part).err().ok_or("copy gave Ok")?;
        assert_eq!(e.to_string(), gpu, "from a part of memory of another GPU");
        let e = cuda.copy(&mut there.part_mut(1..).ok_or("part_mut gave None")?, &tail).err().ok_or("copy gave Ok")?;
        assert_eq!(e.to_string(), gpu, "to a part of memory of another GPU");
        assert!(asked().is_empty(), "the driver was asked");
        Ok(())
    }

    /// No elements are no bytes, and the driver is asked for the copy as for any, with the address it gave the memory.
    #[test]
    fn a_copy_of_no_elements_is_asked_of_the_driver_as_any_copy() -> Result {
        let cuda = gpu()?;
        let (from, mut to) = (cuda.alloc::<u32>(0)?, cuda.alloc::<u32>(0)?);
        asked();
        cuda.copy(&mut to, &from)?;
        assert_eq!(asked().len(), 1);
        Ok(())
    }

    /// What the driver refuses of a copy is the error of the copy, in the driver's words.
    #[test]
    fn what_the_driver_refuses_of_a_copy_is_the_error_of_the_copy() -> Result {
        let cuda = gpu_in(14)?;
        let (from, mut to) = (cuda.alloc::<u32>(4)?, cuda.alloc::<u32>(4)?);
        let e = cuda.copy(&mut to, &from).err().ok_or("copy gave Ok")?;
        let said = "cuMemcpyAsync: CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered";
        assert_eq!(e.to_string(), said);
        Ok(())
    }
}
