//! Memory the host and a GPU's kernels both reach, at one address.

use std::ffi::c_int;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;
use std::slice;

use super::{Cuda, Driver, Error, Memory, MemoryMut, Plain, Ptr, bytes, sealed};

mod placement;

pub use placement::{Advice, Location, Placement};

/// Something a GPU can do if it says so, as `cuDeviceGetAttribute` has it: the attribute's
/// number, and its name without the `CU_DEVICE_ATTRIBUTE_` before it. The GPU says so by an
/// attribute that is not 0.
pub(crate) type Flag = (c_int, &'static str);

/// The GPU allocates managed memory.
const MANAGED_MEMORY: Flag = (83, "MANAGED_MEMORY");

/// The GPU uses the host's addresses for the host's memory, so that page-locked memory has one
/// address for both.
const UNIFIED_ADDRESSING: Flag = (41, "UNIFIED_ADDRESSING");

/// The GPU reaches any pageable memory of the host, as one with HMM does.
const PAGEABLE_MEMORY_ACCESS: Flag = (88, "PAGEABLE_MEMORY_ACCESS");

/// The GPU page-locks memory the host has, where it is: `cuMemHostRegister` does it.
const HOST_REGISTER_SUPPORTED: Flag = (99, "HOST_REGISTER_SUPPORTED");

/// The GPU reaches memory that `cuMemHostRegister` page-locked at the host's own address for it, so that a kernel is
/// given the slice's pointer, not another address that the driver has for the same memory.
const HOST_POINTER_FOR_REGISTERED_MEM: Flag = (91, "CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM");

/// The GPU reaches managed memory while the host does, as the driver needs of a GPU that it moves
/// memory for.
const CONCURRENT_MANAGED_ACCESS: Flag = (89, "CONCURRENT_MANAGED_ACCESS");

/// The error that says the `call` is refused, as a GPU `cannot` do it, and why: its attribute `name` is 0.
fn lacks(call: &'static str, cannot: &str, name: &str) -> Error {
    Error::new(call, format!("{cannot}: its CU_DEVICE_ATTRIBUTE_{name} is 0"))
}

impl Driver {
    /// `Ok` if this GPU has the `flag`; or the error that says the `call` is refused, as the GPU
    /// `cannot` do it, and why.
    pub(crate) fn require(&self, call: &'static str, (attribute, name): Flag, cannot: &str) -> Result<(), Error> {
        if self.attribute(attribute)? == 0 {
            return Err(lacks(call, cannot, name));
        }
        Ok(())
    }

    /// `Ok` if each GPU the driver shows that allocates managed memory reaches it while the host does; or the error
    /// that says `cuMemAllocManaged` is refused, and names the first that does not, by the number [`Cuda::on`] takes.
    /// Managed memory is every GPU's, and a GPU that does not faults the host on it while any of its work runs, a
    /// kernel, a copy or a zeroing, so `managed` makes none where there is one. Each GPU is asked whether it
    /// allocates managed memory, and one that cannot reaches none, so it does not stop `managed`.
    fn require_every_gpu_beside_the_host(&self) -> Result<(), Error> {
        let api = &self.api;
        for ordinal in 0..api.gpus()? {
            let mut device = 0;
            // SAFETY: it writes the device to `device`. `ordinal` is of a GPU the driver shows.
            api.check("cuDeviceGet", unsafe { (api.device_get)(&mut device, ordinal as c_int) })?;
            let has = |(attribute, _): Flag| api.attribute_of(device, attribute);
            if has(MANAGED_MEMORY)? != 0 && has(CONCURRENT_MANAGED_ACCESS)? == 0 {
                let cannot = format!("GPU {ordinal} cannot reach managed memory while the host does");
                return Err(lacks("cuMemAllocManaged", &cannot, CONCURRENT_MANAGED_ACCESS.1));
            }
        }
        Ok(())
    }
}

impl Cuda {
    /// `len` elements of managed memory, all zero: memory that the host and the GPU's kernels
    /// reach at one address, and the driver moves to whichever touches it. The GPU must allocate
    /// managed memory; the error says so of one that does not: `cuMemAllocManaged: the GPU cannot
    /// allocate managed memory: its CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY is 0`.
    ///
    /// Managed memory is every GPU's, so each GPU the driver shows that allocates it must reach it
    /// while the host does too. A GPU that cannot, WSL2's say, faults the host on managed memory
    /// while any of its work runs, a kernel, a copy or a zeroing, and the host may touch the memory
    /// then. `managed` makes none where one GPU cannot, whichever GPU `self` is of, and the error
    /// names the first by the number [`Cuda::on`] takes: `cuMemAllocManaged: GPU 1 cannot reach
    /// managed memory while the host does: its CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is 0`.
    /// [`pinned`](Cuda::pinned) makes memory that such a GPU reaches at the host's addresses.
    /// `CUDA_VISIBLE_DEVICES` leaves a GPU out of those the driver shows, and `managed` does not
    /// ask a GPU that is left out.
    ///
    /// The GPU writes the zeros, so the driver has each page at the GPU when the host first writes
    /// it, and moves it back. Memory that the host or a kernel writes whole before anything reads
    /// it needs no zeros, and [`managed_uninit`](Cuda::managed_uninit) makes it for what the
    /// driver's own call costs.
    ///
    /// The zeros go to the current stream of the GPU, as a launch does, after the work that stream
    /// was given before, and the call returns once they are queued. [`host`](Unified::host) and
    /// [`host_mut`](Unified::host_mut) wait for them, as they wait for a kernel.
    pub fn managed<T: Plain>(&self, len: usize) -> Result<Unified<'static, T>, Error> {
        // SAFETY: the zeros are written below, before the memory is lent.
        let memory = unsafe { self.managed_uninit(len) }?;
        // The driver does not clear it, and the host must not be lent bytes that were never
        // written. The GPU writes the zeros, queued on the current stream, and a call of the host
        // that reaches the memory waits for them.
        memory.driver.zero(memory.addr(), len * size_of::<T>())?;
        Ok(memory)
    }

    /// `len` elements of managed memory as the driver leaves it, with no zeros written: it costs
    /// what `cuMemAllocManaged` does, and no more. [`managed`](Cuda::managed) writes the zeros with
    /// the GPU, which has the driver move each page to the GPU, and back when the host first
    /// writes it. The GPU must allocate managed memory, and each GPU the driver shows that allocates
    /// it must reach it while the host does too, as for `managed`, and the errors are the same.
    ///
    /// # Safety
    ///
    /// The bytes may be those of memory freed before, which the driver hands out again where a
    /// buffer beside it holds the page. The host or a kernel must write each element before
    /// anything reads it: [`host`](Unified::host) and [`host_mut`](Unified::host_mut) lend the
    /// bytes as they are.
    pub unsafe fn managed_uninit<T: Plain>(&self, len: usize) -> Result<Unified<'static, T>, Error> {
        self.driver.require("cuMemAllocManaged", MANAGED_MEMORY, "the GPU cannot allocate managed memory")?;
        self.driver.require_every_gpu_beside_the_host()?;
        let bytes = bytes::<T>("cuMemAllocManaged", len)?;
        let (api, mut addr) = (self.driver.current()?, 0);
        // SAFETY: it writes the address to `addr`. It refuses no bytes, so it is asked for one. 1
        // is CU_MEM_ATTACH_GLOBAL: any stream of any GPU may use it, as the default stream does.
        api.check("cuMemAllocManaged", unsafe { (api.alloc_managed)(&mut addr, bytes.max(1), 1) })?;
        let ptr = ptr::with_exposed_provenance_mut(addr as usize);
        Ok(Unified { driver: self.driver.clone(), ptr, len, kind: Kind::Managed, of: PhantomData })
    }

    /// `len` elements of page-locked memory, all zero: memory in the host that stays where it is,
    /// and that a kernel reads and writes across the bus, at the address the host has for it. The
    /// GPU must use the host's addresses for the host's memory; the error says so of one that
    /// does not: `cuMemHostAlloc: the GPU cannot reach page-locked memory at the host's addresses:
    /// its CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING is 0`.
    ///
    /// The host writes the zeros, a pass of its own before the host's fill. Memory that the host or
    /// a kernel writes whole before anything reads it needs no zeros, and
    /// [`pinned_uninit`](Cuda::pinned_uninit) makes it for what the driver's own call costs.
    pub fn pinned<T: Plain>(&self, len: usize) -> Result<Unified<'static, T>, Error> {
        // SAFETY: the zeros are written below, before the memory is lent.
        let memory = unsafe { self.pinned_uninit::<T>(len) }?;
        // SAFETY: the allocation has room for `len` elements, and any bits are a `T`. No kernel
        // uses memory the host has only now been given.
        unsafe { memory.ptr.write_bytes(0, len) };
        Ok(memory)
    }

    /// `len` elements of page-locked memory as the driver leaves it, with no zeros written: it
    /// costs what `cuMemHostAlloc` does, and no more. [`pinned`](Cuda::pinned) writes the zeros in
    /// a pass of its own. The GPU must use the host's addresses for the host's memory, as for
    /// `pinned`, and the error is the same.
    ///
    /// # Safety
    ///
    /// The bytes may be those of memory freed before, which the driver hands out again where a
    /// buffer beside it holds the page. The host or a kernel must write each element before
    /// anything reads it: [`host`](Unified::host) and [`host_mut`](Unified::host_mut) lend the
    /// bytes as they are.
    pub unsafe fn pinned_uninit<T: Plain>(&self, len: usize) -> Result<Unified<'static, T>, Error> {
        let cannot = "the GPU cannot reach page-locked memory at the host's addresses";
        self.driver.require("cuMemHostAlloc", UNIFIED_ADDRESSING, cannot)?;
        let bytes = bytes::<T>("cuMemHostAlloc", len)?;
        let (api, mut host) = (self.driver.current()?, ptr::null_mut());
        // SAFETY: it writes the address to `host`. It refuses no bytes, so it is asked for one. 2
        // is CU_MEMHOSTALLOC_DEVICEMAP: the GPU reaches it.
        api.check("cuMemHostAlloc", unsafe { (api.alloc_host)(&mut host, bytes.max(1), 2) })?;
        Ok(Unified { driver: self.driver.clone(), ptr: host.cast::<T>(), len, kind: Kind::Pinned, of: PhantomData })
    }

    /// `memory`, a slice of the host's own, pageable memory, lent to the GPU for as long as the
    /// `Unified` lasts: a kernel reads and writes the elements where they are, a `Vec`'s or a
    /// `Box`'s, and the host has them again when it drops the `Unified`, which waits for the
    /// kernels first. The GPU must reach pageable memory, as one with HMM does; the error says so
    /// of one that does not, and a kernel is not left to fault: `cuDeviceGetAttribute: the GPU
    /// cannot reach pageable memory, a slice of the host's: its
    /// CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS is 0`. No call of the driver's is refused, so the
    /// error names the one that told.
    ///
    /// The driver here, NVIDIA's open kernel module 615.71.10 on Linux 7.0, faults a kernel on an
    /// element that has a guard page in its aligned 64 KiB, a page set with
    /// `madvise(MADV_GUARD_INSTALL)`: the kernel fails with `CUDA_ERROR_ILLEGAL_ADDRESS (700)`, as
    /// does every call after it. glibc 2.43 sets such a page below each thread's stack, so lend
    /// the memory of a `Vec`, a `Box` or an anonymous mapping, and not an array on a thread's stack.
    ///
    /// ```no_run
    /// # use kernite_cuda::Cuda;
    /// let cuda = Cuda::new()?;
    /// let mut xs = vec![1.0f32; 1000];
    /// let lent = cuda.lend(&mut xs)?;
    /// // A kernel adds 1 to each element here, launched on `lent.ptr()`.
    /// drop(lent);
    /// assert_eq!(xs[999], 2.0);
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// The slice is borrowed for as long as it is lent, so the compiler refuses the host a
    /// touch of it:
    ///
    /// ```compile_fail,E0502
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// let mut xs = vec![1.0f32; 4];
    /// let lent = cuda.lend(&mut xs)?;
    /// let first = xs[0];
    /// let for_a_kernel = lent.ptr();
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// A kernel that is no `unsafe fn` is lent a slice lent for good alone, as a `mem::forget` of
    /// what `lend` gave would give one lent for less back to the host with no wait for the kernel:
    ///
    /// ```compile_fail,E0597
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// fn lent_to_a_safe_kernel(_: &mut impl kernite_cuda::MemoryMut<f32>) {}
    /// let mut xs = vec![1.0f32; 4];
    /// lent_to_a_safe_kernel(&mut cuda.lend(&mut xs)?);
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// ```no_run
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// fn lent_to_a_safe_kernel(_: &mut impl kernite_cuda::MemoryMut<f32>) {}
    /// lent_to_a_safe_kernel(&mut cuda.lend(Vec::leak(vec![1.0f32; 4]))?);
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    pub fn lend<'a, T: Plain>(&self, memory: &'a mut [T]) -> Result<Unified<'a, T>, Error> {
        let cannot = "the GPU cannot reach pageable memory, a slice of the host's";
        self.driver.require("cuDeviceGetAttribute", PAGEABLE_MEMORY_ACCESS, cannot)?;
        let (ptr, len) = (memory.as_mut_ptr(), memory.len());
        Ok(Unified { driver: self.driver.clone(), ptr, len, kind: Kind::Lent, of: PhantomData })
    }

    /// `memory`, a slice of the host's own memory, page-locked where it is for as long as the `Unified` lasts: a kernel
    /// reads and writes its elements across the bus, a `Vec`'s or a `Box`'s, and the host has the slice back when it
    /// drops the `Unified`, which waits for the kernels first, and then unlocks it. [`lend`](Cuda::lend) needs a GPU
    /// that reaches pageable memory, as one with HMM does, and this does not: it needs a GPU that page-locks memory the
    /// host has, and reaches it at the host's addresses. The error says so of one that does not, and a kernel is not
    /// left to fault: `cuMemHostRegister: the GPU cannot page-lock memory the host has: its
    /// CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED is 0`, or `cuMemHostRegister: the GPU cannot reach page-locked
    /// memory the host has at the host's addresses: its CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM is
    /// 0`.
    ///
    /// The driver refuses memory that is page-locked already, as a slice of what this returns is: `cuMemHostRegister:
    /// CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED (712): part or all of the requested memory range is already mapped`;
    /// and memory that it made, page-locked or managed: `cuMemHostRegister: CUDA_ERROR_INVALID_VALUE (1): invalid
    /// argument`. A slice with no elements has no memory to page-lock, so the driver is asked neither to page-lock it
    /// nor to unlock it.
    ///
    /// The driver here faults a kernel on a lent element that has a guard page in its aligned 64 KiB, as an array on a
    /// thread's stack may, and runs it on such an element registered: so an array on a thread's stack may be
    /// registered, though not lent.
    ///
    /// A mapping of a file is another matter. The driver here refuses a shared one on ext4, in the words it refuses
    /// memory it made in: `cuMemHostRegister: CUDA_ERROR_INVALID_VALUE (1): invalid argument`. It takes a private one,
    /// and Linux copies each page of it as the driver locks it, as the GPU may write it: a kernel's writes are in the
    /// program's copies, and not in the file.
    ///
    /// ```no_run
    /// # use kernite_cuda::Cuda;
    /// let cuda = Cuda::new()?;
    /// let mut xs = vec![1.0f32; 1000];
    /// let registered = cuda.register(&mut xs)?;
    /// // A kernel adds 1 to each element here, launched on `registered.ptr()`.
    /// drop(registered);
    /// assert_eq!(xs[999], 2.0);
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// The slice is borrowed for as long as it is registered, so the compiler refuses the host a touch of it:
    ///
    /// ```compile_fail,E0502
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// let mut xs = vec![1.0f32; 4];
    /// let registered = cuda.register(&mut xs)?;
    /// let first = xs[0];
    /// let for_a_kernel = registered.ptr();
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// A kernel may write the slice, so it is taken as `&mut`, and a slice the host holds as `&` is refused:
    ///
    /// ```compile_fail,E0308
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// let xs = vec![1.0f32; 4];
    /// let registered = cuda.register(&xs)?;
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// A kernel that is no `unsafe fn` is given a slice registered for good alone, as a `mem::forget` of what
    /// `register` gave would give one registered for less back to the host, with no wait for the kernel, and
    /// locked for good:
    ///
    /// ```compile_fail,E0597
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// fn registered_for_a_safe_kernel(_: &mut impl kernite_cuda::MemoryMut<f32>) {}
    /// let mut xs = vec![1.0f32; 4];
    /// registered_for_a_safe_kernel(&mut cuda.register(&mut xs)?);
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// ```no_run
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// fn registered_for_a_safe_kernel(_: &mut impl kernite_cuda::MemoryMut<f32>) {}
    /// registered_for_a_safe_kernel(&mut cuda.register(Vec::leak(vec![1.0f32; 4]))?);
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    pub fn register<'a, T: Plain>(&self, memory: &'a mut [T]) -> Result<Unified<'a, T>, Error> {
        let cannot = "the GPU cannot page-lock memory the host has";
        self.driver.require("cuMemHostRegister", HOST_REGISTER_SUPPORTED, cannot)?;
        let cannot = "the GPU cannot reach page-locked memory the host has at the host's addresses";
        self.driver.require("cuMemHostRegister", HOST_POINTER_FOR_REGISTERED_MEM, cannot)?;
        let (ptr, len, bytes) = (memory.as_mut_ptr(), memory.len(), size_of_val(memory));
        // A slice with no bytes has none to lock, and the driver refuses no bytes.
        if bytes != 0 {
            let api = self.driver.current()?;
            // SAFETY: the slice is `bytes` bytes at `ptr`, which the host owns. 2 is CU_MEMHOSTREGISTER_DEVICEMAP: the
            // GPU reaches it. It is unregistered once, by the drop of the `Unified`, before the host has it back.
            api.check("cuMemHostRegister", unsafe { (api.register)(ptr.cast(), bytes, 2) })?;
        }
        Ok(Unified { driver: self.driver.clone(), ptr, len, kind: Kind::Registered, of: PhantomData })
    }
}

/// What made the memory, which is how it is let go.
enum Kind {
    /// `cuMemAllocManaged` did: `cuMemFree` frees it.
    Managed,
    /// `cuMemHostAlloc` did: `cuMemFreeHost` frees it.
    Pinned,
    /// The host's own slice, which the host frees when it has it back.
    Lent,
    /// The host's own slice, which `cuMemHostRegister` page-locked: `cuMemHostUnregister` unlocks it, and the host
    /// frees it when it has it back. A slice with no bytes has none to lock, and none to unlock.
    Registered,
}

/// `len` elements of `T` that the host and the kernels of one GPU, the one that made it, both
/// reach at one address, with no copy between them. A kernel is given [`ptr`](Unified::ptr), and
/// the host the slices [`host`](Unified::host) and [`host_mut`](Unified::host_mut). It is one of
/// four kinds, by what makes it:
///
/// - [`Cuda::managed`] makes managed memory, which the driver moves to whichever of the host and
///   the GPU touches it.
/// - [`Cuda::pinned`] makes page-locked memory, which stays in the host's memory, and a kernel
///   reads and writes it across the bus.
/// - [`Cuda::lend`] lends a slice of the host's own memory to the GPU, for as long as `'a`, where
///   the GPU reaches pageable memory.
/// - [`Cuda::register`] page-locks a slice of the host's own memory where it is, for as long as
///   `'a`, and a kernel reads and writes it across the bus. The first two are their own:
///   `Unified<'static, T>`.
///
/// The first two are zeros, which cost: the GPU writes the zeros of managed memory, so the driver
/// has each page at the GPU when the host first writes it, and moves it back, and the host writes
/// those of page-locked memory in a pass of its own. Memory that the host or a kernel writes whole
/// before anything reads it needs none, and the `unsafe` [`Cuda::managed_uninit`] and
/// [`Cuda::pinned_uninit`] make it for what the driver's own calls cost.
///
/// Managed memory also moves ahead of its use: [`prefetch`](Unified::prefetch) sends a range of it
/// to the GPU or the host, [`advise`](Unified::advise) tells the driver how a range is used, and
/// [`placement`](Unified::placement) says what the driver has of both. The driver prefetches and
/// places managed memory alone, so those two refuse page-locked memory, made or registered, and a
/// slice lent; it advises a slice lent too, and page-locked memory not. The driver works on whole
/// pages, 4 KiB: a range reaches all of each page it touches, the rest of the page too, whether it
/// is the memory of the same buffer or of another. A managed buffer starts at a multiple of 512
/// bytes, not at a page, so it shares the pages at its ends with the buffers beside it.
///
/// The slices wait for every kernel launched on the GPU to finish, and say whether one failed,
/// before they lend the memory. A slice taken after a launch is never under its kernel, so the host
/// reads what the kernels wrote, and a kernel reads what the host wrote.
///
/// ```no_run
/// # use kernite_cuda::Cuda;
/// let cuda = Cuda::new()?;
/// let mut xs = cuda.managed::<f32>(1000)?;
/// xs.host_mut()?.fill(1.0);
/// // A kernel adds 1 to each element here, launched on `xs.ptr()`.
/// assert_eq!(xs.host()?[999], 2.0);
/// # Ok::<(), kernite_cuda::Error>(())
/// ```
///
/// A slice the host may write is held by one borrow of the memory, and the memory's pointer needs
/// another, so the compiler refuses to take a pointer for a kernel while the host holds a slice:
///
/// ```compile_fail,E0502
/// # let cuda = kernite_cuda::Cuda::new()?;
/// let mut xs = cuda.managed::<f32>(4)?;
/// let slice = xs.host_mut()?;
/// let for_a_kernel = xs.ptr();
/// slice[0] = 1.0;
/// # Ok::<(), kernite_cuda::Error>(())
/// ```
///
/// A [`Ptr`] is `Copy` and borrows nothing, so the compiler cannot refuse a launch on one taken
/// before the slice:
///
/// ```no_run
/// # let cuda = kernite_cuda::Cuda::new()?;
/// let mut xs = cuda.managed::<f32>(4)?;
/// let for_a_kernel = xs.ptr();
/// let slice = xs.host_mut()?;
/// // A launch on `for_a_kernel` here would run under `slice`.
/// let _ = for_a_kernel;
/// slice[0] = 1.0;
/// # Ok::<(), kernite_cuda::Error>(())
/// ```
///
/// A launch on a pointer is `unsafe` for that, as for a [`Buffer`](super::Buffer), and its
/// contract, which [`Module::launch`] has as its `# Safety`, is this: each pointer is to a buffer
/// or to unified memory of the GPU, as long as the kernel reads and writes through it; from the
/// launch until the kernel is done, the host holds no slice of memory the kernel writes, and no
/// slice it may write of memory the kernel reads. The launcher of a kernel that is no `unsafe fn`
/// takes the memory, not a pointer, and borrows it, as [`Memory`](super::Memory) says, so the
/// compiler refuses that launch while the host holds such a slice.
///
/// Dropping the memory waits for the kernels that use it, and then frees it; or, for a slice
/// lent, gives it back to the host, and for a slice registered, unlocks it and gives it back.
///
/// [`Module::launch`]: super::Module::launch
pub struct Unified<'a, T: Plain> {
    driver: Rc<Driver>,
    /// The address, which is the host's and the GPU's.
    ptr: *mut T,
    len: usize,
    kind: Kind,
    /// The slice a lent memory borrows.
    of: PhantomData<&'a mut [T]>,
}

impl<T: Plain> Unified<'_, T> {
    /// Its address, for a kernel to be given.
    pub fn ptr(&self) -> Ptr<T> {
        Ptr::new(self.addr())
    }

    /// The address of its element `i`, for a kernel to be given the part from there on; `None`
    /// if `i` is past its end.
    pub fn at(&self, i: usize) -> Option<Ptr<T>> {
        Ptr::element(self.addr(), self.len, i)
    }

    /// How many elements it has.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// What it holds, as the host reads it, after the kernels launched before have run: it waits
    /// for every kernel launched on the GPU to finish, and says whether one failed.
    pub fn host(&self) -> Result<&[T], Error> {
        self.driver.sync()?;
        // SAFETY: `ptr` is the address of `len` elements of `T`, which last as long as `self`.
        // `sync` waited for the kernels launched, and none launched after it writes the memory
        // while the host holds this slice: the launcher of a kernel that is no `unsafe fn`
        // borrows the memory mutably to write it, which it cannot while this slice borrows `self`,
        // and an `unsafe` launch keeps from it, as its caller promises. Any bits are a `T`, as
        // `Plain` promises.
        Ok(unsafe { slice::from_raw_parts(self.ptr, self.len) })
    }

    /// What it holds, as the host reads and writes it, after the kernels launched before have
    /// run: it waits for every kernel launched on the GPU to finish, and says whether one failed.
    pub fn host_mut(&mut self) -> Result<&mut [T], Error> {
        self.driver.sync()?;
        // SAFETY: as for `host`, and nothing else holds a slice of it, as `self` is borrowed
        // for as long as this one is held. No kernel launched after `sync` reads the memory
        // either, as the host may write it: a launcher that lends it borrows `self`, which it
        // cannot while this slice borrows it mutably, and an `unsafe` launch keeps from it.
        Ok(unsafe { slice::from_raw_parts_mut(self.ptr, self.len) })
    }

    /// The address as the GPU reckons it.
    fn addr(&self) -> u64 {
        self.ptr.addr() as u64
    }
}

impl<T: Plain> Unified<'static, T> {
    /// Sets every element to zero, on the current stream of its GPU, after the work that stream was given before, and
    /// does not wait for it. [`host`](Unified::host) and [`host_mut`](Unified::host_mut) wait for the zeros, as they
    /// wait for a kernel. It is for memory of `'static` alone, as [`Cuda::copy`] is: a slice lent or registered for
    /// less would be the host's again at a `mem::forget` of its `Unified`, with the zeros still to be written.
    ///
    /// ```compile_fail,E0597
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// let mut xs = vec![1.0f32; 4];
    /// cuda.lend(&mut xs)?.zero()?;
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    pub fn zero(&mut self) -> Result<(), Error> {
        self.driver.zero(self.addr(), self.len * size_of::<T>())
    }
}

// Memory of `'static` alone: a slice lent or registered for less would be the host's again at a `mem::forget` of its
// `Unified`, with no wait for the kernel.
impl<T: Plain> sealed::Sealed for Unified<'static, T> {}

impl<T: Plain> Memory<T> for Unified<'static, T> {
    fn lend(&self, module: &super::Module) -> Result<(Ptr<T>, usize), Error> {
        module.reaches(&self.driver)?;
        Ok((self.ptr(), self.len))
    }

    fn reach(&self, cuda: &Cuda) -> Result<(Ptr<T>, usize), Error> {
        cuda.reaches(&self.driver)?;
        if matches!(self.kind, Kind::Lent) {
            // The driver copies pageable memory through memory of its own: it reads it as the call is made, which is
            // before the work the stream was given earlier has run, and writes it after the call waits for the stream.
            let why = "a slice that `lend` lent is not copied from or to, as the driver reads it when the call is \
                       made, and writes it after the call waits for the stream: `register` page-locks the slice where \
                       it is, and its copies are queued in order"
                .to_string();
            return Err(Error::new("cuMemcpyAsync", why));
        }
        Ok((self.ptr(), self.len))
    }
}

impl<T: Plain> MemoryMut<T> for Unified<'static, T> {}

impl<T: Plain> Drop for Unified<'_, T> {
    fn drop(&mut self) {
        // As for a `Buffer`: left, if its GPU's context cannot be made current. The driver waits
        // for the kernels that use memory before it frees or unlocks it; a slice lent is waited for
        // here.
        if let Ok(api) = self.driver.current() {
            // SAFETY: the driver gave the address of managed and of page-locked memory, and each is
            // freed once, here. A slice registered with bytes was registered once, at this address,
            // and is unlocked once, here. A slice lent or registered is the host's, which frees it.
            unsafe {
                match self.kind {
                    Kind::Managed => (api.free)(self.addr()),
                    Kind::Pinned => (api.free_host)(self.ptr.cast()),
                    // A slice with no bytes was not locked: there is nothing to unlock, and no kernel reaches it.
                    Kind::Registered if self.len * size_of::<T>() == 0 => 0,
                    Kind::Registered => (api.unregister)(self.ptr.cast()),
                    Kind::Lent => (api.synchronize)(),
                }
            };
        }
    }
}

impl<T: Plain> fmt::Debug for Unified<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unified({} elements at {:#x})", self.len, self.addr())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::{Cell, RefCell};
    use std::ffi::{c_uint, c_void};

    use super::super::{Api, Buffer, CuResult, Handle, Module};
    use super::*;

    type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

    thread_local! {
        /// What each GPU here says of each attribute: the GPU's number is the row, and the attribute's the index.
        static ATTRIBUTES: Cell<[[c_int; 128]; 2]> = const { Cell::new([[0; 128]; 2]) };
        /// How many GPUs the fake driver shows.
        static GPUS: Cell<c_int> = const { Cell::new(1) };
    }

    /// The fake GPUs' `cuDeviceGetAttribute`.
    unsafe extern "C" fn attribute(value: *mut c_int, attribute: c_int, device: c_int) -> CuResult {
        // SAFETY: the caller gives a place for the value.
        unsafe { *value = ATTRIBUTES.get()[device as usize][attribute as usize] };
        0
    }

    /// The fake driver's `cuDeviceGetCount`.
    unsafe extern "C" fn device_count(count: *mut c_int) -> CuResult {
        // SAFETY: the caller gives a place for the count.
        unsafe { *count = GPUS.get() };
        0
    }

    /// The fake driver's `cuDeviceGet`: GPU `ordinal` is the device `ordinal`.
    unsafe extern "C" fn device_get(device: *mut c_int, ordinal: c_int) -> CuResult {
        // SAFETY: the caller gives a place for the device.
        unsafe { *device = ordinal };
        0
    }

    /// The fake GPU's `cuCtxSetCurrent`.
    unsafe extern "C" fn set_current(_: Handle) -> CuResult {
        0
    }

    /// What a GPU says that says each of `flags` is 1 and every other attribute 0.
    fn says(flags: &[Flag]) -> [c_int; 128] {
        let mut values = [0; 128];
        for (attribute, _) in flags {
            values[*attribute as usize] = 1;
        }
        values
    }

    /// A machine of one GPU that says each of `flags` is 1 and every other attribute 0, and refuses every call but
    /// the four that count the GPUs and number them, tell the attributes of one, and make its context current.
    pub(crate) fn gpu(flags: &[Flag]) -> Result<Cuda> {
        gpu_with(flags, |_| {})
    }

    /// The same GPU, with the calls `replace` puts in the driver's place.
    pub(crate) fn gpu_with(flags: &[Flag], replace: impl FnOnce(&mut Api)) -> Result<Cuda> {
        gpu_in(1, flags, replace)
    }

    /// A machine of two GPUs, GPU 0 and GPU 1, which say each of `first` and each of `second` is 1 and every other
    /// attribute 0, and the `Cuda` of the GPU `on`.
    fn machine(on: c_int, first: &[Flag], second: &[Flag]) -> Result<Cuda> {
        ATTRIBUTES.set([says(first), says(second)]);
        GPUS.set(2);
        cuda_on(on, 1, |_| {})
    }

    /// The same GPU, with `context` as its context: two with two contexts are two GPUs.
    fn gpu_in(context: usize, flags: &[Flag], replace: impl FnOnce(&mut Api)) -> Result<Cuda> {
        ATTRIBUTES.set([says(flags); 2]);
        GPUS.set(1);
        cuda_on(0, context, replace)
    }

    /// The `Cuda` of the GPU `device`, with `context` as its context, and the calls `replace` puts in the driver's
    /// place.
    fn cuda_on(device: c_int, context: usize, replace: impl FnOnce(&mut Api)) -> Result<Cuda> {
        let mut api = Api::refusing();
        api.device_count = device_count;
        api.device_get = device_get;
        api.attribute = attribute;
        api.set_current = set_current;
        replace(&mut api);
        let driver = Rc::new(Driver { api, device, context: ptr::without_provenance_mut(context) });
        Ok(Cuda::of(driver, String::new())?)
    }

    thread_local! {
        /// The block that the fake driver handed out last: its address and its bytes.
        static BLOCK: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    }

    /// `bytes` of 0xAA, which the fake driver hands out as the driver may, with no clearing.
    fn dirty(bytes: usize) -> *mut u8 {
        let block = Box::leak(vec![0xAAu8; bytes].into_boxed_slice());
        BLOCK.set((block.as_mut_ptr() as usize, bytes));
        block.as_mut_ptr()
    }

    /// The fake driver's `cuMemAllocManaged`.
    unsafe extern "C" fn alloc_managed(addr: *mut u64, bytes: usize, _: c_uint) -> CuResult {
        // SAFETY: the caller gives a place for the address.
        unsafe { *addr = dirty(bytes) as u64 };
        0
    }

    /// The fake driver's `cuMemHostAlloc`.
    unsafe extern "C" fn alloc_host(host: *mut *mut c_void, bytes: usize, _: c_uint) -> CuResult {
        // SAFETY: the caller gives a place for the address.
        unsafe { *host = dirty(bytes).cast() };
        0
    }

    /// The fake driver's `cuMemsetD8Async`, which sets the bytes at once.
    unsafe extern "C" fn memset(addr: u64, value: u8, bytes: usize, _: Handle) -> CuResult {
        // SAFETY: the caller gives the address of a block of the fake driver's, of `bytes` bytes.
        unsafe { ptr::with_exposed_provenance_mut::<u8>(addr as usize).write_bytes(value, bytes) };
        0
    }

    thread_local! {
        /// How many times the fake driver was asked for a copy.
        static COPIED: Cell<usize> = const { Cell::new(0) };
    }

    /// The fake driver's `cuMemcpyAsync`: it counts the copies it is asked for, and makes none.
    unsafe extern "C" fn copy_async(_: u64, _: u64, _: usize, _: Handle) -> CuResult {
        COPIED.set(COPIED.get() + 1);
        0
    }

    /// The fake driver's `cuMemFree`, and `cuMemFreeHost`: it frees the block it handed out last.
    unsafe extern "C" fn free(_: u64) -> CuResult {
        let (at, bytes) = BLOCK.get();
        // SAFETY: the block was leaked from a boxed slice of `bytes` bytes, and is freed once.
        drop(unsafe {
            Box::from_raw(ptr::slice_from_raw_parts_mut(ptr::with_exposed_provenance_mut::<u8>(at), bytes))
        });
        0
    }

    /// The fake driver's `cuMemFreeHost`.
    unsafe extern "C" fn free_host(_: *mut c_void) -> CuResult {
        // SAFETY: as for `free`.
        unsafe { free(0) }
    }

    /// The fake driver's `cuCtxSynchronize`: no kernel runs, and the calls are counted.
    unsafe extern "C" fn synchronize() -> CuResult {
        SYNCHRONIZED.set(SYNCHRONIZED.get() + 1);
        0
    }

    thread_local! {
        /// What the fake driver was asked to page-lock, in order: the address, the bytes and the flags.
        static REGISTERED: RefCell<Vec<(usize, usize, c_uint)>> = const { RefCell::new(Vec::new()) };
        /// The address of each range that the fake driver was asked to unlock, in order.
        static UNREGISTERED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
        /// How many times the fake driver was asked to wait for the kernels.
        static SYNCHRONIZED: Cell<usize> = const { Cell::new(0) };
    }

    /// What `cuMemHostRegister` is, for a fake driver to say.
    type Register = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> CuResult;

    /// The fake driver's `cuMemHostRegister`: it keeps what it is asked.
    unsafe extern "C" fn register(at: *mut c_void, bytes: usize, flags: c_uint) -> CuResult {
        REGISTERED.with_borrow_mut(|asked| asked.push((at.addr(), bytes, flags)));
        0
    }

    /// The fake driver's `cuMemHostRegister` of a driver that says the range is registered already.
    unsafe extern "C" fn register_again(_: *mut c_void, _: usize, _: c_uint) -> CuResult {
        // CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED.
        712
    }

    /// The fake driver's `cuMemHostUnregister`: it keeps what it is asked.
    unsafe extern "C" fn unregister(at: *mut c_void) -> CuResult {
        UNREGISTERED.with_borrow_mut(|asked| asked.push(at.addr()));
        0
    }

    /// A GPU that has the attributes `flags`, whose driver answers `cuMemHostRegister` with `register`, and keeps what
    /// it is asked to register, to unregister and to wait for, none from before.
    fn registering(flags: &[Flag], register: Register) -> Result<Cuda> {
        REGISTERED.with_borrow_mut(Vec::clear);
        UNREGISTERED.with_borrow_mut(Vec::clear);
        SYNCHRONIZED.set(0);
        gpu_with(flags, |api| {
            api.register = register;
            api.unregister = unregister;
            api.synchronize = synchronize;
        })
    }

    /// The fake driver's `cuModuleLoadDataEx`: a module with no kernels and no memory.
    unsafe extern "C" fn load_module(
        module: *mut Handle,
        _: *const c_void,
        _: c_uint,
        _: *mut c_int,
        _: *mut *mut c_void,
    ) -> CuResult {
        // SAFETY: the caller gives a place for the module.
        unsafe { *module = ptr::without_provenance_mut(1) };
        0
    }

    /// The fake driver's `cuModuleUnload`.
    unsafe extern "C" fn unload_module(_: Handle) -> CuResult {
        0
    }

    /// Managed memory is every GPU's, and a GPU that does not reach it while the host does faults the host on it while
    /// any of its work runs, so `managed` makes it only where each GPU the driver shows that allocates it reaches it
    /// while the host does: the GPU that `managed` is called on must allocate it, and the error says so of one that
    /// does not, or names the first GPU that cannot reach it, by the number `Cuda::on` takes, in GUIDE.md's words,
    /// before the driver is asked for any, whichever GPU of the machine the call is made on. Each GPU is asked whether
    /// it allocates managed memory, and one that does not reaches none, so it does not stop `managed`. The driver here
    /// refuses everything, so a refusal that is the attribute's is not the driver's.
    #[test]
    fn managed_memory_is_made_only_where_each_gpu_that_allocates_it_reaches_it_while_the_host_does() -> Result {
        let (m, c) = (MANAGED_MEMORY, CONCURRENT_MANAGED_ACCESS);
        let refused = |cuda: Cuda| -> Result<(String, String)> {
            let made = cuda.managed::<u32>(4).err().ok_or("managed gave Ok")?;
            // SAFETY: no memory is made, as the machine refuses it, and none is read.
            let uninit = unsafe { cuda.managed_uninit::<u32>(4) }.err().ok_or("managed_uninit gave Ok")?;
            Ok((made.to_string(), uninit.to_string()))
        };
        let same = |said: String| (said.clone(), said);
        let allocate = "cuMemAllocManaged: the GPU cannot allocate managed memory: \
                        its CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY is 0";
        let reach = |gpu: usize| {
            format!(
                "cuMemAllocManaged: GPU {gpu} cannot reach managed memory while the host does: \
                 its CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is 0"
            )
        };
        // One GPU.
        assert_eq!(refused(gpu(&[])?)?, same(allocate.to_string()), "a GPU that does neither");
        assert_eq!(refused(gpu(&[c])?)?, same(allocate.to_string()), "a GPU that reaches it and does not allocate it");
        assert_eq!(refused(gpu(&[m])?)?, same(reach(0)), "a GPU that allocates it and does not reach it");
        // Two GPUs, and the `Cuda` of each in turn: the GPU that cannot reach managed memory is named whichever GPU
        // `managed` is called on, the other one as well as that one, and the first of two that cannot.
        for (first, second, said) in
            [(&[m, c][..], &[m][..], reach(1)), (&[m], &[m, c], reach(0)), (&[m], &[m], reach(0))]
        {
            for on in [0, 1] {
                let said = same(said.clone());
                assert_eq!(refused(machine(on, first, second)?)?, said, "GPU {on} of {first:?} and {second:?}");
            }
        }
        // The GPU that `managed` is called on allocates it, or the error says so, whatever the other GPU says.
        assert_eq!(refused(machine(0, &[], &[m, c])?)?, same(allocate.to_string()), "GPU 0 does not allocate it");
        assert_eq!(refused(machine(1, &[m, c], &[])?)?, same(allocate.to_string()), "GPU 1 does not allocate it");
        // Neither a GPU that does not allocate managed memory nor one that allocates it and reaches it while the host
        // does stops `managed`: the driver is asked for the memory.
        let asked = |cuda: Cuda| -> Result {
            let e = cuda.managed::<u32>(4).err().ok_or("managed gave Ok")?;
            assert_eq!(e.call, "cuMemAllocManaged");
            assert!(!e.message.contains("CU_DEVICE_ATTRIBUTE_"), "{e}");
            Ok(())
        };
        asked(gpu(&[m, c])?)?;
        for on in [0, 1] {
            asked(machine(on, &[m, c], &[m, c])?)?;
        }
        asked(machine(0, &[m, c], &[])?)?;
        asked(machine(1, &[], &[m, c])?)?;
        asked(machine(0, &[m, c], &[c])?)?;
        asked(machine(1, &[c], &[m, c])?)?;
        Ok(())
    }

    /// A GPU that does not use the host's addresses for the host's memory is refused page-locked
    /// memory, in GUIDE.md's words, before the driver is asked for any.
    #[test]
    fn page_locked_memory_is_made_only_for_a_gpu_that_uses_the_hosts_addresses() -> Result {
        let e = gpu(&[])?.pinned::<u32>(4).err().ok_or("pinned gave Ok")?;
        let said = "the GPU cannot reach page-locked memory at the host's addresses: \
                    its CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING is 0";
        assert_eq!(e.to_string(), format!("cuMemHostAlloc: {said}"));
        let e = gpu(&[UNIFIED_ADDRESSING])?.pinned::<u32>(4).err().ok_or("pinned gave Ok")?;
        assert_eq!(e.call, "cuMemHostAlloc");
        Ok(())
    }

    /// A GPU that does not reach pageable memory is refused a slice, in GUIDE.md's words, and
    /// one that does is lent it: the address of its elements is the slice's.
    #[test]
    fn a_slice_is_lent_only_to_a_gpu_that_reaches_pageable_memory() -> Result {
        let mut host = [1u32, 2, 3];
        let at = host.as_ptr() as u64;
        let e = gpu(&[])?.lend(&mut host).err().ok_or("lend gave Ok")?;
        let said = "the GPU cannot reach pageable memory, a slice of the host's: \
                    its CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS is 0";
        assert_eq!(e.to_string(), format!("cuDeviceGetAttribute: {said}"));
        let lent = gpu(&[PAGEABLE_MEMORY_ACCESS])?.lend(&mut host)?;
        assert_eq!((lent.len(), lent.ptr().addr()), (3, at));
        Ok(())
    }

    /// Unified memory of another GPU is neither copied from nor to, nor lent to a kernel, and the error says so, in
    /// GUIDE.md's words.
    #[test]
    fn unified_memory_of_another_gpu_is_neither_copied_from_or_to_nor_lent_to_a_kernel() -> Result {
        let flags = [HOST_REGISTER_SUPPORTED, HOST_POINTER_FOR_REGISTERED_MEM];
        let replace = |api: &mut Api| {
            api.register = register;
            api.unregister = unregister;
            api.load_module = load_module;
            api.unload_module = unload_module;
        };
        let (here, there) = (gpu_in(1, &flags, replace)?, gpu_in(2, &flags, replace)?);
        let (mut mine, mut theirs) =
            (here.register(Vec::leak(vec![0u32; 4]))?, there.register(Vec::leak(vec![0u32; 4]))?);
        let said = "cuMemcpyAsync: the memory is on another GPU than the copy is made on";
        let e = here.copy(&mut mine, &theirs).err().ok_or("copy gave Ok")?;
        assert_eq!(e.to_string(), said, "from memory of another GPU");
        let e = here.copy(&mut theirs, &mine).err().ok_or("copy gave Ok")?;
        assert_eq!(e.to_string(), said, "to memory of another GPU");
        let module = Module::load(&here, "", &[], &[])?;
        let e = theirs.lend(&module).err().ok_or("lend gave Ok")?;
        assert_eq!(e.to_string(), "cuLaunchKernel: the memory is on another GPU than the kernels are loaded on");
        assert_eq!(mine.lend(&module)?.1, 4, "memory of this GPU is lent");
        Ok(())
    }

    /// A buffer of four words that the fake driver gave no address, as a side of a copy: the driver frees none, and its
    /// `cuMemFree` refuses, which a drop ignores.
    fn buffer_of(cuda: &Cuda) -> Buffer<u32> {
        Buffer { driver: cuda.driver.clone(), ptr: 4096, len: 4, reservation: None, of: PhantomData }
    }

    /// A copy of managed memory, and of page-locked memory, either way, and a zeroing of managed memory are each asked
    /// of the driver, with the memory's own address.
    #[test]
    fn managed_and_page_locked_memory_are_copied_and_managed_memory_is_zeroed_by_the_driver() -> Result {
        let flags = [MANAGED_MEMORY, UNIFIED_ADDRESSING, CONCURRENT_MANAGED_ACCESS];
        let cuda = gpu_with(&flags, |api| {
            api.alloc_managed = alloc_managed;
            api.alloc_host = alloc_host;
            api.memset = memset;
            api.copy_async = copy_async;
            api.synchronize = synchronize;
        })?;
        let (mut managed, mut pinned) = (cuda.managed::<u32>(4)?, cuda.pinned::<u32>(4)?);
        let mut buffer = buffer_of(&cuda);
        managed.host_mut()?.fill(7);
        COPIED.set(0);
        cuda.copy(&mut buffer, &managed)?;
        cuda.copy(&mut managed, &buffer)?;
        cuda.copy(&mut buffer, &pinned)?;
        cuda.copy(&mut pinned, &buffer)?;
        assert_eq!(COPIED.get(), 4);
        managed.zero()?;
        assert_eq!(managed.host()?, [0; 4]);
        Ok(())
    }

    /// A kernel that is no `unsafe fn` is lent a buffer, a copy of buffers is queued and a buffer is zeroed, on a GPU
    /// that reaches managed memory while the host does, with a slice of managed memory held, and the kernel is lent
    /// that memory too. On a GPU that does not there is no slice to hold, as `managed` makes none for it, and the same
    /// work on buffers goes to the driver as on the other.
    #[test]
    fn work_on_buffers_goes_on_beside_a_slice_of_managed_memory_where_there_can_be_one_and_where_not() -> Result {
        for reaches in [false, true] {
            let flags = [MANAGED_MEMORY, UNIFIED_ADDRESSING, CONCURRENT_MANAGED_ACCESS];
            let cuda = gpu_with(&flags[..2 + usize::from(reaches)], |api| {
                api.alloc_managed = alloc_managed;
                api.memset = memset;
                api.copy_async = copy_async;
                api.synchronize = synchronize;
                api.load_module = load_module;
                api.unload_module = unload_module;
            })?;
            let module = Module::load(&cuda, "", &[], &[])?;
            let managed = cuda.managed::<u32>(4);
            let held = managed.as_ref().ok().map(|managed| managed.host()).transpose()?;
            assert_eq!(
                held.is_some(),
                reaches,
                "a slice of managed memory is held where the GPU reaches it with the host"
            );

            let (block, from) = (dirty(16), buffer_of(&cuda));
            let mut to =
                Buffer { driver: cuda.driver.clone(), ptr: block as u64, len: 4, reservation: None, of: PhantomData };
            COPIED.set(0);
            assert_eq!(from.lend(&module)?.1, 4, "a kernel lent a buffer, reaches {reaches}");
            if let Ok(managed) = &managed {
                assert_eq!(managed.lend(&module)?.1, 4, "a kernel lent managed memory");
            }
            cuda.copy(&mut to, &from)?;
            assert_eq!(COPIED.get(), 1, "a copy of buffers, reaches {reaches}");
            to.zero()?;
            // SAFETY: `block` is a block of 16 bytes that the fake driver leaked, and it is read as bytes.
            let zeroed = unsafe { std::slice::from_raw_parts(block, 16) };
            assert_eq!(zeroed, [0; 16], "a zeroing of a buffer, reaches {reaches}");
        }
        Ok(())
    }

    /// A copy from or to a slice that `lend` lent is refused, in GUIDE.md's words, and the driver is not asked: it
    /// reads such a slice as the call is made and writes it after the call waits for the stream, so the copy is not
    /// queued in order. A slice that `register` page-locked is copied, and a zeroing of a lent slice is queued.
    #[test]
    fn a_slice_that_lend_lent_is_not_copied_from_or_to_and_one_that_register_locked_is() -> Result {
        let flags = [PAGEABLE_MEMORY_ACCESS, HOST_REGISTER_SUPPORTED, HOST_POINTER_FOR_REGISTERED_MEM];
        let cuda = gpu_with(&flags, |api| {
            api.register = register;
            api.unregister = unregister;
            api.memset = memset;
            api.copy_async = copy_async;
            api.synchronize = synchronize;
        })?;
        let said = "cuMemcpyAsync: a slice that `lend` lent is not copied from or to, as the driver reads it when the \
                    call is made, and writes it after the call waits for the stream: `register` page-locks the slice \
                    where it is, and its copies are queued in order";
        let (mut lent, mut buffer) = (cuda.lend(Vec::leak(vec![7u32; 4]))?, buffer_of(&cuda));
        COPIED.set(0);
        let e = cuda.copy(&mut buffer, &lent).err().ok_or("copy gave Ok")?;
        assert_eq!(e.to_string(), said, "from a slice lent");
        let e = cuda.copy(&mut lent, &buffer).err().ok_or("copy gave Ok")?;
        assert_eq!(e.to_string(), said, "to a slice lent");
        assert_eq!(COPIED.get(), 0, "the driver was asked for a copy of a slice lent");
        let mut registered = cuda.register(Vec::leak(vec![7u32; 4]))?;
        cuda.copy(&mut buffer, &registered)?;
        cuda.copy(&mut registered, &buffer)?;
        assert_eq!(COPIED.get(), 2, "the driver was not asked for the copies of a slice registered");
        // A zeroing of the slice lent is queued as any is, and the host sees the zeros after the wait.
        assert_eq!(lent.host()?, [7; 4]);
        lent.zero()?;
        assert_eq!(lent.host()?, [0; 4]);
        Ok(())
    }

    /// A GPU that does not page-lock memory the host has, or does not reach it at the host's addresses, is refused a
    /// slice, in GUIDE.md's words, and the driver is not asked to page-lock it.
    #[test]
    fn a_slice_is_registered_only_for_a_gpu_that_page_locks_it_and_reaches_it_at_the_hosts_addresses() -> Result {
        let mut host = [1u32, 2, 3];
        let said = |why: &str| format!("cuMemHostRegister: {why}");
        let cannot =
            "the GPU cannot page-lock memory the host has: its CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED is 0";
        // A GPU that does not page-lock it is refused first, whatever else it says.
        for flags in [&[][..], &[HOST_POINTER_FOR_REGISTERED_MEM]] {
            let e = registering(flags, register)?.register(&mut host).err().ok_or("register gave Ok")?;
            assert_eq!(e.to_string(), said(cannot));
        }
        let cannot = "the GPU cannot reach page-locked memory the host has at the host's addresses: \
                      its CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM is 0";
        let e =
            registering(&[HOST_REGISTER_SUPPORTED], register)?.register(&mut host).err().ok_or("register gave Ok")?;
        assert_eq!(e.to_string(), said(cannot));
        assert!(REGISTERED.with_borrow(Vec::is_empty), "the driver was not asked");
        Ok(())
    }

    /// A slice is page-locked where it is, once, by its address and its bytes, for the GPU to reach: the address of its
    /// elements is the slice's. It is unlocked once, at the same address, when the memory is dropped, and not before.
    #[test]
    fn a_slice_is_registered_once_where_it_is_and_unregistered_once_when_dropped() -> Result {
        let cuda = registering(&[HOST_REGISTER_SUPPORTED, HOST_POINTER_FOR_REGISTERED_MEM], register)?;
        let mut host = [1u32, 2, 3];
        let at = host.as_ptr().addr();
        let registered = cuda.register(&mut host)?;
        assert_eq!((registered.len(), registered.ptr().addr()), (3, at as u64));
        // 2 is CU_MEMHOSTREGISTER_DEVICEMAP.
        assert_eq!(REGISTERED.take(), [(at, 12, 2)]);
        assert!(UNREGISTERED.with_borrow(Vec::is_empty), "it is locked until it is dropped");
        drop(registered);
        assert_eq!(UNREGISTERED.take(), [at]);
        assert!(REGISTERED.with_borrow(Vec::is_empty), "and it is not locked again");
        Ok(())
    }

    /// A slice with no elements has no memory to lock, so the driver is asked neither to lock it nor to unlock it, and
    /// its drop does not wait for the kernels, as that of a slice lent does. It is registered memory, which `prefetch`
    /// says is page-locked, not a slice lent.
    #[test]
    fn a_slice_with_no_elements_is_registered_with_no_call_of_the_driver() -> Result {
        let cuda = registering(&[HOST_REGISTER_SUPPORTED, HOST_POINTER_FOR_REGISTERED_MEM], register)?;
        let none = cuda.register::<u64>(&mut [])?;
        assert!(none.is_empty());
        let e = none.prefetch(0..0, Location::Gpu).err().ok_or("prefetch gave Ok")?;
        let said = "the driver prefetches managed memory alone: this is page-locked memory";
        assert_eq!(e.to_string(), format!("cuMemPrefetchAsync: {said}"));
        drop(none);
        assert!(REGISTERED.with_borrow(Vec::is_empty) && UNREGISTERED.with_borrow(Vec::is_empty));
        assert_eq!(SYNCHRONIZED.get(), 0, "the drop did not wait for the kernels");
        Ok(())
    }

    /// What the driver refuses, it refuses in its own words under the name of the call, and nothing is unlocked that
    /// was never locked.
    #[test]
    fn memory_the_driver_refuses_is_refused_and_not_unregistered() -> Result {
        let cuda = registering(&[HOST_REGISTER_SUPPORTED, HOST_POINTER_FOR_REGISTERED_MEM], register_again)?;
        let e = cuda.register(&mut [1u32, 2, 3]).err().ok_or("register gave Ok")?;
        assert_eq!(e.call, "cuMemHostRegister");
        assert!(e.message.contains("(712)"), "{e}");
        assert!(UNREGISTERED.with_borrow(Vec::is_empty));
        Ok(())
    }

    /// Managed and page-locked memory are zeros, though the driver clears neither: the fake hands
    /// out bytes of 0xAA.
    #[test]
    fn managed_and_page_locked_memory_are_zeros_though_the_driver_clears_neither() -> Result {
        let cuda = gpu_with(&[MANAGED_MEMORY, CONCURRENT_MANAGED_ACCESS, UNIFIED_ADDRESSING], |api| {
            api.alloc_managed = alloc_managed;
            api.alloc_host = alloc_host;
            api.memset = memset;
            api.free = free;
            api.free_host = free_host;
            api.synchronize = synchronize;
        })?;
        assert_eq!(cuda.managed::<u32>(4)?.host()?, [0; 4]);
        assert_eq!(cuda.pinned::<u32>(4)?.host()?, [0; 4]);
        Ok(())
    }

    /// Managed and page-locked memory made with no zeros are as the driver hands them out, the
    /// fake's bytes of 0xAA, and nothing is asked to clear them: the fake has no `cuMemsetD8`.
    #[test]
    fn memory_made_with_no_zeros_is_as_the_driver_hands_it_out() -> Result {
        let cuda = gpu_with(&[MANAGED_MEMORY, CONCURRENT_MANAGED_ACCESS, UNIFIED_ADDRESSING], |api| {
            api.alloc_managed = alloc_managed;
            api.alloc_host = alloc_host;
            api.free = free;
            api.free_host = free_host;
            api.synchronize = synchronize;
        })?;
        // SAFETY: only the test reads the bytes, and it knows them to be the fake's.
        let managed = unsafe { cuda.managed_uninit::<u32>(4) }?;
        assert_eq!(managed.host()?, [0xAAAA_AAAA; 4]);
        drop(managed);
        // SAFETY: as above.
        let pinned = unsafe { cuda.pinned_uninit::<u32>(4) }?;
        assert_eq!(pinned.host()?, [0xAAAA_AAAA; 4]);
        Ok(())
    }

    /// The elements of a buffer from one on are lent with the address of that element and how many are left; a view
    /// from its end is lent with none, and none starts past it.
    #[test]
    fn the_tail_of_a_buffer_is_lent_from_its_element_and_none_starts_past_its_end() -> Result {
        let cuda = gpu_with(&[], |api| {
            api.load_module = load_module;
            api.unload_module = unload_module;
        })?;
        let module = Module::load(&cuda, "", &[], &[])?;
        // The fake driver gave it no address, and frees none: its `cuMemFree` refuses, and a drop ignores that.
        let buffer =
            Buffer::<u32> { driver: cuda.driver.clone(), ptr: 4096, len: 4, reservation: None, of: PhantomData };
        for (from, want) in [(0, (4096, 4)), (1, (4100, 3)), (4, (4112, 0))] {
            let (at, left) = buffer.tail(from).ok_or("tail gave None")?.lend(&module)?;
            assert_eq!((at.addr(), left), want, "from {from}");
        }
        assert!(buffer.tail(5).is_none());
        Ok(())
    }
}
