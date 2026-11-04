//! Memory the host and a GPU's kernels both reach, at one address.

use std::ffi::c_int;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;
use std::slice;

use super::{Cuda, Driver, Error, Plain, Ptr, bytes};

/// Something a GPU can do if it says so, as `cuDeviceGetAttribute` has it: the attribute's
/// number, and its name without the `CU_DEVICE_ATTRIBUTE_` before it. The GPU says so by an
/// attribute that is not 0.
type Flag = (c_int, &'static str);

/// The GPU allocates managed memory.
const MANAGED_MEMORY: Flag = (83, "MANAGED_MEMORY");

/// The GPU uses the host's addresses for the host's memory, so that page-locked memory has one
/// address for both.
const UNIFIED_ADDRESSING: Flag = (41, "UNIFIED_ADDRESSING");

/// The GPU reaches any pageable memory of the host, as one with HMM does.
const PAGEABLE_MEMORY_ACCESS: Flag = (88, "PAGEABLE_MEMORY_ACCESS");

/// The GPU reaches managed memory while the host does, as the driver needs of a GPU that it moves
/// memory for. [`Cuda::concurrent_managed_access`] says it.
pub(crate) const CONCURRENT_MANAGED_ACCESS: Flag = (89, "CONCURRENT_MANAGED_ACCESS");

impl Driver {
    /// `Ok` if this GPU has the `flag`; or the error that says the `call` is refused, as the GPU
    /// `cannot` do it, and why.
    fn require(&self, call: &'static str, (attribute, name): Flag, cannot: &str) -> Result<(), Error> {
        if self.attribute(attribute)? == 0 {
            return Err(Error::new(call, format!("{cannot}: its CU_DEVICE_ATTRIBUTE_{name} is 0")));
        }
        Ok(())
    }
}

impl Cuda {
    /// `len` elements of managed memory, all zero: memory that the host and the GPU's kernels
    /// reach at one address, and the driver moves to whichever touches it. The GPU must allocate
    /// managed memory; the error says so of one that does not: `cuMemAllocManaged: the GPU cannot
    /// allocate managed memory: its CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY is 0`.
    pub fn managed<T: Plain>(&self, len: usize) -> Result<Unified<'static, T>, Error> {
        self.driver.require("cuMemAllocManaged", MANAGED_MEMORY, "the GPU cannot allocate managed memory")?;
        let bytes = bytes::<T>(len);
        let (api, mut addr) = (self.driver.current()?, 0);
        // SAFETY: it writes the address to `addr`. It refuses no bytes, so it is asked for one. 1
        // is CU_MEM_ATTACH_GLOBAL: any stream of any GPU may use it, as the default stream does.
        api.check("cuMemAllocManaged", unsafe { (api.alloc_managed)(&mut addr, bytes.max(1), 1) })?;
        let ptr = ptr::with_exposed_provenance_mut(addr as usize);
        let memory = Unified { driver: self.driver.clone(), ptr, len, kind: Kind::Managed, of: PhantomData };
        // The driver does not clear it, and the host must not be lent bytes that were never
        // written. The GPU writes the zeros: where the GPU does not allow concurrent access, the
        // host must not touch managed memory while a kernel runs, and one may be running.
        memory.driver.zero(addr, bytes)?;
        Ok(memory)
    }

    /// `len` elements of page-locked memory, all zero: memory in the host that stays where it is,
    /// and that a kernel reads and writes across the bus, at the address the host has for it. The
    /// GPU must use the host's addresses for the host's memory; the error says so of one that
    /// does not: `cuMemHostAlloc: the GPU cannot reach page-locked memory at the host's addresses:
    /// its CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING is 0`.
    pub fn pinned<T: Plain>(&self, len: usize) -> Result<Unified<'static, T>, Error> {
        let cannot = "the GPU cannot reach page-locked memory at the host's addresses";
        self.driver.require("cuMemHostAlloc", UNIFIED_ADDRESSING, cannot)?;
        let bytes = bytes::<T>(len);
        let (api, mut host) = (self.driver.current()?, ptr::null_mut());
        // SAFETY: it writes the address to `host`. It refuses no bytes, so it is asked for one. 2
        // is CU_MEMHOSTALLOC_DEVICEMAP: the GPU reaches it.
        api.check("cuMemHostAlloc", unsafe { (api.alloc_host)(&mut host, bytes.max(1), 2) })?;
        let memory =
            Unified { driver: self.driver.clone(), ptr: host.cast::<T>(), len, kind: Kind::Pinned, of: PhantomData };
        // SAFETY: the allocation has room for `len` elements, and any bits are a `T`. No kernel
        // uses memory the host has only now been given.
        unsafe { memory.ptr.write_bytes(0, len) };
        Ok(memory)
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
    /// the memory of a `Vec`, a `Box` or a mapping, and not an array on a thread's stack.
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
    /// # let cuda = kernite_cuda::Cuda::new().unwrap();
    /// let mut xs = vec![1.0f32; 4];
    /// let lent = cuda.lend(&mut xs).unwrap();
    /// let first = xs[0];
    /// let for_a_kernel = lent.ptr();
    /// ```
    pub fn lend<'a, T: Plain>(&self, memory: &'a mut [T]) -> Result<Unified<'a, T>, Error> {
        let cannot = "the GPU cannot reach pageable memory, a slice of the host's";
        self.driver.require("cuDeviceGetAttribute", PAGEABLE_MEMORY_ACCESS, cannot)?;
        let (ptr, len) = (memory.as_mut_ptr(), memory.len());
        Ok(Unified { driver: self.driver.clone(), ptr, len, kind: Kind::Lent, of: PhantomData })
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
}

/// `len` elements of `T` that the host and the kernels of one GPU, the one that made it, both
/// reach at one address, with no copy between them. A kernel is given [`ptr`](Unified::ptr), and
/// the host the slices [`host`](Unified::host) and [`host_mut`](Unified::host_mut). It is one of
/// three kinds, by what makes it:
///
/// - [`Cuda::managed`] makes managed memory, which the driver moves to whichever of the host and
///   the GPU touches it.
/// - [`Cuda::pinned`] makes page-locked memory, which stays in the host's memory, and a kernel
///   reads and writes it across the bus.
/// - [`Cuda::lend`] lends a slice of the host's own memory to the GPU, for as long as `'a`, where
///   the GPU reaches pageable memory. The others are their own: `Unified<'static, T>`.
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
/// # let cuda = kernite_cuda::Cuda::new().unwrap();
/// let mut xs = cuda.managed::<f32>(4).unwrap();
/// let slice = xs.host_mut().unwrap();
/// let for_a_kernel = xs.ptr();
/// slice[0] = 1.0;
/// ```
///
/// A [`Ptr`] is `Copy` and borrows nothing, so the compiler cannot refuse a launch on one taken
/// before the slice:
///
/// ```no_run
/// # let cuda = kernite_cuda::Cuda::new().unwrap();
/// let mut xs = cuda.managed::<f32>(4).unwrap();
/// let for_a_kernel = xs.ptr();
/// let slice = xs.host_mut().unwrap();
/// // A launch on `for_a_kernel` here would run under `slice`.
/// let _ = for_a_kernel;
/// slice[0] = 1.0;
/// ```
///
/// A launch is `unsafe` for that, as for a [`Buffer`](super::Buffer), and its contract, which
/// [`Module::launch`] has as its `# Safety`, is this: each pointer is to a buffer or to unified
/// memory of the GPU, as long as the kernel reads and writes through it; from the launch until the
/// kernel is done, the host holds no slice of memory the kernel writes, and no slice it may write
/// of memory the kernel reads; and where [`Cuda::concurrent_managed_access`] is false, the GPU not
/// reaching managed memory while the host does, the host holds no slice of managed memory at all.
///
/// Dropping the memory waits for the kernels that use it, and then frees it; or, for a slice
/// lent, gives it back to the host.
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
        self.at(0)
    }

    /// The address of its element `i`, for a kernel to be given the part from there on.
    /// Panics if `i` is past its end.
    pub fn at(&self, i: usize) -> Ptr<T> {
        assert!(i <= self.len, "element {i} of memory of {}", self.len);
        Ptr { addr: self.addr() + (i * size_of::<T>()) as u64, of: PhantomData }
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
        // `sync` waited for the kernels launched, and a launch after it is `unsafe` and keeps
        // from writing the memory while the host holds this slice. Any bits are a `T`, as `Plain`
        // promises.
        Ok(unsafe { slice::from_raw_parts(self.ptr, self.len) })
    }

    /// What it holds, as the host reads and writes it, after the kernels launched before have
    /// run: it waits for every kernel launched on the GPU to finish, and says whether one failed.
    pub fn host_mut(&mut self) -> Result<&mut [T], Error> {
        self.driver.sync()?;
        // SAFETY: as for `host`, and nothing else holds a slice of it, as `self` is borrowed
        // for as long as this one is held. A launch after `sync` keeps from reading the memory
        // too, as the host may write it.
        Ok(unsafe { slice::from_raw_parts_mut(self.ptr, self.len) })
    }

    /// The address as the GPU reckons it.
    fn addr(&self) -> u64 {
        self.ptr.addr() as u64
    }
}

impl<T: Plain> Drop for Unified<'_, T> {
    fn drop(&mut self) {
        // As for a `Buffer`: left, if its GPU's context cannot be made current. The driver waits
        // for the kernels that use memory before it frees it; a slice lent is waited for here.
        if let Ok(api) = self.driver.current() {
            // SAFETY: the driver gave the address of managed and of page-locked memory, and each is
            // freed once, here. A slice lent is the host's, which frees it.
            unsafe {
                match self.kind {
                    Kind::Managed => (api.free)(self.addr()),
                    Kind::Pinned => (api.free_host)(self.ptr.cast()),
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
mod tests {
    use std::cell::Cell;
    use std::ffi::{c_uint, c_void};

    use super::super::{Api, CuResult, Handle};
    use super::*;

    thread_local! {
        /// What the GPU here says of each attribute: the attribute's number is the index.
        static ATTRIBUTES: Cell<[c_int; 128]> = const { Cell::new([0; 128]) };
    }

    /// The fake GPU's `cuDeviceGetAttribute`.
    unsafe extern "C" fn attribute(value: *mut c_int, attribute: c_int, _: c_int) -> CuResult {
        // SAFETY: the caller gives a place for the value.
        unsafe { *value = ATTRIBUTES.get()[attribute as usize] };
        0
    }

    /// The fake GPU's `cuCtxSetCurrent`.
    unsafe extern "C" fn set_current(_: Handle) -> CuResult {
        0
    }

    /// A GPU that says each of `flags` is 1 and every other attribute 0, and refuses every call but
    /// the two that tell its attributes and make its context current.
    pub(super) fn gpu(flags: &[Flag]) -> Cuda {
        gpu_with(flags, |_| {})
    }

    /// The same GPU, with the calls `replace` puts in the driver's place.
    pub(super) fn gpu_with(flags: &[Flag], replace: impl FnOnce(&mut Api)) -> Cuda {
        let mut values = [0; 128];
        for (attribute, _) in flags {
            values[*attribute as usize] = 1;
        }
        ATTRIBUTES.set(values);
        let mut api = Api::refusing();
        api.attribute = attribute;
        api.set_current = set_current;
        replace(&mut api);
        let driver = Rc::new(Driver { api, device: 0, context: ptr::without_provenance_mut(1) });
        Cuda::of(driver, String::new()).unwrap()
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

    /// The fake driver's `cuMemsetD8`.
    unsafe extern "C" fn memset(addr: u64, value: u8, bytes: usize) -> CuResult {
        // SAFETY: the caller gives the address of a block of the fake driver's, of `bytes` bytes.
        unsafe { ptr::with_exposed_provenance_mut::<u8>(addr as usize).write_bytes(value, bytes) };
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

    /// The fake driver's `cuCtxSynchronize`: no kernel runs.
    unsafe extern "C" fn synchronize() -> CuResult {
        0
    }

    /// A GPU says it reaches managed memory while the host does as its attribute says, whatever
    /// else it says of itself.
    #[test]
    fn a_gpu_says_whether_it_reaches_managed_memory_while_the_host_does() {
        assert!(gpu(&[CONCURRENT_MANAGED_ACCESS]).concurrent_managed_access());
        assert!(!gpu(&[]).concurrent_managed_access());
        let others = [MANAGED_MEMORY, UNIFIED_ADDRESSING, PAGEABLE_MEMORY_ACCESS];
        assert!(!gpu(&others).concurrent_managed_access());
        assert!(gpu(&[&others[..], &[CONCURRENT_MANAGED_ACCESS]].concat()).concurrent_managed_access());
    }

    /// A GPU that does not allocate managed memory is refused it, in the README's words, before
    /// the driver is asked for any.
    #[test]
    fn managed_memory_is_made_only_for_a_gpu_that_allocates_it() {
        let e = gpu(&[]).managed::<u32>(4).unwrap_err();
        let said = "the GPU cannot allocate managed memory: its CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY is 0";
        assert_eq!(e.to_string(), format!("cuMemAllocManaged: {said}"));
        // One that does is asked, and the driver here refuses everything.
        let e = gpu(&[MANAGED_MEMORY]).managed::<u32>(4).unwrap_err();
        assert_eq!(e.call, "cuMemAllocManaged");
    }

    /// A GPU that does not use the host's addresses for the host's memory is refused page-locked
    /// memory, in the README's words, before the driver is asked for any.
    #[test]
    fn page_locked_memory_is_made_only_for_a_gpu_that_uses_the_hosts_addresses() {
        let e = gpu(&[]).pinned::<u32>(4).unwrap_err();
        let said = "the GPU cannot reach page-locked memory at the host's addresses: \
                    its CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING is 0";
        assert_eq!(e.to_string(), format!("cuMemHostAlloc: {said}"));
        let e = gpu(&[UNIFIED_ADDRESSING]).pinned::<u32>(4).unwrap_err();
        assert_eq!(e.call, "cuMemHostAlloc");
    }

    /// A GPU that does not reach pageable memory is refused a slice, in the README's words, and
    /// one that does is lent it: the address of its elements is the slice's.
    #[test]
    fn a_slice_is_lent_only_to_a_gpu_that_reaches_pageable_memory() {
        let mut host = [1u32, 2, 3];
        let at = host.as_ptr() as u64;
        let e = gpu(&[]).lend(&mut host).unwrap_err();
        let said = "the GPU cannot reach pageable memory, a slice of the host's: \
                    its CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS is 0";
        assert_eq!(e.to_string(), format!("cuDeviceGetAttribute: {said}"));
        let lent = gpu(&[PAGEABLE_MEMORY_ACCESS]).lend(&mut host).unwrap();
        assert_eq!((lent.len(), lent.ptr().addr()), (3, at));
    }

    /// Managed and page-locked memory are zeros, though the driver clears neither: the fake hands
    /// out bytes of 0xAA.
    #[test]
    fn managed_and_page_locked_memory_are_zeros_though_the_driver_clears_neither() {
        let cuda = gpu_with(&[MANAGED_MEMORY, UNIFIED_ADDRESSING], |api| {
            api.alloc_managed = alloc_managed;
            api.alloc_host = alloc_host;
            api.memset = memset;
            api.free = free;
            api.free_host = free_host;
            api.synchronize = synchronize;
        });
        assert_eq!(cuda.managed::<u32>(4).unwrap().host().unwrap(), [0; 4]);
        assert_eq!(cuda.pinned::<u32>(4).unwrap().host().unwrap(), [0; 4]);
    }
}
