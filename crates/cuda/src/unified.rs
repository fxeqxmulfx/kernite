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

/// The GPU reaches managed memory while the host does, as the driver needs of a GPU that it moves
/// memory for. [`Cuda::concurrent_managed_access`] says it.
pub(crate) const CONCURRENT_MANAGED_ACCESS: Flag = (89, "CONCURRENT_MANAGED_ACCESS");

impl Cuda {
    /// `len` elements of managed memory, all zero: memory that the host and the GPU's kernels reach
    /// at one address, and the driver moves to whichever touches it. The GPU must allocate managed
    /// memory, as one from Kepler's `sm_30` on does on Linux; the driver says
    /// `CUDA_ERROR_NOT_SUPPORTED` of one that does not.
    pub fn managed<T: Plain>(&self, len: usize) -> Result<Unified<T>, Error> {
        let bytes = bytes::<T>(len);
        let (api, mut addr) = (self.driver.current()?, 0);
        // SAFETY: it writes the address to `addr`. It refuses no bytes, so it is asked for one. 1
        // is CU_MEM_ATTACH_GLOBAL: any stream of any GPU may use it, as the default stream does.
        api.check("cuMemAllocManaged", unsafe { (api.alloc_managed)(&mut addr, bytes.max(1), 1) })?;
        let memory = Unified { driver: self.driver.clone(), ptr: ptr::with_exposed_provenance_mut(addr as usize), len };
        // The driver does not clear it, and the host must not be lent bytes that were never written.
        memory.driver.zero(addr, bytes)?;
        Ok(memory)
    }
}

/// `len` elements of `T` that the host and the kernels of one GPU, the one that made it, both
/// reach at one address, freed when it is dropped: [`Cuda::managed`] makes it. A kernel is given
/// [`ptr`](Unified::ptr), and the host the slices [`host`](Unified::host) and
/// [`host_mut`](Unified::host_mut) lend, with no copy between them.
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
/// [`Module::launch`]: super::Module::launch
pub struct Unified<T: Plain> {
    driver: Rc<Driver>,
    /// The address, which is the host's and the GPU's.
    ptr: *mut T,
    len: usize,
}

impl<T: Plain> Unified<T> {
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

impl<T: Plain> Drop for Unified<T> {
    fn drop(&mut self) {
        // As for a `Buffer`: left, if its GPU's context cannot be made current. The driver waits for
        // the kernels that use the memory before it frees it.
        if let Ok(api) = self.driver.current() {
            // SAFETY: `cuMemAllocManaged` gave the address, and it is freed once, here.
            unsafe { (api.free)(self.addr()) };
        }
    }
}

impl<T: Plain> fmt::Debug for Unified<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unified({} elements at {:#x})", self.len, self.addr())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

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

    /// A GPU says it reaches managed memory while the host does as its attribute says.
    #[test]
    fn a_gpu_says_whether_it_reaches_managed_memory_while_the_host_does() {
        assert!(gpu(&[CONCURRENT_MANAGED_ACCESS]).concurrent_managed_access());
        assert!(!gpu(&[]).concurrent_managed_access());
    }
}
