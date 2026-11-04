//! Memory the host and a GPU's kernels both reach, at one address.

use std::ffi::c_int;

/// Something a GPU can do if it says so, as `cuDeviceGetAttribute` has it: the attribute's
/// number, and its name without the `CU_DEVICE_ATTRIBUTE_` before it. The GPU says so by an
/// attribute that is not 0.
type Flag = (c_int, &'static str);

/// The GPU reaches managed memory while the host does, as the driver needs of a GPU that it moves
/// memory for. [`Cuda::concurrent_managed_access`] says it.
pub(crate) const CONCURRENT_MANAGED_ACCESS: Flag = (89, "CONCURRENT_MANAGED_ACCESS");

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::ptr;
    use std::rc::Rc;

    use super::super::{Api, CuResult, Cuda, Driver, Handle};
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
