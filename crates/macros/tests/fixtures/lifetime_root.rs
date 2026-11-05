//! A lifetime before a path from the root, `&'static ::core::ffi::CStr`, keeps its space where kernite quotes the
//! type in an error, as it does before any other path. The path is no primitive's, so it is quoted as written.

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn rooted(x: &'static ::core::ffi::CStr) {
        ret();
    }

    pub unsafe extern "sysv64" fn plain(x: &'static core::ffi::CStr) {
        ret();
    }

    pub unsafe extern "sysv64" fn bound(x: for<'a> fn(&'a ::core::ffi::CStr)) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn rooted(x: &'static ::core::ffi::CStr) {}
}
