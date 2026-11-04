//! A reference with a lifetime to a primitive written as a path, `&'static ::core::primitive::u8`, is the type
//! `&'static u8` to kernite: a lifetime's name begins no path, so the path after it, from the root or not, is read as
//! the primitive's, as it is anywhere. A kernel takes no reference, so kernite refuses it, and says the type by the
//! primitive's name, as it does of the same type written by the name, which each module here has beside the paths.

/// x86 kernels, by the paths and by the names.
#[kernite_macros::kernels]
pub mod x86 {
    pub unsafe extern "sysv64" fn rooted(x: &'static ::core::primitive::u8) {}
    pub unsafe extern "sysv64" fn plain(x: &'static core::primitive::u8) {}
    pub unsafe extern "sysv64" fn named(x: &'static u8) {}
    pub unsafe extern "sysv64" fn bound(x: for<'a> fn(&'a ::std::primitive::u8) -> &'a ::std::primitive::u16) {}
    pub unsafe extern "sysv64" fn bound_named(x: for<'a> fn(&'a u8) -> &'a u16) {}
}

/// GPU kernels, by the paths and by the names.
#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn rooted(x: &'static ::std::primitive::str) {}
    pub unsafe extern "ptx-kernel" fn plain(x: &'static std::primitive::str) {}
    pub unsafe extern "ptx-kernel" fn named(x: &'static str) {}
}
