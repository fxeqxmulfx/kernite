//! A kernel whose result is a reference to never, `&'static !`. Its `!` follows a lifetime, so it is the never type
//! and no macro's, and the braces after it are the body. kernite refuses the result, a reference, at the result, and
//! not for the want of a body.

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn reference() -> &'static ! {
        ud2();
    }
}
