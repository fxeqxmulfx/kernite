//! An `r#` that no name follows, in a module of kernels: Rust's lexer refuses it, and stops the crate.

#[kernite_macros::kernels]
pub mod x86 {
    pub unsafe extern "sysv64" fn k(x: u64, r#: u64) -> u64 {
        ret(x);
    }
}
