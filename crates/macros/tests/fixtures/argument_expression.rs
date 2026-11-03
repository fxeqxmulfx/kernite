//! A kernel whose pointer's type is an expression of its compile-time argument, `[u8; N + 1]`. Rust
//! takes no such type in any generic function: an argument goes in a type alone, `N` or `{ N }`. So
//! the macro says nothing of its own, and Rust's error is at the type.

#[kernite_macros::kernels]
pub mod cpu {
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn arr<const N: usize>(p: *mut [u8; N + 1]) {
        mov(byte[p], N);
    }
}
