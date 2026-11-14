//! A const with `#[inline]`, beside a kernel that reads it: the macros hand the const to Rust as written, and Rust says
//! what is wrong of it. kernite adds nothing.

#[kernite_macros::kernels]
pub mod reads_a_const {
    #[inline]
    const N: u32 = 1;

    pub unsafe extern "sysv64" fn k() {
        let r: gpr;
        mov(r, N);
    }
}
