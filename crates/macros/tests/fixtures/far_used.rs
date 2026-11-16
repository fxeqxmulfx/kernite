//! A static past `rip`'s reach is a static of an `extern` block to Rust, which refuses `#[used]` on it.

#[kernite_macros::kernels]
pub mod cpu {
    /// 3 GiB that no kernel names, kept all the same.
    #[used]
    pub static BIG: global<[u8; 3221225472]>;

    /// Does nothing, in a module of x86 kernels.
    pub unsafe extern "sysv64" fn nothing() {}
}
