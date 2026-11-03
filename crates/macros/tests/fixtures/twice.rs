//! A cooperative GPU kernel deprecated twice, which Rust refuses at the second `#[deprecated]`.
//! `k_blocks` keeps both where the kernel has them, so the error is said there alone: for the
//! launcher and for `k_blocks`, at one place, which Cargo's short form prints as one line.

#[kgen_macros::kernels]
pub mod twice {
    #[cooperative]
    #[deprecated]
    #[deprecated]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}
