//! A cooperative GPU kernel whose `#[expect(non_snake_case)]` the launcher meets, in a crate that
//! forbids that lint: Rust refuses the `expect` once, where the kernel has it, for `k_blocks` keeps
//! no lint.

#![forbid(non_snake_case)]

#[kgen_macros::kernels]
pub mod gpu {
    #[cooperative]
    #[expect(non_snake_case)]
    pub unsafe extern "ptx-kernel" fn k(xsPtr: *mut u32) {
        st.global.u32([xsPtr], 1);
    }
}
