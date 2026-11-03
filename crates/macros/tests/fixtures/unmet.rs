//! A cooperative GPU kernel whose `#[expect(non_snake_case)]` nothing meets, in a crate that denies
//! an expectation unfulfilled. The launcher keeps the `expect` as written, so Rust refuses it
//! there, once: `k_blocks` keeps no lint.

#![deny(unfulfilled_lint_expectations)]

#[kernite_macros::kernels]
pub mod gpu {
    #[cooperative]
    #[expect(non_snake_case)]
    pub unsafe extern "ptx-kernel" fn k(x: *mut u32) {
        st.global.u32([x], 1);
    }
}
