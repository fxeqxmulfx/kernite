//! A cooperative GPU kernel with an attribute that Rust refuses where the kernel has it: an
//! `expect` that names no lint, alone and in a `cfg_attr(..)`, and a `cfg_attr(..)` whose condition
//! is malformed. `k_blocks` does not keep the `expect` of no lint, so its error is said once, and
//! not again as an `allow` the user did not write. It keeps the `cfg_attr(..)`, with the lint after
//! its condition, so that error is said once too, where the user wrote it, and not again where the
//! macro is. A malformed attribute stops the build before lints run, so no lint is in this fixture.

/// An `expect` of no lint.
#[kgen_macros::kernels]
pub mod bare {
    #[cooperative]
    #[expect]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// The same, in a `cfg_attr(..)`.
#[kgen_macros::kernels]
pub mod conditional {
    #[cooperative]
    #[cfg_attr(all(), expect)]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// A `cfg_attr(..)` whose condition, `not()`, has nothing to negate, and a lint after it.
#[kgen_macros::kernels]
pub mod condition {
    #[cooperative]
    #[cfg_attr(not(), allow(non_snake_case))]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}
