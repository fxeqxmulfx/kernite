//! A cooperative GPU kernel with an attribute that Rust refuses where the kernel has it: an
//! `expect` that names no lint, alone and in a `cfg_attr(..)`, and a `cfg_attr(..)` whose condition
//! is malformed. `k_blocks` keeps no lint level, so the error of an `expect` is said once, where
//! the kernel has it. It keeps the `cfg_attr(..)`, with `deprecated` after its condition, so that
//! error is said once too, where the user wrote it, and not again where the macro is. A malformed
//! attribute stops the build before lints run, so no lint is in this fixture.

/// An `expect` of no lint.
#[kernite_macros::kernels]
pub mod bare {
    #[cooperative]
    #[expect]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// The same, in a `cfg_attr(..)`.
#[kernite_macros::kernels]
pub mod conditional {
    #[cooperative]
    #[cfg_attr(all(), expect)]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// A `cfg_attr(..)` whose condition, `not()`, has nothing to negate, and `deprecated` after it.
#[kernite_macros::kernels]
pub mod condition {
    #[cooperative]
    #[cfg_attr(not(), deprecated)]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}
