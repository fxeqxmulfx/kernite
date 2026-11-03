//! What `k_blocks` keeps of lints that a build without a word cannot show: each `deny`, `forbid`
//! and `warn`, which say a word when their lint fires. Rust says each lint of `n` twice, for the
//! launcher and for `k_blocks`, at one place: Cargo's short form prints the two as one line, and
//! counts both.

/// A compile-time argument named in lower case, which the kernel denies: `k_blocks` errs with the
/// launcher, and does not warn.
#[kgen_macros::kernels]
pub mod denied {
    #[instances(1, 2)]
    #[cooperative]
    #[deny(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// The same, forbidden.
#[kgen_macros::kernels]
pub mod forbidden {
    #[instances(1, 2)]
    #[cooperative]
    #[forbid(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// A module that denies the lint, with a kernel that only warns of it: `k_blocks` warns with the
/// launcher, and does not err.
#[deny(non_upper_case_globals)]
pub mod strict {
    /// The kernel that warns.
    #[kgen_macros::kernels]
    pub mod warned {
        #[instances(1, 2)]
        #[cooperative]
        #[warn(non_upper_case_globals)]
        pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
            st.global.u32([xs], n);
        }
    }
}
