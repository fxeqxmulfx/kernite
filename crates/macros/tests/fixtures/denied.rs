//! What a kernel says of lints that a build without a word cannot show: each `deny`, `forbid` and
//! `warn`, which say a word when their lint fires. The launcher keeps each, and Rust says each
//! lint of `n` once, there: `k_blocks` keeps none of them, and Rust lints neither its name nor
//! the compile-time arguments it takes.

/// A compile-time argument named in lower case, which the kernel denies: the launcher errs, and
/// `k_blocks` says nothing.
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

/// A module that denies the lint, with a kernel that only warns of it: the launcher warns, and
/// `k_blocks` says nothing.
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
