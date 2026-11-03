//! The method that says how many blocks a cooperative GPU kernel's grid has at most, `k_blocks`,
//! keeps what the kernel says of lints and of whether it is there, and nothing else. It keeps
//! each `allow`, `warn`, `deny` and `forbid` as written, each `expect` as an `allow`, and `cfg`,
//! `deprecated` and `doc(hidden)`; each alone, in a `cfg_attr(..)`, or written by a macro. A name
//! written raw, `r#allow`, is the name. It leaves out the kernel's docs, and what the kernel says
//! of its code or its symbol. Each case is a module of its own, and Rust builds each as written,
//! without a word.

/// A compile-time argument named in lower case, which the kernel allows.
#[kgen_macros::kernels]
pub mod lower {
    #[instances(1, 2)]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// The same, named raw, `r#in`.
#[kgen_macros::kernels]
pub mod raw {
    #[instances(1, 2)]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const r#in: u32>(xs: *mut u32) {
        st.global.u32([xs], r#in);
    }
}

/// A parameter named in camel case, which the kernel expects Rust to warn of: the launcher takes
/// it, and `k_blocks` does not.
#[kgen_macros::kernels]
pub mod expected {
    #[cooperative]
    #[expect(non_snake_case)]
    pub unsafe extern "ptx-kernel" fn k(xsPtr: *mut u32) {
        st.global.u32([xsPtr], 1);
    }
}

/// A compile-time argument named in lower case, which the kernel expects Rust to warn of: both
/// methods take it.
#[kgen_macros::kernels]
pub mod met {
    #[instances(1, 2)]
    #[cooperative]
    #[expect(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// Docs, then lints, one warned of and then allowed: `k_blocks` keeps each lint after the docs,
/// in order, and has docs of its own alone.
#[kgen_macros::kernels]
pub mod listed {
    /// What the kernel does, which `k_blocks` does not say.
    #[instances(1, 2)]
    #[cooperative]
    #[allow(non_snake_case)]
    #[warn(non_upper_case_globals)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xsPtr: *mut u32) {
        st.global.u32([xsPtr], n);
    }
}

/// A kernel left out, with a compile-time argument named in lower case and no lint allowed:
/// `k_blocks` goes with its launcher.
#[kgen_macros::kernels]
pub mod gone {
    #[instances(1, 2)]
    #[cooperative]
    #[cfg(false)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// Lints expected in a `cfg_attr(..)`, with a comma after them, of both kinds: `k_blocks` takes
/// `n` and not `xsPtr`.
#[kgen_macros::kernels]
pub mod conditional {
    #[instances(1, 2)]
    #[cooperative]
    #[cfg_attr(all(), expect(non_snake_case, non_upper_case_globals),)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xsPtr: *mut u32) {
        st.global.u32([xsPtr], n);
    }
}

/// The same, in a `cfg_attr(..)` inside another.
#[kgen_macros::kernels]
pub mod nested {
    #[instances(1, 2)]
    #[cooperative]
    #[cfg_attr(all(), cfg_attr(all(), expect(non_snake_case, non_upper_case_globals)))]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xsPtr: *mut u32) {
        st.global.u32([xsPtr], n);
    }
}

/// The same lints, which a macro writes, `#[$lint]` and `#[cfg_attr(all(), $lint)]`: the fragment
/// arrives in an invisible group, and the macro reads the `expect` in it.
macro_rules! written {
    ($lint:meta) => {
        /// The attribute alone.
        #[kgen_macros::kernels]
        pub mod written {
            #[instances(1, 2)]
            #[cooperative]
            #[$lint]
            pub unsafe extern "ptx-kernel" fn k<const n: u32>(xsPtr: *mut u32) {
                st.global.u32([xsPtr], n);
            }
        }

        /// The attribute in a `cfg_attr(..)`.
        #[kgen_macros::kernels]
        pub mod written_in_cfg_attr {
            #[instances(1, 2)]
            #[cooperative]
            #[cfg_attr(all(), $lint)]
            pub unsafe extern "ptx-kernel" fn k<const n: u32>(xsPtr: *mut u32) {
                st.global.u32([xsPtr], n);
            }
        }
    };
}

written!(expect(non_snake_case, non_upper_case_globals));

/// Docs alone in a `cfg_attr(..)`: `k_blocks` keeps none of it, for a `cfg_attr(..)` of its
/// condition alone is an error.
#[kgen_macros::kernels]
pub mod conditional_docs {
    #[cooperative]
    #[cfg_attr(all(), doc = "What the kernel does, which `k_blocks` does not say.")]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// Docs and a lint in a `cfg_attr(..)`: `k_blocks` keeps the lint, and has docs of its own alone.
#[kgen_macros::kernels]
pub mod conditional_docs_and_lint {
    #[instances(1, 2)]
    #[cooperative]
    #[cfg_attr(all(), doc = "What the kernel does, which `k_blocks` does not say.", allow(non_upper_case_globals))]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// A lint allowed by a name written raw, `r#allow`, which Rust reads as `allow`.
#[kgen_macros::kernels]
pub mod raw_allow {
    #[instances(1, 2)]
    #[cooperative]
    #[r#allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// Docs, then a lint expected in a `cfg_attr(..)`, each name written raw: `k_blocks` keeps the
/// lint, and has docs of its own alone.
#[kgen_macros::kernels]
pub mod raw_names {
    #[r#doc = "What the kernel does, which `k_blocks` does not say."]
    #[instances(1, 2)]
    #[cooperative]
    #[r#cfg_attr(all(), r#expect(non_upper_case_globals))]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// A kernel hidden from the docs: `k_blocks` is hidden with it.
#[kgen_macros::kernels]
pub mod hidden {
    #[cooperative]
    #[doc(hidden)]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// The same, by names written raw, `r#doc(r#hidden)`.
#[kgen_macros::kernels]
pub mod raw_hidden {
    #[cooperative]
    #[r#doc(r#hidden)]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// A kernel hidden from the docs among other words of its `doc(..)`: `k_blocks` is hidden with it,
/// and takes none of the others, here an alias that would be its own name.
#[kgen_macros::kernels]
pub mod hidden_aliased {
    #[cooperative]
    #[doc(hidden, alias = "k_blocks")]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// A kernel found in the docs by another name, `k_blocks`: the method of that name does not take
/// the alias, which would be its own.
#[kgen_macros::kernels]
pub mod aliased {
    #[cooperative]
    #[doc(alias = "k_blocks")]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// A kernel deprecated: `k_blocks` is too, and a call of it warns so, as expected below.
#[kgen_macros::kernels]
pub mod retired {
    #[cooperative]
    #[deprecated]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// How many blocks a grid of `retired`'s kernel has at most, of 256 threads each.
#[expect(deprecated)]
pub fn retired_blocks(kernels: &retired::Kernels) -> Result<u32, kgen_cuda::Error> {
    kernels.k_blocks(256, 0)
}

/// A kernel whose launcher needs a CPU feature: `k_blocks` does not, and is called without
/// `unsafe`, below.
#[kgen_macros::kernels]
pub mod featured {
    #[cooperative]
    #[target_feature(enable = "avx2")]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// How many blocks a grid of `featured`'s kernel has at most, of 256 threads each.
pub fn featured_blocks(kernels: &featured::Kernels) -> Result<u32, kgen_cuda::Error> {
    kernels.k_blocks(256, 0)
}

/// A kernel whose launcher has a symbol of its own: `k_blocks` does not take it as well.
#[kgen_macros::kernels]
pub mod exported {
    #[cooperative]
    #[unsafe(export_name = "blocks_exported_k")]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}
