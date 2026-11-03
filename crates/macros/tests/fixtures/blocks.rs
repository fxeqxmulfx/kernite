//! The method that says how many blocks a cooperative GPU kernel's grid has at most, `k_blocks`,
//! keeps what the kernel says of whether it is there, and nothing else: `cfg`, `deprecated` and
//! `doc(hidden)`; each alone, in a `cfg_attr(..)`, or written by a macro. A name written raw,
//! `r#doc`, is the name. It leaves out the kernel's docs, what the kernel says of its code or its
//! symbol, and what it says of lints: Rust lints neither the name of `k_blocks` nor the
//! compile-time arguments it takes, and says a lint of those at the launcher alone. Each case is
//! a module of its own, and Rust builds each as written, without a word.

/// A compile-time argument named in lower case, which the kernel allows.
#[kernite_macros::kernels]
pub mod lower {
    #[instances(1, 2)]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// The same, named raw, `r#in`.
#[kernite_macros::kernels]
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
#[kernite_macros::kernels]
pub mod expected {
    #[cooperative]
    #[expect(non_snake_case)]
    pub unsafe extern "ptx-kernel" fn k(xsPtr: *mut u32) {
        st.global.u32([xsPtr], 1);
    }
}

/// A compile-time argument named in lower case, which the kernel expects Rust to warn of: both
/// methods take it.
#[kernite_macros::kernels]
pub mod met {
    #[instances(1, 2)]
    #[cooperative]
    #[expect(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// Docs, then lints, one warned of and then allowed: `k_blocks` keeps none of them, and has docs
/// of its own alone.
#[kernite_macros::kernels]
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

/// A kernel left out: `k_blocks` goes with its launcher, and the method of that name that the
/// crate has of its own, below, is the only one.
#[kernite_macros::kernels]
pub mod gone {
    #[cooperative]
    #[cfg(false)]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// `gone`'s `Kernels` with a method `k_blocks` of the crate's own: Rust would say the name is
/// defined twice if the kernel's had stayed.
impl gone::Kernels {
    /// Does nothing.
    pub fn k_blocks(&self) {}
}

/// Lints expected of both kinds and the kernel hidden, in a `cfg_attr(..)` with a comma after
/// them: `k_blocks` takes `n` and not `xsPtr`, keeps none of the lints, and is hidden with the
/// kernel.
#[kernite_macros::kernels]
pub mod conditional {
    #[instances(1, 2)]
    #[cooperative]
    #[cfg_attr(all(), expect(non_snake_case, non_upper_case_globals), doc(hidden),)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xsPtr: *mut u32) {
        st.global.u32([xsPtr], n);
    }
}

/// The same, in a `cfg_attr(..)` inside another.
#[kernite_macros::kernels]
pub mod nested {
    #[instances(1, 2)]
    #[cooperative]
    #[cfg_attr(all(), cfg_attr(all(), expect(non_snake_case, non_upper_case_globals), doc(hidden)))]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xsPtr: *mut u32) {
        st.global.u32([xsPtr], n);
    }
}

/// The kernel hidden by an attribute a macro writes, `#[$kept]` and `#[cfg_attr(all(), $kept)]`:
/// the fragment arrives in an invisible group, and the macro reads the `doc(hidden)` in it.
macro_rules! written {
    ($kept:meta) => {
        /// The attribute alone.
        #[kernite_macros::kernels]
        pub mod written {
            #[cooperative]
            #[$kept]
            pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
                st.global.u32([xs], 1);
            }
        }

        /// The attribute in a `cfg_attr(..)`.
        #[kernite_macros::kernels]
        pub mod written_in_cfg_attr {
            #[cooperative]
            #[cfg_attr(all(), $kept)]
            pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
                st.global.u32([xs], 1);
            }
        }
    };
}

written!(doc(hidden));

/// Docs alone in a `cfg_attr(..)`: `k_blocks` keeps none of it, for a `cfg_attr(..)` of its
/// condition alone is an error.
#[kernite_macros::kernels]
pub mod conditional_docs {
    #[cooperative]
    #[cfg_attr(all(), doc = "What the kernel does, which `k_blocks` does not say.")]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// Docs and a lint in a `cfg_attr(..)`: `k_blocks` keeps neither, and has docs of its own alone.
#[kernite_macros::kernels]
pub mod conditional_docs_and_lint {
    #[instances(1, 2)]
    #[cooperative]
    #[cfg_attr(all(), doc = "What the kernel does, which `k_blocks` does not say.", allow(non_upper_case_globals))]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// A lint allowed by a name written raw, `r#allow`, which Rust reads as `allow`.
#[kernite_macros::kernels]
pub mod raw_allow {
    #[instances(1, 2)]
    #[cooperative]
    #[r#allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const n: u32>(xs: *mut u32) {
        st.global.u32([xs], n);
    }
}

/// Docs, then a lint expected in a `cfg_attr(..)`, each name written raw: `k_blocks` keeps
/// neither, and has docs of its own alone.
#[kernite_macros::kernels]
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
#[kernite_macros::kernels]
pub mod hidden {
    #[cooperative]
    #[doc(hidden)]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// The same, by names written raw, `r#doc(r#hidden)`.
#[kernite_macros::kernels]
pub mod raw_hidden {
    #[cooperative]
    #[r#doc(r#hidden)]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// A kernel hidden from the docs among other words of its `doc(..)`: `k_blocks` is hidden with it,
/// and takes none of the others, here an alias that would be its own name.
#[kernite_macros::kernels]
pub mod hidden_aliased {
    #[cooperative]
    #[doc(hidden, alias = "k_blocks")]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// A kernel found in the docs by another name, `k_blocks`: the method of that name does not take
/// the alias, which would be its own.
#[kernite_macros::kernels]
pub mod aliased {
    #[cooperative]
    #[doc(alias = "k_blocks")]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// A kernel deprecated: `k_blocks` is too, and a call of it warns so, as expected below.
#[kernite_macros::kernels]
pub mod retired {
    #[cooperative]
    #[deprecated]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// How many blocks a grid of `retired`'s kernel has at most, of 256 threads each.
#[expect(deprecated)]
pub fn retired_blocks(kernels: &retired::Kernels) -> Result<u32, kernite_cuda::Error> {
    kernels.k_blocks(256, 0)
}

/// A kernel whose launcher needs a CPU feature: `k_blocks` does not, and is called without
/// `unsafe`, below.
#[kernite_macros::kernels]
pub mod featured {
    #[cooperative]
    #[target_feature(enable = "avx2")]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// How many blocks a grid of `featured`'s kernel has at most, of 256 threads each.
pub fn featured_blocks(kernels: &featured::Kernels) -> Result<u32, kernite_cuda::Error> {
    kernels.k_blocks(256, 0)
}

/// A kernel whose launcher has a symbol of its own: `k_blocks` does not take it as well.
#[kernite_macros::kernels]
pub mod exported {
    #[cooperative]
    #[unsafe(export_name = "blocks_exported_k")]
    pub unsafe extern "ptx-kernel" fn k(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}
