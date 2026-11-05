//! Each crate's README, its example run as a doctest, built against the crates as a crate of a
//! user's is: `cargo test -p readmes --doc`.

#[cfg(doctest)]
#[doc = include_str!("../../../crates/core/README.md")]
struct KerniteCore;

#[cfg(doctest)]
#[doc = include_str!("../../../crates/lang/README.md")]
struct KerniteLang;

#[cfg(doctest)]
#[doc = include_str!("../../../crates/build/README.md")]
struct KerniteBuild;

#[cfg(doctest)]
#[doc = include_str!("../../../crates/kernite/README.md")]
struct Kernite;

#[cfg(doctest)]
#[doc = include_str!("../../../crates/macros/README.md")]
struct KerniteMacros;

#[cfg(doctest)]
#[doc = include_str!("../../../crates/cuda/README.md")]
struct KerniteCuda;

#[cfg(doctest)]
#[doc = include_str!("../../../crates/cpu/README.md")]
struct KerniteCpu;
