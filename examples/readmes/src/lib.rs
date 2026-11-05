//! Each crate's README, its example run as a doctest, built against the crates as a crate of a
//! user's is: `cargo test -p readmes --doc`.

#[cfg(doctest)]
#[doc = include_str!("../../../crates/lang/README.md")]
struct KerniteLang;
