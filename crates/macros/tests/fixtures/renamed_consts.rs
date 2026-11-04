//! A crate that depends on `kernite-macros` under another name, `km`, in its manifest, with a module of consts
//! that no module of kernels takes: it builds without a word, for the macro a module of consts has calls
//! `kernite-macros` back only for a module of kernels given its consts, and none is.

#[km::consts]
pub mod sizes {
    pub const N: u32 = 4;
}
