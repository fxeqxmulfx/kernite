//! A module of kernels that takes consts, and has items of its own of their names: a function
//! named as one, which the macro refuses at its name, and a struct in braces and a type named as
//! others, which Rust keeps in its other namespace, and of which nothing is said.

#[kernite_macros::consts]
pub mod sizes {
    /// How far a value moves.
    pub const STEP: u64 = 3;

    /// How many values a row holds.
    pub const LEN: usize = 4;

    /// How many rows there are.
    pub const ROWS: usize = 2;
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod kernels {
    /// Named as a const the module takes, which its kernels read.
    pub unsafe extern "sysv64" fn STEP() {}

    /// Named as a const the module takes, and no kernel names it. It is in braces: a unit or a
    /// tuple struct is a value too, which Rust refuses beside the const.
    pub struct LEN {}

    /// Named as another const the module takes, and no kernel names it.
    pub type ROWS = u32;
}
