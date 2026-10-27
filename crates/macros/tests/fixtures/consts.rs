//! A module of consts that holds, besides an attribute of its own, only what it refuses: a const
//! that is not `pub`, a `pub` one that is no value known now, and a function. The macro says each
//! where it is, in order, and hands none on, so the module of kernels that takes it says nothing.

#[kgen_macros::consts]
pub mod sizes {
    #![allow(dead_code)]

    /// Half a step, which no module of kernels sees.
    const HALF: u64 = double(1) / 2;

    /// How far a value moves: Rust calls `double` for it, and kgen calls no function for a value.
    pub const STEP: u64 = double(3);

    /// Twice `x`.
    pub const fn double(x: u64) -> u64 {
        2 * x
    }
}

#[kgen_macros::kernels(consts = crate::sizes)]
pub mod kernels {
    /// The module's own, which it may have: `sizes` hands on no `STEP`.
    const STEP: u64 = 3;

    /// `x` a step on.
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}
