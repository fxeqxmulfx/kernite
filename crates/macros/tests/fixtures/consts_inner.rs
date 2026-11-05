//! A module of kernels given consts keeps its own attributes first in it, the docs `//!` and the attributes `#![..]`,
//! as a module that is given none does, and a lint that they forbid is not broken by what the macro writes.

#[kernite_macros::consts]
pub mod sizes {
    /// How far a value moves.
    pub const STEP: u64 = 3;
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod cpu {
    //! Kernels that step.
    #![forbid(missing_docs, unused_imports, dead_code)]

    /// `x` a step on.
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod gpu {
    //! Kernels that step on the GPU.
    #![forbid(missing_docs, unused_imports, dead_code)]

    /// `x` a step on.
    pub unsafe extern "ptx-kernel" fn step(x: *mut u64) {
        let v: u64;
        ld.global.u64(v, [x]);
        add.u64(v, v, STEP);
        st.global.u64([x], v);
    }
}
