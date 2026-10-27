//! Modules of kernels that take their consts by each kind of path Rust has, which build without
//! a word.

extern crate self as paths;

#[kgen_macros::consts]
pub mod sizes {
    /// How far a value moves.
    pub const STEP: u64 = 3;
}

#[kgen_macros::kernels(consts = crate::sizes)]
pub mod by_crate {
    /// `x` a step on.
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

#[kgen_macros::kernels(consts = ::paths::sizes)]
pub mod by_extern {
    /// `x` a step on.
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

#[kgen_macros::kernels(consts = sizes)]
pub mod by_name {
    /// `x` a step on.
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

/// Modules of kernels beside a module of consts of their own.
pub mod model {
    #[kgen_macros::consts]
    pub mod sizes {
        /// How far a value moves here.
        pub const STEP: u64 = 5;
    }

    #[kgen_macros::kernels(consts = self::sizes)]
    pub mod by_self {
        /// `x` a step on.
        pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
            add(x, STEP);
            ret(x);
        }
    }

    #[kgen_macros::kernels(consts = super::sizes)]
    pub mod by_super {
        /// `x` a step on, of the crate's module of consts.
        pub unsafe extern "ptx-kernel" fn step(x: *mut u64) {
            let v: u64;
            ld.global.u64(v, [x]);
            add.u64(v, v, STEP);
            st.global.u64([x], v);
        }
    }
}
