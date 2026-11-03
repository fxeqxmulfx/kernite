//! Modules of kernels that take their consts by each kind of path Rust has, which build without
//! a word. In each, Rust sees the consts of the module the path names, by the `use` the macro
//! writes from inside it, as the fixture asserts as it builds.

extern crate self as paths;

#[kernite_macros::consts]
pub mod sizes {
    /// How far a value moves.
    pub const STEP: u64 = 3;
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod by_crate {
    /// The `STEP` Rust sees here.
    pub const SEEN: u64 = STEP;

    /// `x` a step on.
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

#[kernite_macros::kernels(consts = ::paths::sizes)]
pub mod by_extern {
    /// The `STEP` Rust sees here.
    pub const SEEN: u64 = STEP;

    /// `x` a step on.
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

#[kernite_macros::kernels(consts = sizes)]
pub mod by_name {
    /// The `STEP` Rust sees here.
    pub const SEEN: u64 = STEP;

    /// `x` a step on.
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

/// Modules of kernels beside a module of consts of their own.
pub mod model {
    #[kernite_macros::consts]
    pub mod sizes {
        /// How far a value moves here.
        pub const STEP: u64 = 5;
    }

    #[kernite_macros::kernels(consts = self::sizes)]
    pub mod by_self {
        /// The `STEP` Rust sees here.
        pub const SEEN: u64 = STEP;

        /// `x` a step on.
        pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
            add(x, STEP);
            ret(x);
        }
    }

    #[kernite_macros::kernels(consts = super::sizes)]
    pub mod by_super {
        /// The `STEP` Rust sees here.
        pub const SEEN: u64 = STEP;

        /// `x` a step on, of the crate's module of consts.
        pub unsafe extern "ptx-kernel" fn step(x: *mut u64) {
            let v: u64;
            ld.global.u64(v, [x]);
            add.u64(v, v, STEP);
            st.global.u64([x], v);
        }
    }
}

// The crate's `STEP` by each path but `self::sizes`, which names `model`'s.
const _: () = assert!(by_crate::SEEN == 3 && by_extern::SEEN == 3 && by_name::SEEN == 3);
const _: () = assert!(model::by_self::SEEN == 5 && model::by_super::SEEN == 3);
