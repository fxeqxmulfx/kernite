//! Modules of kernels whose consts' path Rust cannot find, written in each form a path has, and a path of which the
//! first name is there and the second is not, and one that ends at a const: Rust says so once for each, at the name
//! it cannot find, and nothing of the code that names the modules' items.

extern crate self as forms;

pub const SIZE: usize = 4;

mod hidden {
    #[kernite_macros::consts]
    pub mod sizes {
        pub const STEP: u64 = 3;
    }
}

#[kernite_macros::kernels(consts = crate::nope)]
pub mod by_crate {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels(consts = ::forms::nope)]
pub mod by_extern {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels(consts = ::nope::sizes)]
pub mod by_missing_crate {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels(consts = nope)]
pub mod by_name {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels(consts = self::nope)]
pub mod by_self {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels(consts = crate::SIZE)]
pub mod by_const {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels(consts = crate::nope::sizes)]
pub mod by_first {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels(consts = crate::hidden::nope)]
pub mod by_second {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels(consts = crate::hidden::nope::sizes)]
pub mod by_middle {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

/// In a module, where a path beginning `self`, `super` or a name is the module's.
pub mod inner {
    #[kernite_macros::kernels(consts = super::nope)]
    pub mod by_super {
        pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
            ret(x);
        }
    }

    #[kernite_macros::kernels(consts = self::sizes)]
    pub mod by_own {
        pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
            ret(x);
        }
    }

    pub fn from_inside() -> u64 {
        unsafe { by_super::step(1) + by_own::step(2) }
    }
}

/// From the crate's root, each module's kernel by its path.
pub fn from_the_root() -> u64 {
    unsafe {
        by_crate::step(1)
            + by_extern::step(1)
            + by_missing_crate::step(1)
            + by_name::step(1)
            + by_self::step(1)
            + by_const::step(1)
            + by_first::step(1)
            + by_second::step(1)
            + by_middle::step(1)
            + inner::from_inside()
    }
}
