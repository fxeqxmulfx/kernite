//! Modules kernite cannot read, for the head of a kernel it does not take, `bad<T>`, that have a static. Two have read
//! a function that names an ABI, one before `bad` and one after, so they know the static; the third has not, so it
//! knows none. Rust says nothing of the code that names the first two by a path, by `use` or by a glob, nor of the code
//! that names the third by a path or by `use`, and cannot find it by a glob. Nor does Rust ask for `kernite-cuda`,
//! which no module needs.

#[kernite_macros::kernels]
pub mod before_the_head {
    pub unsafe extern "sysv64" fn first(x: u64) -> u64 {
        ret(x);
    }

    pub static COUNTS: global<[u32; 4]>;

    pub unsafe extern "sysv64" fn bad<T>(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod after_the_head {
    pub static COUNTS: global<[u32; 4]>;

    pub unsafe extern "sysv64" fn bad<T>(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "sysv64" fn last(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod no_other {
    pub static COUNTS: global<[u32; 4]>;

    pub unsafe extern "sysv64" fn bad<T>(x: u64) -> u64 {
        ret(x);
    }
}

pub mod by_path_and_use {
    use crate::after_the_head::COUNTS as AFTER;
    use crate::before_the_head::COUNTS as BEFORE;
    use crate::no_other::COUNTS as NONE;

    pub fn f() -> usize {
        BEFORE.len() + AFTER.len() + NONE.len() + crate::no_other::COUNTS.len()
    }
}

pub mod known_by_glob {
    use crate::after_the_head::*;

    pub fn f() -> usize {
        COUNTS.len() + unsafe { last(1) } as usize
    }
}

pub mod also_known_by_glob {
    use crate::before_the_head::*;

    pub fn f() -> usize {
        COUNTS.len() + unsafe { first(1) } as usize
    }
}

/// The import is unused, for it finds nothing.
pub mod unknown_by_glob {
    use crate::no_other::*;

    pub fn f() -> usize {
        COUNTS.len()
    }
}
