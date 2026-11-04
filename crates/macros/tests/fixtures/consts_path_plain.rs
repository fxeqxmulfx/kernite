//! A module of kernels whose consts' path is a module with no `#[kernite_macros::consts]` on it: Rust says so once,
//! at the path, and nothing of the code that names the module's items, wherever it is.

/// A module of consts that lacks its attribute.
pub mod plain {
    pub const STEP: u64 = 3;
}

#[kernite_macros::kernels(consts = crate::plain)]
pub mod cpu {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

/// From the crate's root.
pub fn from_the_root() -> u64 {
    unsafe { cpu::step(1) }
}

/// From another module, by path.
pub mod other {
    pub fn by_path() -> u64 {
        unsafe { crate::cpu::step(2) }
    }
}

/// By the names a glob brings in.
pub mod globbed {
    use crate::cpu::*;

    pub fn all() -> u64 {
        unsafe { step(3) }
    }
}
