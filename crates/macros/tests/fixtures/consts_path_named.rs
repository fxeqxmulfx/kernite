//! A module of kernels whose consts' path Rust cannot find, `crate::nope`, and code that names its items by `use`: Rust
//! says of the path once, as for any other use, and of each `use` of an item by name that the module has no such
//! item, for it has none until the path is mended.

#[kernite_macros::kernels(consts = crate::nope)]
pub mod cpu {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "sysv64" fn twice(x: u64) -> u64 {
        add(x, x);
        ret(x);
    }
}

pub mod named {
    use crate::cpu::step;

    pub fn all() -> u64 {
        unsafe { step(1) }
    }
}

pub mod braced {
    use crate::cpu::{self, step, twice};

    pub fn all() -> u64 {
        unsafe { step(1) + cpu::twice(2) + twice(3) }
    }
}

pub mod reexported {
    pub use crate::cpu::step;
}

/// The module itself, imported or renamed, is there.
pub mod itself {
    use crate::cpu;
    pub use crate::cpu as fast;

    pub fn all() -> u64 {
        unsafe { cpu::step(1) + fast::step(2) }
    }
}
