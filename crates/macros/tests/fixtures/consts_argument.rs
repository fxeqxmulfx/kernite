//! A module of consts given an argument, `#[consts(x)]`: the argument is told once, where it is, and changes nothing
//! else. The module is the one it is without the argument, for Rust to read and for the modules of kernels that take
//! its consts to read them: Rust says nothing of the code that names its items, wherever it is, nor of the module of
//! kernels given them.

#[kernite_macros::consts(x)]
pub mod sizes {
    pub const N: u32 = 4;
    pub const BOTH: [u32; 2] = [N, 2];
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod cpu {
    pub const BYTES: usize = N as usize * 4;

    pub unsafe extern "sysv64" fn read(x: *mut u32) {
        mov(dword[x], N);
        ret();
    }
}

/// From the crate's root, by the modules' paths.
pub fn from_the_root(x: *mut u32) -> u32 {
    unsafe { cpu::read(x) };
    sizes::N + sizes::BOTH[1] + cpu::BYTES as u32
}

/// From another module, by path and by `use`.
pub mod other {
    use crate::sizes::BOTH;

    pub fn by_path(x: *mut u32) -> u32 {
        unsafe { crate::cpu::read(x) };
        crate::sizes::N + BOTH[0]
    }
}

/// By the names a glob brings in.
pub mod globbed {
    use crate::sizes::*;

    pub fn all() -> u32 {
        N + BOTH[0]
    }
}
