//! A module of kernels that takes consts, and that kernite cannot read, for the head of a kernel it does not take,
//! `bad<T>`: it says so alone, though `STEP`, a kernel of its own, has the name of a const it takes. Rust says nothing
//! of the code that names the rest: `good`, and `COUNTS`, a static whose type reads a const it takes, for a kernel
//! comes before `bad`.

#[kernite_macros::consts]
pub mod sizes {
    pub const STEP: u64 = 3;
    pub const WIDTH: usize = 4;
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod cpu {
    pub static COUNTS: global<[u32; WIDTH]>;

    pub unsafe extern "sysv64" fn STEP(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "sysv64" fn good(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }

    pub unsafe extern "sysv64" fn bad<T>(x: u64) -> u64 {
        ret(x);
    }
}

pub fn by_path() -> u64 {
    unsafe { crate::cpu::good(1) + crate::cpu::COUNTS.len() as u64 }
}

pub mod by_glob {
    use crate::cpu::*;

    pub fn f() -> u64 {
        unsafe { good(1) + COUNTS.len() as u64 }
    }
}
