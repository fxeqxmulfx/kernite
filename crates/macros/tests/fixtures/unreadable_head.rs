//! A module kernite cannot read, for the head of a kernel it does not take, `bad<T>`, beside a kernel it does, `good`:
//! code that names `good` by the module's path, by `use` and by a glob.

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn good(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "sysv64" fn bad<T>(x: u64) -> u64 {
        ret(x);
    }
}

pub fn by_path() -> u64 {
    unsafe { crate::cpu::good(1) }
}

pub mod by_use {
    use crate::cpu::good;

    pub fn f() -> u64 {
        unsafe { good(1) }
    }
}

pub mod by_glob {
    use crate::cpu::*;

    pub fn f() -> u64 {
        unsafe { good(1) }
    }
}
