//! A module of consts given an argument, beside other errors of its own: the argument is told where it is, and
//! changes nothing else, so the module's other errors are said as well, for they are other mistakes. The module is the
//! one it is without the argument, so the module of kernels that takes its consts reads the one that is right, and its
//! own error is said.

#[kernite_macros::consts(x)]
pub mod sizes {
    const N: u32 = 4;
    pub const M: u32 = 2;
    pub fn f() {}
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod cpu {
    pub unsafe extern "sysv64" fn read(x: u64) -> u64 {
        add(x, M);
        add(x, nope);
        ret(x);
    }
}
