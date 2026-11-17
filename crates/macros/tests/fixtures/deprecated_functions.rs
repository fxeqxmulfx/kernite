//! A kernel is a function, and Rust tells of a deprecated static at the first place each function writes it: a kernel
//! of instances is a function for each instance, and a helper is part of each kernel that calls it. `deny` counts the
//! errors, where Cargo says a line that two warnings share once.
#![deny(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    #[deprecated]
    pub static EACH: global<[u8; 4096]>;

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(i: u64) {
        let p: gpr;
        lea(p, [EACH]);
        mov(byte[p + i], N);
        lea(p, [EACH]);
        mov(byte[p], N);
    }

    fn touch(p: gpr) {
        lea(p, [EACH]);
    }

    pub unsafe extern "sysv64" fn one(i: u64) {
        let p: gpr;
        mov(p, i);
        touch(p);
        mov(byte[p], 1);
    }

    pub unsafe extern "sysv64" fn two(i: u64) {
        let p: gpr;
        mov(p, i);
        touch(p);
        mov(byte[p], 2);
    }
}
