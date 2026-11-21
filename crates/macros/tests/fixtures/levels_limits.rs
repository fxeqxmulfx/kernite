//! Where Rust cannot hold the level of an x86 helper: at the first place a kernel names a static, in the helper's code.
//! The name is an operand of the kernel's naked function, which Rust tells of at the kernel's level, where the kernel
//! neither allows the lint nor expects it. So `warns_through` is told of the place in `names` as a warning, though
//! `names` sets no level, and `through` of the place in `allows_names` as an error, though `allows_names` allows the
//! lint. `expects` is Rust's own error: no naked function takes `expect`.
#![deny(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    #[deprecated]
    pub static BIG: global<[u8; 4096]>;

    fn names(p: gpr) {
        lea(p, [BIG]);
        mov(byte[p], 1);
    }

    #[allow(deprecated)]
    fn allows_names(p: gpr) {
        lea(p, [BIG]);
        mov(byte[p], 1);
    }

    #[warn(deprecated)]
    pub unsafe extern "sysv64" fn warns_through(last: u64) {
        let p: gpr;
        mov(p, last);
        names(p);
    }

    pub unsafe extern "sysv64" fn through(last: u64) {
        let p: gpr;
        mov(p, last);
        allows_names(p);
    }

    #[expect(deprecated)]
    pub unsafe extern "sysv64" fn expects(last: u64) {
        let p: gpr;
        mov(p, last);
        mov(byte[p], 1);
    }
}
