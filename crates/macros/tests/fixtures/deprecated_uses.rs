//! A use of a deprecated static or function by a kernel is told of by Rust once, at the use, as Rust tells of a use in
//! a function of its own: not at the module's attribute as well, and not of a static that no kernel names. A GPU
//! kernel's use of a deprecated static is told of at the use too, as a use of the static's method, `Kernels::COUNTS`.
//! `deny` counts the uses told of, where Cargo says a line that two errors share once.
#![deny(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    /// Named by a kernel, once.
    #[deprecated]
    pub static ONCE: global<[u8; 4096]>;

    /// Named by no kernel.
    #[deprecated]
    pub static UNUSED: global<[u8; 4096]>;

    /// Named by a kernel, twice.
    #[deprecated = "use `ONCE`"]
    pub static TWICE: global<[u8; 4096]>;

    /// Called by a kernel.
    #[deprecated(note = "use `fresh`")]
    pub unsafe extern "sysv64" fn old() -> u64 {
        let v: gpr;
        mov(v, 8);
        ret(v);
    }

    /// Names each deprecated item, and `TWICE` twice.
    pub unsafe extern "sysv64" fn names(i: u64) -> u64 {
        let p: gpr;
        let q: gpr;
        let v: gpr;
        lea(p, [ONCE]);
        lea(q, [TWICE]);
        mov(byte[p + i], 7);
        mov(byte[q + i], 9);
        lea(q, [TWICE]);
        mov(byte[q], 1);
        call(v, old, ());
        ret(v);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    /// Named by a kernel.
    #[deprecated]
    pub static COUNTS: global<[u32; 16]>;

    /// Named by no kernel.
    #[deprecated]
    pub static UNUSED: global<[u32; 16]>;

    pub unsafe extern "ptx-kernel" fn bump() {
        red.global.add.u32([COUNTS + 8], 1);
    }
}
