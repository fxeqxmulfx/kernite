//! A static that does not compile, which two kernels name: it is said once, where it is.

#[kgen_macros::kernels]
pub mod cpu {
    pub static COUNTS: global<[u32; 0]>;

    pub unsafe extern "sysv64" fn first() {
        let at: gpr;
        lea(at, [COUNTS]);
    }

    pub unsafe extern "sysv64" fn second() {
        let at: gpr;
        lea(at, [COUNTS]);
    }
}

#[kgen_macros::kernels]
pub mod gpu {
    pub static COUNTS: global<[u32; 0]>;

    pub unsafe extern "ptx-kernel" fn first() {
        let at: u64;
        mov.u64(at, COUNTS);
    }

    pub unsafe extern "ptx-kernel" fn second() {
        let at: u64;
        mov.u64(at, COUNTS);
    }
}
