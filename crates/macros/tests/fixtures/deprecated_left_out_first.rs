//! A deprecated helper and a deprecated `extern "ptx-func"` function that a kernel which `cfg` leaves out names first,
//! and a kernel that `cfg` keeps names after: Rust tells of the use in the kernel that is kept, and of none in the one
//! that is left out. `deny` counts the uses told of.
#![deny(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    /// Called by two kernels.
    #[deprecated]
    fn old(p: gpr) {
        mov(byte[p], 1);
    }

    /// Left out by `cfg`, and the first to call `old`.
    #[cfg(any())]
    pub unsafe extern "sysv64" fn gone(p: *mut u8) {
        old(p);
    }

    /// Kept, and calls `old` after it.
    pub unsafe extern "sysv64" fn kept(p: *mut u8) {
        old(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    /// Called by two kernels.
    #[deprecated]
    pub unsafe extern "ptx-func" fn old(n: u32) -> u32 {
        ret(n);
    }

    /// Left out by `cfg`, and the first to call `old`.
    #[cfg(any())]
    pub unsafe extern "ptx-kernel" fn gone(out: *mut u32) {
        let v: u32;
        call(v, old, 1);
        st.global.u32([out], v);
    }

    /// Kept, and calls `old` after it.
    pub unsafe extern "ptx-kernel" fn kept(out: *mut u32) {
        let v: u32;
        call(v, old, 2);
        st.global.u32([out], v);
    }
}
