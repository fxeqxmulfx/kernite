//! A malformed `deprecated` that Rust says once but counts more than once, where the build is incremental, as it counts
//! an error once for each function that has the attribute. In `cpu`, two kernels and a helper that set a level name a
//! helper, and so do two kernels that set none. In `gpu`, a kernel and a function that set a level name a function, and
//! so does a kernel that sets none; two kernels take the address of a kernel, whose launcher has the attribute too.

#[kernite_macros::kernels]
pub mod cpu {
    /// Named by kernels, and by a helper.
    #[deprecated(foo)]
    fn old(p: gpr) {
        mov(byte[p], 1);
    }

    /// Calls `old`, and has a level of its own.
    #[allow(deprecated)]
    fn wrapper(p: gpr) {
        old(p);
    }

    /// Calls `old` twice, and sets a level.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    #[allow(deprecated)]
    pub unsafe extern "sysv64" fn quiet(p: *mut u8) {
        old(p);
        old(p);
    }

    /// Calls `old`, and sets a level.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    #[allow(deprecated)]
    pub unsafe extern "sysv64" fn quieter(p: *mut u8) {
        old(p);
    }

    /// Calls `old` and `wrapper`, and sets no level.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    pub unsafe extern "sysv64" fn loud(p: *mut u8) {
        old(p);
        wrapper(p);
    }

    /// Calls `old`, and sets no level.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    pub unsafe extern "sysv64" fn louder(p: *mut u8) {
        old(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    /// Named by a kernel and a function that set a level, and by a kernel that sets none.
    #[deprecated(baz)]
    pub unsafe extern "ptx-func" fn old(n: u32) -> u32 {
        ret(n);
    }

    /// Calls `old`, and has a level of its own.
    #[allow(deprecated)]
    pub unsafe extern "ptx-func" fn wrapper(n: u32) -> u32 {
        let v: u32;
        call(v, old, n);
        ret(v);
    }

    /// Calls `old`, and sets a level.
    #[allow(deprecated)]
    pub unsafe extern "ptx-kernel" fn quiet(out: *mut u32) {
        let v: u32;
        call(v, old, 1);
        st.global.u32([out], v);
    }

    /// A kernel whose address two kernels take, one of them twice.
    #[deprecated(bar)]
    pub unsafe extern "ptx-kernel" fn spent(out: *mut u32) {
        st.global.u32([out], 1);
    }

    /// Takes the address of `spent` twice.
    pub unsafe extern "ptx-kernel" fn run(out: *mut u32) {
        let fp: u64;
        mov.u64(fp, spent);
        mov.u64(fp, spent);
        st.global.u32([out], 2);
    }

    /// Takes the address of `spent`, and calls `old`.
    pub unsafe extern "ptx-kernel" fn other(out: *mut u32) {
        let fp: u64;
        let v: u32;
        mov.u64(fp, spent);
        call(v, old, 3);
        st.global.u32([out], v);
    }
}
