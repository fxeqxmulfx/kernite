//! Helpers and an `extern "ptx-func"` function whose `deprecated` is malformed, and that kernels name at more than one
//! place: Rust's error at each attribute, in its words, once, and counted once, as it is of a function of its own that
//! two functions call, or one calls twice. `cpu` has the helpers, `gpu` the function.

#[kernite_macros::kernels]
pub mod cpu {
    /// Called by two kernels.
    #[deprecated(foo)]
    fn both(p: gpr) {
        mov(byte[p], 1);
    }

    /// Called twice by a kernel.
    #[deprecated(bar)]
    fn twice(p: gpr) {
        mov(byte[p], 2);
    }

    /// Called by a helper that two kernels call.
    #[deprecated(baz)]
    fn deep(p: gpr) {
        mov(byte[p], 3);
    }

    /// Calls `deep`.
    fn middle(p: gpr) {
        deep(p);
    }

    /// Calls `both`, `twice` twice, and `middle`.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    pub unsafe extern "sysv64" fn first(p: *mut u8) {
        both(p);
        twice(p);
        twice(p);
        middle(p);
    }

    /// Calls `both` and `middle`.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    pub unsafe extern "sysv64" fn second(p: *mut u8) {
        both(p);
        middle(p);
    }

    /// Calls `both`, in each of two instances.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn many<const N: u64>(p: *mut u8) {
        both(p);
        mov(byte[p + 1], N);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    /// Called twice by a kernel, and by another.
    #[deprecated(qux)]
    pub unsafe extern "ptx-func" fn twice(n: u32) -> u32 {
        ret(n);
    }

    /// Calls `twice` twice.
    pub unsafe extern "ptx-kernel" fn run(out: *mut u32) {
        let v: u32;
        call(v, twice, 1);
        call(v, twice, v);
        st.global.u32([out], v);
    }

    /// Calls `twice`.
    pub unsafe extern "ptx-kernel" fn other(out: *mut u32) {
        let v: u32;
        call(v, twice, 2);
        st.global.u32([out], v);
    }
}
