//! A helper and an `extern "ptx-func"` function whose `deprecated` is malformed: Rust's error at the attribute, in its
//! words, once, whether a kernel names the item or not, under a `cfg` or not, and whether the `cfg` holds or not, as it
//! says of a function of its own. It says nothing of one that is not malformed, and nothing of one that `cfg` leaves
//! out, or whose `cfg_attr(..)` does not hold. `cpu` has the helpers, `gpu` the functions.

#[kernite_macros::kernels]
pub mod cpu {
    /// A kernel names it, and so does one that `cfg` leaves out.
    #[deprecated(foo)]
    fn named(p: gpr) {
        mov(byte[p], 1);
    }

    /// No kernel does.
    #[deprecated(bar)]
    fn alone(p: gpr) {
        mov(byte[p], 2);
    }

    /// Malformed in a `cfg_attr(..)` that holds, and called by no kernel.
    #[cfg_attr(all(), deprecated(note = "a", note = "b"))]
    fn conditional(p: gpr) {
        mov(byte[p], 3);
    }

    /// Two `deprecated`, and called by no kernel.
    #[deprecated]
    #[deprecated]
    fn twice(p: gpr) {
        mov(byte[p], 4);
    }

    /// Malformed in a `cfg_attr(..)` that does not hold: Rust leaves the attribute out.
    #[cfg_attr(any(), deprecated(baz))]
    fn off(p: gpr) {
        mov(byte[p], 5);
    }

    /// Malformed, and left out by `cfg`.
    #[cfg(any())]
    #[deprecated(foo)]
    fn gone(p: gpr) {
        mov(byte[p], 6);
    }

    /// Not malformed, and called by no kernel.
    #[deprecated(note = "use `named`")]
    fn fine(p: gpr) {
        mov(byte[p], 7);
    }

    /// Malformed, and named only by a kernel that `cfg` leaves out: Rust reads it as one that no kernel names.
    #[deprecated(qux)]
    fn unplaced(p: gpr) {
        mov(byte[p], 8);
    }

    /// Malformed, and named only by a kernel under a `cfg` that holds: Rust reads it as one that a kernel names.
    #[deprecated(quux)]
    fn guarded(p: gpr) {
        mov(byte[p], 9);
    }

    /// Calls `named`.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    pub unsafe extern "sysv64" fn names(p: *mut u8) {
        named(p);
    }

    /// Calls `unplaced` and `named`, under a `cfg` that does not hold.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    #[cfg(any())]
    pub unsafe extern "sysv64" fn left_out(p: *mut u8) {
        unplaced(p);
        named(p);
    }

    /// Calls `guarded`, under a `cfg` that holds.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    #[cfg(all())]
    pub unsafe extern "sysv64" fn names_guarded(p: *mut u8) {
        guarded(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    /// A kernel names it, and so does one that `cfg` leaves out.
    #[deprecated(note = "a", note = "b")]
    pub unsafe extern "ptx-func" fn old(n: u32) -> u32 {
        ret(n);
    }

    /// No kernel does.
    #[deprecated(baz)]
    pub unsafe extern "ptx-func" fn lone(n: u32) -> u32 {
        ret(n);
    }

    /// Malformed in a `cfg_attr(..)` that holds, and called by no kernel.
    #[cfg_attr(all(), deprecated(since = 1))]
    pub unsafe extern "ptx-func" fn conditional(n: u32) -> u32 {
        ret(n);
    }

    /// Malformed, and left out by `cfg`.
    #[cfg(any())]
    #[deprecated(baz)]
    pub unsafe extern "ptx-func" fn gone(n: u32) -> u32 {
        ret(n);
    }

    /// Not malformed, and called by no kernel.
    #[deprecated(note = "use `old`")]
    pub unsafe extern "ptx-func" fn fine(n: u32) -> u32 {
        ret(n);
    }

    /// Malformed, and named only by a kernel that `cfg` leaves out: Rust reads it as one that no kernel names.
    #[deprecated(since = 2)]
    pub unsafe extern "ptx-func" fn unplaced(n: u32) -> u32 {
        ret(n);
    }

    /// Malformed, and named only by a kernel under a `cfg` that holds: Rust reads it as one that a kernel names.
    #[deprecated(since = 3)]
    pub unsafe extern "ptx-func" fn guarded(n: u32) -> u32 {
        ret(n);
    }

    /// Calls `old`.
    pub unsafe extern "ptx-kernel" fn run(out: *mut u32) {
        let v: u32;
        call(v, old, 21);
        st.global.u32([out], v);
    }

    /// Calls `unplaced` and `old`, under a `cfg` that does not hold.
    #[cfg(any())]
    pub unsafe extern "ptx-kernel" fn left_out(out: *mut u32) {
        let v: u32;
        call(v, unplaced, 1);
        call(v, old, v);
        st.global.u32([out], v);
    }

    /// Calls `guarded`, under a `cfg` that holds.
    #[cfg(all())]
    pub unsafe extern "ptx-kernel" fn names_guarded(out: *mut u32) {
        let v: u32;
        call(v, guarded, 1);
        st.global.u32([out], v);
    }
}
