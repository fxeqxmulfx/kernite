//! Each attribute that a helper and an `extern "ptx-func"` function take, which Rust has of them as it has of a
//! function of its own: a doc, `cfg`, `deprecated`, the level of a lint, and `rustfmt`'s, alone and in a
//! `cfg_attr(..)`. Rust says nothing of any, as there is nothing to say: a `deprecated` that is not malformed is told
//! of at a use, and the only uses here are of `old`, which the code that names it allows or expects. `cpu` has the
//! helpers, `gpu` the functions.

#[kernite_macros::kernels]
pub mod cpu {
    /// A doc.
    fn doc(p: gpr) {
        mov(byte[p], 1);
    }

    #[doc = "A doc as an attribute."]
    fn attribute(p: gpr) {
        mov(byte[p], 2);
    }

    #[doc(hidden)]
    fn hidden(p: gpr) {
        mov(byte[p], 3);
    }

    #[cfg(all())]
    fn configured(p: gpr) {
        mov(byte[p], 4);
    }

    #[cfg_attr(all(), doc = "A doc in a `cfg_attr(..)`.")]
    #[cfg_attr(any(), doc = "A doc that the condition leaves out.")]
    fn conditional(p: gpr) {
        mov(byte[p], 5);
    }

    #[deprecated]
    fn old(p: gpr) {
        mov(byte[p], 6);
    }

    #[deprecated(since = "1.0.0", note = "use `doc`")]
    fn older(p: gpr) {
        mov(byte[p], 7);
    }

    #[cfg_attr(all(), deprecated(note = "use `doc`"))]
    fn old_conditional(p: gpr) {
        mov(byte[p], 8);
    }

    #[allow(unused)]
    #[warn(unused)]
    fn levels(p: gpr) {
        mov(byte[p], 9);
    }

    #[deny(unused)]
    fn denied(p: gpr) {
        mov(byte[p], 10);
    }

    #[forbid(unused)]
    fn forbidden(p: gpr) {
        mov(byte[p], 11);
    }

    #[cfg_attr(all(), allow(deprecated))]
    fn allowed(p: gpr) {
        old(p);
    }

    #[expect(deprecated)]
    fn expected(p: gpr) {
        old(p);
    }

    #[rustfmt::skip]
    fn skipped(p: gpr) {
        mov(byte[p],   12);
    }

    /// Calls the helpers that are named by a kernel, and `old`, whose use this kernel allows.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    #[allow(deprecated)]
    pub unsafe extern "sysv64" fn all(p: *mut u8) {
        doc(p);
        attribute(p);
        hidden(p);
        configured(p);
        conditional(p);
        levels(p);
        denied(p);
        forbidden(p);
        allowed(p);
        expected(p);
        skipped(p);
        old(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    /// A doc.
    pub unsafe extern "ptx-func" fn doc(n: u32) -> u32 {
        ret(n);
    }

    #[doc = "A doc as an attribute."]
    pub unsafe extern "ptx-func" fn attribute(n: u32) -> u32 {
        ret(n);
    }

    #[cfg(all())]
    pub unsafe extern "ptx-func" fn configured(n: u32) -> u32 {
        ret(n);
    }

    #[cfg_attr(all(), doc = "A doc in a `cfg_attr(..)`.")]
    pub unsafe extern "ptx-func" fn conditional(n: u32) -> u32 {
        ret(n);
    }

    #[deprecated]
    pub unsafe extern "ptx-func" fn old(n: u32) -> u32 {
        ret(n);
    }

    #[deprecated(since = "1.0.0", note = "use `doc`")]
    pub unsafe extern "ptx-func" fn older(n: u32) -> u32 {
        ret(n);
    }

    #[allow(unused)]
    #[warn(unused)]
    pub unsafe extern "ptx-func" fn levels(n: u32) -> u32 {
        ret(n);
    }

    #[cfg_attr(all(), allow(deprecated))]
    pub unsafe extern "ptx-func" fn allowed(n: u32) -> u32 {
        let v: u32;
        call(v, old, n);
        ret(v);
    }

    #[expect(deprecated)]
    pub unsafe extern "ptx-func" fn expected(n: u32) -> u32 {
        let v: u32;
        call(v, old, n);
        ret(v);
    }

    #[rustfmt::skip]
    pub unsafe extern "ptx-func" fn skipped(n: u32) -> u32 {
        ret(n);
    }

    /// Calls the functions that are named by a kernel.
    pub unsafe extern "ptx-kernel" fn run(out: *mut u32) {
        let v: u32;
        call(v, doc, 1);
        call(v, attribute, v);
        call(v, configured, v);
        call(v, conditional, v);
        call(v, levels, v);
        call(v, allowed, v);
        call(v, expected, v);
        call(v, skipped, v);
        st.global.u32([out], v);
    }
}
