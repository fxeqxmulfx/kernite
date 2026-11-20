//! Each attribute that is not for a helper, or for an `extern "ptx-func"` function, which are no symbol of the host's,
//! nor a function of Rust's: kernite's error at the attribute's name, once, and Rust says nothing of the attribute,
//! which is not written to it. A helper or a function whose attribute is refused is one as it was, which a kernel
//! calls: Rust says nothing of the call. `cpu` has the helpers, `gpu` the functions. The first six helpers have the
//! attributes of a symbol of the host's and of a function of Rust's.

#[kernite_macros::kernels]
pub mod cpu {
    #[unsafe(no_mangle)]
    fn no_mangle(p: gpr) {
        mov(byte[p], 1);
    }

    #[unsafe(export_name = "EXPORTED")]
    fn export_name(p: gpr) {
        mov(byte[p], 2);
    }

    #[unsafe(link_section = ".mine")]
    fn link_section(p: gpr) {
        mov(byte[p], 3);
    }

    #[cold]
    fn cold(p: gpr) {
        mov(byte[p], 4);
    }

    #[track_caller]
    fn track_caller(p: gpr) {
        mov(byte[p], 5);
    }

    #[used]
    fn used(p: gpr) {
        mov(byte[p], 6);
    }

    // The same, in the forms Rust reads as the same attribute: written raw, on a helper named raw, which the error says
    // as written; in a `cfg_attr(..)` whatever its condition, beside one a helper takes, and inside another.
    #[r#used]
    fn r#type(p: gpr) {
        mov(byte[p], 7);
    }

    #[unsafe(r#no_mangle)]
    fn raw_unsafe(p: gpr) {
        mov(byte[p], 8);
    }

    #[cfg_attr(all(), unsafe(no_mangle))]
    fn holds(p: gpr) {
        mov(byte[p], 9);
    }

    #[cfg_attr(any(), used)]
    fn false_condition(p: gpr) {
        mov(byte[p], 10);
    }

    #[cfg_attr(all(), doc = "A doc, which a helper takes.", used)]
    fn beside(p: gpr) {
        mov(byte[p], 11);
    }

    #[cfg_attr(all(), cfg_attr(any(), cold))]
    fn nested(p: gpr) {
        mov(byte[p], 12);
    }

    // Any other attribute is refused as well, for an item of Rust's of its own, a tool's that is not `rustfmt`'s, and
    // one that no tool or macro reads: a path is said as written, and an attribute's arguments and value are not.
    #[must_use]
    fn must_use(p: gpr) {
        mov(byte[p], 13);
    }

    #[derive(Debug)]
    fn derive(p: gpr) {
        mov(byte[p], 14);
    }

    #[repr(C)]
    fn repr(p: gpr) {
        mov(byte[p], 15);
    }

    #[clippy::cognitive_complexity = "5"]
    fn clippy(p: gpr) {
        mov(byte[p], 16);
    }

    #[foo]
    fn unknown(p: gpr) {
        mov(byte[p], 17);
    }

    // Each attribute of a helper is judged, in the order written, beside the ones a helper takes. A helper that `cfg`
    // leaves out says nothing, and one it keeps says it.
    /// A doc, which a helper takes.
    #[used]
    #[allow(unused)]
    #[cold]
    fn many(p: gpr) {
        mov(byte[p], 18);
    }

    #[cfg(any())]
    #[used]
    fn gone(p: gpr) {
        mov(byte[p], 19);
    }

    #[cfg(all())]
    #[used]
    fn here(p: gpr) {
        mov(byte[p], 20);
    }

    /// Calls each helper that `cfg` keeps, for the helpers to be called, as they are where a kernel needs them.
    ///
    /// # Safety
    /// `p` points to a byte to write.
    pub unsafe extern "sysv64" fn all(p: *mut u8) {
        no_mangle(p);
        export_name(p);
        link_section(p);
        cold(p);
        track_caller(p);
        used(p);
        r#type(p);
        raw_unsafe(p);
        holds(p);
        false_condition(p);
        beside(p);
        nested(p);
        must_use(p);
        derive(p);
        repr(p);
        clippy(p);
        unknown(p);
        many(p);
        here(p);
    }
}

// The same for a function of the GPU's: a few of each form, and the attributes a function takes beside them.
#[kernite_macros::kernels]
pub mod gpu {
    #[foo]
    pub unsafe extern "ptx-func" fn odd(n: u32) -> u32 {
        ret(n);
    }

    #[must_use]
    pub unsafe extern "ptx-func" fn must_use(n: u32) -> u32 {
        ret(n);
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-func" fn no_mangle(n: u32) -> u32 {
        ret(n);
    }

    #[r#used]
    pub unsafe extern "ptx-func" fn r#type(n: u32) -> u32 {
        ret(n);
    }

    #[cfg_attr(any(), cold)]
    pub unsafe extern "ptx-func" fn false_condition(n: u32) -> u32 {
        ret(n);
    }

    /// A doc, which a function takes.
    #[used]
    #[allow(unused)]
    #[cold]
    pub unsafe extern "ptx-func" fn many(n: u32) -> u32 {
        ret(n);
    }

    #[cfg(any())]
    #[used]
    pub unsafe extern "ptx-func" fn gone(n: u32) -> u32 {
        ret(n);
    }

    #[cfg(all())]
    #[used]
    pub unsafe extern "ptx-func" fn here(n: u32) -> u32 {
        ret(n);
    }

    /// Calls each function that `cfg` keeps.
    pub unsafe extern "ptx-kernel" fn run(out: *mut u32) {
        let v: u32;
        call(v, odd, 1);
        call(v, must_use, v);
        call(v, no_mangle, v);
        call(v, r#type, v);
        call(v, false_condition, v);
        call(v, many, v);
        call(v, here, v);
        st.global.u32([out], v);
    }
}
