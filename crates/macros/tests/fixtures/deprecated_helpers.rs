//! A use of a deprecated helper, `extern "ptx-func"` function or GPU kernel is told of by Rust at the name, once for
//! each place a kernel, a function or a helper writes it, as Rust tells of a use of a deprecated function of its own:
//! not at the module's attribute, not of one that nothing names, and not where `cfg` leaves out the code that names it.
//! A helper that two kernels call, or a kernel of two instances, has the place once. `deny` counts the uses told of,
//! where Cargo says a line that two errors share once.
#![deny(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    /// Called by kernels.
    #[deprecated(note = "use `fresh`")]
    fn touch(p: gpr) {
        mov(byte[p], 1);
    }

    /// Called by no kernel.
    #[deprecated]
    fn unused(p: gpr) {
        mov(byte[p], 2);
    }

    /// Deprecated, by `cfg_attr`.
    #[cfg_attr(all(), deprecated)]
    fn loud(p: gpr) {
        mov(byte[p], 3);
    }

    /// Not deprecated, as `cfg_attr` has it.
    #[cfg_attr(any(), deprecated)]
    fn quiet(p: gpr) {
        mov(byte[p], 4);
    }

    /// Deprecated, and named as a raw name.
    #[deprecated]
    fn r#type(p: gpr) {
        mov(byte[p], 5);
    }

    /// Deprecated, and calls a helper that is: Rust tells of it as it does of any use in a deprecated function.
    #[deprecated]
    fn old(p: gpr) {
        touch(p);
    }

    /// Not deprecated, and called by two kernels.
    fn both(p: gpr) {
        touch(p);
    }

    /// Calls each helper once.
    pub unsafe extern "sysv64" fn names(p: *mut u8) {
        touch(p);
        loud(p);
        quiet(p);
        old(p);
        r#type(p);
    }

    /// Calls `touch` at two places.
    pub unsafe extern "sysv64" fn twice(p: *mut u8) {
        touch(p);
        touch(p);
    }

    /// Calls `both`.
    pub unsafe extern "sysv64" fn first(p: *mut u8) {
        both(p);
    }

    /// Calls `both` as well.
    pub unsafe extern "sysv64" fn second(p: *mut u8) {
        both(p);
    }

    /// Of two instances, with one place.
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn many<const N: u64>(p: *mut u8) {
        touch(p);
        mov(byte[p], N);
    }

    /// Left out by `cfg`, so that Rust tells of no use in it.
    #[cfg(any())]
    pub unsafe extern "sysv64" fn gone(p: *mut u8) {
        touch(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    /// Called by a kernel and a function, and its address taken.
    #[deprecated(note = "use `fresh`")]
    pub unsafe extern "ptx-func" fn old(n: u32) -> u32 {
        let v: u32;
        add.u32(v, n, n);
        ret(v);
    }

    /// Called by no kernel.
    #[deprecated]
    pub unsafe extern "ptx-func" fn unused(n: u32) -> u32 {
        ret(n);
    }

    /// Called by a kernel.
    #[deprecated]
    fn bump(p: u64) {
        red.global.add.u32([p], 1);
    }

    /// Calls itself: Rust does not tell of a use inside the item that is deprecated.
    #[deprecated]
    pub unsafe extern "ptx-func" fn gcd(a: u32, b: u32) -> u32 {
        let p: pred;
        setp.eq.u32(p, b, 0);
        if p {
            ret(a);
        }
        let (r, g): (u32, u32);
        rem.u32(r, a, b);
        call(g, gcd, (b, r));
        ret(g);
    }

    /// Calls `old`.
    pub unsafe extern "ptx-func" fn twice(n: u32) -> u32 {
        let v: u32;
        call(v, old, n);
        ret(v);
    }

    /// Calls `old`, takes its address, calls `gcd` and `bump`, and takes the address of `spent`.
    pub unsafe extern "ptx-kernel" fn run(out: *mut u32) {
        let (v, g, fp): (u32, u32, u64);
        call(v, old, 21);
        mov.u64(fp, old);
        call(g, gcd, (v, 14));
        st.global.u32([out], g);
        bump(out);
        mov.u64(fp, spent);
    }

    /// A kernel whose address `run` takes.
    #[deprecated]
    pub unsafe extern "ptx-kernel" fn spent(out: *mut u32) {
        st.global.u32([out], 1);
    }
}
