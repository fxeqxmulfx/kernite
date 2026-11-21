//! The level a helper or an `extern "ptx-func"` function sets holds at each place its own code names a deprecated item,
//! and the level of a kernel that calls it does not reach in. `allows` is told of nothing, though `plain`, which calls
//! it, sets no level and the crate denies the lint; `warns` and `warns_bump` are told of as a warning; `lets` is told
//! of at the crate's level, though `quiet`, which calls it, allows the lint. `expects` meets its expectation at each
//! place, and so is told of nothing; `unmet` names no deprecated item, and is told that its expectation is unfulfilled,
//! once, at the attribute. `gone` is left out by `cfg`, and nothing is said of its expectation. Nor is anything said of
//! `alone`, which no kernel calls, nor of `hidden`, which only `dropped` calls, which `cfg` leaves out: kernite reads
//! the code of a helper only where a kernel calls it. `spare`, an `extern "ptx-func"` function that no kernel calls, is
//! a function of the GPU's all the same, and its expectation is unfulfilled.
#![deny(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    #[deprecated]
    pub static COUNTS: global<[u8; 4]> = [0; 4];

    #[deprecated]
    const K: u64 = 5;

    #[deprecated]
    fn touch(p: gpr) {
        mov(byte[p], 1);
    }

    #[allow(deprecated)]
    fn allows(p: gpr) {
        touch(p);
        mov(byte[p], K);
    }

    #[warn(deprecated)]
    fn warns(p: gpr) {
        touch(p);
    }

    fn lets(p: gpr) {
        touch(p);
        mov(byte[p], K);
    }

    #[expect(deprecated)]
    fn expects(p: gpr) {
        touch(p);
        mov(byte[p], K);
    }

    #[expect(deprecated)]
    fn unmet(p: gpr) {
        mov(byte[p], 1);
    }

    #[cfg(any())]
    #[expect(deprecated)]
    fn gone(p: gpr) {
        mov(byte[p], 1);
    }

    #[expect(deprecated)]
    fn alone(p: gpr) {
        touch(p);
    }

    #[expect(deprecated)]
    fn hidden(p: gpr) {
        touch(p);
    }

    fn names(p: gpr) {
        let q: gpr;
        lea(q, [COUNTS]);
        mov(byte[q], 1);
        mov(byte[p], 1);
    }

    pub unsafe extern "sysv64" fn plain(p: *mut u8) {
        allows(p);
        warns(p);
        expects(p);
        unmet(p);
    }

    #[allow(deprecated)]
    pub unsafe extern "sysv64" fn quiet(p: *mut u8) {
        lets(p);
        names(p);
    }

    #[cfg(any())]
    pub unsafe extern "sysv64" fn dropped(p: *mut u8) {
        hidden(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    #[deprecated]
    fn bump(p: u64) {
        red.global.add.u32([p], 1);
    }

    #[allow(deprecated)]
    fn allows_bump(p: u64) {
        bump(p);
    }

    #[warn(deprecated)]
    fn warns_bump(p: u64) {
        bump(p);
    }

    fn lets_bump(p: u64) {
        bump(p);
    }

    #[expect(deprecated)]
    fn alone(p: u64) {
        bump(p);
    }

    #[cfg(any())]
    #[expect(deprecated)]
    pub unsafe extern "ptx-func" fn gone(n: u32) -> u32 {
        ret(n);
    }

    #[deprecated]
    pub unsafe extern "ptx-func" fn old(n: u32) -> u32 {
        let v: u32;
        add.u32(v, n, n);
        ret(v);
    }

    #[allow(deprecated)]
    pub unsafe extern "ptx-func" fn allows(n: u32) -> u32 {
        let v: u32;
        call(v, old, n);
        ret(v);
    }

    #[warn(deprecated)]
    pub unsafe extern "ptx-func" fn warns(n: u32) -> u32 {
        let v: u32;
        call(v, old, n);
        ret(v);
    }

    pub unsafe extern "ptx-func" fn lets(n: u32) -> u32 {
        let v: u32;
        call(v, old, n);
        ret(v);
    }

    #[expect(deprecated)]
    pub unsafe extern "ptx-func" fn expects(n: u32) -> u32 {
        let v: u32;
        call(v, old, n);
        ret(v);
    }

    #[expect(deprecated)]
    pub unsafe extern "ptx-func" fn unmet(n: u32) -> u32 {
        ret(n);
    }

    #[expect(deprecated)]
    pub unsafe extern "ptx-func" fn spare(n: u32) -> u32 {
        ret(n);
    }

    pub unsafe extern "ptx-kernel" fn plain(out: *mut u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        call(a, allows, 21);
        call(b, warns, 21);
        call(c, expects, 21);
        call(d, unmet, 21);
        allows_bump(out);
        warns_bump(out);
        st.global.u32([out], a);
        st.global.u32([out + 4], b);
        st.global.u32([out + 8], c);
        st.global.u32([out + 12], d);
    }

    #[allow(deprecated)]
    pub unsafe extern "ptx-kernel" fn quiet(out: *mut u32) {
        let v: u32;
        call(v, lets, 21);
        lets_bump(out);
        st.global.u32([out], v);
    }
}
