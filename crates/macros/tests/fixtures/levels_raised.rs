//! The level a kernel sets by `warn`, `deny` or `forbid` holds at each place its code names a deprecated item, where
//! the crate allows the lint: Rust tells of each place as a warning or an error, but of none of `crate_level`, which
//! sets no level. The level of a kernel does not hold at a place in the code of a helper, which has its own:
//! `names_after` is told of its own place of `BIG`, and not of the helper's.
#![allow(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    #[deprecated]
    pub static BIG: global<[u8; 4096]>;

    #[deprecated]
    const K: u64 = 5;

    #[deprecated]
    fn touch(p: gpr) {
        mov(byte[p], 1);
    }

    #[deprecated]
    pub unsafe extern "sysv64" fn leaf(p: *mut u64) {
        mov(qword[p], 7);
    }

    fn names(p: gpr) {
        lea(p, [BIG]);
        mov(byte[p], 1);
    }

    pub unsafe extern "sysv64" fn crate_level(last: u64) {
        let p: gpr;
        let q: gpr;
        lea(p, [BIG]);
        mov(byte[p], K);
        lea(q, [BIG]);
        mov(byte[q + last], 9);
        touch(p);
    }

    #[warn(deprecated)]
    pub unsafe extern "sysv64" fn warned(last: u64) {
        let p: gpr;
        let q: gpr;
        lea(p, [BIG]);
        mov(byte[p], K);
        lea(q, [BIG]);
        mov(byte[q + last], 9);
        touch(p);
    }

    #[warn(deprecated)]
    pub unsafe extern "sysv64" fn names_after(last: u64) {
        let p: gpr;
        let q: gpr;
        mov(p, last);
        names(p);
        lea(q, [BIG]);
        mov(byte[q], 9);
    }

    #[deny(deprecated)]
    pub unsafe extern "sysv64" fn denied(last: u64, out: *mut u64) {
        let p: gpr;
        let q: gpr;
        lea(p, [BIG]);
        mov(byte[p], K);
        lea(q, [BIG]);
        mov(byte[q + last], 9);
        touch(p);
        call(leaf, out);
        call(leaf, out);
    }

    #[forbid(deprecated)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn forbidden<const N: u64>(last: u64, out: *mut u64) {
        let p: gpr;
        lea(p, [BIG]);
        mov(byte[p + last], N);
        lea(p, [BIG]);
        mov(byte[p], K);
        touch(p);
        call(leaf, out);
        call(leaf, out);
    }

    #[deny(deprecated)]
    #[cooperative]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn team<const N: u64>(last: u64) {
        let p: gpr;
        lea(p, [BIG]);
        mov(byte[p + last], N);
        lea(p, [BIG]);
        mov(byte[p], K);
        touch(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    #[deprecated]
    pub static COUNTS: global<[u32; 16]>;

    #[deprecated]
    const K: u32 = 5;

    #[deprecated]
    pub unsafe extern "ptx-func" fn old(n: u32) -> u32 {
        let v: u32;
        add.u32(v, n, n);
        ret(v);
    }

    pub unsafe extern "ptx-kernel" fn crate_level(out: *mut u32) {
        let v: u32;
        red.global.add.u32([COUNTS + 8], K);
        call(v, old, 21);
        red.global.add.u32([COUNTS + 12], 1);
        st.global.u32([out], v);
    }

    #[warn(deprecated)]
    pub unsafe extern "ptx-kernel" fn warned(out: *mut u32) {
        let v: u32;
        red.global.add.u32([COUNTS + 8], K);
        call(v, old, 21);
        red.global.add.u32([COUNTS + 12], 1);
        st.global.u32([out], v);
    }

    #[deny(deprecated)]
    pub unsafe extern "ptx-kernel" fn denied(out: *mut u32) {
        let v: u32;
        red.global.add.u32([COUNTS + 8], K);
        call(v, old, 21);
        red.global.add.u32([COUNTS + 12], 1);
        st.global.u32([out], v);
    }

    #[forbid(deprecated)]
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn forbidden(out: *mut u32) {
        let v: u32;
        red.global.add.u32([COUNTS + 8], K);
        call(v, old, 21);
        red.global.add.u32([COUNTS + 12], 1);
        st.global.u32([out], v);
    }
}
