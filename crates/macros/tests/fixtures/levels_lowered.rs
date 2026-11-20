//! The level a kernel sets by `warn` or `expect` holds at each place its code names a deprecated item, where the crate
//! denies the lint: Rust tells of a place of `warned` as a warning, and of none of `expected` and `once`, as each place
//! meets the expectation, the first place of `BIG` too, whose operand the function of each instance allows; and of
//! `unmet`, whose code names none, that its expectation is unfulfilled, once, at the attribute. `crate_level`, which
//! sets no level, is told of at the crate's, as an error.
#![deny(deprecated)]

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

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn crate_level<const N: u64>(last: u64) {
        let p: gpr;
        mov(p, last);
        mov(byte[p], K);
        mov(byte[p + N], K);
        touch(p);
    }

    #[warn(deprecated)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn warned<const N: u64>(last: u64) {
        let p: gpr;
        mov(p, last);
        mov(byte[p], K);
        mov(byte[p + N], K);
        touch(p);
    }

    #[expect(deprecated)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn expected<const N: u64>(last: u64) {
        let p: gpr;
        let q: gpr;
        lea(p, [BIG]);
        mov(byte[p + last], K);
        lea(q, [BIG]);
        mov(byte[q + N], K);
        touch(p);
    }

    /// Names `BIG` at one place alone, the operand's.
    #[expect(deprecated)]
    #[cooperative]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn once<const N: u64>(last: u64) {
        let p: gpr;
        lea(p, [BIG]);
        mov(byte[p + last], N);
    }

    #[expect(deprecated)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn unmet<const N: u64>(last: u64) {
        let p: gpr;
        mov(p, last);
        mov(byte[p + N], 1);
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

    #[expect(deprecated)]
    pub unsafe extern "ptx-kernel" fn expected(out: *mut u32) {
        let v: u32;
        red.global.add.u32([COUNTS + 8], K);
        call(v, old, 21);
        red.global.add.u32([COUNTS + 12], 1);
        st.global.u32([out], v);
    }

    #[expect(deprecated)]
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn unmet(out: *mut u32) {
        st.global.u32([out], 1);
    }
}
