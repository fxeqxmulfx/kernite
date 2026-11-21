//! A kernel that allows `deprecated` is told of nothing by Rust at any place its code names a deprecated item, as a
//! function of Rust's that allows it is told of nothing: a static at each of its places, a constant, a helper, an x86
//! kernel and an instance of one, and on the GPU a static at each of its places, a constant, a function, a helper and
//! the address of a kernel. So it is of a kernel of instances, whose code is a function for each instance, of a
//! cooperative one, and of one whose `allow` is in a `cfg_attr(..)`. `deny` makes each place Rust tells of an error.
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

    #[deprecated]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn old<const N: u64>(p: *mut u64) {
        mov(qword[p], N);
    }

    #[deprecated]
    pub unsafe extern "sysv64" fn leaf(p: *mut u64) {
        mov(qword[p], 7);
    }

    /// Names `BIG` at two places, `K`, `touch`, an instance of `old` and `leaf` at two.
    #[allow(deprecated)]
    pub unsafe extern "sysv64" fn quiet(last: u64, out: *mut u64) {
        let p: gpr;
        let q: gpr;
        lea(p, [BIG]);
        mov(byte[p], K);
        lea(q, [BIG]);
        mov(byte[q + last], 9);
        touch(p);
        call(old::<2>, out);
        call(leaf, out);
        call(leaf, out);
    }

    /// Names them in each of its instances.
    #[allow(deprecated)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(last: u64, out: *mut u64) {
        let p: gpr;
        lea(p, [BIG]);
        mov(byte[p + last], N);
        lea(p, [BIG]);
        mov(byte[p], K);
        touch(p);
        call(old::<2>, out);
        call(leaf, out);
    }

    /// Names them, as a cooperative kernel of two instances.
    #[allow(deprecated)]
    #[cooperative]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn team<const N: u64>(last: u64, out: *mut u64) {
        let p: gpr;
        lea(p, [BIG]);
        mov(byte[p + last], N);
        lea(p, [BIG]);
        mov(byte[p], K);
        touch(p);
        call(old::<2>, out);
    }

    /// Allows it where `cfg` keeps the attribute, which it does.
    #[cfg_attr(all(), allow(deprecated))]
    pub unsafe extern "sysv64" fn conditional(last: u64) {
        let p: gpr;
        let q: gpr;
        lea(p, [BIG]);
        mov(byte[p], K);
        lea(q, [BIG]);
        mov(byte[q + last], 9);
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

    #[deprecated]
    fn bump(p: u64) {
        red.global.add.u32([p], 1);
    }

    #[deprecated]
    pub unsafe extern "ptx-kernel" fn spent(out: *mut u32) {
        st.global.u32([out], 1);
    }

    /// Names `COUNTS` at two places, `K`, `old`, `bump` and the address of `spent`.
    #[allow(deprecated)]
    pub unsafe extern "ptx-kernel" fn quiet(out: *mut u32) {
        let (v, fp): (u32, u64);
        red.global.add.u32([COUNTS + 8], K);
        call(v, old, 21);
        red.global.add.u32([COUNTS + 12], 1);
        bump(out);
        mov.u64(fp, spent);
        st.global.u32([out], v);
    }

    /// Names them, as a cooperative kernel.
    #[allow(deprecated)]
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn team(out: *mut u32) {
        let (v, fp): (u32, u64);
        red.global.add.u32([COUNTS + 8], K);
        call(v, old, 21);
        red.global.add.u32([COUNTS + 12], 1);
        bump(out);
        mov.u64(fp, spent);
        st.global.u32([out], v);
    }
}
