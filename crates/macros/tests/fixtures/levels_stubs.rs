//! A kernel that Rust does not get the code of is told of at the level it sets at each place its code names a
//! deprecated item, as one it does: `plain` and `tiles` give back a type with a `_` in it, `team` and the GPU's `tiles`
//! have an instance that does not compile, each is a warning at each place though the crate allows the lint, and Rust
//! says its own error beside them.
#![allow(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    #[deprecated]
    const K: u64 = 5;

    #[deprecated]
    fn touch(p: gpr) {
        mov(byte[p], 1);
    }

    #[warn(deprecated)]
    pub unsafe extern "sysv64" fn plain(p: *const u8) -> *const _ {
        touch(p);
        mov(byte[p], K);
        ret(p);
    }

    #[warn(deprecated)]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *const u8) -> *const _ {
        touch(p);
        mov(byte[p], K);
        add(p, N);
        ret(p);
    }

    #[warn(deprecated)]
    #[cooperative]
    #[instances(1, 300)]
    pub unsafe extern "sysv64" fn team<const N: u64>(p: *mut u8) {
        touch(p);
        mov(byte[p], K);
        mov(byte[p], N);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    #[deprecated]
    const K: u32 = 5;

    #[deprecated]
    pub unsafe extern "ptx-func" fn old(n: u32) -> u32 {
        let v: u32;
        add.u32(v, n, n);
        ret(v);
    }

    #[warn(deprecated)]
    #[instances(1, 70000)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(out: *mut u32) {
        let v: u32;
        call(v, old, 21);
        st.global.u32([out], K);
        st.global.u16([out], N);
    }
}
