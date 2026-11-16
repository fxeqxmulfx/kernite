//! Two modules of one name in two functions have one path, so a static past `rip`'s reach of one name in each is one
//! symbol defined twice, which the assembler refuses.

/// Writes 1 to the first byte of its module's `BIG`.
pub fn first() {
    #[kernite_macros::kernels]
    mod m {
        static BIG: global<[u8; 3221225472]>;

        pub unsafe extern "sysv64" fn k() {
            let p: gpr;
            lea(p, [BIG]);
            mov(byte[p], 1);
        }
    }
    // SAFETY: `k` writes only the memory of its module's, which nothing else touches.
    unsafe { m::k() };
}

/// Writes 1 to the first byte of the `BIG` of a module of its own of the same name.
pub fn second() {
    #[kernite_macros::kernels]
    mod m {
        static BIG: global<[u8; 3221225472]>;

        pub unsafe extern "sysv64" fn k() {
            let p: gpr;
            lea(p, [BIG]);
            mov(byte[p], 1);
        }
    }
    // SAFETY: As in `first`.
    unsafe { m::k() };
}
