//! A crate of the `dylib` kind with a static in reach of `rip` and one past it: it links, though a kernel names the
//! first at a displacement from `rip`. `dylib_user` links it, calls the kernel, and names each static, which the crate
//! exports, to read the bytes the kernel wrote.
#![deny(warnings)]

#[kernite_macros::kernels]
pub mod cpu {
    /// 4096 bytes of zeros.
    pub static NEAR: global<[u8; 4096]>;
    /// 3 GiB of zeros, the linker's.
    pub static BIG: global<[u8; 3221225472]>;

    /// Writes 7 to the first byte of each static, and 9 to the byte `last` on.
    pub unsafe extern "sysv64" fn ends(last: u64) {
        let p: gpr;
        lea(p, [NEAR]);
        mov(byte[p], 7);
        mov(byte[p + last], 9);
        lea(p, [BIG]);
        mov(byte[p], 7);
        mov(byte[p + last], 9);
    }
}
