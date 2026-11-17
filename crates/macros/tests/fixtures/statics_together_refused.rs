//! A kernel that names a static past `rip`'s reach at a displacement is kernite's error at the operand, before the
//! linker's, where the static is far for the statics before it in its module: one that `cfg` leaves out counts all the
//! same, so the crate builds no static of 1 GiB. A kernel that takes the static's address is no error.

#[kernite_macros::kernels]
pub mod left_out {
    /// Left out by `cfg`, and counted all the same: 1 GiB, all that is in reach.
    #[cfg(any())]
    pub static A: global<[u8; 1073741824]>;

    /// Past reach, beside `A`.
    pub static B: global<[u8; 1073741824]>;

    /// Names `B` at a displacement: an error.
    pub unsafe extern "sysv64" fn second() {
        mov(byte[B + 1073741823], 2);
    }

    /// Takes the address of `B`: no error.
    pub unsafe extern "sysv64" fn address() {
        let p: gpr;
        lea(p, [B + 1073741823]);
        mov(byte[p], 2);
    }
}
