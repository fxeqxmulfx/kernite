//! A module of x86 kernels as an editor opens it, for `tests/hover.rs` to hover at the instructions a kernel names.

#[kernite_macros::kernels]
pub mod kernels {
    /// The flags that `claim` sets a bit of: memory of the module's.
    pub static FLAGS: global<[u64; 4]>;

    /// Adds the eight floats of `b` to those of `a`.
    pub unsafe extern "sysv64" fn add_row(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vaddps(x, x, y);
        vmovups([a], x);
    }

    /// Sets bit `i` of the `FLAGS` in one atomic step, and copies it to `i` if it was clear.
    pub unsafe extern "sysv64" fn claim(i: u64) {
        let (flags, bit): (gpr, gpr);
        lea(flags, [FLAGS]);
        mov(bit, i);
        lock.bts([flags], i);
        cmovz(i, bit);
    }

    /// Not built: `cfg` leaves it out, and with it the docs of the instructions only it names.
    #[cfg(any())]
    pub unsafe extern "sysv64" fn left_out(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vsubps(x, x, y);
        vmovups([a], x);
    }
}
