//! A module of x86 kernels that name instructions, as a crate builds, tests, documents and lints it: the docs of the
//! instructions, which an editor shows, are nothing to Rust.

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

    /// Built for a test and left out of the library.
    #[cfg(test)]
    pub unsafe extern "sysv64" fn in_tests(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vsubps(x, x, y);
        vmovups([a], x);
    }

    /// Left out always.
    #[cfg(any())]
    pub unsafe extern "sysv64" fn left_out(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vmulps(x, x, y);
        vmovups([a], x);
    }
}
