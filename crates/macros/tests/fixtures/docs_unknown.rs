//! A module of x86 kernels, one of which names an instruction x86 does not have, next to ones it has: the error is that
//! name's alone, as Rust has no doc of it to find.

#[kernite_macros::kernels]
pub mod kernels {
    /// Adds the eight floats of `b` to those of `a`.
    pub unsafe extern "sysv64" fn add_row(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vaddpz(x, x, y);
        vmovups([a], x);
    }

    /// Subtracts the eight floats of `b` from those of `a`.
    pub unsafe extern "sysv64" fn sub_row(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vsubps(x, x, y);
        vmovups([a], x);
    }
}
