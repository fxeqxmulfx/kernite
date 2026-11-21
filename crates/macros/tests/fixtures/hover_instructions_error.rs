//! A module of x86 kernels, one of which kernite refuses, as an editor opens it, for `tests/hover.rs` to hover at the
//! instructions a kernel names.

#[kernite_macros::kernels]
pub mod kernels {
    /// Adds the eight floats of `b` to those of `a`.
    pub unsafe extern "sysv64" fn add_row(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vaddps(x, x, y);
        vmovups([a], x);
    }

    /// Names one instruction that x86 has, and one that it has not.
    pub unsafe extern "sysv64" fn broken(a: *mut f32) {
        let x: ymm;
        vmovups(x, [a]);
        vsubps(x, x, x);
        vaddpz(x, x, x);
    }
}
