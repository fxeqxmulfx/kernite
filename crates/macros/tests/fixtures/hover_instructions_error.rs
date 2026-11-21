//! Modules of x86 and of GPU kernels, one of each that kernite refuses, as an editor opens them, for `tests/hover.rs`
//! to hover at the instructions a kernel names.

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

#[kernite_macros::kernels]
pub mod gpu {
    /// Adds one to the float at `a`.
    pub unsafe extern "ptx-kernel" fn add_one(a: *mut f32) {
        let v: f32;
        ld.global.f32(v, [a]);
        add.rn.f32(v, v, 1.0);
        st.global.f32([a], v);
    }

    /// Names one instruction that the GPU has, with an operand too many, and one that it has not.
    pub unsafe extern "ptx-kernel" fn broken(a: *mut f32) {
        let v: f32;
        ld.global.f32(v, [a]);
        sub.rn.f32(v, v, v, v);
        ld.globl.f32(v, [a]);
    }
}
