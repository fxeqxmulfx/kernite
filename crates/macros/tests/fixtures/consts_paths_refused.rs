//! A path to a constant that Rust has not is Rust's error alone, at the path, in Rust's words. The path without its
//! root, `f32::consts::LOG2_E`, is Rust's only after `use std::f32;`: where the module has none, Rust says
//! "ambiguous associated type" once for each place it is written, a const's, a static's, a kernel's operand on x86 and
//! on a GPU, and a helper's, which two kernels call; and kernite says nothing.

/// The path without its root, in each place a value is written on x86.
#[kernite_macros::kernels]
pub mod rootless_x86 {
    pub const LOG2_E: u64 = f32::consts::LOG2_E.to_bits() as u64;
    pub static TABLE: global<[f32; 2]> = [f32::consts::E, 1.0];

    /// `x` times the constant.
    fn scaled(x: ymm) {
        vmulps(x, x, [[f32::consts::PI; 8]]);
    }

    pub unsafe extern "sysv64" fn first(out: *mut u64) {
        let g: gpr;
        mov(g, LOG2_E);
        mov(qword[out], g);
    }

    pub unsafe extern "sysv64" fn second(out: *mut f32) {
        let v: ymm;
        vmovupd(v, [[f64::consts::TAU; 4]]);
        scaled(v);
        vmovss([out], v.xmm);
    }

    pub unsafe extern "sysv64" fn third(out: *mut f32) {
        let v: ymm;
        vmovups(v, [[1.0; 8]]);
        scaled(v);
        vmovss([out], v.xmm);
    }
}

/// The same on a GPU: a static's data, an operand, and a helper's, which two kernels call.
#[kernite_macros::kernels]
pub mod rootless_gpu {
    pub static SCALES: constant<[f32; 2]> = [f32::consts::E, 1.0];

    /// `x` times the constant.
    fn scaled(x: f64) {
        mul.rn.f64(x, x, f64::consts::LN_2);
    }

    pub unsafe extern "ptx-kernel" fn first(out: *mut f64) {
        let x: f64;
        mov.f64(x, f64::consts::FRAC_PI_2);
        scaled(x);
        st.global.f64([out], x);
    }

    pub unsafe extern "ptx-kernel" fn second(out: *mut f64) {
        let x: f64;
        mov.f64(x, 1.0f64);
        scaled(x);
        st.global.f64([out], x);
    }
}
