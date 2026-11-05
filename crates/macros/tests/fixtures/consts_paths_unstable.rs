//! A constant of `core::f32::consts` or `core::f64::consts` that Rust has unstable, `more_float_constants`, is Rust's
//! error alone, at the path, that the library feature is unstable: in a const, a static's data, a kernel's operand on
//! x86 and on a GPU, and a helper's, which two kernels call and Rust says once. kernite says nothing: it has the value
//! that Rust's nightly gives, for a crate that enables the feature to use.

/// An unstable constant in each place a value is written on x86.
#[kernite_macros::kernels]
pub mod unstable_x86 {
    pub const SQRT_3: u64 = std::f32::consts::SQRT_3.to_bits() as u64;
    pub static TABLE: global<[f32; 2]> = [core::f32::consts::SQRT_5, 1.0];

    /// `x` times the constant.
    fn scaled(x: ymm) {
        vmulps(x, x, [[::std::f32::consts::FRAC_1_SQRT_3; 8]]);
    }

    pub unsafe extern "sysv64" fn first(out: *mut u64) {
        let g: gpr;
        mov(g, SQRT_3);
        mov(qword[out], g);
    }

    pub unsafe extern "sysv64" fn second(out: *mut f32) {
        let v: ymm;
        vmovupd(v, [[::core::f64::consts::FRAC_1_SQRT_2PI; 4]]);
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
pub mod unstable_gpu {
    pub static SCALES: constant<[f32; 2]> = [std::f32::consts::FRAC_1_SQRT_5, 1.0];

    /// `x` times the constant.
    fn scaled(x: f64) {
        mul.rn.f64(x, x, core::f64::consts::FRAC_1_SQRT_PI);
    }

    pub unsafe extern "ptx-kernel" fn first(out: *mut f64) {
        let x: f64;
        mov.f64(x, std::f64::consts::SQRT_3);
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
