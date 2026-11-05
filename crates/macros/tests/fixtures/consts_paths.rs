//! A constant of `core::f32::consts` or `core::f64::consts` is named by Rust's own path, `std::f32::consts::LOG2_E`,
//! `core::f64::consts::PI`, `::std::f32::consts::TAU` or `::core::f64::consts::E`, a name written raw too,
//! `::r#std::f32::consts::r#GOLDEN_RATIO`; and a float's own constants, `f64::EPSILON` and `f64::MIN_POSITIVE`, by the
//! float's path. A module of kernels names them in a const, a static's data, a kernel's operand and a helper's, on x86
//! and on a GPU, and a module of consts in a const that the kernels take, and Rust builds it without a word. Each is
//! Rust's own value, bit for bit, also in a module of kernels that gives `std` another meaning and takes consts that
//! name `std::f32::consts::LOG2_E`: Rust reads the path where the consts are. The halves have constants of kernite's
//! own, `f16::MAX` and `bf16::EPSILON`, which Rust has not, and Rust is not asked about them.

#![deny(warnings)]

/// An x86 module of consts, a static, operands and a helper.
#[kernite_macros::kernels]
pub mod cpu {
    pub const LOG2_E: u64 = std::f32::consts::LOG2_E.to_bits() as u64;
    pub const PI: u64 = core::f64::consts::PI.to_bits();
    pub const EPSILON: u64 = f64::EPSILON.to_bits();
    pub const SMALLEST: u64 = f64::MIN_POSITIVE.to_bits();
    pub static TABLE: global<[f32; 3]> = [std::f32::consts::LOG2_E, ::core::f32::consts::E, f32::EPSILON];
    pub static WIDE: global<[f64; 2]> = [::std::f64::consts::TAU, core::f64::consts::GOLDEN_RATIO];

    /// The consts, as they are, at `out`.
    pub unsafe extern "sysv64" fn consts(out: *mut u64) {
        let g: gpr;
        mov(g, LOG2_E);
        mov(qword[out], g);
        mov(g, PI);
        mov(qword[out + 8], g);
        mov(g, EPSILON);
        mov(qword[out + 16], g);
        mov(g, SMALLEST);
        mov(qword[out + 24], g);
    }

    /// `x` times `FRAC_1_PI`, in each lane.
    fn scaled(x: ymm) {
        vmulps(x, x, [[::core::f32::consts::FRAC_1_PI; 8]]);
    }

    /// Five constants of `f32`, by the four spellings of the path and by one with names written raw, at `out`, and
    /// one by a helper at `out[5]`.
    pub unsafe extern "sysv64" fn singles(out: *mut f32) {
        let v: ymm;
        vmovups(v, [[std::f32::consts::LOG2_E; 8]]);
        vmovss([out], v.xmm);
        vmovups(v, [[core::f32::consts::LN_2; 8]]);
        vmovss([out + 4], v.xmm);
        vmovups(v, [[::std::f32::consts::FRAC_PI_2; 8]]);
        vmovss([out + 8], v.xmm);
        vmovups(v, [[::core::f32::consts::TAU; 8]]);
        vmovss([out + 12], v.xmm);
        vmovups(v, [[::r#std::f32::r#consts::r#GOLDEN_RATIO; 8]]);
        vmovss([out + 16], v.xmm);
        vmovups(v, [[1.0; 8]]);
        scaled(v);
        vmovss([out + 20], v.xmm);
    }

    /// Five constants of `f64`, in the same spellings, at `out`.
    pub unsafe extern "sysv64" fn doubles(out: *mut f64) {
        let v: ymm;
        vmovupd(v, [[std::f64::consts::SQRT_2; 4]]);
        vmovsd([out], v.xmm);
        vmovupd(v, [[core::f64::consts::FRAC_1_SQRT_2; 4]]);
        vmovsd([out + 8], v.xmm);
        vmovupd(v, [[::std::f64::consts::EULER_GAMMA; 4]]);
        vmovsd([out + 16], v.xmm);
        vmovupd(v, [[::core::f64::consts::LOG10_2; 4]]);
        vmovsd([out + 24], v.xmm);
        vmovupd(v, [[std::r#f64::consts::r#FRAC_2_SQRT_PI; 4]]);
        vmovsd([out + 32], v.xmm);
    }
}

/// A GPU module of a static, operands and the halves' constants.
#[kernite_macros::kernels]
pub mod gpu {
    pub static SCALES: constant<[f32; 2]> = [std::f32::consts::SQRT_2, ::core::f32::consts::FRAC_1_SQRT_2];

    /// Four constants of `f32`, in the four spellings, at `out`; and four of `f64` at `wide`.
    pub unsafe extern "ptx-kernel" fn floats(out: *mut f32, wide: *mut f64) {
        let x: f32;
        let y: f64;
        mov.f32(x, std::f32::consts::LOG2_E);
        st.global.f32([out], x);
        mov.f32(x, core::f32::consts::LN_10);
        st.global.f32([out + 4], x);
        mov.f32(x, ::std::f32::consts::FRAC_PI_3);
        st.global.f32([out + 8], x);
        mov.f32(x, ::core::f32::consts::LOG10_E);
        st.global.f32([out + 12], x);
        mov.f64(y, std::f64::consts::PI);
        st.global.f64([wide], y);
        mov.f64(y, core::f64::consts::E);
        st.global.f64([wide + 8], y);
        mov.f64(y, ::std::f64::consts::FRAC_PI_8);
        st.global.f64([wide + 16], y);
        mov.f64(y, ::core::f64::consts::LOG2_10);
        st.global.f64([wide + 24], y);
    }

    /// `f16::MAX` at `half` and `bf16::EPSILON` at `brain`, as they are.
    pub unsafe extern "ptx-kernel" fn halves(half: *mut f16, brain: *mut bf16) {
        let x: f16;
        let y: bf16;
        mov.b16(x, f16::MAX);
        mov.b16(y, bf16::EPSILON);
        st.global.b16([half], x);
        st.global.b16([brain], y);
    }
}

/// A module that has `use std::f32;` has Rust's `f32::consts::LOG2_E` as well, for x86 and for a GPU: Rust finds the
/// path, so kernite takes it.
#[kernite_macros::kernels]
pub mod with_use {
    use std::f32;

    /// `LOG2_E` at `out`.
    pub unsafe extern "sysv64" fn cpu(out: *mut f32) {
        let v: ymm;
        vmovups(v, [[f32::consts::LOG2_E; 8]]);
        vmovss([out], v.xmm);
    }

    /// `LOG2_E` at `out`.
    pub unsafe extern "ptx-kernel" fn gpu(out: *mut f32) {
        let x: f32;
        mov.f32(x, f32::consts::LOG2_E);
        st.global.f32([out], x);
    }
}

/// A module of the crate's own that has a `LOG2_E` of its own, where `std` is, for the module below to give `std` as
/// another meaning.
pub mod fake {
    pub mod f32 {
        pub mod consts {
            pub const LOG2_E: f32 = 3.0;
        }
    }
}

/// A module of consts that names constants of Rust's by their paths, for the kernels that take it to read.
#[kernite_macros::consts]
pub mod given {
    pub const LOG2_E: u64 = std::f32::consts::LOG2_E.to_bits() as u64;
    pub const E: u64 = ::core::f64::consts::E.to_bits();
}

/// An x86 module that takes the consts of `given`, and gives `std` another meaning: the path in `given` is Rust's where
/// `given` is, so Rust says nothing of it here.
#[kernite_macros::kernels(consts = crate::given)]
pub mod from_given {
    #[allow(unused_imports)]
    use crate::fake as std;

    /// The two consts, as they are, at `out`.
    pub unsafe extern "sysv64" fn cpu(out: *mut u64) {
        let g: gpr;
        mov(g, LOG2_E);
        mov(qword[out], g);
        mov(g, E);
        mov(qword[out + 8], g);
    }
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid, bf16, f16};

    use super::{cpu, from_given, gpu, given, with_use};

    /// The bits of each of `values`, as `f32` has them.
    fn singles(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// The bits of each of `values`, as `f64` has them.
    fn doubles(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// The consts, each a constant of Rust's, give Rust's bits to a kernel.
    #[test]
    fn an_x86_kernel_gets_the_bits_of_the_consts_rust_has() {
        let mut out = [0u64; 4];
        // SAFETY: the kernel writes four quad words at `out`, which holds four.
        unsafe { cpu::consts(out.as_mut_ptr()) };
        let want = [
            std::f32::consts::LOG2_E.to_bits() as u64,
            std::f64::consts::PI.to_bits(),
            f64::EPSILON.to_bits(),
            f64::MIN_POSITIVE.to_bits(),
        ];
        assert_eq!(out, want);
        assert_eq!((cpu::LOG2_E, cpu::PI, cpu::EPSILON, cpu::SMALLEST), (want[0], want[1], want[2], want[3]));
    }

    /// The statics' data has Rust's bits.
    #[test]
    fn an_x86_static_holds_the_bits_of_the_constants_rust_has() {
        assert_eq!(
            singles(&cpu::TABLE.read()),
            singles(&[std::f32::consts::LOG2_E, std::f32::consts::E, f32::EPSILON])
        );
        assert_eq!(doubles(&cpu::WIDE.read()), doubles(&[std::f64::consts::TAU, std::f64::consts::GOLDEN_RATIO]));
    }

    /// The operands, in each spelling, have Rust's bits: the four, one with names raw, and a helper's.
    #[test]
    fn an_x86_kernel_operand_has_the_bits_of_the_constant_rust_has() {
        let mut out = [0f32; 6];
        // SAFETY: the kernel writes six floats at `out`, which holds six.
        unsafe { cpu::singles(out.as_mut_ptr()) };
        let want = [
            std::f32::consts::LOG2_E,
            std::f32::consts::LN_2,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::TAU,
            std::f32::consts::GOLDEN_RATIO,
            std::f32::consts::FRAC_1_PI,
        ];
        assert_eq!(singles(&out), singles(&want));
        let mut out = [0f64; 5];
        // SAFETY: the kernel writes five doubles at `out`, which holds five.
        unsafe { cpu::doubles(out.as_mut_ptr()) };
        let want = [
            std::f64::consts::SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::EULER_GAMMA,
            std::f64::consts::LOG10_2,
            std::f64::consts::FRAC_2_SQRT_PI,
        ];
        assert_eq!(doubles(&out), doubles(&want));
    }

    /// A module that has `use std::f32;` names the constant without its root, as Rust reads it.
    #[test]
    fn a_kernel_names_a_constant_without_its_root_where_the_module_has_a_use() {
        let mut out = [0f32; 1];
        // SAFETY: the kernel writes one float at `out`, which holds one.
        unsafe { with_use::cpu(out.as_mut_ptr()) };
        assert_eq!(singles(&out), singles(&[std::f32::consts::LOG2_E]));
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let kernels = with_use::Kernels::load(&cuda).unwrap();
        let out = cuda.alloc::<f32>(1).unwrap();
        // SAFETY: the kernel writes one float at `out`, which holds one, and the launch has ended when it is read.
        unsafe { kernels.gpu(Grid::new(1, 1), out.ptr()).unwrap() };
        cuda.sync().unwrap();
        assert_eq!(singles(&out.read().unwrap()), singles(&[std::f32::consts::LOG2_E]));
    }

    /// A module of consts that name constants of Rust's gives Rust's bits to the kernels that take them.
    #[test]
    fn a_kernel_gets_the_bits_of_a_given_const_that_names_a_constant_rust_has() {
        let mut out = [0u64; 2];
        // SAFETY: the kernel writes two quad words at `out`, which holds two.
        unsafe { from_given::cpu(out.as_mut_ptr()) };
        let want = [std::f32::consts::LOG2_E.to_bits() as u64, std::f64::consts::E.to_bits()];
        assert_eq!(out, want);
        assert_eq!((given::LOG2_E, given::E), (want[0], want[1]));
    }

    /// The GPU's kernels give Rust's bits for the constants. For the halves', `f16::MAX` and `bf16::EPSILON`, they give
    /// the bits that IEEE 754's rules give a float of their widths, 5 bits of exponent and 10 of fraction, and 8 and 7,
    /// for Rust has no such floats to take them from. They skip where there is no GPU.
    #[test]
    fn a_gpu_kernel_gets_the_bits_of_the_constants_rust_has() {
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let kernels = gpu::Kernels::load(&cuda).unwrap();
        let (out, wide) = (cuda.alloc::<f32>(4).unwrap(), cuda.alloc::<f64>(4).unwrap());
        let (half, brain) = (cuda.alloc::<f16>(1).unwrap(), cuda.alloc::<bf16>(1).unwrap());
        // SAFETY: `floats` writes four floats at `out` and four doubles at `wide`, which hold four each, and `halves`
        // one half to each buffer it is given, which holds one; the launches have ended when they are read.
        unsafe {
            kernels.floats(Grid::new(1, 1), out.ptr(), wide.ptr()).unwrap();
            kernels.halves(Grid::new(1, 1), half.ptr(), brain.ptr()).unwrap();
        }
        cuda.sync().unwrap();
        let want = [
            std::f32::consts::LOG2_E,
            std::f32::consts::LN_10,
            std::f32::consts::FRAC_PI_3,
            std::f32::consts::LOG10_E,
        ];
        assert_eq!(singles(&out.read().unwrap()), singles(&want));
        let want = [
            std::f64::consts::PI,
            std::f64::consts::E,
            std::f64::consts::FRAC_PI_8,
            std::f64::consts::LOG2_10,
        ];
        assert_eq!(doubles(&wide.read().unwrap()), doubles(&want));
        assert_eq!(half.read().unwrap()[0].to_bits(), 0x7bff);
        assert_eq!(brain.read().unwrap()[0].to_bits(), 0x3c00);
    }
}
