//! A number's name that the module gives another type, `pub type u64 = i64;` and a `use` of it, in a
//! kernel of every kind: kernite reads the number the name says and Rust passes what the type is, so
//! the macros ask Rust whether each number a kernel takes or gives back is that number. It is not,
//! and Rust says so, once, at the type: in each parameter and result of an x86 kernel, plain, of
//! compile-time arguments and cooperative, and in each parameter of a GPU kernel of those kinds.

#![allow(non_camel_case_types)]

/// The names of `u64` and `f64` as a module takes them from the crate's root, one at a time.
pub type u64 = u32;
/// See `u64`.
pub type f64 = f32;

/// The type each number's name is in the modules below: that of another number.
pub mod alias {
    pub type f32 = core::primitive::f64;
    pub type f64 = core::primitive::f32;
    pub type usize = core::primitive::isize;
    pub type isize = core::primitive::usize;
    pub type u64 = core::primitive::i64;
    pub type i64 = core::primitive::u64;
    pub type u32 = core::primitive::i32;
    pub type i32 = core::primitive::u32;
    pub type u16 = core::primitive::i16;
    pub type i16 = core::primitive::u16;
    pub type u8 = core::primitive::i8;
    pub type i8 = core::primitive::u8;
    pub type bool = core::primitive::u8;
}

#[kernite_macros::kernels]
pub mod ids {
    use crate::{f64, u64};

    pub unsafe extern "sysv64" fn id(x: u64) -> usize {
        ret(x);
    }

    pub unsafe extern "sysv64" fn twice(x: f64) -> f64 {
        vaddsd(x, x, x);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod cpu {
    use crate::alias::*;

    pub unsafe extern "sysv64" fn params(
        a: f32,
        b: f64,
        c: usize,
        d: isize,
        e: u64,
        f: i64,
    ) {
        ret();
    }

    pub unsafe extern "sysv64" fn r_f32(p: *const u8) -> f32 { let r: xmm; vmovss(r, [p]); ret(r); }
    pub unsafe extern "sysv64" fn r_f64(p: *const u8) -> f64 { let r: xmm; vmovsd(r, [p]); ret(r); }
    pub unsafe extern "sysv64" fn r_usize(p: *const u8) -> usize { let r: gpr; mov(r, qword[p]); ret(r); }
    pub unsafe extern "sysv64" fn r_isize(p: *const u8) -> isize { let r: gpr; mov(r, qword[p]); ret(r); }
    pub unsafe extern "sysv64" fn r_u64(p: *const u8) -> u64 { let r: gpr; mov(r, qword[p]); ret(r); }
    pub unsafe extern "sysv64" fn r_i64(p: *const u8) -> i64 { let r: gpr; mov(r, qword[p]); ret(r); }
    pub unsafe extern "sysv64" fn r_u32(p: *const u8) -> u32 { let r: gpr; mov(r, qword[p]); ret(r); }
    pub unsafe extern "sysv64" fn r_i32(p: *const u8) -> i32 { let r: gpr; mov(r, qword[p]); ret(r); }
    pub unsafe extern "sysv64" fn r_u16(p: *const u8) -> u16 { let r: gpr; mov(r, qword[p]); ret(r); }
    pub unsafe extern "sysv64" fn r_i16(p: *const u8) -> i16 { let r: gpr; mov(r, qword[p]); ret(r); }
    pub unsafe extern "sysv64" fn r_u8(p: *const u8) -> u8 { let r: gpr; mov(r, qword[p]); ret(r); }
    pub unsafe extern "sysv64" fn r_i8(p: *const u8) -> i8 { let r: gpr; mov(r, qword[p]); ret(r); }
    pub unsafe extern "sysv64" fn r_bool(p: *const u8) -> bool { let r: gpr; mov(r, qword[p]); ret(r); }
}

#[kernite_macros::kernels]
pub mod cpu_tiles {
    use crate::alias::*;

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn params<const N: usize>(
        a: f32,
        b: f64,
        c: usize,
        d: isize,
        e: u64,
        f: i64,
    ) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn r_f32<const N: usize>(p: *const u8) -> f32 { let r: xmm; vmovss(r, [p]); ret(r); }
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn r_u8<const N: usize>(p: *const u8) -> u8 { let r: gpr; mov(r, qword[p]); ret(r); }
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn r_bool<const N: usize>(p: *const u8) -> bool { let r: gpr; mov(r, qword[p]); ret(r); }
}

#[kernite_macros::kernels]
pub mod cpu_together {
    use crate::alias::*;

    #[cooperative]
    pub unsafe extern "sysv64" fn params(
        a: f32,
        b: f64,
        c: usize,
        d: isize,
        e: u64,
        f: i64,
    ) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn params_tiles<const N: usize>(a: f32, b: u64) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    use crate::alias::*;

    pub unsafe extern "ptx-kernel" fn params(
        a: u8,
        b: u16,
        c: u32,
        d: u64,
        e: i8,
        f: i16,
        g: i32,
        h: i64,
        i: usize,
        j: isize,
        k: f32,
        l: f64,
    ) {
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn params_tiles<const N: u32>(a: u8, b: f64) {
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn params_together(a: u32, b: f32) {
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn params_together_tiles<const N: u32>(a: i64) {
    }
}

/// A name for a type wider than the number it names, as the module's `u32` below is: Rust passes
/// eight bytes where the kernel reads four.
pub mod wide {
    pub type u32 = core::primitive::u64;
}

#[kernite_macros::kernels]
pub mod gpu_wide {
    use crate::wide::u32;

    pub unsafe extern "ptx-kernel" fn store(p: *mut core::primitive::u32, n: u32) {
        st.global.u32([p], n);
    }
}
