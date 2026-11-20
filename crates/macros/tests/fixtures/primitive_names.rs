//! A module that gives the name of a primitive another type, `u8`, `u32`, `f32` or `str`, builds as one
//! that does not: the Rust the macros write names the primitives by their paths, `::core::primitive::u8`,
//! and each kernel's own types here are written so. A GPU module that takes `u8`, a cooperative GPU
//! kernel in one that takes `u32`, an x86 module that takes `str`, and modules of statics that take
//! every number's name and `str`, on x86 and on a GPU; and modules whose kernels name the constants of the
//! numbers they give other types: a number's own, `u32::MAX`, which Rust is not asked of, and a module's,
//! `std::f32::consts::PI`, which it is. Rust builds each without a word.
//! Clippy asks of each public x86 kernel a `# Safety`, as of any `unsafe fn`, which these have not: allowed here.

#![deny(warnings)]
#![allow(clippy::missing_safety_doc)]
#![allow(non_camel_case_types)]

/// The type each name is in the modules below.
pub mod alias {
    pub type u8 = [core::primitive::u8];
    pub type u32 = core::primitive::u64;
    pub type str = core::primitive::u8;
    pub type bool = core::primitive::u8;
    pub type f32 = core::primitive::f64;
    pub type f64 = core::primitive::f32;
    pub type usize = core::primitive::isize;
    pub type isize = core::primitive::usize;
    pub type u16 = core::primitive::i16;
    pub type i16 = core::primitive::u16;
    pub type u64 = core::primitive::i64;
    pub type i64 = core::primitive::u64;
    pub type i32 = core::primitive::u32;
    pub type i8 = core::primitive::u8;
}

/// A GPU module that takes `u8`: the manifest it is built again for, `include_bytes!`, is a `&[u8]`.
#[kernite_macros::kernels]
pub mod gpu_u8 {
    #[allow(unused_imports)]
    use crate::alias::u8;

    pub unsafe extern "ptx-kernel" fn k(p: *mut core::primitive::u32) {
        st.global.u32([p], 1);
    }
}

/// A cooperative GPU kernel in a module that takes `u32`: its `_blocks` method takes `threads` and
/// `shared` as `u32`s and gives back one.
#[kernite_macros::kernels]
pub mod gpu_u32 {
    #[allow(unused_imports)]
    use crate::alias::u32;

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn k(p: *mut core::primitive::u32) {
        st.global.u32([p], 1);
    }
}

/// An x86 module that takes `str`: its `LISTINGS` is a slice of pairs of `&str`.
#[kernite_macros::kernels]
pub mod cpu_str {
    #[allow(unused_imports)]
    use crate::alias::str;

    pub unsafe extern "sysv64" fn k(p: *mut core::primitive::u64) {
        mov(qword[p], 1);
    }
}

/// An x86 module of statics that takes every name: the type of each is a `kernite_cpu::Global` of
/// the primitive, and a float starts as its bits.
#[kernite_macros::kernels]
pub mod cpu_statics {
    #[allow(unused_imports)]
    use crate::alias::*;

    pub static COUNTS: global<[u32; 4]> = [1, 2, 3, 4];
    pub static SIGNED: global<[i8; 2]> = [-1, 2];
    pub static HALVES: constant<[f32; 2]> = [0.5, 1.5];
    pub static WIDE: constant<[f64; 2]> = [0.5, 1.5];
    pub static ZEROS: global<[u16; 8]>;

    pub unsafe extern "sysv64" fn first(p: *mut core::primitive::u64) {
        let at: gpr;
        lea(at, [COUNTS]);
        mov(qword[p], at);
        lea(at, [SIGNED]);
        mov(qword[p], at);
        lea(at, [HALVES]);
        mov(qword[p], at);
        lea(at, [WIDE]);
        mov(qword[p], at);
        lea(at, [ZEROS]);
        mov(qword[p], at);
    }
}

/// A GPU module of statics that takes every name: the method of each gives a `kernite_cuda::Global` of
/// the primitive.
#[kernite_macros::kernels]
pub mod gpu_statics {
    #[allow(unused_imports)]
    use crate::alias::*;

    pub static COUNTS: global<[u32; 4]>;
    pub static HALVES: global<[f32; 2]>;
    pub static BYTES: global<[u8; 4]>;

    pub unsafe extern "ptx-kernel" fn first() {
        let at: u64;
        mov.u64(at, COUNTS);
        mov.u64(at, HALVES);
        mov.u64(at, BYTES);
    }
}

/// An x86 module that takes every name and names constants: a number's own, by its name and by its path, and a
/// module's.
#[kernite_macros::kernels]
pub mod cpu_constants {
    #[allow(unused_imports)]
    use crate::alias::*;

    pub unsafe extern "sysv64" fn k(x: core::primitive::u64) -> core::primitive::u64 {
        add(x, u8::MAX);
        add(x, ::std::primitive::u16::MAX);
        add(x, i8::MIN);
        add(x, u64::BITS);
        let g: gpr;
        mov(g, core::primitive::u32::MAX);
        add(x, g);
        let v: ymm;
        vmovups(v, [[f32::INFINITY; 8]]);
        vmovups(v, [[core::primitive::f32::EPSILON; 8]]);
        vmovups(v, [[std::f32::consts::PI; 8]]);
        vmovups(v, [[::core::f32::consts::E; 8]]);
        ret(x);
    }
}

/// A GPU module that takes every name and names constants, as `cpu_constants` does.
#[kernite_macros::kernels]
pub mod gpu_constants {
    #[allow(unused_imports)]
    use crate::alias::*;

    pub unsafe extern "ptx-kernel" fn k(p: *mut core::primitive::u32) {
        let v: core::primitive::u32;
        let x: core::primitive::f32;
        mov.u32(v, u32::MAX);
        add.u32(v, v, core::primitive::u8::MAX);
        mov.f32(x, f32::MAX);
        mov.f32(x, ::std::primitive::f32::MIN_POSITIVE);
        mov.f32(x, std::f32::consts::LOG2_E);
        mov.f32(x, ::core::f32::consts::LN_2);
        st.global.u32([p], v);
    }
}
