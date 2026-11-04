//! The types that a helper's or an `extern "ptx-func"` function's signature names, written as primitives' paths,
//! `core::primitive::u64`, in modules where a kernel does not compile: Rust is shown them there, each whole, a
//! compile-time argument's too, however many tokens it is written in. Rust says nothing of them, as it says nothing of
//! the same types by their names, which each module of paths here is followed by. A comma after the last
//! compile-time argument, as the x86 `numbers` and the GPU `tiles` have, is no type.

#![allow(dead_code)]

/// A type of the crate's.
pub struct Word(pub u64);

/// x86 helpers by the paths, beside a kernel that does not compile.
#[kernite_macros::kernels]
pub mod paths_x86 {
    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        ret(y);
    }

    fn numbers<const N: ::core::primitive::u64, const B: core::primitive::bool,>(
        a: std::primitive::u32,
        b: ::std::primitive::u8,
    ) -> core::primitive::i64 {
        nop();
    }

    fn memory(p: *const ::core::primitive::u8, q: [std::primitive::u16; 2], r: *mut (core::primitive::f32, u8)) {
        nop();
    }

    fn others<const W: crate::Word>(a: crate::Word) {
        nop();
    }
}

/// The same x86 helpers by the names.
#[kernite_macros::kernels]
pub mod named_x86 {
    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        ret(y);
    }

    fn numbers<const N: u64, const B: bool,>(a: u32, b: u8) -> i64 {
        nop();
    }

    fn memory(p: *const u8, q: [u16; 2], r: *mut (f32, u8)) {
        nop();
    }

    fn others<const W: crate::Word>(a: crate::Word) {
        nop();
    }
}

/// GPU helpers and functions by the paths, beside a kernel that does not compile.
#[kernite_macros::kernels]
pub mod paths_gpu {
    pub unsafe extern "ptx-kernel" fn k(p: *mut u32) {
        st.global.u32([p], y);
    }

    fn numbers<const N: ::core::primitive::u32, const B: core::primitive::bool>(
        a: std::primitive::u32,
        b: ::std::primitive::f32,
    ) {
        nop();
    }

    pub unsafe extern "ptx-func" fn halves(
        a: core::primitive::u32,
        b: ::core::primitive::f32,
    ) -> ::std::primitive::f32 {
        ret(b);
    }

    #[instances(1)]
    pub unsafe extern "ptx-func" fn tiles<const M: ::std::primitive::u32,>(
        a: core::primitive::u64,
    ) -> std::primitive::u64 {
        ret(a);
    }
}

/// The same GPU helpers and functions by the names.
#[kernite_macros::kernels]
pub mod named_gpu {
    pub unsafe extern "ptx-kernel" fn k(p: *mut u32) {
        st.global.u32([p], y);
    }

    fn numbers<const N: u32, const B: bool>(a: u32, b: f32) {
        nop();
    }

    pub unsafe extern "ptx-func" fn halves(a: u32, b: f32) -> f32 {
        ret(b);
    }

    #[instances(1)]
    pub unsafe extern "ptx-func" fn tiles<const M: u32,>(a: u64) -> u64 {
        ret(a);
    }
}
