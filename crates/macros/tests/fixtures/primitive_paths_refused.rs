//! A type written as a path that the place does not take is kernite's error alone, at the type, in the words it has for
//! the same type by its name, and Rust says no more than it does of the name. A primitive the place does not take is
//! named by its name: `core::primitive::str` for a parameter, `::std::primitive::char`, `std::primitive::u128`, and
//! `core::primitive::f32` for a compile-time argument, which Rust refuses as it refuses `f32`. A path that names no
//! primitive is quoted as written: a type of the crate's, `crate::Word`, one that is a `type` of the crate's for `u64`,
//! `crate::Wide`, a path that goes on, `core::primitive::u64::MAX`, one that begins elsewhere,
//! `my::core::primitive::u64`, and one to a type `core::primitive` has not, `core::primitive::f16`. Each module of
//! paths here is a module of names that follows it, in the same kernels.

#![allow(dead_code, non_camel_case_types)]

/// A type of the crate's.
pub struct Word(pub u64);

/// A `type` of the crate's, for `u64`.
pub type Wide = u64;

/// A `core` of the crate's own, in a module of its own: kernite does not know it.
pub mod my {
    pub mod core {
        pub mod primitive {
            pub type u64 = ::core::primitive::u64;
        }
    }
}

/// x86 kernels by the paths.
#[kernite_macros::kernels]
pub mod paths_x86 {
    #[instances(1)]
    pub unsafe extern "sysv64" fn arg_f32<const X: core::primitive::f32>(p: *mut u64) {
        mov(qword[p], 1);
    }
    #[instances(1)]
    pub unsafe extern "sysv64" fn arg_word<const X: crate::Word>(p: *mut u64) {
        mov(qword[p], 1);
    }
    pub unsafe extern "sysv64" fn param_str(x: core::primitive::str) {}
    pub unsafe extern "sysv64" fn param_char(x: ::std::primitive::char) {}
    pub unsafe extern "sysv64" fn param_u128(x: std::primitive::u128) {}
    pub unsafe extern "sysv64" fn param_word(x: crate::Word) {}
    pub unsafe extern "sysv64" fn param_wide(x: crate::Wide) {}
    pub unsafe extern "sysv64" fn param_max(x: core::primitive::u64::MAX) {}
    pub unsafe extern "sysv64" fn param_my(x: my::core::primitive::u64) {}
    pub unsafe extern "sysv64" fn param_f16(x: core::primitive::f16) {}
    pub unsafe extern "sysv64" fn result_i128(x: core::primitive::u64) -> ::std::primitive::i128 {}
    pub unsafe extern "sysv64" fn variable(p: *mut core::primitive::u64) {
        let v: [core::primitive::str; 2];
    }
}

/// The same x86 kernels by the names.
#[kernite_macros::kernels]
pub mod named_x86 {
    use crate::{Word, Wide};

    #[instances(1)]
    pub unsafe extern "sysv64" fn arg_f32<const X: f32>(p: *mut u64) {
        mov(qword[p], 1);
    }
    #[instances(1)]
    pub unsafe extern "sysv64" fn arg_word<const X: Word>(p: *mut u64) {
        mov(qword[p], 1);
    }
    pub unsafe extern "sysv64" fn param_str(x: str) {}
    pub unsafe extern "sysv64" fn param_char(x: char) {}
    pub unsafe extern "sysv64" fn param_u128(x: u128) {}
    pub unsafe extern "sysv64" fn param_word(x: Word) {}
    pub unsafe extern "sysv64" fn param_wide(x: Wide) {}
    pub unsafe extern "sysv64" fn result_i128(x: u64) -> i128 {}
    pub unsafe extern "sysv64" fn variable(p: *mut u64) {
        let v: [str; 2];
    }
}

/// GPU kernels by the paths.
#[kernite_macros::kernels]
pub mod paths_gpu {
    #[instances(1)]
    pub unsafe extern "ptx-kernel" fn arg_f32<const X: core::primitive::f32>(p: *mut core::primitive::u32) {
        st.global.u32([p], 1);
    }
    pub unsafe extern "ptx-kernel" fn param_str(x: core::primitive::str) {}
    pub unsafe extern "ptx-kernel" fn param_char(x: ::std::primitive::char) {}
    pub unsafe extern "ptx-kernel" fn param_u128(x: std::primitive::u128) {}
    pub unsafe extern "ptx-kernel" fn param_word(x: crate::Word) {}
    pub unsafe extern "ptx-kernel" fn param_wide(x: crate::Wide) {}
    pub unsafe extern "ptx-kernel" fn param_f16(x: core::primitive::f16) {}
    pub unsafe extern "ptx-kernel" fn result_i128(x: core::primitive::u64) -> ::std::primitive::i128 {}
    pub unsafe extern "ptx-kernel" fn variable(p: *mut core::primitive::u32) {
        let v: core::primitive::char;
    }
}

/// The same GPU kernels by the names.
#[kernite_macros::kernels]
pub mod named_gpu {
    use crate::{Word, Wide};

    #[instances(1)]
    pub unsafe extern "ptx-kernel" fn arg_f32<const X: f32>(p: *mut u32) {
        st.global.u32([p], 1);
    }
    pub unsafe extern "ptx-kernel" fn param_str(x: str) {}
    pub unsafe extern "ptx-kernel" fn param_char(x: char) {}
    pub unsafe extern "ptx-kernel" fn param_u128(x: u128) {}
    pub unsafe extern "ptx-kernel" fn param_word(x: Word) {}
    pub unsafe extern "ptx-kernel" fn param_wide(x: Wide) {}
    pub unsafe extern "ptx-kernel" fn result_i128(x: u64) -> i128 {}
    pub unsafe extern "ptx-kernel" fn variable(p: *mut u32) {
        let v: char;
    }
}
