//! Kernels kernite refuses, in modules that give the names of numbers another type: the stub the macros
//! leave for each asks Rust whether each number it takes or gives back is that number, so that kernite
//! and Rust each say what they find, where it is: kernite at the kernel's body, Rust at the type.

#![allow(non_camel_case_types)]

/// The type each number's name is in the modules below: that of another number.
pub mod alias {
    pub type f32 = core::primitive::f64;
    pub type f64 = core::primitive::f32;
    pub type usize = core::primitive::isize;
    pub type u64 = core::primitive::i64;
    pub type u32 = core::primitive::i32;
    pub type u8 = core::primitive::i8;
    pub type bool = core::primitive::u8;
}

#[kernite_macros::kernels]
pub mod cpu {
    use crate::alias::*;

    pub unsafe extern "sysv64" fn plain(a: f32, b: usize, c: u64) -> f64 {
        ret(y);
    }

    pub unsafe extern "sysv64" fn plain_bool(a: u64) -> bool {
        ret(y);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: usize>(a: f32, b: u64) -> usize {
        ret(y);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(a: f32, b: u64) {
        ret(y);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: usize>(a: f64) {
        ret(y);
    }

    pub unsafe extern "C" fn other_abi(a: u64) -> f64 {
        ret(a);
    }

    pub extern "sysv64" fn safe(a: f32) -> f32 {
        ret(a);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    use crate::alias::*;

    pub unsafe extern "ptx-kernel" fn plain(a: u32, b: f64, c: u8) {
        st.global.u32([y], a);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(a: u64) {
        st.global.u32([y], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(a: f32) {
        st.global.u32([y], 1);
    }
}
