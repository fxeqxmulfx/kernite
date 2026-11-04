//! Kernels kernite refuses, that take parameters named `_`, in a kernel of every kind: kernite says what is
//! wrong where it is, in the kernel, and Rust says nothing, for the stub the macros leave for each takes
//! the parameters as the kernel does. In modules that give the name of a number another type, `u64` and
//! `u32`, Rust says as well, once, at the type of the number named `_`, that it is not that number.

#![allow(non_camel_case_types)]

/// The type each number's name is in the modules that take them: that of another number.
pub mod alias {
    pub type u64 = core::primitive::i64;
    pub type u32 = core::primitive::i32;
}

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn plain(_: u64, x: usize, _: *const usize) -> usize {
        ret(y);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(_: u64, x: usize, _: *mut usize) -> usize {
        ret(y);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(_: u64, x: usize, _: *mut usize) {
        ret(y);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(_: u64, x: usize, _: *mut usize) {
        ret(y);
    }

    pub unsafe extern "C" fn other_abi(_: u64, x: usize, _: *const usize) -> usize {
        ret(x);
    }

    pub extern "sysv64" fn safe(_: u64, x: usize, _: *const usize) -> usize {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn plain(_: u32, p: *mut u16, _: *const u16) {
        st.global.u32([y], 1);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(_: u32, p: *mut u16, _: *const u16) {
        st.global.u32([y], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(_: u32, p: *mut u16, _: *const u16) {
        st.global.u32([y], 1);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(_: u32, p: *mut u16, _: *const u16) {
        st.global.u32([y], N);
    }
}

#[kernite_macros::kernels]
pub mod cpu_aliased {
    use crate::alias::*;

    pub unsafe extern "sysv64" fn plain(_: u64, x: usize, _: *const usize) -> usize {
        ret(y);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(_: u64, x: usize, _: *mut usize) -> usize {
        ret(y);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(_: u64, x: usize, _: *mut usize) {
        ret(y);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(_: u64, x: usize, _: *mut usize) {
        ret(y);
    }

    pub unsafe extern "C" fn other_abi(_: u64, x: usize, _: *const usize) -> usize {
        ret(x);
    }

    pub extern "sysv64" fn safe(_: u64, x: usize, _: *const usize) -> usize {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu_aliased {
    use crate::alias::*;

    pub unsafe extern "ptx-kernel" fn plain(_: u32, p: *mut u16, _: *const u16) {
        st.global.u32([y], 1);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(_: u32, p: *mut u16, _: *const u16) {
        st.global.u32([y], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(_: u32, p: *mut u16, _: *const u16) {
        st.global.u32([y], 1);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(_: u32, p: *mut u16, _: *const u16) {
        st.global.u32([y], N);
    }
}
