//! A number's name that the module gives a type with no size, `u8` a slice. Rust says of each
//! number a kernel takes or gives back, at the type, that it is not the number and that its size
//! cannot be known, and nothing at the module's attribute, where nothing can be mended. So it does
//! of an x86 kernel with the number as its result or its parameter, and of a GPU kernel with it as
//! a parameter: plain, of compile-time arguments, cooperative, cooperative of compile-time
//! arguments, and one kernite refuses. A name given a type with a size that is no number, a
//! `String`, is Rust's one error that it is not the number, in a cooperative x86 kernel, which
//! copies its arguments to each thread, and in a GPU kernel, which packs them for the driver.

#![allow(non_camel_case_types)]

/// A slice, in place of the number's name: `u8`, which an x86 kernel gives back and a GPU kernel takes,
/// and `usize`, which an x86 kernel takes, as SysV leaves the high bits of a `u8` as they were.
pub mod wide {
    pub type u8 = [core::primitive::u8];
    pub type usize = [core::primitive::usize];
}

#[kernite_macros::kernels]
pub mod cpu {
    use crate::wide::{u8, usize};

    pub unsafe extern "sysv64" fn plain(x: u64) -> u8 {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: u64) -> u8 {
        ret(x);
    }

    pub unsafe extern "sysv64" fn refused(x: u64) -> u8 {
        ret(y);
    }

    pub unsafe extern "sysv64" fn parameter(x: usize) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    use crate::wide::u8;

    pub unsafe extern "ptx-kernel" fn plain(x: u8, p: *mut core::primitive::u32) {
        st.global.u32([p], 1);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: u8, p: *mut core::primitive::u32) {
        st.global.u32([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: u8, p: *mut core::primitive::u32) {
        st.global.u32([p], 1);
    }

    pub unsafe extern "ptx-kernel" fn refused(x: u8, p: *mut core::primitive::u32) {
        st.global.u32([y], 1);
    }
}

/// A number's name given a type with a size, `String`: not `Copy`, and no GPU argument.
pub mod owned {
    pub type f64 = String;
    pub type u16 = String;
}

#[kernite_macros::kernels]
pub mod cpu_kinds {
    use crate::owned::f64;
    use crate::wide::usize;

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: usize) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: usize) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: usize) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn owned(x: f64) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_kinds {
    use crate::owned::u16;
    use crate::wide::u8;

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(x: u8, p: *mut core::primitive::u32) {
        st.global.u32([p], N);
    }

    pub unsafe extern "ptx-kernel" fn owned(x: u16, p: *mut core::primitive::u32) {
        st.global.u32([p], 1);
    }
}
