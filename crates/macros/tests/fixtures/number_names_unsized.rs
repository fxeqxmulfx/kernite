//! A number's name that the module gives a type with no size, `u8` a slice. Rust says what it finds of
//! each number a kernel takes or gives back at the type, and the questions bring no error at the
//! module's attribute, where it would say that a call of a kernel that gives back such a value, or a
//! local that holds it, has no size. So it is for an x86 kernel with the number as its result, plain,
//! of compile-time arguments and one kernite refuses, and for a GPU kernel with it as a parameter,
//! plain, of compile-time arguments, cooperative and one kernite refuses. The errors at the attribute
//! are those of the code the macros write for the kernel itself, an instance's or a launcher's, which
//! the questions do not change.

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
