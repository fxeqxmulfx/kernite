//! A module that gives `core` or `std` another meaning, `use crate::fake as core;`, has Rust say that a number written
//! as its path is not the number, as it says it of the number's name that a module gives another type: once, at the
//! type, for each parameter and result of an x86 kernel and each parameter of a GPU kernel. A path from the root,
//! `::core` or `::std`, is the crate `core` or `std` whatever the module gives, so Rust says nothing of it. Each module
//! of paths here is a module of names that follows it.

#![allow(non_camel_case_types)]

/// The `primitive` of a `core` or `std` of the crate's own, whose numbers are other types.
pub mod fake {
    pub mod primitive {
        pub type u64 = ::core::primitive::u32;
        pub type f32 = ::core::primitive::f64;
        pub type u32 = ::core::primitive::u64;
    }
}

/// The same numbers as the names a module takes.
pub mod alias {
    pub type u64 = ::core::primitive::u32;
    pub type f32 = ::core::primitive::f64;
    pub type u32 = ::core::primitive::u64;
}

/// x86 kernels by the paths: a parameter and a result, of which the first and the last are `core`'s, the pointee, which
/// Rust is not asked of, and one from the root.
#[kernite_macros::kernels]
pub mod paths_x86 {
    use crate::fake as core;

    pub unsafe extern "sysv64" fn both(x: core::primitive::u64) -> core::primitive::u64 {
        ret(x);
    }
    pub unsafe extern "sysv64" fn floats(x: core::primitive::f32) -> ::std::primitive::f32 {
        ret(x);
    }
    pub unsafe extern "sysv64" fn pointee(p: *mut core::primitive::u64) {
        mov(qword[p], 1);
    }
    pub unsafe extern "sysv64" fn root(x: ::core::primitive::u64) -> ::std::primitive::u64 {
        ret(x);
    }
}

/// The same x86 kernels by the names.
#[kernite_macros::kernels]
pub mod named_x86 {
    use crate::alias::*;

    pub unsafe extern "sysv64" fn both(x: u64) -> u64 {
        ret(x);
    }
    pub unsafe extern "sysv64" fn floats(x: f32) -> f32 {
        ret(x);
    }
    pub unsafe extern "sysv64" fn pointee(p: *mut u64) {
        mov(qword[p], 1);
    }
}

/// GPU kernels by the paths, with `std` given the meaning: a parameter, the pointee, and one from the root.
#[kernite_macros::kernels]
pub mod paths_gpu {
    use crate::fake as std;

    pub unsafe extern "ptx-kernel" fn both(x: std::primitive::u32, p: *mut std::primitive::u32) {
        st.global.u32([p], x);
    }
    pub unsafe extern "ptx-kernel" fn root(x: ::std::primitive::u32) {}
}

/// The same GPU kernels by the names.
#[kernite_macros::kernels]
pub mod named_gpu {
    use crate::alias::*;

    pub unsafe extern "ptx-kernel" fn both(x: u32, p: *mut u32) {
        st.global.u32([p], x);
    }
}
