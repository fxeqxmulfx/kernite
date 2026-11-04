//! Kernels that take a pointer to a type with no size, a tuple that ends in `CStr` or a struct whose last field is
//! a slice, that holds a `_` that is no placeholder: a pattern in a block, `[u8; { let _ = 1; 3 }]` or `Tail<{ let
//! _ = 1; 3 }>`; an argument of a macro, `[u8; len!(_)]` or `text!(_)`, which drops it; a pattern in an array's
//! length that is no block, `[u8; if let Some(_) = Some(1) { 3 } else { 4 }]`; and the name of a function
//! pointer's parameter, `fn(_: u8)`. Rust says that the size cannot be known, once, at the type, as it does of
//! one with none of these: in an x86 kernel, plain, of compile-time arguments and cooperative, and in a GPU
//! kernel.

/// A macro that drops what it is given: `len!(_)` is 3.
macro_rules! len {
    ($x:tt) => {
        3
    };
}

/// A macro that gives a type, and drops what it is given: `text!(_)` is `CStr`.
macro_rules! text {
    ($x:tt) => {
        core::ffi::CStr
    };
}

/// A type with a number as its argument, whose last field is a slice.
#[repr(C)]
pub struct Tail<const N: usize> {
    pub head: [u8; N],
    pub rest: [u8],
}

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn block(p: *const ([u8; { let _ = 1; 3 }], core::ffi::CStr), n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn block_tiles<const N: u64>(
        p: *const ([u8; { let _ = 1; 3 }], core::ffi::CStr),
        n: usize,
    ) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn block_together(p: *const ([u8; { let _ = 1; 3 }], core::ffi::CStr), n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn macro_args(p: *const ([u8; len!(_)], core::ffi::CStr), n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn macro_args_tiles<const N: u64>(p: *const ([u8; len!(_)], core::ffi::CStr), n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn macro_args_together(p: *const ([u8; len!(_)], core::ffi::CStr), n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn array_length(
        p: *const ([u8; if let Some(_) = Some(1) { 3 } else { 4 }], core::ffi::CStr),
        n: usize,
    ) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn function(p: *const (fn(_: u8), core::ffi::CStr), n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn const_argument(p: *const crate::Tail<{ let _ = 1; 3 }>, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn type_macro(p: *const text!(_), n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn block(p: *mut ([u8; { let _ = 1; 3 }], core::ffi::CStr), n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn macro_args(p: *mut ([u8; len!(_)], core::ffi::CStr), n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn array_length(
        p: *mut ([u8; if let Some(_) = Some(1) { 3 } else { 4 }], core::ffi::CStr),
        n: u32,
    ) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn function(p: *mut (fn(_: u8), core::ffi::CStr), n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn const_argument(p: *mut crate::Tail<{ let _ = 1; 3 }>, n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn type_macro(p: *mut text!(_), n: u32) {
        st.global.u32([p], n);
    }
}
