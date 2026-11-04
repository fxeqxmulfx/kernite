//! Kernels that take a pointer to a type with no size which kernite cannot tell by what is
//! written, and which a clause on the type could not hold: a type with a `&` in it, as
//! `(&u8, CStr)` and `GTail<[&u8]>` are, and `u8`, which the module takes from the crate, where
//! it is a name for a slice. Rust says the size cannot be known, at the type, in a kernel of
//! every kind: plain, of compile-time arguments and cooperative, on x86 and on a GPU. Each is one
//! error.

/// A name for a slice, in place of the number's.
#[allow(non_camel_case_types)]
pub type u8 = [core::primitive::u8];

/// A struct whose last field is a `T` that may have no size.
pub struct GTail<T: ?Sized> {
    pub head: u32,
    pub rest: T,
}

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn tuple(p: *const (&u8, core::ffi::CStr), n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tuple_tiles<const N: u64>(p: *const (&u8, core::ffi::CStr), n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn tuple_together(p: *const (&u8, core::ffi::CStr), n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn tail(p: *const crate::GTail<[&u8]>, n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tail_tiles<const N: u64>(p: *const crate::GTail<[&u8]>, n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn tail_together(p: *const crate::GTail<[&u8]>, n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod shadowed_cpu {
    use crate::u8;

    pub unsafe extern "sysv64" fn plain(p: *const u8, n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *const u8, n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *const u8, n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn tuple(p: *mut (&u8, core::ffi::CStr), n: u32) {
        st.global.u32([p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tuple_tiles<const N: u32>(p: *mut (&u8, core::ffi::CStr)) {
        st.global.u32([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn tuple_together(p: *mut (&u8, core::ffi::CStr), n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn tail(p: *mut crate::GTail<[&u8]>, n: u32) {
        st.global.u32([p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tail_tiles<const N: u32>(p: *mut crate::GTail<[&u8]>) {
        st.global.u32([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn tail_together(p: *mut crate::GTail<[&u8]>, n: u32) {
        st.global.u32([p], n);
    }
}

#[kernite_macros::kernels]
pub mod shadowed_gpu {
    use crate::u8;

    pub unsafe extern "ptx-kernel" fn plain(p: *mut u8, n: u32) {
        st.global.u32([p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(p: *mut u8) {
        st.global.u32([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(p: *mut u8, n: u32) {
        st.global.u32([p], n);
    }
}
