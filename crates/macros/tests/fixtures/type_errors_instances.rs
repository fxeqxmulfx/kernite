//! Kernels of instances whose types are wrong: a pointer to a type that is not there, in a parameter and
//! in a result, and a pointer to one with no size, in a result. Rust says each once, at the type.

#[kernite_macros::kernels]
pub mod cpu {
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn parameter<const N: usize>(p: *mut Missing) {
        mov(byte[p], N);
    }

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn result<const N: usize>(p: *mut u8) -> *mut Lost {
        mov(byte[p], N);
        ret(p);
    }

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn unsized_result<const N: usize>(p: *mut u8) -> *mut core::ffi::CStr {
        mov(byte[p], N);
        ret(p);
    }
}
