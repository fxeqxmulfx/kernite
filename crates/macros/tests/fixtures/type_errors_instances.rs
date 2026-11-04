//! Kernels of instances whose types name a type that is not there: a pointer to one in a
//! parameter, and in a result. Rust says each once, at the type, as it does of any function.

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
}
