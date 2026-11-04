//! Cooperative x86 kernels, of instances and not, whose types name a type that is not there.
//! Rust says each once, at the type, as it does of any function.

#[kernite_macros::kernels]
pub mod cpu {
    #[cooperative]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn together<const N: usize>(p: *mut Missing) {
        mov(byte[p], N);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn alone(p: *mut Absent) {
        mov(byte[p], 1);
    }
}
