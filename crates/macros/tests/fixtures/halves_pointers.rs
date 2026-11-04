//! x86 kernels that take or give back a pointer to an `f16`: kernite takes a pointer to any type, so
//! Rust's word on the type is the only one, that its `f16` is unstable, said once, at the type, in a
//! kernel of every kind.

#[kernite_macros::kernels]
pub mod halves {
    pub unsafe extern "sysv64" fn plain(x: *mut f16) {
        mov(qword[x], 0);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: *mut f16) {
        mov(qword[x], N);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut f16) {
        mov(qword[x], 0);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: *mut f16) {
        mov(qword[x], N);
    }

    pub unsafe extern "sysv64" fn result(x: u64) -> *mut f16 {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles_result<const N: u64>(x: u64) -> *mut f16 {
        ret(x);
    }
}
