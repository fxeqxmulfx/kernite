//! x86 and GPU kernels that take or give back a pointer to a type kernite would refuse by value. kernite takes
//! a pointer of one word to any type, so an `f128` it points to is Rust's to say is unstable, said once, at the
//! type; and a pointer of two words, to a `[u64]`, is kernite's to refuse, and Rust says nothing of it.

#[kernite_macros::kernels]
pub mod cpu_float {
    pub unsafe extern "sysv64" fn plain(x: *mut f128) {
        mov(qword[x], 0);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: *mut f128) {
        mov(qword[x], N);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut f128) {
        mov(qword[x], 0);
    }

    pub unsafe extern "sysv64" fn result(x: *mut f128) -> *mut f128 {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles_result<const N: u64>(x: *mut f128) -> *mut f128 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod cpu_slice {
    pub unsafe extern "sysv64" fn plain(x: *mut [u64]) {
        mov(qword[x], 0);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: *mut [u64]) {
        mov(qword[x], N);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut [u64]) {
        mov(qword[x], 0);
    }

    pub unsafe extern "sysv64" fn result(x: *mut [u64]) -> *mut [u64] {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles_result<const N: u64>(x: *mut [u64]) -> *mut [u64] {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu_float {
    pub unsafe extern "ptx-kernel" fn plain(x: *mut f128) {
        st.global.u32([x], 0);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: *mut f128) {
        st.global.u32([x], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut f128) {
        st.global.u32([x], 0);
    }
}

#[kernite_macros::kernels]
pub mod gpu_slice {
    pub unsafe extern "ptx-kernel" fn plain(x: *mut [u64]) {
        st.global.u32([x], 0);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: *mut [u64]) {
        st.global.u32([x], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut [u64]) {
        st.global.u32([x], 0);
    }
}
