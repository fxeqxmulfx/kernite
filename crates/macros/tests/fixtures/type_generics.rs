//! Kernels whose generics are types: `<T>`, and `impl` in a parameter's type or the result's, which in a
//! parameter makes the function generic over a type, and in the result hides one. kernite refuses each at
//! the `T` or the first `impl`, in one set of words, and Rust says nothing: it is given no function to read.
//! The first error ends a module, so each is a module of its own, and a call to a kernel is no error of Rust's.

#[kernite_macros::kernels]
pub mod generic {
    pub unsafe extern "sysv64" fn k<T>(p: *const T, n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod plain {
    pub unsafe extern "sysv64" fn k(p: *const impl Copy, n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod instances {
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn k<const N: usize>(p: *const impl Copy, n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod cooperative {
    #[cooperative]
    pub unsafe extern "sysv64" fn k(p: *const impl Copy, n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod cooperative_instances {
    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn k<const N: usize>(p: *const impl Copy, n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn g(p: *mut impl Copy, n: u32) {
        let at: u64;
        mov.u64(at, p);
        st.global.u32([at], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu_instances {
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn g<const N: u32>(p: *mut impl Copy, n: u32) {
        let at: u64;
        mov.u64(at, p);
        st.global.u32([at], n);
    }
}

#[kernite_macros::kernels]
pub mod unsized_pointee {
    pub unsafe extern "sysv64" fn k(p: *const (impl ?Sized + core::fmt::Debug), n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod result {
    pub unsafe extern "sysv64" fn k(p: *mut u8) -> *mut impl Copy {
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod option {
    pub unsafe extern "sysv64" fn k(p: *const Option<impl Copy>, n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod array {
    pub unsafe extern "sysv64" fn k(p: *const [impl Copy; 2], n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod tuple {
    pub unsafe extern "sysv64" fn k(p: *const (u8, impl Copy), n: usize) {
        mov(qword[p], n);
    }
}

#[kernite_macros::kernels]
pub mod by_value {
    pub unsafe extern "sysv64" fn k(p: *mut u64, x: impl Copy) {
        mov(qword[p], 1);
    }
}

#[kernite_macros::kernels]
pub mod by_value_instances {
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn k<const N: usize>(p: *mut u64, x: impl Copy) {
        mov(qword[p], N);
    }
}

#[kernite_macros::kernels]
pub mod by_value_result {
    pub unsafe extern "sysv64" fn k(n: usize) -> impl Copy {
        ret(n);
    }
}

#[kernite_macros::kernels]
pub mod result_option {
    pub unsafe extern "sysv64" fn k(n: usize) -> Option<impl Copy> {
        ret(n);
    }
}

#[kernite_macros::kernels]
pub mod held {
    pub unsafe extern "sysv64" fn k(p: Option<*const impl Copy>, n: usize) {
        mov(qword[p], n);
    }
}

/// What a crate that uses these kernels writes. Rust says nothing of the calls: the errors above are
/// the only ones.
pub fn calls(p: *mut u64) {
    unsafe {
        plain::k(p, 7);
        instances::k::<1>(p, 9);
        result::k(p.cast());
    }
}
