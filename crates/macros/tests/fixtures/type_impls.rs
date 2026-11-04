//! `impl` in the type of a variable, a const or a static is no function's generic, and is said as it was.

#[kernite_macros::kernels]
pub mod lets {
    pub unsafe extern "sysv64" fn k(p: *mut u64) {
        let x: impl Copy;
        mov(qword[p], x);
    }
}

#[kernite_macros::kernels]
pub mod consts {
    pub const N: impl Copy = 3u64;

    pub unsafe extern "sysv64" fn k(p: *mut u64) {
        mov(qword[p], 1);
    }
}

#[kernite_macros::kernels]
pub mod read {
    pub const N: impl Copy = 3u64;

    pub unsafe extern "sysv64" fn k(p: *mut u64) {
        mov(qword[p], N);
    }
}

#[kernite_macros::kernels]
pub mod statics {
    static S: global<impl Copy>;

    pub unsafe extern "ptx-kernel" fn g(p: *mut u32) {
        st.global.u32([p], 1);
    }
}
