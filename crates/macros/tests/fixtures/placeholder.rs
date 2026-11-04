//! Kernels with a `_` in a type that a pointer points to: `*const [_; 3]` in a parameter, `-> *const _`
//! and `-> *mut (u8, _)` in the result, and `*mut (u8, _)` on a GPU; and one with a `_` in a result that
//! is no pointer, `-> [_; 3]`, which kernite refuses. Rust does not allow a `_` in a signature, and
//! says so, once, at the `_`, in a kernel of every kind, and nothing else: nothing at the module's
//! attribute, nor that it cannot tell the type of the `_`. An x86 kernel, plain, of compile-time
//! arguments, cooperative, and one kernite refuses for another reason, and a GPU kernel, plain, of
//! compile-time arguments, cooperative, and one kernite refuses. A result of a cooperative x86 kernel
//! or of a GPU kernel is kernite's error, and Rust says its own beside it, at the `_`, once.

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn plain(p: *const [_; 3], n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn plain_returns(p: *const u8) -> *const _ {
        ret(p);
    }

    pub unsafe extern "sysv64" fn plain_returns_pair(p: *const u8) -> *mut (u8, _) {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *const [_; 3], n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles_returns<const N: u64>(p: *const u8) -> *const _ {
        add(p, N);
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles_returns_pair<const N: u64>(p: *const u8) -> *mut (u8, _) {
        add(p, N);
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *const [_; 3], n: usize) {
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together_returns(p: *const u8) -> *const _ {
        ret(p);
    }

    pub unsafe extern "sysv64" fn refused(p: *const [_; 3], n: usize) {
        mov(qword[p], y);
    }

    pub unsafe extern "sysv64" fn refused_returns(p: *const u8) -> *const _ {
        mov(qword[p], y);
    }

    pub unsafe extern "sysv64" fn refused_returns_pair(p: *const u8) -> *mut (u8, _) {
        mov(qword[p], y);
    }

    pub unsafe extern "sysv64" fn returns_array(p: *const u8) -> [_; 3] {
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn plain(p: *mut (u8, _), n: u32) {
        st.global.u32([p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(p: *const (u8, _)) {
        st.global.u32([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(p: *mut (u8, _), n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn returns(p: *mut u32) -> *mut [_; 3] {
        st.global.u32([p], 1);
    }

    pub unsafe extern "ptx-kernel" fn refused(p: *mut (u8, _), n: u32) {
        st.global.u32([p], y);
    }
}

/// An array's whole length, `*const [u8; _]`, in a plain x86 kernel's parameter and result.
#[kernite_macros::kernels]
pub mod lengths {
    pub unsafe extern "sysv64" fn plain(p: *const [u8; _], n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn plain_returns(p: *const u8) -> *const [u8; _] {
        ret(p);
    }
}
