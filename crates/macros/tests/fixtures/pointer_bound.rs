//! Kernels that take a pointer to a type Rust refuses by itself: `NeedsCopy<String>`, a struct whose
//! argument must be `Copy`, which `String` is not. It is Rust's error, said once, at the type, in a
//! kernel of every kind, and not again at the module's attribute: an x86 kernel, plain, of
//! compile-time arguments, cooperative, and cooperative of compile-time arguments, one that returns
//! the pointer, plain or of compile-time arguments, one with two such pointers, and one kernite
//! refuses for another reason; and a GPU kernel, plain, of compile-time arguments, cooperative, and
//! cooperative of compile-time arguments, and one kernite refuses.

/// A struct whose argument must be `Copy`.
pub struct NeedsCopy<T: Copy>(pub T);

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn plain(p: *const crate::NeedsCopy<String>, n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *mut crate::NeedsCopy<String>, n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *const crate::NeedsCopy<String>, n: usize) {
        mov(qword[p], n);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(p: *mut crate::NeedsCopy<String>, n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn returns(p: *const u8) -> *const crate::NeedsCopy<String> {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn returns_tiles<const N: u64>(p: *const u8) -> *mut crate::NeedsCopy<String> {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn two(p: *const crate::NeedsCopy<String>, q: *mut crate::NeedsCopy<Vec<u8>>) {
        mov(qword[q], p);
    }

    pub unsafe extern "sysv64" fn refused(p: *const crate::NeedsCopy<String>, n: usize) {
        mov(qword[p], y);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn refused_tiles<const N: u64>(p: *const crate::NeedsCopy<String>, n: usize) {
        mov(qword[p], y);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn plain(p: *mut crate::NeedsCopy<String>, n: u32) {
        st.global.u32([p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(p: *const crate::NeedsCopy<String>) {
        st.global.u32([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(p: *mut crate::NeedsCopy<String>, n: u32) {
        st.global.u32([p], n);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(p: *mut crate::NeedsCopy<String>) {
        st.global.u32([p], N);
    }

    pub unsafe extern "ptx-kernel" fn refused(p: *mut crate::NeedsCopy<String>, n: u32) {
        st.global.u32([p], y);
    }
}
