//! Kernels with pointers, in every kind of kernel the macros write a check for, built without a word
//! under `#![deny(warnings)]`: a pointer to a struct with a lifetime, which Rust takes without one
//! in a parameter, `Option<&u8>`, and a tuple with a reference; a pointer a kernel returns, and a
//! kernel that never returns; a parameter and a kernel named `thin`; kernels deprecated, by
//! `#[deprecated]` and by `cfg_attr`, and kernels configured out, by `#[cfg]` and by `cfg_attr`. Each
//! is plain, of compile-time arguments and cooperative, on x86, and a GPU kernel too.

#![deny(warnings)]

/// A struct with a lifetime, which `*const Bar` names without one.
pub struct Bar<'a>(pub &'a u8);

/// The x86 kernels.
#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn bar(p: *const crate::Bar, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn option(p: *const Option<&u8>, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn pair(p: *const (&u8, u8), n: usize) {
        mov(qword[p], n);
    }

    /// A kernel named as the function the macro writes to ask Rust is named, in the plain kernel's own
    /// `naked_asm!`.
    pub unsafe extern "sysv64" fn thin(p: *const u8, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn named(thin: *const u8, n: usize) {
        mov(qword[thin], n);
    }

    pub unsafe extern "sysv64" fn given(p: *const crate::Bar) -> *const crate::Bar {
        ret(p);
    }

    pub unsafe extern "sysv64" fn stop(p: *const u8) -> ! {
        ud2();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *const crate::Bar, n: usize) -> *const crate::Bar {
        add(n, N);
        mov(qword[p], n);
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles_named<const N: u64>(thin: *const Option<&u8>, n: usize) -> *const (&u8, u8) {
        add(n, N);
        mov(qword[thin], n);
        ret(thin);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *const Option<&u8>, n: usize) {
        mov(qword[p], n);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(p: *const (&u8, u8), n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[deprecated]
    pub unsafe extern "sysv64" fn old(p: *const Option<&u8>, n: usize) -> *const (&u8, u8) {
        mov(qword[p], n);
        ret(p);
    }

    #[cfg_attr(all(), deprecated)]
    pub unsafe extern "sysv64" fn older(p: *const Option<&u8>, n: usize) {
        mov(qword[p], n);
    }

    #[deprecated]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn old_tiles<const N: u64>(p: *const crate::Bar, n: usize) -> *const crate::Bar {
        add(n, N);
        ret(p);
    }

    #[cfg_attr(all(), deprecated)]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn older_tiles<const N: u64>(p: *const (&u8, u8), n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[deprecated]
    #[cooperative]
    pub unsafe extern "sysv64" fn old_together(p: *const crate::Bar, n: usize) {
        mov(qword[p], n);
    }

    #[cfg_attr(all(), deprecated)]
    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn older_together<const N: u64>(p: *const Option<&u8>, n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cfg(any())]
    pub unsafe extern "sysv64" fn gone(p: *const Option<&u8>, n: usize) -> *const (&u8, u8) {
        mov(qword[p], n);
        ret(p);
    }

    #[cfg_attr(all(), cfg(any()))]
    pub unsafe extern "sysv64" fn gone_too(p: *const Option<&u8>, n: usize) {
        mov(qword[p], n);
    }

    #[cfg(any())]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn gone_tiles<const N: u64>(p: *const crate::Bar, n: usize) -> *const crate::Bar {
        add(n, N);
        ret(p);
    }

    #[cfg_attr(all(), cfg(any()))]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn gone_too_tiles<const N: u64>(p: *const (&u8, u8), n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cfg(any())]
    #[cooperative]
    pub unsafe extern "sysv64" fn gone_together(p: *const crate::Bar, n: usize) {
        mov(qword[p], n);
    }

    #[cfg_attr(all(), cfg(any()))]
    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn gone_too_together<const N: u64>(p: *const Option<&u8>, n: usize) {
        add(n, N);
        mov(qword[p], n);
    }
}

/// The GPU kernels.
#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn bar(p: *mut crate::Bar, n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn option(p: *mut Option<&u8>, n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn pair(p: *mut (&u8, u8), n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn thin(thin: *mut u8, n: u32) {
        st.global.u32([thin], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(p: *mut crate::Bar) {
        st.global.u32([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(p: *mut Option<&u8>, n: u32) {
        st.global.u32([p], n);
    }

    #[deprecated]
    pub unsafe extern "ptx-kernel" fn old(p: *mut crate::Bar, n: u32) {
        st.global.u32([p], n);
    }

    #[cfg_attr(all(), deprecated)]
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn older_together(p: *mut Option<&u8>, n: u32) {
        st.global.u32([p], n);
    }

    #[cfg(any())]
    pub unsafe extern "ptx-kernel" fn gone(p: *mut crate::Bar, n: u32) {
        st.global.u32([p], n);
    }

    #[cfg_attr(all(), cfg(any()))]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn gone_too_tiles<const N: u32>(p: *mut (&u8, u8)) {
        st.global.u32([p], N);
    }
}
