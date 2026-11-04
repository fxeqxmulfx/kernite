//! Kernels with pointers, in every kind of kernel the macros write a check for, built without a word
//! under `#![deny(warnings)]`: a pointer to a struct with a lifetime, which Rust takes without one
//! in a parameter, `Option<&u8>`, and a tuple with a reference; a pointer a kernel returns, and a
//! kernel that never returns; parameters named as the locals the macros write, `__0` and on, and a
//! struct named as the trait they write; kernels deprecated, by `#[deprecated]` and by `cfg_attr`, and
//! kernels configured out, by `#[cfg]` and by `cfg_attr`. Each is plain, of compile-time arguments and
//! cooperative, on x86, and a GPU kernel too.

#![deny(warnings)]

/// A struct with a lifetime, which `*const Bar` names without one.
pub struct Bar<'a>(pub &'a u8);

/// A struct named as the trait the macros write to ask Rust whether a pointer has a size, which a
/// kernel's type means in a module that has it by its name.
pub struct __KerniteThin(pub u8);

/// The x86 kernels.
#[kernite_macros::kernels]
pub mod cpu {
    use crate::__KerniteThin;

    pub unsafe extern "sysv64" fn bar(p: *const crate::Bar, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn option(p: *const Option<&u8>, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn pair(p: *const (&u8, u8), n: usize) {
        mov(qword[p], n);
    }

    /// Parameters named as the locals the macros write to ask Rust, in the plain kernel's own
    /// `naked_asm!`: `__0` and on, one for each parameter and for the result.
    pub unsafe extern "sysv64" fn locals(__0: *const u8, __1: *const Option<&u8>, __2: usize) -> *const u8 {
        mov(qword[__0], __2);
        ret(__0);
    }

    pub unsafe extern "sysv64" fn trait_named(p: *const __KerniteThin, n: usize) -> *const __KerniteThin {
        mov(qword[p], n);
        ret(p);
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
    pub unsafe extern "sysv64" fn tiles_named<const N: u64>(__0: *const Option<&u8>, __1: usize) -> *const (&u8, u8) {
        add(__1, N);
        mov(qword[__0], __1);
        ret(__0);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn trait_named_tiles<const N: u64>(
        p: *mut __KerniteThin,
        n: usize,
    ) -> *mut __KerniteThin {
        add(n, N);
        mov(qword[p], n);
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *const Option<&u8>, n: usize) {
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn trait_named_together(p: *const __KerniteThin, n: usize) {
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
    use crate::__KerniteThin;

    pub unsafe extern "ptx-kernel" fn bar(p: *mut crate::Bar, n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn option(p: *mut Option<&u8>, n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn pair(p: *mut (&u8, u8), n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn locals(__0: *mut u8, __1: *mut Option<&u8>, __2: u32) {
        st.global.u32([__0], __2);
    }

    pub unsafe extern "ptx-kernel" fn trait_named(p: *mut __KerniteThin, n: u32) {
        st.global.u32([p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn trait_named_tiles<const N: u32>(p: *const __KerniteThin) {
        st.global.u32([p], N);
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
