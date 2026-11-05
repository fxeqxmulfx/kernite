//! A kernel's compile-time argument named by a name Rust lets be written raw in no place, `r#self`, `r#Self`,
//! `r#super`, `r#crate` or `r#_`, in a kernel of every kind that takes one: a GPU kernel, alone and cooperative, and an
//! x86 kernel, alone and cooperative; named by a type as well, in a parameter and in a result, and beside parameters
//! and compile-time arguments named as the macros name a name of these. Rust refuses each at each place it is written,
//! in its own words. The macros make it say no more of the name: not again at the module's attribute, where the macros
//! would write the name as text to pick an instance, and not of the `&self` and `Self` that the launcher of a GPU
//! kernel is written with.

#![allow(unused_braces)]

/// GPU kernels. The launcher is generic over the argument, and picks an instance by its value.
#[kernite_macros::kernels]
pub mod gpu {
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn by_self<const r#self: u32>(p: *mut u32) {
        st.global.u32([p], r#self);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn by_upper_self<const r#Self: u32>(p: *mut u32) {
        st.global.u32([p], r#Self);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn by_super<const r#super: u32>(p: *mut u32) {
        st.global.u32([p], r#super);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn by_crate<const r#crate: u32>(p: *mut u32) {
        st.global.u32([p], r#crate);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn by_underscore<const r#_: u32>(p: *mut u32) {
        st.global.u32([p], r#_);
    }

    /// Named by the type of a parameter, alone and in braces.
    #[instances(2, 3)]
    pub unsafe extern "ptx-kernel" fn in_a_type<const r#crate: usize>(
        p: *mut [u8; r#crate],
        q: *mut [u8; { r#crate }],
    ) {
        st.global.u8([p], r#crate);
        st.global.u8([q], r#crate);
    }

    /// One argument named so beside one that is not.
    #[instances((2, 3), (4, 1))]
    pub unsafe extern "ptx-kernel" fn two<const W: usize, const r#Self: usize>(p: *mut [[u8; W]; r#Self]) {
        st.global.u8([p], W);
        st.global.u8([p + 1], r#Self);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together<const r#self: u32>(p: *mut u32) {
        st.global.u32([p], r#self);
    }

    #[cooperative]
    #[instances(2, 3)]
    pub unsafe extern "ptx-kernel" fn together_in_a_type<const r#super: usize>(p: *mut [u8; r#super]) {
        st.global.u8([p], r#super);
    }

    /// Beside parameters named as the macros name a parameter named `_` or by a name Rust refuses, `__kernite_0`
    /// among them, and an argument named so.
    #[instances((1, 2))]
    pub unsafe extern "ptx-kernel" fn beside<const r#crate: u32, const __kernite_0: u32>(
        r#self: u32,
        _: u32,
        __kernite_1: *mut u32,
        r#_: u32,
    ) {
        add.u32(r#self, r#self, r#crate);
        add.u32(r#self, r#self, __kernite_0);
        st.global.u32([__kernite_1], r#self);
    }
}

/// x86 kernels. The macros write an instance for each value and a call that picks it, by the value of the argument.
#[kernite_macros::kernels]
pub mod cpu {
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn by_self<const r#self: u64>(x: u64) -> u64 {
        add(x, r#self);
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn by_upper_self<const r#Self: u64>(x: u64) -> u64 {
        add(x, r#Self);
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn by_super<const r#super: u64>(x: u64) -> u64 {
        add(x, r#super);
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn by_crate<const r#crate: u64>(x: u64) -> u64 {
        add(x, r#crate);
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn by_underscore<const r#_: u64>(x: u64) -> u64 {
        add(x, r#_);
        ret(x);
    }

    /// Named by the type of a parameter, alone and in braces, and by the type of the result.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn in_a_type<const r#crate: usize>(
        p: *mut [u8; r#crate],
        q: *mut [u8; { r#crate }],
    ) -> *mut [u8; r#crate] {
        mov(byte[p], r#crate);
        mov(byte[q], r#crate);
        ret(p);
    }

    /// One argument named so beside one that is not.
    #[instances((2, 3), (4, 1))]
    pub unsafe extern "sysv64" fn two<const W: usize, const r#Self: usize>(p: *mut [[u8; W]; r#Self]) {
        mov(byte[p], W);
        mov(byte[p + 1], r#Self);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together<const r#self: u64>(p: *mut u64) {
        mov(qword[p], r#self);
    }

    #[cooperative]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn together_in_a_type<const r#super: usize>(p: *mut [u8; r#super]) {
        mov(byte[p], r#super);
    }

    /// Beside parameters named as the macros name a parameter named `_` or by a name Rust refuses, `__kernite_0`
    /// among them, and an argument named so.
    #[instances((1, 2))]
    pub unsafe extern "sysv64" fn beside<const r#crate: u64, const __kernite_0: u64>(
        r#self: u64,
        _: u64,
        __kernite_1: u64,
        r#_: u64,
    ) -> u64 {
        add(r#self, r#crate);
        add(r#self, __kernite_0);
        add(r#self, __kernite_1);
        ret(r#self);
    }
}
