//! A kernel's parameter named by a name Rust lets be written raw in no place, `r#self`, `r#Self`, `r#super`,
//! `r#crate` or `r#_`, in a kernel of every kind: a GPU kernel, alone, of compile-time arguments and cooperative; an
//! x86 kernel, plain, of compile-time arguments and cooperative. Rust refuses each at each place it is written, in its
//! own words. The macros make it say no more of the name: not again at the module's attribute, where the macros
//! would write the name as text, and not of the `&self` and `Self` that the launcher of a GPU kernel is written with.

/// GPU kernels. A pointer is passed by its address and a number by its value, and the launcher is a method with
/// `&self`, which a parameter named `self` would be bound beside, and `Self`, which one named `Self` would be.
#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn by_self(r#self: *mut u64) {
        st.global.u64([r#self], 1);
    }

    pub unsafe extern "ptx-kernel" fn by_upper_self(r#Self: *mut u64) {
        st.global.u64([r#Self], 1);
    }

    pub unsafe extern "ptx-kernel" fn by_super(r#super: *mut u64) {
        st.global.u64([r#super], 1);
    }

    pub unsafe extern "ptx-kernel" fn by_crate(r#crate: *mut u64) {
        st.global.u64([r#crate], 1);
    }

    pub unsafe extern "ptx-kernel" fn by_underscore(r#_: *mut u64) {
        st.global.u64([r#_], 1);
    }

    pub unsafe extern "ptx-kernel" fn a_number(r#self: u64, r#Self: *mut u64) {
        st.global.u64([r#Self], r#self);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn of_arguments<const N: u32>(r#crate: *mut u32, r#super: u32) {
        add.u32(r#super, r#super, N);
        st.global.u32([r#crate], r#super);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(r#self: *mut u64) {
        st.global.u64([r#self], 1);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_of_arguments<const N: u32>(r#Self: *mut u32) {
        st.global.u32([r#Self], N);
    }
}

/// x86 kernels. The macros write Rust of their own for those of compile-time arguments and the cooperative ones,
/// an instance for each value and a call that picks it, and only a copy of the signature for the others.
#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn by_self(r#self: u64) -> u64 {
        ret(r#self);
    }

    pub unsafe extern "sysv64" fn by_upper_self(r#Self: u64) -> u64 {
        ret(r#Self);
    }

    pub unsafe extern "sysv64" fn by_super(r#super: u64) -> u64 {
        ret(r#super);
    }

    pub unsafe extern "sysv64" fn by_crate(r#crate: u64) -> u64 {
        ret(r#crate);
    }

    pub unsafe extern "sysv64" fn by_underscore(r#_: u64) -> u64 {
        ret(r#_);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn of_arguments<const N: u64>(r#crate: u64, r#self: *mut u64) -> u64 {
        add(r#crate, N);
        mov(qword[r#self], r#crate);
        ret(r#crate);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(r#crate: *mut u64) {
        mov(qword[r#crate], 1);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_of_arguments<const N: u64>(r#Self: *mut u64, r#_: u64) {
        mov(qword[r#Self], N);
    }
}
