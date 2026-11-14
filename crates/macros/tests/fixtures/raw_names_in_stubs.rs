//! A kernel that does not compile, with a parameter or a compile-time argument named by a name Rust lets be written raw
//! in no place, `r#self`, `r#Self`, `r#super` or `r#crate`: on x86 and on a GPU, alone, of compile-time arguments and
//! cooperative. The macros write the kernel for Rust with a body that never runs, which uses each parameter, and a
//! launcher that never runs. kernite says that the kernel is not `unsafe`, and Rust refuses each name at each place it
//! is written, in its own words: there is no more, not again at the module's attribute, nor of the `&self` and `Self`
//! of the launcher of a GPU kernel.

#[kernite_macros::kernels]
pub mod cpu {
    pub extern "sysv64" fn plain(r#self: u64, _: u64) -> u64 {
        ret(r#self);
    }

    #[instances(1, 2)]
    pub extern "sysv64" fn of_arguments<const r#crate: u64>(r#Self: *mut u64) {
        mov(qword[r#Self], r#crate);
    }

    #[cooperative]
    pub extern "sysv64" fn together(r#super: *mut u64) {
        mov(qword[r#super], 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub extern "ptx-kernel" fn plain(r#self: *mut u64, _: u64) {
        st.global.u64([r#self], 1);
    }

    #[instances(1, 2)]
    pub extern "ptx-kernel" fn of_arguments<const r#crate: u32>(r#Self: *mut u32) {
        st.global.u32([r#Self], r#crate);
    }

    #[cooperative]
    pub extern "ptx-kernel" fn together(r#super: *mut u64) {
        st.global.u64([r#super], 1);
    }
}
