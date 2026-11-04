//! Each thing wrong with a kernel's tuning directives, which the macro says where it is written,
//! once: a value, for each instance of a kernel of them, in the kernel's directive; a directive on
//! a function that is no GPU kernel, once however many kernels call it, and on a helper nothing
//! calls; and a directive that is not one, in a module of its own, for it stops the module.

#[kernite_macros::kernels]
pub mod values {
    #[maxntid(2048)]
    pub unsafe extern "ptx-kernel" fn too_wide() {}

    #[reqntid(32, 33)]
    pub unsafe extern "ptx-kernel" fn too_many() {}

    #[maxntid(256)]
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn both() {}

    #[minnctapersm(2)]
    pub unsafe extern "ptx-kernel" fn alone() {}

    #[maxnreg(16)]
    pub unsafe extern "ptx-kernel" fn few_registers() {}

    #[instances(64, 2048)]
    #[reqntid(N)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>() {}
}

#[kernite_macros::kernels]
pub mod elsewhere {
    #[maxnreg(32)]
    extern "ptx-func" fn called() {}

    #[maxnreg(32)]
    fn helper() {
        bar.sync(0);
    }

    #[maxnreg(32)]
    fn never_called() {}

    pub unsafe extern "ptx-kernel" fn one() {
        helper();
    }

    pub unsafe extern "ptx-kernel" fn two() {
        helper();
    }
}

#[kernite_macros::kernels]
pub mod cpu {
    #[maxntid(256)]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>() {}
}

#[kernite_macros::kernels]
pub mod old {
    #[maxnctapersm(2)]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod no_number {
    #[maxnreg]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod twice {
    #[maxnreg(32)]
    #[maxnreg(32)]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod not_whole {
    #[maxntid(1.5)]
    pub unsafe extern "ptx-kernel" fn k() {}
}
