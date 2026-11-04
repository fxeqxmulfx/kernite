//! Modules that build, and an import of the crate's that each has: Rust warns that it is unused where
//! nothing names it, and where only a helper that no kernel calls does; it says nothing where a kernel
//! names it. A module that builds has no error of kernite's, so Rust is shown no helper's types.

pub struct Ret(pub u64);

#[kernite_macros::kernels]
pub mod x86_unused {
    use super::Ret;

    fn h(x: gpr) {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        h(x);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod x86_named_by_kernel {
    use super::Ret;

    fn h(x: gpr) {
        nop();
    }

    pub unsafe extern "sysv64" fn k(p: *const Ret) -> *const Ret {
        h(p);
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod x86_uncalled_helper {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu_unused {
    use super::Ret;

    extern "ptx-func" fn g(x: u32) -> u32 {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn k(p: *mut u32, n: u32) {
        let v: u32;
        call(v, g, (n));
        st.global.u32([p], v);
    }
}

#[kernite_macros::kernels]
pub mod gpu_named_by_kernel {
    use super::Ret;

    extern "ptx-func" fn g(x: u32) -> u32 {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn k(p: *mut Ret, n: u32) {
        let v: u32;
        call(v, g, (n));
        st.global.u32([p], v);
    }
}
