//! Helpers and `extern "ptx-func"` functions whose signatures name `Ret`, a type of the crate's that their
//! module imports and kernite refuses: at the argument of a call of the helper, or at the type. The macros
//! write no function for either, but they show Rust the types, so Rust does not call the import unused,
//! and says nothing beside kernite's error, which is said once. Each module is a case: a helper's
//! parameter, its result and its compile-time argument, the types nested in them, and a function's.

pub struct Ret(pub u64);

#[kernite_macros::kernels]
pub mod x86_helper_param {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        h(x);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod x86_helper_result {
    use super::Ret;

    fn h(x: u64) -> Ret {
        ret(x);
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        h(x);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod x86_helper_generic {
    use super::Ret;

    fn h<const K: u64, const N: Ret>(x: gpr) {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        h::<1, 2>(x);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod x86_helper_nested {
    use super::Ret;

    fn array(x: [Ret; 2]) {
        nop();
    }

    fn pointer(x: *const Ret) {
        nop();
    }

    fn option(x: Option<Ret>) {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        array(x);
        pointer(x);
        option(x);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod x86_helper_of_instances {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn k<const N: u64>(x: u64) -> u64 {
        h(x);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod x86_helper_of_helper {
    use super::Ret;

    fn inner(x: Ret) {
        nop();
    }

    fn outer(x: gpr) {
        inner(x);
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        outer(x);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu_func_param {
    use super::Ret;

    pub unsafe extern "ptx-func" fn g(x: Ret) -> u32 {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn k(p: *mut u32, n: u32) {
        st.global.u32([p], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu_func_result {
    use super::Ret;

    pub unsafe extern "ptx-func" fn g(x: u32) -> Ret {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn k(p: *mut u32, n: u32) {
        st.global.u32([p], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu_func_generic {
    use super::Ret;

    #[instances(1)]
    pub unsafe extern "ptx-func" fn g<const N: Ret>(x: u32) -> u32 {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn k(p: *mut u32, n: u32) {
        st.global.u32([p], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu_func_of_instances {
    use super::Ret;

    #[instances(1, 2)]
    pub unsafe extern "ptx-func" fn g<const N: u32>(x: Ret) -> u32 {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn k(p: *mut u32, n: u32) {
        st.global.u32([p], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu_helper_param {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    pub unsafe extern "ptx-kernel" fn k(p: *mut u32, n: u32) {
        h(n);
        st.global.u32([p], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu_helper_result {
    use super::Ret;

    fn h(x: u32) -> Ret {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn k(p: *mut u32, n: u32) {
        h(n);
        st.global.u32([p], n);
    }
}
