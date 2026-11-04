//! Helpers and `extern "ptx-func"` functions under `cfg`, whose signatures name an import of the crate's, in
//! modules where kernite says an error: of a kernel that does not compile, of the module as a whole, or of the
//! GPU's module. Rust sees their types where `cfg` keeps the function and the error is said, there: a function
//! `cfg` leaves out names nothing, and where `cfg` leaves out what the error is of, the module shows Rust none.

pub struct Ret(pub u64);

/// A helper under `a`, which a kernel under `a` calls, and kernite refuses its argument.
#[kernite_macros::kernels]
pub mod together {
    use super::Ret;

    #[cfg(feature = "a")]
    fn h(x: Ret) {
        nop();
    }

    #[cfg(feature = "a")]
    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        h(x);
        ret(x);
    }
}

/// A helper under `a`, and a kernel that does not compile, kept whatever holds.
#[kernite_macros::kernels]
pub mod helper_only {
    use super::Ret;

    #[cfg(feature = "a")]
    fn h(x: Ret) {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        ret(y);
    }
}

/// A helper kept whatever holds, and a kernel that does not compile under `a`.
#[kernite_macros::kernels]
pub mod kernel_only {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    #[cfg(feature = "a")]
    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        ret(y);
    }
}

/// A function under `a` whose result is refused, and a kernel that does not compile, kept whatever holds.
#[kernite_macros::kernels]
pub mod function_only {
    use super::Ret;

    #[cfg(feature = "a")]
    extern "ptx-func" fn g(x: u32) -> Ret {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn k(p: *mut u32) {
        st.global.u32([p], y);
    }
}

/// A function kept whatever holds, a kernel that does not compile under `b`, and a helper under `a`.
#[kernite_macros::kernels]
pub mod either {
    use super::Ret;

    #[cfg(feature = "a")]
    fn h(x: Ret) {
        nop();
    }

    extern "ptx-func" fn g(x: u32) -> u32 {
        ret(x);
    }

    #[cfg(feature = "b")]
    pub unsafe extern "ptx-kernel" fn k(p: *mut u32) {
        st.global.u32([p], y);
    }
}

/// A helper under `a`, a kernel that does not compile under `a`, and another under `b`: kernite says an error where
/// either kernel is kept, and Rust is shown the helper's types where it does and `cfg` keeps the helper: under `a`.
#[kernite_macros::kernels]
pub mod both {
    use super::Ret;

    #[cfg(feature = "a")]
    fn h(x: Ret) {
        nop();
    }

    #[cfg(feature = "a")]
    pub unsafe extern "sysv64" fn ka(x: u64) -> u64 {
        ret(y);
    }

    #[cfg(feature = "b")]
    pub unsafe extern "sysv64" fn kb(x: u64) -> u64 {
        ret(y);
    }
}

/// A helper under `a`, a kernel under `a`, and another under `b` that no module of PTX has with the first: kernite
/// says an error where both are kept, and Rust is shown the helper's types there.
#[kernite_macros::kernels]
pub mod no_module_where_both {
    use super::Ret;

    #[cfg(feature = "a")]
    fn h(x: Ret) {
        nop();
    }

    #[cfg(feature = "a")]
    pub unsafe extern "ptx-kernel" fn old(n: u32) {
        let i: u32;
        shfl.idx.b32(i, n, 0, 31);
    }

    #[cfg(feature = "b")]
    pub unsafe extern "ptx-kernel" fn new(f: f32) {
        let v: f32;
        tanh.approx.f32(v, f);
    }
}

/// A helper under `a`, in a module that is two: kernite says that under no `cfg`, and Rust is shown the helper's
/// types where `cfg` keeps the helper.
#[kernite_macros::kernels]
pub mod two_modules {
    use super::Ret;

    static COUNTS: global<[u32; 4]>;

    #[cfg(feature = "a")]
    fn h(x: Ret) {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn g(p: *mut u32) {
        let v: u32;
        ld.global.u32(v, [COUNTS]);
        st.global.u32([p], v);
    }
}
