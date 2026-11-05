//! A macro's fragment `$t:ty` that holds bounds, `dyn Debug + Send`, is one type behind a pointer or a reference, as
//! Rust reads it, wherever the macros write it back: in a parameter and in a result, of an x86 kernel, of a GPU kernel
//! and of a helper, and in a type that holds the pointer or the reference: a tuple, an array, a function pointer or an
//! argument of a type. kernite refuses each, for a pointer to such a type has two words and a kernel takes no
//! reference. Rust says nothing of the fragment, as it says nothing of it in a macro that does not call kernite: not
//! that its `+` is ambiguous, as it says of `*const dyn Debug + Send` written out. Each is built again with the bounds
//! in parentheses, `(dyn Debug + Send)`, as written by hand: kernite says the same of it.

/// A type of the crate's with an argument that is a type, of any size.
pub struct Wrap<T: ?Sized>(core::marker::PhantomData<T>);

/// x86 kernels that take a pointer or a reference to `$t`.
macro_rules! x86_params {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn a(x: *const $t) {
                ret();
            }
            pub unsafe extern "sysv64" fn b(x: *mut $t) {
                ret();
            }
            pub unsafe extern "sysv64" fn c(x: &$t) {
                ret();
            }
            pub unsafe extern "sysv64" fn d(x: &mut $t) {
                ret();
            }
            pub unsafe extern "sysv64" fn e(x: &'static $t) {
                ret();
            }
        }
    };
}

x86_params!(x86_params_bounds, dyn core::fmt::Debug + Send);
x86_params!(x86_params_parens, (dyn core::fmt::Debug + Send));

/// x86 kernels that give back a pointer or a reference to `$t`. A reference with no lifetime of its own takes the one
/// of the reference that is the parameter, as Rust has it: kernite refuses that parameter, and says nothing of the
/// result.
macro_rules! x86_results {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn a(x: u64) -> *const $t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn b(x: u64) -> *mut $t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn c(x: u64) -> &'static $t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn d(x: u64) -> &'static mut $t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn e(x: &u8) -> &$t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn f(x: &u8) -> &mut $t {
                ret(x);
            }
        }
    };
}

x86_results!(x86_results_bounds, dyn core::fmt::Debug + Send);
x86_results!(x86_results_parens, (dyn core::fmt::Debug + Send));

/// GPU kernels that take a pointer or a reference to `$t`.
macro_rules! gpu_params {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "ptx-kernel" fn a(x: *const $t) {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn b(x: *mut $t) {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn c(x: &$t) {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn d(x: &mut $t) {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn e(x: &'static $t) {
                ret();
            }
        }
    };
}

gpu_params!(gpu_params_bounds, dyn core::fmt::Debug + Send);
gpu_params!(gpu_params_parens, (dyn core::fmt::Debug + Send));

/// GPU kernels that give back a pointer or a reference to `$t`, which a GPU kernel does not.
macro_rules! gpu_results {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "ptx-kernel" fn a() -> *const $t {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn b() -> *mut $t {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn c() -> &$t {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn d() -> &mut $t {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn e() -> &'static $t {
                ret();
            }
        }
    };
}

gpu_results!(gpu_results_bounds, dyn core::fmt::Debug + Send);
gpu_results!(gpu_results_parens, (dyn core::fmt::Debug + Send));

/// Helpers that take a pointer or a reference to `$t`, in a module where a kernel does not compile, for Rust is
/// shown their types there.
macro_rules! helper_params {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn k(x: *const $t) {
                ret();
            }

            fn a(x: *const $t) {
                nop();
            }
            fn b(x: *mut $t) {
                nop();
            }
            fn c(x: &$t) {
                nop();
            }
            fn d(x: &mut $t) {
                nop();
            }
            fn e(x: &'static $t) {
                nop();
            }
        }
    };
}

helper_params!(helper_params_bounds, dyn core::fmt::Debug + Send);
helper_params!(helper_params_parens, (dyn core::fmt::Debug + Send));

/// Helpers that give back a pointer or a reference to `$t`, in a module where a kernel does not compile. A reference
/// with no lifetime of its own takes the one of the reference that is the parameter, as Rust has it.
macro_rules! helper_results {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn k(x: *const $t) {
                ret();
            }

            fn a(x: gpr) -> *const $t {
                nop();
            }
            fn b(x: gpr) -> *mut $t {
                nop();
            }
            fn c(x: gpr) -> &'static $t {
                nop();
            }
            fn d(x: gpr) -> &'static mut $t {
                nop();
            }
            fn e(x: &u8) -> &$t {
                nop();
            }
            fn f(x: &u8) -> &mut $t {
                nop();
            }
        }
    };
}

helper_results!(helper_results_bounds, dyn core::fmt::Debug + Send);
helper_results!(helper_results_parens, (dyn core::fmt::Debug + Send));

/// Kernels and a helper whose type holds `&$t` or `*const $t` in a tuple, an array, a function pointer and an argument
/// of a type, which Rust reads as one type too.
macro_rules! nested {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            use super::Wrap;

            pub unsafe extern "sysv64" fn tuple(x: (u8, &'static $t)) {
                ret();
            }
            pub unsafe extern "sysv64" fn array(x: u64) -> [*const $t; 2] {
                ret(x);
            }
            pub unsafe extern "ptx-kernel" fn function(x: fn(&mut $t) -> u8) {
                ret();
            }
            pub unsafe extern "sysv64" fn argument(x: Wrap<&'static $t>) {
                ret();
            }

            fn helper(x: [(u8, &'static $t); 2], y: fn(*mut $t) -> Wrap<&'static $t>) {
                nop();
            }
        }
    };
}

nested!(nested_bounds, dyn core::fmt::Debug + Send);
nested!(nested_parens, (dyn core::fmt::Debug + Send));
