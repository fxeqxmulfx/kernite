//! A macro's fragment `$t:ty` that holds bounds, `dyn Debug + Send`, is one type as the result of a function pointer or
//! of a trait object's `Fn`, as Rust reads it, wherever the macros write it back: in a parameter and in a result, of an
//! x86 kernel, of a GPU kernel and of a helper, and in a type that holds the function pointer: a tuple, an array or the
//! parameter of another function pointer. kernite refuses each, for a kernel takes pointers, integers and floats. Rust
//! says nothing of the fragment, as it says nothing of it in a macro that does not call kernite: not that its `+` is
//! ambiguous, as it says of `fn() -> dyn Debug + Send` written out. Each is built again with the bounds in parentheses,
//! `(dyn Debug + Send)`, as written by hand: kernite says the same of it, and Rust says that the parentheses are
//! unnecessary, as it does of a macro that does not call kernite. A kernel that gives back a pointer to a type that has
//! the bounds in an argument, `*mut Wrap<dyn Debug + Send>`, builds: that `+` is no bounds of the pointer's own, and
//! the macros open the group after the arrow of its result, as they did.

/// x86 kernels that take a function pointer that gives back `$t`, a pointer to a trait object's `Fn` that does, and a
/// box of one, where a `>` follows the arrow.
macro_rules! x86_params {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn a(x: fn() -> $t) {
                ret();
            }
            pub unsafe extern "sysv64" fn b(x: *const (dyn Fn() -> $t)) {
                ret();
            }
            pub unsafe extern "sysv64" fn c(x: Box<dyn Fn() -> $t>) {
                ret();
            }
        }
    };
}

x86_params!(x86_params_bounds, dyn core::fmt::Debug + Send);
x86_params!(x86_params_parens, (dyn core::fmt::Debug + Send));

/// An x86 kernel that gives back a function pointer that gives back `$t`.
macro_rules! x86_results {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn a(x: u64) -> fn() -> $t {
                ret(x);
            }
        }
    };
}

x86_results!(x86_results_bounds, dyn core::fmt::Debug + Send);
x86_results!(x86_results_parens, (dyn core::fmt::Debug + Send));

/// GPU kernels that take and give back a function pointer that gives back `$t`.
macro_rules! gpu {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "ptx-kernel" fn a(x: fn() -> $t) {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn b() -> fn() -> $t {
                ret();
            }
        }
    };
}

gpu!(gpu_bounds, dyn core::fmt::Debug + Send);
gpu!(gpu_parens, (dyn core::fmt::Debug + Send));

/// Helpers that take and give back a function pointer that gives back `$t`, in a module where a kernel does not
/// compile, for Rust is shown their types there.
macro_rules! helpers {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn k(x: fn() -> $t) {
                ret();
            }

            fn a(x: fn() -> $t) {
                nop();
            }
            fn b(x: gpr) -> fn() -> $t {
                nop();
            }
        }
    };
}

helpers!(helpers_bounds, dyn core::fmt::Debug + Send);
helpers!(helpers_parens, (dyn core::fmt::Debug + Send));

/// Kernels and a helper whose type holds the function pointer in a tuple, an array and the parameter of another
/// function pointer, which Rust reads as one type too.
macro_rules! nested {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn tuple(x: (u8, fn() -> $t)) {
                ret();
            }
            pub unsafe extern "sysv64" fn array(x: u64) -> [fn() -> $t; 2] {
                ret(x);
            }
            pub unsafe extern "ptx-kernel" fn function(x: fn(fn() -> $t) -> u8) {
                ret();
            }

            fn helper(x: [(u8, fn() -> $t); 2], y: fn(fn() -> $t) -> u8) {
                nop();
            }
        }
    };
}

nested!(nested_bounds, dyn core::fmt::Debug + Send);
nested!(nested_parens, (dyn core::fmt::Debug + Send));

/// A type of the crate's with an argument that is a type, of any size.
pub struct Wrap<T: ?Sized>(core::marker::PhantomData<T>);

/// An x86 kernel that takes and gives back `$t`, a pointer to a type that has the bounds in an argument. That `+` is no
/// bounds of the pointer's own, so the macros open the group after the arrow of the kernel's own result, for kernite to
/// read it: the kernel builds, and Rust says nothing of it.
macro_rules! own_result {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            use super::Wrap;

            pub unsafe extern "sysv64" fn a(x: $t) -> $t {
                ret(x);
            }
        }
    };
}

own_result!(own_result_wrap, *mut Wrap<dyn core::fmt::Debug + Send>);
