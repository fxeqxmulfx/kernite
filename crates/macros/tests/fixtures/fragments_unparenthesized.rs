//! A macro's fragment `$t:ty` is named in kernite's errors as it is written, in no parentheses, where it is not the
//! bounds of a trait object behind a `&`, a pointer or an arrow: when the macros open it, `*const [$t]` with `$t` as
//! `u8`; when they keep it as one type, `&$t` and `*const $t` with `$t` as `Wrap<u8>`, or as
//! `Wrap<dyn core::fmt::Debug + Send>`, whose `+` is in an argument; as bounds in an argument of a type, `Wrap<$b>`;
//! and after the arrow of a function pointer, `fn() -> $t`, with each of the three. `fragments_bounds.rs` has the
//! bounds behind a `&` and a pointer, which are named in parentheses, and `fragments_arrows.rs` those after an arrow.

/// A type of the crate's with an argument that is a type, of any size.
pub struct Wrap<T: ?Sized>(core::marker::PhantomData<T>);

/// Kernels whose types hold `$t`, in a slice, behind a pointer a GPU kernel returns, behind a reference and after the
/// arrow of a function pointer, and `$b`, bounds, in an argument of a type.
macro_rules! unparenthesized {
    ($module:ident, $t:ty, $b:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            use super::Wrap;

            pub unsafe extern "sysv64" fn slice(x: *const [$t]) {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn pointer() -> *const $t {
                ret();
            }
            pub unsafe extern "sysv64" fn reference(x: &$t) {
                ret();
            }
            pub unsafe extern "sysv64" fn argument(x: Wrap<$b>) {
                ret();
            }
            pub unsafe extern "sysv64" fn function(x: fn() -> $t) {
                ret();
            }
        }
    };
}

unparenthesized!(opened, u8, dyn core::fmt::Debug + Send);
unparenthesized!(kept, Wrap<u8>, dyn core::fmt::Debug + Send);
unparenthesized!(inner, Wrap<dyn core::fmt::Debug + Send>, dyn core::fmt::Debug + Send);
