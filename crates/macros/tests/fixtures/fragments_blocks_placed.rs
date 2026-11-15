//! Rust says that a result with a `&` before a body that the call gives has no lifetime at the `&` of each kernel of a
//! macro, wherever the macro is defined: below its call, as a `#[macro_export]` macro called by its path is, and in a
//! file of its own. It is so for `&u8` written in the macro and for a `$t:ident`: the body has the place of the call,
//! whoever wrote the type, so it tells nothing of where the ident is.

#[macro_use]
mod macros;

crate::below!(under, { ret(x); }, { ret(x); });
crate::below_idents!(under_idents, u8, { ret(x); }, { ret(x); });

#[macro_export]
macro_rules! below {
    ($module:ident, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &u8 $b
            pub unsafe extern "sysv64" fn second(x: u64) -> &u8 $c
        }
    };
}

#[macro_export]
macro_rules! below_idents {
    ($module:ident, $t:ident, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t $b
            pub unsafe extern "sysv64" fn second(x: u64) -> &$t $c
        }
    };
}

files!(filed, { ret(x); }, { ret(x); });
files_idents!(filed_idents, u8, { ret(x); }, { ret(x); });
