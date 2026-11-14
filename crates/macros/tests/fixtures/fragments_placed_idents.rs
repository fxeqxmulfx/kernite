//! Rust says that a result with a `&` before `$t:ident` or `$t:tt` has no lifetime at the `&` of each kernel of a macro
//! that takes the same `$t`, wherever the macro is defined: below its call, as a `#[macro_export]` macro called by its
//! path is, and in a file of its own, with two kernels or with one. It is the same as for `$t:ty` in
//! `fragments_placed.rs`. So it is for a result the call writes whole, `&u8` as a `$r:ty`, with the macro in a file of
//! its own: the `&` and the ident are the call's, and the token after them is the macro's.

#[macro_use]
mod macros;

crate::below!(under, u8);

#[macro_export]
macro_rules! below {
    ($module:ident, $t:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> Option<&$t> {
                ret(x);
            }
        }
    };
}

files!(filed, u8);
alone!(single, u8);

crate::below_tree!(under_tree, u16);

#[macro_export]
macro_rules! below_tree {
    ($module:ident, $t:tt) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> Option<&$t> {
                ret(x);
            }
        }
    };
}

files_tree!(filed_tree, u16);
whole!(whole_ty, &u8);
