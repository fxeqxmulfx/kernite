//! Rust says that a result with a `&` before `$t` has no lifetime at the `&` of each kernel of a macro that takes the
//! same `$t`, wherever the macro is defined: below its call, as a `#[macro_export]` macro called by its path is, and in
//! a file of its own, with two kernels or with one.

#[macro_use]
mod macros;

crate::below!(under, u8);

#[macro_export]
macro_rules! below {
    ($module:ident, $t:ty) => {
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
