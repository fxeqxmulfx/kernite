//! A fragment that is the user's own pair of parentheses and ends the type behind a `&` is linted as in a plain
//! function: Rust says that the pair is unnecessary, where the user wrote it, in `(u8)`, `(dyn Send)` and `([u8])`, and
//! says nothing of the pair that a `+` needs, in `(dyn Send + Sync)`. Two kernels of a macro take the same fragment,
//! and Rust says that the `&` of each has no lifetime, as it does of two plain functions of another macro.

macro_rules! kernels {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &$t {
                ret(x);
            }
        }
    };
}

kernels!(kernel_byte, (u8));
kernels!(kernel_object, (dyn Send));
kernels!(kernel_slice, ([u8]));
kernels!(kernel_needed, (dyn Send + Sync));

macro_rules! plain {
    ($module:ident, $t:ty) => {
        pub mod $module {
            pub fn first(_x: u64) -> &$t {
                todo!()
            }
            pub fn second(_x: u64) -> &$t {
                todo!()
            }
        }
    };
}

plain!(plain_byte, (u8));
plain!(plain_object, (dyn Send));
plain!(plain_slice, ([u8]));
plain!(plain_needed, (dyn Send + Sync));
