//! A fragment that kernels of one macro take together is one fragment in each. Rust says that a result with a `&`
//! before `$t` has no lifetime at the `&` of each kernel, as for kernels written out, and an error at the fragment
//! once, in the call of the macro. Each macro is above its call here; `fragments_placed.rs` has them elsewhere.

macro_rules! types {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn number(x: u64) -> u64 {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &$t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn third(x: u64) -> &$t {
                ret(x);
            }
        }
    };
}

types!(byte, u8);
types!(bytes, [u8]);
types!(object, dyn Send + Sync);
types!(half, f16);

macro_rules! paths {
    ($module:ident, $t:path) => {
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

paths!(pathed, u8);

// The `&` is not the first token of the result, tokens are between it and the fragment, or the fragment is not the last
// token of the type the `&` is before.
macro_rules! shapes {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn mutable(x: u64) -> &mut $t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn twice(x: u64) -> &&$t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn held(x: u64) -> Option<&$t> {
                ret(x);
            }
            pub unsafe extern "sysv64" fn paired(x: u64) -> (u8, &$t) {
                ret(x);
            }
            pub unsafe extern "sysv64" fn listed(x: u64) -> (&$t, u8) {
                ret(x);
            }
            pub unsafe extern "sysv64" fn arrayed(x: u64) -> [&$t; 3] {
                ret(x);
            }
            pub unsafe extern "sysv64" fn called(x: u64) -> &fn() -> $t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn wrapped(x: u64) -> &Vec<$t> {
                ret(x);
            }
        }
    };
}

shapes!(shaped, u8);

// The last token of the type is the macro's, not the fragment's.
macro_rules! tails {
    ($module:ident, $t:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t<u8> {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &$t<u8> {
                ret(x);
            }
        }
    };
}

tails!(tailed, Vec);

macro_rules! traits {
    ($module:ident, $t:path) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &dyn $t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &dyn $t {
                ret(x);
            }
        }
    };
}

traits!(traited, Send);

macro_rules! two_fragments {
    ($module:ident, $t:ty, $u:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &$u {
                ret(x);
            }
        }
    };
}

two_fragments!(two, u8, u16);

macro_rules! pointers {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> *const $t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> *const $t {
                ret(x);
            }
        }
    };
}

pointers!(sizeless, core::ffi::CStr);

macro_rules! gpu {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "ptx-kernel" fn first(x: u64) -> &$t {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn second(x: u64) -> &$t {
                ret();
            }
        }
    };
}

gpu!(device, u8);

pub trait Tr {
    type Out;
}

// The type behind the `&` ends at the `as` of a qualified path.
macro_rules! qualified {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> <&$t as crate::Tr>::Out {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> <&$t as crate::Tr>::Out {
                ret(x);
            }
        }
    };
}

qualified!(qualifieds, u8);
