//! A fragment that is a token of the call, a `$t:ident` or a `$t:tt`, has no group round it. Rust says that a result
//! with a `&` before `$t` has no lifetime at the `&` of each kernel of a macro that takes it, as for kernels written
//! out, and not at the `$t` of the first. A result written out, or in the body of a macro, is as it was. So is a result
//! the call writes whole, `&u8` as a `$r:ty`, a `$($r:tt)*` or two `tt`s: the `&` and the ident are the call's, and the
//! token after them is the macro's. Each macro is above its call here; `fragments_placed_idents.rs` has them elsewhere.

macro_rules! idents {
    ($module:ident, $t:ident) => {
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

idents!(byte, u8);
idents!(half, f16);

macro_rules! trees {
    ($module:ident, $t:tt) => {
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

trees!(tree, u16);

// The `&` is not the first token of the result, tokens are between it and the fragment, or the fragment is the last
// token of a bracket.
macro_rules! shapes {
    ($module:ident, $t:ident) => {
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

macro_rules! traits {
    ($module:ident, $t:ident) => {
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

pub trait Tr {
    type Out;
}

// The type behind the `&` ends at the `as` of a qualified path.
macro_rules! qualified {
    ($module:ident, $t:ident) => {
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

macro_rules! gpu {
    ($module:ident, $t:ident) => {
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

// The ident is written in place: in kernels written out, with and without a space, and in the body of a macro.
#[kernite_macros::kernels]
pub mod written {
    pub unsafe extern "sysv64" fn first(x: u64) -> &u8 {
        ret(x);
    }
    pub unsafe extern "sysv64" fn second(x: u64) -> & u8 {
        ret(x);
    }
    pub unsafe extern "sysv64" fn third(x: u64) -> &mut u8 {
        ret(x);
    }
    pub unsafe extern "sysv64" fn fourth(x: u64) -> &dyn Send {
        ret(x);
    }
    pub unsafe extern "sysv64" fn fifth(x: u64) -> (u8, &u8) {
        ret(x);
    }
}

macro_rules! owned {
    ($module:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &u8 {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &u8 {
                ret(x);
            }
            pub unsafe extern "sysv64" fn third(x: u64) -> (u8, &u8) {
                ret(x);
            }
        }
    };
}

owned!(own);

// The call writes the whole result, `&u8`: the `&` and the ident are its, and the `{` after them is the macro's. A
// `$r:ty` with two kernels and with one, a `$($r:tt)*`, and two `tt`s, where Rust says the error of the second kernel
// at its own `$a`.
macro_rules! whole {
    ($module:ident, $r:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> $r {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> $r {
                ret(x);
            }
        }
    };
}

whole!(whole_ty, &u8);

macro_rules! single {
    ($module:ident, $r:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn only(x: u64) -> $r {
                ret(x);
            }
        }
    };
}

single!(singled, &u8);

macro_rules! sequence {
    ($module:ident, $($r:tt)*) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> $($r)* {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> $($r)* {
                ret(x);
            }
        }
    };
}

sequence!(sequenced, &u8);

macro_rules! pair {
    ($module:ident, $a:tt $t:tt) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> $a $t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> $a $t {
                ret(x);
            }
        }
    };
}

pair!(paired, &u8);
