//! The last segment of a path that a macro is given, `&$a::$b`, is a token of the call, and no parentheses can go round
//! a segment. Rust says that a result with a `&` before the path has no lifetime at the `&` of each kernel of a macro
//! that takes the same `$b`, as for a path written out, and not at the `$b` of the first. It is so whether the macro is
//! given the first segment, `&$a::$b`, or writes it, `&crate::$b`; behind `mut`; for a `$t:tt`; for a path of three
//! segments; for a raw ident, `r#Local`; where the `&` is in an `Option`, a tuple, behind a `dyn` or before a `Vec`;
//! and where the macro is below its call or in a file of its own. A path written out, in a kernel or in the body of a
//! macro, is as it was. A GPU kernel's `&` takes the lifetime of `self`, so Rust has none to ask for there.

#[macro_use]
mod macros;

pub struct Local;

pub trait Tr {}

pub mod inner {
    pub struct Deep;
}

macro_rules! segments {
    ($module:ident, $a:ident, $b:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$a::$b {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &$a::$b {
                ret(x);
            }
            pub unsafe extern "sysv64" fn third(x: u64) -> &crate::$b {
                ret(x);
            }
            pub unsafe extern "sysv64" fn fourth(x: u64) -> &mut crate::$b {
                ret(x);
            }
            pub unsafe extern "sysv64" fn fifth(x: u64) -> &crate::Local {
                ret(x);
            }
        }
    };
}

segments!(segmented, crate, Local);

macro_rules! trees {
    ($module:ident, $a:tt, $b:tt) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$a::$b {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &$a::$b {
                ret(x);
            }
        }
    };
}

trees!(treed, crate, Local);

macro_rules! deep {
    ($module:ident, $a:ident, $m:ident, $b:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$a::$m::$b {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &$a::$m::$b {
                ret(x);
            }
        }
    };
}

deep!(deeper, crate, inner, Deep);

// The `&` is not the first token of the result, or the path is not the last tokens of it.
macro_rules! shapes {
    ($module:ident, $a:ident, $b:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn held(x: u64) -> Option<&$a::$b> {
                ret(x);
            }
            pub unsafe extern "sysv64" fn paired(x: u64) -> (u8, &$a::$b) {
                ret(x);
            }
            pub unsafe extern "sysv64" fn listed(x: u64) -> (&$a::$b, u8) {
                ret(x);
            }
            pub unsafe extern "sysv64" fn traited(x: u64) -> &dyn $a::$b {
                ret(x);
            }
            pub unsafe extern "sysv64" fn wrapped(x: u64) -> &Vec<$a::$b> {
                ret(x);
            }
        }
    };
}

shapes!(shaped, crate, Tr);

macro_rules! gpu {
    ($module:ident, $a:ident, $b:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "ptx-kernel" fn first(x: u64) -> &$a::$b {
                ret();
            }
            pub unsafe extern "ptx-kernel" fn second(x: u64) -> &$a::$b {
                ret();
            }
        }
    };
}

gpu!(device, crate, Local);

crate::below!(under, crate, Local);

#[macro_export]
macro_rules! below {
    ($module:ident, $a:ident, $b:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$a::$b {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &$a::$b {
                ret(x);
            }
        }
    };
}

files!(filed, crate, Local);

// The path is written in place: in a kernel, with and without a space, and in the body of a macro.
#[kernite_macros::kernels]
pub mod written {
    pub unsafe extern "sysv64" fn first(x: u64) -> &crate::Local {
        ret(x);
    }
    pub unsafe extern "sysv64" fn second(x: u64) -> & crate::Local {
        ret(x);
    }
    pub unsafe extern "sysv64" fn third(x: u64) -> &mut crate::inner::Deep {
        ret(x);
    }
    pub unsafe extern "sysv64" fn fourth(x: u64) -> &dyn crate::Tr {
        ret(x);
    }
}

macro_rules! owned {
    ($module:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &crate::Local {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &crate::inner::Deep {
                ret(x);
            }
        }
    };
}

owned!(own);

// The last segment is a raw ident.
macro_rules! raws {
    ($module:ident, $a:ident, $b:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$a::$b {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &$a::$b {
                ret(x);
            }
        }
    };
}

raws!(raw, crate, r#Local);
