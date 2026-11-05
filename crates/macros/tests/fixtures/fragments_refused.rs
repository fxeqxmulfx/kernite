//! An error at a fragment is said where the fragment is, in the call of the macro that has it: at the name that is
//! not defined, at the operator that has no meaning between its operands, at the type that Rust refuses, and, of the
//! fragment as a whole, where it starts. A fragment is one operand, so `1 - 1` under `10 / $n` is the division by zero
//! it is for Rust, and an array as long as `$n * 2` has no sixth index past its end.

pub struct Wrap<T>(core::marker::PhantomData<T>);

macro_rules! operand {
    ($module:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn scaled(p: *const u8) -> *const u8 {
                add(p, $n * 2);
                ret(p);
            }
        }
    };
}

operand!(unknown, x + 1);
operand!(float, 1.5 + 2);
operand!(string, "a" + 1);

macro_rules! divided {
    ($module:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn k(p: *const u8) -> *const u8 {
                add(p, 10 / $n);
                ret(p);
            }
        }
    };
}

divided!(by_zero, 1 - 1);

macro_rules! lengths {
    ($module:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn past_the_end(x: u64) -> u64 {
                let v: [gpr; $n * 2];
                mov(v[6], 1);
                ret(x);
            }
        }
    };
}

lengths!(six, 1 + 2);

macro_rules! small {
    ($module:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub const C: u8 = $n * 2;

            pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
                add(x, C);
                ret(x);
            }
        }
    };
}

small!(too_wide, 200 + 100);
small!(mismatched, 1u16 + 2);

macro_rules! typed {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            use super::Wrap;

            pub unsafe extern "sysv64" fn wrapped(x: *const Wrap<$t>) {
                ret();
            }
            pub unsafe extern "sysv64" fn taken(x: $t) {
                ret();
            }
        }
    };
}

typed!(unstable, *const f128);
typed!(refused, &f16);

macro_rules! nested {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            use super::Wrap;

            pub unsafe extern "sysv64" fn k(x: Wrap<$t>) {
                ret();
            }
        }
    };
}

nested!(arrays, Wrap<[u8; 3 * 2]>);
nested!(plain, Wrap<u8>);

macro_rules! given {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn k(x: u64) -> $t {
                ret(x);
            }
        }
    };
}

given!(unsized_result, *const core::ffi::CStr);

macro_rules! stray {
    ($module:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn k(p: *const u8) -> *const u8 {
                add(p, 1) $n * 2;
                ret(p);
            }
        }
    };
}

stray!(after_a_call, 1 + 2);
