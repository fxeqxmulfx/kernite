//! A body that the call of a macro gives, a `$b:block`, after a result with a `&`: the kernel's body, which ends the
//! type behind the `&`, and no part of it. Each macro is above its call and has two kernels whose bodies the call
//! gives, with the result `&$t` for a `$t:ident`, a `$t:ty`, a `$t:tt` and a `$t:path`, `&u8` written in the macro,
//! `&mut $t` and `&Vec<$t>`; and a kernel with no result after one with a `&`, and two GPU kernels. kernite refuses
//! each result at its `&`, and Rust says that the `&` of each x86 kernel has no lifetime there, as it says it of the
//! same kernels written out.

macro_rules! idents {
    ($module:ident, $t:ident, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t $b
            pub unsafe extern "sysv64" fn second(x: u64) -> &$t $c
        }
    };
}

idents!(on_idents, u8, { ret(x); }, { ret(x); });

macro_rules! types {
    ($module:ident, $t:ty, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t $b
            pub unsafe extern "sysv64" fn second(x: u64) -> &$t $c
        }
    };
}

types!(on_types, u8, { ret(x); }, { ret(x); });

macro_rules! written {
    ($module:ident, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &u8 $b
            pub unsafe extern "sysv64" fn second(x: u64) -> &u8 $c
        }
    };
}

written!(on_written, { ret(x); }, { ret(x); });

macro_rules! shapes {
    ($module:ident, $t:ident, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &mut $t $b
            pub unsafe extern "sysv64" fn second(x: u64) -> &Vec<$t> $c
        }
    };
}

shapes!(on_shapes, u8, { ret(x); }, { ret(x); });

macro_rules! results {
    ($module:ident, $t:ident, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t $b
            pub unsafe extern "sysv64" fn second(x: u64) $c
        }
    };
}

results!(on_results, u8, { ret(x); }, { ret(); });

macro_rules! gpus {
    ($module:ident, $t:ident, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "ptx-kernel" fn first(x: *mut u32) -> &$t $b
            pub unsafe extern "ptx-kernel" fn second(x: *mut u32) -> &$t $c
        }
    };
}

gpus!(on_gpus, u8, { st.global.u32([x], 1); }, { st.global.u32([x], 2); });

macro_rules! tts {
    ($module:ident, $t:tt, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t $b
            pub unsafe extern "sysv64" fn second(x: u64) -> &$t $c
        }
    };
}

tts!(on_tts, u8, { ret(x); }, { ret(x); });

macro_rules! paths {
    ($module:ident, $t:path, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t $b
            pub unsafe extern "sysv64" fn second(x: u64) -> &$t $c
        }
    };
}

paths!(on_paths, u8, { ret(x); }, { ret(x); });
