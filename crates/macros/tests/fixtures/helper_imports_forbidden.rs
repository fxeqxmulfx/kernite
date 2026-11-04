//! Helpers and `extern "ptx-func"` functions in a crate that forbids lints, in modules where a kernel does
//! not compile: Rust sees their types, the language's and one that kernite refuses, and says nothing but
//! kernite's errors, so the Rust the macros write for it needs no `allow`.

#![forbid(
    dead_code,
    deprecated,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unreachable_patterns,
    unused,
    unused_imports,
    unused_macros,
    unused_parens
)]

pub struct Ret(pub u64);

#[kernite_macros::kernels]
pub mod cpu {
    use super::Ret;

    const N: usize = 4;

    fn refused<const M: u64>(x: Ret, y: [gpr; N], z: ymm) -> Ret {
        nop();
    }

    fn language(x: gpr, y: [r32; 2], z: local<[u32; 4], 16>) {
        nop();
    }

    pub unsafe extern "sysv64" fn broken(x: u64) -> u64 {
        ret(y);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    use super::Ret;

    pub unsafe extern "ptx-func" fn refused(x: Ret, y: u32) -> u32 {
        ret(y);
    }

    pub unsafe extern "ptx-func" fn language(x: u32, y: f32) -> u32 {
        ret(x);
    }

    fn help(x: Ret, y: shared<[f32; 256], 16>) {
        nop();
    }

    pub unsafe extern "ptx-kernel" fn broken(p: *mut u32) {
        st.global.u32([p], y);
    }
}
