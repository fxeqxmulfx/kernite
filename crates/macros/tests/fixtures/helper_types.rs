//! The types that the signature of a helper or of an `extern "ptx-func"` function names, in modules where a kernel
//! does not compile, for Rust is shown them there: types the language has, `gpr` or `local<[u32; 4], 16>`, and types
//! of the crate's that kernite refuses, `Ret`. Rust says of them only that a lifetime is not declared, that a module
//! is where a type is written, and what a macro writes, and it says so of a helper that no kernel calls too.

pub struct Ret(pub u64);
pub trait Tr {}

macro_rules! half {
    () => {
        f16
    };
}

#[kernite_macros::kernels]
pub mod x86 {
    use super::{Ret, Tr};

    const N: usize = 4;

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        ret(y);
    }

    fn language(a: gpr, b: ymm, c: [gpr; N], d: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    fn quiet(a: Ret, b: &mut Ret, c: *const Ret, d: [Ret; 2], e: Option<Ret>, f: Box<dyn Tr>, g: fn(Ret) -> Ret) {
        nop();
    }

    fn unsized_(a: [u64], b: str, c: dyn Tr, d: *const [u8], e: Self, f: !, g: _) {
        nop();
    }

    fn floats(a: f16, b: f128, c: [f16; 4]) -> f16 {
        nop();
    }

    fn lifetime(a: &'a Ret) {
        nop();
    }

    fn module(a: core::f16) {
        nop();
    }

    fn krate(a: core) {
        nop();
    }

    fn written(a: half!()) {
        nop();
    }

    fn result_lifetime(a: gpr) -> &'a Ret {
        nop();
    }

    fn result_module(a: gpr) -> core::f16 {
        nop();
    }

    fn generic<const M: Ret>(a: gpr) {
        nop();
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    use super::{Ret, Tr};

    pub unsafe extern "ptx-kernel" fn k(p: *mut u32) {
        st.global.u32([p], y);
    }

    fn language(a: u32, b: f32, c: local<[u32; 4], 16>, d: shared<[f32; 256], 16>) {
        nop();
    }

    fn quiet(a: Ret, b: &Ret, c: *const Ret, d: [Ret; 2], e: f16, f: f128, g: [u64], h: dyn Tr, i: _, j: !) {
        nop();
    }

    fn lifetime(a: &'a Ret) {
        nop();
    }

    fn module(a: core::f16) {
        nop();
    }

    fn written(a: half!()) {
        nop();
    }

    pub unsafe extern "ptx-func" fn func_quiet(a: Ret, b: &Ret, c: [Ret; 2], d: f128) -> Ret {
        ret(a);
    }

    pub unsafe extern "ptx-func" fn func_lifetime(a: &'a Ret) -> u32 {
        ret(a);
    }

    pub unsafe extern "ptx-func" fn func_module(a: u32) -> core::f16 {
        ret(a);
    }

    pub unsafe extern "ptx-func" fn func_written(a: half!()) -> u32 {
        ret(a);
    }

    #[instances(1)]
    pub unsafe extern "ptx-func" fn func_generic<const M: Ret>(a: u32) -> u32 {
        ret(a);
    }
}
