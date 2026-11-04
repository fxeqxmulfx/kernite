//! Helpers whose signatures name `Ret`, a type of the crate's that their module imports, in a module where kernite
//! says an error of the module as a whole, and of none of its items: that it is two modules, that no module of PTX
//! has its GPU's kernels, that an item of its own has the name of a const it takes, or that its attribute is
//! refused. No kernel calls the helpers, so kernite says nothing of their types; it shows them to Rust, so Rust does
//! not call the import unused beside the error, and says nothing of `gpr` or `local<[u32; 4], 16>`.

pub struct Ret(pub u64);

#[kernite_macros::consts]
pub mod sizes {
    pub const STEP: u64 = 3;
}

#[kernite_macros::kernels]
pub mod two_modules {
    use super::Ret;

    static COUNTS: global<[u32; 4]>;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn g(p: *mut u32) {
        let v: u32;
        ld.global.u32(v, [COUNTS]);
        st.global.u32([p], v);
    }
}

#[kernite_macros::kernels]
pub mod no_module_of_ptx_has_both {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    pub unsafe extern "ptx-kernel" fn old(n: u32) {
        let i: u32;
        shfl.idx.b32(i, n, 0, 31);
    }

    pub unsafe extern "ptx-kernel" fn new(f: f32) {
        let v: f32;
        tanh.approx.f32(v, f);
    }
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod a_const_given_and_its_own {
    use super::Ret;

    pub const STEP: u64 = 4;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }
}

#[kernite_macros::kernels(consts = 5)]
pub mod refused_at_its_attribute {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }
}
