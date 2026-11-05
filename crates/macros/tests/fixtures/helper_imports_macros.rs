//! Helpers that no kernel calls, in modules where the macros find an error in writing the module, and not
//! kernite in compiling an item: a compile-time argument named as a parameter of a launcher, of a cooperative
//! x86 kernel's or a GPU kernel's, or of its `_blocks` method; a `_blocks` method that two items have; statics
//! with no room beside each other; and a name that `cfg` leaves out where a kernel that names it is kept. One
//! helper takes `Ret`, a type of the crate's that the module imports, and the other the language's types,
//! `gpr`, `ymm`, `local<[u32; 4], 16>` and `r32`. Rust is shown the helpers' types, so it does not call the
//! import unused, and says nothing of a name it cannot find.

pub struct Ret(pub u64);

#[kernite_macros::kernels]
pub mod cpu_grid {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    #[instances(1, 2)]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn k<const grid: u64>(x: *mut u64) {
        mov(qword[x], 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu_grid {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    #[instances(1)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const grid: u32>(xs: *mut u32) {
        st.global.u32([xs], grid);
    }
}

#[kernite_macros::kernels]
pub mod gpu_blocks {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    #[cooperative]
    #[instances((1, 2))]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const threads: u32, const shared: u32>(xs: *mut u32) {
        st.global.u32([xs], threads);
    }
}

#[kernite_macros::kernels]
pub mod kernel_twice {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn k() {}

    pub unsafe extern "ptx-kernel" fn k_blocks() {}
}

#[kernite_macros::kernels]
pub mod static_twice {
    use super::Ret;

    #[allow(non_upper_case_globals)]
    static k_blocks: global<u32>;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod no_room {
    use super::Ret;

    pub static A: constant<[u32; 12000]> = [0; 12000];
    pub static B: constant<[u32; 12000]> = [0; 12000];

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    pub unsafe extern "ptx-kernel" fn nothing() {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod left_out {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    #[cfg(any())]
    fn g() {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: *mut u32) {
        g();
        mov(dword[x], 1);
    }
}
