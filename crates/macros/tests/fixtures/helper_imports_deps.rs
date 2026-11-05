//! Helpers that no kernel calls, in a module where the crate lacks a dependency the Rust the macros write needs:
//! a static of GPU kernels needs `kernite-macros` and `kernite-cuda`; a static of x86 kernels and a cooperative x86
//! kernel need `kernite-cpu`; and GPU kernels need `kernite-cuda`. One helper takes `Ret`, a type of the crate's
//! that the module imports, and the other the language's types, `gpr`, `ymm`, `local<[u32; 4], 16>` and `r32`.
//! Rust is shown the helpers' types, so it does not call the import unused, and says nothing of a name it cannot
//! find.

pub struct Ret(pub u64);

#[km::kernels]
pub mod gpu_static {
    use super::Ret;

    pub static TABLE: constant<[u32; 4]>;

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

#[km::kernels]
pub mod x86_static {
    use super::Ret;

    pub static COUNTS: global<[u32; 4]>;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: *mut u32) {
        mov(dword[x], 1);
    }
}

#[km::kernels]
pub mod x86_cooperative {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32) {
        grid.sync();
        mov(dword[x], 1);
    }
}

#[km::kernels]
pub mod gpu_plain {
    use super::Ret;

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
