//! A cooperative kernel under `a`, x86 in `clash` and the GPU's in `clash_gpu`, whose compile-time arguments are named
//! as parameters of its launcher, `grid`, and of its `_blocks` method, `threads`: errors the macros find in writing the
//! module. The helpers are under `b`: `h` takes `Ret`, a type of the crate's that each module imports, and `language`
//! the language's types. Rust is shown their types where `cfg` keeps them and the error is said, so it does not call
//! the import unused there, and says nothing of the types it cannot find; and not otherwise.

pub struct Ret(pub u64);

#[kernite_macros::kernels]
pub mod clash {
    use super::Ret;

    #[cfg(feature = "b")]
    fn h(x: Ret) {
        nop();
    }

    #[cfg(feature = "b")]
    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    #[cfg(feature = "a")]
    #[cooperative]
    #[instances(1)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn k<const grid: u64>(x: *mut u64) {
        mov(qword[x], grid);
    }

    pub unsafe extern "sysv64" fn plain(x: *mut u64) {
        mov(qword[x], 1);
    }
}

#[kernite_macros::kernels]
pub mod clash_gpu {
    use super::Ret;

    #[cfg(feature = "b")]
    fn h(x: Ret) {
        nop();
    }

    #[cfg(feature = "b")]
    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    #[cfg(feature = "a")]
    #[cooperative]
    #[instances((1, 2))]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn g<const grid: u32, const threads: u32>(xs: *mut u32) {
        st.global.u32([xs], grid);
    }

    pub unsafe extern "ptx-kernel" fn plain_gpu(xs: *mut u32) {
        ret();
    }
}
