//! Kernels with numbers, in every kind of kernel the macros write a check for, built without a word
//! under `#![deny(warnings)]`: a result of each kind of number, and a parameter of each; parameters
//! named as the locals the macros write for them; a kernel that never returns; kernels deprecated,
//! by `#[deprecated]` and by `cfg_attr`, and kernels configured out, by `#[cfg]` and by `cfg_attr`.
//! Each is plain, of compile-time arguments and cooperative, on x86, and a GPU kernel too.

#![deny(warnings)]

/// The x86 kernels.
#[kernite_macros::kernels]
pub mod cpu {
    /// A parameter of each integer, and a result.
    pub unsafe extern "sysv64" fn ints(a: u64, b: i64, c: usize, d: isize) -> u64 {
        add(a, b);
        add(a, c);
        add(a, d);
        ret(a);
    }

    pub unsafe extern "sysv64" fn singles(a: f32, b: f32) -> f32 {
        vaddss(a, a, b);
        ret(a);
    }

    pub unsafe extern "sysv64" fn doubles(a: f64, b: f64) -> f64 {
        vaddsd(a, a, b);
        ret(a);
    }

    /// A result of each number narrower than a register, and `bool`.
    pub unsafe extern "sysv64" fn narrow_u8(a: u64) -> u8 {
        ret(a);
    }

    pub unsafe extern "sysv64" fn narrow_i16(a: u64) -> i16 {
        ret(a);
    }

    pub unsafe extern "sysv64" fn narrow_u32(a: u64) -> u32 {
        ret(a);
    }

    pub unsafe extern "sysv64" fn flag(a: u64) -> bool {
        ret(a);
    }

    /// Parameters named as the locals the macros write: the probe's of a plain kernel, `__0`, `__1`
    /// and `__n`, and the parameter of the closure that asks of a type, `__kernite_asked`.
    pub unsafe extern "sysv64" fn locals(__0: u64, __1: u64, __n: u64, __kernite_asked: u64) -> u64 {
        add(__0, __1);
        add(__0, __n);
        add(__0, __kernite_asked);
        ret(__0);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn locals_tiles<const N: u64>(__0: u64, __n: f64, __kernite_asked: u64) -> u64 {
        add(__0, N);
        add(__0, __kernite_asked);
        ret(__0);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn float_tiles<const N: u64>(x: f64) -> f64 {
        vaddsd(x, x, x);
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn flag_tiles<const N: u64>(x: u64) -> bool {
        add(x, N);
        ret(x);
    }

    pub unsafe extern "sysv64" fn stop(a: u64) -> ! {
        ud2();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(a: u64, b: f64) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(__0: f32, a: u64) {
        ret();
    }

    #[deprecated]
    pub unsafe extern "sysv64" fn old(a: u64, b: f64) -> f64 {
        ret(b);
    }

    #[cfg_attr(all(), deprecated)]
    pub unsafe extern "sysv64" fn older(a: u64) -> u64 {
        ret(a);
    }

    #[deprecated]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn old_tiles<const N: u64>(a: u64) -> u8 {
        add(a, N);
        ret(a);
    }

    #[cfg_attr(all(), deprecated)]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn older_tiles<const N: u64>(a: f32) -> f32 {
        ret(a);
    }

    #[deprecated]
    #[cooperative]
    pub unsafe extern "sysv64" fn old_together(a: u64, b: f32) {
        ret();
    }

    #[cfg_attr(all(), deprecated)]
    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn older_together<const N: u64>(a: f64) {
        ret();
    }

    #[cfg(any())]
    pub unsafe extern "sysv64" fn gone(a: u64, b: f64) -> f64 {
        ret(b);
    }

    #[cfg_attr(all(), cfg(any()))]
    pub unsafe extern "sysv64" fn gone_too(a: u64) -> u64 {
        ret(a);
    }

    #[cfg(any())]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn gone_tiles<const N: u64>(a: u64) -> bool {
        ret(a);
    }

    #[cfg_attr(all(), cfg(any()))]
    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn gone_too_together<const N: u64>(a: f32) {
        ret();
    }
}

/// The GPU kernels.
#[kernite_macros::kernels]
pub mod gpu {
    /// A parameter of each number a GPU kernel takes.
    pub unsafe extern "ptx-kernel" fn numbers(
        a: u8,
        b: u16,
        c: u32,
        d: u64,
        e: i8,
        f: i16,
        g: i32,
        h: i64,
        i: usize,
        j: isize,
        k: f32,
        l: f64,
    ) {
        let x: u32;
        mov.u32(x, c);
    }

    /// Parameters named as the locals the macros write.
    pub unsafe extern "ptx-kernel" fn locals(__0: u32, __1: f32, __n: u64, __kernite_asked: u8) {
        let x: u32;
        mov.u32(x, __0);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(a: u8, b: f64) {
        let x: u32;
        mov.u32(x, N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(a: u32, b: f32) {
        let x: u32;
        mov.u32(x, a);
    }

    #[deprecated]
    pub unsafe extern "ptx-kernel" fn old(a: u32, b: f32) {
        let x: u32;
        mov.u32(x, a);
    }

    #[cfg_attr(all(), deprecated)]
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn older_together(a: u32) {
        let x: u32;
        mov.u32(x, a);
    }

    #[cfg(any())]
    pub unsafe extern "ptx-kernel" fn gone(a: u32, b: f32) {
        let x: u32;
        mov.u32(x, a);
    }

    #[cfg_attr(all(), cfg(any()))]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn gone_too_tiles<const N: u32>(a: u16) {
        let x: u32;
        mov.u32(x, N);
    }
}
