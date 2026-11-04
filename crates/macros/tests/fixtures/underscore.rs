//! Parameters named `_`, in a kernel of every kind: values given and not read, as Rust has it. Each kernel
//! takes a number named `_`, a number it reads, a pointer named `_`, and the pointer it writes through and
//! the number it reads after them, so that a kernel that read from the wrong place would give a wrong value.
//! Rust builds the crate without a word, under `#![deny(warnings)]`, and the tests run each kernel: the x86
//! kernels, plain, of compile-time arguments, cooperative, and cooperative of arguments, and the GPU kernels
//! of the same kinds, on the GPU if there is one. Some parameters and compile-time arguments are named as
//! the macros name a parameter named `_`, `__kernite_0` among them, which the macros then name otherwise.

#![deny(warnings)]
#![allow(non_upper_case_globals)]

/// x86 kernels: each gives `a - b` back, or writes it through `p`; those of arguments add `N` as well.
#[kernite_macros::kernels]
pub mod cpu {
    /// `a - b`.
    pub unsafe extern "sysv64" fn plain(_: u64, a: u64, _: *const u64, b: u64) -> u64 {
        sub(a, b);
        ret(a);
    }

    /// `a - __kernite_0`, with a parameter named as the macros name the first parameter named `_`.
    pub unsafe extern "sysv64" fn named_plain(__kernite_0: u64, _: u64, a: u64) -> u64 {
        sub(a, __kernite_0);
        ret(a);
    }

    /// `a - b + N`.
    #[instances(10, 20)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(_: u64, a: u64, _: *const u64, b: u64) -> u64 {
        sub(a, b);
        add(a, N);
        ret(a);
    }

    /// Every thread writes `a - b` through `p`.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(_: u64, a: u64, _: *mut u64, p: *mut u64, b: u64) {
        sub(a, b);
        mov(qword[p], a);
    }

    /// Every thread writes `a - b + N` through `p`.
    #[cooperative]
    #[instances(10, 20)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(_: u64, a: u64, _: *mut u64, p: *mut u64, b: u64) {
        sub(a, b);
        add(a, N);
        mov(qword[p], a);
    }

    /// `a - b + N`, with parameters named as the macros name the ones they name for `_`.
    #[instances(10, 20)]
    pub unsafe extern "sysv64" fn named_tiles<const N: u64>(
        __kernite_1: u64,
        _: u64,
        a: u64,
        __kernite_0: u64,
        _: *const u64,
        b: u64,
    ) -> u64 {
        sub(a, b);
        add(a, N);
        add(a, __kernite_0);
        add(a, __kernite_1);
        ret(a);
    }

    /// Every thread writes `a - b` through `p`, with parameters named as above.
    #[cooperative]
    pub unsafe extern "sysv64" fn named_together(__kernite_0: u64, _: u64, a: u64, _: *mut u64, p: *mut u64) {
        sub(a, __kernite_0);
        mov(qword[p], a);
    }

    /// `a + K`, with a compile-time argument named as the macros name the first parameter named `_`.
    #[instances(10, 20)]
    pub unsafe extern "sysv64" fn counted<const __kernite_0: u64>(_: u64, a: u64) -> u64 {
        add(a, __kernite_0);
        ret(a);
    }

    /// Every thread writes `a + K` through `p`, with a compile-time argument named as above.
    #[cooperative]
    #[instances(10, 20)]
    pub unsafe extern "sysv64" fn counted_together<const __kernite_0: u64>(_: u64, a: u64, p: *mut u64) {
        add(a, __kernite_0);
        mov(qword[p], a);
    }
}

/// GPU kernels: each writes `a - b` through `p`; those of arguments add `N` as well.
#[kernite_macros::kernels]
pub mod gpu {
    /// `a - b`.
    pub unsafe extern "ptx-kernel" fn plain(_: u32, a: u32, _: *mut u32, p: *mut u32, b: u32) {
        let x: u32;
        sub.u32(x, a, b);
        st.global.u32([p], x);
    }

    /// `a - b + N`.
    #[instances(10, 20)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(_: u32, a: u32, _: *mut u32, p: *mut u32, b: u32) {
        let x: u32;
        sub.u32(x, a, b);
        add.u32(x, x, N);
        st.global.u32([p], x);
    }

    /// `a - b`, by every block.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(_: u32, a: u32, _: *mut u32, p: *mut u32, b: u32) {
        let x: u32;
        sub.u32(x, a, b);
        st.global.u32([p], x);
    }

    /// `a - b + N`, by every block.
    #[cooperative]
    #[instances(10, 20)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(_: u32, a: u32, _: *mut u32, p: *mut u32, b: u32) {
        let x: u32;
        sub.u32(x, a, b);
        add.u32(x, x, N);
        st.global.u32([p], x);
    }

    /// `a - b + N`, with parameters named as the macros name the ones they name for `_`.
    #[instances(10, 20)]
    pub unsafe extern "ptx-kernel" fn named_tiles<const N: u32>(
        __kernite_1: u32,
        _: u32,
        a: u32,
        __kernite_0: u32,
        _: *mut u32,
        p: *mut u32,
        b: u32,
    ) {
        let x: u32;
        sub.u32(x, a, b);
        add.u32(x, x, N);
        add.u32(x, x, __kernite_0);
        add.u32(x, x, __kernite_1);
        st.global.u32([p], x);
    }

    /// `a + K`, with a compile-time argument named as the macros name the first parameter named `_`.
    #[instances(10, 20)]
    pub unsafe extern "ptx-kernel" fn counted<const __kernite_0: u32>(_: u32, a: u32, p: *mut u32) {
        let x: u32;
        add.u32(x, a, __kernite_0);
        st.global.u32([p], x);
    }
}

#[test]
fn each_x86_kernel_reads_what_comes_after_the_parameters_named_underscore() {
    // What a number named `_` is given, which the kernels would show if they read it, and a pointer
    // named `_`, to memory they must not write.
    let (junk, mut written) = (999u64, 0u64);
    let (junk, written) = (&raw const junk, &raw mut written);
    let grid = kernite_cpu::Grid::new(1);
    unsafe {
        assert_eq!(cpu::plain(1, 50, junk, 8), 42);
        assert_eq!(cpu::named_plain(1, 2, 50), 49);
        assert_eq!(cpu::tiles::<10>(1, 50, junk, 8), 52);
        assert_eq!(cpu::tiles::<20>(1, 50, junk, 8), 62);
        assert_eq!(cpu::named_tiles::<10>(100, 1, 50, 200, junk, 8), 52 + 200 + 100);
        assert_eq!(cpu::counted::<20>(1, 50), 70);
        // That `$launch`, a cooperative kernel, writes `$want` through `written`.
        macro_rules! writes {
            ($want:expr, $launch:expr) => {{
                *written = 0;
                $launch.unwrap();
                assert_eq!(*written, $want, stringify!($launch));
            }};
        }
        writes!(42, cpu::together(grid, 1, 50, junk.cast_mut(), written, 8));
        writes!(52, cpu::together_tiles::<10>(grid, 1, 50, junk.cast_mut(), written, 8));
        writes!(62, cpu::together_tiles::<20>(grid, 1, 50, junk.cast_mut(), written, 8));
        writes!(49, cpu::named_together(grid, 1, 2, 50, junk.cast_mut(), written));
        writes!(70, cpu::counted_together::<20>(grid, 1, 50, written));
        assert_eq!(*junk, 999);
    }
}

#[test]
fn each_gpu_kernel_reads_what_comes_after_the_parameters_named_underscore() {
    let Ok(cuda) = kernite_cuda::Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
    let kernels = gpu::Kernels::load(&cuda).unwrap();
    let (junk, mut written) = (cuda.upload(&[999u32]).unwrap(), cuda.alloc::<u32>(1).unwrap());
    let grid = kernite_cuda::Grid::new(1, 1);
    // That `$launch` writes `$want` through `written`.
    macro_rules! writes {
        ($want:expr, $launch:expr) => {{
            written.write(&[0]).unwrap();
            unsafe { $launch.unwrap() };
            cuda.sync().unwrap();
            assert_eq!(written.read().unwrap(), [$want], stringify!($launch));
        }};
    }
    writes!(42, kernels.plain(grid, 1, 50, junk.ptr(), written.ptr(), 8));
    writes!(52, kernels.tiles::<10>(grid, 1, 50, junk.ptr(), written.ptr(), 8));
    writes!(62, kernels.tiles::<20>(grid, 1, 50, junk.ptr(), written.ptr(), 8));
    writes!(52 + 200 + 100, kernels.named_tiles::<10>(grid, 100, 1, 50, 200, junk.ptr(), written.ptr(), 8));
    writes!(70, kernels.counted::<20>(grid, 1, 50, written.ptr()));
    if cuda.cooperative() {
        writes!(42, kernels.together(grid, 1, 50, junk.ptr(), written.ptr(), 8));
        writes!(52, kernels.together_tiles::<10>(grid, 1, 50, junk.ptr(), written.ptr(), 8));
        writes!(62, kernels.together_tiles::<20>(grid, 1, 50, junk.ptr(), written.ptr(), 8));
    } else {
        eprintln!("{} launches no kernel cooperatively: skipped", cuda.name());
    }
    assert_eq!(junk.read().unwrap(), [999]);
}
