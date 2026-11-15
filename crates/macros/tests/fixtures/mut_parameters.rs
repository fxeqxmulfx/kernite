//! A `mut` before a parameter's name, in a function of every kind: an x86 kernel, plain, of compile-time arguments,
//! cooperative, and cooperative of arguments, and a helper; and the same for a GPU, with an `extern "ptx-func"`
//! function. Rust builds the crate without a word, under `#![deny(warnings)]`, and the tests run each kernel, the GPU's
//! on the GPU if there is one: each does what it does without the `mut`, for an instruction writes any variable.

#![deny(warnings)]

/// x86 kernels: each gives `a + b` back, or writes it through `p`; those of arguments add `N` as well.
#[kernite_macros::kernels]
pub mod cpu {
    /// `a + b`, with a `mut` before each name.
    pub unsafe extern "sysv64" fn both(mut a: u64, mut b: u64) -> u64 {
        add(a, b);
        ret(a);
    }

    /// `a + b`, with a `mut` before the second name alone.
    pub unsafe extern "sysv64" fn second(a: u64, mut b: u64) -> u64 {
        add(a, b);
        ret(a);
    }

    /// `a + N`.
    #[instances(10, 20)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(mut a: u64) -> u64 {
        add(a, N);
        ret(a);
    }

    /// `a + 1`, by a helper.
    pub unsafe extern "sysv64" fn helped(a: u64) -> u64 {
        bump(a);
        ret(a);
    }

    fn bump(mut x: gpr) {
        add(x, 1);
    }

    /// Every thread writes `a + b` through `p`.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(mut a: u64, p: *mut u64, mut b: u64) {
        add(a, b);
        mov(qword[p], a);
    }

    /// Every thread writes `a + N` through `p`.
    #[cooperative]
    #[instances(10, 20)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(mut a: u64, p: *mut u64) {
        add(a, N);
        mov(qword[p], a);
    }
}

/// GPU kernels: each writes `a + b` through `p`; those of arguments add `N` as well.
#[kernite_macros::kernels]
pub mod gpu {
    /// `a + b`, with a `mut` before each name.
    pub unsafe extern "ptx-kernel" fn both(p: *mut u32, mut a: u32, mut b: u32) {
        add.u32(a, a, b);
        st.global.u32([p], a);
    }

    /// `a + N`.
    #[instances(10, 20)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(p: *mut u32, mut a: u32) {
        add.u32(a, a, N);
        st.global.u32([p], a);
    }

    /// `a + 1`, by a helper.
    pub unsafe extern "ptx-kernel" fn helped(p: *mut u32, a: u32) {
        bump(a);
        st.global.u32([p], a);
    }

    fn bump(mut x: u32) {
        add.u32(x, x, 1);
    }

    /// `a + b`, by a GPU function.
    extern "ptx-func" fn sum(mut a: u32, mut b: u32) -> u32 {
        add.u32(a, a, b);
        ret(a);
    }

    /// `a + b`, by `sum`.
    pub unsafe extern "ptx-kernel" fn called(p: *mut u32, a: u32, b: u32) {
        let r: u32;
        call(r, sum, (a, b));
        st.global.u32([p], r);
    }

    /// `a + b`, by every block.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(p: *mut u32, mut a: u32, b: u32) {
        add.u32(a, a, b);
        st.global.u32([p], a);
    }

    /// `a + N`, by every block.
    #[cooperative]
    #[instances(10, 20)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(p: *mut u32, mut a: u32) {
        add.u32(a, a, N);
        st.global.u32([p], a);
    }
}

#[test]
fn each_x86_kernel_does_what_it_does_without_a_mut() {
    let mut written = 0u64;
    let written = &raw mut written;
    let grid = kernite_cpu::Grid::new(1);
    unsafe {
        assert_eq!(cpu::both(40, 2), 42);
        assert_eq!(cpu::second(40, 2), 42);
        assert_eq!(cpu::tiles::<10>(32), 42);
        assert_eq!(cpu::tiles::<20>(22), 42);
        assert_eq!(cpu::helped(41), 42);
        // That `$launch`, a cooperative kernel, writes 42 through `written`.
        macro_rules! writes {
            ($launch:expr) => {{
                *written = 0;
                $launch.unwrap();
                assert_eq!(*written, 42, stringify!($launch));
            }};
        }
        writes!(cpu::together(grid, 40, written, 2));
        writes!(cpu::together_tiles::<10>(grid, 32, written));
        writes!(cpu::together_tiles::<20>(grid, 22, written));
    }
}

#[test]
fn each_gpu_kernel_does_what_it_does_without_a_mut() {
    let Ok(cuda) = kernite_cuda::Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
    let kernels = gpu::Kernels::load(&cuda).unwrap();
    let mut written = cuda.alloc::<u32>(1).unwrap();
    let grid = kernite_cuda::Grid::new(1, 1);
    // That `$launch` writes 42 through `written`.
    macro_rules! writes {
        ($launch:expr) => {{
            written.write(&[0]).unwrap();
            unsafe { $launch.unwrap() };
            cuda.sync().unwrap();
            assert_eq!(written.read().unwrap(), [42], stringify!($launch));
        }};
    }
    writes!(kernels.both(grid, written.ptr(), 40, 2));
    writes!(kernels.tiles::<10>(grid, written.ptr(), 32));
    writes!(kernels.tiles::<20>(grid, written.ptr(), 22));
    writes!(kernels.helped(grid, written.ptr(), 41));
    writes!(kernels.called(grid, written.ptr(), 40, 2));
    if cuda.cooperative() {
        writes!(kernels.together(grid, written.ptr(), 40, 2));
        writes!(kernels.together_tiles::<10>(grid, written.ptr(), 32));
        writes!(kernels.together_tiles::<20>(grid, written.ptr(), 22));
    } else {
        eprintln!("{} launches no kernel cooperatively: skipped", cuda.name());
    }
}
