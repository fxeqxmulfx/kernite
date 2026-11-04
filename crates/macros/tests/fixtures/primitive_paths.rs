//! A type written as a primitive's path, `core::primitive::u64`, is that primitive wherever a type is written: Rust
//! names each by a path as well as by a name, and so does a module of kernels. Each module here is twice, once by the
//! names and once by the paths, in the four spellings Rust has, `core::primitive::u64`, `::core::primitive::u64`,
//! `std::primitive::u64` and `::std::primitive::u64`, and with a name written raw, `r#core::primitive::u64` and
//! `::core::primitive::r#u64`. Rust builds each without a word, a kernel by a path is called as the same kernel by
//! its name is, and its listing, or its PTX, is the one of the kernel by its name.

#![deny(warnings)]

/// The module by the names, for x86.
#[kernite_macros::kernels]
pub mod plain_x86 {
    const STEP: usize = 3;
    const LAST: u8 = 2 as u8;
    pub const HALF: f32 = 0.5;
    pub static COUNTS: global<[u32; 4]> = [1, 2, 3, 4];

    /// `x` moved `N` on, if `B`.
    fn moved<const N: i32, const B: bool>(x: gpr) {
        if B {
            add(x, N);
        }
    }

    /// `x` moved a step and `N` on, and one at `p`.
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn plus<const N: u64>(x: u64, p: *mut u32) -> u64 {
        let t: local<[u32; 4]>;
        moved::<3, true>(x);
        add(x, N);
        add(x, STEP);
        add(x, LAST);
        mov(dword[p], 1);
        ret(x);
    }

    /// Half of `a` as a float, and `b` for `n` steps.
    pub unsafe extern "sysv64" fn halves(a: f32, b: f64, q: *const [u8; 4]) -> f64 {
        for i in 0..(2 as usize) {
            vaddsd(b, b, b);
        }
        ret(b);
    }

    /// A cooperative kernel: one at `x`, from each thread.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32, n: usize) {
        let at: gpr;
        lea(at, [COUNTS]);
        mov(dword[x], 1);
    }
}

/// The module by the paths, for x86: the kernels of `plain_x86`.
#[kernite_macros::kernels]
pub mod paths_x86 {
    const STEP: core::primitive::usize = 3;
    const LAST: ::core::primitive::u8 = 2 as std::primitive::u8;
    pub const HALF: ::std::primitive::f32 = 0.5;
    pub static COUNTS: global<[core::primitive::u32; 4]> = [1, 2, 3, 4];

    /// `x` moved `N` on, if `B`.
    fn moved<const N: ::core::primitive::i32, const B: std::primitive::bool>(x: gpr) {
        if B {
            add(x, N);
        }
    }

    /// `x` moved a step and `N` on, and one at `p`.
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn plus<const N: ::core::primitive::r#u64>(
        x: r#core::primitive::u64,
        p: *mut ::core::primitive::r#u32,
    ) -> ::std::primitive::u64 {
        let t: local<[std::primitive::u32; 4]>;
        moved::<3, true>(x);
        add(x, N);
        add(x, STEP);
        add(x, LAST);
        mov(dword[p], 1);
        ret(x);
    }

    /// Half of `a` as a float, and `b` for `n` steps.
    pub unsafe extern "sysv64" fn halves(
        a: ::std::primitive::r#f32,
        b: core::primitive::f64,
        q: *const [r#core::primitive::u8; 4],
    ) -> std::primitive::f64 {
        for i in 0..(2 as ::core::primitive::usize) {
            vaddsd(b, b, b);
        }
        ret(b);
    }

    /// A cooperative kernel: one at `x`, from each thread.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut std::primitive::u32, n: ::std::primitive::usize) {
        let at: gpr;
        lea(at, [COUNTS]);
        mov(dword[x], 1);
    }
}

/// The module by the names, for a GPU.
#[kernite_macros::kernels]
pub mod plain_gpu {
    pub const WIDTH: u32 = 4;
    pub const SCALE: f32 = 1.5;
    pub static BYTES: global<[u8; 4]>;

    /// `a` twice.
    extern "ptx-func" fn twice(a: u32) -> u32 {
        add.u32(a, a, a);
        ret(a);
    }

    /// `x` moved `K` on.
    fn moved<const K: u64>(x: u32) {
        add.u32(x, x, K);
    }

    /// `N` and twice `w`, at `out`.
    #[instances(4, 8)]
    pub unsafe extern "ptx-kernel" fn fill<const N: u32>(out: *mut u32, w: u32, f: f32) {
        let r: u32;
        let v: u32;
        let t: shared<[f32; 4], 16>;
        mov.u32(v, N);
        moved::<7>(v);
        call(r, twice, (w,));
        add.u32(v, v, r);
        st.global.u32([out], v);
    }

    /// Two floats and a double.
    pub unsafe extern "ptx-kernel" fn floats(p: *mut f32, d: f64, i: i64, s: usize) {
        let x: f32;
        mov.f32(x, 2.5);
        st.global.f32([p], x);
    }
}

/// The module by the paths, for a GPU: the kernels of `plain_gpu`.
#[kernite_macros::kernels]
pub mod paths_gpu {
    pub const WIDTH: ::core::primitive::u32 = 4;
    pub const SCALE: std::primitive::f32 = 1.5;
    pub static BYTES: global<[::std::primitive::u8; 4]>;

    /// `a` twice.
    extern "ptx-func" fn twice(a: r#core::primitive::u32) -> ::core::primitive::r#u32 {
        add.u32(a, a, a);
        ret(a);
    }

    /// `x` moved `K` on.
    fn moved<const K: ::std::primitive::r#u64>(x: ::std::primitive::u32) {
        add.u32(x, x, K);
    }

    /// `N` and twice `w`, at `out`.
    #[instances(4, 8)]
    pub unsafe extern "ptx-kernel" fn fill<const N: r#core::primitive::u32>(
        out: *mut std::primitive::u32,
        w: ::std::primitive::u32,
        f: core::primitive::f32,
    ) {
        let r: ::core::primitive::u32;
        let v: std::primitive::u32;
        let t: shared<[::std::primitive::f32; 4], 16>;
        mov.u32(v, N);
        moved::<7>(v);
        call(r, twice, (w,));
        add.u32(v, v, r);
        st.global.u32([out], v);
    }

    /// Two floats and a double.
    pub unsafe extern "ptx-kernel" fn floats(
        p: *mut core::primitive::f32,
        d: ::core::primitive::f64,
        i: std::primitive::i64,
        s: ::std::primitive::usize,
    ) {
        let x: ::std::primitive::f32;
        mov.f32(x, 2.5);
        st.global.f32([p], x);
    }
}

#[cfg(test)]
mod tests {
    use kernite_cpu::Grid as CpuGrid;
    use kernite_cuda::{Cuda, Grid};

    use super::{paths_gpu, paths_x86, plain_gpu, plain_x86};

    /// The listings are the same, kernel for kernel.
    #[test]
    fn a_kernel_typed_by_paths_has_the_listing_of_the_kernel_typed_by_names() {
        assert_eq!(paths_x86::LISTINGS, plain_x86::LISTINGS);
        assert_eq!(plain_x86::LISTINGS.len(), 4);
    }

    /// The PTX is the same, but for the module's name, which both mangle into their kernels' names.
    #[test]
    fn a_gpu_kernel_typed_by_paths_has_the_ptx_of_the_kernel_typed_by_names() {
        assert_eq!(paths_gpu::PTX.replace("paths_gpu", "plain_gpu"), plain_gpu::PTX);
        assert!(plain_gpu::PTX.contains(".entry"));
    }

    /// The x86 kernels run as Rust calls them, typed by the paths and by the names, to the same result.
    #[test]
    fn a_kernel_typed_by_paths_is_called_as_the_kernel_typed_by_names_is() {
        let (mut a, mut b) = (0u32, 0u32);
        // SAFETY: each kernel writes one `u32` through the pointer it is given, which is to one.
        let (by_name, by_path) = unsafe { (plain_x86::plus::<2>(10, &mut a), paths_x86::plus::<2>(10, &mut b)) };
        assert_eq!((by_name, a), (20, 1));
        assert_eq!((by_path, b), (20, 1));
        let bytes = [1u8, 2, 3, 4];
        // SAFETY: the kernel reads nothing through the pointer it is given.
        let (by_name, by_path) = unsafe { (plain_x86::halves(0.5, 1.5, &bytes), paths_x86::halves(0.5, 1.5, &bytes)) };
        assert_eq!((by_name, by_path), (6.0, 6.0));
        let (mut a, mut b) = (0u32, 0u32);
        // SAFETY: each thread of the grid writes one `u32` through the pointer it is given, which is to one.
        unsafe {
            plain_x86::together(CpuGrid::new(1), &mut a, 1).unwrap();
            paths_x86::together(CpuGrid::new(1), &mut b, 1).unwrap();
        }
        assert_eq!((a, b), (1, 1));
        assert_eq!(paths_x86::COUNTS.read(), plain_x86::COUNTS.read());
        assert_eq!(plain_x86::COUNTS.read(), [1, 2, 3, 4]);
    }

    /// The GPU kernels run as Rust launches them, typed by the paths and by the names, to the same result.
    #[test]
    fn a_gpu_kernel_typed_by_paths_is_launched_as_the_kernel_typed_by_names_is() {
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let (plain, paths) = (plain_gpu::Kernels::load(&cuda).unwrap(), paths_gpu::Kernels::load(&cuda).unwrap());
        let ints = [cuda.alloc::<u32>(1).unwrap(), cuda.alloc::<u32>(1).unwrap()];
        let floats = [cuda.alloc::<f32>(1).unwrap(), cuda.alloc::<f32>(1).unwrap()];
        // SAFETY: each kernel writes one number to the buffer it is given, which holds one, and the launches have
        // ended when they are read.
        unsafe {
            plain.fill::<8>(Grid::new(1, 1), ints[0].ptr(), 5, 1.5).unwrap();
            paths.fill::<8>(Grid::new(1, 1), ints[1].ptr(), 5, 1.5).unwrap();
            plain.floats(Grid::new(1, 1), floats[0].ptr(), 1.0, -2, 3).unwrap();
            paths.floats(Grid::new(1, 1), floats[1].ptr(), 1.0, -2, 3).unwrap();
        }
        cuda.sync().unwrap();
        for (int, float) in ints.iter().zip(&floats) {
            assert_eq!(int.read().unwrap(), [8 + 7 + 10]);
            assert_eq!(float.read().unwrap(), [2.5]);
        }
        let bytes = [1u8, 2, 3, 4];
        paths.BYTES().write(&bytes).unwrap();
        assert_eq!(paths.BYTES().read().unwrap(), bytes);
    }
}
