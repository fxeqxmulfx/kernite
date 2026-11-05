//! Kernels whose types are written raw, `r#u64`, which Rust reads as `u64`. The x86 ones are the same kernels as
//! the plain ones beside them, but for the `r#`: a number as a parameter and as the result, an `f32` and an `f64`,
//! a pointer, a kernel of compile-time arguments and a cooperative one, variables and a static. Rust builds the
//! crate without a word, each raw kernel has the listing of its plain twin, and the tests run both and find the
//! same. The GPU ones, which are raw alone, run on the GPU: a launcher takes the types Rust reads, and the static
//! is read back as `u32`s. They skip where there is none.

#[kernite_macros::kernels]
pub mod plain {
    pub static COUNTS: global<[u32; 4]>;

    /// `x` and `n` added.
    pub unsafe extern "sysv64" fn sum(x: u64, n: usize) -> u64 {
        add(x, n);
        ret(x);
    }

    /// `a` and `b` added, as `f32`s.
    pub unsafe extern "sysv64" fn sum32(a: f32, b: f32) -> f32 {
        vaddss(a, a, b);
        ret(a);
    }

    /// `a` and `b` added, as `f64`s.
    pub unsafe extern "sysv64" fn sum64(a: f64, b: f64) -> f64 {
        vaddsd(a, a, b);
        ret(a);
    }

    /// The byte after the one at `p`.
    pub unsafe extern "sysv64" fn after(p: *const u8) -> *const u8 {
        add(p, 1);
        ret(p);
    }

    /// `x` and `N` added.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn plus<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    /// Every thread writes 1 through `p`.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut u32) {
        mov(dword[p], 1);
    }

    /// `x` doubled, by variables.
    pub unsafe extern "sysv64" fn doubled(x: u64) -> u64 {
        let t: gpr;
        let w: [gpr; 2];
        mov(t, x);
        mov(w[0], t);
        mov(w[1], t);
        add(w[0], w[1]);
        ret(w[0]);
    }

    /// Adds 1 to the first of the counts.
    pub unsafe extern "sysv64" fn count() {
        let at: gpr;
        lea(at, [COUNTS]);
        add(dword[at], 1);
    }
}

#[kernite_macros::kernels]
pub mod raw {
    pub static COUNTS: r#global<[r#u32; 4]>;

    /// `x` and `n` added.
    pub unsafe extern "sysv64" fn sum(x: r#u64, n: r#usize) -> r#u64 {
        add(x, n);
        ret(x);
    }

    /// `a` and `b` added, as `f32`s.
    pub unsafe extern "sysv64" fn sum32(a: r#f32, b: r#f32) -> r#f32 {
        vaddss(a, a, b);
        ret(a);
    }

    /// `a` and `b` added, as `f64`s.
    pub unsafe extern "sysv64" fn sum64(a: r#f64, b: r#f64) -> r#f64 {
        vaddsd(a, a, b);
        ret(a);
    }

    /// The byte after the one at `p`.
    pub unsafe extern "sysv64" fn after(p: *const r#u8) -> *const r#u8 {
        add(p, 1);
        ret(p);
    }

    /// `x` and `N` added.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn plus<const N: r#u64>(x: r#u64) -> r#u64 {
        add(x, N);
        ret(x);
    }

    /// Every thread writes 1 through `p`.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut r#u32) {
        mov(dword[p], 1);
    }

    /// `x` doubled, by variables.
    pub unsafe extern "sysv64" fn doubled(x: r#u64) -> r#u64 {
        let t: r#gpr;
        let w: [r#gpr; 2];
        mov(t, x);
        mov(w[0], t);
        mov(w[1], t);
        add(w[0], w[1]);
        ret(w[0]);
    }

    /// Adds 1 to the first of the counts.
    pub unsafe extern "sysv64" fn count() {
        let at: r#gpr;
        lea(at, [COUNTS]);
        add(dword[at], 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub static COUNTS: r#global<[r#u32; 4]>;

    /// Writes `a + b` at `out`.
    pub unsafe extern "ptx-kernel" fn total(out: *mut r#u32, a: r#u32, b: r#u32) {
        let sum: r#u32;
        add.u32(sum, a, b);
        st.global.u32([out], sum);
    }

    /// Writes twice `x` at `out`.
    pub unsafe extern "ptx-kernel" fn doubled(out: *mut r#f32, x: r#f32) {
        let twice: r#f32;
        add.rn.f32(twice, x, x);
        st.global.f32([out], twice);
    }

    /// Writes `base + N` at `out`.
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn offset<const N: r#u32>(out: *mut r#u64, base: r#u64) {
        let at: r#u64;
        add.u64(at, base, N);
        st.global.u64([out], at);
    }

    /// Adds `n` to the first of the counts, once from each thread of the grid.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(n: r#u32) {
        let at: r#u64;
        mov.u64(at, COUNTS);
        red.global.add.u32([at], n);
    }
}

/// The same checks for each module of x86 kernels.
#[cfg(test)]
macro_rules! x86_kernels_run {
    ($kernels:ident) => {
        let mut word = 0u32;
        let bytes = [0u8; 2];
        // SAFETY: each kernel reads or writes what it is given, which is there, and `together` writes one `u32`.
        unsafe {
            assert_eq!($kernels::sum(2, 3), 5);
            assert_eq!($kernels::sum32(1.5, 2.25), 3.75);
            assert_eq!($kernels::sum64(1.5, 2.25), 3.75);
            assert_eq!($kernels::after(bytes.as_ptr()), bytes.as_ptr().add(1));
            assert_eq!(($kernels::plus::<2>(10), $kernels::plus::<3>(10)), (12, 13));
            $kernels::together(kernite_cpu::Grid::new(1), &mut word).unwrap();
            assert_eq!($kernels::doubled(21), 42);
        }
        assert_eq!(word, 1);
    };
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid};

    use super::{gpu, plain, raw};

    #[test]
    fn a_kernel_written_raw_has_the_listing_of_the_same_kernel_written_plain() {
        // `sum`, `sum32`, `sum64`, `after`, two of `plus`, `together`, `doubled` and `count`.
        assert_eq!(plain::LISTINGS.len(), 9);
        assert_eq!(raw::LISTINGS, plain::LISTINGS);
    }

    #[test]
    fn a_kernel_written_plain_runs() {
        x86_kernels_run!(plain);
    }

    #[test]
    fn a_kernel_written_raw_runs() {
        x86_kernels_run!(raw);
    }

    #[test]
    fn a_gpu_kernel_written_raw_runs() {
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let kernels = gpu::Kernels::load(&cuda).unwrap();
        let (word, float, longs) =
            (cuda.alloc::<u32>(1).unwrap(), cuda.alloc::<f32>(1).unwrap(), cuda.alloc::<u64>(2).unwrap());
        let grid = Grid::new(1, 1);
        // SAFETY: each kernel writes one element to a buffer that has it, or adds to the first of the counts,
        // and every launch has ended when its result is read.
        unsafe {
            kernels.total(grid, word.ptr(), 2, 3).unwrap();
            kernels.doubled(grid, float.ptr(), 1.25).unwrap();
            kernels.offset::<1>(grid, longs.ptr(), 10).unwrap();
            kernels.offset::<2>(grid, longs.at(1), 20).unwrap();
            kernels.together(grid, 5).unwrap();
        }
        cuda.sync().unwrap();
        assert_eq!(word.read().unwrap(), [5]);
        assert_eq!(float.read().unwrap(), [2.5]);
        assert_eq!(longs.read().unwrap(), [11, 22]);
        assert_eq!(kernels.COUNTS().read().unwrap(), [5, 0, 0, 0]);
    }
}
