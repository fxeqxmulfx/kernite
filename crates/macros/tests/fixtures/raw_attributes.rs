//! Attributes kernite reads, written raw: a kernel's `#[r#instances(..)]` and `#[r#cooperative]`, a GPU kernel's
//! `#[r#maxntid(..)]`, `#[r#reqntid(..)]`, `#[r#minnctapersm(..)]` and `#[r#maxnreg(..)]`, and the module's own,
//! `#[kernels(r#consts = ..)]`. Rust reads `r#instances` as `instances`, so each is the attribute written plain. Rust
//! builds the crate without a word, and the tests find: each x86 module written raw has the `LISTINGS` of the one
//! written plain, and runs as it does; the GPU module written raw is the PTX of the one written plain, has the
//! launchers of the one written plain, `together_blocks` among them, and the driver holds a launch to its directives;
//! and the module that takes its consts as `r#consts = crate::sizes` reads them as the one that takes them as
//! `consts = crate::sizes` does. The tests that need a GPU skip where there is none.

/// x86 kernels: one of instances, one cooperative, and one that is both.
#[kernite_macros::kernels]
pub mod plain {
    /// `x` and `N` added.
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    /// Every thread writes 1 through `p`.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut u64) {
        grid.sync();
        mov(qword[p], 1);
    }

    /// Every thread writes `N` through `p`.
    #[instances(1, 2)]
    #[cooperative]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(p: *mut u64) {
        grid.sync();
        mov(qword[p], N);
    }
}

/// The same kernels, with the attributes written raw.
#[kernite_macros::kernels]
pub mod raw {
    /// `x` and `N` added.
    #[r#instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    /// Every thread writes 1 through `p`.
    #[r#cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut u64) {
        grid.sync();
        mov(qword[p], 1);
    }

    /// Every thread writes `N` through `p`.
    #[r#instances(1, 2)]
    #[r#cooperative]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(p: *mut u64) {
        grid.sync();
        mov(qword[p], N);
    }
}

/// GPU kernels with each directive, one of instances that sizes its block by its compile-time argument, and a
/// cooperative one.
#[kernite_macros::kernels]
pub mod plain_gpu {
    /// Every thread adds 1 to the count at `n`.
    fn count(n: u64) {
        red.global.add.u32([n], 1);
    }

    /// At most 256 threads in a block.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn most(n: *mut u32) {
        count(n);
    }

    /// Exactly 128 threads in a block.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn exactly(n: *mut u32) {
        count(n);
    }

    /// At most 256 threads in a block, and two blocks an SM, at least.
    #[maxntid(256)]
    #[minnctapersm(2)]
    pub unsafe extern "ptx-kernel" fn held(n: *mut u32) {
        count(n);
    }

    /// At most 32 registers a thread.
    #[maxnreg(32)]
    pub unsafe extern "ptx-kernel" fn few(n: *mut u32) {
        count(n);
    }

    /// Exactly `N` threads in a block, for each `N` the kernel is compiled for.
    #[instances(64, 96)]
    #[reqntid(N)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>(n: *mut u32) {
        count(n);
    }

    /// Every thread adds 1 to the count at `n`, and the grid waits for each.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(n: *mut u32) {
        count(n);
        grid.sync();
    }
}

/// The same kernels, with the attributes written raw.
#[kernite_macros::kernels]
pub mod raw_gpu {
    /// Every thread adds 1 to the count at `n`.
    fn count(n: u64) {
        red.global.add.u32([n], 1);
    }

    /// At most 256 threads in a block.
    #[r#maxntid(256)]
    pub unsafe extern "ptx-kernel" fn most(n: *mut u32) {
        count(n);
    }

    /// Exactly 128 threads in a block.
    #[r#reqntid(128)]
    pub unsafe extern "ptx-kernel" fn exactly(n: *mut u32) {
        count(n);
    }

    /// At most 256 threads in a block, and two blocks an SM, at least.
    #[r#maxntid(256)]
    #[r#minnctapersm(2)]
    pub unsafe extern "ptx-kernel" fn held(n: *mut u32) {
        count(n);
    }

    /// At most 32 registers a thread.
    #[r#maxnreg(32)]
    pub unsafe extern "ptx-kernel" fn few(n: *mut u32) {
        count(n);
    }

    /// Exactly `N` threads in a block, for each `N` the kernel is compiled for.
    #[r#instances(64, 96)]
    #[r#reqntid(N)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>(n: *mut u32) {
        count(n);
    }

    /// Every thread adds 1 to the count at `n`, and the grid waits for each.
    #[r#cooperative]
    pub unsafe extern "ptx-kernel" fn together(n: *mut u32) {
        count(n);
        grid.sync();
    }
}

/// A module of consts, which two modules of kernels take.
#[kernite_macros::consts]
pub mod sizes {
    pub const STEP: u64 = 3;
}

/// A module of kernels that takes its consts as `consts = crate::sizes`.
#[kernite_macros::kernels(consts = crate::sizes)]
pub mod by_name {
    /// `x` a step on.
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

/// The same, as `r#consts = crate::sizes`.
#[kernite_macros::kernels(r#consts = crate::sizes)]
pub mod by_raw {
    /// `x` a step on.
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

/// The same checks for each module of x86 kernels.
#[cfg(test)]
macro_rules! x86_kernels_run {
    ($kernels:ident) => {
        let (mut word, mut other) = (0u64, 0u64);
        // SAFETY: each cooperative kernel writes the word it is given, which is there; `tile` is given none.
        unsafe {
            assert_eq!(($kernels::tile::<1>(10), $kernels::tile::<2>(10)), (11, 12));
            $kernels::together(kernite_cpu::Grid::new(1), &mut word).unwrap();
            assert_eq!(word, 1);
            $kernels::together_tiles::<1>(kernite_cpu::Grid::new(1), &mut other).unwrap();
            assert_eq!(other, 1);
            $kernels::together_tiles::<2>(kernite_cpu::Grid::new(1), &mut other).unwrap();
            assert_eq!(other, 2);
        }
    };
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Error, Grid, Ptr};

    use super::{by_name, by_raw, plain, plain_gpu, raw, raw_gpu};

    #[test]
    fn a_kernel_with_attributes_written_raw_has_the_listing_of_the_same_kernel_written_plain() {
        // Two of `tile`, `together` and two of `together_tiles`.
        assert_eq!(plain::LISTINGS.len(), 5);
        assert_eq!(raw::LISTINGS, plain::LISTINGS);
    }

    #[test]
    fn a_kernel_with_attributes_written_plain_runs() {
        x86_kernels_run!(plain);
    }

    #[test]
    fn a_kernel_with_attributes_written_raw_runs() {
        x86_kernels_run!(raw);
    }

    #[test]
    fn a_module_given_its_consts_by_a_key_written_raw_reads_them_as_one_given_them_by_the_key_written_plain() {
        assert_eq!(by_name::LISTINGS.len(), 1);
        assert_eq!(by_raw::LISTINGS, by_name::LISTINGS);
        // SAFETY: the kernel takes a number and gives one.
        assert_eq!(unsafe { by_raw::step(10) }, 13);
    }

    #[test]
    fn a_gpu_module_with_attributes_written_raw_is_the_ptx_of_the_one_written_plain() {
        // The PTX names each kernel by its module's path, `9plain_gpu` or `7raw_gpu`, and differs in nothing else.
        let raw = raw_gpu::PTX.replace("7raw_gpu", "9plain_gpu");
        let directives =
            [".maxntid 256", ".reqntid 128", ".minnctapersm 2", ".maxnreg 32", ".reqntid 64", ".reqntid 96"];
        for directive in directives {
            assert!(plain_gpu::PTX.contains(directive), "{directive} in the PTX:\n{}", plain_gpu::PTX);
        }
        assert_eq!(raw, plain_gpu::PTX);
    }

    /// `blocks` blocks of `threads` threads along x.
    fn grid(blocks: u32, threads: u32) -> Grid {
        Grid { blocks: [blocks, 1, 1], threads: [threads, 1, 1], shared: 0 }
    }

    /// How many threads ran, when `launch` is given a count to add to; or why none did.
    fn ran(cuda: &Cuda, launch: impl Fn(Ptr<u32>) -> Result<(), Error>) -> Result<u32, Error> {
        let mut count = cuda.alloc::<u32>(1)?;
        count.zero()?;
        launch(count.ptr())?;
        cuda.sync()?;
        Ok(count.read()?[0])
    }

    #[test]
    fn a_gpu_kernel_with_attributes_written_raw_runs_over_the_blocks_its_directives_allow() {
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let k = raw_gpu::Kernels::load(&cuda).unwrap();
        // What the driver refuses of a launch, past `CUDA_ERROR_INVALID_VALUE (1): invalid argument`.
        let refused = |launched: Result<u32, Error>| {
            let e = launched.unwrap_err();
            let said = e.message.strip_prefix("CUDA_ERROR_INVALID_VALUE (1): invalid argument: ");
            said.unwrap_or_else(|| panic!("{e}")).to_string()
        };
        // SAFETY: each kernel adds to the count it is given, which lives until the launch has run.
        unsafe {
            assert_eq!(ran(&cuda, |n| k.most(grid(3, 256), n)).unwrap(), 768);
            assert_eq!(ran(&cuda, |n| k.exactly(grid(2, 128), n)).unwrap(), 256);
            assert_eq!(ran(&cuda, |n| k.held(grid(4, 256), n)).unwrap(), 1024);
            assert_eq!(ran(&cuda, |n| k.few(grid(4, 1024), n)).unwrap(), 4096);
            assert_eq!(ran(&cuda, |n| k.tile::<64>(grid(3, 64), n)).unwrap(), 192);
            assert_eq!(ran(&cuda, |n| k.tile::<96>(grid(3, 96), n)).unwrap(), 288);
            // A launch past a directive is refused, with the directive the PTX says.
            let said = refused(ran(&cuda, |n| k.most(grid(1, 512), n)));
            assert_eq!(said, "each block has 512 threads, and the kernel's `.maxntid 256` takes 256 at most");
            let said = refused(ran(&cuda, |n| k.exactly(grid(1, 64), n)));
            assert_eq!(said, "each block has 64 threads along x, and the kernel's `.reqntid 128` takes 128 exactly");
            let said = refused(ran(&cuda, |n| k.tile::<64>(grid(1, 96), n)));
            assert_eq!(said, "each block has 96 threads along x, and the kernel's `.reqntid 64` takes 64 exactly");
        }
        cuda.sync().unwrap();
    }

    #[test]
    fn a_cooperative_gpu_kernel_written_raw_has_a_method_for_the_blocks_of_its_grid_and_runs() {
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let k = raw_gpu::Kernels::load(&cuda).unwrap();
        let blocks = k.together_blocks(32, 0).unwrap();
        assert!(blocks > 0);
        // SAFETY: the kernel adds to the count it is given, which lives until the launch has run.
        let threads = unsafe { ran(&cuda, |n| k.together(grid(blocks, 32), n)) };
        assert_eq!(threads.unwrap(), blocks * 32);
    }
}
