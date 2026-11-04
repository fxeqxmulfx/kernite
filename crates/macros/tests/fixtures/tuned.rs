//! GPU kernels that say their tuning directives, `#[maxntid(..)]`, `#[reqntid(..)]`,
//! `#[minnctapersm(..)]` and `#[maxnreg(..)]`, of one size, two and three, and a kernel of
//! instances that sizes its block by its compile-time argument. Rust builds the crate without a
//! word, though a const is read by a directive alone, and the tests run each kernel on the GPU over
//! the blocks its directives allow. A kernel that uses the registers `#[maxnreg(..)]` allows
//! launches over its most threads at the edge of what a launch gives a block, and not a thread or a
//! step of registers past it. The tests skip where there is no GPU.

#[kernite_macros::kernels]
pub mod gpu {
    /// Read by a directive alone.
    const ROW: u32 = 16;

    /// Every thread adds 1 to the count at `n`.
    fn count(n: u64) {
        red.global.add.u32([n], 1);
    }

    /// At most 256 threads in a block.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn max_x(n: *mut u32) {
        count(n);
    }

    /// At most `ROW * ROW` threads in a block, whatever its shape.
    #[maxntid(ROW, ROW)]
    pub unsafe extern "ptx-kernel" fn max_xy(n: *mut u32) {
        count(n);
    }

    /// At most 256 threads in a block, of three sizes.
    #[maxntid(8, 8, 4)]
    pub unsafe extern "ptx-kernel" fn max_xyz(n: *mut u32) {
        count(n);
    }

    /// Exactly 256 threads in a block.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn req_x(n: *mut u32) {
        count(n);
    }

    /// Exactly `ROW` by `ROW` threads in a block.
    #[reqntid(ROW, ROW)]
    pub unsafe extern "ptx-kernel" fn req_xy(n: *mut u32) {
        count(n);
    }

    /// Exactly 8 by 8 by 4 threads in a block.
    #[reqntid(8, 8, 4)]
    pub unsafe extern "ptx-kernel" fn req_xyz(n: *mut u32) {
        count(n);
    }

    /// At most 256 threads in a block, and two blocks an SM, at least.
    #[maxntid(256)]
    #[minnctapersm(2)]
    pub unsafe extern "ptx-kernel" fn max_blocks(n: *mut u32) {
        count(n);
    }

    /// At most 32 registers a thread, with a block of 1024 threads.
    #[maxnreg(32)]
    pub unsafe extern "ptx-kernel" fn regs(n: *mut u32) {
        count(n);
    }

    /// Exactly `N` threads in a block, for each `N` the kernel is compiled for.
    #[instances(64, 128)]
    #[reqntid(N)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>(n: *mut u32) {
        count(n);
    }

    /// How many floats `heavy` loads.
    const LOADS: u32 = 250;

    /// Loads the `LOADS` floats `v[k]` at `x[tid.x]`, 1024 apart, and writes to `y[tid.x]` the sum of their
    /// products in pairs, `v[k] * v[LOADS - 1 - k]`. All are loaded before the first is used, so that `ptxas`
    /// gives the kernel every register `#[maxnreg(..)]` allows.
    fn heavy(x: u64, y: u64) {
        let (i, at): (u32, u64);
        mov.u32(i, tid.x);
        mad.wide.u32(at, i, 4, x);
        let v: [f32; LOADS];
        for k in 0..LOADS {
            ld.global.f32(v[k], [at + 4096 * k]);
        }
        let s: f32;
        mov.f32(s, 0);
        for k in 0..LOADS {
            fma.rn.f32(s, v[k], v[LOADS - 1 - k], s);
        }
        mad.wide.u32(at, i, 4, y);
        st.global.f32([at], s);
    }

    /// At most `T` threads in a block, with `R` registers a thread: the edge of what a launch gives.
    #[instances((256, 255), (288, 168), (384, 168), (1000, 64))]
    #[maxntid(T)]
    #[maxnreg(R)]
    pub unsafe extern "ptx-kernel" fn heavy_max<const T: u32, const R: u32>(x: *const f32, y: *mut f32) {
        heavy(x, y);
    }

    /// Exactly `T` threads in a block, with 255 registers a thread.
    #[instances(256, 33)]
    #[reqntid(T)]
    #[maxnreg(255)]
    pub unsafe extern "ptx-kernel" fn heavy_req<const T: u32>(x: *const f32, y: *mut f32) {
        heavy(x, y);
    }
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Error, Grid, Module, Ptr, arg};

    use super::gpu;

    /// The GPU and the kernels on it, or none where there is no GPU.
    fn gpu() -> Option<(Cuda, gpu::Kernels)> {
        let cuda = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")).ok()?;
        let kernels = gpu::Kernels::load(&cuda).unwrap();
        Some((cuda, kernels))
    }

    /// `blocks` blocks of `threads` threads along x, y and z.
    fn grid(blocks: u32, threads: [u32; 3]) -> Grid {
        Grid { blocks: [blocks, 1, 1], threads, shared: 0 }
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
    fn a_kernel_runs_over_the_blocks_its_directives_allow() {
        let Some((cuda, k)) = gpu() else { return };
        // SAFETY: each kernel adds to the count it is given, which lives until the launch has run.
        unsafe {
            // `.maxntid` takes a block of that many threads at most in all, of any shape.
            assert_eq!(ran(&cuda, |n| k.max_x(grid(3, [256, 1, 1]), n)).unwrap(), 768);
            assert_eq!(ran(&cuda, |n| k.max_x(grid(3, [1, 1, 1]), n)).unwrap(), 3);
            assert_eq!(ran(&cuda, |n| k.max_xy(grid(2, [16, 16, 1]), n)).unwrap(), 512);
            assert_eq!(ran(&cuda, |n| k.max_xy(grid(2, [256, 1, 1]), n)).unwrap(), 512);
            assert_eq!(ran(&cuda, |n| k.max_xyz(grid(2, [8, 8, 4]), n)).unwrap(), 512);
            assert_eq!(ran(&cuda, |n| k.max_xyz(grid(2, [4, 4, 16]), n)).unwrap(), 512);
            // `.reqntid` takes a block of its shape.
            assert_eq!(ran(&cuda, |n| k.req_x(grid(3, [256, 1, 1]), n)).unwrap(), 768);
            assert_eq!(ran(&cuda, |n| k.req_xy(grid(2, [16, 16, 1]), n)).unwrap(), 512);
            assert_eq!(ran(&cuda, |n| k.req_xyz(grid(2, [8, 8, 4]), n)).unwrap(), 512);
            // `.minnctapersm` and `.maxnreg` ask for more of the GPU, and launch as the kernel does.
            assert_eq!(ran(&cuda, |n| k.max_blocks(grid(4, [256, 1, 1]), n)).unwrap(), 1024);
            assert_eq!(ran(&cuda, |n| k.regs(grid(4, [1024, 1, 1]), n)).unwrap(), 4096);
            // Each instance has the size it is compiled for.
            assert_eq!(ran(&cuda, |n| k.tile::<64>(grid(3, [64, 1, 1]), n)).unwrap(), 192);
            assert_eq!(ran(&cuda, |n| k.tile::<128>(grid(3, [128, 1, 1]), n)).unwrap(), 384);
        }
    }

    /// The name of the kernel in `gpu::PTX` that has `part`, as its `.entry` says it.
    fn entry(part: &str) -> &'static str {
        let mut names = gpu::PTX.lines().filter_map(|l| l.strip_prefix(".visible .entry ")?.strip_suffix('('));
        names.find(|name| name.contains(part)).unwrap_or_else(|| panic!("no kernel has {part}"))
    }

    /// `gpu::PTX` with the directive `from` of the kernel `name` written `to`, by hand.
    fn rewritten(name: &str, from: &str, to: &str) -> String {
        let start = gpu::PTX.find(&format!(".entry {name}(")).unwrap();
        let end = start + gpu::PTX[start..].find("\n{\n").unwrap();
        let head = &gpu::PTX[start..end];
        assert_eq!(head.matches(from).count(), 1, "{name} says {from} once");
        format!("{}{}{}", &gpu::PTX[..start], head.replace(from, to), &gpu::PTX[end..])
    }

    #[test]
    fn a_kernel_at_the_edge_of_the_registers_a_launch_gives_launches_over_its_most_threads() {
        let Some((cuda, k)) = gpu() else { return };
        let (x, y) = (cuda.alloc::<f32>(1024 * 250).unwrap(), cuda.alloc::<f32>(1024).unwrap());
        let (xs, ys) = (x.ptr(), y.ptr());
        let launched = |threads: u32, launch: &dyn Fn(Grid) -> Result<(), Error>| {
            launch(grid(1, [threads, 1, 1])).and_then(|()| cuda.sync())
        };
        // kernite takes each of these. A launch counts a thread's registers by eights and a block's warps
        // by fours: 255 registers for 8 warps, 168 for 9, counted as 12, and 64 for 32, are 65536 at most.
        // SAFETY: each kernel reads `x[i + 1024 j]` for `j` below 250 and writes `y[i]`, for the thread `i`,
        // below 1024; both buffers have room for that.
        unsafe {
            launched(256, &|g| k.heavy_max::<256, 255>(g, xs, ys)).unwrap();
            launched(288, &|g| k.heavy_max::<288, 168>(g, xs, ys)).unwrap();
            launched(384, &|g| k.heavy_max::<384, 168>(g, xs, ys)).unwrap();
            launched(1000, &|g| k.heavy_max::<1000, 64>(g, xs, ys)).unwrap();
            launched(256, &|g| k.heavy_req::<256>(g, xs, ys)).unwrap();
            launched(33, &|g| k.heavy_req::<33>(g, xs, ys)).unwrap();
        }
        // The same PTX by hand, with a step more of registers or a thread more, which kernite refuses: the
        // kernel keeps every register it is given, `ptxas` does not lower `.maxnreg` for the threads of a
        // block, and the launch has fewer. Each is a module of its own.
        let past = [
            ("heavy_maxILj256ELj255EE", ".maxntid 256", ".maxntid 257", 257),
            ("heavy_maxILj288ELj168EE", ".maxnreg 168", ".maxnreg 169", 288),
            ("heavy_maxILj384ELj168EE", ".maxnreg 168", ".maxnreg 169", 384),
            ("heavy_maxILj1000ELj64EE", ".maxnreg 64", ".maxnreg 65", 1000),
            ("heavy_reqILj256EE", ".reqntid 256", ".reqntid 257", 257),
        ];
        for (part, from, to, threads) in past {
            let name = entry(part);
            let module = Module::load(&cuda, &rewritten(name, from, to), &[name], &[]).unwrap();
            // SAFETY: as above.
            let result = unsafe { module.launch(0, grid(1, [threads, 1, 1]), &[arg(&xs), arg(&ys)]) };
            let e = result.and_then(|()| cuda.sync()).unwrap_err();
            assert_eq!(e.call, "cuLaunchKernel", "{part}: {e}");
            assert!(e.message.starts_with("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (701)"), "{part} {to}: {e}");
        }
        // The launches that failed did not harm the GPU.
        cuda.sync().unwrap();
    }
}
