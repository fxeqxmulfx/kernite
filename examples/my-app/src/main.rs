//! An example user of kgen: kernels written where they are called, in `#[kgen_macros::kernels]`,
//! and safe functions over them. The tests hold each kernel to a scalar reference that does
//! the same float operations in the same order, bit for bit, and its listing to the one in
//! `snapshots/`.

/// The kernels. Each reads the floats its pointers point at, and needs AVX2 and FMA.
#[kgen_macros::kernels]
mod kernels {
    /// The sum of the eight floats at `xs`.
    pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
        let v: ymm;
        vmovups(v, [xs]);
        ret_sum(v);
    }

    /// The dot product of the `n` floats at `x` and the `n` at `y`, `n` a multiple of 8.
    pub unsafe extern "sysv64" fn dot1(x: *const f32, y: *const f32, n: usize) -> f32 {
        dot::<1>(x, y, n);
    }

    /// The same, `n` a multiple of 32, by four accumulators.
    pub unsafe extern "sysv64" fn dot4(x: *const f32, y: *const f32, n: usize) -> f32 {
        dot::<4>(x, y, n);
    }

    /// `UNROLL` accumulators of eight lanes take turns, so that a fused multiply-add does not
    /// wait for the one before it; then they are added in order, and the lanes summed as in
    /// `sum8`. Each holds a register through the loop and the load needs one more, so sixteen
    /// would take 17 of the 16 there are: kgen does not spill, it says which variables hold
    /// them.
    fn dot<const UNROLL: usize>(x: gpr, y: gpr, n: gpr) {
        test(n, n);
        jz(empty);
        let acc: [ymm; UNROLL];
        for k in 0..UNROLL {
            zero(acc[k]);
        }
        label(top);
        for k in 0..UNROLL {
            let xs: ymm;
            vmovups(xs, [x + 32 * k]);
            vfmadd231ps(acc[k], xs, [y + 32 * k]);
        }
        add(x, 32 * UNROLL);
        add(y, 32 * UNROLL);
        sub(n, 8 * UNROLL);
        jnz(top);

        note("the accumulators, in order");
        for k in 1..UNROLL {
            vaddps(acc[0], acc[0], acc[k]);
        }
        ret_sum(acc[0]);

        label(empty);
        let zero: ymm;
        zero(zero);
        ret(zero.xmm);
    }

    /// Returns the sum of `v`'s lanes: the high half added to the low, then the high pair of
    /// that to the low pair, then the second float to the first.
    fn ret_sum(v: ymm) {
        note("the lanes");
        let high: ymm;
        vextractf128(high.xmm, v, 1);
        let sum: ymm;
        vaddps(sum.xmm, v.xmm, high.xmm);
        let pair: ymm;
        vmovhlps(pair.xmm, sum.xmm, sum.xmm);
        vaddps(sum.xmm, sum.xmm, pair.xmm);
        vmovshdup(pair.xmm, sum.xmm);
        vaddss(sum.xmm, sum.xmm, pair.xmm);
        ret(sum.xmm);
    }
}

/// The GPU kernels, for NVIDIA GPUs: a thread each element.
#[kgen_macros::kernels]
mod gpu {
    /// `y[i] = a * x[i] + y[i]` for each `i < n`, rounded once.
    pub unsafe extern "ptx-kernel" fn saxpy(a: f32, x: *const f32, y: *mut f32, n: u32) {
        let i: u32;
        global_index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let (at_x, at_y): (u64, u64);
        mad.wide.u32(at_x, i, 4, x);
        mad.wide.u32(at_y, i, 4, y);
        let (xi, yi): (f32, f32);
        ld.global.f32(xi, [at_x]);
        ld.global.f32(yi, [at_y]);
        fma.rn.f32(yi, a, xi, yi);
        st.global.f32([at_y], yi);
    }

    /// `sums[b]`, the sum of the 256 floats of `x` from `256 b`: a block of 256 threads each,
    /// adding in a tree in the shared memory it declares, half as many threads each level.
    pub unsafe extern "ptx-kernel" fn block_sums(x: *const f32, sums: *mut f32) {
        let tile: shared<[f32; 256], 16>;
        let (t, b, i, mine): (u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(b, ctaid.x);
        mad.lo.u32(i, b, 256, t);
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        let (v, w): (f32, f32);
        ld.global.f32(v, [at]);
        mov.u32(mine, tile);
        mad.lo.u32(mine, t, 4, mine);
        st.shared.f32([mine], v);
        let lower: pred;
        for half in [128, 64, 32, 16, 8, 4, 2, 1] {
            bar.sync(0);
            setp.lt.u32(lower, t, half);
            if lower {
                ld.shared.f32(w, [mine + 4 * half]);
                add.rn.f32(v, v, w);
                st.shared.f32([mine], v);
            }
        }
        setp.eq.u32(lower, t, 0);
        if lower {
            mad.wide.u32(at, b, 4, sums);
            st.global.f32([at], v);
        }
    }

    /// Each block's run of floats of `x`, a thread each, reversed in place through the shared
    /// memory the launch gives: 4 bytes a thread.
    pub unsafe extern "ptx-kernel" fn reverse_runs(x: *mut f32) {
        let run: shared<[f32]>;
        let (t, n, i, j, base): (u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        global_index(i);
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        let v: f32;
        ld.global.f32(v, [at]);
        mov.u32(base, run);
        mad.lo.u32(j, t, 4, base);
        st.shared.f32([j], v);
        bar.sync(0);
        sub.u32(j, n, t);
        sub.u32(j, j, 1);
        mad.lo.u32(j, j, 4, base);
        ld.shared.f32(v, [j]);
        st.global.f32([at], v);
    }

    /// `x[i] *= a` for each `i < n`, in `f16`, rounded once.
    pub unsafe extern "ptx-kernel" fn scale_f16(x: *mut f16, a: f16, n: u32) {
        let i: u32;
        global_index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let at: u64;
        mad.wide.u32(at, i, 2, x);
        let v: f16;
        ld.global.b16(v, [at]);
        mul.rn.f16(v, v, a);
        st.global.b16([at], v);
    }

    /// The index of this thread among all of the grid's.
    fn global_index(i: u32) {
        let (block, size, thread): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        mad.lo.u32(i, block, size, thread);
    }
}

use kgen_cuda::{Cuda, Grid, f16};

/// Whether this cpu runs the kernels.
fn supported() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// The sum of `xs`.
fn sum8(xs: &[f32; 8]) -> f32 {
    assert!(supported(), "the kernels need AVX2 and FMA");
    // SAFETY: eight floats are there to read, and the cpu has what the kernel needs.
    unsafe { kernels::sum8(xs.as_ptr()) }
}

/// The dot product of `x` and `y`, which must be as long as each other and a multiple of
/// eight floats: by four accumulators when it is a multiple of 32, else by one.
fn dot(x: &[f32], y: &[f32]) -> f32 {
    assert!(supported(), "the kernels need AVX2 and FMA");
    assert_eq!(x.len(), y.len(), "x and y differ in length");
    assert!(x.len().is_multiple_of(8), "{} floats are not a multiple of eight", x.len());
    let kernel = if x.len().is_multiple_of(32) { kernels::dot4 } else { kernels::dot1 };
    // SAFETY: `x.len()` floats are there to read behind each pointer, a whole number of the
    // kernel's steps, and the cpu has what it needs.
    unsafe { kernel(x.as_ptr(), y.as_ptr(), x.len()) }
}

/// `y = a * x + y` on the GPU, each element rounded once; `x` and `y` must be as long as each
/// other.
fn saxpy(cuda: &Cuda, kernels: &gpu::Kernels, a: f32, x: &[f32], y: &mut [f32]) -> Result<(), kgen_cuda::Error> {
    assert_eq!(x.len(), y.len(), "x and y differ in length");
    let n = u32::try_from(x.len()).expect("fewer than 2^32 floats");
    if n == 0 {
        return Ok(());
    }
    let (xs, ys) = (cuda.upload(x)?, cuda.upload(y)?);
    // SAFETY: `n` floats behind each pointer, in buffers that last until `read_into` has
    // waited for the kernel.
    unsafe { kernels.saxpy(Grid::new(n.div_ceil(256), 256), a, xs.ptr(), ys.ptr(), n)? };
    ys.read_into(y)
}

/// The sum of each run of 256 floats of `x`, on the GPU, added in a tree; `x` must be a whole
/// number of runs.
fn block_sums(cuda: &Cuda, kernels: &gpu::Kernels, x: &[f32]) -> Result<Vec<f32>, kgen_cuda::Error> {
    assert!(x.len().is_multiple_of(256), "{} floats are no whole number of runs of 256", x.len());
    let blocks = u32::try_from(x.len() / 256).expect("fewer than 2^32 runs");
    let mut sums = vec![0.0; blocks as usize];
    if blocks == 0 {
        return Ok(sums);
    }
    let (xs, out) = (cuda.upload(x)?, cuda.alloc::<f32>(sums.len())?);
    // SAFETY: 256 floats a block behind `xs`, and a float a block behind `out`, in buffers
    // that last until `read_into` has waited for the kernel.
    unsafe { kernels.block_sums(Grid::new(blocks, 256), xs.ptr(), out.ptr())? };
    out.read_into(&mut sums)?;
    Ok(sums)
}

/// Each run of `run` floats of `x` reversed, on the GPU; `x` must be a whole number of runs,
/// and a run no more threads than a block has.
fn reverse_runs(cuda: &Cuda, kernels: &gpu::Kernels, x: &mut [f32], run: usize) -> Result<(), kgen_cuda::Error> {
    assert!(run > 0 && x.len().is_multiple_of(run), "{} floats are no whole number of runs of {run}", x.len());
    let (blocks, threads) = (u32::try_from(x.len() / run).expect("fewer than 2^32 runs"), run as u32);
    if blocks == 0 {
        return Ok(());
    }
    let xs = cuda.upload(x)?;
    // SAFETY: `run` floats a block behind `xs`, and 4 bytes of shared memory a thread, in a
    // buffer that lasts until `read_into` has waited for the kernel.
    unsafe { kernels.reverse_runs(Grid::new(blocks, threads).with_shared(4 * threads), xs.ptr())? };
    xs.read_into(x)
}

/// `x *= a` on the GPU, each element in `f16`, rounded once.
fn scale_f16(cuda: &Cuda, kernels: &gpu::Kernels, a: f16, x: &mut [f16]) -> Result<(), kgen_cuda::Error> {
    let n = u32::try_from(x.len()).expect("fewer than 2^32 halves");
    if n == 0 {
        return Ok(());
    }
    let xs = cuda.upload(x)?;
    // SAFETY: `n` halves behind the pointer, in a buffer that lasts until `read_into` has
    // waited for the kernel.
    unsafe { kernels.scale_f16(Grid::new(n.div_ceil(256), 256), xs.ptr(), a, n)? };
    xs.read_into(x)
}

fn main() {
    if !supported() {
        eprintln!("this cpu lacks AVX2 or FMA, which the kernels are written in");
        std::process::exit(1);
    }
    println!("sum8(1, ..., 8) = {}", sum8(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
    for n in [1024, 1000] {
        let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32 - 3.0).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32 / 2.0).collect();
        let naive: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        println!("dot of {n} floats = {}; a loop in Rust says {naive}", dot(&x, &y));
    }
    let gpu = Cuda::new().and_then(|cuda| gpu::Kernels::load(&cuda).map(|kernels| (cuda, kernels)));
    match gpu {
        Ok((cuda, kernels)) => {
            let x: Vec<f32> = (0..1000).map(|i| i as f32 / 8.0).collect();
            let mut y = vec![1.0; 1000];
            match saxpy(&cuda, &kernels, 2.0, &x, &mut y) {
                Ok(()) => println!("saxpy on the {}: 2 * {} + 1 = {}", cuda.name(), x[999], y[999]),
                Err(e) => println!("saxpy failed on the {}: {e}", cuda.name()),
            }
            let ones: Vec<f32> = (0..1024).map(|i| i as f32).collect();
            match block_sums(&cuda, &kernels, &ones) {
                Ok(sums) => println!("sums of 0..1024 in runs of 256, in shared memory: {sums:?}"),
                Err(e) => println!("block_sums failed: {e}"),
            }
            let mut run: Vec<f32> = (0..8).map(|i| i as f32).collect();
            match reverse_runs(&cuda, &kernels, &mut run, 4) {
                Ok(()) => println!("0..8 reversed in runs of 4, in shared memory the launch gives: {run:?}"),
                Err(e) => println!("reverse_runs failed: {e}"),
            }
            let mut halves = [f16::from_f32(0.1), f16::from_f32(1000.0)];
            match scale_f16(&cuda, &kernels, f16::from_f32(3.0), &mut halves) {
                Ok(()) => println!("[0.1, 1000] times 3 in f16: {halves:?}"),
                Err(e) => println!("scale_f16 failed: {e}"),
            }
        }
        Err(e) => println!("no GPU to run saxpy on: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use kgen_cuda::{Cuda, f16};

    use super::{gpu, kernels};

    /// Each kernel's listing against the one in `snapshots/`, and the GPU kernels' PTX, so that
    /// a change to kgen or to a kernel that moves an instruction shows in review. With
    /// `KGEN_BLESS=1` the files are written instead, for the diff to be reviewed and committed.
    #[test]
    fn listings_match_their_snapshots() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("snapshots");
        let bless = std::env::var_os("KGEN_BLESS").is_some();
        let mut changed = vec![];
        let files = kernels::LISTINGS.iter().map(|&(name, text)| (format!("{name}.s"), text));
        for (name, text) in files.chain([("gpu.ptx".to_string(), gpu::PTX)]) {
            let path = dir.join(&name);
            if bless {
                fs::create_dir_all(&dir).unwrap();
                fs::write(&path, text).unwrap();
            } else if fs::read_to_string(&path).ok().as_deref() != Some(text) {
                changed.push(name);
            }
        }
        assert!(changed.is_empty(), "listings changed: {changed:?}; rerun with KGEN_BLESS=1 and review the diff");
    }

    /// Pseudo-random floats in [-1, 1), the same for the same seed.
    fn floats(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 8) as f32 / (1 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// What `sum8` does to eight floats, in its order: the high half added to the low, then
    /// the high pair of that to the low pair, then the second float to the first.
    fn sum_lanes(v: [f32; 8]) -> f32 {
        let s: [f32; 4] = std::array::from_fn(|j| v[j] + v[j + 4]);
        (s[0] + s[2]) + (s[1] + s[3])
    }

    /// What `dot(unroll)` computes, in its order: `unroll` accumulators of eight lanes, each
    /// product added with one rounding, then the accumulators added in order, then the lanes
    /// as `sum8` adds them.
    fn dot(x: &[f32], y: &[f32], unroll: usize) -> f32 {
        let mut acc = vec![[0f32; 8]; unroll];
        for (xs, ys) in x.chunks(8 * unroll).zip(y.chunks(8 * unroll)) {
            for (k, lanes) in acc.iter_mut().enumerate() {
                for (j, lane) in lanes.iter_mut().enumerate() {
                    *lane = xs[8 * k + j].mul_add(ys[8 * k + j], *lane);
                }
            }
        }
        let mut v = acc[0];
        for other in &acc[1..] {
            for (lane, o) in v.iter_mut().zip(other) {
                *lane += o;
            }
        }
        sum_lanes(v)
    }

    #[test]
    fn sum8_is_its_reference_bit_for_bit() {
        assert!(super::supported(), "the kernels need AVX2 and FMA");
        for seed in 1..=1000 {
            let xs: [f32; 8] = floats(8, seed).try_into().unwrap();
            // SAFETY: eight floats, and a cpu that runs the kernels.
            let got = unsafe { kernels::sum8(xs.as_ptr()) };
            assert_eq!(got.to_bits(), sum_lanes(xs).to_bits(), "{xs:?}");
        }
    }

    #[test]
    fn dot_is_its_reference_bit_for_bit() {
        assert!(super::supported(), "the kernels need AVX2 and FMA");
        for n in [0, 32, 96, 1024, 4000] {
            let (x, y) = (floats(n, 2 * n as u32 + 1), floats(n, 2 * n as u32 + 2));
            // SAFETY: `n` floats behind each pointer, a multiple of 32, and a cpu that runs
            // the kernels.
            let (one, four) =
                unsafe { (kernels::dot1(x.as_ptr(), y.as_ptr(), n), kernels::dot4(x.as_ptr(), y.as_ptr(), n)) };
            assert_eq!(one.to_bits(), dot(&x, &y, 1).to_bits(), "{n} floats, one accumulator");
            assert_eq!(four.to_bits(), dot(&x, &y, 4).to_bits(), "{n} floats, four accumulators");
        }
    }

    /// The GPU, and its kernels; on a machine with none, `None`, and the test is skipped.
    fn cuda() -> Option<(Cuda, gpu::Kernels)> {
        let cuda = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")).ok()?;
        let kernels = gpu::Kernels::load(&cuda).unwrap();
        Some((cuda, kernels))
    }

    /// What `block_sums` does to each run of 256, in its order: each float of the lower half
    /// added to the one as far into the upper, then the same of what that leaves.
    fn tree(run: &[f32]) -> f32 {
        let mut tile = run.to_vec();
        for half in [128, 64, 32, 16, 8, 4, 2, 1] {
            for t in 0..half {
                tile[t] += tile[t + half];
            }
        }
        tile[0]
    }

    #[test]
    fn block_sums_is_its_tree_bit_for_bit() {
        let Some((cuda, kernels)) = cuda() else { return };
        for runs in [0, 1, 2, 37] {
            let x = floats(256 * runs, runs as u32 + 7);
            let want: Vec<u32> = x.chunks(256).map(|run| tree(run).to_bits()).collect();
            let got = super::block_sums(&cuda, &kernels, &x).unwrap();
            assert_eq!(got.iter().map(|s| s.to_bits()).collect::<Vec<_>>(), want, "{runs} runs");
        }
    }

    #[test]
    fn reverse_runs_reverses_each_run() {
        let Some((cuda, kernels)) = cuda() else { return };
        // A run of 1024 is as many threads as a block has, and 4 KB of shared memory.
        for (n, run) in [(12, 3), (1024, 1), (4096, 1024), (1000, 40)] {
            let mut x: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let want: Vec<f32> = x.chunks(run).flat_map(|r| r.iter().rev().copied()).collect();
            super::reverse_runs(&cuda, &kernels, &mut x, run).unwrap();
            assert_eq!(x, want, "{n} floats in runs of {run}");
        }
    }

    /// `scale_f16` on every `f16` against the product in `f32`, which is exact, rounded to an
    /// `f16` once as `mul.rn.f16` rounds it.
    #[test]
    fn scale_f16_rounds_the_product_once() {
        let Some((cuda, kernels)) = cuda() else { return };
        for a in [3.0, -0.1, 1.0e-3, 600.0] {
            let a = f16::from_f32(a);
            let mut x: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
            let want: Vec<f16> = x.iter().map(|x| f16::from_f32(x.to_f32() * a.to_f32())).collect();
            super::scale_f16(&cuda, &kernels, a, &mut x).unwrap();
            for (bits, (got, want)) in (0..=u16::MAX).zip(x.iter().zip(&want)) {
                match want.to_f32().is_nan() {
                    true => assert!(got.to_f32().is_nan(), "{bits:#06x} times {a}"),
                    false => assert_eq!(got.to_bits(), want.to_bits(), "{bits:#06x} times {a}"),
                }
            }
        }
    }

    /// `saxpy` on the GPU against `f32::mul_add`, which rounds once as `fma.rn.f32` does; on a
    /// machine with no GPU, skipped.
    #[test]
    fn saxpy_is_mul_add_bit_for_bit() {
        let cuda = match Cuda::new() {
            Ok(cuda) => cuda,
            Err(e) => return eprintln!("no GPU: skipped. {e}"),
        };
        let kernels = gpu::Kernels::load(&cuda).unwrap();
        for n in [0, 1, 255, 256, 257, 10_000] {
            let (x, mut y) = (floats(n, 3 * n as u32 + 1), floats(n, 3 * n as u32 + 2));
            let want: Vec<u32> = x.iter().zip(&y).map(|(x, y)| 1.5f32.mul_add(*x, *y).to_bits()).collect();
            super::saxpy(&cuda, &kernels, 1.5, &x, &mut y).unwrap();
            assert_eq!(y.iter().map(|y| y.to_bits()).collect::<Vec<_>>(), want, "{n} floats");
        }
    }
}
