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

    /// The index of this thread among all of the grid's.
    fn global_index(i: u32) {
        let (block, size, thread): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        mad.lo.u32(i, block, size, thread);
    }
}

use kgen_cuda::{Cuda, Grid};

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
        }
        Err(e) => println!("no GPU to run saxpy on: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use kgen_cuda::Cuda;

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
