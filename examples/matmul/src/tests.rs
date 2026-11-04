//! The kernels against the reference, and their listings against `snapshots/`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use kernite_cuda::{Buffer, Cuda, Error, Plain, f16};

use crate::{Shape, cpu, floats, gpu, reference, tensor};

/// The products the GPU's kernels are held to: the least, none, and of each size up to a few
/// hundred.
const SHAPES: [Shape; 7] = [(1, 8, 1), (0, 8, 4), (5, 8, 0), (3, 16, 5), (6, 24, 33), (64, 64, 64), (100, 200, 300)];

/// Asserts that `got` is `want` bit for bit, as products of `shape`.
fn same_bits(got: &[f32], want: &[f32], shape: Shape) {
    let bits = |x: &[f32]| x.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(got), bits(want), "{shape:?}");
}

/// The GPU; on a machine with none, `None`, and the test is skipped.
fn cuda() -> Option<Cuda> {
    Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")).ok()
}

/// The product of `a` and `b` of `shape` that `launch` computes on the GPU, given `c`, `a` and
/// `b` there.
fn on_gpu<T: Plain>(
    cuda: &Cuda,
    a: &[T],
    b: &[T],
    (m, n, _): Shape,
    launch: impl FnOnce(&Buffer<f32>, &Buffer<T>, &Buffer<T>) -> Result<(), Error>,
) -> Vec<f32> {
    let (a, b, c) = (cuda.upload(a).unwrap(), cuda.upload(b).unwrap(), cuda.alloc(m * n).unwrap());
    launch(&c, &a, &b).unwrap();
    c.read().unwrap()
}

/// The products `cpu::sgemm` is held to: none in each of `m`, `n` and `k`, and the least; then
/// each size the kernel works in, one less, equal and one more, the others small: `MR` and
/// `PACK` of `m`, `NR` and `NC` of `n`, `UNROLL` and `KC` of `k`; then `n` of each number of
/// slivers up to two for each of 12 threads and two more, three columns short, so that the
/// strips come in each width, the last one short; and one past a block in each at once.
fn cpu_shapes() -> Vec<Shape> {
    use cpu::kernels::{KC, MR, NC, NR, PACK, UNROLL};
    let around = |x: usize| [x - 1, x, x + 1];
    let mut shapes = vec![(0, NR, 3), (MR, 0, 3), (MR, NR, 0), (1, 1, 1)];
    shapes.extend([MR, PACK].into_iter().flat_map(around).map(|m| (m, NR + 1, 9)));
    shapes.extend([NR, NC].into_iter().flat_map(around).map(|n| (MR + 1, n, 9)));
    shapes.extend([UNROLL, KC].into_iter().flat_map(around).map(|k| (MR + 1, NR + 1, k)));
    shapes.extend((1..=2 * 12 + 2).map(|slivers| (MR + 1, slivers * NR - 3, 9)));
    shapes.push((3 * MR + 1, NC + NR + 5, 2 * KC + 1));
    shapes
}

/// The numbers of threads `cpu::sgemm` is held to the reference on, those of them the machine
/// runs at once: one, two, and six and twelve, the cores and the threads of a six-core CPU with
/// two threads a core.
const THREADS: [usize; 4] = [1, 2, 6, 12];

#[test]
fn cpu_sgemm_is_the_reference_bit_for_bit() {
    let counts = cpu::runnable(&THREADS).unwrap();
    for shape @ (m, n, k) in cpu_shapes() {
        let (a, b) = (floats(m * k, 1), floats(k * n, 2));
        for &threads in &counts {
            // `c`, then a row and a tile more, which the kernel leaves as they are.
            let mut c = vec![f32::NAN; m * n + n + cpu::kernels::NR];
            let (c, past) = c.split_at_mut(m * n);
            cpu::sgemm(c, &a, &b, shape, threads).unwrap();
            same_bits(c, &reference::sgemm(&a, &b, shape), shape);
            assert!(past.iter().all(|x| x.is_nan()), "{shape:?} on {threads} threads: c written past its end");
        }
    }
}

#[test]
#[should_panic(expected = "c is not 4294967296 rows of 4294967296")]
fn cpu_sgemm_refuses_a_shape_too_large_to_count() {
    cpu::sgemm(&mut [], &[], &[], (1 << 32, 1 << 32, 0), 1).unwrap();
}

#[test]
fn cpu_sgemm_takes_usize_max_rows_of_nothing() {
    cpu::sgemm(&mut [], &[], &[], (usize::MAX, 0, 0), 1).unwrap();
}

#[test]
fn cpu_runnable_keeps_as_many_threads_as_the_machine_runs_at_once() {
    let most = kernite_cpu::most().unwrap();
    assert_eq!(cpu::runnable(&[1, most]).unwrap(), [1, most]);
}

#[test]
fn cpu_sgemm_fails_where_its_grid_cannot_start() {
    let most = kernite_cpu::most().unwrap();
    for threads in [0, most + 1] {
        assert!(cpu::sgemm(&mut [0.0], &[1.0], &[1.0], (1, 1, 1), threads).is_err(), "{threads} threads");
    }
}

/// The products the GPU's cores are held to: none in each of `m`, `n` and `k`; then each size the
/// kernel works in, one less, equal and one more, in both ways it loads, four floats at a time
/// where `n` and `k` are multiples of 4 and one at a time where they are not: `BM` of `m`, `BN`
/// of `n`, and every `k` up to four tiles, so that each short first tile and each number of
/// whole tiles after it is met; then tiles in a grid of each shape, the last short in each
/// size; and `SHAPES`.
fn gpu_shapes() -> Vec<Shape> {
    use gpu::kernels::{BK, BM, BN};
    let mut shapes = vec![(0, 8, 4), (5, 0, 4), (1, 1, 1)];
    shapes.extend((BM - 1..=BM + 1).flat_map(|m| [(m, 12, 12), (m, 13, 13)]));
    shapes.extend((BN - 4..=BN + 4).flat_map(|n| [(5, n, 12), (5, n, 13)]));
    shapes.extend((0..=4 * BK + 1).flat_map(|k| [(5, 12, k), (5, 13, k)]));
    shapes.extend([(BM + 1, BN + 4, BK + 4), (BM + 1, BN + 1, BK + 1), (2 * BM + 1, 3 * BN - 4, 3 * BK + 4)]);
    shapes.extend(SHAPES);
    shapes
}

#[test]
fn gpu_sgemm_is_the_reference_bit_for_bit() {
    let Some(cuda) = cuda() else { return };
    let kernels = gpu::kernels::Kernels::load(&cuda).unwrap();
    for shape @ (m, n, k) in gpu_shapes() {
        let (a, b) = (floats(m * k, 1), floats(k * n, 2));
        // SAFETY: `on_gpu` keeps the buffers until it reads `c`, which waits for the kernel.
        let c = on_gpu(&cuda, &a, &b, shape, |c, a, b| unsafe { gpu::launch(&kernels, c, a, b, shape) });
        same_bits(&c, &reference::sgemm(&a, &b, shape), shape);
    }
}

#[test]
fn gpu_sgemm_takes_more_tiles_than_a_grid_has_blocks_along_x() {
    let Some(cuda) = cuda() else { return };
    let kernels = gpu::kernels::Kernels::load(&cuda).unwrap();
    // A module for the oldest GPU has 65535 blocks along x at most: one tile more, down and across,
    // in the kernel that loads 16 bytes, `n` and `k` multiples of 4, and in the one that loads 4.
    use gpu::kernels::{BM, BN};
    let shapes = [(65535 * BM + 1, 1, 1), (1, 65535 * BN + 1, 1), (65535 * BM + 1, 4, 4), (4, 65535 * BN + 4, 4)];
    for shape @ (m, n, k) in shapes {
        let (a, b) = (floats(m * k, 1), floats(k * n, 2));
        // SAFETY: `on_gpu` keeps the buffers until it reads `c`, which waits for the kernel.
        let c = on_gpu(&cuda, &a, &b, shape, |c, a, b| unsafe { gpu::launch(&kernels, c, a, b, shape) });
        same_bits(&c, &reference::sgemm(&a, &b, shape), shape);
    }
}

#[test]
fn gpu_sgemm_never_steps_past_k_with_zeros() {
    let Some(cuda) = cuda() else { return };
    let kernels = gpu::kernels::Kernels::load(&cuda).unwrap();
    // Each product, -1e-30 by 1e-30, is less than the least float, so each chain is -0 at every
    // step: a step of a zero past `k` would make it +0.
    for k in 1..=2 * gpu::kernels::BK + 1 {
        for shape @ (m, n, _) in [(3, 5, k), (3, 8, k)] {
            let (a, b) = (vec![-1e-30; m * k], vec![1e-30; k * n]);
            // SAFETY: `on_gpu` keeps the buffers until it reads `c`, which waits for the kernel.
            let c = on_gpu(&cuda, &a, &b, shape, |c, a, b| unsafe { gpu::launch(&kernels, c, a, b, shape) });
            same_bits(&c, &reference::sgemm(&a, &b, shape), shape);
            assert!(c.iter().all(|x| x.to_bits() == (-0.0f32).to_bits()), "{shape:?}: {c:?}");
        }
    }
}

/// Integers from -8 to 8, of which `k` products, at most 64 each, sum exactly in an `f32` in
/// any order while there are fewer than 2^18.
fn integers(n: usize, seed: u32) -> Vec<f16> {
    floats(n, seed).iter().map(|x| f16::from_f32((x * 8.0).round())).collect()
}

/// The products the tensor cores' kernel is held to, `n` and `k` multiples of 8: none in each
/// of `m`, `n` and `k`; then each size the kernel works in, one less, equal and one more where
/// its doc allows: `BM` of `m`, `BN` of `n`, and every `k` up to a few tiles more than the
/// stages hold, so that each short last tile and each number of tiles in the pipeline is met;
/// then tiles in grids of each shape, with a group of tile rows whole and one short, the last
/// tile short in each size; and a few larger.
fn tensor_shapes() -> Vec<Shape> {
    use tensor::kernels::{BK, BM, BN, GROUP, STAGES};
    let mut shapes = vec![(0, 8, 8), (5, 0, 8), (5, 8, 0), (1, 8, 8)];
    shapes.extend((BM - 1..=BM + 1).map(|m| (m, 16, 24)));
    shapes.extend((BN - 8..=BN + 8).step_by(8).map(|n| (5, n, 24)));
    shapes.extend((0..=(STAGES + 2) * BK).step_by(8).map(|k| (5, 24, k)));
    shapes.extend([(BM + 1, BN + 8, BK + 8), (BM - 1, 2 * BN - 8, 3 * BK - 8)]);
    shapes.extend([(GROUP * BM + 1, 3 * BN - 8, BK + 8), ((2 * GROUP + 1) * BM - 1, BN + 8, 8)]);
    shapes.extend([(64, 64, 64), (100, 200, 304), (512, 512, 512)]);
    shapes
}

/// The GPU if it runs `tensor`'s kernels; on one older than Ampere, `None`, and the test is
/// skipped.
fn ampere() -> Option<Cuda> {
    let cuda = cuda()?;
    if cuda.sm() < tensor::SM {
        eprintln!("the {} is older than Ampere, sm_80: skipped", cuda.name());
        return None;
    }
    Some(cuda)
}

/// For halves of integers of `shape`, the `c` that `hgemm` computes, and the reference's.
fn hgemm_and_reference(
    cuda: &Cuda,
    kernels: &tensor::kernels::Kernels,
    shape @ (m, n, k): Shape,
) -> (Vec<f32>, Vec<f32>) {
    let (a, b) = (integers(m * k, 1), integers(k * n, 2));
    let widen = |x: &[f16]| x.iter().map(|h| h.to_f32()).collect::<Vec<_>>();
    // SAFETY: `on_gpu` keeps the buffers until it reads `c`, which waits for the kernel.
    let c = on_gpu(cuda, &a, &b, shape, |c, a, b| unsafe { tensor::launch(kernels, c, a, b, shape) });
    (c, reference::sgemm(&widen(&a), &widen(&b), shape))
}

#[test]
fn hgemm_is_exact_on_integers() {
    let Some(cuda) = ampere() else { return };
    let kernels = tensor::kernels::Kernels::load(&cuda).unwrap();
    for shape in tensor_shapes() {
        let (c, want) = hgemm_and_reference(&cuda, &kernels, shape);
        same_bits(&c, &want, shape);
    }
}

#[test]
fn hgemm_is_exact_over_65536_tiles_down_and_across() {
    let Some(cuda) = ampere() else { return };
    let kernels = tensor::kernels::Kernels::load(&cuda).unwrap();
    // 65536 tiles, one more than the 65535 that 16 bits hold, the last of them short: a count of
    // blocks, or of tiles down or across, held in 16 bits would be 0.
    use tensor::kernels::{BM, BN};
    for shape in [(65535 * BM + 1, 8, 8), (1, 65535 * BN + 8, 8)] {
        let (c, want) = hgemm_and_reference(&cuda, &kernels, shape);
        same_bits(&c, &want, shape);
    }
}

#[test]
fn hgemm_refuses_n_and_k_that_are_not_multiples_of_8() {
    let Some(cuda) = ampere() else { return };
    let kernels = tensor::kernels::Kernels::load(&cuda).unwrap();
    for shape @ (m, n, k) in [(1, 7, 8), (1, 8, 7), (1, 9, 16)] {
        let (c, a, b) =
            (cuda.alloc::<f32>(m * n).unwrap(), cuda.alloc::<f16>(m * k).unwrap(), cuda.alloc::<f16>(k * n).unwrap());
        // SAFETY: nothing is launched: `launch` panics before it.
        let refused = std::panic::catch_unwind(|| unsafe { tensor::launch(&kernels, &c, &a, &b, shape) });
        let message = *refused.unwrap_err().downcast::<String>().unwrap();
        assert_eq!(message, format!("n is {n} and k is {k}: both must be multiples of 8"));
    }
}

/// `ptxas`, if CUDA is here: `$PTXAS`, or on the path, or in a toolkit under `/usr/local`.
fn ptxas() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("PTXAS") {
        return Some(p.into());
    }
    if Command::new("ptxas").arg("--version").output().is_ok_and(|o| o.status.success()) {
        return Some("ptxas".into());
    }
    let mut found: Vec<PathBuf> = fs::read_dir("/usr/local")
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path().join("bin/ptxas"))
        .filter(|p| p.exists())
        .collect();
    found.sort();
    found.pop()
}

#[test]
fn hgemm_spills_nothing() {
    let Some(ptxas) = ptxas() else { return eprintln!("no ptxas here: skipped; set PTXAS to run it") };
    let src = std::env::temp_dir().join(format!("matmul-hgemm-{}.ptx", std::process::id()));
    fs::write(&src, tensor::kernels::PTX).unwrap();
    let run = Command::new(ptxas).args(["-v", "-arch=sm_80", "-o", "/dev/null"]).arg(&src).output();
    // Before the unwrap and the asserts, so that a ptxas that cannot be run, one that fails, or a
    // spill leaves nothing in the temp dir.
    fs::remove_file(src).unwrap();
    let run = run.unwrap();
    let report = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "{report}");
    assert!(report.contains("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"), "{report}");
}

/// Each x86 kernel's listing against the one in `snapshots/`, and the GPU kernels' PTX, so that
/// a change to kernite or to a kernel that moves an instruction shows in review. With
/// `KERNITE_BLESS=1` the files are written instead, for the diff to be reviewed and committed.
#[test]
fn listings_match_their_snapshots() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("snapshots");
    let bless = std::env::var_os("KERNITE_BLESS").is_some();
    let mut changed = vec![];
    let files = cpu::kernels::LISTINGS.iter().map(|&(name, text)| (format!("{name}.s"), text));
    let ptx = [("gpu.ptx".to_string(), gpu::kernels::PTX), ("tensor.ptx".to_string(), tensor::kernels::PTX)];
    for (name, text) in files.chain(ptx) {
        let path = dir.join(&name);
        if bless {
            fs::create_dir_all(&dir).unwrap();
            fs::write(&path, text).unwrap();
        } else if fs::read_to_string(&path).ok().as_deref() != Some(text) {
            changed.push(name);
        }
    }
    assert!(changed.is_empty(), "listings changed: {changed:?}; rerun with KERNITE_BLESS=1 and review the diff");
}
