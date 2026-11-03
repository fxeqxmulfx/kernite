//! The kernels against the reference, and their listings against `snapshots/`.

use std::fs;
use std::path::Path;

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
/// each size the kernel works in, one less, equal and one more, the others small: `MR` of `m`,
/// `NR` and `NC` of `n`, `UNROLL` and `KC` of `k`; and one past a block in each at once.
fn cpu_shapes() -> Vec<Shape> {
    use cpu::kernels::{KC, MR, NC, NR, UNROLL};
    let around = |x: usize| [x - 1, x, x + 1];
    let mut shapes = vec![(0, NR, 3), (MR, 0, 3), (MR, NR, 0), (1, 1, 1)];
    shapes.extend(around(MR).map(|m| (m, NR + 1, 9)));
    shapes.extend([NR, NC].into_iter().flat_map(around).map(|n| (MR + 1, n, 9)));
    shapes.extend([UNROLL, KC].into_iter().flat_map(around).map(|k| (MR + 1, NR + 1, k)));
    shapes.push((3 * MR + 1, NC + NR + 5, 2 * KC + 1));
    shapes
}

#[test]
fn cpu_sgemm_is_the_reference_bit_for_bit() {
    for shape @ (m, n, k) in cpu_shapes() {
        let (a, b) = (floats(m * k, 1), floats(k * n, 2));
        let mut c = vec![f32::NAN; m * n];
        cpu::sgemm(&mut c, &a, &b, shape);
        same_bits(&c, &reference::sgemm(&a, &b, shape), shape);
    }
}

#[test]
#[should_panic(expected = "c is not 4294967296 rows of 4294967296")]
fn cpu_sgemm_refuses_a_shape_too_large_to_count() {
    cpu::sgemm(&mut [], &[], &[], (1 << 32, 1 << 32, 0));
}

#[test]
fn gpu_sgemm_is_the_reference_bit_for_bit() {
    let Some(cuda) = cuda() else { return };
    let kernels = gpu::kernels::Kernels::load(&cuda).unwrap();
    for shape @ (m, n, k) in SHAPES {
        let (a, b) = (floats(m * k, 1), floats(k * n, 2));
        // SAFETY: `on_gpu` keeps the buffers until it reads `c`, which waits for the kernel.
        let c = on_gpu(&cuda, &a, &b, shape, |c, a, b| unsafe { gpu::launch(&kernels, c, a, b, shape) });
        same_bits(&c, &reference::sgemm(&a, &b, shape), shape);
    }
}

/// Integers from -8 to 8, of which `k` products, at most 64 each, sum exactly in an `f32` in
/// any order while there are fewer than 2^18.
fn integers(n: usize, seed: u32) -> Vec<f16> {
    floats(n, seed).iter().map(|x| f16::from_f32((x * 8.0).round())).collect()
}

#[test]
fn hgemm_is_exact_on_integers() {
    let Some(cuda) = cuda() else { return };
    if cuda.sm() < 80 {
        return eprintln!("the {} is older than Ampere, sm_80: skipped", cuda.name());
    }
    let kernels = tensor::kernels::Kernels::load(&cuda).unwrap();
    let widen = |x: &[f16]| x.iter().map(|h| h.to_f32()).collect::<Vec<_>>();
    for shape @ (m, n, k) in SHAPES {
        let (a, b) = (integers(m * k, 1), integers(k * n, 2));
        // SAFETY: `on_gpu` keeps the buffers until it reads `c`, which waits for the kernel.
        let c = on_gpu(&cuda, &a, &b, shape, |c, a, b| unsafe { tensor::launch(&kernels, c, a, b, shape) });
        same_bits(&c, &reference::sgemm(&widen(&a), &widen(&b), shape), shape);
    }
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
