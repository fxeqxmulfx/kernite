//! Matrix products, `c = a b` into `f32`, by kernels written in kernite's language: for the CPU
//! in x86, and for the GPU in PTX, on its cores in `f32` and, on an Ampere GPU or a newer, on its
//! tensor cores from `f16`. The tests hold each kernel to a scalar reference that does the same
//! float operations in the same order, bit for bit, or the tensor cores, which add in an order of
//! their own, to products of integers, exact in any order. `main` times them:
//!
//! ```text
//! cargo run --release -p matmul -- cpu             # the x86 kernels
//! cargo run --release -p matmul -- gpu             # the PTX kernels, on the GPU
//! cargo run --release -p matmul -- gpu 1024 4096   # at these sizes
//! ```

mod cpu;
mod gpu;
#[cfg(test)]
mod reference;
mod tensor;
#[cfg(test)]
mod tests;

use std::process::ExitCode;
use std::time::{Duration, Instant};

use kernite_cuda::{Buffer, Cuda, Error, Grid, Plain, f16};

/// A product's size, `(m, n, k)`: of `a`, `m` rows of `k`, and `b`, `k` rows of `n`, into `c`,
/// `m` rows of `n`.
type Shape = (usize, usize, usize);

/// The sizes timed when none are given: products of `n` rows of `n` by `n` rows of `n`.
const SIZES: [usize; 4] = [256, 512, 1024, 2048];

/// The numbers of threads the x86 kernel is timed on, those of them the machine runs at once: one,
/// then six and twelve, the cores and the threads of a six-core CPU with two threads a core.
const THREADS: [usize; 3] = [1, 6, 12];

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let target = args.next().unwrap_or_default();
    let Ok(mut sizes) = args.map(|arg| arg.parse()).collect::<Result<Vec<usize>, _>>() else {
        return usage();
    };
    if sizes.is_empty() {
        sizes = SIZES.to_vec();
    }
    match target.as_str() {
        "cpu" => on_cpu(&sizes).unwrap_or_else(|e| {
            eprintln!("the CPU failed: {e}");
            ExitCode::FAILURE
        }),
        "gpu" => on_gpu(&sizes).unwrap_or_else(|e| {
            eprintln!("the GPU failed: {e}");
            ExitCode::FAILURE
        }),
        _ => usage(),
    }
}

/// What the arguments may be.
fn usage() -> ExitCode {
    eprintln!("usage: matmul cpu|gpu [n]...");
    ExitCode::FAILURE
}

/// Times the x86 kernel at each size, on each count of [`THREADS`] the machine runs at once; a
/// line on standard error says which it skips.
fn on_cpu(sizes: &[usize]) -> Result<ExitCode, kernite_cpu::Error> {
    if !cpu::supported() {
        eprintln!("the kernels need AVX2 and FMA");
        return Ok(ExitCode::FAILURE);
    }
    let counts = cpu::runnable(&THREADS)?;
    println!("{:>6} {:>7} {:>14}", "n", "threads", "sgemm GFLOPS");
    for &n in sizes {
        let shape = (n, n, n);
        let (a, b, mut c) = (floats(n * n, 1), floats(n * n, 2), vec![0.0; n * n]);
        for &threads in &counts {
            let time = average(|count| {
                for _ in 0..count {
                    cpu::sgemm(&mut c, &a, &b, shape, threads).unwrap_or_else(|e| panic!("the grid failed: {e}"));
                }
            });
            println!("{n:>6} {threads:>7} {:>14.1}", gflops(shape, time));
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// Times the PTX kernels at each size, on the first GPU. `hgemm` is timed where the GPU is
/// Ampere's or newer and `n` is a multiple of 8; the table has a dash for the others.
fn on_gpu(sizes: &[usize]) -> Result<ExitCode, Error> {
    let cuda = Cuda::new()?;
    let cores = gpu::kernels::Kernels::load(&cuda)?;
    let tensor_cores = if cuda.sm() >= tensor::SM { Some(tensor::kernels::Kernels::load(&cuda)?) } else { None };
    println!("{}", cuda.name());
    println!("{:>6} {:>14} {:>14}", "n", "sgemm GFLOPS", "hgemm GFLOPS");
    for &n in sizes {
        let shape = (n, n, n);
        let (a, b) = (floats(n * n, 1), floats(n * n, 2));
        let halves = |x: &[f32]| x.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>();
        // SAFETY: `gpu_gflops` keeps the buffers until it has waited for the kernels.
        let sgemm = gpu_gflops(&cuda, &a, &b, shape, |c, a, b| unsafe { gpu::launch(&cores, c, a, b, shape) })?;
        let hgemm = match &tensor_cores {
            Some(kernels) if n % 8 == 0 => {
                // SAFETY: as above.
                let launch = |c: &_, a: &_, b: &_| unsafe { tensor::launch(kernels, c, a, b, shape) };
                format!("{:.1}", gpu_gflops(&cuda, &halves(&a), &halves(&b), shape, launch)?)
            }
            _ => "-".to_string(),
        };
        println!("{n:>6} {sgemm:>14.1} {hgemm:>14}");
    }
    Ok(ExitCode::SUCCESS)
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

/// Asserts that `lens`, the lengths of `a`, `b` and `c`, are those of a product of `shape`.
fn operands((a, b, c): (usize, usize, usize), (m, n, k): Shape) {
    assert_eq!(Some(a), m.checked_mul(k), "a is not {m} rows of {k}");
    assert_eq!(Some(b), k.checked_mul(n), "b is not {k} rows of {n}");
    assert_eq!(Some(c), m.checked_mul(n), "c is not {m} rows of {n}");
}

/// As [`operands`] asserts, for a GPU kernel, which counts in 32 bits: and that each of them
/// has fewer than 2^32 elements. `shape` in 32 bits.
fn gpu_operands(lens: (usize, usize, usize), shape: Shape) -> (u32, u32, u32) {
    operands(lens, shape);
    let (m, n, k) = shape;
    let fits = |len: usize| u32::try_from(len).is_ok();
    assert!(fits(m * k) && fits(k * n) && fits(m * n), "{m} by {n} by {k} has 2^32 elements or more");
    (m as u32, n as u32, k as u32)
}

/// The grid for `blocks` blocks of `threads` threads, `blocks` 1 or more: a rectangle, since a
/// module that runs on the oldest GPU, as kernite writes it, has at most 65535 blocks along x.
/// A kernel counts its blocks `ctaid.y * nctaid.x + ctaid.x`, and those past `blocks` do nothing.
fn gpu_grid(blocks: u32, threads: u32) -> Grid {
    let x = blocks.min(65535);
    Grid { blocks: [x, blocks.div_ceil(x), 1], threads: [threads, 1, 1], shared: 0 }
}

/// How long a run takes at the least: the 3 that warm up take half a minute, by which time the
/// machine is hot and its clocks are near those it keeps.
const RUN: Duration = Duration::from_secs(10);

/// The average time of one call, where `calls(count)` makes `count` calls back to back and waits
/// for them: of 10 runs after 3 to warm up. A run is as many calls as take [`RUN`] or more: the
/// count doubles until a run takes a second or more, then grows in proportion.
fn average(mut calls: impl FnMut(u32)) -> Duration {
    let mut run = |count| {
        let start = Instant::now();
        calls(count);
        start.elapsed()
    };
    let (mut count, mut time) = (1, run(1));
    while time < Duration::from_secs(1) {
        count *= 2;
        time = run(count);
    }
    let count = count * RUN.div_duration_f64(time).ceil() as u32;
    for _ in 0..3 {
        run(count);
    }
    (0..10).map(|_| run(count)).sum::<Duration>() / (10 * count)
}

/// The GFLOPS of the product of `a` and `b` of `shape` that `launch` starts on the GPU, given
/// `c`, `a` and `b` there, timed by [`average`]. There is no going on once the GPU fails.
fn gpu_gflops<T: Plain>(
    cuda: &Cuda,
    a: &[T],
    b: &[T],
    shape: Shape,
    launch: impl Fn(&Buffer<f32>, &Buffer<T>, &Buffer<T>) -> Result<(), Error>,
) -> Result<f64, Error> {
    let (m, n, _) = shape;
    let (a, b, c) = (cuda.upload(a)?, cuda.upload(b)?, cuda.alloc::<f32>(m * n)?);
    let check = |done: Result<(), Error>| done.unwrap_or_else(|e| panic!("the GPU failed: {e}"));
    let time = average(|count| {
        for _ in 0..count {
            check(launch(&c, &a, &b));
        }
        check(cuda.sync());
    });
    Ok(gflops(shape, time))
}

/// The GFLOPS of a product of `shape` in `time`.
fn gflops((m, n, k): Shape, time: Duration) -> f64 {
    (2 * m * n * k) as f64 / time.as_secs_f64() / 1e9
}
