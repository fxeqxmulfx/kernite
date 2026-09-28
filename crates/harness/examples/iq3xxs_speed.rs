//! Compare IQ3_XXS's grid row with its one-time nibble repack on the same data.
//! Run through `./make.py speed SPEED_EXAMPLE=iq3xxs_speed` for a quiet, pinned core.

use std::hint::black_box;
use std::time::Instant;

use kgen_harness::{Block, iq3_xxs, iq3_xxs_nibble};

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

fn main() {
    let (rows, cols, secs): (usize, usize, f64) = (env("ROWS", 64), env("COLS", 2560), env("SECS", 3.0));
    assert_eq!(cols % 256, 0);
    let sbs = cols / 256;
    let mut seed = 0x7124_3bea_2d19_605fu64;
    let mut byte = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 24) as u8
    };
    let levels = [4u8, 12, 20, 28, 36, 44, 52, 62];
    let grid: [u32; 256] =
        std::array::from_fn(|_| u32::from_le_bytes(std::array::from_fn(|_| levels[(byte() & 7) as usize])));
    let mut original = vec![0u8; rows * sbs * iq3_xxs::SUPERBLOCK];
    for (i, b) in original.iter_mut().enumerate() {
        *b = match i % iq3_xxs::SUPERBLOCK {
            0 => 0,
            1 => 0x30,
            _ => byte(),
        };
    }
    let packed = iq3_xxs_nibble::repack(&original, &grid);
    let quants: Vec<Block> = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| byte() as i8))).collect();
    let scales: Vec<f32> = (0..sbs * 8).map(|i| 0.01 + i as f32 * 1e-4).collect();
    let only = std::env::var("KERNEL").unwrap_or_default();
    if only == "?" {
        for (name, _) in iq3_xxs::KERNELS {
            println!("grid-{name}");
        }
        for (name, _) in iq3_xxs_nibble::KERNELS {
            println!("nibble-{name}");
        }
        for (name, _) in iq3_xxs_nibble::PAIRS {
            println!("nibble-{name}");
        }
        return;
    }
    for (name, kernel) in iq3_xxs::KERNELS {
        let label = format!("grid-{name}");
        if !only.is_empty() && only != label {
            continue;
        }
        let start = Instant::now();
        let (mut passes, mut check) = (0u64, 0.0f32);
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in original.chunks_exact(sbs * iq3_xxs::SUPERBLOCK) {
                    check += iq3_xxs::dot(kernel, black_box(row), &quants, &scales, &grid);
                }
            }
            passes += 64;
        }
        report(&label, rows, cols, passes, start.elapsed().as_secs_f64(), check);
    }
    for (name, kernel) in iq3_xxs_nibble::KERNELS {
        let label = format!("nibble-{name}");
        if !only.is_empty() && only != label {
            continue;
        }
        let start = Instant::now();
        let (mut passes, mut check) = (0u64, 0.0f32);
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in packed.chunks_exact(sbs * iq3_xxs_nibble::SUPERBLOCK) {
                    check += iq3_xxs_nibble::dot(kernel, black_box(row), &quants, &scales);
                }
            }
            passes += 64;
        }
        report(&label, rows, cols, passes, start.elapsed().as_secs_f64(), check);
    }
    let pair_quants = kgen_harness::interleave(&quants, &quants.iter().rev().copied().collect::<Vec<_>>());
    let pair_scales = kgen_harness::interleave(&scales, &scales.iter().rev().copied().collect::<Vec<_>>());
    for (name, kernel) in iq3_xxs_nibble::PAIRS {
        let label = format!("nibble-{name}");
        if !only.is_empty() && only != label {
            continue;
        }
        let start = Instant::now();
        let (mut passes, mut check) = (0u64, 0.0f32);
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in packed.chunks_exact(sbs * iq3_xxs_nibble::SUPERBLOCK) {
                    let [a, b] = iq3_xxs_nibble::dot2(kernel, black_box(row), &pair_quants, &pair_scales);
                    check += a + b;
                }
            }
            passes += 64;
        }
        report(&label, rows, cols, passes, start.elapsed().as_secs_f64(), check);
    }
}

fn report(name: &str, rows: usize, cols: usize, passes: u64, seconds: f64, check: f32) {
    let weights = passes * (rows * cols) as u64;
    println!(
        "iq3_xxs {name}: {rows} rows of {cols}, {:.2} G weights/s, {:.1} ns a row, {weights} weights (check {check})",
        weights as f64 / seconds / 1e9,
        seconds * 1e9 / (passes * rows as u64) as f64
    );
}
