//! The IQ2_XS packed kernels against rows of 2560 weights in cache, one CPU.

use std::hint::black_box;
use std::time::Instant;

use kgen_harness::Block;
use kgen_harness::iq2_xs_packed::{self, KERNELS, PAIRS, SUPERBLOCK};

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

fn main() {
    let (rows, cols, secs): (usize, usize, f64) = (env("ROWS", 64), env("COLS", 2560), env("SECS", 3.0));
    let only = std::env::var("KERNEL").unwrap_or_default();
    if only == "?" {
        for (name, _) in KERNELS {
            println!("{name}");
        }
        for (name, _) in PAIRS {
            println!("{name}");
        }
        return;
    }
    assert_eq!(cols % 256, 0);
    let sbs = cols / 256;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut byte = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 24) as u8
    };
    let grid: [u64; 512] =
        std::array::from_fn(|_| u64::from_le_bytes(std::array::from_fn(|_| [8, 25, 43][byte() as usize % 3])));
    let mut file = vec![0u8; rows * sbs * 74];
    for (i, b) in file.iter_mut().enumerate() {
        *b = match i % 74 {
            0 => 0x00,
            1 => 0x30,
            _ => byte(),
        };
    }
    let weights = iq2_xs_packed::repack(&file);
    assert_eq!(weights.len(), rows * sbs * SUPERBLOCK);
    let quants: Vec<Block> = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| byte() as i8))).collect();
    let scales: Vec<f32> = (0..sbs * 8).map(|i| 0.01 + i as f32 * 1e-4).collect();
    for (name, kernel) in KERNELS {
        if !only.is_empty() && name != only {
            continue;
        }
        let (start, mut passes) = (Instant::now(), 0u64);
        let mut check = 0.0f32;
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in weights.chunks_exact(sbs * SUPERBLOCK) {
                    check += iq2_xs_packed::dot(kernel, black_box(row), &quants, &scales, &grid);
                }
            }
            passes += 64;
        }
        let took = start.elapsed().as_secs_f64();
        let done = passes * (rows * cols) as u64;
        println!(
            "iq2_xs {name}: {rows} rows of {cols}, {:.2} G weights/s, {:.1} ns a row, {done} weights (check {check})",
            done as f64 / took / 1e9,
            took * 1e9 / (passes as f64 * rows as f64)
        );
    }
    let pair_quants = kgen_harness::interleave(&quants, &quants.iter().rev().copied().collect::<Vec<_>>());
    let pair_scales = kgen_harness::interleave(&scales, &scales.iter().rev().copied().collect::<Vec<_>>());
    for (name, kernel) in PAIRS {
        if !only.is_empty() && name != only {
            continue;
        }
        let (start, mut passes) = (Instant::now(), 0u64);
        let mut check = 0.0f32;
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in weights.chunks_exact(sbs * SUPERBLOCK) {
                    let [a, b] = iq2_xs_packed::dot2(kernel, black_box(row), &pair_quants, &pair_scales, &grid);
                    check += a + b;
                }
            }
            passes += 64;
        }
        let took = start.elapsed().as_secs_f64();
        let done = passes * (rows * cols) as u64;
        println!(
            "iq2_xs {name}: {rows} rows of {cols}, {:.2} G weights/s, {:.1} ns a row, {done} weights (check {check})",
            done as f64 / took / 1e9,
            took * 1e9 / (passes as f64 * rows as f64)
        );
    }
}
