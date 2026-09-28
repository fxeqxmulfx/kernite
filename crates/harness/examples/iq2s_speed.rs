//! Each generated `IQ2_S` kernel on its own, against one vector, over rows few enough to stay
//! in the caches: `ROWS` rows of `COLS` (64 of 2560 by default, as `examples/iqdot.rs` in
//! `roofline` runs them), again and again for `SECS` each. `KERNEL` picks one by name, and `KERNEL=?` lists them. Run it
//! pinned:
//!
//!     taskset -c 0 target/release/examples/speed

use std::hint::black_box;
use std::time::Instant;

use kgen_harness::Block;
use kgen_harness::iq2_s::{self, SUPERBLOCK};
use kgen_harness::{iq2_s_nibble, iq2_s_packed, iq2_s_planes};

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

fn main() {
    let (rows, cols, secs): (usize, usize, f64) = (env("ROWS", 64), env("COLS", 2560), env("SECS", 3.0));
    let only = std::env::var("KERNEL").unwrap_or_default();
    if only == "?" {
        for (name, _) in iq2_s::KERNELS.into_iter().chain(iq2_s_packed::KERNELS) {
            println!("{name}");
        }
        for (name, _) in iq2_s_nibble::KERNELS {
            println!("nibble-{name}");
        }
        for (name, _) in iq2_s_planes::KERNELS {
            println!("planes-{name}");
        }
        return;
    }
    assert_eq!(cols % 256, 0, "whole superblocks");
    let sbs = cols / 256;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut byte = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 24) as u8
    };
    let grid: [u64; 1024] =
        std::array::from_fn(|_| u64::from_le_bytes(std::array::from_fn(|_| [8, 25, 43][byte() as usize % 3])));
    let mut weights = vec![0u8; rows * sbs * SUPERBLOCK];
    for (i, b) in weights.iter_mut().enumerate() {
        // A small positive scale in every superblock's first two bytes, noise after.
        *b = match i % SUPERBLOCK {
            0 => 0x00,
            1 => 0x30,
            _ => byte(),
        };
    }
    let quants: Vec<Block> = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| byte() as i8))).collect();
    let scales: Vec<f32> = (0..sbs * 8).map(|i| 0.01 + i as f32 * 1e-4).collect();
    let packed = iq2_s_packed::repack(&weights);
    let nibble = iq2_s_nibble::repack(&weights, &grid);
    let planes = iq2_s_planes::repack(&weights, &grid);
    for (name, kernel) in iq2_s::KERNELS {
        if !only.is_empty() && name != only {
            continue;
        }
        let (start, mut passes) = (Instant::now(), 0u64);
        let mut check = 0.0f32;
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in weights.chunks_exact(sbs * SUPERBLOCK) {
                    check += iq2_s::dot(kernel, black_box(row), &quants, &scales, &grid);
                }
            }
            passes += 64;
        }
        let took = start.elapsed().as_secs_f64();
        let done = passes * (rows * cols) as u64;
        println!(
            "iq2_s {name}: {rows} rows of {cols}, {:.2} G weights/s, {:.1} ns a row, {done} weights  (check {check})",
            done as f64 / took / 1e9,
            took * 1e9 / (passes as f64 * rows as f64)
        );
    }
    for (name, kernel) in iq2_s_packed::KERNELS {
        if !only.is_empty() && name != only {
            continue;
        }
        let (start, mut passes) = (Instant::now(), 0u64);
        let mut check = 0.0f32;
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in packed.chunks_exact(sbs * SUPERBLOCK) {
                    check += iq2_s_packed::dot(kernel, black_box(row), &quants, &scales, &grid);
                }
            }
            passes += 64;
        }
        let took = start.elapsed().as_secs_f64();
        let done = passes * (rows * cols) as u64;
        println!(
            "iq2_s {name}: {rows} rows of {cols}, {:.2} G weights/s, {:.1} ns a row, {done} weights  (check {check})",
            done as f64 / took / 1e9,
            took * 1e9 / (passes as f64 * rows as f64)
        );
    }
    for (kind, data, row_bytes, kernels, dot) in [
        (
            "nibble",
            &nibble,
            sbs * iq2_s_nibble::SUPERBLOCK,
            &iq2_s_nibble::KERNELS[..],
            iq2_s_nibble::dot as fn(iq2_s_nibble::Kernel, &[u8], &[Block], &[f32]) -> f32,
        ),
        (
            "planes",
            &planes,
            sbs * iq2_s_planes::SUPERBLOCK,
            &iq2_s_planes::KERNELS[..],
            iq2_s_planes::dot as fn(iq2_s_planes::Kernel, &[u8], &[Block], &[f32]) -> f32,
        ),
    ] {
        for &(name, kernel) in kernels {
            let label = format!("{kind}-{name}");
            if !only.is_empty() && label != only {
                continue;
            }
            let (start, mut passes) = (Instant::now(), 0u64);
            let mut check = 0.0f32;
            while start.elapsed().as_secs_f64() < secs {
                for _ in 0..64 {
                    for row in data.chunks_exact(row_bytes) {
                        check += dot(kernel, black_box(row), &quants, &scales);
                    }
                }
                passes += 64;
            }
            let took = start.elapsed().as_secs_f64();
            let done = passes * (rows * cols) as u64;
            println!(
                "iq2_s {label}: {cols} columns, {:.2} G weights/s, {:.1} ns a row, {done} weights (check {check})",
                done as f64 / took / 1e9,
                took * 1e9 / (passes as f64 * rows as f64),
            );
        }
    }
}
