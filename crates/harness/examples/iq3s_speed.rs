//! IQ3_S's original grid kernel against the one-time nibble repack, on the same rows, and the
//! repack against two vectors at once; the rate counts row weights, whatever the vectors.
//! `ROWS`, `COLS`, `SECS` and `KERNEL` select the case; run pinned after moving other
//! user processes off the measured core with `../make.py quiet`.

use std::hint::black_box;
use std::time::Instant;

use kgen_harness::Block;
use kgen_harness::{iq3_s, iq3_s_nibble};

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

fn main() {
    let (rows, cols, secs): (usize, usize, f64) = (env("ROWS", 64), env("COLS", 2560), env("SECS", 3.0));
    assert_eq!(cols % 256, 0);
    let sbs = cols / 256;
    let mut seed = 0x713e_a8f3_eacb_4671u64;
    let mut byte = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 24) as u8
    };
    let grid: [u32; 512] = std::array::from_fn(|_| u32::from_le_bytes(std::array::from_fn(|_| 1 + 2 * (byte() % 8))));
    let mut original = vec![0u8; rows * sbs * iq3_s::SUPERBLOCK];
    for (i, b) in original.iter_mut().enumerate() {
        *b = match i % iq3_s::SUPERBLOCK {
            0 => 0,
            1 => 0x30,
            _ => byte(),
        };
    }
    let packed = iq3_s_nibble::repack(&original, &grid);
    let quants: Vec<Block> = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| byte() as i8))).collect();
    let scales: Vec<f32> = (0..sbs * 8).map(|i| 0.01 + i as f32 * 1e-4).collect();
    let only = std::env::var("KERNEL").unwrap_or_default();
    if only == "?" {
        for (name, _) in iq3_s::KERNELS {
            println!("grid-{name}");
        }
        for (name, _) in iq3_s_nibble::KERNELS {
            println!("nibble-{name}");
        }
        for (name, _) in iq3_s_nibble::PAIRS {
            println!("nibble-{name}");
        }
        return;
    }
    for (name, kernel) in iq3_s::KERNELS {
        let label = format!("grid-{name}");
        if !only.is_empty() && only != label {
            continue;
        }
        let (start, mut passes) = (Instant::now(), 0u64);
        let mut check = 0.0f32;
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in original.chunks_exact(sbs * iq3_s::SUPERBLOCK) {
                    check += iq3_s::dot(kernel, black_box(row), &quants, &scales, &grid);
                }
            }
            passes += 64;
        }
        report(&label, rows, cols, passes, start.elapsed().as_secs_f64(), check);
    }
    for (name, kernel) in iq3_s_nibble::KERNELS {
        let label = format!("nibble-{name}");
        if !only.is_empty() && only != label {
            continue;
        }
        let (start, mut passes) = (Instant::now(), 0u64);
        let mut check = 0.0f32;
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in packed.chunks_exact(sbs * iq3_s_nibble::SUPERBLOCK) {
                    check += iq3_s_nibble::dot(kernel, black_box(row), &quants, &scales);
                }
            }
            passes += 64;
        }
        report(&label, rows, cols, passes, start.elapsed().as_secs_f64(), check);
    }
    let pair_quants = iq3_s_nibble::interleave(&quants, &quants.iter().rev().copied().collect::<Vec<_>>());
    let pair_scales = iq3_s_nibble::interleave(&scales, &scales.iter().rev().copied().collect::<Vec<_>>());
    for (name, kernel) in iq3_s_nibble::PAIRS {
        let label = format!("nibble-{name}");
        if !only.is_empty() && only != label {
            continue;
        }
        let (start, mut passes) = (Instant::now(), 0u64);
        let mut check = 0.0f32;
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in packed.chunks_exact(sbs * iq3_s_nibble::SUPERBLOCK) {
                    let [a, b] = iq3_s_nibble::dot2(kernel, black_box(row), &pair_quants, &pair_scales);
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
        "iq3_s {name}: {rows} rows of {cols}, {:.2} G weights/s, {:.1} ns a row, {weights} weights  (check {check})",
        weights as f64 / seconds / 1e9,
        seconds * 1e9 / (passes * rows as u64) as f64
    );
}
