//! IQ4_XS's generated row kernels against one vector and against two, on the same rows; the
//! rate counts row weights, whatever the vectors. `./make.py cycles SPEED_EXAMPLE=iq4xs_speed`.

use std::hint::black_box;
use std::time::Instant;

use kgen_harness::{Block, interleave, iq4_xs};

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

fn main() {
    let (rows, cols, secs): (usize, usize, f64) = (env("ROWS", 64), env("COLS", 2560), env("SECS", 3.0));
    assert_eq!(cols % 256, 0);
    let sbs = cols / 256;
    let mut seed = 0x52d1_7a0c_e936_48b5u64;
    let mut byte = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 24) as u8
    };
    let mut weights = vec![0u8; rows * sbs * iq4_xs::SUPERBLOCK];
    for (i, b) in weights.iter_mut().enumerate() {
        *b = match i % iq4_xs::SUPERBLOCK {
            0 => 0,
            1 => 0x30,
            _ => byte(),
        };
    }
    let quants: Vec<Block> = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| byte() as i8))).collect();
    let scales: Vec<f32> = (0..sbs * 8).map(|i| 0.01 + i as f32 * 1e-4).collect();
    let pair_quants = interleave(&quants, &quants.iter().rev().copied().collect::<Vec<_>>());
    let pair_scales = interleave(&scales, &scales.iter().rev().copied().collect::<Vec<_>>());
    let only = std::env::var("KERNEL").unwrap_or_default();
    if only == "?" {
        for (name, _) in iq4_xs::KERNELS {
            println!("{name}");
        }
        for (name, _) in iq4_xs::PAIRS {
            println!("{name}");
        }
        return;
    }
    let stride = sbs * iq4_xs::SUPERBLOCK;
    for (name, kernel) in iq4_xs::KERNELS {
        if !only.is_empty() && only != name {
            continue;
        }
        let (start, mut passes, mut check) = (Instant::now(), 0u64, 0.0f32);
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in weights.chunks_exact(stride) {
                    check += iq4_xs::dot(kernel, black_box(row), &quants, &scales);
                }
            }
            passes += 64;
        }
        report(name, rows, cols, passes, start.elapsed().as_secs_f64(), check);
    }
    for (name, kernel) in iq4_xs::PAIRS {
        if !only.is_empty() && only != name {
            continue;
        }
        let (start, mut passes, mut check) = (Instant::now(), 0u64, 0.0f32);
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in weights.chunks_exact(stride) {
                    let [a, b] = iq4_xs::dot2(kernel, black_box(row), &pair_quants, &pair_scales);
                    check += a + b;
                }
            }
            passes += 64;
        }
        report(name, rows, cols, passes, start.elapsed().as_secs_f64(), check);
    }
}

fn report(name: &str, rows: usize, cols: usize, passes: u64, seconds: f64, check: f32) {
    let weights = passes * (rows * cols) as u64;
    println!(
        "iq4_xs {name}: {rows} rows of {cols}, {:.2} G weights/s, {:.1} ns a row, {weights} weights  (check {check})",
        weights as f64 / seconds / 1e9,
        seconds * 1e9 / (passes * rows as u64) as f64
    );
}
