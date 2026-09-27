//! IQ4_NL's pipelined row variants at the routed down projection's width.
//! `./make.py cycles SPEED_EXAMPLE=iq4nl_speed COLS=640` measures a quiet core.

use std::hint::black_box;
use std::time::Instant;

use kgen_harness::{Block, iq4_nl};

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

fn main() {
    let (rows, cols, secs): (usize, usize, f64) = (env("ROWS", 64), env("COLS", 640), env("SECS", 3.0));
    assert_eq!(cols % 128, 0);
    let mut seed = 0x917c_298e_3a44_108fu64;
    let mut byte = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 24) as u8
    };
    let stride = cols / 128 * iq4_nl::GROUP;
    let mut weights = vec![0u8; rows * stride];
    for (i, b) in weights.iter_mut().enumerate() {
        *b = match i % 18 {
            0 => 0,
            1 => 0x30,
            _ => byte(),
        };
    }
    let quants: Vec<Block> = (0..cols / 32).map(|_| Block(std::array::from_fn(|_| byte() as i8))).collect();
    let scales: Vec<f32> = (0..cols / 32).map(|i| 0.01 + i as f32 * 1e-4).collect();
    let only = std::env::var("KERNEL").unwrap_or_default();
    if only == "?" {
        for (name, _) in iq4_nl::KERNELS {
            println!("{name}");
        }
        return;
    }
    for (name, kernel) in iq4_nl::KERNELS {
        if !only.is_empty() && only != name {
            continue;
        }
        let (start, mut passes, mut check) = (Instant::now(), 0u64, 0.0f32);
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in weights.chunks_exact(stride) {
                    check += iq4_nl::dot(kernel, black_box(row), &quants, &scales);
                }
            }
            passes += 64;
        }
        let seconds = start.elapsed().as_secs_f64();
        let done = passes * (rows * cols) as u64;
        println!(
            "iq4_nl {name}: {rows} rows of {cols}, {:.2} G weights/s, {:.1} ns a row, {done} weights (check {check})",
            done as f64 / seconds / 1e9,
            seconds * 1e9 / (passes * rows as u64) as f64,
        );
    }
}
