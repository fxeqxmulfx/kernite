//! The Q6_K file and front-scale layouts against one Q8 activation vector.

use std::hint::black_box;
use std::time::Instant;

use kgen_harness::{Block, q6_k};

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

fn main() {
    let (rows, cols, secs): (usize, usize, f64) = (env("ROWS", 64), env("COLS", 2560), env("SECS", 2.0));
    assert_eq!(cols % 256, 0);
    let sbs = cols / 256;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut byte = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 24) as u8
    };
    let mut file = vec![0u8; rows * sbs * q6_k::SUPERBLOCK];
    for block in file.as_chunks_mut::<{ q6_k::SUPERBLOCK }>().0 {
        for b in &mut block[..208] {
            *b = byte();
        }
        block[208..210].copy_from_slice(&0x3000u16.to_le_bytes());
    }
    let front = q6_k::repack_front(&file);
    let quants: Vec<Block> = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| byte() as i8))).collect();
    let scales: Vec<f32> = (0..sbs * 8).map(|i| 0.01 + i as f32 * 1e-4).collect();
    let sums: Vec<f32> = (0..sbs * 16).map(|i| -0.01 + i as f32 * 1e-4).collect();
    for (name, weights, stride, dot) in [
        ("file", &file, sbs * q6_k::SUPERBLOCK, q6_k::dot as fn(&[u8], &[Block], &[f32], &[f32]) -> f32),
        ("front", &front, sbs * q6_k::FRONT_SUPERBLOCK, q6_k::dot_front as fn(&[u8], &[Block], &[f32], &[f32]) -> f32),
    ] {
        let (start, mut passes, mut check) = (Instant::now(), 0u64, 0.0f32);
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..64 {
                for row in weights.chunks_exact(stride) {
                    check += dot(black_box(row), &quants, &scales, &sums);
                }
            }
            passes += 64;
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "q6_k {name}: {rows} rows of {cols}, {:.2} G weights/s, {:.1} ns a row, {passes} passes (check {check})",
            passes as f64 * (rows * cols) as f64 / elapsed / 1e9,
            elapsed * 1e9 / (passes as f64 * rows as f64),
        );
    }
}
