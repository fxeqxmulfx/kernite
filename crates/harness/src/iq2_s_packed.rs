//! IQ2_S with four 10-bit indices packed into five consecutive bytes per group.

use crate::{Block, supported};

/// Row, superblock count, activation quants and scales, and the 1024-entry grid.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32, *const u64) -> f32;

#[kgen_macros::kernel(iq2_s_packed(ahead = 1))]
unsafe extern "sysv64" fn ahead1(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_s_packed(ahead = 2))]
unsafe extern "sysv64" fn ahead2(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_s_packed(ahead = 3))]
unsafe extern "sysv64" fn ahead3(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

/// The scheduled variants to compare.
pub const KERNELS: [(&str, Kernel); 3] = [("packed1", ahead1), ("packed2", ahead2), ("packed3", ahead3)];
/// Bytes per 256 weights, unchanged from the file.
pub const SUPERBLOCK: usize = 82;

/// Run a generated kernel on checked slices.
pub fn dot(kernel: Kernel, row: &[u8], quants: &[Block], scales: &[f32], grid: &[u64; 1024]) -> f32 {
    assert_eq!(row.len() % SUPERBLOCK, 0, "whole superblocks");
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 8 && scales.len() >= sbs * 8, "activations for every group");
    assert!(supported(), "AVX2, FMA and F16C");
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr(), grid.as_ptr()) }
}

/// Row, superblocks, two vectors' quants and scales stored together block by block, the grid,
/// and the two dots' destination.
pub type Pair = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32, *const u64, *mut f32);

#[kgen_macros::kernel(iq2_s_packed2(ahead = 1))]
unsafe extern "sysv64" fn pair_ahead1(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
    out: *mut f32,
);

#[kgen_macros::kernel(iq2_s_packed2(ahead = 2))]
unsafe extern "sysv64" fn pair_ahead2(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
    out: *mut f32,
);

/// Two-vector variants to compare by name.
pub const PAIRS: [(&str, Pair); 2] = [("pair-ahead1", pair_ahead1), ("pair-ahead2", pair_ahead2)];

/// Run a two-vector kernel on checked slices, which hold both vectors' activations as
/// [`crate::interleave`] lays them out.
pub fn dot2(kernel: Pair, row: &[u8], quants: &[Block], scales: &[f32], grid: &[u64; 1024]) -> [f32; 2] {
    assert_eq!(row.len() % SUPERBLOCK, 0, "whole superblocks");
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 16 && scales.len() >= sbs * 16, "both vectors' activations for every group");
    assert!(supported(), "AVX2, FMA and F16C");
    let mut out = [0.0; 2];
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr(), grid.as_ptr(), out.as_mut_ptr()) };
    out
}

/// Rearrange the 32 low bytes and eight high-index bytes into eight groups of five bytes.
/// Scales stay at byte 74; signs move from byte 34 to 42. Nothing else changes.
pub fn repack(original: &[u8]) -> Vec<u8> {
    assert_eq!(original.len() % SUPERBLOCK, 0);
    let mut out = vec![0u8; original.len()];
    for (src, dst) in original.as_chunks::<SUPERBLOCK>().0.iter().zip(out.as_chunks_mut::<SUPERBLOCK>().0) {
        dst[..2].copy_from_slice(&src[..2]);
        for g in 0..8 {
            let high = src[66 + g];
            let mut indices = 0u64;
            for l in 0..4 {
                let index = u64::from(src[2 + 4 * g + l]) | (u64::from((high >> (2 * l)) & 3) << 8);
                indices |= index << (10 * l);
            }
            dst[2 + 5 * g..][..5].copy_from_slice(&indices.to_le_bytes()[..5]);
        }
        dst[42..74].copy_from_slice(&src[34..66]);
        dst[74..82].copy_from_slice(&src[74..82]);
    }
    out
}
