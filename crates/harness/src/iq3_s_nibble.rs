//! The IQ3_S kernel on a row repacked to signed odd-level nibbles.

use crate::{Block, supported};

/// Row, superblocks, activation quants, activation scales.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32) -> f32;

#[kgen_macros::kernel(iq3_s_nibble(ahead = 1))]
unsafe extern "sysv64" fn ahead1(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq3_s_nibble(ahead = 2))]
unsafe extern "sysv64" fn ahead2(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq3_s_nibble(ahead = 3))]
unsafe extern "sysv64" fn ahead3(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

/// Variants to compare by name.
pub const KERNELS: [(&str, Kernel); 3] = [("ahead1", ahead1), ("ahead2", ahead2), ("ahead3", ahead3)];
/// Bytes per group of 256 weights after repacking.
pub const SUPERBLOCK: usize = 134;

/// Run a kernel on checked slices.
pub fn dot(kernel: Kernel, row: &[u8], quants: &[Block], scales: &[f32]) -> f32 {
    assert_eq!(row.len() % SUPERBLOCK, 0, "whole superblocks");
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 8 && scales.len() >= sbs * 8, "activations for every group");
    assert!(supported(), "AVX2, FMA and F16C");
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr()) }
}

/// Row, superblocks, two vectors' quants and scales stored together block by block, and the
/// two dots' destination.
pub type Pair = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32, *mut f32);

#[kgen_macros::kernel(iq3_s_nibble2(ahead = 1))]
unsafe extern "sysv64" fn pair_ahead1(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, out: *mut f32);

#[kgen_macros::kernel(iq3_s_nibble2(ahead = 2))]
unsafe extern "sysv64" fn pair_ahead2(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, out: *mut f32);

#[kgen_macros::kernel(iq3_s_nibble2(ahead = 3))]
unsafe extern "sysv64" fn pair_ahead3(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, out: *mut f32);

/// Two-vector variants to compare by name.
pub const PAIRS: [(&str, Pair); 3] =
    [("pair-ahead1", pair_ahead1), ("pair-ahead2", pair_ahead2), ("pair-ahead3", pair_ahead3)];

/// Run a two-vector kernel on checked slices, which hold both vectors' activations as
/// [`crate::interleave`] lays them out.
pub fn dot2(kernel: Pair, row: &[u8], quants: &[Block], scales: &[f32]) -> [f32; 2] {
    assert_eq!(row.len() % SUPERBLOCK, 0, "whole superblocks");
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 16 && scales.len() >= sbs * 16, "both vectors' activations for every group");
    assert!(supported(), "AVX2, FMA and F16C");
    let mut out = [0.0; 2];
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr(), out.as_mut_ptr()) };
    out
}

/// Repack the original IQ3_S row once, keeping its half and four scale bytes. Each
/// codebook level must be one of the format's positive odd values 1..15.
pub fn repack(original: &[u8], grid: &[u32; 512]) -> Vec<u8> {
    assert_eq!(original.len() % 110, 0, "whole IQ3_S superblocks");
    let mut out = vec![0u8; original.len() / 110 * SUPERBLOCK];
    for (src, dst) in original.as_chunks::<110>().0.iter().zip(out.as_chunks_mut::<SUPERBLOCK>().0) {
        dst[..2].copy_from_slice(&src[..2]);
        dst[2..6].copy_from_slice(&src[106..110]);
        for g in 0..8 {
            let signs = u32::from_le_bytes(src[74 + 4 * g..][..4].try_into().unwrap());
            let code = |j: usize| {
                let entry = j / 4;
                let high = usize::from(src[66 + g] >> entry & 1) << 8;
                let level = (grid[src[2 + 8 * g + entry] as usize | high] >> (8 * (j % 4))) as u8;
                assert!(level <= 15 && level & 1 == 1, "IQ3_S levels are odd 1..15");
                ((level - 1) / 2) | (((signs >> j & 1) as u8) << 3)
            };
            for i in 0..16 {
                dst[6 + 16 * g + i] = code(i) | code(16 + i) << 4;
            }
        }
    }
    out
}
