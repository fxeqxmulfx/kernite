//! IQ3_XXS rows repacked once to signed four-bit level codes.

use crate::{Block, supported};

/// Row, superblocks, activation quants and activation scales.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32) -> f32;

#[kgen_macros::kernel(iq3_xxs_nibble(ahead = 1))]
unsafe extern "sysv64" fn ahead1(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq3_xxs_nibble(ahead = 2))]
unsafe extern "sysv64" fn ahead2(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq3_xxs_nibble(ahead = 3))]
unsafe extern "sysv64" fn ahead3(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

/// Variants with one, two or three groups' products started ahead of their sums.
pub const KERNELS: [(&str, Kernel); 3] = [("ahead1", ahead1), ("ahead2", ahead2), ("ahead3", ahead3)];
/// Bytes per superblock of 256 weights after repacking.
pub const SUPERBLOCK: usize = 134;

/// Run a generated kernel on checked slices.
pub fn dot(kernel: Kernel, row: &[u8], quants: &[Block], scales: &[f32]) -> f32 {
    assert_eq!(row.len() % SUPERBLOCK, 0, "whole superblocks");
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 8 && scales.len() >= sbs * 8);
    assert!(supported(), "AVX2, FMA and F16C");
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr()) }
}

/// Keep the half, collect each group's scale nibble, and put signed level codes in the same
/// order as IQ3_S's nibble layout. The grid's eight possible magnitudes are 4, 12, ..., 52, 62.
pub fn repack(original: &[u8], grid: &[u32; 256]) -> Vec<u8> {
    assert_eq!(original.len() % 98, 0, "whole IQ3_XXS superblocks");
    let mut out = vec![0u8; original.len() / 98 * SUPERBLOCK];
    for (src, dst) in original.as_chunks::<98>().0.iter().zip(out.as_chunks_mut::<SUPERBLOCK>().0) {
        dst[..2].copy_from_slice(&src[..2]);
        let mut scales = 0u32;
        for g in 0..8 {
            let word = u32::from_le_bytes(src[66 + 4 * g..][..4].try_into().unwrap());
            scales |= (word >> 28) << (4 * g);
            let code = |j: usize| {
                let level = (grid[src[2 + 8 * g + j / 4] as usize] >> (8 * (j % 4))) as u8;
                assert!(matches!(level, 4 | 12 | 20 | 28 | 36 | 44 | 52 | 62), "IQ3_XXS level {level}");
                let field = word >> (7 * (j / 8)) & 127;
                let signs = field | (field.count_ones() & 1) << 7;
                (level >> 3) | (((signs >> (j % 8)) as u8 & 1) << 3)
            };
            for i in 0..16 {
                dst[6 + 16 * g + i] = code(i) | code(i + 16) << 4;
            }
        }
        dst[2..6].copy_from_slice(&scales.to_le_bytes());
    }
    out
}
