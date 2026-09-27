//! IQ2_S repacked to signed level nibbles once when the weights are loaded.

use crate::{Block, supported};

/// Row, superblocks, activation quants and activation scales.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32) -> f32;

#[kgen_macros::kernel(iq2_s_nibble(ahead = 1))]
unsafe extern "sysv64" fn ahead1(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq2_s_nibble(ahead = 2))]
unsafe extern "sysv64" fn ahead2(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq2_s_nibble(ahead = 3))]
unsafe extern "sysv64" fn ahead3(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

/// The three instruction schedules.
pub const KERNELS: [(&str, Kernel); 3] = [("ahead1", ahead1), ("ahead2", ahead2), ("ahead3", ahead3)];
/// Bytes per 256 weights after repacking.
pub const SUPERBLOCK: usize = 138;

/// Run a generated kernel on checked slices.
pub fn dot(kernel: Kernel, row: &[u8], quants: &[Block], scales: &[f32]) -> f32 {
    assert_eq!(row.len() % SUPERBLOCK, 0);
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 8 && scales.len() >= sbs * 8);
    assert!(supported());
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr()) }
}

/// Keep the half and group scales, replacing grid indices and signs with level codes.
pub fn repack(original: &[u8], grid: &[u64; 1024]) -> Vec<u8> {
    assert_eq!(original.len() % 82, 0);
    let mut out = vec![0u8; original.len() / 82 * SUPERBLOCK];
    for (src, dst) in original.as_chunks::<82>().0.iter().zip(out.as_chunks_mut::<SUPERBLOCK>().0) {
        dst[..2].copy_from_slice(&src[..2]);
        dst[2..10].copy_from_slice(&src[74..82]);
        for g in 0..8 {
            let signs = u32::from_le_bytes(src[34 + 4 * g..][..4].try_into().unwrap());
            let code = |j: usize| {
                let entry = j / 8;
                let high = ((usize::from(src[66 + g]) >> (2 * entry)) & 3) << 8;
                let gridrow = grid[usize::from(src[2 + 4 * g + entry]) | high];
                let level = (gridrow >> (8 * (j % 8))) as u8;
                let magnitude = match level {
                    8 => 0,
                    25 => 1,
                    43 => 2,
                    _ => panic!("IQ2_S level {level}"),
                };
                magnitude | (((signs >> j & 1) as u8) << 3)
            };
            for j in 0..16 {
                dst[10 + 16 * g + j] = code(j) | code(j + 16) << 4;
            }
        }
    }
    out
}
