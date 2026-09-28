//! IQ4_XS rows as the file has them, against one or two vectors of Q8 activations.

use crate::{Block, half, supported};

/// Row, superblocks, activation quants, activation scales.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32) -> f32;
/// Row, superblocks, two vectors' quants and scales stored together block by block, and the
/// two dots' destination.
pub type Pair = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32, *mut f32);

#[kgen_macros::kernel(iq4_xs(ahead = 1))]
unsafe extern "sysv64" fn ahead1(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq4_xs(ahead = 2))]
unsafe extern "sysv64" fn ahead2(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq4_xs(ahead = 3))]
unsafe extern "sysv64" fn ahead3(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq4_xs2(ahead = 1))]
unsafe extern "sysv64" fn pair_ahead1(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, out: *mut f32);

#[kgen_macros::kernel(iq4_xs2(ahead = 2))]
unsafe extern "sysv64" fn pair_ahead2(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, out: *mut f32);

#[kgen_macros::kernel(iq4_xs2(ahead = 3))]
unsafe extern "sysv64" fn pair_ahead3(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, out: *mut f32);

/// One-vector variants to compare by name.
pub const KERNELS: [(&str, Kernel); 3] = [("ahead1", ahead1), ("ahead2", ahead2), ("ahead3", ahead3)];
/// Two-vector variants to compare by name.
pub const PAIRS: [(&str, Pair); 3] =
    [("pair-ahead1", pair_ahead1), ("pair-ahead2", pair_ahead2), ("pair-ahead3", pair_ahead3)];
/// Bytes per superblock of 256 weights.
pub const SUPERBLOCK: usize = 136;
/// The level each four-bit code names.
pub const LEVELS: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// Run a one-vector kernel on checked slices.
pub fn dot(kernel: Kernel, row: &[u8], quants: &[Block], scales: &[f32]) -> f32 {
    assert_eq!(row.len() % SUPERBLOCK, 0, "whole superblocks");
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 8 && scales.len() >= sbs * 8, "activations for every group");
    assert!(supported(), "AVX2, FMA and F16C");
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr()) }
}

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

/// The saturated pairs, scale products, FMA order and horizontal sum of the original kernel:
/// group `g`'s scale is `d * (ls - 32)` times its activation scale, `ls` its low nibble at
/// byte `4 + g / 2` with its high two bits at bit `2g` of the word at 2.
pub fn reference(row: &[u8], quants: &[Block], scales: &[f32]) -> f32 {
    if row.is_empty() {
        return 0.0;
    }
    let mut acc = [0.0f32; 8];
    for (sb, blk) in row.as_chunks::<SUPERBLOCK>().0.iter().enumerate() {
        let d = half(u16::from_le_bytes([blk[0], blk[1]]));
        let high = u16::from_le_bytes([blk[2], blk[3]]);
        for g in 0..8 {
            let ls = (blk[4 + g / 2] >> (4 * (g % 2)) & 15) | ((high >> (2 * g) & 3) as u8) << 4;
            let scale = (d * (i32::from(ls) - 32) as f32) * scales[8 * sb + g];
            let x = &quants[8 * sb + g].0;
            let weight = |j: usize| {
                let byte = blk[8 + 16 * g + j % 16];
                LEVELS[if j < 16 { (byte & 15) as usize } else { (byte >> 4) as usize }]
            };
            let pair = |m: usize| {
                let sum = (0..2)
                    .map(|k| {
                        let j = 2 * m + k;
                        let w = weight(j);
                        let signed_x = if w < 0 { x[j].wrapping_neg() } else { x[j] };
                        i32::from(w.unsigned_abs()) * i32::from(signed_x)
                    })
                    .sum::<i32>();
                sum.clamp(i16::MIN.into(), i16::MAX.into())
            };
            for (lane, a) in acc.iter_mut().enumerate() {
                let lanes = pair(2 * lane) + pair(2 * lane + 1);
                *a = (lanes as f32).mul_add(scale, *a);
            }
        }
    }
    let v = acc.map(|a| a + 0.0);
    let s = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
    (s[0] + s[2]) + (s[1] + s[3])
}
