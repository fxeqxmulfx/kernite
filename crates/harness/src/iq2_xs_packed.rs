//! IQ2_XS with compact grid indices and expanded sign bytes.

use crate::{Block, half, supported};

/// Row, superblocks, activation quants and scales, and the 512-entry grid.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32, *const u64) -> f32;

#[kgen_macros::kernel(iq2_xs_packed(ahead = 1))]
unsafe extern "sysv64" fn ahead1(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_xs_packed(ahead = 2))]
unsafe extern "sysv64" fn ahead2(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_xs_packed(ahead = 3))]
unsafe extern "sysv64" fn ahead3(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

/// Schedules for the same repacked row.
pub const KERNELS: [(&str, Kernel); 3] = [("packed1", ahead1), ("packed2", ahead2), ("packed3", ahead3)];
/// Bytes per 256 weights, four more than in the file.
pub const SUPERBLOCK: usize = 78;

/// Run a generated kernel on checked slices.
pub fn dot(kernel: Kernel, row: &[u8], quants: &[Block], scales: &[f32], grid: &[u64; 512]) -> f32 {
    assert_eq!(row.len() % SUPERBLOCK, 0);
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 8 && scales.len() >= sbs * 8);
    assert!(supported());
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr(), grid.as_ptr()) }
}

/// Store eight 9-bit indices in nine bytes per pair of groups and expand each 7-bit sign field
/// to its eight sign bits, including the odd-parity eighth bit.
pub fn repack(original: &[u8]) -> Vec<u8> {
    assert_eq!(original.len() % 74, 0);
    let mut out = vec![0u8; original.len() / 74 * SUPERBLOCK];
    for (src, dst) in original.as_chunks::<74>().0.iter().zip(out.as_chunks_mut::<SUPERBLOCK>().0) {
        dst[..2].copy_from_slice(&src[..2]);
        for p in 0..4 {
            let mut indices = 0u128;
            for j in 0..8 {
                let (g, l) = (2 * p + j / 4, j % 4);
                let at = 2 + 8 * g + 2 * l;
                let word = u16::from_le_bytes([src[at], src[at + 1]]);
                indices |= u128::from(word & 511) << (9 * j);
                let sign7 = (word >> 9) as u8;
                dst[38 + 4 * g + l] = sign7 | (((sign7.count_ones() & 1) as u8) << 7);
            }
            dst[2 + 9 * p..][..9].copy_from_slice(&indices.to_le_bytes()[..9]);
        }
        dst[70..78].copy_from_slice(&src[66..74]);
    }
    out
}

/// The file layout's pairwise products and FMA order, including sign parity.
pub fn reference(row: &[u8], quants: &[Block], scales: &[f32], grid: &[u64; 512]) -> f32 {
    assert_eq!(row.len() % 74, 0);
    if row.is_empty() {
        return 0.0;
    }
    let mut acc = [0.0f32; 8];
    for (sb, blk) in row.as_chunks::<74>().0.iter().enumerate() {
        let d = half(u16::from_le_bytes([blk[0], blk[1]]));
        for g in 0..8 {
            let packed = blk[66 + g];
            let half_scale = |h: usize| {
                let nibble = if h == 0 { packed & 15 } else { packed >> 4 };
                ((d * (2 * u32::from(nibble) + 1) as f32) * 0.125) * scales[8 * sb + g]
            };
            let x = &quants[8 * sb + g].0;
            let weight_sign = |j: usize| {
                let entry = j / 8;
                let at = 2 + 8 * g + 2 * entry;
                let word = u16::from_le_bytes([blk[at], blk[at + 1]]);
                let signs = (word >> 9) as u8;
                let bit = if j % 8 == 7 { signs.count_ones() & 1 } else { u32::from(signs >> (j % 8) & 1) };
                let level = (grid[(word & 511) as usize] >> (8 * (j % 8))) as u8;
                (level, bit)
            };
            let pair = |m: usize| {
                let sum = (0..2)
                    .map(|b| {
                        let j = 2 * m + b;
                        let (v, sign) = weight_sign(j);
                        let q = if sign == 1 { x[j].wrapping_neg() } else { x[j] };
                        i32::from(v) * i32::from(q)
                    })
                    .sum::<i32>();
                sum.clamp(i16::MIN.into(), i16::MAX.into())
            };
            for (lane, a) in acc.iter_mut().enumerate() {
                let lanes = pair(2 * lane) + pair(2 * lane + 1);
                *a = (lanes as f32).mul_add(half_scale(lane / 4), *a);
            }
        }
    }
    let v = acc.map(|a| a + 0.0);
    let s = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
    (s[0] + s[2]) + (s[1] + s[3])
}
