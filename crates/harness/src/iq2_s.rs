//! `IQ2_S` rows against one activation vector.

use crate::{Block, half, supported};

/// Row, superblock count, activation quants and scales, and the 1024-entry grid.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32, *const u64) -> f32;

#[kgen_macros::kernel(iq2_s(ahead = 1))]
unsafe extern "sysv64" fn ahead1(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_s(ahead = 2))]
unsafe extern "sysv64" fn ahead2(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_s(ahead = 3))]
unsafe extern "sysv64" fn ahead3(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_s_gather(ahead = 1))]
unsafe extern "sysv64" fn gather1(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_s_gather(ahead = 2))]
unsafe extern "sysv64" fn gather2(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_s_gather(ahead = 3))]
unsafe extern "sysv64" fn gather3(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

/// The scheduled variants to compare.
pub const KERNELS: [(&str, Kernel); 6] = [
    ("ahead1", ahead1),
    ("ahead2", ahead2),
    ("ahead3", ahead3),
    ("gather1", gather1),
    ("gather2", gather2),
    ("gather3", gather3),
];
/// Bytes per 256 weights.
pub const SUPERBLOCK: usize = 82;

/// Run a kernel on checked slices.
pub fn dot(kernel: Kernel, row: &[u8], quants: &[Block], scales: &[f32], grid: &[u64; 1024]) -> f32 {
    assert_eq!(row.len() % SUPERBLOCK, 0, "whole superblocks");
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 8 && scales.len() >= sbs * 8, "activations for every group");
    assert!(supported(), "AVX2, FMA and F16C");
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr(), grid.as_ptr()) }
}

/// The original kernel's saturated pair products, per-half scales and FMA order.
pub fn reference(row: &[u8], quants: &[Block], scales: &[f32], grid: &[u64; 1024]) -> f32 {
    if row.is_empty() {
        return 0.0;
    }
    let mut acc = [0.0f32; 8];
    for (sb, blk) in row.as_chunks::<SUPERBLOCK>().0.iter().enumerate() {
        let d = half(u16::from_le_bytes([blk[0], blk[1]]));
        for g in 0..8 {
            let packed = blk[74 + g];
            let half_scale = |h: usize| {
                let nibble = if h == 0 { packed & 15 } else { packed >> 4 };
                ((d * (2 * u32::from(nibble) + 1) as f32) * 0.125) * scales[8 * sb + g]
            };
            let signs = u32::from_le_bytes(blk[34 + 4 * g..][..4].try_into().unwrap());
            let x = &quants[8 * sb + g].0;
            let weight = |j: usize| {
                let entry = j / 8;
                let high = ((usize::from(blk[66 + g]) >> (2 * entry)) & 3) << 8;
                (grid[usize::from(blk[2 + 4 * g + entry]) | high] >> (8 * (j % 8))) as u8
            };
            let signed = |j: usize| if signs >> j & 1 == 1 { x[j].wrapping_neg() } else { x[j] };
            let pair = |m: usize| {
                let sum = (0..2).map(|b| i32::from(weight(2 * m + b)) * i32::from(signed(2 * m + b))).sum::<i32>();
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
