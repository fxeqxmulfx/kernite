//! `IQ2_XXS` rows against one activation vector.

use crate::{Block, half, supported};

/// The kernels' type: `row`, its superblocks, the activations' quants and scales, the grid.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32, *const u64) -> f32;

#[kgen_macros::kernel(iq2_xxs(ahead = 1))]
unsafe extern "sysv64" fn ahead1(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_xxs(ahead = 2))]
unsafe extern "sysv64" fn ahead2(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_xxs(ahead = 3))]
unsafe extern "sysv64" fn ahead3(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_xxs(ahead = 4))]
unsafe extern "sysv64" fn ahead4(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_xxs(ahead = 3, pairs = 1))]
unsafe extern "sysv64" fn ahead3_pairs1(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

#[kgen_macros::kernel(iq2_xxs(ahead = 3, pairs = 2))]
unsafe extern "sysv64" fn ahead3_pairs2(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
) -> f32;

/// Every kernel, by name: `aheadD` makes each group's products `D` groups ahead of its sum, and
/// `pairsP` loads `P` of its two pairs of grid rows as pairs.
pub const KERNELS: [(&str, Kernel); 6] = [
    ("ahead1", ahead1),
    ("ahead2", ahead2),
    ("ahead3", ahead3),
    ("ahead4", ahead4),
    ("ahead3-pairs1", ahead3_pairs1),
    ("ahead3-pairs2", ahead3_pairs2),
];

/// Row, superblocks, two vectors' quants and scales stored together block by block, the grid,
/// and the two dots' destination.
pub type Pair = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32, *const u64, *mut f32);

#[kgen_macros::kernel(iq2_xxs2(ahead = 1))]
unsafe extern "sysv64" fn pair_ahead1(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    grid: *const u64,
    out: *mut f32,
);

#[kgen_macros::kernel(iq2_xxs2(ahead = 2))]
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

/// Bytes to a superblock of 256 weights.
pub const SUPERBLOCK: usize = 66;

/// `kernel` on `row`, checked: whole superblocks, activations for all of them, a CPU that
/// runs it.
pub fn dot(kernel: Kernel, row: &[u8], quants: &[Block], scales: &[f32], grid: &[u64; 256]) -> f32 {
    assert_eq!(row.len() % SUPERBLOCK, 0, "whole superblocks");
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 8 && scales.len() >= sbs * 8, "activations for every group");
    assert!(supported(), "AVX2, FMA and F16C");
    // The kernel reads `sbs` superblocks of `row`, 8 blocks of `quants` (aligned, as `Block`
    // is) and 8 of `scales` for each, and entries of `grid` by byte: all checked above.
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr(), grid.as_ptr()) }
}

/// Run a two-vector kernel on checked slices, which hold both vectors' activations as
/// [`crate::interleave`] lays them out.
pub fn dot2(kernel: Pair, row: &[u8], quants: &[Block], scales: &[f32], grid: &[u64; 256]) -> [f32; 2] {
    assert_eq!(row.len() % SUPERBLOCK, 0, "whole superblocks");
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 16 && scales.len() >= sbs * 16, "both vectors' activations for every group");
    assert!(supported(), "AVX2, FMA and F16C");
    let mut out = [0.0; 2];
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr(), grid.as_ptr(), out.as_mut_ptr()) };
    out
}

/// What the kernels compute, one operation at a time in their order.
///
/// Group `g` of superblock `sb` has its scale `((d * (2k + 1)) * 0.125) * scales[8 sb + g]`,
/// `k` the top nibble of its second word; its 32 levels are bytes of four grid entries, one
/// per index byte; its signs, 7-bit fields of that word with a parity bit on top, negate its
/// activations (`-(-128)` stays -128, as `vpsignb` has it). The products are summed as
/// `vpmaddubsw` does, in pairs saturated to 16 bits, then as `vpmaddwd` does, into eight lanes
/// of four; each lane is converted and added in with one fused multiply-add, group after group.
/// The lanes are summed at the end as the kernels do: `acc + 0.0`, the high four onto the low,
/// the high pair onto the low, lane 1 onto lane 0.
pub fn reference(row: &[u8], quants: &[Block], scales: &[f32], grid: &[u64; 256]) -> f32 {
    let sbs = row.len() / SUPERBLOCK;
    if sbs == 0 {
        return 0.0;
    }
    let mut acc = [0.0f32; 8];
    for (sb, blk) in row.as_chunks::<SUPERBLOCK>().0.iter().enumerate() {
        let d = half(u16::from_le_bytes([blk[0], blk[1]]));
        for g in 0..8 {
            let at = 2 + 8 * g;
            let word = u32::from_le_bytes(blk[at + 4..at + 8].try_into().unwrap());
            let scale = ((d * (2 * (word >> 28) + 1) as f32) * 0.125) * scales[8 * sb + g];
            let x = &quants[8 * sb + g].0;
            let weight = |j: usize| (grid[usize::from(blk[at + j / 8])] >> (8 * (j % 8))) as u8;
            let signed = |j: usize| {
                let field = word >> (7 * (j / 8)) & 0x7f;
                let byte = field | (field.count_ones() & 1) << 7;
                if byte >> (j % 8) & 1 == 1 { x[j].wrapping_neg() } else { x[j] }
            };
            let pair = |m: usize| {
                let sum = (0..2).map(|b| i32::from(weight(2 * m + b)) * i32::from(signed(2 * m + b))).sum::<i32>();
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
