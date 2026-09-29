//! Q4_K rows against two or four vectors of Q8 activations, each with its half sums.

use crate::{Block, half, supported};

/// Row, superblocks, the vectors' quants and scales stored together block by block, their half
/// sums vector by vector, and where the dots go.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32, *const f32, *mut f32);

#[kgen_macros::kernel(q4_k(vectors = 2))]
unsafe extern "sysv64" fn two(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q4_k(vectors = 4))]
unsafe extern "sysv64" fn four(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q4_k(vectors = 4, spread = 0))]
unsafe extern "sysv64" fn four_flat(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q4_k(vectors = 2, ahead = 1024, hint = 1))]
unsafe extern "sysv64" fn two_ahead(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q4_k(vectors = 4, ahead = 1024, hint = 0))]
unsafe extern "sysv64" fn four_ahead(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

/// The kernel for each width and way of building it, to compare by name: the vectors it takes.
pub const KERNELS: [(usize, Kernel); 5] = [(2, two), (4, four), (4, four_flat), (2, two_ahead), (4, four_ahead)];
/// Bytes per superblock of 256 weights.
pub const SUPERBLOCK: usize = 144;

/// Run a kernel of `vectors` on checked slices: `quants` and `scales` as [`crate::interleave_all`]
/// lays the vectors' out, `half_sums` sixteen floats a superblock for vector 0, then vector 1, and
/// so on.
pub fn dot(
    vectors: usize,
    kernel: Kernel,
    row: &[u8],
    quants: &[Block],
    scales: &[f32],
    half_sums: &[f32],
) -> Vec<f32> {
    dot_of(SUPERBLOCK, vectors, kernel, row, quants, scales, half_sums)
}

/// [`dot`] for a format whose superblocks are `superblock` bytes.
pub(crate) fn dot_of(
    superblock: usize,
    vectors: usize,
    kernel: Kernel,
    row: &[u8],
    quants: &[Block],
    scales: &[f32],
    half_sums: &[f32],
) -> Vec<f32> {
    assert_eq!(row.len() % superblock, 0, "whole superblocks");
    let sbs = row.len() / superblock;
    assert!(quants.len() >= sbs * 8 * vectors && scales.len() >= sbs * 8 * vectors, "every vector's blocks");
    assert!(half_sums.len() >= sbs * 16 * vectors, "every vector's half sums");
    assert!(supported(), "AVX2, FMA and F16C");
    let mut out = vec![0.0; vectors];
    unsafe { kernel(row.as_ptr(), sbs, quants.as_ptr().cast(), scales.as_ptr(), half_sums.as_ptr(), out.as_mut_ptr()) };
    out
}

/// The scale and min of sub-block `j` from the twelve bytes at 4, as ggml's `get_scale_min_k4`.
///
/// Also `Q5_K`'s.
fn scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// One vector's dot with the row, as the original row kernel takes it: per block one saturating
/// pair sum, one exact conversion and one fused multiply-add at `d * sc` times the activation
/// scale; per superblock two fused multiply-adds of `-dmin * m` with the half sums; the two
/// accumulators added and summed across lanes at the end.
pub fn reference(row: &[u8], quants: &[Block], scales: &[f32], half_sums: &[f32]) -> f32 {
    reference_of::<SUPERBLOCK>(false, row, quants, scales, half_sums)
}

/// [`reference`] for a format with superblocks of `N` bytes, whose nibbles are the last 128 of
/// them, and which with `five` has 32 bytes of fifth bits, bit `j` of each for block `j`, before
/// them.
pub(crate) fn reference_of<const N: usize>(
    five: bool,
    row: &[u8],
    quants: &[Block],
    scales: &[f32],
    half_sums: &[f32],
) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut off = [0.0f32; 8];
    for (sb, blk) in row.as_chunks::<N>().0.iter().enumerate() {
        let d = half(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = half(u16::from_le_bytes([blk[2], blk[3]]));
        let mut wo = [0.0f32; 8];
        for j in 0..8 {
            let (sc, m) = scale_min(j, &blk[4..16]);
            let ws = d * f32::from(sc);
            wo[j] = -dmin * f32::from(m);
            let x = &quants[8 * sb + j].0;
            for (lane, a) in acc.iter_mut().enumerate() {
                let mut p = 0i32;
                for i in 0..4 {
                    let byte = blk[N - 128 + 32 * (j / 2) + 4 * lane + i];
                    let mut w = if j % 2 == 0 { byte & 15 } else { byte >> 4 };
                    if five {
                        w |= (blk[16 + 4 * lane + i] >> j & 1) << 4;
                    }
                    p += i32::from(w) * i32::from(x[4 * lane + i]);
                }
                *a = (p as f32).mul_add(ws * scales[8 * sb + j], *a);
            }
        }
        for (lane, o) in off.iter_mut().enumerate() {
            *o = wo[lane / 2].mul_add(half_sums[16 * sb + lane], *o);
        }
        for (lane, o) in off.iter_mut().enumerate() {
            *o = wo[4 + lane / 2].mul_add(half_sums[16 * sb + 8 + lane], *o);
        }
    }
    let v = std::array::from_fn::<_, 8, _>(|i| acc[i] + off[i]);
    let a = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
    (a[0] + a[2]) + (a[1] + a[3])
}
