//! One Q6_K row against Q8 activations, with the two half sums per block.

use crate::{Block, half, supported};

/// The generated row function's ABI.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32, *const f32) -> f32;

#[kgen_macros::kernel(q6_k)]
unsafe extern "sysv64" fn generated(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
) -> f32;

#[kgen_macros::kernel(q6_k_front)]
unsafe extern "sysv64" fn generated_front(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
) -> f32;

/// The generated Q6_K kernel.
pub const KERNEL: Kernel = generated;
/// Bytes per superblock of 256 weights.
pub const SUPERBLOCK: usize = 210;
/// Bytes per 256 weights after moving scales to an aligned header.
pub const FRONT_SUPERBLOCK: usize = 224;

/// Put the half scale and the sixteen signed group scales first, then the two code planes
/// at aligned offsets. The last fourteen header bytes are padding.
pub fn repack_front(row: &[u8]) -> Vec<u8> {
    assert_eq!(row.len() % SUPERBLOCK, 0);
    let mut out = vec![0u8; row.len() / SUPERBLOCK * FRONT_SUPERBLOCK];
    for (src, dst) in row.as_chunks::<SUPERBLOCK>().0.iter().zip(out.as_chunks_mut::<FRONT_SUPERBLOCK>().0) {
        dst[..2].copy_from_slice(&src[208..210]);
        dst[2..18].copy_from_slice(&src[192..208]);
        dst[32..224].copy_from_slice(&src[..192]);
    }
    out
}

/// Generated Q6_K kernel on the front-scale layout.
pub fn dot_front(row: &[u8], quants: &[Block], scales: &[f32], half_sums: &[f32]) -> f32 {
    assert_eq!(row.len() % FRONT_SUPERBLOCK, 0);
    let blocks = row.len() / FRONT_SUPERBLOCK * 8;
    assert!(quants.len() >= blocks && scales.len() >= blocks && half_sums.len() >= blocks * 2);
    assert!(supported());
    unsafe {
        generated_front(
            row.as_ptr(),
            row.len() / FRONT_SUPERBLOCK,
            quants.as_ptr().cast(),
            scales.as_ptr(),
            half_sums.as_ptr(),
        )
    }
}

/// Run the generated kernel on checked slices.
pub fn dot(row: &[u8], quants: &[Block], scales: &[f32], half_sums: &[f32]) -> f32 {
    assert_eq!(row.len() % SUPERBLOCK, 0);
    let blocks = row.len() / SUPERBLOCK * 8;
    assert!(quants.len() >= blocks && scales.len() >= blocks && half_sums.len() >= blocks * 2);
    assert!(supported());
    unsafe { KERNEL(row.as_ptr(), row.len() / SUPERBLOCK, quants.as_ptr().cast(), scales.as_ptr(), half_sums.as_ptr()) }
}

/// The same integer pairs, scale multiplication, offset FMAs and horizontal sum.
pub fn reference(row: &[u8], quants: &[Block], scales: &[f32], half_sums: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut off = [0.0f32; 8];
    for (sb, blk) in row.as_chunks::<SUPERBLOCK>().0.iter().enumerate() {
        let d = half(u16::from_le_bytes([blk[208], blk[209]]));
        for g in 0..8 {
            let n = g / 4;
            let k = g % 4;
            let ab = [d * (blk[192 + g * 2] as i8) as f32, d * (blk[193 + g * 2] as i8) as f32];
            for lane in 0..8 {
                let mut p = 0i32;
                for j in 0..4 {
                    let idx = lane * 4 + j;
                    let ql = blk[n * 64 + 32 * (k % 2) + idx];
                    let qh = blk[128 + n * 32 + idx];
                    let nib = if k < 2 { ql & 15 } else { ql >> 4 };
                    let w = i32::from(nib | ((qh >> (2 * k)) & 3) << 4);
                    p += w * i32::from(quants[sb * 8 + g].0[idx]);
                }
                let ws = ab[usize::from(lane >= 4)] * scales[sb * 8 + g];
                acc[lane] = (p as f32).mul_add(ws, acc[lane]);
            }
        }
        for half in 0..2 {
            for (lane, out) in off.iter_mut().enumerate() {
                let i = half * 8 + lane;
                let wo = -32.0 * (d * (blk[192 + i] as i8) as f32);
                *out = wo.mul_add(half_sums[sb * 16 + i], *out);
            }
        }
    }
    let v = std::array::from_fn::<_, 8, _>(|i| acc[i] + off[i]);
    let a = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
    (a[0] + a[2]) + (a[1] + a[3])
}
