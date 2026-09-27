//! IQ4_NL rows against one vector of Q8 activations.

use crate::{Block, half, supported};

/// Row, four-block groups, activation quants and activation scales.
pub type Kernel = unsafe extern "sysv64" fn(*const u8, usize, *const i8, *const f32) -> f32;

#[kgen_macros::kernel(iq4_nl(ahead = 1))]
unsafe extern "sysv64" fn ahead1(row: *const u8, groups: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq4_nl(ahead = 2))]
unsafe extern "sysv64" fn ahead2(row: *const u8, groups: usize, quants: *const i8, scales: *const f32) -> f32;

#[kgen_macros::kernel(iq4_nl(ahead = 3))]
unsafe extern "sysv64" fn ahead3(row: *const u8, groups: usize, quants: *const i8, scales: *const f32) -> f32;

/// Variants with one to three blocks' products in flight.
pub const KERNELS: [(&str, Kernel); 3] = [("ahead1", ahead1), ("ahead2", ahead2), ("ahead3", ahead3)];
/// Bytes per group of 128 weights.
pub const GROUP: usize = 72;

/// Run a generated kernel on checked slices.
pub fn dot(kernel: Kernel, row: &[u8], quants: &[Block], scales: &[f32]) -> f32 {
    assert_eq!(row.len() % GROUP, 0, "four-block groups");
    let blocks = row.len() / 18;
    assert!(quants.len() >= blocks && scales.len() >= blocks);
    assert!(supported(), "AVX2, FMA and F16C");
    unsafe { kernel(row.as_ptr(), blocks / 4, quants.as_ptr().cast(), scales.as_ptr()) }
}

/// The saturated pairs, scale products, FMA order and horizontal sum of the original kernel.
pub fn reference(row: &[u8], quants: &[Block], scales: &[f32]) -> f32 {
    if row.is_empty() {
        return 0.0;
    }
    assert_eq!(row.len() % GROUP, 0);
    let levels = [-127i8, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
    let mut acc = [0.0f32; 8];
    for (b, blk) in row.as_chunks::<18>().0.iter().enumerate() {
        let d = half(u16::from_le_bytes([blk[0], blk[1]]));
        let scale = d * scales[b];
        let x = &quants[b].0;
        let weight = |j: usize| {
            let byte = blk[2 + j % 16];
            levels[if j < 16 { (byte & 15) as usize } else { (byte >> 4) as usize }]
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
    let v = acc.map(|a| a + 0.0);
    let s = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
    (s[0] + s[2]) + (s[1] + s[3])
}
