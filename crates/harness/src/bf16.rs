//! Pairs of BF16 rows against one vector of floats.

use crate::supported;

/// The kernels' type: the first row, `x`, `out`, the pairs, bytes from a row to the next, and
/// blocks of 32 values a row.
pub type Kernel = unsafe extern "sysv64" fn(*const u16, *const f32, *mut f32, usize, usize, usize);

#[kgen_macros::kernel(bf16_pairs)]
unsafe extern "sysv64" fn plain(
    w: *const u16,
    x: *const f32,
    out: *mut f32,
    pairs: usize,
    stride: usize,
    blocks: usize,
);

#[kgen_macros::kernel(bf16_pairs(ahead = 512))]
unsafe extern "sysv64" fn ahead512(
    w: *const u16,
    x: *const f32,
    out: *mut f32,
    pairs: usize,
    stride: usize,
    blocks: usize,
);

#[kgen_macros::kernel(bf16_pairs(ahead = 1024))]
unsafe extern "sysv64" fn ahead1024(
    w: *const u16,
    x: *const f32,
    out: *mut f32,
    pairs: usize,
    stride: usize,
    blocks: usize,
);

#[kgen_macros::kernel(bf16_pairs(ahead = 512, hint = 1))]
unsafe extern "sysv64" fn ahead512_t1(
    w: *const u16,
    x: *const f32,
    out: *mut f32,
    pairs: usize,
    stride: usize,
    blocks: usize,
);

#[kgen_macros::kernel(bf16_triples(ahead = 2048))]
unsafe extern "sysv64" fn triples_ahead2048(
    w: *const u16,
    x: *const f32,
    out: *mut f32,
    triples: usize,
    stride: usize,
    blocks: usize,
);

/// Every kernel, by name: `aheadN` asks for each row's line `N` bytes on, into L1 or, with
/// `t1`, L2.
pub const KERNELS: [(&str, Kernel); 4] =
    [("plain", plain), ("ahead512", ahead512), ("ahead1024", ahead1024), ("ahead512-t1", ahead512_t1)];

/// Three-row kernel with its rows asked for 2048 bytes ahead.
pub const TRIPLES: Kernel = triples_ahead2048;

/// The two-vector kernels' type: the first row, the two vectors back to back, `out`, bytes from
/// the first vector's outputs to the second's, the rows, and blocks of 32 values a row.
pub type Two = unsafe extern "sysv64" fn(*const u16, *const f32, *mut f32, usize, usize, usize);

#[kgen_macros::kernel(bf16_two)]
unsafe extern "sysv64" fn two_plain(
    w: *const u16,
    x: *const f32,
    out: *mut f32,
    gap: usize,
    rows: usize,
    blocks: usize,
);

#[kgen_macros::kernel(bf16_two(ahead = 2048))]
unsafe extern "sysv64" fn two_ahead2048(
    w: *const u16,
    x: *const f32,
    out: *mut f32,
    gap: usize,
    rows: usize,
    blocks: usize,
);

/// Every two-vector kernel, by name.
pub const TWOS: [(&str, Two); 2] = [("two", two_plain), ("two-ahead2048", two_ahead2048)];

/// `kernel` on the rows of `w` against the two vectors of `x`, back to back, checked: the first
/// vector's products go into `first` and the second's into `second`, one a row.
pub fn two(kernel: Two, w: &[u16], x: &[f32], [first, second]: [&mut [f32]; 2]) {
    let cols = x.len() / 2;
    assert!(cols > 0 && cols.is_multiple_of(32) && x.len() == 2 * cols, "two vectors of whole blocks of 32");
    assert!(!first.is_empty() && first.len() == second.len(), "one output a row for each vector");
    assert_eq!(w.len(), first.len() * cols, "rows as long as a vector");
    assert!(supported(), "AVX2, FMA and F16C");
    let gap = (second.as_mut_ptr() as usize).wrapping_sub(first.as_mut_ptr() as usize);
    // The kernel reads the rows back to back and both vectors, all checked above, and writes one
    // float a row into `first` and the float `gap` bytes on, in `second`.
    unsafe { kernel(w.as_ptr(), x.as_ptr(), first.as_mut_ptr(), gap, first.len(), cols / 32) }
}

/// `kernel` on the rows of `w`, `x.len()` values each, into `out`, checked: whole blocks of 32,
/// a whole number of pairs of rows, one output a row, a CPU that runs it.
pub fn rows(kernel: Kernel, w: &[u16], x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    assert!(cols > 0 && cols.is_multiple_of(32), "whole blocks of 32");
    assert_eq!(w.len(), out.len() * cols, "one output a row");
    assert!(out.len() >= 2 && out.len().is_multiple_of(2), "whole pairs");
    assert!(supported(), "AVX2, FMA and F16C");
    // The kernel reads `out.len()` rows of `cols` values, `2 * cols` bytes apart, and `cols`
    // values of `x`, and writes `out.len()` floats: all checked above. What it prefetches past
    // the rows it never reads.
    unsafe { kernel(w.as_ptr(), x.as_ptr(), out.as_mut_ptr(), out.len() / 2, 2 * cols, cols / 32) }
}

/// Three rows per call, with the same argument checks as [`rows`].
pub fn triples(w: &[u16], x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    assert!(cols > 0 && cols.is_multiple_of(32), "whole blocks of 32");
    assert_eq!(w.len(), out.len() * cols, "one output a row");
    assert!(out.len() >= 3 && out.len().is_multiple_of(3), "whole triples");
    assert!(supported(), "AVX2, FMA and F16C");
    unsafe { TRIPLES(w.as_ptr(), x.as_ptr(), out.as_mut_ptr(), out.len() / 3, 2 * cols, cols / 32) }
}

/// What the kernels compute for one row, one operation at a time in their order.
///
/// A weight is the float whose top half it is. Value `i` goes into lane `i % 8` of accumulator
/// `i / 8 % 4` with one fused multiply-add, in order; the lanes are `(a0 + a1) + (a2 + a3)`,
/// summed from lane 0 up.
pub fn reference(row: &[u16], x: &[f32]) -> f32 {
    let mut acc = [[0.0f32; 8]; 4];
    for (i, (&w, &x)) in row.iter().zip(x).enumerate() {
        let a = &mut acc[i / 8 % 4][i % 8];
        *a = f32::from_bits(u32::from(w) << 16).mul_add(x, *a);
    }
    let v: [f32; 8] = std::array::from_fn(|l| (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]));
    v[1..].iter().fold(v[0], |sum, &lane| sum + lane)
}
