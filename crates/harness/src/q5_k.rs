//! Q5_K rows against two or four vectors of Q8 activations, each with its half sums.

use crate::Block;
use crate::q4_k::{Kernel, dot_of, reference_of};

#[kgen_macros::kernel(q5_k(vectors = 2))]
unsafe extern "sysv64" fn two(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q5_k(vectors = 4))]
unsafe extern "sysv64" fn four(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q5_k(vectors = 4, ahead = 1024, hint = 0))]
unsafe extern "sysv64" fn four_ahead(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

/// The kernel for each width and way of building it, to compare by name: the vectors it takes.
pub const KERNELS: [(usize, Kernel); 3] = [(2, two), (4, four), (4, four_ahead)];
/// Bytes per superblock of 256 weights.
pub const SUPERBLOCK: usize = 176;

/// Run a kernel of `vectors` on checked slices; the layouts are [`crate::q4_k::dot`]'s.
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

/// One vector's dot with the row: [`crate::q4_k::reference`] with each weight's fifth bit.
pub fn reference(row: &[u8], quants: &[Block], scales: &[f32], half_sums: &[f32]) -> f32 {
    reference_of::<SUPERBLOCK>(true, row, quants, scales, half_sums)
}
