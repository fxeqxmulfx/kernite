//! Q6_K rows on the front-scale layout against two or four vectors of Q8 activations, each with
//! its half sums.

use crate::q4_k::{Kernel, dot_of};
use crate::{Block, q6_k, supported};

#[kgen_macros::kernel(q6_k_few(vectors = 2))]
unsafe extern "sysv64" fn two(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q6_k_few(vectors = 4))]
unsafe extern "sysv64" fn four(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q6_k_few(vectors = 4, spread = 0))]
unsafe extern "sysv64" fn four_flat(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q6_k_few(vectors = 2, ahead = 1024, hint = 1))]
unsafe extern "sysv64" fn two_ahead(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q6_k_few(vectors = 4, ahead = 2048, hint = 0))]
unsafe extern "sysv64" fn four_ahead(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

#[kgen_macros::kernel(q6_k_few(vectors = 4, group = 8, ahead = 2048, hint = 0))]
unsafe extern "sysv64" fn four_of_eight(
    row: *const u8,
    sbs: usize,
    quants: *const i8,
    scales: *const f32,
    half_sums: *const f32,
    out: *mut f32,
);

/// The kernel for each width and way of building it, to compare by name: the vectors it takes,
/// which for the last is a group of eight, in two calls.
pub const KERNELS: [(usize, Kernel); 6] =
    [(2, two), (4, four), (4, four_flat), (2, two_ahead), (4, four_ahead), (8, four_of_eight)];
/// Bytes per superblock of 256 weights on the front layout.
pub const SUPERBLOCK: usize = q6_k::FRONT_SUPERBLOCK;

/// Run a kernel of `vectors` on checked slices; the layouts are [`crate::q4_k::dot`]'s. Eight
/// vectors are a group the four-wide kernel takes in two calls, its second four vectors in.
pub fn dot(
    vectors: usize,
    kernel: Kernel,
    row: &[u8],
    quants: &[Block],
    scales: &[f32],
    half_sums: &[f32],
) -> Vec<f32> {
    if vectors != 8 {
        return dot_of(SUPERBLOCK, vectors, kernel, row, quants, scales, half_sums);
    }
    assert_eq!(row.len() % SUPERBLOCK, 0, "whole superblocks");
    let sbs = row.len() / SUPERBLOCK;
    assert!(quants.len() >= sbs * 8 * 8 && scales.len() >= sbs * 8 * 8, "every vector's blocks");
    assert!(half_sums.len() >= sbs * 16 * 8, "every vector's half sums");
    assert!(supported(), "AVX2, FMA and F16C");
    let mut out = vec![0.0; 8];
    for half in 0..2 {
        unsafe {
            kernel(
                row.as_ptr(),
                sbs,
                quants.as_ptr().add(4 * half).cast(),
                scales.as_ptr().add(4 * half),
                half_sums.as_ptr().add(4 * half * sbs * 16),
                out.as_mut_ptr().add(4 * half),
            )
        };
    }
    out
}

/// One vector's dot with the row, as the original row kernel takes it: [`q6_k::reference`] on the
/// row put back into the file's layout.
pub fn reference(row: &[u8], quants: &[Block], scales: &[f32], half_sums: &[f32]) -> f32 {
    assert_eq!(row.len() % SUPERBLOCK, 0);
    let mut file = vec![0u8; row.len() / SUPERBLOCK * q6_k::SUPERBLOCK];
    for (front, file) in row.as_chunks::<SUPERBLOCK>().0.iter().zip(file.as_chunks_mut::<{ q6_k::SUPERBLOCK }>().0) {
        file[..192].copy_from_slice(&front[32..224]);
        file[192..208].copy_from_slice(&front[2..18]);
        file[208..210].copy_from_slice(&front[..2]);
    }
    q6_k::reference(&file, quants, scales, half_sums)
}
