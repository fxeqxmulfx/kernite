//! Every generated kernel against its scalar reference, to the bit, on random rows.

use kgen_harness::bf16;
use kgen_harness::iq2_s;
use kgen_harness::iq2_s_nibble;
use kgen_harness::iq2_s_packed;
use kgen_harness::iq2_s_planes;
use kgen_harness::iq2_xs_packed;
use kgen_harness::iq2_xxs::{KERNELS, SUPERBLOCK, dot, reference};
use kgen_harness::iq3_s;
use kgen_harness::iq3_s_nibble;
use kgen_harness::iq3_xxs;
use kgen_harness::iq3_xxs_nibble;
use kgen_harness::iq4_nl;
use kgen_harness::iq4_xs;
use kgen_harness::q6_k;
use kgen_harness::{Block, supported};

/// `f` called `depth` frames further down the stack, each frame holding a little more.
#[inline(never)]
fn at_depth(depth: usize, f: &mut dyn FnMut()) {
    let pad = [depth as u8; 24];
    std::hint::black_box(&pad);
    match depth {
        0 => f(),
        _ => at_depth(depth - 1, f),
    }
}

/// xorshift64*: the same numbers on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }
}

/// Random finite half bits.
fn finite_half(rng: &mut Rng) -> u16 {
    loop {
        let bits = (rng.next() >> 48) as u16;
        if bits >> 10 & 0x1f != 0x1f {
            return bits;
        }
    }
}

/// A grid like the format's, levels 8, 25 and 43, or any bytes at all, which drive
/// `vpmaddubsw` into saturation.
fn grid(rng: &mut Rng, wild: bool) -> [u64; 256] {
    std::array::from_fn(|_| {
        u64::from_le_bytes(std::array::from_fn(
            |_| if wild { rng.byte() } else { [8, 25, 43][(rng.next() % 3) as usize] },
        ))
    })
}

fn case(rng: &mut Rng, sbs: usize) -> (Vec<u8>, Vec<Block>, Vec<f32>) {
    let mut row = vec![0u8; sbs * SUPERBLOCK];
    for blk in row.as_chunks_mut::<SUPERBLOCK>().0 {
        blk[..2].copy_from_slice(&finite_half(rng).to_le_bytes());
        for b in &mut blk[2..] {
            *b = rng.byte();
        }
    }
    let quants = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
    let scales = (0..sbs * 8)
        .map(|_| {
            let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            unit * 2f32.powi((rng.next() % 21) as i32 - 10)
        })
        .collect();
    (row, quants, scales)
}

#[test]
fn q6_k_matches_its_reference_to_the_bit() {
    if !supported() {
        return;
    }
    let mut rng = Rng(0x8a1b_cdef_0472_3956);
    for sbs in [0, 1, 2, 3, 4, 8, 10] {
        for trial in 0..50 {
            let mut row = vec![0u8; sbs * q6_k::SUPERBLOCK];
            for blk in row.as_chunks_mut::<{ q6_k::SUPERBLOCK }>().0 {
                for b in &mut blk[..208] {
                    *b = rng.byte();
                }
                blk[208..210].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
            }
            let quants: Vec<Block> = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
            let scales: Vec<f32> =
                (0..sbs * 8).map(|_| ((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) / 32.0).collect();
            let sums: Vec<f32> =
                (0..sbs * 16).map(|_| ((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) / 8.0).collect();
            let want = q6_k::reference(&row, &quants, &scales, &sums);
            let got = q6_k::dot(&row, &quants, &scales, &sums);
            assert_eq!(got.to_bits(), want.to_bits(), "{sbs} superblocks, trial {trial}: {got} versus {want}");
            let front = q6_k::repack_front(&row);
            let got = q6_k::dot_front(&front, &quants, &scales, &sums);
            assert_eq!(got.to_bits(), want.to_bits(), "front {sbs} superblocks, trial {trial}: {got} versus {want}");
        }
    }
}

#[test]
fn iq2_s_repacked_matches_grid_reference_to_the_bit() {
    if !supported() {
        return;
    }
    let mut rng = Rng(0x7128_8d23_64fb_129c);
    let grid: [u64; 1024] =
        std::array::from_fn(|_| u64::from_le_bytes(std::array::from_fn(|_| [8, 25, 43][(rng.next() % 3) as usize])));
    for sbs in [0, 1, 2, 3, 4, 8, 10] {
        for trial in 0..50 {
            let mut row = vec![0u8; sbs * iq2_s::SUPERBLOCK];
            for blk in row.as_chunks_mut::<{ iq2_s::SUPERBLOCK }>().0 {
                blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                for b in &mut blk[2..] {
                    *b = rng.byte();
                }
            }
            let packed = iq2_s_nibble::repack(&row, &grid);
            let planes = iq2_s_planes::repack(&row, &grid);
            let quants: Vec<Block> = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
            let scales: Vec<f32> =
                (0..sbs * 8).map(|_| ((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) / 32.0).collect();
            let want = iq2_s::reference(&row, &quants, &scales, &grid);
            for (name, kernel) in iq2_s_nibble::KERNELS {
                let got = iq2_s_nibble::dot(kernel, &packed, &quants, &scales);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{name}, {sbs} superblocks, trial {trial}: {got} versus {want}"
                );
            }
            for (name, kernel) in iq2_s_planes::KERNELS {
                let got = iq2_s_planes::dot(kernel, &planes, &quants, &scales);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "planes {name}, {sbs} superblocks, trial {trial}: {got} versus {want}"
                );
            }
        }
    }
}

#[test]
fn iq2_xxs_matches_its_reference_to_the_bit() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut checked = 0;
    for wild in [false, true] {
        let grid = grid(&mut rng, wild);
        for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
            for trial in 0..40 {
                let (row, quants, scales) = case(&mut rng, sbs);
                let want = reference(&row, &quants, &scales, &grid);
                for (name, kernel) in KERNELS {
                    let got = dot(kernel, &row, &quants, &scales, &grid);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{name}, {sbs} superblocks, trial {trial}, wild grid {wild}: {got} against {want}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 2 * 8 * 40 * KERNELS.len());
}

#[test]
fn iq2_s_matches_its_reference_to_the_bit() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0xf4c9_e2da_37b0_1586);
    let mut checked = 0;
    for wild in [false, true] {
        let grid: [u64; 1024] = std::array::from_fn(|_| {
            u64::from_le_bytes(std::array::from_fn(|_| {
                if wild { rng.byte() } else { [8, 25, 43][(rng.next() % 3) as usize] }
            }))
        });
        for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
            for trial in 0..40 {
                let mut row = vec![0u8; sbs * iq2_s::SUPERBLOCK];
                for blk in row.as_chunks_mut::<{ iq2_s::SUPERBLOCK }>().0 {
                    blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                    for b in &mut blk[2..] {
                        *b = rng.byte();
                    }
                }
                let quants: Vec<Block> =
                    (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
                let scales: Vec<f32> = (0..sbs * 8)
                    .map(|_| {
                        let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                        unit * 2f32.powi((rng.next() % 21) as i32 - 10)
                    })
                    .collect();
                let want = iq2_s::reference(&row, &quants, &scales, &grid);
                let packed = iq2_s_packed::repack(&row);
                for (name, kernel) in iq2_s_packed::KERNELS {
                    let got = iq2_s_packed::dot(kernel, &packed, &quants, &scales, &grid);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "IQ2_S {name}, {sbs} superblocks, trial {trial}, wild grid {wild}: {got} against {want}"
                    );
                }
                for (name, kernel) in iq2_s::KERNELS {
                    let got = iq2_s::dot(kernel, &row, &quants, &scales, &grid);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "IQ2_S {name}, {sbs} superblocks, trial {trial}, wild grid {wild}: {got} against {want}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 2 * 8 * 40 * iq2_s::KERNELS.len());
}

#[test]
fn iq2_s_packed_pairs_match_the_original_row_to_the_bit_for_each_vector() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0x3b9e_71c4_d05a_28f6);
    let mut checked = 0;
    for wild in [false, true] {
        let grid: [u64; 1024] = std::array::from_fn(|_| {
            u64::from_le_bytes(std::array::from_fn(|_| {
                if wild { rng.byte() } else { [8, 25, 43][(rng.next() % 3) as usize] }
            }))
        });
        for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
            for trial in 0..40 {
                let mut row = vec![0u8; sbs * iq2_s::SUPERBLOCK];
                for blk in row.as_chunks_mut::<{ iq2_s::SUPERBLOCK }>().0 {
                    blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                    for b in &mut blk[2..] {
                        *b = rng.byte();
                    }
                }
                let vectors: [(Vec<Block>, Vec<f32>); 2] = std::array::from_fn(|_| {
                    let quants = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
                    let scales = (0..sbs * 8)
                        .map(|_| {
                            let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                            unit * 2f32.powi((rng.next() % 21) as i32 - 10)
                        })
                        .collect();
                    (quants, scales)
                });
                let quants = kgen_harness::interleave(&vectors[0].0, &vectors[1].0);
                let scales = kgen_harness::interleave(&vectors[0].1, &vectors[1].1);
                let want = vectors.each_ref().map(|(q, s)| iq2_s::reference(&row, q, s, &grid));
                let packed = iq2_s_packed::repack(&row);
                for (name, kernel) in iq2_s_packed::PAIRS {
                    // Its frame is aligned inside, whatever the caller's stack: called from a few
                    // depths, it has to be right from each.
                    let mut got = [0.0; 2];
                    at_depth(trial % 8, &mut || got = iq2_s_packed::dot2(kernel, &packed, &quants, &scales, &grid));
                    for v in 0..2 {
                        assert_eq!(
                            got[v].to_bits(),
                            want[v].to_bits(),
                            "IQ2_S {name}, vector {v}, {sbs} superblocks, trial {trial}, wild grid {wild}: {} against {}",
                            got[v],
                            want[v]
                        );
                    }
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 2 * 8 * 40 * iq2_s_packed::PAIRS.len());
}

#[test]
fn iq2_xs_packed_matches_file_reference_to_the_bit() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0x7a86_92f1_a7e4_5311);
    for wild in [false, true] {
        let grid: [u64; 512] = std::array::from_fn(|_| {
            u64::from_le_bytes(std::array::from_fn(|_| {
                if wild { rng.byte() } else { [8, 25, 43][(rng.next() % 3) as usize] }
            }))
        });
        for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
            for trial in 0..40 {
                let mut row = vec![0u8; sbs * 74];
                for blk in row.as_chunks_mut::<74>().0 {
                    blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                    for b in &mut blk[2..] {
                        *b = rng.byte();
                    }
                }
                let quants: Vec<Block> =
                    (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
                let scales: Vec<f32> = (0..sbs * 8)
                    .map(|_| {
                        let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                        unit * 2f32.powi((rng.next() % 21) as i32 - 10)
                    })
                    .collect();
                let want = iq2_xs_packed::reference(&row, &quants, &scales, &grid);
                let packed = iq2_xs_packed::repack(&row);
                for (name, kernel) in iq2_xs_packed::KERNELS {
                    let got = iq2_xs_packed::dot(kernel, &packed, &quants, &scales, &grid);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "IQ2_XS {name}, {sbs} superblocks, trial {trial}, wild grid {wild}: {got} against {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn iq3_xxs_matches_its_reference_to_the_bit() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0x713e_a8f3_eacb_4671);
    let mut checked = 0;
    for wild in [false, true] {
        let grid: [u32; 256] = std::array::from_fn(|_| {
            u32::from_le_bytes(std::array::from_fn(|_| {
                if wild { rng.byte() } else { [8, 25, 43][(rng.next() % 3) as usize] }
            }))
        });
        for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
            for trial in 0..40 {
                let mut row = vec![0u8; sbs * iq3_xxs::SUPERBLOCK];
                for blk in row.as_chunks_mut::<{ iq3_xxs::SUPERBLOCK }>().0 {
                    blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                    for b in &mut blk[2..] {
                        *b = rng.byte();
                    }
                }
                let quants: Vec<Block> =
                    (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
                let scales: Vec<f32> = (0..sbs * 8)
                    .map(|_| {
                        let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                        unit * 2f32.powi((rng.next() % 21) as i32 - 10)
                    })
                    .collect();
                let want = iq3_xxs::reference(&row, &quants, &scales, &grid);
                for (name, kernel) in iq3_xxs::KERNELS {
                    let got = iq3_xxs::dot(kernel, &row, &quants, &scales, &grid);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "IQ3_XXS {name}, {sbs} superblocks, trial {trial}, wild grid {wild}: {got} against {want}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 2 * 8 * 40 * iq3_xxs::KERNELS.len());
}

#[test]
fn iq3_s_matches_its_reference_to_the_bit() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0xcfa6_e8d7_14b2_0491);
    let mut checked = 0;
    for wild in [false, true] {
        let grid: [u32; 512] = std::array::from_fn(|_| {
            u32::from_le_bytes(std::array::from_fn(|_| {
                if wild { rng.byte() } else { [8, 25, 43][(rng.next() % 3) as usize] }
            }))
        });
        for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
            for trial in 0..40 {
                let mut row = vec![0u8; sbs * iq3_s::SUPERBLOCK];
                for blk in row.as_chunks_mut::<{ iq3_s::SUPERBLOCK }>().0 {
                    blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                    for b in &mut blk[2..] {
                        *b = rng.byte();
                    }
                }
                let quants: Vec<Block> =
                    (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
                let scales: Vec<f32> = (0..sbs * 8)
                    .map(|_| {
                        let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                        unit * 2f32.powi((rng.next() % 21) as i32 - 10)
                    })
                    .collect();
                let want = iq3_s::reference(&row, &quants, &scales, &grid);
                for (name, kernel) in iq3_s::KERNELS {
                    let got = iq3_s::dot(kernel, &row, &quants, &scales, &grid);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "IQ3_S {name}, {sbs} superblocks, trial {trial}, wild grid {wild}: {got} against {want}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 2 * 8 * 40 * iq3_s::KERNELS.len());
}

#[test]
fn iq3_s_nibbles_match_the_original_row_to_the_bit() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0xeb40_2ad3_75f8_196c);
    let grid: [u32; 512] =
        std::array::from_fn(|_| u32::from_le_bytes(std::array::from_fn(|_| 1 + 2 * (rng.byte() % 8))));
    let mut checked = 0;
    for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
        for trial in 0..60 {
            let mut original = vec![0u8; sbs * iq3_s::SUPERBLOCK];
            for blk in original.as_chunks_mut::<{ iq3_s::SUPERBLOCK }>().0 {
                blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                for b in &mut blk[2..] {
                    *b = rng.byte();
                }
            }
            let packed = iq3_s_nibble::repack(&original, &grid);
            let quants: Vec<Block> = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
            let scales: Vec<f32> = (0..sbs * 8)
                .map(|_| {
                    let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    unit * 2f32.powi((rng.next() % 21) as i32 - 10)
                })
                .collect();
            let want = iq3_s::reference(&original, &quants, &scales, &grid);
            for (name, kernel) in iq3_s_nibble::KERNELS {
                let got = iq3_s_nibble::dot(kernel, &packed, &quants, &scales);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "IQ3_S nibble {name}, {sbs} superblocks, trial {trial}: {got} against {want}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 8 * 60 * iq3_s_nibble::KERNELS.len());
}

#[test]
fn iq3_s_nibble_pairs_match_the_original_row_to_the_bit_for_each_vector() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0x5a1c_93e0_27b4_d86f);
    let grid: [u32; 512] =
        std::array::from_fn(|_| u32::from_le_bytes(std::array::from_fn(|_| 1 + 2 * (rng.byte() % 8))));
    let mut checked = 0;
    for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
        for trial in 0..60 {
            let mut original = vec![0u8; sbs * iq3_s::SUPERBLOCK];
            for blk in original.as_chunks_mut::<{ iq3_s::SUPERBLOCK }>().0 {
                blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                for b in &mut blk[2..] {
                    *b = rng.byte();
                }
            }
            let packed = iq3_s_nibble::repack(&original, &grid);
            let vectors: [(Vec<Block>, Vec<f32>); 2] = std::array::from_fn(|_| {
                let quants = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
                let scales = (0..sbs * 8)
                    .map(|_| {
                        let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                        unit * 2f32.powi((rng.next() % 21) as i32 - 10)
                    })
                    .collect();
                (quants, scales)
            });
            let quants = kgen_harness::interleave(&vectors[0].0, &vectors[1].0);
            let scales = kgen_harness::interleave(&vectors[0].1, &vectors[1].1);
            let want = vectors.each_ref().map(|(q, s)| iq3_s::reference(&original, q, s, &grid));
            for (name, kernel) in iq3_s_nibble::PAIRS {
                // Its frame is aligned inside, whatever the caller's stack: called from a few
                // depths, it has to be right from each.
                let mut got = [0.0; 2];
                at_depth(trial % 8, &mut || got = iq3_s_nibble::dot2(kernel, &packed, &quants, &scales));
                for v in 0..2 {
                    assert_eq!(
                        got[v].to_bits(),
                        want[v].to_bits(),
                        "IQ3_S nibble {name}, vector {v}, {sbs} superblocks, trial {trial}: {} against {}",
                        got[v],
                        want[v]
                    );
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 8 * 60 * iq3_s_nibble::PAIRS.len());
}

/// Random IQ4_XS superblocks with finite halves, and a vector's activations for them.
fn iq4_xs_case(rng: &mut Rng, sbs: usize) -> (Vec<u8>, Vec<Block>, Vec<f32>) {
    let mut row = vec![0u8; sbs * iq4_xs::SUPERBLOCK];
    for blk in row.as_chunks_mut::<{ iq4_xs::SUPERBLOCK }>().0 {
        blk[..2].copy_from_slice(&finite_half(rng).to_le_bytes());
        for b in &mut blk[2..] {
            *b = rng.byte();
        }
    }
    let (quants, scales) = activations(rng, sbs * 8);
    (row, quants, scales)
}

/// `blocks` random activation blocks, and scales across twenty binades of either sign.
fn activations(rng: &mut Rng, blocks: usize) -> (Vec<Block>, Vec<f32>) {
    let quants = (0..blocks).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
    let scales = (0..blocks)
        .map(|_| {
            let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            unit * 2f32.powi((rng.next() % 21) as i32 - 10)
        })
        .collect();
    (quants, scales)
}

#[test]
fn iq4_xs_rows_match_the_original_kernel_to_the_bit() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0x3c7e_19a4_d250_8b61);
    let mut checked = 0;
    for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
        for trial in 0..60 {
            let (row, quants, scales) = iq4_xs_case(&mut rng, sbs);
            let want = iq4_xs::reference(&row, &quants, &scales);
            for (name, kernel) in iq4_xs::KERNELS {
                let got = iq4_xs::dot(kernel, &row, &quants, &scales);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "IQ4_XS {name}, {sbs} superblocks, trial {trial}: {got} against {want}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 8 * 60 * iq4_xs::KERNELS.len());
}

#[test]
fn iq4_xs_pairs_match_the_original_kernel_to_the_bit_for_each_vector() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0x8d05_f3b2_6e19_c47a);
    let mut checked = 0;
    for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
        for trial in 0..60 {
            let (row, first_quants, first_scales) = iq4_xs_case(&mut rng, sbs);
            let (second_quants, second_scales) = activations(&mut rng, sbs * 8);
            let quants = kgen_harness::interleave(&first_quants, &second_quants);
            let scales = kgen_harness::interleave(&first_scales, &second_scales);
            let want = [
                iq4_xs::reference(&row, &first_quants, &first_scales),
                iq4_xs::reference(&row, &second_quants, &second_scales),
            ];
            for (name, kernel) in iq4_xs::PAIRS {
                // Called from a few depths, as the IQ3_S pairs are: its frame is aligned inside.
                let mut got = [0.0; 2];
                at_depth(trial % 8, &mut || got = iq4_xs::dot2(kernel, &row, &quants, &scales));
                for v in 0..2 {
                    assert_eq!(
                        got[v].to_bits(),
                        want[v].to_bits(),
                        "IQ4_XS {name}, vector {v}, {sbs} superblocks, trial {trial}: {} against {}",
                        got[v],
                        want[v]
                    );
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 8 * 60 * iq4_xs::PAIRS.len());
}

#[test]
fn iq3_xxs_nibbles_match_the_original_row_to_the_bit() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0xcb72_8fa1_5234_0c9e);
    let levels = [4u8, 12, 20, 28, 36, 44, 52, 62];
    let grid: [u32; 256] =
        std::array::from_fn(|_| u32::from_le_bytes(std::array::from_fn(|_| levels[(rng.byte() & 7) as usize])));
    for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
        for trial in 0..60 {
            let mut original = vec![0u8; sbs * iq3_xxs::SUPERBLOCK];
            for blk in original.as_chunks_mut::<{ iq3_xxs::SUPERBLOCK }>().0 {
                blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                for b in &mut blk[2..] {
                    *b = rng.byte();
                }
            }
            let packed = iq3_xxs_nibble::repack(&original, &grid);
            let quants: Vec<Block> = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
            let scales: Vec<f32> = (0..sbs * 8)
                .map(|_| {
                    let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    unit * 2f32.powi((rng.next() % 21) as i32 - 10)
                })
                .collect();
            let want = iq3_xxs::reference(&original, &quants, &scales, &grid);
            for (name, kernel) in iq3_xxs_nibble::KERNELS {
                let got = iq3_xxs_nibble::dot(kernel, &packed, &quants, &scales);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "IQ3_XXS nibble {name}, {sbs} superblocks, trial {trial}: {got} against {want}"
                );
            }
        }
    }
}

#[test]
fn iq3_xxs_nibble_pairs_match_the_original_row_to_the_bit_for_each_vector() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0x71d3_0e9a_c4b2_58f1);
    let levels = [4u8, 12, 20, 28, 36, 44, 52, 62];
    let grid: [u32; 256] =
        std::array::from_fn(|_| u32::from_le_bytes(std::array::from_fn(|_| levels[(rng.byte() & 7) as usize])));
    let mut checked = 0;
    for sbs in [0, 1, 2, 3, 4, 5, 8, 10] {
        for trial in 0..60 {
            let mut original = vec![0u8; sbs * iq3_xxs::SUPERBLOCK];
            for blk in original.as_chunks_mut::<{ iq3_xxs::SUPERBLOCK }>().0 {
                blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                for b in &mut blk[2..] {
                    *b = rng.byte();
                }
            }
            let packed = iq3_xxs_nibble::repack(&original, &grid);
            let vectors: [(Vec<Block>, Vec<f32>); 2] = std::array::from_fn(|_| {
                let quants = (0..sbs * 8).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
                let scales = (0..sbs * 8)
                    .map(|_| {
                        let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                        unit * 2f32.powi((rng.next() % 21) as i32 - 10)
                    })
                    .collect();
                (quants, scales)
            });
            let quants = kgen_harness::interleave(&vectors[0].0, &vectors[1].0);
            let scales = kgen_harness::interleave(&vectors[0].1, &vectors[1].1);
            let want = vectors.each_ref().map(|(q, s)| iq3_xxs::reference(&original, q, s, &grid));
            for (name, kernel) in iq3_xxs_nibble::PAIRS {
                // Its frame is aligned inside, whatever the caller's stack: called from a few
                // depths, it has to be right from each.
                let mut got = [0.0; 2];
                at_depth(trial % 8, &mut || got = iq3_xxs_nibble::dot2(kernel, &packed, &quants, &scales));
                for v in 0..2 {
                    assert_eq!(
                        got[v].to_bits(),
                        want[v].to_bits(),
                        "IQ3_XXS nibble {name}, vector {v}, {sbs} superblocks, trial {trial}: {} against {}",
                        got[v],
                        want[v]
                    );
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 8 * 60 * iq3_xxs_nibble::PAIRS.len());
}

#[test]
fn iq4_nl_matches_its_reference_to_the_bit() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0xb2ec_6117_4ca9_e503);
    for groups in [0, 1, 2, 3, 5, 10] {
        for trial in 0..60 {
            let mut row = vec![0u8; groups * iq4_nl::GROUP];
            for blk in row.as_chunks_mut::<18>().0 {
                blk[..2].copy_from_slice(&finite_half(&mut rng).to_le_bytes());
                for b in &mut blk[2..] {
                    *b = rng.byte();
                }
            }
            let blocks = 4 * groups;
            let quants: Vec<Block> = (0..blocks).map(|_| Block(std::array::from_fn(|_| rng.byte() as i8))).collect();
            let scales: Vec<f32> = (0..blocks)
                .map(|_| {
                    let unit = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    unit * 2f32.powi((rng.next() % 21) as i32 - 10)
                })
                .collect();
            let want = iq4_nl::reference(&row, &quants, &scales);
            for (name, kernel) in iq4_nl::KERNELS {
                let got = iq4_nl::dot(kernel, &row, &quants, &scales);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "IQ4_NL {name}, {groups} groups, trial {trial}: {got} against {want}"
                );
            }
        }
    }
}

/// BF16 bits with an exponent within 2^-40 of one or a subnormal's, or a signed zero: no sum of
/// their products overflows.
fn bf16_bits(rng: &mut Rng) -> u16 {
    let sign = (rng.next() >> 63) as u16;
    let man = (rng.next() >> 57) as u16;
    let exp = match rng.next() % 16 {
        0 => 0,
        1 => return sign << 15,
        _ => (127 - 40 + rng.next() % 81) as u16,
    };
    sign << 15 | exp << 7 | man
}

#[test]
fn bf16_pairs_match_their_reference_to_the_bit() {
    if !supported() {
        eprintln!("skipped: no AVX2, FMA or F16C");
        return;
    }
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut checked = 0;
    for cols in [32, 64, 96, 320, 2560] {
        for rows in [2, 4, 6, 10] {
            for trial in 0..10 {
                let w: Vec<u16> = (0..rows * cols).map(|_| bf16_bits(&mut rng)).collect();
                let x: Vec<f32> = (0..cols)
                    .map(|_| f32::from_bits(u32::from(bf16_bits(&mut rng)) << 16 | (rng.next() >> 48) as u32))
                    .collect();
                let want: Vec<f32> = w.chunks(cols).map(|row| bf16::reference(row, &x)).collect();
                for (name, kernel) in bf16::KERNELS {
                    let mut got = vec![f32::NAN; rows];
                    bf16::rows(kernel, &w, &x, &mut got);
                    for (r, (g, want)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            want.to_bits(),
                            "{name}, {rows} rows of {cols}, trial {trial}, row {r}: {g} against {want}"
                        );
                    }
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 5 * 4 * 10 * bf16::KERNELS.len());
}

#[test]
fn bf16_two_vectors_match_the_reference_for_each_to_the_bit() {
    if !supported() {
        return;
    }
    let mut rng = Rng(0x1f83_d9ab_5be0_cd19);
    let mut checked = 0;
    for cols in [32, 64, 96, 320, 2560] {
        for rows in [1, 2, 3, 7] {
            for trial in 0..10 {
                let w: Vec<u16> = (0..rows * cols).map(|_| bf16_bits(&mut rng)).collect();
                let x: Vec<f32> = (0..2 * cols)
                    .map(|_| f32::from_bits(u32::from(bf16_bits(&mut rng)) << 16 | (rng.next() >> 48) as u32))
                    .collect();
                for (name, kernel) in bf16::TWOS {
                    // Room between the two outputs, so the gap is not just the rows.
                    let mut out = vec![f32::NAN; 3 * rows + 5];
                    let (first, rest) = out.split_at_mut(rows);
                    let second = &mut rest[rows + 5..];
                    bf16::two(kernel, &w, &x, [first, second]);
                    for (v, got) in [&out[..rows], &out[2 * rows + 5..]].into_iter().enumerate() {
                        let x = &x[v * cols..][..cols];
                        for (r, (g, row)) in got.iter().zip(w.chunks(cols)).enumerate() {
                            let want = bf16::reference(row, x);
                            assert_eq!(
                                g.to_bits(),
                                want.to_bits(),
                                "{name}, {rows} rows of {cols}, trial {trial}, vector {v}, row {r}: {g} against {want}"
                            );
                        }
                    }
                    assert!(out[rows..2 * rows + 5].iter().all(|v| v.is_nan()), "{name} wrote between the outputs");
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 5 * 4 * 10 * bf16::TWOS.len());
}

#[test]
fn bf16_triples_match_their_reference_to_the_bit() {
    if !supported() {
        return;
    }
    let mut rng = Rng(0x6789_0413_26da_13f2);
    for cols in [32, 64, 96, 320, 2560] {
        for rows in [3, 6, 9, 15] {
            for trial in 0..10 {
                let w: Vec<u16> = (0..rows * cols).map(|_| bf16_bits(&mut rng)).collect();
                let x: Vec<f32> = (0..cols)
                    .map(|_| f32::from_bits(u32::from(bf16_bits(&mut rng)) << 16 | (rng.next() >> 48) as u32))
                    .collect();
                let want: Vec<f32> = w.chunks(cols).map(|row| bf16::reference(row, &x)).collect();
                let mut got = vec![f32::NAN; rows];
                bf16::triples(&w, &x, &mut got);
                for (r, (g, want)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g.to_bits(), want.to_bits(), "{rows} rows of {cols}, trial {trial}, row {r}");
                }
            }
        }
    }
}
