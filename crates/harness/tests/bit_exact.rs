//! Every generated kernel against its scalar reference, to the bit, on random rows.

use kgen_harness::iq2_xxs::{KERNELS, SUPERBLOCK, dot, reference};
use kgen_harness::{Block, supported};

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
                for (ahead, kernel) in KERNELS {
                    let got = dot(kernel, &row, &quants, &scales, &grid);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "ahead {ahead}, {sbs} superblocks, trial {trial}, wild grid {wild}: {got} against {want}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 2 * 8 * 40 * KERNELS.len());
}
