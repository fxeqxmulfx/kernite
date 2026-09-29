//! `Q4_K` rows against a group of two or four activation vectors, every block of weights decoded
//! once for all of them.
//!
//! The arithmetic is the original row kernel's, to the bit: per block and vector one
//! `vpmaddubsw`, one `vpmaddwd`, one `vcvtdq2ps` and one fused multiply-add at the product of the
//! weight scale and the activation scale, each vector into an accumulator of its own; the
//! offsets, `-dmin * m` against the activations' half sums, into accumulators of their own, all
//! added at the end. What changes is where the work goes. The products of the scales are not
//! made a block at a time: a superblock's are made in a few wide multiplications and stored,
//! and each block's fused multiply-add picks its own out of that store with a broadcast load,
//! which takes no execution port. The scales are decoded and the products made for the next
//! superblock while this one's blocks run.

use kgen_core::Program;
use kgen_x86::{Addr, Asm, Gpr, Op, Ymm};

/// Bytes of a `Q4_K` superblock: `d`, `dmin`, twelve bytes of 6-bit scales and mins, 128 of nibbles.
const BYTES: i32 = 144;
/// Where a half of the scratch slot keeps the superblock's `[d * sc, -dmin * m]` pairs, and
/// where the products with the activations' scales.
const PAIRS: i32 = 0;
const PRODUCTS: i32 = 64;
/// A half of the scratch slot: the pairs, then a scale product for every block and vector.
const HALF: i32 = 256;

/// `Q4_K` against `vectors` (2 or 4) activation vectors, one row a call. With `spread` the next
/// superblock's scales are decoded among the blocks of this one; without, in one piece before them.
///
/// `fn(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, half_sums: *const f32,
/// out: *mut f32)`
///
/// The vectors lie block by block together, as a group of that width is stored: block `b` of
/// vector `k` at byte `32 * (vectors * b + k)` of `quants`, its scale at float `vectors * b + k`
/// of `scales`. The half sums lie vector by vector, sixteen floats a superblock, from
/// `half_sums` for vector 0. `out[k]` gets the bits the one-vector kernel gives vector `k`.
pub fn q4_k(vectors: usize, spread: bool) -> Program<Op> {
    assert!(matches!(vectors, 2 | 4), "{vectors} vectors");
    let g = vectors as i32;
    let mut a = Asm::new(format!("q4_k_x{vectors}{}", if spread { "" } else { "_flat" }));
    let row = a.param("row");
    let n = a.param("sbs");
    let quants = a.param("quants");
    let scales = a.param("scales");
    let sums = a.param("half_sums");
    let out = a.param("out");
    let (empty, top, last) = (a.label(), a.label(), a.label());
    a.test(n, n);
    a.jz(empty);

    a.note("constants");
    let low = a.ymm("nibble_mask");
    let one = a.ymm("one16");
    let low_at = a.u32s(&[0x0f0f_0f0f]);
    let one_at = a.u32s(&[0x0001_0001]);
    a.vpbroadcastd(low, low_at);
    a.vpbroadcastd(one, one_at);

    // One pointer to each vector's half sums, which lie a row's length apart.
    let mut hs = vec![sums];
    if vectors > 1 {
        let stride = a.gpr("stride");
        a.mov(stride, n);
        a.shl(stride, 6);
        for k in 1..vectors {
            let next = a.gpr(format!("sums{k}"));
            a.lea(next, hs[k - 1].at(0).index(stride, 1));
            hs.push(next);
        }
    }

    let acc: Vec<Ymm> = (0..vectors).map(|k| a.ymm(format!("acc{k}"))).collect();
    let offs = a.slot(32 * g as u32, 32);
    for (k, &v) in acc.iter().enumerate() {
        a.zero(v);
        let zero = a.ymm("zero");
        a.zero(zero);
        a.store_vmovaps(offs.at(32 * k as i32), zero);
    }

    // The scratch a superblock's products wait in: this one's half and the next one's, which
    // trade places by flipping one address bit.
    let slot = a.slot(2 * HALF as u32, 2 * HALF as u32);
    let (cur, nxt) = (a.gpr("cur"), a.gpr("nxt"));
    a.lea(cur, slot.at(0));
    a.lea(nxt, slot.at(HALF));

    let k = Constants::new(&mut a);
    prepare(&mut a, &k, row, scales, g, 0, cur);
    a.dec(n);
    a.jz(last);

    a.place(top);
    if spread {
        let mut next = a.capture(|a| prepare(a, &k, row, scales, g, 1, nxt));
        let share = next.len().div_ceil(8);
        blocks(&mut a, row, quants, cur, &acc, offs, &hs, low, one, g, Some((&mut next, share)));
    } else {
        prepare(&mut a, &k, row, scales, g, 1, nxt);
        blocks(&mut a, row, quants, cur, &acc, offs, &hs, low, one, g, None);
    }
    a.add(row, BYTES);
    a.add(quants, 256 * g);
    a.add(scales, 32 * g);
    for &h in &hs {
        a.add(h, 64);
    }
    a.xor(cur, HALF);
    a.xor(nxt, HALF);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    blocks(&mut a, row, quants, cur, &acc, offs, &hs, low, one, g, None);
    for (v, &acc) in acc.iter().enumerate() {
        let sum = a.ymm("sum");
        a.vaddps(sum, acc, offs.at(32 * v as i32));
        let high = a.ymm("high");
        a.vextractf128(high.xmm(), sum, 1);
        a.vaddps(sum.xmm(), sum.xmm(), high.xmm());
        let pair = a.ymm("pair");
        a.vmovhlps(pair.xmm(), sum.xmm(), sum.xmm());
        a.vaddps(sum.xmm(), sum.xmm(), pair.xmm());
        a.vmovshdup(pair.xmm(), sum.xmm());
        let result = a.ymm("result");
        a.vaddss(result.xmm(), sum.xmm(), pair.xmm());
        a.store_vmovss(out.at(4 * v as i32), result.xmm());
    }
    a.ret();

    a.place(empty);
    let nothing = a.ymm("nothing");
    a.zero(nothing);
    for v in 0..vectors {
        a.store_vmovss(out.at(4 * v as i32), nothing.xmm());
    }
    a.ret();
    a.finish()
}

/// What decoding a superblock's scales takes, as bytes in memory.
struct Constants {
    /// The low six bits of every byte, and bits 4 and 5 of every byte.
    six: Addr,
    top_two: Addr,
    nibble: Addr,
    /// The shifts that take a byte's low nibble and its high one: `[0, 4, 0, 4]`.
    halves: Addr,
    /// The half `dmin`'s sign bit, to negate it with.
    sign: Addr,
    /// Bytes `[sc0, m0, sc1, m1, sc2, m2, sc3, m3]` and the same for blocks 4 to 7, from
    /// `[sc0..sc7, m0..m7]`.
    first_four: Addr,
    last_four: Addr,
}

impl Constants {
    fn new(a: &mut Asm) -> Self {
        let interleave = |from: u8| -> Vec<u8> {
            let mut bytes = vec![0x80u8; 16];
            for j in 0..4u8 {
                bytes[2 * j as usize] = from + j;
                bytes[2 * j as usize + 1] = 8 + from + j;
            }
            bytes
        };
        Constants {
            six: a.u32s(&[0x3f3f_3f3f; 4]),
            top_two: a.u32s(&[0x3030_3030; 4]),
            nibble: a.u32s(&[0x0f0f_0f0f; 4]),
            halves: a.u32s(&[0, 4, 0, 4]),
            sign: a.u32s(&[0x8000_0000; 4]),
            first_four: a.u8s(&interleave(0)),
            last_four: a.u8s(&interleave(4)),
        }
    }
}

/// Superblock `+sb`'s scales decoded, stored to the half of the slot at `to`: the eight
/// `[d * sc, -dmin * m]` pairs, then every block's scale product with every vector's scale.
///
/// The twelve bytes of 6-bit values come apart with a handful of byte operations on all of
/// them at once. Sub-blocks 0 to 3 keep six bits of their scale and min in a byte; 4 to 7 keep
/// four in the nibbles of the last four bytes and the top two of each in the bits 6 and 7 of the
/// first eight.
fn prepare(a: &mut Asm, k: &Constants, row: Gpr, scales: Gpr, g: i32, sb: i32, to: Gpr) {
    a.note(format!("scales of superblock +{sb}"));
    let at = BYTES * sb;
    let packed = a.ymm("packed");
    a.vmovdqu(packed.xmm(), row.at(at + 4));
    let low = a.ymm("low_six");
    a.vpand(low.xmm(), packed.xmm(), k.six);
    let top = a.ymm("top_bits");
    a.vpsrlw(top.xmm(), packed.xmm(), 2);
    a.vpand(top.xmm(), top.xmm(), k.top_two);
    let last = a.ymm("last_four");
    a.vpbroadcastd(last.xmm(), row.at(at + 12));
    a.vpsrlvd(last.xmm(), last.xmm(), k.halves);
    a.vpand(last.xmm(), last.xmm(), k.nibble);
    a.vpor(top.xmm(), top.xmm(), last.xmm());
    // [sc0..sc7, m0..m7].
    a.vpunpckldq(low.xmm(), low.xmm(), top.xmm());
    let (first, second) = (a.ymm("blocks_0_3"), a.ymm("blocks_4_7"));
    a.vpshufb(first.xmm(), low.xmm(), k.first_four);
    a.vpshufb(second.xmm(), low.xmm(), k.last_four);
    a.vpmovzxbd_reg(first, first.xmm());
    a.vpmovzxbd_reg(second, second.xmm());
    a.vcvtdq2ps(first, first);
    a.vcvtdq2ps(second, second);
    // [d, -dmin] four times over.
    let halves = a.ymm("d_dmin");
    a.vpbroadcastd(halves.xmm(), row.at(at));
    a.vpxor(halves.xmm(), halves.xmm(), k.sign);
    let both = a.ymm("d_and_minus_dmin");
    a.vcvtph2ps(both, halves.xmm());
    a.vmulps(first, first, both);
    a.vmulps(second, second, both);
    a.store_vmovaps(to.at(PAIRS), first);
    a.store_vmovaps(to.at(PAIRS + 32), second);

    // The products with the activations' scales, two blocks (four for two vectors) a register.
    let scales_at = 32 * g * sb;
    if g == 4 {
        // Each register takes the scale of two blocks, four times over: the even floats of
        // `first` or `second` are the scales, and these say which two.
        let pairs = [a.u32s(&[0, 0, 0, 0, 2, 2, 2, 2]), a.u32s(&[4, 4, 4, 4, 6, 6, 6, 6])];
        for i in 0..4i32 {
            let idx = a.ymm("pair_of_blocks");
            a.vmovdqa(idx, pairs[i as usize % 2]);
            let product = a.ymm("scale_products");
            a.vpermps(product, idx, if i < 2 { first } else { second });
            a.vmulps(product, product, scales.at(scales_at + 32 * i));
            a.store_vmovaps(to.at(PRODUCTS + 32 * i), product);
        }
    } else {
        for (i, pairs) in [first, second].into_iter().enumerate() {
            let product = a.ymm("scale_products");
            a.vmovsldup(product, pairs);
            a.vmulps(product, product, scales.at(scales_at + 32 * i as i32));
            a.store_vmovaps(to.at(PRODUCTS + 32 * i as i32), product);
        }
    }
}

/// The eight blocks of the superblock at `row`, against every vector, and its offsets.
#[allow(clippy::too_many_arguments)]
fn blocks(
    a: &mut Asm,
    row: Gpr,
    quants: Gpr,
    cur: Gpr,
    acc: &[Ymm],
    offs: kgen_x86::Slot,
    hs: &[Gpr],
    low: Ymm,
    one: Ymm,
    g: i32,
    mut next: Option<(&mut kgen_x86::Stream, usize)>,
) {
    a.note("offsets");
    let (lo, hi) = (a.ymm("offsets_lo"), a.ymm("offsets_hi"));
    a.vmovshdup(lo, cur.at(PAIRS));
    a.vmovshdup(hi, cur.at(PAIRS + 32));
    for (v, &h) in hs.iter().enumerate() {
        let off = a.ymm("off");
        a.vmovdqa(off, offs.at(32 * v as i32));
        a.vfmadd231ps(off, lo, h.at(0));
        a.vfmadd231ps(off, hi, h.at(32));
        a.store_vmovaps(offs.at(32 * v as i32), off);
    }
    for pair in 0..4 {
        let codes = a.ymm("codes");
        a.vmovdqu(codes, row.at(16 + 32 * pair));
        let wu = [a.ymm("low_nibbles"), a.ymm("high_nibbles")];
        a.vpand(wu[0], codes, low);
        a.vpsrlw(wu[1], codes, 4);
        a.vpand(wu[1], wu[1], low);
        for (half, &wu) in wu.iter().enumerate() {
            let j = 2 * pair + half as i32;
            a.note(format!("block {j}"));
            if let Some((stream, share)) = next.as_mut() {
                a.splice(stream, *share);
            }
            for (v, &acc) in acc.iter().enumerate() {
                let p = a.ymm("products");
                a.vpmaddubsw(p, wu, quants.at(32 * (g * j + v as i32)));
                a.vpmaddwd(p, p, one);
                a.vcvtdq2ps(p, p);
                let scale = a.ymm("block_scale");
                a.vbroadcastss(scale, cur.at(PRODUCTS + 4 * (g * j + v as i32)));
                a.vfmadd231ps(acc, p, scale);
            }
        }
    }
}
