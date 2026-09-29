//! `Q6_K` rows on the front-scale layout against a group of two or four activation vectors, every
//! block of weights decoded once for all of them.
//!
//! The arithmetic is the original row kernel's, to the bit: per block and vector one
//! `vpmaddubsw`, one `vpmaddwd`, one `vcvtdq2ps` and one fused multiply-add, each vector into an
//! accumulator of its own, at the product of the activation's scale and the weight scale, which
//! is `d` times a signed byte for each sixteen weights; the offsets, `-32 * d * sc` against the
//! activations' half sums, into accumulators of their own, all added at the end.
//!
//! As in [`crate::k4`] the scale products are made a superblock ahead in a few wide
//! multiplications and stored. A block's two weight scales `[a, b]` reach its eight lanes as
//! `[a, a, a, a, b, b, b, b]`, four to each sixteen weights, so a block's product with a vector's
//! scale `x` is `[a x, a x, a x, a x, b x, b x, b x, b x]`: the block's products with every
//! vector's scale are made and stored as `[a x0, a x1, .., b x0, b x1, ..]` and each fused
//! multiply-add takes its vector's out of that with one `vpermps`, which runs on a port the
//! multiplications and conversions leave free.

use kgen_core::Program;
use kgen_x86::{Addr, Asm, Gpr, Op, Slot, Stream, Ymm};

use crate::k4::Shape;

/// A superblock on the front layout: the half `d`, sixteen signed scales, padding to 32 bytes,
/// 128 bytes of low nibbles, then 64 of bit pairs.
const BYTES: i32 = 224;
const SCALES: i32 = 2;
const NIBBLES: i32 = 32;
const PAIRS: i32 = 160;

/// Where a half of the scratch slot keeps the eight blocks' scale products, each a cache line
/// of its own so that the loads that pick a vector's out of it, which begin up to three floats in,
/// never straddle two; the sixteen scales `d * sc`, and the same times -32, which the offsets
/// take.
const PRODUCTS: i32 = 0;
const WIDE: i32 = 512;
const FACTORS: i32 = 576;
/// A half of the scratch slot.
const HALF: i32 = 1024;

/// `Q6_K` on the front layout against `shape.vectors` activation vectors, one row a call.
///
/// `fn(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, half_sums: *const f32,
/// out: *mut f32)`, with the vectors laid out as [`crate::q4_k`]'s are: block `b` of vector `k`
/// at byte `32 * (vectors * b + k)` of `quants`, its scale at float `vectors * b + k` of `scales`,
/// and the half sums vector by vector, sixteen floats a superblock. `out[k]` gets the bits the
/// one-vector kernel gives vector `k`.
pub fn q6_k_few(shape: Shape) -> Program<Op> {
    q6_k_of(shape, shape.vectors)
}

/// [`q6_k_few`] on vectors that lie in a group of `group`, which is `shape.vectors` or, for four,
/// eight: the layout above with `group` in place of `vectors`. A group of eight is two calls,
/// the second with `quants` moved on 128 bytes, `scales` on four floats and `half_sums` on four
/// vectors' worth, and `out` on four floats.
pub fn q6_k_of(shape: Shape, group: usize) -> Program<Op> {
    let Shape { vectors, spread, ahead, hint } = shape;
    assert!(matches!(vectors, 2 | 4), "{vectors} vectors");
    assert!(group == vectors || (vectors == 4 && group == 8), "{vectors} vectors of a group of {group}");
    let g = vectors as i32;
    let mut name = format!("q6_k_front_x{vectors}{}", if spread { "" } else { "_flat" });
    if group != vectors {
        name += &format!("_of{group}");
    }
    if ahead > 0 {
        name += &format!("_ahead{ahead}_{}", hint.suffix());
    }
    let mut a = Asm::new(name);
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
    let k = Constants::new(&mut a, vectors);

    // One pointer to each vector's half sums, which lie a row's length apart.
    let mut hs = vec![sums];
    let stride = a.gpr("stride");
    a.mov(stride, n);
    a.shl(stride, 6);
    for v in 1..vectors {
        let next = a.gpr(format!("sums{v}"));
        a.lea(next, hs[v - 1].at(0).index(stride, 1));
        hs.push(next);
    }

    let acc: Vec<Ymm> = (0..vectors).map(|v| a.ymm(format!("acc{v}"))).collect();
    let offs = a.slot(32 * g as u32, 32);
    for (v, &acc) in acc.iter().enumerate() {
        a.zero(acc);
        let zero = a.ymm("zero");
        a.zero(zero);
        a.store_vmovaps(offs.at(32 * v as i32), zero);
    }

    // The scratch a superblock's products wait in: this one's half and the next one's, which
    // trade places by flipping one address bit.
    let slot = a.slot(2 * HALF as u32, 2 * HALF as u32);
    let (cur, nxt) = (a.gpr("cur"), a.gpr("nxt"));
    a.lea(cur, slot.at(0));
    a.lea(nxt, slot.at(HALF));

    let (one, lane) = (a.ymm("one16"), a.ymm("lane"));
    a.vmovdqa(one, k.one);
    a.vmovdqa(lane, k.lane);
    let l = Loop { shape, group: group as i32, row, quants, cur, acc, offs, hs, one, lane };
    prepare(&mut a, &k, &l, scales, 0, cur);
    a.dec(n);
    a.jz(last);

    a.place(top);
    if spread {
        let mut next = a.capture(|a| prepare(a, &k, &l, scales, 1, nxt));
        let share = next.len().div_ceil(8);
        blocks(&mut a, &k, &l, true, Some((&mut next, share)));
    } else {
        prepare(&mut a, &k, &l, scales, 1, nxt);
        blocks(&mut a, &k, &l, true, None);
    }
    a.add(row, BYTES);
    a.add(quants, 256 * group as i32);
    a.add(scales, 32 * group as i32);
    for &h in &l.hs {
        a.add(h, 64);
    }
    a.xor(cur, HALF);
    a.xor(nxt, HALF);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    blocks(&mut a, &k, &l, false, None);
    for (v, &acc) in l.acc.iter().enumerate() {
        let sum = a.ymm("sum");
        a.vaddps(sum, acc, l.offs.at(32 * v as i32));
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

struct Loop {
    shape: Shape,
    /// Vectors in the group the activations lie in, which the strides through them follow.
    group: i32,
    row: Gpr,
    quants: Gpr,
    cur: Gpr,
    acc: Vec<Ymm>,
    offs: Slot,
    hs: Vec<Gpr>,
    /// A 1 in every 16-bit lane, and the `vpermps` indices that take a vector's product with
    /// `a` to the low four lanes and with `b` to the high four from a load that starts at its
    /// first.
    one: Ymm,
    lane: Ymm,
}

/// What the loop reads from memory that never changes.
struct Constants {
    /// 0x0f and 0x30 in every byte, and a 1 in every 16-bit lane.
    nibble: Addr,
    top_two: Addr,
    one: Addr,
    minus32: Addr,
    /// `vpermps` indices that spread a block's pair of scales `[a, b]` out of the four pairs of
    /// half a superblock's sixteen, for each block of the four: to `[a, a, a, a, b, b, b, b]`,
    /// or with two vectors `[a, a, b, b, a, a, b, b]`.
    spread: [Addr; 4],
    /// The indices that pick a vector's product with `a` into the low four lanes and its
    /// product with `b` into the high four, out of `[a x0, a x1, .., b x0, b x1, ..]` loaded from
    /// where the vector's `a x` is.
    lane: Addr,
}

impl Constants {
    fn new(a: &mut Asm, vectors: usize) -> Self {
        let g = vectors as u32;
        let spread = std::array::from_fn(|j| {
            let index: Vec<u32> = (0..8).map(|l| 2 * j as u32 + if g == 4 { l / 4 } else { l / 2 % 2 }).collect();
            a.u32s(&index)
        });
        let lane = a.u32s(&(0..8).map(|l| g * (l / 4)).collect::<Vec<u32>>());
        Constants {
            nibble: a.u32s(&[0x0f0f_0f0f; 8]),
            top_two: a.u32s(&[0x3030_3030; 8]),
            one: a.u32s(&[0x0001_0001; 8]),
            minus32: a.f32s(&[-32.0; 8]),
            spread,
            lane,
        }
    }
}

/// Superblock `+sb`'s scale products, stored to the half of the slot at `to`: for each block the
/// scales `[a, b]` times every vector's scale, the sixteen scales `d * sc` themselves, and those
/// times -32.
fn prepare(a: &mut Asm, k: &Constants, l: &Loop, scales: Gpr, sb: i32, to: Gpr) {
    a.note(format!("scales of superblock +{sb}"));
    let (row, g, group) = (l.row, l.shape.vectors as i32, l.group);
    let at = BYTES * sb;
    let half = a.ymm("half_d");
    a.vpbroadcastw(half.xmm(), row.at(at));
    let d = a.ymm("d");
    a.vcvtph2ps(d, half.xmm());
    for h in 0..2 {
        let wide = a.ymm("d_times_scales");
        a.vpmovsxbd(wide, row.at(at + SCALES + 8 * h));
        a.vcvtdq2ps(wide, wide);
        a.vmulps(wide, wide, d);
        a.store_vmovaps(to.at(WIDE + 32 * h), wide);
        a.vmulps(wide, wide, k.minus32);
        a.store_vmovaps(to.at(FACTORS + 32 * h), wide);
    }
    let scales_at = 32 * group * sb;
    for j in 0..8i32 {
        let idx = a.ymm("spread");
        a.vmovdqa(idx, k.spread[(j % 4) as usize]);
        let product = a.ymm("scale_products");
        a.vpermps(product, idx, to.at(WIDE + 32 * (j / 4)));
        let x = a.ymm("activation_scales");
        if g == 4 {
            a.vbroadcastf128(x, scales.at(scales_at + 4 * group * j));
        } else {
            a.vbroadcastsd(x, scales.at(scales_at + 4 * group * j));
        }
        a.vmulps(product, product, x);
        a.store_vmovaps(to.at(PRODUCTS + 64 * j), product);
    }
}

/// The eight blocks of the superblock at `row`, against every vector, and its offsets. With
/// `more`, another superblock follows, whose row's reads are asked for ahead of time when the
/// shape says so.
fn blocks(a: &mut Asm, k: &Constants, l: &Loop, more: bool, mut next: Option<(&mut Stream, usize)>) {
    let Loop { shape, group, row, quants, cur, acc, offs, hs, one, lane } = l;
    let (row, quants, cur, offs, one, lane) = (*row, *quants, *cur, *offs, *one, *lane);
    let group = *group;
    if more && shape.ahead > 0 {
        for line in 0..(BYTES + 63) / 64 {
            a.prefetch(shape.hint, row.at(shape.ahead + 64 * line));
        }
    }
    a.note("offsets");
    let (lo, hi) = (a.ymm("offsets_lo"), a.ymm("offsets_hi"));
    a.vmovdqa(lo, cur.at(FACTORS));
    a.vmovdqa(hi, cur.at(FACTORS + 32));
    for (v, &h) in hs.iter().enumerate() {
        let off = a.ymm("off");
        a.vmovdqa(off, offs.at(32 * v as i32));
        a.vfmadd231ps(off, lo, h.at(0));
        a.vfmadd231ps(off, hi, h.at(32));
        a.store_vmovaps(offs.at(32 * v as i32), off);
    }
    for half in 0..2 {
        let ql = [a.ymm("nibbles_even"), a.ymm("nibbles_odd")];
        let qh = a.ymm("bit_pairs");
        a.vmovdqu(ql[0], row.at(NIBBLES + 64 * half));
        a.vmovdqu(ql[1], row.at(NIBBLES + 64 * half + 32));
        a.vmovdqu(qh, row.at(PAIRS + 32 * half));
        for i in 0..4 {
            let j = 4 * half + i;
            a.note(format!("block {j}"));
            // Bits 0 to 3 from a nibble, 4 and 5 from bit pair `i` of the plane.
            let wu = a.ymm("weights");
            if i < 2 {
                a.vpand(wu, ql[i as usize], k.nibble);
            } else {
                a.vpsrlw(wu, ql[i as usize % 2], 4);
                a.vpand(wu, wu, k.nibble);
            }
            let bits = a.ymm("bit_pair");
            match i {
                0 => a.vpsllw(bits, qh, 4),
                1 => a.vpsllw(bits, qh, 2),
                2 => {}
                _ => a.vpsrlw(bits, qh, 2),
            }
            a.vpand(bits, if i == 2 { qh } else { bits }, k.top_two);
            a.vpor(wu, wu, bits);
            if let Some((stream, share)) = next.as_mut() {
                a.splice(stream, *share);
            }
            for (v, &acc) in acc.iter().enumerate() {
                let p = a.ymm("products");
                a.vpmaddubsw(p, wu, quants.at(32 * (group * j + v as i32)));
                a.vpmaddwd(p, p, one);
                a.vcvtdq2ps(p, p);
                let scale = a.ymm("block_scale");
                a.vpermps(scale, lane, cur.at(PRODUCTS + 64 * j + 4 * v as i32));
                a.vfmadd231ps(acc, p, scale);
            }
        }
    }
}
