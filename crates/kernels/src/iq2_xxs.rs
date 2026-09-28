use kgen_core::Program;
use kgen_x86::{Addr, Asm, Gpr, Op, Ymm};

use crate::lanes::lanes_summed;

/// The constants every group uses, in registers.
struct Group {
    /// Byte `l` of a word to the eight bytes of weights `8l..8l + 8`.
    spread: Ymm,
    /// Bit `j % 8` for byte `j`.
    bit: Ymm,
    /// 0x7F, the largest signed byte, in every byte.
    max8: Ymm,
    one16: Ymm,
    /// How many of the two pairs of grid rows are loaded as pairs: see [`products`].
    pairs: usize,
}

/// The constants a superblock's scales and signs use once each, in memory.
struct Superblock {
    fields: [Addr; 4],
    tops: Addr,
    one: Addr,
    eighth: Addr,
}

/// The parameters.
struct Row {
    row: Gpr,
    quants: Gpr,
    scales: Gpr,
    grid: Gpr,
}

/// An `IQ2_XXS` row against one activation vector:
///
/// `fn(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, grid: *const u64) -> f32`
///
/// `row` holds `sbs` superblocks of 66 bytes, 256 weights each. `quants` and `scales` are
/// the activations, 32 bytes and one float to every 32 weights, and `quants` is 32-byte
/// aligned. `grid` is the format's 256 rows of eight levels.
///
/// The result is the bits of `dot_iq_ahead::<IQ2_XXS, 66, D>` in the `roofline` kernels:
/// each group's scale is `((d * (2k + 1)) * 0.125) * scale` of its activations; each
/// group's products are summed in 16-bit pairs, in 32-bit lanes, converted, and added in with
/// one fused multiply-add per group, in group order, into one accumulator; that is summed as
/// `hsum(acc + 0.0)`, the high 128 bits onto the low, then the high pair, then lane 1 onto
/// lane 0.
///
/// As there, each group's integer work runs `ahead` groups before its sum, and each
/// superblock's scales and signs one superblock before its groups. They go to one of two
/// 64-byte halves of a 128-byte slot, which `cur` and `nxt` swap between with an `xor`.
///
/// `pairs` of each group's two pairs of grid rows are loaded as pairs: see [`products`].
pub fn iq2_xxs(ahead: usize, pairs: usize) -> Program<Op> {
    assert!((1..8).contains(&ahead), "ahead {ahead}");
    assert!(pairs <= 2, "pairs {pairs}");
    let mut a = Asm::new(match pairs {
        0 => format!("iq2_xxs_ahead{ahead}"),
        _ => format!("iq2_xxs_ahead{ahead}_pairs{pairs}"),
    });
    let row = a.param("row");
    let n = a.param("sbs");
    let quants = a.param("quants");
    let scales = a.param("scales");
    let grid = a.param("grid");
    let r = Row { row, quants, scales, grid };
    let (empty, top, last) = (a.label(), a.label(), a.label());
    a.test(n, n);
    a.jz(empty);

    a.note("constants");
    let spread: Vec<u8> = (0..32).map(|j| j / 8).collect();
    let (spread_at, bit_at) = (a.u8s(&spread), a.u64s(&[0x8040_2010_0804_0201]));
    let (max8_at, one16_at) = (a.u32s(&[0x7F7F_7F7F]), a.u32s(&[0x0001_0001]));
    let k = Group { spread: a.ymm("spread"), bit: a.ymm("bit"), max8: a.ymm("max8"), one16: a.ymm("one16"), pairs };
    a.vmovdqa(k.spread, spread_at);
    a.vpbroadcastq(k.bit, bit_at);
    a.vpbroadcastd(k.max8, max8_at);
    a.vpbroadcastd(k.one16, one16_at);
    let s = Superblock {
        fields: [0x7F, 0x7F00, 0x7F_0000, 0x7F00_0000].map(|m| a.u32s(&[m; 8])),
        tops: a.u32s(&[0x8080_8080; 8]),
        one: a.u32s(&[1; 8]),
        eighth: a.f32s(&[0.125; 8]),
    };
    let acc = a.ymm("acc");
    a.zero(acc);
    let slot = a.slot(128, 128);
    let (cur, nxt) = (a.gpr("cur"), a.gpr("nxt"));
    a.lea(cur, slot.at(0));
    a.lea(nxt, slot.at(64));

    // `ring[g % 8]` holds group `g`'s products from when they are made until they are summed.
    let ring: [Ymm; 8] = std::array::from_fn(|g| a.ymm(format!("p{g}")));
    prepare(&mut a, &s, &r, 0, cur);
    for (g, &p) in ring.iter().enumerate().take(ahead) {
        products(&mut a, &k, &r, p, 0, cur, g as i32);
    }
    a.dec(n);
    a.jz(last);

    a.place(top);
    // The next superblock's signs and scales, chains of instructions that wait on each other,
    // spread over the groups: the signs over those before the first that reads them, the
    // scales, which only the next pass reads, over the rest, so that the two chains' variables
    // are not alive together.
    let steps = a.ymm("steps");
    let mut next_signs = a.capture(|a| group_signs(a, &s, &r, 1, steps, nxt.at(32)));
    let mut next_scales = a.capture(|a| group_scales(a, &s, &r, steps, 1, nxt));
    let (share, scales_share) = (next_signs.len().div_ceil(8 - ahead), next_scales.len().div_ceil(ahead));
    for g in 0..8 {
        if g < 8 - ahead {
            a.splice(&mut next_signs, share);
        } else {
            a.splice(&mut next_scales, scales_share);
        }
        match g + ahead {
            j if j < 8 => products(&mut a, &k, &r, ring[j], 0, cur, j as i32),
            j => products(&mut a, &k, &r, ring[j - 8], 1, nxt, (j - 8) as i32),
        }
        sum(&mut a, &k, acc, ring[g], cur, g as i32);
    }
    a.add(r.row, 66);
    a.add(r.quants, 256);
    a.add(r.scales, 32);
    a.xor(cur, 64);
    a.xor(nxt, 64);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    for g in 0..8 {
        if g + ahead < 8 {
            products(&mut a, &k, &r, ring[g + ahead], 0, cur, (g + ahead) as i32);
        }
        sum(&mut a, &k, acc, ring[g], cur, g as i32);
    }
    finish(&mut a, acc);

    a.place(empty);
    let nothing = a.ymm("nothing");
    a.zero(nothing);
    a.ret_f32(nothing.xmm());
    a.finish()
}

/// An `IQ2_XXS` row against two activation vectors, each group's grid rows and signs decoded
/// once for both:
///
/// `fn(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, grid: *const u64, out: *mut f32)`
///
/// The vectors lie block by block together, the way a group of two is stored: block `b` of the
/// first, then block `b` of the second, and their scales the same way. `out` gets the two dots,
/// each the bits [`iq2_xxs`] makes of its vector alone. A slot half holds a superblock's sixteen
/// scales, float `2g + v` for group `g` and vector `v`, then its eight words of signs.
pub fn iq2_xxs2(ahead: usize) -> Program<Op> {
    assert!((1..8).contains(&ahead), "ahead {ahead}");
    let mut a = Asm::new(format!("iq2_xxs2_ahead{ahead}"));
    let row = a.param("row");
    let n = a.param("sbs");
    let quants = a.param("quants");
    let scales = a.param("scales");
    let grid = a.param("grid");
    let out = a.param("out");
    let r = Row { row, quants, scales, grid };
    let (empty, top, last) = (a.label(), a.label(), a.label());
    a.test(n, n);
    a.jz(empty);

    a.note("constants");
    let spread: Vec<u8> = (0..32).map(|j| j / 8).collect();
    let (spread_at, bit_at) = (a.u8s(&spread), a.u64s(&[0x8040_2010_0804_0201]));
    let (max8_at, one16_at) = (a.u32s(&[0x7F7F_7F7F]), a.u32s(&[0x0001_0001]));
    let k = Group { spread: a.ymm("spread"), bit: a.ymm("bit"), max8: a.ymm("max8"), one16: a.ymm("one16"), pairs: 2 };
    a.vmovdqa(k.spread, spread_at);
    a.vpbroadcastq(k.bit, bit_at);
    a.vpbroadcastd(k.max8, max8_at);
    a.vpbroadcastd(k.one16, one16_at);
    let s = Superblock {
        fields: [0x7F, 0x7F00, 0x7F_0000, 0x7F00_0000].map(|m| a.u32s(&[m; 8])),
        tops: a.u32s(&[0x8080_8080; 8]),
        one: a.u32s(&[1; 8]),
        eighth: a.f32s(&[0.125; 8]),
    };
    let acc = [a.ymm("acc0"), a.ymm("acc1")];
    for v in acc {
        a.zero(v);
    }
    // Aligned to its whole size: the two halves trade places by flipping one address bit.
    let slot = a.slot(256, 256);
    let (cur, nxt) = (a.gpr("cur"), a.gpr("nxt"));
    a.lea(cur, slot.at(0));
    a.lea(nxt, slot.at(128));

    let ring: [[Ymm; 2]; 8] = std::array::from_fn(|g| [a.ymm(format!("p{g}a")), a.ymm(format!("p{g}b"))]);
    let steps = a.ymm("steps");
    group_signs(&mut a, &s, &r, 0, steps, cur.at(64));
    pair_scales(&mut a, &s, &r, steps, 0, cur);
    for (g, p) in ring.iter().enumerate().take(ahead) {
        pair_products(&mut a, &k, &r, p, 0, cur, g as i32);
    }
    a.dec(n);
    a.jz(last);

    a.place(top);
    // As in [`iq2_xxs`]: the next superblock's signs over the groups before the first that reads
    // them, its scales over the rest.
    let steps = a.ymm("steps");
    let mut next_signs = a.capture(|a| group_signs(a, &s, &r, 1, steps, nxt.at(64)));
    let mut next_scales = a.capture(|a| pair_scales(a, &s, &r, steps, 1, nxt));
    let (share, scales_share) = (next_signs.len().div_ceil(8 - ahead), next_scales.len().div_ceil(ahead));
    for g in 0..8 {
        // The sums first: the accumulators' chains are the longest, and the oldest uop ready
        // is the one a port takes.
        pair_sums(&mut a, &k, acc, ring[g], cur, g as i32);
        if g < 8 - ahead {
            a.splice(&mut next_signs, share);
        } else {
            a.splice(&mut next_scales, scales_share);
        }
        match g + ahead {
            j if j < 8 => pair_products(&mut a, &k, &r, &ring[j], 0, cur, j as i32),
            j => pair_products(&mut a, &k, &r, &ring[j - 8], 1, nxt, (j - 8) as i32),
        }
    }
    a.add(r.row, 66);
    a.add(r.quants, 512);
    a.add(r.scales, 64);
    a.xor(cur, 128);
    a.xor(nxt, 128);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    for g in 0..8 {
        pair_sums(&mut a, &k, acc, ring[g], cur, g as i32);
        if g + ahead < 8 {
            pair_products(&mut a, &k, &r, &ring[g + ahead], 0, cur, (g + ahead) as i32);
        }
    }
    for (v, acc) in acc.into_iter().enumerate() {
        let result = lanes_summed(&mut a, acc);
        a.store_vmovss(out.at(4 * v as i32), result);
    }
    a.ret();

    a.place(empty);
    let nothing = a.ymm("nothing");
    a.zero(nothing);
    a.store_vmovss(out.at(0), nothing.xmm());
    a.store_vmovss(out.at(4), nothing.xmm());
    a.ret();
    a.finish()
}

/// Superblock `sb` (0 this one, 1 the next)'s eight words of signs and eight group scales,
/// times the activations', to the 64 bytes at `to`.
fn prepare(a: &mut Asm, s: &Superblock, r: &Row, sb: i32, to: Gpr) {
    let steps = a.ymm("steps");
    group_signs(a, s, r, sb, steps, to.at(32));
    group_scales(a, s, r, steps, sb, to);
}

/// Superblock `sb`'s eight words of signs, to the 32 bytes at `to`, and `steps` of its group
/// scales.
///
/// Each group's second word holds its scale in the top nibble and four 7-bit fields of signs
/// below; `vshufps` and `vpermq` gather the eight second words in group order. `(w >> 27) | 1`
/// is `2 * (w >> 28) + 1`, the odd step.
fn group_signs(a: &mut Asm, s: &Superblock, r: &Row, sb: i32, steps: Ymm, to: Addr) {
    a.note(format!("signs of superblock +{sb}"));
    let (at, w) = (66 * sb, a.ymm("words"));
    a.vmovdqu(w, r.row.at(at + 2));
    a.vshufps(w, w, r.row.at(at + 34), 0xdd);
    a.vpermq(w, w, 0xd8);
    // Field `l`, bits `7l..7l + 7`, moved up `l` bits to the bottom of byte `l`.
    let (y, t) = (a.ymm("fields"), a.ymm("field"));
    a.vpand(y, w, s.fields[0]);
    a.vpaddd(t, w, w);
    a.vpand(t, t, s.fields[1]);
    a.vpor(y, y, t);
    a.vpslld(t, w, 2);
    a.vpand(t, t, s.fields[2]);
    a.vpor(y, y, t);
    a.vpslld(t, w, 3);
    a.vpand(t, t, s.fields[3]);
    a.vpor(y, y, t);
    a.vpsrld(steps, w, 27);
    // Bit 7 of each byte is the parity of the seven below: bit `k` of `z` is the XOR of bits
    // `k - 7..=k` of `y`, and what the shifts carry across bytes lands below the tops.
    let z = a.ymm("parity");
    a.vpaddw(z, y, y);
    a.vpxor(z, z, y);
    a.vpsllw(t, z, 2);
    a.vpxor(z, z, t);
    a.vpsllw(t, z, 4);
    a.vpxor(z, z, t);
    a.vpand(z, z, s.tops);
    a.vpor(y, y, z);
    a.store_vmovdqa(to, y);
}

/// Superblock `sb`'s eight group scales, from `steps` of them, times the activations', to the
/// 32 bytes at `to`.
fn group_scales(a: &mut Asm, s: &Superblock, r: &Row, steps: Ymm, sb: i32, to: Gpr) {
    a.note(format!("scales of superblock +{sb}"));
    let d = a.ymm("d");
    a.vpbroadcastw(d.xmm(), r.row.at(66 * sb));
    a.vcvtph2ps(d, d.xmm());
    let scale = a.ymm("scale");
    a.vpor(scale, steps, s.one);
    a.vcvtdq2ps(scale, scale);
    a.vmulps(scale, d, scale);
    a.vmulps(scale, scale, s.eighth);
    a.vmulps(scale, scale, r.scales.at(32 * sb));
    a.store_vmovaps(to.at(0), scale);
}

/// Superblock `sb`'s group scales, from `steps` of them, times each vector's, to the 64 bytes
/// at `to`: float `2g + v` for group `g` and vector `v`, each rounded as [`group_scales`]
/// rounds it for one vector.
fn pair_scales(a: &mut Asm, s: &Superblock, r: &Row, steps: Ymm, sb: i32, to: Gpr) {
    a.note(format!("pair scales of superblock +{sb}"));
    let d = a.ymm("d");
    a.vpbroadcastw(d.xmm(), r.row.at(66 * sb));
    a.vcvtph2ps(d, d.xmm());
    let scale = a.ymm("scale");
    a.vpor(scale, steps, s.one);
    a.vcvtdq2ps(scale, scale);
    a.vmulps(scale, d, scale);
    a.vmulps(scale, scale, s.eighth);
    // Groups 0, 1, 4, 5, 2, 3, 6, 7, so that `vshufps` pairs up each lane's first two, or its
    // last two, in group order: groups 0-3 each twice, then 4-7.
    a.vpermq(scale, scale, 0xd8);
    for (half, control) in [(0, 0x50), (1, 0xfa)] {
        let pair = a.ymm("pair_scale");
        a.vshufps(pair, scale, scale, control);
        a.vmulps(pair, pair, r.scales.at(64 * sb + 32 * half));
        a.store_vmovaps(to.at(32 * half), pair);
    }
}

/// Group `g` of superblock `sb`: its grid rows and signs once, then each vector's signed
/// activations times the levels, in 16-bit pairs, into `p[v]`. Its signs are word `g` of the
/// 32 bytes at `from + 64`.
#[allow(clippy::too_many_arguments)]
fn pair_products(a: &mut Asm, k: &Group, r: &Row, p: &[Ymm; 2], sb: i32, from: Gpr, g: i32) {
    a.note(format!("products of group {g} of superblock +{sb}"));
    let at = 66 * sb + 2 + 8 * g;
    let index: [Gpr; 4] = std::array::from_fn(|l| a.gpr(format!("index{l}")));
    for (l, &i) in index.iter().enumerate() {
        a.movzx_byte(i, r.row.at(at + l as i32));
    }
    let grid = |i: Gpr| r.grid.at(0).index(i, 8);
    let (levels, rows) = (a.ymm("levels"), a.ymm("rows"));
    a.vmovq(levels.xmm(), grid(index[0]));
    a.vmovhps(levels.xmm(), levels.xmm(), grid(index[1]));
    a.vmovq(rows.xmm(), grid(index[2]));
    a.vmovhps(rows.xmm(), rows.xmm(), grid(index[3]));
    a.vinserti128(levels, levels, rows.xmm(), 1);
    let signs = a.ymm("signs");
    a.vpbroadcastd(signs, from.at(64 + 4 * g));
    a.vpshufb(signs, signs, k.spread);
    a.vpand(signs, signs, k.bit);
    a.vpaddb(signs, signs, k.max8);
    for (v, &p) in p.iter().enumerate() {
        let x = a.ymm("x");
        a.vmovdqa(x, r.quants.at(512 * sb + 32 * (2 * g + v as i32)));
        a.vpsignb(x, x, signs);
        a.vpmaddubsw(p, levels, x);
    }
}

/// Group `g`'s pairs of products summed into lanes and added into each vector's accumulator at
/// its scale, float `2g + v` of `from`.
fn pair_sums(a: &mut Asm, k: &Group, acc: [Ymm; 2], p: [Ymm; 2], from: Gpr, g: i32) {
    a.note(format!("sums of group {g}"));
    for v in 0..2 {
        let lanes = a.ymm("lanes");
        a.vpmaddwd(lanes, p[v], k.one16);
        a.vcvtdq2ps(lanes, lanes);
        let scale = a.ymm("group_scale");
        a.vbroadcastss(scale, from.at(4 * (2 * g + v as i32)));
        a.vfmadd231ps(acc[v], lanes, scale);
    }
}

/// Group `g` of superblock `sb` (0 this one, 1 the next): its 32 levels times its signed
/// activations, in 16-bit pairs, into `p`. Its signs are word `g` of the second half of
/// `from`.
#[allow(clippy::too_many_arguments)]
fn products(a: &mut Asm, k: &Group, r: &Row, p: Ymm, sb: i32, from: Gpr, g: i32) {
    a.note(format!("products of group {g} of superblock +{sb}"));
    let at = 66 * sb + 2 + 8 * g;
    let index: [Gpr; 4] = std::array::from_fn(|l| a.gpr(format!("index{l}")));
    for (l, &i) in index.iter().enumerate() {
        a.movzx_byte(i, r.row.at(at + l as i32));
    }
    // The four rows, quadword `l` from index `l`. A pair is two rows in a 128-bit lane, the
    // second loaded over its high quadword by `vmovhps`: one fused uop, on port 5. The high
    // lane's pair goes in above the low one's by `vinserti128`, port 5 again. A row not in a
    // pair is broadcast and blended in: two uops, on any vector port.
    let rows = a.ymm("rows");
    let grid = |i: Gpr| r.grid.at(0).index(i, 8);
    a.vmovq(p.xmm(), grid(index[0]));
    if k.pairs >= 1 {
        a.vmovhps(p.xmm(), p.xmm(), grid(index[1]));
    }
    if k.pairs == 2 {
        a.vmovq(rows.xmm(), grid(index[2]));
        a.vmovhps(rows.xmm(), rows.xmm(), grid(index[3]));
        a.vinserti128(p, p, rows.xmm(), 1);
    } else {
        for (l, &i) in index.iter().enumerate().skip(1 + k.pairs) {
            a.vpbroadcastq(rows, grid(i));
            a.vpblendd(p, p, rows, 0b11 << (2 * l));
        }
    }
    let signs = a.ymm("signs");
    a.vpbroadcastd(signs, from.at(32 + 4 * g));
    a.vpshufb(signs, signs, k.spread);
    a.vpand(signs, signs, k.bit);
    // A byte's bit, where set, plus 0x7F is negative, and 0x7F alone positive: `vpsignb`
    // negates the activation under the one and keeps it under the other.
    a.vpaddb(signs, signs, k.max8);
    let x = a.ymm("x");
    a.vmovdqa(x, r.quants.at(256 * sb + 32 * g));
    a.vpsignb(x, x, signs);
    a.vpmaddubsw(p, p, x);
}

/// Group `g`'s pairs `p` summed into lanes and added into `acc` at the group's scale, float
/// `g` of `from`.
fn sum(a: &mut Asm, k: &Group, acc: Ymm, p: Ymm, from: Gpr, g: i32) {
    a.note(format!("sum of group {g}"));
    let lanes = a.ymm("lanes");
    a.vpmaddwd(lanes, p, k.one16);
    a.vcvtdq2ps(lanes, lanes);
    let scale = a.ymm("group_scale");
    a.vbroadcastss(scale, from.at(4 * g));
    a.vfmadd231ps(acc, lanes, scale);
}

/// `hsum(acc + 0.0)`, returned.
fn finish(a: &mut Asm, acc: Ymm) {
    let result = lanes_summed(a, acc);
    a.ret_f32(result);
}
