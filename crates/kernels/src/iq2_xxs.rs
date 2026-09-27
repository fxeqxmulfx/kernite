use kgen_core::Program;
use kgen_x86::{Addr, Asm, Gpr, Op, Ymm};

/// The constants every group uses, in registers.
struct Group {
    /// Byte `l` of a word to the eight bytes of weights `8l..8l + 8`.
    spread: Ymm,
    /// Bit `j % 8` for byte `j`.
    bit: Ymm,
    one8: Ymm,
    one16: Ymm,
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
pub fn iq2_xxs(ahead: usize) -> Program<Op> {
    assert!((1..8).contains(&ahead), "ahead {ahead}");
    let mut a = Asm::new(format!("iq2_xxs_ahead{ahead}"));
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
    let (one8_at, one16_at) = (a.u32s(&[0x0101_0101]), a.u32s(&[0x0001_0001]));
    let k = Group { spread: a.ymm("spread"), bit: a.ymm("bit"), one8: a.ymm("one8"), one16: a.ymm("one16") };
    a.vmovdqa(k.spread, spread_at);
    a.vpbroadcastq(k.bit, bit_at);
    a.vpbroadcastd(k.one8, one8_at);
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
    prepare(&mut a, &s, &r, 1, nxt);
    for g in 0..8 {
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

/// Superblock `sb` (0 this one, 1 the next)'s eight group scales, times the activations',
/// then its eight words of signs, to the 64 bytes at `to`.
///
/// Each group's second word holds its scale in the top nibble and four 7-bit fields of signs
/// below; `vshufps` and `vpermq` gather the eight second words in group order.
fn prepare(a: &mut Asm, s: &Superblock, r: &Row, sb: i32, to: Gpr) {
    a.note(format!("scales and signs of superblock +{sb}"));
    let at = 66 * sb;
    let w = a.ymm("words");
    a.vmovdqu(w, r.row.at(at + 2));
    a.vshufps(w, w, r.row.at(at + 34), 0xdd);
    a.vpermq(w, w, 0xd8);

    // `(w >> 27) | 1` is `2 * (w >> 28) + 1`, the odd step.
    let d = a.ymm("d");
    a.vpbroadcastw(d.xmm(), r.row.at(at));
    a.vcvtph2ps(d, d.xmm());
    let scale = a.ymm("scale");
    a.vpsrld(scale, w, 27);
    a.vpor(scale, scale, s.one);
    a.vcvtdq2ps(scale, scale);
    a.vmulps(scale, d, scale);
    a.vmulps(scale, scale, s.eighth);
    a.vmulps(scale, scale, r.scales.at(32 * sb));
    a.store_vmovaps(to.at(0), scale);

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
    a.store_vmovdqa(to.at(32), y);
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
    // The four rows, quadword `l` from index `l`.
    let row = a.ymm("rows");
    a.vmovq(p.xmm(), r.grid.at(0).index(index[0], 8));
    for (l, &i) in index.iter().enumerate().skip(1) {
        a.vpbroadcastq(row, r.grid.at(0).index(i, 8));
        a.vpblendd(p, p, row, 0b11 << (2 * l));
    }
    let signs = a.ymm("signs");
    a.vpbroadcastd(signs, from.at(32 + 4 * g));
    a.vpshufb(signs, signs, k.spread);
    a.vpand(signs, signs, k.bit);
    a.vpcmpeqb(signs, signs, k.bit);
    a.vpor(signs, signs, k.one8);
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
    a.note("the lanes summed");
    let zero = a.ymm("zero");
    a.zero(zero);
    a.vaddps(acc, acc, zero);
    let high = a.ymm("high");
    a.vextractf128(high.xmm(), acc, 1);
    let sum = a.ymm("sum");
    a.vaddps(sum.xmm(), acc.xmm(), high.xmm());
    let pair = a.ymm("pair");
    a.vmovhlps(pair.xmm(), sum.xmm(), sum.xmm());
    a.vaddps(sum.xmm(), sum.xmm(), pair.xmm());
    a.vmovshdup(pair.xmm(), sum.xmm());
    let result = a.ymm("result");
    a.vaddss(result.xmm(), sum.xmm(), pair.xmm());
    a.ret_f32(result.xmm());
}
