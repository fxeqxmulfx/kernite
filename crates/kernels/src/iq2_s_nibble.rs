use kgen_core::Program;
use kgen_x86::{Addr, Asm, Gpr, Op, Ymm};

struct Row {
    row: Gpr,
    quants: Gpr,
    scales: Gpr,
}

struct Codec {
    levels: Ymm,
    one16: Ymm,
    nibble: Option<Ymm>,
    three: Option<Ymm>,
    spread: Option<Ymm>,
    bit: Option<Ymm>,
    max8: Option<Ymm>,
}

/// IQ2_S repacked to four-bit signed level codes, against one Q8 vector.
///
/// `fn(row, superblocks, quants, scales) -> f32`. A 138-byte superblock has
/// the original half scale, eight packed group scale bytes, then eight groups
/// of sixteen signed level nibbles. The six possible levels are ±8, ±25, ±43.
/// The two halves of each group keep their distinct scale and the original
/// row kernel's order of products, multiplications and FMAs.
pub fn iq2_s_nibble(ahead: usize) -> Program<Op> {
    packed(ahead, false)
}

/// IQ2_S's three levels in two-bit planes, with its original sign bits.
/// Each superblock is 106 bytes: the same scale header, 64 bytes of levels,
/// then 32 bytes of signs. Four groups share each plane's 32-byte load.
pub fn iq2_s_planes(ahead: usize) -> Program<Op> {
    packed(ahead, true)
}

fn packed(ahead: usize, planes: bool) -> Program<Op> {
    assert!((1..8).contains(&ahead));
    let mut a = Asm::new(format!("iq2_s_{}_ahead{ahead}", if planes { "planes" } else { "nibble" }));
    let row = a.param("row");
    let n = a.param("superblocks");
    let quants = a.param("quants");
    let scales = a.param("scales");
    let r = Row { row, quants, scales };
    let (empty, top, last) = (a.label(), a.label(), a.label());
    a.test(n, n);
    a.jz(empty);

    let mut signed = [0i8; 16];
    for i in 0..3 {
        signed[i] = [8, 25, 43][i];
        signed[i + 8] = -signed[i];
    }
    let levels_at =
        a.u8s(&std::array::from_fn::<_, 32, _>(|i| if planes { [8, 25, 43, 0][i % 4] } else { signed[i % 16] as u8 }));
    let one16_at = a.u32s(&[0x0001_0001]);
    let levels = a.ymm("levels");
    let one16 = a.ymm("one16");
    a.vmovdqa(levels, levels_at);
    a.vpbroadcastd(one16, one16_at);
    let nibble = (!planes).then(|| {
        let at = a.u32s(&[0x0f0f_0f0f]);
        let v = a.ymm("nibble");
        a.vpbroadcastd(v, at);
        v
    });
    let (three, spread, bit, max8) = if planes {
        let three_at = a.u32s(&[0x0303_0303]);
        let spread_at = a.u8s(&(0..32).map(|j| j / 8).collect::<Vec<u8>>());
        let bit_at = a.u64s(&[0x8040_2010_0804_0201]);
        let max8_at = a.u32s(&[0x7f7f_7f7f]);
        let (three, spread, bit, max8) = (a.ymm("three"), a.ymm("spread"), a.ymm("bit"), a.ymm("max8"));
        a.vpbroadcastd(three, three_at);
        a.vmovdqa(spread, spread_at);
        a.vpbroadcastq(bit, bit_at);
        a.vpbroadcastd(max8, max8_at);
        (Some(three), Some(spread), Some(bit), Some(max8))
    } else {
        (None, None, None, None)
    };
    let codec = Codec { levels, one16, nibble, three, spread, bit, max8 };
    let stride = if planes { 106 } else { 138 };
    let scale_mask = a.u32s(&[15; 8]);
    let scale_one = a.u32s(&[1; 8]);
    let eighth = a.f32s(&[0.125; 8]);
    let acc = a.ymm("acc");
    a.zero(acc);
    let slot = a.slot(128, 128);
    let (cur, nxt) = (a.gpr("cur"), a.gpr("nxt"));
    a.lea(cur, slot.at(0));
    a.lea(nxt, slot.at(64));
    let ring: [Ymm; 8] = std::array::from_fn(|g| a.ymm(format!("p{g}")));
    group_scales(&mut a, &r, scale_mask, scale_one, eighth, stride, 0, cur);
    for (g, &p) in ring.iter().enumerate().take(ahead) {
        products(&mut a, &r, &codec, stride, p, 0, g as i32);
    }
    a.dec(n);
    a.jz(last);

    a.place(top);
    let mut next_scales = a.capture(|a| group_scales(a, &r, scale_mask, scale_one, eighth, stride, 1, nxt));
    let share = next_scales.len().div_ceil(8);
    for g in 0..8 {
        a.splice(&mut next_scales, share);
        match g + ahead {
            j if j < 8 => products(&mut a, &r, &codec, stride, ring[j], 0, j as i32),
            j => products(&mut a, &r, &codec, stride, ring[j - 8], 1, (j - 8) as i32),
        }
        sum(&mut a, acc, codec.one16, ring[g], cur, g as i32);
    }
    a.add(r.row, stride);
    a.add(r.quants, 256);
    a.add(r.scales, 32);
    a.xor(cur, 64);
    a.xor(nxt, 64);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    for g in 0..8 {
        if g + ahead < 8 {
            products(&mut a, &r, &codec, stride, ring[g + ahead], 0, (g + ahead) as i32);
        }
        sum(&mut a, acc, codec.one16, ring[g], cur, g as i32);
    }
    finish(&mut a, acc);
    a.place(empty);
    let zero = a.ymm("zero");
    a.zero(zero);
    a.ret_f32(zero.xmm());
    a.finish()
}

#[allow(clippy::too_many_arguments)]
fn group_scales(a: &mut Asm, r: &Row, mask: Addr, one: Addr, eighth: Addr, stride: i32, sb: i32, to: Gpr) {
    let d = a.ymm("d");
    a.vpbroadcastw(d.xmm(), r.row.at(stride * sb));
    a.vcvtph2ps(d, d.xmm());
    let steps = a.ymm("steps");
    a.vpmovzxbd(steps, r.row.at(stride * sb + 2));
    let low = a.ymm("low_scales");
    let high = a.ymm("high_scales");
    a.vpand(low, steps, mask);
    a.vpsrld(high, steps, 4);
    for v in [low, high] {
        a.vpaddd(v, v, v);
        a.vpor(v, v, one);
        a.vcvtdq2ps(v, v);
        a.vmulps(v, d, v);
        a.vmulps(v, v, eighth);
        a.vmulps(v, v, r.scales.at(32 * sb));
    }
    a.store_vmovaps(to.at(0), low);
    a.store_vmovaps(to.at(32), high);
}

fn products(a: &mut Asm, r: &Row, codec: &Codec, stride: i32, p: Ymm, sb: i32, g: i32) {
    if let Some(three) = codec.three {
        let codes = a.ymm("codes");
        a.vmovdqu(codes, r.row.at(stride * sb + 10 + 32 * (g / 4)));
        if g % 4 != 0 {
            a.vpsrlw(codes, codes, (2 * (g % 4)) as u8);
        }
        a.vpand(codes, codes, three);
        a.vpshufb(codes, codec.levels, codes);
        let signs = a.ymm("signs");
        a.vpbroadcastd(signs, r.row.at(stride * sb + 74 + 4 * g));
        a.vpshufb(signs, signs, codec.spread.unwrap());
        a.vpand(signs, signs, codec.bit.unwrap());
        a.vpaddb(signs, signs, codec.max8.unwrap());
        let x = a.ymm("x");
        a.vmovdqa(x, r.quants.at(256 * sb + 32 * g));
        a.vpsignb(x, x, signs);
        a.vpmaddubsw(p, codes, x);
    } else {
        let raw = a.ymm("packed");
        let low = a.ymm("low_codes");
        a.vmovdqu(raw.xmm(), r.row.at(stride * sb + 10 + 16 * g));
        a.vpand(low.xmm(), raw.xmm(), codec.nibble.unwrap().xmm());
        a.vpsrlw(raw.xmm(), raw.xmm(), 4);
        a.vpand(raw.xmm(), raw.xmm(), codec.nibble.unwrap().xmm());
        a.vinserti128(p, low, raw.xmm(), 1);
        a.vpshufb(p, codec.levels, p);
        let abs = a.ymm("unsigned_levels");
        a.vpabsb(abs, p);
        let x = a.ymm("x");
        a.vmovdqa(x, r.quants.at(256 * sb + 32 * g));
        a.vpsignb(x, x, p);
        a.vpmaddubsw(p, abs, x);
    }
}

fn sum(a: &mut Asm, acc: Ymm, one16: Ymm, p: Ymm, from: Gpr, g: i32) {
    let lanes = a.ymm("lanes");
    a.vpmaddwd(lanes, p, one16);
    a.vcvtdq2ps(lanes, lanes);
    let low = a.ymm("low_scale");
    let high = a.ymm("high_scale");
    let scale = a.ymm("scale");
    a.vbroadcastss(low, from.at(4 * g));
    a.vbroadcastss(high, from.at(32 + 4 * g));
    a.vpblendd(scale, low, high, 0xf0);
    a.vfmadd231ps(acc, lanes, scale);
}

fn finish(a: &mut Asm, acc: Ymm) {
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
