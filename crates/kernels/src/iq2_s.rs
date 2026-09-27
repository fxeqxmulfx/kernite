use kgen_core::Program;
use kgen_x86::{Addr, Asm, Gpr, Op, Ymm};

struct Group {
    spread: Ymm,
    bit: Ymm,
    max8: Ymm,
    one16: Ymm,
}

struct Superblock {
    nibble: Addr,
    one: Addr,
    eighth: Addr,
}

struct Row {
    row: Gpr,
    quants: Gpr,
    scales: Gpr,
    grid: Gpr,
}

/// An IQ2_S row against one vector of Q8 activations, exactly as the original format stores it.
///
/// `fn(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, grid: *const u64) -> f32`
///
/// A superblock has 82 bytes. Its 32 index bytes and eight high-index bytes address four grid
/// rows per group; its 32 sign bytes give one sign bit to each level. The scale byte of a group
/// gives separate odd multipliers to the first and last sixteen weights. The kernel keeps the
/// same pairwise saturation, float products, FMA order and final horizontal sum as the original
/// `dot_iq_ahead::<IQ2_S, 82, 2>` kernel.
///
/// `ahead` groups' products are started before their sums. The next superblock's signs and
/// scales are prepared while the current one is consumed.
pub fn iq2_s(ahead: usize) -> Program<Op> {
    build(ahead, IndexMode::Original)
}

/// IQ2_S in its original 82-byte layout, with four grid rows gathered by AVX2.
pub fn iq2_s_gather(ahead: usize) -> Program<Op> {
    build(ahead, IndexMode::Gather)
}

/// IQ2_S with its four 10-bit indices packed into five consecutive bytes per group.
/// The 82-byte superblock stores exactly the original information and reads the same bytes.
pub fn iq2_s_packed(ahead: usize) -> Program<Op> {
    build(ahead, IndexMode::Packed)
}

#[derive(Clone, Copy)]
enum IndexMode {
    Original,
    Gather,
    Packed,
}

fn build(ahead: usize, mode: IndexMode) -> Program<Op> {
    assert!((1..8).contains(&ahead), "ahead {ahead}");
    let prefix = match mode {
        IndexMode::Original => "",
        IndexMode::Gather => "gather_",
        IndexMode::Packed => "packed_",
    };
    let mut a = Asm::new(format!("iq2_s_{prefix}ahead{ahead}"));
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
    let k = Group { spread: a.ymm("spread"), bit: a.ymm("bit"), max8: a.ymm("max8"), one16: a.ymm("one16") };
    let (spread_at, bit_at) = (a.u8s(&spread), a.u64s(&[0x8040_2010_0804_0201]));
    let (max8_at, one16_at) = (a.u32s(&[0x7F7F_7F7F]), a.u32s(&[0x0001_0001]));
    a.vmovdqa(k.spread, spread_at);
    a.vpbroadcastq(k.bit, bit_at);
    a.vpbroadcastd(k.max8, max8_at);
    a.vpbroadcastd(k.one16, one16_at);
    let s = Superblock { nibble: a.u32s(&[15; 8]), one: a.u32s(&[1; 8]), eighth: a.f32s(&[0.125; 8]) };
    let acc = a.ymm("acc");
    a.zero(acc);
    let slot = a.slot(256, 256);
    let (cur, nxt) = (a.gpr("cur"), a.gpr("nxt"));
    a.lea(cur, slot.at(0));
    a.lea(nxt, slot.at(128));

    let ring: [Ymm; 8] = std::array::from_fn(|g| a.ymm(format!("p{g}")));
    prepare(&mut a, &s, &r, 0, cur, mode);
    for (g, &p) in ring.iter().enumerate().take(ahead) {
        products(&mut a, &k, &r, p, 0, cur, g as i32, mode);
    }
    a.dec(n);
    a.jz(last);

    a.place(top);
    let mut next_signs = a.capture(|a| group_signs(a, &r, 1, nxt, mode));
    let mut next_scales = a.capture(|a| group_scales(a, &s, &r, 1, nxt));
    let (sign_share, scale_share) = (next_signs.len().div_ceil(8 - ahead), next_scales.len().div_ceil(ahead));
    for g in 0..8 {
        if g < 8 - ahead {
            a.splice(&mut next_signs, sign_share);
        } else {
            a.splice(&mut next_scales, scale_share);
        }
        match g + ahead {
            j if j < 8 => products(&mut a, &k, &r, ring[j], 0, cur, j as i32, mode),
            j => products(&mut a, &k, &r, ring[j - 8], 1, nxt, (j - 8) as i32, mode),
        }
        sum(&mut a, &k, acc, ring[g], cur, g as i32);
    }
    a.add(r.row, 82);
    a.add(r.quants, 256);
    a.add(r.scales, 32);
    a.xor(cur, 128);
    a.xor(nxt, 128);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    for g in 0..8 {
        if g + ahead < 8 {
            products(&mut a, &k, &r, ring[g + ahead], 0, cur, (g + ahead) as i32, mode);
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

fn prepare(a: &mut Asm, s: &Superblock, r: &Row, sb: i32, to: Gpr, mode: IndexMode) {
    group_signs(a, r, sb, to, mode);
    group_scales(a, s, r, sb, to);
}

fn group_signs(a: &mut Asm, r: &Row, sb: i32, to: Gpr, mode: IndexMode) {
    a.note(format!("signs of superblock +{sb}"));
    let signs = a.ymm("packed_signs");
    let at = if matches!(mode, IndexMode::Packed) { 42 } else { 34 };
    a.vmovdqu(signs, r.row.at(82 * sb + at));
    a.store_vmovdqa(to.at(64), signs);
}

/// Scale low and high halves, in group order, to the two aligned 32-byte halves of `to`.
fn group_scales(a: &mut Asm, s: &Superblock, r: &Row, sb: i32, to: Gpr) {
    a.note(format!("scales of superblock +{sb}"));
    let d = a.ymm("d");
    a.vpbroadcastw(d.xmm(), r.row.at(82 * sb));
    a.vcvtph2ps(d, d.xmm());
    let steps = a.ymm("steps");
    a.vpmovzxbd(steps, r.row.at(82 * sb + 74));
    let low = a.ymm("low_scales");
    let high = a.ymm("high_scales");
    a.vpand(low, steps, s.nibble);
    a.vpsrld(high, steps, 4);
    for v in [low, high] {
        a.vpaddd(v, v, v);
        a.vpor(v, v, s.one);
        a.vcvtdq2ps(v, v);
        a.vmulps(v, d, v);
        a.vmulps(v, v, s.eighth);
        a.vmulps(v, v, r.scales.at(32 * sb));
    }
    a.store_vmovaps(to.at(0), low);
    a.store_vmovaps(to.at(32), high);
}

/// Four grid quadwords, with each of their eight activations signed and multiplied.
#[allow(clippy::too_many_arguments)]
fn products(a: &mut Asm, k: &Group, r: &Row, p: Ymm, sb: i32, from: Gpr, g: i32, mode: IndexMode) {
    a.note(format!("products of group {g} of superblock +{sb}"));
    if matches!(mode, IndexMode::Gather) {
        let high = a.ymm("high_index_bits");
        let shifts = a.u32s(&[0, 2, 4, 6]);
        let mask3 = a.u32s(&[3; 4]);
        a.vpbroadcastb(high.xmm(), r.row.at(82 * sb + 66 + g));
        a.vpsrlvd(high.xmm(), high.xmm(), shifts);
        a.vpand(high.xmm(), high.xmm(), mask3);
        a.vpslld(high.xmm(), high.xmm(), 8);
        let index = a.ymm("index");
        a.vpmovzxbd_xmm(index.xmm(), r.row.at(82 * sb + 2 + 4 * g));
        a.vpor(index.xmm(), index.xmm(), high.xmm());
        let gather_mask = a.ymm("gather_mask");
        a.ones(gather_mask);
        a.zero(p);
        a.vpgatherdq(p, r.grid.at(0).index_xmm(index.xmm(), 8), gather_mask);
    } else {
        let index: [Gpr; 4] = std::array::from_fn(|l| a.gpr(format!("index{l}")));
        if matches!(mode, IndexMode::Packed) {
            let bits = a.gpr("packed_indices");
            a.load_u64(bits, r.row.at(82 * sb + 2 + 5 * g));
            for (l, &i) in index.iter().enumerate() {
                a.mov(i, bits);
                if l != 0 {
                    a.shr(i, (10 * l) as u8);
                }
                a.and(i, 1023);
            }
        } else {
            let high = a.gpr("high_index_bits");
            a.movzx_byte(high, r.row.at(82 * sb + 66 + g));
            for (l, &i) in index.iter().enumerate() {
                a.movzx_byte(i, r.row.at(82 * sb + 2 + 4 * g + l as i32));
                let bits = a.gpr("high_for_entry");
                a.mov(bits, high);
                if l != 0 {
                    a.shr(bits, (2 * l) as u8);
                }
                a.and(bits, 3);
                a.shl(bits, 8);
                a.or_reg(i, bits);
            }
        }
        let grid = |i: Gpr| r.grid.at(0).index(i, 8);
        let rows = a.ymm("rows");
        a.vmovq(p.xmm(), grid(index[0]));
        a.vmovhps(p.xmm(), p.xmm(), grid(index[1]));
        a.vmovq(rows.xmm(), grid(index[2]));
        a.vmovhps(rows.xmm(), rows.xmm(), grid(index[3]));
        a.vinserti128(p, p, rows.xmm(), 1);
    }
    let signs = a.ymm("signs");
    a.vpbroadcastd(signs, from.at(64 + 4 * g));
    a.vpshufb(signs, signs, k.spread);
    a.vpand(signs, signs, k.bit);
    a.vpaddb(signs, signs, k.max8);
    let x = a.ymm("x");
    a.vmovdqa(x, r.quants.at(256 * sb + 32 * g));
    a.vpsignb(x, x, signs);
    a.vpmaddubsw(p, p, x);
}

fn sum(a: &mut Asm, k: &Group, acc: Ymm, p: Ymm, from: Gpr, g: i32) {
    a.note(format!("sum of group {g}"));
    let lanes = a.ymm("lanes");
    a.vpmaddwd(lanes, p, k.one16);
    a.vcvtdq2ps(lanes, lanes);
    let (low, high, scale) = (a.ymm("low_scale"), a.ymm("high_scale"), a.ymm("scale"));
    a.vbroadcastss(low, from.at(4 * g));
    a.vbroadcastss(high, from.at(32 + 4 * g));
    a.vpblendd(scale, low, high, 0xf0);
    a.vfmadd231ps(acc, lanes, scale);
}

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
