use kgen_core::Program;
use kgen_x86::{Asm, Gpr, Op, Ymm};

struct Constants {
    levels: Ymm,
    nibble: Ymm,
    one16: Ymm,
}

struct Row {
    row: Gpr,
    quants: Gpr,
    scales: Gpr,
}

/// Four IQ4_NL blocks (128 weights) of a row against Q8 activations, bit for bit as the
/// original row kernel computes them.
///
/// `fn(row: *const u8, groups: usize, quants: *const i8, scales: *const f32) -> f32`
///
/// Each group is 72 bytes of weights, four aligned Q8 blocks and four float activation
/// scales. The codebook is the signed levels of IQ4_NL. Unsigned absolute levels times
/// signed activations saturate in 16-bit pairs; the pairs become 32-bit lanes and enter
/// one FMA per 32 weights, in row order. `ahead` blocks start their integer products before
/// the current block's FMA. The final horizontal sum has the same order as `q8::hsum`.
pub fn iq4_nl(ahead: usize) -> Program<Op> {
    assert!((1..4).contains(&ahead), "ahead {ahead}");
    let mut a = Asm::new(format!("iq4_nl_ahead{ahead}"));
    let row = a.param("row");
    let n = a.param("groups");
    let quants = a.param("quants");
    let scales = a.param("scales");
    let r = Row { row, quants, scales };
    let (empty, top, last) = (a.label(), a.label(), a.label());
    a.test(n, n);
    a.jz(empty);

    let lut: [u8; 32] = std::array::from_fn(|i| {
        [-127i8, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113][i % 16] as u8
    });
    let (levels_at, nibble_at, one16_at) = (a.u8s(&lut), a.u32s(&[0x0f0f_0f0f]), a.u32s(&[0x0001_0001]));
    let k = Constants { levels: a.ymm("levels"), nibble: a.ymm("nibble"), one16: a.ymm("one16") };
    a.vmovdqa(k.levels, levels_at);
    a.vpbroadcastd(k.nibble, nibble_at);
    a.vpbroadcastd(k.one16, one16_at);
    let acc = a.ymm("acc");
    a.zero(acc);
    let ring: [Ymm; 4] = std::array::from_fn(|i| a.ymm(format!("p{i}")));
    for (b, &p) in ring.iter().enumerate().take(ahead) {
        products(&mut a, &k, &r, p, 0, b as i32);
    }
    a.dec(n);
    a.jz(last);

    a.place(top);
    for b in 0..4 {
        match b + ahead {
            j if j < 4 => products(&mut a, &k, &r, ring[j], 0, j as i32),
            j => products(&mut a, &k, &r, ring[j - 4], 1, (j - 4) as i32),
        }
        sum(&mut a, &k, &r, acc, ring[b], b as i32);
    }
    a.add(r.row, 72);
    a.add(r.quants, 128);
    a.add(r.scales, 16);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    for b in 0..4 {
        if b + ahead < 4 {
            products(&mut a, &k, &r, ring[b + ahead], 0, (b + ahead) as i32);
        }
        sum(&mut a, &k, &r, acc, ring[b], b as i32);
    }
    finish(&mut a, acc);

    a.place(empty);
    let nothing = a.ymm("nothing");
    a.zero(nothing);
    a.ret_f32(nothing.xmm());
    a.finish()
}

fn products(a: &mut Asm, k: &Constants, r: &Row, p: Ymm, group: i32, b: i32) {
    a.note(format!("products of block {b} of group +{group}"));
    let at = 72 * group + 18 * b;
    let raw = a.ymm("packed");
    let low = a.ymm("low_codes");
    a.vmovdqu(raw.xmm(), r.row.at(at + 2));
    a.vpand(low.xmm(), raw.xmm(), k.nibble.xmm());
    a.vpsrlw(raw.xmm(), raw.xmm(), 4);
    a.vpand(raw.xmm(), raw.xmm(), k.nibble.xmm());
    let signed = a.ymm("signed_levels");
    a.vinserti128(signed, low, raw.xmm(), 1);
    a.vpshufb(signed, k.levels, signed);
    let levels = a.ymm("unsigned_levels");
    a.vpabsb(levels, signed);
    let x = a.ymm("x");
    a.vmovdqa(x, r.quants.at(128 * group + 32 * b));
    a.vpsignb(x, x, signed);
    a.vpmaddubsw(p, levels, x);
}

fn sum(a: &mut Asm, k: &Constants, r: &Row, acc: Ymm, p: Ymm, b: i32) {
    a.note(format!("sum of block {b}"));
    let lanes = a.ymm("lanes");
    a.vpmaddwd(lanes, p, k.one16);
    a.vcvtdq2ps(lanes, lanes);
    let d = a.ymm("d");
    a.vpbroadcastw(d.xmm(), r.row.at(18 * b));
    a.vcvtph2ps(d, d.xmm());
    let dx = a.ymm("dx");
    a.vbroadcastss(dx, r.scales.at(4 * b));
    a.vmulps(d, d, dx);
    a.vfmadd231ps(acc, lanes, d);
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
