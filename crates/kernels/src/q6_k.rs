use kgen_core::Program;
use kgen_x86::{Asm, Gpr, Op, Ymm};

/// Q6_K's 210-byte superblocks against one Q8 activation vector.
///
/// `fn(row, superblocks, quants, scales, half_sums) -> f32`. The half sums are
/// two floats for each 32 activation quants. The affine offset is accumulated
/// separately from the quant products, in the same order as the original row
/// kernel, so the returned bits agree with it.
pub fn q6_k() -> Program<Op> {
    let mut a = Asm::new("q6_k");
    let row = a.param("row");
    let n = a.param("superblocks");
    let quants = a.param("quants");
    let scales = a.param("scales");
    let sums = a.param("half_sums");
    let (empty, top) = (a.label(), a.label());
    a.test(n, n);
    a.jz(empty);

    let low = a.ymm("nibble_mask");
    let three = a.ymm("pair_mask");
    let one = a.ymm("one16");
    let low_at = a.u32s(&[0x0f0f_0f0f]);
    let three_at = a.u32s(&[0x0303_0303]);
    let one_at = a.u32s(&[0x0001_0001]);
    a.vpbroadcastd(low, low_at);
    a.vpbroadcastd(three, three_at);
    a.vpbroadcastd(one, one_at);
    let pair_shuffle = a.u8s(&[0, 0, 0, 0, 1, 1, 1, 1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]);
    let minus32 = a.f32s(&[-32.0; 8]);
    let (acc, off) = (a.ymm("acc"), a.ymm("off"));
    a.zero(acc);
    a.zero(off);

    a.place(top);
    let dh = a.ymm("half_d");
    a.vpbroadcastw(dh.xmm(), row.at(208));
    let d = a.ymm("d");
    a.vcvtph2ps(d, dh.xmm());

    for half in 0..2 {
        let ql0 = a.ymm("ql0");
        let ql1 = a.ymm("ql1");
        let qh = a.ymm("qh");
        a.vmovdqu(ql0, row.at(64 * half));
        a.vmovdqu(ql1, row.at(64 * half + 32));
        a.vmovdqu(qh, row.at(128 + 32 * half));
        for k in 0..4 {
            let g = half * 4 + k;
            group(&mut a, row, quants, scales, acc, d, ql0, ql1, qh, low, three, one, pair_shuffle, g);
        }
    }

    for half in 0..2 {
        let eight = a.ymm("eight_scales");
        a.vmovq(eight.xmm(), row.at(192 + half * 8));
        let wide = a.ymm("wide_scales");
        a.vpmovsxbd_reg(wide, eight.xmm());
        a.vcvtdq2ps(wide, wide);
        a.vmulps(wide, wide, d);
        a.vmulps(wide, wide, minus32);
        let dsums = a.ymm("dsums");
        a.vmovups(dsums, sums.at(half * 32));
        a.vfmadd231ps(off, wide, dsums);
    }
    a.add(row, 210);
    a.add(quants, 256);
    a.add(scales, 32);
    a.add(sums, 64);
    a.dec(n);
    a.jnz(top);
    a.vaddps(acc, acc, off);
    finish(&mut a, acc);

    a.place(empty);
    let zero = a.ymm("zero");
    a.zero(zero);
    a.ret_f32(zero.xmm());
    a.finish()
}

#[allow(clippy::too_many_arguments)]
fn group(
    a: &mut Asm,
    row: Gpr,
    quants: Gpr,
    scales: Gpr,
    acc: Ymm,
    d: Ymm,
    ql0: Ymm,
    ql1: Ymm,
    qh: Ymm,
    low: Ymm,
    three: Ymm,
    one: Ymm,
    pair_shuffle: kgen_x86::Addr,
    g: i32,
) {
    let k = g % 4;
    let ql = if k % 2 == 0 { ql0 } else { ql1 };
    let nib = a.ymm("nib");
    if k < 2 {
        a.vpand(nib, ql, low);
    } else {
        a.vpsrlw(nib, ql, 4);
        a.vpand(nib, nib, low);
    }
    let high = a.ymm("high");
    if k == 0 {
        a.vpand(high, qh, three);
    } else {
        a.vpsrlw(high, qh, (2 * k) as u8);
        a.vpand(high, high, three);
    }
    a.vpsllw(high, high, 4);
    a.vpor(nib, nib, high);

    let x = a.ymm("x");
    a.vmovdqa(x, quants.at(g * 32));
    let p = a.ymm("pairs");
    a.vpmaddubsw(p, nib, x);
    a.vpmaddwd(p, p, one);
    a.vcvtdq2ps(p, p);

    let pair = a.ymm("pair_scales");
    a.vmovd(pair.xmm(), row.at(192 + 2 * g));
    a.vpshufb(pair.xmm(), pair.xmm(), pair_shuffle);
    let ws = a.ymm("ws");
    a.vpmovsxbd_reg(ws, pair.xmm());
    a.vcvtdq2ps(ws, ws);
    a.vmulps(ws, ws, d);
    let dx = a.ymm("dx");
    a.vbroadcastss(dx, scales.at(4 * g));
    a.vmulps(ws, ws, dx);
    a.vfmadd231ps(acc, p, ws);
}

fn finish(a: &mut Asm, acc: Ymm) {
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
