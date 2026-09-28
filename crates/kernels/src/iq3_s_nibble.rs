use kgen_core::Program;
use kgen_x86::{Addr, Asm, Gpr, Op, Xmm, Ymm};

struct Group {
    /// The signed odd levels indexed by each packed sign and magnitude nibble.
    levels: Ymm,
    nibble: Ymm,
    one16: Ymm,
}

struct Superblock {
    scale_pairs: Addr,
    nibble_shifts: Addr,
    nibble: Addr,
    one: Addr,
    quarter: Option<Addr>,
}

struct Row {
    row: Gpr,
    quants: Gpr,
    scales: Gpr,
}

/// IQ3_S repacked once into four-bit signed levels, against one activation vector.
///
/// `fn(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32`
///
/// Each of `sbs` superblocks has 134 bytes: the original half `d` at 0, its four packed
/// scale bytes at 2, then eight groups of sixteen bytes at 6. A group's byte `i` holds
/// weights `i` and `16 + i` in its low and high nibble. The low three bits name an odd
/// level `1..15`; bit 3 is the sign. `quants` has eight aligned 32-byte blocks per
/// superblock, and `scales` has eight floats. The kernel returns the exact bits of the
/// IQ3_S row kernel on the original, grid-indexed weights: it keeps the same `vpmaddubsw`
/// saturation and the same order of scale products, FMAs and final horizontal sum.
///
/// `ahead` groups' integer products are made before their sums. The next superblock's
/// scales are prepared alongside this one's groups, in the other half of a 64-byte slot.
pub fn iq3_s_nibble(ahead: usize) -> Program<Op> {
    nibble(ahead, false)
}

/// IQ3_XXS repacked to signed four-bit levels. The levels and scale factor differ from IQ3_S,
/// while its eight groups, scale nibbles and floating-point accumulation have the same shape.
pub fn iq3_xxs_nibble(ahead: usize) -> Program<Op> {
    nibble(ahead, true)
}

/// IQ3_S repacked as for [`iq3_s_nibble`], against two activation vectors: each group is
/// decoded once for both.
///
/// `fn(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, out: *mut f32)`
///
/// The vectors lie block by block together, the way a group of two is stored: block `b` of
/// vector `k` at byte `64 * b + 32 * k` of `quants`, its scale at float `2 * b + k` of
/// `scales`. `out[k]` gets the exact bits [`iq3_s_nibble`] returns for vector `k` alone: each
/// vector has its own accumulator, and the same products, FMAs and final sum in the same order.
pub fn iq3_s_nibble2(ahead: usize) -> Program<Op> {
    nibble2(ahead, false)
}

fn nibble(ahead: usize, xxs: bool) -> Program<Op> {
    assert!((1..8).contains(&ahead), "ahead {ahead}");
    let mut a = Asm::new(format!("iq3_{}_nibble_ahead{ahead}", if xxs { "xxs" } else { "s" }));
    let row = a.param("row");
    let n = a.param("sbs");
    let quants = a.param("quants");
    let scales = a.param("scales");
    let r = Row { row, quants, scales };
    let (empty, top, last) = (a.label(), a.label(), a.label());
    a.test(n, n);
    a.jz(empty);

    a.note("constants");
    let levels = if xxs { [4i8, 12, 20, 28, 36, 44, 52, 62] } else { [1i8, 3, 5, 7, 9, 11, 13, 15] };
    let signed = std::array::from_fn::<i8, 16, _>(|i| if i < 8 { levels[i] } else { -levels[i - 8] });
    let lut: [u8; 32] = std::array::from_fn(|i| signed[i % 16] as u8);
    let levels_at = a.u8s(&lut);
    let nibble_at = a.u32s(&[0x0f0f_0f0f]);
    let one16_at = a.u32s(&[0x0001_0001]);
    let k = Group { levels: a.ymm("levels"), nibble: a.ymm("nibble"), one16: a.ymm("one16") };
    a.vmovdqa(k.levels, levels_at);
    a.vpbroadcastd(k.nibble, nibble_at);
    a.vpbroadcastd(k.one16, one16_at);
    let s = Superblock {
        scale_pairs: a.u8s(&[0, 0, 1, 1, 2, 2, 3, 3, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]),
        nibble_shifts: a.u32s(&[0, 4, 0, 4, 0, 4, 0, 4]),
        nibble: a.u32s(&[15; 8]),
        one: a.u32s(&[1; 8]),
        quarter: xxs.then(|| a.f32s(&[0.25; 8])),
    };
    let acc = a.ymm("acc");
    a.zero(acc);
    let slot = a.slot(64, 64);
    let (cur, nxt) = (a.gpr("cur"), a.gpr("nxt"));
    a.lea(cur, slot.at(0));
    a.lea(nxt, slot.at(32));

    let ring: [Ymm; 8] = std::array::from_fn(|g| a.ymm(format!("p{g}")));
    group_scales(&mut a, &s, &r, 0, cur);
    for (g, &p) in ring.iter().enumerate().take(ahead) {
        products(&mut a, &k, &r, p, 0, g as i32);
    }
    a.dec(n);
    a.jz(last);

    a.place(top);
    let mut next_scales = a.capture(|a| group_scales(a, &s, &r, 1, nxt));
    let share = next_scales.len().div_ceil(8);
    for g in 0..8 {
        a.splice(&mut next_scales, share);
        match g + ahead {
            j if j < 8 => products(&mut a, &k, &r, ring[j], 0, j as i32),
            j => products(&mut a, &k, &r, ring[j - 8], 1, (j - 8) as i32),
        }
        sum(&mut a, &k, acc, ring[g], cur, g as i32);
    }
    a.add(r.row, 134);
    a.add(r.quants, 256);
    a.add(r.scales, 32);
    a.xor(cur, 32);
    a.xor(nxt, 32);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    for g in 0..8 {
        if g + ahead < 8 {
            products(&mut a, &k, &r, ring[g + ahead], 0, (g + ahead) as i32);
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

fn nibble2(ahead: usize, xxs: bool) -> Program<Op> {
    assert!((1..8).contains(&ahead), "ahead {ahead}");
    let mut a = Asm::new(format!("iq3_{}_nibble2_ahead{ahead}", if xxs { "xxs" } else { "s" }));
    let row = a.param("row");
    let n = a.param("sbs");
    let quants = a.param("quants");
    let scales = a.param("scales");
    let out = a.param("out");
    let r = Row { row, quants, scales };
    let (empty, top, last) = (a.label(), a.label(), a.label());
    a.test(n, n);
    a.jz(empty);

    a.note("constants");
    let levels = if xxs { [4i8, 12, 20, 28, 36, 44, 52, 62] } else { [1i8, 3, 5, 7, 9, 11, 13, 15] };
    let signed = std::array::from_fn::<i8, 16, _>(|i| if i < 8 { levels[i] } else { -levels[i - 8] });
    let lut: [u8; 32] = std::array::from_fn(|i| signed[i % 16] as u8);
    let levels_at = a.u8s(&lut);
    let nibble_at = a.u32s(&[0x0f0f_0f0f]);
    let one16_at = a.u32s(&[0x0001_0001]);
    let k = Group { levels: a.ymm("levels"), nibble: a.ymm("nibble"), one16: a.ymm("one16") };
    a.vmovdqa(k.levels, levels_at);
    a.vpbroadcastd(k.nibble, nibble_at);
    a.vpbroadcastd(k.one16, one16_at);
    // Groups 0, 1, 4, 5 in the low lane and 2, 3, 6, 7 in the high one: each scale doubled in
    // place then comes out in group order, one half of the groups at a time.
    let s = Superblock {
        scale_pairs: a.u8s(&[0, 0, 2, 2, 1, 1, 3, 3, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]),
        nibble_shifts: a.u32s(&[0, 4, 0, 4, 0, 4, 0, 4]),
        nibble: a.u32s(&[15; 8]),
        one: a.u32s(&[1; 8]),
        quarter: xxs.then(|| a.f32s(&[0.25; 8])),
    };
    let acc = [a.ymm("acc0"), a.ymm("acc1")];
    for v in acc {
        a.zero(v);
    }
    // Aligned to its whole size: the two halves trade places by flipping one address bit.
    let slot = a.slot(128, 128);
    let (cur, nxt) = (a.gpr("cur"), a.gpr("nxt"));
    a.lea(cur, slot.at(0));
    a.lea(nxt, slot.at(64));

    let ring: [[Ymm; 2]; 8] = std::array::from_fn(|g| [a.ymm(format!("p{g}a")), a.ymm(format!("p{g}b"))]);
    pair_scales(&mut a, &s, &r, 0, cur);
    for (g, &p) in ring.iter().enumerate().take(ahead) {
        products2(&mut a, &k, &r, p, 0, g as i32);
    }
    a.dec(n);
    a.jz(last);

    a.place(top);
    let mut next_scales = a.capture(|a| pair_scales(a, &s, &r, 1, nxt));
    let share = next_scales.len().div_ceil(8);
    for g in 0..8 {
        // The sums first: the accumulators' chains are the longest, and the oldest uop ready
        // is the one a port takes.
        sum2(&mut a, &k, acc, ring[g], cur, g as i32);
        a.splice(&mut next_scales, share);
        match g + ahead {
            j if j < 8 => products2(&mut a, &k, &r, ring[j], 0, j as i32),
            j => products2(&mut a, &k, &r, ring[j - 8], 1, (j - 8) as i32),
        }
    }
    a.add(r.row, 134);
    a.add(r.quants, 512);
    a.add(r.scales, 64);
    a.xor(cur, 64);
    a.xor(nxt, 64);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    for g in 0..8 {
        sum2(&mut a, &k, acc, ring[g], cur, g as i32);
        if g + ahead < 8 {
            products2(&mut a, &k, &r, ring[g + ahead], 0, (g + ahead) as i32);
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

/// Group `g`'s scale times vector `k`'s activation scale, at float `2 * g + k` of the half-slot
/// `to`: the eight odd group scales as [`group_scales`] makes them, each doubled for the two
/// vectors, whose block scales lie in that order already.
fn pair_scales(a: &mut Asm, s: &Superblock, r: &Row, sb: i32, to: Gpr) {
    a.note(format!("scales of superblock +{sb}"));
    let d = a.ymm("d");
    a.vpbroadcastw(d.xmm(), r.row.at(134 * sb));
    a.vcvtph2ps(d, d.xmm());
    let steps = a.ymm("steps");
    a.vmovd(steps.xmm(), r.row.at(134 * sb + 2));
    a.vpshufb(steps.xmm(), steps.xmm(), s.scale_pairs);
    a.vpmovzxbd_reg(steps, steps.xmm());
    a.vpsrlvd(steps, steps, s.nibble_shifts);
    a.vpand(steps, steps, s.nibble);
    let scale = a.ymm("scale");
    a.vpaddd(scale, steps, steps);
    a.vpor(scale, scale, s.one);
    a.vcvtdq2ps(scale, scale);
    a.vmulps(scale, d, scale);
    if let Some(quarter) = s.quarter {
        a.vmulps(scale, scale, quarter);
    }
    for (half, control) in [(0, 0x50), (1, 0xfa)] {
        let pair = a.ymm("pair_scale");
        a.vshufps(pair, scale, scale, control);
        a.vmulps(pair, pair, r.scales.at(64 * sb + 32 * half));
        a.store_vmovaps(to.at(32 * half), pair);
    }
}

/// Group `g`'s packed nibbles expanded to signed odd levels once, then multiplied by each
/// vector's correspondingly signed activations, into `p[k]` for vector `k`.
fn products2(a: &mut Asm, k: &Group, r: &Row, p: [Ymm; 2], sb: i32, g: i32) {
    a.note(format!("products of group {g} of superblock +{sb}"));
    let raw = a.ymm("packed");
    let low = a.ymm("low_codes");
    a.vmovdqu(raw.xmm(), r.row.at(134 * sb + 6 + 16 * g));
    a.vpand(low.xmm(), raw.xmm(), k.nibble.xmm());
    a.vpsrlw(raw.xmm(), raw.xmm(), 4);
    a.vpand(raw.xmm(), raw.xmm(), k.nibble.xmm());
    let signed = a.ymm("signed_levels");
    a.vinserti128(signed, low, raw.xmm(), 1);
    a.vpshufb(signed, k.levels, signed);
    let levels = a.ymm("unsigned_levels");
    a.vpabsb(levels, signed);
    for (v, p) in p.into_iter().enumerate() {
        let x = a.ymm("x");
        a.vmovdqa(x, r.quants.at(512 * sb + 64 * g + 32 * v as i32));
        a.vpsignb(x, x, signed);
        a.vpmaddubsw(p, levels, x);
    }
}

fn sum2(a: &mut Asm, k: &Group, acc: [Ymm; 2], p: [Ymm; 2], from: Gpr, g: i32) {
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

/// Eight odd group scales, times the activation scales, to the half-slot `to`.
fn group_scales(a: &mut Asm, s: &Superblock, r: &Row, sb: i32, to: Gpr) {
    a.note(format!("scales of superblock +{sb}"));
    let d = a.ymm("d");
    a.vpbroadcastw(d.xmm(), r.row.at(134 * sb));
    a.vcvtph2ps(d, d.xmm());
    let steps = a.ymm("steps");
    a.vmovd(steps.xmm(), r.row.at(134 * sb + 2));
    a.vpshufb(steps.xmm(), steps.xmm(), s.scale_pairs);
    a.vpmovzxbd_reg(steps, steps.xmm());
    a.vpsrlvd(steps, steps, s.nibble_shifts);
    a.vpand(steps, steps, s.nibble);
    let scale = a.ymm("scale");
    a.vpaddd(scale, steps, steps);
    a.vpor(scale, scale, s.one);
    a.vcvtdq2ps(scale, scale);
    a.vmulps(scale, d, scale);
    if let Some(quarter) = s.quarter {
        a.vmulps(scale, scale, quarter);
    }
    a.vmulps(scale, scale, r.scales.at(32 * sb));
    a.store_vmovaps(to.at(0), scale);
}

/// Group `g`'s packed nibbles expanded to signed odd levels, then multiplied by the
/// correspondingly signed activations in saturated 16-bit pairs.
fn products(a: &mut Asm, k: &Group, r: &Row, p: Ymm, sb: i32, g: i32) {
    a.note(format!("products of group {g} of superblock +{sb}"));
    let raw = a.ymm("packed");
    let low = a.ymm("low_codes");
    a.vmovdqu(raw.xmm(), r.row.at(134 * sb + 6 + 16 * g));
    a.vpand(low.xmm(), raw.xmm(), k.nibble.xmm());
    a.vpsrlw(raw.xmm(), raw.xmm(), 4);
    a.vpand(raw.xmm(), raw.xmm(), k.nibble.xmm());
    a.vinserti128(p, low, raw.xmm(), 1);
    a.vpshufb(p, k.levels, p);
    let levels = a.ymm("unsigned_levels");
    a.vpabsb(levels, p);
    let x = a.ymm("x");
    a.vmovdqa(x, r.quants.at(256 * sb + 32 * g));
    a.vpsignb(x, x, p);
    a.vpmaddubsw(p, levels, x);
}

fn sum(a: &mut Asm, k: &Group, acc: Ymm, p: Ymm, from: Gpr, g: i32) {
    a.note(format!("sum of group {g}"));
    let lanes = a.ymm("lanes");
    a.vpmaddwd(lanes, p, k.one16);
    a.vcvtdq2ps(lanes, lanes);
    let scale = a.ymm("group_scale");
    a.vbroadcastss(scale, from.at(4 * g));
    a.vfmadd231ps(acc, lanes, scale);
}

fn finish(a: &mut Asm, acc: Ymm) {
    let result = lanes_summed(a, acc);
    a.ret_f32(result);
}

/// `acc`'s lanes added as the row kernels add them, into the low lane of what it returns.
fn lanes_summed(a: &mut Asm, acc: Ymm) -> Xmm {
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
    result.xmm()
}
