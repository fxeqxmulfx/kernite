//! Formats whose weights are four-bit codes into sixteen signed levels, sixteen code bytes a
//! group of 32 weights: byte `i` of a group holds weights `i` and `16 + i` in its low and high
//! nibble. They differ in the levels and in the superblock of 256 weights around the codes,
//! where each keeps its half `d` at 0 and its eight group scales its own way.
//!
//! Each kernel returns the exact bits of the format's original row kernel: the same
//! `vpmaddubsw` saturation, the same order of scale products and FMAs, one accumulator a
//! vector, and the same final horizontal sum.

use kgen_core::Program;
use kgen_x86::{Addr, Asm, Gpr, Op, Ymm};

use crate::lanes::{lanes_summed, pair_sums, sum};

/// What sets one format apart from another.
struct Format {
    /// The kernel's name, before the vectors and `ahead`.
    name: &'static str,
    /// Bytes a superblock, and where in it the first group's codes start.
    bytes: i32,
    codes: i32,
    /// The level each code names.
    levels: [i8; 16],
    scales: Scales,
}

/// How a superblock keeps its eight group scales.
#[derive(Clone, Copy)]
enum Scales {
    /// A nibble `k` a group in the four bytes at 2, group `g` in byte `g / 2`, low nibble
    /// first: `d * (2k + 1)`, then a quarter of that when `quarter`.
    Odd { quarter: bool },
    /// Six bits `ls` a group: its low nibble as `Odd` keeps them but in the four bytes at 4, and
    /// its high two bits at bit `2g` of the 16-bit word at 2: `d * (ls - 32)`.
    Six,
}

/// IQ3_S repacked once: the low three bits of a code name an odd level `1..15`, bit 3 its sign.
const IQ3_S: Format = Format {
    name: "iq3_s_nibble",
    bytes: 134,
    codes: 6,
    levels: signed([1, 3, 5, 7, 9, 11, 13, 15]),
    scales: Scales::Odd { quarter: false },
};

/// IQ3_XXS repacked once, the same way; its levels and scale factor differ.
const IQ3_XXS: Format = Format {
    name: "iq3_xxs_nibble",
    bytes: 134,
    codes: 6,
    levels: signed([4, 12, 20, 28, 36, 44, 52, 62]),
    scales: Scales::Odd { quarter: true },
};

/// IQ4_XS as the file has it: the half, the high scale bits, the low scale nibbles, the codes.
const IQ4_XS: Format = Format {
    name: "iq4_xs",
    bytes: 136,
    codes: 8,
    levels: [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    scales: Scales::Six,
};

/// Eight magnitudes, then the same negated: the levels of a code whose bit 3 is a sign.
const fn signed(levels: [i8; 8]) -> [i8; 16] {
    let mut out = [0; 16];
    let mut i = 0;
    while i < 8 {
        out[i] = levels[i];
        out[i + 8] = -levels[i];
        i += 1;
    }
    out
}

/// IQ3_S repacked once into four-bit signed levels, against one activation vector.
///
/// `fn(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32) -> f32`
///
/// Each of `sbs` superblocks has 134 bytes: the original half `d` at 0, its four packed
/// scale bytes at 2, then eight groups of sixteen bytes at 6. The low three bits of a code
/// name an odd level `1..15`; bit 3 is the sign. `quants` has eight aligned 32-byte blocks per
/// superblock, and `scales` has eight floats. The kernel returns the exact bits of the IQ3_S
/// row kernel on the original, grid-indexed weights.
///
/// `ahead` groups' integer products are made before their sums. The next superblock's
/// scales are prepared alongside this one's groups, in the other half of a 64-byte slot.
pub fn iq3_s_nibble(ahead: usize) -> Program<Op> {
    one(&IQ3_S, ahead)
}

/// IQ3_XXS repacked to signed four-bit levels. The levels and scale factor differ from IQ3_S,
/// while its eight groups, scale nibbles and floating-point accumulation have the same shape.
pub fn iq3_xxs_nibble(ahead: usize) -> Program<Op> {
    one(&IQ3_XXS, ahead)
}

/// IQ4_XS rows as the file has them, against one activation vector; the same arguments as
/// [`iq3_s_nibble`], with 136-byte superblocks.
pub fn iq4_xs(ahead: usize) -> Program<Op> {
    one(&IQ4_XS, ahead)
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
    two(&IQ3_S, ahead)
}

/// IQ3_XXS repacked as for [`iq3_xxs_nibble`], against two vectors, as [`iq3_s_nibble2`] is
/// IQ3_S's.
pub fn iq3_xxs_nibble2(ahead: usize) -> Program<Op> {
    two(&IQ3_XXS, ahead)
}

/// IQ4_XS against two vectors, as [`iq3_s_nibble2`] is IQ3_S's.
pub fn iq4_xs2(ahead: usize) -> Program<Op> {
    two(&IQ4_XS, ahead)
}

struct Group {
    /// The signed levels indexed by each code.
    levels: Ymm,
    nibble: Ymm,
    one16: Ymm,
}

/// A superblock's constants for its group scales, which come out with group `order[j]` in
/// lane `j`.
struct Superblock {
    scale_pairs: Addr,
    nibble_shifts: Addr,
    nibble: Addr,
    kind: Kind,
}

enum Kind {
    Odd { one: Addr, quarter: Option<Addr> },
    Six { high_shifts: Addr, pair: Addr, less: Addr },
}

/// Group order, one to a lane.
const IN_ORDER: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
/// Groups 0, 1, 4, 5 in the low lane and 2, 3, 6, 7 in the high one: each scale doubled in place
/// then comes out in group order, one half of the groups at a time.
const FOR_PAIRS: [u8; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

struct Row {
    row: Gpr,
    quants: Gpr,
    scales: Gpr,
}

/// The constants every group uses, loaded.
fn group_constants(a: &mut Asm, f: &Format) -> Group {
    a.note("constants");
    let lut: [u8; 32] = std::array::from_fn(|i| f.levels[i % 16] as u8);
    let levels_at = a.u8s(&lut);
    let nibble_at = a.u32s(&[0x0f0f_0f0f]);
    let one16_at = a.u32s(&[0x0001_0001]);
    let k = Group { levels: a.ymm("levels"), nibble: a.ymm("nibble"), one16: a.ymm("one16") };
    a.vmovdqa(k.levels, levels_at);
    a.vpbroadcastd(k.nibble, nibble_at);
    a.vpbroadcastd(k.one16, one16_at);
    k
}

fn superblock_constants(a: &mut Asm, f: &Format, order: [u8; 8]) -> Superblock {
    let mut pairs = [0x80u8; 16];
    for (j, &g) in order.iter().enumerate() {
        pairs[j] = g / 2;
    }
    let scale_pairs = a.u8s(&pairs);
    let nibble_shifts = a.u32s(&order.map(|g| 4 * u32::from(g % 2)));
    let nibble = a.u32s(&[15; 8]);
    let kind = match f.scales {
        Scales::Odd { quarter } => Kind::Odd { one: a.u32s(&[1; 8]), quarter: quarter.then(|| a.f32s(&[0.25; 8])) },
        Scales::Six => Kind::Six {
            high_shifts: a.u32s(&order.map(|g| 2 * u32::from(g))),
            pair: a.u32s(&[3; 8]),
            less: a.u32s(&[32u32.wrapping_neg(); 8]),
        },
    };
    Superblock { scale_pairs, nibble_shifts, nibble, kind }
}

fn one(f: &Format, ahead: usize) -> Program<Op> {
    assert!((1..8).contains(&ahead), "ahead {ahead}");
    let mut a = Asm::new(format!("{}_ahead{ahead}", f.name));
    let row = a.param("row");
    let n = a.param("sbs");
    let quants = a.param("quants");
    let scales = a.param("scales");
    let r = Row { row, quants, scales };
    let (empty, top, last) = (a.label(), a.label(), a.label());
    a.test(n, n);
    a.jz(empty);

    let k = group_constants(&mut a, f);
    let s = superblock_constants(&mut a, f, IN_ORDER);
    let acc = a.ymm("acc");
    a.zero(acc);
    let slot = a.slot(64, 64);
    let (cur, nxt) = (a.gpr("cur"), a.gpr("nxt"));
    a.lea(cur, slot.at(0));
    a.lea(nxt, slot.at(32));

    let ring: [Ymm; 8] = std::array::from_fn(|g| a.ymm(format!("p{g}")));
    let scales_to = |a: &mut Asm, sb: i32, to: Gpr| {
        let ds = group_scales(a, f, &s, &r, sb);
        a.vmulps(ds, ds, r.scales.at(32 * sb));
        a.store_vmovaps(to.at(0), ds);
    };
    scales_to(&mut a, 0, cur);
    for (g, &p) in ring.iter().enumerate().take(ahead) {
        products(&mut a, f, &k, &r, &[p], 0, g as i32);
    }
    a.dec(n);
    a.jz(last);

    a.place(top);
    let mut next_scales = a.capture(|a| scales_to(a, 1, nxt));
    let share = next_scales.len().div_ceil(8);
    for g in 0..8 {
        a.splice(&mut next_scales, share);
        match g + ahead {
            j if j < 8 => products(&mut a, f, &k, &r, &[ring[j]], 0, j as i32),
            j => products(&mut a, f, &k, &r, &[ring[j - 8]], 1, (j - 8) as i32),
        }
        a.note(format!("sum of group {g}"));
        sum(&mut a, k.one16, acc, ring[g], cur.at(4 * g as i32));
    }
    a.add(r.row, f.bytes);
    a.add(r.quants, 256);
    a.add(r.scales, 32);
    a.xor(cur, 32);
    a.xor(nxt, 32);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    for g in 0..8 {
        if g + ahead < 8 {
            products(&mut a, f, &k, &r, &[ring[g + ahead]], 0, (g + ahead) as i32);
        }
        a.note(format!("sum of group {g}"));
        sum(&mut a, k.one16, acc, ring[g], cur.at(4 * g as i32));
    }
    let result = lanes_summed(&mut a, acc);
    a.ret_f32(result);

    a.place(empty);
    let nothing = a.ymm("nothing");
    a.zero(nothing);
    a.ret_f32(nothing.xmm());
    a.finish()
}

fn two(f: &Format, ahead: usize) -> Program<Op> {
    assert!((1..8).contains(&ahead), "ahead {ahead}");
    let mut a = Asm::new(format!("{}2_ahead{ahead}", f.name));
    let row = a.param("row");
    let n = a.param("sbs");
    let quants = a.param("quants");
    let scales = a.param("scales");
    let out = a.param("out");
    let r = Row { row, quants, scales };
    let (empty, top, last) = (a.label(), a.label(), a.label());
    a.test(n, n);
    a.jz(empty);

    let k = group_constants(&mut a, f);
    let s = superblock_constants(&mut a, f, FOR_PAIRS);
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
    // Group `g`'s scale times vector `k`'s at float `2 * g + k` of the half-slot: the vectors'
    // block scales lie in that order already.
    let scales_to = |a: &mut Asm, sb: i32, to: Gpr| {
        let ds = group_scales(a, f, &s, &r, sb);
        for (half, control) in [(0, 0x50), (1, 0xfa)] {
            let pair = a.ymm("pair_scale");
            a.vshufps(pair, ds, ds, control);
            a.vmulps(pair, pair, r.scales.at(64 * sb + 32 * half));
            a.store_vmovaps(to.at(32 * half), pair);
        }
    };
    scales_to(&mut a, 0, cur);
    for (g, p) in ring.iter().enumerate().take(ahead) {
        products(&mut a, f, &k, &r, p, 0, g as i32);
    }
    a.dec(n);
    a.jz(last);

    a.place(top);
    let mut next_scales = a.capture(|a| scales_to(a, 1, nxt));
    let share = next_scales.len().div_ceil(8);
    for g in 0..8 {
        // The sums first: the accumulators' chains are the longest, and the oldest uop ready
        // is the one a port takes.
        pair_sums(&mut a, k.one16, acc, ring[g], cur, g as i32);
        a.splice(&mut next_scales, share);
        match g + ahead {
            j if j < 8 => products(&mut a, f, &k, &r, &ring[j], 0, j as i32),
            j => products(&mut a, f, &k, &r, &ring[j - 8], 1, (j - 8) as i32),
        }
    }
    a.add(r.row, f.bytes);
    a.add(r.quants, 512);
    a.add(r.scales, 64);
    a.xor(cur, 64);
    a.xor(nxt, 64);
    a.dec(n);
    a.jnz(top);

    a.place(last);
    for g in 0..8 {
        pair_sums(&mut a, k.one16, acc, ring[g], cur, g as i32);
        if g + ahead < 8 {
            products(&mut a, f, &k, &r, &ring[g + ahead], 0, (g + ahead) as i32);
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

/// The eight group scales of superblock `+sb`, each `d` times its group's factor, in the lane
/// order `s` was made for.
fn group_scales(a: &mut Asm, f: &Format, s: &Superblock, r: &Row, sb: i32) -> Ymm {
    a.note(format!("scales of superblock +{sb}"));
    let at = f.bytes * sb;
    let d = a.ymm("d");
    a.vpbroadcastw(d.xmm(), r.row.at(at));
    a.vcvtph2ps(d, d.xmm());
    let steps = a.ymm("steps");
    let nibbles = match f.scales {
        Scales::Odd { .. } => 2,
        Scales::Six => 4,
    };
    a.vmovd(steps.xmm(), r.row.at(at + nibbles));
    a.vpshufb(steps.xmm(), steps.xmm(), s.scale_pairs);
    a.vpmovzxbd_reg(steps, steps.xmm());
    a.vpsrlvd(steps, steps, s.nibble_shifts);
    a.vpand(steps, steps, s.nibble);
    let scale = a.ymm("scale");
    match s.kind {
        Kind::Odd { one, quarter } => {
            a.vpaddd(scale, steps, steps);
            a.vpor(scale, scale, one);
            a.vcvtdq2ps(scale, scale);
            a.vmulps(scale, d, scale);
            if let Some(quarter) = quarter {
                a.vmulps(scale, scale, quarter);
            }
        }
        Kind::Six { high_shifts, pair, less } => {
            // Every dword holds the word twice; a shift below 16 reads the low copy.
            a.vpbroadcastw(scale, r.row.at(at + 2));
            a.vpsrlvd(scale, scale, high_shifts);
            a.vpand(scale, scale, pair);
            a.vpslld(scale, scale, 4);
            a.vpor(scale, scale, steps);
            a.vpaddd(scale, scale, less);
            a.vcvtdq2ps(scale, scale);
            a.vmulps(scale, d, scale);
        }
    }
    scale
}

/// Group `g`'s codes expanded to signed levels once, then multiplied by each vector's
/// correspondingly signed activations in saturated 16-bit pairs, into `p[k]` for vector `k`.
fn products(a: &mut Asm, f: &Format, k: &Group, r: &Row, p: &[Ymm], sb: i32, g: i32) {
    a.note(format!("products of group {g} of superblock +{sb}"));
    let raw = a.ymm("packed");
    let low = a.ymm("low_codes");
    a.vmovdqu(raw.xmm(), r.row.at(f.bytes * sb + f.codes + 16 * g));
    a.vpand(low.xmm(), raw.xmm(), k.nibble.xmm());
    a.vpsrlw(raw.xmm(), raw.xmm(), 4);
    a.vpand(raw.xmm(), raw.xmm(), k.nibble.xmm());
    // With one vector the signed levels go straight into its product's register.
    let signed = if p.len() == 1 { p[0] } else { a.ymm("signed_levels") };
    a.vinserti128(signed, low, raw.xmm(), 1);
    a.vpshufb(signed, k.levels, signed);
    let levels = a.ymm("unsigned_levels");
    a.vpabsb(levels, signed);
    let vectors = p.len() as i32;
    for (v, &p) in p.iter().enumerate() {
        let x = a.ymm("x");
        a.vmovdqa(x, r.quants.at(256 * vectors * sb + 32 * (vectors * g + v as i32)));
        a.vpsignb(x, x, signed);
        a.vpmaddubsw(p, levels, x);
    }
}
