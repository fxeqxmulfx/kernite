//! How the row kernels add their products up: each group's into the lanes, the lanes at the end.

use kgen_x86::{Addr, Asm, Gpr, Xmm, Ymm};

/// A group's products `p`, in 16-bit pairs, summed into 32-bit lanes (`one16` is 1 in every
/// 16-bit lane), converted, and added into `acc` at the scale at `scale_at` with one fused
/// multiply-add.
pub(crate) fn sum(a: &mut Asm, one16: Ymm, acc: Ymm, p: Ymm, scale_at: Addr) {
    let lanes = a.ymm("lanes");
    a.vpmaddwd(lanes, p, one16);
    a.vcvtdq2ps(lanes, lanes);
    let scale = a.ymm("group_scale");
    a.vbroadcastss(scale, scale_at);
    a.vfmadd231ps(acc, lanes, scale);
}

/// Group `g`'s products for two vectors, `p[v]`, each [`sum`]med into its accumulator at float
/// `2g + v` of `from`.
pub(crate) fn pair_sums(a: &mut Asm, one16: Ymm, acc: [Ymm; 2], p: [Ymm; 2], from: Gpr, g: i32) {
    a.note(format!("sums of group {g}"));
    for v in 0..2 {
        sum(a, one16, acc[v], p[v], from.at(4 * (2 * g + v as i32)));
    }
}

/// `hsum(acc + 0.0)`, as the row kernels add their lanes: the high 128 bits onto the low, then
/// the high pair, then lane 1 onto lane 0, into the low lane of what it returns.
pub(crate) fn lanes_summed(a: &mut Asm, acc: Ymm) -> Xmm {
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
