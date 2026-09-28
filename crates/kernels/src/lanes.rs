//! What the row kernels share at their end.

use kgen_x86::{Asm, Xmm, Ymm};

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
