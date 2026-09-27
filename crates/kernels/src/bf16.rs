use kgen_core::Program;
use kgen_x86::{Asm, Hint, Op, Xmm, Ymm};

/// Pairs of BF16 rows against one vector of floats:
///
/// `fn(w: *const u16, x: *const f32, out: *mut f32, pairs: usize, stride: usize, blocks: usize)`
///
/// Row `r` starts `r * stride` bytes into `w`, and pair `p` is rows `2p` and `2p + 1`. A row
/// and `x` are `32 * blocks` values; `pairs` and `blocks` are at least one. `out[r]` gets row
/// `r` times `x`.
///
/// The bits are those of `dot_bf16_f32x2` in the `roofline` kernels: a weight `w` is the float
/// `w << 16`; a row has four accumulators, and values `8k..8k + 8` of every 32 go into
/// accumulator `k`, one fused multiply-add each, in order; they are summed as
/// `(a0 + a1) + (a2 + a3)`, then its eight lanes in order from lane 0.
///
/// With `ahead` bytes, each row asks for the line `ahead` bytes on, as `hint` says, each time
/// it reads a line. That is for rows that sit in L3 rather than memory: there the L2 streamer
/// starts over at every 4 KB page, and on its own kept about five lines in flight a core in the
/// model's hyper-connection rows. Past a row's end it asks for the next row's lines, and past
/// the last row's for whatever lies there, which a prefetch may do.
pub fn bf16_pairs(ahead: i32, hint: Hint) -> Program<Op> {
    assert!((0..=4096).contains(&ahead), "ahead {ahead}");
    let mut a = Asm::new(match ahead {
        0 => "bf16_pairs".to_string(),
        _ => format!("bf16_pairs_ahead{ahead}_{}", hint.suffix()),
    });
    let w = a.param("w");
    let x = a.param("x");
    let out = a.param("out");
    let pairs = a.param("pairs");
    let stride = a.param("stride");
    let blocks = a.param("blocks");
    let (pair, block) = (a.label(), a.label());
    let rows = [a.gpr("row0"), a.gpr("row1")];
    let (xs, left) = (a.gpr("xs"), a.gpr("left"));

    a.place(pair);
    a.note("a pair of rows");
    let acc: [[Ymm; 4]; 2] = std::array::from_fn(|r| std::array::from_fn(|k| a.ymm(format!("acc{r}{k}"))));
    for &v in acc.iter().flatten() {
        a.zero(v);
    }
    a.mov(rows[0], w);
    a.lea(rows[1], w.at(0).index(stride, 1));
    a.mov(xs, x);
    a.mov(left, blocks);

    a.place(block);
    a.note("32 values of each row");
    if ahead > 0 {
        for &row in &rows {
            a.prefetch(hint, row.at(ahead));
        }
    }
    for k in 0..4 {
        let xv = a.ymm("x8");
        a.vmovups(xv, xs.at(32 * k));
        for (r, &row) in rows.iter().enumerate() {
            let v = a.ymm(format!("w{r}"));
            a.vpmovzxwd(v, row.at(16 * k));
            a.vpslld(v, v, 16);
            a.vfmadd231ps(acc[r][k as usize], v, xv);
        }
    }
    for &row in &rows {
        a.add(row, 64);
    }
    a.add(xs, 128);
    a.dec(left);
    a.jnz(block);

    for (r, acc) in acc.iter().enumerate() {
        let sum = reduce(&mut a, acc);
        a.store_vmovss(out.at(4 * r as i32), sum);
    }
    a.add(out, 8);
    a.lea(w, w.at(0).index(stride, 2));
    a.dec(pairs);
    a.jnz(pair);
    a.ret();
    a.finish()
}

/// `(acc0 + acc1) + (acc2 + acc3)`, then its lanes added in order from lane 0, into the low
/// lane of what it returns.
fn reduce(a: &mut Asm, acc: &[Ymm; 4]) -> Xmm {
    a.note("the accumulators summed, then the lanes in order");
    let (s, t) = (a.ymm("s"), a.ymm("t"));
    a.vaddps(s, acc[0], acc[1]);
    a.vaddps(t, acc[2], acc[3]);
    a.vaddps(s, s, t);
    let high = a.ymm("high");
    a.vextractf128(high.xmm(), s, 1);
    let (sum, lane) = (a.ymm("sum").xmm(), a.ymm("lane").xmm());
    let half = |a: &mut Asm, v: Xmm, sum: Xmm, first: bool| {
        if first {
            a.vmovshdup(lane, v);
            a.vaddss(sum, v, lane);
        } else {
            a.vaddss(sum, sum, v);
            a.vmovshdup(lane, v);
            a.vaddss(sum, sum, lane);
        }
        a.vmovhlps(lane, v, v);
        a.vaddss(sum, sum, lane);
        a.vshufps(lane, v, v, 0xff);
        a.vaddss(sum, sum, lane);
    };
    half(a, s.xmm(), sum, true);
    half(a, high.xmm(), sum, false);
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_say_the_distance_and_the_hint() {
        assert_eq!(bf16_pairs(0, Hint::T0).name, "bf16_pairs");
        assert_eq!(bf16_pairs(1024, Hint::T1).name, "bf16_pairs_ahead1024_t1");
    }

    #[test]
    fn prefetches_only_when_asked() {
        let prefetches = |p: &Program<Op>| {
            p.body.iter().filter(|i| matches!(i, kgen_core::Item::Inst(inst) if inst.op == Op::Prefetcht0)).count()
        };
        assert_eq!(prefetches(&bf16_pairs(0, Hint::T0)), 0);
        assert_eq!(prefetches(&bf16_pairs(512, Hint::T0)), 2);
    }
}
