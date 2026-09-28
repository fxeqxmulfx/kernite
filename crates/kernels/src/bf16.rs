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
    bf16_rows(2, ahead, hint)
}

/// Three BF16 rows per activation load, with each row reduced in the same order as
/// [`bf16_pairs`]. The third row uses the otherwise idle vector registers.
pub fn bf16_triples(ahead: i32, hint: Hint) -> Program<Op> {
    bf16_rows(3, ahead, hint)
}

/// BF16 rows against two vectors of floats, each row read once for both:
///
/// `fn(w: *const u16, x: *const f32, out: *mut f32, gap: usize, rows: usize, blocks: usize)`
///
/// The rows lie back to back from `w`, `32 * blocks` values each, and so do the two vectors
/// from `x`, as long as a row each; `rows` and `blocks` are at least one. `out[r]` gets row `r`
/// times the first vector, and the float `gap` bytes on from it row `r` times the second.
///
/// Every product has the bits [`bf16_pairs`] gives it: four accumulators to a row and a vector,
/// summed the same way. A row's values are widened once for both vectors. With `ahead` bytes,
/// the row asks for its line that far on each time it reads one, as `hint` says.
pub fn bf16_two(ahead: i32, hint: Hint) -> Program<Op> {
    assert!((0..=4096).contains(&ahead), "ahead {ahead}");
    let mut a = Asm::new(match ahead {
        0 => "bf16_two".to_string(),
        _ => format!("bf16_two_ahead{ahead}_{}", hint.suffix()),
    });
    let w = a.param("w");
    let x = a.param("x");
    let out = a.param("out");
    let gap = a.param("gap");
    let rows = a.param("rows");
    let blocks = a.param("blocks");
    let (row, block) = (a.label(), a.label());
    let (second, x0, x1, left) = (a.gpr("second"), a.gpr("x0"), a.gpr("x1"), a.gpr("left"));
    a.note("the second vector, 128 bytes a block past the first");
    a.mov(second, blocks);
    a.shl(second, 7);
    a.lea(second, x.at(0).index(second, 1));

    a.place(row);
    a.note("a row");
    let acc: Vec<[Ymm; 4]> = (0..2).map(|v| std::array::from_fn(|k| a.ymm(format!("acc{v}{k}")))).collect();
    for &v in acc.iter().flatten() {
        a.zero(v);
    }
    a.mov(x0, x);
    a.mov(x1, second);
    a.mov(left, blocks);

    a.place(block);
    a.note("32 values of the row");
    if ahead > 0 {
        a.prefetch(hint, w.at(ahead));
    }
    for k in 0..4 {
        let v = a.ymm("w8");
        a.vpmovzxwd(v, w.at(16 * k));
        a.vpslld(v, v, 16);
        a.vfmadd231ps(acc[0][k as usize], v, x0.at(32 * k));
        a.vfmadd231ps(acc[1][k as usize], v, x1.at(32 * k));
    }
    a.add(w, 64);
    a.add(x0, 128);
    a.add(x1, 128);
    a.dec(left);
    a.jnz(block);

    let sum = reduce(&mut a, &acc[0]);
    a.store_vmovss(out.at(0), sum);
    let sum = reduce(&mut a, &acc[1]);
    a.store_vmovss(out.at(0).index(gap, 1), sum);
    a.add(out, 4);
    a.dec(rows);
    a.jnz(row);
    a.ret();
    a.finish()
}

fn bf16_rows(group: usize, ahead: i32, hint: Hint) -> Program<Op> {
    assert!(group == 2 || group == 3);
    assert!((0..=4096).contains(&ahead), "ahead {ahead}");
    let base = if group == 2 { "bf16_pairs" } else { "bf16_triples" };
    let mut a = Asm::new(match ahead {
        0 => base.to_string(),
        _ => format!("{base}_ahead{ahead}_{}", hint.suffix()),
    });
    let w = a.param("w");
    let x = a.param("x");
    let out = a.param("out");
    let pairs = a.param("pairs");
    let stride = a.param("stride");
    let blocks = a.param("blocks");
    let (row_group, block) = (a.label(), a.label());
    let rows: Vec<_> = (0..group).map(|r| a.gpr(format!("row{r}"))).collect();
    let (xs, left) = (a.gpr("xs"), a.gpr("left"));

    a.place(row_group);
    a.note(if group == 2 { "a pair of rows" } else { "three rows" });
    let acc: Vec<[Ymm; 4]> = (0..group).map(|r| std::array::from_fn(|k| a.ymm(format!("acc{r}{k}")))).collect();
    for &v in acc.iter().flatten() {
        a.zero(v);
    }
    a.mov(rows[0], w);
    for (r, &row) in rows.iter().enumerate().skip(1) {
        a.lea(row, w.at(0).index(stride, r as u8));
    }
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
    a.add(out, (4 * group) as i32);
    a.lea(w, w.at(0).index(stride, 2));
    if group == 3 {
        a.lea(w, w.at(0).index(stride, 1));
    }
    a.dec(pairs);
    a.jnz(row_group);
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
        assert_eq!(bf16_two(0, Hint::T0).name, "bf16_two");
        assert_eq!(bf16_two(2048, Hint::T0).name, "bf16_two_ahead2048_t0");
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
