//! The product on the CPU: a kernel in x86, which needs AVX2 and FMA, blocked as GotoBLAS and
//! BLIS block it, so that each operand is read from the cache that holds it.

/// The kernels.
#[kernite_macros::kernels]
pub mod kernels {
    /// Rows of `c` the micro-kernel keeps in registers.
    pub const MR: usize = 6;
    /// Columns of `c` it keeps in registers: two vectors of eight a row, twelve of the sixteen
    /// `ymm`s, with two for a row of `b` and one for a float of `a`.
    pub const NR: usize = 16;
    /// Steps of `k` a block takes: a sliver of `b`'s panel, `KC` rows of `NR` floats, 16 KB,
    /// stays in L1 while the slivers of `a`'s block pass it.
    pub const KC: usize = 256;
    /// Rows of `a` a block takes: `MC` by `KC` floats, 144 KB, stay in L2.
    pub const MC: usize = 144;
    /// Columns of `b` a panel takes: `KC` by `NC` floats, 4 MB, stay in L3.
    pub const NC: usize = 4096;
    /// Steps of `k` a turn of the micro-kernel's loop takes.
    pub const UNROLL: usize = 4;

    /// `c = a b`, of `a`, `m` rows of `k` floats, and `b`, `k` rows of `n`, into `c`, `m` rows
    /// of `n`, where `shape` points at `m`, `n` and `k`, in a row: each element of `c` a chain
    /// of fused multiply-adds over `k` in order, from +0.
    ///
    /// `b` is packed a panel at a time, `KC` rows of `NC` columns at most, into `packed`, which
    /// has room for `min(k, KC)` rows of `min(n, NC)` columns rounded up to `NR`, and for 15
    /// floats more before them, to reach a multiple of 64 bytes. `a` is packed a block at a
    /// time, `MC` rows of `KC`, into the kernel's stack memory. The micro-kernel multiplies a
    /// sliver of each, `MR` rows of `a` and `NR` columns of `b`, into a tile of `c` in
    /// registers: from +0 for the first block of `k`, and from what the tile holds for each
    /// after, so that each chain goes on in order. Past the edges of `c` the slivers hold
    /// zeros, and of a tile there only what lies in `c` is read and written.
    pub unsafe extern "sysv64" fn sgemm(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        shape: *const usize,
        packed: *mut f32,
    ) {
        let block: local<[f32; MC * KC], 64>;
        let (m, n, k): (gpr, gpr, gpr);
        mov(m, [shape]);
        mov(n, [shape + 8]);
        mov(k, [shape + 16]);
        test(m, m);
        jz(done);
        test(n, n);
        jz(done);
        add(packed, 63);
        and(packed, -64);
        note("the bytes from a row of b or c to the next");
        let ld: gpr;
        mov(ld, n);
        shl(ld, 2);

        note("panels of NC columns of b, and of c");
        let (bj, cj, nleft): (gpr, gpr, gpr);
        mov(bj, b);
        mov(cj, c);
        mov(nleft, n);
        label(panel);
        let nc: gpr;
        mov(nc, nleft);
        at_most::<NC>(nc);

        note("blocks of KC steps of k from step pc, each chain going on from the block before");
        let (pc, bk): (gpr, gpr);
        zero(pc);
        mov(bk, bj);
        label(depth);
        let kc: gpr;
        mov(kc, k);
        sub(kc, pc);
        at_most::<KC>(kc);
        pack_b(packed, bk, ld, nc, kc);

        note("blocks of MC rows of a, and of c");
        let (ai, ci, mleft): (gpr, gpr, gpr);
        lea(ai, [a + pc * 4]);
        mov(ci, cj);
        mov(mleft, m);
        label(rows);
        let (mc, pa): (gpr, gpr);
        mov(mc, mleft);
        at_most::<MC>(mc);
        lea(pa, [block]);
        pack_a(pa, ai, k, mc, kc);

        note("slivers of NR columns of the panel");
        let (pb, cs, wleft): (gpr, gpr, gpr);
        mov(pb, packed);
        mov(cs, ci);
        mov(wleft, nc);
        label(columns);
        let (ct, hleft): (gpr, gpr);
        lea(pa, [block]);
        mov(ct, cs);
        mov(hleft, mc);
        note("slivers of MR rows of the block, a tile of c each");
        label(tile);
        micro(ct, ld, pa, pb, kc, pc, hleft, wleft);
        lea(ct, [ct + ld * 4]);
        lea(ct, [ct + ld * 2]);
        sub(hleft, MR);
        ja(tile);
        let t: gpr;
        mov(t, kc);
        shl(t, 6);
        add(pb, t);
        add(cs, 4 * NR);
        sub(wleft, NR);
        ja(columns);

        imul(t, k, 4 * MC);
        add(ai, t);
        imul(t, ld, MC);
        add(ci, t);
        sub(mleft, MC);
        ja(rows);

        imul(t, ld, KC);
        add(bk, t);
        add(pc, KC);
        cmp(pc, k);
        jb(depth);

        add(bj, 4 * NC);
        add(cj, 4 * NC);
        sub(nleft, NC);
        ja(panel);
        label(done);
    }

    /// `x = min(x, MOST)`.
    fn at_most<const MOST: usize>(x: gpr) {
        let most: gpr;
        mov(most, MOST);
        cmp(x, most);
        cmova(x, most);
    }

    /// Packs `kc` rows of `nc` floats at `from`, rows `ld` bytes apart, into slivers of `NR`
    /// columns at `to`, a sliver's rows one after another: zeros past `nc`.
    fn pack_b(to: gpr, from: gpr, ld: gpr, nc: gpr, kc: gpr) {
        test(kc, kc);
        jz(end);
        let (q, col, left): (gpr, gpr, gpr);
        mov(q, to);
        mov(col, from);
        mov(left, nc);
        label(sliver);
        let mask: [ymm; 2];
        masks(mask, left);
        let (p, l): (gpr, gpr);
        mov(p, col);
        mov(l, kc);
        label(row);
        for half in 0..2 {
            let v: ymm;
            vmaskmovps(v, mask[half], [p + 32 * half]);
            vmovaps([q + 32 * half], v);
        }
        add(p, ld);
        add(q, 4 * NR);
        dec(l);
        jnz(row);
        add(col, 4 * NR);
        sub(left, NR);
        ja(sliver);
        label(end);
    }

    /// Packs `mc` rows of `kc` floats at `from`, rows `k` floats apart, into slivers of `MR`
    /// rows at `to`, a sliver's steps of `k` one after another, `MR` floats a step: zeros past
    /// `mc`.
    fn pack_a(to: gpr, from: gpr, k: gpr, mc: gpr, kc: gpr) {
        test(kc, kc);
        jz(end);
        let (lda, q, p, left): (gpr, gpr, gpr, gpr);
        mov(lda, k);
        shl(lda, 2);
        mov(q, to);
        mov(p, from);
        mov(left, mc);
        label(sliver);
        let (r, at): (gpr, gpr);
        zero(r);
        mov(at, q);
        label(row);
        let (s, d, l, f): (gpr, gpr, gpr, gpr);
        mov(d, at);
        mov(l, kc);
        cmp(r, left);
        jae(pad);
        mov(s, p);
        label(each);
        mov(f.r32, [s]);
        mov([d], f.r32);
        add(s, 4);
        add(d, 4 * MR);
        dec(l);
        jnz(each);
        jmp(next);
        label(pad);
        mov(dword[d], 0);
        add(d, 4 * MR);
        dec(l);
        jnz(pad);
        label(next);
        add(p, lda);
        add(at, 4);
        inc(r);
        cmp(r, MR);
        jb(row);
        lea(q, [d - 4 * (MR - 1)]);
        sub(left, MR);
        ja(sliver);
        label(end);
    }

    /// In `mask`, the masks of the first `min(w, NR)` of `NR` lanes.
    fn masks(mask: [ymm; 2], w: gpr) {
        let t: gpr;
        mov(t, w);
        at_most::<NR>(t);
        let lanes: ymm;
        vmovd(lanes.xmm, t.r32);
        vpbroadcastd(lanes, lanes.xmm);
        vpcmpgtd(mask[0], lanes, [[0, 1, 2, 3, 4, 5, 6, 7]]);
        vpcmpgtd(mask[1], lanes, [[8, 9, 10, 11, 12, 13, 14, 15]]);
    }

    /// The tile of `c` at `c`, rows `ld` bytes apart, plus the product of the `kc` steps of the
    /// slivers at `x`, of `a`, and at `y`, of `b`; or the product alone where `pc` is 0, the
    /// first block of `k`. Of the tile, `h` rows and `w` columns lie in `c`. `x` ends at the
    /// next sliver of `a`.
    fn micro(c: gpr, ld: gpr, x: gpr, y: gpr, kc: gpr, pc: gpr, h: gpr, w: gpr) {
        let acc: [ymm; 2 * MR];
        for i in 0..2 * MR {
            zero(acc[i]);
        }
        test(pc, pc);
        jz(steps);
        tile::<false>(acc, c, ld, h, w);

        label(steps);
        note("UNROLL steps of k a turn");
        let (p, l): (gpr, gpr);
        mov(p, y);
        mov(l, kc);
        sub(l, UNROLL);
        jb(rest);
        label(turn);
        for s in 0..UNROLL {
            step::<s>(acc, x, p);
        }
        add(x, 4 * MR * UNROLL);
        add(p, 4 * NR * UNROLL);
        sub(l, UNROLL);
        jae(turn);
        label(rest);
        note("the steps past the last whole turn");
        add(l, UNROLL);
        jz(store);
        label(one);
        step::<0>(acc, x, p);
        add(x, 4 * MR);
        add(p, 4 * NR);
        dec(l);
        jnz(one);
        label(store);
        tile::<true>(acc, c, ld, h, w);
    }

    /// Step `S` of a turn: a row of the sliver of `b` at `y` by each of the `MR` floats of the
    /// sliver of `a` at `x`, added to `acc`, two vectors a row.
    fn step<const S: usize>(acc: [ymm; 2 * MR], x: gpr, y: gpr) {
        let (b0, b1): (ymm, ymm);
        vmovaps(b0, [y + 4 * NR * S]);
        vmovaps(b1, [y + 4 * NR * S + 32]);
        for r in 0..MR {
            let s: ymm;
            vbroadcastss(s, [x + 4 * (MR * S + r)]);
            vfmadd231ps(acc[2 * r], s, b0);
            vfmadd231ps(acc[2 * r + 1], s, b1);
        }
    }

    /// Stores `acc` to the tile of `c` at `c`, rows `ld` bytes apart, if `STORE`, or else
    /// loads it from there: the whole tile where it lies in `c`, `h` rows of `MR` or more and
    /// `w` columns of `NR` or more; else, by masks, only what lies in `c`.
    fn tile<const STORE: bool>(acc: [ymm; 2 * MR], c: gpr, ld: gpr, h: gpr, w: gpr) {
        let p: gpr;
        mov(p, c);
        cmp(h, MR);
        jb(edge);
        cmp(w, NR);
        jb(edge);
        for r in 0..MR {
            for half in 0..2 {
                if STORE {
                    vmovups([p + 32 * half], acc[2 * r + half]);
                } else {
                    vmovups(acc[2 * r + half], [p + 32 * half]);
                }
            }
            add(p, ld);
        }
        jmp(done);
        label(edge);
        let mask: [ymm; 2];
        masks(mask, w);
        for r in 0..MR {
            if r > 0 {
                cmp(h, r);
                jbe(done);
            }
            for half in 0..2 {
                if STORE {
                    vmaskmovps([p + 32 * half], mask[half], acc[2 * r + half]);
                } else {
                    vmaskmovps(acc[2 * r + half], mask[half], [p + 32 * half]);
                }
            }
            add(p, ld);
        }
        label(done);
    }
}

/// Whether this CPU runs the kernels.
pub fn supported() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// `c = a b` by the kernel, of the operands of `shape`.
pub fn sgemm(c: &mut [f32], a: &[f32], b: &[f32], shape: crate::Shape) {
    let (m, n, k) = shape;
    assert!(supported(), "the kernels need AVX2 and FMA");
    crate::operands((a.len(), b.len(), c.len()), shape);
    use kernels::{KC, NC, NR};
    let mut packed = Vec::with_capacity(k.min(KC) * n.min(NC).next_multiple_of(NR) + 15);
    // SAFETY: `m` rows of `k` floats behind `a`, `k` rows of `n` behind `b` and `m` rows of `n`
    // behind `c`, as `shape` says; room for a panel of `b` in `packed`, and the CPU has what the
    // kernel needs.
    unsafe { kernels::sgemm(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), [m, n, k].as_ptr(), packed.as_mut_ptr()) }
}
