//! The product on the CPU: a kernel in x86, which needs AVX2 and FMA, blocked as GotoBLAS and
//! OpenBLAS block it, so that each operand is read from the cache that holds it.

/// The kernels.
#[kernite_macros::kernels]
pub mod kernels {
    /// Rows of `c` the micro-kernel keeps in registers.
    pub const MR: usize = 6;
    /// Columns of `c` it keeps in registers: two vectors of eight a row, twelve of the sixteen
    /// `ymm`s, with two for a row of `b` and one for a float of `a`.
    pub const NR: usize = 16;
    /// Steps of `k` a block takes: a sliver of `a`, `MR` rows of `KC` floats, 18 KB, is read from
    /// L1 while the slivers of `b`'s block pass it.
    pub const KC: usize = 768;
    /// Columns of `b` a strip takes at most: `KC` rows of `NC` floats, 192 KB, stay in L2 while
    /// the slivers of `a` pass them.
    pub const NC: usize = 64;
    /// Rows of `a` a thread packs at a time: whole slivers.
    pub const PACK: usize = 4 * MR;
    /// Steps of `k` a turn of the micro-kernel's loop takes.
    pub const UNROLL: usize = 4;

    /// `c = a b`, of `a`, `m` rows of `k` floats, and `b`, `k` rows of `n`, into `c`, `m` rows
    /// of `n`, on the threads of a grid: each element of `c` a chain of fused multiply-adds over
    /// `k` in order, from +0. `m` and `n` are 1 or more.
    ///
    /// `shape` points at six words: `m`, `n` and `k`; how many parts of `PACK` rows `a` has; how
    /// many pieces `c` has; and where the counters are. The pieces follow, four words each: the
    /// first column, and how many, which is `NC` or fewer; the first row, a multiple of `MR`, and
    /// how many, which lie in `c`. The pieces lie side by side in `c`, and none shares an element
    /// with another. The counters are all zero, four for each block of `KC` steps of `k`: the parts
    /// taken and packed, the pieces taken and done. Each is a line of the cache to itself, 64 bytes
    /// from a multiple of 64, so that threads taking from one do not slow those taking from
    /// another.
    ///
    /// For each block of `KC` steps of `k`, the threads pack `a` into slivers of `MR` rows at
    /// `packed`, which has room for `m` rows, rounded up to `MR`, of `min(k, KC)` floats: they grow
    /// with `m`, so the caller hands them in, from a multiple of 64 bytes, so that threads packing
    /// parts side by side share a line of it only where a block takes an odd number of steps. Each
    /// part of `a` goes to the first thread to come for it. When every part is packed, the threads
    /// take the pieces of `c` the same way: `b`, the columns of a piece, is packed into slivers of
    /// `NR` columns in the kernel's stack memory, which stays a thread's own, in its core's L2. The
    /// micro-kernel multiplies a sliver of each into a tile of `c` in registers: from +0 for the
    /// first block of `k`, and from what the tile holds for each after, so that each chain goes on
    /// in order, whichever thread has the piece. When every piece is done, the next block of `k`
    /// packs over `a`'s. A thread waits for work to be done, not for threads to come: one that
    /// comes late, or is kept back, holds up only the part or the piece it took. Past the edges of
    /// `c` the slivers hold zeros, and of a tile there only what lies in `c` is read and written.
    #[cooperative]
    pub unsafe extern "sysv64" fn sgemm(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        shape: *const usize,
        packed: *mut f32,
    ) {
        let block: local<[f32; KC * NC], 64>;
        let (m, n, k): (gpr, gpr, gpr);
        mov(m, [shape]);
        mov(n, [shape + 8]);
        mov(k, [shape + 16]);
        note("the bytes from a row of b or c to the next");
        let ld: gpr;
        mov(ld, n);
        shl(ld, 2);

        note("blocks of KC steps of k from step pc, each chain going on from the block before");
        let (pc, ak, bk, counters): (gpr, gpr, gpr, gpr);
        zero(pc);
        mov(ak, a);
        mov(bk, b);
        note("a block's counters, a line each: parts taken, +64 packed; +128 pieces taken, +192 done");
        mov(counters, [shape + 40]);
        label(depth);
        let kc: gpr;
        mov(kc, k);
        sub(kc, pc);
        at_most::<KC>(kc);

        note("parts of PACK rows of a, packed by the first thread to come for each");
        label(parts);
        let part: gpr;
        mov(part, 1);
        lock.xadd([counters], part);
        cmp(part, [shape + 24]);
        jae(packed_all);
        let (sliver, row, mine): (gpr, gpr, gpr);
        imul(row, part, PACK);
        mov(mine, m);
        sub(mine, row);
        at_most::<PACK>(mine);
        mov(sliver, row);
        imul(sliver, kc);
        lea(sliver, [packed + sliver * 4]);
        imul(row, k);
        lea(row, [ak + row * 4]);
        pack_a(sliver, row, k, mine, kc);
        lock.inc(qword[counters + 64]);
        jmp(parts);
        label(packed_all);
        let total: gpr;
        mov(total, [shape + 24]);
        wait_for::<64>(counters, total);

        note("pieces of c, columns and rows, each taken by the first thread to come for it");
        label(pieces);
        let (piece, cj, nc, ri, hleft): (gpr, gpr, gpr, gpr, gpr);
        mov(piece, 1);
        lock.xadd([counters + 128], piece);
        cmp(piece, [shape + 32]);
        jae(joined);
        shl(piece, 5);
        mov(cj, [shape + piece + 48]);
        mov(nc, [shape + piece + 56]);
        mov(ri, [shape + piece + 64]);
        mov(hleft, [shape + piece + 72]);
        let (bj, to): (gpr, gpr);
        lea(bj, [bk + cj * 4]);
        lea(cj, [c + cj * 4]);
        lea(to, [block]);
        pack_b(to, bj, ld, nc, kc);

        note("slivers of MR rows of a, and of c");
        let (pa, ci): (gpr, gpr);
        mov(pa, ri);
        imul(pa, kc);
        lea(pa, [packed + pa * 4]);
        imul(ri, ld);
        lea(ci, [cj + ri]);
        label(rows);
        note("slivers of NR columns of the piece, a tile of c each");
        let (pb, ct, wleft, x): (gpr, gpr, gpr, gpr);
        lea(pb, [block]);
        mov(ct, ci);
        mov(wleft, nc);
        label(tile);
        mov(x, pa);
        micro(ct, ld, x, pb, kc, pc, hleft, wleft);
        add(ct, 4 * NR);
        sub(wleft, NR);
        ja(tile);
        mov(pa, x);
        lea(ci, [ci + ld * 4]);
        lea(ci, [ci + ld * 2]);
        sub(hleft, MR);
        ja(rows);
        lock.inc(qword[counters + 192]);
        jmp(pieces);

        label(joined);
        mov(total, [shape + 32]);
        wait_for::<192>(counters, total);
        add(ak, 4 * KC);
        let t: gpr;
        imul(t, ld, KC);
        add(bk, t);
        add(counters, 256);
        add(pc, KC);
        cmp(pc, k);
        jb(depth);
    }

    /// Waits for the counter at `counters + AT` to reach `total`.
    fn wait_for<const AT: usize>(counters: gpr, total: gpr) {
        label(again);
        cmp([counters + AT], total);
        jae(done);
        pause();
        jmp(again);
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
        test(mc, mc);
        jz(end);
        let (lda, q, p, left): (gpr, gpr, gpr, gpr);
        mov(lda, k);
        shl(lda, 2);
        mov(q, to);
        mov(p, from);
        mov(left, mc);
        label(sliver);
        cmp(left, MR);
        jb(edge);
        note("a whole sliver, a step at a time: rows 0 to 2 from s, and 3 to 5 from s3");
        let (s, s3, l): (gpr, gpr, gpr);
        mov(s, p);
        lea(s3, [p + lda * 2]);
        add(s3, lda);
        mov(l, kc);
        label(step);
        for r in 0..MR {
            let f: gpr;
            if r < 3 {
                mov(f.r32, [s + lda * r]);
            } else {
                mov(f.r32, [s3 + lda * (r - 3)]);
            }
            mov([q + 4 * r], f.r32);
        }
        add(s, 4);
        add(s3, 4);
        add(q, 4 * MR);
        dec(l);
        jnz(step);
        lea(p, [p + lda * 4]);
        lea(p, [p + lda * 2]);
        sub(left, MR);
        ja(sliver);
        jmp(end);

        note("the last sliver, short of MR rows: a row at a time, then zeros");
        label(edge);
        let (r, at): (gpr, gpr);
        zero(r);
        mov(at, q);
        label(row);
        let (from_row, d, n, g): (gpr, gpr, gpr, gpr);
        mov(d, at);
        mov(n, kc);
        cmp(r, left);
        jae(pad);
        mov(from_row, p);
        label(each);
        mov(g.r32, [from_row]);
        mov([d], g.r32);
        add(from_row, 4);
        add(d, 4 * MR);
        dec(n);
        jnz(each);
        jmp(next);
        label(pad);
        mov(dword[d], 0);
        add(d, 4 * MR);
        dec(n);
        jnz(pad);
        label(next);
        add(p, lda);
        add(at, 4);
        inc(r);
        cmp(r, MR);
        jb(row);
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
    /// first block of `k`. Of the tile, `h` rows and `w` columns lie in `c`. `x` and `y` end
    /// past their slivers.
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
        let l: gpr;
        mov(l, kc);
        sub(l, UNROLL);
        jb(rest);
        label(turn);
        for s in 0..UNROLL {
            step::<s>(acc, x, y);
        }
        add(x, 4 * MR * UNROLL);
        add(y, 4 * NR * UNROLL);
        sub(l, UNROLL);
        jae(turn);
        label(rest);
        note("the steps past the last whole turn");
        add(l, UNROLL);
        jz(store);
        label(one);
        step::<0>(acc, x, y);
        add(x, 4 * MR);
        add(y, 4 * NR);
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

/// Floats in a line of the CPU's cache, which is 64 bytes.
const LINE: usize = 16;

/// A line of the CPU's cache: 64 bytes, from a multiple of 64. The kernel's counters have one
/// each, and `packed` is a run of them.
#[derive(Clone)]
#[repr(C, align(64))]
struct Line {
    _floats: [f32; LINE],
}

/// Whether this CPU runs the kernels.
pub fn supported() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// Of the numbers of threads in `counts`, those the machine runs at once, in order, which a grid
/// may have. For each of the others, a line on standard error says that it is skipped.
pub fn runnable(counts: &[usize]) -> Result<Vec<usize>, kernite_cpu::Error> {
    let most = kernite_cpu::most()?;
    let (run, skipped): (Vec<usize>, Vec<usize>) = counts.iter().partition(|&&threads| threads <= most);
    for threads in skipped {
        eprintln!("{threads} threads: skipped, the machine runs {most} at once");
    }
    Ok(run)
}

/// `c = a b` by the kernel, on `threads` threads, of the operands of `shape`. It fails where the
/// grid cannot start: with no threads, or with more than the machine runs at once.
pub fn sgemm(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    shape: crate::Shape,
    threads: usize,
) -> Result<(), kernite_cpu::Error> {
    let (m, n, k) = shape;
    assert!(supported(), "the kernels need AVX2 and FMA");
    crate::operands((a.len(), b.len(), c.len()), shape);
    if m * n == 0 {
        return Ok(());
    }
    use kernels::{KC, MR, NC, NR, PACK};
    // The pieces of `c` the threads take in turn, from big to small, so that they finish
    // together: strips of columns, wide at first and one sliver at the end; the narrower a
    // strip, the more pieces its rows are split in.
    let mut pieces = Vec::new();
    let mut at = 0;
    while at < n {
        let slivers = (n - at).div_ceil(NR).div_ceil(2 * threads.max(1)).clamp(1, NC / NR);
        let end = n.min(at + slivers * NR);
        let rows = m.div_ceil(MR).div_ceil((NC / NR).div_ceil(slivers)) * MR;
        for row in (0..m).step_by(rows) {
            pieces.extend([at, end - at, row, rows.min(m - row)]);
        }
        at = end;
    }
    let mut counters = vec![Line { _floats: [0.0; LINE] }; 4 * k.div_ceil(KC).max(1)];
    let mut work = vec![m, n, k, m.div_ceil(PACK), pieces.len() / 4, counters.as_mut_ptr() as usize];
    work.extend(pieces);
    let mut packed = Vec::<Line>::with_capacity((MR * k.min(KC) * m.div_ceil(MR)).div_ceil(LINE));
    // SAFETY: `m` rows of `k` floats behind `a`, `k` rows of `n` behind `b` and `m` rows of `n`
    // behind `c`, as `shape` says; room for `a`'s slivers in `packed`, from a multiple of 64
    // bytes; the pieces lie inside `c` and share no element; four counters, zero, a line each,
    // for each block of `k`; and the CPU has what the kernel needs.
    unsafe {
        kernels::sgemm(
            kernite_cpu::Grid::new(threads),
            c.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            work.as_ptr(),
            packed.as_mut_ptr().cast(),
        )
    }
}
