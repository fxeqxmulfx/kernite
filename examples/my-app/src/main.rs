//! An example user of kernite: kernels written where they are called, in
//! `#[kernite_macros::kernels]`, and safe functions over them. The tests hold each kernel to a
//! scalar reference that does the same float operations in the same order, bit for bit, or the
//! tensor cores, which add in an order of their own, to products of integers, exact in any order;
//! and its listing to the one in `snapshots/`.

/// The kernels for the cpu, and the memory of the module's, `COUNTS` and `WEIGHTS`, which they
/// keep from call to call. Each reads what its pointers point at, and all but `gcds`, `gcd` and
/// `histogram`, which take integers, need AVX2 and FMA.
#[kernite_macros::kernels]
mod kernels {
    /// The sum of the eight floats at `xs`.
    pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
        let v: ymm;
        vmovups(v, [xs]);
        ret_sum(v);
    }

    /// The sum of the sixteen floats at `xs`: `sum8` of each half, added. A call may change
    /// every vector, so the first half's sum waits out the second call in the kernel's stack
    /// memory, as the listing says; and `xs` in rbx, which a call leaves as it was, and which
    /// the kernel saves for its own caller.
    pub unsafe extern "sysv64" fn sum16(xs: *const f32) -> f32 {
        let (low, high): (ymm, ymm);
        call(low.xmm, sum8, xs);
        add(xs, 32);
        call(high.xmm, sum8, xs);
        vaddss(high.xmm, low.xmm, high.xmm);
        ret(high.xmm);
    }

    /// The dot product of the `n` floats at `x` and the `n` at `y`, `n` a multiple of
    /// `8 UNROLL`, compiled for one accumulator, for four and for sixteen. `UNROLL`
    /// accumulators of eight lanes take turns, so that a fused multiply-add does not wait for
    /// the one before it; then they are added in order, and the lanes summed as in `sum8`.
    /// Each holds a register through the loop and the load needs one more, so sixteen take 17
    /// of the 16 there are: two spill to the stack, loaded before their multiply-adds and
    /// stored after, as the listing says.
    #[instances(1, 4, 16)]
    pub unsafe extern "sysv64" fn dot<const UNROLL: usize>(x: *const f32, y: *const f32, n: usize) -> f32 {
        test(n, n);
        jz(empty);
        let acc: [ymm; UNROLL];
        for k in 0..UNROLL {
            zero(acc[k]);
        }
        label(top);
        for k in 0..UNROLL {
            let xs: ymm;
            vmovups(xs, [x + 32 * k]);
            vfmadd231ps(acc[k], xs, [y + 32 * k]);
        }
        add(x, 32 * UNROLL);
        add(y, 32 * UNROLL);
        sub(n, 8 * UNROLL);
        jnz(top);

        note("the accumulators, in order");
        for k in 1..UNROLL {
            vaddps(acc[0], acc[0], acc[k]);
        }
        ret_sum(acc[0]);

        label(empty);
        let zero: ymm;
        zero(zero);
        ret(zero.xmm);
    }

    /// Adds to `counts[b]` how many of the `n` bytes at `bytes` are `b`, for each of the 256
    /// values, wrapping past `u32::MAX`. `TABLES` tables of 256 counts in the kernel's stack
    /// memory take turns, a byte each, so that a run of one value does not wait on its own last
    /// count; the bytes past a whole turn go to the first. Stack memory holds whatever was
    /// there, so the tables are cleared first, and added to `counts` last. Compiled for 4
    /// tables, 4 KB, and for 16, whose 16 KB the prologue touches a page at a time in a loop.
    #[instances(4, 16)]
    pub unsafe extern "sysv64" fn count_bytes<const TABLES: usize>(bytes: *const u8, n: usize, counts: *mut u32) {
        let tables: local<[u32; 256 * TABLES], 32>;
        note("the tables cleared");
        let z: ymm;
        zero(z);
        let i: gpr;
        zero(i);
        label(clear);
        for k in 0..4 {
            vmovaps([tables + i + 32 * k], z);
        }
        add(i, 128);
        cmp(i, 1024 * TABLES);
        jnz(clear);

        note("a byte to each table in turn");
        cmp(n, TABLES);
        jb(tail);
        label(turn);
        for k in 0..TABLES {
            let b: gpr;
            movzx(b.r32, byte[bytes + k]);
            let count: gpr;
            mov(count.r32, [tables + b * 4 + 1024 * k]);
            inc(count);
            mov([tables + b * 4 + 1024 * k], count.r32);
        }
        add(bytes, TABLES);
        sub(n, TABLES);
        cmp(n, TABLES);
        jae(turn);

        label(tail);
        test(n, n);
        jz(fold);
        label(rest);
        let b: gpr;
        movzx(b.r32, byte[bytes]);
        let count: gpr;
        mov(count.r32, [tables + b * 4]);
        inc(count);
        mov([tables + b * 4], count.r32);
        inc(bytes);
        dec(n);
        jnz(rest);

        label(fold);
        note("the tables added to the counts");
        zero(i);
        label(sum);
        let s: ymm;
        vmovdqa(s, [tables + i]);
        for k in 1..TABLES {
            vpaddd(s, s, [tables + i + 1024 * k]);
        }
        vpaddd(s, s, [counts + i]);
        vmovdqu([counts + i], s);
        add(i, 32);
        cmp(i, 1024);
        jnz(sum);
    }

    /// `x[i] /= s` for each `i < n`, `s` the sum of `x`, in one launch on the threads of a grid,
    /// bit for bit as `pascal::normalize` computes it on a GPU. The floats are in chunks of
    /// 1024, as the GPU's are in blocks of 256 threads of four, and the threads take the chunks
    /// in turn, each its index's and every `grid.threads`th after. A chunk's sum is its
    /// block's: a lane of 32 vectors for each GPU thread, whose four floats, 256 apart, are
    /// added in order; then the vectors in the block's tree, the upper half added to the lower
    /// down to one, whose lanes are added as `sum8` adds them, which is the tree's order too.
    /// The last chunk, if it is partial, is summed in the kernel's stack memory, with zeros
    /// after its floats. The sums go to `sums`, a float a chunk; the grid waits; then each
    /// thread adds them in order, and divides its floats by that, rounded once.
    #[cooperative]
    pub unsafe extern "sysv64" fn normalize(x: *mut f32, sums: *mut f32, n: usize) {
        let tail: local<[f32; 1024], 32>;
        let (whole, rest, c, p, i): (gpr, gpr, gpr, gpr, gpr);
        mov(whole, n);
        shr(whole, 10);
        mov(rest, n);
        and(rest, 1023);

        note("each of the thread's chunks summed");
        mov(c, grid.thread);
        label(chunk);
        mov(p, c);
        shl(p, 12);
        add(p, x);
        cmp(c, whole);
        jb(sum);
        jnz(summed);
        test(rest, rest);
        jz(summed);

        note("the last chunk, partial, copied to the stack, zeros after it");
        let f: ymm;
        zero(f);
        zero(i);
        label(clear);
        for k in 0..4 {
            vmovaps([tail + i + 32 * k], f);
        }
        add(i, 128);
        cmp(i, 4096);
        jnz(clear);
        zero(i);
        label(copy);
        vmovss(f.xmm, [p + i * 4]);
        vmovss([tail + i * 4], f.xmm);
        inc(i);
        cmp(i, rest);
        jnz(copy);
        lea(p, [tail]);

        label(sum);
        note("a GPU thread's four floats to a lane, then the tree's first two halvings");
        let tree: [ymm; 8];
        for j in 0..8 {
            let v: [ymm; 4];
            for q in 0..4 {
                vmovups(v[q], [p + 32 * (j + 8 * q)]);
                for k in 1..4 {
                    vaddps(v[q], v[q], [p + 1024 * k + 32 * (j + 8 * q)]);
                }
            }
            vaddps(v[0], v[0], v[2]);
            vaddps(v[1], v[1], v[3]);
            vaddps(tree[j], v[0], v[1]);
        }
        note("the rest of the tree");
        for half in [4, 2, 1] {
            for j in 0..half {
                vaddps(tree[j], tree[j], tree[j + half]);
            }
        }
        let s: ymm;
        add_lanes(s, tree[0]);
        vmovss([sums + c * 4], s.xmm);
        add(c, grid.threads);
        jmp(chunk);

        label(summed);
        grid.sync();

        note("the chunks' sums, in order");
        let all: gpr;
        lea(all, [n + 1023]);
        shr(all, 10);
        zero(s);
        zero(c);
        label(next);
        cmp(c, all);
        jz(divide);
        vaddss(s.xmm, s.xmm, [sums + c * 4]);
        inc(c);
        jmp(next);

        label(divide);
        note("the thread's floats over the sum");
        let d: ymm;
        vbroadcastss(d, s.xmm);
        mov(c, grid.thread);
        label(over);
        mov(p, c);
        shl(p, 12);
        add(p, x);
        cmp(c, whole);
        jae(part);
        zero(i);
        label(each);
        for k in 0..4 {
            let v: ymm;
            vmovups(v, [p + i + 32 * k]);
            vdivps(v, v, d);
            vmovups([p + i + 32 * k], v);
        }
        add(i, 128);
        cmp(i, 4096);
        jnz(each);
        add(c, grid.threads);
        jmp(over);

        label(part);
        jnz(done);
        test(rest, rest);
        jz(done);
        zero(i);
        label(one);
        vmovss(f.xmm, [p + i * 4]);
        vdivss(f.xmm, f.xmm, s.xmm);
        vmovss([p + i * 4], f.xmm);
        inc(i);
        cmp(i, rest);
        jnz(one);
        label(done);
    }

    /// `steps` steps of heat spreading along the `n` floats of `x`, in one launch on the threads
    /// of a grid, bit for bit as `pascal::diffuse` computes them on a GPU: in each, each float
    /// becomes `x[i] + (x[i - 1] - x[i]) / 4 + (x[i + 1] - x[i]) / 4` of the floats before,
    /// each operation rounded once, an end's missing neighbour as warm as the end. The threads
    /// take chunks of 1024 floats in turn, as in `normalize`, eight floats at once where their
    /// neighbours are all there, else one. A step reads `x` and writes `y`, the next the other
    /// way, and the grid waits between them; after an odd number, each thread copies its chunks
    /// back to `x`. `y` is as long as `x`.
    #[cooperative]
    pub unsafe extern "sysv64" fn diffuse(x: *mut f32, y: *mut f32, n: usize, steps: usize) {
        let (from, to, c, i, end, j): (gpr, gpr, gpr, gpr, gpr, gpr);
        mov(from, x);
        mov(to, y);
        let quarter: ymm;
        vbroadcastss(quarter, [0.25]);
        test(steps, steps);
        jz(done);

        label(step);
        mov(c, grid.thread);
        label(chunk);
        mov(i, c);
        shl(i, 10);
        cmp(i, n);
        jae(stepped);
        lea(end, [i + 1024]);
        cmp(end, n);
        jbe(each);
        mov(end, n);
        label(each);
        cmp(i, end);
        jz(chunked);
        note("eight floats at once, if their neighbours are all there");
        test(i, i);
        jz(one);
        lea(j, [i + 9]);
        cmp(j, n);
        ja(one);
        lea(j, [i + 8]);
        cmp(j, end);
        ja(one);
        let (v, l, r): (ymm, ymm, ymm);
        vmovups(v, [from + i * 4]);
        vmovups(l, [from + i * 4 - 4]);
        vmovups(r, [from + i * 4 + 4]);
        vsubps(l, l, v);
        vmulps(l, l, quarter);
        vsubps(r, r, v);
        vmulps(r, r, quarter);
        vaddps(v, v, l);
        vaddps(v, v, r);
        vmovups([to + i * 4], v);
        add(i, 8);
        jmp(each);

        label(one);
        note("one float, an end's missing neighbour the end itself");
        vmovss(v.xmm, [from + i * 4]);
        vmovaps(l.xmm, v.xmm);
        vmovaps(r.xmm, v.xmm);
        test(i, i);
        jz(left);
        vmovss(l.xmm, [from + i * 4 - 4]);
        label(left);
        lea(j, [i + 1]);
        cmp(j, n);
        jz(right);
        vmovss(r.xmm, [from + i * 4 + 4]);
        label(right);
        vsubss(l.xmm, l.xmm, v.xmm);
        vmulss(l.xmm, l.xmm, quarter.xmm);
        vsubss(r.xmm, r.xmm, v.xmm);
        vmulss(r.xmm, r.xmm, quarter.xmm);
        vaddss(v.xmm, v.xmm, l.xmm);
        vaddss(v.xmm, v.xmm, r.xmm);
        vmovss([to + i * 4], v.xmm);
        inc(i);
        jmp(each);

        label(chunked);
        add(c, grid.threads);
        jmp(chunk);

        label(stepped);
        grid.sync();
        mov(j, from);
        mov(from, to);
        mov(to, j);
        dec(steps);
        jnz(step);

        note("after an odd number of steps, the thread's chunks back to `x`");
        cmp(from, x);
        jz(done);
        mov(c, grid.thread);
        label(back);
        mov(i, c);
        shl(i, 10);
        cmp(i, n);
        jae(done);
        lea(end, [i + 1024]);
        cmp(end, n);
        jbe(copy);
        mov(end, n);
        label(copy);
        vmovss(v.xmm, [from + i * 4]);
        vmovss([x + i * 4], v.xmm);
        inc(i);
        cmp(i, end);
        jnz(copy);
        add(c, grid.threads);
        jmp(back);
        label(done);
    }

    /// `g[i]`, the greatest common divisor of `a[i]` and `b[i]`, for each `i < n`, by `gcd`.
    /// What lives across the calls, the pointers, `n` and `i`, is in the registers a call
    /// leaves as they were, which the kernel saves for its own caller.
    pub unsafe extern "sysv64" fn gcds(a: *const u32, b: *const u32, g: *mut u32, n: usize) {
        test(n, n);
        jz(done);
        let i: gpr;
        zero(i);
        label(next);
        let (x, y, d): (gpr, gpr, gpr);
        mov(x.r32, [a + i * 4]);
        mov(y.r32, [b + i * 4]);
        call(d, gcd, (x, y));
        mov([g + i * 4], d.r32);
        inc(i);
        cmp(i, n);
        jnz(next);
        label(done);
    }

    /// The greatest common divisor of `a` and `b` by Stein's algorithm, which halves and
    /// subtracts where Euclid's divides, a call of itself a step. A step takes a bit off one of
    /// them at least, so the calls go no deeper than the two have bits, 128.
    pub unsafe extern "sysv64" fn gcd(a: u64, b: u64) -> u64 {
        let (t, g): (gpr, gpr);
        note("`a` the larger");
        cmp(a, b);
        jae(ordered);
        mov(t, a);
        mov(a, b);
        mov(b, t);
        label(ordered);
        test(b, b);
        jnz(step);
        ret(a);

        label(step);
        test(a, 1);
        jnz(a_odd);
        test(b, 1);
        jnz(halve_a);
        note("both even: twice the gcd of their halves");
        shr(a, 1);
        shr(b, 1);
        call(g, gcd, (a, b));
        add(g, g);
        ret(g);

        label(a_odd);
        test(b, 1);
        jz(halve_b);
        note("both odd: their difference, which is even, for `a`");
        sub(a, b);
        label(halve_a);
        note("`a` even and `b` odd: `a` halved");
        shr(a, 1);
        call(g, gcd, (a, b));
        ret(g);

        label(halve_b);
        note("`b` even and `a` odd: `b` halved");
        shr(b, 1);
        call(g, gcd, (a, b));
        ret(g);
    }

    /// How many of the integers `histogram` has seen fall in each of 16 bins, by their top four
    /// bits: memory of the module's, in the process's, kept from one call to the next.
    pub static COUNTS: global<[u32; 16]>;

    /// What `weigh` multiplies each float by, by its place in a run of four: constant memory,
    /// which the kernel only reads, and the host writes.
    pub static WEIGHTS: constant<[f32; 4]> = [1.0, 0.5, 0.25, 2.0];

    /// Adds each of the `n` integers at `x` to its bin of `COUNTS`, the one its top four bits
    /// say, wrapping past `u32::MAX`, as `histogram` does on the GPU.
    pub unsafe extern "sysv64" fn histogram(x: *const u32, n: usize) {
        test(n, n);
        jz(done);
        let (bins, bin, count): (gpr, gpr, gpr);
        note("the bins' address, for an index");
        lea(bins, [COUNTS]);
        label(next);
        mov(bin.r32, [x]);
        shr(bin, 28);
        mov(count.r32, [bins + bin * 4]);
        inc(count);
        mov([bins + bin * 4], count.r32);
        add(x, 4);
        dec(n);
        jnz(next);
        label(done);
    }

    /// `x[i] *= WEIGHTS[i % 4]` for each `i < n`, rounded once, as `weigh` does on the GPU:
    /// eight floats at a time, by the four weights twice; then four, by them once; then each of
    /// the rest by its own.
    pub unsafe extern "sysv64" fn weigh(x: *mut f32, n: usize) {
        let (w, v): (ymm, ymm);
        vbroadcastf128(w, [WEIGHTS]);
        cmp(n, 8);
        jb(four);
        label(eight);
        vmulps(v, w, [x]);
        vmovups([x], v);
        add(x, 32);
        sub(n, 8);
        cmp(n, 8);
        jae(eight);

        label(four);
        cmp(n, 4);
        jb(rest);
        vmulps(v.xmm, w.xmm, [x]);
        vmovups([x], v.xmm);
        add(x, 16);
        sub(n, 4);

        label(rest);
        test(n, n);
        jz(done);
        let (weights, i): (gpr, gpr);
        note("the weights' address, for an index");
        lea(weights, [WEIGHTS]);
        zero(i);
        label(one);
        vmovss(v.xmm, [x + i * 4]);
        vmulss(v.xmm, v.xmm, [weights + i * 4]);
        vmovss([x + i * 4], v.xmm);
        inc(i);
        cmp(i, n);
        jnz(one);
        label(done);
    }

    /// Returns the sum of `v`'s lanes, as `add_lanes` adds them.
    fn ret_sum(v: ymm) {
        let sum: ymm;
        add_lanes(sum, v);
        ret(sum.xmm);
    }

    /// The sum of `v`'s lanes, in `sum`'s first: the high half added to the low, then the high
    /// pair of that to the low pair, then the second float to the first.
    fn add_lanes(sum: ymm, v: ymm) {
        note("the lanes");
        let high: ymm;
        vextractf128(high.xmm, v, 1);
        vaddps(sum.xmm, v.xmm, high.xmm);
        let pair: ymm;
        vmovhlps(pair.xmm, sum.xmm, sum.xmm);
        vaddps(sum.xmm, sum.xmm, pair.xmm);
        vmovshdup(pair.xmm, sum.xmm);
        vaddss(sum.xmm, sum.xmm, pair.xmm);
    }
}

/// The GPU kernels, for NVIDIA GPUs: a thread each element.
#[kernite_macros::kernels]
mod gpu {
    /// `y[i] = a * x[i] + y[i]` for each `i < n`, rounded once.
    pub unsafe extern "ptx-kernel" fn saxpy(a: f32, x: *const f32, y: *mut f32, n: u32) {
        let i: u32;
        global_index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let (at_x, at_y): (u64, u64);
        mad.wide.u32(at_x, i, 4, x);
        mad.wide.u32(at_y, i, 4, y);
        let (xi, yi): (f32, f32);
        ld.global.f32(xi, [at_x]);
        ld.global.f32(yi, [at_y]);
        fma.rn.f32(yi, a, xi, yi);
        st.global.f32([at_y], yi);
    }

    /// `sums[b]`, the sum of the `N` floats of `x` from `N b`: a block of `N` threads each,
    /// adding in a tree in the shared memory it declares, half as many threads each level.
    /// `N` is a power of two, compiled for 256 and for 1024, as many threads as a block has.
    #[instances(256, 1024)]
    pub unsafe extern "ptx-kernel" fn block_sums<const N: u32>(x: *const f32, sums: *mut f32) {
        let tile: shared<[f32; N], 16>;
        let (t, b, i, mine): (u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(b, ctaid.x);
        mad.lo.u32(i, b, N, t);
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        let (v, w): (f32, f32);
        ld.global.f32(v, [at]);
        mov.u32(mine, tile);
        mad.lo.u32(mine, t, 4, mine);
        st.shared.f32([mine], v);
        let lower: pred;
        for half in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1] {
            if half < N {
                bar.sync(0);
                setp.lt.u32(lower, t, half);
                if lower {
                    ld.shared.f32(w, [mine + 4 * half]);
                    add.rn.f32(v, v, w);
                    st.shared.f32([mine], v);
                }
            }
        }
        setp.eq.u32(lower, t, 0);
        if lower {
            mad.wide.u32(at, b, 4, sums);
            st.global.f32([at], v);
        }
    }

    /// Each block's run of floats of `x`, a thread each, reversed in place through the shared
    /// memory the launch gives: 4 bytes a thread.
    pub unsafe extern "ptx-kernel" fn reverse_runs(x: *mut f32) {
        let run: shared<[f32]>;
        let (t, n, i, j, base): (u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        global_index(i);
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        let v: f32;
        ld.global.f32(v, [at]);
        mov.u32(base, run);
        mad.lo.u32(j, t, 4, base);
        st.shared.f32([j], v);
        bar.sync(0);
        sub.u32(j, n, t);
        sub.u32(j, j, 1);
        mad.lo.u32(j, j, 4, base);
        ld.shared.f32(v, [j]);
        st.global.f32([at], v);
    }

    /// `x[i] *= a` for each `i < n`, in `f16`, rounded once.
    pub unsafe extern "ptx-kernel" fn scale_f16(x: *mut f16, a: f16, n: u32) {
        let i: u32;
        global_index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let at: u64;
        mad.wide.u32(at, i, 2, x);
        let v: f16;
        ld.global.b16(v, [at]);
        mul.rn.f16(v, v, a);
        st.global.b16([at], v);
    }

    /// `g[i]`, the greatest common divisor of `a[i]` and `b[i]`, for each `i < n`, by `gcd`.
    pub unsafe extern "ptx-kernel" fn gcds(a: *const u32, b: *const u32, g: *mut u32, n: u32) {
        let i: u32;
        global_index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let (at_a, at_b, at_g): (u64, u64, u64);
        mad.wide.u32(at_a, i, 4, a);
        mad.wide.u32(at_b, i, 4, b);
        mad.wide.u32(at_g, i, 4, g);
        let (x, y, d): (u32, u32, u32);
        ld.global.u32(x, [at_a]);
        ld.global.u32(y, [at_b]);
        call(d, gcd, (x, y));
        st.global.u32([at_g], d);
    }

    /// The greatest common divisor of `a` and `b` by Euclid's algorithm, a call of itself a
    /// step: a `.func`, which the GPU calls, not a helper, inlined, since nothing could inline
    /// a call that recurses as deep as the values it is given.
    extern "ptx-func" fn gcd(a: u32, b: u32) -> u32 {
        let done: pred;
        setp.eq.u32(done, b, 0);
        if done {
            ret(a);
        }
        let (r, g): (u32, u32);
        rem.u32(r, a, b);
        call(g, gcd, (b, r));
        ret(g);
    }

    /// How many of the integers `histogram` has seen fall in each of 16 bins, by their top four
    /// bits: memory of the module's, kept on the GPU from one launch to the next.
    pub static COUNTS: global<[u32; 16]>;

    /// What `weigh` multiplies each float by, by its place in a run of four: in constant
    /// memory, which every thread reads and the host writes.
    pub static WEIGHTS: constant<[f32; 4]> = [1.0, 0.5, 0.25, 2.0];

    /// Adds each `x[i]`, `i < n`, to its bin of `COUNTS`, the one its top four bits say.
    pub unsafe extern "ptx-kernel" fn histogram(x: *const u32, n: u32) {
        let i: u32;
        global_index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        let bin: u32;
        ld.global.u32(bin, [at]);
        shr.u32(bin, bin, 28);
        mov.u64(at, COUNTS);
        mad.wide.u32(at, bin, 4, at);
        red.global.add.u32([at], 1);
    }

    /// `x[i] *= WEIGHTS[i % 4]` for each `i < n`, rounded once.
    pub unsafe extern "ptx-kernel" fn weigh(x: *mut f32, n: u32) {
        let i: u32;
        global_index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        let (v, w): (f32, f32);
        ld.global.f32(v, [at]);
        let weight: u32;
        and.b32(weight, i, 3);
        let table: u32;
        mov.u32(table, WEIGHTS);
        mad.lo.u32(weight, weight, 4, table);
        ld.r#const.f32(w, [weight]);
        mul.rn.f32(v, v, w);
        st.global.f32([at], v);
    }

    /// The index of this thread among all of the grid's.
    fn global_index(i: u32) {
        let (block, size, thread): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        mad.lo.u32(i, block, size, thread);
    }
}

/// The kernels for the tensor cores of Ampere's GPUs and newer, `sm_80`, in a module of their
/// own: a module's `.target` is the newest any of its kernels needs, and `gpu`'s load on
/// older GPUs.
#[kernite_macros::kernels]
mod ampere {
    /// `c = a b` in `f32`, of `a`, rows of `k` halves, and `b`, `k` rows of `n`, given as its
    /// transpose `bt`, `n` rows of `k`: a warp a block, for 16 rows by 8 columns of `c`. It
    /// takes 16 of `k` at a time: `cp.async` copies them of the tile's rows of `a` and of `bt`
    /// from global memory to shared, with no register between; `ldmatrix` loads each thread's
    /// fragments of them; and `mma` multiplies them, adding to the tile in `f32`. `n` must be
    /// a multiple of 8, and `k` a positive multiple of 16.
    pub unsafe extern "ptx-kernel" fn matmul(a: *const f16, bt: *const f16, c: *mut f32, n: u32, k: u32) {
        let a_tile: shared<[f16; 256], 16>;
        let b_tile: shared<[f16; 128], 16>;
        let (lane, tiles, row, col, i, j): (u32, u32, u32, u32, u32, u32);
        mov.u32(lane, tid.x);
        note("the tile's first row and column");
        mov.u32(i, ctaid.x);
        shr.u32(tiles, n, 3);
        div.u32(row, i, tiles);
        rem.u32(col, i, tiles);
        shl.b32(row, row, 4);
        shl.b32(col, col, 3);

        note("what each thread copies: half a row, 8 halves, of `a`'s, and of `bt`'s if one of 16");
        let (half, at_a, at_b, to_a, to_b): (u32, u64, u64, u32, u32);
        and.b32(half, lane, 1);
        shl.b32(half, half, 3);
        shr.u32(i, lane, 1);
        add.u32(j, i, row);
        mad.lo.u32(j, j, k, half);
        mad.wide.u32(at_a, j, 2, a);
        add.u32(j, i, col);
        mad.lo.u32(j, j, k, half);
        mad.wide.u32(at_b, j, 2, bt);
        mov.u32(to_a, a_tile);
        mad.lo.u32(to_a, lane, 16, to_a);
        mov.u32(to_b, b_tile);
        mad.lo.u32(to_b, lane, 16, to_b);
        let copies_b: pred;
        setp.lt.u32(copies_b, lane, 16);

        note("the row each thread gives `ldmatrix`: of the four 8 by 8 of `a`'s, and the two of `bt`'s");
        let (from_a, from_b): (u32, u32);
        and.b32(i, lane, 15);
        shr.u32(j, lane, 4);
        mov.u32(from_a, a_tile);
        mad.lo.u32(from_a, i, 32, from_a);
        mad.lo.u32(from_a, j, 16, from_a);
        and.b32(i, lane, 7);
        shr.u32(j, lane, 3);
        and.b32(j, j, 1);
        mov.u32(from_b, b_tile);
        mad.lo.u32(from_b, i, 32, from_b);
        mad.lo.u32(from_b, j, 16, from_b);

        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        mov.f32(d0, 0);
        mov.f32(d1, 0);
        mov.f32(d2, 0);
        mov.f32(d3, 0);
        let (a0, a1, a2, a3, b0, b1): (b32, b32, b32, b32, b32, b32);
        let more: pred;
        label(step);
        cp.r#async.cg.shared.global([to_a], [at_a], 16);
        if copies_b {
            cp.r#async.cg.shared.global([to_b], [at_b], 16);
        }
        cp.r#async.wait_all();
        bar.sync(0);
        ldmatrix.sync.aligned.m8n8.x4.shared.b16((a0, a1, a2, a3), [from_a]);
        ldmatrix.sync.aligned.m8n8.x2.shared.b16((b0, b1), [from_b]);
        mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(
            (d0, d1, d2, d3),
            (a0, a1, a2, a3),
            (b0, b1),
            (d0, d1, d2, d3),
        );
        note("before the next copies overwrite what the other threads load");
        bar.sync(0);
        add.u64(at_a, at_a, 32);
        add.u64(at_b, at_b, 32);
        sub.u32(k, k, 16);
        setp.ne.u32(more, k, 0);
        if more {
            bra(step);
        }

        note("each thread's two of row `lane / 4` from column `2 (lane % 4)`, and the two 8 rows below");
        shr.u32(i, lane, 2);
        add.u32(i, i, row);
        and.b32(j, lane, 3);
        shl.b32(j, j, 1);
        add.u32(j, j, col);
        mad.lo.u32(i, i, n, j);
        let at_c: u64;
        mad.wide.u32(at_c, i, 4, c);
        st.global.v2.f32([at_c], (d0, d1));
        mad.lo.u32(i, n, 8, i);
        mad.wide.u32(at_c, i, 4, c);
        st.global.v2.f32([at_c], (d2, d3));
    }
}

/// The kernels whose blocks wait for each other, `#[cooperative]`, which a GPU from Pascal's
/// on, `sm_60`, launches all at once, in a module of their own: a module's `.target` is the
/// newest any of its kernels needs, and `gpu`'s load on older GPUs.
#[kernite_macros::kernels]
mod pascal {
    /// How many floats each thread of `normalize` keeps in its registers.
    pub const K: u32 = 4;

    /// `x[i] /= s` for each `i < n`, `s` the sum of `x`, in one launch of blocks of 256
    /// threads, where it would take two kernels if the grid could not wait: one for the sum,
    /// and the floats to memory and back before the other divides them. Each thread loads `K`
    /// floats, `x[256 (K b + k) + t]` for each `k < K`, 0 past `n`, and keeps them in its
    /// registers; their sum, in order, goes in a tree in the block's shared memory, as in
    /// `gpu::block_sums`, and the block's to `sums[b]`. The grid waits; then the block's first
    /// thread adds the blocks' sums, in order, and each thread divides its floats by that,
    /// rounded once.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn normalize(x: *mut f32, sums: *mut f32, n: u32) {
        let tile: shared<[f32; 256], 16>;
        let (t, b, i, j, mine): (u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(b, ctaid.x);
        mad.lo.u32(i, b, 256 * K, t);
        let (at, inside): (u64, pred);
        mad.wide.u32(at, i, 4, x);
        let v: [f32; K];
        for k in 0..K {
            mov.f32(v[k], 0);
            add.u32(j, i, 256 * k);
            setp.lt.u32(inside, j, n);
            if inside {
                ld.global.f32(v[k], [at + 1024 * k]);
            }
        }

        note("the thread's sum, then the block's, in a tree");
        let (s, w): (f32, f32);
        mov.f32(s, v[0]);
        for k in 1..K {
            add.rn.f32(s, s, v[k]);
        }
        mov.u32(mine, tile);
        mad.lo.u32(mine, t, 4, mine);
        st.shared.f32([mine], s);
        let lower: pred;
        for half in [128, 64, 32, 16, 8, 4, 2, 1] {
            bar.sync(0);
            setp.lt.u32(lower, t, half);
            if lower {
                ld.shared.f32(w, [mine + 4 * half]);
                add.rn.f32(s, s, w);
                st.shared.f32([mine], s);
            }
        }
        let (others, to): (pred, u64);
        setp.ne.u32(others, t, 0);
        if !others {
            mad.wide.u32(to, b, 4, sums);
            st.global.f32([to], s);
        }
        grid.sync();

        note("the first thread adds the blocks' sums, in order, for the others to read");
        if others {
            bra(added);
        }
        let (blocks, more): (u32, pred);
        mov.u32(blocks, nctaid.x);
        mov.f32(s, 0);
        mov.u64(to, sums);
        label(next);
        ld.global.f32(w, [to]);
        add.rn.f32(s, s, w);
        add.u64(to, to, 4);
        sub.u32(blocks, blocks, 1);
        setp.ne.u32(more, blocks, 0);
        if more {
            bra(next);
        }
        st.shared.f32([tile], s);
        label(added);
        bar.sync(0);
        ld.shared.f32(s, [tile]);
        for k in 0..K {
            add.u32(j, i, 256 * k);
            setp.lt.u32(inside, j, n);
            if inside {
                div.rn.f32(v[k], v[k], s);
                st.global.f32([at + 1024 * k], v[k]);
            }
        }
    }

    /// `steps` steps of heat spreading along the `n` floats of `x`, in one launch of blocks of
    /// 256 threads, where it would take a launch a step: in each, each float becomes
    /// `x[i] + (x[i - 1] - x[i]) / 4 + (x[i + 1] - x[i]) / 4` of the floats before, each
    /// operation rounded once, an end's missing neighbour as warm as the end. Each thread keeps
    /// its float in a register, and writes it for its neighbours to `x` and to `y` by turns,
    /// the one they are not reading; the grid waits between steps. `y` is as long as `x`.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn diffuse(x: *mut f32, y: *mut f32, n: u32, steps: u32) {
        let (i, j): (u32, u32);
        mov.u32(i, ctaid.x);
        mov.u32(j, tid.x);
        mad.lo.u32(i, i, 256, j);
        let (outside, end, more): (pred, pred, pred);
        setp.ge.u32(outside, i, n);
        let (from, to, at): (u64, u64, u64);
        mov.u64(from, x);
        mov.u64(to, y);
        let (v, l, r): (f32, f32, f32);
        mov.f32(v, 0);
        if !outside {
            mad.wide.u32(at, i, 4, x);
            ld.global.f32(v, [at]);
        }
        setp.eq.u32(end, steps, 0);
        if end {
            bra(done);
        }

        label(step);
        note("a thread past `n` has no float, and only waits");
        if outside {
            bra(wrote);
        }
        mad.wide.u32(at, i, 4, from);
        mov.f32(l, v);
        mov.f32(r, v);
        setp.ne.u32(end, i, 0);
        if end {
            ld.global.f32(l, [at - 4]);
        }
        add.u32(j, i, 1);
        setp.ne.u32(end, j, n);
        if end {
            ld.global.f32(r, [at + 4]);
        }
        sub.rn.f32(l, l, v);
        mul.rn.f32(l, l, 0.25);
        sub.rn.f32(r, r, v);
        mul.rn.f32(r, r, 0.25);
        add.rn.f32(v, v, l);
        add.rn.f32(v, v, r);
        mad.wide.u32(at, i, 4, to);
        st.global.f32([at], v);
        label(wrote);
        grid.sync();
        mov.u64(at, from);
        mov.u64(from, to);
        mov.u64(to, at);
        sub.u32(steps, steps, 1);
        setp.ne.u32(more, steps, 0);
        if more {
            bra(step);
        }

        label(done);
        if !outside {
            mad.wide.u32(at, i, 4, x);
            st.global.f32([at], v);
        }
    }
}

use std::sync::{Mutex, PoisonError};

use kernite_cuda::{Cuda, Grid, f16};

/// Whether this cpu runs the kernels.
fn supported() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// The sum of `xs`.
fn sum8(xs: &[f32; 8]) -> f32 {
    assert!(supported(), "the kernels need AVX2 and FMA");
    // SAFETY: eight floats are there to read, and the cpu has what the kernel needs.
    unsafe { kernels::sum8(xs.as_ptr()) }
}

/// The sum of `xs`, by `sum8` of each half.
fn sum16(xs: &[f32; 16]) -> f32 {
    assert!(supported(), "the kernels need AVX2 and FMA");
    // SAFETY: sixteen floats are there to read, and the cpu has what the kernel needs.
    unsafe { kernels::sum16(xs.as_ptr()) }
}

/// The dot product of `x` and `y`, which must be as long as each other and a multiple of
/// eight floats: by four accumulators when it is a multiple of 32, else by one.
fn dot(x: &[f32], y: &[f32]) -> f32 {
    assert!(supported(), "the kernels need AVX2 and FMA");
    assert_eq!(x.len(), y.len(), "x and y differ in length");
    assert!(x.len().is_multiple_of(8), "{} floats are not a multiple of eight", x.len());
    let kernel = if x.len().is_multiple_of(32) { kernels::dot::<4> } else { kernels::dot::<1> };
    // SAFETY: `x.len()` floats are there to read behind each pointer, a whole number of the
    // kernel's steps, and the cpu has what it needs.
    unsafe { kernel(x.as_ptr(), y.as_ptr(), x.len()) }
}

/// Adds to `counts[b]` how many of `bytes` are `b`, for each byte `b`, wrapping past
/// `u32::MAX`, by four tables in turn.
fn count_bytes(bytes: &[u8], counts: &mut [u32; 256]) {
    assert!(supported(), "the kernels need AVX2 and FMA");
    // SAFETY: `bytes.len()` bytes are there to read and 256 counts to write, and the cpu has
    // what the kernel needs.
    unsafe { kernels::count_bytes::<4>(bytes.as_ptr(), bytes.len(), counts.as_mut_ptr()) }
}

/// The greatest common divisor of each `a[i]` and `b[i]` on the cpu, by Stein's algorithm in
/// a function that calls itself; `a` and `b` must be as long as each other.
fn gcds_on_cpu(a: &[u32], b: &[u32]) -> Vec<u32> {
    assert_eq!(a.len(), b.len(), "a and b differ in length");
    let mut g = vec![0; a.len()];
    // SAFETY: `a.len()` integers behind each pointer, to read and to write, and the kernel
    // needs no more of the cpu than x86-64 has.
    unsafe { kernels::gcds(a.as_ptr(), b.as_ptr(), g.as_mut_ptr(), a.len()) };
    g
}

/// Keeps the module's memory, `kernels::COUNTS` and `kernels::WEIGHTS`, to one call of the
/// kernels that name it at a time, as their safety contract asks: each call holds it.
static MEMORY: Mutex<()> = Mutex::new(());

/// Adds each of `x` to its bin of `kernels::COUNTS`, by its top four bits, on the cpu, which
/// keeps the counts from one call to the next, as `histogram` does on the GPU: the host reads,
/// writes and zeroes them there.
fn histogram_on_cpu(x: &[u32]) {
    let _held = MEMORY.lock().unwrap_or_else(PoisonError::into_inner);
    // SAFETY: `x.len()` integers are there to read, no other call touches `COUNTS` while the
    // lock is held, and the kernel needs no more of the cpu than x86-64 has.
    unsafe { kernels::histogram(x.as_ptr(), x.len()) }
}

/// `x[i] *= w[i % 4]` on the cpu, each product rounded once, as `weigh` does on the GPU, of `w`
/// the weights in the module's constant memory, `kernels::WEIGHTS`: `[1, 0.5, 0.25, 2]` until
/// the host writes others.
fn weigh_on_cpu(x: &mut [f32]) {
    assert!(supported(), "the kernels need AVX2 and FMA");
    let _held = MEMORY.lock().unwrap_or_else(PoisonError::into_inner);
    // SAFETY: `x.len()` floats are there to read and write, no other call touches `WEIGHTS`
    // while the lock is held, and the cpu has what the kernel needs.
    unsafe { kernels::weigh(x.as_mut_ptr(), x.len()) }
}

/// `x / s` on `threads` threads of the cpu, `s` the sum of `x`, in one launch: each thread
/// sums its chunks of 1024 floats while the others sum theirs, then waits for them to add the
/// chunks' sums, bit for bit as `normalize` does on the GPU. `threads` is one at least, and at
/// most as many as the machine runs at once, `kernite_cpu::most()`; else it is an error.
fn normalize_on_threads(x: &mut [f32], threads: usize) -> Result<(), kernite_cpu::Error> {
    assert!(supported(), "the kernels need AVX2 and FMA");
    let mut sums = vec![0.0; x.len().div_ceil(1024)];
    // SAFETY: `x.len()` floats behind `x` and a float a chunk behind `sums`, which nothing
    // else touches while the threads run, and the cpu has what the kernel needs.
    unsafe { kernels::normalize(kernite_cpu::Grid::new(threads), x.as_mut_ptr(), sums.as_mut_ptr(), x.len()) }
}

/// `steps` steps of heat spreading along `x` on `threads` threads of the cpu, in one launch,
/// the threads waiting for each other between steps, bit for bit as `diffuse` does on the
/// GPU. `threads` is one at least, and at most as many as the machine runs at once,
/// `kernite_cpu::most()`; else it is an error.
fn diffuse_on_threads(x: &mut [f32], steps: usize, threads: usize) -> Result<(), kernite_cpu::Error> {
    assert!(supported(), "the kernels need AVX2 and FMA");
    let mut y = vec![0.0; x.len()];
    // SAFETY: `x.len()` floats behind each pointer, which nothing else touches while the
    // threads run, and the cpu has what the kernel needs.
    unsafe { kernels::diffuse(kernite_cpu::Grid::new(threads), x.as_mut_ptr(), y.as_mut_ptr(), x.len(), steps) }
}

/// `y = a * x + y` on the GPU, each element rounded once; `x` and `y` must be as long as each
/// other.
fn saxpy(cuda: &Cuda, kernels: &gpu::Kernels, a: f32, x: &[f32], y: &mut [f32]) -> Result<(), kernite_cuda::Error> {
    assert_eq!(x.len(), y.len(), "x and y differ in length");
    let n = u32::try_from(x.len()).expect("fewer than 2^32 floats");
    if n == 0 {
        return Ok(());
    }
    let (xs, ys) = (cuda.upload(x)?, cuda.upload(y)?);
    // SAFETY: `n` floats behind each pointer, in buffers that last until `read_into` has
    // waited for the kernel.
    unsafe { kernels.saxpy(Grid::new(n.div_ceil(256), 256), a, xs.ptr(), ys.ptr(), n)? };
    ys.read_into(y)
}

/// The sum of each run of `N` floats of `x`, on the GPU, added in a tree; `x` must be a whole
/// number of runs, and `N` a run `gpu::block_sums` is compiled for, 256 or 1024: another is
/// a compile error.
fn block_sums<const N: u32>(cuda: &Cuda, kernels: &gpu::Kernels, x: &[f32]) -> Result<Vec<f32>, kernite_cuda::Error> {
    let run = N as usize;
    assert!(x.len().is_multiple_of(run), "{} floats are no whole number of runs of {N}", x.len());
    let blocks = u32::try_from(x.len() / run).expect("fewer than 2^32 runs");
    let mut sums = vec![0.0; blocks as usize];
    if blocks == 0 {
        return Ok(sums);
    }
    let (xs, out) = (cuda.upload(x)?, cuda.alloc::<f32>(sums.len())?);
    // SAFETY: `N` floats a block behind `xs`, and a float a block behind `out`, in buffers
    // that last until `read_into` has waited for the kernel.
    unsafe { kernels.block_sums::<N>(Grid::new(blocks, N), xs.ptr(), out.ptr())? };
    out.read_into(&mut sums)?;
    Ok(sums)
}

/// Each run of `run` floats of `x` reversed, on the GPU; `x` must be a whole number of runs,
/// and a run no more threads than a block has.
fn reverse_runs(cuda: &Cuda, kernels: &gpu::Kernels, x: &mut [f32], run: usize) -> Result<(), kernite_cuda::Error> {
    assert!(run > 0 && x.len().is_multiple_of(run), "{} floats are no whole number of runs of {run}", x.len());
    let (blocks, threads) = (u32::try_from(x.len() / run).expect("fewer than 2^32 runs"), run as u32);
    if blocks == 0 {
        return Ok(());
    }
    let xs = cuda.upload(x)?;
    // SAFETY: `run` floats a block behind `xs`, and 4 bytes of shared memory a thread, in a
    // buffer that lasts until `read_into` has waited for the kernel.
    unsafe { kernels.reverse_runs(Grid::new(blocks, threads).with_shared(4 * threads), xs.ptr())? };
    xs.read_into(x)
}

/// `x *= a` on the GPU, each element in `f16`, rounded once.
fn scale_f16(cuda: &Cuda, kernels: &gpu::Kernels, a: f16, x: &mut [f16]) -> Result<(), kernite_cuda::Error> {
    let n = u32::try_from(x.len()).expect("fewer than 2^32 halves");
    if n == 0 {
        return Ok(());
    }
    let xs = cuda.upload(x)?;
    // SAFETY: `n` halves behind the pointer, in a buffer that lasts until `read_into` has
    // waited for the kernel.
    unsafe { kernels.scale_f16(Grid::new(n.div_ceil(256), 256), xs.ptr(), a, n)? };
    xs.read_into(x)
}

/// The greatest common divisor of each `a[i]` and `b[i]`, on the GPU, by a function that calls
/// itself; `a` and `b` must be as long as each other.
fn gcds(cuda: &Cuda, kernels: &gpu::Kernels, a: &[u32], b: &[u32]) -> Result<Vec<u32>, kernite_cuda::Error> {
    assert_eq!(a.len(), b.len(), "a and b differ in length");
    let n = u32::try_from(a.len()).expect("fewer than 2^32 pairs");
    let mut g = vec![0; a.len()];
    if n == 0 {
        return Ok(g);
    }
    let (xs, ys, out) = (cuda.upload(a)?, cuda.upload(b)?, cuda.alloc::<u32>(g.len())?);
    // SAFETY: `n` integers behind each pointer, in buffers that last until `read_into` has
    // waited for the kernel.
    unsafe { kernels.gcds(Grid::new(n.div_ceil(256), 256), xs.ptr(), ys.ptr(), out.ptr(), n)? };
    out.read_into(&mut g)?;
    Ok(g)
}

/// Adds each of `x` to its bin of `gpu::COUNTS`, by its top four bits, on the GPU, which keeps
/// the counts from one call to the next: `kernels.COUNTS()` reads them, writes them, and
/// zeroes them.
fn histogram(cuda: &Cuda, kernels: &gpu::Kernels, x: &[u32]) -> Result<(), kernite_cuda::Error> {
    let n = u32::try_from(x.len()).expect("fewer than 2^32 integers");
    if n == 0 {
        return Ok(());
    }
    let xs = cuda.upload(x)?;
    // SAFETY: `n` integers behind the pointer, in a buffer that lasts until `sync` has waited
    // for the kernel.
    unsafe { kernels.histogram(Grid::new(n.div_ceil(256), 256), xs.ptr(), n)? };
    cuda.sync()
}

/// `x[i] *= w[i % 4]` on the GPU, each product rounded once, of `w` the weights in its
/// constant memory, `gpu::WEIGHTS`: `[1, 0.5, 0.25, 2]` until `kernels.WEIGHTS()` writes others.
fn weigh(cuda: &Cuda, kernels: &gpu::Kernels, x: &mut [f32]) -> Result<(), kernite_cuda::Error> {
    let n = u32::try_from(x.len()).expect("fewer than 2^32 floats");
    if n == 0 {
        return Ok(());
    }
    let xs = cuda.upload(x)?;
    // SAFETY: `n` floats behind the pointer, in a buffer that lasts until `read_into` has
    // waited for the kernel.
    unsafe { kernels.weigh(Grid::new(n.div_ceil(256), 256), xs.ptr(), n)? };
    xs.read_into(x)
}

/// `a b` in `f32`, of `a`, `m` rows of `k` halves, and `b`, `k` rows of `n`, given as its
/// transpose `bt`, `n` rows of `k`, on the tensor cores; `m` must be a multiple of 16, `n` of
/// 8 and `k` of 16. The kernels load only on an Ampere GPU or newer, `sm_80`.
fn matmul(
    cuda: &Cuda,
    kernels: &ampere::Kernels,
    a: &[f16],
    bt: &[f16],
    (m, n, k): (usize, usize, usize),
) -> Result<Vec<f32>, kernite_cuda::Error> {
    let whole = m.is_multiple_of(16) && n.is_multiple_of(8) && k.is_multiple_of(16);
    assert!(whole, "{m} by {n} by {k} is no whole number of tiles of 16 by 8 by 16");
    assert_eq!(a.len(), m * k, "a is not {m} rows of {k}");
    assert_eq!(bt.len(), n * k, "bt is not {n} rows of {k}");
    let fits = |len: usize| u32::try_from(len).is_ok();
    assert!(fits(m * k) && fits(n * k) && fits(m * n), "{m} by {n} by {k} has 2^32 elements or more");
    let mut c = vec![0.0; m * n];
    if c.is_empty() || k == 0 {
        return Ok(c);
    }
    let (xs, ys, out) = (cuda.upload(a)?, cuda.upload(bt)?, cuda.alloc::<f32>(c.len())?);
    let tiles = (m / 16 * (n / 8)) as u32;
    // SAFETY: `m` rows of `k` halves behind `xs`, `n` behind `ys`, and `m` rows of `n` floats
    // behind `out`, in whole tiles, in buffers that last until `read_into` has waited for the
    // kernel.
    unsafe { kernels.matmul(Grid::new(tiles, 32), xs.ptr(), ys.ptr(), out.ptr(), n as u32, k as u32)? };
    out.read_into(&mut c)?;
    Ok(c)
}

/// `x / s` on the GPU, `s` the sum of `x`, in one launch: each thread keeps its floats in its
/// registers while the grid waits for every block's sum. `x` is at most as many floats as the
/// GPU holds at once, `256 pascal::K` a block of as many blocks as it runs at once,
/// `kernels.normalize_blocks(256, 0)`; more is an error.
fn normalize(cuda: &Cuda, kernels: &pascal::Kernels, x: &mut [f32]) -> Result<(), kernite_cuda::Error> {
    let n = u32::try_from(x.len()).expect("fewer than 2^32 floats");
    if n == 0 {
        return Ok(());
    }
    let blocks = n.div_ceil(256 * pascal::K);
    let (xs, sums) = (cuda.upload(x)?, cuda.alloc::<f32>(blocks as usize)?);
    // SAFETY: `n` floats behind `xs`, and a float a block behind `sums`, in buffers that last
    // until `read_into` has waited for the kernel.
    unsafe { kernels.normalize(Grid::new(blocks, 256), xs.ptr(), sums.ptr(), n)? };
    xs.read_into(x)
}

/// `steps` steps of heat spreading along `x` on the GPU, in one launch: in each, each float
/// becomes `x[i] + (x[i - 1] - x[i]) / 4 + (x[i + 1] - x[i]) / 4` of the floats before, an
/// end's missing neighbour as warm as the end. `x` is at most a float a thread of as many
/// blocks of 256 as the GPU runs at once, `kernels.diffuse_blocks(256, 0)`; more is an error.
fn diffuse(cuda: &Cuda, kernels: &pascal::Kernels, x: &mut [f32], steps: u32) -> Result<(), kernite_cuda::Error> {
    let n = u32::try_from(x.len()).expect("fewer than 2^32 floats");
    if n == 0 {
        return Ok(());
    }
    let (xs, ys) = (cuda.upload(x)?, cuda.alloc::<f32>(x.len())?);
    // SAFETY: `n` floats behind each pointer, in buffers that last until `read_into` has
    // waited for the kernel.
    unsafe { kernels.diffuse(Grid::new(n.div_ceil(256), 256), xs.ptr(), ys.ptr(), n, steps)? };
    xs.read_into(x)
}

fn main() {
    if !supported() {
        eprintln!("this cpu lacks AVX2 or FMA, which the kernels are written in");
        std::process::exit(1);
    }
    println!("sum8(1, ..., 8) = {}", sum8(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
    let ones: [f32; 16] = std::array::from_fn(|i| (i + 1) as f32);
    println!("sum16(1, ..., 16), which calls sum8 for each half = {}", sum16(&ones));
    for n in [1024, 1000] {
        let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32 - 3.0).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32 / 2.0).collect();
        let naive: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        println!("dot of {n} floats = {}; a loop in Rust says {naive}", dot(&x, &y));
    }
    let mut counts = [0; 256];
    count_bytes(b"hello, world", &mut counts);
    println!(
        "in \"hello, world\", counted in tables on the stack: {} l, {} o and {} h",
        counts[usize::from(b'l')],
        counts[usize::from(b'o')],
        counts[usize::from(b'h')]
    );
    let g = gcds_on_cpu(&[1071, 2_971_215_073], &[462, 1_836_311_903]);
    println!("by Stein's algorithm on the cpu: gcd(1071, 462) = {}, and of two Fibonacci numbers: {}", g[0], g[1]);
    let seen: Vec<Vec<u32>> = (0..2)
        .map(|call| {
            histogram_on_cpu(&[call << 28, 0x1000_0000, u32::MAX]);
            kernels::COUNTS.read()[..4].to_vec()
        })
        .collect();
    println!("4 of the 16 bins in the process's memory after a call of 3 integers, and after another: {seen:?}");
    let mut x = [1.0; 6];
    weigh_on_cpu(&mut x);
    println!("six ones weighed on the cpu by the weights in the module's constant memory: {x:?}");
    let cooperative = kernite_cpu::most().and_then(|most| {
        let threads = most.min(4);
        let mut x = [1.0, 2.0, 3.0, 4.0];
        normalize_on_threads(&mut x, threads)?;
        let mut heat = [0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0];
        diffuse_on_threads(&mut heat, 2, threads)?;
        Ok((threads, x, heat))
    });
    match cooperative {
        Ok((threads, x, heat)) => {
            println!("1 to 4 over their sum, on {threads} threads of the cpu that wait for each other: {x:?}");
            println!("16 among cold floats, after two steps of heat spreading on {threads} threads: {heat:?}");
        }
        Err(e) => println!("the cpu's cooperative kernels failed: {e}"),
    }
    let gpu = Cuda::new().and_then(|cuda| gpu::Kernels::load(&cuda).map(|kernels| (cuda, kernels)));
    match gpu {
        Ok((cuda, kernels)) => {
            let x: Vec<f32> = (0..1000).map(|i| i as f32 / 8.0).collect();
            let mut y = vec![1.0; 1000];
            match saxpy(&cuda, &kernels, 2.0, &x, &mut y) {
                Ok(()) => println!("saxpy on the {}: 2 * {} + 1 = {}", cuda.name(), x[999], y[999]),
                Err(e) => println!("saxpy failed on the {}: {e}", cuda.name()),
            }
            let ones: Vec<f32> = (0..1024).map(|i| i as f32).collect();
            match block_sums::<256>(&cuda, &kernels, &ones)
                .and_then(|q| Ok((q, block_sums::<1024>(&cuda, &kernels, &ones)?)))
            {
                Ok((quarters, whole)) => {
                    println!(
                        "sums of 0..1024 in runs of 256, in shared memory: {quarters:?}; in one of 1024: {whole:?}"
                    )
                }
                Err(e) => println!("block_sums failed: {e}"),
            }
            let mut run: Vec<f32> = (0..8).map(|i| i as f32).collect();
            match reverse_runs(&cuda, &kernels, &mut run, 4) {
                Ok(()) => println!("0..8 reversed in runs of 4, in shared memory the launch gives: {run:?}"),
                Err(e) => println!("reverse_runs failed: {e}"),
            }
            let mut halves = [f16::from_f32(0.1), f16::from_f32(1000.0)];
            match scale_f16(&cuda, &kernels, f16::from_f32(3.0), &mut halves) {
                Ok(()) => println!("[0.1, 1000] times 3 in f16: {halves:?}"),
                Err(e) => println!("scale_f16 failed: {e}"),
            }
            match gcds(&cuda, &kernels, &[1071, 2_971_215_073], &[462, 1_836_311_903]) {
                Ok(g) => println!("gcd(1071, 462) = {}, and of two Fibonacci numbers, 46 calls deep: {}", g[0], g[1]),
                Err(e) => println!("gcds failed: {e}"),
            }
            let seen = (0..2).try_fold(vec![], |mut seen, launch| {
                histogram(&cuda, &kernels, &[launch << 28, 0x1000_0000, u32::MAX])?;
                seen.push(kernels.COUNTS().read()?[..4].to_vec());
                Ok::<_, kernite_cuda::Error>(seen)
            });
            match seen {
                Ok(seen) => println!("4 of the 16 bins after a launch of 3 integers, and after another: {seen:?}"),
                Err(e) => println!("histogram failed: {e}"),
            }
            let mut x = [1.0; 6];
            match weigh(&cuda, &kernels, &mut x) {
                Ok(()) => println!("six ones weighed by the weights in constant memory: {x:?}"),
                Err(e) => println!("weigh failed: {e}"),
            }
            if cuda.sm() >= 80 {
                let a: Vec<f16> = (0..64 * 64).map(|i| f16::from_f32((i % 7) as f32 - 3.0)).collect();
                let bt: Vec<f16> = (0..64 * 64).map(|i| f16::from_f32((i % 5) as f32 - 2.0)).collect();
                let naive: f32 = (0..64).map(|l| a[l].to_f32() * bt[l].to_f32()).sum();
                let c = ampere::Kernels::load(&cuda).and_then(|kernels| matmul(&cuda, &kernels, &a, &bt, (64, 64, 64)));
                match c {
                    Ok(c) => {
                        println!("64 by 64 by 64 on the tensor cores: c[0][0] = {}; a loop in Rust says {naive}", c[0])
                    }
                    Err(e) => println!("matmul failed: {e}"),
                }
            } else {
                println!("the {} is older than Ampere, sm_80, for the tensor cores' matmul", cuda.name());
            }
            if cuda.sm() >= 60 {
                let cooperative = pascal::Kernels::load(&cuda).and_then(|kernels| {
                    let mut x = [1.0, 2.0, 3.0, 4.0];
                    normalize(&cuda, &kernels, &mut x)?;
                    let mut heat = [0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0];
                    diffuse(&cuda, &kernels, &mut heat, 2)?;
                    Ok((x, heat, kernels.normalize_blocks(256, 0)? * 256 * pascal::K))
                });
                match cooperative {
                    Ok((x, heat, most)) => {
                        println!("1 to 4 over their sum, in a launch whose grid waits for every block's: {x:?}");
                        println!("the {} keeps {most} floats in its registers so, at most", cuda.name());
                        println!("16 among cold floats, after two steps of heat spreading in one launch: {heat:?}");
                    }
                    Err(e) => println!("the cooperative kernels failed: {e}"),
                }
            } else {
                println!("the {} is older than Pascal, sm_60, for a kernel whose grid waits", cuda.name());
            }
        }
        Err(e) => println!("no GPU to run saxpy on: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use kernite_cuda::{Cuda, Grid, f16};

    use super::{ampere, gpu, kernels, pascal};

    /// Each kernel's listing against the one in `snapshots/`, and the GPU kernels' PTX, so that a
    /// change to kernite or to a kernel that moves an instruction shows in review. With
    /// `KERNITE_BLESS=1` the files are written instead, for the diff to be reviewed and committed.
    #[test]
    fn listings_match_their_snapshots() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("snapshots");
        let bless = std::env::var_os("KERNITE_BLESS").is_some();
        let mut changed = vec![];
        // An instance's listing, of `dot::<4>`, is `dot4.s`.
        let files =
            kernels::LISTINGS.iter().map(|&(name, text)| (format!("{}.s", name.replace(['<', '>', ':'], "")), text));
        let ptx = [
            ("gpu.ptx".to_string(), gpu::PTX),
            ("ampere.ptx".to_string(), ampere::PTX),
            ("pascal.ptx".to_string(), pascal::PTX),
        ];
        for (name, text) in files.chain(ptx) {
            let path = dir.join(&name);
            if bless {
                fs::create_dir_all(&dir).unwrap();
                fs::write(&path, text).unwrap();
            } else if fs::read_to_string(&path).ok().as_deref() != Some(text) {
                changed.push(name);
            }
        }
        assert!(changed.is_empty(), "listings changed: {changed:?}; rerun with KERNITE_BLESS=1 and review the diff");
    }

    /// Pseudo-random integers, the same for the same seed.
    fn words(n: usize, seed: u32) -> Vec<u32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                s
            })
            .collect()
    }

    /// Pseudo-random floats in [-1, 1), the same for the same seed.
    fn floats(n: usize, seed: u32) -> Vec<f32> {
        words(n, seed).iter().map(|s| (s >> 8) as f32 / (1 << 23) as f32 - 1.0).collect()
    }

    /// What `sum8` does to eight floats, in its order: the high half added to the low, then
    /// the high pair of that to the low pair, then the second float to the first.
    fn sum_lanes(v: [f32; 8]) -> f32 {
        let s: [f32; 4] = std::array::from_fn(|j| v[j] + v[j + 4]);
        (s[0] + s[2]) + (s[1] + s[3])
    }

    /// What `dot(unroll)` computes, in its order: `unroll` accumulators of eight lanes, each
    /// product added with one rounding, then the accumulators added in order, then the lanes
    /// as `sum8` adds them.
    fn dot(x: &[f32], y: &[f32], unroll: usize) -> f32 {
        let mut acc = vec![[0f32; 8]; unroll];
        for (xs, ys) in x.chunks(8 * unroll).zip(y.chunks(8 * unroll)) {
            for (k, lanes) in acc.iter_mut().enumerate() {
                for (j, lane) in lanes.iter_mut().enumerate() {
                    *lane = xs[8 * k + j].mul_add(ys[8 * k + j], *lane);
                }
            }
        }
        let mut v = acc[0];
        for other in &acc[1..] {
            for (lane, o) in v.iter_mut().zip(other) {
                *lane += o;
            }
        }
        sum_lanes(v)
    }

    #[test]
    fn sum8_is_its_reference_bit_for_bit() {
        assert!(super::supported(), "the kernels need AVX2 and FMA");
        for seed in 1..=1000 {
            let xs: [f32; 8] = floats(8, seed).try_into().unwrap();
            // SAFETY: eight floats, and a cpu that runs the kernels.
            let got = unsafe { kernels::sum8(xs.as_ptr()) };
            assert_eq!(got.to_bits(), sum_lanes(xs).to_bits(), "{xs:?}");
        }
    }

    /// For one accumulator and for four, and for sixteen, two of which spill, where `n` is a
    /// multiple of the 128 floats they take in a turn.
    /// `sum16` against `sum8`'s reference of each half, added: what it returns has been in the
    /// kernel's stack memory and back, across a call.
    #[test]
    fn sum16_is_sum8_of_each_half_bit_for_bit() {
        assert!(super::supported(), "the kernels need AVX2 and FMA");
        for seed in 1..=1000 {
            let xs: [f32; 16] = floats(16, seed).try_into().unwrap();
            let want = sum_lanes(xs[..8].try_into().unwrap()) + sum_lanes(xs[8..].try_into().unwrap());
            // SAFETY: sixteen floats, and a cpu that runs the kernels.
            let got = unsafe { kernels::sum16(xs.as_ptr()) };
            assert_eq!(got.to_bits(), want.to_bits(), "{xs:?}");
        }
    }

    /// The greatest common divisor of `a` and `b`, by Euclid's algorithm, as the GPU's `gcd`
    /// takes it.
    fn euclid(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { euclid(b, a % b) }
    }

    /// Pairs for `gcds`, and their greatest common divisors by `euclid`: zeros, pairs with a
    /// factor in common, and those of the most steps of Euclid's, consecutive Fibonacci
    /// numbers, up to the largest pair below 2^32, which takes the GPU's `gcd` 46 calls deep.
    fn gcd_pairs() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut fibonacci = vec![1u32, 2];
        while let Some(next) = fibonacci[fibonacci.len() - 2].checked_add(fibonacci[fibonacci.len() - 1]) {
            fibonacci.push(next);
        }
        let mut pairs: Vec<(u32, u32)> = fibonacci.windows(2).map(|w| (w[1], w[0])).collect();
        pairs.extend([(0, 0), (0, 7), (7, 0), (u32::MAX, u32::MAX), (u32::MAX, 1 << 31), (1071, 462)]);
        for w in words(30_000, 5).chunks(3) {
            let common = w[0] % 1000 + 1;
            pairs.push((w[1] % (u32::MAX / common) * common, w[2] % (u32::MAX / common) * common));
        }
        // A divisor of two `u32`s is a `u32`.
        let want = pairs.iter().map(|&(x, y)| euclid(x.into(), y.into()) as u32).collect();
        let (a, b) = pairs.into_iter().unzip();
        (a, b, want)
    }

    /// `gcds` on the cpu against Euclid's algorithm in Rust, on the GPU's pairs; and `gcd`
    /// alone on 64 bits, at the ends of the range and on pairs with a factor in common, a
    /// power of two in it, which Stein's algorithm halves away a call a bit.
    #[test]
    fn gcds_on_the_cpu_are_euclids() {
        let (a, b, want) = gcd_pairs();
        assert_eq!(super::gcds_on_cpu(&a, &b), want);
        let mut pairs =
            vec![(u64::MAX, u64::MAX), (u64::MAX, 1 << 63), (1 << 63, 1 << 62), (0, u64::MAX), (u64::MAX, 0)];
        for w in words(60_000, 9).chunks(6) {
            let common = u64::from(w[0] % 1000 + 1) << (w[1] % 24);
            let wide = |hi: u32, lo: u32| (u64::from(hi) << 32 | u64::from(lo)) % (u64::MAX / common) * common;
            pairs.push((wide(w[2], w[3]), wide(w[4], w[5])));
        }
        for (x, y) in pairs {
            // SAFETY: `gcd` reaches no memory but its own stack's.
            assert_eq!(unsafe { kernels::gcd(x, y) }, euclid(x, y), "gcd({x}, {y})");
        }
    }

    /// `histogram` on the cpu against the counts of each bin in Rust, as on the GPU: kept in the
    /// module's memory from one call to the next, and from what the host writes, wrapping past
    /// `u32::MAX`.
    #[test]
    fn histogram_on_the_cpu_keeps_its_counts_from_call_to_call() {
        let counts = &kernels::COUNTS;
        assert_eq!((counts.len(), counts.is_empty()), (16, false));
        assert_eq!(counts.read(), [0; 16], "zeros until something writes them");
        let mut want = [0u32; 16];
        for (n, seed) in [(1000, 1), (0, 2), (1, 3), (100_000, 4)] {
            let x = words(n, seed);
            for v in &x {
                want[(v >> 28) as usize] += 1;
            }
            super::histogram_on_cpu(&x);
            assert_eq!(counts.read(), want, "after {n} more integers");
        }
        counts.write(&[7; 16]);
        super::histogram_on_cpu(&[0, u32::MAX, u32::MAX]);
        let mut want = [7; 16];
        want[0] += 1;
        want[15] += 2;
        assert_eq!(counts.read(), want, "after the host's");
        counts.write(&[u32::MAX; 16]);
        super::histogram_on_cpu(&[0x1234_5678]);
        let mut read = [0; 16];
        counts.read_into(&mut read);
        assert_eq!((read[0], read[1]), (u32::MAX, 0), "past `u32::MAX`");
        counts.zero();
        assert_eq!(counts.read(), [0; 16]);
    }

    /// `weigh` on the cpu against each float times its weight in Rust, rounded once, as on the
    /// GPU: by the weights the module starts with, and then by those the host writes; eight at
    /// a time, four, and one.
    #[test]
    fn weigh_on_the_cpu_multiplies_by_the_weights_in_constant_memory() {
        let weights = &kernels::WEIGHTS;
        assert_eq!(weights.read(), [1.0, 0.5, 0.25, 2.0]);
        for w in [[1.0, 0.5, 0.25, 2.0], [3.0, -0.1, 1.0e-30, 7.5]] {
            weights.write(&w);
            for n in [0, 1, 3, 4, 7, 8, 12, 15, 257, 10_000] {
                let mut x = floats(n, n as u32 + 9);
                let want: Vec<u32> = x.iter().enumerate().map(|(i, x)| (x * w[i % 4]).to_bits()).collect();
                super::weigh_on_cpu(&mut x);
                assert_eq!(x.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want, "{n} floats by {w:?}");
            }
        }
    }

    #[test]
    fn dot_is_its_reference_bit_for_bit() {
        assert!(super::supported(), "the kernels need AVX2 and FMA");
        for n in [0, 32, 96, 128, 1024, 3968, 4000] {
            let (x, y) = (floats(n, 2 * n as u32 + 1), floats(n, 2 * n as u32 + 2));
            // SAFETY: `n` floats behind each pointer, a multiple of 32, and a cpu that runs
            // the kernels.
            let (one, four) =
                unsafe { (kernels::dot::<1>(x.as_ptr(), y.as_ptr(), n), kernels::dot::<4>(x.as_ptr(), y.as_ptr(), n)) };
            assert_eq!(one.to_bits(), dot(&x, &y, 1).to_bits(), "{n} floats, one accumulator");
            assert_eq!(four.to_bits(), dot(&x, &y, 4).to_bits(), "{n} floats, four accumulators");
            if n.is_multiple_of(128) {
                // SAFETY: as above, and a multiple of 128.
                let sixteen = unsafe { kernels::dot::<16>(x.as_ptr(), y.as_ptr(), n) };
                assert_eq!(sixteen.to_bits(), dot(&x, &y, 16).to_bits(), "{n} floats, sixteen accumulators");
            }
        }
    }

    /// Each count, from counts that start anywhere, some near `u32::MAX`, and called again and
    /// again on the same stack, so that tables left uncleared would show; at lengths short of a
    /// turn of the tables, past one, and a run of one byte.
    #[test]
    fn count_bytes_counts_each_byte() {
        assert!(super::supported(), "the kernels need AVX2 and FMA");
        let start: [u32; 256] = std::array::from_fn(|b| (b as u32).wrapping_mul(0x9e37_79b9));
        let mut inputs: Vec<Vec<u8>> = [0, 1, 3, 4, 5, 15, 16, 17, 1000, 4097, 100_000]
            .iter()
            .map(|&n| words(n, n as u32 + 7).iter().map(|w| (w >> 24) as u8).collect())
            .collect();
        inputs.push(vec![7; 1001]);
        for bytes in &inputs {
            let mut want = start;
            for &b in bytes {
                want[usize::from(b)] = want[usize::from(b)].wrapping_add(1);
            }
            let (mut four, mut sixteen) = (start, start);
            // SAFETY: `bytes.len()` bytes and 256 counts, and a cpu that runs the kernels.
            unsafe {
                kernels::count_bytes::<4>(bytes.as_ptr(), bytes.len(), four.as_mut_ptr());
                kernels::count_bytes::<16>(bytes.as_ptr(), bytes.len(), sixteen.as_mut_ptr());
            }
            assert_eq!(four, want, "{} bytes, four tables", bytes.len());
            assert_eq!(sixteen, want, "{} bytes, sixteen tables", bytes.len());
        }
    }

    /// A kernel for the tests alone.
    #[kernite_macros::kernels]
    mod deep {
        /// Clears the first and the last word of a megabyte of stack memory.
        pub unsafe extern "sysv64" fn deep() {
            let words: local<[u64; 131072]>;
            let zero: gpr;
            zero(zero);
            mov([words], zero);
            mov([words + 8 * 131071], zero);
        }
    }

    /// A cooperative kernel for the tests alone.
    #[kernite_macros::kernels]
    mod framed {
        /// Each thread writes `t + 1` to the first word of a frame of 192 KB, and `t + 2` to the
        /// last; waits for the others; and stores the sum of the two words it reads back in
        /// `out[t]`, `t` its index in the grid.
        #[cooperative]
        pub unsafe extern "sysv64" fn framed(out: *mut u64) {
            let words: local<[u64; 24576]>;
            let (t, v): (gpr, gpr);
            mov(t, grid.thread);
            lea(v, [t + 1]);
            mov([words], v);
            lea(v, [t + 2]);
            mov([words + 8 * 24575], v);
            grid.sync();
            mov(v, [words]);
            add(v, [words + 8 * 24575]);
            mov([out + t * 8], v);
        }
    }

    /// A frame past the end of a thread's stack faults at the guard page below the stack, which
    /// Rust reports as the overflow it is, since the prologue touches the frame a page at a
    /// time: a frame taken at once would reach past the guard page, into memory of another's
    /// or none. The test runs itself again in a process of its own, which the overflow aborts.
    #[test]
    fn a_frame_past_the_end_of_the_stack_faults_at_its_guard_page() {
        if std::env::var_os("MY_APP_DEEP").is_some() {
            let thread = std::thread::Builder::new().name("deep".into()).stack_size(64 << 10);
            // SAFETY: the kernel reaches no memory but its own stack's.
            thread.spawn(|| unsafe { deep::deep() }).unwrap().join().unwrap();
            return;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::a_frame_past_the_end_of_the_stack_faults_at_its_guard_page", "--nocapture"])
            .env("MY_APP_DEEP", "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        let overflowed = stderr.contains("thread 'deep'") && stderr.contains("has overflowed its stack");
        assert!(!out.status.success() && overflowed, "{stderr}");
    }

    /// The GPU, and its kernels; on a machine with none, `None`, and the test is skipped.
    fn cuda() -> Option<(Cuda, gpu::Kernels)> {
        let cuda = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")).ok()?;
        let kernels = gpu::Kernels::load(&cuda).unwrap();
        Some((cuda, kernels))
    }

    /// What `block_sums` does to each run, a power of two long, in its order: each float of
    /// the lower half added to the one as far into the upper, then the same of what that
    /// leaves.
    fn tree(run: &[f32]) -> f32 {
        let mut tile = run.to_vec();
        let mut half = tile.len() / 2;
        while half > 0 {
            for t in 0..half {
                tile[t] += tile[t + half];
            }
            half /= 2;
        }
        tile[0]
    }

    #[test]
    fn block_sums_is_its_tree_bit_for_bit() {
        let Some((cuda, kernels)) = cuda() else { return };
        for runs in [0, 1, 2, 37] {
            let x = floats(1024 * runs, runs as u32 + 7);
            let sums = |run: usize| x.chunks(run).map(|run| tree(run).to_bits()).collect::<Vec<_>>();
            let bits = |got: Vec<f32>| got.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(super::block_sums::<256>(&cuda, &kernels, &x).unwrap()), sums(256), "{runs} runs of 1024");
            assert_eq!(
                bits(super::block_sums::<1024>(&cuda, &kernels, &x).unwrap()),
                sums(1024),
                "{runs} runs of 1024"
            );
        }
    }

    #[test]
    fn reverse_runs_reverses_each_run() {
        let Some((cuda, kernels)) = cuda() else { return };
        // A run of 1024 is as many threads as a block has, and 4 KB of shared memory.
        for (n, run) in [(12, 3), (1024, 1), (4096, 1024), (1000, 40)] {
            let mut x: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let want: Vec<f32> = x.chunks(run).flat_map(|r| r.iter().rev().copied()).collect();
            super::reverse_runs(&cuda, &kernels, &mut x, run).unwrap();
            assert_eq!(x, want, "{n} floats in runs of {run}");
        }
    }

    /// `scale_f16` on every `f16` against the product in `f32`, which is exact, rounded to an
    /// `f16` once as `mul.rn.f16` rounds it.
    #[test]
    fn scale_f16_rounds_the_product_once() {
        let Some((cuda, kernels)) = cuda() else { return };
        for a in [3.0, -0.1, 1.0e-3, 600.0] {
            let a = f16::from_f32(a);
            let mut x: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
            let want: Vec<f16> = x.iter().map(|x| f16::from_f32(x.to_f32() * a.to_f32())).collect();
            super::scale_f16(&cuda, &kernels, a, &mut x).unwrap();
            for (bits, (got, want)) in (0..=u16::MAX).zip(x.iter().zip(&want)) {
                match want.to_f32().is_nan() {
                    true => assert!(got.to_f32().is_nan(), "{bits:#06x} times {a}"),
                    false => assert_eq!(got.to_bits(), want.to_bits(), "{bits:#06x} times {a}"),
                }
            }
        }
    }

    /// `gcds` on the GPU against Euclid's algorithm in Rust, on `gcd_pairs`.
    #[test]
    fn gcds_are_euclids() {
        let Some((cuda, kernels)) = cuda() else { return };
        let (a, b, want) = gcd_pairs();
        assert_eq!(super::gcds(&cuda, &kernels, &a, &b).unwrap(), want);
    }

    /// `histogram` against the counts of each bin in Rust, kept on the GPU from one launch to
    /// the next, and from what the host writes.
    #[test]
    fn histogram_keeps_its_counts_from_launch_to_launch() {
        let Some((cuda, kernels)) = cuda() else { return };
        let counts = kernels.COUNTS();
        assert_eq!(counts.len(), 16);
        assert_eq!(counts.read().unwrap(), [0; 16], "zeros until something writes them");
        let mut want = [0u32; 16];
        for (n, seed) in [(1000, 1), (0, 2), (1, 3), (100_000, 4)] {
            let x = words(n, seed);
            for v in &x {
                want[(v >> 28) as usize] += 1;
            }
            super::histogram(&cuda, &kernels, &x).unwrap();
            assert_eq!(counts.read().unwrap(), want, "after {n} more integers");
        }
        counts.write(&[7; 16]).unwrap();
        super::histogram(&cuda, &kernels, &[0, u32::MAX, u32::MAX]).unwrap();
        let mut want = [7; 16];
        want[0] += 1;
        want[15] += 2;
        assert_eq!(counts.read().unwrap(), want, "after the host's");
        counts.zero().unwrap();
        assert_eq!(counts.read().unwrap(), [0; 16]);
    }

    /// `weigh` against each float times its weight in Rust, which rounds once as `mul.rn.f32`
    /// does: by the weights the module starts with, and then by those the host writes.
    #[test]
    fn weigh_multiplies_by_the_weights_in_constant_memory() {
        let Some((cuda, kernels)) = cuda() else { return };
        let weights = kernels.WEIGHTS();
        assert_eq!(weights.read().unwrap(), [1.0, 0.5, 0.25, 2.0]);
        for w in [[1.0, 0.5, 0.25, 2.0], [3.0, -0.1, 1.0e-30, 7.5]] {
            weights.write(&w).unwrap();
            for n in [0, 1, 257, 10_000] {
                let mut x = floats(n, n as u32 + 9);
                let want: Vec<u32> = x.iter().enumerate().map(|(i, x)| (x * w[i % 4]).to_bits()).collect();
                super::weigh(&cuda, &kernels, &mut x).unwrap();
                assert_eq!(x.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want, "{n} floats by {w:?}");
            }
        }
    }

    /// Integers from -8 to 8, as halves, the same for the same seed.
    fn integers(n: usize, seed: u32) -> Vec<f16> {
        floats(n, seed).iter().map(|x| f16::from_f32((x * 8.0).round())).collect()
    }

    /// `matmul` against the product of integers, of which each sum is exact in an `f32` in any
    /// order the tensor cores add: of `k` products of at most 64, less than 2^24 in all.
    #[test]
    fn matmul_is_exact_on_integers() {
        let Some((cuda, _)) = cuda() else { return };
        if cuda.sm() < 80 {
            return eprintln!("the {} is older than Ampere, sm_80: skipped", cuda.name());
        }
        let kernels = ampere::Kernels::load(&cuda).unwrap();
        for (m, n, k) in [(16, 8, 16), (0, 8, 16), (16, 8, 0), (32, 24, 48), (48, 40, 160), (128, 128, 1024)] {
            let (a, bt) = (integers(m * k, (m + k) as u32 + 1), integers(n * k, (n + k) as u32 + 2));
            let want: Vec<f32> = (0..m * n)
                .map(|ij| {
                    let (i, j) = (ij / n, ij % n);
                    let sum: i32 = (0..k).map(|l| a[i * k + l].to_f32() as i32 * bt[j * k + l].to_f32() as i32).sum();
                    sum as f32
                })
                .collect();
            let got = super::matmul(&cuda, &kernels, &a, &bt, (m, n, k)).unwrap();
            assert_eq!(got, want, "{m} by {n} by {k}");
        }
    }

    /// The GPU, and its kernels whose blocks wait for each other; on a machine with none, or
    /// with one older than Pascal, `sm_60`, `None`, and the test is skipped.
    fn pascal() -> Option<(Cuda, pascal::Kernels)> {
        let (cuda, _) = cuda()?;
        if cuda.sm() < 60 {
            eprintln!("the {} is older than Pascal, sm_60: skipped", cuda.name());
            return None;
        }
        let kernels = pascal::Kernels::load(&cuda).unwrap();
        Some((cuda, kernels))
    }

    /// What `normalize` computes, in its order: each thread's `K` floats added in order, 0 past
    /// the end, then the sums of a block's threads in a tree, then the blocks' in order from 0,
    /// and each float divided by that.
    fn normalized(x: &[f32]) -> Vec<f32> {
        let k = pascal::K as usize;
        let sum = x
            .chunks(256 * k)
            .map(|block| {
                let at = |j: usize| block.get(j).copied().unwrap_or(0.0);
                let threads: Vec<f32> = (0..256).map(|t| (1..k).fold(at(t), |s, k| s + at(256 * k + t))).collect();
                tree(&threads)
            })
            .fold(0.0, |s, b| s + b);
        x.iter().map(|v| v / sum).collect()
    }

    /// The end of the error of a cooperative launch of `blocks` blocks, which is more than the
    /// GPU runs at once, `most`.
    fn too_large(e: kernite_cuda::Error, blocks: u32, most: u32) {
        assert!(e.message.starts_with("CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (720): "), "{e}");
        assert!(
            e.message.ends_with(&format!(": the grid has {blocks} blocks, and the GPU runs {most} of these at once")),
            "{e}"
        );
    }

    /// `normalize` against [`normalized`], bit for bit, each time of ten: of a float, of a block
    /// and a float more, and of as many as the GPU holds at once, so that a block that went on
    /// before the others had written their sums would show; and one more is an error.
    #[test]
    fn normalize_is_its_reference_bit_for_bit() {
        let Some((cuda, kernels)) = pascal() else { return };
        let most = kernels.normalize_blocks(256, 0).unwrap();
        let all = (most * 256 * pascal::K) as usize;
        for n in [1, 1023, 1025, 10_000, all] {
            // In [0.5, 1.5), so that the sum is far from 0.
            let x: Vec<f32> = floats(n, n as u32 + 11).iter().map(|x| x / 2.0 + 1.0).collect();
            let want: Vec<u32> = normalized(&x).iter().map(|v| v.to_bits()).collect();
            for _ in 0..10 {
                let mut got = x.clone();
                super::normalize(&cuda, &kernels, &mut got).unwrap();
                assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want, "{n} floats");
            }
        }
        too_large(super::normalize(&cuda, &kernels, &mut vec![1.0; all + 1]).unwrap_err(), most + 1, most);
    }

    /// What `diffuse` computes, a step at a time, in its order.
    fn diffused(x: &[f32], steps: u32) -> Vec<f32> {
        let mut x = x.to_vec();
        for _ in 0..steps {
            x = (0..x.len())
                .map(|i| {
                    let v = x[i];
                    let (l, r) = (if i > 0 { x[i - 1] } else { v }, x.get(i + 1).copied().unwrap_or(v));
                    v + (l - v) * 0.25 + (r - v) * 0.25
                })
                .collect();
        }
        x
    }

    /// `diffuse` against [`diffused`], bit for bit: of a float, of a block and a float more,
    /// and of as many as the GPU holds at once over a thousand steps, each a wait of the grid,
    /// so that a thread that read a neighbour of another step would show; and one more is an
    /// error.
    #[test]
    fn diffuse_is_its_reference_bit_for_bit() {
        let Some((cuda, kernels)) = pascal() else { return };
        let most = kernels.diffuse_blocks(256, 0).unwrap();
        let all = (most * 256) as usize;
        for (n, steps) in [(1, 3), (2, 1), (257, 2), (1000, 0), (all, 1000)] {
            let x = floats(n, n as u32 + 13);
            let want: Vec<u32> = diffused(&x, steps).iter().map(|v| v.to_bits()).collect();
            let mut got = x.clone();
            super::diffuse(&cuda, &kernels, &mut got, steps).unwrap();
            assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want, "{n} floats, {steps} steps");
        }
        too_large(super::diffuse(&cuda, &kernels, &mut vec![0.0; all + 1], 1).unwrap_err(), most + 1, most);
    }

    /// How many threads the cpu tests run on at most: four, or as many as the machine runs at
    /// once if fewer, since `cargo test` runs tests beside each other and each thread spins at
    /// a barrier.
    fn threads() -> usize {
        kernite_cpu::most().unwrap().min(4)
    }

    /// The error of a grid of more threads than the machine runs at once.
    fn too_many(e: kernite_cpu::Error) {
        let most = kernite_cpu::most().unwrap();
        assert_eq!(e.message, format!("the grid has {} threads, and the machine runs {most} at once", most + 1));
    }

    /// The cpu's `normalize` against [`normalized`], the GPU's reference, bit for bit, on one
    /// thread to four, each time of ten: of no float and of one, of a chunk and a float less and
    /// more, and of chunks enough that each thread has several, the last partial, so that a
    /// thread that went on before the others had written their sums would show; and a grid of
    /// more threads than the machine runs is an error.
    #[test]
    fn normalize_on_threads_is_the_gpus_reference_bit_for_bit() {
        assert!(super::supported(), "the kernels need AVX2 and FMA");
        for threads in 1..=threads() {
            for n in [0, 1, 1023, 1024, 1025, 10_000, 41_000] {
                // In [0.5, 1.5), so that the sum is far from 0.
                let x: Vec<f32> = floats(n, n as u32 + 11).iter().map(|x| x / 2.0 + 1.0).collect();
                let want: Vec<u32> = normalized(&x).iter().map(|v| v.to_bits()).collect();
                for _ in 0..10 {
                    let mut got = x.clone();
                    super::normalize_on_threads(&mut got, threads).unwrap();
                    assert_eq!(
                        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        want,
                        "{n} floats, {threads} threads"
                    );
                }
            }
        }
        too_many(super::normalize_on_threads(&mut [1.0], kernite_cpu::most().unwrap() + 1).unwrap_err());
    }

    /// The cpu's `diffuse` against [`diffused`], the GPU's reference, bit for bit, on one thread
    /// to four: of floats fewer than a vector and its neighbours, of a chunk and one more, and
    /// of chunks enough that each thread has several, over a thousand steps, each a wait of the
    /// grid, so that a thread that read a neighbour of another step would show; and a grid of
    /// more threads than the machine runs is an error.
    #[test]
    fn diffuse_on_threads_is_the_gpus_reference_bit_for_bit() {
        assert!(super::supported(), "the kernels need AVX2 and FMA");
        for threads in 1..=threads() {
            for (n, steps) in [(1, 3), (2, 1), (9, 2), (10, 3), (17, 4), (1000, 0), (1025, 1), (9000, 1000)] {
                let x = floats(n, n as u32 + 13);
                let want: Vec<u32> = diffused(&x, steps).iter().map(|v| v.to_bits()).collect();
                let mut got = x.clone();
                super::diffuse_on_threads(&mut got, steps as usize, threads).unwrap();
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want,
                    "{n} floats, {steps} steps, {threads} threads"
                );
            }
        }
        too_many(super::diffuse_on_threads(&mut [1.0], 1, kernite_cpu::most().unwrap() + 1).unwrap_err());
    }

    /// A frame of 192 KB is on the stack of the thread the kernel runs on, each of the grid's,
    /// whose own it is, launch after launch on the threads the launches before kept: each reads
    /// back what it wrote, though all wrote before any read.
    #[test]
    fn a_frame_of_192_kb_is_each_threads_own_launch_after_launch() {
        for threads in 1..=threads() {
            let want: Vec<u64> = (0..threads as u64).map(|t| 2 * t + 3).collect();
            for launch in 0..50 {
                let mut out = vec![0u64; threads];
                // SAFETY: a word a thread behind `out`, which nothing else touches while the
                // threads run, and the kernel reaches nothing else.
                unsafe { framed::framed(kernite_cpu::Grid::new(threads), out.as_mut_ptr()) }.unwrap();
                assert_eq!(out, want, "{threads} threads, launch {launch}");
            }
        }
    }

    /// Launches from two threads at once, one of `normalize` and one of `diffuse`, each wait at
    /// a barrier of their grid's own: each result is the reference's, bit for bit, as when
    /// they launch alone, a hundred times over.
    #[test]
    fn launches_from_two_threads_at_once_wait_at_barriers_of_their_own() {
        assert!(super::supported(), "the kernels need AVX2 and FMA");
        let threads = threads().min(2);
        let x = floats(41_000, 17).iter().map(|x| x / 2.0 + 1.0).collect::<Vec<f32>>();
        let heat = floats(9000, 19);
        let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let (normal, spread) = (bits(&normalized(&x)), bits(&diffused(&heat, 50)));
        std::thread::scope(|s| {
            s.spawn(|| {
                for _ in 0..100 {
                    let mut got = x.clone();
                    super::normalize_on_threads(&mut got, threads).unwrap();
                    assert_eq!(bits(&got), normal);
                }
            });
            s.spawn(|| {
                for _ in 0..100 {
                    let mut got = heat.clone();
                    super::diffuse_on_threads(&mut got, 50, threads).unwrap();
                    assert_eq!(bits(&got), spread);
                }
            });
        });
    }

    /// `saxpy` on the GPU against `f32::mul_add`, which rounds once as `fma.rn.f32` does; on a
    /// machine with no GPU, skipped.
    #[test]
    fn saxpy_is_mul_add_bit_for_bit() {
        let cuda = match Cuda::new() {
            Ok(cuda) => cuda,
            Err(e) => return eprintln!("no GPU: skipped. {e}"),
        };
        let kernels = gpu::Kernels::load(&cuda).unwrap();
        for n in [0, 1, 255, 256, 257, 10_000] {
            let (x, mut y) = (floats(n, 3 * n as u32 + 1), floats(n, 3 * n as u32 + 2));
            let want: Vec<u32> = x.iter().zip(&y).map(|(x, y)| 1.5f32.mul_add(*x, *y).to_bits()).collect();
            super::saxpy(&cuda, &kernels, 1.5, &x, &mut y).unwrap();
            assert_eq!(y.iter().map(|y| y.to_bits()).collect::<Vec<_>>(), want, "{n} floats");
        }
    }

    /// `Kernels::load` on the `Cuda` of each GPU, all held at once on this thread, and `saxpy`
    /// launched on each in turn against `f32::mul_add`, twice around, so that each launch follows
    /// one on another GPU. On a machine with one GPU it runs on that one; with none, skipped.
    #[test]
    fn saxpy_is_mul_add_bit_for_bit_on_each_gpu() {
        let Some(first) = cuda() else { return };
        let others = (1..Cuda::count().unwrap()).map(|index| {
            let cuda = Cuda::on(index).unwrap();
            gpu::Kernels::load(&cuda).map(|kernels| (cuda, kernels)).unwrap()
        });
        let gpus: Vec<(Cuda, gpu::Kernels)> = std::iter::once(first).chain(others).collect();
        for round in 0..2 {
            for (index, (cuda, kernels)) in gpus.iter().enumerate() {
                let seed = 10 * round + index as u32;
                let (x, mut y) = (floats(10_000, 3 * seed + 1), floats(10_000, 3 * seed + 2));
                let want: Vec<u32> = x.iter().zip(&y).map(|(x, y)| 1.5f32.mul_add(*x, *y).to_bits()).collect();
                super::saxpy(cuda, kernels, 1.5, &x, &mut y).unwrap();
                assert_eq!(y.iter().map(|y| y.to_bits()).collect::<Vec<_>>(), want, "GPU {index}, round {round}");
            }
        }
    }

    /// GPU kernels for the tests alone, of nothing past `sm_20`.
    #[kernite_macros::kernels]
    mod oldest {
        /// `y[i] = a * x[i] + y[i]` for each `i < n`, rounded once.
        pub unsafe extern "ptx-kernel" fn saxpy(a: f32, x: *const f32, y: *mut f32, n: u32) {
            let (block, size, thread, i): (u32, u32, u32, u32);
            mov.u32(block, ctaid.x);
            mov.u32(size, ntid.x);
            mov.u32(thread, tid.x);
            mad.lo.u32(i, block, size, thread);
            let past: pred;
            setp.ge.u32(past, i, n);
            if past {
                ret();
            }
            let (at_x, at_y): (u64, u64);
            mad.wide.u32(at_x, i, 4, x);
            mad.wide.u32(at_y, i, 4, y);
            let (xi, yi): (f32, f32);
            ld.global.f32(xi, [at_x]);
            ld.global.f32(yi, [at_y]);
            fma.rn.f32(yi, a, xi, yi);
            st.global.f32([at_y], yi);
        }

        /// Stores `ctaid.x` of the last block, the one whose number is one less than the blocks
        /// along x, in `out[0]`.
        pub unsafe extern "ptx-kernel" fn last(out: *mut u32) {
            let (block, blocks): (u32, u32);
            mov.u32(block, ctaid.x);
            mov.u32(blocks, nctaid.x);
            sub.u32(blocks, blocks, 1);
            let other: pred;
            setp.ne.u32(other, block, blocks);
            if other {
                ret();
            }
            st.global.u32([out], block);
        }
    }

    /// A module of nothing past `sm_20` is written for `sm_30`, as every module is: the driver
    /// starts 65535 blocks along x of one written for `sm_20`, whatever the GPU, and 2^31 - 1 of
    /// one written for `sm_30`. `saxpy` over 65536 blocks against `f32::mul_add`, and the last of
    /// 2^31 - 1 blocks of one thread; on a machine with no GPU, skipped.
    #[test]
    fn a_kernel_of_nothing_past_sm_20_starts_every_grid_the_gpu_takes() {
        let cuda = match Cuda::new() {
            Ok(cuda) => cuda,
            Err(e) => return eprintln!("no GPU: skipped. {e}"),
        };
        assert!(oldest::PTX.contains(".version 3.0\n.target sm_30\n"), "{}", oldest::PTX);
        let kernels = oldest::Kernels::load(&cuda).unwrap();
        // 65536 blocks of 256 threads, the last with one float to its name.
        let n = 65535 * 256 + 1;
        let (x, y) = (floats(n, 7), floats(n, 8));
        let want: Vec<u32> = x.iter().zip(&y).map(|(x, y)| 1.5f32.mul_add(*x, *y).to_bits()).collect();
        let (xs, ys) = (cuda.upload(&x).unwrap(), cuda.upload(&y).unwrap());
        // SAFETY: `n` floats behind each pointer, in buffers that last until `read` has waited
        // for the kernel.
        unsafe { kernels.saxpy(Grid::new(65536, 256), 1.5, xs.ptr(), ys.ptr(), n as u32) }.unwrap();
        assert_eq!(ys.read().unwrap().iter().map(|y| y.to_bits()).collect::<Vec<_>>(), want);
        let most = (1 << 31) - 1;
        let out = cuda.alloc::<u32>(1).unwrap();
        // SAFETY: a word at `out`, which the last block writes.
        unsafe { kernels.last(Grid::new(most, 1), out.ptr()) }.unwrap();
        assert_eq!(out.read().unwrap(), [most - 1]);
    }

    /// GPU kernels for the tests alone, of 20 KB of shared memory of their own and the launch's
    /// beside it, by a launch and by a cooperative one.
    #[kernite_macros::kernels]
    mod beside {
        /// The last word of its 20 KB of shared memory, `fixed`, and the last of the `bytes` the
        /// launch gives, `launched`, each written and read back, to `out[0]` and `out[1]`.
        pub unsafe extern "ptx-kernel" fn both(out: *mut u32, bytes: u32) {
            write_both(out, bytes);
        }

        /// As `both`, and then the grid waits.
        #[cooperative]
        pub unsafe extern "ptx-kernel" fn together(out: *mut u32, bytes: u32) {
            write_both(out, bytes);
            grid.sync();
        }

        /// Writes the last word of each memory, reads it back, and stores it at `out`.
        fn write_both(out: u64, bytes: u32) {
            let fixed: shared<[u32; 5120], 4>;
            let launched: shared<[u32]>;
            let (seven, nine, mine, theirs, at): (u32, u32, u32, u32, u32);
            mov.u32(seven, 7);
            st.shared.u32([fixed + 20476], seven);
            ld.shared.u32(mine, [fixed + 20476]);
            st.global.u32([out], mine);
            mov.u32(at, launched);
            add.u32(at, at, bytes);
            mov.u32(nine, 9);
            st.shared.u32([at - 4], nine);
            ld.shared.u32(theirs, [at - 4]);
            st.global.u32([out + 4], theirs);
        }
    }

    /// What a launch past the shared memory a block has says: the sizes, and what the GPU takes.
    fn said(bytes: u32, most: u32) -> String {
        format!(
            ": each block has {} bytes of shared memory, 20480 of the kernel's and {bytes} of the launch's, and the \
             GPU takes {most} at most",
            20480 + u64::from(bytes)
        )
    }

    /// A kernel of 20 KB of shared memory of its own is given the launch's beside it, up to all a
    /// block has: from 16 KB to 79 KB, 36 KB to 99 KB in all on an RTX 30, each launch runs and the
    /// kernel reads what it wrote, by a launch and by a cooperative one; and a launch past what a
    /// block has is an error that says how much it has and how much the GPU takes. On a machine
    /// with no GPU, with one older than Pascal, `sm_60`, or whose blocks have less than 99 KB,
    /// skipped.
    #[test]
    fn a_kernel_with_shared_memory_of_its_own_is_given_the_launchs_beside_it() {
        let cuda = match Cuda::new() {
            Ok(cuda) => cuda,
            Err(e) => return eprintln!("no GPU: skipped. {e}"),
        };
        let most = cuda.max_shared();
        if cuda.sm() < 60 || most < 99 << 10 {
            eprintln!("the {}, sm_{}, gives a block {most} bytes: skipped", cuda.name(), cuda.sm());
            return;
        }
        let kernels = beside::Kernels::load(&cuda).unwrap();
        let out = cuda.alloc::<u32>(2).unwrap();
        let all = most - 20480;
        // 28 KB is what a launch is given unasked beside the kernel's own 20 KB; the rest the
        // kernel must be let have.
        for bytes in [16 << 10, 28 << 10, 29 << 10, 40 << 10, 48 << 10, 49 << 10, 79 << 10, all] {
            let grid = Grid::new(1, 1).with_shared(bytes);
            // SAFETY: `both` and `together` write two words at `out`, which has them, and the last
            // word of the `bytes` of shared memory they are given.
            unsafe { kernels.both(grid, out.ptr(), bytes) }.unwrap();
            assert_eq!(out.read().unwrap(), [7, 9], "{bytes} bytes");
            // SAFETY: as above.
            unsafe { kernels.together(grid, out.ptr(), bytes) }.unwrap();
            assert_eq!(out.read().unwrap(), [7, 9], "{bytes} bytes, cooperatively");
            assert!(kernels.together_blocks(1, bytes).unwrap() >= 1, "{bytes} bytes");
        }
        // 80 KB, 100 KB in all.
        for bytes in [all + 1, all + 1024, u32::MAX] {
            let grid = Grid::new(1, 1).with_shared(bytes);
            // SAFETY: as above; the launches are refused.
            let launched =
                [unsafe { kernels.both(grid, out.ptr(), bytes) }, unsafe { kernels.together(grid, out.ptr(), bytes) }];
            let blocks = kernels.together_blocks(1, bytes).map(|_| ());
            for e in launched.into_iter().chain([blocks]).map(Result::unwrap_err) {
                assert_eq!(e.call, "cuFuncSetAttribute", "{e}");
                assert!(e.message.starts_with("CUDA_ERROR_INVALID_VALUE (1): "), "{e}");
                assert!(e.message.ends_with(&said(bytes, most)), "{e}");
            }
        }
        cuda.sync().unwrap();
    }
}
