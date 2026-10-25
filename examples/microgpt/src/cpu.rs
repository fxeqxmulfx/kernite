//! The model on the CPU: its buffers in memory, and kernels in x86 over them, which need AVX2
//! and FMA. Each computes what [`crate::reference`] does, in its order, so that the two agree
//! bit for bit: a sum in order is a chain of scalar fused multiply-adds, and what is the same
//! for each of a row's elements is done eight at a time.

use crate::model::{Adam, Buf, D, E, FC1, FC2, H, LM, Layer, M, Ops, PARAMS, QKV, T, Then, V, VP, WO, WPE, WTE};
use crate::reference;

/// The kernels. Pointers are to rows of the buffers [`Buf`] names, and a count `n` is of rows
/// unless it says otherwise; each kernel trusts that its pointers have that many rows to read
/// and write.
#[kgen_macros::kernels]
pub mod kernels {
    /// The model's sizes, as [`crate::model`] has them, which the crate checks as it compiles.
    pub const E: usize = 16;
    #[allow(missing_docs)]
    pub const H: usize = 4;
    #[allow(missing_docs)]
    pub const D: usize = 4;
    #[allow(missing_docs)]
    pub const T: usize = 16;
    #[allow(missing_docs)]
    pub const V: usize = 27;
    #[allow(missing_docs)]
    pub const VP: usize = 32;
    #[allow(missing_docs)]
    pub const M: usize = 64;
    #[allow(missing_docs)]
    pub const WPE: usize = 432;
    /// Where `QKV`'s queries, keys and values are in a row of them.
    pub const QKV_COLS: usize = 3 * E;

    /// `reference::exp`'s constants.
    pub const LN2_HI: f32 = 355.0 / 512.0;
    #[allow(missing_docs)]
    pub const LN2_LO: f32 = -2.121_944_4e-4;
    #[allow(missing_docs)]
    pub const TAYLOR: [f32; 7] = [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0];

    /// What `matmul` does to each output before it stores it: nothing, adds the residual
    /// stream's, or keeps it only above zero. Only the kernels read these, which Rust does not
    /// see.
    #[allow(dead_code)]
    pub const STORE: usize = 0;
    #[allow(dead_code, missing_docs)]
    pub const ADD: usize = 1;
    #[allow(dead_code, missing_docs)]
    pub const RELU: usize = 2;

    /// `x0 = wte[token] + wpe[t]` for rows `t0..n`, `p` the parameters.
    pub unsafe extern "sysv64" fn embed(x0: *mut f32, p: *const f32, tokens: *const u32, t0: usize, n: usize) {
        cmp(t0, n);
        jae(done);
        label(row);
        let (tok, at): (gpr, gpr);
        mov(tok.r32, [tokens + 4 * t0]);
        shl(tok, 6);
        mov(at, t0);
        shl(at, 6);
        for k in 0..2 {
            let x: ymm;
            vmovups(x, [p + tok + 32 * k]);
            vaddps(x, x, [p + at + (4 * WPE + 32 * k)]);
            vmovups([x0 + at + 32 * k], x);
        }
        inc(t0);
        cmp(t0, n);
        jb(row);
        label(done);
    }

    /// Each of `n` rows of `x` divided by its root mean square, into `y`, and the scale it
    /// took into `r`.
    pub unsafe extern "sysv64" fn rmsnorm(y: *mut f32, r: *mut f32, x: *const f32, n: usize) {
        test(n, n);
        jz(done);
        let one: ymm;
        vmovss(one.xmm, [1.0]);
        label(row);
        note("the mean square");
        let ss: ymm;
        zero(ss);
        for e in 0..E {
            let v: ymm;
            vmovss(v.xmm, [x + 4 * e]);
            vfmadd231ss(ss.xmm, v.xmm, v.xmm);
        }
        vmulss(ss.xmm, ss.xmm, [0.0625]);
        note("its root's reciprocal, and the row by it");
        vaddss(ss.xmm, ss.xmm, [1e-5]);
        vsqrtss(ss.xmm, ss.xmm, ss.xmm);
        let s: ymm;
        vdivss(s.xmm, one.xmm, ss.xmm);
        vmovss([r], s.xmm);
        vbroadcastss(s, s.xmm);
        for k in 0..2 {
            let v: ymm;
            vmulps(v, s, [x + 32 * k]);
            vmovups([y + 32 * k], v);
        }
        add(x, 4 * E);
        add(y, 4 * E);
        add(r, 4);
        dec(n);
        jnz(row);
        label(done);
    }

    /// `y = x W`, `W` the queries', keys' and values' weights side by side.
    pub unsafe extern "sysv64" fn qkv(y: *mut f32, x: *const f32, w: *const f32, n: usize) {
        matmul::<E, QKV_COLS, STORE>(y, x, w, y, n);
    }

    /// `y = x W + res`, `W` attention's output projection.
    pub unsafe extern "sysv64" fn wo(y: *mut f32, x: *const f32, w: *const f32, res: *const f32, n: usize) {
        matmul::<E, E, ADD>(y, x, w, res, n);
    }

    /// `y = relu(x W)`, `W` the MLP's first layer.
    pub unsafe extern "sysv64" fn fc1(y: *mut f32, x: *const f32, w: *const f32, n: usize) {
        matmul::<E, M, RELU>(y, x, w, y, n);
    }

    /// `y = x W + res`, `W` the MLP's second layer.
    pub unsafe extern "sysv64" fn fc2(y: *mut f32, x: *const f32, w: *const f32, res: *const f32, n: usize) {
        matmul::<M, E, ADD>(y, x, w, res, n);
    }

    /// `y = x W`, `W` the logits' head.
    pub unsafe extern "sysv64" fn lm(y: *mut f32, x: *const f32, w: *const f32, n: usize) {
        matmul::<E, VP, STORE>(y, x, w, y, n);
    }

    /// `n` rows of `y = x W`, `W` of `ROWS` rows of `COLS`, then what `THEN` says: `res` is
    /// the residual stream's rows when it adds them. Each output is a chain over the inputs in
    /// order, eight outputs to a vector.
    fn matmul<const ROWS: usize, const COLS: usize, const THEN: usize>(y: gpr, x: gpr, w: gpr, res: gpr, n: gpr) {
        test(n, n);
        jz(done);
        let z: ymm;
        if THEN == RELU {
            zero(z);
        }
        label(row);
        let acc: [ymm; COLS / 8];
        for k in 0..COLS / 8 {
            zero(acc[k]);
        }
        for i in 0..ROWS {
            let b: ymm;
            vbroadcastss(b, [x + 4 * i]);
            for k in 0..COLS / 8 {
                vfmadd231ps(acc[k], b, [w + 4 * (i * COLS + 8 * k)]);
            }
        }
        for k in 0..COLS / 8 {
            if THEN == ADD {
                vaddps(acc[k], acc[k], [res + 32 * k]);
            }
            if THEN == RELU {
                vmaxps(acc[k], acc[k], z);
            }
            vmovups([y + 32 * k], acc[k]);
        }
        add(x, 4 * ROWS);
        add(y, 4 * COLS);
        if THEN == ADD {
            add(res, 4 * COLS);
        }
        dec(n);
        jnz(row);
        label(done);
    }

    /// Each head of rows `t0..n` attending to the rows up to it, from `qkv`'s queries, keys
    /// and values: its weights into `aw`, and the values they weigh, summed, into `att`.
    pub unsafe extern "sysv64" fn attention(att: *mut f32, aw: *mut f32, qkv: *const f32, t0: usize, n: usize) {
        cmp(t0, n);
        jae(done);
        let one: ymm;
        vmovss(one.xmm, [1.0]);
        label(row);
        let (q, w, out): (gpr, gpr, gpr);
        imul(q, t0, 4 * QKV_COLS);
        add(q, qkv);
        imul(w, t0, 4 * H * T);
        add(w, aw);
        imul(out, t0, 4 * E);
        add(out, att);
        for h in 0..H {
            head::<h>(out, w, q, qkv, t0, one);
        }
        inc(t0);
        cmp(t0, n);
        jb(row);
        label(done);
    }

    /// Head `HEAD` of row `t`: its query at `q`, its weights' row at `w` and its output's at
    /// `out`; `qkv` the first row's query.
    fn head<const HEAD: usize>(out: gpr, w: gpr, q: gpr, qkv: gpr, t: gpr, one: ymm) {
        note("head {HEAD}: the logits, and the largest");
        let qj: [ymm; D];
        for j in 0..D {
            vmovss(qj[j].xmm, [q + 4 * (HEAD * D + j)]);
        }
        let m: ymm;
        vmovss(m.xmm, [f32::NEG_INFINITY]);
        let (ks, ws, left): (gpr, gpr, gpr);
        lea(ks, [qkv + 4 * (E + HEAD * D)]);
        lea(ws, [w + 4 * HEAD * T]);
        lea(left, [t + 1]);
        label(logits);
        let l: ymm;
        zero(l);
        for j in 0..D {
            vfmadd231ss(l.xmm, qj[j].xmm, [ks + 4 * j]);
        }
        vmulss(l.xmm, l.xmm, [0.5]);
        vmovss([ws], l.xmm);
        vmaxss(m.xmm, l.xmm, m.xmm);
        add(ks, 4 * QKV_COLS);
        add(ws, 4);
        dec(left);
        jnz(logits);

        note("head {HEAD}: their exponentials, and the sum");
        let total: ymm;
        zero(total);
        lea(ws, [w + 4 * HEAD * T]);
        lea(left, [t + 1]);
        label(exps);
        let x: ymm;
        vmovss(x.xmm, [ws]);
        vsubss(x.xmm, x.xmm, m.xmm);
        exp(x);
        vmovss([ws], x.xmm);
        vaddss(total.xmm, total.xmm, x.xmm);
        add(ws, 4);
        dec(left);
        jnz(exps);

        note("head {HEAD}: the weights, and the values they weigh");
        let (inv, sum): (ymm, ymm);
        vdivss(inv.xmm, one.xmm, total.xmm);
        zero(sum);
        lea(ks, [qkv + 4 * (2 * E + HEAD * D)]);
        lea(ws, [w + 4 * HEAD * T]);
        lea(left, [t + 1]);
        label(values);
        let ww: ymm;
        vmulss(ww.xmm, inv.xmm, [ws]);
        vmovss([ws], ww.xmm);
        vbroadcastss(ww.xmm, ww.xmm);
        vfmadd231ps(sum.xmm, ww.xmm, [ks]);
        add(ks, 4 * QKV_COLS);
        add(ws, 4);
        dec(left);
        jnz(values);
        vmovups([out + 4 * HEAD * D], sum.xmm);
    }

    /// `probs = softmax(logits * inv_temp)` over the first `V` of each of `n` rows, the rest
    /// zero.
    pub unsafe extern "sysv64" fn softmax(probs: *mut f32, logits: *const f32, inv_temp: f32, n: usize) {
        test(n, n);
        jz(done);
        let one: ymm;
        vmovss(one.xmm, [1.0]);
        vbroadcastss(inv_temp.ymm, inv_temp);
        label(row);
        note("scaled, and the largest");
        for k in 0..VP / 8 {
            let p: ymm;
            vmulps(p, inv_temp.ymm, [logits + 32 * k]);
            vmovups([probs + 32 * k], p);
        }
        let m: ymm;
        vmovss(m.xmm, [f32::NEG_INFINITY]);
        for j in 0..V {
            let p: ymm;
            vmovss(p.xmm, [probs + 4 * j]);
            vmaxss(m.xmm, p.xmm, m.xmm);
        }
        note("the exponentials, and their sum");
        vbroadcastss(m, m.xmm);
        for k in 0..VP / 8 {
            let p: ymm;
            vmovups(p, [probs + 32 * k]);
            vsubps(p, p, m);
            exp(p);
            vmovups([probs + 32 * k], p);
        }
        let total: ymm;
        zero(total);
        for j in 0..V {
            vaddss(total.xmm, total.xmm, [probs + 4 * j]);
        }
        note("each divided by it, past V zero");
        let inv: ymm;
        vdivss(inv.xmm, one.xmm, total.xmm);
        vbroadcastss(inv, inv.xmm);
        for k in 0..VP / 8 {
            let p: ymm;
            vmulps(p, inv, [probs + 32 * k]);
            // V is 27: the last vector keeps three lanes.
            if k == V / 8 {
                vandps(p, p, [[-1, -1, -1, 0, 0, 0, 0, 0]]);
            }
            vmovups([probs + 32 * k], p);
        }
        add(probs, 4 * VP);
        add(logits, 4 * VP);
        dec(n);
        jnz(row);
        label(done);
    }

    /// `x = e^x` in each lane, for `x <= 0`, as `reference::exp` works it out: `k ln 2 + r`,
    /// `e^r` by its Taylor series, `2^k` by its bits.
    fn exp(x: ymm) {
        vmaxps(x, x, [[-87.0; 8]]);
        let (k, p): (ymm, ymm);
        vmulps(k, x, [[f32::consts::LOG2_E; 8]]);
        vroundps(k, k, 0x08);
        vfnmadd231ps(x, k, [[LN2_HI; 8]]);
        vfnmadd231ps(x, k, [[LN2_LO; 8]]);
        vmovaps(p, [[TAYLOR[0]; 8]]);
        for c in 1..7 {
            vfmadd213ps(p, x, [[TAYLOR[c]; 8]]);
        }
        vcvtps2dq(k, k);
        vpaddd(k, k, [[127; 8]]);
        vpslld(k, k, 23);
        vmulps(x, p, k);
    }

    /// `d = (probs - onehot(target)) * inv_n` for each of `n` rows, the targets at `targets`.
    pub unsafe extern "sysv64" fn xent(d: *mut f32, probs: *const f32, targets: *const u32, inv_n: f32, n: usize) {
        test(n, n);
        jz(done);
        vbroadcastss(inv_n.ymm, inv_n);
        label(row);
        for k in 0..VP / 8 {
            let g: ymm;
            vmulps(g, inv_n.ymm, [probs + 32 * k]);
            vmovups([d + 32 * k], g);
        }
        note("the target's");
        let j: gpr;
        mov(j.r32, [targets]);
        let g: ymm;
        vmovss(g.xmm, [probs + 4 * j]);
        vsubss(g.xmm, g.xmm, [1.0]);
        vmulss(g.xmm, g.xmm, inv_n.xmm);
        vmovss([d + 4 * j], g.xmm);
        add(d, 4 * VP);
        add(probs, 4 * VP);
        add(targets, 4);
        dec(n);
        jnz(row);
        label(done);
    }

    /// `dx += dy Wᵀ` for `QKV`.
    pub unsafe extern "sysv64" fn qkv_bwd_x(dx: *mut f32, dy: *const f32, w: *const f32, n: usize) {
        bwd_x::<E, QKV_COLS>(dx, dy, w, n);
    }

    /// `dx += dy Wᵀ` for `WO`.
    pub unsafe extern "sysv64" fn wo_bwd_x(dx: *mut f32, dy: *const f32, w: *const f32, n: usize) {
        bwd_x::<E, E>(dx, dy, w, n);
    }

    /// `dx += dy Wᵀ` for `FC1`.
    pub unsafe extern "sysv64" fn fc1_bwd_x(dx: *mut f32, dy: *const f32, w: *const f32, n: usize) {
        bwd_x::<E, M>(dx, dy, w, n);
    }

    /// `dx += dy Wᵀ` for `FC2`.
    pub unsafe extern "sysv64" fn fc2_bwd_x(dx: *mut f32, dy: *const f32, w: *const f32, n: usize) {
        bwd_x::<M, E>(dx, dy, w, n);
    }

    /// `dx += dy Wᵀ` for `LM`.
    pub unsafe extern "sysv64" fn lm_bwd_x(dx: *mut f32, dy: *const f32, w: *const f32, n: usize) {
        bwd_x::<E, VP>(dx, dy, w, n);
    }

    /// `n` rows of `dx += dy Wᵀ`, `W` of `ROWS` rows of `COLS`: each input's gradient a chain
    /// over the outputs in order, from where it was, four inputs at a time.
    fn bwd_x<const ROWS: usize, const COLS: usize>(dx: gpr, dy: gpr, w: gpr, n: gpr) {
        test(n, n);
        jz(done);
        label(row);
        let (wi, xi, left): (gpr, gpr, gpr);
        mov(wi, w);
        mov(xi, dx);
        mov(left, ROWS / 4);
        label(four);
        let acc: [ymm; 4];
        for k in 0..4 {
            vmovss(acc[k].xmm, [xi + 4 * k]);
        }
        for o in 0..COLS {
            let g: ymm;
            vmovss(g.xmm, [dy + 4 * o]);
            for k in 0..4 {
                vfmadd231ss(acc[k].xmm, g.xmm, [wi + 4 * (k * COLS + o)]);
            }
        }
        for k in 0..4 {
            vmovss([xi + 4 * k], acc[k].xmm);
        }
        add(wi, 16 * COLS);
        add(xi, 16);
        dec(left);
        jnz(four);
        add(dx, 4 * ROWS);
        add(dy, 4 * COLS);
        dec(n);
        jnz(row);
        label(done);
    }

    /// `g += xᵀ dy` for `QKV`.
    pub unsafe extern "sysv64" fn qkv_bwd_w(g: *mut f32, x: *const f32, dy: *const f32, n: usize) {
        bwd_w::<E, QKV_COLS>(g, x, dy, n);
    }

    /// `g += xᵀ dy` for `WO`.
    pub unsafe extern "sysv64" fn wo_bwd_w(g: *mut f32, x: *const f32, dy: *const f32, n: usize) {
        bwd_w::<E, E>(g, x, dy, n);
    }

    /// `g += xᵀ dy` for `FC1`.
    pub unsafe extern "sysv64" fn fc1_bwd_w(g: *mut f32, x: *const f32, dy: *const f32, n: usize) {
        bwd_w::<E, M>(g, x, dy, n);
    }

    /// `g += xᵀ dy` for `FC2`.
    pub unsafe extern "sysv64" fn fc2_bwd_w(g: *mut f32, x: *const f32, dy: *const f32, n: usize) {
        bwd_w::<M, E>(g, x, dy, n);
    }

    /// `g += xᵀ dy` for `LM`.
    pub unsafe extern "sysv64" fn lm_bwd_w(g: *mut f32, x: *const f32, dy: *const f32, n: usize) {
        bwd_w::<E, VP>(g, x, dy, n);
    }

    /// `g += xᵀ dy` over `n` rows, `g` of `ROWS` rows of `COLS`: each weight's gradient a
    /// chain over the rows in order, from where it was, eight weights to a vector.
    fn bwd_w<const ROWS: usize, const COLS: usize>(g: gpr, x: gpr, dy: gpr, n: gpr) {
        test(n, n);
        jz(done);
        let i: gpr;
        mov(i, ROWS);
        label(input);
        let acc: [ymm; COLS / 8];
        for k in 0..COLS / 8 {
            vmovups(acc[k], [g + 32 * k]);
        }
        let (xt, dyt, left): (gpr, gpr, gpr);
        mov(xt, x);
        mov(dyt, dy);
        mov(left, n);
        label(row);
        let b: ymm;
        vbroadcastss(b, [xt]);
        for k in 0..COLS / 8 {
            vfmadd231ps(acc[k], b, [dyt + 32 * k]);
        }
        add(xt, 4 * ROWS);
        add(dyt, 4 * COLS);
        dec(left);
        jnz(row);
        for k in 0..COLS / 8 {
            vmovups([g + 32 * k], acc[k]);
        }
        add(g, 4 * COLS);
        add(x, 4);
        dec(i);
        jnz(input);
        label(done);
    }

    /// `dh = 0` where `h` is not above zero, over `n` vectors of eight.
    pub unsafe extern "sysv64" fn relu_bwd(dh: *mut f32, h: *const f32, n: usize) {
        test(n, n);
        jz(done);
        let z: ymm;
        zero(z);
        label(top);
        let keep: ymm;
        note("0 < h, which no NaN is");
        vcmpps(keep, z, [h], 0x11);
        vandps(keep, keep, [dh]);
        vmovups([dh], keep);
        add(h, 32);
        add(dh, 32);
        dec(n);
        jnz(top);
        label(done);
    }

    /// `dx +=` the gradient by `x` of normalizing it, given the gradient `dy` by the result and
    /// the scale `r` each of `n` rows took.
    pub unsafe extern "sysv64" fn rmsnorm_bwd(dx: *mut f32, dy: *const f32, x: *const f32, r: *const f32, n: usize) {
        test(n, n);
        jz(done);
        label(row);
        let dot: ymm;
        zero(dot);
        for e in 0..E {
            let v: ymm;
            vmovss(v.xmm, [dy + 4 * e]);
            vfmadd231ss(dot.xmm, v.xmm, [x + 4 * e]);
        }
        note("c = dot r^3 / E");
        let (s, c): (ymm, ymm);
        vmovss(s.xmm, [r]);
        vmulss(c.xmm, s.xmm, s.xmm);
        vmulss(c.xmm, c.xmm, s.xmm);
        vmulss(c.xmm, dot.xmm, c.xmm);
        vmulss(c.xmm, c.xmm, [0.0625]);
        vbroadcastss(c, c.xmm);
        vbroadcastss(s, s.xmm);
        note("dx += dy r - c x");
        for k in 0..2 {
            let g: ymm;
            vmulps(g, s, [dy + 32 * k]);
            vfnmadd231ps(g, c, [x + 32 * k]);
            vaddps(g, g, [dx + 32 * k]);
            vmovups([dx + 32 * k], g);
        }
        add(dx, 4 * E);
        add(dy, 4 * E);
        add(x, 4 * E);
        add(r, 4);
        dec(n);
        jnz(row);
        label(done);
    }

    /// `dqkv` from `datt`, attention's gradient, over rows `0..n`: each row's query's, and
    /// the keys and values it attended to added to.
    pub unsafe extern "sysv64" fn attention_bwd(
        dqkv: *mut f32,
        qkv: *const f32,
        aw: *const f32,
        datt: *const f32,
        n: usize,
    ) {
        test(n, n);
        jz(done);
        for h in 0..H {
            head_bwd::<h>(dqkv, qkv, aw, datt, n);
        }
        label(done);
    }

    /// Head `HEAD`'s part of `attention_bwd`, a row at a time in order.
    fn head_bwd<const HEAD: usize>(dqkv: gpr, qkv: gpr, aw: gpr, datt: gpr, n: gpr) {
        let t: gpr;
        zero(t);
        label(row);
        note("head {HEAD}: the output's gradient");
        let (w, at): (gpr, gpr);
        imul(w, t, 4 * H * T);
        add(w, aw);
        imul(at, t, 4 * E);
        let dout: [ymm; D];
        for j in 0..D {
            vmovss(dout[j].xmm, [datt + at + 4 * (HEAD * D + j)]);
        }
        let douts: ymm;
        vmovups(douts.xmm, [datt + at + 4 * HEAD * D]);

        note("head {HEAD}: the sum of the weights by their gradients");
        let (vs, ws, left): (gpr, gpr, gpr);
        lea(vs, [qkv + 4 * (2 * E + HEAD * D)]);
        lea(ws, [w + 4 * HEAD * T]);
        lea(left, [t + 1]);
        let sum: ymm;
        zero(sum);
        label(sums);
        let (dw, ww): (ymm, ymm);
        weight_grad(dw, dout, vs);
        vmovss(ww.xmm, [ws]);
        vfmadd231ss(sum.xmm, ww.xmm, dw.xmm);
        add(vs, 4 * QKV_COLS);
        add(ws, 4);
        dec(left);
        jnz(sums);

        note("head {HEAD}: the query's gradient, and the keys' and values'");
        let (q, ks, dks): (gpr, gpr, gpr);
        imul(q, t, 4 * QKV_COLS);
        let qs: ymm;
        vmovups(qs.xmm, [qkv + q + 4 * HEAD * D]);
        lea(ks, [qkv + 4 * (E + HEAD * D)]);
        lea(dks, [dqkv + 4 * (E + HEAD * D)]);
        lea(vs, [qkv + 4 * (2 * E + HEAD * D)]);
        lea(ws, [w + 4 * HEAD * T]);
        lea(left, [t + 1]);
        let dq: ymm;
        zero(dq);
        label(grads);
        let (dw, ww, acc): (ymm, ymm, ymm);
        weight_grad(dw, dout, vs);
        vsubss(dw.xmm, dw.xmm, sum.xmm);
        vmovss(ww.xmm, [ws]);
        vmulss(dw.xmm, ww.xmm, dw.xmm);
        vmulss(dw.xmm, dw.xmm, [0.5]);
        vbroadcastss(dw.xmm, dw.xmm);
        vfmadd231ps(dq.xmm, dw.xmm, [ks]);
        vmovups(acc.xmm, [dks]);
        vfmadd231ps(acc.xmm, dw.xmm, qs.xmm);
        vmovups([dks], acc.xmm);
        vbroadcastss(ww.xmm, ww.xmm);
        vmovups(acc.xmm, [dks + 4 * E]);
        vfmadd231ps(acc.xmm, ww.xmm, douts.xmm);
        vmovups([dks + 4 * E], acc.xmm);
        add(ks, 4 * QKV_COLS);
        add(vs, 4 * QKV_COLS);
        add(dks, 4 * QKV_COLS);
        add(ws, 4);
        dec(left);
        jnz(grads);
        vmovups([dqkv + q + 4 * HEAD * D], dq.xmm);
        inc(t);
        cmp(t, n);
        jb(row);
    }

    /// `dw`, the gradient by one weight: the output's gradient `dout`, a lane each, by the
    /// value at `v`, a chain in order.
    fn weight_grad(dw: ymm, dout: [ymm; D], v: gpr) {
        zero(dw);
        for j in 0..D {
            vfmadd231ss(dw.xmm, dout[j].xmm, [v + 4 * j]);
        }
    }

    /// The embeddings' gradients, `g` the parameters': `wte[token] += dx0` and `wpe[t] +=
    /// dx0` for rows `0..n`, in order.
    pub unsafe extern "sysv64" fn embed_bwd(g: *mut f32, dx0: *const f32, tokens: *const u32, n: usize) {
        test(n, n);
        jz(done);
        let at: gpr;
        lea(at, [g + 4 * WPE]);
        label(row);
        let tok: gpr;
        mov(tok.r32, [tokens]);
        shl(tok, 6);
        for k in 0..2 {
            let (d, s): (ymm, ymm);
            vmovups(d, [dx0 + 32 * k]);
            vaddps(s, d, [g + tok + 32 * k]);
            vmovups([g + tok + 32 * k], s);
            vaddps(s, d, [at + 32 * k]);
            vmovups([at + 32 * k], s);
        }
        add(tokens, 4);
        add(dx0, 4 * E);
        add(at, 4 * E);
        dec(n);
        jnz(row);
        label(done);
    }

    /// Adam's step on `n` vectors of eight parameters `p`, by their gradients `g`, which it
    /// zeroes, and their moments `m` and `v`.
    pub unsafe extern "sysv64" fn adam(
        p: *mut f32,
        g: *mut f32,
        m: *mut f32,
        v: *mut f32,
        n: usize,
        lr: f32,
        b1: f32,
        ob1: f32,
        b2: f32,
        ob2: f32,
        bc1: f32,
        bc2: f32,
        eps: f32,
    ) {
        test(n, n);
        jz(done);
        vbroadcastss(lr.ymm, lr);
        vbroadcastss(b1.ymm, b1);
        vbroadcastss(ob1.ymm, ob1);
        vbroadcastss(b2.ymm, b2);
        vbroadcastss(ob2.ymm, ob2);
        vbroadcastss(bc1.ymm, bc1);
        vbroadcastss(bc2.ymm, bc2);
        vbroadcastss(eps.ymm, eps);
        let z: ymm;
        zero(z);
        label(top);
        let (gs, ms, vs, ps): (ymm, ymm, ymm, ymm);
        vmovups(gs, [g]);
        note("m = b1 m + (1 - b1) g; v = b2 v + (1 - b2) g^2");
        vmulps(ms, b1.ymm, [m]);
        vfmadd231ps(ms, ob1.ymm, gs);
        vmovups([m], ms);
        vmulps(gs, gs, gs);
        vmulps(vs, b2.ymm, [v]);
        vfmadd231ps(vs, ob2.ymm, gs);
        vmovups([v], vs);
        note("p -= lr m^ / (sqrt(v^) + eps)");
        vmulps(ms, ms, bc1.ymm);
        vmulps(vs, vs, bc2.ymm);
        vsqrtps(vs, vs);
        vaddps(vs, vs, eps.ymm);
        vmulps(ms, lr.ymm, ms);
        vdivps(ms, ms, vs);
        vmovups(ps, [p]);
        vsubps(ps, ps, ms);
        vmovups([p], ps);
        vmovups([g], z);
        add(p, 32);
        add(g, 32);
        add(m, 32);
        add(v, 32);
        dec(n);
        jnz(top);
        label(done);
    }
}

/// The model's buffers in memory, and the kernels over them.
pub struct Cpu {
    bufs: Vec<Vec<f32>>,
    tokens: Vec<u32>,
}

impl Cpu {
    /// A model of `params`, Adam's moments zero; `None` if this CPU has not AVX2 and FMA,
    /// which the kernels are written in.
    pub fn new(params: &[f32]) -> Option<Cpu> {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return None;
        }
        let mut cpu = Cpu { bufs: Buf::ALL.iter().map(|b| vec![0.0; b.len()]).collect(), tokens: vec![] };
        cpu.write(Buf::Params, params);
        Some(cpu)
    }

    /// Where element `at` of `b` is, for a kernel, `len` elements from there being in it.
    /// The pointers into a buffer come from the one pointer `as_mut_ptr` gives, so that none
    /// invalidates another.
    fn at(&mut self, b: Buf, at: usize, len: usize) -> *mut f32 {
        let buf = &mut self.bufs[b as usize];
        assert!(at + len <= buf.len(), "{len} floats at {at} of {b:?}, which has {}", buf.len());
        buf.as_mut_ptr().wrapping_add(at)
    }

    /// Row `t` of `b`, `rows` rows of it from there being in it.
    fn row(&mut self, b: Buf, t: usize, rows: usize) -> *mut f32 {
        self.at(b, t * b.width(), rows * b.width())
    }

    /// The weights of `l`.
    fn weights(&mut self, b: Buf, l: Layer) -> *mut f32 {
        self.at(b, l.at, l.rows * l.cols)
    }

    /// The tokens from `t`, `n - t` of them being there.
    fn tokens_at(&self, t: usize, n: usize) -> *const u32 {
        self.tokens[t..n].as_ptr()
    }
}

/// `t0..n` as rows of a document: within it, `T` at most.
fn rows(t0: usize, n: usize) -> usize {
    assert!(t0 <= n && n <= T, "rows {t0}..{n} of a document of {T}");
    n - t0
}

/// The kernels' sizes and constants are the model's.
const _: () = {
    assert!(kernels::QKV_COLS == QKV.cols);
    assert!(kernels::LN2_HI == reference::LN2_HI && kernels::LN2_LO == reference::LN2_LO);
    let mut i = 0;
    while i < kernels::TAYLOR.len() {
        assert!(kernels::TAYLOR[i] == reference::TAYLOR[i]);
        i += 1;
    }
    assert!(kernels::E == E && kernels::H == H && kernels::D == D && kernels::T == T && kernels::M == M);
    assert!(kernels::V == V && kernels::VP == VP && kernels::WPE == WPE && WTE == 0);
    assert!(E * 4 == 1 << 6, "embed and embed_bwd find a token's row by shifting it left 6");
    assert!(V % 8 == 3, "softmax keeps three lanes of the last vector");
    assert!(PARAMS.is_multiple_of(8) && M.is_multiple_of(8));
};

impl Ops for Cpu {
    fn write(&mut self, b: Buf, data: &[f32]) {
        self.bufs[b as usize].copy_from_slice(data);
    }

    fn read(&mut self, b: Buf) -> Vec<f32> {
        self.bufs[b as usize].clone()
    }

    fn tokens(&mut self, tokens: &[u32]) {
        assert!(tokens.len() <= T + 1 && tokens.iter().all(|&t| (t as usize) < V), "tokens of a document");
        self.tokens = tokens.to_vec();
    }

    fn embed(&mut self, t0: usize, n: usize) {
        rows(t0, n);
        let (x0, p, tokens) = (self.row(Buf::X0, 0, n), self.at(Buf::Params, 0, PARAMS), self.tokens_at(0, n));
        // SAFETY: `n` rows of `X0` and tokens, each token's row of `wte` and the rows of `wpe`
        // up to `n`, and a CPU that runs the kernels.
        unsafe { kernels::embed(x0, p, tokens, t0, n) }
    }

    fn rmsnorm(&mut self, y: Buf, r: Buf, x: Buf, t0: usize, n: usize) {
        let k = rows(t0, n);
        let (y, r, x) = (self.row(y, t0, k), self.row(r, t0, k), self.row(x, t0, k));
        // SAFETY: `k` rows of each from `t0`, and a CPU that runs the kernels.
        unsafe { kernels::rmsnorm(y, r, x, k) }
    }

    fn matmul(&mut self, l: Layer, y: Buf, x: Buf, then: Then, t0: usize, n: usize) {
        let k = rows(t0, n);
        assert_eq!((x.width(), y.width()), (l.rows, l.cols), "{x:?} by {l:?} into {y:?}");
        let (w, out, x) = (self.weights(Buf::Params, l), self.row(y, t0, k), self.row(x, t0, k));
        // SAFETY: `k` rows of each from `t0`, as wide as the layer says, its weights, and a
        // CPU that runs the kernels.
        unsafe {
            match (l, then) {
                (QKV, Then::Store) => kernels::qkv(out, x, w, k),
                (WO, Then::Add(res)) => kernels::wo(out, x, w, self.row(res, t0, k), k),
                (FC1, Then::Relu) => kernels::fc1(out, x, w, k),
                (FC2, Then::Add(res)) => kernels::fc2(out, x, w, self.row(res, t0, k), k),
                (LM, Then::Store) => kernels::lm(out, x, w, k),
                _ => panic!("no kernel computes {l:?} then {then:?}"),
            }
        }
    }

    fn attention(&mut self, t0: usize, n: usize) {
        rows(t0, n);
        let (att, aw, qkv) = (self.row(Buf::Att, 0, n), self.row(Buf::Aw, 0, n), self.row(Buf::Qkv, 0, n));
        // SAFETY: rows up to `n` of each, and a CPU that runs the kernels.
        unsafe { kernels::attention(att, aw, qkv, t0, n) }
    }

    fn softmax(&mut self, inv_temp: f32, t0: usize, n: usize) {
        let k = rows(t0, n);
        let (probs, logits) = (self.row(Buf::Probs, t0, k), self.row(Buf::Logits, t0, k));
        // SAFETY: `k` rows of each from `t0`, and a CPU that runs the kernels.
        unsafe { kernels::softmax(probs, logits, inv_temp, k) }
    }

    fn clear(&mut self) {
        for b in Buf::SUMS {
            self.bufs[b as usize].fill(0.0);
        }
    }

    fn xent(&mut self, n: usize) {
        rows(0, n);
        assert!(self.tokens.len() > n, "a target for each of {n} rows");
        let (d, probs, targets) = (self.row(Buf::DLogits, 0, n), self.row(Buf::Probs, 0, n), self.tokens_at(1, n + 1));
        // SAFETY: `n` rows of each and `n` targets, each below `V`, and a CPU that runs the
        // kernels.
        unsafe { kernels::xent(d, probs, targets, 1.0 / n as f32, n) }
    }

    fn bwd_x(&mut self, l: Layer, dx: Buf, dy: Buf, n: usize) {
        rows(0, n);
        assert_eq!((dx.width(), dy.width()), (l.rows, l.cols), "{dy:?} by {l:?} into {dx:?}");
        let (w, dx, dy) = (self.weights(Buf::Params, l), self.row(dx, 0, n), self.row(dy, 0, n));
        // SAFETY: `n` rows of each, as wide as the layer says, its weights, and a CPU that runs
        // the kernels.
        unsafe {
            match l {
                QKV => kernels::qkv_bwd_x(dx, dy, w, n),
                WO => kernels::wo_bwd_x(dx, dy, w, n),
                FC1 => kernels::fc1_bwd_x(dx, dy, w, n),
                FC2 => kernels::fc2_bwd_x(dx, dy, w, n),
                LM => kernels::lm_bwd_x(dx, dy, w, n),
                _ => panic!("no kernel for {l:?}"),
            }
        }
    }

    fn bwd_w(&mut self, l: Layer, x: Buf, dy: Buf, n: usize) {
        rows(0, n);
        assert_eq!((x.width(), dy.width()), (l.rows, l.cols), "{x:?} by {dy:?} for {l:?}");
        let (g, x, dy) = (self.weights(Buf::Grads, l), self.row(x, 0, n), self.row(dy, 0, n));
        // SAFETY: `n` rows of each, as wide as the layer says, its gradients, and a CPU that
        // runs the kernels.
        unsafe {
            match l {
                QKV => kernels::qkv_bwd_w(g, x, dy, n),
                WO => kernels::wo_bwd_w(g, x, dy, n),
                FC1 => kernels::fc1_bwd_w(g, x, dy, n),
                FC2 => kernels::fc2_bwd_w(g, x, dy, n),
                LM => kernels::lm_bwd_w(g, x, dy, n),
                _ => panic!("no kernel for {l:?}"),
            }
        }
    }

    fn relu_bwd(&mut self, n: usize) {
        rows(0, n);
        let (dh, h) = (self.row(Buf::DHid, 0, n), self.row(Buf::Hid, 0, n));
        // SAFETY: `n` rows of `M` floats of each, `M / 8` vectors a row, and a CPU that runs the
        // kernels.
        unsafe { kernels::relu_bwd(dh, h, n * M / 8) }
    }

    fn rmsnorm_bwd(&mut self, dx: Buf, dy: Buf, x: Buf, r: Buf, n: usize) {
        rows(0, n);
        let (dx, dy, x, r) = (self.row(dx, 0, n), self.row(dy, 0, n), self.row(x, 0, n), self.row(r, 0, n));
        // SAFETY: `n` rows of each, and a CPU that runs the kernels.
        unsafe { kernels::rmsnorm_bwd(dx, dy, x, r, n) }
    }

    fn attention_bwd(&mut self, n: usize) {
        rows(0, n);
        let dqkv = self.row(Buf::DQkv, 0, n);
        let (qkv, aw, datt) = (self.row(Buf::Qkv, 0, n), self.row(Buf::Aw, 0, n), self.row(Buf::DAtt, 0, n));
        // SAFETY: `n` rows of each, and a CPU that runs the kernels.
        unsafe { kernels::attention_bwd(dqkv, qkv, aw, datt, n) }
    }

    fn embed_bwd(&mut self, n: usize) {
        rows(0, n);
        assert!(self.tokens.len() >= n, "a token for each of {n} rows");
        let (g, dx0, tokens) = (self.at(Buf::Grads, 0, PARAMS), self.row(Buf::DX0, 0, n), self.tokens_at(0, n));
        // SAFETY: `n` rows of `DX0` and tokens, each token's row of `wte` and the rows of `wpe`
        // up to `n`, and a CPU that runs the kernels.
        unsafe { kernels::embed_bwd(g, dx0, tokens, n) }
    }

    fn adam(&mut self, a: &Adam) {
        let (p, g) = (self.at(Buf::Params, 0, PARAMS), self.at(Buf::Grads, 0, PARAMS));
        let (m, v) = (self.at(Buf::Moment1, 0, PARAMS), self.at(Buf::Moment2, 0, PARAMS));
        // SAFETY: `PARAMS` floats of each, `PARAMS / 8` vectors, and a CPU that runs the
        // kernels.
        unsafe { kernels::adam(p, g, m, v, PARAMS / 8, a.lr, a.b1, a.ob1, a.b2, a.ob2, a.bc1, a.bc2, a.eps) }
    }
}
