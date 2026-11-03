//! The model on the GPU: its buffers in the GPU's memory, and kernels in PTX over them. Each
//! computes what [`crate::reference`] does, in its order, so that the two agree bit for bit:
//! a sum in order is a chain in one thread, and the threads take apart only what is the same
//! for each element.

use kernite_cuda::{Buffer, Cuda, Error, Grid, Ptr};

use crate::model::{Adam, Buf, E, FC1, FC2, H, LM, Layer, M, Ops, PARAMS, QKV, T, Then, V, VP, WO, WTE};

/// The kernels. Pointers are to the first row of the buffers [`Buf`] names; a kernel works
/// out its rows from its block and thread, and trusts that they are there. They read the
/// model's sizes and `exp`'s constants as [`crate::model::consts`] has them.
#[kernite_macros::kernels(consts = crate::model::consts)]
pub mod kernels {
    /// What `matmul` does to each output before it stores it: nothing, adds the residual
    /// stream's, or keeps it only above zero.
    pub const STORE: usize = 0;
    #[allow(missing_docs)]
    pub const ADD: usize = 1;
    #[allow(missing_docs)]
    pub const RELU: usize = 2;

    /// `x0 = wte[token] + wpe[t]`, `p` the parameters: row `t0` and on a block each, an
    /// element a thread.
    pub unsafe extern "ptx-kernel" fn embed(x0: *mut f32, p: *const f32, tokens: *const u32, t0: u32) {
        let (t, e, tok, i): (u32, u32, u32, u32);
        mov.u32(t, ctaid.x);
        add.u32(t, t, t0);
        mov.u32(e, tid.x);
        let at: u64;
        mad.wide.u32(at, t, 4, tokens);
        ld.global.u32(tok, [at]);
        let (a, b): (f32, f32);
        mad.lo.u32(i, tok, E, e);
        load(a, p, i);
        mad.lo.u32(i, t, E, e);
        add.u32(i, i, WPE);
        load(b, p, i);
        add.rn.f32(a, a, b);
        mad.lo.u32(i, t, E, e);
        store(x0, i, a);
    }

    /// Each row of `x` divided by its root mean square, into `y`, and the scale it took into
    /// `r`: row `t0` and on a thread each.
    pub unsafe extern "ptx-kernel" fn rmsnorm(y: *mut f32, r: *mut f32, x: *const f32, t0: u32) {
        let t: u32;
        mov.u32(t, tid.x);
        add.u32(t, t, t0);
        let (xs, ys): (u64, u64);
        mad.wide.u32(xs, t, 4 * E, x);
        mad.wide.u32(ys, t, 4 * E, y);
        let v: [f32; E];
        let ss: f32;
        mov.f32(ss, 0.0);
        for e in 0..E {
            ld.global.f32(v[e], [xs + 4 * e]);
            fma.rn.f32(ss, v[e], v[e], ss);
        }
        mul.rn.f32(ss, ss, 0.0625);
        add.rn.f32(ss, ss, 1e-5);
        sqrt.rn.f32(ss, ss);
        let s: f32;
        div.rn.f32(s, 1.0, ss);
        store(r, t, s);
        for e in 0..E {
            mul.rn.f32(v[e], v[e], s);
            st.global.f32([ys + 4 * e], v[e]);
        }
    }

    /// `y = x W`, `W` the queries', keys' and values' weights side by side.
    pub unsafe extern "ptx-kernel" fn qkv(y: *mut f32, x: *const f32, w: *const f32, t0: u32) {
        matmul::<E, QKV_COLS, STORE>(y, x, w, y, t0);
    }

    /// `y = x W + res`, `W` attention's output projection.
    pub unsafe extern "ptx-kernel" fn wo(y: *mut f32, x: *const f32, w: *const f32, res: *const f32, t0: u32) {
        matmul::<E, E, ADD>(y, x, w, res, t0);
    }

    /// `y = relu(x W)`, `W` the MLP's first layer.
    pub unsafe extern "ptx-kernel" fn fc1(y: *mut f32, x: *const f32, w: *const f32, t0: u32) {
        matmul::<E, M, RELU>(y, x, w, y, t0);
    }

    /// `y = x W + res`, `W` the MLP's second layer.
    pub unsafe extern "ptx-kernel" fn fc2(y: *mut f32, x: *const f32, w: *const f32, res: *const f32, t0: u32) {
        matmul::<M, E, ADD>(y, x, w, res, t0);
    }

    /// `y = x W`, `W` the logits' head.
    pub unsafe extern "ptx-kernel" fn lm(y: *mut f32, x: *const f32, w: *const f32, t0: u32) {
        matmul::<E, VP, STORE>(y, x, w, y, t0);
    }

    /// `y = x W`, `W` of `ROWS` rows of `COLS`, then what `THEN` says: `res` is the residual
    /// stream when it adds it. Row `t0` and on a block each, an output a thread, each a chain
    /// over the inputs in order.
    fn matmul<const ROWS: usize, const COLS: usize, const THEN: usize>(y: u64, x: u64, w: u64, res: u64, t0: u32) {
        let (t, o, j): (u32, u32, u32);
        mov.u32(t, ctaid.x);
        add.u32(t, t, t0);
        mov.u32(o, tid.x);
        let (xs, ws): (u64, u64);
        mad.wide.u32(xs, t, 4 * ROWS, x);
        mad.wide.u32(ws, o, 4, w);
        let acc: f32;
        mov.f32(acc, 0.0);
        for i in 0..ROWS {
            let (xi, wi): (f32, f32);
            ld.global.f32(xi, [xs + 4 * i]);
            ld.global.f32(wi, [ws + 4 * i * COLS]);
            fma.rn.f32(acc, xi, wi, acc);
        }
        mad.lo.u32(j, t, COLS, o);
        if THEN == ADD {
            let r: f32;
            load(r, res, j);
            add.rn.f32(acc, acc, r);
        }
        if THEN == RELU {
            let above: pred;
            setp.gt.f32(above, acc, 0.0);
            selp.f32(acc, acc, 0.0, above);
        }
        store(y, j, acc);
    }

    /// Each head of row `t0` and on attending to the rows up to it, from `qkv`'s queries, keys
    /// and values: its weights into `aw`, and the values they weigh, summed, into `att`. A
    /// block each row, a thread each head.
    pub unsafe extern "ptx-kernel" fn attention(att: *mut f32, aw: *mut f32, qkv: *const f32, t0: u32) {
        let (t, h, i): (u32, u32, u32);
        mov.u32(t, ctaid.x);
        add.u32(t, t, t0);
        mov.u32(h, tid.x);
        let (qs, w0, ks, vs, ws): (u64, u64, u64, u64, u64);
        mul.lo.u32(i, h, D);
        mad.lo.u32(i, t, QKV_COLS, i);
        mad.wide.u32(qs, i, 4, qkv);
        let q: [f32; D];
        for j in 0..D {
            ld.global.f32(q[j], [qs + 4 * j]);
        }
        mad.lo.u32(i, t, H, h);
        mul.lo.u32(i, i, T);
        mad.wide.u32(w0, i, 4, aw);
        let (s, again): (u32, pred);

        note("the logits, and the largest");
        let m: f32;
        mov.f32(m, f32::NEG_INFINITY);
        mad.lo.u32(i, h, D, E);
        mad.wide.u32(ks, i, 4, qkv);
        mov.u64(ws, w0);
        mov.u32(s, 0);
        label(logits);
        let (l, above): (f32, pred);
        mov.f32(l, 0.0);
        for j in 0..D {
            let k: f32;
            ld.global.f32(k, [ks + 4 * j]);
            fma.rn.f32(l, q[j], k, l);
        }
        mul.rn.f32(l, l, 0.5);
        st.global.f32([ws], l);
        setp.gt.f32(above, l, m);
        if above {
            mov.f32(m, l);
        }
        add.u64(ks, ks, 4 * QKV_COLS);
        add.u64(ws, ws, 4);
        add.u32(s, s, 1);
        setp.le.u32(again, s, t);
        if again {
            bra(logits);
        }

        note("their exponentials, and the sum");
        let total: f32;
        mov.f32(total, 0.0);
        mov.u64(ws, w0);
        mov.u32(s, 0);
        label(exps);
        let x: f32;
        ld.global.f32(x, [ws]);
        sub.rn.f32(x, x, m);
        exp(x);
        st.global.f32([ws], x);
        add.rn.f32(total, total, x);
        add.u64(ws, ws, 4);
        add.u32(s, s, 1);
        setp.le.u32(again, s, t);
        if again {
            bra(exps);
        }

        note("the weights, and the values they weigh");
        let inv: f32;
        div.rn.f32(inv, 1.0, total);
        let out: [f32; D];
        for j in 0..D {
            mov.f32(out[j], 0.0);
        }
        mad.lo.u32(i, h, D, 2 * E);
        mad.wide.u32(vs, i, 4, qkv);
        mov.u64(ws, w0);
        mov.u32(s, 0);
        label(values);
        let ww: f32;
        ld.global.f32(ww, [ws]);
        mul.rn.f32(ww, ww, inv);
        st.global.f32([ws], ww);
        for j in 0..D {
            let v: f32;
            ld.global.f32(v, [vs + 4 * j]);
            fma.rn.f32(out[j], ww, v, out[j]);
        }
        add.u64(vs, vs, 4 * QKV_COLS);
        add.u64(ws, ws, 4);
        add.u32(s, s, 1);
        setp.le.u32(again, s, t);
        if again {
            bra(values);
        }
        let outs: u64;
        mul.lo.u32(i, h, D);
        mad.lo.u32(i, t, E, i);
        mad.wide.u32(outs, i, 4, att);
        for j in 0..D {
            st.global.f32([outs + 4 * j], out[j]);
        }
    }

    /// `probs = softmax(logits * inv_temp)` over the first `V` of each row, the rest zero: row
    /// `t0` and on a thread each.
    pub unsafe extern "ptx-kernel" fn softmax(probs: *mut f32, logits: *const f32, inv_temp: f32, t0: u32) {
        let t: u32;
        mov.u32(t, tid.x);
        add.u32(t, t, t0);
        let (ls, ps): (u64, u64);
        mad.wide.u32(ls, t, 4 * VP, logits);
        mad.wide.u32(ps, t, 4 * VP, probs);
        note("scaled, and the largest");
        let p: [f32; V];
        let m: f32;
        mov.f32(m, f32::NEG_INFINITY);
        for j in 0..V {
            let above: pred;
            ld.global.f32(p[j], [ls + 4 * j]);
            mul.rn.f32(p[j], p[j], inv_temp);
            setp.gt.f32(above, p[j], m);
            if above {
                mov.f32(m, p[j]);
            }
        }
        note("the exponentials, and their sum");
        let total: f32;
        mov.f32(total, 0.0);
        for j in 0..V {
            sub.rn.f32(p[j], p[j], m);
            exp(p[j]);
            add.rn.f32(total, total, p[j]);
        }
        note("each divided by it, past V zero");
        let (inv, z): (f32, f32);
        div.rn.f32(inv, 1.0, total);
        for j in 0..V {
            mul.rn.f32(p[j], p[j], inv);
            st.global.f32([ps + 4 * j], p[j]);
        }
        mov.f32(z, 0.0);
        for j in V..VP {
            st.global.f32([ps + 4 * j], z);
        }
    }

    /// `x = e^x`, for `x <= 0`, as `reference::exp` works it out: `k ln 2 + r`, `e^r` by its
    /// Taylor series, `2^k` by its bits.
    fn exp(x: f32) {
        let above: pred;
        setp.gt.f32(above, x, -87.0);
        if !above {
            mov.f32(x, -87.0);
        }
        let (k, nk, p, two): (f32, f32, f32, f32);
        mul.rn.f32(k, x, f32::consts::LOG2_E);
        cvt.rni.f32.f32(k, k);
        neg.f32(nk, k);
        fma.rn.f32(x, nk, LN2_HI, x);
        fma.rn.f32(x, nk, LN2_LO, x);
        mov.f32(p, TAYLOR[0]);
        for c in 1..7 {
            fma.rn.f32(p, p, x, TAYLOR[c]);
        }
        let bits: s32;
        cvt.rni.s32.f32(bits, k);
        add.s32(bits, bits, 127);
        shl.b32(bits, bits, 23);
        mov.b32(two, bits);
        mul.rn.f32(x, p, two);
    }

    /// `d = (probs - onehot(target)) * inv_n`, the targets at `targets`: a block each row, a
    /// thread each token.
    pub unsafe extern "ptx-kernel" fn xent(d: *mut f32, probs: *const f32, targets: *const u32, inv_n: f32) {
        let (t, j, i, target): (u32, u32, u32, u32);
        mov.u32(t, ctaid.x);
        mov.u32(j, tid.x);
        let at: u64;
        mad.wide.u32(at, t, 4, targets);
        ld.global.u32(target, [at]);
        let hit: pred;
        setp.eq.u32(hit, j, target);
        let (p, y): (f32, f32);
        selp.f32(y, 1.0, 0.0, hit);
        mad.lo.u32(i, t, VP, j);
        load(p, probs, i);
        sub.rn.f32(p, p, y);
        mul.rn.f32(p, p, inv_n);
        store(d, i, p);
    }

    /// `dx += dy Wᵀ` for `QKV`.
    pub unsafe extern "ptx-kernel" fn qkv_bwd_x(dx: *mut f32, dy: *const f32, w: *const f32) {
        bwd_x::<E, QKV_COLS>(dx, dy, w);
    }

    /// `dx += dy Wᵀ` for `WO`.
    pub unsafe extern "ptx-kernel" fn wo_bwd_x(dx: *mut f32, dy: *const f32, w: *const f32) {
        bwd_x::<E, E>(dx, dy, w);
    }

    /// `dx += dy Wᵀ` for `FC1`.
    pub unsafe extern "ptx-kernel" fn fc1_bwd_x(dx: *mut f32, dy: *const f32, w: *const f32) {
        bwd_x::<E, M>(dx, dy, w);
    }

    /// `dx += dy Wᵀ` for `FC2`.
    pub unsafe extern "ptx-kernel" fn fc2_bwd_x(dx: *mut f32, dy: *const f32, w: *const f32) {
        bwd_x::<M, E>(dx, dy, w);
    }

    /// `dx += dy Wᵀ` for `LM`.
    pub unsafe extern "ptx-kernel" fn lm_bwd_x(dx: *mut f32, dy: *const f32, w: *const f32) {
        bwd_x::<E, VP>(dx, dy, w);
    }

    /// `dx += dy Wᵀ`, `W` of `ROWS` rows of `COLS`: a block each row, a thread each input,
    /// whose gradient is a chain over the outputs in order, from where it was.
    fn bwd_x<const ROWS: usize, const COLS: usize>(dx: u64, dy: u64, w: u64) {
        let (t, i, at): (u32, u32, u32);
        mov.u32(t, ctaid.x);
        mov.u32(i, tid.x);
        mad.lo.u32(at, t, ROWS, i);
        let acc: f32;
        load(acc, dx, at);
        let (dys, ws): (u64, u64);
        mad.wide.u32(dys, t, 4 * COLS, dy);
        mad.wide.u32(ws, i, 4 * COLS, w);
        for o in 0..COLS {
            let (g, wi): (f32, f32);
            ld.global.f32(g, [dys + 4 * o]);
            ld.global.f32(wi, [ws + 4 * o]);
            fma.rn.f32(acc, g, wi, acc);
        }
        store(dx, at, acc);
    }

    /// `g += xᵀ dy` over `n` rows, for `QKV`.
    pub unsafe extern "ptx-kernel" fn qkv_bwd_w(g: *mut f32, x: *const f32, dy: *const f32, n: u32) {
        bwd_w::<E, QKV_COLS>(g, x, dy, n);
    }

    /// `g += xᵀ dy` over `n` rows, for `WO`.
    pub unsafe extern "ptx-kernel" fn wo_bwd_w(g: *mut f32, x: *const f32, dy: *const f32, n: u32) {
        bwd_w::<E, E>(g, x, dy, n);
    }

    /// `g += xᵀ dy` over `n` rows, for `FC1`.
    pub unsafe extern "ptx-kernel" fn fc1_bwd_w(g: *mut f32, x: *const f32, dy: *const f32, n: u32) {
        bwd_w::<E, M>(g, x, dy, n);
    }

    /// `g += xᵀ dy` over `n` rows, for `FC2`.
    pub unsafe extern "ptx-kernel" fn fc2_bwd_w(g: *mut f32, x: *const f32, dy: *const f32, n: u32) {
        bwd_w::<M, E>(g, x, dy, n);
    }

    /// `g += xᵀ dy` over `n` rows, for `LM`.
    pub unsafe extern "ptx-kernel" fn lm_bwd_w(g: *mut f32, x: *const f32, dy: *const f32, n: u32) {
        bwd_w::<E, VP>(g, x, dy, n);
    }

    /// `g += xᵀ dy` over `n` rows, `g` of `ROWS` rows of `COLS`: a block each input, a thread
    /// each output, each weight's gradient a chain over the rows in order, from where it was.
    fn bwd_w<const ROWS: usize, const COLS: usize>(g: u64, x: u64, dy: u64, n: u32) {
        let (i, o, at, t): (u32, u32, u32, u32);
        mov.u32(i, ctaid.x);
        mov.u32(o, tid.x);
        mad.lo.u32(at, i, COLS, o);
        let acc: f32;
        load(acc, g, at);
        let (xs, dys): (u64, u64);
        mad.wide.u32(xs, i, 4, x);
        mad.wide.u32(dys, o, 4, dy);
        let again: pred;
        setp.eq.u32(again, n, 0);
        if again {
            bra(done);
        }
        mov.u32(t, 0);
        label(row);
        let (xt, dyt): (f32, f32);
        ld.global.f32(xt, [xs]);
        ld.global.f32(dyt, [dys]);
        fma.rn.f32(acc, xt, dyt, acc);
        add.u64(xs, xs, 4 * ROWS);
        add.u64(dys, dys, 4 * COLS);
        add.u32(t, t, 1);
        setp.lt.u32(again, t, n);
        if again {
            bra(row);
        }
        label(done);
        store(g, at, acc);
    }

    /// `dh = 0` where `h` is not above zero: a block each row, a thread each element.
    pub unsafe extern "ptx-kernel" fn relu_bwd(dh: *mut f32, h: *const f32) {
        let (t, j, i): (u32, u32, u32);
        mov.u32(t, ctaid.x);
        mov.u32(j, tid.x);
        mad.lo.u32(i, t, M, j);
        let (x, z): (f32, f32);
        load(x, h, i);
        let above: pred;
        setp.gt.f32(above, x, 0.0);
        let at: u64;
        mad.wide.u32(at, i, 4, dh);
        mov.f32(z, 0.0);
        if !above {
            st.global.f32([at], z);
        }
    }

    /// `dx +=` the gradient by `x` of normalizing it, given the gradient `dy` by the result and
    /// the scale `r` each row took: a thread each row.
    pub unsafe extern "ptx-kernel" fn rmsnorm_bwd(dx: *mut f32, dy: *const f32, x: *const f32, r: *const f32) {
        let t: u32;
        mov.u32(t, tid.x);
        let (dxs, dys, xs): (u64, u64, u64);
        mad.wide.u32(dxs, t, 4 * E, dx);
        mad.wide.u32(dys, t, 4 * E, dy);
        mad.wide.u32(xs, t, 4 * E, x);
        let g: [f32; E];
        let xe: [f32; E];
        let dot: f32;
        mov.f32(dot, 0.0);
        for e in 0..E {
            ld.global.f32(g[e], [dys + 4 * e]);
            ld.global.f32(xe[e], [xs + 4 * e]);
            fma.rn.f32(dot, g[e], xe[e], dot);
        }
        note("c = dot r^3 / E");
        let (s, c, nc): (f32, f32, f32);
        load(s, r, t);
        mul.rn.f32(c, s, s);
        mul.rn.f32(c, c, s);
        mul.rn.f32(c, dot, c);
        mul.rn.f32(c, c, 0.0625);
        neg.f32(nc, c);
        note("dx += dy r - c x");
        for e in 0..E {
            let d: f32;
            mul.rn.f32(g[e], g[e], s);
            fma.rn.f32(g[e], nc, xe[e], g[e]);
            ld.global.f32(d, [dxs + 4 * e]);
            add.rn.f32(d, d, g[e]);
            st.global.f32([dxs + 4 * e], d);
        }
    }

    /// `dqkv` from `datt`, attention's gradient, over rows `0..n`: each row's query's, and the
    /// keys and values it attended to added to, a row at a time in order. A thread each head.
    pub unsafe extern "ptx-kernel" fn attention_bwd(
        dqkv: *mut f32,
        qkv: *const f32,
        aw: *const f32,
        datt: *const f32,
        n: u32,
    ) {
        let (h, t, s, i): (u32, u32, u32, u32);
        mov.u32(h, tid.x);
        let again: pred;
        setp.eq.u32(again, n, 0);
        if again {
            ret();
        }
        mov.u32(t, 0);
        label(row);
        note("the query, its weights, and the output's gradient");
        let (qs, w0, ds): (u64, u64, u64);
        mul.lo.u32(i, h, D);
        mad.lo.u32(i, t, QKV_COLS, i);
        mad.wide.u32(qs, i, 4, qkv);
        mad.lo.u32(i, t, H, h);
        mul.lo.u32(i, i, T);
        mad.wide.u32(w0, i, 4, aw);
        mul.lo.u32(i, h, D);
        mad.lo.u32(i, t, E, i);
        mad.wide.u32(ds, i, 4, datt);
        let q: [f32; D];
        let dout: [f32; D];
        for j in 0..D {
            ld.global.f32(q[j], [qs + 4 * j]);
            ld.global.f32(dout[j], [ds + 4 * j]);
        }

        note("the sum of the weights by their gradients");
        let (ks, vs, dks, ws): (u64, u64, u64, u64);
        let (sum, w, dw): (f32, f32, f32);
        mov.f32(sum, 0.0);
        mad.lo.u32(i, h, D, 2 * E);
        mad.wide.u32(vs, i, 4, qkv);
        mov.u64(ws, w0);
        mov.u32(s, 0);
        label(sums);
        weight_grad(dw, dout, vs);
        ld.global.f32(w, [ws]);
        fma.rn.f32(sum, w, dw, sum);
        add.u64(vs, vs, 4 * QKV_COLS);
        add.u64(ws, ws, 4);
        add.u32(s, s, 1);
        setp.le.u32(again, s, t);
        if again {
            bra(sums);
        }

        note("the query's gradient, and the keys' and values'");
        let dq: [f32; D];
        for j in 0..D {
            mov.f32(dq[j], 0.0);
        }
        mad.lo.u32(i, h, D, E);
        mad.wide.u32(ks, i, 4, qkv);
        mad.wide.u32(dks, i, 4, dqkv);
        mad.lo.u32(i, h, D, 2 * E);
        mad.wide.u32(vs, i, 4, qkv);
        mov.u64(ws, w0);
        mov.u32(s, 0);
        label(grads);
        let g: f32;
        weight_grad(dw, dout, vs);
        sub.rn.f32(dw, dw, sum);
        ld.global.f32(w, [ws]);
        mul.rn.f32(g, w, dw);
        mul.rn.f32(g, g, 0.5);
        for j in 0..D {
            let (k, dk, dv): (f32, f32, f32);
            ld.global.f32(k, [ks + 4 * j]);
            fma.rn.f32(dq[j], g, k, dq[j]);
            ld.global.f32(dk, [dks + 4 * j]);
            fma.rn.f32(dk, g, q[j], dk);
            st.global.f32([dks + 4 * j], dk);
            ld.global.f32(dv, [dks + 4 * (E + j)]);
            fma.rn.f32(dv, w, dout[j], dv);
            st.global.f32([dks + 4 * (E + j)], dv);
        }
        add.u64(ks, ks, 4 * QKV_COLS);
        add.u64(vs, vs, 4 * QKV_COLS);
        add.u64(dks, dks, 4 * QKV_COLS);
        add.u64(ws, ws, 4);
        add.u32(s, s, 1);
        setp.le.u32(again, s, t);
        if again {
            bra(grads);
        }
        let dqs: u64;
        mul.lo.u32(i, h, D);
        mad.lo.u32(i, t, QKV_COLS, i);
        mad.wide.u32(dqs, i, 4, dqkv);
        for j in 0..D {
            st.global.f32([dqs + 4 * j], dq[j]);
        }
        add.u32(t, t, 1);
        setp.lt.u32(again, t, n);
        if again {
            bra(row);
        }
    }

    /// `dw`, the gradient by one weight: the output's gradient `dout` by the value at `v`, a
    /// chain in order.
    fn weight_grad(dw: f32, dout: [f32; D], v: u64) {
        mov.f32(dw, 0.0);
        for j in 0..D {
            let x: f32;
            ld.global.f32(x, [v + 4 * j]);
            fma.rn.f32(dw, dout[j], x, dw);
        }
    }

    /// The embeddings' gradients, `g` the parameters': `wte[token] += dx0` and `wpe[t] +=
    /// dx0` for rows `0..n` in order, a thread each element of a row.
    pub unsafe extern "ptx-kernel" fn embed_bwd(g: *mut f32, dx0: *const f32, tokens: *const u32, n: u32) {
        let (e, t, tok, i): (u32, u32, u32, u32);
        mov.u32(e, tid.x);
        let again: pred;
        setp.eq.u32(again, n, 0);
        if again {
            ret();
        }
        mov.u32(t, 0);
        label(row);
        let at: u64;
        mad.wide.u32(at, t, 4, tokens);
        ld.global.u32(tok, [at]);
        let (d, x): (f32, f32);
        mad.lo.u32(i, t, E, e);
        load(d, dx0, i);
        mad.lo.u32(i, tok, E, e);
        load(x, g, i);
        add.rn.f32(x, x, d);
        store(g, i, x);
        mad.lo.u32(i, t, E, e);
        add.u32(i, i, WPE);
        load(x, g, i);
        add.rn.f32(x, x, d);
        store(g, i, x);
        add.u32(t, t, 1);
        setp.lt.u32(again, t, n);
        if again {
            bra(row);
        }
    }

    /// Adam's step on the `n` parameters `p`, by their gradients `g`, which it zeroes, and
    /// their moments `m` and `v`: a thread each.
    pub unsafe extern "ptx-kernel" fn adam(
        p: *mut f32,
        g: *mut f32,
        m: *mut f32,
        v: *mut f32,
        n: u32,
        lr: f32,
        b1: f32,
        ob1: f32,
        b2: f32,
        ob2: f32,
        bc1: f32,
        bc2: f32,
        eps: f32,
    ) {
        let (i, block, size): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(i, tid.x);
        mad.lo.u32(i, block, size, i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let (gi, mi, vi, pi, x): (f32, f32, f32, f32, f32);
        load(gi, g, i);
        load(mi, m, i);
        load(vi, v, i);
        note("m = b1 m + (1 - b1) g; v = b2 v + (1 - b2) g^2");
        mul.rn.f32(x, mi, b1);
        fma.rn.f32(mi, ob1, gi, x);
        store(m, i, mi);
        mul.rn.f32(gi, gi, gi);
        mul.rn.f32(x, vi, b2);
        fma.rn.f32(vi, ob2, gi, x);
        store(v, i, vi);
        note("p -= lr m^ / (sqrt(v^) + eps)");
        mul.rn.f32(mi, mi, bc1);
        mul.rn.f32(vi, vi, bc2);
        sqrt.rn.f32(vi, vi);
        add.rn.f32(vi, vi, eps);
        mul.rn.f32(mi, lr, mi);
        div.rn.f32(mi, mi, vi);
        load(pi, p, i);
        sub.rn.f32(pi, pi, mi);
        store(p, i, pi);
        mov.f32(x, 0.0);
        store(g, i, x);
    }

    /// `x = base[i]`.
    fn load(x: f32, base: u64, i: u32) {
        let at: u64;
        mad.wide.u32(at, i, 4, base);
        ld.global.f32(x, [at]);
    }

    /// `base[i] = x`.
    fn store(base: u64, i: u32, x: f32) {
        let at: u64;
        mad.wide.u32(at, i, 4, base);
        st.global.f32([at], x);
    }
}

/// The kernels' sizes and constants are the model's.
const _: () = assert!(WTE == 0, "embed and embed_bwd find a token's row as its offset from the parameters");

/// The model's buffers on the GPU, and the kernels over them.
pub struct Gpu {
    kernels: kernels::Kernels,
    bufs: Vec<Buffer<f32>>,
    /// The document's tokens, `T + 1` whatever its length, and how many of them are its.
    tokens: Buffer<u32>,
    count: usize,
}

impl Gpu {
    /// A model of `params` on the first GPU, Adam's moments zero; or why there is none to run
    /// the kernels.
    pub fn new(params: &[f32]) -> Result<Gpu, Error> {
        let cuda = Cuda::new()?;
        let kernels = kernels::Kernels::load(&cuda)?;
        let bufs = Buf::ALL.iter().map(|b| cuda.alloc(b.len())).collect::<Result<Vec<_>, _>>()?;
        let mut gpu = Gpu { kernels, bufs, tokens: cuda.alloc(T + 1)?, count: 0 };
        gpu.bufs[Buf::Params as usize].write(params)?;
        Ok(gpu)
    }

    /// `b`, from its first row: a kernel finds its own rows.
    fn at(&self, b: Buf) -> Ptr<f32> {
        self.bufs[b as usize].ptr()
    }

    /// The weights of `l` in `b`: the parameters, or their gradients.
    fn weights(&self, b: Buf, l: Layer) -> Ptr<f32> {
        self.bufs[b as usize].at(l.at)
    }
}

/// What the driver says, once it says a call failed: there is no going on without the GPU.
fn check<R>(r: Result<R, Error>) -> R {
    r.unwrap_or_else(|e| panic!("the GPU failed: {e}"))
}

/// `t0..n` as rows of a document: within it, `T` at most. How many, for a grid.
fn rows(t0: usize, n: usize) -> u32 {
    assert!(t0 <= n && n <= T, "rows {t0}..{n} of a document of {T}");
    (n - t0) as u32
}

/// Launches `kernel` over `blocks` blocks of `threads` threads, if that is any: a grid of none
/// cannot be launched, and would have nothing to do.
fn launch(blocks: u32, threads: u32, kernel: impl FnOnce(Grid) -> Result<(), Error>) {
    if blocks > 0 && threads > 0 {
        check(kernel(Grid::new(blocks, threads)));
    }
}

impl Ops for Gpu {
    fn write(&mut self, b: Buf, data: &[f32]) {
        check(self.bufs[b as usize].write(data));
    }

    fn read(&mut self, b: Buf) -> Vec<f32> {
        check(self.bufs[b as usize].read())
    }

    fn tokens(&mut self, tokens: &[u32]) {
        assert!(tokens.len() <= T + 1 && tokens.iter().all(|&t| (t as usize) < V), "tokens of a document");
        let mut all = [0; T + 1];
        all[..tokens.len()].copy_from_slice(tokens);
        check(self.tokens.write(&all));
        self.count = tokens.len();
    }

    fn embed(&mut self, t0: usize, n: usize) {
        assert!(self.count >= n, "a token for each of {n} rows");
        let (ks, x0, p, tokens) = (&self.kernels, self.at(Buf::X0), self.at(Buf::Params), self.tokens.ptr());
        // SAFETY: a block for each of the rows `t0..n` of `X0` and of the tokens, a thread for
        // each element; each token's row of `wte`, and the rows of `wpe` up to `n`.
        launch(rows(t0, n), E as u32, |grid| unsafe { ks.embed(grid, x0, p, tokens, t0 as u32) });
    }

    fn rmsnorm(&mut self, y: Buf, r: Buf, x: Buf, t0: usize, n: usize) {
        let (ks, y, r, x) = (&self.kernels, self.at(y), self.at(r), self.at(x));
        // SAFETY: a thread for each of the rows `t0..n` of each.
        launch(1, rows(t0, n), |grid| unsafe { ks.rmsnorm(grid, y, r, x, t0 as u32) });
    }

    fn matmul(&mut self, l: Layer, y: Buf, x: Buf, then: Then, t0: usize, n: usize) {
        assert_eq!((x.width(), y.width()), (l.rows, l.cols), "{x:?} by {l:?} into {y:?}");
        let (ks, w, out, x, t0) = (&self.kernels, self.weights(Buf::Params, l), self.at(y), self.at(x), t0 as u32);
        let res = match then {
            Then::Add(res) => self.at(res),
            _ => out,
        };
        // SAFETY: a block for each of the rows `t0..n` of each, as wide as the layer says, a
        // thread for each output; and its weights.
        launch(rows(t0 as usize, n), l.cols as u32, |grid| unsafe {
            match (l, then) {
                (QKV, Then::Store) => ks.qkv(grid, out, x, w, t0),
                (WO, Then::Add(_)) => ks.wo(grid, out, x, w, res, t0),
                (FC1, Then::Relu) => ks.fc1(grid, out, x, w, t0),
                (FC2, Then::Add(_)) => ks.fc2(grid, out, x, w, res, t0),
                (LM, Then::Store) => ks.lm(grid, out, x, w, t0),
                _ => panic!("no kernel computes {l:?} then {then:?}"),
            }
        });
    }

    fn attention(&mut self, t0: usize, n: usize) {
        let (ks, att, aw, qkv) = (&self.kernels, self.at(Buf::Att), self.at(Buf::Aw), self.at(Buf::Qkv));
        // SAFETY: a block for each of the rows `t0..n`, a thread for each head, which reads the
        // rows up to its own.
        launch(rows(t0, n), H as u32, |grid| unsafe { ks.attention(grid, att, aw, qkv, t0 as u32) });
    }

    fn softmax(&mut self, inv_temp: f32, t0: usize, n: usize) {
        let (ks, probs, logits) = (&self.kernels, self.at(Buf::Probs), self.at(Buf::Logits));
        // SAFETY: a thread for each of the rows `t0..n` of each.
        launch(1, rows(t0, n), |grid| unsafe { ks.softmax(grid, probs, logits, inv_temp, t0 as u32) });
    }

    fn clear(&mut self) {
        for b in Buf::SUMS {
            check(self.bufs[b as usize].zero());
        }
    }

    fn xent(&mut self, n: usize) {
        assert!(self.count > n, "a target for each of {n} rows");
        let (ks, d, probs, targets) = (&self.kernels, self.at(Buf::DLogits), self.at(Buf::Probs), self.tokens.at(1));
        // SAFETY: a block for each of `n` rows of each and of the targets, the tokens after the
        // first, and a thread for each token.
        launch(rows(0, n), VP as u32, |grid| unsafe { ks.xent(grid, d, probs, targets, 1.0 / n as f32) });
    }

    fn bwd_x(&mut self, l: Layer, dx: Buf, dy: Buf, n: usize) {
        assert_eq!((dx.width(), dy.width()), (l.rows, l.cols), "{dy:?} by {l:?} into {dx:?}");
        let (ks, w, dx, dy) = (&self.kernels, self.weights(Buf::Params, l), self.at(dx), self.at(dy));
        // SAFETY: a block for each of `n` rows of each, as wide as the layer says, a thread for
        // each input; and its weights.
        launch(rows(0, n), l.rows as u32, |grid| unsafe {
            match l {
                QKV => ks.qkv_bwd_x(grid, dx, dy, w),
                WO => ks.wo_bwd_x(grid, dx, dy, w),
                FC1 => ks.fc1_bwd_x(grid, dx, dy, w),
                FC2 => ks.fc2_bwd_x(grid, dx, dy, w),
                LM => ks.lm_bwd_x(grid, dx, dy, w),
                _ => panic!("no kernel for {l:?}"),
            }
        });
    }

    fn bwd_w(&mut self, l: Layer, x: Buf, dy: Buf, n: usize) {
        assert_eq!((x.width(), dy.width()), (l.rows, l.cols), "{x:?} by {dy:?} for {l:?}");
        let (ks, g, x, dy, n) = (&self.kernels, self.weights(Buf::Grads, l), self.at(x), self.at(dy), rows(0, n));
        // SAFETY: a block for each input and a thread for each output of the layer's gradients,
        // over `n` rows of each, as wide as the layer says.
        launch(l.rows as u32, l.cols as u32, |grid| unsafe {
            match l {
                QKV => ks.qkv_bwd_w(grid, g, x, dy, n),
                WO => ks.wo_bwd_w(grid, g, x, dy, n),
                FC1 => ks.fc1_bwd_w(grid, g, x, dy, n),
                FC2 => ks.fc2_bwd_w(grid, g, x, dy, n),
                LM => ks.lm_bwd_w(grid, g, x, dy, n),
                _ => panic!("no kernel for {l:?}"),
            }
        });
    }

    fn relu_bwd(&mut self, n: usize) {
        let (ks, dh, h) = (&self.kernels, self.at(Buf::DHid), self.at(Buf::Hid));
        // SAFETY: a block for each of `n` rows of each, a thread for each element.
        launch(rows(0, n), M as u32, |grid| unsafe { ks.relu_bwd(grid, dh, h) });
    }

    fn rmsnorm_bwd(&mut self, dx: Buf, dy: Buf, x: Buf, r: Buf, n: usize) {
        let (ks, dx, dy, x, r) = (&self.kernels, self.at(dx), self.at(dy), self.at(x), self.at(r));
        // SAFETY: a thread for each of `n` rows of each.
        launch(1, rows(0, n), |grid| unsafe { ks.rmsnorm_bwd(grid, dx, dy, x, r) });
    }

    fn attention_bwd(&mut self, n: usize) {
        let (ks, dqkv, n) = (&self.kernels, self.at(Buf::DQkv), rows(0, n));
        let (qkv, aw, datt) = (self.at(Buf::Qkv), self.at(Buf::Aw), self.at(Buf::DAtt));
        // SAFETY: a thread for each head, over `n` rows of each.
        launch(1, H as u32, |grid| unsafe { ks.attention_bwd(grid, dqkv, qkv, aw, datt, n) });
    }

    fn embed_bwd(&mut self, n: usize) {
        assert!(self.count >= n, "a token for each of {n} rows");
        let (ks, g, dx0, tokens, n) =
            (&self.kernels, self.at(Buf::Grads), self.at(Buf::DX0), self.tokens.ptr(), rows(0, n));
        // SAFETY: a thread for each element of a row, over `n` rows of `DX0` and of the tokens;
        // each token's row of `wte`, and the rows of `wpe` up to `n`.
        launch(1, E as u32, |grid| unsafe { ks.embed_bwd(grid, g, dx0, tokens, n) });
    }

    fn adam(&mut self, a: &Adam) {
        let (ks, p, g) = (&self.kernels, self.at(Buf::Params), self.at(Buf::Grads));
        let (m, v, n) = (self.at(Buf::Moment1), self.at(Buf::Moment2), PARAMS as u32);
        // SAFETY: a thread for each of the `PARAMS` floats of each, the ones past them doing
        // nothing.
        launch(n.div_ceil(256), 256, |grid| unsafe {
            ks.adam(grid, p, g, m, v, n, a.lr, a.b1, a.ob1, a.b2, a.ob2, a.bc1, a.bc2, a.eps)
        });
    }
}
