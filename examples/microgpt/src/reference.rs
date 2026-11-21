//! The model's operations in plain Rust: what each kernel computes, operation for operation
//! and rounding for rounding, so that the kernels can be held to it bit for bit.
//!
//! Every sum is a chain from zero in the order written, each product added by a fused
//! multiply-add, which rounds once. `e^x` is [`exp`], a polynomial the kernels evaluate the
//! same way, where a library's would differ by machine.

use crate::model::{
    Adam, Buf, D, E, H, LN2_HI, LN2_LO, Layer, M, Op, Ops, PARAMS, QKV_COLS, T, TAYLOR, Then, V, VP, WPE, WTE, rows,
    span, span_mut,
};

/// e^x for `x <= 0`, as every backend works it out: x is cut to `k ln 2 + r`, `k` the nearest
/// integer to `x / ln 2` and `|r| <= ln 2 / 2`; e^r by its Taylor series; and 2^k by its bits.
/// Below -87 the result would be subnormal, and is taken at -87.
pub fn exp(x: f32) -> f32 {
    let r = if x > -87.0 { x } else { -87.0 };
    let k = (r * std::f32::consts::LOG2_E).round_ties_even();
    let r = (-k).mul_add(LN2_HI, r);
    let r = (-k).mul_add(LN2_LO, r);
    let p = TAYLOR[1..].iter().fold(TAYLOR[0], |p, &c| p.mul_add(r, c));
    p * f32::from_bits(((k as i32 + 127) as u32) << 23)
}

/// The model's buffers as vectors, and the operations on them.
pub struct Reference {
    bufs: Vec<Vec<f32>>,
    tokens: Vec<u32>,
}

impl Reference {
    /// A model of `params`, Adam's moments zero; or why `params` are not a model's.
    pub fn new(params: &[f32]) -> Op<Reference> {
        let mut r = Reference { bufs: Buf::ALL.iter().map(|b| vec![0.0; b.len()]).collect(), tokens: vec![] };
        r.write(Buf::Params, params)?;
        Ok(r)
    }
}

/// Two of `bufs` at once, `a` to read and `b` to write; or why they are one.
fn pair(bufs: &mut [Vec<f32>], a: Buf, b: Buf) -> Op<(&[f32], &mut [f32])> {
    if a == b {
        return Err(format!("{a:?} is read and written at once"));
    }
    let [from, to] = bufs.get_disjoint_mut([a as usize, b as usize]).map_err(|e| format!("{a:?} and {b:?}: {e}"))?;
    Ok((from.as_slice(), to.as_mut_slice()))
}

/// The rows of `l`'s weights among `params`: the outputs of each input.
fn weights(params: &[f32], l: Layer) -> Op<Vec<&[f32]>> {
    (0..l.rows).map(|i| span(params, l.at + i * l.cols, l.cols)).collect()
}

impl Ops for Reference {
    fn write(&mut self, b: Buf, data: &[f32]) -> Op {
        b.store(b.of_mut(&mut self.bufs)?, data)
    }

    fn read(&mut self, b: Buf) -> Op<Vec<f32>> {
        Ok(b.of(&self.bufs)?.to_vec())
    }

    fn tokens(&mut self, tokens: &[u32]) -> Op {
        self.tokens = tokens.to_vec();
        Ok(())
    }

    fn embed(&mut self, t0: usize, n: usize) -> Op {
        let k = rows(t0, n)?;
        if self.tokens.len() < n {
            return Err(format!("a token for each of {n} rows"));
        }
        let (p, x) = pair(&mut self.bufs, Buf::Params, Buf::X0)?;
        for ((t, &token), x) in self.tokens.iter().enumerate().zip(x.as_chunks_mut::<E>().0.iter_mut()).skip(t0).take(k)
        {
            let (wte, wpe) = (span(p, WTE + token as usize * E, E)?, span(p, WPE + t * E, E)?);
            for ((x, a), b) in x.iter_mut().zip(wte).zip(wpe) {
                *x = a + b;
            }
        }
        Ok(())
    }

    fn rmsnorm(&mut self, y: Buf, r: Buf, x: Buf, t0: usize, n: usize) -> Op {
        let k = rows(t0, n)?;
        let (x, mut scales) = (self.read(x)?, self.read(r)?);
        let out = y.of_mut(&mut self.bufs)?;
        for ((row, out), scale) in
            x.as_chunks::<E>().0.iter().zip(out.as_chunks_mut::<E>().0.iter_mut()).zip(&mut scales).skip(t0).take(k)
        {
            let ss = row.iter().fold(0.0f32, |ss, &v| v.mul_add(v, ss));
            let ms = ss * 0.0625;
            let s = 1.0 / (ms + 1e-5).sqrt();
            *scale = s;
            for (out, &v) in out.iter_mut().zip(row) {
                *out = v * s;
            }
        }
        self.write(r, &scales)
    }

    fn matmul(&mut self, l: Layer, y: Buf, x: Buf, then: Then, t0: usize, n: usize) -> Op {
        rows(t0, n)?;
        let params = self.read(Buf::Params)?;
        let w = weights(&params, l)?;
        let x = self.read(x)?;
        let res = match then {
            Then::Add(b) => self.read(b)?,
            _ => vec![],
        };
        let out = y.of_mut(&mut self.bufs)?;
        for t in t0..n {
            let out = span_mut(out, t * l.cols, l.cols)?;
            out.fill(0.0);
            for (&xi, w_row) in span(&x, t * l.rows, l.rows)?.iter().zip(&w) {
                for (acc, &wio) in out.iter_mut().zip(*w_row) {
                    *acc = xi.mul_add(wio, *acc);
                }
            }
            match then {
                Then::Store => {}
                Then::Add(_) => {
                    for (acc, &r) in out.iter_mut().zip(span(&res, t * l.cols, l.cols)?) {
                        *acc += r;
                    }
                }
                Then::Relu => {
                    for acc in out.iter_mut() {
                        *acc = if *acc > 0.0 { *acc } else { 0.0 };
                    }
                }
            }
        }
        Ok(())
    }

    fn attention(&mut self, t0: usize, n: usize) -> Op {
        rows(t0, n)?;
        let qkv = self.read(Buf::Qkv)?;
        let (mut aw, mut att) = (self.read(Buf::Aw)?, self.read(Buf::Att)?);
        // The `D` floats of head `h` of position `s`'s query, key or value: part 0, 1 or 2.
        let head = |s: usize, part: usize, h: usize| span(&qkv, s * QKV_COLS + part * E + h * D, D);
        for t in t0..n {
            for h in 0..H {
                let q = head(t, 0, h)?;
                let w = span_mut(&mut aw, (t * H + h) * T, T)?;
                let mut m = f32::NEG_INFINITY;
                for (s, ws) in w.iter_mut().enumerate().take(t + 1) {
                    let dot = q.iter().zip(head(s, 1, h)?).fold(0.0f32, |acc, (&q, &k)| q.mul_add(k, acc));
                    let l = dot * 0.5;
                    *ws = l;
                    if l > m {
                        m = l;
                    }
                }
                let mut total = 0.0f32;
                for ws in w.iter_mut().take(t + 1) {
                    *ws = exp(*ws - m);
                    total += *ws;
                }
                let inv = 1.0 / total;
                let out = span_mut(&mut att, t * E + h * D, D)?;
                out.fill(0.0);
                for (s, ws) in w.iter_mut().enumerate().take(t + 1) {
                    *ws *= inv;
                    for (o, &v) in out.iter_mut().zip(head(s, 2, h)?) {
                        *o = ws.mul_add(v, *o);
                    }
                }
            }
        }
        self.write(Buf::Aw, &aw)?;
        self.write(Buf::Att, &att)
    }

    fn softmax(&mut self, inv_temp: f32, t0: usize, n: usize) -> Op {
        let k = rows(t0, n)?;
        let (logits, probs) = pair(&mut self.bufs, Buf::Logits, Buf::Probs)?;
        for (l, p) in logits.as_chunks::<VP>().0.iter().zip(probs.as_chunks_mut::<VP>().0.iter_mut()).skip(t0).take(k) {
            let mut m = f32::NEG_INFINITY;
            for (pj, &lj) in p.iter_mut().zip(l).take(V) {
                *pj = lj * inv_temp;
                if *pj > m {
                    m = *pj;
                }
            }
            let mut total = 0.0f32;
            for pj in p.iter_mut().take(V) {
                *pj = exp(*pj - m);
                total += *pj;
            }
            let inv = 1.0 / total;
            for pj in p.iter_mut().take(V) {
                *pj *= inv;
            }
            for pj in p.iter_mut().skip(V) {
                *pj = 0.0;
            }
        }
        Ok(())
    }

    fn clear(&mut self) -> Op {
        for b in Buf::SUMS {
            b.of_mut(&mut self.bufs)?.fill(0.0);
        }
        Ok(())
    }

    fn xent(&mut self, n: usize) -> Op {
        rows(0, n)?;
        let inv_n = 1.0 / n as f32;
        let targets = self.tokens.get(1..=n).ok_or_else(|| format!("a target for each of {n} rows"))?;
        let (probs, d) = pair(&mut self.bufs, Buf::Probs, Buf::DLogits)?;
        for ((&target, probs), d) in
            targets.iter().zip(probs.as_chunks::<VP>().0.iter()).zip(d.as_chunks_mut::<VP>().0.iter_mut())
        {
            for (j, (d, &p)) in d.iter_mut().zip(probs).enumerate() {
                let y = if j == target as usize { 1.0 } else { 0.0 };
                *d = (p - y) * inv_n;
            }
        }
        Ok(())
    }

    fn bwd_x(&mut self, l: Layer, dx: Buf, dy: Buf, n: usize) -> Op {
        rows(0, n)?;
        let params = self.read(Buf::Params)?;
        let w = weights(&params, l)?;
        let (dy, dx) = pair(&mut self.bufs, dy, dx)?;
        for t in 0..n {
            let dy = span(dy, t * l.cols, l.cols)?;
            for (dx, w_row) in span_mut(dx, t * l.rows, l.rows)?.iter_mut().zip(&w) {
                *dx = dy.iter().zip(*w_row).fold(*dx, |acc, (&dy, &w)| dy.mul_add(w, acc));
            }
        }
        Ok(())
    }

    fn bwd_w(&mut self, l: Layer, x: Buf, dy: Buf, n: usize) -> Op {
        rows(0, n)?;
        let (x, dy) = (self.read(x)?, self.read(dy)?);
        let g = span_mut(Buf::Grads.of_mut(&mut self.bufs)?, l.at, l.rows * l.cols)?;
        for t in 0..n {
            let dy = span(&dy, t * l.cols, l.cols)?;
            for (i, &xi) in span(&x, t * l.rows, l.rows)?.iter().enumerate() {
                for (g, &dy) in span_mut(g, i * l.cols, l.cols)?.iter_mut().zip(dy) {
                    *g = xi.mul_add(dy, *g);
                }
            }
        }
        Ok(())
    }

    fn relu_bwd(&mut self, n: usize) -> Op {
        rows(0, n)?;
        let (h, dh) = pair(&mut self.bufs, Buf::Hid, Buf::DHid)?;
        for (d, &x) in dh.iter_mut().zip(h).take(n * M) {
            // Not `x <= 0.0`: the ReLU passed no NaN either.
            *d = if x > 0.0 { *d } else { 0.0 };
        }
        Ok(())
    }

    fn rmsnorm_bwd(&mut self, dx: Buf, dy: Buf, x: Buf, r: Buf, n: usize) -> Op {
        rows(0, n)?;
        let (dy, x, r) = (self.read(dy)?, self.read(x)?, self.read(r)?);
        let dx = dx.of_mut(&mut self.bufs)?;
        let by_row = dy
            .as_chunks::<E>()
            .0
            .iter()
            .zip(x.as_chunks::<E>().0.iter())
            .zip(&r)
            .zip(dx.as_chunks_mut::<E>().0.iter_mut());
        for (((dy, x), &r), dx) in by_row.take(n) {
            let dot = dy.iter().zip(x).fold(0.0f32, |acc, (&dy, &x)| dy.mul_add(x, acc));
            let r2 = r * r;
            let r3 = r2 * r;
            let c = dot * r3;
            let c = c * 0.0625;
            for ((dx, &dy), &x) in dx.iter_mut().zip(dy).zip(x) {
                let g = dy * r;
                let g = (-c).mul_add(x, g);
                *dx += g;
            }
        }
        Ok(())
    }

    fn attention_bwd(&mut self, n: usize) -> Op {
        rows(0, n)?;
        let (qkv, aw, datt) = (self.read(Buf::Qkv)?, self.read(Buf::Aw)?, self.read(Buf::DAtt)?);
        let dqkv = Buf::DQkv.of_mut(&mut self.bufs)?;
        // The `D` floats of head `h` of position `s`'s query, key or value: part 0, 1 or 2.
        let head = |s: usize, part: usize, h: usize| span(&qkv, s * QKV_COLS + part * E + h * D, D);
        for h in 0..H {
            for t in 0..n {
                let q = head(t, 0, h)?;
                let w = span(&aw, (t * H + h) * T, T)?;
                let dout = span(&datt, t * E + h * D, D)?;
                // The gradient by each weight: the output's gradient by its value.
                let dw = |s: usize| -> Op<f32> {
                    Ok(dout.iter().zip(head(s, 2, h)?).fold(0.0f32, |acc, (&d, &v)| d.mul_add(v, acc)))
                };
                let mut sum = 0.0f32;
                for (s, &ws) in w.iter().enumerate().take(t + 1) {
                    sum = ws.mul_add(dw(s)?, sum);
                }
                let mut dq = [0.0f32; D];
                for (s, &ws) in w.iter().enumerate().take(t + 1) {
                    let dl = ws * (dw(s)? - sum);
                    let g = dl * 0.5;
                    for (dq, &k) in dq.iter_mut().zip(head(s, 1, h)?) {
                        *dq = g.mul_add(k, *dq);
                    }
                    for (dk, &q) in span_mut(dqkv, s * QKV_COLS + E + h * D, D)?.iter_mut().zip(q) {
                        *dk = g.mul_add(q, *dk);
                    }
                    for (dv, &o) in span_mut(dqkv, s * QKV_COLS + 2 * E + h * D, D)?.iter_mut().zip(dout) {
                        *dv = ws.mul_add(o, *dv);
                    }
                }
                for (d, &v) in span_mut(dqkv, t * QKV_COLS + h * D, D)?.iter_mut().zip(&dq) {
                    *d = v;
                }
            }
        }
        Ok(())
    }

    fn embed_bwd(&mut self, n: usize) -> Op {
        rows(0, n)?;
        if self.tokens.len() < n {
            return Err(format!("a token for each of {n} rows"));
        }
        let dx = self.read(Buf::DX0)?;
        let g = Buf::Grads.of_mut(&mut self.bufs)?;
        for ((t, &token), dx) in self.tokens.iter().enumerate().zip(dx.as_chunks::<E>().0.iter()).take(n) {
            for (g, &dx) in span_mut(g, WTE + token as usize * E, E)?.iter_mut().zip(dx) {
                *g += dx;
            }
            for (g, &dx) in span_mut(g, WPE + t * E, E)?.iter_mut().zip(dx) {
                *g += dx;
            }
        }
        Ok(())
    }

    fn adam(&mut self, a: &Adam) -> Op {
        let four = [Buf::Params, Buf::Grads, Buf::Moment1, Buf::Moment2].map(|b| b as usize);
        let [p, g, m, v] =
            self.bufs.get_disjoint_mut(four).map_err(|e| format!("the parameters' four buffers: {e}"))?;
        for (((p, g), m), v) in p.iter_mut().zip(g).zip(m).zip(v).take(PARAMS) {
            let t = *m * a.b1;
            *m = a.ob1.mul_add(*g, t);
            let gg = *g * *g;
            let u = *v * a.b2;
            *v = a.ob2.mul_add(gg, u);
            let mh = *m * a.bc1;
            let vh = *v * a.bc2;
            let d = vh.sqrt() + a.eps;
            let step = (a.lr * mh) / d;
            *p -= step;
            *g = 0.0;
        }
        Ok(())
    }
}
