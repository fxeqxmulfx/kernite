//! The model's operations in plain Rust: what each kernel computes, operation for operation
//! and rounding for rounding, so that the kernels can be held to it bit for bit.
//!
//! Every sum is a chain from zero in the order written, each product added by a fused
//! multiply-add, which rounds once. `e^x` is [`exp`], a polynomial the kernels evaluate the
//! same way, where a library's would differ by machine.

use crate::model::{Adam, Buf, D, E, H, Layer, M, Ops, PARAMS, T, Then, V, VP, WPE, WTE};

/// ln 2 in two parts, Cephes's: the first of nine bits, so that `k ln 2` loses nothing for the
/// `k` that `exp` takes, and the rest.
pub const LN2_HI: f32 = 355.0 / 512.0;
/// The rest of ln 2.
pub const LN2_LO: f32 = -2.121_944_4e-4;
/// The Taylor series of e^r to the sixth power, highest first.
pub const TAYLOR: [f32; 7] = [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0];

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
    /// A model of `params`, Adam's moments zero.
    pub fn new(params: &[f32]) -> Reference {
        let mut r = Reference { bufs: Buf::ALL.iter().map(|b| vec![0.0; b.len()]).collect(), tokens: vec![] };
        r.write(Buf::Params, params);
        r
    }

    fn buf(&self, b: Buf) -> &[f32] {
        &self.bufs[b as usize]
    }

    fn buf_mut(&mut self, b: Buf) -> &mut Vec<f32> {
        &mut self.bufs[b as usize]
    }

    /// Two buffers at once, `a` to read and `b` to write.
    fn pair(&mut self, a: Buf, b: Buf) -> (&[f32], &mut [f32]) {
        assert_ne!(a, b, "a buffer read and written at once");
        let (a, b) = (a as usize, b as usize);
        if a < b {
            let (lo, hi) = self.bufs.split_at_mut(b);
            (&lo[a], &mut hi[0])
        } else {
            let (lo, hi) = self.bufs.split_at_mut(a);
            (&hi[0], &mut lo[b])
        }
    }
}

impl Ops for Reference {
    fn write(&mut self, b: Buf, data: &[f32]) {
        self.buf_mut(b).copy_from_slice(data);
    }

    fn read(&mut self, b: Buf) -> Vec<f32> {
        self.buf(b).to_vec()
    }

    fn tokens(&mut self, tokens: &[u32]) {
        self.tokens = tokens.to_vec();
    }

    fn embed(&mut self, t0: usize, n: usize) {
        let p = self.buf(Buf::Params).to_vec();
        let tokens = self.tokens.clone();
        let x = self.buf_mut(Buf::X0);
        for t in t0..n {
            for e in 0..E {
                x[t * E + e] = p[WTE + tokens[t] as usize * E + e] + p[WPE + t * E + e];
            }
        }
    }

    fn rmsnorm(&mut self, y: Buf, r: Buf, x: Buf, t0: usize, n: usize) {
        let x = self.buf(x).to_vec();
        let mut scales = self.buf(r).to_vec();
        let out = self.buf_mut(y);
        for t in t0..n {
            let row = &x[t * E..][..E];
            let ss = row.iter().fold(0.0f32, |ss, &v| v.mul_add(v, ss));
            let ms = ss * 0.0625;
            let s = 1.0 / (ms + 1e-5).sqrt();
            scales[t] = s;
            for e in 0..E {
                out[t * E + e] = row[e] * s;
            }
        }
        self.write(r, &scales);
    }

    fn matmul(&mut self, l: Layer, y: Buf, x: Buf, then: Then, t0: usize, n: usize) {
        let (w, x) = (self.buf(Buf::Params)[l.at..][..l.rows * l.cols].to_vec(), self.buf(x).to_vec());
        let res = match then {
            Then::Add(b) => self.buf(b).to_vec(),
            _ => vec![],
        };
        let out = self.buf_mut(y);
        for t in t0..n {
            for o in 0..l.cols {
                let mut acc = 0.0f32;
                for i in 0..l.rows {
                    acc = x[t * l.rows + i].mul_add(w[i * l.cols + o], acc);
                }
                out[t * l.cols + o] = match then {
                    Then::Store => acc,
                    Then::Add(_) => acc + res[t * l.cols + o],
                    Then::Relu if acc > 0.0 => acc,
                    Then::Relu => 0.0,
                };
            }
        }
    }

    fn attention(&mut self, t0: usize, n: usize) {
        let qkv = self.buf(Buf::Qkv).to_vec();
        let (mut aw, mut att) = (self.buf(Buf::Aw).to_vec(), self.buf(Buf::Att).to_vec());
        for t in t0..n {
            for h in 0..H {
                let q = &qkv[t * 3 * E + h * D..][..D];
                let w = &mut aw[(t * H + h) * T..][..T];
                let mut m = f32::NEG_INFINITY;
                for s in 0..=t {
                    let k = &qkv[s * 3 * E + E + h * D..][..D];
                    let dot = (0..D).fold(0.0f32, |acc, j| q[j].mul_add(k[j], acc));
                    let l = dot * 0.5;
                    w[s] = l;
                    if l > m {
                        m = l;
                    }
                }
                let mut total = 0.0f32;
                for ws in &mut w[..=t] {
                    *ws = exp(*ws - m);
                    total += *ws;
                }
                let inv = 1.0 / total;
                let mut out = [0.0f32; D];
                for s in 0..=t {
                    w[s] *= inv;
                    let v = &qkv[s * 3 * E + 2 * E + h * D..][..D];
                    for j in 0..D {
                        out[j] = w[s].mul_add(v[j], out[j]);
                    }
                }
                att[t * E + h * D..][..D].copy_from_slice(&out);
            }
        }
        self.write(Buf::Aw, &aw);
        self.write(Buf::Att, &att);
    }

    fn softmax(&mut self, inv_temp: f32, t0: usize, n: usize) {
        let (logits, probs) = self.pair(Buf::Logits, Buf::Probs);
        for t in t0..n {
            let (l, p) = (&logits[t * VP..][..VP], &mut probs[t * VP..][..VP]);
            let mut m = f32::NEG_INFINITY;
            for j in 0..V {
                p[j] = l[j] * inv_temp;
                if p[j] > m {
                    m = p[j];
                }
            }
            let mut total = 0.0f32;
            for pj in &mut p[..V] {
                *pj = exp(*pj - m);
                total += *pj;
            }
            let inv = 1.0 / total;
            for pj in &mut p[..V] {
                *pj *= inv;
            }
            p[V..].fill(0.0);
        }
    }

    fn clear(&mut self) {
        for b in Buf::SUMS {
            self.buf_mut(b).fill(0.0);
        }
    }

    fn xent(&mut self, n: usize) {
        let inv_n = 1.0 / n as f32;
        let targets = self.tokens[1..=n].to_vec();
        let (probs, d) = self.pair(Buf::Probs, Buf::DLogits);
        for t in 0..n {
            for j in 0..VP {
                let y = if j == targets[t] as usize { 1.0 } else { 0.0 };
                d[t * VP + j] = (probs[t * VP + j] - y) * inv_n;
            }
        }
    }

    fn bwd_x(&mut self, l: Layer, dx: Buf, dy: Buf, n: usize) {
        let w = self.buf(Buf::Params)[l.at..][..l.rows * l.cols].to_vec();
        let (dy, dx) = self.pair(dy, dx);
        for t in 0..n {
            for i in 0..l.rows {
                let mut acc = dx[t * l.rows + i];
                for o in 0..l.cols {
                    acc = dy[t * l.cols + o].mul_add(w[i * l.cols + o], acc);
                }
                dx[t * l.rows + i] = acc;
            }
        }
    }

    fn bwd_w(&mut self, l: Layer, x: Buf, dy: Buf, n: usize) {
        let (x, dy) = (self.buf(x).to_vec(), self.buf(dy).to_vec());
        let g = &mut self.buf_mut(Buf::Grads)[l.at..][..l.rows * l.cols];
        for i in 0..l.rows {
            for o in 0..l.cols {
                let mut acc = g[i * l.cols + o];
                for t in 0..n {
                    acc = x[t * l.rows + i].mul_add(dy[t * l.cols + o], acc);
                }
                g[i * l.cols + o] = acc;
            }
        }
    }

    fn relu_bwd(&mut self, n: usize) {
        let (h, dh) = self.pair(Buf::Hid, Buf::DHid);
        for (d, &x) in dh[..n * M].iter_mut().zip(&h[..n * M]) {
            // Not `x <= 0.0`: the ReLU passed no NaN either.
            *d = if x > 0.0 { *d } else { 0.0 };
        }
    }

    fn rmsnorm_bwd(&mut self, dx: Buf, dy: Buf, x: Buf, r: Buf, n: usize) {
        let (dy, x, r) = (self.buf(dy).to_vec(), self.buf(x).to_vec(), self.buf(r).to_vec());
        let dx = self.buf_mut(dx);
        for t in 0..n {
            let (dy, x) = (&dy[t * E..][..E], &x[t * E..][..E]);
            let dot = (0..E).fold(0.0f32, |acc, e| dy[e].mul_add(x[e], acc));
            let r2 = r[t] * r[t];
            let r3 = r2 * r[t];
            let c = dot * r3;
            let c = c * 0.0625;
            for e in 0..E {
                let g = dy[e] * r[t];
                let g = (-c).mul_add(x[e], g);
                dx[t * E + e] += g;
            }
        }
    }

    fn attention_bwd(&mut self, n: usize) {
        let (qkv, aw, datt) = (self.buf(Buf::Qkv).to_vec(), self.buf(Buf::Aw).to_vec(), self.buf(Buf::DAtt).to_vec());
        let dqkv = self.buf_mut(Buf::DQkv);
        for h in 0..H {
            for t in 0..n {
                let q = &qkv[t * 3 * E + h * D..][..D];
                let w = &aw[(t * H + h) * T..][..T];
                let dout = &datt[t * E + h * D..][..D];
                // The gradient by each weight: the output's gradient by its value.
                let dw = |s: usize| {
                    let v = &qkv[s * 3 * E + 2 * E + h * D..][..D];
                    (0..D).fold(0.0f32, |acc, j| dout[j].mul_add(v[j], acc))
                };
                let sum = (0..=t).fold(0.0f32, |sum, s| w[s].mul_add(dw(s), sum));
                let mut dq = [0.0f32; D];
                for (s, &ws) in w[..=t].iter().enumerate() {
                    let dl = ws * (dw(s) - sum);
                    let g = dl * 0.5;
                    for j in 0..D {
                        let k = s * 3 * E + E + h * D + j;
                        dq[j] = g.mul_add(qkv[k], dq[j]);
                        dqkv[k] = g.mul_add(q[j], dqkv[k]);
                        dqkv[k + E] = ws.mul_add(dout[j], dqkv[k + E]);
                    }
                }
                dqkv[t * 3 * E + h * D..][..D].copy_from_slice(&dq);
            }
        }
    }

    fn embed_bwd(&mut self, n: usize) {
        let (dx, tokens) = (self.buf(Buf::DX0).to_vec(), self.tokens.clone());
        let g = self.buf_mut(Buf::Grads);
        for t in 0..n {
            for e in 0..E {
                g[WTE + tokens[t] as usize * E + e] += dx[t * E + e];
                g[WPE + t * E + e] += dx[t * E + e];
            }
        }
    }

    fn adam(&mut self, a: &Adam) {
        let four = [Buf::Params, Buf::Grads, Buf::Moment1, Buf::Moment2].map(|b| b as usize);
        let [p, g, m, v] = self.bufs.get_disjoint_mut(four).expect("the parameters' four buffers");
        for i in 0..PARAMS {
            let t = m[i] * a.b1;
            m[i] = a.ob1.mul_add(g[i], t);
            let gg = g[i] * g[i];
            let u = v[i] * a.b2;
            v[i] = a.ob2.mul_add(gg, u);
            let mh = m[i] * a.bc1;
            let vh = v[i] * a.bc2;
            let d = vh.sqrt() + a.eps;
            let step = (a.lr * mh) / d;
            p[i] -= step;
            g[i] = 0.0;
        }
    }
}
