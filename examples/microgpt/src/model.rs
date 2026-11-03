//! microgpt.py's model as the kernels compute it: its sizes, where each weight is, and the
//! order of the operations a step of training is made of, written once for every backend.
//!
//! Each backend computes the same operations in the same order, rounding at the same points,
//! so that all of them agree bit for bit; [`crate::reference`] says what each one does in
//! plain Rust. The numbers are `f32` where microgpt.py's are `f64`, so they agree with
//! Python's only as far as that allows.

use crate::random::Random;

pub use consts::*;

/// What the kernels read of the model, once for every backend: its sizes, where its
/// embeddings are, and `exp`'s constants. Each module of kernels takes them as its own,
/// `#[kernite_macros::kernels(consts = crate::model::consts)]`, and the rest of the crate as
/// the model's, `model::E`.
#[kernite_macros::consts]
pub mod consts {
    /// Tokens: the 26 letters, and `BOS`.
    pub const V: usize = 27;
    /// The vocabulary padded to a multiple of eight: the width of a row of logits.
    pub const VP: usize = 32;
    /// The width of the embedding.
    pub const E: usize = 16;
    /// Heads of attention.
    pub const H: usize = 4;
    /// The width of each head.
    pub const D: usize = E / H;
    /// The longest context: positions 0 to 15.
    pub const T: usize = 16;
    /// The width of the MLP's hidden layer.
    pub const M: usize = 4 * E;
    /// The token that begins and ends each name.
    pub const BOS: u32 = 26;
    /// The token embeddings, a row of `E` for each token.
    pub const WTE: usize = 0;
    /// The position embeddings, a row of `E` for each position.
    pub const WPE: usize = WTE + V * E;
    /// The width of a row of queries, keys and values, side by side.
    pub const QKV_COLS: usize = 3 * E;
    /// ln 2 in two parts, Cephes's: the first of nine bits, so that `k ln 2` loses nothing for
    /// the `k` that [`exp`](crate::reference::exp) takes, and the rest.
    pub const LN2_HI: f32 = 355.0 / 512.0;
    /// The rest of ln 2.
    pub const LN2_LO: f32 = -2.121_944_4e-4;
    /// The Taylor series of e^r to the sixth power, highest first.
    pub const TAYLOR: [f32; 7] = [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0];
}

/// A matrix of weights, `rows` inputs by `cols` outputs, stored a row of outputs to each
/// input: the transpose of microgpt.py's, so that a row's outputs are side by side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layer {
    /// Where it starts among the parameters.
    pub at: usize,
    /// Inputs.
    pub rows: usize,
    /// Outputs.
    pub cols: usize,
}

/// Queries, keys and values side by side: `wq`, `wk` and `wv`.
pub const QKV: Layer = Layer { at: WPE + T * E, rows: E, cols: QKV_COLS };
/// Attention's output projection.
pub const WO: Layer = Layer { at: QKV.at + E * 3 * E, rows: E, cols: E };
/// The MLP's first layer.
pub const FC1: Layer = Layer { at: WO.at + E * E, rows: E, cols: M };
/// The MLP's second layer.
pub const FC2: Layer = Layer { at: FC1.at + E * M, rows: M, cols: E };
/// The logits' head, `lm_head`, its outputs past `V` zero.
pub const LM: Layer = Layer { at: FC2.at + M * E, rows: E, cols: VP };
/// The parameters, padding and all.
pub const PARAMS: usize = LM.at + E * VP;

/// The weights as microgpt.py draws them, each from N(0, 0.08) in its order: `wte`, `wpe`,
/// `lm_head`, then the layer's `wq`, `wk`, `wv`, `wo`, `mlp_fc1` and `mlp_fc2`, each a row of
/// inputs to an output; rounded to `f32`, and stored as [`Layer`] says.
pub fn init(rng: &mut Random) -> Vec<f32> {
    let mut p = vec![0.0; PARAMS];
    let mut draw = |outs: usize, ins: usize, at: &dyn Fn(usize, usize) -> usize| {
        for o in 0..outs {
            for i in 0..ins {
                p[at(o, i)] = rng.gauss(0.0, 0.08) as f32;
            }
        }
    };
    let layer = |l: Layer, from: usize| move |o: usize, i: usize| l.at + i * l.cols + from + o;
    draw(V, E, &|t, e| WTE + t * E + e);
    draw(T, E, &|t, e| WPE + t * E + e);
    draw(V, E, &layer(LM, 0));
    for part in 0..3 {
        draw(E, E, &layer(QKV, part * E));
    }
    draw(E, E, &layer(WO, 0));
    draw(M, E, &layer(FC1, 0));
    draw(E, M, &layer(FC2, 0));
    p
}

/// What a backend holds, by name: the activations of a document's positions and their
/// gradients, `T` rows of each, and the parameters with their gradients and Adam's moments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Buf {
    /// The token's embedding plus the position's.
    X0,
    /// That normalized: the residual stream into attention.
    X1,
    /// Normalized again: attention's input.
    X2,
    /// Each position's query, key and value.
    Qkv,
    /// Attention's weights: for each head of each position, one for each position it attends.
    Aw,
    /// Attention's output, the heads side by side.
    Att,
    /// The residual stream after attention.
    X3,
    /// That normalized: the MLP's input.
    X4,
    /// The MLP's hidden layer, after its ReLU.
    Hid,
    /// The residual stream after the MLP.
    X5,
    /// The logits.
    Logits,
    /// The probabilities of the next token.
    Probs,
    /// The scale each normalization gave its row, the reciprocal of its root mean square:
    /// `X1`'s.
    R1,
    /// `X2`'s.
    R2,
    /// `X4`'s.
    R3,
    /// The gradient of the loss by the logits.
    DLogits,
    /// The residual stream's gradient: by `X5`, then by `X3`, then by `X1`.
    D5,
    /// The gradient by `Hid`.
    DHid,
    /// By `X4`.
    DX4,
    /// By `Att`.
    DAtt,
    /// By `Qkv`.
    DQkv,
    /// By `X2`.
    DX2,
    /// By `X0`.
    DX0,
    /// The parameters.
    Params,
    /// Their gradients.
    Grads,
    /// Adam's first moments.
    Moment1,
    /// Adam's second moments.
    Moment2,
}

impl Buf {
    /// Every buffer.
    pub const ALL: [Buf; 27] = {
        use Buf::*;
        [
            X0, X1, X2, Qkv, Aw, Att, X3, X4, Hid, X5, Logits, Probs, R1, R2, R3, DLogits, D5, DHid, DX4, DAtt, DQkv,
            DX2, DX0, Params, Grads, Moment1, Moment2,
        ]
    };

    /// The gradients a backward pass adds to, which start each step at zero.
    pub const SUMS: [Buf; 7] = [Buf::D5, Buf::DHid, Buf::DX4, Buf::DAtt, Buf::DQkv, Buf::DX2, Buf::DX0];

    /// Floats in each of its rows.
    pub fn width(self) -> usize {
        use Buf::*;
        match self {
            X0 | X1 | X2 | Att | X3 | X4 | X5 | D5 | DX4 | DAtt | DX2 | DX0 => E,
            Qkv | DQkv => 3 * E,
            Aw => H * T,
            Hid | DHid => M,
            Logits | Probs | DLogits => VP,
            R1 | R2 | R3 => 1,
            Params | Grads | Moment1 | Moment2 => PARAMS,
        }
    }

    /// Floats in all.
    pub fn len(self) -> usize {
        match self {
            Buf::Params | Buf::Grads | Buf::Moment1 | Buf::Moment2 => PARAMS,
            b => T * b.width(),
        }
    }
}

/// What a matrix product does to each output before it is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Then {
    /// Nothing.
    Store,
    /// Adds the residual stream's row.
    Add(Buf),
    /// Keeps it if it is above zero, else zero.
    Relu,
}

/// Adam's step: microgpt.py's rate and betas, and the corrections of the moments' bias.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adam {
    /// The learning rate, decayed in a line to zero.
    pub lr: f32,
    /// β₁, and 1 − β₁.
    pub b1: f32,
    #[allow(missing_docs)]
    pub ob1: f32,
    /// β₂, and 1 − β₂.
    pub b2: f32,
    #[allow(missing_docs)]
    pub ob2: f32,
    /// 1 / (1 − β₁ᵗ) and 1 / (1 − β₂ᵗ), at step t from 1.
    pub bc1: f32,
    #[allow(missing_docs)]
    pub bc2: f32,
    /// ε.
    pub eps: f32,
}

impl Adam {
    /// The step after `step` of `steps`, counting from 0, worked out in `f64` as microgpt.py
    /// does and then rounded.
    pub fn at(step: usize, steps: usize) -> Adam {
        let (lr, b1, b2) = (0.01, 0.85f64, 0.99f64);
        let t = (step + 1) as f64;
        Adam {
            lr: (lr * (1.0 - step as f64 / steps as f64)) as f32,
            b1: b1 as f32,
            ob1: (1.0 - b1) as f32,
            b2: b2 as f32,
            ob2: (1.0 - b2) as f32,
            bc1: (1.0 / (1.0 - b1.powf(t))) as f32,
            bc2: (1.0 / (1.0 - b2.powf(t))) as f32,
            eps: 1e-8,
        }
    }
}

/// The operations the model is made of, which a backend computes: the reference in Rust, the
/// others by their kernels. Each forward operation computes rows `t0..n`, so that a sample can
/// add one position to the ones before; each backward one, rows `0..n`.
pub trait Ops {
    /// Copies `data` into `b`, all of it.
    fn write(&mut self, b: Buf, data: &[f32]);
    /// All of `b`.
    fn read(&mut self, b: Buf) -> Vec<f32>;
    /// The document's tokens, from `BOS`: what the positions read, and the targets they are
    /// trained toward.
    fn tokens(&mut self, tokens: &[u32]);

    /// `X0`: each position's token's embedding plus its own.
    fn embed(&mut self, t0: usize, n: usize);
    /// `y` is `x` normalized by its root mean square, and `r` the scale it took.
    fn rmsnorm(&mut self, y: Buf, r: Buf, x: Buf, t0: usize, n: usize);
    /// `y` is `x` by `layer`, and then what `then` says.
    fn matmul(&mut self, layer: Layer, y: Buf, x: Buf, then: Then, t0: usize, n: usize);
    /// `Att` and `Aw`, from `Qkv`: each head of each position attending to it and the ones
    /// before.
    fn attention(&mut self, t0: usize, n: usize);
    /// `Probs`: the softmax of the logits times `inv_temp`, past `V` zero.
    fn softmax(&mut self, inv_temp: f32, t0: usize, n: usize);

    /// Sets the buffers in [`Buf::SUMS`] to zero.
    fn clear(&mut self);
    /// `DLogits`: the cross-entropy's gradient toward each next token, over `n` positions.
    fn xent(&mut self, n: usize);
    /// `dx += dy` by `layer` transposed: the gradient by its input.
    fn bwd_x(&mut self, layer: Layer, dx: Buf, dy: Buf, n: usize);
    /// `layer`'s gradient `+= x` transposed by `dy`.
    fn bwd_w(&mut self, layer: Layer, x: Buf, dy: Buf, n: usize);
    /// `DHid` where the ReLU passed its input, else zero.
    fn relu_bwd(&mut self, n: usize);
    /// `dx +=` the gradient by `x` of the normalization, `r` its scale, given `dy`.
    fn rmsnorm_bwd(&mut self, dx: Buf, dy: Buf, x: Buf, r: Buf, n: usize);
    /// `DQkv`, from `DAtt`.
    fn attention_bwd(&mut self, n: usize);
    /// The embeddings' gradients `+= DX0`.
    fn embed_bwd(&mut self, n: usize);
    /// Adam's step on every parameter, which zeroes its gradient.
    fn adam(&mut self, adam: &Adam);
}

/// Positions `t0..n` of the model, the earlier ones' keys and values in place: the
/// probabilities of the next token at each, the logits scaled by `inv_temp` first.
pub fn forward(ops: &mut impl Ops, t0: usize, n: usize, inv_temp: f32) {
    use Buf::*;
    ops.embed(t0, n);
    // Normalized once into the residual stream, and again into attention, as microgpt.py
    // does: not redundant, by the gradient the residual stream carries.
    ops.rmsnorm(X1, R1, X0, t0, n);
    ops.rmsnorm(X2, R2, X1, t0, n);
    ops.matmul(QKV, Qkv, X2, Then::Store, t0, n);
    ops.attention(t0, n);
    ops.matmul(WO, X3, Att, Then::Add(X1), t0, n);
    ops.rmsnorm(X4, R3, X3, t0, n);
    ops.matmul(FC1, Hid, X4, Then::Relu, t0, n);
    ops.matmul(FC2, X5, Hid, Then::Add(X3), t0, n);
    ops.matmul(LM, Logits, X5, Then::Store, t0, n);
    ops.softmax(inv_temp, t0, n);
}

/// The gradients of the mean cross-entropy over positions `0..n`, added to `Grads`.
pub fn backward(ops: &mut impl Ops, n: usize) {
    use Buf::*;
    ops.clear();
    ops.xent(n);
    ops.bwd_x(LM, D5, DLogits, n);
    ops.bwd_w(LM, X5, DLogits, n);
    ops.bwd_x(FC2, DHid, D5, n);
    ops.bwd_w(FC2, Hid, D5, n);
    ops.relu_bwd(n);
    ops.bwd_x(FC1, DX4, DHid, n);
    ops.bwd_w(FC1, X4, DHid, n);
    // `D5` was the gradient by `X5`, which is `X3` plus the MLP's: now it is by `X3`.
    ops.rmsnorm_bwd(D5, DX4, X3, R3, n);
    ops.bwd_x(WO, DAtt, D5, n);
    ops.bwd_w(WO, Att, D5, n);
    ops.attention_bwd(n);
    ops.bwd_x(QKV, DX2, DQkv, n);
    ops.bwd_w(QKV, X2, DQkv, n);
    // And `X3` is `X1` plus attention's: now `D5` is by `X1`.
    ops.rmsnorm_bwd(D5, DX2, X1, R2, n);
    ops.rmsnorm_bwd(DX0, D5, X0, R1, n);
    ops.embed_bwd(n);
}

/// The tokens of `doc`, between a `BOS` on each side, each letter by its place in `chars`.
pub fn tokenize(doc: &str, chars: &[char]) -> Vec<u32> {
    let letters = doc.chars().map(|c| chars.iter().position(|&k| k == c).expect("a letter of the names") as u32);
    std::iter::once(BOS).chain(letters).chain([BOS]).collect()
}

/// Step `step` of `steps` of training on `tokens`: forward over its first `T` positions at
/// most, backward, Adam. Its loss, the mean cross-entropy, worked out in `f64` as microgpt.py
/// does, from the probabilities.
pub fn train(ops: &mut impl Ops, tokens: &[u32], step: usize, steps: usize) -> f64 {
    let n = T.min(tokens.len() - 1);
    ops.tokens(&tokens[..=n]);
    forward(ops, 0, n, 1.0);
    let probs = ops.read(Buf::Probs);
    let losses: f64 = (0..n).map(|t| -f64::from(probs[t * VP + tokens[t + 1] as usize]).ln()).sum();
    backward(ops, n);
    ops.adam(&Adam::at(step, steps));
    (1.0 / n as f64) * losses
}

/// A name, as microgpt.py samples one: from `BOS`, each next token drawn from the
/// probabilities the logits times `inv_temp` give, until `BOS` again or `T` letters.
pub fn sample(ops: &mut impl Ops, rng: &mut Random, inv_temp: f32, chars: &[char]) -> String {
    let (mut tokens, mut name) = (vec![BOS], String::new());
    for pos in 0..T {
        ops.tokens(&tokens);
        forward(ops, pos, pos + 1, inv_temp);
        let probs = ops.read(Buf::Probs);
        let weights: Vec<f64> = probs[pos * VP..][..V].iter().map(|&p| f64::from(p)).collect();
        let token = rng.choice(&weights) as u32;
        if token == BOS {
            break;
        }
        name.push(chars[token as usize]);
        tokens.push(token);
    }
    name
}
