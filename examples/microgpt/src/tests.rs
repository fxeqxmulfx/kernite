//! The port held to microgpt.py, by what Python drew and printed; and the backends held to
//! the reference, bit for bit.

use std::fs;
use std::path::Path;

use crate::cpu::{self, Cpu};
use crate::gpu::{self, Gpu};
use crate::model::{self, Adam, Buf, Layer, Ops, Then, V};
use crate::random::Random;
use crate::reference::Reference;
use crate::{Names, SAMPLES, STEPS, lines, run};

/// How many names microgpt.py shuffles: all of `input.txt`'s.
const DOCS: usize = 32033;

/// `fixtures/{name}`.
fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures").join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The draws of `kind` that `fixtures/random.txt` has from Python: the words after the first
/// on its line.
fn draws<T: std::str::FromStr<Err: std::fmt::Debug>>(kind: &str) -> Vec<T> {
    let text = fixture("random.txt");
    let line = text.lines().find(|l| l.split(' ').next() == Some(kind)).unwrap_or_else(|| panic!("no `{kind}`"));
    line.split(' ').skip(1).map(|w| w.parse().unwrap()).collect()
}

/// Where microgpt.py's run starts, but for its download: the generator after it shuffles the
/// names, and the first thousand of them in that order, the ones its thousand steps train on.
fn python_start() -> (Random, Names) {
    let mut rng = Random::new(42);
    // What the shuffle draws depends only on how many names there are.
    rng.shuffle(&mut [(); DOCS]);
    (rng, Names::new(lines(&fixture("docs.txt"))))
}

/// A model of `params` on the GPU; or, where there is no GPU to test, `None`, and why.
fn on_gpu(params: &[f32]) -> Option<Gpu> {
    match kernite_cuda::Cuda::new() {
        Ok(_) => Some(Gpu::new(params).unwrap_or_else(|e| panic!("{e}"))),
        Err(e) => {
            eprintln!("no GPU: skipped. {e}");
            None
        }
    }
}

/// A model of `params` in plain Rust: the reference.
fn in_rust(params: &[f32]) -> Option<Reference> {
    Some(Reference::new(params))
}

/// A model of `params` on the CPU, which must run the kernels.
fn on_cpu(params: &[f32]) -> Option<Cpu> {
    Some(Cpu::new(params).expect("a CPU with AVX2 and FMA, which the kernels are written in"))
}

#[test]
fn draws_are_pythons() {
    let mut rng = Random::new(42);
    assert_eq!((0..6).map(|_| rng.random()).collect::<Vec<_>>(), draws::<f64>("random"));
    let mut rng = Random::new(42);
    assert_eq!([1, 5, 15, 31, 32].map(|k| rng.bits(k)).to_vec(), draws::<u32>("bits"));
    let mut rng = Random::new(42);
    assert_eq!((0..7).map(|_| rng.gauss(0.0, 0.08)).collect::<Vec<_>>(), draws::<f64>("gauss"));
    let (mut rng, weights) = (Random::new(42), [0.1, 0.25, 0.0, 0.4, 0.25]);
    assert_eq!((0..24).map(|_| rng.choice(&weights)).collect::<Vec<_>>(), draws::<usize>("choices"));
    // What microgpt.py draws, in its order: the shuffle of its names, then its 4192 weights.
    let (mut rng, mut order) = (Random::new(42), (0..DOCS).collect::<Vec<_>>());
    rng.shuffle(&mut order);
    assert_eq!(order[..12], draws::<usize>("shuffle"));
    let weights: Vec<f64> = (0..4192).map(|_| rng.gauss(0.0, 0.08)).collect();
    let want = draws::<f64>("weights");
    assert_eq!([weights[0], weights[1], weights[2], weights[3], weights[4191]], want[..5]);
    // Python 3.12 adds floats with compensation, which a sum in order differs from in the
    // last bits; a draw out of place would differ by far more.
    let sum: f64 = weights.iter().sum();
    assert!((sum - want[5]).abs() < 1e-9, "the weights sum to {sum}; Python's to {}", want[5]);
}

/// Two backends side by side: each operation done by both, and every buffer the same in both,
/// bit for bit, after it.
struct Both<A, B> {
    a: A,
    b: B,
}

impl<A: Ops, B: Ops> Both<A, B> {
    /// Panics at the first float the two differ in after `op`, if there is one.
    fn agree(&mut self, op: &str) {
        for buf in Buf::ALL {
            let (a, b) = (self.a.read(buf), self.b.read(buf));
            if let Some(i) = (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits()) {
                panic!("after `{op}`, {buf:?}[{i}] is {} in one and {} in the other", a[i], b[i]);
            }
        }
    }
}

/// The operations of [`Ops`] that change a buffer, each done by both and then compared.
macro_rules! both {
    ($($op:ident($($arg:ident: $ty:ty),*);)*) => {$(
        fn $op(&mut self, $($arg: $ty),*) {
            self.a.$op($($arg),*);
            self.b.$op($($arg),*);
            self.agree(stringify!($op));
        }
    )*};
}

impl<A: Ops, B: Ops> Ops for Both<A, B> {
    fn read(&mut self, b: Buf) -> Vec<f32> {
        self.a.read(b)
    }

    both! {
        write(b: Buf, data: &[f32]);
        tokens(tokens: &[u32]);
        embed(t0: usize, n: usize);
        rmsnorm(y: Buf, r: Buf, x: Buf, t0: usize, n: usize);
        matmul(l: Layer, y: Buf, x: Buf, then: Then, t0: usize, n: usize);
        attention(t0: usize, n: usize);
        softmax(inv_temp: f32, t0: usize, n: usize);
        clear();
        xent(n: usize);
        bwd_x(l: Layer, dx: Buf, dy: Buf, n: usize);
        bwd_w(l: Layer, x: Buf, dy: Buf, n: usize);
        relu_bwd(n: usize);
        rmsnorm_bwd(dx: Buf, dy: Buf, x: Buf, r: Buf, n: usize);
        attention_bwd(n: usize);
        embed_bwd(n: usize);
        adam(step: &Adam);
    }
}

/// Steps of training, the first on a document longer than the model sees, and names made up:
/// each operation of the backend `new` makes held to the reference's.
fn agrees_op_by_op<A: Ops>(new: impl FnOnce(&[f32]) -> Option<A>) {
    let (mut rng, names) = python_start();
    let params = model::init(&mut rng);
    let Some(a) = new(&params) else { return };
    let mut both = Both { a, b: Reference::new(&params) };
    let mut docs = vec!["abcdefghijklmnopqrstuvwxyz".to_string()];
    docs.extend_from_slice(&names.docs[..24]);
    run(&mut both, &Names { docs, chars: names.chars }, &mut rng, 25, 5, |_, _| {});
}

#[test]
fn cpu_agrees_with_the_reference_op_by_op() {
    agrees_op_by_op(on_cpu);
}

#[test]
fn gpu_agrees_with_the_reference_op_by_op() {
    agrees_op_by_op(on_gpu);
}

/// microgpt.py's run, on the backend `new` makes: the loss at each step, and the names made up
/// after. `None` if there is no backend.
fn python_run<A: Ops>(new: impl FnOnce(&[f32]) -> Option<A>) -> Option<(Vec<f64>, Vec<String>)> {
    let (mut rng, names) = python_start();
    let params = model::init(&mut rng);
    let mut ops = new(&params)?;
    let mut losses = vec![];
    let samples = run(&mut ops, &names, &mut rng, STEPS, SAMPLES, |_, loss| losses.push(loss));
    Some((losses, samples))
}

/// The reference's run is what microgpt.py printed: the same header, the same names made up,
/// and the same loss at each step, but for what `f32` rounds where Python's `f64` does not.
#[test]
fn reference_runs_as_python_does() {
    let text = fixture("python.txt");
    let params = model::init(&mut python_start().0);
    let nonzero = params.iter().filter(|&&p| p != 0.0).count();
    let header = [format!("num docs: {DOCS}"), format!("vocab size: {V}"), format!("num params: {nonzero}")];
    assert_eq!(text.lines().take(3).collect::<Vec<_>>(), header);

    let (losses, samples) = python_run(in_rust).unwrap();
    let printed: Vec<f64> =
        text.lines().filter_map(|l| l.split("| loss ").nth(1)).map(|x| x.parse().unwrap()).collect();
    let names: Vec<&str> = text.lines().filter_map(|l| l.strip_prefix("sample ")?.split(": ").nth(1)).collect();
    assert_eq!(samples, names);
    assert_eq!(losses.len(), printed.len());
    let gaps: Vec<f64> = losses.iter().zip(&printed).map(|(a, b)| (a - b).abs()).collect();
    // Python printed four decimals, which are off by up to 0.00005 alone; f32 makes the worst
    // just over that.
    let worst = (0..gaps.len()).max_by(|&i, &j| gaps[i].total_cmp(&gaps[j])).unwrap();
    let (step, got, want) = (worst + 1, losses[worst], printed[worst]);
    assert!(gaps[worst] < 1e-4, "the loss at step {step} is {got}; Python printed {want}");
}

/// `got` is `want`, loss for loss, bit for bit, and name for name.
fn same_run((losses, samples): (Vec<f64>, Vec<String>), (want_losses, want_samples): (Vec<f64>, Vec<String>)) {
    let step = losses.iter().zip(&want_losses).position(|(a, b)| a.to_bits() != b.to_bits());
    if let Some(i) = step {
        panic!("the loss at step {} is {}, and {} by the reference", i + 1, losses[i], want_losses[i]);
    }
    assert_eq!(samples, want_samples);
}

#[test]
fn cpu_runs_as_the_reference_does() {
    same_run(python_run(on_cpu).unwrap(), python_run(in_rust).unwrap());
}

#[test]
fn gpu_runs_as_the_reference_does() {
    if let Some(gpu) = python_run(on_gpu) {
        same_run(gpu, python_run(in_rust).unwrap());
    }
}

/// Each x86 kernel's listing against the one in `snapshots/`, and the GPU kernels' PTX, so that
/// a change to kernite or to a kernel that moves an instruction shows in review. With
/// `KERNITE_BLESS=1` the files are written instead, for the diff to be reviewed and committed.
#[test]
fn listings_match_their_snapshots() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("snapshots");
    let bless = std::env::var_os("KERNITE_BLESS").is_some();
    let mut changed = vec![];
    let files = cpu::kernels::LISTINGS.iter().map(|&(name, text)| (format!("{name}.s"), text));
    for (name, text) in files.chain([("gpu.ptx".to_string(), gpu::kernels::PTX)]) {
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
