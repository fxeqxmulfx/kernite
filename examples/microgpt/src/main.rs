//! Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95),
//! trained and sampled by kernels written in kgen's language: x86 ones on the CPU, PTX ones on
//! the GPU.
//!
//! microgpt.py trains a GPT of one layer to make up names, in pure Python, a scalar at a time.
//! This is the same model, its weights drawn from the same Mersenne Twister in the same order
//! and trained on the same names in the same order, so that what it prints can be held to what
//! microgpt.py prints. Its numbers are `f32` where Python's are `f64`, so the two agree only
//! as far as that allows. The backends agree with each other bit for bit: each computes the
//! operations [`model`] lists in the order [`reference`] gives.
//!
//! ```text
//! cargo run --release -p microgpt              # the x86 kernels
//! cargo run --release -p microgpt -- gpu       # the PTX kernels, on the GPU
//! cargo run --release -p microgpt -- reference # plain Rust
//! ```

mod cpu;
mod gpu;
mod model;
mod random;
mod reference;

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::process::{Command, ExitCode};

use cpu::Cpu;
use gpu::Gpu;
use model::{BOS, Ops, V};
use random::Random;
use reference::Reference;

/// Where microgpt.py downloads the names from.
const NAMES: &str = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";
/// Steps of training, a name each.
const STEPS: usize = 1000;
/// Names made up after.
const SAMPLES: usize = 20;
/// 1 / microgpt.py's temperature, 0.5: what the logits are multiplied by, exactly as Python's
/// division by it does.
const INV_TEMP: f32 = 2.0;

/// The names microgpt.py trains on, in the order it shuffles them, and their letters.
struct Names {
    docs: Vec<String>,
    chars: Vec<char>,
}

impl Names {
    /// `docs`, each a name, and the letters among them in order: tokens 0 to 25.
    fn new(docs: Vec<String>) -> Names {
        let chars: Vec<char> = docs.iter().flat_map(|d| d.chars()).collect::<BTreeSet<_>>().into_iter().collect();
        assert_eq!(chars.len() as u32, BOS, "the names are of the 26 letters");
        Names { docs, chars }
    }
}

/// The lines of `text` that are not blank, trimmed, as microgpt.py reads `input.txt`.
fn lines(text: &str) -> Vec<String> {
    text.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect()
}

/// microgpt.py's names: `input.txt` beside this crate, downloaded as microgpt.py does when it
/// is not there.
fn read_names() -> Result<Vec<String>, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("input.txt");
    if !path.exists() {
        eprintln!("downloading {NAMES}");
        let status = Command::new("curl").arg("-fsSLo").arg(&path).arg(NAMES).status();
        if !status.as_ref().is_ok_and(|s| s.success()) {
            let _ = std::fs::remove_file(&path);
            return Err(format!("curl could not download {NAMES}: {status:?}"));
        }
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(lines(&text))
}

/// microgpt.py's run on `ops`: `steps` steps of training, the names in turn, then `samples`
/// names made up by the model. `step` is told each step's loss when it is known.
fn run(
    ops: &mut impl Ops,
    names: &Names,
    rng: &mut Random,
    steps: usize,
    samples: usize,
    mut step: impl FnMut(usize, f64),
) -> Vec<String> {
    for i in 0..steps {
        let tokens = model::tokenize(&names.docs[i % names.docs.len()], &names.chars);
        step(i, model::train(ops, &tokens, i, steps));
    }
    (0..samples).map(|_| model::sample(ops, rng, INV_TEMP, &names.chars)).collect()
}

/// Which computes the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Cpu,
    Gpu,
    Reference,
}

fn main() -> ExitCode {
    let backend = match std::env::args().nth(1).as_deref() {
        None | Some("cpu") => Backend::Cpu,
        Some("gpu") => Backend::Gpu,
        Some("reference") => Backend::Reference,
        Some(other) => {
            eprintln!("microgpt [cpu | gpu | reference]: `{other}` is none of them");
            return ExitCode::FAILURE;
        }
    };
    let mut docs = match read_names() {
        Ok(docs) => docs,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut rng = Random::new(42);
    rng.shuffle(&mut docs);
    let names = Names::new(docs);
    println!("num docs: {}", names.docs.len());
    println!("vocab size: {V}");
    let params = model::init(&mut rng);
    println!("num params: {}", params.iter().filter(|&&p| p != 0.0).count());
    let print = |i: usize, loss: f64| {
        print!("step {:4} / {STEPS:4} | loss {loss:.4}\r", i + 1);
        let _ = std::io::stdout().flush();
    };
    let samples = match backend {
        Backend::Cpu => match Cpu::new(&params) {
            Some(mut cpu) => run(&mut cpu, &names, &mut rng, STEPS, SAMPLES, print),
            None => {
                eprintln!("this CPU lacks AVX2 or FMA, which the kernels are written in");
                return ExitCode::FAILURE;
            }
        },
        Backend::Gpu => match Gpu::new(&params) {
            Ok(mut gpu) => run(&mut gpu, &names, &mut rng, STEPS, SAMPLES, print),
            Err(e) => {
                eprintln!("no GPU to run the kernels on: {e}");
                return ExitCode::FAILURE;
            }
        },
        Backend::Reference => run(&mut Reference::new(&params), &names, &mut rng, STEPS, SAMPLES, print),
    };
    println!("\n--- inference (new, hallucinated names) ---");
    for (i, name) in samples.iter().enumerate() {
        println!("sample {:2}: {name}", i + 1);
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests;
