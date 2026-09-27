//! The kernels, written in [`kgen_x86::Asm`], and [`Kernel`], which names each with its
//! parameters the way the macro spells them: `iq2_xxs(ahead = 3)`.

mod iq2_xxs;

pub use iq2_xxs::iq2_xxs;
use kgen_core::Program;
use kgen_x86::Op;

/// A kernel and its parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// [`iq2_xxs`].
    Iq2Xxs {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
}

impl Kernel {
    /// The kernel `name` with `params`, each `key = value`; a parameter left out takes its
    /// default.
    pub fn parse(name: &str, params: &[(String, i64)]) -> Result<Kernel, String> {
        let mut take = Params(params.to_vec());
        let kernel = match name {
            "iq2_xxs" => {
                let ahead = take.int("ahead", 3, 1..=7)? as usize;
                Kernel::Iq2Xxs { ahead }
            }
            _ => return Err(format!("no kernel `{name}`; there is `iq2_xxs`")),
        };
        take.done(name)?;
        Ok(kernel)
    }

    /// Its program.
    pub fn program(&self) -> Program<Op> {
        match *self {
            Kernel::Iq2Xxs { ahead } => iq2_xxs(ahead),
        }
    }

    /// The kernels whose listings are kept as snapshots.
    pub fn snapshots() -> Vec<Kernel> {
        (1..=4).map(|ahead| Kernel::Iq2Xxs { ahead }).collect()
    }
}

/// Parameters not yet taken.
struct Params(Vec<(String, i64)>);

impl Params {
    fn int(&mut self, key: &str, default: i64, range: std::ops::RangeInclusive<i64>) -> Result<i64, String> {
        let Some(at) = self.0.iter().position(|(k, _)| k == key) else {
            return Ok(default);
        };
        let (_, value) = self.0.remove(at);
        if self.0.iter().any(|(k, _)| k == key) {
            return Err(format!("`{key}` given twice"));
        }
        if !range.contains(&value) {
            return Err(format!("`{key}` is {value}, outside {range:?}"));
        }
        Ok(value)
    }

    fn done(self, name: &str) -> Result<(), String> {
        match self.0.first() {
            Some((key, _)) => Err(format!("`{name}` takes no parameter `{key}`")),
            None => Ok(()),
        }
    }
}
