//! The kernels, written in [`kgen_x86::Asm`], and [`Kernel`], which names each with its
//! parameters the way the macro spells them: `iq2_xxs(ahead = 3, pairs = 2)`.

mod bf16;
mod iq2_xxs;
mod iq3_s;
mod iq3_s_nibble;
mod iq3_xxs;

pub use bf16::bf16_pairs;
pub use iq2_xxs::iq2_xxs;
pub use iq3_s::iq3_s;
pub use iq3_s_nibble::iq3_s_nibble;
pub use iq3_xxs::iq3_xxs;
use kgen_core::Program;
use kgen_x86::{Hint, Op};

/// A kernel and its parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// [`iq2_xxs`].
    Iq2Xxs {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
        /// How many of each group's two pairs of grid rows are loaded as pairs.
        pairs: usize,
    },
    /// [`iq3_xxs`].
    Iq3Xxs {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq3_s`].
    Iq3S {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq3_s_nibble`].
    Iq3SNibble {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`bf16_pairs`].
    Bf16Pairs {
        /// Bytes ahead of each row's reads that it asks for, or none.
        ahead: i32,
        /// How close to the core.
        hint: Hint,
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
                let pairs = take.int("pairs", 0, 0..=2)? as usize;
                Kernel::Iq2Xxs { ahead, pairs }
            }
            "iq3_xxs" => {
                let ahead = take.int("ahead", 1, 1..=7)? as usize;
                Kernel::Iq3Xxs { ahead }
            }
            "iq3_s" => {
                let ahead = take.int("ahead", 1, 1..=7)? as usize;
                Kernel::Iq3S { ahead }
            }
            "iq3_s_nibble" => {
                let ahead = take.int("ahead", 1, 1..=7)? as usize;
                Kernel::Iq3SNibble { ahead }
            }
            "bf16_pairs" => {
                let ahead = take.int("ahead", 0, 0..=4096)? as i32;
                let hint = [Hint::T0, Hint::T1, Hint::T2, Hint::Nta][take.int("hint", 0, 0..=3)? as usize];
                Kernel::Bf16Pairs { ahead, hint }
            }
            _ => {
                return Err(format!(
                    "no kernel `{name}`; there are `iq2_xxs`, `iq3_xxs`, `iq3_s`, `iq3_s_nibble` and `bf16_pairs`"
                ));
            }
        };
        take.done(name)?;
        Ok(kernel)
    }

    /// Its program.
    pub fn program(&self) -> Program<Op> {
        match *self {
            Kernel::Iq2Xxs { ahead, pairs } => iq2_xxs(ahead, pairs),
            Kernel::Iq3Xxs { ahead } => iq3_xxs(ahead),
            Kernel::Iq3S { ahead } => iq3_s(ahead),
            Kernel::Iq3SNibble { ahead } => iq3_s_nibble(ahead),
            Kernel::Bf16Pairs { ahead, hint } => bf16_pairs(ahead, hint),
        }
    }

    /// The kernels whose listings are kept as snapshots.
    pub fn snapshots() -> Vec<Kernel> {
        let mut kernels: Vec<Kernel> = (1..=4).map(|ahead| Kernel::Iq2Xxs { ahead, pairs: 0 }).collect();
        kernels.extend((1..=2).map(|pairs| Kernel::Iq2Xxs { ahead: 3, pairs }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq3Xxs { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq3S { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq3SNibble { ahead }));
        kernels.push(Kernel::Bf16Pairs { ahead: 0, hint: Hint::T0 });
        kernels.extend(
            [(512, Hint::T0), (1024, Hint::T0), (512, Hint::T1)].map(|(ahead, hint)| Kernel::Bf16Pairs { ahead, hint }),
        );
        kernels
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
