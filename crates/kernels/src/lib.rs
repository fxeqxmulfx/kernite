//! The kernels, written in [`kgen_x86::Asm`], and [`Kernel`], which names each with its
//! parameters the way the macro spells them: `iq2_xxs(ahead = 3, pairs = 2)`.

mod bf16;
mod iq2_s;
mod iq2_s_nibble;
mod iq2_xxs;
mod iq3_s;
mod iq3_xxs;
mod iq4_nl;
mod lanes;
mod nibbles;
mod q4_k;
mod q6_k;

pub use bf16::{bf16_pairs, bf16_triples, bf16_two};
pub use iq2_s::iq2_s;
pub use iq2_s::iq2_s_gather;
pub use iq2_s::iq2_s_packed;
pub use iq2_s::iq2_s_packed2;
pub use iq2_s::iq2_xs_packed;
pub use iq2_s::iq2_xs_packed2;
pub use iq2_s_nibble::iq2_s_nibble;
pub use iq2_s_nibble::iq2_s_planes;
pub use iq2_xxs::iq2_xxs;
pub use iq2_xxs::iq2_xxs2;
pub use iq3_s::iq3_s;
pub use iq3_xxs::{iq3_xxs, iq3_xxs2};
pub use iq4_nl::iq4_nl;
use kgen_core::Program;
use kgen_x86::{Hint, Op};
pub use nibbles::{iq3_s_nibble, iq3_s_nibble2, iq3_xxs_nibble, iq3_xxs_nibble2, iq4_xs, iq4_xs2};
pub use q4_k::q4_k;
pub use q6_k::q6_k;
pub use q6_k::q6_k_front;

/// A kernel and its parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// [`iq2_s`].
    Iq2S {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq2_s_gather`].
    Iq2SGather {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq2_s_packed`].
    Iq2SPacked {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq2_s_packed2`].
    Iq2SPacked2 {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq2_xs_packed`].
    Iq2XsPacked {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq2_xs_packed2`].
    Iq2XsPacked2 {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq2_s_nibble`].
    Iq2SNibble {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq2_s_planes`].
    Iq2SPlanes {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq2_xxs`].
    Iq2Xxs {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
        /// How many of each group's two pairs of grid rows are loaded as pairs.
        pairs: usize,
    },
    /// [`iq2_xxs2`].
    Iq2Xxs2 {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq3_xxs`].
    Iq3Xxs {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq3_xxs2`].
    Iq3Xxs2 {
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
    /// [`iq3_s_nibble2`].
    Iq3SNibble2 {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq3_xxs_nibble`].
    Iq3XxsNibble {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq3_xxs_nibble2`].
    Iq3XxsNibble2 {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq4_xs`].
    Iq4Xs {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq4_xs2`].
    Iq4Xs2 {
        /// Groups whose products are made ahead of their sums.
        ahead: usize,
    },
    /// [`iq4_nl`].
    Iq4Nl {
        /// Blocks whose integer products are made ahead of their sums.
        ahead: usize,
    },
    /// [`q4_k`].
    Q4K {
        /// Activation vectors a row is multiplied with at once: 2 or 4.
        vectors: usize,
        /// Whether the next superblock's scales are decoded among this one's blocks.
        spread: bool,
    },
    /// [`q6_k`].
    Q6K,
    /// [`q6_k_front`].
    Q6KFront,
    /// [`bf16_pairs`].
    Bf16Pairs {
        /// Bytes ahead of each row's reads that it asks for, or none.
        ahead: i32,
        /// How close to the core.
        hint: Hint,
    },
    /// [`bf16_triples`].
    Bf16Triples {
        /// Bytes ahead of each row's reads that it asks for, or none.
        ahead: i32,
        /// How close to the core.
        hint: Hint,
    },
    /// [`bf16_two`].
    Bf16Two {
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
            "iq2_s" => {
                let ahead = take.int("ahead", 2, 1..=7)? as usize;
                Kernel::Iq2S { ahead }
            }
            "iq2_s_gather" => {
                let ahead = take.int("ahead", 2, 1..=7)? as usize;
                Kernel::Iq2SGather { ahead }
            }
            "iq2_s_packed" => {
                let ahead = take.int("ahead", 2, 1..=7)? as usize;
                Kernel::Iq2SPacked { ahead }
            }
            "iq2_s_packed2" => {
                let ahead = take.int("ahead", 2, 1..=7)? as usize;
                Kernel::Iq2SPacked2 { ahead }
            }
            "iq2_xs_packed" => {
                let ahead = take.int("ahead", 2, 1..=7)? as usize;
                Kernel::Iq2XsPacked { ahead }
            }
            "iq2_xs_packed2" => {
                let ahead = take.int("ahead", 2, 1..=7)? as usize;
                Kernel::Iq2XsPacked2 { ahead }
            }
            "iq2_s_nibble" => {
                let ahead = take.int("ahead", 2, 1..=7)? as usize;
                Kernel::Iq2SNibble { ahead }
            }
            "iq2_s_planes" => {
                let ahead = take.int("ahead", 2, 1..=7)? as usize;
                Kernel::Iq2SPlanes { ahead }
            }
            "iq2_xxs" => {
                let ahead = take.int("ahead", 3, 1..=7)? as usize;
                let pairs = take.int("pairs", 0, 0..=2)? as usize;
                Kernel::Iq2Xxs { ahead, pairs }
            }
            "iq2_xxs2" => {
                let ahead = take.int("ahead", 2, 1..=7)? as usize;
                Kernel::Iq2Xxs2 { ahead }
            }
            "iq3_xxs" => {
                let ahead = take.int("ahead", 1, 1..=7)? as usize;
                Kernel::Iq3Xxs { ahead }
            }
            "iq3_xxs2" => {
                let ahead = take.int("ahead", 1, 1..=7)? as usize;
                Kernel::Iq3Xxs2 { ahead }
            }
            "iq3_s" => {
                let ahead = take.int("ahead", 1, 1..=7)? as usize;
                Kernel::Iq3S { ahead }
            }
            "iq3_s_nibble" => {
                let ahead = take.int("ahead", 1, 1..=7)? as usize;
                Kernel::Iq3SNibble { ahead }
            }
            "iq3_s_nibble2" => {
                let ahead = take.int("ahead", 1, 1..=7)? as usize;
                Kernel::Iq3SNibble2 { ahead }
            }
            "iq3_xxs_nibble" => {
                let ahead = take.int("ahead", 1, 1..=7)? as usize;
                Kernel::Iq3XxsNibble { ahead }
            }
            "iq3_xxs_nibble2" => {
                let ahead = take.int("ahead", 1, 1..=7)? as usize;
                Kernel::Iq3XxsNibble2 { ahead }
            }
            "iq4_xs" => {
                let ahead = take.int("ahead", 2, 1..=7)? as usize;
                Kernel::Iq4Xs { ahead }
            }
            "iq4_xs2" => {
                let ahead = take.int("ahead", 3, 1..=7)? as usize;
                Kernel::Iq4Xs2 { ahead }
            }
            "iq4_nl" => {
                let ahead = take.int("ahead", 2, 1..=3)? as usize;
                Kernel::Iq4Nl { ahead }
            }
            "q4_k" => {
                let vectors = take.int("vectors", 4, 2..=4)? as usize;
                if vectors == 3 {
                    return Err("`vectors` is 2 or 4".to_owned());
                }
                let spread = take.int("spread", 1, 0..=1)? == 1;
                Kernel::Q4K { vectors, spread }
            }
            "q6_k" => Kernel::Q6K,
            "q6_k_front" => Kernel::Q6KFront,
            "bf16_pairs" => {
                let ahead = take.int("ahead", 0, 0..=4096)? as i32;
                let hint = [Hint::T0, Hint::T1, Hint::T2, Hint::Nta][take.int("hint", 0, 0..=3)? as usize];
                Kernel::Bf16Pairs { ahead, hint }
            }
            "bf16_triples" => {
                let ahead = take.int("ahead", 0, 0..=4096)? as i32;
                let hint = [Hint::T0, Hint::T1, Hint::T2, Hint::Nta][take.int("hint", 0, 0..=3)? as usize];
                Kernel::Bf16Triples { ahead, hint }
            }
            "bf16_two" => {
                let ahead = take.int("ahead", 0, 0..=4096)? as i32;
                let hint = [Hint::T0, Hint::T1, Hint::T2, Hint::Nta][take.int("hint", 0, 0..=3)? as usize];
                Kernel::Bf16Two { ahead, hint }
            }
            _ => {
                return Err(format!(
                    "no kernel `{name}`; there are `iq2_s`, `iq2_s_gather`, `iq2_s_packed`, `iq2_s_packed2`, `iq2_xs_packed`, `iq2_xs_packed2`, `iq2_s_nibble`, `iq2_s_planes`, `iq2_xxs`, `iq2_xxs2`, `iq3_xxs`, `iq3_xxs2`, `iq3_xxs_nibble`, `iq3_xxs_nibble2`, `iq3_s`, `iq3_s_nibble`, `iq3_s_nibble2`, `iq4_xs`, `iq4_xs2`, `iq4_nl`, `q4_k`, `q6_k`, `bf16_pairs`, `bf16_triples` and `bf16_two`"
                ));
            }
        };
        take.done(name)?;
        Ok(kernel)
    }

    /// Its program.
    pub fn program(&self) -> Program<Op> {
        match *self {
            Kernel::Iq2S { ahead } => iq2_s(ahead),
            Kernel::Iq2SGather { ahead } => iq2_s_gather(ahead),
            Kernel::Iq2SPacked { ahead } => iq2_s_packed(ahead),
            Kernel::Iq2SPacked2 { ahead } => iq2_s_packed2(ahead),
            Kernel::Iq2XsPacked { ahead } => iq2_xs_packed(ahead),
            Kernel::Iq2XsPacked2 { ahead } => iq2_xs_packed2(ahead),
            Kernel::Iq2SNibble { ahead } => iq2_s_nibble(ahead),
            Kernel::Iq2SPlanes { ahead } => iq2_s_planes(ahead),
            Kernel::Iq2Xxs { ahead, pairs } => iq2_xxs(ahead, pairs),
            Kernel::Iq2Xxs2 { ahead } => iq2_xxs2(ahead),
            Kernel::Iq3Xxs { ahead } => iq3_xxs(ahead),
            Kernel::Iq3Xxs2 { ahead } => iq3_xxs2(ahead),
            Kernel::Iq3S { ahead } => iq3_s(ahead),
            Kernel::Iq3SNibble { ahead } => iq3_s_nibble(ahead),
            Kernel::Iq3SNibble2 { ahead } => iq3_s_nibble2(ahead),
            Kernel::Iq3XxsNibble { ahead } => iq3_xxs_nibble(ahead),
            Kernel::Iq3XxsNibble2 { ahead } => iq3_xxs_nibble2(ahead),
            Kernel::Iq4Xs { ahead } => iq4_xs(ahead),
            Kernel::Iq4Xs2 { ahead } => iq4_xs2(ahead),
            Kernel::Iq4Nl { ahead } => iq4_nl(ahead),
            Kernel::Q4K { vectors, spread } => q4_k(vectors, spread),
            Kernel::Q6K => q6_k(),
            Kernel::Q6KFront => q6_k_front(),
            Kernel::Bf16Pairs { ahead, hint } => bf16_pairs(ahead, hint),
            Kernel::Bf16Triples { ahead, hint } => bf16_triples(ahead, hint),
            Kernel::Bf16Two { ahead, hint } => bf16_two(ahead, hint),
        }
    }

    /// The kernels whose listings are kept as snapshots.
    pub fn snapshots() -> Vec<Kernel> {
        let mut kernels: Vec<Kernel> = (1..=3).map(|ahead| Kernel::Iq2S { ahead }).collect();
        kernels.extend((1..=3).map(|ahead| Kernel::Iq2SGather { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq2SPacked { ahead }));
        kernels.extend((1..=2).map(|ahead| Kernel::Iq2SPacked2 { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq2XsPacked { ahead }));
        kernels.extend((1..=2).map(|ahead| Kernel::Iq2XsPacked2 { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq2SNibble { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq2SPlanes { ahead }));
        kernels.extend((1..=4).map(|ahead| Kernel::Iq2Xxs { ahead, pairs: 0 }));
        kernels.extend((1..=2).map(|pairs| Kernel::Iq2Xxs { ahead: 3, pairs }));
        kernels.extend((1..=2).map(|ahead| Kernel::Iq2Xxs2 { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq3Xxs { ahead }));
        kernels.extend((1..=2).map(|ahead| Kernel::Iq3Xxs2 { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq3S { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq3SNibble { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq3SNibble2 { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq3XxsNibble { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq3XxsNibble2 { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq4Xs { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq4Xs2 { ahead }));
        kernels.extend((1..=3).map(|ahead| Kernel::Iq4Nl { ahead }));
        kernels.extend([2, 4].map(|vectors| Kernel::Q4K { vectors, spread: true }));
        kernels.push(Kernel::Q4K { vectors: 4, spread: false });
        kernels.push(Kernel::Q6K);
        kernels.push(Kernel::Q6KFront);
        kernels.push(Kernel::Bf16Pairs { ahead: 0, hint: Hint::T0 });
        kernels.extend(
            [(512, Hint::T0), (1024, Hint::T0), (512, Hint::T1)].map(|(ahead, hint)| Kernel::Bf16Pairs { ahead, hint }),
        );
        kernels.extend([(0, Hint::T0), (2048, Hint::T0)].map(|(ahead, hint)| Kernel::Bf16Two { ahead, hint }));
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
