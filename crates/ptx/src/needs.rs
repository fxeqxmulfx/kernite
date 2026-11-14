//! What a kernel needs of the PTX it is written in and of the GPU it runs on.

/// What a kernel needs of the PTX it is written in and of the GPU it runs on, as `ptxas` has
/// it; a module is of [`Needs::FLOOR`] at the least.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Needs {
    /// The version of PTX, as ten times its major and its minor: 78 for 7.8. 0 for any.
    pub ptx: u32,
    /// The GPU, by its `sm_` number: 80 for Ampere's first. 0 for any.
    pub sm: u32,
}

/// The GPUs kernite writes for, by their `sm_` numbers, and the version of PTX that came with
/// each, which a module for it is of at least.
pub const GPUS: [(u32, u32); 15] = [
    (30, 30),
    (32, 40),
    (35, 31),
    (37, 41),
    (50, 40),
    (52, 41),
    (53, 42),
    (60, 50),
    (61, 50),
    (62, 50),
    (70, 60),
    (72, 61),
    (75, 63),
    (80, 70),
    (86, 71),
];

impl Needs {
    /// What every module is written for: Kepler, `sm_30`, and PTX 3.0, which came with it. The
    /// driver starts at most 65535 blocks along x of a module written for an older GPU, Fermi's
    /// `sm_20` or `sm_21`, whatever the GPU it runs on, and 2^31 - 1 of one written for Kepler.
    pub const FLOOR: Needs = Needs { ptx: 30, sm: 30 };

    /// What both need: the newer version of PTX, and the newer GPU.
    pub fn and(self, other: Needs) -> Needs {
        Needs { ptx: self.ptx.max(other.ptx), sm: self.sm.max(other.sm) }
    }

    /// The version of PTX a module of it is of, as a module's `.version` writes it: `7.8`. That
    /// is what it needs, and what its GPU came with.
    pub fn version(self) -> String {
        let sm = self.sm.max(Needs::FLOOR.sm);
        let came = GPUS.iter().find(|&&(g, _)| g == sm).map_or(0, |&(_, v)| v);
        let v = self.ptx.max(came).max(Needs::FLOOR.ptx);
        format!("{}.{}", v / 10, v % 10)
    }

    /// The GPU as a module's `.target` writes it: `sm_80`.
    pub fn target(self) -> String {
        format!("sm_{}", self.sm.max(Needs::FLOOR.sm))
    }
}
