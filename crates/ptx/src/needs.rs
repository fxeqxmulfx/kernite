//! What a kernel needs of the PTX it is written in and of the GPU it runs on.

/// What a kernel needs of the PTX it is written in and of the GPU it runs on, as `ptxas` has
/// it; a module is of [`Needs::FLOOR`] at the least.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Needs {
    /// The version of PTX, as ten times its major and its minor: 78 for 7.8. 0 for any.
    pub ptx: u32,
    /// The GPU, by its `sm_` number: 80 for Ampere's first. 0 for any.
    pub sm: u32,
    /// A GPU older than this, by its `sm_` number, in a module of PTX [`Needs::since`] or
    /// newer: `shfl` without `.sync` is for none from Volta's on, `sm_70`, from PTX 6.4. 0 for
    /// any.
    pub below: u32,
    /// The version of PTX from which [`Needs::below`] holds, as [`Needs::ptx`] writes it.
    pub since: u32,
}

/// The GPUs kernite writes for, by their `sm_` numbers, and the version of PTX that came with
/// each, which a module for it is of at least.
pub const GPUS: [(u32, u32); 17] = [
    (20, 20),
    (21, 20),
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
    /// What every module is written for: PTX 2.3, the first with `.address_size 64`, which
    /// every module has, and Fermi, `sm_20`, the oldest GPU the driver compiles PTX for.
    pub const FLOOR: Needs = Needs { ptx: 23, sm: 20, below: 0, since: 0 };

    /// What both need, and the narrower of the GPUs they are for.
    pub fn and(self, other: Needs) -> Needs {
        let least = |a: u32, b: u32| match (a, b) {
            (0, x) | (x, 0) => x,
            (a, b) => a.min(b),
        };
        Needs {
            ptx: self.ptx.max(other.ptx),
            sm: self.sm.max(other.sm),
            below: least(self.below, other.below),
            since: least(self.since, other.since),
        }
    }

    /// Whether some module has it all: one for a GPU older than [`Needs::below`], or of a
    /// version older than [`Needs::since`].
    pub fn possible(self) -> bool {
        self.below == 0 || self.sm.max(Needs::FLOOR.sm) < self.below || self.written() < self.since
    }

    /// The version of PTX a module of it is of, as [`Needs::ptx`] writes it: what it needs,
    /// and what its GPU does.
    pub fn written(self) -> u32 {
        let sm = self.sm.max(Needs::FLOOR.sm);
        let came = GPUS.iter().find(|&&(g, _)| g == sm).map_or(0, |&(_, v)| v);
        self.ptx.max(came).max(Needs::FLOOR.ptx)
    }

    /// The version of PTX as a module's `.version` writes it: `7.8`.
    pub fn version(self) -> String {
        dotted(self.written())
    }

    /// The GPU as a module's `.target` writes it: `sm_80`.
    pub fn target(self) -> String {
        format!("sm_{}", self.sm.max(Needs::FLOOR.sm))
    }
}

/// A version of PTX as [`Needs::ptx`] writes it, as PTX does: `7.8`.
fn dotted(v: u32) -> String {
    format!("{}.{}", v / 10, v % 10)
}

/// Why no module has all of `needs`, each what an instruction needs and that instruction as
/// it is to be named: one is for GPUs older than another needs, in the version of PTX a
/// module for that GPU, or for a third, is of.
pub fn clash(needs: &[(Needs, String)]) -> String {
    let all = needs.iter().fold(Needs::default(), |n, (m, _)| n.and(*m));
    let (old, o) = needs.iter().find(|(n, _)| n.below == all.below).expect("one is for older GPUs");
    let first = format!("{o} is for GPUs older than sm_{} from PTX {} on", old.below, dotted(old.since));
    let gpu = needs.iter().max_by_key(|(n, _)| n.sm).expect("one needs a newer GPU");
    let ptx = needs.iter().max_by_key(|(n, _)| n.ptx).expect("one needs the version");
    let ((new, g), (v, w)) = (gpu, ptx);
    let came = Needs { sm: new.sm, ..Needs::default() }.written();
    if came >= all.since {
        format!("{first}, and {g} needs sm_{}, which PTX has from {} on: no module has both", new.sm, dotted(came))
    } else if std::ptr::eq(gpu, ptx) {
        format!("{first}, and {g} needs sm_{} and PTX {}: no module has both", new.sm, dotted(v.ptx))
    } else {
        format!("{first}; {g} needs sm_{}, and {w} PTX {}: no module has them all", new.sm, dotted(v.ptx))
    }
}
