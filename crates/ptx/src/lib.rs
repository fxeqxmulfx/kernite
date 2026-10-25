//! PTX for NVIDIA GPUs, as `kgen` writes it: the adapter between `kgen-lang` and the virtual
//! instruction set that the CUDA driver compiles for whichever GPU it finds.
//!
//! - [`lang::Kernel`] is PTX for `kgen-lang`: variables of PTX's register types, and
//!   instructions by their PTX names, `ld.global.f32(x, [p + 4])`, each checked against the
//!   forms of [`forms`], the instructions PTX has for the GPUs from Pascal, `sm_61`, to
//!   Ampere, `sm_86`.
//! - [`Entry`] is a kernel written out: the program `kgen-compile` checks, and the `.entry`
//!   it prints as. [`module`] puts entries in a module for the driver to load, of the PTX
//!   version and for the GPU their instructions need.
//!
//! PTX's registers have no number, and `ptxas` allocates them, so nothing here allocates: a
//! variable is a register named after it.

pub mod forms;
pub mod lang;

use kgen_core::{Item, Program};

/// A register type of PTX, which is what a variable is declared as, and the types an
/// instruction's name ends in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ty {
    /// One bit, set by `setp`, guarding what `if p { .. }` holds.
    Pred,
    /// 8 bits of no type in particular, for `ld`, `st`, `cvt` and `mov`'s parts.
    B8,
    /// 16 bits of no type in particular.
    B16,
    /// 32 bits of no type in particular: they fit any instruction of 32 bits.
    B32,
    /// 64 bits of no type in particular.
    B64,
    /// 128 bits, for `ld`, `st` and `mov`.
    B128,
    /// An unsigned integer of 8 bits.
    U8,
    /// An unsigned integer of 16 bits.
    U16,
    /// An unsigned integer of 32 bits.
    U32,
    /// An unsigned integer of 64 bits, an address among them.
    U64,
    /// A signed integer of 8 bits.
    S8,
    /// A signed integer of 16 bits.
    S16,
    /// A signed integer of 32 bits.
    S32,
    /// A signed integer of 64 bits.
    S64,
    /// A float of 16 bits, IEEE's half.
    F16,
    /// Two `f16` in 32 bits, the first in the low half.
    F16x2,
    /// A float of 32 bits.
    F32,
    /// A float of 64 bits.
    F64,
    /// A float of 16 bits with `f32`'s exponent; PTX keeps it in a `b16` register.
    Bf16,
    /// Two `bf16` in 32 bits, in a `b32` register.
    Bf16x2,
    /// `f32` with the precision of `f16`, as the tensor cores read it, in a `b32` register.
    Tf32,
}

/// What a type is, for which types it fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A predicate.
    Pred,
    /// Bits of no type in particular.
    Bit,
    /// An unsigned integer.
    Unsigned,
    /// A signed integer.
    Signed,
    /// A float, or floats packed together.
    Float,
}

impl Ty {
    /// Every type, in the order an entry declares its registers.
    pub const ALL: [Ty; 21] = [
        Ty::Pred,
        Ty::B8,
        Ty::U8,
        Ty::S8,
        Ty::B16,
        Ty::U16,
        Ty::S16,
        Ty::F16,
        Ty::Bf16,
        Ty::B32,
        Ty::U32,
        Ty::S32,
        Ty::F32,
        Ty::F16x2,
        Ty::Bf16x2,
        Ty::Tf32,
        Ty::B64,
        Ty::U64,
        Ty::S64,
        Ty::F64,
        Ty::B128,
    ];

    /// The type PTX writes `name`, if it is one.
    pub fn parse(name: &str) -> Option<Ty> {
        Ty::ALL.into_iter().find(|t| t.name() == name)
    }

    /// Its name, as PTX writes it after a dot.
    pub fn name(self) -> &'static str {
        match self {
            Ty::Pred => "pred",
            Ty::B8 => "b8",
            Ty::B16 => "b16",
            Ty::B32 => "b32",
            Ty::B64 => "b64",
            Ty::B128 => "b128",
            Ty::U8 => "u8",
            Ty::U16 => "u16",
            Ty::U32 => "u32",
            Ty::U64 => "u64",
            Ty::S8 => "s8",
            Ty::S16 => "s16",
            Ty::S32 => "s32",
            Ty::S64 => "s64",
            Ty::F16 => "f16",
            Ty::F16x2 => "f16x2",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            Ty::Bf16 => "bf16",
            Ty::Bf16x2 => "bf16x2",
            Ty::Tf32 => "tf32",
        }
    }

    /// The type its register is declared as: PTX has no `bf16`, `bf16x2` or `tf32`
    /// registers, and keeps them in bits of their size.
    pub fn decl(self) -> Ty {
        match self {
            Ty::Bf16 => Ty::B16,
            Ty::Bf16x2 | Ty::Tf32 => Ty::B32,
            t => t,
        }
    }

    /// Bits: 1 for a predicate.
    pub fn bits(self) -> u32 {
        match self {
            Ty::Pred => 1,
            Ty::B8 | Ty::U8 | Ty::S8 => 8,
            Ty::B16 | Ty::U16 | Ty::S16 | Ty::F16 | Ty::Bf16 => 16,
            Ty::B32 | Ty::U32 | Ty::S32 | Ty::F32 | Ty::F16x2 | Ty::Bf16x2 | Ty::Tf32 => 32,
            Ty::B64 | Ty::U64 | Ty::S64 | Ty::F64 => 64,
            Ty::B128 => 128,
        }
    }

    /// What it is.
    pub fn kind(self) -> Kind {
        match self {
            Ty::Pred => Kind::Pred,
            Ty::B8 | Ty::B16 | Ty::B32 | Ty::B64 | Ty::B128 => Kind::Bit,
            Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64 => Kind::Unsigned,
            Ty::S8 | Ty::S16 | Ty::S32 | Ty::S64 => Kind::Signed,
            Ty::F16 | Ty::F16x2 | Ty::F32 | Ty::F64 | Ty::Bf16 | Ty::Bf16x2 | Ty::Tf32 => Kind::Float,
        }
    }

    /// Whether it is an integer, signed or not.
    pub fn int(self) -> bool {
        matches!(self.kind(), Kind::Unsigned | Kind::Signed)
    }

    /// Whether a register of this type is an operand an instruction of type `want` takes, as
    /// PTX checks most instructions: one of the same size that is the same type, or untyped
    /// bits, or an integer where an integer is wanted.
    pub fn fits(self, want: Ty) -> bool {
        if self == want {
            return true;
        }
        if self.bits() != want.bits() || self == Ty::Pred || want == Ty::Pred {
            return false;
        }
        match (self.kind(), want.kind()) {
            (Kind::Bit, _) | (_, Kind::Bit) => true,
            (a, b) => a != Kind::Float && b != Kind::Float,
        }
    }

    /// Whether it fits as PTX checks `ld`, `st` and `cvt`: a register may be wider than the
    /// instruction's type, and bits fit any type of no more bits, an integer bits and
    /// integers, a float bits and itself. Not for `bf16`, `bf16x2` and `tf32`, which take
    /// [`Ty::fits`].
    pub fn fits_wide(self, want: Ty) -> bool {
        if matches!(want, Ty::Bf16 | Ty::Bf16x2 | Ty::Tf32) || matches!(self, Ty::Bf16 | Ty::Bf16x2 | Ty::Tf32) {
            return self.fits(want);
        }
        if self == Ty::Pred || want == Ty::Pred {
            return self == want;
        }
        if self.bits() < want.bits() {
            return false;
        }
        match (self.kind(), want.kind()) {
            (Kind::Bit, _) => true,
            (Kind::Unsigned | Kind::Signed, k) => k != Kind::Float,
            (Kind::Float, Kind::Bit) => true,
            (Kind::Float, Kind::Float) => self == want,
            _ => false,
        }
    }
}

/// What a kernel needs of the PTX it is written in and of the GPU it runs on: the least it
/// needs is [`Needs::FLOOR`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Needs {
    /// The version of PTX, as ten times its major and its minor: 78 for 7.8. 0 for any.
    pub ptx: u32,
    /// The GPU, by its `sm_` number: 80 for Ampere's first. 0 for any.
    pub sm: u32,
    /// A GPU older than this, by its `sm_` number: `shfl` without `.sync` is gone from
    /// Volta's on, `sm_70`. 0 for any.
    pub below: u32,
}

impl Needs {
    /// What every module is written for: PTX 7.5, which every driver of CUDA 11.5 and later
    /// reads, and Pascal, `sm_61`, the oldest GPU kgen writes for.
    pub const FLOOR: Needs = Needs { ptx: 75, sm: 61, below: 0 };

    /// What both need.
    pub fn and(self, other: Needs) -> Needs {
        let below = match (self.below, other.below) {
            (0, b) | (b, 0) => b,
            (a, b) => a.min(b),
        };
        Needs { ptx: self.ptx.max(other.ptx), sm: self.sm.max(other.sm), below }
    }

    /// Whether some GPU has it all.
    pub fn possible(self) -> bool {
        self.below == 0 || self.sm.max(Needs::FLOOR.sm) < self.below
    }

    /// The version of PTX as a module's `.version` writes it: `7.8`.
    pub fn version(self) -> String {
        let v = self.ptx.max(Needs::FLOOR.ptx);
        format!("{}.{}", v / 10, v % 10)
    }

    /// The GPU as a module's `.target` writes it: `sm_80`.
    pub fn target(self) -> String {
        format!("sm_{}", self.sm.max(Needs::FLOOR.sm))
    }
}

/// A PTX instruction as it is written, guard, operands and all; the operands of the
/// [`kgen_core::Inst`] it is the opcode of say what it reads and writes, for the checks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Op(pub String);

/// A kernel's parameter: the type it is passed as, and the register it is loaded into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    /// The type of its register; the `.param` is of [`Param::passed`].
    pub ty: Ty,
    /// The variable it is loaded into, by [`kgen_core::VReg::id`].
    pub reg: usize,
    /// Whether it is a pointer, to be made a global address.
    pub pointer: bool,
}

impl Param {
    /// The type it is passed as, and loaded by `ld.param`: its register's, but bits for a
    /// float of 16, which `ld` has no type for.
    pub fn passed(&self) -> Ty {
        match self.ty {
            Ty::F16 => Ty::B16,
            t => t.decl(),
        }
    }
}

/// A kernel written in PTX.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The kernel as a program, for the checks: each instruction's opcode is its PTX.
    pub program: Program<Op>,
    /// Its parameters, in order.
    pub params: Vec<Param>,
    /// Each variable's register, its name unique in the entry, and its type, by
    /// [`kgen_core::VReg::id`].
    pub regs: Vec<(String, Ty)>,
    /// What it needs, above [`Needs::FLOOR`], and the first instruction or type that needs
    /// it, for when another kernel of the module needs what cannot go with it.
    pub needs: Vec<(Needs, String)>,
}

impl Entry {
    /// What it needs altogether.
    pub fn needs(&self) -> Needs {
        self.needs.iter().fold(Needs::default(), |n, (m, _)| n.and(*m))
    }

    /// The `.entry`: its parameters, its registers, the loads of its parameters into them,
    /// and its body.
    pub fn text(&self) -> String {
        let name = &self.program.name;
        let mut out = format!(".visible .entry {name}(");
        for (i, p) in self.params.iter().enumerate() {
            out += if i == 0 { "\n" } else { ",\n" };
            out += &format!("\t.param .{} {name}_param_{i}", p.passed().name());
        }
        out += "\n)\n{\n";
        for ty in Ty::ALL.into_iter().filter(|t| t.decl() == *t) {
            let regs: Vec<String> =
                self.regs.iter().filter(|(_, t)| t.decl() == ty).map(|(reg, _)| format!("%{reg}")).collect();
            for some in regs.chunks(8) {
                out += &format!("\t.reg .{} {};\n", ty.name(), some.join(", "));
            }
        }
        out += "\n";
        for (i, p) in self.params.iter().enumerate() {
            let reg = &self.regs[p.reg].0;
            out += &format!("\tld.param.{} %{reg}, [{name}_param_{i}];\n", p.passed().name());
            if p.pointer {
                out += &format!("\tcvta.to.global.u64 %{reg}, %{reg};\n");
            }
        }
        for item in &self.program.body {
            match item {
                Item::Label(l) => out += &format!("$L{}:\n", l.0),
                Item::Note(note) => out += &format!("\t// {}\n", note.replace('\n', " ")),
                Item::Inst(inst) => out += &format!("\t{}\n", inst.op.0),
            }
        }
        out + "}\n"
    }
}

/// The PTX module of `entries`, of the version and for the GPU the instructions they use
/// need, and no newer: [`Needs::FLOOR`] if they need nothing more. The driver compiles it for
/// the GPU it runs on, which may be any as new as its `.target` or newer, unless an
/// instruction is gone from newer ones.
///
/// It fails if one kernel needs a GPU older than another kernel needs: PTX has no module
/// for both.
pub fn module(entries: &[Entry]) -> Result<String, String> {
    let needs = entries.iter().fold(Needs::default(), |n, e| n.and(e.needs()));
    if !needs.possible() {
        let all = || entries.iter().flat_map(|e| e.needs.iter().map(move |(n, why)| (n, why, &e.program.name)));
        let old = all().find(|(n, ..)| n.below == needs.below);
        let new = all().filter(|(n, ..)| n.sm >= needs.below).max_by_key(|(n, ..)| n.sm);
        if let (Some((_, a, ka)), Some((n, b, kb))) = (old, new) {
            return Err(format!(
                "`{a}` in `{ka}` is for GPUs older than sm_{}, and `{b}` in `{kb}` needs sm_{} or newer: one module \
                 cannot have both",
                needs.below, n.sm
            ));
        }
    }
    let mut out =
        format!("// Written by kgen.\n\n.version {}\n.target {}\n.address_size 64\n", needs.version(), needs.target());
    for e in entries {
        out += "\n";
        out += &e.text();
    }
    Ok(out)
}
