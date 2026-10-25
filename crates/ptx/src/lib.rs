//! PTX for NVIDIA GPUs, as `kgen` writes it: the adapter between `kgen-lang` and the virtual
//! instruction set that the CUDA driver compiles for whichever GPU it finds.
//!
//! - [`lang::Kernel`] is PTX for `kgen-lang`: variables of PTX's register types, and
//!   instructions by their PTX names, `ld.global.f32(x, [p + 4])`, each checked against the
//!   types its name gives its operands.
//! - [`Entry`] is a kernel written out: the program `kgen-compile` checks, and the `.entry`
//!   it prints as. [`module`] puts entries in a module for the driver to load.
//!
//! PTX's registers have no number, and `ptxas` allocates them, so nothing here allocates: a
//! variable is a register named after it.

pub mod lang;

use kgen_core::{Item, Program};

/// A register type of PTX, which is what a variable is declared as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ty {
    /// One bit, set by `setp`, guarding what `if p { .. }` holds.
    Pred,
    /// 32 bits, of no type in particular: they fit any instruction of 32 bits.
    B32,
    /// An unsigned integer of 32 bits.
    U32,
    /// A signed integer of 32 bits.
    S32,
    /// A float of 32 bits.
    F32,
    /// 64 bits of no type in particular.
    B64,
    /// An unsigned integer of 64 bits, an address among them.
    U64,
    /// A signed integer of 64 bits.
    S64,
}

impl Ty {
    /// Every type, in the order an entry declares its registers.
    pub const ALL: [Ty; 8] = [Ty::Pred, Ty::B32, Ty::U32, Ty::S32, Ty::F32, Ty::B64, Ty::U64, Ty::S64];

    /// The type PTX writes `name`, if it is one.
    pub fn parse(name: &str) -> Option<Ty> {
        Ty::ALL.into_iter().find(|t| t.name() == name)
    }

    /// Its name, as PTX writes it after a dot.
    pub fn name(self) -> &'static str {
        match self {
            Ty::Pred => "pred",
            Ty::B32 => "b32",
            Ty::U32 => "u32",
            Ty::S32 => "s32",
            Ty::F32 => "f32",
            Ty::B64 => "b64",
            Ty::U64 => "u64",
            Ty::S64 => "s64",
        }
    }

    /// Bits: 1 for a predicate.
    pub fn bits(self) -> u32 {
        match self {
            Ty::Pred => 1,
            Ty::B32 | Ty::U32 | Ty::S32 | Ty::F32 => 32,
            Ty::B64 | Ty::U64 | Ty::S64 => 64,
        }
    }

    /// Whether a register of this type is an operand an instruction of type `want` takes:
    /// one of the same size that is the same type, or untyped bits, or an integer where an
    /// integer is wanted.
    pub fn fits(self, want: Ty) -> bool {
        if self == want {
            return true;
        }
        if self.bits() != want.bits() || self == Ty::Pred || want == Ty::Pred {
            return false;
        }
        matches!(self, Ty::B32 | Ty::B64) || matches!(want, Ty::B32 | Ty::B64) || (self != Ty::F32 && want != Ty::F32)
    }
}

/// A PTX instruction as it is written, guard, operands and all; the operands of the
/// [`kgen_core::Inst`] it is the opcode of say what it reads and writes, for the checks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Op(pub String);

/// A kernel's parameter: the type it is passed as, and the register it is loaded into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    /// The type of the `.param`, and of its register.
    pub ty: Ty,
    /// The variable it is loaded into, by [`kgen_core::VReg::id`].
    pub reg: usize,
    /// Whether it is a pointer, to be made a global address.
    pub pointer: bool,
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
}

impl Entry {
    /// The `.entry`: its parameters, its registers, the loads of its parameters into them,
    /// and its body.
    pub fn text(&self) -> String {
        let name = &self.program.name;
        let mut out = format!(".visible .entry {name}(");
        for (i, p) in self.params.iter().enumerate() {
            out += if i == 0 { "\n" } else { ",\n" };
            out += &format!("\t.param .{} {name}_param_{i}", p.ty.name());
        }
        out += "\n)\n{\n";
        for ty in Ty::ALL {
            let regs: Vec<String> =
                self.regs.iter().filter(|(_, t)| *t == ty).map(|(reg, _)| format!("%{reg}")).collect();
            for some in regs.chunks(8) {
                out += &format!("\t.reg .{} {};\n", ty.name(), some.join(", "));
            }
        }
        out += "\n";
        for (i, p) in self.params.iter().enumerate() {
            let reg = &self.regs[p.reg].0;
            out += &format!("\tld.param.{} %{reg}, [{name}_param_{i}];\n", p.ty.name());
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

/// The PTX module of `entries`. It targets `sm_75`, the oldest GPU the CUDA 13 compilers
/// know, and the driver compiles it for the one it runs on.
pub fn module(entries: &[Entry]) -> String {
    let mut out = String::from("// Written by kgen.\n\n.version 7.5\n.target sm_75\n.address_size 64\n");
    for e in entries {
        out += "\n";
        out += &e.text();
    }
    out
}
