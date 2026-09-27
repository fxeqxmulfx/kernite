use crate::program::{ConstId, Label, SlotId};
use crate::reg::Register;

/// How many bits an operand covers: which view of a register, or how much memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Width {
    /// A byte.
    W8,
    /// A word.
    W16,
    /// A doubleword.
    W32,
    /// A quadword.
    W64,
    /// 128 bits.
    W128,
    /// 256 bits.
    W256,
}

impl Width {
    /// The bytes it covers.
    pub fn bytes(self) -> u32 {
        match self {
            Width::W8 => 1,
            Width::W16 => 2,
            Width::W32 => 4,
            Width::W64 => 8,
            Width::W128 => 16,
            Width::W256 => 32,
        }
    }
}

/// Where a memory operand's address starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Base<R> {
    /// A register.
    Reg(R),
    /// A slot of the stack frame, whose place the frame's layout decides.
    Slot(SlotId),
    /// A constant of the program, whose place the machine decides.
    Const(ConstId),
}

/// A memory operand: `width` bits at `base + index * scale + disp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mem<R> {
    /// Where the address starts.
    pub base: Base<R>,
    /// A register and the scale it is multiplied by, 1, 2, 4 or 8.
    pub index: Option<(R, u8)>,
    /// Bytes added.
    pub disp: i32,
    /// How much is read or written.
    pub width: Width,
}

/// An immediate. `hex` only says how it reads best: a shuffle's control byte in hex, a count
/// in decimal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Imm {
    /// The value.
    pub value: i64,
    /// Whether a listing shows it in hex.
    pub hex: bool,
}

/// An instruction's operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operand<R> {
    /// A register, seen at a width: `eax` of `rax`, `xmm3` of `ymm3`.
    Reg(R, Width),
    /// Memory.
    Mem(Mem<R>),
    /// A constant in the instruction.
    Imm(Imm),
    /// A jump's target.
    Label(Label),
}

/// What an instruction does with an operand's register. The registers that form a memory
/// operand's address are read whatever the access says, which is about the memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    /// Reads it.
    Read,
    /// Replaces all of it.
    Write,
    /// Reads it, then writes it: a two-operand instruction's destination, or a write that
    /// keeps part of what was there.
    ReadWrite,
    /// Names it without depending on it: the sources of `vpxor x, x, x`, whose result is
    /// zero whatever `x` held.
    Ignored,
}

/// An operand and what the instruction does with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Arg<R> {
    /// The operand.
    pub operand: Operand<R>,
    /// What happens to it.
    pub access: Access,
}

impl<R: Register> Arg<R> {
    /// The registers it reads: a read register, or the registers of an address.
    pub fn reads(&self) -> impl Iterator<Item = R> {
        let (first, second) = match self.operand {
            Operand::Reg(r, _) if matches!(self.access, Access::Read | Access::ReadWrite) => (Some(r), None),
            Operand::Mem(m) => (
                match m.base {
                    Base::Reg(r) => Some(r),
                    Base::Slot(_) | Base::Const(_) => None,
                },
                m.index.map(|(r, _)| r),
            ),
            _ => (None, None),
        };
        first.into_iter().chain(second)
    }

    /// Every register it names, whatever happens to it.
    pub fn regs(&self) -> impl Iterator<Item = R> {
        let (first, second) = match self.operand {
            Operand::Reg(r, _) => (Some(r), None),
            Operand::Mem(m) => (
                match m.base {
                    Base::Reg(r) => Some(r),
                    Base::Slot(_) | Base::Const(_) => None,
                },
                m.index.map(|(r, _)| r),
            ),
            Operand::Imm(_) | Operand::Label(_) => (None, None),
        };
        first.into_iter().chain(second)
    }

    /// The register it writes, if it writes one.
    pub fn writes(&self) -> Option<R> {
        match self.operand {
            Operand::Reg(r, _) if matches!(self.access, Access::Write | Access::ReadWrite) => Some(r),
            _ => None,
        }
    }

    /// The same argument with every register replaced by `f` of it.
    pub fn map<S>(self, mut f: impl FnMut(R) -> S) -> Arg<S> {
        let operand = match self.operand {
            Operand::Reg(r, w) => Operand::Reg(f(r), w),
            Operand::Mem(m) => Operand::Mem(Mem {
                base: match m.base {
                    Base::Reg(r) => Base::Reg(f(r)),
                    Base::Slot(s) => Base::Slot(s),
                    Base::Const(c) => Base::Const(c),
                },
                index: m.index.map(|(r, scale)| (f(r), scale)),
                disp: m.disp,
                width: m.width,
            }),
            Operand::Imm(i) => Operand::Imm(i),
            Operand::Label(l) => Operand::Label(l),
        };
        Arg { operand, access: self.access }
    }
}
