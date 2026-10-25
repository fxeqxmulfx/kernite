use crate::operand::Arg;
use crate::reg::{Class, PReg, Register, VReg};

/// A place in the body that jumps can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label(pub u32);

/// A slot of the stack frame, by its index in [`Program::slots`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlotId(pub u32);

/// A constant, by its index in [`Program::consts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstId(pub u32);

/// Where control goes after an instruction.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Flow {
    /// To the next item.
    Next,
    /// To the label.
    Jump(Label),
    /// To the label or to the next item, as the flags decide.
    Branch(Label),
    /// To one of the labels, as a value decides, through a table: PTX's `brx.idx`.
    Table {
        /// The labels.
        to: Vec<Label>,
        /// Whether it may go to the next item instead, as where a predicate it runs under
        /// does not hold.
        or_next: bool,
    },
    /// Out of the kernel, with its result where [`Program::results`] puts it.
    Return,
}

/// One machine instruction: an opcode of the instruction set `Op`, its operands and where
/// control goes after it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Inst<Op, R> {
    /// The opcode.
    pub op: Op,
    /// The operands, in the order the instruction set writes them.
    pub args: Vec<Arg<R>>,
    /// Operands it uses without writing them out: the value a `ret` hands back.
    pub implicit: Vec<Arg<R>>,
    /// Where control goes next.
    pub flow: Flow,
}

impl<Op, R: Register> Inst<Op, R> {
    /// Every operand, written out or implicit.
    pub fn operands(&self) -> impl Iterator<Item = &Arg<R>> + '_ {
        self.args.iter().chain(&self.implicit)
    }

    /// The registers it reads, in operand order; a register read twice shows up twice.
    pub fn reads(&self) -> impl Iterator<Item = R> + '_ {
        self.operands().flat_map(Arg::reads)
    }

    /// The registers it writes.
    pub fn writes(&self) -> impl Iterator<Item = R> + '_ {
        self.operands().filter_map(Arg::writes)
    }
}

/// An element of a body.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Item<Op, R> {
    /// A jump target.
    Label(Label),
    /// An instruction.
    Inst(Inst<Op, R>),
    /// A remark for whoever reads the listing. No code.
    Note(String),
}

/// What a variable is: the file it takes its register from, and the name diagnostics and
/// listings call it by.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RegInfo {
    /// The file.
    pub class: Class,
    /// A name for people.
    pub name: String,
}

/// Stack memory a kernel asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot {
    /// Bytes.
    pub size: u32,
    /// Alignment, a power of two.
    pub align: u32,
}

/// Read-only data a kernel reads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Const {
    /// The bytes.
    pub bytes: Vec<u8>,
    /// Alignment, a power of two.
    pub align: u32,
}

/// What a kernel takes and gives, as the files the calling convention passes them in.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    /// The parameters, in order.
    pub params: Vec<Class>,
    /// The result, if there is one.
    pub result: Option<Class>,
}

/// A kernel before allocation: variables, not registers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program<Op> {
    /// What it is called.
    pub name: String,
    /// What it takes and gives.
    pub signature: Signature,
    /// Every variable, by [`VReg::id`].
    pub regs: Vec<RegInfo>,
    /// The parameters, in order, each with the register it arrives in.
    pub params: Vec<(VReg, PReg)>,
    /// The values returned, each with the register it must be in at its return.
    ///
    /// Parameters and results keep their registers throughout.
    pub results: Vec<(VReg, PReg)>,
    /// The code.
    pub body: Vec<Item<Op, VReg>>,
    /// Stack memory, by [`SlotId`].
    pub slots: Vec<Slot>,
    /// Read-only data, by [`ConstId`].
    pub consts: Vec<Const>,
}

/// The stack frame of an allocated kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Frame {
    /// The callee-saved registers the body writes, to save on entry and restore at every
    /// return, in the order they are saved.
    pub saved: Vec<PReg>,
    /// Each slot's offset from the stack pointer once it is aligned.
    pub offsets: Vec<u32>,
    /// Bytes the slots take together, a multiple of `align`; 0 with no slots.
    pub size: u32,
    /// The alignment the slots need; 0 with no slots.
    pub align: u32,
}

/// A kernel after allocation: machine registers only, and a frame laid out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compiled<Op> {
    /// What it is called.
    pub name: String,
    /// What it takes and gives.
    pub signature: Signature,
    /// The code.
    pub body: Vec<Item<Op, PReg>>,
    /// Its stack frame.
    pub frame: Frame,
    /// Read-only data, by [`ConstId`].
    pub consts: Vec<Const>,
    /// Every variable of the program and the register it got, by [`VReg::id`]; `None` for
    /// one that is never used.
    pub assignment: Vec<(RegInfo, Option<PReg>)>,
}
