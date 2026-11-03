//! The language's port: [`Target`], which an instruction set implements, and what crosses it.

use std::fmt;

use crate::token::Span;

/// An instruction set, as the evaluator sees it: variables of the machine's types, and
/// instructions by name. It checks what it is given and builds the program; the evaluator
/// knows no machine.
pub trait Target {
    /// A variable, seen as one of the machine's types: `ymm`, `v.xmm`, `u32`.
    type Var: Copy + PartialEq + fmt::Debug;

    /// The function's next parameter, of the type `ty` as Rust writes it: Rust's, `*const f32` or
    /// `usize`, or the machine's. [`pointer`](crate::pointer) says whether it is a pointer.
    fn param(&mut self, name: &str, ty: &str) -> Result<Self::Var, String>;

    /// What the function returns, of the type `ty` as written: `()`, `!` for never, Rust's or
    /// the machine's.
    fn result(&mut self, ty: &str) -> Result<(), String>;

    /// A new variable of the machine's type `ty`.
    fn var(&mut self, name: &str, ty: &str) -> Result<Self::Var, String>;

    /// The module's memory `name`, which a `static` of the module declares, as a variable, the
    /// first time the function names it. A machine has none unless it says so.
    fn global(&mut self, name: &str) -> Result<Self::Var, String> {
        Err(format!(
            "`{name}` is memory of the module's, and this machine has none: its kernels take memory by pointer"
        ))
    }

    /// `var` seen as `view`: a field, as in `v.xmm`, or the type of a helper's parameter.
    fn view(&mut self, var: Self::Var, view: &str) -> Result<Self::Var, String>;

    /// `var.field`: `var` seen as `field`, unless the machine has parts of a variable too, as
    /// PTX selects a byte of one, `r.b0`.
    fn field(&mut self, var: Self::Var, field: &str) -> Result<Self::Var, String> {
        self.view(var, field)
    }

    /// Places label `label` here. The evaluator numbers labels, each once.
    fn label(&mut self, label: u32);

    /// A remark for whoever reads the listing.
    fn note(&mut self, text: &str);

    /// The instruction `name` over `args`, run only where `guard` holds if there is one.
    fn inst(&mut self, name: &str, args: &[Operand<Self::Var>], guard: Option<Guard<Self::Var>>) -> Result<(), Fault>;

    /// The end of the body, for whatever the machine puts there.
    fn end(&mut self) -> Result<(), String>;

    /// `var`, a parameter or a variable the body declares, is declared at `at`: where an error
    /// the machine finds with it later is placed. A machine need not keep it.
    fn declared_at(&mut self, _var: Self::Var, _at: Span) {}

    /// The instruction that `inst` is given next is at `at`: where an error the machine finds
    /// with it later is placed. A machine need not keep it.
    fn inst_at(&mut self, _at: Span) {}
}

/// What an instruction is given.
#[derive(Clone, Debug, PartialEq)]
pub enum Operand<V> {
    /// A variable.
    Var(V),
    /// A name that is nothing in scope, for a machine that has registers of its own by name:
    /// `tid.x`. `_` is one, for an operand the machine lets be left out.
    Special(String),
    /// An integer known now, from -2^63 to 2^64 - 1, as the 64 bits of an `i64` or a `u64` are.
    Int(i128),
    /// An `f32` known now.
    F32(f32),
    /// An `f64` known now.
    F64(f64),
    /// An `f16` known now, as its bits.
    F16(u16),
    /// A `bf16` known now, as its bits.
    Bf16(u16),
    /// Memory at an address: `[x + i * 4 + 16]`.
    Mem(Mem<V>),
    /// Memory holding a value known now, as bytes: `[1.0]`, `[[0.5; 8]]`.
    Data(Vec<u8>),
    /// Memory, `Mem` or `Data`, after a name that is nothing in scope, for a machine that names
    /// how much of it an instruction reaches, as x86 does: `dword[x + 4]`.
    Sized(String, Box<Operand<V>>),
    /// A label, for a jump.
    Label(u32),
    /// A function of the module that names an ABI, by its name: one the machine calls, as
    /// PTX's `call` does, or whose address it takes.
    Func(String),
    /// The code of a function of the module that names an ABI, `[f]`: memory whose address x86's
    /// `lea` takes.
    Code(String),
    /// Operands taken together: `(a, b, c, d)`, or variables declared together, `acc`.
    Vector(Vec<Operand<V>>),
    /// `a | b`: two operands in the place of one, as the result and the predicate PTX's `setp`
    /// writes.
    Pair(Box<Operand<V>>, Box<Operand<V>>),
    /// `!p`: an operand read inverted.
    Not(Box<Operand<V>>),
    /// `-a`: an operand read negated.
    Neg(Box<Operand<V>>),
    /// `[a, b, ..]`: operands in brackets, more than one, as a texture's handle and where it
    /// is read.
    List(Vec<Operand<V>>),
}

/// An address: `base + index * scale + disp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mem<V> {
    /// The variable added as it is.
    pub base: Option<V>,
    /// The variable multiplied, and by what.
    pub index: Option<(V, i64)>,
    /// Bytes added.
    pub disp: i64,
}

/// The predicate an instruction runs under: `if p { .. }`, `if !p { .. }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Guard<V> {
    /// The predicate.
    pub pred: V,
    /// Whether the instruction runs where it is false.
    pub negated: bool,
}

/// Why a target refuses an instruction, and which argument it blames, if one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fault {
    /// The argument at fault, by position.
    pub arg: Option<usize>,
    /// What is wrong.
    pub message: String,
}

impl Fault {
    /// The instruction as a whole is wrong.
    pub fn new(message: impl Into<String>) -> Self {
        Fault { arg: None, message: message.into() }
    }

    /// Argument `arg` is wrong.
    pub fn at(arg: usize, message: impl Into<String>) -> Self {
        Fault { arg: Some(arg), message: message.into() }
    }
}
