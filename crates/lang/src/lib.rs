//! The language kernels are written in: Rust's syntax, read as instructions over variables,
//! so that a kernel is a function written where it is called.
//!
//! ```text
//! pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
//!     let v: ymm;
//!     vmovups(v, [xs]);
//!     ret_sum(v);
//! }
//!
//! fn ret_sum(v: ymm) {
//!     note("the lanes");
//!     let high: ymm;
//!     vextractf128(high.xmm, v, 1);
//!     ..
//! }
//! ```
//!
//! What runs is instructions; everything else happens while the kernel compiles:
//!
//! - **Variables** are declared with a type and no value, `let v: ymm;`, or many at once:
//!   `let acc: [ymm; 4];`, which are `acc0` to `acc3`, or `let (x, y): (u64, f32);`, each
//!   of its own type. Any number of instructions may write
//!   one; the allocator gives each a register, or says which were alive together when there
//!   were not enough. A field is a view of one: `v.xmm`, `g.r32`. A type of the machine's
//!   that takes arguments is written as Rust writes a generic one, `shared<[f32; N * 4], 16>`,
//!   and the machine is given it with values in place of the names they are bound to:
//!   `shared<[f32; 256], 16>`.
//! - **Values** are known now: `const` items, `let k = 4;`, a helper's `<const N: usize>`,
//!   a loop's counter. Integers are checked for overflow; floats are `f32` unless written or
//!   declared `f64`, `f16` or `bf16`, and their arithmetic is the machine's. The 16-bit ones
//!   are worked out in `f64` and rounded once, to the nearest, as the machine rounds them.
//! - **Instructions** are calls by name, `vaddps(sum, sum, pair)`, or by words joined with
//!   dots, `ld.global.f32(x, [p + 4])`. An operand is a variable, a value, a label, or memory
//!   in brackets: `[x + i * 4 + 16]` at an address, `[1.0]` or `[[0.5; 8]]` holding data.
//!   A name that is nothing in scope goes to the machine as it is, for registers of its own:
//!   `tid.x`, or `_` for a result nothing reads; the name of a function that names an ABI is
//!   that function, for the machine to call. Operands taken together, a vector or a fragment,
//!   are a tuple, `(a, b)`, `()` for none, or variables declared together; `d | p` is two
//!   written at once, `!p` a predicate read inverted, `-a` an operand negated, and
//!   `[t, s, (x, y)]` operands in brackets, which the machine reads as one.
//! - **Names of instructions** are Rust's words: one that starts with a digit takes a `_`,
//!   `tex._2d`; one that holds a `::` has `__` there, `ld.global.L2__64B`; and a keyword
//!   takes `r#`, `cp.r#async`. The machine is given them as it writes them.
//! - **Helpers** are functions with no ABI, inlined where they are called. A parameter takes
//!   the caller's variable, seen as its type; `<const N: usize>` takes a value, as in
//!   `dot::<4>(x, y, n)`. A helper sees its own names and the module's constants.
//! - **Functions that name an ABI** are compiled each on its own: kernels, and functions the
//!   machine calls by an instruction, as PTX's `call(r, f, (a, b))` calls a `.func`.
//! - **`for k in 0..n { .. }`** is unrolled, over a range, an array or variables.
//!   **`if c { .. } else { .. }`** chooses now when `c` is a bool; when `c` is a predicate
//!   variable it guards each instruction inside, `if !p` the other way.
//! - **Labels** are placed by `label(top);` and named by jumps, before or after. Each block
//!   numbers its labels afresh every time it runs, so a loop's body or a helper's gets new
//!   ones each time. `note("acc {k}")` is a remark for the listing, `{name}` a value's.
//!
//! This crate knows no machine: [`kernel`] runs a function's body against a [`Target`], which
//! an instruction set implements to check and build its instructions. Tokens come from
//! `proc_macro` through [`Tree`], or from a string through [`lex`]; spans are numbers
//! whoever made the tokens can map back.

mod ast;
mod eval;
mod parse;
mod token;
mod value;

use std::fmt;

pub use ast::{
    BinOp, Block, Const, Expr, Fn, Generic, Instances, Item, ItemKind, Module, Param, Ret, Stmt, Type, UnOp,
};
pub use eval::{Fault, Guard, Mem, Operand, Run, Target, instance, instances, kernel};
pub use parse::module;
pub use token::{Delim, Span, Tree, lex, text};
pub use value::{IntTy, Value};

/// Why a module does not compile, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// Where.
    pub span: Span,
    /// What is wrong.
    pub message: String,
}

impl Error {
    /// The message with the line and column of a span that is a byte offset into `src`.
    pub fn at(&self, src: &str) -> String {
        let offset = (self.span.0 as usize).min(src.len());
        let before = &src[..offset];
        let line = before.matches('\n').count() + 1;
        let column = before.rsplit('\n').next().map_or(0, |l| l.chars().count()) + 1;
        format!("{line}:{column}: {}", self.message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// The module in `src`, as a string holds it.
pub fn parse(src: &str) -> Result<Module, Error> {
    module(&lex(src)?)
}

#[cfg(test)]
mod tests;
