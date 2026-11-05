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
//!   one; the allocator gives each a register, and where there are not enough, keeps some on
//!   the stack, which the listing says. A field is a view of one: `v.xmm`, `g.r32`. A type of the machine's
//!   that takes arguments is written as Rust writes a generic one, `shared<[f32; N * 4], 16>`,
//!   and the machine is given it with values in place of the names they are bound to:
//!   `shared<[f32; 256], 16>`.
//! - **Values** are known now: `const` items, each of which [`constant`] works out on its own,
//!   `let k = 4;`, the `<const N: usize>` of a helper or of an instance, a loop's counter.
//!   Integers are checked for overflow; floats are `f32` unless written or declared `f64`,
//!   `f16` or `bf16`, and their arithmetic is the machine's. The 16-bit ones are worked out in
//!   `f64` and rounded once, to the nearest, as the machine rounds them.
//! - **Instructions** are calls by name, `vaddps(sum, sum, pair)`, or by words joined with
//!   dots, `ld.global.f32(x, [p + 4])`. An operand is a variable, a value, a label, or memory
//!   in brackets: `[x + i * 4 + 16]` at an address, `[1.0]` or `[[0.5; 8]]` holding data.
//!   Before the brackets, a name that is nothing in scope goes to the machine with the memory,
//!   for one that names how much of it an instruction reaches: `dword[x]`.
//!   A name that is nothing in scope goes to the machine as it is, for registers of its own:
//!   `tid.x`, or `_` for a result nothing reads; the name of a function that names an ABI is
//!   that function, for the machine to call. Operands taken together, a vector or a fragment,
//!   are a tuple, `(a, b)`, `()` for none, or variables declared together; `d | p` is two
//!   written at once, `!p` a predicate read inverted, `-a` an operand negated, and
//!   `[t, s, (x, y)]` operands in brackets, which the machine reads as one.
//! - **Names of instructions** are Rust's words: one that starts with a digit takes a `_`,
//!   `tex._2d`; one that holds a `::` has `__` there, `ld.global.L2__64B`; and a keyword
//!   takes `r#`, `cp.r#async`. The machine is given them as it writes them.
//! - **Names written raw** are the names Rust reads, in a type, a value's name and an attribute's: `r#u64` is `u64`,
//!   `r#x` is `x` and `#[r#cooperative]` is `#[cooperative]`. [`unraw`] gives a name so, and [`read_type`] a type,
//!   `[r#f16; 4]` as `[f16; 4]`. A keyword written raw is a name, and no keyword: [`read_type`] leaves its `r#`, for
//!   [`pointer()`] to count one word in a pointer to `r#dyn::Foo`, a struct, where a pointer to `dyn Foo` has two. So
//!   it is for the 48 keywords Rust lets be written raw, the strict and the reserved ones. A weak keyword, `union` say,
//!   is a name as it is, and loses its `r#`.
//! - **Helpers** are functions with no ABI, inlined where they are called. A parameter takes
//!   the caller's variable, seen as its type; `<const N: usize>` takes a value, as in
//!   `dot::<4>(x, y, n)`. A helper sees its own names and the module's constants.
//! - **Functions that name an ABI** are compiled each on its own: kernels, and functions the
//!   machine calls by an instruction, as PTX's `call(r, f, (a, b))` calls a `.func`. Each runs
//!   on its machine, never while Rust compiles, so it is not `const`; and no function is
//!   `async`, for a function runs when it is called. One with `<const N: u32>`, an integer or
//!   a `bool`, is compiled for each instance its `#[instances(256, 1024)]` lists, of values
//!   known now, or for two `#[instances((3, true), (-2, false))]`; [`instances`] lists them,
//!   and an instruction names one as `f::<256>`. A generic is a value: `<T>` is an error, and so is
//!   `impl` in a parameter's type or the result's, wherever in it, `x: impl Copy`,
//!   `Option<impl Copy>` or `-> *mut impl Copy`: in a parameter it makes the function generic over
//!   a type, and in the result it hides one. `#[cooperative]` on a kernel says that its parts wait
//!   for each other at `grid.sync()`, a GPU kernel's blocks or an x86 kernel's threads, so that it
//!   is launched with them all running at once. A GPU kernel's directives, `#[maxntid(16, 16)]`
//!   and the like, give whole numbers known now, which the machine takes before the parameters.
//! - **`for k in 0..n { .. }`** is unrolled, over a range, an array or variables.
//!   **`if c { .. } else { .. }`** chooses now when `c` is a bool; when `c` is a predicate
//!   variable it guards each instruction inside, `if !p` the other way.
//! - **Labels** are placed by `label(top);` and named by jumps, before or after. Each block
//!   numbers its labels afresh every time it runs, so a loop's body or a helper's gets new
//!   ones each time. `note("acc {k}")` is a remark for the listing, `{name}` a value's.
//! - **Statics** are memory of the module's, of a type of the machine's that takes arguments,
//!   `static COUNTS: global<[u32; 16]>;`, which each of its functions names, helpers among
//!   them, as memory a function declares is named, and which keeps what they write from one
//!   run to the next. `= [1.5, 2.0]` is what it holds until something writes it, zeros if it
//!   says nothing; [`memory`] works out its type and that value. A machine without such
//!   memory refuses it where a function names it.
//!
//! This crate knows no machine: [`kernel`] runs a function's body against a [`Target`], which
//! an instruction set implements to check and build its instructions. Tokens come from
//! `proc_macro` through [`Tree`], or from a string through [`lex`]; spans are numbers
//! whoever made the tokens can map back.

mod ast;
mod eval;
mod parse;
mod target;
mod token;
mod value;

use std::fmt;

pub use ast::{
    BinOp, Block, Const, Cooperative, Directive, Expr, Fn, Generic, Instances, Item, ItemKind, Module, Param, Ret,
    Static, Stmt, Type, UnOp,
};
pub use eval::{Memory, Mention, Run, constant, instance, instances, kernel, memory};
pub use parse::{module, read};
pub use target::{Fault, Guard, Mem, Operand, Target};
pub use token::{Delim, Span, Tree, lex, pointer, read_type, text, unraw};
pub use value::{IntTy, Value, article};

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

/// How deep helpers call helpers, and how deep brackets, types, expressions, blocks and the arrays
/// of a value nest, at most.
const DEPTH: usize = 64;

/// Nothing, if a level inside `open` others is within the most; else why not, at `span`, where
/// the level starts.
pub fn nested(open: usize, span: Span) -> Result<(), Error> {
    if open >= DEPTH {
        return Err(Error { span, message: format!("this nests {DEPTH} deep at most") });
    }
    Ok(())
}

/// The module in `src`, as a string holds it.
pub fn parse(src: &str) -> Result<Module, Error> {
    module(&lex(src)?, 0)
}

#[cfg(test)]
mod tests;
