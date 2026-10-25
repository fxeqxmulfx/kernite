//! A module of kernels as the parser reads it: items, statements, expressions.

use std::ops::Range;

use crate::Error;
use crate::token::Span;

/// The items of a module, in order.
#[derive(Clone, Debug)]
pub struct Module {
    /// Each item.
    pub items: Vec<Item>,
}

/// An item, and the tokens it spans by index into the module's tokens.
#[derive(Clone, Debug)]
pub struct Item {
    /// Its tokens, attributes included.
    pub tokens: Range<usize>,
    /// What it is.
    pub kind: ItemKind,
}

/// What an item is.
#[derive(Clone, Debug)]
pub enum ItemKind {
    /// A constant, which the kernels read at compile time and the module keeps.
    Const(Const),
    /// A kernel if it names an ABI, else a helper, inlined where it is called.
    Fn(Fn),
    /// Anything else: Rust that passes through.
    Other,
}

/// `const NAME: TYPE = VALUE;`.
#[derive(Clone, Debug)]
pub struct Const {
    /// Its name.
    pub name: String,
    /// Where its name is.
    pub span: Span,
    /// Its type.
    pub ty: Type,
    /// Its value.
    pub value: Expr,
}

/// A function: a kernel or a helper.
#[derive(Clone, Debug)]
pub struct Fn {
    /// Its outer attributes, doc comments among them, by index into the module's tokens.
    pub attrs: Range<usize>,
    /// What comes before `fn`: visibility, `unsafe`, `extern "abi"`.
    pub head: Range<usize>,
    /// The ABI it names, which makes it a kernel for the machine that ABI belongs to.
    pub abi: Option<(String, Span)>,
    /// Its name.
    pub name: String,
    /// Where its name is.
    pub span: Span,
    /// Its const parameters, a helper's compile-time arguments.
    pub generics: Vec<Generic>,
    /// Its parameters.
    pub params: Vec<Param>,
    /// The index of the parameters' group among the module's tokens.
    pub params_at: usize,
    /// What it returns.
    pub ret: Option<Ret>,
    /// Its body, or why it does not parse: the rest of the module is read all the same.
    pub body: Result<Block, Error>,
}

/// `const N: usize` among a helper's generics.
#[derive(Clone, Debug)]
pub struct Generic {
    /// Its name.
    pub name: String,
    /// Where it is.
    pub span: Span,
    /// Its type.
    pub ty: String,
}

/// A parameter.
#[derive(Clone, Debug)]
pub struct Param {
    /// Its name.
    pub name: String,
    /// Where its name is.
    pub span: Span,
    /// Its type.
    pub ty: Type,
}

/// What a function returns.
#[derive(Clone, Debug)]
pub struct Ret {
    /// The type, as written.
    pub ty: String,
    /// Where it is.
    pub span: Span,
    /// Its tokens, by index into the module's tokens.
    pub tokens: Range<usize>,
}

/// A type: a name, an array of variables, `[ymm; N]`, or a name with arguments,
/// `shared<[f32; 256], 16>`.
#[derive(Clone, Debug)]
pub enum Type {
    /// A type by its name, as written: `ymm`, `*const f32`.
    Name(String, Span),
    /// `[elem; len]`.
    Array(Box<Type>, Expr, Span),
    /// `[elem]`, of a length the program learns when it runs.
    Slice(Box<Type>, Span),
    /// `name<args>`, its arguments types or values: `shared<[f32; 256], 16>`.
    Generic(String, Vec<Type>, Span),
}

impl Type {
    /// Where it is.
    pub fn span(&self) -> Span {
        match self {
            Type::Name(_, span) | Type::Array(_, _, span) | Type::Slice(_, span) | Type::Generic(_, _, span) => *span,
        }
    }
}

/// Statements in braces, which scope the names and labels declared in them.
#[derive(Clone, Debug)]
pub struct Block {
    /// The statements.
    pub stmts: Vec<Stmt>,
    /// Where the block opens.
    pub span: Span,
}

/// A statement.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// `let v: T;` declares a variable, `let a: [T; N];` as many, `let k = value;` names a value
    /// or a variable.
    Let {
        /// The name.
        name: String,
        /// Where it is.
        span: Span,
        /// The type.
        ty: Option<Type>,
        /// The value.
        value: Option<Expr>,
    },
    /// `const K: T = value;`.
    Const(Const),
    /// `for k in range { .. }`, unrolled.
    For {
        /// The name each element takes.
        var: String,
        /// Where it is.
        span: Span,
        /// What it runs over.
        iter: Expr,
        /// The body.
        body: Block,
    },
    /// `if cond { .. } else { .. }`: at compile time if `cond` is a value, by predicate if it is
    /// a variable.
    If {
        /// The condition.
        cond: Expr,
        /// What runs if it holds.
        then: Block,
        /// What runs if it does not.
        otherwise: Option<Box<Stmt>>,
        /// Where it is.
        span: Span,
    },
    /// `{ .. }`.
    Block(Block),
    /// An instruction, a helper's call, `label(name)` or `note("..")`.
    Expr(Expr),
}

/// A unary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    /// `-`.
    Neg,
    /// `!`.
    Not,
}

/// A binary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `+`.
    Add,
    /// `-`.
    Sub,
    /// `*`.
    Mul,
    /// `/`.
    Div,
    /// `%`.
    Rem,
    /// `<<`.
    Shl,
    /// `>>`.
    Shr,
    /// `&`.
    And,
    /// `|`.
    Or,
    /// `^`.
    Xor,
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
    /// `&&`.
    AndAnd,
    /// `||`.
    OrOr,
}

/// An expression: a value at compile time, or an operand.
#[derive(Clone, Debug)]
pub enum Expr {
    /// A number, as written.
    Lit(String, Span),
    /// A string.
    Str(String, Span),
    /// `true` or `false`.
    Bool(bool, Span),
    /// A name, or `a::b`.
    Path(Vec<String>, Span),
    /// `base.name`: a view of a variable, or part of an instruction's name.
    Field(Box<Expr>, String, Span),
    /// `base[index]`.
    Index(Box<Expr>, Box<Expr>, Span),
    /// `func(args)` or `func::<generics>(args)`.
    Call {
        /// What is called.
        func: Box<Expr>,
        /// Its compile-time arguments.
        generics: Vec<Expr>,
        /// Its arguments.
        args: Vec<Expr>,
        /// Where it is.
        span: Span,
    },
    /// `recv.name(args)`.
    Method {
        /// What it is called on.
        recv: Box<Expr>,
        /// The method.
        name: String,
        /// Its arguments.
        args: Vec<Expr>,
        /// Where the method's name is.
        span: Span,
    },
    /// `-a`, `!a`.
    Unary(UnOp, Box<Expr>, Span),
    /// `a op b`, spanned at the operator.
    Binary(BinOp, Box<Expr>, Box<Expr>, Span),
    /// `lo..hi`, `lo..=hi`.
    Range {
        /// Where it starts.
        lo: Box<Expr>,
        /// Where it ends.
        hi: Box<Expr>,
        /// Whether `hi` is in it.
        inclusive: bool,
        /// Where it is.
        span: Span,
    },
    /// `[a, b, ..]`: an array, or an address.
    Array(Vec<Expr>, Span),
    /// `[value; len]`.
    Repeat(Box<Expr>, Box<Expr>, Span),
    /// `(a)`.
    Paren(Box<Expr>, Span),
    /// `(a, b, ..)` or `(a,)`: operands taken together, as a vector.
    Tuple(Vec<Expr>, Span),
    /// `a as T`.
    Cast(Box<Expr>, String, Span),
}

impl Expr {
    /// Where it is.
    pub fn span(&self) -> Span {
        match self {
            Expr::Lit(_, s)
            | Expr::Str(_, s)
            | Expr::Bool(_, s)
            | Expr::Path(_, s)
            | Expr::Field(_, _, s)
            | Expr::Index(_, _, s)
            | Expr::Call { span: s, .. }
            | Expr::Method { span: s, .. }
            | Expr::Unary(_, _, s)
            | Expr::Binary(_, _, _, s)
            | Expr::Range { span: s, .. }
            | Expr::Array(_, s)
            | Expr::Repeat(_, _, s)
            | Expr::Paren(_, s)
            | Expr::Tuple(_, s)
            | Expr::Cast(_, _, s) => *s,
        }
    }
}
