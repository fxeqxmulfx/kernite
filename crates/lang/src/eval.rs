//! The evaluator: a kernel's body run at compile time, its loops unrolled and its helpers
//! inlined, as calls into the [`Target`] that builds the machine's instructions.

use std::fmt;

use crate::Error;
use crate::ast::{BinOp, Block, Const, Expr, Fn, ItemKind, Module, Stmt, Type, UnOp};
use crate::token::Span;
use crate::value::{self, Value, Want};

/// An instruction set, as the evaluator sees it: variables of the machine's types, and
/// instructions by name. It checks what it is given and builds the program; the evaluator
/// knows no machine.
pub trait Target {
    /// A variable, seen as one of the machine's types: `ymm`, `v.xmm`, `u32`.
    type Var: Copy + PartialEq + fmt::Debug;

    /// The kernel's next parameter, of the Rust type `ty` as written: `*const f32`, `usize`.
    fn param(&mut self, name: &str, ty: &str) -> Result<Self::Var, String>;

    /// What the kernel returns, of the Rust type `ty`.
    fn result(&mut self, ty: &str) -> Result<(), String>;

    /// A new variable of the machine's type `ty`.
    fn var(&mut self, name: &str, ty: &str) -> Result<Self::Var, String>;

    /// `var` seen as `view`: a field, as in `v.xmm`, or the type of a helper's parameter.
    fn view(&mut self, var: Self::Var, view: &str) -> Result<Self::Var, String>;

    /// Places label `label` here. The evaluator numbers labels, each once.
    fn label(&mut self, label: u32);

    /// A remark for whoever reads the listing.
    fn note(&mut self, text: &str);

    /// The instruction `name` over `args`, run only where `guard` holds if there is one.
    fn inst(&mut self, name: &str, args: &[Operand<Self::Var>], guard: Option<Guard<Self::Var>>) -> Result<(), Fault>;

    /// The end of the body, for whatever the machine puts there.
    fn end(&mut self) -> Result<(), String>;
}

/// What an instruction is given.
#[derive(Clone, Debug, PartialEq)]
pub enum Operand<V> {
    /// A variable.
    Var(V),
    /// A name that is nothing in scope, for a machine that has registers of its own by name:
    /// `tid.x`. `_` is one, for an operand the machine lets be left out.
    Special(String),
    /// An integer known now.
    Int(i64),
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
    /// A label, for a jump.
    Label(u32),
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

/// What running a kernel found, besides what it asked of its target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    /// Every variable declared, with where: the place to blame for what goes wrong with one
    /// later, when its registers are allocated.
    pub decls: Vec<(String, Span)>,
    /// The module's constants it read, itself or through others, in the module's order. Rust
    /// never sees a kernel's body run, so it sees none of these reads.
    pub consts: Vec<String>,
}

/// Runs kernel `f` of `module` on `target`: its parameters declared, its body evaluated.
pub fn kernel<T: Target>(module: &Module, f: &Fn, target: &mut T) -> Result<Run, Error> {
    let mut e = Eval::new(module, target);
    if let Some(g) = f.generics.first() {
        return Err(err(g.span, "a kernel takes no compile-time arguments; a helper does"));
    }
    let mut scope = vec![];
    for p in &f.params {
        let Type::Name(ty, span) = &p.ty else {
            return Err(err(p.ty.span(), "a kernel's parameters are values the machine passes, not arrays"));
        };
        let v = e.target.param(&p.name, ty).map_err(|m| err(*span, m))?;
        e.decls.push((p.name.clone(), p.span));
        scope.push((p.name.clone(), Binding::Var(v)));
    }
    if let Some(r) = &f.ret {
        e.target.result(&r.ty).map_err(|m| err(r.span, m))?;
    }
    e.scopes.push(scope);
    e.block(f.body.as_ref().map_err(Clone::clone)?)?;
    e.target.end().map_err(|m| err(f.span, m))?;
    // A constant is evaluated the first time it is named, so the ones evaluated are the ones read.
    let consts = e.consts.iter().filter(|(_, s)| matches!(s, State::Done(_))).map(|(c, _)| c.name.clone()).collect();
    Ok(Run { decls: e.decls, consts })
}

/// How deep helpers may call helpers.
const DEPTH: usize = 64;

fn err(span: Span, message: impl Into<String>) -> Error {
    Error { span, message: message.into() }
}

/// What a name stands for.
#[derive(Clone, Debug)]
enum Binding<V> {
    /// A value known now.
    Value(Value),
    /// A variable.
    Var(V),
    /// Variables declared together, `let acc: [ymm; 4]`.
    Vars(Vec<V>),
    /// A label of the block.
    Label(u32),
}

impl<V> Binding<V> {
    fn kind(&self) -> String {
        match self {
            Binding::Value(v) => v.kind().to_string(),
            Binding::Var(_) => "a variable".into(),
            Binding::Vars(vs) => format!("{} variables", vs.len()),
            Binding::Label(_) => "a label".into(),
        }
    }
}

type Scope<V> = Vec<(String, Binding<V>)>;

/// A module constant: waiting to be named, being evaluated, or known.
enum State {
    Waiting,
    Busy,
    Done(Value),
}

/// An address on its way: variables with what each is multiplied by, and bytes added.
enum Lin<V> {
    Value(Value),
    Sum(Vec<(V, i64)>, i64),
}

struct Eval<'a, T: Target> {
    target: &'a mut T,
    consts: Vec<(&'a Const, State)>,
    helpers: Vec<&'a Fn>,
    kernels: Vec<&'a str>,
    /// The names in scope, innermost last. A helper runs in scopes of its own.
    scopes: Vec<Scope<T::Var>>,
    labels: u32,
    guard: Option<Guard<T::Var>>,
    depth: usize,
    decls: Vec<(String, Span)>,
}

impl<'a, T: Target> Eval<'a, T> {
    fn new(module: &'a Module, target: &'a mut T) -> Self {
        let (mut consts, mut helpers, mut kernels) = (vec![], vec![], vec![]);
        for item in &module.items {
            match &item.kind {
                ItemKind::Const(c) => consts.push((c, State::Waiting)),
                ItemKind::Fn(f) if f.abi.is_some() => kernels.push(f.name.as_str()),
                ItemKind::Fn(f) => helpers.push(f),
                ItemKind::Other => {}
            }
        }
        Eval { target, consts, helpers, kernels, scopes: vec![], labels: 0, guard: None, depth: 0, decls: vec![] }
    }

    fn bind(&mut self, name: &str, b: Binding<T::Var>) {
        self.scopes.last_mut().expect("a scope").push((name.to_string(), b));
    }

    fn lookup(&mut self, name: &str, span: Span) -> Result<Option<Binding<T::Var>>, Error> {
        for scope in self.scopes.iter().rev() {
            if let Some((_, b)) = scope.iter().rev().find(|(n, _)| n == name) {
                return Ok(Some(b.clone()));
            }
        }
        Ok(self.global(name, span)?.map(Binding::Value))
    }

    /// The module constant `name`, evaluated the first time it is named, in a scope of
    /// constants alone.
    fn global(&mut self, name: &str, span: Span) -> Result<Option<Value>, Error> {
        let Some(i) = self.consts.iter().position(|(c, _)| c.name == name) else {
            return Ok(None);
        };
        match &self.consts[i].1 {
            State::Done(v) => return Ok(Some(v.clone())),
            State::Busy => return Err(err(span, format!("`{name}` is defined in terms of itself"))),
            State::Waiting => {}
        }
        self.consts[i].1 = State::Busy;
        let c = self.consts[i].0;
        let outer = std::mem::take(&mut self.scopes);
        let v = self.constant(c);
        self.scopes = outer;
        let v = v?;
        self.consts[i].1 = State::Done(v.clone());
        Ok(Some(v))
    }

    fn constant(&mut self, c: &Const) -> Result<Value, Error> {
        let v = self.value(&c.value, want(&c.ty))?;
        self.typed(v, &c.ty, c.value.span())
    }

    /// `v` as a value of type `ty`, or why it is not one.
    fn typed(&mut self, v: Value, ty: &Type, span: Span) -> Result<Value, Error> {
        match ty {
            Type::Name(name, _) => {
                if let Some(elem) = name.strip_prefix("&[").and_then(|n| n.strip_suffix(']')) {
                    let Value::Array(xs) = v else {
                        return Err(err(span, format!("this is {}, not {name}", v.kind())));
                    };
                    let elem = Type::Name(elem.to_string(), span);
                    return xs
                        .into_iter()
                        .map(|x| self.typed(x, &elem, span))
                        .collect::<Result<_, _>>()
                        .map(Value::Array);
                }
                value::typed(v, name).map_err(|m| err(span, m))
            }
            Type::Array(elem, len, _) => {
                let n = self.count(len)?;
                match v {
                    Value::Array(xs) if xs.len() == n => {
                        xs.into_iter().map(|x| self.typed(x, elem, span)).collect::<Result<_, _>>().map(Value::Array)
                    }
                    Value::Array(xs) => Err(err(span, format!("this has {} elements, not {n}", xs.len()))),
                    other => Err(err(span, format!("this is {}, not an array", other.kind()))),
                }
            }
            Type::Slice(..) | Type::Generic(..) => {
                Err(err(ty.span(), "a value known now is a number, a bool, or an array of them, `[T; N]`"))
            }
        }
    }

    /// The type `t` as the target reads it, with what its arguments name worked out:
    /// `shared<[f32; 256], 16>`.
    fn type_name(&mut self, t: &Type) -> Result<String, Error> {
        Ok(match t {
            Type::Name(name, _) => name.clone(),
            Type::Array(elem, len, _) => format!("[{}; {}]", self.type_name(elem)?, self.count(len)?),
            Type::Slice(elem, _) => format!("[{}]", self.type_name(elem)?),
            Type::Generic(name, args, _) => {
                let mut out = vec![];
                for a in args {
                    out.push(match a {
                        Type::Name(n, span) if n.starts_with(|c: char| c.is_ascii_digit()) => {
                            value::literal(n, Want::Any).map_err(|m| err(*span, m))?.to_string()
                        }
                        Type::Name(n, span) => match self.lookup(n, *span)? {
                            Some(Binding::Value(v)) => v.to_string(),
                            _ => n.clone(),
                        },
                        _ => self.type_name(a)?,
                    });
                }
                format!("{name}<{}>", out.join(", "))
            }
        })
    }

    /// A length or a count: a `usize` known now.
    fn count(&mut self, e: &Expr) -> Result<usize, Error> {
        match self.value(e, Want::Int(value::IntTy::Usize))? {
            Value::Int(n, _) if n >= 0 => usize::try_from(n).map_err(|_| err(e.span(), format!("{n} is too many"))),
            other => Err(err(e.span(), format!("a count is a whole number; this is {other}"))),
        }
    }

    /// The value of `e`, which must be known now.
    fn value(&mut self, e: &Expr, want: Want) -> Result<Value, Error> {
        let at = |m: String| err(e.span(), m);
        match e {
            Expr::Lit(text, _) => value::literal(text, want).map_err(at),
            Expr::Bool(b, _) => Ok(Value::Bool(*b)),
            Expr::Str(..) => Err(at("a string is no value here; `note(\"..\")` takes one".into())),
            Expr::Path(path, span) if path.len() == 1 => match self.lookup(&path[0], *span)? {
                Some(Binding::Value(v)) => Ok(v),
                Some(b) => Err(at(format!(
                    "`{}` is {}, known only when the kernel runs; this wants a value known now",
                    path[0],
                    b.kind()
                ))),
                None => Err(at(format!("`{}` is not defined", path[0]))),
            },
            Expr::Path(path, _) => {
                let p = path.join("::");
                value::path(&p).ok_or_else(|| at(format!("`{p}` is not a constant kgen knows")))
            }
            Expr::Field(..) => Err(at("a field is a view of a variable, not a value".into())),
            Expr::Index(base, index, _) => {
                let i = self.count(index)?;
                match self.value(base, want)? {
                    Value::Array(xs) => xs
                        .get(i)
                        .cloned()
                        .ok_or_else(|| at(format!("index {i} is past the end of {} elements", xs.len()))),
                    other => Err(at(format!("{} has no elements to index", other.kind()))),
                }
            }
            Expr::Call { func, generics, args, .. } => match (&**func, generics.as_slice(), args.as_slice()) {
                (Expr::Path(p, _), [], [arg]) if p.len() == 2 && p[1] == "from_bits" => {
                    let bits = self.value(arg, Want::Any)?;
                    match (p[0].as_str(), bits) {
                        ("f32", Value::Int(b, _)) => u32::try_from(b)
                            .map(|b| Value::F32(f32::from_bits(b)))
                            .map_err(|_| at(format!("{b} is not 32 bits"))),
                        ("f64", Value::Int(b, _)) => u64::try_from(b)
                            .map(|b| Value::F64(f64::from_bits(b)))
                            .map_err(|_| at(format!("{b} is not 64 bits"))),
                        ("f16", Value::Int(b, _)) => {
                            u16::try_from(b).map(Value::F16).map_err(|_| at(format!("{b} is not 16 bits")))
                        }
                        ("bf16", Value::Int(b, _)) => {
                            u16::try_from(b).map(Value::Bf16).map_err(|_| at(format!("{b} is not 16 bits")))
                        }
                        _ => Err(at("`from_bits` makes an f32, f64, f16 or bf16 of an integer".into())),
                    }
                }
                _ => Err(at("a call is a statement: an instruction or a helper, not a value".into())),
            },
            Expr::Method { recv, name, args, .. } => {
                let recv = self.value(recv, want)?;
                let args = args.iter().map(|a| self.value(a, Want::Any)).collect::<Result<Vec<_>, _>>()?;
                value::method(&recv, name, &args).map_err(at)
            }
            Expr::Unary(op, inner, _) => {
                let v = self.value(inner, want)?;
                value::unary(*op, &v).map_err(at)
            }
            Expr::Binary(op, a, b, _) => {
                let a = self.value(a, want)?;
                // The right side takes the left's type, as an unsuffixed literal does in Rust;
                // a shift's count is a type of its own.
                let want_b = match op {
                    BinOp::Shl | BinOp::Shr => Want::Any,
                    _ => Want::like(&a, want),
                };
                let b = self.value(b, want_b)?;
                value::binary(*op, &a, &b).map_err(at)
            }
            Expr::Range { lo, hi, inclusive, .. } => {
                let lo = self.value(lo, want)?;
                let want_hi = match lo {
                    Value::Int(_, Some(t)) => Want::Int(t),
                    _ => want,
                };
                let hi = self.value(hi, want_hi)?;
                let (Value::Int(l, tl), Value::Int(h, th)) = (&lo, &hi) else {
                    return Err(at(format!("a range runs between integers, not {} and {}", lo.kind(), hi.kind())));
                };
                let ty = match (tl, th) {
                    (Some(a), Some(b)) if a != b => {
                        return Err(at(format!("a range from {} to {}", a.name(), b.name())));
                    }
                    _ => tl.or(*th),
                };
                let end = if *inclusive { *h + 1 } else { *h };
                if end - *l > 1 << 20 {
                    return Err(at(format!("{l}..{end} unrolls to too many")));
                }
                Ok(Value::Array((*l..end).map(|x| Value::Int(x, ty)).collect()))
            }
            Expr::Array(items, _) => {
                items.iter().map(|x| self.value(x, want)).collect::<Result<Vec<_>, _>>().map(Value::Array)
            }
            Expr::Repeat(item, len, _) => {
                let v = self.value(item, want)?;
                let n = self.count(len)?;
                if n > 1 << 20 {
                    return Err(at(format!("{n} elements are too many")));
                }
                Ok(Value::Array(vec![v; n]))
            }
            Expr::Paren(inner, _) => self.value(inner, want),
            Expr::Tuple(..) => Err(at("a tuple is operands taken together, not a value".into())),
            Expr::Cast(inner, ty, _) => {
                let v = self.value(inner, Want::Any)?;
                value::cast(&v, ty).map_err(at)
            }
        }
    }

    /// The variable `e` names, if it names one: `v`, `acc[2]`, `v.xmm`.
    fn place(&mut self, e: &Expr) -> Result<Option<T::Var>, Error> {
        match e {
            Expr::Path(path, span) if path.len() == 1 => match self.lookup(&path[0], *span)? {
                Some(Binding::Var(v)) => Ok(Some(v)),
                _ => Ok(None),
            },
            Expr::Index(base, index, span) => {
                let Expr::Path(path, pspan) = &**base else { return Ok(None) };
                if path.len() != 1 {
                    return Ok(None);
                }
                let Some(Binding::Vars(vs)) = self.lookup(&path[0], *pspan)? else { return Ok(None) };
                let i = self.count(index)?;
                match vs.get(i) {
                    Some(&v) => Ok(Some(v)),
                    None => Err(err(*span, format!("`{}` has {} variables; there is no [{i}]", path[0], vs.len()))),
                }
            }
            Expr::Field(base, view, span) => match self.place(base)? {
                Some(v) => self.target.view(v, view).map(Some).map_err(|m| err(*span, m)),
                None => Ok(None),
            },
            Expr::Paren(inner, _) => self.place(inner),
            _ => Ok(None),
        }
    }

    /// What `e` stands for, for `let` and `for`: a variable, many, or a value.
    fn binding(&mut self, e: &Expr) -> Result<Binding<T::Var>, Error> {
        if let Expr::Path(path, span) = e
            && path.len() == 1
            && let Some(b) = self.lookup(&path[0], *span)?
        {
            return Ok(b);
        }
        match self.place(e)? {
            Some(v) => Ok(Binding::Var(v)),
            None => Ok(Binding::Value(self.value(e, Want::Any)?)),
        }
    }

    /// Whether `e` is known only when the kernel runs: a variable or a name the machine has,
    /// or one of them inverted, negated or joined to another.
    fn runtime(&mut self, e: &Expr) -> Result<bool, Error> {
        Ok(match e {
            Expr::Path(path, span) if path.len() == 1 => {
                !matches!(self.lookup(&path[0], *span)?, Some(Binding::Value(_)))
            }
            Expr::Field(..) => self.place(e)?.is_some() || dotted(e).is_some(),
            Expr::Index(..) => self.place(e)?.is_some(),
            Expr::Unary(_, inner, _) | Expr::Paren(inner, _) => self.runtime(inner)?,
            Expr::Binary(BinOp::Or, a, b, _) => self.runtime(a)? || self.runtime(b)?,
            Expr::Tuple(..) => true,
            _ => false,
        })
    }

    /// What `e` gives an instruction.
    fn operand(&mut self, e: &Expr) -> Result<Operand<T::Var>, Error> {
        match e {
            Expr::Path(path, span) if path.len() == 1 => match self.lookup(&path[0], *span)? {
                Some(Binding::Var(v)) => Ok(Operand::Var(v)),
                Some(Binding::Value(v)) => scalar(v, *span),
                Some(Binding::Vars(vs)) => Ok(Operand::Vector(vs.into_iter().map(Operand::Var).collect())),
                Some(Binding::Label(l)) => Ok(Operand::Label(l)),
                None => Ok(Operand::Special(path[0].clone())),
            },
            Expr::Field(..) => match self.place(e)? {
                Some(v) => Ok(Operand::Var(v)),
                None => match dotted(e) {
                    Some(name) if self.lookup(name.split('.').next().unwrap_or(""), e.span())?.is_none() => {
                        Ok(Operand::Special(name))
                    }
                    _ => Err(err(e.span(), "a field is a view of a variable, as in `v.xmm`")),
                },
            },
            Expr::Index(..) => match self.place(e)? {
                Some(v) => Ok(Operand::Var(v)),
                None => scalar(self.value(e, Want::Any)?, e.span()),
            },
            Expr::Array(items, span) => match items.as_slice() {
                [inner] => self.memory(inner, *span),
                [] => Err(err(*span, "memory is one address in brackets, as in `[x + 4]`")),
                _ => items.iter().map(|x| self.operand(x)).collect::<Result<_, _>>().map(Operand::List),
            },
            Expr::Tuple(items, _) => {
                items.iter().map(|x| self.operand(x)).collect::<Result<_, _>>().map(Operand::Vector)
            }
            Expr::Paren(inner, _) => self.operand(inner),
            Expr::Unary(op, inner, _) if self.runtime(inner)? => {
                let inner = Box::new(self.operand(inner)?);
                Ok(match op {
                    UnOp::Not => Operand::Not(inner),
                    UnOp::Neg => Operand::Neg(inner),
                })
            }
            Expr::Binary(BinOp::Or, a, b, _) if self.runtime(e)? => {
                Ok(Operand::Pair(Box::new(self.operand(a)?), Box::new(self.operand(b)?)))
            }
            Expr::Repeat(_, _, span) => {
                Err(err(*span, "an array is data; in brackets, `[[x; n]]`, it is memory that holds it"))
            }
            Expr::Str(_, span) => Err(err(*span, "a string is no operand; `note(\"..\")` takes one")),
            _ => scalar(self.value(e, Want::Any)?, e.span()),
        }
    }

    /// `[inner]`: memory at an address, or memory holding a value known now.
    fn memory(&mut self, inner: &Expr, span: Span) -> Result<Operand<T::Var>, Error> {
        let (terms, disp) = match self.linear(inner)? {
            Lin::Value(v) => return value::bytes(&v).map(Operand::Data).map_err(|m| err(inner.span(), m)),
            Lin::Sum(terms, disp) => (terms, disp),
        };
        let mut merged: Vec<(T::Var, i64)> = vec![];
        for (v, k) in terms {
            match merged.iter_mut().find(|(u, _)| *u == v) {
                Some((_, sum)) => *sum += k,
                None => merged.push((v, k)),
            }
        }
        merged.retain(|&(_, k)| k != 0);
        let base = merged.iter().position(|&(_, k)| k == 1).map(|i| merged.remove(i).0);
        match merged.as_slice() {
            [] if base.is_none() => Err(err(span, "this address has no variable in it")),
            [] => Ok(Operand::Mem(Mem { base, index: None, disp })),
            &[index] => Ok(Operand::Mem(Mem { base, index: Some(index), disp })),
            _ => Err(err(span, "an address adds two variables at most, one of them multiplied")),
        }
    }

    /// An address as a sum of variables times numbers, plus a number.
    fn linear(&mut self, e: &Expr) -> Result<Lin<T::Var>, Error> {
        match e {
            Expr::Binary(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul), a, b, span) => {
                let (la, lb) = (self.linear(a)?, self.linear(b)?);
                let (ta, da, tb, db) = match (la, lb) {
                    (Lin::Value(x), Lin::Value(y)) => {
                        return value::binary(*op, &x, &y).map(Lin::Value).map_err(|m| err(*span, m));
                    }
                    (la, lb) => {
                        let (ta, da) = sum(la, a.span())?;
                        let (tb, db) = sum(lb, b.span())?;
                        (ta, da, tb, db)
                    }
                };
                let over = || err(*span, "this address overflows");
                let scale = |terms: Vec<(T::Var, i64)>, by: i64| -> Result<Vec<(T::Var, i64)>, Error> {
                    terms.into_iter().map(|(v, k)| k.checked_mul(by).map(|k| (v, k)).ok_or_else(over)).collect()
                };
                Ok(match op {
                    BinOp::Add => Lin::Sum([ta, tb].concat(), da.checked_add(db).ok_or_else(over)?),
                    BinOp::Sub => Lin::Sum([ta, scale(tb, -1)?].concat(), da.checked_sub(db).ok_or_else(over)?),
                    _ if ta.is_empty() => Lin::Sum(scale(tb, da)?, db.checked_mul(da).ok_or_else(over)?),
                    _ if tb.is_empty() => Lin::Sum(scale(ta, db)?, da.checked_mul(db).ok_or_else(over)?),
                    _ => return Err(err(*span, "an address multiplies a variable by a number, not by a variable")),
                })
            }
            Expr::Paren(inner, _) => self.linear(inner),
            Expr::Unary(UnOp::Neg, inner, span) => match self.linear(inner)? {
                Lin::Value(v) => value::unary(UnOp::Neg, &v).map(Lin::Value).map_err(|m| err(*span, m)),
                Lin::Sum(terms, disp) => Ok(Lin::Sum(terms.into_iter().map(|(v, k)| (v, -k)).collect(), -disp)),
            },
            _ => match self.place(e)? {
                Some(v) => Ok(Lin::Sum(vec![(v, 1)], 0)),
                None => Ok(Lin::Value(self.value(e, Want::Any)?)),
            },
        }
    }

    fn block(&mut self, b: &Block) -> Result<(), Error> {
        self.scopes.push(vec![]);
        let r = self.statements(b);
        self.scopes.pop();
        r
    }

    fn statements(&mut self, b: &Block) -> Result<(), Error> {
        // The labels the block places, numbered afresh each time it runs, so that a loop's
        // body or a helper's has new ones each time; jumps may name them before they are placed.
        for stmt in &b.stmts {
            let Some((name, span)) = placed_label(stmt) else { continue };
            let scope = self.scopes.last().expect("a scope");
            if scope.iter().any(|(n, b)| n == name && matches!(b, Binding::Label(_))) {
                return Err(err(span, format!("label `{name}` is placed twice in this block")));
            }
            self.labels += 1;
            self.bind(name, Binding::Label(self.labels - 1));
        }
        b.stmts.iter().try_for_each(|s| self.stmt(s))
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), Error> {
        match s {
            Stmt::Let { name, span, ty, value } => self.let_(name, *span, ty.as_ref(), value.as_ref()),
            Stmt::Const(c) => {
                let v = self.constant(c)?;
                self.bind(&c.name, Binding::Value(v));
                Ok(())
            }
            Stmt::For { var, iter, body, .. } => {
                let items: Vec<Binding<T::Var>> = match self.binding(iter)? {
                    Binding::Value(Value::Array(xs)) => xs.into_iter().map(Binding::Value).collect(),
                    Binding::Vars(vs) => vs.into_iter().map(Binding::Var).collect(),
                    other => {
                        return Err(err(
                            iter.span(),
                            format!("`for` runs over a range, an array or variables, not {}", other.kind()),
                        ));
                    }
                };
                for item in items {
                    self.scopes.push(vec![(var.clone(), item)]);
                    let r = self.block(body);
                    self.scopes.pop();
                    r?;
                }
                Ok(())
            }
            Stmt::If { cond, then, otherwise, span } => self.if_(cond, then, otherwise.as_deref(), *span),
            Stmt::Block(b) => self.block(b),
            Stmt::Expr(e) => self.call(e),
        }
    }

    fn let_(&mut self, name: &str, span: Span, ty: Option<&Type>, value: Option<&Expr>) -> Result<(), Error> {
        let b = match (ty, value) {
            (Some(t @ (Type::Name(..) | Type::Generic(..))), None) => {
                let ty = self.type_name(t)?;
                let v = self.target.var(name, &ty).map_err(|m| err(t.span(), m))?;
                self.decls.push((name.to_string(), span));
                Binding::Var(v)
            }
            (Some(Type::Slice(_, tspan)), None) => {
                return Err(err(*tspan, format!("variables are declared so many at once: `let {name}: [u32; 4];`")));
            }
            (Some(Type::Array(elem, len, _)), None) => {
                let Type::Name(t, tspan) = &**elem else {
                    return Err(err(elem.span(), "variables are declared in arrays one deep"));
                };
                let n = self.count(len)?;
                let mut vs = Vec::with_capacity(n);
                for i in 0..n {
                    let each = format!("{name}{i}");
                    vs.push(self.target.var(&each, t).map_err(|m| err(*tspan, m))?);
                    self.decls.push((each, span));
                }
                Binding::Vars(vs)
            }
            (None, Some(e)) => self.binding(e)?,
            (Some(t), Some(e)) => {
                let v = self.value(e, want(t))?;
                Binding::Value(self.typed(v, t, e.span())?)
            }
            (None, None) => {
                return Err(err(
                    span,
                    format!("`let {name};` declares nothing: give it a type, as in `let {name}: ymm;`, or a value"),
                ));
            }
        };
        self.bind(name, b);
        Ok(())
    }

    fn if_(&mut self, cond: &Expr, then: &Block, otherwise: Option<&Stmt>, span: Span) -> Result<(), Error> {
        let Some((pred, negated)) = self.predicate(cond)? else {
            return match self.value(cond, Want::Any)? {
                Value::Bool(true) => self.block(then),
                Value::Bool(false) => otherwise.map_or(Ok(()), |s| self.stmt(s)),
                other => Err(err(cond.span(), format!("a condition is a bool or a predicate, not {}", other.kind()))),
            };
        };
        if self.guard.is_some() {
            return Err(err(span, "a predicate's `if` inside another's: one predicate guards an instruction"));
        }
        self.guard = Some(Guard { pred, negated });
        let r = self.block(then).and_then(|()| match otherwise {
            Some(s) => {
                self.guard = Some(Guard { pred, negated: !negated });
                self.stmt(s)
            }
            None => Ok(()),
        });
        self.guard = None;
        r
    }

    /// The variable a condition tests, and whether it is negated: `p`, `!p`.
    fn predicate(&mut self, cond: &Expr) -> Result<Option<(T::Var, bool)>, Error> {
        match cond {
            Expr::Unary(UnOp::Not, inner, _) => Ok(self.predicate(inner)?.map(|(v, n)| (v, !n))),
            Expr::Paren(inner, _) => self.predicate(inner),
            _ => Ok(self.place(cond)?.map(|v| (v, false))),
        }
    }

    /// A statement: an instruction, a helper's call, `label(name)` or `note("..")`.
    fn call(&mut self, e: &Expr) -> Result<(), Error> {
        match e {
            Expr::Call { func, generics, args, span } => {
                let Expr::Path(path, _) = &**func else {
                    return Err(err(*span, "what is called is an instruction or a helper, by name"));
                };
                let [name] = path.as_slice() else {
                    return Err(err(*span, format!("`{}` names nothing kgen calls", path.join("::"))));
                };
                if generics.is_empty() && name == "label" {
                    return self.place_label(args, *span);
                }
                if generics.is_empty() && name == "note" {
                    return self.note(args, *span);
                }
                if let Some(&f) = self.helpers.iter().find(|f| f.name == *name) {
                    return self.inline(f, generics, args, *span);
                }
                if self.kernels.contains(&name.as_str()) {
                    return Err(err(
                        *span,
                        format!("`{name}` is a kernel; a kernel calls helpers, functions with no ABI"),
                    ));
                }
                if !generics.is_empty() {
                    return Err(err(*span, format!("`{name}` is no helper, and only helpers take `::<..>`")));
                }
                self.inst(name, args, *span)
            }
            Expr::Method { recv, name, args, span } => {
                let Some(prefix) = dotted(recv) else {
                    return Err(err(*span, "an instruction's name is words joined by dots, as in `ld.global.f32`"));
                };
                self.inst(&format!("{prefix}.{name}"), args, *span)
            }
            _ => Err(err(e.span(), "a statement is an instruction, a helper's call, `label(name)` or `note(\"..\")`")),
        }
    }

    fn inst(&mut self, name: &str, args: &[Expr], span: Span) -> Result<(), Error> {
        let ops = args.iter().map(|a| self.operand(a)).collect::<Result<Vec<_>, _>>()?;
        self.target
            .inst(&spelled(name), &ops, self.guard)
            .map_err(|f| Error { span: f.arg.and_then(|i| args.get(i)).map_or(span, Expr::span), message: f.message })
    }

    fn place_label(&mut self, args: &[Expr], span: Span) -> Result<(), Error> {
        let [Expr::Path(path, pspan)] = args else {
            return Err(err(span, "`label` takes the label's name, as in `label(top)`"));
        };
        if self.guard.is_some() {
            return Err(err(span, "a label takes no predicate: control arrives there whatever it holds"));
        }
        let scope = self.scopes.last().expect("a scope");
        let found = scope.iter().rev().find_map(|(n, b)| match b {
            Binding::Label(l) if path.len() == 1 && *n == path[0] => Some(*l),
            _ => None,
        });
        let Some(l) = found else {
            return Err(err(*pspan, "a label is placed by a statement of its own, `label(name);`, in a block"));
        };
        self.target.label(l);
        Ok(())
    }

    fn note(&mut self, args: &[Expr], span: Span) -> Result<(), Error> {
        let [Expr::Str(text, sspan)] = args else {
            return Err(err(span, "`note` takes a string, as in `note(\"the lanes\")`"));
        };
        let mut out = String::new();
        let mut rest = text.as_str();
        while let Some(i) = rest.find(['{', '}']) {
            out += &rest[..i];
            let tail = &rest[i..];
            if tail.starts_with("{{") || tail.starts_with("}}") {
                out.push(tail.as_bytes()[0] as char);
                rest = &tail[2..];
                continue;
            }
            let close = tail.find('}').filter(|_| tail.starts_with('{'));
            let Some(close) = close else {
                return Err(err(*sspan, "a `{` in a note opens a name and a `}` closes it; `{{` and `}}` are braces"));
            };
            let name = tail[1..close].trim();
            match self.lookup(name, *sspan)? {
                Some(Binding::Value(v)) => out += &v.to_string(),
                Some(b) => return Err(err(*sspan, format!("`{name}` is {}, and a note shows values", b.kind()))),
                None => return Err(err(*sspan, format!("`{name}` is not defined"))),
            }
            rest = &tail[close + 1..];
        }
        out += rest;
        self.target.note(&out);
        Ok(())
    }

    /// `f`'s body, here, over the caller's variables: each argument seen as its parameter's
    /// type. It sees its own names and the module's constants, nothing of the caller's.
    fn inline(&mut self, f: &'a Fn, generics: &[Expr], args: &[Expr], span: Span) -> Result<(), Error> {
        if self.depth >= DEPTH {
            return Err(err(span, format!("helpers call helpers {DEPTH} deep here; does `{}` call itself?", f.name)));
        }
        if let Some(r) = &f.ret {
            return Err(err(r.span, "a helper returns nothing; it writes the variables it is given"));
        }
        if generics.len() != f.generics.len() {
            return Err(err(
                span,
                format!("`{}` takes {} values in `::<..>`; this gives {}", f.name, f.generics.len(), generics.len()),
            ));
        }
        if args.len() != f.params.len() {
            return Err(err(
                span,
                format!("`{}` takes {} arguments; this gives {}", f.name, f.params.len(), args.len()),
            ));
        }
        let mut scope = vec![];
        for (g, e) in f.generics.iter().zip(generics) {
            let ty = Type::Name(g.ty.clone(), g.span);
            let v = self.value(e, want(&ty))?;
            scope.push((g.name.clone(), Binding::Value(self.typed(v, &ty, e.span())?)));
        }
        let given = args.iter().map(|a| self.binding(a)).collect::<Result<Vec<_>, _>>()?;
        let outer = std::mem::replace(&mut self.scopes, vec![scope]);
        let r = self.params(f, args, given).and_then(|()| {
            self.depth += 1;
            let r = f.body.as_ref().map_err(Clone::clone).and_then(|b| self.block(b));
            self.depth -= 1;
            r
        });
        self.scopes = outer;
        r
    }

    /// Binds `f`'s parameters to what the caller gave, in `f`'s scope.
    fn params(&mut self, f: &Fn, args: &[Expr], given: Vec<Binding<T::Var>>) -> Result<(), Error> {
        for ((p, e), b) in f.params.iter().zip(args).zip(given) {
            let at = |m: String| err(e.span(), format!("`{}`: {m}", p.name));
            let b = match (&p.ty, b) {
                (t @ (Type::Name(..) | Type::Generic(..)), Binding::Var(v)) => {
                    let t = self.type_name(t)?;
                    Binding::Var(self.target.view(v, &t).map_err(at)?)
                }
                (Type::Array(elem, len, _), Binding::Vars(vs)) => {
                    let Type::Name(t, _) = &**elem else {
                        return Err(err(elem.span(), "variables are passed in arrays one deep"));
                    };
                    let n = self.count(len)?;
                    if vs.len() != n {
                        return Err(at(format!("wants {n} variables; this is {}", vs.len())));
                    }
                    Binding::Vars(vs.into_iter().map(|v| self.target.view(v, t)).collect::<Result<_, _>>().map_err(at)?)
                }
                (t @ (Type::Name(..) | Type::Generic(..)), b) => {
                    let t = self.type_name(t)?;
                    return Err(at(format!("wants a variable of type `{t}`; this is {}", b.kind())));
                }
                (Type::Array(..), b) => {
                    return Err(at(format!("wants variables declared together; this is {}", b.kind())));
                }
                (Type::Slice(..), _) => {
                    return Err(err(p.ty.span(), "variables are passed so many at once: `[u32; N]`"));
                }
            };
            self.bind(&p.name, b);
        }
        Ok(())
    }
}

/// What a value known now gives an instruction.
fn scalar<V>(v: Value, span: Span) -> Result<Operand<V>, Error> {
    match v {
        Value::Int(x, _) => i64::try_from(x)
            .or_else(|_| u64::try_from(x).map(|x| x as i64))
            .map(Operand::Int)
            .map_err(|_| err(span, format!("{x} is wider than 64 bits"))),
        Value::F32(x) => Ok(Operand::F32(x)),
        Value::F64(x) => Ok(Operand::F64(x)),
        Value::F16(x) => Ok(Operand::F16(x)),
        Value::Bf16(x) => Ok(Operand::Bf16(x)),
        Value::Bool(_) => Err(err(span, "a bool is no operand")),
        Value::Array(_) => Err(err(span, "an array is data; in brackets, `[[..]]`, it is memory that holds it")),
    }
}

/// An address's piece as variables and a number.
fn sum<V>(l: Lin<V>, span: Span) -> Result<(Vec<(V, i64)>, i64), Error> {
    match l {
        Lin::Sum(terms, disp) => Ok((terms, disp)),
        Lin::Value(Value::Int(x, _)) => {
            i64::try_from(x).map(|x| (vec![], x)).map_err(|_| err(span, format!("{x} is too far for an address")))
        }
        Lin::Value(v) => Err(err(span, format!("an address adds whole numbers of bytes, not {}", v.kind()))),
    }
}

/// `a.b.c` as it reads, if `e` is names joined by dots.
fn dotted(e: &Expr) -> Option<String> {
    match e {
        Expr::Path(path, _) if path.len() == 1 => Some(path[0].clone()),
        Expr::Field(base, name, _) => Some(format!("{}.{name}", dotted(base)?)),
        _ => None,
    }
}

/// An instruction's name as the machine writes it: a word that starts with a digit has a `_`
/// before it in Rust, as in `tex._2d`, and a `::` is `__`, as in `L2__64B`.
fn spelled(name: &str) -> String {
    let words = name.split('.').map(|w| {
        let w = w.strip_prefix('_').filter(|w| w.starts_with(|c: char| c.is_ascii_digit())).unwrap_or(w);
        w.replace("__", "::")
    });
    words.collect::<Vec<_>>().join(".")
}

/// The label a statement places, if it is `label(name)`.
fn placed_label(s: &Stmt) -> Option<(&str, Span)> {
    let Stmt::Expr(Expr::Call { func, generics, args, .. }) = s else { return None };
    match (&**func, generics.as_slice(), args.as_slice()) {
        (Expr::Path(f, _), [], [Expr::Path(name, span)]) if f.len() == 1 && f[0] == "label" && name.len() == 1 => {
            Some((&name[0], *span))
        }
        _ => None,
    }
}

/// The type a literal takes where a value of type `ty` is wanted.
fn want(ty: &Type) -> Want {
    match ty {
        Type::Name(name, _) => Want::of(name.trim_start_matches("&[").trim_end_matches(']')),
        Type::Array(elem, ..) | Type::Slice(elem, _) => want(elem),
        Type::Generic(..) => Want::Any,
    }
}
