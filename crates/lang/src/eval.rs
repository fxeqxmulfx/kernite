//! The evaluator: a kernel's body run at compile time, its loops unrolled and its helpers
//! inlined, as calls into the [`Target`] that builds the machine's instructions.

use std::rc::Rc;

use crate::ast::{BinOp, Block, Const, Expr, Fn, Generic, ItemKind, Module, Static, Stmt, Type, UnOp};
use crate::target::{Fault, Guard, Mem, Operand, Target};
use crate::token::Span;
use crate::value::{self, Value, Want};
use crate::{DEPTH, Error, nested};

/// What running a kernel found, besides what it asked of its target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    /// The module's constants it read, itself or through others, in its body or its
    /// `#[instances(..)]`, in the module's order. Rust never sees a kernel's body run, nor its
    /// list worked out, so it sees none of these reads.
    pub consts: Vec<String>,
}

/// A static of a module, as [`memory`] evaluates it.
#[derive(Clone, Debug, PartialEq)]
pub struct Memory {
    /// Its type as the target reads it, with what its arguments name worked out among the
    /// module's constants: `global<[f32; 256], 16>`.
    pub ty: String,
    /// What it holds until something writes it, if it says: a value known now, its literals
    /// of the type its type's first argument holds, `[0.5; 256]`.
    pub value: Option<Value>,
    /// The module's constants it read, itself or through others, in the module's order. Rust
    /// never sees its type or its value worked out, so it sees none of these reads.
    pub consts: Vec<String>,
}

/// `s`, a static of `module`, worked out among the module's constants; or why not.
pub fn memory(module: &Module, s: &Static) -> Result<Memory, Error> {
    let mut consts = Consts;
    let mut e = Eval::new(module, &mut consts);
    let ty = e.type_name(&s.ty)?;
    let held = match &s.ty {
        Type::Generic(_, args, _) => args.first().map_or(Want::Any, want),
        t => want(t),
    };
    let value = s.value.as_ref().map(|v| e.value(v, held)).transpose()?;
    let consts = e.consts.iter().filter(|(_, state)| matches!(state, State::Done(_))).map(|(c, _)| c.name.clone());
    Ok(Memory { ty, value, consts: consts.collect() })
}

/// `c`, a constant of `module`, worked out among the module's constants; or why it is no value
/// known now.
pub fn constant(module: &Module, c: &Const) -> Result<Value, Error> {
    let mut consts = Consts;
    let mut e = Eval::new(module, &mut consts);
    let i = e.consts.iter().position(|(k, _)| std::ptr::eq(*k, c)).expect("a constant of the module");
    e.consts[i].1 = State::Busy;
    e.constant(c)
}

/// Runs `f` of `module` on `target`, a kernel or another function that names an ABI, for the
/// compile-time arguments `args`, those of one of its [`instances`]: its parameters declared,
/// its body evaluated.
pub fn kernel<T: Target>(module: &Module, f: &Fn, args: &[Value], target: &mut T) -> Result<Run, Error> {
    instance(f, args)?;
    let mut e = Eval::new(module, target);
    e.listed(f)?;
    let mut scope: Scope<T::Var> =
        f.generics.iter().zip(args).map(|(g, a)| (g.name.clone(), Binding::Value(Rc::new(a.clone())))).collect();
    for p in &f.params {
        let Type::Name(ty, span) = &p.ty else {
            return Err(err(p.ty.span(), "a function that names an ABI takes values the machine passes, not arrays"));
        };
        let v = e.target.param(&p.name, ty).map_err(|m| err(*span, m))?;
        e.target.declared_at(v, p.span);
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
    Ok(Run { consts })
}

/// The compile-time arguments of each instance of `f`, a function of `module`, in the order
/// of its generics: what its `#[instances(..)]` lists, each a value of its generic's type,
/// evaluated among the module's constants. A function that names an ABI and takes none has one
/// instance, of none; a helper has none, being compiled where it is called.
pub fn instances(module: &Module, f: &Fn) -> Result<Vec<Vec<Value>>, Error> {
    Eval::new(module, &mut Consts).listed(f)
}

/// What `f` is called for the compile-time arguments `args`, as Rust names its instance:
/// `tile::<4>`, or `saxpy` for none; or why they are not of its generics' types, each an
/// integer or a `bool`.
pub fn instance(f: &Fn, args: &[Value]) -> Result<String, Error> {
    if args.len() != f.generics.len() {
        return Err(err(
            f.span,
            format!("`{}` takes {} compile-time arguments; this gives {}", f.name, f.generics.len(), args.len()),
        ));
    }
    for (g, a) in f.generics.iter().zip(args) {
        machine_generic(g)?;
        if value::typed(a.clone(), &g.ty).as_ref() != Ok(a) {
            return Err(err(g.span, format!("`{}` takes a `{}` for `{}`; this is {}", f.name, g.ty, g.name, a.kind())));
        }
    }
    Ok(named(&f.name, args))
}

/// `name::<args>`, an instance's name; `name` for none.
fn named(name: &str, args: &[Value]) -> String {
    if args.is_empty() {
        return name.to_string();
    }
    let args: Vec<String> = args.iter().map(Value::to_string).collect();
    format!("{name}::<{}>", args.join(", "))
}

/// Why `g`, a generic of a function that names an ABI, cannot be one, if it cannot: Rust's
/// const generics, which the kernel's instances are picked by, are integers and `bool`s.
fn machine_generic(g: &Generic) -> Result<(), Error> {
    if value::IntTy::parse(&g.ty).is_some() || g.ty == "bool" {
        return Ok(());
    }
    Err(err(
        g.span,
        format!(
            "a function that names an ABI takes integers and `bool`s at compile time, as Rust's const generics are; \
             `{}` is neither",
            g.ty
        ),
    ))
}

/// No machine, for values known now alone: the instances a function lists. Nothing that
/// evaluates them reaches it.
struct Consts;

/// What [`Consts`] says to whatever reaches it.
const NO_MACHINE: &str = "this is evaluated for a value known now, with no machine to run on";

impl Target for Consts {
    type Var = ();

    fn param(&mut self, _: &str, _: &str) -> Result<(), String> {
        Err(NO_MACHINE.into())
    }

    fn result(&mut self, _: &str) -> Result<(), String> {
        Err(NO_MACHINE.into())
    }

    fn var(&mut self, _: &str, _: &str) -> Result<(), String> {
        Err(NO_MACHINE.into())
    }

    fn view(&mut self, (): (), _: &str) -> Result<(), String> {
        Err(NO_MACHINE.into())
    }

    fn label(&mut self, _: u32) {}

    fn note(&mut self, _: &str) {}

    fn inst(&mut self, _: &str, _: &[Operand<()>], _: Option<Guard<()>>) -> Result<(), Fault> {
        Err(Fault::new(NO_MACHINE))
    }

    fn end(&mut self) -> Result<(), String> {
        Err(NO_MACHINE.into())
    }
}

/// The most variables a kernel declares by `let`, however its loops and helpers repeat them: x86's
/// allocator keeps a set of them for each of them, a bit each, so this many take 512 MiB.
const VARIABLES: usize = 1 << 16;

/// The most steps working out a function takes. A step is a turn of a `for`; an instruction, a
/// call, a label or a note; a number of a range; an item of an array listed, as `[a, b]`; or an
/// element of an `[x; n]`, of an array a method makes, of one copied from a name, as
/// `let b: [u32; 4] = a;` and `for x in a` copy it, or of a value an index takes or a note shows. A
/// number read by name or worked out costs none. Loops that nest and helpers that call themselves
/// repeat each other's work, and this holds the work to what finishes.
const STEPS: u128 = 1 << 22;

fn err(span: Span, message: impl Into<String>) -> Error {
    Error { span, message: message.into() }
}

/// What a name stands for.
#[derive(Clone, Debug)]
enum Binding<V> {
    /// A value known now.
    Value(Rc<Value>),
    /// A variable.
    Var(V),
    /// Variables declared together, `let acc: [ymm; 4]`.
    Vars(Vec<V>),
    /// A label of the block.
    Label(u32),
    /// Memory of the module's, by its place among the module's statics: a variable the
    /// target makes the first time the function names it.
    Static(usize),
}

impl<V> Binding<V> {
    fn kind(&self) -> String {
        match self {
            Binding::Value(v) => v.kind().to_string(),
            Binding::Var(_) => "a variable".into(),
            Binding::Vars(vs) => format!("{} variables", vs.len()),
            Binding::Label(_) => "a label".into(),
            Binding::Static(_) => "memory of the module's".into(),
        }
    }
}

type Scope<V> = Vec<(String, Binding<V>)>;

/// A module constant: waiting to be named, being evaluated, or known.
enum State {
    Waiting,
    Busy,
    Done(Rc<Value>),
}

/// An address on its way: variables with what each is multiplied by, and bytes added.
enum Lin<V> {
    Value(Value),
    Sum(Vec<(V, i64)>, i64),
}

struct Eval<'a, T: Target> {
    target: &'a mut T,
    consts: Vec<(&'a Const, State)>,
    /// The module's statics, each with its variable once the function names it.
    statics: Vec<(&'a Static, Option<T::Var>)>,
    helpers: Vec<&'a Fn>,
    /// The functions that name an ABI, each compiled on its own: kernels, and functions the
    /// machine calls.
    compiled: Vec<&'a Fn>,
    /// The names in scope, innermost last. A helper runs in scopes of its own.
    scopes: Vec<Scope<T::Var>>,
    labels: u32,
    guard: Option<Guard<T::Var>>,
    /// How many instructions have run.
    insts: usize,
    /// How many helpers are running, each in the one before.
    depth: usize,
    /// How many levels of expressions and statements are running, each in the one before.
    nest: usize,
    /// How many variables `let` has declared, in the kernel and in the helpers it called.
    declared: usize,
    /// How many steps working out has taken, in the kernel and in the helpers it called.
    steps: u128,
}

impl<'a, T: Target> Eval<'a, T> {
    fn new(module: &'a Module, target: &'a mut T) -> Self {
        let (mut consts, mut statics, mut helpers, mut compiled) = (vec![], vec![], vec![], vec![]);
        for item in &module.items {
            match &item.kind {
                ItemKind::Const(c) => consts.push((c, State::Waiting)),
                ItemKind::Static(s) => statics.push((s, None)),
                ItemKind::Fn(f) if f.abi.is_some() => compiled.push(f),
                ItemKind::Fn(f) => helpers.push(f),
                ItemKind::Other => {}
            }
        }
        Eval {
            target,
            consts,
            statics,
            helpers,
            compiled,
            scopes: vec![],
            labels: 0,
            guard: None,
            insts: 0,
            depth: 0,
            nest: 0,
            declared: 0,
            steps: 0,
        }
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
        if let Some(v) = self.global(name, span)? {
            return Ok(Some(Binding::Value(v)));
        }
        Ok(self.statics.iter().position(|(s, _)| s.name == name).map(Binding::Static))
    }

    /// The value of `e`, which must be known now: lent where it is held, if `e` names it.
    fn lent(&mut self, e: &Expr, want: Want) -> Result<Rc<Value>, Error> {
        let at = |m: String| err(e.span(), m);
        match e {
            Expr::Path(path, span) if path.len() == 1 => match self.lookup(&path[0], *span)? {
                Some(Binding::Value(v)) => Ok(v),
                Some(b) => Err(at(format!(
                    "`{}` is {}, known only when the kernel runs; this wants a value known now",
                    path[0],
                    b.kind()
                ))),
                None => Err(at(format!("`{}` is not defined", path[0]))),
            },
            _ => self.value(e, want).map(Rc::new),
        }
    }

    /// The variable the module's static `i` is, named at `span`: the target makes it the first
    /// time.
    fn static_var(&mut self, i: usize, span: Span) -> Result<T::Var, Error> {
        if let Some(v) = self.statics[i].1 {
            return Ok(v);
        }
        let v = self.target.global(&self.statics[i].0.name).map_err(|m| err(span, m))?;
        self.statics[i].1 = Some(v);
        Ok(v)
    }

    /// The module constant `name`, evaluated the first time it is named, in a scope of
    /// constants alone.
    fn global(&mut self, name: &str, span: Span) -> Result<Option<Rc<Value>>, Error> {
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
        let v = Rc::new(v?);
        self.consts[i].1 = State::Done(v.clone());
        Ok(Some(v))
    }

    /// The compile-time arguments of each of `f`'s instances, as [`instances`] has them: those
    /// its `#[instances(..)]` lists, evaluated in a scope of constants alone.
    fn listed(&mut self, f: &Fn) -> Result<Vec<Vec<Value>>, Error> {
        let name = &f.name;
        let Some(list) = &f.instances else {
            return match (&f.abi, f.generics.first()) {
                (None, _) => Ok(vec![]),
                (Some(_), None) => Ok(vec![vec![]]),
                (Some(_), Some(g)) => Err(err(
                    g.span,
                    format!(
                        "`{name}` takes compile-time arguments, so it is compiled for each instance its \
                         `#[instances(..)]` lists, as in `#[instances(4, 8)]`"
                    ),
                )),
            };
        };
        if f.abi.is_none() {
            return Err(err(
                list.span,
                "a helper is compiled where it is called, for what `::<..>` gives it there: `#[instances(..)]` lists \
                 what a function that names an ABI is compiled for",
            ));
        }
        if f.generics.is_empty() {
            return Err(err(
                list.span,
                format!("`{name}` takes no compile-time arguments, so it is compiled once, with no `#[instances(..)]`"),
            ));
        }
        for g in &f.generics {
            machine_generic(g)?;
        }
        if list.values.is_empty() {
            return Err(err(list.span, "`#[instances(..)]` lists each instance, and this lists none"));
        }
        let outer = std::mem::take(&mut self.scopes);
        let listed = self.instances_of(f, &list.values);
        self.scopes = outer;
        listed
    }

    /// The compile-time arguments of `f` that each of `values` gives, as `#[instances(..)]`
    /// lists them, each instance once.
    fn instances_of(&mut self, f: &Fn, values: &[Expr]) -> Result<Vec<Vec<Value>>, Error> {
        let mut out: Vec<Vec<Value>> = vec![];
        for e in values {
            let given = match (f.generics.len(), e) {
                (1, e) => vec![e],
                (n, Expr::Tuple(items, _)) if items.len() == n => items.iter().collect(),
                (_, e) => {
                    let names: Vec<&str> = f.generics.iter().map(|g| g.name.as_str()).collect();
                    return Err(err(
                        e.span(),
                        format!(
                            "an instance of `{}` gives its {} compile-time arguments in parentheses, as `({})`",
                            f.name,
                            names.len(),
                            names.join(", ")
                        ),
                    ));
                }
            };
            let args = self.generics(f, given)?;
            if out.contains(&args) {
                return Err(err(e.span(), format!("`{}` is listed twice", named(&f.name, &args))));
            }
            out.push(args);
        }
        Ok(out)
    }

    /// The values `given` for `f`'s generics, each of its generic's type.
    fn generics<'e>(&mut self, f: &Fn, given: impl IntoIterator<Item = &'e Expr>) -> Result<Vec<Value>, Error> {
        let mut out = vec![];
        for (g, e) in f.generics.iter().zip(given) {
            let ty = Type::Name(g.ty.clone(), g.span);
            let v = self.value(e, want(&ty))?;
            out.push(self.typed(v, &ty, e.span())?);
        }
        Ok(out)
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

    /// Runs `f` one level down in what is running, or says that it nests too deep, where its
    /// level starts at `span`.
    fn within<R>(&mut self, span: Span, f: impl FnOnce(&mut Self) -> Result<R, Error>) -> Result<R, Error> {
        nested(self.nest, span)?;
        self.nest += 1;
        let r = f(self);
        self.nest -= 1;
        r
    }

    /// The value of `e`, which must be known now.
    fn value(&mut self, e: &Expr, want: Want) -> Result<Value, Error> {
        self.within(e.span(), |this| this.value_of(e, want))
    }

    fn value_of(&mut self, e: &Expr, want: Want) -> Result<Value, Error> {
        let at = |m: String| err(e.span(), m);
        match e {
            Expr::Lit(text, _) => value::literal(text, want).map_err(at),
            Expr::Bool(b, _) => Ok(Value::Bool(*b)),
            Expr::Str(..) => Err(at("a string is no value here; `note(\"..\")` takes one".into())),
            Expr::Path(path, _) if path.len() == 1 => {
                let v = self.lent(e, want)?;
                self.elements(&v, e.span())?;
                Ok(Rc::unwrap_or_clone(v))
            }
            Expr::Path(path, _) => {
                let p = path.join("::");
                value::path(&p).ok_or_else(|| at(format!("`{p}` is not a constant kernite knows")))
            }
            Expr::Field(..) => Err(at("a field is a view of a variable, not a value".into())),
            Expr::Instance { .. } => {
                Err(at("an instance of a function is for an instruction to name, as `call` does, not a value".into()))
            }
            Expr::Index(base, index, _) => {
                let i = self.count(index)?;
                let lent = self.lent(base, want)?;
                match &*lent {
                    Value::Array(xs) => {
                        let x = xs
                            .get(i)
                            .ok_or_else(|| at(format!("index {i} is past the end of {} elements", xs.len())))?;
                        self.step(value::held(x), e.span())?;
                        Ok(x.clone())
                    }
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
                let recv = self.lent(recv, want)?;
                let args = args.iter().map(|a| self.value(a, Want::Any)).collect::<Result<Vec<_>, _>>()?;
                let v = value::method(&recv, name, &args).map_err(at)?;
                self.elements(&v, e.span())?;
                Ok(v)
            }
            // A literal negated is one number, as Rust reads `-128i8`.
            Expr::Unary(UnOp::Neg, inner, _) if matches!(**inner, Expr::Lit(..)) => {
                let Expr::Lit(text, _) = &**inner else { unreachable!("a literal") };
                value::negated(text, want).map_err(at)
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
                let n = (end - *l).max(0) as u128;
                value::fits(n).map_err(at)?;
                self.step(n, e.span())?;
                Ok(Value::Array((*l..end).map(|x| Value::Int(x, ty)).collect()))
            }
            Expr::Array(items, _) => {
                // Items past the most are worked out only to count what they hold, for the error,
                // and not kept.
                let (mut xs, mut held) = (vec![], 0);
                for x in items {
                    let x = self.value(x, want)?;
                    nested(value::nests(&x), e.span())?;
                    held += value::held(&x);
                    if value::fits(held).is_ok() {
                        xs.push(x);
                    }
                }
                value::fits(value::array(held, items.len())).map_err(at)?;
                // Its items were counted where they were made: it holds them, and copies none.
                self.step(items.len() as u128, e.span())?;
                Ok(Value::Array(xs))
            }
            Expr::Repeat(item, len, _) => {
                let v = self.value(item, want)?;
                nested(value::nests(&v), e.span())?;
                let n = self.count(len)?;
                let held = value::array(n as u128 * value::held(&v), n);
                value::fits(held).map_err(at)?;
                self.step(held, e.span())?;
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
                Some(Binding::Static(i)) => self.static_var(i, *span).map(Some),
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
            Expr::Field(base, field, span) => match self.place(base)? {
                Some(v) => self.target.field(v, field).map(Some).map_err(|m| err(*span, m)),
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
            return match b {
                Binding::Static(i) => self.static_var(i, *span).map(Binding::Var),
                b => Ok(b),
            };
        }
        match self.place(e)? {
            Some(v) => Ok(Binding::Var(v)),
            None => Ok(Binding::Value(Rc::new(self.value(e, Want::Any)?))),
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
                Some(Binding::Static(i)) => self.static_var(i, *span).map(Operand::Var),
                Some(Binding::Value(v)) => scalar(&v, *span),
                Some(Binding::Vars(vs)) => Ok(Operand::Vector(vs.into_iter().map(Operand::Var).collect())),
                Some(Binding::Label(l)) => Ok(Operand::Label(l)),
                None => self.unscoped(&path[0], *span),
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
            Expr::Index(base, index, span) => match self.place(e)? {
                Some(v) => Ok(Operand::Var(v)),
                None => match self.unbound(base)? {
                    Some(word) => Ok(Operand::Sized(word, Box::new(self.memory(index, *span)?))),
                    None => scalar(&self.value(e, Want::Any)?, e.span()),
                },
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
            Expr::Instance { func, generics, span } => self.instance_named(func, generics, *span),
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
            _ => scalar(&self.value(e, Want::Any)?, e.span()),
        }
    }

    /// The name `e` is, if it is nothing in scope and no function of the module: `dword` of
    /// `dword[x]`.
    fn unbound(&mut self, e: &Expr) -> Result<Option<String>, Error> {
        let Expr::Path(path, span) = e else { return Ok(None) };
        let [name] = path.as_slice() else { return Ok(None) };
        let function = self.compiled.iter().chain(&self.helpers).any(|f| f.name == *name);
        Ok((self.lookup(name, *span)?.is_none() && !function).then(|| name.clone()))
    }

    /// What a name that is nothing in scope gives an instruction: a function of the module
    /// that names an ABI, or a name the machine has.
    fn unscoped(&self, name: &str, span: Span) -> Result<Operand<T::Var>, Error> {
        if let Some(f) = self.compiled.iter().find(|f| f.name == name) {
            if !f.generics.is_empty() {
                return Err(err(
                    span,
                    format!("`{name}` is compiled for each of its instances: an instruction names one, `{name}::<..>`"),
                ));
            }
            return Ok(Operand::Func(name.to_string()));
        }
        if self.helpers.iter().any(|f| f.name == name) {
            return Err(err(
                span,
                format!("`{name}` is a helper, inlined where it is called, `{name}(..)`: no instruction calls it"),
            ));
        }
        Ok(Operand::Special(name.to_string()))
    }

    /// `func::<generics>`, an instance of a function of the module that names an ABI, as an
    /// instruction names it: one its `#[instances(..)]` lists.
    fn instance_named(&mut self, func: &Expr, generics: &[Expr], span: Span) -> Result<Operand<T::Var>, Error> {
        let name = match func {
            Expr::Path(path, _) if path.len() == 1 => &path[0],
            _ => return Err(err(span, "an instance is of a function of the module, by its name, as `tile::<4>`")),
        };
        let Some(&f) = self.compiled.iter().find(|f| f.name == *name) else {
            let what = match self.helpers.iter().any(|f| f.name == *name) {
                true => "a helper, inlined where it is called",
                false => "no function of the module",
            };
            return Err(err(span, format!("`{name}` is {what}: an instruction names a function that names an ABI")));
        };
        if f.generics.is_empty() {
            return Err(err(
                span,
                format!("`{name}` takes no compile-time arguments: an instruction names it, `{name}`"),
            ));
        }
        if generics.len() != f.generics.len() {
            return Err(err(
                span,
                format!("`{name}` takes {} values in `::<..>`; this gives {}", f.generics.len(), generics.len()),
            ));
        }
        let args = self.generics(f, generics)?;
        let listed = self.listed(f)?;
        if !listed.contains(&args) {
            let listed: Vec<String> = listed.iter().map(|a| format!("`{}`", named(name, a))).collect();
            return Err(err(
                span,
                format!(
                    "`{}` is not compiled: `{name}`'s `#[instances(..)]` lists {}",
                    named(name, &args),
                    listed.join(", ")
                ),
            ));
        }
        Ok(Operand::Func(named(name, &args)))
    }

    /// `[inner]`: memory at an address, memory holding a value known now, or the code of a
    /// function of the module that names an ABI, `[f]` or `[tile::<4>]`.
    fn memory(&mut self, inner: &Expr, span: Span) -> Result<Operand<T::Var>, Error> {
        let func = match inner {
            Expr::Path(path, at) if path.len() == 1 && self.lookup(&path[0], *at)?.is_none() => {
                Some(self.unscoped(&path[0], *at)?)
            }
            Expr::Instance { func, generics, span } => Some(self.instance_named(func, generics, *span)?),
            _ => None,
        };
        if let Some(Operand::Func(f)) = func {
            return Ok(Operand::Code(f));
        }
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

    /// A statement. One that holds others, `for`, `if` or a block, is a level of nesting for
    /// them, and so is each `else if`; a helper's body and a kernel's are not.
    fn stmt(&mut self, s: &Stmt) -> Result<(), Error> {
        match s {
            Stmt::For { span, .. } | Stmt::If { span, .. } => self.within(*span, |this| this.stmt_of(s)),
            Stmt::Block(b) => self.within(b.span, |this| this.stmt_of(s)),
            _ => self.stmt_of(s),
        }
    }

    fn stmt_of(&mut self, s: &Stmt) -> Result<(), Error> {
        match s {
            Stmt::Let { name, span, ty, value } => self.let_(name, *span, ty.as_ref(), value.as_ref()),
            Stmt::Const(c) => {
                let v = self.constant(c)?;
                self.bind(&c.name, Binding::Value(Rc::new(v)));
                Ok(())
            }
            Stmt::For { var, iter, body, .. } => {
                let items: Vec<Binding<T::Var>> = match self.binding(iter)? {
                    Binding::Value(ref v) if let Value::Array(xs) = &**v => {
                        // An array made by the header and one that is lent are both copied below,
                        // item by item. The count of the one made stands for its copy, and the one
                        // lent is counted here.
                        if Rc::strong_count(v) > 1 {
                            self.step(value::held(v), iter.span())?;
                        }
                        xs.iter().map(|x| Binding::Value(Rc::new(x.clone()))).collect()
                    }
                    Binding::Vars(vs) => vs.into_iter().map(Binding::Var).collect(),
                    other => {
                        return Err(err(
                            iter.span(),
                            format!("`for` runs over a range, an array or variables, not {}", other.kind()),
                        ));
                    }
                };
                for item in items {
                    self.step(1, iter.span())?;
                    self.scopes.push(vec![(var.clone(), item)]);
                    let r = self.block(body);
                    self.scopes.pop();
                    r?;
                }
                Ok(())
            }
            Stmt::If { cond, then, otherwise, span } => self.if_(cond, then, otherwise.as_deref(), *span),
            Stmt::Block(b) => self.block(b),
            Stmt::Expr(e) => {
                self.step(1, e.span())?;
                self.call(e)
            }
        }
    }

    /// Counts `n` more variables declared by `let`. The kernel that would pass the most is refused
    /// at `at`, before any is made.
    fn declare(&mut self, n: usize, at: Span) -> Result<(), Error> {
        let all = self.declared as u128 + n as u128;
        if all > VARIABLES as u128 {
            return Err(err(at, format!("{all} variables are too many: a kernel declares {VARIABLES} at most")));
        }
        self.declared += n;
        Ok(())
    }

    /// Counts `n` more steps. The function that would pass the most is refused at `at`, where it
    /// takes the step.
    fn step(&mut self, n: u128, at: Span) -> Result<(), Error> {
        self.steps += n;
        if self.steps > STEPS {
            return Err(err(at, format!("this takes more than {STEPS} steps to work out")));
        }
        Ok(())
    }

    /// Counts the elements of `v` at `at`, if it is an array copied from a name or made by a
    /// method. A number read by name or made by a method costs none.
    fn elements(&mut self, v: &Value, at: Span) -> Result<(), Error> {
        match v {
            Value::Array(_) => self.step(value::held(v), at),
            _ => Ok(()),
        }
    }

    fn let_(&mut self, name: &str, span: Span, ty: Option<&Type>, value: Option<&Expr>) -> Result<(), Error> {
        let b = match (ty, value) {
            (Some(t @ (Type::Name(..) | Type::Generic(..))), None) => {
                let ty = self.type_name(t)?;
                self.declare(1, span)?;
                let v = self.target.var(name, &ty).map_err(|m| err(t.span(), m))?;
                self.target.declared_at(v, span);
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
                self.declare(n, len.span())?;
                let mut vs = Vec::with_capacity(n);
                for i in 0..n {
                    let each = format!("{name}{i}");
                    let v = self.target.var(&each, t).map_err(|m| err(*tspan, m))?;
                    self.target.declared_at(v, span);
                    vs.push(v);
                }
                Binding::Vars(vs)
            }
            (None, Some(e)) => self.binding(e)?,
            (Some(t), Some(e)) => {
                let v = self.value(e, want(t))?;
                Binding::Value(Rc::new(self.typed(v, t, e.span())?))
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
        // The target checks the variable as it guards an instruction: a block that runs none
        // checks nothing. Both blocks run before either count is read, so what is wrong inside
        // either is said first.
        let ran = self.guarded(Guard { pred, negated }, |e| e.block(then))?;
        let other = otherwise.map_or(Ok(true), |s| self.guarded(Guard { pred, negated: !negated }, |e| e.stmt(s)))?;
        let which = match (ran, other) {
            (true, true) => return Ok(()),
            (false, _) => "its block",
            (true, false) => "its `else`",
        };
        Err(err(span, format!("an `if` on a variable guards each instruction inside, and {which} runs none")))
    }

    /// Runs `body`, each instruction it runs under `guard`: whether it ran one.
    fn guarded(
        &mut self,
        guard: Guard<T::Var>,
        body: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<bool, Error> {
        let ran = self.insts;
        self.guard = Some(guard);
        let r = body(self);
        self.guard = None;
        r.map(|()| self.insts > ran)
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
                    return Err(err(*span, format!("`{}` names nothing kernite calls", path.join("::"))));
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
                if self.compiled.iter().any(|f| f.name == *name) {
                    return Err(err(
                        *span,
                        format!(
                            "`{name}` names an ABI, so it is compiled on its own, not inlined as a helper is: an \
                             instruction calls it, `call(r, {name}, (a, b))`"
                        ),
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
        self.insts += 1;
        self.target.inst_at(span);
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
                Some(Binding::Value(v)) => {
                    self.step(value::held(&v), *sspan)?;
                    out += &v.to_string();
                }
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
        let values = self.generics(f, generics)?;
        let scope = f.generics.iter().zip(values).map(|(g, v)| (g.name.clone(), Binding::Value(Rc::new(v)))).collect();
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
fn scalar<V>(v: &Value, span: Span) -> Result<Operand<V>, Error> {
    match *v {
        Value::Int(x, _) if (i128::from(i64::MIN)..=i128::from(u64::MAX)).contains(&x) => Ok(Operand::Int(x)),
        Value::Int(x, _) => Err(err(span, format!("{x} is wider than 64 bits"))),
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
