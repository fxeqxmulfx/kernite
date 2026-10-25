//! Values the compiler computes with: what constants, generics and loop counters hold. Floats
//! are `f32` unless written or declared `f64`, and their arithmetic is the machine's, so a
//! constant comes out as Rust computes it.

use std::fmt;

use crate::ast::{BinOp, UnOp};

/// An integer type, as a suffix or a declaration names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntTy {
    /// `i8`.
    I8,
    /// `i16`.
    I16,
    /// `i32`.
    I32,
    /// `i64`.
    I64,
    /// `isize`.
    Isize,
    /// `u8`.
    U8,
    /// `u16`.
    U16,
    /// `u32`.
    U32,
    /// `u64`.
    U64,
    /// `usize`.
    Usize,
}

const INT_TYS: [(&str, IntTy); 10] = [
    ("usize", IntTy::Usize),
    ("isize", IntTy::Isize),
    ("u16", IntTy::U16),
    ("u32", IntTy::U32),
    ("u64", IntTy::U64),
    ("i16", IntTy::I16),
    ("i32", IntTy::I32),
    ("i64", IntTy::I64),
    ("u8", IntTy::U8),
    ("i8", IntTy::I8),
];

impl IntTy {
    /// The type `name` names.
    pub fn parse(name: &str) -> Option<IntTy> {
        INT_TYS.iter().find(|(n, _)| *n == name).map(|&(_, t)| t)
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        INT_TYS.iter().find(|(_, t)| *t == self).map(|&(n, _)| n).expect("every type is in the table")
    }

    /// Its width in bits.
    pub fn bits(self) -> u32 {
        match self {
            IntTy::I8 | IntTy::U8 => 8,
            IntTy::I16 | IntTy::U16 => 16,
            IntTy::I32 | IntTy::U32 => 32,
            IntTy::I64 | IntTy::U64 | IntTy::Isize | IntTy::Usize => 64,
        }
    }

    fn signed(self) -> bool {
        matches!(self, IntTy::I8 | IntTy::I16 | IntTy::I32 | IntTy::I64 | IntTy::Isize)
    }

    fn range(self) -> (i128, i128) {
        let bits = self.bits();
        if self.signed() { (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) } else { (0, (1 << bits) - 1) }
    }

    /// `x` cut to its width, as `as` does.
    fn wrap(self, x: i128) -> i128 {
        let bits = self.bits();
        let low = x & ((1 << bits) - 1);
        if self.signed() && low >= 1 << (bits - 1) { low - (1 << bits) } else { low }
    }
}

/// A value known at compile time.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// An integer, with its type if a suffix or a declaration gave it one.
    Int(i128, Option<IntTy>),
    /// An `f32`.
    F32(f32),
    /// An `f64`.
    F64(f64),
    /// A `bool`.
    Bool(bool),
    /// An array: a range, a list, data.
    Array(Vec<Value>),
}

impl Value {
    /// What kind of value it is, for messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Int(..) => "an integer",
            Value::F32(_) => "an f32",
            Value::F64(_) => "an f64",
            Value::Bool(_) => "a bool",
            Value::Array(_) => "an array",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(x, _) => write!(f, "{x}"),
            Value::F32(x) => write!(f, "{x}"),
            Value::F64(x) => write!(f, "{x}"),
            Value::Bool(x) => write!(f, "{x}"),
            Value::Array(xs) => {
                write!(f, "[")?;
                for (i, x) in xs.iter().enumerate() {
                    write!(f, "{}{x}", if i > 0 { ", " } else { "" })?;
                }
                write!(f, "]")
            }
        }
    }
}

/// The type a value is wanted at, which an unsuffixed literal takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// Whatever it is: integers untyped, floats `f32`.
    Any,
    /// An integer of this type.
    Int(IntTy),
    /// An `f64`.
    F64,
}

impl Want {
    /// What a declaration of type `ty` wants.
    pub fn of(ty: &str) -> Want {
        match (IntTy::parse(ty), ty) {
            (Some(t), _) => Want::Int(t),
            (None, "f64") => Want::F64,
            _ => Want::Any,
        }
    }
}

/// A number literal: `16`, `0x3f`, `8u8`, `0.5`, `1e-5`, `2.0f64`.
pub fn literal(text: &str, want: Want) -> Result<Value, String> {
    let t: String = text.chars().filter(|&c| c != '_').collect();
    let int_want = match want {
        Want::Int(ty) => Some(ty),
        _ => None,
    };
    let radix = match t.get(..2) {
        Some("0x") => 16,
        Some("0o") => 8,
        Some("0b") => 2,
        _ => 10,
    };
    let bad = || format!("`{text}` is not a number kgen reads");
    if radix != 10 {
        let body = &t[2..];
        let split = body.find(|c: char| !c.is_digit(radix)).unwrap_or(body.len());
        let (digits, suffix) = body.split_at(split);
        let ty = if suffix.is_empty() { int_want } else { Some(IntTy::parse(suffix).ok_or_else(bad)?) };
        let x = i128::from_str_radix(digits, radix).map_err(|_| bad())?;
        return int(x, ty);
    }
    if let Some(digits) = t.strip_suffix("f32") {
        return digits.parse().map(Value::F32).map_err(|_| bad());
    }
    if let Some(digits) = t.strip_suffix("f64") {
        return digits.parse().map(Value::F64).map_err(|_| bad());
    }
    if let Some(&(name, ty)) = INT_TYS.iter().find(|(name, _)| t.ends_with(name)) {
        let x = t[..t.len() - name.len()].parse().map_err(|_| bad())?;
        return int(x, Some(ty));
    }
    if t.contains(['.', 'e', 'E']) {
        return match want {
            Want::F64 => t.parse().map(Value::F64).map_err(|_| bad()),
            _ => t.parse().map(Value::F32).map_err(|_| bad()),
        };
    }
    int(t.parse().map_err(|_| bad())?, int_want)
}

/// `x` as an integer of type `ty`, if it is in range.
fn int(x: i128, ty: Option<IntTy>) -> Result<Value, String> {
    let (lo, hi) = ty.map_or((i64::MIN.into(), u64::MAX.into()), IntTy::range);
    if x < lo || x > hi {
        return Err(format!("{x} is out of range for {}", ty.map_or("an integer", IntTy::name)));
    }
    Ok(Value::Int(x, ty))
}

/// The type two integers share.
fn unify(a: Option<IntTy>, b: Option<IntTy>) -> Result<Option<IntTy>, String> {
    match (a, b) {
        (Some(a), Some(b)) if a != b => {
            Err(format!("{} and {} are different types; cast one with `as`", a.name(), b.name()))
        }
        _ => Ok(a.or(b)),
    }
}

fn op_text(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::And => "&",
        BinOp::Or => "|",
        BinOp::Xor => "^",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::AndAnd => "&&",
        BinOp::OrOr => "||",
    }
}

/// `a op b`.
pub fn binary(op: BinOp, a: &Value, b: &Value) -> Result<Value, String> {
    use BinOp::*;
    let cmp = |o: std::cmp::Ordering| {
        Value::Bool(match op {
            Eq => o.is_eq(),
            Ne => o.is_ne(),
            Lt => o.is_lt(),
            Le => o.is_le(),
            Gt => o.is_gt(),
            _ => o.is_ge(),
        })
    };
    let mismatch = || format!("`{}` has no meaning between {} and {}", op_text(op), a.kind(), b.kind());
    match (a, b) {
        (&Value::Int(x, tx), &Value::Int(y, ty)) => {
            let t = if matches!(op, Shl | Shr) { tx } else { unify(tx, ty)? };
            let over = || format!("{x} {} {y} overflows", op_text(op));
            let v = match op {
                Add => x.checked_add(y).ok_or_else(over)?,
                Sub => x.checked_sub(y).ok_or_else(over)?,
                Mul => x.checked_mul(y).ok_or_else(over)?,
                Div | Rem if y == 0 => return Err(format!("{x} {} 0", op_text(op))),
                Div => x / y,
                Rem => x % y,
                Shl | Shr => {
                    let bits = tx.map_or(64, IntTy::bits);
                    if !(0..bits.into()).contains(&y) {
                        return Err(format!("a shift by {y} is wider than its {bits} bits"));
                    }
                    let shifted = if op == Shl { x << y } else { x >> y };
                    return Ok(Value::Int(tx.map_or(shifted, |t| t.wrap(shifted)), tx));
                }
                And => x & y,
                Or => x | y,
                Xor => x ^ y,
                Eq | Ne | Lt | Le | Gt | Ge => return Ok(cmp(x.cmp(&y))),
                AndAnd | OrOr => return Err(mismatch()),
            };
            int(v, t)
        }
        (&Value::F32(x), &Value::F32(y)) => Ok(match op {
            Add => Value::F32(x + y),
            Sub => Value::F32(x - y),
            Mul => Value::F32(x * y),
            Div => Value::F32(x / y),
            Rem => Value::F32(x % y),
            Eq | Ne | Lt | Le | Gt | Ge => match x.partial_cmp(&y) {
                Some(o) => cmp(o),
                None => Value::Bool(op == Ne),
            },
            _ => return Err(mismatch()),
        }),
        (&Value::F64(x), &Value::F64(y)) => Ok(match op {
            Add => Value::F64(x + y),
            Sub => Value::F64(x - y),
            Mul => Value::F64(x * y),
            Div => Value::F64(x / y),
            Rem => Value::F64(x % y),
            Eq | Ne | Lt | Le | Gt | Ge => match x.partial_cmp(&y) {
                Some(o) => cmp(o),
                None => Value::Bool(op == Ne),
            },
            _ => return Err(mismatch()),
        }),
        (&Value::Bool(x), &Value::Bool(y)) => Ok(Value::Bool(match op {
            And | AndAnd => x && y,
            Or | OrOr => x || y,
            Xor | Ne => x != y,
            Eq => x == y,
            _ => return Err(mismatch()),
        })),
        _ => Err(mismatch()),
    }
}

/// `op a`.
pub fn unary(op: UnOp, a: &Value) -> Result<Value, String> {
    match (op, a) {
        (UnOp::Neg, &Value::Int(_, Some(t))) if !t.signed() => Err(format!("a {} cannot be negated", t.name())),
        (UnOp::Neg, &Value::Int(x, t)) => int(-x, t),
        (UnOp::Neg, &Value::F32(x)) => Ok(Value::F32(-x)),
        (UnOp::Neg, &Value::F64(x)) => Ok(Value::F64(-x)),
        (UnOp::Not, &Value::Bool(x)) => Ok(Value::Bool(!x)),
        (UnOp::Not, &Value::Int(x, t)) => Ok(Value::Int(t.map_or(!x, |t| t.wrap(!x)), t)),
        (UnOp::Neg, _) => Err(format!("{} cannot be negated", a.kind())),
        (UnOp::Not, _) => Err(format!("`!` has no meaning on {}", a.kind())),
    }
}

/// `a as ty`, as Rust casts.
pub fn cast(a: &Value, ty: &str) -> Result<Value, String> {
    let fail = || Err(format!("{} cannot be cast to {ty}", a.kind()));
    if let Some(t) = IntTy::parse(ty) {
        let (lo, hi) = t.range();
        return Ok(Value::Int(
            match *a {
                Value::Int(x, _) => t.wrap(x),
                Value::F32(x) => (x as i128).clamp(lo, hi),
                Value::F64(x) => (x as i128).clamp(lo, hi),
                Value::Bool(x) => x.into(),
                Value::Array(_) => return fail(),
            },
            Some(t),
        ));
    }
    match (ty, a) {
        ("f32", &Value::Int(x, _)) => Ok(Value::F32(x as f32)),
        ("f32", &Value::F32(x)) => Ok(Value::F32(x)),
        ("f32", &Value::F64(x)) => Ok(Value::F32(x as f32)),
        ("f64", &Value::Int(x, _)) => Ok(Value::F64(x as f64)),
        ("f64", &Value::F32(x)) => Ok(Value::F64(x.into())),
        ("f64", &Value::F64(x)) => Ok(Value::F64(x)),
        _ => fail(),
    }
}

/// The value `a` takes as a constant of type `ty`, or why it cannot.
pub fn typed(a: Value, ty: &str) -> Result<Value, String> {
    let wrong = |a: &Value| Err(format!("this is {}, not {ty}", a.kind()));
    match (IntTy::parse(ty), ty, a) {
        (Some(t), _, Value::Int(x, None)) => int(x, Some(t)),
        (Some(t), _, a @ Value::Int(_, Some(u))) => {
            if t == u {
                Ok(a)
            } else {
                wrong(&a)
            }
        }
        (None, "f32", a @ Value::F32(_)) | (None, "f64", a @ Value::F64(_)) | (None, "bool", a @ Value::Bool(_)) => {
            Ok(a)
        }
        (_, _, a) => wrong(&a),
    }
}

/// `recv.name(args)`, for the methods a constant has.
pub fn method(recv: &Value, name: &str, args: &[Value]) -> Result<Value, String> {
    match (name, recv, args) {
        ("step_by", Value::Array(xs), &[Value::Int(n, _)]) if n > 0 => {
            Ok(Value::Array(xs.iter().step_by(n as usize).cloned().collect()))
        }
        ("rev", Value::Array(xs), []) => Ok(Value::Array(xs.iter().rev().cloned().collect())),
        ("len", Value::Array(xs), []) => Ok(Value::Int(xs.len() as i128, Some(IntTy::Usize))),
        ("min" | "max", a, [b]) => {
            let less = binary(BinOp::Lt, a, b)? == Value::Bool(true);
            Ok(if less == (name == "min") { a.clone() } else { b.clone() })
        }
        ("to_bits", &Value::F32(x), []) => Ok(Value::Int(x.to_bits().into(), Some(IntTy::U32))),
        ("to_bits", &Value::F64(x), []) => Ok(Value::Int(x.to_bits().into(), Some(IntTy::U64))),
        ("sqrt", &Value::F32(x), []) => Ok(Value::F32(x.sqrt())),
        ("sqrt", &Value::F64(x), []) => Ok(Value::F64(x.sqrt())),
        ("recip", &Value::F32(x), []) => Ok(Value::F32(1.0 / x)),
        ("recip", &Value::F64(x), []) => Ok(Value::F64(1.0 / x)),
        _ => Err(format!("{} has no method `{name}` taking {} arguments", recv.kind(), args.len())),
    }
}

/// The constants named by paths: `f32::INFINITY`, `u32::MAX`.
pub fn path(p: &str) -> Option<Value> {
    Some(match p {
        "f32::INFINITY" => Value::F32(f32::INFINITY),
        "f32::NEG_INFINITY" => Value::F32(f32::NEG_INFINITY),
        "f32::MAX" => Value::F32(f32::MAX),
        "f32::MIN" => Value::F32(f32::MIN),
        "f32::MIN_POSITIVE" => Value::F32(f32::MIN_POSITIVE),
        "f32::EPSILON" => Value::F32(f32::EPSILON),
        "f64::INFINITY" => Value::F64(f64::INFINITY),
        "f64::NEG_INFINITY" => Value::F64(f64::NEG_INFINITY),
        "f64::MAX" => Value::F64(f64::MAX),
        "f64::MIN" => Value::F64(f64::MIN),
        "f32::consts::LN_2" => Value::F32(std::f32::consts::LN_2),
        "f32::consts::LOG2_E" => Value::F32(std::f32::consts::LOG2_E),
        "f32::consts::PI" => Value::F32(std::f32::consts::PI),
        "f32::consts::E" => Value::F32(std::f32::consts::E),
        "f64::consts::LN_2" => Value::F64(std::f64::consts::LN_2),
        "f64::consts::LOG2_E" => Value::F64(std::f64::consts::LOG2_E),
        "f64::consts::PI" => Value::F64(std::f64::consts::PI),
        "f64::consts::E" => Value::F64(std::f64::consts::E),
        _ => {
            let (ty, which) = p.split_once("::")?;
            let t = IntTy::parse(ty)?;
            let (lo, hi) = t.range();
            match which {
                "MIN" => Value::Int(lo, Some(t)),
                "MAX" => Value::Int(hi, Some(t)),
                "BITS" => Value::Int(t.bits().into(), Some(IntTy::U32)),
                _ => return None,
            }
        }
    })
}

/// A value as little-endian bytes, the way data in memory holds it. Integers without a type
/// take the type of the first that has one, as Rust infers an array's, else `i32`.
pub fn bytes(v: &Value) -> Result<Vec<u8>, String> {
    fn first_type(v: &Value) -> Option<IntTy> {
        match v {
            Value::Int(_, t) => *t,
            Value::Array(xs) => xs.iter().find_map(first_type),
            _ => None,
        }
    }
    fn push(v: &Value, default: IntTy, out: &mut Vec<u8>) -> Result<(), String> {
        match *v {
            Value::Int(x, t) => {
                let t = t.unwrap_or(default);
                int(x, Some(t))?;
                out.extend_from_slice(&x.to_le_bytes()[..t.bits() as usize / 8]);
            }
            Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::Bool(_) => return Err("a bool is not data".into()),
            Value::Array(ref xs) => {
                for x in xs {
                    push(x, default, out)?;
                }
            }
        }
        Ok(())
    }
    let mut out = vec![];
    push(v, first_type(v).unwrap_or(IntTy::I32), &mut out)?;
    if out.is_empty() {
        return Err("empty data".into());
    }
    Ok(out)
}
