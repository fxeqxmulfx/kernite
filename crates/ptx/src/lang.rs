//! PTX as the language writes it: [`Kernel`] is the [`Target`] that checks each call against
//! the forms of [`crate::forms`], and writes the instruction out.
//!
//! The types are PTX's: `pred`; `b8` to `b128`, bits of no type in particular; `u8` to `u64`
//! and `s8` to `s64`; `f32` and `f64`; and the floats of 16 bits, `f16` and `bf16`, alone or
//! two together, `f16x2` and `bf16x2`, and `tf32`. An instruction is called by its PTX name,
//! words joined with dots, with the operands PTX writes: `add.rn.f32(acc, acc, x)`,
//! `setp.lt.u32(p, i, n)`, `ld.global.v4.f32((a, b, c, d), [p + 16])`.
//!
//! - A float `add`, `sub`, `mul`, `mad`, `fma`, `div`, `sqrt` or `rcp` says how it rounds,
//!   `.rn`, so that ptxas neither fuses nor approximates it: what it computes is what it says.
//! - An address is a variable and a displacement, `[p + 16]`; a generic or a global one is
//!   in 64 bits. A kernel's pointers arrive as global addresses, for `ld.global` and
//!   `st.global`.
//! - `mov` and `cvt` read the special registers: `tid.x`, `laneid`, `clock64`.
//! - Operands taken together are a tuple, `(a, b)`, or variables declared together; two
//!   results written at once are `p | q`; `_` is a result left unwritten, where PTX lets one
//!   be; and `!p` is a predicate read inverted.
//! - `if p { .. }` runs each instruction inside where `p`, a `pred`, holds, and `if !p` where
//!   it does not; a guarded `bra(label)` is a branch.
//! - A kernel returns nothing: `ret()`, which the end of the body adds if control reaches it.
//!
//! What an instruction needs of PTX and of the GPU, its [`Needs`], the kernel needs: an
//! instruction that no GPU has with what the kernel needs already is an error where it is.

use std::collections::BTreeSet;

use kgen_core::Width::{W8, W16, W32, W64, W128};
use kgen_core::{
    Access, Arg, Base, Class, Flow, Inst, Item, Label, Mem, Operand, PReg, Program, RegInfo, Signature, VReg, Width,
};
use kgen_lang::{Fault, Guard, Operand as Given, Target};

use crate::forms::{self, Bound, Count, Imm, Role, Spec};
use crate::{Entry, Kind, Needs, Op, Param, Ty};

/// A variable: a register of the type it was declared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var {
    reg: VReg,
    ty: Ty,
}

/// A kernel being written in the language, for PTX.
pub struct Kernel {
    p: Program<Op>,
    params: Vec<Param>,
    regs: Vec<(String, Ty)>,
    taken: BTreeSet<String>,
    partial: Vec<Partial>,
    needs: Vec<(Needs, String)>,
}

/// A write under a guard, which leaves what was there where the guard does not hold. It is
/// taken for a write while nothing reads the variable but under the same guard: a read under
/// another, or after the guard changes, may find what was there before, so then the write
/// reads it too. A write under the opposite guard, the other side of `if p { .. } else
/// { .. }`, makes the two a write on every path.
struct Partial {
    var: VReg,
    /// The guard, while it holds as it did; `None` once it is written again, or once control
    /// can come from elsewhere.
    side: Option<(VReg, bool)>,
    /// The instruction, by its place in the body, and the operand among its arguments.
    at: usize,
    arg: usize,
}

/// An instruction's operands written out: their text, what they read and write, and the
/// special registers among them with what those need.
#[derive(Default)]
struct Written {
    texts: Vec<String>,
    args: Vec<Arg<VReg>>,
    needs: Needs,
    sregs: Vec<&'static str>,
}

/// Why operands are not those of a form.
enum Miss {
    /// There are too few or too many: it takes from the first to the second.
    Count(usize, usize),
    /// One is wrong: how many before it were right, whether it has the shape the form wants
    /// there, and what is wrong.
    At(usize, bool, Fault),
}

impl Kernel {
    /// An empty kernel called `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Kernel {
            p: Program {
                name: name.into(),
                signature: Signature { params: vec![], result: None },
                regs: vec![],
                params: vec![],
                results: vec![],
                body: vec![],
                slots: vec![],
                consts: vec![],
            },
            params: vec![],
            regs: vec![],
            taken: BTreeSet::new(),
            partial: vec![],
            needs: vec![],
        }
    }

    /// The kernel as written.
    pub fn finish(self) -> Entry {
        Entry { program: self.p, params: self.params, regs: self.regs, needs: self.needs }
    }

    /// A new variable, its register named after it: with `_` after a special register's
    /// name, and then a number if another has the name.
    fn new_var(&mut self, name: &str, ty: Ty) -> Var {
        let class = match ty {
            Ty::Pred => Class::Pred,
            Ty::F32 => Class::Float,
            _ => Class::Gpr,
        };
        let id = self.p.regs.len() as u32;
        self.p.regs.push(RegInfo { class, name: name.to_string() });
        let base = if forms::reserved(name) { format!("{name}_") } else { name.to_string() };
        let (mut reg, mut k) = (base.clone(), 1);
        while self.taken.contains(&reg) {
            reg = format!("{base}_{k}");
            k += 1;
        }
        self.taken.insert(reg.clone());
        self.regs.push((reg, ty));
        Var { reg: VReg { id, class }, ty }
    }

    fn reg(&self, v: Var) -> &str {
        &self.regs[v.reg.id as usize].0
    }

    /// What the kernel needs so far.
    fn needed(&self) -> Needs {
        self.needs.iter().fold(Needs::default(), |n, (m, _)| n.and(*m))
    }

    /// Notes that the kernel needs `needs` for `what`, an instruction; or says why no GPU has
    /// it as well as what the kernel needs already.
    fn require(&mut self, needs: Needs, what: &str) -> Result<(), String> {
        let had = self.needed();
        let all = had.and(needs);
        if !all.possible() {
            let older = needs.below != 0 && had.sm >= needs.below;
            let message = if older {
                let (n, other) = self.needs.iter().max_by_key(|(n, _)| n.sm).expect("something needs that GPU");
                format!(
                    "`{what}` is for GPUs older than sm_{}, and `{other}` before it needs sm_{} or newer: no GPU has \
                     both",
                    needs.below, n.sm
                )
            } else {
                let (n, other) =
                    self.needs.iter().find(|(n, _)| n.below != 0 && needs.sm >= n.below).expect("something is older");
                format!(
                    "`{what}` needs sm_{} or newer, and `{other}` before it is for GPUs older than sm_{}: no GPU has \
                     both",
                    needs.sm, n.below
                )
            };
            return Err(message);
        }
        if all.and(Needs::FLOOR) != had.and(Needs::FLOOR) {
            self.needs.push((needs, what.to_string()));
        }
        Ok(())
    }

    /// Adds the instruction, keeping account of the writes under guards.
    fn push(&mut self, line: String, args: Vec<Arg<VReg>>, guard: Option<Guard<Var>>, flow: Flow) {
        let at = self.p.body.len();
        let side = guard.map(|g| (g.pred.reg, g.negated));
        let reads: Vec<VReg> = args.iter().flat_map(Arg::reads).chain(guard.map(|g| g.pred.reg)).collect();
        let body = &mut self.p.body;
        self.partial.retain(|w| {
            let stale = reads.contains(&w.var) && (w.side.is_none() || w.side != side);
            if stale && let Item::Inst(inst) = &mut body[w.at] {
                inst.args[w.arg].access = Access::ReadWrite;
            }
            !stale
        });
        for (i, a) in args.iter().enumerate() {
            let Some(var) = a.writes() else { continue };
            match side {
                Some((pred, negated))
                    if !self.partial.iter().any(|w| w.var == var && w.side == Some((pred, !negated))) =>
                {
                    self.partial.push(Partial { var, side, at, arg: i });
                }
                _ => self.partial.retain(|w| w.var != var),
            }
            for w in &mut self.partial {
                if w.side.is_some_and(|(p, _)| p == var) {
                    w.side = None;
                }
            }
        }
        let implicit = guard.map(|g| Arg { operand: Operand::Reg(g.pred.reg, W32), access: Access::Read });
        self.p.body.push(Item::Inst(Inst { op: Op(line), args, implicit: implicit.into_iter().collect(), flow }));
    }

    /// `args` as the operands of `b`, the form of the instruction `name`.
    fn operands(&self, b: &Bound, name: &str, args: &[Given<Var>]) -> Result<Written, Miss> {
        let specs: Vec<&Spec> = b.form.operands.iter().filter(|s| s.when.is_none_or(|w| b.has(w))).collect();
        let least = specs.iter().filter(|s| !s.optional).count();
        if !(least..=specs.len()).contains(&args.len()) {
            return Err(Miss::Count(least, specs.len()));
        }
        let mut w = Written::default();
        for (i, (spec, a)) in specs.iter().zip(args).enumerate() {
            let text =
                self.operand(b, spec, a, name, &mut w).map_err(|m| Miss::At(i, shaped(b, spec, a), Fault::at(i, m)))?;
            w.texts.push(text);
        }
        Ok(w)
    }

    /// `a` as the operand `spec` of `b`, written out, with what it reads and writes added to
    /// `w`; or what is wrong with it.
    fn operand(&self, b: &Bound, spec: &Spec, a: &Given<Var>, name: &str, w: &mut Written) -> Result<String, String> {
        match (spec.role, a) {
            (Role::Label, Given::Label(l)) => {
                w.args.push(Arg { operand: Operand::Label(Label(*l)), access: Access::Read });
                Ok(format!("$L{l}"))
            }
            (Role::Addr(access), Given::Mem(m)) => self.address(b, m, access, name, w),
            (Role::Imm(imm), Given::Int(x)) => known(imm, *x, name),
            (Role::Dst | Role::Src, _) => self.value(b, spec, a, name, w),
            (_, Given::Data(_)) => Err(format!("`{name}` takes a value as it is: `1.5`, not `[1.5]`")),
            (_, a) => Err(format!("`{name}` wants {}; this is {}", describe(b, spec), given(a))),
        }
    }

    /// `a` as the register operand `spec`: one, a few together, or two written at once.
    fn value(&self, b: &Bound, spec: &Spec, a: &Given<Var>, name: &str, w: &mut Written) -> Result<String, String> {
        let Some((second, optional)) = &spec.second else {
            return self.several(b, spec, a, name, w);
        };
        match a {
            Given::Pair(x, y) => {
                let x = self.several(b, spec, x, name, w)?;
                let y = self.several(b, second, y, name, w)?;
                if x == "_" && y == "_" {
                    return Err(format!("`{name}` writes one of the two at least: `p | _` or `_ | q`"));
                }
                Ok(format!("{x}|{y}"))
            }
            _ if *optional => self.several(b, spec, a, name, w),
            _ => Err(format!("`{name}` writes two results at once, `p | q`; this is {}", given(a))),
        }
    }

    /// `a` as `spec`, as many registers as it takes together.
    fn several(&self, b: &Bound, spec: &Spec, a: &Given<Var>, name: &str, w: &mut Written) -> Result<String, String> {
        let n = lanes(b, spec);
        if n == 1 {
            return self.one(b, spec, a, name, w);
        }
        let Given::Vector(parts) = a else {
            return Err(format!("`{name}` wants {}; this is {}", describe(b, spec), given(a)));
        };
        if parts.len() != n {
            return Err(format!("`{name}` wants {}; this is {}", describe(b, spec), given(a)));
        }
        let mut texts = vec![];
        for part in parts {
            texts.push(self.one(b, spec, part, name, w)?);
        }
        if texts.iter().all(|t| t == "_") {
            return Err(format!("`{name}` writes one of them at least"));
        }
        Ok(format!("{{{}}}", texts.join(", ")))
    }

    /// `a` as one register of `spec`: a variable, or what else `spec` lets it be.
    fn one(&self, b: &Bound, spec: &Spec, a: &Given<Var>, name: &str, w: &mut Written) -> Result<String, String> {
        let t = b.ty(spec.ty.expect("a register's operand has a type"));
        let write = spec.role == Role::Dst;
        let var = |v: &Var, w: &mut Written| {
            let access = if write { Access::Write } else { Access::Read };
            w.args.push(Arg { operand: Operand::Reg(v.reg, width(v.ty)), access });
            format!("%{}", self.reg(*v))
        };
        match a {
            Given::Var(v) => {
                let fits = if b.form.wide { v.ty.fits_wide(t) } else { v.ty.fits(t) };
                if !fits {
                    return Err(format!("`{name}` wants a `{}` variable; this is a `{}`", t.name(), v.ty.name()));
                }
                Ok(var(v, w))
            }
            Given::Special(s) if s == "_" => match spec.sink {
                true => Ok("_".into()),
                false => Err(format!("`{name}` has no result here that it may leave unwritten, `_`")),
            },
            Given::Special(s) => match forms::sreg(s) {
                None => match forms::beyond(s) {
                    Some(why) => Err(format!("`{s}` is a special register of PTX's that kgen does not read: {why}")),
                    None => Err(format!("`{s}` is not defined")),
                },
                Some(_) if write => {
                    Err(format!("`{name}` writes a variable; `{s}` is a special register, which it reads"))
                }
                Some(r) if !spec.sreg => Err(format!("`{name}` reads no special register; `mov.{}` does", r.ty.name())),
                Some(r) if !r.read_as(t) => {
                    Err(format!("`{name}` reads a `{}`; `{s}` is a `{}`", t.name(), r.ty.name()))
                }
                Some(r) => {
                    w.needs = w.needs.and(r.needs);
                    if r.needs != Needs::default() {
                        w.sregs.push(&r.name);
                    }
                    Ok(format!("%{s}"))
                }
            },
            Given::Not(p) if spec.not => match **p {
                Given::Var(v) if v.ty == Ty::Pred => Ok(format!("!{}", var(&v, w))),
                _ => Err(format!("`{name}` reads a predicate inverted, `!p`; this is {} inverted", given(p))),
            },
            Given::Not(_) => Err(format!("`{name}` takes no operand inverted")),
            Given::Neg(_) => Err(format!("`{name}` takes no operand negated")),
            Given::Data(_) => Err(format!("`{name}` takes a value as it is: `1.5`, not `[1.5]`")),
            Given::Int(_) | Given::F32(_) | Given::F64(_) | Given::F16(_) | Given::Bf16(_)
                if write || spec.reg_only =>
            {
                Err(format!("`{name}` wants {}; this is {}", describe(b, spec), given(a)))
            }
            Given::Int(x) => {
                if let Some(below) = spec.below
                    && !(0..below).contains(x)
                {
                    return Err(format!("`{name}` takes 0 to {} here; {x} is not one of them", below - 1));
                }
                if let Some(step) = spec.step
                    && (*x <= 0 || x % step != 0)
                {
                    return Err(format!("`{name}` takes a multiple of {step} here, and more than 0; {x} is not one"));
                }
                if spec.fraction && *x != 1 {
                    return Err(fraction(name, x));
                }
                int(t, *x, name)
            }
            Given::F32(x) if spec.fraction && !(*x > 0.0 && *x <= 1.0) => Err(fraction(name, x)),
            Given::F32(x) => match t {
                Ty::F32 | Ty::B32 => Ok(format!("0f{:08X}", x.to_bits())),
                Ty::F64 => Err(format!("`{name}` wants an `f64`; {x} is an `f32`: `{x}f64`")),
                _ if t.kind() == Kind::Float => Err(half(t, name)),
                _ => Err(format!("`{name}` wants {}; this is {}", describe(b, spec), given(a))),
            },
            Given::F64(x) => match t {
                Ty::F64 | Ty::B64 => Ok(format!("0d{:016X}", x.to_bits())),
                Ty::F32 => Err(format!("`{name}` wants an `f32`; {x}f64 is an `f64`")),
                _ if t.kind() == Kind::Float => Err(half(t, name)),
                _ => Err(format!("`{name}` wants {}; this is {}", describe(b, spec), given(a))),
            },
            Given::F16(x) | Given::Bf16(x) => match t {
                Ty::B16 => Ok(format!("0x{x:04X}")),
                _ if t.kind() == Kind::Float => Err(half(t, name)),
                _ => Err(format!("`{name}` wants {}; this is {}", describe(b, spec), given(a))),
            },
            a => Err(format!("`{name}` wants {}; this is {}", describe(b, spec), given(a))),
        }
    }

    /// The address `m`, in the state space `b` names.
    fn address(
        &self,
        b: &Bound,
        m: &kgen_lang::Mem<Var>,
        access: Access,
        name: &str,
        w: &mut Written,
    ) -> Result<String, String> {
        if m.index.is_some() {
            return Err(format!(
                "`{name}` takes an address of a variable and a displacement; work out `p + 4 * i` first, as \
                 `mad.wide.u32(a, i, 4, p)` does"
            ));
        }
        let wide = b.space().wide();
        let Some(base) = m.base else {
            let bits = if wide { "a 64-bit" } else { "a" };
            return Err(format!("`{name}` takes an address that starts at {bits} variable"));
        };
        if base.ty.kind() == Kind::Float || base.ty.kind() == Kind::Pred {
            return Err(format!("`{name}` takes an address in an integer variable; this is a `{}`", base.ty.name()));
        }
        if wide && base.ty.bits() != 64 {
            return Err(format!("`{name}` takes an address in a 64-bit variable; this is a `{}`", base.ty.name()));
        }
        if !wide && !matches!(base.ty.bits(), 32 | 64) {
            return Err(format!("`{name}` takes an address in 32 bits or 64; this is a `{}`", base.ty.name()));
        }
        let disp = i32::try_from(m.disp).map_err(|_| format!("{} bytes are too far for an address", m.disp))?;
        let mem = Mem { base: Base::Reg(base.reg), index: None, disp, width: W32 };
        w.args.push(Arg { operand: Operand::Mem(mem), access });
        Ok(match disp {
            0 => format!("[%{}]", self.reg(base)),
            d => format!("[%{}+{d}]", self.reg(base)),
        })
    }
}

/// How many registers `spec` of `b` is.
fn lanes(b: &Bound, spec: &Spec) -> usize {
    match spec.count {
        Count::One => 1,
        Count::Fixed(n) => n,
        Count::Vec => b.lanes(),
    }
}

/// Whether `a` is what `spec` is, but for what it holds: a few together where it takes a
/// few, an address where it takes one.
fn shaped(b: &Bound, spec: &Spec, a: &Given<Var>) -> bool {
    match (spec.role, a) {
        (Role::Addr(_), Given::Mem(_)) | (Role::Label, Given::Label(_)) | (Role::Imm(_), Given::Int(_)) => true,
        (Role::Dst | Role::Src, Given::Vector(parts)) => lanes(b, spec) == parts.len(),
        (Role::Dst | Role::Src, Given::Pair(..)) => spec.second.is_some(),
        (Role::Dst | Role::Src, Given::Mem(_) | Given::Label(_)) => false,
        (Role::Dst | Role::Src, _) => lanes(b, spec) == 1,
        _ => false,
    }
}

/// The width of a register of type `t`, for what reads and writes it.
fn width(t: Ty) -> Width {
    match t.decl().bits() {
        8 => W8,
        16 => W16,
        64 => W64,
        128 => W128,
        _ => W32,
    }
}

/// The integer `x` as an operand of type `t`.
fn int(t: Ty, x: i64, name: &str) -> Result<String, String> {
    match t {
        Ty::Pred => Err(format!("`{name}` wants a predicate, which is a variable `setp` sets")),
        Ty::F32 if (x as f32) as i128 == i128::from(x) => Ok(format!("0f{:08X}", (x as f32).to_bits())),
        Ty::F64 if (x as f64) as i128 == i128::from(x) => Ok(format!("0d{:016X}", (x as f64).to_bits())),
        Ty::F32 | Ty::F64 => Err(format!("`{name}` wants an `{}`; {x} is not one", t.name())),
        _ if t.kind() == Kind::Float => Err(half(t, name)),
        Ty::B128 => Err(format!("`{name}` takes a `b128` only from a variable")),
        _ => {
            let bits = t.bits();
            if bits < 64 && !(-(1i64 << (bits - 1))..(1i64 << bits)).contains(&x) {
                return Err(format!("`{name}` wants {bits} bits; {x} does not fit them"));
            }
            Ok(x.to_string())
        }
    }
}

/// Why `x` is no fraction for `name`.
fn fraction(name: &str, x: impl std::fmt::Display) -> String {
    format!("`{name}` takes a fraction here, more than 0 and at most 1; {x} is not one")
}

/// Why `name` takes no value known now as its operand of type `t`, a float PTX keeps in bits.
fn half(t: Ty, name: &str) -> String {
    let fill = match t {
        Ty::F16 => "mov.b16(h, 1.5f16)",
        Ty::Bf16 => "mov.b16(h, 1.5bf16)",
        Ty::F16x2 => "mov.b32(h, (1.5f16, 2.0f16))",
        Ty::Bf16x2 => "mov.b32(h, (1.5bf16, 2.0bf16))",
        _ => "mov.b32(t, 1.5)",
    };
    format!("`{name}` takes a `{}` only from a variable: `{fill}` puts one in it", t.name())
}

/// The integer `x` as an operand that must be known now, of what `imm` lets it be.
fn known(imm: Imm, x: i64, name: &str) -> Result<String, String> {
    let fits = match imm {
        Imm::Bits(bits) => (0..1i64 << bits).contains(&x),
        Imm::Exactly(n) => x == n,
        Imm::Pow2(p) => x > 0 && x & (x - 1) == 0 && x <= 1 << p,
    };
    if fits {
        return Ok(x.to_string());
    }
    Err(match imm {
        Imm::Bits(bits) => {
            format!("`{name}` takes {bits} bits here, 0 to {}; {x} is not one of them", (1i64 << bits) - 1)
        }
        Imm::Exactly(n) => format!("`{name}` takes {n} here, the one value PTX has for it; not {x}"),
        Imm::Pow2(p) => format!("`{name}` takes a power of two here, 2^{p} at most; {x} is not one"),
    })
}

/// What `spec` of `b` wants, for an error.
fn describe(b: &Bound, spec: &Spec) -> String {
    let t = spec.ty.map(|t| b.ty(t).name()).unwrap_or_default();
    let one = match spec.role {
        Role::Addr(_) => return "an address, as in `[p + 4]`".into(),
        Role::Label => return "a label".into(),
        Role::Imm(_) => return "an integer known now".into(),
        Role::Dst if spec.sink => format!("a `{t}` variable to write, or `_`"),
        Role::Dst => format!("a `{t}` variable to write"),
        Role::Src if spec.reg_only => format!("a `{t}` variable"),
        Role::Src if spec.sreg => format!("a `{t}` variable, a value or a special register"),
        Role::Src => format!("a `{t}` variable or a value"),
    };
    match lanes(b, spec) {
        1 => one,
        n => format!("{n} operands taken together, each {one}"),
    }
}

fn given(a: &Given<Var>) -> String {
    match a {
        Given::Var(v) => format!("a `{}` variable", v.ty.name()),
        Given::Special(name) if name == "_" => "`_`".into(),
        Given::Special(name) => format!("`{name}`"),
        Given::Int(x) => format!("the integer {x}"),
        Given::F32(x) => format!("the float {x}"),
        Given::F64(x) => format!("the f64 {x}"),
        Given::F16(x) => format!("the f16 {}", kgen_lang::Value::F16(*x)),
        Given::Bf16(x) => format!("the bf16 {}", kgen_lang::Value::Bf16(*x)),
        Given::Mem(_) => "an address".into(),
        Given::Data(_) => "memory holding data".into(),
        Given::Label(_) => "a label".into(),
        Given::Vector(vs) => format!("{} operands taken together", vs.len()),
        Given::Pair(..) => "two operands joined by `|`".into(),
        Given::Not(_) => "an operand inverted".into(),
        Given::Neg(_) => "an operand negated".into(),
        Given::List(_) => "operands in brackets".into(),
    }
}

/// How many operands the forms take, `1 or 2 operands`, from each one's fewest and most.
fn counts(ranges: &[(usize, usize)]) -> String {
    let all: BTreeSet<usize> = ranges.iter().flat_map(|&(a, b)| a..=b).collect();
    let words: Vec<String> = all.iter().map(usize::to_string).collect();
    match words.as_slice() {
        [one] if one == "0" => "no operands".into(),
        [one] if one == "1" => "1 operand".into(),
        [one] => format!("{one} operands"),
        [first @ .., last] => format!("{} or {last} operands", first.join(", ")),
        [] => unreachable!("an instruction has forms"),
    }
}

/// The types a variable may be, for an error about one that is none.
const TYPES: &str = "`pred`, `b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, `bf16`, `bf16x2`, \
                     `tf32`, `f32` and `f64`";

impl Target for Kernel {
    type Var = Var;

    fn param(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        let pointer = ty.starts_with("*const ") || ty.starts_with("*mut ");
        let t = match ty {
            _ if pointer => Ty::U64,
            "u8" => Ty::U8,
            "i8" => Ty::S8,
            "u16" => Ty::U16,
            "i16" => Ty::S16,
            "u32" => Ty::U32,
            "i32" => Ty::S32,
            "u64" | "usize" => Ty::U64,
            "i64" | "isize" => Ty::S64,
            "f16" => Ty::F16,
            "bf16" => Ty::Bf16,
            "f32" => Ty::F32,
            "f64" => Ty::F64,
            _ => {
                return Err(format!(
                    "a GPU kernel takes pointers, integers and the floats `f16`, `bf16`, `f32` and `f64`, not `{ty}`"
                ));
            }
        };
        let v = self.new_var(name, t);
        self.p.params.push((v.reg, PReg { class: v.reg.class, hw: 0 }));
        self.p.signature.params.push(v.reg.class);
        self.params.push(Param { ty: t, reg: v.reg.id as usize, pointer });
        Ok(v)
    }

    fn result(&mut self, ty: &str) -> Result<(), String> {
        match ty {
            "()" => Ok(()),
            _ => Err(format!("a GPU kernel returns nothing, not `{ty}`: it writes what it works out to memory")),
        }
    }

    fn var(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        let Some(t) = Ty::parse(ty) else {
            return Err(format!("PTX variables are {TYPES}, not `{ty}`"));
        };
        if t == Ty::B128 {
            self.require(Needs { ptx: 83, sm: 70, below: 0 }, &format!("let {name}: b128"))?;
        }
        Ok(self.new_var(name, t))
    }

    fn view(&mut self, v: Var, view: &str) -> Result<Var, String> {
        match Ty::parse(view) {
            Some(t) if v.ty.fits(t) => Ok(v),
            Some(t) => Err(format!("a `{}` variable is no `{}`", v.ty.name(), t.name())),
            None => Err(format!("PTX variables are {TYPES}, not `{view}`")),
        }
    }

    fn label(&mut self, label: u32) {
        for w in &mut self.partial {
            w.side = None;
        }
        self.p.body.push(Item::Label(Label(label)));
    }

    fn note(&mut self, text: &str) {
        self.p.body.push(Item::Note(text.to_string()));
    }

    fn inst(&mut self, name: &str, args: &[Given<Var>], guard: Option<Guard<Var>>) -> Result<(), Fault> {
        if let Some(g) = guard
            && g.pred.ty != Ty::Pred
        {
            return Err(Fault::new(format!("an instruction is guarded by a `pred`; this is a `{}`", g.pred.ty.name())));
        }
        let bound = forms::find(name).map_err(Fault::new)?;
        let (mut ranges, mut best, mut found) = (vec![], None::<(usize, bool, Fault)>, None);
        for b in &bound {
            match self.operands(b, name, args) {
                Ok(w) => {
                    found = Some((b, w));
                    break;
                }
                Err(Miss::Count(least, most)) => ranges.push((least, most)),
                Err(Miss::At(at, shape, fault)) => {
                    if best.as_ref().is_none_or(|(a, s, _)| (at, shape) > (*a, *s)) {
                        best = Some((at, shape, fault));
                    }
                }
            }
        }
        let Some((b, w)) = found else {
            return Err(match best {
                Some((.., fault)) => fault,
                None => Fault::new(format!("`{name}` takes {}, not {}", counts(&ranges), args.len())),
            });
        };
        let mut what = name.to_string();
        for s in &w.sregs {
            what += &format!(" %{s}");
        }
        self.require(b.needs.and(w.needs), &what).map_err(Fault::new)?;
        let label = args.iter().find_map(|a| match a {
            Given::Label(l) => Some(Label(*l)),
            _ => None,
        });
        let flow = match (b.form.op, label, guard) {
            ("bra", Some(l), None) => Flow::Jump(l),
            ("bra", Some(l), Some(_)) => Flow::Branch(l),
            ("ret" | "exit", _, None) => Flow::Return,
            _ => Flow::Next,
        };
        let mut line = match guard {
            Some(g) => format!("@{}%{} ", if g.negated { "!" } else { "" }, self.reg(g.pred)),
            None => String::new(),
        };
        line += name;
        if !w.texts.is_empty() {
            line += " ";
            line += &w.texts.join(", ");
        }
        line += ";";
        self.push(line, w.args, guard, flow);
        Ok(())
    }

    fn end(&mut self) -> Result<(), String> {
        let open = match self.p.body.iter().rev().find(|item| !matches!(item, Item::Note(_))) {
            Some(Item::Inst(inst)) => !matches!(inst.flow, Flow::Return | Flow::Jump(_)),
            _ => true,
        };
        if open {
            self.push("ret;".into(), vec![], None, Flow::Return);
        }
        Ok(())
    }
}
