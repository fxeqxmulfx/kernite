//! PTX as the language writes it: [`Kernel`] is the [`Target`] that checks each call against
//! the types its name gives its operands, and writes the instruction out.
//!
//! The types are PTX's: `pred`, `b32`, `u32`, `s32`, `f32`, `b64`, `u64` and `s64`. An
//! instruction is called by its PTX name, words joined with dots, with the operands PTX
//! writes: `add.rn.f32(acc, acc, x)`, `setp.lt.u32(p, i, n)`, `ld.global.f32(x, [p + 4])`.
//! Its operands' type is the type its name ends in; `cvt` names two, the result's first, and
//! `mul.wide` and `mad.wide` write twice the bits they read.
//!
//! - A float `add`, `sub`, `mul`, `mad`, `fma`, `div`, `sqrt` or `rcp` says how it rounds,
//!   `.rn`, so that ptxas neither fuses nor approximates it: what it computes is what it says.
//! - An address is a 64-bit variable and a displacement, `[p + 16]`. A kernel's pointers
//!   arrive as global addresses, for `ld.global` and `st.global`.
//! - `tid.x`, `ntid.x`, `ctaid.x` and `nctaid.x`, and their `y` and `z`, are read by `mov`.
//! - `if p { .. }` runs each instruction inside where `p`, a `pred`, holds, and `if !p` where
//!   it does not; a guarded `bra(label)` is a branch.
//! - A kernel returns nothing: `ret()`, which the end of the body adds if control reaches it.

use std::collections::BTreeSet;

use kgen_core::Width::{W32, W64};
use kgen_core::{
    Access, Arg, Base, Class, Flow, Inst, Item, Label, Mem, Operand, PReg, Program, RegInfo, Signature, VReg,
};
use kgen_lang::{Fault, Guard, Operand as Given, Target};

use crate::{Entry, Op, Param, Ty};

/// A variable: a register of the type it was declared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var {
    reg: VReg,
    ty: Ty,
}

/// What an operand of an instruction may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A variable it writes, of a type that fits.
    Dst(Ty),
    /// A variable it reads, of a type that fits, or a value known now.
    Src(Ty),
    /// That, or a special register: `tid.x`.
    Sreg(Ty),
    /// An address it reads or writes at.
    Addr(Access),
    /// A label it jumps to.
    Goto,
    /// An integer known now.
    Int,
}

use Kind::{Addr, Dst, Goto, Int, Src, Sreg};

/// The special registers `mov` reads.
const SPECIAL: &[&str] = &[
    "tid.x", "tid.y", "tid.z", "ntid.x", "ntid.y", "ntid.z", "ctaid.x", "ctaid.y", "ctaid.z", "nctaid.x", "nctaid.y",
    "nctaid.z", "laneid", "warpid",
];

/// Names PTX gives special registers, which a variable's register does not take.
const RESERVED: &[&str] = &[
    "tid",
    "ntid",
    "ctaid",
    "nctaid",
    "laneid",
    "warpid",
    "nwarpid",
    "smid",
    "nsmid",
    "gridid",
    "clock",
    "clock64",
    "clock_hi",
    "globaltimer",
    "globaltimer_lo",
    "globaltimer_hi",
    "total_smem_size",
    "dynamic_smem_size",
    "aggr_smem_size",
    "current_graph_exec",
    "is_explicit_cluster",
    "clusterid",
    "nclusterid",
    "cluster_ctaid",
    "cluster_nctaid",
    "cluster_ctarank",
    "cluster_nctarank",
];

/// The comparisons of `setp`.
const COMPARISONS: &[&str] = &[
    "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan",
];

/// The float instructions that must say how they round.
const ROUNDED: &[&str] = &["add", "sub", "mul", "mad", "fma", "div", "sqrt", "rcp"];

/// A kernel being written in the language, for PTX.
pub struct Kernel {
    p: Program<Op>,
    params: Vec<Param>,
    regs: Vec<(String, Ty)>,
    taken: BTreeSet<String>,
    partial: Vec<Partial>,
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
        }
    }

    /// The kernel as written.
    pub fn finish(self) -> Entry {
        Entry { program: self.p, params: self.params, regs: self.regs }
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
        let base = if RESERVED.contains(&name) { format!("{name}_") } else { name.to_string() };
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

    /// `a` as the operand `kind`, written out, and what it reads or writes.
    fn operand(&self, kind: Kind, a: &Given<Var>) -> Result<(String, Option<Arg<VReg>>), String> {
        let var = |v: &Var, access: Access| {
            let width = if v.ty.bits() == 64 { W64 } else { W32 };
            (format!("%{}", self.reg(*v)), Some(Arg { operand: Operand::Reg(v.reg, width), access }))
        };
        match (kind, a) {
            (Dst(t), Given::Var(v)) if v.ty.fits(t) => Ok(var(v, Access::Write)),
            (Src(t) | Sreg(t), Given::Var(v)) if v.ty.fits(t) => Ok(var(v, Access::Read)),
            (Dst(t) | Src(t) | Sreg(t), Given::Var(v)) => {
                Err(format!("wants a `{}` variable; this is a `{}`", t.name(), v.ty.name()))
            }
            (Src(t) | Sreg(t), Given::Int(x)) => Ok((int(t, *x)?, None)),
            (Src(Ty::F32 | Ty::B32) | Sreg(Ty::F32 | Ty::B32), Given::F32(x)) => Ok((float(*x), None)),
            (Sreg(_), Given::Special(s)) if SPECIAL.contains(&s.as_str()) => Ok((format!("%{s}"), None)),
            (_, Given::Data(_)) => Err("takes a value as it is: `1.5`, not `[1.5]`".into()),
            (Addr(access), Given::Mem(m)) => self.address(m, access),
            (Goto, Given::Label(l)) => {
                Ok((format!("$L{l}"), Some(Arg { operand: Operand::Label(Label(*l)), access: Access::Read })))
            }
            (Int, Given::Int(x)) => Ok((x.to_string(), None)),
            (kind, a) => Err(format!("wants {}; this is {}", describe(kind), given(a))),
        }
    }

    fn address(&self, m: &kgen_lang::Mem<Var>, access: Access) -> Result<(String, Option<Arg<VReg>>), String> {
        if m.index.is_some() {
            return Err("takes an address of a variable and a displacement; work out `p + 4 * i` first, \
                        as `mad.wide.u32(a, i, 4, p)` does"
                .into());
        }
        let Some(base) = m.base else {
            return Err("takes an address that starts at a 64-bit variable".into());
        };
        if base.ty.bits() != 64 {
            return Err(format!("takes an address in a 64-bit variable; this is a `{}`", base.ty.name()));
        }
        let disp = i32::try_from(m.disp).map_err(|_| format!("{} bytes are too far for an address", m.disp))?;
        let text = match disp {
            0 => format!("[%{}]", self.reg(base)),
            d => format!("[%{}+{d}]", self.reg(base)),
        };
        let mem = Mem { base: Base::Reg(base.reg), index: None, disp, width: W32 };
        Ok((text, Some(Arg { operand: Operand::Mem(mem), access })))
    }
}

/// The operands of the instruction `name`, from the words of its name: the modifiers PTX has
/// for it, and the types.
fn operands(name: &str) -> Result<Vec<Kind>, String> {
    let mut words = name.split('.');
    let op = words.next().unwrap_or_default();
    let (mut types, mut mods) = (vec![], vec![]);
    for w in words {
        match Ty::parse(w) {
            Some(t) => types.push(t),
            None => mods.push(w),
        }
    }
    let allowed: &[&str] = match op {
        "mov" | "and" | "or" | "xor" | "not" | "cnot" | "shl" | "shr" | "selp" | "rem" => &[],
        "add" | "sub" | "fma" => &["rn", "rz", "rm", "rp", "ftz", "sat"],
        "mul" | "mad" => &["lo", "hi", "wide", "rn", "rz", "rm", "rp", "ftz", "sat"],
        "div" => &["rn", "rz", "rm", "rp", "approx", "full", "ftz"],
        "sqrt" | "rcp" => &["rn", "rz", "rm", "rp", "approx", "ftz"],
        "min" | "max" | "neg" | "abs" => &["ftz"],
        "setp" => &[
            "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu", "ltu", "leu", "gtu", "geu",
            "num", "nan", "ftz",
        ],
        "cvt" => &["rn", "rz", "rm", "rp", "rni", "rzi", "rmi", "rpi", "ftz", "sat"],
        "ld" => &["global", "shared", "local", "const", "volatile"],
        "st" => &["global", "shared", "local", "volatile"],
        "atom" => &["global", "shared", "add", "min", "max", "inc", "dec", "and", "or", "xor", "exch", "cas"],
        "red" => &["global", "shared", "add", "min", "max", "inc", "dec", "and", "or", "xor"],
        "bra" | "ret" | "exit" => &["uni"],
        "bar" => &["sync"],
        _ => return Err(format!("PTX has no instruction `{op}` that kgen knows")),
    };
    if let Some(m) = mods.iter().find(|m| !allowed.contains(m)) {
        return Err(format!("PTX's `{op}` has no `.{m}` that kgen knows"));
    }
    let has = |m: &str| mods.contains(&m);
    match op {
        "bra" | "ret" | "exit" | "bar" if !types.is_empty() => return Err(format!("`{op}` takes no type")),
        "bra" => return Ok(vec![Goto]),
        "ret" | "exit" => return Ok(vec![]),
        "bar" if has("sync") => return Ok(vec![Int]),
        "bar" => return Err("kgen knows `bar.sync`".into()),
        "cvt" => {
            return match types[..] {
                [to, from] => Ok(vec![Dst(to), Src(from)]),
                _ => Err(format!("`{name}` names two types, the result's and then the source's: `cvt.rn.f32.s32`")),
            };
        }
        _ => {}
    }
    let [t] = types[..] else {
        return Err(format!("`{name}` ends in its operands' type, one of b32, u32, s32, f32, b64, u64, s64 and pred"));
    };
    if t == Ty::F32
        && ROUNDED.contains(&op)
        && !mods.iter().any(|m| ["rn", "rz", "rm", "rp", "approx", "full"].contains(m))
    {
        return Err(format!("`{name}` leaves ptxas to choose how it rounds, and to fuse it; say so: `{op}.rn.f32`"));
    }
    let int = !matches!(t, Ty::F32 | Ty::Pred);
    if int && ["mul", "mad"].contains(&op) && !(has("lo") || has("hi") || has("wide")) {
        return Err(format!(
            "`{name}` keeps the low half of the product, the high or both: `{op}.lo`, `{op}.hi`, `{op}.wide`"
        ));
    }
    if ["ld", "st", "atom", "red"].contains(&op) && !["global", "shared", "local", "const"].iter().any(|s| has(s)) {
        return Err(format!("`{name}` says where it reads or writes: `{op}.global`"));
    }
    if op == "setp" && mods.iter().filter(|m| COMPARISONS.contains(m)).count() != 1 {
        return Err(format!("`{name}` names one comparison, as in `setp.lt.u32`"));
    }
    let wide = |t: Ty| match t {
        Ty::U32 => Ok(Ty::U64),
        Ty::S32 => Ok(Ty::S64),
        _ => Err(format!("`{name}` widens a 32-bit integer: `{op}.wide.u32`")),
    };
    Ok(match op {
        "mov" => vec![Dst(t), Sreg(t)],
        "setp" => vec![Dst(Ty::Pred), Src(t), Src(t)],
        "selp" => vec![Dst(t), Src(t), Src(t), Src(Ty::Pred)],
        "ld" => vec![Dst(t), Addr(Access::Read)],
        "st" => vec![Addr(Access::Write), Src(t)],
        "atom" if has("cas") => vec![Dst(t), Addr(Access::ReadWrite), Src(t), Src(t)],
        "atom" => vec![Dst(t), Addr(Access::ReadWrite), Src(t)],
        "red" => vec![Addr(Access::ReadWrite), Src(t)],
        "shl" | "shr" => vec![Dst(t), Src(t), Src(Ty::U32)],
        "mul" if has("wide") => vec![Dst(wide(t)?), Src(t), Src(t)],
        "mad" if has("wide") => vec![Dst(wide(t)?), Src(t), Src(t), Src(wide(t)?)],
        "mad" | "fma" => vec![Dst(t), Src(t), Src(t), Src(t)],
        "not" | "cnot" | "neg" | "abs" | "sqrt" | "rcp" => vec![Dst(t), Src(t)],
        _ => vec![Dst(t), Src(t), Src(t)],
    })
}

/// The integer `x` as an operand of type `t`.
fn int(t: Ty, x: i64) -> Result<String, String> {
    match t {
        Ty::F32 if (x as f32) as i64 == x => Ok(float(x as f32)),
        Ty::F32 => Err(format!("wants an `f32`; {x} is not one")),
        Ty::Pred => Err("wants a predicate, which is a variable `setp` sets".into()),
        _ if t.bits() == 32 && !(i64::from(i32::MIN)..=i64::from(u32::MAX)).contains(&x) => {
            Err(format!("wants 32 bits; {x} does not fit them"))
        }
        _ => Ok(x.to_string()),
    }
}

/// A float as PTX writes it exactly: its bits.
fn float(x: f32) -> String {
    format!("0f{:08X}", x.to_bits())
}

fn describe(kind: Kind) -> String {
    match kind {
        Dst(t) => format!("a `{}` variable to write", t.name()),
        Src(t) => format!("a `{}` variable or a value", t.name()),
        Sreg(t) => format!("a `{}` variable, a value or a special register", t.name()),
        Addr(_) => "an address, as in `[p + 4]`".into(),
        Goto => "a label".into(),
        Int => "an integer".into(),
    }
}

fn given(a: &Given<Var>) -> String {
    match a {
        Given::Var(v) => format!("a `{}` variable", v.ty.name()),
        Given::Special(name) => format!("`{name}`"),
        Given::Int(x) => format!("the integer {x}"),
        Given::F32(x) => format!("the float {x}"),
        Given::F64(x) => format!("the f64 {x}"),
        Given::Mem(_) => "an address".into(),
        Given::Data(_) => "memory holding data".into(),
        Given::Label(_) => "a label".into(),
    }
}

impl Target for Kernel {
    type Var = Var;

    fn param(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        let pointer = ty.starts_with("*const ") || ty.starts_with("*mut ");
        let t = match ty {
            _ if pointer => Ty::U64,
            "u32" => Ty::U32,
            "i32" => Ty::S32,
            "u64" | "usize" => Ty::U64,
            "i64" | "isize" => Ty::S64,
            "f32" => Ty::F32,
            _ => return Err(format!("a GPU kernel takes pointers, integers of 32 and 64 bits and `f32`, not `{ty}`")),
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
        match Ty::parse(ty) {
            Some(t) => Ok(self.new_var(name, t)),
            None => {
                Err(format!("PTX variables are `pred`, `b32`, `u32`, `s32`, `f32`, `b64`, `u64` and `s64`, not `{ty}`"))
            }
        }
    }

    fn view(&mut self, v: Var, view: &str) -> Result<Var, String> {
        match Ty::parse(view) {
            Some(t) if v.ty.fits(t) => Ok(v),
            Some(t) => Err(format!("a `{}` variable is no `{}`", v.ty.name(), t.name())),
            None => Err(format!(
                "PTX variables are `pred`, `b32`, `u32`, `s32`, `f32`, `b64`, `u64` and `s64`, not `{view}`"
            )),
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
        let kinds = operands(name).map_err(Fault::new)?;
        if kinds.len() != args.len() {
            return Err(Fault::new(format!("`{name}` takes {} operands, not {}", kinds.len(), args.len())));
        }
        let (mut texts, mut built) = (vec![], vec![]);
        for (i, (&kind, a)) in kinds.iter().zip(args).enumerate() {
            if let Given::Special(s) = a {
                let known = SPECIAL.contains(&s.as_str());
                if !known || !matches!(kind, Sreg(_)) {
                    let m = if known {
                        format!("`{name}` reads no special register; `mov.u32` does")
                    } else {
                        format!("`{s}` is not defined")
                    };
                    return Err(Fault::at(i, m));
                }
            }
            let (text, arg) = self.operand(kind, a).map_err(|m| Fault::at(i, format!("`{name}` {m}")))?;
            texts.push(text);
            built.extend(arg);
        }
        let op = name.split('.').next().unwrap_or_default();
        let label = args.iter().find_map(|a| match a {
            Given::Label(l) => Some(Label(*l)),
            _ => None,
        });
        let flow = match (op, label, guard) {
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
        if !texts.is_empty() {
            line += " ";
            line += &texts.join(", ");
        }
        line += ";";
        self.push(line, built, guard, flow);
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
