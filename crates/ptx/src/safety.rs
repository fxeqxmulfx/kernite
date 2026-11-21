//! What proves a kernel with no `unsafe` safe, as its instructions are written: each load and store lies inside its
//! memory at its alignment, no two threads write one address, and none reads what another writes.
//!
//! The proof follows the kernel straight through, for a kernel has no jump: it knows each variable as a whole number
//! of the thread's and the launch's numbers, or as an address `offset` bytes from a pointer parameter. What it cannot
//! prove at once, because it is about a number the launch gives, it leaves as a condition on the launch, which the host
//! checks before the kernel runs: that the memory has as many elements as the kernel reaches, and that no number wraps.
//! What no launch can mend, or the kernel's instructions are none that it follows, is an error at the instruction.

use std::collections::BTreeMap;

use kernite_build::{Quantity, Requirement, Why};
use kernite_core::VReg;
use kernite_lang::{Fault, Operand as Given};

use crate::forms::{Bound, Space};
use crate::ty::Ty;
use crate::var::Var;

/// The instructions that work on registers and nothing else, which no thread can use against another.
const PURE: &str = "add sub mul mad clmad mul24 mad24 sad div rem abs neg min max popc clz bfind fns brev bfe bfi \
                    szext bmsk dp4a dp2a addc subc madc testp copysign fma rcp sqrt rsqrt sin cos lg2 ex2 tanh setp \
                    set selp slct and or xor not cnot lop3 shf shl shr vadd vsub vabsdiff vmin vmax vshl vshr vmad \
                    vset vadd2 vsub2 vavrg2 vabsdiff2 vmin2 vmax2 vset2 vadd4 vsub4 vavrg4 vabsdiff4 vmin4 vmax4 \
                    vset4 mov prmt isspacep cvt brkpt nanosleep pmevent";

/// What the instructions that are neither [`PURE`] nor a plain load or store, nor the end of a thread, do, which
/// kernite cannot prove safe; `call` and `grid` are the language's, and no instruction of the table.
const REFUSED: &[(&str, &str)] = &[
    ("bar barrier membar fence mbarrier grid", "makes threads wait for each other, or orders what they write"),
    (
        "shfl vote match activemask redux ldmatrix movmatrix mma wmma",
        "works on the registers of the threads of a warp together",
    ),
    ("atom red cp sured", "reaches memory that other threads reach at once"),
    ("tex tld4 txq istypep suld sust suq", "reaches a texture or a surface"),
    ("bra brx", "jumps"),
    ("call", "calls a function"),
    (
        "cvta ldu prefetch prefetchu applypriority discard createpolicy stacksave stackrestore alloca",
        "reaches memory in a way kernite does not follow",
    ),
];

/// The instructions that end a thread: with a guard, the threads that go on know it did not hold.
const ENDS: [&str; 3] = ["ret", "exit", "trap"];

/// Why the proof does not follow the instruction `op`, `None` for one it does.
fn why_not(op: &str) -> Option<&'static str> {
    let has = |list: &str| list.split_whitespace().any(|o| o == op);
    if matches!(op, "ld" | "st") || ENDS.contains(&op) || has(PURE) {
        return None;
    }
    Some(
        REFUSED.iter().find(|(ops, _)| has(ops)).map_or("is none of the instructions kernite follows", |&(_, why)| why),
    )
}

/// The error for `name`, which does `what`, where it cannot be proved safe: a kernel that does is an `unsafe fn`.
fn cannot(name: &str, what: &str) -> Fault {
    Fault::new(format!(
        "`{name}` {what}, and kernite cannot prove that this is safe, so a kernel that does is an `unsafe fn`"
    ))
}

/// The error for `name`, an instruction of `op` that the proof does not follow.
pub(crate) fn refused(name: &str, op: &str) -> Fault {
    cannot(name, why_not(op).unwrap_or("is none of the instructions kernite follows"))
}

/// Which memory, by its place among the kernel's, `a` names, if it names one: shared and local memory of the kernel's
/// and the module's statics are named by a variable.
pub(crate) fn memory(a: &Given<Var>) -> Option<usize> {
    let of = |v: &Var| match v {
        Var::Mem(i) => Some(*i),
        _ => None,
    };
    match a {
        Given::Var(v) => of(v),
        Given::Mem(m) => m.base.as_ref().and_then(of).or_else(|| m.index.as_ref().and_then(|(v, _)| of(v))),
        Given::Sized(_, a) | Given::Not(a) | Given::Neg(a) => memory(a),
        Given::Pair(a, b) => memory(a).or_else(|| memory(b)),
        Given::Vector(v) | Given::List(v) => v.iter().find_map(memory),
        _ => None,
    }
}

/// A number the proof reasons about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Atom {
    /// The thread's index in its block, `tid.x`: from 0 to one less than the block's threads.
    Thread,
    /// The block's index in its grid, `ctaid.x`: from 0 to one less than the grid's blocks.
    Block,
    /// A number the launch gives, not less than 0, and not less than 1 for a count of threads or of blocks.
    Launch(Quantity),
}

/// Whether a product of atoms is not the same for each thread.
fn varies(atoms: &[Atom]) -> bool {
    atoms.iter().any(|a| matches!(a, Atom::Thread | Atom::Block))
}

/// A whole number, as a sum of products of atoms, each with its factor. No sum has a factor of 0.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Poly(BTreeMap<Vec<Atom>, i128>);

impl Poly {
    fn int(x: i128) -> Poly {
        Poly(BTreeMap::from([(vec![], x)]).into_iter().filter(|&(_, c)| c != 0).collect())
    }

    fn atom(a: Atom) -> Poly {
        Poly(BTreeMap::from([(vec![a], 1)]))
    }

    fn launch(q: Quantity) -> Poly {
        Poly::atom(Atom::Launch(q))
    }

    /// `None` where a factor is past `i128`.
    fn add(&self, other: &Poly) -> Option<Poly> {
        let mut sum = self.0.clone();
        for (k, c) in &other.0 {
            let e = sum.entry(k.clone()).or_insert(0);
            *e = e.checked_add(*c)?;
        }
        sum.retain(|_, c| *c != 0);
        Some(Poly(sum))
    }

    fn scale(&self, by: i128) -> Option<Poly> {
        let mut out = BTreeMap::new();
        for (k, c) in &self.0 {
            out.insert(k.clone(), c.checked_mul(by)?);
        }
        out.retain(|_, c| *c != 0);
        Some(Poly(out))
    }

    fn sub(&self, other: &Poly) -> Option<Poly> {
        self.add(&other.scale(-1)?)
    }

    fn mul(&self, other: &Poly) -> Option<Poly> {
        let mut out = Poly::default();
        for (a, x) in &self.0 {
            for (b, y) in &other.0 {
                let mut k: Vec<Atom> = a.iter().chain(b).copied().collect();
                k.sort();
                out = out.add(&Poly(BTreeMap::from([(k, x.checked_mul(*y)?)])))?;
            }
        }
        Some(out)
    }

    /// Whether it is not the same for each thread.
    fn varies(&self) -> bool {
        self.0.keys().any(|k| varies(k))
    }

    /// The number it is, if it holds no atom.
    fn constant(&self) -> Option<i128> {
        match self.0.iter().next() {
            None => Some(0),
            Some((k, c)) if k.is_empty() && self.0.len() == 1 => Some(*c),
            _ => None,
        }
    }

    /// What it is but for its constant, and the constant.
    fn split(&self) -> (Poly, i128) {
        let mut rest = self.0.clone();
        let c = rest.remove(&vec![]).unwrap_or(0);
        (Poly(rest), c)
    }

    /// The most, or the least, it is over every thread of a launch, which is a sum of its own: each product of the
    /// thread's or the block's index, with a factor above 0, takes the index's greatest value, and one below, its
    /// least, 0, or the other way about; the launch's numbers are not less than 0.
    fn bound(&self, high: bool) -> Option<Poly> {
        let mut out = Poly::default();
        for (atoms, &c) in &self.0 {
            let term = if !varies(atoms) {
                Poly(BTreeMap::from([(atoms.clone(), c)]))
            } else if (c > 0) != high {
                continue;
            } else {
                let mut t = Poly::int(c);
                for a in atoms {
                    let last = |q| Poly::launch(q).sub(&Poly::int(1));
                    t = t.mul(&match a {
                        Atom::Thread => last(Quantity::Threads)?,
                        Atom::Block => last(Quantity::Blocks)?,
                        a => Poly::atom(*a),
                    })?;
                }
                t
            };
            out = out.add(&term)?;
        }
        Some(out)
    }
}

/// How two numbers compare, as `setp` says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// What the proof knows of a variable.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Val {
    /// A whole number, exactly: no sum of it wraps, as the conditions on the launch have it.
    Num(Poly),
    /// An address `offset` bytes from where the pointer parameter `pointer` points.
    Addr { pointer: usize, offset: Poly },
    /// What `setp` compared: `left` and `right` are whole numbers.
    Cmp { left: Poly, cmp: Cmp, right: Poly },
}

impl Val {
    /// A number not less than 0 that holds where `left cmp right` holds, if `holds`, and where it does not, if not.
    fn fact(&self, holds: bool) -> Option<Poly> {
        let Val::Cmp { left, cmp, right } = self else { return None };
        let (up, down) = (right.sub(left)?, left.sub(right)?);
        let one = Poly::int(1);
        match (cmp, holds) {
            (Cmp::Lt, true) | (Cmp::Ge, false) => up.sub(&one),
            (Cmp::Le, true) | (Cmp::Gt, false) => Some(up),
            (Cmp::Gt, true) | (Cmp::Le, false) => down.sub(&one),
            (Cmp::Ge, true) | (Cmp::Lt, false) => Some(down),
            (Cmp::Eq | Cmp::Ne, _) => None,
        }
    }
}

/// The part of a pointer's memory that a thread has for itself, as each access to it says: its offset is `shift`, the
/// same for every thread, and `block` times the thread's block, and `thread` times its index in its block, and its
/// bytes are in the `thread` bytes from there, if there are any.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Frame {
    shift: Poly,
    block: Poly,
    thread: i128,
}

/// What the proof knows of a pointer parameter.
#[derive(Debug)]
struct Pointer {
    /// The bytes of what it points to, and so how it is aligned.
    bytes: i128,
    writable: bool,
    /// The part of its memory a thread has for itself, once an access says it.
    frame: Option<Frame>,
}

/// The bytes of the type `ty`, a type of a pointer's target: 0 for one kernite does not know the size of.
fn size(ty: &str) -> i128 {
    match ty {
        "u8" | "i8" => 1,
        "u16" | "i16" | "f16" | "bf16" => 2,
        "u32" | "i32" | "f32" => 4,
        "u64" | "i64" | "f64" | "usize" | "isize" => 8,
        _ => 0,
    }
}

/// One instruction the proof reads: its name, what the language bound it to, its operands and the guard it runs under,
/// the registers it writes, and the bytes it reaches at an address.
pub(crate) struct Step<'a> {
    pub name: &'a str,
    pub bound: &'a Bound,
    pub args: &'a [Given<Var>],
    pub guard: Option<(VReg, bool)>,
    pub written: Vec<u32>,
    pub reach: u64,
}

/// What a kernel with no `unsafe` is proved to do so far.
#[derive(Debug)]
pub(crate) struct Proof {
    /// What each variable holds, by its place among the kernel's, where the proof follows it.
    vals: BTreeMap<u32, Val>,
    /// Numbers that are not less than 0 where the kernel is, from the compares that end the threads that fail them.
    facts: Vec<Poly>,
    /// Each pointer parameter, by its place among the parameters.
    pointers: BTreeMap<usize, Pointer>,
    /// What each launch must hold: for each sum that is not less than 0, but for its constant, the least constant.
    needs: BTreeMap<Poly, (i128, Why)>,
}

/// What every launch of a kernel with no `unsafe` holds, whether the kernel says it or not: its grid has fewer than
/// 2^32 threads, and so the index `ctaid.x * ntid.x + tid.x` numbers each apart in 32 bits.
fn grid() -> (Poly, i128) {
    let threads = Poly::launch(Quantity::Threads);
    (threads.mul(&Poly::launch(Quantity::Blocks)).and_then(|p| p.scale(-1)).unwrap_or_default(), (1 << 32) - 1)
}

impl Proof {
    /// The proof of a kernel that has no instruction yet.
    pub(crate) fn new() -> Proof {
        let (grid, most) = grid();
        let needs = BTreeMap::from([(grid, (most, Why::Fits))]);
        Proof { vals: BTreeMap::new(), facts: vec![], pointers: BTreeMap::new(), needs }
    }

    /// The parameter number `index` of the kernel, of type `ty` as written, which is in register `reg`, of type `t`.
    pub(crate) fn param(&mut self, reg: VReg, index: usize, ty: &str, t: Ty) {
        let pointer =
            ty.strip_prefix("*const ").map(|to| (to, false)).or_else(|| ty.strip_prefix("*mut ").map(|to| (to, true)));
        if let Some((to, writable)) = pointer {
            let bytes = size(to);
            if bytes > 0 {
                self.pointers.insert(index, Pointer { bytes, writable, frame: None });
                self.vals.insert(reg.id, Val::Addr { pointer: index, offset: Poly::default() });
            }
        } else if matches!(t, Ty::U32 | Ty::U64) {
            self.vals.insert(reg.id, Val::Num(Poly::launch(Quantity::Param(index))));
        }
    }

    /// What each launch must hold for the kernel to be safe, once its instructions are all in.
    pub(crate) fn requires(self) -> Vec<Requirement> {
        let skip = grid();
        self.needs
            .into_iter()
            .filter(|(p, (c, _))| (p, *c) != (&skip.0, skip.1))
            .map(|(p, (c, why))| {
                // No sum holds an atom of a thread's, for `need` refuses one that varies.
                let launch = |k: Vec<Atom>| {
                    k.into_iter().filter_map(|a| if let Atom::Launch(q) = a { Some(q) } else { None }).collect()
                };
                let mut terms: Vec<(i128, Vec<Quantity>)> = p.0.into_iter().map(|(k, f)| (f, launch(k))).collect();
                terms.extend((c != 0).then_some((c, vec![])));
                Requirement { terms, why }
            })
            .collect()
    }

    /// Takes in the instruction `i`: an error if it is one the proof does not follow, or reaches what it cannot prove
    /// safe, and else what it makes of the variables, the compares it ends threads by, and what each launch must hold.
    pub(crate) fn inst(&mut self, i: &Step) -> Result<(), Fault> {
        let op = i.bound.form.op;
        if why_not(op).is_some() {
            return Err(refused(i.name, op));
        }
        // The instruction runs only where its guard holds.
        let own = i.guard.and_then(|(p, negated)| self.vals.get(&p.id)?.fact(!negated));
        let facts = self.facts.len();
        self.facts.extend(own);
        let done = if matches!(op, "ld" | "st") { self.access(i) } else { Ok(()) };
        self.facts.truncate(facts);
        done?;
        let value = if i.guard.is_none() { self.value(i) } else { None };
        for r in &i.written {
            self.vals.remove(r);
        }
        if let Some((r, v)) = value {
            self.vals.insert(r, v);
        }
        // The threads that go on past a guarded end did not meet its guard.
        if let Some((p, negated)) = i.guard
            && ENDS.contains(&op)
        {
            self.facts.extend(self.vals.get(&p.id).and_then(|v| v.fact(negated)));
        }
        Ok(())
    }

    /// What the variable `a` holds, if the proof follows it.
    fn val(&self, a: &Given<Var>) -> Option<Val> {
        match a {
            Given::Var(Var::Reg { reg, .. }) => self.vals.get(&reg.id).cloned(),
            Given::Int(x) if *x >= 0 => Some(Val::Num(Poly::int(*x))),
            Given::Special(s) => {
                let atom = match s.as_str() {
                    "tid.x" => Atom::Thread,
                    "ctaid.x" => Atom::Block,
                    "ntid.x" => Atom::Launch(Quantity::Threads),
                    "nctaid.x" => Atom::Launch(Quantity::Blocks),
                    _ => return None,
                };
                Some(Val::Num(Poly::atom(atom)))
            }
            _ => None,
        }
    }

    /// The register `i` writes whole, and what it holds then, if the proof follows it: the sums and products of
    /// numbers, `mov`, `cvt` from 32 bits to 64, and the addresses `mad.wide` and `add` make of a pointer and a number.
    fn value(&mut self, i: &Step) -> Option<(u32, Val)> {
        let Some(Given::Var(Var::Reg { reg, .. })) = i.args.first() else { return None };
        let words: Vec<&str> = i.name.split('.').collect();
        let bits = match words.last().copied() {
            Some("u32" | "b32") => 32,
            Some("u64" | "b64") => 64,
            _ => 0,
        };
        let src: Vec<Option<Val>> = i.args.iter().skip(1).map(|a| self.val(a)).collect();
        // `cvt` cuts a register wider than 32 bits to its low 32 bits, which need not be the number it holds.
        let cut = matches!(i.args.get(1), Some(Given::Var(Var::Reg { ty, .. })) if ty.bits() > 32);
        let num = |v: &Option<Val>| match v {
            Some(Val::Num(p)) => Some(p.clone()),
            _ => None,
        };
        let v = match (words.as_slice(), src.as_slice()) {
            (["mov", _], [Some(v @ (Val::Num(_) | Val::Addr { .. }))]) if bits > 0 => v.clone(),
            (["add", _], [Some(a), Some(b)]) if bits > 0 => self.sum(a, b, bits, false)?,
            (["sub", _], [Some(a), Some(b)]) if bits > 0 => self.sum(a, b, bits, true)?,
            (["mul", "lo", _], [a, b]) if bits > 0 => self.whole(num(a)?.mul(&num(b)?)?, bits)?,
            (["mul", "wide", "u32"], [a, b]) => self.whole(num(a)?.mul(&num(b)?)?, 64)?,
            (["mad", "lo", _], [a, b, c]) if bits > 0 => self.whole(num(a)?.mul(&num(b)?)?.add(&num(c)?)?, bits)?,
            (["mad", "wide", "u32"], [a, b, Some(c)]) => self.sum(&Val::Num(num(a)?.mul(&num(b)?)?), c, 64, false)?,
            (["shl", _], [a, Some(Val::Num(k))]) if bits > 0 => {
                let k = u32::try_from(k.constant()?).ok().filter(|&k| k < bits)?;
                self.whole(num(a)?.scale(1 << k)?, bits)?
            }
            (["cvt", "u64", "u32"], [Some(v @ Val::Num(_))]) if !cut => v.clone(),
            (["setp", cmp, "u32" | "u64"], [a, b]) => {
                let cmp = match *cmp {
                    "lt" => Cmp::Lt,
                    "le" => Cmp::Le,
                    "gt" => Cmp::Gt,
                    "ge" => Cmp::Ge,
                    "eq" => Cmp::Eq,
                    "ne" => Cmp::Ne,
                    _ => return None,
                };
                Val::Cmp { left: num(a)?, cmp, right: num(b)? }
            }
            _ => return None,
        };
        Some((reg.id, v))
    }

    /// The sum or the difference of `a` and `b` in `bits` bits, if the proof follows it.
    fn sum(&mut self, a: &Val, b: &Val, bits: u32, minus: bool) -> Option<Val> {
        let with = |x: &Poly, y: &Poly| if minus { x.sub(y) } else { x.add(y) };
        match (a, b) {
            (Val::Num(x), Val::Num(y)) => self.whole(with(x, y)?, bits),
            (Val::Addr { pointer, offset }, Val::Num(n)) if bits == 64 => {
                Some(Val::Addr { pointer: *pointer, offset: with(offset, n)? })
            }
            (Val::Num(n), Val::Addr { pointer, offset }) if bits == 64 && !minus => {
                Some(Val::Addr { pointer: *pointer, offset: offset.add(n)? })
            }
            _ => None,
        }
    }

    /// The whole number `p`, which a result of `bits` bits holds exactly if no thread of a launch wraps it past 0 or
    /// past what the bits hold: a condition on the launch; none if one wraps whatever the launch is.
    fn whole(&mut self, p: Poly, bits: u32) -> Option<Val> {
        let low = p.bound(false)?;
        let high = Poly::int((1 << bits) - 1).sub(&p.bound(true)?)?;
        if [&low, &high].iter().any(|b| b.constant().is_some_and(|c| c < 0)) {
            return None;
        }
        self.need(low, Why::Fits);
        self.need(high, Why::Fits);
        Some(Val::Num(p))
    }

    /// Makes each launch hold `p >= 0`, a sum of the launch's numbers alone. `false` if no launch can: it does not
    /// hold, whatever they are.
    fn need(&mut self, p: Poly, why: Why) -> bool {
        if p.varies() {
            return false;
        }
        let (rest, c) = p.split();
        let g = rest.0.values().fold(0, |g, &k| gcd(g, k.abs()));
        if g == 0 {
            return c >= 0;
        }
        // Whole numbers: `g * q + c >= 0` is `q + floor(c / g) >= 0`.
        let (rest, c) = (Poly(rest.0.into_iter().map(|(k, f)| (k, f / g)).collect()), c.div_euclid(g));
        // The numbers are not less than 0.
        if c >= 0 && rest.0.values().all(|&f| f > 0) {
            return true;
        }
        match self.needs.get_mut(&rest) {
            Some((least, _)) if *least <= c => {}
            Some(entry) => *entry = (c, why),
            None => {
                self.needs.insert(rest, (c, why));
            }
        }
        true
    }

    /// A load or a store: each of its checks is an error at the instruction, or a condition on the launch.
    fn access(&mut self, i: &Step) -> Result<(), Fault> {
        let name = i.name;
        if let Some(w) = i.bound.words().find(|w| !plain(w)) {
            return Err(cannot(name, &format!("has `.{w}`, which a plain load or store has not")));
        }
        match i.bound.spaces().first().copied().unwrap_or(Space::Generic) {
            Space::Global => {}
            Space::Generic => {
                let any = "reaches memory by a generic address, which may be shared or local memory as well as global";
                return Err(cannot(name, any));
            }
            s => return Err(cannot(name, &format!("reaches {} memory, which no pointer of a kernel is to", s.noun()))),
        }
        let Some((at, m)) = i.args.iter().enumerate().find_map(|(at, a)| match a {
            Given::Mem(m) => Some((at, m)),
            _ => None,
        }) else {
            return Err(cannot(name, "reaches memory at no address"));
        };
        // From here on, the address is what is at fault.
        let here = |what: &str| Fault { arg: Some(at), ..cannot(name, what) };
        let from = match m.base {
            Some(Var::Reg { reg, .. }) => self.vals.get(&reg.id).cloned(),
            _ => None,
        };
        let pointer = match &from {
            Some(Val::Addr { pointer, .. }) => self.pointers.get(pointer).map(|p| (*pointer, p.bytes, p.writable)),
            _ => None,
        };
        let (Some(Val::Addr { offset, .. }), Some((pointer, target, writable))) = (from, pointer) else {
            return Err(here("reaches memory at an address that kernite cannot follow to a pointer of the kernel's"));
        };
        let write = i.bound.form.op == "st";
        if write && !writable {
            return Err(here("writes through a `*const` pointer, memory lent to be read"));
        }
        let bytes = i128::from(i.reach);
        let Some(a) = offset.add(&Poly::int(i128::from(m.disp))) else {
            return Err(here("reaches an address too far for kernite to follow"));
        };
        if !(1..=target).contains(&bytes) {
            let what = format!("reaches {bytes} bytes at once, more than the {target} of what its pointer is to");
            return Err(here(&what));
        }
        if a.0.values().any(|c| c % bytes != 0) {
            let what = format!("reaches an address that is not a multiple of {bytes} bytes from its pointer");
            return Err(here(&what));
        }
        self.inside(pointer, target, &a, bytes).map_err(here)?;
        if writable {
            self.apart(pointer, &a, bytes, write).map_err(here)?;
        }
        Ok(())
    }

    /// That `bytes` at `a` bytes from pointer `pointer` are in its memory, which holds `Len` of its target, `target`
    /// bytes each: as a number the thread has, the launch gives, and the compares of the kernel keep below one that
    /// the launch gives.
    fn inside(&mut self, pointer: usize, target: i128, a: &Poly, bytes: i128) -> Result<(), &'static str> {
        const BEFORE: &str = "reaches an address too far for kernite to follow";
        let size = Poly::launch(Quantity::Len(pointer)).scale(target).ok_or(BEFORE)?;
        let slack = size.sub(a).and_then(|s| s.sub(&Poly::int(bytes))).ok_or(BEFORE)?;
        let low = a.bound(false).ok_or(BEFORE)?;
        // Each thread's index is below what a compare says: the slack is that many times the compare's own.
        let by_fact = self.facts.iter().rev().find_map(|f| {
            let (k, &c) = slack.0.iter().find(|(k, _)| varies(k))?;
            let g = *f.0.get(k)?;
            let m = c.checked_div(g).filter(|&m| m > 0 && c % g == 0)?;
            slack.sub(&f.scale(m)?).filter(|r| !r.varies())
        });
        // Or its index is a block's or a thread's alone, as many as the launch says there are.
        let alone = |(k, &c): (&Vec<Atom>, &i128)| c > 0 && matches!(k.as_slice(), [Atom::Thread] | [Atom::Block]);
        let rows = a.0.iter().filter(|(k, _)| varies(k)).all(alone);
        let high = match slack.varies() {
            false => slack,
            true => match (by_fact, rows) {
                (Some(r), _) => r,
                (None, true) => {
                    let last = a.bound(true).ok_or(BEFORE)?;
                    size.sub(&last).and_then(|s| s.sub(&Poly::int(bytes))).ok_or(BEFORE)?
                }
                (None, false) => {
                    return Err("reaches an element at an index that no compare keeps below the memory's length");
                }
            },
        };
        match self.need(low, Why::Inside(pointer)) && self.need(high, Why::Inside(pointer)) {
            true => Ok(()),
            false => Err("reaches memory before it starts, or past where it ends, whatever the launch is"),
        }
    }

    /// That `bytes` at `a` bytes from pointer `pointer`, which the kernel writes, are in the part of its memory that
    /// only this thread reaches, which each access to it says the same.
    fn apart(&mut self, pointer: usize, a: &Poly, bytes: i128, write: bool) -> Result<(), &'static str> {
        let not = match write {
            true => "writes at an address that is not only its thread's",
            false => "reads memory the kernel may write, at an address that is not only its thread's",
        };
        let (mut shift, mut block, mut thread) = (Poly::default(), Poly::default(), Poly::default());
        for (k, &c) in &a.0 {
            let count = |x| k.iter().filter(|&&a| a == x).count();
            let rest = k.iter().copied().filter(|a| !matches!(a, Atom::Thread | Atom::Block)).collect();
            let term = Poly(BTreeMap::from([(rest, c)]));
            let part = match (count(Atom::Thread), count(Atom::Block)) {
                (0, 0) => &mut shift,
                (1, 0) => &mut thread,
                (0, 1) => &mut block,
                _ => return Err(not),
            };
            *part = part.add(&term).ok_or(not)?;
        }
        let width = thread.constant().filter(|&w| w >= 0).ok_or(not)?;
        let (shift, at) = shift.split();
        let end = at.checked_add(bytes).ok_or(not)?;
        let frame = Frame { shift, block: block.clone(), thread: width };
        let known = self.pointers.get(&pointer).and_then(|p| p.frame.clone());
        if known.as_ref().is_some_and(|f| *f != frame) {
            return Err(not);
        }
        let (threads, blocks) = (Poly::launch(Quantity::Threads), Poly::launch(Quantity::Blocks));
        let at_most_one = |q: &Poly| Poly::int(1).sub(q).ok_or(not);
        if known.is_none() {
            // Each thread has a part of its own: a block has one thread if the offset has no thread, one block if it
            // has no block, and else the part of a block holds its threads' parts.
            let mut holds = true;
            if width == 0 {
                holds &= self.need(at_most_one(&threads)?, Why::Apart(pointer));
            }
            if block.0.is_empty() {
                holds &= self.need(at_most_one(&blocks)?, Why::Apart(pointer));
            } else if width > 0 {
                holds &= self.need(block.sub(&threads.scale(width).ok_or(not)?).ok_or(not)?, Why::Apart(pointer));
            }
            if !holds {
                return Err(not);
            }
            if let Some(p) = self.pointers.get_mut(&pointer) {
                p.frame = Some(frame);
            }
        }
        // And the bytes of the access are in that part.
        let inside = match (width, block.0.is_empty()) {
            (0, true) => true,
            (0, false) => at >= 0 && self.need(block.sub(&Poly::int(end)).ok_or(not)?, Why::Apart(pointer)),
            (w, _) => at >= 0 && end <= w,
        };
        if inside { Ok(()) } else { Err(not) }
    }
}

/// Whether `word` of a load's or a store's name is one that leaves it a plain access: its state space, how many it
/// loads, a type, or how the cache treats it.
fn plain(word: &str) -> bool {
    Ty::parse(word).is_some()
        || word.starts_with("L1::")
        || word.starts_with("L2::")
        || ["weak", "global", "v2", "v4", "ca", "cg", "cs", "lu", "cv", "wb", "wt"].contains(&word)
}

/// The greatest number that divides `a` and `b`.
fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::forms::forms;

    /// The instructions the proof sorts, each by the list it is in.
    fn sorted() -> Vec<&'static str> {
        let words = |list: &'static str| list.split_whitespace().collect::<Vec<_>>();
        let mut sorted = [words(PURE), vec!["ld", "st"], ENDS.to_vec()].concat();
        sorted.extend(REFUSED.iter().flat_map(|(ops, _)| words(ops)));
        sorted
    }

    /// Every instruction of PTX's, as kernite has them, is sorted, once: followed, or refused for a reason. What is
    /// sorted is PTX's, but for `call` and `grid`, which the language has.
    #[test]
    fn each_instruction_of_the_table_is_sorted_once_and_each_sorted_is_in_the_table() {
        let table: BTreeSet<&str> = forms().iter().map(|f| f.op).collect();
        let sorted = sorted();
        for op in &table {
            let times = sorted.iter().filter(|s| *s == op).count();
            assert_eq!(times, 1, "`{op}` is sorted {times} times");
        }
        let stale: Vec<&&str> =
            sorted.iter().filter(|s| !table.contains(**s) && !["call", "grid"].contains(*s)).collect();
        assert!(stale.is_empty(), "no instruction of PTX's, as kernite has them: {stale:?}");
    }

    #[test]
    fn the_proof_follows_the_instructions_that_use_no_memory_but_a_pointers_and_refuses_the_rest() {
        for op in ["add", "mad", "setp", "mov", "cvt", "ld", "st", "ret", "exit", "trap"] {
            assert_eq!(why_not(op), None, "{op}");
        }
        for op in ["bar", "shfl", "atom", "red", "tex", "bra", "cvta", "call", "grid", "mma"] {
            assert!(why_not(op).is_some(), "{op}");
        }
        let refused = refused("bar.sync", "bar").message;
        let what = "makes threads wait for each other, or orders what they write";
        let end = "and kernite cannot prove that this is safe, so a kernel that does is an `unsafe fn`";
        assert_eq!(refused, format!("`bar.sync` {what}, {end}"));
    }
}
