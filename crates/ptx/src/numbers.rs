//! The whole numbers a kernel works out, as its instructions come: each variable as a sum of products of the thread's
//! and the launch's numbers, or as an address `offset` bytes from a pointer parameter. What proves a kernel with no
//! `unsafe` safe, and what says what a kernel moves, both read them, so that kernite follows numbers by one code.
//!
//! The kernel is followed straight through, for a kernel with no jump. What it cannot know at once, because the launch
//! gives it, is a condition on the launch, which the host checks before a kernel with no `unsafe` runs: that the memory
//! has as many elements as the kernel reaches, and that no number wraps.

use std::collections::BTreeMap;

use kernite_build::{Quantity, Requirement, Why};
use kernite_core::VReg;
use kernite_lang::Operand as Given;

use crate::forms::Bound;
use crate::ty::Ty;
use crate::var::Var;

/// A number the proof reasons about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Atom {
    /// The thread's index in its block, `tid.x`: from 0 to one less than the block's threads.
    Thread,
    /// The block's index in its grid, `ctaid.x`: from 0 to one less than the grid's blocks.
    Block,
    /// A number the launch gives, not less than 0, and not less than 1 for a count of threads or of blocks.
    Launch(Quantity),
}

/// Whether a product of atoms is not the same for each thread.
pub(crate) fn varies(atoms: &[Atom]) -> bool {
    atoms.iter().any(|a| matches!(a, Atom::Thread | Atom::Block))
}

/// A whole number, as a sum of products of atoms, each with its factor. No sum has a factor of 0, nor of `i128::MIN`,
/// whose negation and absolute value are past `i128`: each factor can be negated and divided by another.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Poly(pub(crate) BTreeMap<Vec<Atom>, i128>);

impl Poly {
    pub(crate) fn int(x: i128) -> Poly {
        Poly(BTreeMap::from([(vec![], x)]).into_iter().filter(|&(_, c)| c != 0).collect())
    }

    pub(crate) fn atom(a: Atom) -> Poly {
        Poly(BTreeMap::from([(vec![a], 1)]))
    }

    pub(crate) fn launch(q: Quantity) -> Poly {
        Poly::atom(Atom::Launch(q))
    }

    /// `None` where a factor is past `i128`, or is `i128::MIN`.
    pub(crate) fn add(&self, other: &Poly) -> Option<Poly> {
        let mut sum = self.0.clone();
        for (k, c) in &other.0 {
            let e = sum.entry(k.clone()).or_insert(0);
            *e = e.checked_add(*c).filter(|&e| e != i128::MIN)?;
        }
        sum.retain(|_, c| *c != 0);
        Some(Poly(sum))
    }

    pub(crate) fn scale(&self, by: i128) -> Option<Poly> {
        let mut out = BTreeMap::new();
        for (k, c) in &self.0 {
            out.insert(k.clone(), c.checked_mul(by).filter(|&c| c != i128::MIN)?);
        }
        out.retain(|_, c| *c != 0);
        Some(Poly(out))
    }

    pub(crate) fn sub(&self, other: &Poly) -> Option<Poly> {
        self.add(&other.scale(-1)?)
    }

    pub(crate) fn mul(&self, other: &Poly) -> Option<Poly> {
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
    pub(crate) fn varies(&self) -> bool {
        self.0.keys().any(|k| varies(k))
    }

    /// The number it is, if it holds no atom.
    pub(crate) fn constant(&self) -> Option<i128> {
        match self.0.iter().next() {
            None => Some(0),
            Some((k, c)) if k.is_empty() && self.0.len() == 1 => Some(*c),
            _ => None,
        }
    }

    /// What it is but for its constant, and the constant.
    pub(crate) fn split(&self) -> (Poly, i128) {
        let mut rest = self.0.clone();
        let c = rest.remove(&vec![]).unwrap_or(0);
        (Poly(rest), c)
    }

    /// The most, or the least, it is over every thread of a launch, which is a sum of its own: each product of the
    /// thread's or the block's index, with a factor above 0, takes the index's greatest value, and one below, its
    /// least, 0, or the other way about; the launch's numbers are not less than 0.
    pub(crate) fn bound(&self, high: bool) -> Option<Poly> {
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
pub(crate) enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// What is known of a variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Val {
    /// A whole number, exactly: no sum of it wraps, as the conditions on the launch have it.
    Num(Poly),
    /// An address `offset` bytes from where the pointer parameter `pointer` points.
    Addr { pointer: usize, offset: Poly },
    /// What `setp` compared: `left` and `right` are whole numbers.
    Cmp { left: Poly, cmp: Cmp, right: Poly },
}

impl Val {
    /// A number not less than 0 that holds where `left cmp right` holds, if `holds`, and where it does not, if not.
    pub(crate) fn fact(&self, holds: bool) -> Option<Poly> {
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

/// What is known of a pointer parameter.
#[derive(Debug)]
pub(crate) struct Pointer {
    /// The bytes of what it points to, and so how it is aligned.
    pub(crate) bytes: i128,
    pub(crate) writable: bool,
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

/// One instruction that is followed: its name, what the language bound it to, its operands and the guard it runs
/// under, the registers it writes, and the bytes it reaches at an address.
pub(crate) struct Step<'a> {
    pub name: &'a str,
    pub bound: &'a Bound,
    pub args: &'a [Given<Var>],
    pub guard: Option<(VReg, bool)>,
    pub written: Vec<u32>,
    pub reach: u64,
}

/// What every launch of a kernel with no `unsafe` holds, whether the kernel says it or not: its grid has fewer than
/// 2^32 threads, and so the index `ctaid.x * ntid.x + tid.x` numbers each apart in 32 bits.
fn grid() -> (Poly, i128) {
    let threads = Poly::launch(Quantity::Threads);
    (threads.mul(&Poly::launch(Quantity::Blocks)).and_then(|p| p.scale(-1)).unwrap_or_default(), (1 << 32) - 1)
}

/// What is known of a kernel's variables so far.
#[derive(Debug)]
pub(crate) struct Numbers {
    /// What each variable holds, by its place among the kernel's, where it is followed.
    pub(crate) vals: BTreeMap<u32, Val>,
    /// Each pointer parameter, by its place among the parameters.
    pub(crate) pointers: BTreeMap<usize, Pointer>,
    /// What each launch must hold: for each sum that is not less than 0, but for its constant, the least constant.
    pub(crate) needs: BTreeMap<Poly, (i128, Why)>,
}

impl Numbers {
    /// What is known of a kernel that has no instruction yet.
    pub(crate) fn new() -> Numbers {
        let (grid, most) = grid();
        let needs = BTreeMap::from([(grid, (most, Why::Fits))]);
        Numbers { vals: BTreeMap::new(), pointers: BTreeMap::new(), needs }
    }

    /// The parameter number `index` of the kernel, of type `ty` as written, which is in register `reg`, of type `t`.
    pub(crate) fn param(&mut self, reg: VReg, index: usize, ty: &str, t: Ty) {
        let pointer =
            ty.strip_prefix("*const ").map(|to| (to, false)).or_else(|| ty.strip_prefix("*mut ").map(|to| (to, true)));
        if let Some((to, writable)) = pointer {
            let bytes = size(to);
            if bytes > 0 {
                self.pointers.insert(index, Pointer { bytes, writable });
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

    /// Takes in the instruction `i`: what it makes of the variables.
    pub(crate) fn follow(&mut self, i: &Step) {
        let value = if i.guard.is_none() { self.value(i) } else { None };
        for r in &i.written {
            self.vals.remove(r);
        }
        if let Some((r, v)) = value {
            self.vals.insert(r, v);
        }
    }

    /// What the variable `a` holds, if it is followed.
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

    /// The register `i` writes whole, and what it holds then, if it is followed: the sums and products of numbers,
    /// `mov`, `cvt` from 32 bits to 64, and the addresses `mad.wide` and `add` make of a pointer and a number.
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

    /// The sum or the difference of `a` and `b` in `bits` bits, if it is followed.
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
    pub(crate) fn need(&mut self, p: Poly, why: Why) -> bool {
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
}

/// The greatest number that divides `a` and `b`.
fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a } else { gcd(b, a % b) }
}
