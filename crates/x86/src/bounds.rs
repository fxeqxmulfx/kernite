//! Whether a safe kernel keeps each address it reaches inside the slice it reaches it in.
//!
//! The kernel's gpr variables that the proof needs, each length and each index and the gprs that a copy, a compare or a
//! `lea` relates to them, are followed over its flow, each with how far it may be above each other and above zero:
//! `i - n <= -8` says that `i` is 8 below `n` at least. A compare tells of two variables, on the way a jump goes and
//! on the way it falls through; a copy or an addition of a number known now carries what is known of the variable it
//! changes; any other write forgets the variable. Where ways meet, only what holds on every one is kept, and what
//! changes round a loop is dropped.
//!
//! A variable is a number from 0 to 2^64 - 1, as a gpr holds it, so an addition that may wrap forgets the variable.
//! The bounds are those of numbers that never wrap: a slice is no longer than memory, `isize::MAX` bytes, so an index
//! below its length is far from 2^64.

use std::collections::{BTreeMap, BTreeSet};

use kernite_compile::forward;
use kernite_core::Access::{ReadWrite, Write};
use kernite_core::{Arg, Base, Class, Imm, Inst, Item, Mem, Operand, Program, VReg, Width};

use crate::op::Op;

/// An address a kernel reaches in a slice, which it must keep inside it.
#[derive(Clone, Debug)]
pub(crate) struct Inside {
    /// The item of the body that reaches it.
    pub(crate) at: usize,
    /// The operand of the instruction that is the address, counted as the source writes them.
    pub(crate) operand: usize,
    /// The variable that holds the slice's length.
    pub(crate) len: VReg,
    /// The variable that holds the index, if the address has one.
    pub(crate) index: Option<VReg>,
    /// How many elements it reaches, from the index.
    pub(crate) reaches: u64,
    /// What is wrong if it is not kept inside, as the error says it.
    pub(crate) message: String,
}

/// A slice's length: the variable that holds it, and the most it can be, as long as memory.
pub(crate) struct Length {
    /// The variable.
    pub(crate) var: VReg,
    /// The most elements the slice has.
    pub(crate) most: u64,
}

/// No most is known.
const NONE: i128 = i128::MAX;

/// The node of the number zero, which the bounds of a variable on its own are against.
const ZERO: usize = 0;

/// The sum of two bounds, which is none if either is.
fn plus(a: i128, b: i128) -> i128 {
    if a == NONE || b == NONE { NONE } else { a + b }
}

/// What the flags say: the last compare, if the flags are still its.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flags {
    /// Nothing is known.
    Unknown,
    /// The flags of comparing `a` with `b + k`, as unsigned numbers, `a` and `b` being nodes.
    Cmp { a: usize, b: usize, k: i128 },
}

/// How a condition compares the first of two numbers with the second, as unsigned numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rel {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Rel {
    /// The condition code `cc` says, of a jump or a move; none for one of the signed or the other flags.
    fn of(cc: &str) -> Option<Rel> {
        Some(match cc {
            "e" | "z" => Rel::Eq,
            "ne" | "nz" => Rel::Ne,
            "b" | "c" | "nae" => Rel::Lt,
            "ae" | "nb" | "nc" => Rel::Ge,
            "be" | "na" => Rel::Le,
            "a" | "nbe" => Rel::Gt,
            _ => return None,
        })
    }

    /// The one that holds where this does not.
    fn not(self) -> Rel {
        match self {
            Rel::Eq => Rel::Ne,
            Rel::Ne => Rel::Eq,
            Rel::Lt => Rel::Ge,
            Rel::Ge => Rel::Lt,
            Rel::Le => Rel::Gt,
            Rel::Gt => Rel::Le,
        }
    }
}

/// What is known of the variables at a place of the body.
#[derive(Clone, Debug)]
struct State {
    /// How many nodes: zero, and the variables.
    n: usize,
    /// `bound[a * n + b]` is the most that `a - b` is, or `NONE`.
    bound: Vec<i128>,
    /// What the flags say.
    flags: Flags,
    /// Whether no way of running the kernel comes here: what is known contradicts itself.
    dead: bool,
    /// Whether `bound` has all that follows from it: the bounds kept by [`State::limit`] have, and those that
    /// [`State::widen`] leaves, or [`State::join`] of a state that has not, are closed before they are used.
    closed: bool,
}

impl PartialEq for State {
    fn eq(&self, other: &State) -> bool {
        self.dead == other.dead && (self.dead || (self.bound == other.bound && self.flags == other.flags))
    }
}

impl State {
    /// A state of `n` nodes, of numbers from 0 up that are all that is known.
    fn new(n: usize) -> State {
        let mut state = State { n, bound: vec![NONE; n * n], flags: Flags::Unknown, dead: false, closed: true };
        for a in 0..n {
            state.put(a, a, 0);
            state.put(ZERO, a, 0);
        }
        state
    }

    fn at(&self, a: usize, b: usize) -> i128 {
        self.bound.get(a * self.n + b).copied().unwrap_or(NONE)
    }

    fn put(&mut self, a: usize, b: usize, c: i128) {
        if let Some(slot) = self.bound.get_mut(a * self.n + b) {
            *slot = c;
        }
    }

    /// All that follows from the bounds, as the shortest ways between the nodes say.
    fn close(&mut self) {
        if self.closed {
            return;
        }
        for k in 0..self.n {
            for x in 0..self.n {
                let xk = self.at(x, k);
                if xk == NONE {
                    continue;
                }
                for y in 0..self.n {
                    let by = plus(xk, self.at(k, y));
                    if by < self.at(x, y) {
                        self.put(x, y, by);
                    }
                }
            }
        }
        self.dead |= (0..self.n).any(|x| self.at(x, x) < 0);
        self.closed = true;
    }

    /// Says that `a - b <= c`, and all that follows. A bound that no number keeps makes the state dead.
    fn limit(&mut self, a: usize, b: usize, c: i128) {
        if self.dead || c >= self.at(a, b) {
            return;
        }
        if plus(self.at(b, a), c) < 0 {
            self.dead = true;
            return;
        }
        for x in 0..self.n {
            let xa = plus(self.at(x, a), c);
            if xa == NONE {
                continue;
            }
            for y in 0..self.n {
                let by = plus(xa, self.at(b, y));
                if by < self.at(x, y) {
                    self.put(x, y, by);
                }
            }
        }
    }

    /// Forgets all of `a`, which is a number of any value.
    fn forget(&mut self, a: usize) {
        for x in 0..self.n {
            self.put(a, x, NONE);
            self.put(x, a, NONE);
        }
        self.put(a, a, 0);
        self.limit(ZERO, a, 0);
        self.unflag(a);
    }

    /// The flags say nothing of `a`, which is not what they compared.
    fn unflag(&mut self, a: usize) {
        if matches!(self.flags, Flags::Cmp { a: x, b: y, .. } if x == a || y == a) {
            self.flags = Flags::Unknown;
        }
    }

    /// `a = b + c`, as numbers, where that does not wrap; else `a` is a number of any value. `b` may be `a`.
    fn assign(&mut self, a: usize, b: usize, c: i128) {
        let fits = if c >= 0 { self.at(b, ZERO) <= i128::from(u64::MAX) - c } else { self.at(ZERO, b) <= c };
        if !fits || self.dead {
            return self.forget(a);
        }
        if a == b {
            // The same variable, moved by `c`: what is known of it moves as well.
            for x in (0..self.n).filter(|&x| x != a) {
                let (above, below) = (self.at(a, x), self.at(x, a));
                self.put(a, x, plus(above, c));
                self.put(x, a, plus(below, -c));
            }
            self.unflag(a);
        } else {
            self.forget(a);
            self.limit(a, b, c);
            self.limit(b, a, -c);
        }
    }

    /// `a = value`, a number known now.
    fn set(&mut self, a: usize, value: u64) {
        self.forget(a);
        self.limit(a, ZERO, i128::from(value));
        self.limit(ZERO, a, -i128::from(value));
    }

    /// What the compare of `flags` says on the way a jump `cc` goes, `taken`, or on the way it falls through.
    fn refine(&mut self, cc: &str, taken: bool) {
        let Flags::Cmp { a, b, k } = self.flags else { return };
        let Some(rel) = Rel::of(cc) else { return };
        match if taken { rel } else { rel.not() } {
            Rel::Eq => {
                self.limit(a, b, k);
                self.limit(b, a, -k);
            }
            Rel::Lt => self.limit(a, b, k - 1),
            Rel::Le => self.limit(a, b, k),
            Rel::Gt => self.limit(b, a, -k - 1),
            Rel::Ge => self.limit(b, a, -k),
            // Not equal says more where `a` is known to be on one side of `b + k`, as a count that reaches its end
            // by steps of 1 is.
            Rel::Ne => {
                if self.at(b, a) <= -k {
                    self.limit(b, a, -k - 1);
                } else if self.at(a, b) <= k {
                    self.limit(a, b, k - 1);
                }
            }
        }
    }

    /// What holds in this state and in `other`: each bound the greater of the two.
    fn join(&self, other: &State) -> State {
        if self.dead {
            return other.clone();
        }
        if other.dead {
            return self.clone();
        }
        let bound = self.bound.iter().zip(&other.bound).map(|(&a, &b)| a.max(b)).collect();
        State {
            bound,
            flags: if self.flags == other.flags { self.flags } else { Flags::Unknown },
            closed: self.closed && other.closed,
            ..self.clone()
        }
    }

    /// What holds in this state, the one a place had, and in `other`, one more way to it: each bound that `other`
    /// keeps, and none that it does not, so that a bound changes only so many times round a loop.
    fn widen(&self, other: &State) -> State {
        if other.dead {
            return self.clone();
        }
        if self.dead {
            return other.clone();
        }
        let bound = self.bound.iter().zip(&other.bound).map(|(&a, &b)| if b <= a { a } else { NONE }).collect();
        State {
            bound,
            flags: if self.flags == other.flags { self.flags } else { Flags::Unknown },
            closed: false,
            ..self.clone()
        }
    }
}

/// The kernel's variables as the nodes of the bounds.
struct Bounds {
    /// The node of each variable that has one, by its number: those that [`needed`] says.
    nodes: BTreeMap<u32, usize>,
}

impl Bounds {
    fn node(&self, v: VReg) -> Option<usize> {
        self.nodes.get(&v.id).copied()
    }

    /// The node of the gpr `a` as the whole register, if it is one.
    fn whole(&self, a: &Arg<VReg>) -> Option<usize> {
        match a.operand {
            Operand::Reg(v, Width::W64) => self.node(v),
            _ => None,
        }
    }

    /// The number `a` is, if it is one known now.
    fn imm(a: &Arg<VReg>) -> Option<i64> {
        match a.operand {
            Operand::Imm(Imm { value, .. }) => Some(value),
            _ => None,
        }
    }

    /// Whether the address `o` is kept inside, where the state is `now`.
    fn holds(&self, now: &State, o: &Inside) -> bool {
        let index = o.index.map_or(Some(ZERO), |i| self.node(i));
        let len = self.node(o.len);
        now.dead || index.zip(len).is_some_and(|(i, l)| now.at(i, l) <= -i128::from(o.reaches))
    }

    /// Carries out the instruction `name` on `now`, where it is one of those whose result is known of the variables;
    /// whether it is.
    fn model(&self, name: &str, args: &[Arg<VReg>], now: &mut State) -> bool {
        match (name, args) {
            ("mov", [to, from]) => match (self.whole(to), self.whole(from), Self::imm(from)) {
                (Some(a), Some(b), _) => now.assign(a, b, 0),
                (Some(a), None, Some(value)) => now.set(a, value as u64),
                _ => return false,
            },
            // `xor` of a gpr with itself, whole or its low 32 bits, as `zero` is: 0.
            ("xor", [to, from]) if to.operand == from.operand => match to.operand {
                Operand::Reg(a, Width::W64 | Width::W32) => match self.node(a) {
                    Some(a) => now.set(a, 0),
                    None => return false,
                },
                _ => return false,
            },
            ("lea", [to, from]) => match (self.whole(to), from.operand) {
                (Some(a), Operand::Mem(Mem { base: Base::Reg(b), index: None, disp, .. })) => match self.node(b) {
                    Some(b) => now.assign(a, b, i128::from(disp)),
                    None => return false,
                },
                _ => return false,
            },
            ("add" | "sub", [to, by]) => match (self.whole(to), Self::imm(by)) {
                (Some(a), Some(c)) => now.assign(a, a, i128::from(c) * if name == "add" { 1 } else { -1 }),
                _ => return false,
            },
            ("inc" | "dec", [to]) => match self.whole(to) {
                Some(a) => now.assign(a, a, if name == "inc" { 1 } else { -1 }),
                None => return false,
            },
            _ => return false,
        }
        true
    }

    /// The state after the instruction `inst`, given the one before it.
    fn apply(&self, inst: &Inst<Op, VReg>, now: &mut State) {
        let (name, args) = (inst.op.name(), inst.args.as_slice());
        if let Some(cc) = name.strip_prefix("cmov")
            && let [to, from] = args
            && let (Some(a), Some(b)) = (self.whole(to), self.whole(from))
        {
            // Where the condition holds `a` is `b`, and where it does not `a` is as it was: what is known of either.
            let (mut moved, mut stayed) = (now.clone(), now.clone());
            moved.refine(cc, true);
            moved.assign(a, b, 0);
            stayed.refine(cc, false);
            *now = moved.join(&stayed);
            return;
        }
        let compared = match (name, args) {
            ("cmp", [a, b]) => match (self.whole(a), self.whole(b), Self::imm(b)) {
                (Some(a), Some(b), _) => Some(Flags::Cmp { a, b, k: 0 }),
                (Some(a), None, Some(value)) => Some(Flags::Cmp { a, b: ZERO, k: i128::from(value as u64) }),
                _ => Some(Flags::Unknown),
            },
            ("test", [a, b]) if a.operand == b.operand => {
                Some(self.whole(a).map_or(Flags::Unknown, |a| Flags::Cmp { a, b: ZERO, k: 0 }))
            }
            _ => None,
        };
        if !self.model(name, args, now) {
            for arg in inst.operands() {
                if let (Operand::Reg(v, _), Write | ReadWrite) = (arg.operand, arg.access)
                    && let Some(a) = self.node(v)
                {
                    now.forget(a);
                }
            }
        }
        // The instructions that leave the flags as they were: `mov`, `lea`, the jumps and the moves on a condition.
        let keeps = matches!(name, "mov" | "lea") || name.starts_with('j') || name.starts_with("cmov");
        match compared {
            Some(flags) => now.flags = flags,
            None if !keeps => now.flags = Flags::Unknown,
            None => {}
        }
    }
}

/// The two gprs, whole, that `inst` relates, if it is a `mov`, a `cmp` or a move on a condition of one to the other, or
/// a `lea` of one at a displacement from the other: what the bounds of one say of the other.
fn relates(inst: &Inst<Op, VReg>) -> Option<(VReg, VReg)> {
    let name = inst.op.name();
    let [Arg { operand: Operand::Reg(a, Width::W64), .. }, from] = inst.args.as_slice() else { return None };
    let b = match from.operand {
        Operand::Reg(b, Width::W64) if matches!(name, "mov" | "cmp") || name.starts_with("cmov") => b,
        Operand::Mem(Mem { base: Base::Reg(b), index: None, .. }) if name == "lea" => b,
        _ => return None,
    };
    (a.class == Class::Gpr && b.class == Class::Gpr).then_some((*a, b))
}

/// The gprs the proof needs a node for: each length and each index, and each gpr that a `mov`, a `cmp`, a `cmov` or a
/// `lea` relates to one of them, and so on. A gpr of no node is a number of any value, which costs the proof nothing,
/// however many the kernel has.
fn needed(p: &Program<Op>, lengths: &[Length], inside: &[Inside]) -> BTreeSet<VReg> {
    let mut related: BTreeMap<VReg, Vec<VReg>> = BTreeMap::new();
    for item in &p.body {
        if let Item::Inst(inst) = item
            && let Some((a, b)) = relates(inst)
        {
            related.entry(a).or_default().push(b);
            related.entry(b).or_default().push(a);
        }
    }
    let mut todo: Vec<VReg> = lengths.iter().map(|l| l.var).chain(inside.iter().filter_map(|o| o.index)).collect();
    let mut needed = BTreeSet::new();
    while let Some(v) = todo.pop() {
        if needed.insert(v) {
            todo.extend(related.get(&v).into_iter().flatten());
        }
    }
    needed
}

/// The first address of `inside` that the kernel `p` does not keep inside its slice, or write of a slice's length, if
/// there is one: the item of the body, the operand of the instruction it is at, and what is wrong. `lengths` are the
/// variables that hold the lengths of the slices the kernel is given. A body that is not whole has none: allocating
/// registers says what is wrong with it.
pub(crate) fn unproven(p: &Program<Op>, lengths: &[Length], inside: &[Inside]) -> Option<(usize, usize, String)> {
    let nodes = needed(p, lengths, inside).into_iter().zip(1..).map(|(v, node)| (v.id, node)).collect();
    let bounds = Bounds { nodes };
    let mut start = State::new(bounds.nodes.len() + 1);
    for length in lengths {
        if let Some(len) = bounds.node(length.var) {
            start.limit(len, ZERO, i128::from(length.most));
        }
    }
    let said = forward(
        p,
        start,
        |old: &State, new: &State, back: bool| if back { old.widen(new) } else { old.join(new) },
        |i, inst, now: &mut State| {
            now.close();
            let mut wrong: Vec<(usize, String)> = inside
                .iter()
                .filter(|o| o.at == i && !bounds.holds(now, o))
                .map(|o| (o.operand, o.message.clone()))
                .collect();
            // The lengths stay as the kernel was given them.
            for (k, arg) in inst.args.iter().enumerate() {
                if let (Operand::Reg(v, _), Write | ReadWrite) = (arg.operand, arg.access)
                    && let Some(length) = lengths.iter().find(|l| l.var == v)
                {
                    let len = p.var_name(length.var);
                    wrong.push((
                        k,
                        format!(
                            "`{len}` is the length of a slice, which kernite keeps as it was given to prove each \
                             address inside the slice, so a kernel that writes it is an `unsafe fn`; `mov(n, {len})` \
                             copies it to a variable `n`, which may change"
                        ),
                    ));
                }
            }
            bounds.apply(inst, now);
            wrong
        },
        |inst, taken, state: &mut State| {
            if let Some(cc) = inst.op.name().strip_prefix('j') {
                state.refine(cc, taken);
            }
        },
    )
    .ok()?;
    p.body.iter().zip(said).enumerate().find_map(|(i, (item, said))| {
        let Item::Inst(_) = item else { return None };
        let (operand, message) = said?.into_iter().min_by_key(|(operand, _)| *operand)?;
        Some((i, operand, message))
    })
}
