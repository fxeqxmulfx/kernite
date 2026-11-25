//! What proves a kernel with no `unsafe` safe, as its instructions are written: each load and store reaches one whole
//! element, inside its memory at its alignment, no two threads write one address, and none reads what another writes.
//!
//! The proof reads what [`Numbers`] knows of the kernel's variables, the thread's and the launch's numbers and the
//! addresses they make of a pointer parameter. What it cannot prove at once, because it is about a number the launch
//! gives, it leaves as a condition on the launch, which the host checks before the kernel runs: that the memory has as
//! many elements as the kernel reaches, and that no number wraps. What no launch can mend, or the kernel's instructions
//! are none that it follows, is an error at the instruction.

use std::collections::BTreeMap;

use kernite_build::{Quantity, Why};
use kernite_lang::{Fault, Operand as Given};

use crate::forms::Space;
use crate::numbers::{Atom, Numbers, Poly, Step, Val, varies};
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

/// The part of a pointer's memory that a thread has for itself, as each access to it says: its offset is `shift`, the
/// same for every thread, and `block` times the thread's block, and `thread` times its index in its block, and its
/// bytes are in the `thread` bytes from there, if there are any.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Frame {
    shift: Poly,
    block: Poly,
    thread: i128,
}

/// What a kernel with no `unsafe` is proved to do so far.
#[derive(Debug, Default)]
pub(crate) struct Proof {
    /// Numbers that are not less than 0 where the kernel is, from the compares that end the threads that fail them.
    facts: Vec<Poly>,
    /// The part of each pointer's memory, by its place among the parameters, that a thread has for itself, once an
    /// access says it.
    frames: BTreeMap<usize, Frame>,
}

impl Proof {
    /// Takes in the instruction `i`, before `n` follows it: an error if it is one the proof does not follow, or reaches
    /// what it cannot prove safe, and else the compares it ends threads by.
    pub(crate) fn inst(&mut self, n: &mut Numbers, i: &Step) -> Result<(), Fault> {
        let op = i.bound.form.op;
        if why_not(op).is_some() {
            return Err(refused(i.name, op));
        }
        // The instruction runs only where its guard holds.
        let own = i.guard.and_then(|(p, negated)| n.vals.get(&p.id)?.fact(!negated));
        let facts = self.facts.len();
        self.facts.extend(own);
        let done = if matches!(op, "ld" | "st") { self.access(n, i) } else { Ok(()) };
        self.facts.truncate(facts);
        done?;
        // The threads that go on past a guarded end did not meet its guard.
        if let Some((p, negated)) = i.guard
            && ENDS.contains(&op)
        {
            self.facts.extend(n.vals.get(&p.id).and_then(|v| v.fact(negated)));
        }
        Ok(())
    }

    /// A load or a store: each of its checks is an error at the instruction, or a condition on the launch.
    fn access(&mut self, n: &mut Numbers, i: &Step) -> Result<(), Fault> {
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
            Some(Var::Reg { reg, .. }) => n.vals.get(&reg.id).cloned(),
            _ => None,
        };
        let pointer = match &from {
            Some(Val::Addr { pointer, .. }) => n.pointers.get(pointer).map(|p| (*pointer, p.bytes, p.writable)),
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
        // PTX takes a vector as one access for each of its scalars, so one of fewer bytes than the element reaches part
        // of it, as a scalar of fewer bytes does: the host may store the whole element, by its atomics, while the
        // kernel runs, and a race of two sizes is outside PTX's memory model.
        let scalar = i128::from(i.bound.scalar().unwrap_or(i.reach));
        if scalar < target {
            let what = format!("reaches part of an element, {scalar} of the {target} bytes of what its pointer is to");
            return Err(here(&what));
        }
        if a.0.values().any(|c| c % bytes != 0) {
            let what = format!("reaches an address that is not a multiple of {bytes} bytes from its pointer");
            return Err(here(&what));
        }
        self.inside(n, pointer, target, &a, bytes).map_err(here)?;
        if writable {
            self.apart(n, pointer, &a, bytes, write).map_err(here)?;
        }
        Ok(())
    }

    /// That `bytes` at `a` bytes from pointer `pointer` are in its memory, which holds `Len` of its target, `target`
    /// bytes each: as a number the thread has, the launch gives, and the compares of the kernel keep below one that
    /// the launch gives.
    fn inside(&self, n: &mut Numbers, pointer: usize, target: i128, a: &Poly, bytes: i128) -> Result<(), &'static str> {
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
        match n.need(low, Why::Inside(pointer)) && n.need(high, Why::Inside(pointer)) {
            true => Ok(()),
            false => Err("reaches memory before it starts, or past where it ends, whatever the launch is"),
        }
    }

    /// That `bytes` at `a` bytes from pointer `pointer`, which the kernel writes, are in the part of its memory that
    /// only this thread reaches, which each access to it says the same.
    fn apart(
        &mut self,
        n: &mut Numbers,
        pointer: usize,
        a: &Poly,
        bytes: i128,
        write: bool,
    ) -> Result<(), &'static str> {
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
        let known = self.frames.get(&pointer).cloned();
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
                holds &= n.need(at_most_one(&threads)?, Why::Apart(pointer));
            }
            if block.0.is_empty() {
                holds &= n.need(at_most_one(&blocks)?, Why::Apart(pointer));
            } else if width > 0 {
                holds &= n.need(block.sub(&threads.scale(width).ok_or(not)?).ok_or(not)?, Why::Apart(pointer));
            }
            if !holds {
                return Err(not);
            }
            self.frames.insert(pointer, frame);
        }
        // And the bytes of the access are in that part.
        let inside = match (width, block.0.is_empty()) {
            (0, true) => true,
            (0, false) => at >= 0 && n.need(block.sub(&Poly::int(end)).ok_or(not)?, Why::Apart(pointer)),
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
