use kernite_core::{Flow, Item, PReg, Program, VReg};

use crate::Error;
use crate::bits::Bits;
use crate::live::{analyse, first_mentions, interference};
use crate::spill;
use crate::{Machine, Spill};

/// `p` as allocated, and each of its variables' registers, by [`VReg::id`]; `None` for a
/// variable the body never names.
///
/// Liveness is solved over the blocks to a fixed point. Two variables interfere when one is
/// written while the other is live, and a variable live across an instruction that changes
/// registers besides its operands, as a call does, may take none of those. The variables the
/// calling convention places take their registers first: the parameters, the results, and a
/// call's arguments and result. Then each other variable, in the order the body first names
/// it, takes the register of a call's argument or result it is copied to or from, if it may,
/// or else the first register the machine offers that none of the variables it interferes with
/// holds.
///
/// A parameter or a result that cannot keep its register throughout, since another placed
/// variable needs it while both are live, or an instruction it is live across changes it,
/// splits: [`split`] moves it to a variable of its own, and the program is allocated again.
///
/// A variable left without a register spills, or one of those holding the registers it could
/// have does, whichever [`spill::choose`] finds cheapest to keep on the stack; [`spill::rewrite`]
/// keeps it in a slot of its own, and the program is allocated again, until every variable has
/// a register. Each of the program's variables splits once at most and spills once at most, and
/// none that splitting or spilling made, nor a call's argument or result. No choice depends on
/// anything but the program, so the same program always gets the same registers and the same
/// spills.
pub(crate) fn allocate<Op: Clone + PartialEq>(
    p: &Program<Op>,
    m: &dyn Machine,
    s: &dyn Spill<Op>,
) -> Result<(Program<Op>, Vec<Option<PReg>>), Error> {
    let mut p = p.clone();
    // Which variables may still spill: the program's own, until they do, but a call's.
    let mut may = vec![true; p.regs.len()];
    // The variables placed for a while alone, which a variable copied to or from one would
    // rather share a register with: a call's arguments and results, and those splitting makes.
    let mut short = vec![];
    for &(v, _) in &p.pinned {
        may[v.id as usize] = false;
        short.push(v.id as usize);
    }
    loop {
        let (blocks, live) = analyse(&p)?;
        let (adj, changed) = interference(&p, &blocks, &live);
        let regs = match fixed(&p, &adj, &changed) {
            Ok(regs) => regs,
            Err(clash) => {
                let v = clash.mended(&p, &short).ok_or_else(|| clash.error(&p))?;
                short.push(split(&mut p, v, s));
                continue;
            }
        };
        let prefer = preferences(&p, s, &short);
        let (v, held) = match colour(&p, m, &adj, &changed, &prefer, regs) {
            Ok(regs) => return Ok((p, regs)),
            Err(short) => short,
        };
        let class = p.regs[v].class;
        let offered = m.allocatable(class);
        let have = offered.len();
        // Those holding a register it could have had, which one of them spilling would free.
        let taken: Vec<(usize, PReg)> =
            held.into_iter().filter(|(_, r)| offered.contains(r) && !changed[v].contains(r)).collect();
        let candidates: Vec<usize> = std::iter::once(v)
            .chain(taken.iter().map(|&(u, _)| u))
            .filter(|&u| have > 0 && may.get(u) == Some(&true) && spill::may(&p, u))
            .collect();
        let Some(spilled) = spill::choose(&p, &blocks, &live, &candidates) else {
            let name = |i: usize| p.regs[i].name.clone();
            let vreg = p.vreg(v);
            let first = first_mentions(&p)[v].expect("a variable coloured is named");
            let near = p.body[..first].iter().rev().find_map(|item| match item {
                Item::Note(note) => Some(note.clone()),
                _ => None,
            });
            let changed = offered.iter().filter(|r| changed[v].contains(r)).copied().collect();
            let taken = taken.into_iter().map(|(u, r)| (name(u), r)).collect();
            return Err(Error::OutOfRegisters { class, var: name(v), vreg, near, taken, changed, have });
        };
        may[spilled] = false;
        spill::rewrite(&mut p, spilled, s);
    }
}

/// Why the places the calling convention gives variables cannot all hold.
#[derive(Clone, Copy, Debug)]
enum Clash {
    /// A variable is placed in two registers: a parameter returned in another register.
    Twice(usize),
    /// Two are placed in one register, the first, and the second needs it while the first is
    /// live.
    Shared(usize, usize, PReg),
    /// One is placed in a register that an instruction it is live across changes.
    Changed(usize, PReg),
}

impl Clash {
    /// The parameter or result whose splitting would mend it, if one may split: one the
    /// program has, not one placed for a while alone; a result before a parameter, a result
    /// moving into its register just before it returns.
    fn mended<Op>(self, p: &Program<Op>, short: &[usize]) -> Option<usize> {
        let result = |v: usize| p.results.iter().any(|(r, _)| r.id as usize == v);
        let param = |v: usize| p.params.iter().any(|(q, _)| q.id as usize == v);
        let may = |v: &usize| !short.contains(v) && (result(*v) || param(*v));
        let these = match self {
            Clash::Twice(v) | Clash::Changed(v, _) => vec![v],
            Clash::Shared(a, b, _) => vec![a, b],
        };
        these.iter().copied().filter(may).find(|&v| result(v)).or_else(|| these.into_iter().find(may))
    }

    /// The error it is, when nothing may split to mend it.
    fn error<Op>(self, p: &Program<Op>) -> Error {
        let name = |i: usize| p.regs[i].name.clone();
        match self {
            Clash::Twice(v) => Error::FixedTwice { var: name(v), vreg: p.vreg(v) },
            Clash::Shared(a, b, reg) => Error::FixedClash { a: name(a), vreg: p.vreg(a), b: name(b), reg },
            Clash::Changed(v, reg) => Error::FixedChanged { var: name(v), vreg: p.vreg(v), reg },
        }
    }
}

/// The registers of the variables the calling convention places, the parameters, the results
/// and a call's arguments and result, once it is known that it places none twice, nor two in
/// one register while both are live, nor one in a register an instruction it is live across
/// changes.
fn fixed<Op>(p: &Program<Op>, adj: &[Bits], changed: &[Vec<PReg>]) -> Result<Vec<Option<PReg>>, Clash> {
    let placed = || p.params.iter().chain(&p.results).chain(&p.pinned).map(|&(v, r)| (v.id as usize, r));
    let mut regs: Vec<Option<PReg>> = vec![None; p.regs.len()];
    for (v, r) in placed() {
        match regs[v] {
            Some(other) if other != r => return Err(Clash::Twice(v)),
            _ => regs[v] = Some(r),
        }
    }
    for (v, r) in placed() {
        if let Some(u) = adj[v].iter().find(|&u| regs[u] == Some(r)) {
            return Err(Clash::Shared(v, u, r));
        }
    }
    match placed().find(|(v, r)| changed[*v].contains(r)) {
        Some((v, r)) => Err(Clash::Changed(v, r)),
        None => Ok(regs),
    }
}

/// `p` with `v`, a parameter or a result, moved by `s`'s copy to a variable made for it, of
/// its name, which the calling convention places instead: so that `v` is a variable like any
/// other. A result is copied there before each return that reads it, which reads that instead;
/// a parameter arrives there and is copied to `v` on entry. A variable both is a result first.
/// It returns the variable made.
fn split<Op>(p: &mut Program<Op>, v: usize, s: &dyn Spill<Op>) -> usize {
    let info = p.regs[v].clone();
    let var = p.vreg(v);
    let made = VReg { id: p.regs.len() as u32, class: info.class };
    p.regs.push(info);
    match p.results.iter().position(|&(r, _)| r == var) {
        Some(at) => {
            p.results[at].0 = made;
            for item in std::mem::take(&mut p.body) {
                match item {
                    Item::Inst(inst) if inst.flow == Flow::Return && inst.reads().any(|r| r == var) => {
                        p.body.push(Item::Inst(s.copy(made, var)));
                        p.body.push(Item::Inst(spill::renamed(inst, var, made)));
                    }
                    item => p.body.push(item),
                }
            }
        }
        None => {
            let at = p.params.iter().position(|&(q, _)| q == var).expect("a parameter or a result splits");
            p.params[at].0 = made;
            p.body.insert(0, Item::Inst(s.copy(var, made)));
        }
    }
    made.id as usize
}

/// For each variable, the variables of `short` it is copied to or from by `s`'s copy, in body
/// order: it would rather have the register of one, so that the copy moves nothing.
fn preferences<Op: PartialEq>(p: &Program<Op>, s: &dyn Spill<Op>, short: &[usize]) -> Vec<Vec<usize>> {
    let mut prefer = vec![vec![]; p.regs.len()];
    for item in &p.body {
        let Item::Inst(inst) = item else { continue };
        let (writes, reads): (Vec<VReg>, Vec<VReg>) = (inst.writes().collect(), inst.reads().collect());
        let ([to], [from]) = (writes.as_slice(), reads.as_slice()) else { continue };
        if to == from || to.class != from.class || *inst != s.copy(*to, *from) {
            continue;
        }
        let (to, from) = (to.id as usize, from.id as usize);
        if short.contains(&from) {
            prefer[to].push(from);
        }
        if short.contains(&to) {
            prefer[from].push(to);
        }
    }
    prefer
}

/// A variable colouring left without a register, and the variables beside it that hold one,
/// each with the register it holds.
type Short = (usize, Vec<(usize, PReg)>);

/// `regs`, the placed variables' registers, with one for every other variable the body names;
/// or the first variable left without one. A variable takes none that an instruction it is live
/// across changes, of `changed`; it takes the register of one it would rather share it with, of
/// `prefer`, if it may.
fn colour<Op>(
    p: &Program<Op>,
    m: &dyn Machine,
    adj: &[Bits],
    changed: &[Vec<PReg>],
    prefer: &[Vec<usize>],
    mut regs: Vec<Option<PReg>>,
) -> Result<Vec<Option<PReg>>, Short> {
    let first = first_mentions(p);
    let mut order: Vec<usize> = (0..p.regs.len()).filter(|&v| first[v].is_some()).collect();
    order.sort_by_key(|&v| first[v]);
    for v in order {
        if regs[v].is_some() {
            continue;
        }
        let held: Vec<(usize, PReg)> = adj[v].iter().filter_map(|u| regs[u].map(|r| (u, r))).collect();
        let offered = m.allocatable(p.regs[v].class);
        let free = |r: &PReg| offered.contains(r) && !changed[v].contains(r) && held.iter().all(|(_, h)| h != r);
        let wanted = prefer[v].iter().filter_map(|&u| regs[u]).find(free);
        match wanted.or_else(|| offered.iter().copied().find(free)) {
            Some(r) => regs[v] = Some(r),
            None => return Err((v, held)),
        }
    }
    for (v, row) in adj.iter().enumerate() {
        for u in row.iter() {
            debug_assert!(
                regs[v] != regs[u],
                "{} and {} interfere and share {:?}",
                p.regs[v].name,
                p.regs[u].name,
                regs[v]
            );
        }
        debug_assert!(
            regs[v].is_none_or(|r| !changed[v].contains(&r)),
            "{} is in {:?}, which an instruction it is live across changes",
            p.regs[v].name,
            regs[v]
        );
    }
    Ok(regs)
}
