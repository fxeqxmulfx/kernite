//! Spilling: which variable goes to the stack when colouring runs out of registers, the
//! program that keeps it there, and which of its moves stay once every variable has a register.

use kernite_core::{
    Arg, Base, Clobber, Flow, Inst, Item, Mem, Operand, PReg, Program, RegInfo, Slot, SlotId, VReg, Width,
};

use crate::cfg::{self, Block, forward};
use crate::live::{Live, before, liveness};
use crate::{Error, Spill};

/// How wide each variable of `p` is kept, as `s` says for the widest operand that names it: by the width the operand
/// has, or all of its register where an address takes it, as a gather takes its indices, for an address does not say
/// how much of it is read.
pub(crate) fn kept<Op>(p: &Program<Op>, s: &dyn Spill<Op>) -> Vec<Width> {
    let mut named = vec![Width::W8; p.regs.len()];
    for item in &p.body {
        let Item::Inst(inst) = item else { continue };
        for arg in inst.operands() {
            for v in arg.regs() {
                let width = match arg.operand {
                    Operand::Reg(_, width) => width,
                    _ => s.width(v.class),
                };
                if let Some(at) = named.get_mut(v.id as usize) {
                    *at = (*at).max(width);
                }
            }
        }
    }
    p.regs.iter().zip(named).map(|(info, named)| s.kept(info.class, named)).collect()
}

/// Whether variable `v` of `p` may spill: every instruction that writes it goes on to the next
/// item, where its store goes.
pub(crate) fn may<Op>(p: &Program<Op>, v: usize) -> bool {
    p.body.iter().all(|item| match item {
        Item::Inst(inst) => inst.flow == Flow::Next || !inst.writes().any(|w| w.id as usize == v),
        Item::Label(_) | Item::Note(_) => true,
    })
}

/// Of `candidates`, the variable that costs least to keep on the stack for each instruction it
/// would hold a register across: the loads and stores spilling it adds, over the instructions
/// it is live after, plus one. One in a loop costs ten times one outside it, a loop being the
/// blocks from a label to the last jump back to it. Of two that cost the same, the one later
/// in [`Program::regs`]; none if there are no candidates.
pub(crate) fn choose<Op>(p: &Program<Op>, blocks: &[Block], live: &Live, candidates: &[usize]) -> Option<usize> {
    // For each block a jump goes back to, the last block that jumps there.
    let mut back: Vec<Option<usize>> = vec![None; blocks.len()];
    for (b, block) in blocks.iter().enumerate() {
        for &s in block.succs.iter().filter(|&&s| s <= b) {
            if let Some(last) = back.get_mut(s) {
                *last = Some(last.map_or(b, |e| e.max(b)));
            }
        }
    }
    let often: Vec<u128> = Vec::from_fn(blocks.len(), |b| {
        let loops = back.iter().enumerate().filter(|&(top, end)| top <= b && end.is_some_and(|e| b <= e)).count();
        10u128.saturating_pow(loops as u32)
    });
    // Each candidate, with what it costs and how many instructions it is live after. A parameter costs a store on
    // entry.
    let mut tally: Vec<(usize, u128, u128)> =
        candidates.iter().map(|&c| (c, u128::from(p.params.iter().any(|(v, _)| v.id as usize == c)), 0)).collect();
    for ((block, out), &often) in blocks.iter().zip(&live.outs).zip(&often) {
        let mut now = out.clone();
        for item in block.items_in(&p.body).iter().rev() {
            let Item::Inst(inst) = item else { continue };
            for (c, cost, size) in &mut tally {
                *size += u128::from(now.contains(*c));
                let moves = u128::from(inst.reads().any(|r| r.id as usize == *c))
                    + u128::from(inst.writes().any(|w| w.id as usize == *c));
                *cost = cost.saturating_add(often.saturating_mul(moves));
            }
            for w in inst.writes() {
                now.remove(w.id as usize);
            }
            for r in inst.reads() {
                now.insert(r.id as usize);
            }
        }
    }
    let weighs = |(_, cost, _): &(usize, u128, u128), (_, _, by): &(usize, u128, u128)| cost.saturating_mul(by + 1);
    tally.iter().min_by(|a, b| weighs(a, b).cmp(&weighs(b, a)).then(b.0.cmp(&a.0))).map(|&(c, ..)| c)
}

/// `p` with its variable `v` kept in a slot of its own, as wide as [`kept`] says, `s` storing it there after each
/// instruction that writes it and loading it before each one that reads it. Each instruction that names it names
/// instead a variable made for that instruction alone, of the same name, which the store and the load move. A parameter
/// is stored on entry, from the register it arrives in; a result is loaded into itself before each return that reads
/// it, in the register it leaves in. [`tidy`] leaves out the moves that the colouring makes needless.
pub(crate) fn rewrite<Op>(p: &mut Program<Op>, v: usize, s: &dyn Spill<Op>) {
    // A number that is no variable of the program has nothing to keep in a slot.
    let (Some(var), Some(&width)) = (p.vreg(v), kept(p, s).get(v)) else { return };
    let (class, name) = (var.class, p.var_name(var).to_string());
    let bytes = width.bytes();
    let slot = SlotId(p.slots.len() as u32);
    p.slots.push(Slot { name: name.clone(), size: bytes, align: bytes, spill: true });
    let result = p.results.iter().any(|&(r, _)| r == var);
    let mut body = Vec::with_capacity(p.body.len());
    if p.params.iter().any(|&(q, _)| q == var) {
        body.push(Item::Inst(s.store(slot, var, width)));
    }
    for item in std::mem::take(&mut p.body) {
        let inst = match item {
            Item::Inst(inst) if inst.operands().any(|a| a.regs().any(|r| r == var)) => inst,
            item => {
                body.push(item);
                continue;
            }
        };
        if result && inst.flow == Flow::Return {
            body.extend([Item::Inst(s.load(var, slot, width)), Item::Inst(inst)]);
            continue;
        }
        let t = VReg { id: p.regs.len() as u32, class };
        p.regs.push(RegInfo { class, name: name.clone() });
        let (reads, writes) = (inst.reads().any(|r| r == var), inst.writes().any(|w| w == var));
        if reads {
            body.push(Item::Inst(s.load(t, slot, width)));
        }
        body.push(Item::Inst(renamed(inst, |r| if r == var { t } else { r })));
        if writes {
            body.push(Item::Inst(s.store(slot, t, width)));
        }
    }
    p.body = body;
}

/// `inst` with each variable it names replaced by what `to` says of it.
pub(crate) fn renamed<Op>(inst: Inst<Op, VReg>, to: impl Fn(VReg) -> VReg) -> Inst<Op, VReg> {
    let rename = |a: Arg<VReg>| a.map(&to);
    Inst {
        op: inst.op,
        args: inst.args.into_iter().map(rename).collect(),
        implicit: inst.implicit.into_iter().map(rename).collect(),
        flow: inst.flow,
        clobbers: inst.clobbers,
    }
}

/// A variable's move between its register and a slot that keeps a spilled variable: `s`'s store of it to the slot, or
/// its load of it from there.
struct Move {
    slot: SlotId,
    var: VReg,
    width: Width,
    /// Whether it loads, and else it stores.
    load: bool,
}

/// `inst` as a [`Move`], if it is one: a move is known by its being equal to what `s` stores or loads.
fn moved<Op: PartialEq>(p: &Program<Op>, inst: &Inst<Op, VReg>, s: &dyn Spill<Op>) -> Option<Move> {
    let (slot, width) = inst.args.iter().find_map(|arg| match arg.operand {
        Operand::Mem(Mem { base: Base::Slot(slot), width, .. })
            if p.slots.get(slot.0 as usize).is_some_and(|slot| slot.spill) =>
        {
            Some((slot, width))
        }
        _ => None,
    })?;
    let mut named = inst.operands().flat_map(Arg::regs);
    let (Some(var), None) = (named.next(), named.next()) else { return None };
    let load = *inst == s.load(var, slot, width);
    (load || *inst == s.store(slot, var, width)).then_some(Move { slot, var, width, load })
}

/// What a register holds of a slot: the slot's content, in the variables that were stored there or loaded from there,
/// one on each way that comes to where the register holds it, sorted and no two alike.
#[derive(Clone, PartialEq)]
struct Mirror {
    slot: SlotId,
    by: Vec<VReg>,
}

/// The registers that hold what a slot keeps, each with the slot and the variables.
type Mirrors = Vec<(PReg, Mirror)>;

/// What becomes of the load of a spilled variable.
enum Load<Op> {
    /// It stays: no register holds what it loads.
    Stays,
    /// It is left out, and the instruction it is for reads, in place of the first, the variable of the register that
    /// holds what it loads: one of the second, which are made one.
    Reads(VReg, Vec<VReg>),
    /// It is this copy, from a register that holds what it loads, in one of these variables, which are made one.
    Becomes(Inst<Op, VReg>, Vec<VReg>),
}

/// `p`, once every variable has a register in `regs`, with the moves of spilled variables that are not needed left out:
/// loads where a register holds the variable, stores that no load follows before another store to the slot, and the
/// slots no move uses. A variable that no instruction names any more has no register in `regs`.
pub(crate) fn tidy<Op: PartialEq>(
    p: &mut Program<Op>,
    regs: &mut [Option<PReg>],
    s: &dyn Spill<Op>,
) -> Result<(), Error> {
    if p.slots.iter().any(|slot| slot.spill) {
        reload(p, regs, s)?;
        drop_dead_stores(p, s)?;
        drop_unused_slots(p, s);
    }
    Ok(())
}

/// `p` with each load of a spilled variable left out or made a copy where a register holds what it loads. A store puts
/// the slot's content in the register it stores from, and a load in the register it loads to, and the register holds it
/// until an instruction writes the register, a call changes the part of it that the slot keeps, or a store to the slot
/// comes. Where ways meet, at a label, a register holds it if each way brings it there. Where the ways bring it in
/// variables of their own, those are made one, a parameter if one of them is, as they take one register, so that the
/// one read after the label is written on every way. A load is for the instruction after it and after the loads
/// between. Where that instruction is the only other one that names the variable the load loads, it reads the variable
/// of the register instead, and the load is left out, unless another load for it writes that register, as each does
/// that is not left out. The variable of a load left out is named no more, and has no register in `regs`. That
/// instruction does not write the variable: the spill stores each variable an instruction writes, and the store would
/// name it too. Else the load is a copy, which the allocator leaves out where the two take one register.
fn reload<Op: PartialEq>(p: &mut Program<Op>, regs: &mut [Option<PReg>], s: &dyn Spill<Op>) -> Result<(), Error> {
    // A variable the body names has a register, and [`compile`](crate::compile) refuses one that has none: here it
    // holds nothing, and a load of it stays.
    let reg = |v: VReg| regs.get(v.id as usize).copied().flatten();
    // How many instructions name each variable.
    let mut named = vec![0; p.regs.len()];
    for item in &p.body {
        let Item::Inst(inst) = item else { continue };
        let mut vars: Vec<VReg> = inst.operands().flat_map(Arg::regs).collect();
        vars.sort();
        vars.dedup();
        for v in vars {
            if let Some(count) = named.get_mut(v.id as usize) {
                *count += 1;
            }
        }
    }
    // The load of a spilled variable that an item is, if it is one.
    let loads = |item: &Item<Op, VReg>| match item {
        Item::Inst(inst) => moved(p, inst, s).filter(|m| m.load),
        _ => None,
    };
    // What becomes of the load `m` for `next`, where the loads for `next` that are not left out write `written`, with
    // `held` made what the registers hold after it.
    let fate = |m: &Move, next: Option<&Inst<Op, VReg>>, written: &[PReg], held: &mut Mirrors| {
        let Some(at) = reg(m.var) else { return Load::Stays };
        let mut holders = held.iter().filter(|(_, h)| h.slot == m.slot);
        let from = holders.clone().find(|(r, _)| *r == at).or_else(|| holders.next()).cloned();
        let once = named.get(m.var.id as usize) == Some(&2) && !p.results.iter().any(|&(r, _)| r == m.var);
        if let Some((held_at, h)) = &from
            && once
            && next.is_some_and(|next| next.reads().any(|v| v == m.var))
            && !written.contains(held_at)
        {
            return Load::Reads(m.var, h.by.clone());
        }
        held.retain(|(r, _)| *r != at);
        held.push((at, Mirror { slot: m.slot, by: vec![m.var] }));
        let Some((_, h)) = from else { return Load::Stays };
        // A mirror has the variable that made it: one that has none is a mirror of nothing, and the load stays.
        let Some(&by) = h.by.first() else { return Load::Stays };
        Load::Becomes(s.copy(m.var, by, m.width), h.by)
    };
    let step = |i: usize, inst: &Inst<Op, VReg>, held: &mut Mirrors| {
        let Some(Move { slot, var, load, .. }) = moved(p, inst, s) else {
            for r in inst.writes().filter_map(reg) {
                held.retain(|(at, _)| *at != r);
            }
            for &clobber in &inst.clobbers {
                match clobber {
                    Clobber::All(r) => held.retain(|(at, _)| *at != r),
                    // The lower half stays, and holds all that a slot of less than the register keeps.
                    Clobber::Upper(r) => {
                        held.retain(|(at, m)| {
                            *at != r
                                || p.slots
                                    .get(m.slot.0 as usize)
                                    .is_some_and(|slot| slot.size < s.width(r.class).bytes())
                        });
                    }
                }
            }
            return Load::Stays;
        };
        if !load {
            held.retain(|(_, m)| m.slot != slot);
            if let Some(at) = reg(var) {
                held.push((at, Mirror { slot, by: vec![var] }));
            }
            return Load::Stays;
        }
        // This load and the loads after it are for the instruction after them. The registers that those not left out
        // write are found by running the loads with none written, then with those the run before wrote, until a run
        // writes the same: no run writes fewer than the one before.
        let group: Vec<Move> = p.body.get(i..).unwrap_or_default().iter().map_while(loads).collect();
        let next = match p.body.get(i + group.len()) {
            Some(Item::Inst(next)) => Some(next),
            _ => None,
        };
        let mut written = Vec::new();
        loop {
            let mut now = held.clone();
            let wrote: Vec<PReg> = group
                .iter()
                .filter(|m| !matches!(fate(m, next, &written, &mut now), Load::Reads(..)))
                .filter_map(|m| reg(m.var))
                .collect();
            if wrote == written {
                break group.first().map_or(Load::Stays, |first| fate(first, next, &written, held));
            }
            written = wrote;
        }
    };
    let meet = |a: &Mirrors, b: &Mirrors, _: bool| -> Mirrors {
        let both = |(at, m): &(PReg, Mirror)| {
            let (_, o) = b.iter().find(|(r, o)| r == at && o.slot == m.slot)?;
            let mut by = [&m.by[..], &o.by[..]].concat();
            by.sort();
            by.dedup();
            Some((*at, Mirror { slot: m.slot, by }))
        };
        a.iter().filter_map(both).collect()
    };
    let fates = forward(p, Vec::new(), meet, step, |_, _, _| {})?;
    // The variables made one: those in which the ways to a load bring the slot's content to one register, and the
    // variable of each load left out, with the one read in place of it. Each is renamed to the first of its set by
    // `rank`: the parameter, if one is in it, and else the one of the lowest number.
    let mut one: Vec<usize> = (0..p.regs.len()).collect();
    let root = |one: &[usize], mut v: usize| {
        while let Some(&up) = one.get(v)
            && up != v
        {
            v = up;
        }
        v
    };
    let rank = |v: usize| (!p.params.iter().any(|(q, _)| q.id as usize == v), v);
    for fate in fates.iter().flatten() {
        let (Load::Reads(_, by) | Load::Becomes(_, by)) = fate else { continue };
        for (a, b) in by.iter().zip(by.iter().skip(1)) {
            let (a, b) = (root(&one, a.id as usize), root(&one, b.id as usize));
            let (low, high) = if rank(a) <= rank(b) { (a, b) } else { (b, a) };
            if let Some(up) = one.get_mut(high) {
                *up = low;
            }
        }
    }
    for fate in fates.iter().flatten() {
        if let Load::Reads(var, by) = fate
            && let Some(first) = by.first()
            && let Some(up) = one.get_mut(var.id as usize)
        {
            *up = first.id as usize;
        }
    }
    let to: Vec<Option<VReg>> = (0..one.len()).map(|v| p.vreg(root(&one, v))).collect();
    for (v, r) in regs.iter_mut().enumerate() {
        if root(&one, v) != v {
            *r = None;
        }
    }
    let rename = |inst| renamed(inst, |v| to.get(v.id as usize).copied().flatten().unwrap_or(v));
    p.body = std::mem::take(&mut p.body)
        .into_iter()
        .zip(fates)
        .filter_map(|(item, fate)| match (item, fate) {
            (_, Some(Load::Reads(..))) => None,
            (_, Some(Load::Becomes(copy, _))) => {
                // Made one, the two are one variable, and the copy moves it to where it is.
                let copy = rename(copy);
                let moves = copy.writes().next() != copy.reads().next();
                moves.then_some(Item::Inst(copy))
            }
            (Item::Inst(inst), _) => Some(Item::Inst(rename(inst))),
            (item, _) => Some(item),
        })
        .collect();
    Ok(())
}

/// `p` without the stores of a spilled variable that no load follows before another store to its slot, on any path.
fn drop_dead_stores<Op: PartialEq>(p: &mut Program<Op>, s: &dyn Spill<Op>) -> Result<(), Error> {
    let blocks = cfg::blocks(&p.body)?;
    let touches = |inst: &Inst<Op, VReg>| match moved(p, inst, s) {
        Some(Move { slot, load: true, .. }) => (vec![slot.0 as usize], vec![]),
        Some(Move { slot, .. }) => (vec![], vec![slot.0 as usize]),
        None => (vec![], vec![]),
    };
    let live = liveness(&p.body, &blocks, p.slots.len(), &touches);
    let mut dead = vec![false; p.body.len()];
    for (block, out) in blocks.iter().zip(&live.outs) {
        let mut now = out.clone();
        for i in block.items.clone().rev() {
            let Some(Item::Inst(inst)) = p.body.get(i) else { continue };
            let (reads, writes) = touches(inst);
            if let Some(dead) = dead.get_mut(i) {
                *dead = writes.iter().any(|&slot| !now.contains(slot));
            }
            before(&mut now, &reads, &writes);
        }
    }
    let mut dead = dead.into_iter();
    p.body.retain(|_| dead.next() != Some(true));
    Ok(())
}

/// `p` without the slots of a spilled variable that no move uses, and with the moves that stay numbered again. The
/// program's own slots come first, for a spill adds its slots after them, and all stay, so that an instruction that
/// names one keeps its number: only a move names a slot that a spill adds.
fn drop_unused_slots<Op: PartialEq>(p: &mut Program<Op>, s: &dyn Spill<Op>) {
    let mut used = vec![false; p.slots.len()];
    for item in &p.body {
        if let Item::Inst(inst) = item
            && let Some(m) = moved(p, inst, s)
            && let Some(at) = used.get_mut(m.slot.0 as usize)
        {
            *at = true;
        }
    }
    let keep: Vec<bool> = p.slots.iter().zip(&used).map(|(slot, &used)| used || !slot.spill).collect();
    let number: Vec<SlotId> = keep
        .iter()
        .scan(0, |next, &keep| {
            *next += u32::from(keep);
            Some(SlotId(*next - u32::from(keep)))
        })
        .collect();
    let body = std::mem::take(&mut p.body);
    p.body = body
        .into_iter()
        .map(|item| match item {
            Item::Inst(inst) => {
                let numbered = moved(p, &inst, s).and_then(|m| {
                    let slot = *number.get(m.slot.0 as usize)?;
                    Some(if m.load { s.load(m.var, slot, m.width) } else { s.store(slot, m.var, m.width) })
                });
                Item::Inst(numbered.unwrap_or(inst))
            }
            item => item,
        })
        .collect();
    let mut keep = keep.into_iter();
    p.slots.retain(|_| keep.next() == Some(true));
}
