//! Liveness: which variables are live where, and which interfere.

use kernite_core::{Arg, Clobber, Inst, Item, Operand, PReg, Program, VReg, Width};

use crate::bits::Bits;
use crate::cfg::{self, Block};
use crate::{Error, Spill};

/// Which variables are live where, block by block: at the top of each and at the bottom.
pub(crate) struct Live {
    ins: Vec<Bits>,
    pub(crate) outs: Vec<Bits>,
}

/// What an instruction touches: the numbers of what it reads, and of what it writes: variables, or slots.
pub(crate) type Touches = (Vec<usize>, Vec<usize>);

/// The body's blocks and what is live at their edges, once control is known to stay inside
/// the body and every variable but the parameters to be written before any path reads it.
pub(crate) fn analyse<Op>(p: &Program<Op>) -> Result<(Vec<Block>, Live), Error> {
    let blocks = cfg::blocks(&p.body)?;
    let live = liveness(&p.body, &blocks, p.regs.len(), &touched);
    let params: Vec<usize> = p.params.iter().map(|(v, _)| v.id as usize).collect();
    let read = live.ins.first().and_then(|top| top.iter().find(|i| !params.contains(i)));
    if let Some(vreg) = read.and_then(|i| p.vreg(i)) {
        return Err(Error::ReadBeforeWrite { var: p.var_name(vreg).to_string(), vreg });
    }
    Ok((blocks, live))
}

/// Which variables are live where, as far as their upper halves go: a variable is live until an instruction writes it,
/// if one after reads its upper half. See [`upper_touched`].
pub(crate) fn upper_live<Op>(p: &Program<Op>, blocks: &[Block], s: &dyn Spill<Op>) -> Live {
    liveness(&p.body, blocks, p.regs.len(), &|inst| upper_touched(inst, s))
}

/// The variables whose upper halves `inst` reads, and those it writes. It reads the upper half of a variable where it
/// reads all of its register, as `s` says how wide that is, and where it reads the registers of an address. It replaces
/// the upper half of each it writes, though it writes part of it, as [`Clobber::Upper`] says.
pub(crate) fn upper_touched<Op>(inst: &Inst<Op, VReg>, s: &dyn Spill<Op>) -> Touches {
    let part = |arg: &&Arg<VReg>| matches!(arg.operand, Operand::Reg(v, width) if width < s.width(v.class));
    let reads = inst.operands().filter(|arg| !part(arg)).flat_map(Arg::reads);
    (reads.map(|v| v.id as usize).collect(), inst.writes().map(|v| v.id as usize).collect())
}

/// The variables `inst` reads and the variables it writes.
pub(crate) fn touched<Op>(inst: &Inst<Op, VReg>) -> Touches {
    (inst.reads().map(|v| v.id as usize).collect(), inst.writes().map(|v| v.id as usize).collect())
}

/// `now`, the variables live just after an instruction that reads `reads` and writes `writes`, as it is just before it:
/// the ones it writes are not live, and the ones it reads are.
pub(crate) fn before(now: &mut Bits, reads: &[usize], writes: &[usize]) {
    for &w in writes {
        now.remove(w);
    }
    for &r in reads {
        now.insert(r);
    }
}

/// Solves liveness backwards to a fixed point, of the `n` things that `touches` says an instruction reads and writes:
/// the variables, or the slots. A block's reads of what it has not written yet make them live at its top; its writes
/// end the liveness that flows up from below.
pub(crate) fn liveness<Op>(
    body: &[Item<Op, VReg>],
    blocks: &[Block],
    n: usize,
    touches: &dyn Fn(&Inst<Op, VReg>) -> Touches,
) -> Live {
    // What each block reads before it writes it, and what it writes.
    let (uses, defs): (Vec<Bits>, Vec<Bits>) = blocks
        .iter()
        .map(|block| {
            let (mut uses, mut defs) = (Bits::new(n), Bits::new(n));
            for item in block.items_in(body) {
                let Item::Inst(inst) = item else { continue };
                let (reads, writes) = touches(inst);
                for r in reads {
                    if !defs.contains(r) {
                        uses.insert(r);
                    }
                }
                for w in writes {
                    defs.insert(w);
                }
            }
            (uses, defs)
        })
        .unzip();
    let mut live = Live { ins: uses.clone(), outs: vec![Bits::new(n); blocks.len()] };
    loop {
        let mut changed = false;
        for (b, ((block, uses), defs)) in blocks.iter().zip(&uses).zip(&defs).enumerate().rev() {
            let mut out = Bits::new(n);
            for to in block.succs.iter().filter_map(|&s| live.ins.get(s)) {
                out.union_with(to);
            }
            let mut ins = out.clone();
            ins.subtract(defs);
            ins.union_with(uses);
            if let (Some(was_out), Some(was_in)) = (live.outs.get_mut(b), live.ins.get_mut(b))
                && (out != *was_out || ins != *was_in)
            {
                (*was_out, *was_in) = (out, ins);
                changed = true;
            }
        }
        if !changed {
            return live;
        }
    }
}

/// The variables `inst` copies, `to` from `from`, and how much, if it is what `s`'s copy makes of the one to the other.
/// That is a copy of as much as `to` is kept, as `kept` says. Where `from` is a call's result, it is a copy of as much
/// as `from` is kept, what the call gives back, though `to` may be kept in more.
pub(crate) fn copied<Op: PartialEq>(
    p: &Program<Op>,
    inst: &Inst<Op, VReg>,
    kept: &[Width],
    s: &dyn Spill<Op>,
) -> Option<(VReg, VReg, Width)> {
    let (writes, reads): (Vec<VReg>, Vec<VReg>) = (inst.writes().collect(), inst.reads().collect());
    let ([to], [from]) = (writes.as_slice(), reads.as_slice()) else { return None };
    let of = if p.pinned.iter().any(|&(v, _)| v == *from) { from } else { to };
    let width = *kept.get(of.id as usize)?;
    let copy = to != from && to.class == from.class && *inst == s.copy(*to, *from, width);
    copy.then_some((*to, *from, width))
}

/// The interference graph, as a set of neighbours per variable: every variable an instruction writes against every
/// other of its file live just after it, the ones written beside it included. But the copy of a variable to a call's
/// argument makes no edge between the two, for they hold one value: they interfere where another instruction writes one
/// while the other is live. And for each variable, the registers of its file that an instruction it is live across
/// changes besides its operands, sorted: it is live after the instruction, which does not write it. One that clears
/// only the upper half of a register changes it for a variable whose upper half is live after it, as `upper` says.
pub(crate) fn interference<Op: PartialEq>(
    p: &Program<Op>,
    blocks: &[Block],
    live: &Live,
    upper: &Live,
    kept: &[Width],
    s: &dyn Spill<Op>,
) -> (Vec<Bits>, Vec<Vec<PReg>>) {
    let n = p.regs.len();
    let mut adj = vec![Bits::new(n); n];
    let mut changed: Vec<Vec<PReg>> = vec![vec![]; n];
    let class = |v: usize| p.regs.get(v).map(|info| info.class);
    let mut edge = |a: usize, b: usize| {
        if a != b && class(a) == class(b) {
            for (x, y) in [(a, b), (b, a)] {
                if let Some(near) = adj.get_mut(x) {
                    near.insert(y);
                }
            }
        }
    };
    for ((block, out), high_out) in blocks.iter().zip(&live.outs).zip(&upper.outs) {
        let (mut now, mut high) = (out.clone(), high_out.clone());
        for item in block.items_in(&p.body).iter().rev() {
            let Item::Inst(inst) = item else { continue };
            let (reads, writes) = touched(inst);
            // The variable that the copy to a call's argument reads, if `inst` is one: it writes the argument alone.
            let passed = copied(p, inst, kept, s)
                .filter(|&(to, _, _)| p.pinned.iter().any(|&(v, _)| v == to))
                .map(|(_, from, _)| from.id as usize);
            for clobber in &inst.clobbers {
                let (reg, alive) = match *clobber {
                    Clobber::All(reg) => (reg, &now),
                    Clobber::Upper(reg) => (reg, &high),
                };
                for l in alive.iter().filter(|&l| class(l) == Some(reg.class)) {
                    if !writes.contains(&l)
                        && let Some(regs) = changed.get_mut(l)
                    {
                        regs.push(reg);
                    }
                }
            }
            for &w in &writes {
                for l in now.iter().filter(|&l| passed != Some(l)) {
                    edge(w, l);
                }
                for &other in &writes {
                    edge(w, other);
                }
            }
            before(&mut now, &reads, &writes);
            let (reads, writes) = upper_touched(inst, s);
            before(&mut high, &reads, &writes);
        }
    }
    for regs in &mut changed {
        regs.sort();
        regs.dedup();
    }
    (adj, changed)
}

/// The index of the item that first names each variable, reading or writing.
pub(crate) fn first_mentions<Op>(p: &Program<Op>) -> Vec<Option<usize>> {
    let mut first = vec![None; p.regs.len()];
    for (i, item) in p.body.iter().enumerate() {
        let Item::Inst(inst) = item else { continue };
        for v in inst.operands().flat_map(Arg::regs) {
            if let Some(at) = first.get_mut(v.id as usize) {
                at.get_or_insert(i);
            }
        }
    }
    first
}
