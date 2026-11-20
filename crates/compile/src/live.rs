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

/// What an instruction touches: the variables it reads, and the variables it writes.
type Touches = (Vec<VReg>, Vec<VReg>);

/// The body's blocks and what is live at their edges, once control is known to stay inside
/// the body and every variable but the parameters to be written before any path reads it.
pub(crate) fn analyse<Op>(p: &Program<Op>) -> Result<(Vec<Block>, Live), Error> {
    let blocks = cfg::blocks(&p.body)?;
    let live = liveness(p, &blocks, &touched);
    let params: Vec<usize> = p.params.iter().map(|(v, _)| v.id as usize).collect();
    if let Some(i) = live.ins[0].iter().find(|i| !params.contains(i)) {
        return Err(Error::ReadBeforeWrite { var: p.regs[i].name.clone(), vreg: p.vreg(i) });
    }
    Ok((blocks, live))
}

/// Which variables are live where, as far as their upper halves go: a variable is live until an instruction writes it,
/// if one after reads its upper half. See [`upper_touched`].
pub(crate) fn upper_live<Op>(p: &Program<Op>, blocks: &[Block], s: &dyn Spill<Op>) -> Live {
    liveness(p, blocks, &|inst| upper_touched(inst, s))
}

/// The variables whose upper halves `inst` reads, and those it writes. It reads the upper half of a variable where it
/// reads all of its register, as `s` says how wide that is, and where it reads the registers of an address. It replaces
/// the upper half of each it writes, though it writes part of it, as [`Clobber::Upper`] says.
pub(crate) fn upper_touched<Op>(inst: &Inst<Op, VReg>, s: &dyn Spill<Op>) -> Touches {
    let part = |arg: &&Arg<VReg>| matches!(arg.operand, Operand::Reg(v, width) if width < s.width(v.class));
    (inst.operands().filter(|arg| !part(arg)).flat_map(Arg::reads).collect(), inst.writes().collect())
}

/// The variables `inst` reads and the variables it writes.
pub(crate) fn touched<Op>(inst: &Inst<Op, VReg>) -> Touches {
    (inst.reads().collect(), inst.writes().collect())
}

/// `now`, the variables live just after an instruction that reads `reads` and writes `writes`, as it is just before it:
/// the ones it writes are not live, and the ones it reads are.
pub(crate) fn before(now: &mut Bits, reads: &[VReg], writes: &[VReg]) {
    for w in writes {
        now.remove(w.id as usize);
    }
    for r in reads {
        now.insert(r.id as usize);
    }
}

/// Solves liveness backwards to a fixed point, as `touches` says what an instruction reads and writes. A block's reads
/// of variables it has not written yet make them live at its top; its writes end the liveness that flows up from below.
fn liveness<Op>(p: &Program<Op>, blocks: &[Block], touches: &dyn Fn(&Inst<Op, VReg>) -> Touches) -> Live {
    let n = p.regs.len();
    let (mut uses, mut defs) = (vec![Bits::new(n); blocks.len()], vec![Bits::new(n); blocks.len()]);
    for (b, block) in blocks.iter().enumerate() {
        for item in &p.body[block.items.clone()] {
            let Item::Inst(inst) = item else { continue };
            let (reads, writes) = touches(inst);
            for r in reads {
                if !defs[b].contains(r.id as usize) {
                    uses[b].insert(r.id as usize);
                }
            }
            for w in writes {
                defs[b].insert(w.id as usize);
            }
        }
    }
    let mut live = Live { ins: uses.clone(), outs: vec![Bits::new(n); blocks.len()] };
    loop {
        let mut changed = false;
        for b in (0..blocks.len()).rev() {
            let mut out = Bits::new(n);
            for &s in &blocks[b].succs {
                out.union_with(&live.ins[s]);
            }
            let mut ins = out.clone();
            ins.subtract(&defs[b]);
            ins.union_with(&uses[b]);
            if out != live.outs[b] || ins != live.ins[b] {
                (live.outs[b], live.ins[b]) = (out, ins);
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
    let width = kept[of.id as usize];
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
    let mut edge = |a: usize, b: usize| {
        if a != b && p.regs[a].class == p.regs[b].class {
            adj[a].insert(b);
            adj[b].insert(a);
        }
    };
    for (b, block) in blocks.iter().enumerate() {
        let (mut now, mut high) = (live.outs[b].clone(), upper.outs[b].clone());
        for item in p.body[block.items.clone()].iter().rev() {
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
                for l in alive.iter().filter(|&l| p.regs[l].class == reg.class) {
                    if writes.iter().all(|w| w.id as usize != l) {
                        changed[l].push(reg);
                    }
                }
            }
            for w in &writes {
                for l in now.iter().filter(|&l| passed != Some(l)) {
                    edge(w.id as usize, l);
                }
                for other in &writes {
                    edge(w.id as usize, other.id as usize);
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
            first[v.id as usize].get_or_insert(i);
        }
    }
    first
}
