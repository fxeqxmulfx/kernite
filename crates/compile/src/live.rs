//! Liveness: which variables are live where, and which interfere.

use kgen_core::{Arg, Item, PReg, Program, VReg};

use crate::Error;
use crate::bits::Bits;
use crate::cfg::{self, Block};

/// Which variables are live where, block by block: at the top of each and at the bottom.
pub(crate) struct Live {
    ins: Vec<Bits>,
    pub(crate) outs: Vec<Bits>,
}

/// The body's blocks and what is live at their edges, once control is known to stay inside
/// the body and every variable but the parameters to be written before any path reads it.
pub(crate) fn analyse<Op>(p: &Program<Op>) -> Result<(Vec<Block>, Live), Error> {
    let blocks = cfg::blocks(&p.body)?;
    let live = liveness(p, &blocks);
    let params: Vec<usize> = p.params.iter().map(|(v, _)| v.id as usize).collect();
    if let Some(i) = live.ins[0].iter().find(|i| !params.contains(i)) {
        return Err(Error::ReadBeforeWrite { var: p.regs[i].name.clone(), vreg: p.vreg(i) });
    }
    Ok((blocks, live))
}

/// Solves liveness backwards to a fixed point. A block's reads of variables it has not
/// written yet make them live at its top; its writes end the liveness that flows up from
/// below.
fn liveness<Op>(p: &Program<Op>, blocks: &[Block]) -> Live {
    let n = p.regs.len();
    let (mut uses, mut defs) = (vec![Bits::new(n); blocks.len()], vec![Bits::new(n); blocks.len()]);
    for (b, block) in blocks.iter().enumerate() {
        for item in &p.body[block.items.clone()] {
            let Item::Inst(inst) = item else { continue };
            for r in inst.reads() {
                if !defs[b].contains(r.id as usize) {
                    uses[b].insert(r.id as usize);
                }
            }
            for w in inst.writes() {
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

/// The interference graph, as a set of neighbours per variable: every variable an
/// instruction writes against every other of its file live just after it, the ones written
/// beside it included. And for each variable, the registers of its file that an instruction it
/// is live across changes besides its operands, sorted: it is live after the instruction, which
/// does not write it.
pub(crate) fn interference<Op>(p: &Program<Op>, blocks: &[Block], live: &Live) -> (Vec<Bits>, Vec<Vec<PReg>>) {
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
        let mut now = live.outs[b].clone();
        for item in p.body[block.items.clone()].iter().rev() {
            let Item::Inst(inst) = item else { continue };
            let writes: Vec<VReg> = inst.writes().collect();
            if !inst.clobbers.is_empty() {
                for l in now.iter().filter(|&l| writes.iter().all(|w| w.id as usize != l)) {
                    let class = p.regs[l].class;
                    changed[l].extend(inst.clobbers.iter().filter(|r| r.class == class));
                }
            }
            for w in &writes {
                for l in now.iter() {
                    edge(w.id as usize, l);
                }
                for other in &writes {
                    edge(w.id as usize, other.id as usize);
                }
            }
            for w in &writes {
                now.remove(w.id as usize);
            }
            for r in inst.reads() {
                now.insert(r.id as usize);
            }
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
