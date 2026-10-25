use kgen_core::{Arg, Item, Machine, PReg, Program, VReg};

use crate::Error;
use crate::bits::Bits;
use crate::cfg::{self, Block};

/// Which variables are live where, block by block: at the top of each and at the bottom.
pub(crate) struct Live {
    ins: Vec<Bits>,
    outs: Vec<Bits>,
}

/// Each variable's register, by [`VReg::id`]; `None` for a variable the body never names.
///
/// Liveness is solved over the blocks to a fixed point; two variables interfere when one is
/// written while the other is live; then each variable, in the order the body first names it
/// and after the ones the calling convention places, takes the first register the machine
/// offers that none of the variables it interferes with holds. No choice depends on anything
/// but the program, so the same program always gets the same registers. Nothing spills: a
/// variable left without a register is an [`Error::OutOfRegisters`], for the program to fix.
pub(crate) fn allocate<Op>(p: &Program<Op>, m: &dyn Machine) -> Result<Vec<Option<PReg>>, Error> {
    let (blocks, live) = analyse(p)?;
    let name = |i: usize| p.regs[i].name.clone();
    let adj = interference(p, &blocks, &live);

    let fixed = || p.params.iter().chain(&p.results);
    let mut regs: Vec<Option<PReg>> = vec![None; p.regs.len()];
    for &(v, r) in fixed() {
        match regs[v.id as usize] {
            Some(other) if other != r => return Err(Error::FixedTwice { var: name(v.id as usize) }),
            _ => regs[v.id as usize] = Some(r),
        }
    }
    for &(v, r) in fixed() {
        if let Some(u) = adj[v.id as usize].iter().find(|&u| regs[u] == Some(r)) {
            return Err(Error::FixedClash { a: name(v.id as usize), b: name(u), reg: r });
        }
    }
    let first = first_mentions(p);
    let mut order: Vec<usize> = (0..p.regs.len()).filter(|&v| first[v].is_some()).collect();
    order.sort_by_key(|&v| first[v]);
    for v in order {
        if regs[v].is_some() {
            continue;
        }
        let class = p.regs[v].class;
        let held: Vec<(usize, PReg)> = adj[v].iter().filter_map(|u| regs[u].map(|r| (u, r))).collect();
        let offered = m.allocatable(class);
        match offered.iter().find(|r| held.iter().all(|(_, h)| h != *r)) {
            Some(&r) => regs[v] = Some(r),
            None => {
                let near = p.body[..first[v].expect("mentioned")].iter().rev().find_map(|item| match item {
                    Item::Note(note) => Some(note.clone()),
                    _ => None,
                });
                return Err(Error::OutOfRegisters {
                    class,
                    var: name(v),
                    near,
                    taken: held.into_iter().map(|(u, r)| (name(u), r)).collect(),
                    have: offered.len(),
                });
            }
        }
    }
    for (v, row) in adj.iter().enumerate() {
        for u in row.iter() {
            debug_assert!(regs[v] != regs[u], "{} and {} interfere and share {:?}", name(v), name(u), regs[v]);
        }
    }
    Ok(regs)
}

/// The body's blocks and what is live at their edges, once control is known to stay inside
/// the body and every variable but the parameters to be written before any path reads it.
pub(crate) fn analyse<Op>(p: &Program<Op>) -> Result<(Vec<Block>, Live), Error> {
    let blocks = cfg::blocks(&p.body)?;
    let live = liveness(p, &blocks);
    let params: Vec<usize> = p.params.iter().map(|(v, _)| v.id as usize).collect();
    if let Some(i) = live.ins[0].iter().find(|i| !params.contains(i)) {
        return Err(Error::ReadBeforeWrite { var: p.regs[i].name.clone() });
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
/// beside it included.
fn interference<Op>(p: &Program<Op>, blocks: &[Block], live: &Live) -> Vec<Bits> {
    let n = p.regs.len();
    let mut adj = vec![Bits::new(n); n];
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
    adj
}

/// The index of the item that first names each variable, reading or writing.
fn first_mentions<Op>(p: &Program<Op>) -> Vec<Option<usize>> {
    let mut first = vec![None; p.regs.len()];
    for (i, item) in p.body.iter().enumerate() {
        let Item::Inst(inst) = item else { continue };
        for v in inst.operands().flat_map(Arg::regs) {
            first[v.id as usize].get_or_insert(i);
        }
    }
    first
}
