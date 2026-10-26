use kgen_core::{Arg, Item, Machine, PReg, Program, Spill, VReg};

use crate::Error;
use crate::bits::Bits;
use crate::cfg::{self, Block};
use crate::spill;

/// Which variables are live where, block by block: at the top of each and at the bottom.
pub(crate) struct Live {
    ins: Vec<Bits>,
    pub(crate) outs: Vec<Bits>,
}

/// `p` as allocated, and each of its variables' registers, by [`VReg::id`]; `None` for a
/// variable the body never names.
///
/// Liveness is solved over the blocks to a fixed point; two variables interfere when one is
/// written while the other is live; then each variable, in the order the body first names it
/// and after the ones the calling convention places, takes the first register the machine
/// offers that none of the variables it interferes with holds.
///
/// A variable left without one spills, or one of those holding the registers it could have
/// does, whichever [`spill::choose`] finds cheapest to keep on the stack; [`spill::rewrite`]
/// keeps it in a slot of its own, and the program is allocated again, until every variable has
/// a register. Each of the program's variables spills once at most, and none that spilling
/// made. No choice depends on anything but the program, so the same program always gets the
/// same registers and the same spills.
pub(crate) fn allocate<Op: Clone>(
    p: &Program<Op>,
    m: &dyn Machine,
    s: &dyn Spill<Op>,
) -> Result<(Program<Op>, Vec<Option<PReg>>), Error> {
    let mut p = p.clone();
    // Which variables may still spill: the program's own, until they do.
    let mut may = vec![true; p.regs.len()];
    loop {
        let (blocks, live) = analyse(&p)?;
        let adj = interference(&p, &blocks, &live);
        let (v, held) = match colour(&p, m, &adj, fixed(&p, &adj)?) {
            Ok(regs) => return Ok((p, regs)),
            Err(short) => short,
        };
        let class = p.regs[v].class;
        let have = m.allocatable(class).len();
        let candidates: Vec<usize> = std::iter::once(v)
            .chain(held.iter().map(|&(u, _)| u))
            .filter(|&u| have > 0 && may.get(u) == Some(&true) && spill::may(&p, u))
            .collect();
        let Some(spilled) = spill::choose(&p, &blocks, &live, &candidates) else {
            let name = |i: usize| p.regs[i].name.clone();
            let first = first_mentions(&p)[v].expect("a variable coloured is named");
            let near = p.body[..first].iter().rev().find_map(|item| match item {
                Item::Note(note) => Some(note.clone()),
                _ => None,
            });
            let taken = held.into_iter().map(|(u, r)| (name(u), r)).collect();
            return Err(Error::OutOfRegisters { class, var: name(v), near, taken, have });
        };
        may[spilled] = false;
        spill::rewrite(&mut p, spilled, s);
    }
}

/// The registers of the variables the calling convention places, the parameters and the
/// results, once it is known that it places none twice, nor two in one register while both
/// are live.
fn fixed<Op>(p: &Program<Op>, adj: &[Bits]) -> Result<Vec<Option<PReg>>, Error> {
    let name = |i: usize| p.regs[i].name.clone();
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
    Ok(regs)
}

/// A variable colouring left without a register, and the variables beside it that hold one,
/// each with the register it holds.
type Short = (usize, Vec<(usize, PReg)>);

/// `regs`, the placed variables' registers, with one for every other variable the body names;
/// or the first variable left without one.
fn colour<Op>(
    p: &Program<Op>,
    m: &dyn Machine,
    adj: &[Bits],
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
        match m.allocatable(p.regs[v].class).iter().find(|r| held.iter().all(|(_, h)| h != *r)) {
            Some(&r) => regs[v] = Some(r),
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
