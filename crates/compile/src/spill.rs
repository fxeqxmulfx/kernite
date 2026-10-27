//! Spilling: which variable goes to the stack when colouring runs out of registers, and the
//! program that keeps it there.

use kgen_core::{Arg, Flow, Inst, Item, Program, RegInfo, Slot, SlotId, VReg};

use crate::Spill;
use crate::cfg::Block;
use crate::live::Live;

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
            back[s] = Some(back[s].map_or(b, |e| e.max(b)));
        }
    }
    let often: Vec<u128> = (0..blocks.len())
        .map(|b| {
            let loops = back.iter().enumerate().filter(|&(top, end)| top <= b && end.is_some_and(|e| b <= e)).count();
            10u128.saturating_pow(loops as u32)
        })
        .collect();
    let (mut cost, mut size) = (vec![0u128; candidates.len()], vec![0u128; candidates.len()]);
    for (k, &c) in candidates.iter().enumerate() {
        // A parameter's store on entry.
        cost[k] = u128::from(p.params.iter().any(|(v, _)| v.id as usize == c));
    }
    for (b, block) in blocks.iter().enumerate() {
        let mut now = live.outs[b].clone();
        for item in p.body[block.items.clone()].iter().rev() {
            let Item::Inst(inst) = item else { continue };
            for (k, &c) in candidates.iter().enumerate() {
                size[k] += u128::from(now.contains(c));
                let moves = u128::from(inst.reads().any(|r| r.id as usize == c))
                    + u128::from(inst.writes().any(|w| w.id as usize == c));
                cost[k] = cost[k].saturating_add(often[b].saturating_mul(moves));
            }
            for w in inst.writes() {
                now.remove(w.id as usize);
            }
            for r in inst.reads() {
                now.insert(r.id as usize);
            }
        }
    }
    let weighs = |k: usize, by: usize| cost[k].saturating_mul(size[by] + 1);
    (0..candidates.len())
        .min_by(|&a, &b| weighs(a, b).cmp(&weighs(b, a)).then(candidates[b].cmp(&candidates[a])))
        .map(|k| candidates[k])
}

/// `p` with its variable `v` kept in a slot of its own, `s` storing it there after each
/// instruction that writes it and loading it before each one that reads it. Each instruction
/// that names it names instead a variable made for that instruction alone, of the same name,
/// which the store and the load move. A parameter is stored on entry, from the register it
/// arrives in; a result is loaded into itself before each return that reads it, in the register
/// it leaves in.
pub(crate) fn rewrite<Op>(p: &mut Program<Op>, v: usize, s: &dyn Spill<Op>) {
    let RegInfo { class, name } = p.regs[v].clone();
    let var = VReg { id: v as u32, class };
    let bytes = s.width(class).bytes();
    let slot = SlotId(p.slots.len() as u32);
    p.slots.push(Slot { name: name.clone(), size: bytes, align: bytes, spill: true });
    let result = p.results.iter().any(|&(r, _)| r == var);
    let mut body = Vec::with_capacity(p.body.len());
    if p.params.iter().any(|&(q, _)| q == var) {
        body.push(Item::Inst(s.store(slot, var)));
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
            body.extend([Item::Inst(s.load(var, slot)), Item::Inst(inst)]);
            continue;
        }
        let t = VReg { id: p.regs.len() as u32, class };
        p.regs.push(RegInfo { class, name: name.clone() });
        let (reads, writes) = (inst.reads().any(|r| r == var), inst.writes().any(|w| w == var));
        if reads {
            body.push(Item::Inst(s.load(t, slot)));
        }
        body.push(Item::Inst(renamed(inst, var, t)));
        if writes {
            body.push(Item::Inst(s.store(slot, t)));
        }
    }
    p.body = body;
}

/// `inst` with `to` wherever it names `from`.
pub(crate) fn renamed<Op>(inst: Inst<Op, VReg>, from: VReg, to: VReg) -> Inst<Op, VReg> {
    let rename = |a: Arg<VReg>| a.map(|r| if r == from { to } else { r });
    Inst {
        op: inst.op,
        args: inst.args.into_iter().map(rename).collect(),
        implicit: inst.implicit.into_iter().map(rename).collect(),
        flow: inst.flow,
        clobbers: inst.clobbers,
    }
}
