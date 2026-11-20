use std::collections::BTreeMap;

use kernite_core::{Flow, Inst, Item, Label, Program, VReg};

use crate::Error;
use crate::bits::Bits;

/// A run of items that control enters only at the top and leaves only at the bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    /// Its items, as a range of the body.
    pub(crate) items: std::ops::Range<usize>,
    /// The blocks control can go to next.
    pub(crate) succs: Vec<usize>,
}

/// A body cut into [`Block`]s, in body order: a block starts at the top, at every label and
/// after every instruction that can jump.
pub(crate) fn blocks<Op, R>(body: &[Item<Op, R>]) -> Result<Vec<Block>, Error> {
    let mut at = BTreeMap::new();
    for (i, item) in body.iter().enumerate() {
        if let Item::Label(Label(l)) = *item
            && at.insert(l, i).is_some()
        {
            return Err(Error::LabelTwice { label: l });
        }
    }
    let mut starts = vec![0];
    for (i, item) in body.iter().enumerate() {
        match item {
            Item::Label(_) if i > 0 => starts.push(i),
            Item::Inst(inst) if inst.flow != Flow::Next && i + 1 < body.len() => starts.push(i + 1),
            _ => {}
        }
    }
    starts.dedup();
    let ends = starts.iter().skip(1).copied().chain([body.len()]);
    let ranges: Vec<_> = starts.iter().copied().zip(ends).map(|(s, e)| s..e).collect();
    let block_at = |label: u32| {
        let i = *at.get(&label).ok_or(Error::NoSuchLabel { label })?;
        Ok(ranges.iter().position(|r| r.start == i).expect("labels start blocks"))
    };
    let mut blocks = Vec::with_capacity(ranges.len());
    for (b, items) in ranges.iter().enumerate() {
        let last = body[items.clone()].iter().rev().find_map(|item| match item {
            Item::Inst(inst) => Some(&inst.flow),
            _ => None,
        });
        let next = || if b + 1 < ranges.len() { Ok(b + 1) } else { Err(Error::FallsOffEnd) };
        let flow = last.unwrap_or(&Flow::Next);
        let succs = flow.labels().iter().map(|&Label(l)| block_at(l)).chain(flow.falls_through().then(next));
        blocks.push(Block { items: items.clone(), succs: succs.collect::<Result<_, _>>()? });
    }
    Ok(blocks)
}

/// For each item of `p`'s body, whether every path from the start to the end runs it: whether it
/// post-dominates the start. It fails, as [`check`](crate::check) does, if control leaves the
/// body or a jump has no label.
///
/// A path may end at any of `leaves`, the indexes of the items where a thread may end, or never
/// come back, while the others go on: a guarded `ret`, or a call. What follows one in its block
/// is on no path that ends there. A loop that has no way out is taken as one that may end after
/// any round, at the jump that goes back. These ends are the analysis's alone: the body keeps
/// the flow it has.
pub fn on_every_path<Op>(p: &Program<Op>, leaves: &[usize]) -> Result<Vec<bool>, Error> {
    let blocks = blocks(&p.body)?;
    // The end is a block of its own, after the last.
    let end = blocks.len();
    let mut succs: Vec<Vec<usize>> = blocks.iter().map(|b| b.succs.clone()).collect();
    // How far every path into each block runs it: through its first leave, if it has one.
    let mut runs: Vec<usize> = blocks.iter().map(|b| b.items.end).collect();
    for &leave in leaves {
        let b = blocks.iter().position(|b| b.items.contains(&leave)).expect("a leave is an item of the body");
        succs[b].push(end);
        runs[b] = runs[b].min(leave + 1);
    }
    for s in succs.iter_mut().filter(|s| s.is_empty()) {
        s.push(end);
    }
    let mut reaches = vec![false; end + 1];
    reaches[end] = true;
    let mut grown = true;
    while grown {
        grown = false;
        for b in (0..end).rev() {
            if !reaches[b] && succs[b].iter().any(|&s| reaches[s]) {
                reaches[b] = true;
                grown = true;
            }
        }
    }
    for (b, s) in succs.iter_mut().enumerate() {
        if !reaches[b] && s.iter().any(|&to| to <= b) {
            s.push(end);
        }
    }
    // The blocks that every path from `b` to the end passes through, `b` among them.
    let mut after = vec![Bits::full(end + 1); end + 1];
    after[end] = Bits::new(end + 1);
    after[end].insert(end);
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..end).rev() {
            let mut now = Bits::full(end + 1);
            for &s in &succs[b] {
                now.intersect_with(&after[s]);
            }
            now.insert(b);
            if now != after[b] {
                after[b] = now;
                changed = true;
            }
        }
    }
    let mut on = vec![false; p.body.len()];
    for b in after[0].iter().filter(|&b| b < end) {
        on[blocks[b].items.start..runs[b]].fill(true);
    }
    Ok(on)
}

/// Follows a value over `p`'s body from its start to a fixed point: `start` is the value at the top of the body, `step`
/// what an instruction does to it, and `meet` what the value at a label is of the values the ways to it bring, a jump
/// back among them. An item that no way comes to is not stepped. `step` must give no more than it is given, and `meet`
/// no more than either of its values, for the loop to end. It fails, as [`check`](crate::check) does, if control
/// leaves the body or a jump has no label.
///
/// `step` fails where what the instruction does cannot be done, and does what it can to the value. Of the failures once
/// the values are as small as they get, the first in the body is given: its item's index, and what `step` said.
pub fn forward<Op, S: Clone + PartialEq, E>(
    p: &Program<Op>,
    start: S,
    meet: impl Fn(&S, &S) -> S,
    step: impl Fn(&Inst<Op, VReg>, &mut S) -> Result<(), E>,
) -> Result<Option<(usize, E)>, Error> {
    let blocks = blocks(&p.body)?;
    // The value at the top of each block, once a way to it is known.
    let mut tops: Vec<Option<S>> = vec![None; blocks.len()];
    tops[0] = Some(start);
    // The value at the bottom of block `b`, and its first failure.
    let run = |b: usize, top: &S| {
        let mut now = top.clone();
        let mut failed = None;
        for i in blocks[b].items.clone() {
            if let Item::Inst(inst) = &p.body[i]
                && let Err(e) = step(inst, &mut now)
            {
                failed.get_or_insert((i, e));
            }
        }
        (now, failed)
    };
    let mut changed = true;
    while changed {
        changed = false;
        for (b, block) in blocks.iter().enumerate() {
            let Some(top) = &tops[b] else { continue };
            let (bottom, _) = run(b, top);
            for &to in &block.succs {
                let met = tops[to].as_ref().map_or_else(|| bottom.clone(), |t| meet(t, &bottom));
                if tops[to].as_ref() != Some(&met) {
                    tops[to] = Some(met);
                    changed = true;
                }
            }
        }
    }
    Ok(tops.iter().enumerate().find_map(|(b, top)| run(b, top.as_ref()?).1))
}
