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

impl Block {
    /// Its items in `body`, the body [`blocks`] cut it from, which has them all.
    pub(crate) fn items_in<'a, Op, R>(&self, body: &'a [Item<Op, R>]) -> &'a [Item<Op, R>] {
        body.get(self.items.clone()).unwrap_or_default()
    }
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
        ranges.iter().position(|r| r.start == i).ok_or(Error::NoSuchLabel { label })
    };
    let mut blocks = Vec::with_capacity(ranges.len());
    for (b, items) in ranges.iter().enumerate() {
        let last = body.get(items.clone()).unwrap_or_default().iter().rev().find_map(|item| match item {
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
    // The first place a thread may end in a block, if one does. An index that is no item of the body is no such place.
    let leaving = |b: &Block| leaves.iter().copied().filter(|leave| b.items.contains(leave)).min();
    // How far every path into each block runs it: through its first leave, if it has one.
    let runs: Vec<usize> = blocks.iter().map(|b| leaving(b).map_or(b.items.end, |leave| leave + 1)).collect();
    let mut succs: Vec<Vec<usize>> =
        blocks.iter().map(|b| b.succs.iter().copied().chain(leaving(b).map(|_| end)).collect()).collect();
    for s in succs.iter_mut().filter(|s| s.is_empty()) {
        s.push(end);
    }
    let mut reaches = Bits::new(end + 1);
    reaches.insert(end);
    let mut grown = true;
    while grown {
        grown = false;
        for (b, s) in succs.iter().enumerate().rev() {
            if !reaches.contains(b) && s.iter().any(|&to| reaches.contains(to)) {
                reaches.insert(b);
                grown = true;
            }
        }
    }
    for (b, s) in succs.iter_mut().enumerate() {
        if !reaches.contains(b) && s.iter().any(|&to| to <= b) {
            s.push(end);
        }
    }
    // The blocks that every path from `b` to the end passes through, `b` among them.
    let mut last = Bits::new(end + 1);
    last.insert(end);
    let mut after = vec![Bits::full(end + 1); end];
    after.push(last);
    let mut changed = true;
    while changed {
        changed = false;
        for (b, s) in succs.iter().enumerate().rev() {
            let mut now = Bits::full(end + 1);
            for to in s.iter().filter_map(|&to| after.get(to)) {
                now.intersect_with(to);
            }
            now.insert(b);
            if let Some(was) = after.get_mut(b)
                && *was != now
            {
                *was = now;
                changed = true;
            }
        }
    }
    // The blocks tile the body in order: a block that every path runs has its items on every path, as far as it runs.
    let first = after.first();
    Ok(blocks
        .iter()
        .zip(&runs)
        .enumerate()
        .flat_map(|(b, (block, &run))| {
            let on = first.is_some_and(|every| every.contains(b));
            block.items.clone().map(move |i| on && i < run)
        })
        .collect())
}

/// Follows a value over `p`'s body to a fixed point: `start` is the value at the top of the body, `step` what the
/// instruction at an index of the body does to it, and `meet` what the value is at a label, of the values the ways to
/// it bring, a jump back among them. It gives what `step` said of each item, the last time `step` ran on it, once the
/// values stop changing: `None` for an item that no way comes to, which is not stepped, and for one that is not an
/// instruction. For the loop to end, `meet` gives no more than its first value, and a value goes down only so many
/// times. It fails, as [`check`](crate::check) does, if control leaves the body or a jump has no label.
pub fn forward<Op, S: Clone + PartialEq, T>(
    p: &Program<Op>,
    start: S,
    meet: impl Fn(&S, &S) -> S,
    step: impl Fn(usize, &Inst<Op, VReg>, &mut S) -> T,
) -> Result<Vec<Option<T>>, Error> {
    let blocks = blocks(&p.body)?;
    // The value at the top of each block, once a way to it is known: the first block's is `start`.
    let mut tops: Vec<Option<S>> = std::iter::once(Some(start)).chain(blocks.iter().skip(1).map(|_| None)).collect();
    // The value at the bottom of a block, and what `step` said of each of its items.
    let run = |block: &Block, top: &S| {
        let mut now = top.clone();
        let items = block.items_in(&p.body);
        let said: Vec<Option<T>> = items
            .iter()
            .zip(block.items.clone())
            .map(|(item, i)| match item {
                Item::Inst(inst) => Some(step(i, inst, &mut now)),
                _ => None,
            })
            .collect();
        (now, said)
    };
    let mut changed = true;
    while changed {
        changed = false;
        for (b, block) in blocks.iter().enumerate() {
            let Some(Some(top)) = tops.get(b) else { continue };
            let (bottom, _) = run(block, top);
            for &to in &block.succs {
                let Some(at) = tops.get_mut(to) else { continue };
                let met = at.as_ref().map_or_else(|| bottom.clone(), |t| meet(t, &bottom));
                if at.as_ref() != Some(&met) {
                    *at = Some(met);
                    changed = true;
                }
            }
        }
    }
    // The blocks tile the body in order, so their items' results, put end to end, are the body's.
    let said = tops.iter().zip(&blocks).flat_map(|(top, block)| match top {
        Some(top) => run(block, top).1,
        None => block.items.clone().map(|_| None).collect(),
    });
    Ok(said.collect())
}
