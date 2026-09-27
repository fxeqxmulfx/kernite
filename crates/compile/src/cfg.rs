use std::collections::BTreeMap;

use kgen_core::{Flow, Item, Label};

use crate::Error;

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
            Item::Inst(inst) => Some(inst.flow),
            _ => None,
        });
        let next = || if b + 1 < ranges.len() { Ok(b + 1) } else { Err(Error::FallsOffEnd) };
        let succs = match last.unwrap_or(Flow::Next) {
            Flow::Next => vec![next()?],
            Flow::Jump(Label(l)) => vec![block_at(l)?],
            Flow::Branch(Label(l)) => vec![block_at(l)?, next()?],
            Flow::Return => vec![],
        };
        blocks.push(Block { items: items.clone(), succs });
    }
    Ok(blocks)
}
