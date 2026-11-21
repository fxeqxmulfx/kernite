//! Whether a kernel returns with a store that x86 has not ordered yet. A non-temporal store, `movnti` or `vmovntps`
//! say, goes around the caches, and x86 lets a later store pass it, so it may land after the kernel returns: after what
//! its caller does next, a flag it sets for another thread say. An `sfence` or an `mfence` orders every store before
//! it. [`check`] follows that over the body's flow, and gives the first store that a way to a `ret` leaves unordered.

use kernite_compile::forward;
use kernite_core::{Inst, Item, Program, VReg};

use crate::op::Op;

/// The first non-temporal store that may not be ordered yet, by its place in the body.
type Unordered = Option<usize>;

/// What `inst`, at `at` in the body, does to `now`: a non-temporal store leaves one unordered, an `sfence` or an
/// `mfence` leaves none. It says the store that is, at a `ret`.
fn step(at: usize, inst: &Inst<Op, VReg>, now: &mut Unordered) -> Unordered {
    match inst.op.name() {
        "movnti" | "vmovntps" | "vmovntpd" | "vmovntdq" => {
            now.get_or_insert(at);
        }
        "sfence" | "mfence" => *now = None,
        _ => {}
    }
    now.filter(|_| inst.op == Op::Ret)
}

/// The first non-temporal store of `p` that a `ret` may come to unordered, if any: its item's index, and what is wrong.
/// A body that is not whole, that control leaves or a jump has no label for, has none: allocating registers says what
/// is wrong with it.
pub(crate) fn check(p: &Program<Op>) -> Option<(usize, String)> {
    let meet = |a: &Unordered, b: &Unordered, _: bool| a.iter().chain(b).min().copied();
    let said = forward(p, None, meet, step, |_, _, _| {}).ok()?;
    let store = said.into_iter().flatten().flatten().min()?;
    let Some(Item::Inst(inst)) = p.body.get(store) else { return None };
    let message = format!(
        "`{}` is a non-temporal store, and the kernel may return before an `sfence` orders it, so a kernel that does \
         is an `unsafe fn`",
        inst.op.name()
    );
    Some((store, message))
}
