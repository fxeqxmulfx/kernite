//! A kernel's frame made instructions, before it is printed: what it runs on entry, and before
//! each `ret` and each jump to a callee that stands for one.

use std::collections::{BTreeMap, BTreeSet};
use std::mem;

use kernite_compile::{Allocated, Frame};
use kernite_core::{Access, Arg, Base, Flow, Inst, Item, Label, Mem, Operand, PReg, Width};

use crate::forms::{self, Feature};
use crate::op::Op;
use crate::operand::{number, ymm};
use crate::sysv::{CALL_CLOBBERS, PAGE, R11, RBP, RSP};

/// An allocated kernel with its frame made instructions, as the printer prints it.
#[derive(Debug)]
pub(crate) struct Lowered {
    /// The code, the frame's instructions among it, as [`print`](crate::intel::print) says.
    pub(crate) body: Vec<Item<Op, PReg>>,
    /// The CPU features past x86-64's own the code needs.
    pub(crate) features: Vec<Feature>,
    /// Bytes of the frame besides its slots, so that a kernel that calls calls at a multiple of 16, and a frame aligned
    /// to 16 lies at one.
    pub(crate) padding: u32,
    /// How many bytes of the frame lie below the stack pointer, in the red zone, which the prologue does not move the
    /// stack pointer for: its slots are at `[rsp - below + offset]`.
    pub(crate) below: u32,
    /// Whether control returns: whether the kernel has a `ret`, though a jump to a callee may stand for it.
    pub(crate) returns: bool,
    /// The registers a call of it changes that a caller does not keep, as `Listing::changes` says.
    pub(crate) changes: Vec<PReg>,
    /// Whether it runs `vzeroupper`, as `Listing::clears_upper` says.
    pub(crate) clears_upper: bool,
    /// Whether a call of it may run SSE code, as `Listing::reaches_sse` says.
    pub(crate) reaches_sse: bool,
}

/// `c` with its frame made instructions, as [`print`](crate::intel::print) says; or why not.
pub(crate) fn lower(c: &Allocated<Op>) -> Result<Lowered, String> {
    let jumps = jumps(c);
    let calls = insts(&c.body).filter(|inst| inst.op == Op::Call).count() > jumps.len();
    let returns = insts(&c.body).any(|inst| inst.op == Op::Ret);
    let uppers = upper_set(&c.body, &jumps);
    // The prologue's loop takes the least value no label of the body has: going up the body's
    // labels in order, each one that has the value looked for moves it one on.
    let labels: BTreeSet<u32> = c
        .body
        .iter()
        .filter_map(|item| match item {
            Item::Label(l) => Some(l.0),
            Item::Inst(_) | Item::Note(_) => None,
        })
        .collect();
    let free = Label(labels.into_iter().fold(0, |free, l| if l == free { free + 1 } else { free }));
    let padding = padding(&c.frame, calls);
    let below = below(&c.frame, padding, calls);
    let mut body = prologue(&c.frame, padding, free, below)?;
    let done: BTreeSet<usize> = jumps.values().flatten().copied().collect();
    // Whether the call that comes next has a `vzeroupper` written before it.
    let mut clear = false;
    for (i, (item, &upper)) in c.body.iter().zip(&uppers).enumerate() {
        match item {
            Item::Inst(call) if jumps.contains_key(&i) => {
                let mut jump =
                    Inst { op: Op::Jmp, implicit: vec![], flow: Flow::Return, clobbers: vec![], ..call.clone() };
                // An address in a register the epilogue restores goes first to `r11`, which passes no argument and
                // which the epilogue leaves as it is.
                if let Some(&Arg { operand: Operand::Reg(at, _), .. }) = call.args.first()
                    && c.frame.saved.contains(&at)
                {
                    body.push(inst(Op::Mov, vec![reg(R11, Access::Write), reg(at, Access::Read)]));
                    jump.args = vec![reg(R11, Access::Read)];
                }
                // The callee, passed all of a ymm, clears the upper halves before it returns. A `vzeroupper` written
                // before the call goes after the frame is taken down, where it goes before a `ret`.
                let clears = (mem::take(&mut clear) || upper) && !call.implicit.iter().any(ymm);
                body.extend(epilogue(&c.frame, padding, below, clears));
                body.push(Item::Inst(jump));
                continue;
            }
            Item::Inst(_) if done.contains(&i) => continue,
            Item::Inst(inst) if inst.op == Op::Ret => body.extend(epilogue(&c.frame, padding, below, upper)),
            // The `vzeroupper` written before a call waits for it, and goes before it where an upper half may be set.
            Item::Inst(inst) if inst.op == Op::Vzeroupper => {
                clear = upper;
                continue;
            }
            Item::Inst(call) if call.op == Op::Call && mem::take(&mut clear) => {
                body.push(inst(Op::Vzeroupper, vec![]));
            }
            Item::Inst(inst) if idle(inst) => continue,
            Item::Inst(inst) => {
                let spill = inst.args.iter().find_map(|a| match a.operand {
                    Operand::Mem(Mem { base: Base::Slot(s), .. }) => {
                        let slot = c.frame.slots.get(s.0 as usize).filter(|slot| slot.spill)?;
                        Some((&slot.name, a.access))
                    }
                    _ => None,
                });
                match spill {
                    Some((var, Access::Write)) => body.push(Item::Note(format!("{var} spilled"))),
                    Some((var, _)) => body.push(Item::Note(format!("{var} reloaded"))),
                    None => {}
                }
            }
            Item::Label(_) | Item::Note(_) => {}
        }
        body.push(item.clone());
    }
    // The frame's instructions need none, but for the epilogue's `vzeroupper`, which is AVX's: a kernel that writes it
    // needs AVX, unless it lists a feature that builds on it.
    let mut features: Vec<Feature> = vec![];
    for inst in insts(&c.body).filter(|inst| !idle(inst)) {
        features.extend(forms::feature(inst.op, &inst.args, &inst.implicit)?);
    }
    let implies_avx = [Feature::Avx2, Feature::Fma, Feature::F16c];
    let clears_upper = insts(&body).any(|inst| inst.op == Op::Vzeroupper);
    let reaches_sse = insts(&c.body).any(|inst| inst.op == Op::Vzeroupper);
    if clears_upper && !features.iter().any(|f| implies_avx.contains(f)) {
        features.push(Feature::Avx);
    }
    features.sort_by_key(|f| f.name());
    features.dedup();
    let changes = insts(&body).flat_map(Inst::writes).filter(|r| CALL_CLOBBERS.contains(r)).collect::<BTreeSet<_>>();
    let changes = changes.into_iter().collect();
    Ok(Lowered { body, features, padding, below, returns, changes, clears_upper, reaches_sse })
}

/// The calls of `c` that are jumps, by their places in its body, each with the place of the `ret` it does without, if
/// no other way comes to that `ret`. A call that `ret` follows at once, with nothing between but labels, notes and
/// moves that change nothing, returns what the callee gives back: the kernel takes its frame down, as before a `ret`,
/// which clears the upper halves of the vector registers where the kernel may have set one, and jumps to the callee,
/// which returns to the kernel's caller. A call stays one where the jump would not do what it does: where the kernel
/// has taken the address of its stack memory, which the callee may use, and the frame the epilogue takes down holds;
/// and where it takes all of a ymm back from a function it does not know: the epilogue's `vzeroupper` runs before the
/// callee sets the upper half of what it gives back. A jump that passes all of a ymm has no `vzeroupper` before it: the
/// callee, passed all of a ymm, clears the upper halves before it returns.
fn jumps(c: &Allocated<Op>) -> BTreeMap<usize, Option<usize>> {
    let stack = |a: &Arg<PReg>| matches!(a.operand, Operand::Mem(Mem { base: Base::Slot(_), .. }));
    let mut jumps = BTreeMap::new();
    if insts(&c.body).any(|inst| inst.op == Op::Lea && inst.args.iter().any(stack)) {
        return jumps;
    }
    for (i, item) in c.body.iter().enumerate() {
        let Item::Inst(call) = item else { continue };
        if call.op != Op::Call {
            continue;
        }
        let next = c.body.iter().enumerate().skip(i + 1).find_map(|(j, item)| match item {
            Item::Inst(inst) if !idle(inst) => Some((j, inst)),
            _ => None,
        });
        let Some((j, ret)) = next else { continue };
        if ret.op == Op::Ret && !call.implicit.iter().any(|a| ymm(a) && a.access == Access::Write) {
            let alone = !c.body.iter().take(j).skip(i + 1).any(|item| matches!(item, Item::Label(_)));
            jumps.insert(i, alone.then_some(j));
        }
    }
    jumps
}

/// The instructions of `body`, in order.
fn insts(body: &[Item<Op, PReg>]) -> impl Iterator<Item = &Inst<Op, PReg>> {
    body.iter().filter_map(|item| match item {
        Item::Inst(inst) => Some(inst),
        Item::Label(_) | Item::Note(_) => None,
    })
}

/// Whether the upper half of a vector register may be set where each item of `body` runs: by an instruction that reads
/// or writes all of a ymm, on some way to it, with no `vzeroupper` since. The `vzeroupper` that `finish` writes before
/// a call that may run SSE code stands for the one the frame writes there. A call has such an operand where it passes
/// all of a ymm to a function it does not know, or takes all of one back. A move that the listing leaves out sets none.
/// A way ends at a call of `jumps`, as at a `ret`: the frame makes it a jump.
fn upper_set(body: &[Item<Op, PReg>], jumps: &BTreeMap<usize, Option<usize>>) -> Vec<bool> {
    // The labels that control comes to with one set. Each round finds those that the last did not, until a round finds
    // none.
    let mut at = BTreeSet::new();
    loop {
        let (mut set, mut now, mut grown) = (Vec::with_capacity(body.len()), false, false);
        for (i, item) in body.iter().enumerate() {
            let Item::Inst(inst) = item else {
                if let Item::Label(l) = item {
                    now |= at.contains(&l.0);
                }
                set.push(false);
                continue;
            };
            set.push(now);
            if inst.op == Op::Vzeroupper {
                now = false;
            } else if !idle(inst) && inst.operands().any(ymm) {
                now = true;
            }
            for l in inst.flow.labels().iter().filter(|_| now) {
                grown |= at.insert(l.0);
            }
            now &= inst.flow.falls_through() && !jumps.contains_key(&i);
        }
        if !grown {
            return set;
        }
    }
}

/// Whether `inst` changes nothing, so the listing leaves it out: a move of all 64 bits, or all
/// 256, to the same register. But one of 32 bits clears the upper half, and one of an xmm the
/// upper lanes.
fn idle(inst: &Inst<Op, PReg>) -> bool {
    matches!(inst.op, Op::Mov | Op::Vmovaps | Op::Vmovups | Op::Vmovdqa | Op::Vmovdqu)
        && matches!(
            inst.args.as_slice(),
            [Arg { operand: Operand::Reg(a, w), .. }, Arg { operand: Operand::Reg(b, v), .. }]
                if a == b && w == v && matches!(w, Width::W64 | Width::W256)
        )
}

/// Pages past this many are touched in a loop.
const UNROLLED: u32 = 3;

/// How many bytes below the stack pointer SysV lets a function use without moving it, its red zone: a signal handler
/// leaves them as they are, and only a call, which pushes its return address there, writes them.
const RED_ZONE: u32 = 128;

/// How many bytes of `frame` lie below the stack pointer once the prologue has run, in the red zone: all of it, up to
/// [`RED_ZONE`], and the rest above, where the kernel calls nothing, as a call writes its return address there, and the
/// frame is aligned to 16 or less, for `and` makes any other. The pages that [`prologue`] touches are above it too: the
/// stack pointer moves by them.
fn below(frame: &Frame, pad: u32, calls: bool) -> u32 {
    if calls || realigns(frame) {
        return 0;
    }
    (frame.size + pad - probes(frame, pad) * PAGE).min(RED_ZONE)
}

/// How many bytes the stack pointer is a multiple of at a call, as SysV asks. A frame aligned to no more is aligned by
/// moving the stack pointer down a few bytes more, one aligned to more by `and`.
const CALLED: u32 = 16;

/// Whether the prologue aligns the stack pointer with `and`, and keeps the caller's in `rbp` for the epilogue to
/// restore: where `frame` is aligned to more than the stack pointer is at a call.
fn realigns(frame: &Frame) -> bool {
    frame.align > CALLED
}

/// How many pages the prologue touches on the way down to `frame`'s bottom, a page apart: as
/// many as leave no more than a page untouched above the bottom, counting `pad`, and the
/// alignment of a frame that `and` aligns: the `and` moves the stack pointer by less, but to
/// no lower page.
fn probes(frame: &Frame, pad: u32) -> u32 {
    let aligned = if realigns(frame) { frame.align } else { 0 };
    match frame.size {
        0 => 0,
        size => (size + pad + aligned).saturating_sub(PAGE).div_ceil(PAGE),
    }
}

/// Bytes of the frame besides its slots, above them, so that the frame lies at a multiple of 16: where the kernel
/// `calls`, as SysV asks of the stack pointer, which `sub rsp` leaves at the frame, and where `frame` is aligned to 16,
/// whose slots then lie at multiples of 16. On entry the stack pointer is 8 past a multiple of 16, the return address
/// below it, and each push moves it 8 more. A frame aligned to more needs none: the prologue aligns it.
fn padding(frame: &Frame, calls: bool) -> u32 {
    if realigns(frame) || !(calls || frame.align == CALLED) {
        return 0;
    }
    let pushes = frame.saved.len() as u32;
    (CALLED - (8 + 8 * pushes + frame.size) % CALLED) % CALLED
}

/// Saves the callee-saved registers the body writes; then, if there are slots, moves the stack pointer down past them
/// and `pad`; else down past `pad` alone. A frame aligned to more than 16 keeps the caller's stack pointer in `rbp`
/// first, and aligns the stack pointer last. It does not move the stack pointer for the `below` bytes of the frame that
/// lie below it.
///
/// No byte of the frame lies more than a page below memory the kernel has touched, so that
/// a frame that runs past the end of the stack faults at the guard page there rather than step
/// over it, as Rust's own functions do: a frame of more, its alignment counted, is touched a
/// page at a time on the way down, past [`UNROLLED`] pages in a loop at label `free`, which
/// keeps its end in `r11`. The alignment comes last, a page at most, unless the stack pointer
/// has it already. A frame that extends more than 2^31 bytes past the stack pointer, more than an
/// address reaches, says so.
fn prologue(frame: &Frame, pad: u32, free: Label, below: u32) -> Result<Vec<Item<Op, PReg>>, String> {
    let mut items: Vec<_> = frame.saved.iter().map(|&r| inst(Op::Push, vec![reg(r, Access::Read)])).collect();
    if frame.size == 0 {
        if pad > 0 {
            items.push(sp(Op::Sub, pad));
        }
        return Ok(items);
    }
    // No kernel's frame is aligned past a page: the language refuses one, and so does `Asm::finish`; one
    // that was might step over a page.
    if frame.size > (1 << 31) + below {
        return Err(format!(
            "the kernel's frame, its stack memory and the variables that spilled, reaches {} bytes past the stack \
             pointer: more than an address does",
            frame.size - below
        ));
    }
    if realigns(frame) {
        items.extend([
            inst(Op::Push, vec![reg(RBP, Access::Read)]),
            inst(Op::Mov, vec![reg(RBP, Access::Write), reg(RSP, Access::Read)]),
        ]);
    }
    let pages = probes(frame, pad);
    if pages > 0 {
        items.push(Item::Note("the frame a page at a time, each touched, so that an overflow faults".into()));
        let top = Mem { base: Base::Reg(RSP), index: None, disp: 0, width: Width::W64 };
        let touch = Arg { operand: Operand::Mem(top), access: Access::ReadWrite };
        let probe = [sp(Op::Sub, PAGE), inst(Op::Or, vec![touch, number(0)])];
        if pages <= UNROLLED {
            (0..pages).for_each(|_| items.extend(probe.clone()));
        } else {
            // The loop ends a whole number of pages down, above the frame's bottom, its padding and alignment counted.
            // The frame reaches 2^31 bytes past the stack pointer at most, and the bytes below the stack pointer, the
            // padding and the alignment add a page at most: so the loop ends 2^31 bytes down at most, a displacement of
            // `i32::MIN`.
            let disp = 0_i32.wrapping_sub_unsigned(pages * PAGE);
            let end = Arg { operand: Operand::Mem(Mem { disp, ..top }), access: Access::Read };
            items.extend([inst(Op::Lea, vec![reg(R11, Access::Write), end]), Item::Label(free)]);
            items.extend(probe);
            let back = Arg { operand: Operand::Label(free), access: Access::Read };
            items.extend([
                inst(Op::Cmp, vec![reg(RSP, Access::Read), reg(R11, Access::Read)]),
                Item::Inst(Inst {
                    op: Op::Jne,
                    args: vec![back],
                    implicit: vec![],
                    flow: Flow::Branch(free),
                    clobbers: vec![],
                }),
            ]);
        }
    }
    let rest = frame.size + pad - below - pages * PAGE;
    if rest > 0 {
        items.push(sp(Op::Sub, rest));
    }
    if realigns(frame) {
        items.push(inst(Op::And, vec![reg(RSP, Access::ReadWrite), number(-i64::from(frame.align))]));
    }
    Ok(items)
}

/// What runs before a `ret`, and before a jump to a callee that stands for one: undoes [`prologue`], and clears the
/// upper halves of the vector registers if `upper`, as [`upper_set`] says, so the caller's SSE code does not pay for
/// them.
fn epilogue(frame: &Frame, pad: u32, below: u32, upper: bool) -> Vec<Item<Op, PReg>> {
    let mut items = vec![];
    if realigns(frame) {
        items.extend([
            inst(Op::Mov, vec![reg(RSP, Access::Write), reg(RBP, Access::Read)]),
            inst(Op::Pop, vec![reg(RBP, Access::Write)]),
        ]);
    } else {
        // An `add` takes 2^31 - 1 at most, and a frame may be 2^31 bytes, with `pad` more.
        let mut back = frame.size + pad - below;
        while back > 0 {
            let by = back.min(i32::MAX as u32);
            items.push(sp(Op::Add, by));
            back -= by;
        }
    }
    items.extend(frame.saved.iter().rev().map(|&r| inst(Op::Pop, vec![reg(r, Access::Write)])));
    if upper {
        items.push(inst(Op::Vzeroupper, vec![]));
    }
    items
}

/// `op` of `args`, which goes on to the next item.
fn inst(op: Op, args: Vec<Arg<PReg>>) -> Item<Op, PReg> {
    Item::Inst(Inst { op, args, implicit: vec![], flow: Flow::Next, clobbers: vec![] })
}

/// All 64 bits of `r`.
fn reg(r: PReg, access: Access) -> Arg<PReg> {
    Arg { operand: Operand::Reg(r, Width::W64), access }
}

/// The stack pointer moved by `bytes`: down for `sub`, up for `add`.
fn sp(op: Op, bytes: u32) -> Item<Op, PReg> {
    inst(op, vec![reg(RSP, Access::ReadWrite), number(bytes.into())])
}

#[cfg(test)]
mod tests {
    use kernite_core::{Class, Signature, Slot, SlotId};

    use super::*;
    use crate::intel;

    type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

    const RAX: PReg = PReg { class: Class::Gpr, hw: 0 };
    const RCX: PReg = PReg { class: Class::Gpr, hw: 1 };
    const RBX: PReg = PReg { class: Class::Gpr, hw: 3 };
    const RDI: PReg = PReg { class: Class::Gpr, hw: 7 };
    const YMM0: PReg = PReg { class: Class::Vec, hw: 0 };

    /// A kernel of `body` in `frame`, allocated by hand.
    fn kernel(body: Vec<Item<Op, PReg>>, frame: Frame) -> Allocated<Op> {
        let signature = Signature { params: vec![], result: None };
        Allocated {
            name: "k".into(),
            signature,
            body,
            frame,
            consts: vec![],
            funcs: vec![],
            statics: vec![],
            assignment: vec![],
        }
    }

    /// Stack memory a kernel declares, or a slot of a variable that spilled.
    fn slot(name: &str, size: u32, spill: bool) -> Slot {
        Slot { name: name.into(), size, align: 8, spill }
    }

    /// `op` of `to`, written, and `from`, read, at `width`.
    fn mov(op: Op, to: PReg, from: PReg, width: Width) -> Item<Op, PReg> {
        let arg = |r, access| Arg { operand: Operand::Reg(r, width), access };
        inst(op, vec![arg(to, Access::Write), arg(from, Access::Read)])
    }

    /// The code of `c`'s listing, a line each.
    fn code(c: &Allocated<Op>) -> Result<String> {
        Ok(intel::print(c, vec![])?.code().collect::<Vec<_>>().join("\n"))
    }

    /// The prologue and the epilogue are instructions of the body before the printer sees it,
    /// which it prints as it prints the rest.
    #[test]
    fn the_frame_is_instructions_before_it_is_printed() -> Result {
        let b = Slot { align: 32, ..slot("b", 32, false) };
        let frame = Frame { saved: vec![RBX], slots: vec![b], offsets: vec![0], size: 32, align: 32 };
        let store = Mem { base: Base::Slot(SlotId(0)), index: None, disp: 0, width: Width::W64 };
        let body = vec![
            mov(Op::Mov, RBX, RDI, Width::W64),
            inst(Op::Mov, vec![Arg { operand: Operand::Mem(store), access: Access::Write }, reg(RBX, Access::Read)]),
            inst(Op::Ret, vec![]),
        ];
        let c = kernel(body, frame);
        let lowered = lower(&c)?;
        let ops: Vec<_> = lowered
            .body
            .iter()
            .map(|item| match item {
                Item::Inst(inst) => inst.op.name(),
                Item::Label(_) | Item::Note(_) => "not an instruction",
            })
            .collect();
        assert_eq!(ops, ["push", "push", "mov", "sub", "and", "mov", "mov", "mov", "pop", "pop", "ret"]);
        let want = "push rbx\npush rbp\nmov rbp, rsp\nsub rsp, 32\nand rsp, -32\nmov rbx, rdi\n\
                    mov qword ptr [rsp], rbx\nmov rsp, rbp\npop rbp\npop rbx\nret";
        assert_eq!(code(&c)?, want);
        assert_eq!((lowered.returns, lowered.padding, lowered.features), (true, 0, vec![]));
        Ok(())
    }

    /// A frame that touches more than `UNROLLED` pages touches them in a loop, at a label of its
    /// own, placed first so that it takes the first number: the least value no label of the body
    /// has, 2 beside the body's 1, 0 and `grid.sync()`'s `u32::MAX`, which come out of order.
    #[test]
    fn the_probe_loops_label_is_one_no_label_of_the_body_has() -> Result {
        let frame = Frame {
            saved: vec![],
            slots: vec![slot("b", 5 * PAGE, false)],
            offsets: vec![0],
            size: 5 * PAGE,
            align: 8,
        };
        let body =
            vec![Item::Label(Label(1)), Item::Label(Label(0)), Item::Label(Label(u32::MAX)), inst(Op::Ret, vec![])];
        let c = kernel(body, frame);
        let lowered = lower(&c)?;
        let labels: Vec<_> =
            lowered.body.iter().filter_map(|item| if let Item::Label(l) = item { Some(l.0) } else { None }).collect();
        assert_eq!(labels, [2, 1, 0, u32::MAX]);
        let jne = lowered.body.iter().find_map(|item| match item {
            Item::Inst(inst) if inst.op == Op::Jne => Some(&inst.flow),
            _ => None,
        });
        assert_eq!(jne, Some(&Flow::Branch(Label(2))));
        let want = "lea r11, [rsp - 16384]\n2:\nsub rsp, 4096\nor qword ptr [rsp], 0\ncmp rsp, r11\njne 2b\n\
                    sub rsp, 3968\n3:\n4:\n5:\nadd rsp, 20352\nret";
        assert_eq!(code(&c)?, want);
        Ok(())
    }

    /// An address reaches 2^31 bytes past the stack pointer, and the bytes of a frame that lie below it are not past
    /// it: a frame of 2^31 bytes and the 128 below the stack pointer is taken, and one of a byte more is not.
    #[test]
    fn the_bytes_of_a_frame_below_the_stack_pointer_are_not_reached_past_it() {
        let lowered = |size| {
            let frame = Frame { saved: vec![], slots: vec![slot("b", size, false)], offsets: vec![0], size, align: 8 };
            lower(&kernel(vec![inst(Op::Ret, vec![])], frame))
        };
        assert!(lowered((1 << 31) + RED_ZONE).is_ok());
        let want = "the kernel's frame, its stack memory and the variables that spilled, reaches 2147483649 bytes past \
                    the stack pointer: more than an address does";
        assert_eq!(lowered((1 << 31) + RED_ZONE + 1).err().as_deref(), Some(want));
    }

    /// A move of a register to itself, of all 64 bits by `mov` or all 256 by `vmovaps`, `vmovups`, `vmovdqa` or
    /// `vmovdqu`, changes nothing, and is left out; one of 32 bits clears the upper half, and one of an xmm the upper
    /// lanes, and stay. A move left out asks for no `vzeroupper`, nor does one of an xmm, which sets no upper half.
    #[test]
    fn a_move_of_a_register_to_itself_is_left_out() -> Result {
        let frame = Frame { saved: vec![], slots: vec![], offsets: vec![], size: 0, align: 0 };
        let wide = [Op::Vmovaps, Op::Vmovups, Op::Vmovdqa, Op::Vmovdqu].map(|op| mov(op, YMM0, YMM0, Width::W256));
        let ret = inst(Op::Ret, vec![]);
        let mut body = vec![mov(Op::Mov, RAX, RAX, Width::W64), mov(Op::Mov, RAX, RAX, Width::W32)];
        body.extend(wide.clone());
        body.extend([mov(Op::Vmovaps, YMM0, YMM0, Width::W128), mov(Op::Mov, RAX, RCX, Width::W64), ret.clone()]);
        assert_eq!(code(&kernel(body, frame.clone()))?, "mov eax, eax\nvmovaps xmm0, xmm0\nmov rax, rcx\nret");
        assert_eq!(code(&kernel([&wide[..], &[ret]].concat(), frame))?, "ret");
        Ok(())
    }
}
