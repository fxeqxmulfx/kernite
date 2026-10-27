//! A kernel's frame made instructions, before it is printed: what it runs on entry, and before
//! each `ret`.

use std::collections::BTreeSet;

use kgen_compile::{Compiled, Frame};
use kgen_core::{Access, Arg, Base, Class, Flow, Inst, Item, Label, Mem, Operand, PReg, Width};

use crate::forms::{self, Feature};
use crate::op::Op;
use crate::operand::number;
use crate::sysv::{PAGE, R11, RBP, RSP};

/// An allocated kernel with its frame made instructions, as the printer prints it.
#[derive(Debug)]
pub(crate) struct Lowered {
    /// The code, the frame's instructions among it, as [`print`](crate::intel::print) says.
    pub(crate) body: Vec<Item<Op, PReg>>,
    /// The CPU features past x86-64's own the code needs.
    pub(crate) features: Vec<Feature>,
    /// Bytes the prologue moves the stack pointer down besides the frame's.
    pub(crate) padding: u32,
    /// Whether control returns: whether the body has a `ret`.
    pub(crate) returns: bool,
}

/// `c` with its frame made instructions, as [`print`](crate::intel::print) says; or why not.
pub(crate) fn lower(c: &Compiled<Op>) -> Result<Lowered, String> {
    let insts = || {
        c.body.iter().filter_map(|item| match item {
            Item::Inst(inst) => Some(inst),
            Item::Label(_) | Item::Note(_) => None,
        })
    };
    let calls = insts().any(|inst| inst.op == Op::Call);
    let returns = insts().any(|inst| inst.op == Op::Ret);
    let vector = insts().any(|inst| inst.operands().any(|a| a.regs().any(|r| r.class == Class::Vec)));
    let mut features: Vec<Feature> =
        insts().filter_map(|inst| forms::feature(inst.op, &inst.args, &inst.implicit)).collect();
    // The epilogue's `vzeroupper` is AVX's: a kernel that writes it needs AVX, unless it lists AVX
    // or a feature that builds on it.
    let implies_avx = [Feature::Avx, Feature::Avx2, Feature::Fma, Feature::F16c];
    if vector && returns && !features.iter().any(|f| implies_avx.contains(f)) {
        features.push(Feature::Avx);
    }
    features.sort_by_key(|f| f.name());
    features.dedup();
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
    let mut body = prologue(&c.frame, padding, free)?;
    for item in &c.body {
        match item {
            Item::Inst(inst) if inst.op == Op::Ret => body.extend(epilogue(&c.frame, padding, vector)),
            // All 64 bits, or all 256, to the same register: but one of 32 bits clears the upper
            // half, and one of an xmm the upper lanes.
            Item::Inst(inst)
                if matches!(inst.op, Op::Mov | Op::Vmovaps | Op::Vmovups | Op::Vmovdqa | Op::Vmovdqu)
                    && matches!(
                        inst.args.as_slice(),
                        [Arg { operand: Operand::Reg(a, w), .. }, Arg { operand: Operand::Reg(b, v), .. }]
                            if a == b && w == v && matches!(w, Width::W64 | Width::W256)
                    ) =>
            {
                continue;
            }
            Item::Inst(inst) => {
                let spill = inst.args.iter().find_map(|a| match a.operand {
                    Operand::Mem(Mem { base: Base::Slot(s), .. }) if c.frame.slots[s.0 as usize].spill => {
                        Some((&c.frame.slots[s.0 as usize].name, a.access))
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
    Ok(Lowered { body, features, padding, returns })
}

/// Pages past this many are touched in a loop.
const UNROLLED: u32 = 3;

/// How many bytes the stack pointer is a multiple of, always: SysV calls a function at a
/// multiple of 16, and each push moves it by 8.
const ALIGNED: u32 = 8;

/// How many pages the prologue touches on the way down to `frame`'s bottom, a page apart: as
/// many as leave no more than a page untouched above the bottom, counting `pad` and the most
/// the alignment may add.
fn probes(frame: &Frame, pad: u32) -> u32 {
    match frame.size {
        0 => 0,
        size => (size + pad + frame.align.saturating_sub(ALIGNED)).saturating_sub(PAGE).div_ceil(PAGE),
    }
}

/// Bytes the prologue moves the stack pointer down besides `frame`, if the kernel `calls`, so
/// that it calls at a multiple of 16: on entry it is 8 past one, the return address below it,
/// and each push moves it 8 more, the frame's `rbp` among them. A frame aligned to 16 or more
/// needs none.
fn padding(frame: &Frame, calls: bool) -> u32 {
    if !calls || frame.align >= 16 {
        return 0;
    }
    let pushes = frame.saved.len() as u32 + u32::from(frame.size > 0);
    (16 - (8 + 8 * pushes + frame.size) % 16) % 16
}

/// Saves the callee-saved registers the body writes; then, if there are slots, keeps the
/// caller's stack pointer in `rbp`, and moves the stack pointer down past them and `pad`, and
/// to their alignment; else down past `pad` alone.
///
/// The stack pointer never moves more than a page below memory the kernel has touched, so that
/// a frame that runs past the end of the stack faults at the guard page there rather than step
/// over it, as Rust's own functions do: a frame of more, its alignment counted, is touched a
/// page at a time on the way down, past [`UNROLLED`] pages in a loop at label `free`, which
/// keeps its end in `r11`. The alignment comes last, a page at most, unless the stack pointer
/// has it already. A frame of more than 2^31 bytes, more than an address reaches past the stack
/// pointer, says so.
fn prologue(frame: &Frame, pad: u32, free: Label) -> Result<Vec<Item<Op, PReg>>, String> {
    let mut items: Vec<_> = frame.saved.iter().map(|&r| inst(Op::Push, vec![reg(r, Access::Read)])).collect();
    if frame.size == 0 {
        if pad > 0 {
            items.push(sp(Op::Sub, pad));
        }
        return Ok(items);
    }
    // No kernel's frame is aligned past a page: the language refuses one, and `Asm::slot` asserts
    // it.
    assert!(frame.align <= PAGE, "a frame aligned to {}: the prologue might step over a page", frame.align);
    if frame.size > 1 << 31 {
        return Err(format!(
            "the kernel's frame, its stack memory and the variables that spilled, is {} bytes: more than an \
             address reaches past the stack pointer",
            frame.size
        ));
    }
    items.extend([
        inst(Op::Push, vec![reg(RBP, Access::Read)]),
        inst(Op::Mov, vec![reg(RBP, Access::Write), reg(RSP, Access::Read)]),
    ]);
    let pages = probes(frame, pad);
    if pages > 0 {
        items.push(Item::Note("the frame a page at a time, each touched, so that an overflow faults".into()));
        let top = Mem { base: Base::Reg(RSP), index: None, disp: 0, width: Width::W64 };
        let touch = Arg { operand: Operand::Mem(top), access: Access::ReadWrite };
        let probe = [sp(Op::Sub, PAGE), inst(Op::Or, vec![touch, number(0)])];
        if pages <= UNROLLED {
            (0..pages).for_each(|_| items.extend(probe.clone()));
        } else {
            // The padding and the alignment add less than a page, so the loop ends no further down
            // than the frame rounded up to a page: 2^31 bytes at most, a displacement of
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
    let rest = frame.size + pad - pages * PAGE;
    if rest > 0 {
        items.push(sp(Op::Sub, rest));
    }
    if frame.align > ALIGNED {
        items.push(inst(Op::And, vec![reg(RSP, Access::ReadWrite), number(-i64::from(frame.align))]));
    }
    Ok(items)
}

/// What runs before a `ret`: undoes [`prologue`], and clears the vector registers' upper halves
/// if the body used any, so the caller's SSE code does not pay for them.
fn epilogue(frame: &Frame, pad: u32, vector: bool) -> Vec<Item<Op, PReg>> {
    let mut items = vec![];
    if frame.size > 0 {
        items.extend([
            inst(Op::Mov, vec![reg(RSP, Access::Write), reg(RBP, Access::Read)]),
            inst(Op::Pop, vec![reg(RBP, Access::Write)]),
        ]);
    } else if pad > 0 {
        items.push(sp(Op::Add, pad));
    }
    items.extend(frame.saved.iter().rev().map(|&r| inst(Op::Pop, vec![reg(r, Access::Write)])));
    if vector {
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
    use kgen_core::{Signature, Slot, SlotId};

    use super::*;
    use crate::intel;

    const RAX: PReg = PReg { class: Class::Gpr, hw: 0 };
    const RCX: PReg = PReg { class: Class::Gpr, hw: 1 };
    const RBX: PReg = PReg { class: Class::Gpr, hw: 3 };
    const RDI: PReg = PReg { class: Class::Gpr, hw: 7 };
    const YMM0: PReg = PReg { class: Class::Vec, hw: 0 };

    /// A kernel of `body` in `frame`, allocated by hand.
    fn kernel(body: Vec<Item<Op, PReg>>, frame: Frame) -> Compiled<Op> {
        let signature = Signature { params: vec![], result: None };
        Compiled {
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
    fn code(c: &Compiled<Op>) -> String {
        intel::print(c).unwrap().code().collect::<Vec<_>>().join("\n")
    }

    /// The prologue and the epilogue are instructions of the body before the printer sees it,
    /// which it prints as it prints the rest.
    #[test]
    fn the_frame_is_instructions_before_it_is_printed() {
        let b = Slot { align: 32, ..slot("b", 32, false) };
        let frame = Frame { saved: vec![RBX], slots: vec![b], offsets: vec![0], size: 32, align: 32 };
        let store = Mem { base: Base::Slot(SlotId(0)), index: None, disp: 0, width: Width::W64 };
        let body = vec![
            mov(Op::Mov, RBX, RDI, Width::W64),
            inst(Op::Mov, vec![Arg { operand: Operand::Mem(store), access: Access::Write }, reg(RBX, Access::Read)]),
            inst(Op::Ret, vec![]),
        ];
        let c = kernel(body, frame);
        let lowered = lower(&c).unwrap();
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
        assert_eq!(code(&c), want);
        assert_eq!((lowered.returns, lowered.padding, lowered.features), (true, 0, vec![]));
    }

    /// A frame that touches more than `UNROLLED` pages touches them in a loop, at a label of its
    /// own, placed first so that it takes the first number: the least value no label of the body
    /// has, 2 beside the body's 1, 0 and `grid.sync()`'s `u32::MAX`, which come out of order.
    #[test]
    fn the_probe_loops_label_is_one_no_label_of_the_body_has() {
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
        let lowered = lower(&c).unwrap();
        let labels: Vec<_> =
            lowered.body.iter().filter_map(|item| if let Item::Label(l) = item { Some(l.0) } else { None }).collect();
        assert_eq!(labels, [2, 1, 0, u32::MAX]);
        let jne = lowered.body.iter().find_map(|item| match item {
            Item::Inst(inst) if inst.op == Op::Jne => Some(&inst.flow),
            _ => None,
        });
        assert_eq!(jne, Some(&Flow::Branch(Label(2))));
        let want = "push rbp\nmov rbp, rsp\nlea r11, [rsp - 16384]\n2:\nsub rsp, 4096\nor qword ptr [rsp], 0\n\
                    cmp rsp, r11\njne 2b\nsub rsp, 4096\n3:\n4:\n5:\nmov rsp, rbp\npop rbp\nret";
        assert_eq!(code(&c), want);
    }

    /// A move of a register to itself, of all 64 bits by `mov` or all 256 by `vmovaps`, `vmovups`,
    /// `vmovdqa` or `vmovdqu`, changes nothing, and is left out; one of 32 bits clears the upper
    /// half, and one of an xmm the upper lanes, and stay.
    #[test]
    fn a_move_of_a_register_to_itself_is_left_out() {
        let frame = Frame { saved: vec![], slots: vec![], offsets: vec![], size: 0, align: 0 };
        let body = vec![
            mov(Op::Mov, RAX, RAX, Width::W64),
            mov(Op::Mov, RAX, RAX, Width::W32),
            mov(Op::Vmovaps, YMM0, YMM0, Width::W256),
            mov(Op::Vmovups, YMM0, YMM0, Width::W256),
            mov(Op::Vmovdqa, YMM0, YMM0, Width::W256),
            mov(Op::Vmovdqu, YMM0, YMM0, Width::W256),
            mov(Op::Vmovaps, YMM0, YMM0, Width::W128),
            mov(Op::Mov, RAX, RCX, Width::W64),
            inst(Op::Ret, vec![]),
        ];
        assert_eq!(code(&kernel(body, frame)), "mov eax, eax\nvmovaps xmm0, xmm0\nmov rax, rcx\nvzeroupper\nret");
    }
}
