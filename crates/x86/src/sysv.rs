use kgen_compile::{Machine, Spill};
use kgen_core::{Access, Arg, Base, Class, Flow, Inst, Mem, Operand, PReg, SlotId, VReg, Width};

use crate::op::Op;

const fn gpr(hw: u8) -> PReg {
    PReg { class: Class::Gpr, hw }
}

const fn vec(hw: u8) -> PReg {
    PReg { class: Class::Vec, hw }
}

pub(crate) const RSP: PReg = gpr(4);
pub(crate) const RBP: PReg = gpr(5);

/// `r11`, which the prologue may use before the body runs: no parameter arrives in it, and the
/// caller keeps nothing there.
pub(crate) const R11: PReg = gpr(11);

/// A page: below the end of a thread's stack lies one the thread may not touch, at least, so
/// that running past the end faults. A frame touches its pages in turn, never moving the stack
/// pointer more than this below memory it has touched, or it could step over that page into
/// memory of another's.
pub(crate) const PAGE: u32 = 4096;

/// Where the integer and pointer parameters arrive, in order.
pub(crate) const INT_PARAMS: [PReg; 6] = [gpr(7), gpr(6), gpr(2), gpr(1), gpr(8), gpr(9)];

/// Where the float parameters arrive, in order, counted apart from the integers.
pub(crate) const FLOAT_PARAMS: [PReg; 8] = [vec(0), vec(1), vec(2), vec(3), vec(4), vec(5), vec(6), vec(7)];

/// Where a float result goes back.
pub(crate) const FLOAT_RESULT: PReg = vec(0);

/// Where an integer or pointer result goes back.
pub(crate) const INT_RESULT: PReg = gpr(0);

/// The caller-saved registers first, so that a kernel that fits in them saves nothing. The
/// stack pointer is never handed out, and neither is `rbp`, which holds the caller's stack
/// pointer while a frame is realigned.
const GPRS: [PReg; 14] = [
    gpr(0),
    gpr(1),
    gpr(2),
    gpr(6),
    gpr(7),
    gpr(8),
    gpr(9),
    gpr(10),
    gpr(11),
    gpr(3),
    gpr(12),
    gpr(13),
    gpr(14),
    gpr(15),
];

const VECS: [PReg; 16] = [
    vec(0),
    vec(1),
    vec(2),
    vec(3),
    vec(4),
    vec(5),
    vec(6),
    vec(7),
    vec(8),
    vec(9),
    vec(10),
    vec(11),
    vec(12),
    vec(13),
    vec(14),
    vec(15),
];

const SAVED: [PReg; 5] = [gpr(3), gpr(12), gpr(13), gpr(14), gpr(15)];

/// The registers a callee may change, which a call clobbers: every one but the callee-saved, the
/// stack pointer and `rbp`.
pub(crate) const CALL_CLOBBERS: [PReg; 25] = [
    gpr(0),
    gpr(1),
    gpr(2),
    gpr(6),
    gpr(7),
    gpr(8),
    gpr(9),
    gpr(10),
    gpr(11),
    vec(0),
    vec(1),
    vec(2),
    vec(3),
    vec(4),
    vec(5),
    vec(6),
    vec(7),
    vec(8),
    vec(9),
    vec(10),
    vec(11),
    vec(12),
    vec(13),
    vec(14),
    vec(15),
];

/// x86-64 under the System V calling convention: sixteen general registers and sixteen
/// vector ones, every vector register the caller's to save. A general register spills with
/// `mov`, a vector one with `vmovups`, all 256 bits, to a slot aligned to its size, and
/// moves to another register whole too, with `mov` or `vmovaps`; none touches the flags.
#[derive(Clone, Copy, Debug, Default)]
pub struct SysV;

impl Machine for SysV {
    fn allocatable(&self, class: Class) -> &[PReg] {
        match class {
            Class::Gpr => &GPRS,
            Class::Vec => &VECS,
            Class::Pred | Class::Float => &[],
        }
    }

    fn callee_saved(&self) -> &[PReg] {
        &SAVED
    }
}

impl SysV {
    /// The instruction that moves all of `var` to `slot`, or back.
    fn spill(&self, slot: SlotId, var: VReg, store: bool) -> Inst<Op, VReg> {
        let width = self.width(var.class);
        let op = if var.class == Class::Gpr { Op::Mov } else { Op::Vmovups };
        let mem = Operand::Mem(Mem { base: Base::Slot(slot), index: None, disp: 0, width });
        let reg = Operand::Reg(var, width);
        let (to, from) = if store { (mem, reg) } else { (reg, mem) };
        let args = vec![Arg { operand: to, access: Access::Write }, Arg { operand: from, access: Access::Read }];
        Inst { op, args, implicit: vec![], flow: Flow::Next, clobbers: vec![] }
    }
}

impl Spill<Op> for SysV {
    fn width(&self, class: Class) -> Width {
        match class {
            Class::Gpr => Width::W64,
            Class::Vec => Width::W256,
            Class::Pred | Class::Float => unreachable!("x86 has no {class:?} registers"),
        }
    }

    fn store(&self, slot: SlotId, var: VReg) -> Inst<Op, VReg> {
        self.spill(slot, var, true)
    }

    fn load(&self, var: VReg, slot: SlotId) -> Inst<Op, VReg> {
        self.spill(slot, var, false)
    }

    fn copy(&self, to: VReg, from: VReg) -> Inst<Op, VReg> {
        let width = self.width(to.class);
        let op = if to.class == Class::Gpr { Op::Mov } else { Op::Vmovaps };
        let arg = |v, access| Arg { operand: Operand::Reg(v, width), access };
        Inst {
            op,
            args: vec![arg(to, Access::Write), arg(from, Access::Read)],
            implicit: vec![],
            flow: Flow::Next,
            clobbers: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A call changes every register the allocator hands out but those SysV has the callee save.
    #[test]
    fn a_call_changes_every_register_sysv_does_not_save() {
        let mut want: Vec<PReg> = GPRS.iter().chain(&VECS).filter(|r| !SAVED.contains(r)).copied().collect();
        let mut clobbers = CALL_CLOBBERS.to_vec();
        want.sort();
        clobbers.sort();
        assert_eq!(clobbers, want);
    }
}
