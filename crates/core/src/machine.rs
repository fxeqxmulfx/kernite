use crate::operand::Width;
use crate::program::{Inst, SlotId};
use crate::reg::{Class, PReg, VReg};

/// What the use cases need to know about a machine and its calling convention: which
/// registers they may hand out, and which of those a kernel must give back as it found them.
///
/// The instruction sets implement it; the allocator's tests implement it with a handful of
/// registers, to run out on purpose.
pub trait Machine {
    /// The registers of `class` a kernel may use, in the order the allocator tries them.
    fn allocatable(&self, class: Class) -> &[PReg];

    /// The registers a kernel must restore before it returns.
    fn callee_saved(&self) -> &[PReg];
}

/// How a machine of the instruction set `Op` keeps a variable on the stack when the allocator
/// runs out of registers: how wide a register is, and the instructions that store one in a
/// slot of the frame and load it back.
///
/// The allocator puts them between any two instructions, between a comparison and the jump
/// that reads its flags among them, so neither may change anything but what it writes: the
/// slot, or the register.
pub trait Spill<Op> {
    /// How wide a register of `class` is, a file [`Machine::allocatable`] has registers of:
    /// the bytes of a slot that keeps all of one, and the alignment it has.
    fn width(&self, class: Class) -> Width;

    /// An instruction that stores all of `var`'s register in `slot`.
    fn store(&self, slot: SlotId, var: VReg) -> Inst<Op, VReg>;

    /// An instruction that loads all of `var`'s register from `slot`.
    fn load(&self, var: VReg, slot: SlotId) -> Inst<Op, VReg>;
}
