use kernite_core::{Class, Inst, PReg, SlotId, VReg, Width};

/// What the allocator needs to know about a machine and its calling convention: which
/// registers it may hand out, and which of those a kernel must give back as it found them.
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
/// runs out of registers, and moves one to another: how wide a register is, and the
/// instructions that store one in a slot of the frame, load it back, and copy it.
///
/// The allocator puts them between any two instructions, between a comparison and the jump
/// that reads its flags among them, so none may change anything but what it writes: the
/// slot, or the register.
pub trait Spill<Op> {
    /// How wide a register of `class` is, a file [`Machine::allocatable`] has registers of:
    /// the bytes of a slot that keeps all of one, and the alignment it has.
    fn width(&self, class: Class) -> Width;

    /// An instruction that stores all of `var`'s register in `slot`.
    fn store(&self, slot: SlotId, var: VReg) -> Inst<Op, VReg>;

    /// An instruction that loads all of `var`'s register from `slot`.
    fn load(&self, var: VReg, slot: SlotId) -> Inst<Op, VReg>;

    /// An instruction that copies all of `from`'s register to `to`'s, of the same file. The
    /// allocator moves a parameter or a result with it; a machine's own code may move a call's
    /// arguments and its result with it too, and the allocator, seeing such a move, places the
    /// two in one register where it can.
    ///
    /// The allocator knows such a move only by its being equal to what this returns: one of
    /// another width, as x86's `mov(to.r32, from.r32)` is, stays a move.
    fn copy(&self, to: VReg, from: VReg) -> Inst<Op, VReg>;
}
