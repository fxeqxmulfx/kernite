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
/// runs out of registers, and moves one to another: how wide a register is, how wide a variable is kept, and the
/// instructions that store one in a slot of the frame, load it back, and copy it.
///
/// The allocator puts them between any two instructions, between a comparison and the jump
/// that reads its flags among them, so none may change anything but what it writes: the
/// slot, or the register.
pub trait Spill<Op> {
    /// How wide a register of `class` is, a file [`Machine::allocatable`] has registers of:
    /// all of one. An operand of less names part of it.
    fn width(&self, class: Class) -> Width;

    /// How wide a variable of `class` is kept, in its slot and in the moves that copy it, where the widest operand that
    /// names it is `named`: at least that, and at most [`Spill::width`]. A machine may keep a variable in part of a
    /// register where no instruction names more of it, a vector register's lower half say, and then its slot has as
    /// many bytes, and the alignment they say. The registers of an address count as all of a register, for an address
    /// does not say how much of one it reads.
    fn kept(&self, class: Class, named: Width) -> Width;

    /// An instruction that stores `width` of `var`'s register in `slot`.
    fn store(&self, slot: SlotId, var: VReg, width: Width) -> Inst<Op, VReg>;

    /// An instruction that loads `width` of `var`'s register from `slot`.
    fn load(&self, var: VReg, slot: SlotId, width: Width) -> Inst<Op, VReg>;

    /// An instruction that copies `width` of `from`'s register to `to`'s, of the same file. The allocator moves a
    /// parameter or a result with it, and a spilled variable that a register holds, in place of its load; a machine's
    /// own code may move a call's arguments and its result with it too.
    fn copy(&self, to: VReg, from: VReg, width: Width) -> Inst<Op, VReg>;

    /// What `inst` copies, if it moves `width` of a register to another of the same file, or to itself: the variable
    /// it writes, the one it reads, and `width`. Each instruction [`Spill::copy`] makes is one, and so is each other
    /// spelling of the move, as x86's `vmovups` between two vector registers is.
    ///
    /// The allocator leaves a copy out where it changes nothing, and has it read another register that holds the value,
    /// where that is as good; where one of its two variables is a call's argument or result, or one that splitting
    /// made, it places the two in one register where it can. It knows a move as a copy only by this, at the width `to`
    /// is kept: one of another width, as x86's `mov(to.r32, from.r32)` is, stays a move. Where `from` is pinned, a
    /// call's result, the width is that `from` is kept at, what the callee gives back, though `to` may be kept in more.
    /// A copy of a variable to itself is left out where it moves all of the register, which is as wide as
    /// [`Spill::width`] says, and stays where it moves less, which may change the rest, as x86's `mov` of 32 bits
    /// clears the upper half.
    fn copies(&self, inst: &Inst<Op, VReg>) -> Option<(VReg, VReg, Width)>;
}
