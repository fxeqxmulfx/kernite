use kgen_core::{Const, Item, PReg, RegInfo, Signature, Slot};

/// The stack frame of an allocated kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Frame {
    /// The callee-saved registers the body writes, to save on entry and restore at every
    /// return, in the order they are saved.
    pub saved: Vec<PReg>,
    /// The slots, by [`kgen_core::SlotId`].
    pub slots: Vec<Slot>,
    /// Each slot's offset from the stack pointer once it is aligned.
    pub offsets: Vec<u32>,
    /// Bytes the slots take together, a multiple of `align`; 0 with no slots.
    pub size: u32,
    /// The alignment the slots need; 0 with no slots.
    pub align: u32,
}

/// A kernel after allocation: machine registers only, and a frame laid out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allocated<Op> {
    /// What it is called.
    pub name: String,
    /// What it takes and gives.
    pub signature: Signature,
    /// The code.
    pub body: Vec<Item<Op, PReg>>,
    /// Its stack frame.
    pub frame: Frame,
    /// Read-only data, by [`kgen_core::ConstId`].
    pub consts: Vec<Const>,
    /// The functions it names, by [`kgen_core::FuncId`].
    pub funcs: Vec<String>,
    /// The memory of the module's it names, by [`kgen_core::StaticId`].
    pub statics: Vec<String>,
    /// Every variable and the register it got, by [`kgen_core::VReg::id`]: the program's, then
    /// those the allocator made, each named as the one it was made for is: to load and store one
    /// that spilled, or to hold a parameter or a result that moved. `None` for one the code does
    /// not name, as one that spilled is not.
    pub assignment: Vec<(RegInfo, Option<PReg>)>,
}
