//! The entities of `kgen`: a kernel as instructions over registers, before allocation and
//! after.
//!
//! Nothing here knows an instruction set. An instruction's opcode is a type parameter the
//! instruction set supplies, and what an instruction does to its operands -- reads, writes --
//! is spelled out on the operands themselves, which is all an allocator needs to know.

mod operand;
mod program;
mod reg;

pub use operand::{Access, Arg, Base, Imm, Mem, Operand, Width};
pub use program::{
    Const, ConstId, Flow, FuncId, Inst, Item, Label, Program, RegInfo, Signature, Slot, SlotId, StaticId,
};
pub use reg::{Class, PReg, Register, VReg};
