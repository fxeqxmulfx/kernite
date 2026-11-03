//! The entities of `kernite`: a kernel as instructions over registers, before allocation and
//! after.
//!
//! Nothing here knows an instruction set. An instruction's opcode is a type parameter the
//! instruction set supplies, and what an instruction does to its operands -- reads, writes --
//! is spelled out on the operands themselves, which is all an allocator needs to know.

mod operand;
mod program;
mod reg;

pub use operand::{Access, Arg, Base, ConstId, FuncId, Imm, Label, Mem, Operand, SlotId, StaticId, Width};
pub use program::{Const, Flow, Inst, Item, Program, RegInfo, Signature, Slot};
pub use reg::{Class, PReg, Register, VReg};
