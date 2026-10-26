//! The entities of `kgen`: a kernel as instructions over registers, before allocation and
//! after; [`Machine`], the port through which the use cases learn a machine's registers; and
//! [`Spill`], through which they keep a variable on its stack.
//!
//! Nothing here knows an instruction set. An instruction's opcode is a type parameter the
//! instruction set supplies, and what an instruction does to its operands -- reads, writes --
//! is spelled out on the operands themselves, which is all an allocator needs to know.

mod machine;
mod operand;
mod program;
mod reg;

pub use machine::{Machine, Spill};
pub use operand::{Access, Arg, Base, Imm, Mem, Operand, Width};
pub use program::{
    Compiled, Const, ConstId, Flow, Frame, Inst, Item, Label, Program, RegInfo, Signature, Slot, SlotId,
};
pub use reg::{Class, PReg, Register, VReg};
