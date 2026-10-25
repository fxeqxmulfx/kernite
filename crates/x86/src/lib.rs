//! x86-64 with AVX2, FMA and F16C, as `kgen` sees it: the adapter between the entities of
//! `kgen-core` and this instruction set.
//!
//! - [`lang::Kernel`] is x86 for `kgen-lang`, the language kernels are written in where
//!   they are called: its types and views, and every instruction it calls by name, checked
//!   against the forms x86 has for it.
//! - [`Asm`] builds the same programs from Rust: one method per instruction, operands typed
//!   by what the instruction accepts, variables instead of registers.
//! - [`SysV`] is the [`kgen_core::Machine`] the allocator runs against: the registers of the
//!   System V calling convention and the ones it wants back.
//! - [`intel::print`] turns an allocated kernel into Intel-syntax lines, as `naked_asm!`
//!   takes them, and a listing for people.

mod asm;
pub mod intel;
pub mod lang;
mod op;
mod operand;
mod sysv;

pub use asm::{Asm, Stream};
pub use op::Op;
pub use operand::{Addr, Gpr, Hint, Slot, Src, Vector, Xmm, Ymm};
pub use sysv::SysV;
