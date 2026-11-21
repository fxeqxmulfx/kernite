//! A variable of a kernel, which an operand of an instruction names: what the language's binding to PTX and the proof
//! that a kernel is safe both read.

use kernite_core::VReg;

use crate::forms::Part;
use crate::ty::Ty;

/// A variable: a register of the type it was declared, or a part of one, or memory the
/// kernel declares in a state space, which stands for its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Var {
    /// A register.
    Reg {
        /// Which.
        reg: VReg,
        /// Its type.
        ty: Ty,
    },
    /// A part of a 32-bit register, as a video instruction selects it: `r.b0`.
    Part {
        /// Which register.
        reg: VReg,
        /// Its type.
        ty: Ty,
        /// Which part.
        part: Part,
    },
    /// Memory, by its place among the kernel's.
    Mem(usize),
}
