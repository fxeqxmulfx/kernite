//! PTX for NVIDIA GPUs, as `kernite` writes it: the adapter between `kernite-lang` and the virtual
//! instruction set that the CUDA driver compiles for whichever GPU it finds.
//!
//! - [`lang::Kernel`] is PTX for `kernite-lang`: variables of PTX's register types, and
//!   instructions by their PTX names, `ld.global.f32(x, [p + 4])`, each checked against the
//!   forms of [`forms`], the instructions PTX has for the GPUs from Fermi, `sm_20`, to
//!   Ampere, `sm_86`; and `call`, against the [`Prototype`] of the function it calls.
//! - [`Entry`] is a kernel or a function written out: the program `kernite-compile` checks,
//!   and the `.entry` or the `.func` it prints as. [`module`] puts them in a module for the
//!   driver to load, of the PTX version and for the GPU their instructions need, with the
//!   memory of the module's its kernels name, [`Static`].
//! - [`Ptx`] is PTX for `kernite-build`, the machine its use cases compile a module's kernels
//!   and statics with: kernels and memory a driver loads as one module.
//!
//! PTX's registers have no number, and `ptxas` allocates them, so nothing here allocates: a
//! variable is a register named after it. Memory a kernel declares, [`Memory`], is named
//! after its variable too.

mod backend;
mod call;
mod entry;
pub mod forms;
pub mod lang;
mod naming;
mod needs;
mod statics;
mod ty;

pub use backend::Ptx;
pub use call::{Prototype, Returns, formal, passable, passed};
pub use entry::{Entry, Op, Param, module};
pub use naming::{ident, scope, symbol};
pub use needs::{GPUS, Needs, clash};
pub use statics::{Memory, Static, fits};
pub use ty::{Kind, Ty};
