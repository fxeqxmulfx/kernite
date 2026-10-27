//! What kgen does with a module of kernels, whatever machine compiles them. It decides what the
//! module's items are: the ABI each function names, whose memory its statics are, which kernels
//! are cooperative, and which function an instance's name names. It has the ports a machine
//! implements, [`Backend`] with [`Native`] or [`Device`], and the data they hand on. Its use
//! cases call them: a module's [`statics`] laid out, what [`prepare`] gathers for every
//! kernel, an instance [`compile`]d, and the cores of kgen's façades, [`one`] kernel by its
//! name and [`all`] that a device runs. It depends on the language alone, so it cannot reach a
//! machine's code: it names the machines only in words.

mod classify;
mod dto;
mod facade;
mod function;
mod port;
mod present;

pub use classify::{Abi, Owner, abi, cooperative, function, instance, owner};
pub use dto::{Bytes, Compiled, DeviceStatic, HostStatic, Naked};
pub use facade::{all, one};
pub use function::{Prepared, compile, prepare, statics};
pub use port::{Backend, Called, Device, Native, StaticError, Unfinished};

#[cfg(test)]
mod tests;
