//! What kgen decides of a module of kernels, whatever machine compiles them: the ABI each
//! function names, whose memory its statics are, which kernels are cooperative, and which
//! function an instance's name names. It depends on the language alone, so it cannot reach a
//! machine's code: it names the machines only in words.

mod classify;
mod dto;

pub use classify::{Abi, Owner, abi, cooperative, function, instance, owner};
pub use dto::Compiled;

#[cfg(test)]
mod tests;
