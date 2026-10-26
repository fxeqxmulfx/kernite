//! The use cases of `kgen`: from a [`Program`] over variables to a [`Compiled`] kernel over
//! machine registers, what did not fit in them spilled to the stack, with its frame laid out.
//!
//! What they know of the machine comes through the [`Machine`] and [`Spill`] ports, so they
//! run the same for any instruction set, and their tests run them on a toy machine with a few
//! registers.

mod alloc;
mod bits;
mod cfg;
mod error;
mod frame;
mod spill;

pub use error::Error;
use kgen_core::{Compiled, Inst, Item, Machine, PReg, Program, Spill};

/// What [`compile`] checks before it allocates, for a machine that allocates its own
/// registers: that control stays inside the body, that every jump has its label, and that
/// every variable but the parameters is written before any path reads it.
pub fn check<Op>(p: &Program<Op>) -> Result<(), Error> {
    alloc::analyse(p).map(|_| ())
}

/// Allocates `p`'s variables to `m`'s registers, spilling to the stack the ones that do not
/// fit, and lays out its frame.
pub fn compile<Op: Clone + PartialEq>(p: &Program<Op>, m: &(impl Machine + Spill<Op>)) -> Result<Compiled<Op>, Error> {
    let (p, regs) = alloc::allocate(p, m, m)?;
    let reg = |v: kgen_core::VReg| regs[v.id as usize].expect("every variable named has a register");
    let body = p
        .body
        .iter()
        .map(|item| match item {
            Item::Label(l) => Item::Label(*l),
            Item::Note(note) => Item::Note(note.clone()),
            Item::Inst(inst) => Item::Inst(Inst {
                op: inst.op.clone(),
                args: inst.args.iter().map(|arg| arg.map(reg)).collect(),
                implicit: inst.implicit.iter().map(|arg| arg.map(reg)).collect(),
                flow: inst.flow.clone(),
                clobbers: inst.clobbers.clone(),
            }),
        })
        .collect();
    let mut used: Vec<PReg> = regs.iter().flatten().copied().collect();
    used.sort();
    used.dedup();
    Ok(Compiled {
        name: p.name.clone(),
        signature: p.signature.clone(),
        body,
        frame: frame::layout(&p.slots, &used, m),
        consts: p.consts.clone(),
        funcs: p.funcs.clone(),
        assignment: p.regs.iter().cloned().zip(regs).collect(),
    })
}

#[cfg(test)]
mod tests;
