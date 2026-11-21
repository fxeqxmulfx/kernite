//! The allocator of `kernite`: from a [`Program`] over variables to an [`Allocated`] kernel over
//! machine registers, what did not fit in them spilled to the stack, with its frame laid out.
//!
//! What it knows of the machine comes through the [`Machine`] and [`Spill`] ports, so it
//! runs the same for any instruction set, and its tests run it on a toy machine with a few
//! registers.
//!
//! [`on_every_path`] says which items of a program every path from its start to its end runs, and [`forward`] follows a
//! value over its flow.

mod alloc;
mod allocated;
mod bits;
mod cfg;
mod error;
mod frame;
mod live;
mod port;
mod spill;

pub use allocated::{Allocated, Frame};
pub use cfg::{forward, on_every_path};
pub use error::Error;
use kernite_core::{Inst, Item, PReg, Program};
pub use port::{Machine, Spill};

/// What [`compile`] checks before it allocates, for a machine that allocates its own
/// registers: that control stays inside the body, that every jump has its label, and that
/// every variable but the parameters is written before any path reads it.
pub fn check<Op>(p: &Program<Op>) -> Result<(), Error> {
    live::analyse(p).map(|_| ())
}

/// Allocates `p`'s variables to `m`'s registers, spilling to the stack the ones that do not
/// fit, and lays out its frame.
pub fn compile<Op: Clone + PartialEq>(p: &Program<Op>, m: &(impl Machine + Spill<Op>)) -> Result<Allocated<Op>, Error> {
    let (p, regs) = alloc::allocate(p, m, m)?;
    let reg = |v: kernite_core::VReg| {
        let at = regs.get(v.id as usize).copied().flatten();
        at.ok_or_else(|| Error::NoRegister { var: p.var_name(v).to_string(), vreg: v })
    };
    let body = p
        .body
        .iter()
        .map(|item| match item {
            Item::Label(l) => Ok(Item::Label(*l)),
            Item::Note(note) => Ok(Item::Note(note.clone())),
            Item::Inst(inst) => Ok(Item::Inst(Inst {
                op: inst.op.clone(),
                args: inst.args.iter().map(|arg| arg.try_map(reg)).collect::<Result<_, _>>()?,
                implicit: inst.implicit.iter().map(|arg| arg.try_map(reg)).collect::<Result<_, _>>()?,
                flow: inst.flow.clone(),
                clobbers: inst.clobbers.clone(),
            })),
        })
        .collect::<Result<_, Error>>()?;
    let mut used: Vec<PReg> = regs.iter().flatten().copied().collect();
    used.sort();
    used.dedup();
    Ok(Allocated {
        name: p.name.clone(),
        signature: p.signature.clone(),
        body,
        frame: frame::layout(&p.slots, &used, m),
        consts: p.consts.clone(),
        funcs: p.funcs.clone(),
        statics: p.statics.clone(),
        assignment: p.regs.iter().cloned().zip(regs).collect(),
    })
}

#[cfg(test)]
mod tests;
