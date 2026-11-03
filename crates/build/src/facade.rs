//! The cores of kgen's façades, whatever machine compiles: one kernel of a module by its
//! name, or every function a device runs, in one module.

use kgen_lang::{Error, ItemKind, Module};

use crate::classify::{helper, trusted};
use crate::present::{Refusal, said};
use crate::{Abi, Compiled, Device, Native, abi, compile, cooperative, instance, owner, prepare};

/// Function `name` of `module`, or its instance `tile::<4>`, compiled by `native`, or why not;
/// `None` if `name` names no instance of a function the process runs: a helper has none, and
/// `native` compiles no function of the GPU's nor one of an ABI kgen knows nothing of. The first
/// static that is not laid out is the error, whatever the name. Once the instance has compiled,
/// what is wrong with the rest of the module is the error as well, as the macro says it: the
/// module is two modules, a helper does not parse, a function names an ABI kgen knows nothing
/// of, a kernel is not an `unsafe fn`, or a function's `#[cooperative]` or `#[instances(..)]` is
/// wrong.
pub fn one<N: Native>(module: &Module, native: &N, name: &str) -> Result<Option<Compiled<N::Code>>, Error> {
    let prepared = prepare(module, native);
    for s in &prepared.statics {
        s.as_ref().map_err(Clone::clone)?;
    }
    let Some((f, args)) = instance(module, name)? else { return Ok(None) };
    if !abi(f).is_some_and(Abi::native) {
        return Ok(None);
    }
    let compiled = compile(module, &prepared, native, f, &args)?;
    rules(module)?;
    Ok(Some(compiled))
}

/// Each function of `module` that the GPU runs, each instance of each, compiled by `device`, in
/// one module with the statics; or the first error, in the module's order, a static's before any
/// function's, or why the device makes none. A helper has no instance, but its
/// `#[instances(..)]` and its `#[cooperative]` are errors. Once the device has made its module,
/// what is wrong with the rest of the module is the error as well, as `one` says.
pub fn all<D: Device>(module: &Module, device: &D) -> Result<Result<String, String>, Error> {
    let prepared = prepare(module, device);
    for s in &prepared.statics {
        s.as_ref().map_err(Clone::clone)?;
    }
    let mut codes = vec![];
    for item in &module.items {
        let ItemKind::Fn(f) = &item.kind else { continue };
        // Passed over: a function the process runs, and one of an ABI kgen knows nothing of. A
        // helper stays, whose attributes are errors here.
        if f.abi.is_some() && abi(f).is_none_or(Abi::native) {
            continue;
        }
        cooperative(f)?;
        for args in kgen_lang::instances(module, f)? {
            codes.push(compile(module, &prepared, device, f, &args)?.code);
        }
    }
    let made = device.module(&codes, &prepared.made);
    if made.is_ok() {
        rules(module)?;
    }
    Ok(made)
}

/// What is wrong with `module` that no compile finds, as the macro says it, the first error, in
/// this order: that it is two modules; then, for each function in the module's order, a
/// helper's first error, an ABI kgen knows nothing of, a kernel that is not an `unsafe fn`, its
/// `#[cooperative]` and its `#[instances(..)]`.
fn rules(module: &Module) -> Result<(), Error> {
    owner(module)?;
    for item in &module.items {
        let ItemKind::Fn(f) = &item.kind else { continue };
        let Some((name, at)) = &f.abi else {
            if let Some(e) = helper(module, f).into_iter().next() {
                return Err(e);
            }
            continue;
        };
        let Some(abi) = Abi::of(name) else { return Err(said(Refusal::UnknownAbi { abi: name }, *at)) };
        if !trusted(f, abi) {
            return Err(said(Refusal::NotUnsafe, f.span));
        }
        cooperative(f)?;
        kgen_lang::instances(module, f)?;
    }
    Ok(())
}
