//! The cores of kernite's façades, whatever machine compiles: one kernel of a module by its
//! name, or every function a device runs, in one module.

use kernite_lang::{Error, ItemKind, Module};

use crate::classify::{attributes, helper, named, trusted};
use crate::present::{Refusal, said};
use crate::{Abi, Compiled, Device, Native, abi, compile, instance, owner, prepare};

/// Why a name names no instance of a function the process runs, for [`one`] to compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Absent<'a> {
    /// The GPU runs the function of this name, `extern "ptx-kernel"` or `extern "ptx-func"`, and the
    /// name given is its own or an instance's of it, one it lists or not.
    Gpu(&'a str),
    /// The process runs the function of this name, and the name given is of no instance of it:
    /// `sum4::<2>`, where `sum4` takes no compile-time arguments or lists no 2 in its
    /// `#[instances(..)]`, or `tile`, where `tile` takes some.
    Instance(&'a str),
    /// No function the process runs has the name: no function has it, or it is a helper's, or its
    /// function names an ABI kernite knows nothing of.
    Kernel,
}

/// Function `name` of `module`, or its instance `tile::<4>`, compiled by `native`, or why not:
/// where `name` names no instance of a function the process runs, [`Absent`] says why. The first
/// static that is not laid out is the error, whatever the name. Once the instance has compiled,
/// what is wrong with the rest of the module is the error as well, as the macro says it: the module
/// is two modules, a helper does not parse, a function names an ABI kernite knows nothing of, a
/// kernel is not an `unsafe fn`, or a function's `#[cooperative]`, directive or `#[instances(..)]` is
/// wrong.
pub fn one<'a, N: Native>(
    module: &'a Module,
    native: &N,
    name: &str,
) -> Result<Result<Compiled<N::Code>, Absent<'a>>, Error> {
    let prepared = prepare(module, native);
    for s in &prepared.statics {
        s.as_ref().map_err(Clone::clone)?;
    }
    let Some(f) = named(module, name) else { return Ok(Err(Absent::Kernel)) };
    let found = instance(module, name)?;
    match abi(f) {
        None => return Ok(Err(Absent::Kernel)),
        Some(a) if !a.native() => return Ok(Err(Absent::Gpu(&f.name))),
        Some(_) => {}
    }
    let Some((_, args, _)) = found else { return Ok(Err(Absent::Instance(&f.name))) };
    let compiled = compile(module, &prepared, native, f, &args)?;
    rules(module)?;
    Ok(Ok(compiled))
}

/// Each function of `module` that the GPU runs, each instance of each, compiled by `device`, in
/// one module with the statics; or the first error, in the module's order, a static's before any
/// function's, or why the device makes none. A helper has no instance, but its
/// `#[instances(..)]`, its `#[cooperative]` and its directives are errors. Once the device has
/// made its module, what is wrong with the rest of the module is the error as well, as `one`
/// says.
pub fn all<D: Device>(module: &Module, device: &D) -> Result<Result<String, String>, Error> {
    let mut prepared = prepare(module, device);
    prepared.refuse_room(module, device);
    for s in &prepared.statics {
        s.as_ref().map_err(Clone::clone)?;
    }
    let mut codes = vec![];
    for item in &module.items {
        let ItemKind::Fn(f) = &item.kind else { continue };
        // Passed over: a function the process runs, and one of an ABI kernite knows nothing of. A
        // helper stays, whose attributes are errors here.
        if f.abi.is_some() && abi(f).is_none_or(Abi::native) {
            continue;
        }
        attributes(f)?;
        for args in kernite_lang::instances(module, f)? {
            codes.push(compile(module, &prepared, device, f, &args)?.code);
        }
    }
    let made = device.module(&codes, &prepared.made).whole();
    if made.is_ok() {
        rules(module)?;
    }
    Ok(made)
}

/// What is wrong with `module` that no compile finds, as the macro says it, the first error, in
/// this order: that it is two modules; then, for each function in the module's order, a
/// helper's first error, an ABI kernite knows nothing of, a kernel that is not an `unsafe fn`,
/// its `#[cooperative]`, its directives and its `#[instances(..)]`.
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
        attributes(f)?;
        kernite_lang::instances(module, f)?;
    }
    Ok(())
}
