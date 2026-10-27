//! The cores of kgen's façades, whatever machine compiles: one kernel of a module by its
//! name, or every function a device runs, in one module.

use kgen_lang::{Error, ItemKind, Module};

use crate::{Abi, Backend, Compiled, Device, abi, compile, cooperative, instance, prepare};

/// Function `name` of `module`, or its instance `tile::<4>`, compiled by `backend`, or why not;
/// `None` if `name` names no instance of a function, as of a helper, which has none. The first
/// static that is not laid out is the error, whatever the name.
pub fn one<B: Backend>(module: &Module, backend: &B, name: &str) -> Result<Option<Compiled<B::Code>>, Error> {
    let prepared = prepare(module, backend);
    for s in &prepared.statics {
        s.as_ref().map_err(Clone::clone)?;
    }
    let Some((f, args)) = instance(module, name)? else { return Ok(None) };
    compile(module, &prepared, backend, f, &args).map(Some)
}

/// Each function of `module` that the process does not run, each instance of each, compiled by
/// `device`, in one module with the statics; or the first error, in the module's order, a
/// static's before any function's, or why the device makes none. A helper has no instance, but
/// its `#[instances(..)]` and its `#[cooperative]` are errors.
pub fn all<D: Device>(module: &Module, device: &D) -> Result<Result<String, String>, Error> {
    let prepared = prepare(module, device);
    for s in &prepared.statics {
        s.as_ref().map_err(Clone::clone)?;
    }
    let mut codes = vec![];
    for item in &module.items {
        let ItemKind::Fn(f) = &item.kind else { continue };
        if abi(f).is_some_and(Abi::native) {
            continue;
        }
        cooperative(f)?;
        for args in kgen_lang::instances(module, f)? {
            codes.push(compile(module, &prepared, device, f, &args)?.code);
        }
    }
    Ok(device.module(&codes, &prepared.made))
}
