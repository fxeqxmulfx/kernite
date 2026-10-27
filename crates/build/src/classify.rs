//! What a module's items are, whatever machine compiles them: the ABI each function names, whose
//! memory its statics are, which kernels are cooperative, and which function an instance's name
//! names.

use kgen_lang::{Fn, ItemKind, Module, Value};

/// The ABIs a kernel names, or a function kernels call, each of which kgen compiles for a
/// machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abi {
    /// `extern "sysv64"`: x86-64 under the System V calling convention.
    Sysv64,
    /// `extern "ptx-kernel"`: an NVIDIA GPU, in PTX.
    PtxKernel,
    /// `extern "ptx-func"`: a function the PTX kernels of its module call, `.func`, of PTX's
    /// types.
    PtxFunc,
}

impl Abi {
    /// The ABI Rust writes as `name`, if kgen compiles for it.
    pub fn of(name: &str) -> Option<Abi> {
        match name {
            "sysv64" => Some(Abi::Sysv64),
            "ptx-kernel" => Some(Abi::PtxKernel),
            "ptx-func" => Some(Abi::PtxFunc),
            _ => None,
        }
    }
}

/// The ABI `f` names, if kgen compiles for it.
pub fn abi(f: &Fn) -> Option<Abi> {
    f.abi.as_ref().and_then(|(abi, _)| Abi::of(abi))
}

/// Whose memory a module's statics are: its kernels'.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    /// The process's, for x86 kernels.
    Process,
    /// The GPU's, for PTX kernels.
    Gpu,
}

/// Whose memory the statics of `module` are, its kernels': the process's when its functions of
/// an ABI kgen compiles for are all `extern "sysv64"`, and else the GPU's, those of a module of
/// statics and no kernels among them. A module of both kinds of kernel and a static is two
/// modules, one of each, which the error at its first static says; its statics are then the
/// GPU's, which its x86 kernels do not reach.
pub fn owner(module: &Module) -> Result<Owner, kgen_lang::Error> {
    let abis: Vec<Abi> = module
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fn(f) => abi(f),
            _ => None,
        })
        .collect();
    if !abis.contains(&Abi::Sysv64) {
        return Ok(Owner::Gpu);
    }
    if abis.iter().all(|&a| a == Abi::Sysv64) {
        return Ok(Owner::Process);
    }
    match module.items.iter().find_map(|item| match &item.kind {
        ItemKind::Static(s) => Some(s),
        _ => None,
    }) {
        Some(s) => Err(kgen_lang::Error {
            span: s.span,
            message: "a static is memory of the module's kernels: the GPU's, for PTX kernels, or the process's, \
                      for x86 ones; and this module has both, so it is two modules, one of each kind of kernel and \
                      the statics they name"
                .to_string(),
        }),
        None => Ok(Owner::Gpu),
    }
}

/// Whether `f` is a kernel whose parts wait for each other, `#[cooperative]`, which every
/// launch runs all at once: a GPU kernel's blocks, or an x86 kernel's threads. Or why it may
/// not say so, if it says so and is no kernel. One of an ABI kgen knows nothing of is none,
/// whatever it says.
pub fn cooperative(f: &Fn) -> Result<bool, kgen_lang::Error> {
    let Some(c) = &f.cooperative else { return Ok(false) };
    let what = match (abi(f), &f.abi) {
        (Some(Abi::PtxKernel | Abi::Sysv64), _) => return Ok(true),
        (Some(Abi::PtxFunc), _) => "a function the kernels call, which runs as the kernel that calls it is launched",
        (None, Some(_)) => return Ok(false),
        (None, None) => "a helper, inlined where it is called",
    };
    let message = format!(
        "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \"ptx-kernel\"` or \
         `extern \"sysv64\"`; `{}` is {what}",
        f.name
    );
    Err(kgen_lang::Error { span: c.span, message })
}

/// The function `name` of `module`.
pub fn function<'a>(module: &'a Module, name: &str) -> Option<&'a Fn> {
    module.items.iter().find_map(|item| match &item.kind {
        ItemKind::Fn(f) if f.name == name => Some(f),
        _ => None,
    })
}

/// The function of `module` that `name` names, with the compile-time arguments of its instance
/// that `name` is, as `tile::<4>` is; none for a function that takes none.
pub fn instance<'a>(module: &'a Module, name: &str) -> Result<Option<(&'a Fn, Vec<Value>)>, kgen_lang::Error> {
    let Some(f) = function(module, name.split("::<").next().unwrap_or(name)) else { return Ok(None) };
    for args in kgen_lang::instances(module, f)? {
        if kgen_lang::instance(f, &args)? == name {
            return Ok(Some((f, args)));
        }
    }
    Ok(None)
}
