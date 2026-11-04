//! What a module's items are, whatever machine compiles them: the ABI each function names,
//! whether a kernel is the `unsafe fn` it must be, whose memory the module's statics are, which
//! kernels are cooperative, what is wrong with a helper, and which function an instance's name
//! names.

use kernite_lang::{Fn, ItemKind, Module, Value};

use crate::present::{Of, Refusal, said};

/// The ABIs a kernel names, or a function kernels call, each of which kernite compiles for a
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
    /// The ABI Rust writes as `name`, if kernite compiles for it.
    pub fn of(name: &str) -> Option<Abi> {
        match name {
            "sysv64" => Some(Abi::Sysv64),
            "ptx-kernel" => Some(Abi::PtxKernel),
            "ptx-func" => Some(Abi::PtxFunc),
            _ => None,
        }
    }

    /// Whether the process runs functions of this ABI, as Rust calls them: `extern "sysv64"`
    /// alone. A GPU runs the others.
    pub fn native(self) -> bool {
        match self {
            Abi::Sysv64 => true,
            Abi::PtxKernel | Abi::PtxFunc => false,
        }
    }
}

/// The ABI `f` names, if kernite compiles for it.
pub fn abi(f: &Fn) -> Option<Abi> {
    f.abi.as_ref().and_then(|(abi, _)| Abi::of(abi))
}

/// Whether `f`, of ABI `abi`, is an `unsafe fn` if it must be: a kernel trusts the pointers it is
/// given, so it is one. A function the kernels call is called by nothing else, so it promises
/// Rust nothing, and need not be.
pub(crate) fn trusted(f: &Fn, abi: Abi) -> bool {
    abi == Abi::PtxFunc || f.unsafe_fn
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
/// an ABI kernite compiles for are all `extern "sysv64"`, and else the GPU's, those of a module
/// of statics and no kernels among them. A module of both kinds of kernel and a static is two
/// modules, one of each, which the error at its first static says; its statics are then the
/// GPU's, which its x86 kernels do not reach.
pub fn owner(module: &Module) -> Result<Owner, kernite_lang::Error> {
    let abis: Vec<Abi> = module
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fn(f) => abi(f),
            _ => None,
        })
        .collect();
    if !abis.iter().any(|a| a.native()) {
        return Ok(Owner::Gpu);
    }
    if abis.iter().all(|a| a.native()) {
        return Ok(Owner::Process);
    }
    match module.items.iter().find_map(|item| match &item.kind {
        ItemKind::Static(s) => Some(s),
        _ => None,
    }) {
        Some(s) => Err(said(Refusal::TwoKinds, s.span)),
        None => Ok(Owner::Gpu),
    }
}

/// Whether `f` is a kernel whose parts wait for each other, `#[cooperative]`, which every
/// launch runs all at once: a GPU kernel's blocks, or an x86 kernel's threads. Or why it may
/// not say so, if it says so and is no kernel. One of an ABI kernite knows nothing of is none,
/// whatever it says.
pub fn cooperative(f: &Fn) -> Result<bool, kernite_lang::Error> {
    let Some(c) = &f.cooperative else { return Ok(false) };
    let of = match (abi(f), &f.abi) {
        (Some(Abi::PtxKernel | Abi::Sysv64), _) => return Ok(true),
        (Some(Abi::PtxFunc), _) => Of::Func,
        (None, Some(_)) => return Ok(false),
        (None, None) => Of::Helper,
    };
    Err(said(Refusal::Cooperative { name: &f.name, of }, c.span))
}

/// Whether `f` is cooperative, as [`cooperative`] has it, if its attributes hold: a directive,
/// `#[maxntid(..)]` and the like, goes on a GPU kernel alone, so a function that is none and says
/// one is refused, at the first it says, after its `#[cooperative]`.
pub(crate) fn attributes(f: &Fn) -> Result<bool, kernite_lang::Error> {
    let cooperative = cooperative(f)?;
    let Some(d) = f.directives.first() else { return Ok(cooperative) };
    let of = match (abi(f), &f.abi) {
        (Some(Abi::PtxKernel), _) | (None, Some(_)) => return Ok(cooperative),
        (Some(Abi::Sysv64), _) => Of::Native,
        (Some(Abi::PtxFunc), _) => Of::Func,
        (None, None) => Of::Helper,
    };
    Err(said(Refusal::Directive { directive: &d.name, name: &f.name, of }, d.span))
}

/// What is wrong with `f`, a helper of `module`, whether it is called or not: its body, if it
/// does not parse, its `#[instances(..)]`, then its attributes, in that order. The rest is
/// checked where it is called.
pub(crate) fn helper(module: &Module, f: &Fn) -> Vec<kernite_lang::Error> {
    let errors = [f.body.as_ref().err().cloned(), kernite_lang::instances(module, f).err(), attributes(f).err()];
    errors.into_iter().flatten().collect()
}

/// The function `name` of `module`.
pub fn function<'a>(module: &'a Module, name: &str) -> Option<&'a Fn> {
    module.items.iter().find_map(|item| match &item.kind {
        ItemKind::Fn(f) if f.name == name => Some(f),
        _ => None,
    })
}

/// What [`instance`] finds.
type Found<'a> = (&'a Fn, Vec<Value>, usize);

/// The function of `module` that `name` names, the compile-time arguments of the instance of it
/// that `name` is, as `tile::<4>` is, and the instance's place among those the function lists.
/// A function that names an ABI and takes no compile-time arguments has one instance, of none,
/// and a helper has none.
pub fn instance<'a>(module: &'a Module, name: &str) -> Result<Option<Found<'a>>, kernite_lang::Error> {
    let Some(f) = function(module, name.split("::<").next().unwrap_or(name)) else { return Ok(None) };
    for (at, args) in kernite_lang::instances(module, f)?.into_iter().enumerate() {
        if kernite_lang::instance(f, &args)? == name {
            return Ok(Some((f, args, at)));
        }
    }
    Ok(None)
}
