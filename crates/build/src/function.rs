//! The use cases for a function of a module, whatever machine compiles it: the module's
//! statics laid out, what every kernel is told, and an instance compiled.

use kernite_lang::{Error, Expr, Fn, ItemKind, Module, Value};

use crate::classify::attributes;
use crate::present::within;
use crate::{Backend, Called, Compiled, StaticError};

/// Each static of `module`, in order, laid out by `backend` after those before it that were;
/// or why not, where it can be mended: at its name, its type or what it holds, as the machine
/// says, or where the language says.
pub fn statics<B: Backend>(module: &Module, backend: &B) -> Vec<Result<Compiled<B::Static>, Error>> {
    prepare(module, backend).statics
}

/// What every kernel of a module is told: its statics, and the functions it may call.
pub struct Prepared<B: Backend> {
    /// Each static of the module, as [`statics`] has it.
    pub(crate) statics: Vec<Result<Compiled<B::Static>, Error>>,
    /// The statics laid out, in order.
    pub(crate) made: Vec<B::Static>,
    /// Each instance of each function a kernel may call, in the module's order.
    pub(crate) callees: Vec<Called<B::Callee>>,
}

/// What [`compile`] tells every kernel of `module`: the statics `backend` lays out, each after
/// those before it that were, and each instance of each function `backend` lets its kernels
/// call, each in the module's order. Beside them, each static as [`statics`] has it. A function
/// whose instances are wrong has no instance here: it says why where it is compiled.
pub fn prepare<B: Backend>(module: &Module, backend: &B) -> Prepared<B> {
    let mut statics = vec![];
    let mut made = vec![];
    for item in &module.items {
        let ItemKind::Static(s) = &item.kind else { continue };
        let laid = kernite_lang::memory(module, s).and_then(|memory| {
            let code = backend.lay_out(&s.name, &memory, &made).map_err(|e| match e {
                StaticError::Name(message) => Error { span: s.span, message },
                StaticError::Type(message) => Error { span: s.ty.span(), message },
                StaticError::Value(message) => Error { span: s.value.as_ref().map_or(s.span, Expr::span), message },
            })?;
            Ok(Compiled { code, consts: memory.consts })
        });
        if let Ok(laid) = &laid {
            made.push(laid.code.clone());
        }
        statics.push(laid);
    }
    let mut callees = vec![];
    for item in &module.items {
        let ItemKind::Fn(g) = &item.kind else { continue };
        let Some(callee) = backend.callee(g) else { continue };
        for args in kernite_lang::instances(module, g).unwrap_or_default() {
            if let Ok(name) = kernite_lang::instance(g, &args) {
                callees.push(Called { name, function: g.name.clone(), args, callee: callee.clone() });
            }
        }
    }
    Prepared { statics, made, callees }
}

/// Instance `args` of `f`, a function of `module`, compiled by `backend` and told what
/// `prepared` holds; or why not, where it can be mended: an instruction where it is called,
/// what the machine finds wrong with the code where the variable it blames is declared, or at
/// `f` if it blames none. For an instance, an error in its body or its code says which.
pub fn compile<B: Backend>(
    module: &Module,
    prepared: &Prepared<B>,
    backend: &B,
    f: &Fn,
    args: &[Value],
) -> Result<Compiled<B::Code>, Error> {
    let name = kernite_lang::instance(f, args)?;
    let cooperative = attributes(f)?;
    let mut kernel = backend.kernel(f, &name, args, cooperative, &prepared.made, &prepared.callees);
    let run = kernite_lang::kernel(module, f, args, &mut kernel).map_err(|e| within(&name, args, e))?;
    let code = backend
        .finish(kernel)
        .map_err(|u| within(&name, args, Error { span: u.at.unwrap_or(f.span), message: u.message }))?;
    Ok(Compiled { code, consts: run.consts })
}
