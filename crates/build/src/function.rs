//! The use cases for a function of a module, whatever machine compiles it: the module's
//! statics laid out, what every kernel is told, and an instance compiled.

use kernite_lang::{Error, Expr, Fn, ItemKind, Module, Static, Value};

use crate::classify::{attributes, function};
use crate::present::within;
use crate::{Backend, Called, Compiled, Done, StaticError};

/// Each static of `module`, in order, laid out by `backend`; or why not, where it can be
/// mended: at its name, its type or what it holds, as the machine says, or where the language
/// says. A module that keeps every static has room for each beside those before it that were
/// laid out, or the static is none, at its type: [`Backend::room`] says.
pub fn statics<B: Backend>(module: &Module, backend: &B) -> Vec<Result<Compiled<B::Static>, Error>> {
    let mut prepared = prepare(module, backend);
    prepared.refuse_room(module, backend);
    prepared.statics
}

/// How many bytes `statics` take of their module's memory together, each a pair of its bytes and its alignment: its
/// bytes, and the padding its alignment may need, one byte less than it. A machine that has room for so many bytes in
/// its module counts the statics it lays out with this, each beside those before it that are in the module.
pub fn taken(statics: impl IntoIterator<Item = (u64, u64)>) -> u64 {
    statics.into_iter().fold(0, |all, (bytes, align)| all.saturating_add(bytes).saturating_add(align - 1))
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

/// What [`compile`] tells every kernel of `module`: the statics `backend` lays out, and each
/// instance of each function `backend` lets its kernels call, each in the module's order. Beside
/// them, each static as [`statics`] has it, but for the room: which statics the module keeps is
/// not for this to say. A function whose instances are wrong, or whose parameters' types cannot be worked out, has no
/// instance here: it says why where it is compiled. `backend` sees each instance by the types of its parameters, worked
/// out.
pub fn prepare<B: Backend>(module: &Module, backend: &B) -> Prepared<B> {
    let (mut statics, mut made) = (vec![], vec![]);
    for item in &module.items {
        let ItemKind::Static(s) = &item.kind else { continue };
        statics.push(kernite_lang::memory(module, s).and_then(|memory| {
            let code = backend.lay_out(&s.name, &memory, &made).map_err(|e| match e {
                StaticError::Name(message) => Error { span: s.span, message },
                StaticError::Type(message) => Error { span: s.ty.span(), message },
                StaticError::Value(message) => Error { span: s.value.as_ref().map_or(s.span, Expr::span), message },
            })?;
            made.push(code.clone());
            Ok(Compiled { code, mentions: memory.mentions })
        }));
    }
    let mut callees = vec![];
    for item in &module.items {
        let ItemKind::Fn(g) = &item.kind else { continue };
        for args in kernite_lang::instances(module, g).unwrap_or_default() {
            let (Ok(name), Ok(params)) =
                (kernite_lang::instance(g, &args), kernite_lang::param_types(module, g, &args))
            else {
                continue;
            };
            if let Some(callee) = backend.callee(g, &params) {
                callees.push(Called { name, function: g.name.clone(), args, callee });
            }
        }
    }
    Prepared { statics, made, callees }
}

/// What stands for `s`, a static of `module` that `backend` does not lay out, in the code that names
/// it, beside the error that says why, if there is one: `s` as its type has it, holding zeros, if the
/// language works its type out and `backend` takes it and its name, whatever it holds, which is not
/// read; else none.
pub(crate) fn stub<B: Backend>(module: &Module, s: &Static, backend: &B) -> Option<B::Static> {
    let memory = kernite_lang::memory(module, &Static { value: None, ..s.clone() }).ok()?;
    backend.lay_out(&s.name, &memory, &[]).ok()
}

impl<B: Backend> Prepared<B> {
    /// Refuses each static `backend` has no room for beside those before it that it has, as
    /// [`Backend::room`] says: it is none, at its type. A static that was not laid out takes no
    /// room. A module that has one is not compiled, so `made` is left as it was.
    pub(crate) fn refuse_room(&mut self, module: &Module, backend: &B) {
        let mut refused = backend.room(&self.made).into_iter();
        let types = module.items.iter().filter_map(|item| match &item.kind {
            ItemKind::Static(s) => Some(s.ty.span()),
            _ => None,
        });
        for (laid, span) in self.statics.iter_mut().zip(types) {
            if laid.is_ok()
                && let Some(message) = refused.next().flatten()
            {
                *laid = Err(Error { span, message });
            }
        }
    }
}

/// Instance `args` of `f`, a function of `module`, compiled by `backend` and told what
/// `prepared` holds; or why not, where it can be mended: an instruction where it is called,
/// what the machine finds wrong with the code where the variable it blames is declared, or at
/// `f` if it blames none. For an instance, an error in its body or its code says which. The functions `backend` says it
/// calls are compiled first, each once, and the machine is given their code to finish it by: one that does not compile
/// is left out, and its error is said where it is compiled. `done` holds what compiling each instance gave so far, by
/// its name, and takes this one's and theirs: one it holds is not compiled again.
pub fn compile<B: Backend>(
    module: &Module,
    prepared: &Prepared<B>,
    backend: &B,
    f: &Fn,
    args: &[Value],
    done: &mut Vec<Done<B::Code>>,
) -> Result<Compiled<B::Code>, Error> {
    let name = kernite_lang::instance(f, args)?;
    if let Some((_, compiled)) = done.iter().find(|(done, _)| *done == name) {
        return compiled.clone();
    }
    let compiled = build(module, prepared, backend, f, args, &mut vec![], done);
    done.push((name, compiled.clone()));
    compiled
}

/// [`compile`] of an instance `done` does not hold, with `busy`, the instances whose compiling waits for this one's,
/// each calling the next.
fn build<B: Backend>(
    module: &Module,
    prepared: &Prepared<B>,
    backend: &B,
    f: &Fn,
    args: &[Value],
    busy: &mut Vec<String>,
    done: &mut Vec<Done<B::Code>>,
) -> Result<Compiled<B::Code>, Error> {
    let name = kernite_lang::instance(f, args)?;
    let cooperative = attributes(f)?;
    let mut kernel = backend.kernel(f, &name, args, cooperative, &prepared.made, &prepared.callees);
    let run = kernite_lang::kernel(module, f, args, &mut kernel).map_err(|e| within(&name, args, e))?;
    busy.push(name.clone());
    for callee in backend.calls(&kernel) {
        if busy.contains(&callee) || done.iter().any(|(called, _)| *called == callee) {
            continue;
        }
        if let Some(called) = prepared.callees.iter().find(|c| c.name == callee)
            && let Some(g) = function(module, &called.function)
        {
            let compiled = build(module, prepared, backend, g, &called.args, busy, done);
            done.push((callee, compiled));
        }
    }
    busy.pop();
    let code = backend
        .finish(kernel, done)
        .map_err(|u| within(&name, args, Error { span: u.at.unwrap_or(f.span), message: u.message }))?;
    Ok(Compiled { code, mentions: run.mentions })
}
