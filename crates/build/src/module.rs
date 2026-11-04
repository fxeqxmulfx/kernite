//! The use case for a module of kernels as a whole, whatever machines compile it: what each of
//! its items is once its kernels are compiled, for whatever writes the module out for Rust.

use std::cell::LazyCell;

use kernite_lang::{Error, Item, ItemKind, Module};

use crate::classify::{helper, trusted};
use crate::present::{Refusal, said};
use crate::{
    Abi, Call, Compiled, Device, DeviceFnOut, DeviceOut, Given, ItemOut, Launcher, ModuleOut, Naked, Native,
    NativeInstance, NativeOut, Owner, compile, cooperative, instance, owner, prepare,
};

/// `module`, given the consts `given` says, its kernels compiled: each function the process runs
/// by `native`, and each the GPU runs by `device` into one module the GPU loads. Its statics are
/// laid out by the machine whose memory they are, and by `native` for its kernels to name. What
/// is wrong is said where it can be mended. A kernel the process runs is written for Rust to call
/// when it is an `unsafe fn`, its attributes hold, each of its instances compiled, and it names no
/// static of the GPU's. The GPU's module has the code of each instance that compiled, of each
/// function the GPU runs whose attributes hold; Rust launches a kernel of it when it is an
/// `unsafe fn`, its attributes hold, and each of its instances compiled. Why the GPU's module is
/// none is said at the module's first item that is the GPU's.
pub fn module<N: Native, D: Device>(mut module: Module, given: Option<Given>, native: &N, device: &D) -> ModuleOut {
    let twice = given.as_ref().map_or_else(Vec::new, |given| take(&mut module, given));
    let before = given.map_or(0, |given| given.before);
    let whose = owner(&module);
    let process = whose == Ok(Owner::Process);
    let m = &module;
    // Each machine's statics and callees, worked out when something first needs them.
    let n = LazyCell::new(|| prepare(m, native));
    let d = LazyCell::new(|| prepare(m, device));
    // The statics are the process's or the GPU's.
    let (mut host, mut gpu) = match process {
        true => (n.statics.iter(), [].iter()),
        false => ([].iter(), d.statics.iter()),
    };
    let (mut items, mut listings, mut codes, mut symbols, mut first) = (vec![], vec![], vec![], vec![], None);
    for item in &m.items {
        let f = match &item.kind {
            ItemKind::Const(_) if item.tokens.start < before => {
                items.push(ItemOut::Given);
                continue;
            }
            ItemKind::Const(_) | ItemKind::Other => {
                items.push(ItemOut::Rust);
                continue;
            }
            ItemKind::Static(_) if process => {
                let laid = host.next().expect("each static is laid out or refused");
                items.push(ItemOut::HostStatic(
                    laid.as_ref()
                        .map(|s| Compiled { code: native.host_static(&s.code), consts: s.consts.clone() })
                        .map_err(Clone::clone),
                ));
                continue;
            }
            ItemKind::Static(s) => {
                first.get_or_insert(s.span);
                let laid = gpu.next().expect("each static is laid out or refused");
                items.push(ItemOut::DeviceStatic(
                    laid.as_ref()
                        .map(|s| Compiled { code: device.device_static(&s.code), consts: s.consts.clone() })
                        .map_err(Clone::clone),
                ));
                continue;
            }
            ItemKind::Fn(f) => f,
        };
        let Some((name, at)) = &f.abi else {
            items.push(ItemOut::Helper(helper(m, f)));
            continue;
        };
        let Some(abi) = Abi::of(name) else {
            items.push(ItemOut::Unknown(said(Refusal::UnknownAbi { abi: name }, *at)));
            continue;
        };
        let not_unsafe = (!trusted(f, abi)).then(|| said(Refusal::NotUnsafe, f.span));
        let (said, listed) = match cooperative(f).and(kernite_lang::instances(m, f)) {
            Ok(listed) => (None, listed),
            Err(e) => (Some(e), vec![]),
        };
        let (count, mut errors, mut consts) = (listed.len(), vec![], vec![]);
        if abi.native() {
            let mut instances = vec![];
            for args in listed {
                match compile(m, &n, native, f, &args) {
                    Ok(k) => {
                        consts.extend(k.consts);
                        instances.push(NativeInstance { args, naked: native.naked(&k.code), calls: vec![] });
                    }
                    Err(e) => errors.push(e),
                }
            }
            // In a module of both kinds of kernel, the statics are the GPU's, which a kernel of
            // the process's does not reach, as the first of them says.
            let reaches = whose.is_ok() || instances.iter().all(|k| k.naked.statics.is_empty());
            let written = said.is_none() && not_unsafe.is_none() && errors.is_empty() && reaches;
            if written {
                for k in &mut instances {
                    k.calls = calls(m, &k.naked.funcs);
                    listings.push((kernite_lang::instance(f, &k.args).unwrap_or_default(), k.naked.text.clone()));
                }
            }
            let consts = in_order(m, &consts);
            let out = NativeOut { not_unsafe, said, instances, consts, written, count, touched: vec![], errors };
            items.push(ItemOut::Native(out));
            continue;
        }
        // A function the GPU runs.
        first.get_or_insert(f.span);
        let mut compiled = vec![];
        for args in listed {
            match compile(m, &d, device, f, &args) {
                Ok(k) => {
                    consts.extend(k.consts);
                    compiled.push((args, k.code));
                }
                Err(e) => errors.push(e),
            }
        }
        let launches = abi == Abi::PtxKernel && said.is_none() && not_unsafe.is_none() && errors.is_empty();
        let mut launched = vec![];
        for (args, code) in compiled {
            if launches {
                launched.push((args, symbols.len()));
                symbols.push(device.symbol(&code));
            }
            codes.push(code);
        }
        let launcher = (abi == Abi::PtxKernel).then_some(match launches {
            true => Launcher::Of(launched),
            false => Launcher::Never,
        });
        let consts = in_order(m, &consts);
        items.push(ItemOut::Device(DeviceFnOut { not_unsafe, said, errors, consts, launcher }));
    }
    // Once every function the process runs is compiled, for a kernel to name the statics of the
    // functions it calls, which may come after it.
    if process {
        let touched = touched(m, &items);
        for (item, named) in items.iter_mut().zip(touched) {
            if let ItemOut::Native(out) = item {
                out.touched = named;
            }
        }
    }
    let loaded = first.map(|span| {
        let (text, error) = match device.module(&codes, &d.made) {
            Ok(text) => (text, None),
            Err(message) => (String::new(), Some(Error { span, message })),
        };
        DeviceOut { text, error, symbols }
    });
    ModuleOut { module, given: twice, owner: whose.err(), items, listings, device: loaded }
}

/// The consts of `module` that `read` names, each once, in the module's order: the instances of a
/// function each read some of them, and it reads the consts that any of them did.
fn in_order(module: &Module, read: &[String]) -> Vec<String> {
    let consts = module.items.iter().filter_map(|item| match &item.kind {
        ItemKind::Const(c) => Some(&c.name),
        _ => None,
    });
    consts.filter(|name| read.contains(name)).cloned().collect()
}

/// What an item of a module of consts is to the modules of kernels that take its consts.
#[derive(Debug, PartialEq)]
pub enum ConstItem {
    /// A const that is a value known now, which kernels may read.
    Given,
    /// A const that is no value known now, and why.
    Wrong(Error),
    /// An item that is no const.
    NotAConst,
}

/// What each item of `module`, a module of consts, is, in order: each const worked out on its
/// own, `pub` or not: only its tokens say which, and the macro reads them.
pub fn consts(module: &Module) -> Vec<ConstItem> {
    let item = |item: &Item| match &item.kind {
        ItemKind::Const(c) => kernite_lang::constant(module, c).map_or_else(ConstItem::Wrong, |_| ConstItem::Given),
        _ => ConstItem::NotAConst,
    };
    module.items.iter().map(item).collect()
}

/// Leaves out of `module` each const `given` gives that an item of the module's own is named as
/// well, and says so at the module's own, which keeps the name.
fn take(module: &mut Module, given: &Given) -> Vec<Error> {
    let named = |item: &Item| match &item.kind {
        ItemKind::Const(c) => Some((c.name.clone(), c.span)),
        ItemKind::Fn(f) => Some((f.name.clone(), f.span)),
        ItemKind::Static(s) => Some((s.name.clone(), s.span)),
        ItemKind::Other => None,
    };
    let (theirs, own): (Vec<&Item>, Vec<&Item>) =
        module.items.iter().partition(|item| item.tokens.start < given.before);
    let (mut errors, mut twice) = (vec![], vec![]);
    for (name, span) in own.into_iter().filter_map(named) {
        if theirs.iter().any(|&item| named(item).is_some_and(|(g, _)| g == name)) {
            errors.push(said(Refusal::GivenTwice { name: &name, from: &given.from }, span));
            twice.push(name);
        }
    }
    module
        .items
        .retain(|item| item.tokens.start >= given.before || named(item).is_none_or(|(g, _)| !twice.contains(&g)));
    errors
}

/// The statics of `module` each function the process runs names, by its index among `items`, in
/// the module's order: those the code of its instances names, those the functions that code
/// names or may reach name, and on.
fn touched(module: &Module, items: &[ItemOut]) -> Vec<Vec<String>> {
    // Each instance that compiled, by name, `gcd` or `tile::<4>`, and its code.
    let mut named: Vec<(String, &Naked)> = vec![];
    for (item, out) in module.items.iter().zip(items) {
        let (ItemKind::Fn(f), ItemOut::Native(out)) = (&item.kind, out) else { continue };
        for k in &out.instances {
            if let Ok(name) = kernite_lang::instance(f, &k.args) {
                named.push((name, &k.naked));
            }
        }
    }
    let statics: Vec<&str> = module
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Static(s) => Some(s.name.as_str()),
            _ => None,
        })
        .collect();
    let touched = |out: &NativeOut| {
        // Its instances' code, then that of the functions they name or may reach, and on.
        let mut reached: Vec<&Naked> = out.instances.iter().map(|k| &k.naked).collect();
        let mut at = 0;
        while at < reached.len() {
            let code = reached[at];
            for name in code.funcs.iter().chain(&code.reach) {
                if let Some(&(_, called)) = named.iter().find(|(n, _)| n == name)
                    && !reached.iter().any(|r| std::ptr::eq(*r, called))
                {
                    reached.push(called);
                }
            }
            at += 1;
        }
        statics
            .iter()
            .filter(|s| reached.iter().any(|code| code.statics.iter().any(|n| n == *s)))
            .map(|s| s.to_string())
            .collect()
    };
    items
        .iter()
        .map(|item| match item {
            ItemOut::Native(out) => touched(out),
            _ => vec![],
        })
        .collect()
}

/// Each function of `module` that `funcs` names, `gcd` or `tile::<4>`, as the code of an
/// instance names them: by the function's name, and the index of the instance among those its
/// `#[instances(..)]` lists, if it takes compile-time arguments.
fn calls(module: &Module, funcs: &[String]) -> Vec<Call> {
    let call = |name: &String| {
        // A function that takes no compile-time arguments and lists instances is wrong at its
        // own item, and `instance` refuses it: the call names it whole, as it names any
        // function of none.
        let Ok(found) = instance(module, name) else { return Call { function: name.clone(), instance: None } };
        let (g, _, k) = found.expect("a kernel calls an instance its function lists");
        Call { function: g.name.clone(), instance: (!g.generics.is_empty()).then_some(k) }
    };
    funcs.iter().map(call).collect()
}
