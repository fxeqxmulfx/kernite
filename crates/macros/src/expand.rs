//! What a module of kernels becomes once the macro has compiled its kernels, item by item.

use std::path::Path;

use proc_macro::{Delimiter, Group, Span, TokenStream, TokenTree};

use kgen::lang::{self, ItemKind, Tree, Value};
use kgen::{Abi, Naked, Owner};

use crate::gpu::{HOLE, gpu};
use crate::rust::{const_names, error, flatten, inner_attribute, is_ident, is_punct, parse, site, stub, tree};
use crate::workspace::workspace;
use crate::x86::{cpu_static, naked, x86_stub};

/// Each instance of an x86 function, by its compile-time arguments, compiled or not.
type X86Instances = Vec<(Vec<Value>, Result<kgen::Compiled<Naked>, lang::Error>)>;

/// What the module's braces hold once its kernels are compiled; `given`, the consts it takes
/// from outside, and the path of their module, as `#[kernels(consts = path)]` writes it.
pub(crate) fn expand(given: Option<(TokenStream, &[TokenTree])>, body: TokenStream) -> TokenStream {
    // The consts given come first, for the kernels to read as the module's own.
    let outside = given.as_ref().map_or_else(Vec::new, |(consts, _)| flatten(consts.clone()));
    let n = outside.len();
    let top: Vec<TokenTree> = outside.into_iter().chain(flatten(body)).collect();
    let mut spans = vec![];
    let trees: Vec<Tree> = top.iter().map(|t| tree(t, &mut spans)).collect();
    let at = |s: lang::Span| spans.get(s.0 as usize).copied().unwrap_or_else(site);
    let mut module = match lang::module(&trees) {
        Ok(module) => module,
        Err(e) => return error(at(e.span), &e.message),
    };
    // The module's own attributes, `#![..]`, before anything the macro writes, where Rust has
    // them.
    let mut out: TokenStream = module
        .items
        .iter()
        .filter(|item| inner_attribute(&top[item.tokens.clone()]))
        .flat_map(|item| top[item.tokens.clone()].iter().cloned())
        .collect();
    if let Some((_, path)) = given {
        out.extend(take(&mut module, n, &top, path, &at));
    }
    // The GPU kernels' namespaces: the workspace's, then the module's path.
    let (workspace, manifest) = workspace();
    let names: Vec<&str> = [workspace.as_str()].into_iter().filter(|n| !n.is_empty()).collect();
    let scope = kgen::ptx_scope(&names) + HOLE;
    let (mut listings, mut entries, mut launchers, mut read) = (vec![], vec![], vec![], vec![]);
    // The first function for a GPU, or static of the GPU's, where an error of the module as a
    // whole goes; the symbol of each kernel it launches, by index; each static of the GPU's,
    // compiled or not, as the host reads it; and those that compiled, as PTX lays them out.
    let (mut first, mut symbols, mut memory, mut laid) = (None, vec![], vec![], vec![]);
    // The statics are the x86 kernels' when the module has no others, and else the GPU's; a
    // module of both kinds of kernel and a static is an error, and its x86 kernels name none.
    let whose = kgen::owner(&module);
    if let Err(e) = &whose {
        out.extend(error(at(e.span), &e.message));
    }
    let cpu = whose == Ok(Owner::Process);
    let (mut statics, mut cpu_statics) = match cpu {
        true => (vec![].into_iter(), kgen::x86_statics(&module).into_iter()),
        false => (kgen::ptx_statics(&module, &scope).into_iter(), vec![].into_iter()),
    };
    // Each x86 function's instances compiled, by its index among the items, unless what it says
    // of them is wrong: all before any is written out, for each kernel's safety contract to
    // name the memory of the module's that the functions it calls name.
    let mut x86_compiled: Vec<Option<X86Instances>> = module
        .items
        .iter()
        .map(|item| {
            let ItemKind::Fn(f) = &item.kind else { return None };
            if kgen::abi(f) != Some(Abi::Sysv64) {
                return None;
            }
            let instances = kgen::cooperative(f).and(lang::instances(&module, f)).ok()?;
            let plain = |k: kgen::Compiled<_>| kgen::Compiled { code: kgen::naked(&k.code), consts: k.consts };
            Some(instances.into_iter().map(|args| (args.clone(), kgen::x86(&module, f, &args).map(plain))).collect())
        })
        .collect();
    let touched = match cpu {
        true => touched(&module, &x86_compiled),
        false => vec![vec![]; module.items.len()],
    };
    for (i, item) in module.items.iter().enumerate() {
        let named = &touched[i];
        let f = match &item.kind {
            // A const given is Rust's where its module is; the module's own attributes are first.
            ItemKind::Const(_) if item.tokens.start < n => continue,
            ItemKind::Other if inner_attribute(&top[item.tokens.clone()]) => continue,
            ItemKind::Const(_) | ItemKind::Other => {
                out.extend(top[item.tokens.clone()].iter().cloned());
                continue;
            }
            ItemKind::Static(s) if cpu => {
                match cpu_statics.next().expect("each static is compiled") {
                    Ok(made) => {
                        read.extend(made.consts);
                        out.extend(cpu_static(s, Some(&kgen::host_static(&made.code)), &top));
                    }
                    Err(e) => {
                        out.extend(error(at(e.span), &e.message));
                        out.extend(cpu_static(s, None, &top));
                    }
                }
                continue;
            }
            ItemKind::Static(s) => {
                first.get_or_insert(s.span);
                match statics.next().expect("each static is compiled") {
                    Ok(made) => {
                        read.extend(made.consts);
                        memory.push((s, Some(kgen::device_static(&made.code, &scope))));
                        laid.push(made.code);
                    }
                    Err(e) => {
                        out.extend(error(at(e.span), &e.message));
                        memory.push((s, None));
                    }
                }
                continue;
            }
            ItemKind::Fn(f) => f,
        };
        let instances = lang::instances(&module, f);
        let cooperative = kgen::cooperative(f);
        let Some((name, abi_span)) = &f.abi else {
            // A helper is checked where it is called; one that does not parse is an error
            // whether it is called or not, as one that lists instances is, or says it is
            // cooperative.
            let errors = [f.body.as_ref().err(), instances.as_ref().err(), cooperative.as_ref().err()];
            for e in errors.into_iter().flatten() {
                out.extend(error(at(e.span), &e.message));
            }
            continue;
        };
        let Some(abi) = Abi::of(name) else {
            let message = format!(
                "kgen compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \"ptx-kernel\"` ones, for \
                 NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"{name}\"` is none of \
                 these"
            );
            out.extend(error(at(*abi_span), &message));
            out.extend(stub(f, &top));
            continue;
        };
        // A function the kernels call is called by nothing else, so it promises Rust nothing.
        let trusted = abi == Abi::PtxFunc || top[f.head.clone()].iter().any(|t| is_ident(t, "unsafe"));
        if !trusted {
            out.extend(error(at(f.span), "a kernel trusts the pointers it is given, so it is an `unsafe fn`"));
        }
        if abi != Abi::Sysv64 {
            first.get_or_insert(f.span);
        }
        // A `#[cooperative]` that is no GPU kernel's is said once, not for each instance.
        let instances = match cooperative.and(instances) {
            Ok(instances) => instances,
            Err(e) => {
                out.extend(error(at(e.span), &e.message));
                match abi {
                    Abi::Sysv64 => out.extend(x86_stub(f, 0, named, &top)),
                    Abi::PtxKernel => launchers.push((f, None)),
                    Abi::PtxFunc => {}
                }
                continue;
            }
        };
        // Each instance compiled, or the errors of those that do not compile.
        let mut errors = vec![];
        match abi {
            Abi::Sysv64 => {
                let mut compiled = vec![];
                for (args, kernel) in x86_compiled[i].take().expect("each x86 function is compiled first") {
                    match kernel {
                        Ok(kernel) => {
                            read.extend(kernel.consts);
                            compiled.push((args, kernel.code));
                        }
                        Err(e) => errors.push(e),
                    }
                }
                // In a module of both kinds of kernel, the statics are the GPU's, which an x86
                // kernel does not reach, as the first of them says.
                let reaches = whose.is_ok() || compiled.iter().all(|(_, listing)| listing.statics.is_empty());
                if trusted && errors.is_empty() && reaches {
                    out.extend(naked(&module, f, &compiled, named, &top));
                    for (args, listing) in &compiled {
                        listings.push((lang::instance(f, args).unwrap_or_default(), listing.text.clone()));
                    }
                } else {
                    out.extend(x86_stub(f, instances.len(), named, &top));
                }
            }
            Abi::PtxKernel | Abi::PtxFunc => {
                let mut compiled = vec![];
                for args in instances {
                    match kgen::ptx(&module, f, &args, &scope) {
                        Ok(kernel) => {
                            read.extend(kernel.consts);
                            compiled.push((args, kernel.code));
                        }
                        Err(e) => errors.push(e),
                    }
                }
                let ok = trusted && errors.is_empty();
                let mut launched = vec![];
                for (args, entry) in compiled {
                    if ok && abi == Abi::PtxKernel {
                        launched.push((args, symbols.len()));
                        symbols.push(entry.program.name.clone());
                    }
                    entries.push(entry);
                }
                if abi == Abi::PtxKernel {
                    launchers.push((f, ok.then_some(launched)));
                }
            }
        }
        for e in errors {
            out.extend(error(at(e.span), &e.message));
        }
    }
    out.extend(used(&module, &read, &top));
    if !listings.is_empty() {
        let mut s = String::from(
            "/// Each x86 kernel's listing, by name, an instance's as `tile::<4>`: where its variables\n\
             /// live, and its code with its notes, for snapshots.\n#[allow(dead_code)]\n\
             pub const LISTINGS: &[(&str, &str)] = &[",
        );
        for (name, text) in &listings {
            s += &format!("({name:?}, {text:?}), ");
        }
        out.extend(parse(&(s + "];")));
    }
    if let Some(span) = first {
        let ptx = kgen::ptx_module(&entries, &laid).unwrap_or_else(|e| {
            out.extend(error(at(span), &e));
            String::new()
        });
        out.extend(gpu(&launchers, &symbols, &memory, &ptx, &top, &at));
        // The workspace's name is in the PTX, so it is built again when it changes.
        if let Some(manifest) = manifest.as_deref().and_then(Path::to_str) {
            out.extend(parse(&format!("const _: &[u8] = ::core::include_bytes!({manifest:?});")));
        }
    }
    out
}

/// What Rust's code in `module` sees the consts it is given by, the first of its items, those of
/// its first `n` tokens: `use path::{E, H};`, the path written from inside the module. A name
/// both have is an error at the module's own, whose the name is for kgen too.
fn take(
    module: &mut lang::Module,
    n: usize,
    top: &[TokenTree],
    path: &[TokenTree],
    at: &dyn std::ops::Fn(lang::Span) -> Span,
) -> TokenStream {
    let name_of = |kind: &ItemKind| match kind {
        ItemKind::Const(c) => Some((c.name.clone(), c.span)),
        ItemKind::Fn(f) => Some((f.name.clone(), f.span)),
        ItemKind::Static(s) => Some((s.name.clone(), s.span)),
        ItemKind::Other => None,
    };
    let (given, own): (Vec<_>, Vec<_>) = module.items.iter().partition(|item| item.tokens.start < n);
    let written: String = path.iter().map(TokenTree::to_string).collect();
    let mut out = TokenStream::new();
    let mut twice = vec![];
    for (name, span) in own.iter().filter_map(|item| name_of(&item.kind)) {
        if given.iter().any(|item| name_of(&item.kind).is_some_and(|(g, _)| g == name)) {
            let message = format!("`{name}` names one of the consts the module takes from `{written}` as well");
            out.extend(error(at(span), &message));
            twice.push(name);
        }
    }
    // Each const given by its name as written, `r#type` as well.
    let mut names = TokenStream::new();
    for item in &given {
        let Some((name, _)) = name_of(&item.kind) else { continue };
        if twice.contains(&name) {
            continue;
        }
        names.extend([const_name(item, top)]);
        names.extend(parse(","));
    }
    module.items.retain(|item| item.tokens.start >= n || name_of(&item.kind).is_none_or(|(g, _)| !twice.contains(&g)));
    out.extend(parse("#[allow(unused_imports)] use"));
    match path {
        [first, ..] if is_ident(first, "crate") || is_punct(first, ':') => out.extend(path.iter().cloned()),
        [first, rest @ ..] if is_ident(first, "self") => {
            out.extend(parse("super"));
            out.extend(rest.iter().cloned());
        }
        _ => {
            out.extend(parse("super::"));
            out.extend(path.iter().cloned());
        }
    }
    out.extend(parse("::"));
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, names))]);
    out.extend(parse(";"));
    out
}

/// The memory of the module's that each of its x86 functions names, by the function's index
/// among the items, each static by name, in the module's order: what the code of each of its
/// instances `compiled` names, and what the functions that code names name, and on.
fn touched<'m>(module: &'m lang::Module, compiled: &[Option<X86Instances>]) -> Vec<Vec<&'m str>> {
    // Each instance that compiled, by name, `gcd` or `tile::<4>`, and its listing.
    let mut listings: Vec<(String, &Naked)> = vec![];
    for (item, instances) in module.items.iter().zip(compiled) {
        let (ItemKind::Fn(f), Some(instances)) = (&item.kind, instances) else { continue };
        for (args, kernel) in instances {
            if let (Ok(name), Ok(kernel)) = (lang::instance(f, args), kernel) {
                listings.push((name, &kernel.code));
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
    let touched = |instances: &X86Instances| {
        // Its instances' code, then that of the functions they name, and on.
        let mut reached: Vec<&Naked> = instances.iter().filter_map(|(_, k)| Some(&k.as_ref().ok()?.code)).collect();
        let mut at = 0;
        while at < reached.len() {
            let listing = reached[at];
            for name in &listing.funcs {
                if let Some(&(_, l)) = listings.iter().find(|(n, _)| n == name)
                    && !reached.iter().any(|r| std::ptr::eq(*r, l))
                {
                    reached.push(l);
                }
            }
            at += 1;
        }
        statics.iter().copied().filter(|s| reached.iter().any(|l| l.statics.iter().any(|n| n == s))).collect()
    };
    compiled.iter().map(|instances| instances.as_ref().map_or_else(Vec::new, touched)).collect()
}

/// A `const _` that names each constant of `module` in `read`, by its name as `top` writes it,
/// `r#type` as well. The kernels read them, but Rust never sees a kernel's body run, so it
/// would find them unused.
fn used(module: &lang::Module, read: &[String], top: &[TokenTree]) -> TokenStream {
    let mut lets = TokenStream::new();
    for item in &module.items {
        if let ItemKind::Const(c) = &item.kind
            && read.contains(&c.name)
        {
            lets.extend(parse("let _ ="));
            lets.extend([const_name(item, top)]);
            lets.extend(parse(";"));
        }
    }
    if lets.is_empty() {
        return TokenStream::new();
    }
    let mut out = parse("const _: () =");
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, lets))]);
    out.extend(parse(";"));
    out
}

/// The name of `item`, a const, as `top` writes it, `r#type` as well, of the span the macro
/// writes.
fn const_name(item: &lang::Item, top: &[TokenTree]) -> TokenTree {
    let mut name = const_names(&top[item.tokens.clone()]).next().expect("a const").clone();
    name.set_span(site());
    name
}
