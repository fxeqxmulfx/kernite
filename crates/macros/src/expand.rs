//! What a module of kernels becomes once kernite has compiled its kernels: each item as what
//! `kernite::module` says it is, written out for Rust. A module kernite refuses whole, at its
//! attribute, is written the same way, with the error in it. kernite checks none of its items, so
//! each kernel is as it is when it does not compile, each static stands as its type has it where
//! kernite works that out, and nothing kernite would find in them is said. Nothing of it is
//! compiled, so a workspace `name` that is not a string is no error at it. No room is counted for
//! the statics, so the crate is not told it needs `kernite-macros` for them. What this finds in
//! writing the module for Rust is said, as for any module, such as the dependency on `kernite-cuda`
//! or `kernite-cpu` it needs or a compile-time argument named as a parameter of a launcher. A module
//! kernite cannot read is its error alone.

use std::path::Path;

use proc_macro::{Delimiter, Group, Span, TokenStream, TokenTree};

use kernite::lang::{self, ItemKind};
use kernite::{DeviceOut, ItemOut, Kept, ModuleOut};

use crate::cond::Cond;
use crate::gpu::{HOLE, gpu};
use crate::room;
use crate::rust::{
    cfg_of, error, error_under, flatten, from_inside, inner_attribute, lex, name_at, parse, position, seen, site, stub,
};
use crate::workspace::workspace;
use crate::x86::{cpu_static, naked, x86_stub};

/// What the module's braces hold once its kernels are compiled; `given`, the consts it takes
/// from outside, and the path of their module, as `#[kernels(consts = path)]` writes it. A module
/// kernite refuses whole, at its attribute, has `why`, the error, which is in the module, said
/// once. It is given no consts, and kernite checks none of its items, for the consts its kernels
/// read may not be given: each kernel is as it is when it does not compile, each static stands as
/// its type has it where kernite works that out, and nothing kernite would find in them is said.
/// Nothing of it is compiled, so it reads no workspace, and a `name` that is not a string is no
/// error at it. No room is counted for the statics, so the crate is not told it needs
/// `kernite-macros` for them. What this finds in writing the module for Rust is said, as for any
/// module: the dependency on `kernite-cuda` or `kernite-cpu` it needs, a compile-time argument named
/// as a parameter of a launcher or of a `_blocks` method, and a GPU kernel or a static named as a
/// cooperative kernel's `_blocks` method. A module kernite cannot read has `why` alone.
pub(crate) fn expand(
    given: Option<(TokenStream, &[TokenTree])>,
    why: Option<TokenStream>,
    body: TokenStream,
) -> TokenStream {
    // The consts given come first, for the kernels to read as the module's own.
    let outside = given.as_ref().map_or_else(Vec::new, |(consts, _)| flatten(consts.clone()));
    let n = outside.len();
    let top: Vec<TokenTree> = outside.into_iter().chain(flatten(body)).collect();
    let (trees, spans) = lex(&top);
    let at = |s: lang::Span| spans.get(s.0 as usize).copied().unwrap_or_else(site);
    let said = |e: &lang::Error| error(at(e.span), &e.message);
    let module = match trees.and_then(|trees| lang::module(&trees, n)) {
        Ok(module) => module,
        // A module kernite refuses has its error alone: it is what can be mended first.
        Err(e) => return why.unwrap_or_else(|| said(&e)),
    };
    // The error at the module itself, and the GPU kernels' namespaces: the workspace's, then the
    // module's path. A name that is not a string is an error at the module; the macro goes on without
    // it, so that the error is the only one. A refused module has the error it is refused for, and no
    // namespace, for nothing of it is compiled: it reads no workspace, so a name that is not a string
    // is none of its errors.
    let refused = why.is_some();
    let (workspace, manifest, at_module) = match why {
        Some(why) => (String::new(), None, Some(why)),
        None => match workspace() {
            Ok((workspace, manifest)) => (workspace, manifest, None),
            Err(message) => (String::new(), None, Some(error(site(), &message))),
        },
    };
    let names: Vec<&str> = [workspace.as_str()].into_iter().filter(|n| !n.is_empty()).collect();
    let scope = kernite::ptx_scope(&names) + HOLE;
    let consts = given
        .as_ref()
        .map(|(_, path)| kernite::Given { before: n, from: path.iter().map(TokenTree::to_string).collect() });
    let made = kernite::module(module, consts, refused, &scope);
    let module = &made.module;
    // The module's own attributes, `#![..]`, before anything the macro writes, where Rust has
    // them. A module given consts has them from `#[kernels]`, which writes this inside the module.
    let mut out: TokenStream = module
        .items
        .iter()
        .filter(|item| given.is_none() && inner_attribute(&top[item.tokens.clone()]))
        .flat_map(|item| top[item.tokens.clone()].iter().cloned())
        .collect();
    // The errors of the module as a whole, said under no `cfg`: at its attribute or at a workspace's name that is not a
    // string; an item that has the name of a const the module takes; and a module that is two.
    let whole = at_module.is_some() || !made.given.is_empty() || made.owner.is_some();
    out.extend(at_module);
    if let Some((_, path)) = given {
        out.extend(made.given.iter().map(said));
        out.extend(take(&made, &top, path));
    }
    out.extend(made.owner.as_ref().map(said));
    // The `cfg` of each item, what is kept of its attributes, and the condition it says. An item says
    // what is wrong with it under its `cfg`, as what is written for it is: Rust says nothing of a
    // function it leaves out.
    let cfgs: Vec<TokenStream> = module.items.iter().map(|item| cfg_of(item, &top)).collect();
    let conds: Vec<Cond> = cfgs.iter().map(Cond::of).collect();
    // Where `cfg` keeps what a part of the GPU's module is of: the code of an item, or a static of the GPU's.
    let kept = |device: &DeviceOut, it: Kept| match it {
        Kept::Code(i) => conds[device.code_items[i]].clone(),
        Kept::Static(j) => conds[device.static_items[j]].clone(),
    };
    // Why the GPU's module is none, each reason where `cfg` keeps what it is of.
    let mut unloaded = vec![];
    for device in made.device.iter() {
        for (when, message) in &device.assembly.refusals {
            unloaded.push((Cond::when(when, &|it| kept(device, it)), message.as_str()));
        }
    }
    // Where Rust is silent about a name it cannot find, `gpr` say, in the signature of a helper: where a macro writes
    // an error as an item of the module. That is where kernite says an error of an item, under the item's `cfg`; why
    // the GPU's module is none, under the `cfg` of what it is of; and the errors of the module as a whole.
    let items = made.items.iter().zip(&conds).filter(|(item, _)| item.failed()).map(|(_, c)| c.clone());
    let failing = Cond::any(items.chain(unloaded.iter().map(|(c, _)| c.clone())).chain(whole.then_some(Cond::Always)));
    // Where Rust finds a name of the kernel's, as it would find it in the kernel's own body: at the
    // name, with the macro's hygiene, for Rust to resolve it as it resolves what the macro writes.
    let place = |m: &lang::Mention| at(m.span).resolved_at(site());
    // How each kernel the GPU launches is, and each static of the GPU's, as the host reads it; when
    // each GPU item is there; and what shows Rust what each item names, and the types of each signature
    // that nothing else writes.
    let (mut launchers, mut memory, mut on_gpu, mut reads) = (vec![], vec![], vec![], TokenStream::new());
    // A field for each static of the GPU's that is laid out, under its `cfg`, for Rust to count the room
    // of those it keeps.
    let mut fields = TokenStream::new();
    // Each listing the process's kernels have, by name, under the `cfg` of the kernel it is of, as
    // what `naked` writes for it is.
    let mut listed = vec![];
    for (i, (item, made_item)) in module.items.iter().zip(&made.items).enumerate() {
        let tokens = &top[item.tokens.clone()];
        let cfg = &cfgs[i];
        let said = |e: &lang::Error| error_under(cfg, at(e.span), &e.message);
        // What shows Rust the types of a signature that nothing else writes, where `cfg` keeps the item and kernite
        // says an error.
        let shown = |f: &lang::Fn| Cond::all([conds[i].clone(), failing.clone()]).guard().map(|g| seen(f, &top, &g));
        reads.extend(named(&made, &conds, i, &top, &place));
        match (&item.kind, made_item) {
            // The module's own attributes are first, and a const given is Rust's where its
            // module is.
            (_, ItemOut::Rust) if inner_attribute(tokens) => {}
            (_, ItemOut::Rust) => out.extend(tokens.iter().cloned()),
            (_, ItemOut::Given) => {}
            (ItemKind::Static(s), ItemOut::HostStatic(laid)) => {
                out.extend(laid.as_ref().err().and_then(|failed| failed.error.as_ref()).map(said));
                let held = laid.as_ref().map_or_else(|failed| failed.stub.as_ref(), |c| Some(&c.code));
                out.extend(cpu_static(s, held, &top));
            }
            (ItemKind::Static(s), ItemOut::DeviceStatic(laid)) => {
                out.extend(laid.as_ref().err().and_then(|failed| failed.error.as_ref()).map(said));
                memory.push((s, laid, conds[i].clone()));
                on_gpu.push(conds[i].clone());
                if let Ok(laid) = laid {
                    fields.extend(room::field(cfg, i, &laid.code.ty, at(s.ty.span())));
                }
            }
            (ItemKind::Fn(f), ItemOut::Helper(errors)) => {
                out.extend(errors.iter().map(said));
                reads.extend(shown(f));
            }
            (ItemKind::Fn(f), ItemOut::Unknown(e)) => {
                out.extend(e.as_ref().map(said));
                out.extend(stub(f, &top));
            }
            (ItemKind::Fn(f), ItemOut::Native(k)) => {
                out.extend(k.not_unsafe.as_ref().map(said));
                out.extend(k.said.as_ref().map(said));
                out.extend(match k.written {
                    true => naked(module, f, &k.instances, &k.touched, cfg, &top),
                    false => x86_stub(f, k.count, &k.touched, cfg, &top),
                });
                out.extend(k.errors.iter().map(said));
                listed.extend(k.listings.iter().map(|listing| (cfg, listing)));
            }
            (ItemKind::Fn(f), ItemOut::Device(d)) => {
                out.extend(d.not_unsafe.as_ref().map(said));
                out.extend(d.said.as_ref().map(said));
                out.extend(d.errors.iter().map(said));
                on_gpu.push(conds[i].clone());
                launchers.extend(d.launcher.as_ref().map(|l| (f, l)));
                if d.launcher.is_none() {
                    reads.extend(shown(f));
                }
            }
            _ => unreachable!("kernite::module says what each item is"),
        }
    }
    out.extend(room::carrier(fields, &Cond::any(memory.iter().filter(|(_, m, _)| m.is_ok()).map(|(.., c)| c.clone()))));
    if !reads.is_empty() {
        out.extend(parse("const _: () ="));
        out.extend([TokenTree::Group(Group::new(Delimiter::Brace, reads))]);
        out.extend(parse(";"));
    }
    // A module with an x86 kernel has `LISTINGS` whether its kernels compile or not, and a module of GPU
    // kernels alone has none.
    if made.items.iter().any(|item| matches!(item, ItemOut::Native(_))) {
        let mut listings = TokenStream::new();
        for (cfg, (name, text)) in listed {
            listings.extend(cfg.clone());
            listings.extend(parse(&format!("({name:?}, {text:?}),")));
        }
        out.extend(parse(
            "/// Each x86 kernel's listing that `cfg` keeps, by name, an instance's as `tile::<4>`: where its\n\
             /// variables live, and its code with its notes, for snapshots. A kernel that does not compile has\n\
             /// none.\n\
             pub const LISTINGS: &[(&::core::primitive::str, &::core::primitive::str)] = &",
        ));
        out.extend([TokenTree::Group(Group::new(Delimiter::Bracket, listings))]);
        out.extend(parse(";"));
    }
    if let Some(device) = &made.device {
        // Each part of the module is there where `cfg` keeps the code and the statics it is of.
        let parts: Vec<(Cond, &str)> = device
            .assembly
            .parts
            .iter()
            .map(|(when, text)| (Cond::when(when, &|it| kept(device, it)), text.as_str()))
            .collect();
        out.extend(unloadable(&unloaded, at(device.at)));
        let symbols: Vec<(Cond, &str)> =
            device.symbols.iter().map(|(code, symbol)| (kept(device, Kept::Code(*code)), symbol.as_str())).collect();
        out.extend(gpu(&launchers, &symbols, &memory, &parts, &top, &at, &on_gpu));
        // The workspace's name is in the PTX, so it is built again when it changes.
        if let Some(manifest) = manifest.as_deref().and_then(Path::to_str) {
            out.extend(parse(&format!("const _: &[::core::primitive::u8] = ::core::include_bytes!({manifest:?});")));
        }
    }
    out
}

/// Why the GPU's module is none, at the module's first item that is the GPU's, `at`: each of `unloaded`
/// where its condition holds. It is said once: where an earlier reason holds, a later is not.
fn unloadable(unloaded: &[(Cond, &str)], at: Span) -> TokenStream {
    let (mut out, mut before) = (TokenStream::new(), vec![]);
    for (cond, message) in unloaded {
        let now = Cond::all([cond.clone(), Cond::any(before.iter().cloned()).not()]);
        out.extend(now.guard().map(|guard| error_under(&guard, at, message)));
        before.push(cond.clone());
    }
    out
}

/// What Rust's code in the module sees the consts it is given by, those `made` says are the
/// module's given: `use path::{E, H};`, the path written from inside the module.
fn take(made: &ModuleOut, top: &[TokenTree], path: &[TokenTree]) -> TokenStream {
    // Each const given by its name as written, `r#type` as well.
    let mut names = TokenStream::new();
    for (item, made_item) in made.module.items.iter().zip(&made.items) {
        if matches!(made_item, ItemOut::Given) {
            names.extend([name_at(item, top, site())]);
            names.extend(parse(","));
        }
    }
    let mut out = parse("use");
    out.extend(from_inside(path, 1));
    out.extend(parse("::"));
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, names))]);
    out.extend(parse(";"));
    out
}

/// What says, at each place item `i` of `made` names an item of its module, that `cfg` may leave the
/// item out; `place` is the span of the place. Rust has a name for a const, a static of the process's
/// and an x86 kernel that takes no compile-time arguments, and finds it where it is written: a
/// `let _ = NAME;` there, for Rust to say it cannot find the name if `cfg` leaves out the item where
/// it keeps item `i`. For any other item kernite says so itself, as an error where `cfg` keeps item `i`
/// and leaves that one out. Both are under the `cfg` of item `i` and of the code that has the place,
/// where item `i` runs it: [`runs`].
fn named(
    made: &ModuleOut,
    conds: &[Cond],
    i: usize,
    top: &[TokenTree],
    place: &dyn Fn(&lang::Mention) -> Span,
) -> TokenStream {
    let module = &made.module;
    let mut out = TokenStream::new();
    for m in made.items[i].mentions() {
        let j = position(module, &m.name);
        let kept = Cond::all([conds[i].clone(), runs(made, conds, i, module.item_at(m.span), &[])]);
        let rust = match (&module.items[j].kind, &made.items[j]) {
            (_, ItemOut::Rust | ItemOut::Given | ItemOut::HostStatic(_)) => true,
            (ItemKind::Fn(f), ItemOut::Native(_)) => f.generics.is_empty(),
            _ => false,
        };
        if rust {
            let Some(guard) = kept.guard() else { continue };
            out.extend(guard);
            out.extend(parse("let _ ="));
            out.extend([name_at(&module.items[j], top, place(m))]);
            out.extend(parse(";"));
        } else if let Some(guard) = Cond::all([kept, conds[j].clone().not()]).guard() {
            let reader = module.items[i].name().expect("an item that names others has a name");
            let message = format!("`{}` is left out by `cfg` where `{reader}`, which names it, is kept", m.name);
            out.extend(error_under(&guard, place(m), &message));
        }
    }
    out
}

/// The condition under which item `i` of `made` runs the code of item `c`, but for the `cfg` of `i`:
/// always for its own code, and for a helper's, where `cfg` keeps the helper and `i` runs the code that
/// calls it, at any place `i` names the helper. `way` holds the items followed back from the place to
/// `c`: a place in one of them adds nothing, for `i` reaches that item by the places already counted.
/// So a helper that calls itself, or two that call each other, do not go on without end.
fn runs(made: &ModuleOut, conds: &[Cond], i: usize, c: usize, way: &[usize]) -> Cond {
    if c == i {
        return Cond::Always;
    }
    let module = &made.module;
    let way = [way, &[c]].concat();
    let calls = made.items[i].mentions().iter().filter(|m| module.items[c].name() == Some(&m.name));
    let callers = calls.map(|m| module.item_at(m.span)).filter(|caller| !way.contains(caller));
    Cond::all([conds[c].clone(), Cond::any(callers.map(|caller| runs(made, conds, i, caller, &way)))])
}
