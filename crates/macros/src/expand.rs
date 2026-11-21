//! What a module of kernels becomes once kernite has compiled its kernels: each item as what
//! `kernite::module` says it is, written out for Rust. A module kernite refuses whole, at its
//! attribute or for an item it cannot read, is written the same way, with the error in it: of the
//! second, the items `lang::read` reads. kernite checks none of its items, so each kernel is as it
//! is when it does not compile, each static stands as its type has it where kernite works that out,
//! and nothing kernite would find in them is said. Nothing of it is compiled, so a workspace `name`
//! that is not a string is no error at it. No room is counted for the statics, so the crate is not
//! told it needs `kernite-macros` for them. What this finds in writing the module for Rust is said,
//! as for any module, such as the dependency on `kernite-cuda` or `kernite-cpu` it needs or a
//! compile-time argument named as a parameter of a launcher. Brackets nested too deep leave a module
//! no item: it is its error alone.

use std::path::Path;

use proc_macro::{Delimiter, Group, Span, TokenStream, TokenTree};

use kernite::lang::{self, ItemKind};
use kernite::{DeviceOut, ItemOut, Kept, ModuleOut};

use crate::cond::Cond;
use crate::gpu::{HOLE, gpu, refused_attrs, used};
use crate::room;
use crate::rust::{
    cfg_of, error, error_under, flatten, for_rust, from_inside, grouped, ident_at, inner_attribute, lex, name_at,
    parse, passable, position, seen, site, stub, takes, unnamed, within,
};
use crate::workspace::workspace;
use crate::x86::{cpu_static, naked, operands, x86_stub};

/// What the module's braces, `body`, hold once its kernels are compiled; `given`, the consts it
/// takes from outside, and the path of their module, as `#[kernels(consts = path)]` writes it. A
/// module kernite refuses whole has its error, which is in the module, said once: `why`, for a
/// module refused at its attribute, which is given no consts, for the consts its kernels read may
/// not be given; or the error `lang::read` gives, for a module kernite cannot read, which has the
/// items it reads and the consts it is given. kernite checks none of the items of a module refused
/// whole: each kernel is as it is when it does not compile, each static stands as its type has it
/// where kernite works that out, and nothing kernite would find in them is said. Nothing of it is
/// compiled, so it reads no workspace, and a `name` that is not a string is no error at it. No room
/// is counted for the statics, so the crate is not told it needs `kernite-macros` for them. What
/// this finds in writing the module for Rust is said, as for any module: the dependency on
/// `kernite-cuda` or `kernite-cpu` it needs, a compile-time argument named as a parameter of a
/// launcher or of a `_blocks` method, and a GPU kernel or a static named as a cooperative kernel's
/// `_blocks` method. Brackets nested too deep leave a module no item: it has its error alone, or
/// `why`.
///
/// Every error the macros write is an item of the module, so Rust is silent about a name it cannot find where there is
/// one. The types a helper's signature names, which no kernel needs to compile, are shown to Rust there alone.
pub(crate) fn expand(
    given: Option<(TokenStream, &[TokenTree])>,
    why: Option<TokenStream>,
    body: &Group,
) -> TokenStream {
    // The consts given come first, for the kernels to read as the module's own.
    let end = body.span_close();
    let outside = given.as_ref().map_or_else(Vec::new, |(consts, _)| grouped(flatten(consts.clone()), end));
    let n = outside.len();
    let top: Vec<TokenTree> = outside.into_iter().chain(grouped(flatten(body.stream()), end)).collect();
    let (trees, spans) = lex(&top);
    let mut top = for_rust(top);
    let at = |s: lang::Span| spans.get(s.0 as usize).copied().unwrap_or_else(site);
    let said = |e: &lang::Error| error(at(e.span), &e.message);
    // A module kernite cannot read is refused whole, as at its attribute, with the items it reads: its error is what
    // can be mended first. Brackets nested too deep leave it no item, and its error alone.
    let (module, unread) = match trees {
        Ok(trees) => lang::read(&trees, n),
        Err(e) => return why.unwrap_or_else(|| said(&e)),
    };
    let why = why.or_else(|| unread.as_ref().map(said));
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
    // The `cfg` of each item, what is kept of its attributes, and the condition it says. An item says
    // what is wrong with it under its `cfg`, as what is written for it is: Rust says nothing of a
    // function it leaves out.
    let cfgs: Vec<TokenStream> = module.items.iter().map(|item| cfg_of(item, &top)).collect();
    // Every signature the macros write has the names Rust can be given. A parameter named `self` or `Self` is Rust's
    // error as of a function that is no method, said by a function of its own that never runs, under the `cfg` of the
    // function that has it.
    let mut free = TokenStream::new();
    for (item, cfg) in module.items.iter().zip(&cfgs) {
        if let ItemKind::Fn(f) = &item.kind
            && let Some(stub) = passable(&mut top, f)
        {
            free.extend(cfg.clone());
            free.extend(stub);
        }
    }
    // The module's own attributes, `#![..]`, before anything the macro writes, where Rust has
    // them. A module given consts has them from `#[kernels]`, which writes this inside the module.
    let mut out: TokenStream = (module.items.iter().map(|item| within(&top, &item.tokens)))
        .filter(|tokens| given.is_none() && inner_attribute(tokens))
        .flat_map(|tokens| tokens.iter().cloned())
        .collect();
    // The errors of the module as a whole, said under no `cfg`: at its attribute or at a workspace's name that is not a
    // string; an item that has the name of a const the module takes; and a module that is two.
    out.extend(at_module);
    if let Some((_, path)) = given {
        out.extend(made.given.iter().map(said));
        out.extend(take(&made, &top, path));
    }
    out.extend(made.owner.as_ref().map(said));
    out.extend(free);
    let conds: Vec<Cond> = cfgs.iter().map(Cond::of).collect();
    // Where `cfg` keeps what a part of the GPU's module is of: the code of an item, or a static of the GPU's.
    let kept = |device: &DeviceOut, it: Kept| {
        let item = match it {
            Kept::Code(i) => device.code_items.get(i),
            Kept::Static(j) => device.static_items.get(j),
        };
        item.map_or(Cond::Never, |&at| cond_of(&conds, at))
    };
    // Why the GPU's module is none, each reason where `cfg` keeps what it is of.
    let mut unloaded = vec![];
    for device in made.device.iter() {
        for (when, message) in &device.assembly.refusals {
            unloaded.push((Cond::when(when, &|it| kept(device, it)), message.as_str()));
        }
    }
    // Where Rust finds a name of the kernel's, as it would find it in the kernel's own body: at the
    // name, with the macro's hygiene, for Rust to resolve it as it resolves what the macro writes.
    let place = |s: lang::Span| at(s).resolved_at(site());
    // How each kernel the GPU launches is, and each static of the GPU's, as the host reads it; when
    // each GPU item is there; what shows Rust what each item names; and the errors of a name that `cfg` leaves out.
    let (mut launchers, mut memory, mut on_gpu) = (vec![], vec![], vec![]);
    let (mut reads, mut left_out) = (TokenStream::new(), TokenStream::new());
    // A field for each static of the GPU's that is laid out, under its `cfg`, for Rust to count the room
    // of those it keeps.
    let mut fields = TokenStream::new();
    // The functions whose types nothing else shows Rust: each helper, and each GPU function, which is not launched,
    // whether a kernel calls it or not.
    let mut helpers = vec![];
    // Each listing the process's kernels have, by name, under the `cfg` of the kernel it is of, as
    // what `naked` writes for it is.
    let mut listed = vec![];
    for (i, (((item, made_item), cfg), cond)) in module.items.iter().zip(&made.items).zip(&cfgs).zip(&conds).enumerate()
    {
        let tokens = within(&top, &item.tokens);
        let said = |e: &lang::Error| error_under(cfg, at(e.span), &e.message);
        let (names, errors) = named(&made, &conds, i, &top, &place);
        reads.extend(names);
        left_out.extend(errors);
        match (&item.kind, made_item) {
            // The module's own attributes are first, and a const given is Rust's where its
            // module is.
            (_, ItemOut::Rust) if inner_attribute(tokens) => {}
            (_, ItemOut::Rust) => out.extend(tokens.iter().cloned()),
            (_, ItemOut::Given) => {}
            (ItemKind::Static(s), ItemOut::HostStatic(laid)) => {
                out.extend(laid.as_ref().err().and_then(|failed| failed.error.as_ref()).map(said));
                let held = laid.as_ref().map_or_else(|failed| failed.stub.as_ref(), |c| Some(&c.code));
                out.extend(cpu_static(s, held, cfg, &top));
            }
            (ItemKind::Static(s), ItemOut::DeviceStatic(laid)) => {
                out.extend(laid.as_ref().err().and_then(|failed| failed.error.as_ref()).map(said));
                out.extend(refused_attrs(s, cfg, &top));
                memory.push((s, laid, cond.clone()));
                on_gpu.push(cond.clone());
                if let Ok(laid) = laid {
                    fields.extend(room::room(cfg, i, &laid.code.ty, at(s.ty.span())));
                }
            }
            (ItemKind::Fn(f), ItemOut::Helper(errors)) => {
                out.extend(errors.iter().map(said));
                helpers.push((i, cond, f));
            }
            (ItemKind::Fn(f), ItemOut::Unknown(e)) => {
                out.extend(e.as_ref().map(said));
                out.extend(stub(f, &top));
            }
            (ItemKind::Fn(f), ItemOut::Native(k)) => {
                out.extend(k.not_unsafe.as_ref().map(said));
                out.extend(k.said.as_ref().map(said));
                let firsts = operands(k);
                // The kernel writes what its code names; the macro's call site stands for what it does not.
                let first = |name: &str| firsts.iter().find(|m| m.name == name).map_or_else(site, |m| place(m.span));
                out.extend(match k.written {
                    true => naked(module, f, &k.instances, &k.touched, cfg, &top, &first),
                    false => x86_stub(f, k.count, &k.touched, cfg, &top),
                });
                out.extend(k.errors.iter().map(said));
                listed.extend(k.listings.iter().map(|listing| (cfg, listing)));
            }
            (ItemKind::Fn(f), ItemOut::Device(d)) => {
                out.extend(d.not_unsafe.as_ref().map(said));
                out.extend(d.said.as_ref().map(said));
                out.extend(d.errors.iter().map(said));
                on_gpu.push(cond.clone());
                launchers.extend(d.launcher.as_ref().map(|l| (f, l)));
                if d.launcher.is_none() {
                    helpers.push((i, cond, f));
                }
            }
            _ => out.extend(error_under(cfg, at(item.span), "kernite made this item what it is not: that is a bug")),
        }
    }
    let laid = Cond::any(memory.iter().filter(|(_, m, _)| m.is_ok()).map(|(.., c)| c.clone()));
    out.extend(room::lacking(&laid));
    // The rest of the module, after the room and the names `cfg` leaves out.
    let mut tail = TokenStream::new();
    // A module with an x86 kernel has `LISTINGS` whether its kernels compile or not, and a module of GPU
    // kernels alone has none.
    if made.items.iter().any(|item| matches!(item, ItemOut::Native(_))) {
        let mut listings = TokenStream::new();
        for (cfg, (name, text)) in listed {
            listings.extend(cfg.clone());
            listings.extend(parse(&format!("({name:?}, {text:?}),")));
        }
        tail.extend(parse(
            "/// Each x86 kernel's listing that `cfg` keeps, by name, an instance's as `tile::<4>`: where its\n\
             /// variables live, and its code with its notes, for snapshots. A kernel that does not compile has\n\
             /// none.\n\
             pub const LISTINGS: &[(&::core::primitive::str, &::core::primitive::str)] = &",
        ));
        tail.extend([TokenTree::Group(Group::new(Delimiter::Bracket, listings))]);
        tail.extend(parse(";"));
    }
    if let Some(device) = &made.device {
        // Each part of the module is there where `cfg` keeps the code and the statics it is of.
        let parts: Vec<(Cond, &str)> = device
            .assembly
            .parts
            .iter()
            .map(|(when, text)| (Cond::when(when, &|it| kept(device, it)), text.as_str()))
            .collect();
        tail.extend(unloadable(&unloaded, at(device.at)));
        let symbols: Vec<(Cond, &str)> =
            device.symbols.iter().map(|(code, symbol)| (kept(device, Kept::Code(*code)), symbol.as_str())).collect();
        tail.extend(gpu(&launchers, &symbols, &memory, &parts, &top, &at, &on_gpu));
        // The workspace's name is in the PTX, so it is built again when it changes.
        if let Some(manifest) = manifest.as_deref().and_then(Path::to_str) {
            tail.extend(parse(&format!("const _: &[::core::primitive::u8] = ::core::include_bytes!({manifest:?});")));
        }
    }
    // Where Rust is silent about a name it cannot find, `gpr` say, in the signature of a helper: where the module
    // has an error as an item, as every error kernite and the macros find is. There the types of the helpers are
    // shown, in a block of the module, under the `cfg` of the helper. The room of the statics is wrong where Rust
    // decides, so `__Room` shows them there, as a field of each helper in the struct says.
    let failing = Cond::any([Cond::errors_in(&out), Cond::errors_in(&left_out), Cond::errors_in(&tail)]);
    let mut shown = TokenStream::new();
    for (i, cond, f) in helpers {
        reads.extend(Cond::all([cond.clone(), failing.clone()]).guard().map(|g| seen(f, &top, &g)));
        shown.extend(cond.guard().map(|g| room::shown(&g, i, seen(f, &top, &TokenStream::new()))));
    }
    out.extend(room::carrier(fields, shown));
    out.extend(left_out);
    if !reads.is_empty() {
        out.extend(unnamed(reads));
    }
    out.extend(docs(&made, &conds, &at));
    out.extend(tail);
    out
}

/// What shows an editor the doc of an instruction where a kernel names it: a block of the module that has a function
/// for each instruction `made` has a doc of, which is that doc and is never called, and, at each word of a kernel that
/// names the instruction, a `let _ = NAME;` of the word, for Rust to resolve it there as it resolves a name of the
/// kernel's own. Each is under the `cfg` of the item that names the instruction, and the function under that of any of
/// them: Rust has no word of an item that `cfg` leaves out, and nothing is never used. Rust has nothing to say of any
/// of it, and an instruction that its machine has no doc of has no function: if it is no instruction, its error is its
/// own. The function is named as the instruction is, with a `_` for each character that is no part of a name,
/// `lock.bts` as `lock_bts`; a name that two have, `add` of x86 and of the GPU, or `a.b` and `a_b`, is told apart by a
/// number.
fn docs(made: &ModuleOut, conds: &[Cond], at: &dyn Fn(lang::Span) -> Span) -> TokenStream {
    let (mut items, mut reads) = (TokenStream::new(), TokenStream::new());
    let mut taken: Vec<String> = vec![];
    for documented in &made.instructions {
        let words: String = documented.name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        let (mut name, mut n) = (words.clone(), 1);
        while taken.contains(&name) {
            n += 1;
            name = format!("{words}_{n}");
        }
        let kept = Cond::any(documented.places.iter().map(|&(i, _)| cond_of(conds, i)));
        let Some(guard) = kept.guard() else { continue };
        items.extend(guard);
        items.extend(parse(&format!("#[doc = {:?}] fn", documented.doc)));
        items.extend(ident_at(&name, site()));
        items.extend(parse("() {}"));
        for &(i, word) in &documented.places {
            if let Some(guard) = cond_of(conds, i).guard() {
                reads.extend(guard);
                reads.extend(parse("let _ ="));
                reads.extend(ident_at(&name, at(word)));
                reads.extend(parse(";"));
            }
        }
        taken.push(name);
    }
    if items.is_empty() {
        return items;
    }
    items.extend(reads);
    let mut out = parse("const _: () =");
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, items))]);
    out.extend(parse(";"));
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
            names.extend(name_at(item, top, site()));
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

/// `let _ = what;` under `guard`: Rust finds `what` there, and says it cannot if it cannot.
fn find(guard: TokenStream, what: impl IntoIterator<Item = TokenTree>) -> TokenStream {
    let mut find = guard;
    find.extend(parse("let _ ="));
    find.extend(what);
    find.extend(parse(";"));
    find
}

/// What says, at each place item `i` of `made` names an item of its module, that `cfg` may leave the
/// item out; `place` is the span of the place. Rust has a name for a const, a static of the process's
/// and an x86 kernel that takes no compile-time arguments, and finds it where it is written: a
/// `let _ = NAME;` there, for Rust to say it cannot find the name if `cfg` leaves out the item where
/// it keeps item `i`; these are the first of the pair this gives back. They tell of a deprecated item at the place, as
/// Rust does of any use, so a place that [`operands`] gives, where `naked` writes the operand that tells of it, has
/// none; but where `cfg` leaves the item out, the operand is `const 0`, which names nothing, and the `let _` says that
/// Rust cannot find the item. For any other item kernite says so itself, as an error where `cfg` keeps item `i` and
/// leaves that one out; these are the second, for they are items of the module. A static of GPU kernels has a method
/// with its attributes, so the first of the pair has `let _ = Kernels::NAME;` for it too, where `cfg` keeps the static,
/// for Rust to tell of a deprecated one at the place. Both are under the `cfg` of item `i` and of the code that has the
/// place, where item `i` runs it: [`runs`]. A constant of a module of Rust's that item `i` names by its path,
/// `std::f32::consts::PI`, is Rust's to say as well, in the first of the pair, by [`takes`]: that the path gives the
/// constant kernite took. So is one that a const that item `i` names has in its definition, where `cfg` keeps the
/// const; but not one in a const that the module is given, whose path is Rust's where the consts are, in a module that
/// holds nothing to give `std` another meaning.
fn named(
    made: &ModuleOut,
    conds: &[Cond],
    i: usize,
    top: &[TokenTree],
    place: &dyn Fn(lang::Span) -> Span,
) -> (TokenStream, TokenStream) {
    let module = &made.module;
    let (mut names, mut errors) = (TokenStream::new(), TokenStream::new());
    let Some((reader, made_reader)) = module.items.get(i).zip(made.items.get(i)) else { return (names, errors) };
    let covered = match made_reader {
        ItemOut::Native(k) => operands(k),
        _ => vec![],
    };
    for m in made_reader.mentions() {
        let at = module.item_at(m.span);
        let kept = Cond::all([cond_of(conds, i), runs(made, conds, i, at, &[])]);
        if m.value.is_some() {
            let given = matches!(made.items.get(at), Some(ItemOut::Given));
            names.extend(kept.guard().filter(|_| !given).map(|guard| takes(guard, m, place)));
            continue;
        }
        let Some(j) = position(module, &m.name) else { continue };
        let Some((item, made_item)) = module.items.get(j).zip(made.items.get(j)) else { continue };
        let rust = match (&item.kind, made_item) {
            (_, ItemOut::Rust | ItemOut::Given | ItemOut::HostStatic(_)) => true,
            (ItemKind::Fn(f), ItemOut::Native(_)) => f.generics.is_empty(),
            _ => false,
        };
        if rust {
            let kept = match covered.iter().any(|x| x.span == m.span) {
                true => Cond::all([kept, cond_of(conds, j).not()]),
                false => kept,
            };
            let Some(guard) = kept.guard() else { continue };
            names.extend(find(guard, name_at(item, top, place(m.span))));
            continue;
        }
        if let Some(guard) = Cond::all([kept.clone(), cond_of(conds, j).not()]).guard() {
            let reader = reader.name().unwrap_or_default();
            let message = format!("`{}` is left out by `cfg` where `{reader}`, which names it, is kept", m.name);
            errors.extend(error_under(&guard, place(m.span), &message));
        }
        if let ItemOut::DeviceStatic(laid) = made_item
            && let Some(path) = used(item, laid, top, place(m.span))
            && let Some(guard) = Cond::all([kept, cond_of(conds, j)]).guard()
        {
            names.extend(find(guard, path));
        }
    }
    (names, errors)
}

/// The condition under which item `i` of `made` runs the code of item `c`, but for the `cfg` of `i`:
/// always for its own code, and for a helper's, where `cfg` keeps the helper and `i` runs the code that
/// calls it, at any place `i` names the helper. A const's definition is worked out wherever `i` names
/// the const, by that const or by another: where `cfg` keeps the const. `way` holds the items followed
/// back from the place to `c`: a place in one of them adds nothing, for `i` reaches that item by the
/// places already counted. So a helper that calls itself, or two that call each other, do not go on
/// without end.
fn runs(made: &ModuleOut, conds: &[Cond], i: usize, c: usize, way: &[usize]) -> Cond {
    let module = &made.module;
    if c == i {
        return Cond::Always;
    }
    let (Some(made_reader), Some(callee)) = (made.items.get(i), module.items.get(c)) else { return Cond::Never };
    if matches!(callee.kind, ItemKind::Const(_)) {
        return cond_of(conds, c);
    }
    let way = [way, &[c]].concat();
    let calls = made_reader.mentions().iter().filter(|m| callee.name() == Some(&m.name));
    let callers = calls.map(|m| module.item_at(m.span)).filter(|caller| !way.contains(caller));
    Cond::all([cond_of(conds, c), Cond::any(callers.map(|caller| runs(made, conds, i, caller, &way)))])
}

/// When item `i` is there, `conds` saying it of each item of the module: never for an item the module has not.
fn cond_of(conds: &[Cond], i: usize) -> Cond {
    conds.get(i).cloned().unwrap_or(Cond::Never)
}
