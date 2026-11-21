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
use crate::gpu::{HOLE, gpu, used};
use crate::room;
use crate::rust::{
    LEVELS, attrs, cfg_of, deprecated, error, error_under, flatten, for_rust, from_inside, grouped, inner_attribute,
    levels, lex, name_at, parse, passable, position, refused_attrs, seen, site, stand_in, stub, takes, unnamed,
};
use crate::workspace::workspace;
use crate::x86::{cpu_static, instance_used, naked, operands, x86_stub};

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
    let mut out: TokenStream = module
        .items
        .iter()
        .filter(|item| given.is_none() && inner_attribute(&top[item.tokens.clone()]))
        .flat_map(|item| top[item.tokens.clone()].iter().cloned())
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
    // Which items are those, and, for each, the places that name an item in its code, as `named` says them: Rust has no
    // function of a helper or a GPU function to put them in.
    let helper: Vec<bool> = made
        .items
        .iter()
        .map(|item| matches!(item, ItemOut::Helper(_)) || matches!(item, ItemOut::Device(d) if d.launcher.is_none()))
        .collect();
    let mut carried = vec![TokenStream::new(); helper.len()];
    // Where each of them has a stand-in, as `named` says, in which Rust reads its `deprecated`.
    let mut stood_in = vec![Cond::Never; helper.len()];
    // Which functions and statics set the level of a lint: only their places are written where the level holds.
    let leveled: Vec<bool> = module
        .items
        .iter()
        .map(|item| match &item.kind {
            ItemKind::Fn(f) => !levels(attrs(f, &top), &LEVELS).is_empty(),
            ItemKind::Static(s) => !levels(top[s.attrs.clone()].iter().cloned(), &LEVELS).is_empty(),
            _ => false,
        })
        .collect();
    // Each listing the process's kernels have, by name, under the `cfg` of the kernel it is of, as
    // what `naked` writes for it is.
    let mut listed = vec![];
    for (i, (item, made_item)) in module.items.iter().zip(&made.items).enumerate() {
        let tokens = &top[item.tokens.clone()];
        let cfg = &cfgs[i];
        let said = |e: &lang::Error| error_under(cfg, at(e.span), &e.message);
        // Where the kernel first writes the name of each item its code names by an operand.
        let firsts = match made_item {
            ItemOut::Native(k) => operands(k, &|m| module.item_at(m.span) == i),
            _ => vec![],
        };
        let (names, errors) = named(&made, &conds, i, &top, &place, &firsts, &mut stood_in);
        left_out.extend(errors);
        // Each place is where the code that writes it is, for the level its item sets to hold there, as in Rust: in the
        // body of the kernel, or in the static, which `own` has, or in what holds the places of the helper or the GPU
        // function. A place of an item that sets no level, or of a const, is in a block of the module, where Rust tells
        // of it at the level of the module.
        let mut own = TokenStream::new();
        for (at, statement) in names {
            if helper[at] && leveled[at] {
                carried[at].extend(statement);
            } else if at == i && leveled[i] {
                own.extend(statement);
            } else {
                reads.extend(statement);
            }
        }
        // A helper or an `extern "ptx-func"` function is no item of Rust's: it takes the attributes that say how Rust
        // has it.
        if helper[i]
            && let ItemKind::Fn(f) = &item.kind
        {
            let is = if f.abi.is_some() { "an `extern \"ptx-func\"` function" } else { "a helper" };
            out.extend(refused_attrs(attrs(f, &top), &top[f.head.end + 1], (is, is), cfg));
        }
        match (&item.kind, made_item) {
            // The module's own attributes are first, and a const given is Rust's where its
            // module is.
            (_, ItemOut::Rust) if inner_attribute(tokens) => {}
            (_, ItemOut::Rust) => out.extend(tokens.iter().cloned()),
            (_, ItemOut::Given) => {}
            (ItemKind::Static(s), ItemOut::HostStatic(laid)) => {
                out.extend(laid.as_ref().err().and_then(|failed| failed.error.as_ref()).map(said));
                let held = laid.as_ref().map_or_else(|failed| failed.stub.as_ref(), |c| Some(&c.code));
                out.extend(cpu_static(s, held, cfg, &top, own));
            }
            (ItemKind::Static(s), ItemOut::DeviceStatic(laid)) => {
                out.extend(laid.as_ref().err().and_then(|failed| failed.error.as_ref()).map(said));
                let is = ("memory on the GPU", "a static of GPU kernels");
                out.extend(refused_attrs(top[s.attrs.clone()].iter().cloned(), &top[s.vis.end + 1], is, cfg));
                memory.push((s, laid, own, conds[i].clone()));
                on_gpu.push(conds[i].clone());
                if let Ok(laid) = laid {
                    fields.extend(room::room(cfg, i, &laid.code.ty, at(s.ty.span())));
                }
            }
            (ItemKind::Fn(f), ItemOut::Helper(errors)) => {
                out.extend(errors.iter().map(said));
                helpers.push((i, f));
            }
            (ItemKind::Fn(f), ItemOut::Unknown(e)) => {
                out.extend(e.as_ref().map(said));
                out.extend(stub(f, &top, TokenStream::new()));
            }
            (ItemKind::Fn(f), ItemOut::Native(k)) => {
                out.extend(k.not_unsafe.as_ref().map(said));
                out.extend(k.said.as_ref().map(said));
                let first = |name: &str| {
                    place(firsts.iter().find(|m| m.name == name).expect("the kernel writes what its code names").span)
                };
                out.extend(match k.written {
                    true => naked(module, f, k, cfg, &top, &first, own),
                    false => x86_stub(f, k.count, &k.touched, cfg, &top, own),
                });
                out.extend(k.errors.iter().map(said));
                listed.extend(k.listings.iter().map(|listing| (cfg, listing)));
            }
            (ItemKind::Fn(f), ItemOut::Device(d)) => {
                out.extend(d.not_unsafe.as_ref().map(said));
                out.extend(d.said.as_ref().map(said));
                out.extend(d.errors.iter().map(said));
                on_gpu.push(conds[i].clone());
                launchers.extend(d.launcher.as_ref().map(|l| (f, l, own)));
                if d.launcher.is_none() {
                    helpers.push((i, f));
                }
            }
            _ => unreachable!("kernite::module says what each item is"),
        }
    }
    let laid = Cond::any(memory.iter().filter(|(_, m, ..)| m.is_ok()).map(|(.., c)| c.clone()));
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
    for (i, f) in helpers {
        reads.extend(Cond::all([conds[i].clone(), failing.clone()]).guard().map(|g| seen(f, &top, &g)));
        shown.extend(conds[i].guard().map(|g| room::shown(&g, i, seen(f, &top, &TokenStream::new()))));
        // What Rust reads of a helper or a GPU function, whose code it never sees, is in items of no name that have it.
        // Its `deprecated` is in one under the item's own `cfg`, but for where a stand-in has it, as `stood_in` says:
        // so Rust says what is wrong of it once, whether a kernel names the item or not. The level it sets, `expect`
        // too, which is unfulfilled where no place the code names has the lint, is in another, with the places of its
        // code that name an item. The code of a helper is read only where an item `cfg` keeps runs it, and only there
        // is that item.
        let says = deprecated(attrs(f, &top));
        if !says.is_empty()
            && let Some(guard) = Cond::all([conds[i].clone(), stood_in[i].clone().not()]).guard()
        {
            out.extend(guard);
            out.extend(says);
            out.extend(parse("const _: () = {};"));
        }
        let read = match made.items[i] {
            ItemOut::Helper(_) => Cond::any(
                (0..made.items.len())
                    .filter(|&k| k != i)
                    .map(|k| Cond::all([conds[k].clone(), runs(&made, &conds, k, i, &[])])),
            ),
            _ => conds[i].clone(),
        };
        if leveled[i]
            && let Some(guard) = read.guard()
        {
            let places = std::mem::take(&mut carried[i]);
            out.extend(guard);
            out.extend(levels(attrs(f, &top), &LEVELS));
            out.extend(if places.is_empty() { parse("const _: () = {};") } else { unnamed(places) });
        }
    }
    out.extend(room::carrier(fields, shown));
    out.extend(left_out);
    out.extend(unnamed(reads));
    out.extend(tail);
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
/// for Rust to tell of a deprecated one at the place. So has an instance of an x86 kernel, `let _ = old::<2>;`: the
/// code names the function of the instance, which has no attributes, and Rust cannot work out the arguments of `old`
/// alone. So has a deprecated helper, `extern "ptx-func"` function or GPU kernel, whose code Rust never sees: one of
/// its name and its `deprecated` stands in for it, in a block, [`stand_in`]. A place in the item itself has none, for
/// Rust does not tell of a use inside the item that is deprecated. Both are under the `cfg` of item `i` and of the code
/// that has the place, where item `i` runs it: [`runs`]. A constant of a module of Rust's that item `i` names by its
/// path, `std::f32::consts::PI`, is Rust's to say as well, in the first of the pair, by [`takes`]: that the path gives
/// the constant kernite took. So is one that a const that item `i` names has in its definition, where `cfg` keeps the
/// const; but not one in a const that the module is given, whose path is Rust's where the consts are, in a module that
/// holds nothing to give `std` another meaning.
///
/// Each statement of the first is given with the item whose code writes the place, for the level that item sets to hold
/// there. The operand of a place of a kernel is in the function of each instance, where the kernel's levels hold, but a
/// helper's do not, and where an `expect` is an `allow`: so where a kernel allows a lint or expects it, each place has
/// the statement too, to be told of at the level of its code, as one that is not an operand is.
///
/// `stood_in` of an item gains where each stand-in of it is there: Rust reads the item's `deprecated` in it.
fn named(
    made: &ModuleOut,
    conds: &[Cond],
    i: usize,
    top: &[TokenTree],
    place: &dyn Fn(lang::Span) -> Span,
    covered: &[&lang::Mention],
    stood_in: &mut [Cond],
) -> (Vec<(usize, TokenStream)>, TokenStream) {
    let module = &made.module;
    let (mut names, mut errors) = (vec![], TokenStream::new());
    let lowers =
        matches!(&module.items[i].kind, ItemKind::Fn(f) if !levels(attrs(f, top), &["allow", "expect"]).is_empty());
    for m in made.items[i].mentions() {
        let at = module.item_at(m.span);
        let kept = Cond::all([conds[i].clone(), runs(made, conds, i, at, &[])]);
        if m.value.is_some() {
            let given = matches!(made.items[at], ItemOut::Given);
            names.extend(kept.guard().filter(|_| !given).map(|guard| (at, takes(guard, m, place))));
            continue;
        }
        let j = position(module, &m.name);
        let rust = match (&module.items[j].kind, &made.items[j]) {
            (_, ItemOut::Rust | ItemOut::Given | ItemOut::HostStatic(_)) => true,
            (ItemKind::Fn(f), ItemOut::Native(_)) => f.generics.is_empty(),
            _ => false,
        };
        if rust {
            let kept = match covered.iter().any(|x| x.span == m.span) && !lowers {
                true => Cond::all([kept, conds[j].clone().not()]),
                false => kept,
            };
            let Some(guard) = kept.guard() else { continue };
            names.push((at, find(guard, [name_at(&module.items[j], top, place(m.span))])));
            continue;
        }
        if let Some(guard) = Cond::all([kept.clone(), conds[j].clone().not()]).guard() {
            let reader = module.items[i].name().expect("an item that names others has a name");
            let message = format!("`{}` is left out by `cfg` where `{reader}`, which names it, is kept", m.name);
            errors.extend(error_under(&guard, place(m.span), &message));
        }
        let path = match &made.items[j] {
            ItemOut::DeviceStatic(laid) => used(&module.items[j], laid, top, place(m.span)),
            ItemOut::Native(_) if at != j => Some(instance_used(&module.items[j], &m.args, top, place(m.span))),
            ItemOut::Helper(_) | ItemOut::Device(_) if at != j => stand_in(&module.items[j], top, place(m.span)),
            _ => None,
        };
        if let Some(path) = path {
            let there = Cond::all([kept, conds[j].clone()]);
            if let Some(guard) = there.guard() {
                names.push((at, find(guard, path)));
                stood_in[j] = Cond::any([stood_in[j].clone(), there]);
            }
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
    if matches!(module.items[c].kind, ItemKind::Const(_)) {
        return conds[c].clone();
    }
    let way = [way, &[c]].concat();
    let calls = made.items[i].mentions().iter().filter(|m| module.items[c].name() == Some(&m.name));
    let callers = calls.map(|m| module.item_at(m.span)).filter(|caller| !way.contains(caller));
    Cond::all([conds[c].clone(), Cond::any(callers.map(|caller| runs(made, conds, i, caller, &way)))])
}
