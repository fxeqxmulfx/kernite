//! What a module of kernels becomes once kgen has compiled its kernels: each item as what
//! `kgen::module` says it is, written out for Rust.

use std::path::Path;

use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

use kgen::lang::{self, ItemKind};
use kgen::{ItemOut, Launcher, ModuleOut};

use crate::gpu::{HOLE, gpu};
use crate::rust::{const_names, error, flatten, inner_attribute, is_ident, is_punct, lex, parse, site, stub};
use crate::workspace::workspace;
use crate::x86::{cpu_static, naked, x86_stub};

/// What the module's braces hold once its kernels are compiled; `given`, the consts it takes
/// from outside, and the path of their module, as `#[kernels(consts = path)]` writes it.
pub(crate) fn expand(given: Option<(TokenStream, &[TokenTree])>, body: TokenStream) -> TokenStream {
    // The consts given come first, for the kernels to read as the module's own.
    let outside = given.as_ref().map_or_else(Vec::new, |(consts, _)| flatten(consts.clone()));
    let n = outside.len();
    let top: Vec<TokenTree> = outside.into_iter().chain(flatten(body)).collect();
    let (trees, spans) = lex(&top);
    let at = |s: lang::Span| spans.get(s.0 as usize).copied().unwrap_or_else(site);
    let said = |e: &lang::Error| error(at(e.span), &e.message);
    let module = match trees.and_then(|trees| lang::module(&trees, n)) {
        Ok(module) => module,
        Err(e) => return said(&e),
    };
    // The GPU kernels' namespaces: the workspace's, then the module's path. A name that is not a
    // string is an error at the module; the macro goes on without it, so that the error is the
    // only one.
    let (workspace, manifest, refused) = match workspace() {
        Ok((workspace, manifest)) => (workspace, manifest, None),
        Err(message) => (String::new(), None, Some(message)),
    };
    let names: Vec<&str> = [workspace.as_str()].into_iter().filter(|n| !n.is_empty()).collect();
    let scope = kgen::ptx_scope(&names) + HOLE;
    let consts = given
        .as_ref()
        .map(|(_, path)| kgen::Given { before: n, from: path.iter().map(TokenTree::to_string).collect() });
    let made = kgen::module(module, consts, &scope);
    let module = &made.module;
    // The module's own attributes, `#![..]`, before anything the macro writes, where Rust has
    // them.
    let mut out: TokenStream = module
        .items
        .iter()
        .filter(|item| inner_attribute(&top[item.tokens.clone()]))
        .flat_map(|item| top[item.tokens.clone()].iter().cloned())
        .collect();
    out.extend(refused.map(|message| error(site(), &message)));
    if let Some((_, path)) = given {
        out.extend(made.given.iter().map(said));
        out.extend(take(&made, &top, path));
    }
    out.extend(made.owner.as_ref().map(said));
    // How each kernel the GPU launches is, and each static of the GPU's, as the host reads it.
    let (mut launchers, mut memory) = (vec![], vec![]);
    for (item, made_item) in module.items.iter().zip(&made.items) {
        let tokens = &top[item.tokens.clone()];
        match (&item.kind, made_item) {
            // The module's own attributes are first, and a const given is Rust's where its
            // module is.
            (_, ItemOut::Rust) if inner_attribute(tokens) => {}
            (_, ItemOut::Rust) => out.extend(tokens.iter().cloned()),
            (_, ItemOut::Given) => {}
            (ItemKind::Static(s), ItemOut::HostStatic(laid)) => {
                out.extend(laid.as_ref().err().map(said));
                out.extend(cpu_static(s, laid.as_ref().ok(), &top));
            }
            (ItemKind::Static(s), ItemOut::DeviceStatic(laid)) => {
                out.extend(laid.as_ref().err().map(said));
                memory.push((s, laid.as_ref().ok()));
            }
            (ItemKind::Fn(_), ItemOut::Helper(errors)) => out.extend(errors.iter().map(said)),
            (ItemKind::Fn(f), ItemOut::Unknown(e)) => {
                out.extend(said(e));
                out.extend(stub(f, &top));
            }
            (ItemKind::Fn(f), ItemOut::Native(k)) => {
                out.extend(k.not_unsafe.as_ref().map(said));
                out.extend(k.said.as_ref().map(said));
                out.extend(match k.written {
                    true => naked(module, f, &k.instances, &k.touched, &top),
                    false => x86_stub(f, k.count, &k.touched, &top),
                });
                out.extend(k.errors.iter().map(said));
            }
            (ItemKind::Fn(f), ItemOut::Device(d)) => {
                out.extend(d.not_unsafe.as_ref().map(said));
                out.extend(d.said.as_ref().map(said));
                out.extend(d.errors.iter().map(said));
                launchers.extend(d.launcher.as_ref().map(|l| match l {
                    Launcher::Never => (f, None),
                    Launcher::Of(launched) => (f, Some(launched.as_slice())),
                }));
            }
            _ => unreachable!("kgen::module says what each item is"),
        }
    }
    out.extend(used(module, &made.read, &top));
    if !made.listings.is_empty() {
        let mut s = String::from(
            "/// Each x86 kernel's listing, by name, an instance's as `tile::<4>`: where its variables\n\
             /// live, and its code with its notes, for snapshots.\n\
             pub const LISTINGS: &[(&str, &str)] = &[",
        );
        for (name, text) in &made.listings {
            s += &format!("({name:?}, {text:?}), ");
        }
        out.extend(parse(&(s + "];")));
    }
    if let Some(device) = &made.device {
        out.extend(device.error.as_ref().map(said));
        out.extend(gpu(&launchers, &device.symbols, &memory, &device.text, &top, &at));
        // The workspace's name is in the PTX, so it is built again when it changes.
        if let Some(manifest) = manifest.as_deref().and_then(Path::to_str) {
            out.extend(parse(&format!("const _: &[u8] = ::core::include_bytes!({manifest:?});")));
        }
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
            names.extend([const_name(item, top)]);
            names.extend(parse(","));
        }
    }
    let mut out = parse("use");
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

/// A `const _` that names each constant of `module` that `read` gives by its index among the
/// items, by its name as `top` writes it, `r#type` as well. The kernels read them, but Rust never
/// sees a kernel's body run, so it would find them unused.
fn used(module: &lang::Module, read: &[usize], top: &[TokenTree]) -> TokenStream {
    let mut lets = TokenStream::new();
    for &i in read {
        lets.extend(parse("let _ ="));
        lets.extend([const_name(&module.items[i], top)]);
        lets.extend(parse(";"));
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
