//! Rust's tokens: the language's trees made of them, the parts of a module or a function as
//! written, and tokens the macro makes, each at the span that says where Rust resolves it.

use std::cell::Cell;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

use kernite::lang::{self, Delim, Fn, ItemKind, Tree, Value};

thread_local! {
    /// The span of each token the macro writes, which says where Rust resolves its names, and
    /// whether it lints them, as it does the user's tokens and not the macro's: none for the
    /// macro's own call site; or, for a module given consts, the path's first token with the
    /// attribute's hygiene, which `#[kernels(consts = path)]` hands on in the group of the path.
    /// That is where the module is, since `__kernels!` is called from inside the consts' macro,
    /// whose call site is not; and Rust, which takes it for the macro's, lints none of it.
    static SITE: Cell<Option<Span>> = const { Cell::new(None) };
}

/// `SITE` at a span until this is dropped, the macro panicking or not: a span of one call of
/// the macro is none in the next, and the thread that runs it may run the next.
pub(crate) struct Site;

impl Site {
    pub(crate) fn at(span: Span) -> Site {
        SITE.set(Some(span));
        Site
    }
}

impl Drop for Site {
    fn drop(&mut self) {
        SITE.set(None);
    }
}

/// The braces of the module `tokens` are, and what is before them, if it is written in braces.
pub(crate) fn braced(tokens: &[TokenTree]) -> Option<(&Group, &[TokenTree])> {
    match tokens.split_last() {
        Some((TokenTree::Group(braces), head))
            if braces.delimiter() == Delimiter::Brace && head.iter().any(|t| is_ident(t, "mod")) =>
        {
            Some((braces, head))
        }
        _ => None,
    }
}

/// The module `head` begins, its braces at `braces`, holding `inner`.
pub(crate) fn module(head: &[TokenTree], braces: &Group, inner: TokenStream) -> TokenStream {
    let mut out: TokenStream = head.iter().cloned().collect();
    let mut group = Group::new(Delimiter::Brace, inner);
    group.set_span(braces.span());
    out.extend([TokenTree::Group(group)]);
    out
}

/// Whether `tokens` are a path, `crate::model::sizes`.
pub(crate) fn is_path(tokens: &[TokenTree]) -> bool {
    let mut rest = match tokens {
        [a, b, rest @ ..] if is_punct(a, ':') && is_punct(b, ':') => rest,
        _ => tokens,
    };
    loop {
        match rest {
            [TokenTree::Ident(_)] => return true,
            [TokenTree::Ident(_), a, b, tail @ ..] if is_punct(a, ':') && is_punct(b, ':') => rest = tail,
            _ => return false,
        }
    }
}

/// Whether `tokens`, an item's, are an attribute of the module's own, `#![..]`.
pub(crate) fn inner_attribute(tokens: &[TokenTree]) -> bool {
    matches!(tokens, [hash, bang, ..] if is_punct(hash, '#') && is_punct(bang, '!'))
}

/// How many of `tokens`, an item's, are its attributes, `#[..]` each.
pub(crate) fn attrs_len(tokens: &[TokenTree]) -> usize {
    let mut n = 0;
    while let [hash, TokenTree::Group(g), ..] = &tokens[n..]
        && is_punct(hash, '#')
        && g.delimiter() == Delimiter::Bracket
    {
        n += 2;
    }
    n
}

/// `f`'s visibility as written, `pub` or `pub(..)`, or none.
pub(crate) fn visibility<'a>(f: &Fn, top: &'a [TokenTree]) -> &'a [TokenTree] {
    let head = &top[f.head.clone()];
    let vis = match head {
        [pub_, TokenTree::Group(g), ..] if is_ident(pub_, "pub") && g.delimiter() == Delimiter::Parenthesis => 2,
        [pub_, ..] if is_ident(pub_, "pub") => 1,
        _ => 0,
    };
    &head[..vis]
}

/// `const { if .. }` on `f`'s compile-time arguments, by their names as written, `r#in` as well:
/// what `arms` gives for each instance, by its arguments. Another is no instance, and Rust
/// refuses to compile the call. It is a chain of `if`, not a `match`: where the instances are
/// every value, `true` and `false` of a `bool`, no value reaches a `match`'s last arm, and Rust
/// warns of it; and the macro states no rule of which values a type has.
pub(crate) fn pick(f: &Fn, arms: &[(Vec<Value>, String)], top: &[TokenTree]) -> String {
    let names: Vec<String> = const_names(written_generics(f, top)).map(TokenTree::to_string).collect();
    let mut out = String::from("const { ");
    let mut listed = vec![];
    for (args, then) in arms {
        let equal: Vec<String> = names.iter().zip(args).map(|(name, arg)| format!("{name} == {arg}")).collect();
        out += &format!("if {} {{ {then} }} else ", equal.join(" && "));
        listed.push(format!("`{}`", lang::instance(f, args).unwrap_or_default()));
    }
    let message = format!("`{}` is compiled for {} alone, as its `#[instances(..)]` lists", f.name, listed.join(", "));
    out + &format!("{{ ::core::panic!({message:?}) }} }}")
}

/// `f`'s generics as written, `<const N: usize>`; none if it has none.
pub(crate) fn written_generics<'a>(f: &Fn, top: &'a [TokenTree]) -> &'a [TokenTree] {
    &top[f.head.end + 2..f.params_at]
}

/// The name after each `const` of `tokens`, as written, `r#type` as well: a const's, or those of
/// a function's compile-time arguments.
pub(crate) fn const_names(tokens: &[TokenTree]) -> impl Iterator<Item = &TokenTree> {
    tokens.windows(2).filter(|w| is_ident(&w[0], "const")).map(|w| &w[1])
}

/// The place of the item of `module` named `name`, among its items.
pub(crate) fn position(module: &lang::Module, name: &str) -> usize {
    module.items.iter().position(|item| item.name() == Some(name)).expect("an item of the module")
}

/// What [`carried`] keeps of `item`'s attributes when it offers nothing: its `cfg`.
pub(crate) fn cfg_of(item: &lang::Item, top: &[TokenTree]) -> TokenStream {
    let tokens = &top[item.tokens.clone()];
    carried(tokens[..attrs_len(tokens)].iter().cloned(), false)
}

/// The name of `item`, a const, a static or a function, as `top` writes it, `r#type` as well, of the
/// span `span`.
pub(crate) fn name_at(item: &lang::Item, top: &[TokenTree], span: Span) -> TokenTree {
    let mut name = match &item.kind {
        ItemKind::Const(_) => const_names(&top[item.tokens.clone()]).next().expect("a const has a name"),
        ItemKind::Static(s) => &top[s.vis.end + 1],
        ItemKind::Fn(f) => &top[f.head.end + 1],
        ItemKind::Other => unreachable!("only an item with a name is named"),
    }
    .clone();
    name.set_span(span);
    name
}

/// An error at each of `f`'s compile-time arguments named as one of `own`, the locals the Rust of
/// `method` binds for itself, under `f`'s `cfg`. A compile-time argument is no local, and no span of
/// stable Rust keeps it apart from one of its name, as it does a parameter's.
pub(crate) fn clashes(f: &Fn, own: &[&str], method: &str, top: &[TokenTree]) -> TokenStream {
    let cfg = carried(attrs(f, top), false);
    let mut errors = TokenStream::new();
    for name in const_names(written_generics(f, top)) {
        let written = name.to_string();
        let name_of = written.trim_start_matches("r#");
        if own.contains(&name_of) {
            errors.extend(error_under(
                &cfg,
                name.span(),
                &format!("`{name_of}` names a parameter of {method} as well"),
            ));
        }
    }
    errors
}

/// `f`'s parameters as written, each its tokens, `x: *mut f32`, and the place of its colon. A comma
/// ends a parameter unless it is among the arguments of a type's, `x: *mut Pair<u8, 2>`. The `>` of
/// an arrow, `fn() -> u8`, closes no `<` of them. Rust cannot pass a parameter named `_` on, so
/// each is named here, for every signature and body the macros write for `f`: it takes the next of
/// `__kernite_0`, `__kernite_1` and on that no parameter or compile-time argument of `f` is named.
pub(crate) fn written_params(f: &Fn, top: &[TokenTree]) -> Vec<(Vec<TokenTree>, usize)> {
    let TokenTree::Group(group) = &top[f.params_at] else { unreachable!("parameters are in parentheses") };
    let tokens = flatten(group.stream());
    let (mut open, mut dash) = (0usize, false);
    let params = tokens
        .split(|t| {
            if is_punct(t, '<') {
                open += 1;
            } else if is_punct(t, '>') && !dash {
                open = open.saturating_sub(1);
            }
            // A `>` after a `-` joint to it is an arrow's, `->`.
            dash = matches!(t, TokenTree::Punct(p) if p.as_char() == '-' && p.spacing() == Spacing::Joint);
            open == 0 && is_punct(t, ',')
        })
        .filter(|p| !p.is_empty());
    let taken: Vec<&str> =
        f.generics.iter().map(|g| &g.name).chain(f.params.iter().map(|p| &p.name)).map(String::as_str).collect();
    let mut free = (0..).map(|n| format!("__kernite_{n}")).filter(|name| !taken.contains(&name.as_str()));
    params
        .map(|p| {
            let mut p = p.to_vec();
            let colon = p.iter().position(|t| is_punct(t, ':')).expect("a parameter has a type");
            if is_ident(&p[colon - 1], "_") {
                p[colon - 1] = TokenTree::Ident(Ident::new(&free.next().expect("names do not run out"), site()));
            }
            (p, colon)
        })
        .collect()
}

/// `"const"` or `"mut"`, and the tokens of the type it points to, if `ty` is a pointer, `*const T`
/// or `*mut T`. The macros read a pointer here and nowhere else.
pub(crate) fn pointer(ty: &[TokenTree]) -> Option<(&'static str, &[TokenTree])> {
    match ty {
        [star, kind, to @ ..] if is_punct(star, '*') => {
            ["const", "mut"].into_iter().find(|k| is_ident(kind, k)).map(|kind| (kind, to))
        }
        _ => None,
    }
}

/// The name of the function the macros write into a body to ask Rust whether a pointer has a size.
const THIN: &str = "__kernite_thin";

/// What Rust is asked of a type of a kernel's, and where the type is.
#[derive(Clone, Copy)]
enum Ask {
    /// `"const"` or `"mut"`, and where the type it points to begins, for a pointer of one word,
    /// `*const T` or `*mut T`: whether `T` has a size.
    Pointer(&'static str, Span),
    /// A number by its name, and where the name is: whether it is that number, the primitive
    /// `::core::primitive::u64`, and not another type the module gives the name. Rust is asked of
    /// the type alone, never of a value of it.
    Number(&'static str, Span),
}

/// The number `ty` is, and where its name is, if `ty` is one by its name: `u8` to `u64`, `i8` to
/// `i64`, `usize`, `isize`, `f32`, `f64` or `bool`, which the language reads by their names.
fn number_name(ty: &[TokenTree]) -> Option<(&'static str, Span)> {
    let [name] = ty else { return None };
    let text = name.to_string();
    let known = lang::IntTy::parse(&text).map(lang::IntTy::name);
    known.or_else(|| ["bool", "f32", "f64"].into_iter().find(|n| *n == text)).map(|n| (n, name.span()))
}

/// For each of `f`'s parameters, and last for its result: what Rust is asked of its type, if it is
/// asked. It is asked of a number by its name, as [`number_name`] reads it, and not of `f16` and
/// `bf16`, which Rust has not; and of a pointer of one word, `*const T` or `*mut T`. The language
/// says which: a pointer to a slice, a `str` or a `dyn` has two words, and is its error alone,
/// which Rust is not asked of, for it would say the same twice.
fn asked(f: &Fn, top: &[TokenTree]) -> Vec<Option<Ask>> {
    let ask = |ty: &[TokenTree]| {
        if let Some((name, at)) = number_name(ty) {
            return Some(Ask::Number(name, at));
        }
        let (kind, to) = pointer(ty)?;
        let one = lex(ty).0.is_ok_and(|trees| lang::pointer(&lang::text(&trees)) == Ok(true));
        one.then(|| Ask::Pointer(kind, to[0].span()))
    };
    let params = written_params(f, top).into_iter().map(|(param, colon)| ask(&param[colon + 1..]));
    params.chain([f.ret.as_ref().and_then(|r| ask(&top[r.tokens.clone()]))]).collect()
}

/// What asks Rust of each type `f` takes or gives back, in a body that has the parameters by their
/// names: [`numbers`] asks of the numbers and [`pointers`] of the pointers, for which a function
/// that takes a pointer as `ptr` is written. A result that is a pointer is asked of where it is
/// given back. Nothing if `f` has none to ask of.
pub(crate) fn checks(f: &Fn, top: &[TokenTree], ptr: &str) -> TokenStream {
    let names: Vec<TokenStream> =
        written_params(f, top).iter().map(|(param, colon)| param[colon - 1].clone().into()).collect();
    let asked = asked(f, top);
    let mut out = numbers(&asked);
    out.extend(pointers(&asked, ptr, &names));
    out
}

/// The block of a `const` operand of a naked kernel's own `naked_asm!`, which asks Rust of each
/// type the kernel takes or gives back: a closure for each number, which Rust checks and never
/// runs, and, if the kernel has a pointer to ask of, one that calls the kernel on its parameters,
/// and asks of them and of what it gives back. A call is made for the pointers alone: a number is
/// asked of by its name, and a pointer by a value of it. Nothing if the kernel has none to ask of.
/// A naked function has no body to write the questions in, and no name of the kernel's finds the
/// closure's locals.
pub(crate) fn probe(f: &Fn, top: &[TokenTree]) -> Option<TokenStream> {
    let n = f.params.len();
    let asked = asked(f, top);
    // The parameters are `__0`, `__1` and on. Rust says of a type that is not well formed wherever a
    // value of it is, so each is located at the type it is asked of, as Rust says it there.
    let local = |i: usize| -> TokenTree {
        let at = asked[i].map_or_else(Span::mixed_site, |(Ask::Pointer(_, at) | Ask::Number(_, at))| {
            at.resolved_at(Span::mixed_site())
        });
        Ident::new(&format!("__{i}"), at).into()
    };
    let mut values: Vec<TokenStream> = (0..n).map(|i| local(i).into()).collect();
    values.push(hygienic(&format!("$__{n}()"), true));
    let mut block = numbers(&asked);
    let calls = pointers(&asked, "*const T", &values);
    if !calls.is_empty() {
        let args: TokenStream = (0..n).flat_map(|i| [local(i), Punct::new(',', Spacing::Alone).into()]).collect();
        // `__n` calls the kernel on them, in a closure of its own, and first, for Rust to know the
        // types of the parameters when it is asked, and for a kernel that never returns to leave no
        // question unreachable.
        let mut call = parse(&top[f.head.end + 1].to_string());
        call.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, args.clone()))]);
        let mut body = hygienic(&format!("let $__{n} = || unsafe"), true);
        body.extend([TokenTree::Group(Group::new(Delimiter::Brace, call))]);
        body.extend(parse(";"));
        body.extend(calls);
        block.extend(parse("let _ = |"));
        block.extend(args);
        block.extend(parse("|"));
        block.extend([TokenTree::Group(Group::new(Delimiter::Brace, body))]);
        block.extend(parse(";"));
    }
    (!block.is_empty()).then(|| {
        block.extend(parse("0"));
        block
    })
}

/// What asks Rust of each number `asked` has, a parameter's or the result's, by [`number`]: in an item
/// of its own, `const _: () = { .. };`, and not among the statements of the body, for Rust numbers the
/// closures of a body in the order they come, and the names of the symbols of the code hold the
/// numbers: the closure a launcher runs, say, would be another's, and the code of a release build
/// would differ in its names. Nothing if there is no number to ask of.
fn numbers(asked: &[Option<Ask>]) -> TokenStream {
    let mut inner = TokenStream::new();
    for ask in asked {
        if let Some(Ask::Number(name, at)) = *ask {
            inner.extend(number(name, at));
        }
    }
    if inner.is_empty() {
        return inner;
    }
    let mut out = respan(parse("const _: () ="), Span::mixed_site());
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, inner))]);
    out.extend(respan(parse(";"), Span::mixed_site()));
    out
}

/// `let _ = |__kernite_asked: u64| { let _: ::core::primitive::u64 = __kernite_asked; };`, which
/// asks Rust whether the type the kernel writes, `u64` at `at`, is the number `name`, as a module's
/// `u64` is not when it gives the name another type. A closure that Rust checks and never calls takes
/// a value of the type as written; so Rust says of a type that is not the number that it is not, once,
/// at the type, and in the order a kernel that compiles says it: it expected the number and found the
/// type. It asks of the type and not of any value of it, which a type that has no size has none of.
/// Each token but the type is the macro's, located at `at`.
fn number(name: &str, at: Span) -> TokenStream {
    let mut out = placed("let _ = | __kernite_asked :", at);
    out.extend([TokenTree::Ident(Ident::new(name, at))]);
    out.extend(placed(&format!("| {{ let _: {} = __kernite_asked; }};", primitive(name)), at));
    out
}

/// What asks Rust of each pointer of one word `asked` has, among the parameters, by the values that
/// `values` gives of them: a call of a function that takes a pointer as `ptr`, which Rust cannot make
/// on a pointer to a type with no size: it says that the size cannot be known. Nothing if there is
/// nothing to ask.
fn pointers(asked: &[Option<Ask>], ptr: &str, values: &[TokenStream]) -> TokenStream {
    let mut calls = TokenStream::new();
    for (value, ask) in values.iter().zip(asked) {
        if let Some(Ask::Pointer(_, to)) = *ask {
            calls.extend(thin(value.clone(), to));
        }
    }
    if calls.is_empty() {
        return calls;
    }
    // The function and its calls are of one span, for Rust to find the one by the other.
    let mut out = respan(parse(&format!("#[inline(always)] fn {THIN}<T>(_: {ptr}) {{}}")), Span::mixed_site());
    out.extend(calls);
    out
}

/// The tokens `s` writes, each located at `at` and of the macro's hygiene.
fn placed(s: &str, at: Span) -> TokenStream {
    respan(parse(s), at.resolved_at(Span::mixed_site()))
}

/// `__kernite_thin(value);` with each token located at `to`, where the type `value` points to
/// begins, for Rust to say there that the type has no size. `value` is resolved as it is written;
/// the function is the macro's, which no name of the kernel's finds.
fn thin(value: TokenStream, to: Span) -> TokenStream {
    let value = value.into_iter().map(|mut t| {
        t.set_span(to.resolved_at(t.span()));
        t
    });
    let at = to.resolved_at(Span::mixed_site());
    let mut args = Group::new(Delimiter::Parenthesis, value.collect());
    args.set_span(at);
    let mut end = Punct::new(';', Spacing::Alone);
    end.set_span(at);
    [TokenTree::Ident(Ident::new(THIN, at)), args.into(), end.into()].into_iter().collect()
}

/// `call`, which gives back the result of an instance of `f`, as `f` gives it back. A pointer is
/// cast back at its type, for Rust to say there that the type has no size. Rust is asked of a
/// number by [`checks`].
pub(crate) fn result(f: &Fn, top: &[TokenTree], call: TokenStream) -> TokenStream {
    let mut out = call;
    if let Some(Ask::Pointer(_, to)) = asked(f, top).pop().flatten() {
        out.extend(parse("."));
        out.extend([TokenTree::Ident(Ident::new("cast", to.resolved_at(site())))]);
        out.extend(parse("()"));
    }
    out
}

/// `f` with a body that never runs, for a kernel that does not compile. It asks Rust of each type `f`
/// takes or gives back, as [`checks`] says: of a number by a closure of the type as written, which it
/// never calls, so that a result reads as it does in a kernel that compiles, "expected `f64`, found
/// `f32`", and not the other way round, as a literal of the number would read. It ends with `dangling`,
/// the value of a result that is a pointer, and with `unreachable!()` for any other.
pub(crate) fn stub(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out = attrs(f, top);
    out.extend(signature(f, top));
    let mut body = checks(f, top, "*const T");
    body.extend(parse(&never(f, top)));
    body.extend(match asked(f, top).pop().flatten() {
        Some(Ask::Pointer(kind, to)) => {
            let mutable = if kind == "mut" { "_mut" } else { "" };
            respan(parse(&format!("::core::ptr::dangling{mutable}()")), to.resolved_at(site()))
        }
        _ => parse("::core::unreachable!()"),
    });
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, body))]);
    out
}

/// What a body of `f`'s parameters does when it never runs, for a kernel that does not compile: it
/// uses each parameter, by its name, which Rust would find unused.
pub(crate) fn never(f: &Fn, top: &[TokenTree]) -> String {
    written_params(f, top).iter().map(|(param, colon)| format!("let _ = {};", param[colon - 1])).collect()
}

/// `f`'s attributes as written, but `#[instances(..)]`, `#[cooperative]` and its directives,
/// `#[maxntid(..)]` and the like, which Rust does not know.
pub(crate) fn attrs(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let instances = f.instances.as_ref().map_or(0..0, |i| i.tokens.clone());
    let cooperative = f.cooperative.as_ref().map_or(0..0, |c| c.tokens.clone());
    let kernite = |i: &usize| {
        [&instances, &cooperative].into_iter().chain(f.directives.iter().map(|d| &d.tokens)).any(|r| r.contains(i))
    };
    f.attrs.clone().filter(|i| !kernite(i)).map(|i| top[i].clone()).collect()
}

/// What the code that stands for an item, an error said of it or a method it gives, keeps of the
/// item's attributes, `attrs`, `#` and brackets each: `cfg`, which says whether the item is there,
/// alone or in a `cfg_attr(..)`; and, if `offered`, `deprecated` and `doc(hidden)`, which say how it is
/// offered. Nothing else: a lint or a symbol is the item's own.
pub(crate) fn carried(attrs: impl IntoIterator<Item = TokenTree>, offered: bool) -> TokenStream {
    let attrs: Vec<TokenTree> = attrs.into_iter().collect();
    let mut out = TokenStream::new();
    for attr in attrs.chunks(2) {
        let [hash, TokenTree::Group(group)] = attr else { unreachable!("an attribute is `#` and brackets") };
        let Some(meta) = keep(flatten(group.stream()), offered) else { continue };
        let mut kept = Group::new(Delimiter::Bracket, meta.into_iter().collect());
        kept.set_span(group.span());
        out.extend([hash.clone(), kept.into()]);
    }
    out
}

/// What [`carried`] keeps of an attribute, `meta` the tokens in its brackets; `None` if nothing. Of a
/// `cfg_attr(..)`, it keeps the condition and what it keeps of the list, or nothing if that is
/// nothing: a `cfg_attr(..)` of a condition alone is an error.
fn keep(meta: Vec<TokenTree>, offered: bool) -> Option<Vec<TokenTree>> {
    match &meta[..] {
        [doc, TokenTree::Group(list)] if offered && named(doc, "doc") => flatten(list.stream())
            .iter()
            .any(|t| named(t, "hidden"))
            .then(|| parse("doc(hidden)").into_iter().collect()),
        [cfg_attr, TokenTree::Group(list)] if named(cfg_attr, "cfg_attr") => {
            let tokens = flatten(list.stream());
            let mut parts = tokens.split(|t| is_punct(t, ','));
            let mut kept = parts.next().expect("a split has a first part").to_vec();
            let condition = kept.len();
            for part in parts.filter_map(|part| keep(part.to_vec(), offered)) {
                kept.extend(parse(","));
                kept.extend(part);
            }
            if kept.len() == condition {
                return None;
            }
            let mut group = Group::new(Delimiter::Parenthesis, kept.into_iter().collect());
            group.set_span(list.span());
            Some(vec![cfg_attr.clone(), group.into()])
        }
        [name, ..] if named(name, "cfg") || (offered && named(name, "deprecated")) => Some(meta),
        _ => None,
    }
}

/// Whether `t` is `name`, or `r#name`: Rust reads an attribute's name written raw as the name.
fn named(t: &TokenTree, name: &str) -> bool {
    is_ident(t, name) || is_ident(t, &format!("r#{name}"))
}

/// The condition under which an item is there, from `cfg`, what [`carried`] keeps of its attributes
/// when it offers nothing: [`all`] its `cfg`s say, where a `cfg_attr(p, cfg(q))` is `any(not(p), q)`,
/// and one in another likewise. `None` for an item with none: it is always there.
pub(crate) fn condition(cfg: &TokenStream) -> Option<TokenStream> {
    let attrs: Vec<TokenTree> = cfg.clone().into_iter().collect();
    all(attrs.chunks(2).map(|attr| {
        let [_, TokenTree::Group(meta)] = attr else { unreachable!("an attribute is `#` and brackets") };
        predicate(&flatten(meta.stream()))
    }))
}

/// The condition that all of `parts` hold, each as [`condition`] has it: `None` if all are always
/// there.
pub(crate) fn all(parts: impl IntoIterator<Item = Option<TokenStream>>) -> Option<TokenStream> {
    let parts: Vec<TokenStream> = parts.into_iter().flatten().collect();
    (!parts.is_empty()).then(|| join("all", parts))
}

/// The condition that any of `parts` holds, each as [`condition`] has it: `None` if one is always.
pub(crate) fn any(parts: impl IntoIterator<Item = Option<TokenStream>>) -> Option<TokenStream> {
    Some(join("any", parts.into_iter().collect::<Option<Vec<_>>>()?))
}

/// The condition that `part` does not hold.
pub(crate) fn not(part: TokenStream) -> TokenStream {
    call("not", vec![part])
}

/// `#[cfg(condition)]`: what is written under it is there where the condition holds. Nothing for `None`,
/// which always does.
pub(crate) fn when(condition: Option<TokenStream>) -> TokenStream {
    let Some(condition) = condition else { return TokenStream::new() };
    let mut out = parse("#");
    out.extend([TokenTree::Group(Group::new(Delimiter::Bracket, call("cfg", vec![condition])))]);
    out
}

/// The predicate that one attribute [`carried`] keeps says, `meta` the tokens in its brackets:
/// `cfg(p)` says `p`, and `cfg_attr(p, ..)` that what it keeps of the list holds, or `p` does not.
/// `None` for an attribute that says nothing, as a `cfg` Rust cannot read: it says what is wrong with
/// it where it is written, and takes the item to be there.
fn predicate(meta: &[TokenTree]) -> Option<TokenStream> {
    let [name, TokenTree::Group(list)] = meta else { return None };
    let tokens = flatten(list.stream());
    if named(name, "cfg") {
        return Some(tokens.into_iter().collect());
    }
    let mut parts = tokens.split(|t| is_punct(t, ','));
    let when: TokenStream = parts.next().expect("a split has a first part").iter().cloned().collect();
    let then: Vec<TokenStream> = parts.filter_map(predicate).collect();
    (!then.is_empty()).then(|| call("any", vec![not(when), join("all", then)]))
}

/// `name(parts)` for a predicate's `all` or `any`, or the part alone where there is one.
fn join(name: &str, mut parts: Vec<TokenStream>) -> TokenStream {
    match parts.len() {
        1 => parts.remove(0),
        _ => call(name, parts),
    }
}

/// `name(a, b, ..)`, the `parts` apart by commas.
fn call(name: &str, parts: Vec<TokenStream>) -> TokenStream {
    let mut args = TokenStream::new();
    for (i, part) in parts.into_iter().enumerate() {
        if i > 0 {
            args.extend(parse(","));
        }
        args.extend(part);
    }
    let mut out = parse(name);
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, args))]);
    out
}

/// `f`'s signature as written, from its visibility to its result, but for its parameters, as
/// [`value_params`] has them, and a number it gives back, which is [`value`]'s.
pub(crate) fn signature(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out: TokenStream = top[f.head.start..f.params_at].iter().cloned().collect();
    let mut params = Group::new(Delimiter::Parenthesis, value_params(f, top));
    params.set_span(top[f.params_at].span());
    out.extend([TokenTree::Group(params)]);
    if let Some(r) = &f.ret {
        // The arrow, then the result.
        out.extend(top[f.params_at + 1..r.tokens.start].iter().cloned());
        out.extend(value(&top[r.tokens.clone()]));
    }
    out
}

/// `f`'s parameters as [`written_params`] has them, a comma after each, but for each number, which
/// is [`value`]'s.
pub(crate) fn value_params(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out = TokenStream::new();
    for (param, colon) in written_params(f, top) {
        out.extend(param[..=colon].iter().cloned());
        out.extend(value(&param[colon + 1..]));
        out.extend(parse(","));
    }
    out
}

/// The tokens of `stream`, with the invisible groups a macro's fragments arrive in opened.
pub(crate) fn flatten(stream: TokenStream) -> Vec<TokenTree> {
    let mut out = vec![];
    for t in stream {
        match t {
            TokenTree::Group(g) if g.delimiter() == Delimiter::None => out.extend(flatten(g.stream())),
            t => out.push(t),
        }
    }
    out
}

/// `tokens` as the language reads them, each span numbered by its place in the spans beside
/// them. The spans come back with an error too, to place it: a bracket that nests too deep.
pub(crate) fn lex(tokens: &[TokenTree]) -> (Result<Vec<Tree>, lang::Error>, Vec<Span>) {
    let mut spans = vec![];
    let trees = tokens.iter().map(|t| tree(t, 0, &mut spans)).collect();
    (trees, spans)
}

/// `t` as the language reads it, `depth` brackets down, each span numbered by its place in
/// `spans`.
fn tree(t: &TokenTree, depth: usize, spans: &mut Vec<Span>) -> Result<Tree, lang::Error> {
    let span = lang::Span(spans.len() as u32);
    Ok(match t {
        TokenTree::Ident(i) => {
            spans.push(i.span());
            Tree::Ident { name: i.to_string(), span }
        }
        TokenTree::Punct(p) => {
            spans.push(p.span());
            Tree::Punct { ch: p.as_char(), joint: p.spacing() == Spacing::Joint, span }
        }
        TokenTree::Literal(l) => {
            spans.push(l.span());
            Tree::Lit { text: l.to_string(), span }
        }
        TokenTree::Group(g) => {
            spans.push(g.span_open());
            lang::nested(depth, span)?;
            let delim = match g.delimiter() {
                Delimiter::Parenthesis => Delim::Paren,
                Delimiter::Bracket => Delim::Bracket,
                Delimiter::Brace | Delimiter::None => Delim::Brace,
            };
            let trees = flatten(g.stream()).iter().map(|t| tree(t, depth + 1, spans)).collect::<Result<_, _>>()?;
            Tree::Group { delim, trees, span }
        }
    })
}

/// `::core::compile_error! { message }`, reported where `span` is.
pub(crate) fn error(span: Span, message: &str) -> TokenStream {
    let mut message = Literal::string(message);
    message.set_span(span);
    let tokens = [
        TokenTree::Punct(Punct::new(':', Spacing::Joint)),
        TokenTree::Punct(Punct::new(':', Spacing::Alone)),
        TokenTree::Ident(Ident::new("core", span)),
        TokenTree::Punct(Punct::new(':', Spacing::Joint)),
        TokenTree::Punct(Punct::new(':', Spacing::Alone)),
        TokenTree::Ident(Ident::new("compile_error", span)),
        TokenTree::Punct(Punct::new('!', Spacing::Alone)),
        TokenTree::Group(Group::new(Delimiter::Brace, TokenTree::Literal(message).into())),
    ];
    tokens
        .into_iter()
        .map(|mut t| {
            t.set_span(span);
            t
        })
        .collect()
}

/// [`error`] under `cfg`, what [`carried`] keeps of an item's attributes: Rust says nothing of an item
/// it leaves out, and so does the error of one that does not compile.
pub(crate) fn error_under(cfg: &TokenStream, span: Span, message: &str) -> TokenStream {
    let mut out = cfg.clone();
    out.extend(error(span, message));
    out
}

/// Whether the crate being compiled has a dependency named `name`, `kernite-cpu` or `kernite-cuda`,
/// which is how the Rust the macro writes names it. One renamed in `Cargo.toml` has another name.
pub(crate) fn depends_on(name: &str) -> bool {
    given(&std::env::args_os().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>(), name)
}

/// Whether `args`, the arguments Rust runs with, give the crate `name` by an `--extern`, as
/// `--extern kernite_cpu=..` gives `kernite-cpu`. When they give no crate at all, as when an editor
/// runs the macro in a process of its own, nothing says it is missing.
fn given(args: &[String], name: &str) -> bool {
    let name = name.replace('-', "_");
    // What each `--extern` gives, `[options:]name[=path]`, written `--extern x` or `--extern=x`.
    let externs: Vec<&str> = args
        .iter()
        .enumerate()
        .filter_map(|(i, a)| match a.strip_prefix("--extern")? {
            "" => args.get(i + 1).map(String::as_str),
            value => value.strip_prefix('='),
        })
        .collect();
    externs.is_empty() || externs.iter().any(|e| extern_name(e) == name)
}

/// The name of the crate that `value`, an `--extern`'s `[options:]name[=path]`, gives.
fn extern_name(value: &str) -> &str {
    let spec = value.split_once('=').map_or(value, |(spec, _path)| spec);
    spec.rsplit_once(':').map_or(spec, |(_options, name)| name)
}

/// A type that stands for one of the crate `name`'s in a crate that has no dependency named so, and
/// says at `span` that `what` needs one. It is an array whose length is the error, which Rust
/// reports once and takes for a type it has reported, so that it says nothing of the code that
/// uses it.
pub(crate) fn missing(span: Span, what: &str, name: &str) -> TokenStream {
    let mut length = parse("();");
    length.extend(error(span, &format!("{what} needs a dependency named `{name}`, which this crate does not have")));
    TokenTree::Group(Group::new(Delimiter::Bracket, length)).into()
}

/// The type `ty`, a kernel's, as the host names it: `f16` and `bf16`, which Rust has not, are
/// kernite-cuda's. Any other is as the kernel writes it, and means what it means in its module.
pub(crate) fn host(ty: &[TokenTree]) -> TokenStream {
    match ty {
        [t] if is_ident(t, "f16") || is_ident(t, "bf16") => parse(&primitive(&t.to_string())),
        _ => ty.iter().cloned().collect(),
    }
}

/// The number `ty` is, if it is one by its name, as the Rust the macros write for a kernel names
/// it: the primitive by its path, `::core::primitive::u64`, located at the name. A module may give
/// the name another type, and Rust says so, once, at the type, where [`checks`] asks it. What the
/// macros write besides, an instance, a dispatcher or a launcher, takes the number and not the
/// name, so it says nothing of the type: a value of one with no size, or that a launcher cannot
/// pass on, would be an error at the module's attribute, where nothing can be mended.
pub(crate) fn number_type(ty: &[TokenTree]) -> Option<TokenStream> {
    number_name(ty).map(|(name, at)| placed(&primitive(name), at))
}

/// `ty`, a parameter's or a result's, as the Rust the macros write has it: [`number_type`] if it is
/// a number's name, `self::f16` if it is `f16`, else as written. It is for the signatures
/// [`signature`] writes and an x86 kernel's instances and launcher. A GPU launcher's numbers are
/// [`number_type`]'s, or [`host`]'s.
///
/// An x86 kernel takes no `f16`: kernite says so at the type, and leaves a stub. Rust would say too
/// that its `f16` is unstable, as it does wherever the type is named, so the stub names `self::f16`,
/// which Rust cannot find. Where a macro has failed in a module, Rust says nothing of a name it cannot
/// find, as it says nothing of a `bf16`; and it says nothing of a call of the stub, whose type it
/// takes for an error.
pub(crate) fn value(ty: &[TokenTree]) -> TokenStream {
    match ty {
        [t] if is_ident(t, "f16") => placed("self::f16", t.span()),
        _ => number_type(ty).unwrap_or_else(|| ty.iter().cloned().collect()),
    }
}

/// The number `name`, by the language's name, as the Rust the macros write names it. A module may
/// give a name such as `u32` another type, so it is the primitive by its path,
/// `::core::primitive::u32`. `f16` and `bf16`, which Rust has not, are kernite-cuda's.
pub(crate) fn primitive(name: &str) -> String {
    match name {
        "f16" | "bf16" => format!("::kernite_cuda::{name}"),
        _ => format!("::core::primitive::{name}"),
    }
}

/// `tokens` where they were written, but of the macro's span: Rust lints what a user wrote, and
/// these only repeat it. An error or a reader finds them where they were written.
pub(crate) fn quiet(tokens: &[TokenTree]) -> impl Iterator<Item = TokenTree> + '_ {
    tokens.iter().map(|t| {
        let mut t = t.clone();
        t.set_span(t.span().resolved_at(site()));
        t
    })
}

/// The tokens `s` writes, each of the span [`SITE`] says.
pub(crate) fn parse(s: &str) -> TokenStream {
    let tokens: TokenStream = s.parse().expect("kernite-macros writes tokens that parse");
    match SITE.get() {
        Some(span) => respan(tokens, span),
        None => tokens,
    }
}

/// The tokens `s` writes, as [`parse`] gives them, but for each name after a `$`: a local the
/// macro binds for itself, of the span `Span::mixed_site()`. No name the user wrote finds it, nor
/// does it find theirs, so a parameter of a kernel may be named as the macro's launcher names its
/// own. Where the Rust never `runs`, each is `_`, for a compile-time argument may be named so, and
/// Rust reads a binding of a const's name as that const.
pub(crate) fn hygienic(s: &str, runs: bool) -> TokenStream {
    fn own(tokens: TokenStream, runs: bool) -> TokenStream {
        let mut tokens = tokens.into_iter();
        let mut out = TokenStream::new();
        while let Some(t) = tokens.next() {
            out.extend([match t {
                TokenTree::Punct(p) if p.as_char() == '$' => {
                    let name = tokens.next().expect("a `$` has a name after it");
                    let name = if runs { name.to_string() } else { "_".to_string() };
                    TokenTree::Ident(Ident::new(&name, Span::mixed_site()))
                }
                TokenTree::Group(g) => {
                    let mut inner = Group::new(g.delimiter(), own(g.stream(), runs));
                    inner.set_span(g.span());
                    TokenTree::Group(inner)
                }
                t => t,
            }]);
        }
        out
    }
    own(parse(s), runs)
}

/// The span of the tokens the macro writes: [`SITE`]'s, or its call site.
pub(crate) fn site() -> Span {
    SITE.get().unwrap_or_else(Span::call_site)
}

/// `tokens`, each of the span `span`.
fn respan(tokens: TokenStream, span: Span) -> TokenStream {
    tokens
        .into_iter()
        .map(|t| match t {
            TokenTree::Group(g) => {
                let mut inner = Group::new(g.delimiter(), respan(g.stream(), span));
                inner.set_span(span);
                TokenTree::Group(inner)
            }
            mut t => {
                t.set_span(span);
                t
            }
        })
        .collect()
}

pub(crate) fn is_ident(t: &TokenTree, name: &str) -> bool {
    matches!(t, TokenTree::Ident(i) if i.to_string() == name)
}

pub(crate) fn is_punct(t: &TokenTree, ch: char) -> bool {
    matches!(t, TokenTree::Punct(p) if p.as_char() == ch)
}

#[cfg(test)]
mod tests {
    use super::given;

    /// Whether `given` says the crate `name` is there, of the arguments of a build that has
    /// `kernite_macros` and no other crate, and `args` besides.
    fn run(args: &str, name: &str) -> bool {
        let build = format!("rustc --extern kernite_macros=/t/libkernite_macros.so {args}");
        given(&build.split_whitespace().map(String::from).collect::<Vec<_>>(), name)
    }

    #[test]
    fn nothing_but_an_extern_gives_a_crate() {
        // A crate's own name, a directory, a path to search and a crate of a longer name hold the
        // crate's, and so does a flag of rustdoc's: none of them is an `--extern` of it.
        for (args, name) in [
            ("--crate-name my_kernite_cuda_app", "kernite-cuda"),
            ("--out-dir /home/u/kernite_cpu_demo/target/debug/deps", "kernite-cpu"),
            ("-L dependency=/home/u/kernite_cuda/target/debug/deps", "kernite-cuda"),
            ("--extern kernite_cuda_demo=/t/libkernite_cuda_demo.rlib", "kernite-cuda"),
            ("--extern-html-root-url kernite_cuda=https://docs.rs/kernite-cuda", "kernite-cuda"),
        ] {
            assert!(!run(args, name), "{args}");
        }
    }

    #[test]
    fn a_crate_renamed_in_cargo_toml_is_given_by_its_new_name() {
        // `cuda = { package = "kernite-cuda", .. }`
        assert!(!run("--extern cuda=/t/libkernite_cuda-1a2b.rlib", "kernite-cuda"));
    }

    #[test]
    fn each_spelling_of_extern_gives_its_crate() {
        for spelling in [
            "--extern kernite_cuda=/t/a.rlib",
            "--extern=kernite_cuda=/t/a.rlib",
            "--extern priv:kernite_cuda=/t/a.rlib",
            "--extern=noprelude:kernite_cuda=C:/t/a.rlib",
            "--extern kernite_cuda",
            "--extern=kernite_cuda",
        ] {
            assert!(run(spelling, "kernite-cuda"), "{spelling}");
        }
        // Each crate is found by its own name: the other's is not it.
        assert!(run("--extern kernite_cpu=/t/a.rlib", "kernite-cpu"));
        assert!(!run("--extern kernite_cpu=/t/a.rlib", "kernite-cuda"));
    }

    #[test]
    fn a_crate_is_there_when_rust_gives_no_crate_at_all() {
        // An editor runs the macro in a process of its own, which Rust's arguments do not reach.
        assert!(given(&["rust-analyzer-proc-macro-srv".to_string()], "kernite-cuda"));
    }
}
