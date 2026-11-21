//! Rust's tokens: the language's trees made of them, the parts of a module or a function as
//! written, and tokens the macro makes, each at the span that says where Rust resolves it.

use std::cell::Cell;
use std::mem::DropGuard;
use std::ops::Range;

use proc_macro::{Delimiter, Group, Ident, Literal, Spacing, Span, TokenStream, TokenTree};

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

/// `SITE` at `span` until the guard this gives is dropped, the macro panicking or not: a span of
/// one call of the macro is none in the next, and the thread that runs it may run the next.
pub(crate) fn site_at(span: Span) -> DropGuard<(), impl FnOnce(())> {
    SITE.set(Some(span));
    DropGuard::new((), |()| SITE.set(None))
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

/// `path`, the path of a module of consts as `#[kernels(consts = path)]` is written where the module of kernels is,
/// written from `depth` modules inside it: from the module itself, 1.
pub(crate) fn from_inside(path: &[TokenTree], depth: usize) -> TokenStream {
    let up = parse(&vec!["super"; depth].join("::"));
    match path {
        [first, ..] if is_ident(first, "crate") || is_punct(first, ':') => path.iter().cloned().collect(),
        [first, rest @ ..] if is_ident(first, "self") => up.into_iter().chain(rest.iter().cloned()).collect(),
        _ => up.into_iter().chain(parse("::")).chain(path.iter().cloned()).collect(),
    }
}

/// What the module `__consts`, which `#[kernels(consts = path)]` writes in its module, holds to find the macro of the
/// module of consts at `path`. A `use` for each name on the path takes the module and a glob of it. A last `use` takes
/// the macro and a glob of `__kernite`, the module `#[consts]` writes. If Rust cannot find a name, the `use` that ends
/// at it says so, once, at the name. That `use` is a glob that fails at its last name, so Rust says nothing of a name
/// it then cannot find: the macro, or a kernel that code calls. A path that fails at a name before its last would leave
/// those errors. Each `use` starts from the module the one before it found, by the name it gave it, so a `use` after
/// one that failed says nothing. That name is `__`, the module's own name, `_` and the place of its token on the path.
/// No two on a path are the same, nor is one the name of an item that a glob brings in, for Rust would not know which
/// the name is. Rust's error says the name, and so it tells which module. A module with no `#[consts]` on it fails the
/// last `use`, at its glob of `__kernite`.
pub(crate) fn path_uses(path: &[TokenTree]) -> TokenStream {
    let (mut out, mut run) = (TokenStream::new(), TokenStream::new());
    for (i, token) in from_inside(path, 2).into_iter().enumerate() {
        let name = match &token {
            // `crate` and `super` are modules that are there, and not names: a `use` ends at the next name.
            TokenTree::Ident(id) if !matches!(id.to_string().as_str(), "crate" | "super") => Some(id.to_string()),
            _ => None,
        };
        run.extend([token]);
        if let Some(name) = name {
            // The `r#` of a raw name is not a part of the name made of it.
            let module = format!("__{}_{i}", name.trim_start_matches("r#"));
            out.extend(parse("use"));
            out.extend(run);
            out.extend(parse(&format!("::{{self as {module}, *}};")));
            run = parse(&module);
        }
    }
    out.extend(parse("pub(super) use"));
    out.extend(run);
    out.extend(parse("::{__kernite_consts, __kernite::*};"));
    out
}

/// Whether `tokens`, an item's, are an attribute of the module's own, `#![..]`.
pub(crate) fn inner_attribute(tokens: &[TokenTree]) -> bool {
    matches!(tokens, [hash, bang, ..] if is_punct(hash, '#') && is_punct(bang, '!'))
}

/// `tokens`, an item's, as its attributes, `#[..]` each, and what follows them.
pub(crate) fn attrs_of(tokens: &[TokenTree]) -> (&[TokenTree], &[TokenTree]) {
    let mut rest = tokens;
    while let [hash, TokenTree::Group(g), after @ ..] = rest
        && is_punct(hash, '#')
        && g.delimiter() == Delimiter::Bracket
    {
        rest = after;
    }
    (tokens.get(..tokens.len() - rest.len()).unwrap_or_default(), rest)
}

/// `f`'s visibility as written, `pub` or `pub(..)`, or none.
pub(crate) fn visibility<'a>(f: &Fn, top: &'a [TokenTree]) -> &'a [TokenTree] {
    let head = within(top, &f.head);
    let vis = match head {
        [pub_, TokenTree::Group(g), ..] if is_ident(pub_, "pub") && g.delimiter() == Delimiter::Parenthesis => 2,
        [pub_, ..] if is_ident(pub_, "pub") => 1,
        _ => 0,
    };
    head.get(..vis).unwrap_or_default()
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

/// The tokens of `top` that `range` covers, a range of tokens that the parser gave; none for one it did not.
pub(crate) fn within<'a>(top: &'a [TokenTree], range: &Range<usize>) -> &'a [TokenTree] {
    top.get(range.clone()).unwrap_or_default()
}

/// The name of a function or a static as written, `r#type` as well: the token after the keyword, `fn` or `static`,
/// that comes after `before`, its head or its visibility.
pub(crate) fn name_after<'a>(before: &Range<usize>, top: &'a [TokenTree]) -> Option<&'a TokenTree> {
    top.get(before.end + 1)
}

/// Where `token` is; the macro's call site for none.
pub(crate) fn span_of(token: Option<&TokenTree>) -> Span {
    token.map_or_else(Span::call_site, TokenTree::span)
}

/// `f`'s generics as written, `<const N: usize>`; none if it has none.
pub(crate) fn written_generics<'a>(f: &Fn, top: &'a [TokenTree]) -> &'a [TokenTree] {
    top.get(f.head.end + 2..f.params_at).unwrap_or_default()
}

/// The name after each `const` of `tokens`, as written, `r#type` as well: a const's, or those of
/// a function's compile-time arguments.
pub(crate) fn const_names(tokens: &[TokenTree]) -> impl Iterator<Item = &TokenTree> {
    tokens.windows(2).filter_map(|w| match w {
        [keyword, name] if is_ident(keyword, "const") => Some(name),
        _ => None,
    })
}

/// The place of the item of `module` named `name`, among its items; none if it has none.
pub(crate) fn position(module: &lang::Module, name: &str) -> Option<usize> {
    module.items.iter().position(|item| item.name() == Some(name))
}

/// What [`carried`] keeps of `item`'s attributes when it offers nothing: its `cfg`.
pub(crate) fn cfg_of(item: &lang::Item, top: &[TokenTree]) -> TokenStream {
    carried(attrs_of(within(top, &item.tokens)).0.iter().cloned(), false)
}

/// The function that stands in for `item`, a helper, an `extern "ptx-func"` function or a GPU kernel, whose code Rust
/// never sees, where a kernel names it: one of its name that has its `deprecated`,
/// `#[deprecated(note = "..")] fn old() {}`, for Rust to tell of a use of `old` as it tells of a use of any deprecated
/// function. A block that names the item has it once for all its uses, as Rust reads the `deprecated` of each function
/// that has it. None where `item` is not deprecated.
pub(crate) fn stand_in(item: &lang::Item, top: &[TokenTree]) -> Option<TokenStream> {
    let mut stand_in = deprecated(attrs_of(within(top, &item.tokens)).0.iter().cloned());
    if stand_in.is_empty() {
        return None;
    }
    stand_in.extend(parse("fn"));
    stand_in.extend([name_at(item, top, site())?]);
    stand_in.extend(parse("() {}"));
    Some(stand_in)
}

/// The name of `item`, a const, a static or a function, as `top` writes it, `r#type` as well, of the
/// span `span`; none for an item that has no name.
pub(crate) fn name_at(item: &lang::Item, top: &[TokenTree], span: Span) -> Option<TokenTree> {
    let mut name = match &item.kind {
        ItemKind::Const(_) => const_names(within(top, &item.tokens)).next()?,
        ItemKind::Static(s) => name_after(&s.vis, top)?,
        ItemKind::Fn(f) => name_after(&f.head, top)?,
        ItemKind::Other => return None,
    }
    .clone();
    name.set_span(span);
    Some(name)
}

/// An error at each of `f`'s compile-time arguments named as one of `own`, the locals the Rust of
/// `method` binds for itself, under `cfg`, `f`'s own. A compile-time argument is no local, and no span of
/// stable Rust keeps it apart from one of its name, as it does a parameter's.
pub(crate) fn clashes(f: &Fn, cfg: &TokenStream, own: &[&str], method: &str, top: &[TokenTree]) -> TokenStream {
    let mut errors = TokenStream::new();
    for name in const_names(written_generics(f, top)) {
        let written = name.to_string();
        let name_of = lang::unraw(&written);
        if own.contains(&name_of) {
            errors.extend(error_under(cfg, name.span(), &format!("`{name_of}` names a parameter of {method} as well")));
        }
    }
    errors
}

/// `tokens` split at each token `ends` says ends a part, that is not among the arguments of a type's, as the comma of
/// `Pair<u8, 2>` is: a comma between parameters, or a plus between a trait object's bounds, `dyn Debug + Send`. A `>`
/// that closes no `<` of the tokens' own closes the arguments of a type that holds them, as the `>` after `u8` in
/// `Option<u8>` does: it is asked of `ends`, as any token is. The `>` of an arrow, `fn() -> u8`, closes no `<`, and
/// ends nothing.
fn split_top(tokens: &[TokenTree], ends: impl std::ops::Fn(&TokenTree) -> bool) -> impl Iterator<Item = &[TokenTree]> {
    let (mut open, mut dash) = (0usize, false);
    tokens.split(move |t| {
        // A `>` after a `-` joint to it is an arrow's, `->`.
        let arrow = dash && is_punct(t, '>');
        let end = open == 0 && !arrow && ends(t);
        if is_punct(t, '<') {
            open += 1;
        } else if is_punct(t, '>') && !arrow {
            open = open.saturating_sub(1);
        }
        dash = matches!(t, TokenTree::Punct(p) if p.as_char() == '-' && p.spacing() == Spacing::Joint);
        end
    })
}

/// `f`'s signature in `top`, written so that Rust can be given every name in it: a name Rust cannot be given is named
/// `__kernite_0`, or the next of `__kernite_0`, `__kernite_1` and on that no parameter or compile-time argument of `f`
/// is named. Rust cannot pass on a parameter named `_`. And it refuses a name written raw that it lets be written in no
/// place, `r#self`, `r#Self`, `r#super`, `r#crate` or `r#_`, with an error at each place it is written: written again
/// by the macros as text, in a call or a signature, it is refused again at the module's attribute, where Rust lexes
/// the text; and a GPU kernel's launcher binds `self` and `Self` for itself, which a parameter or a compile-time
/// argument so named would take for its own. So the macros never write such a name, and Rust has the places the user
/// wrote it, for its own error. A compile-time argument is named so wherever the signature names it, in a type too,
/// `[u8; r#crate]`. All the macros write for `f`, a signature or a body, has the names as they are here.
///
/// A `mut` before a parameter's name is taken and means nothing, in a function of every kind: an instruction may write
/// any variable, which Rust cannot see. So the macros write no `mut` for Rust, which would say that the variable does
/// not need to be mutable, and in a naked function that patterns are not allowed there.
///
/// A parameter named `self` or `Self`, written plain, is named so as well: what the macros write for a kernel binds
/// both in ways of its own, a launcher being a method, and Rust says other words of it, or more; of a helper's or a GPU
/// function's, for which they write no function, Rust says nothing. Rust's error is that of a function that is no
/// method, at the name, once. So the macros write a function for Rust to say it in, which never runs, in an item of no
/// name: `fn f(_: (), self: ()) {}`, with the name in its place among the parameters of `f`, `_` in the place of each
/// other, and each of the type `()`, so that Rust says nothing of a type as written, `gpr` or `f128`, nor that a
/// parameter is unused. It is given back, if `f` has such a parameter.
pub(crate) fn passable(top: &mut [TokenTree], f: &Fn) -> Option<TokenStream> {
    let taken: Vec<&str> =
        f.generics.iter().map(|g| &g.name).chain(f.params.iter().map(|p| &p.name)).map(String::as_str).collect();
    let mut free = (0..).map(|n| format!("__kernite_{n}")).filter(|name| !taken.contains(&name.as_str()));
    let mut fresh = || free.next().map(|name| Ident::new(&name, site()));
    let consts: Vec<(String, Ident)> = const_names(written_generics(f, top))
        .map(TokenTree::to_string)
        .filter(|name| lang::never_raw(name))
        .filter_map(|name| Some((name, fresh()?)))
        .collect();
    let own = |name: &TokenTree| is_ident(name, "self") || is_ident(name, "Self");
    let written = written_params(f, top);
    let free_function = written.iter().any(|p| own(&p.name)).then(|| {
        let params = written.iter().flat_map(|p| {
            let name = if own(&p.name) { p.name.clone().into() } else { parse("_") };
            name.into_iter().chain(parse(": (),"))
        });
        let mut inner = parse("fn f");
        inner.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, params.collect()))]);
        inner.extend(parse("{}"));
        unnamed(inner)
    });
    let mut params = TokenStream::new();
    for Written { head, name, ty } in written {
        let name = match own(&name) || is_ident(&name, "_") || lang::never_raw(&name.to_string()) {
            true => fresh().map_or(name, TokenTree::from),
            false => name,
        };
        // The name, then the colon as written, and the type.
        params.extend([name]);
        params.extend(head.last().cloned());
        params.extend(ty);
        params.extend(parse(","));
    }
    if let Some(at) = top.get_mut(f.params_at) {
        let mut group = Group::new(Delimiter::Parenthesis, params);
        group.set_span(at.span());
        *at = group.into();
    }
    let end = f.ret.as_ref().map_or(f.params_at + 1, |r| r.tokens.end);
    if !consts.is_empty()
        && let Some(signature) = top.get_mut(f.head.end + 2..end)
    {
        rename(signature, &consts);
    }
    free_function
}

/// `tokens`, with the second of a pair of `names` for each ident written as the first, in brackets too.
fn rename(tokens: &mut [TokenTree], names: &[(String, Ident)]) {
    for token in tokens {
        match token {
            TokenTree::Ident(ident) => {
                if let Some((_, to)) = names.iter().find(|(from, _)| *from == ident.to_string()) {
                    *token = to.clone().into();
                }
            }
            TokenTree::Group(group) => {
                let mut inside: Vec<TokenTree> = group.stream().into_iter().collect();
                rename(&mut inside, names);
                let mut renamed = Group::new(group.delimiter(), inside.into_iter().collect());
                renamed.set_span(group.span());
                *token = renamed.into();
            }
            TokenTree::Punct(_) | TokenTree::Literal(_) => {}
        }
    }
}

/// A parameter as written, `mut x: *mut f32`.
pub(crate) struct Written {
    /// Its tokens up to its colon, the colon too: `mut x:`.
    pub(crate) head: Vec<TokenTree>,
    /// Its name, `x`: the token before the colon.
    pub(crate) name: TokenTree,
    /// Its type, `*mut f32`: the tokens after the colon.
    pub(crate) ty: Vec<TokenTree>,
}

/// `f`'s parameters as written, those that have a name and a colon. A fragment that stays one type is in parentheses,
/// as [`for_rust`] has it, for the macros write each parameter as it is here, and Rust reads `*const $t`, with `$t` as
/// `dyn Debug + Send`, as one type only so. A comma ends a parameter unless [`split_top`] says it is among the
/// arguments of a type's. Each name is as [`passable`] has it.
pub(crate) fn written_params(f: &Fn, top: &[TokenTree]) -> Vec<Written> {
    let Some(TokenTree::Group(group)) = top.get(f.params_at) else { return vec![] };
    let tokens = contents(group);
    split_top(&tokens, |t| is_punct(t, ','))
        .filter_map(|p| {
            let (head, ty) = p.split_at_checked(p.iter().position(|t| is_punct(t, ':'))? + 1)?;
            let name = head.iter().nth_back(1)?.clone();
            Some(Written { head: head.to_vec(), name, ty: ty.to_vec() })
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

/// `"const"` or `"mut"`, the tokens of the type of the elements, and whether they are a slice's, if `ty` is a
/// reference to an array, `&[T; N]` or `&mut [T; N]`, or to a slice, `&[T]` or `&mut [T]`: memory a kernel is given.
/// The macros read one here and nowhere else.
pub(crate) fn array_ref(ty: &[TokenTree]) -> Option<(&'static str, Vec<TokenTree>, bool)> {
    let (kind, group) = match ty {
        [amp, mut_, TokenTree::Group(g)] if is_punct(amp, '&') && is_ident(mut_, "mut") => ("mut", g),
        [amp, TokenTree::Group(g)] if is_punct(amp, '&') => ("const", g),
        _ => return None,
    };
    if group.delimiter() != Delimiter::Bracket {
        return None;
    }
    let inside = flatten(group.stream());
    Some(match inside.iter().position(|t| is_punct(t, ';')) {
        Some(end) => (kind, inside.get(..end)?.to_vec(), false),
        None => (kind, inside, true),
    })
}

/// Where in a type a token is, that a caller looks for.
enum Place {
    /// Where a type is written: a type of its own, `f16`, or the first segment of a path, `f16::X`, which may
    /// be a module's name.
    Type,
    /// The whole length of an array, `[u8; _]`.
    Length,
}

/// `tokens` as written, at any depth, but for the tokens `leaf` picks where a type is written: `found` is told
/// which token it is and where in the type, and answers the tokens to write in its place, or `None` to leave it
/// as it is. It is the one rule, for the macros, of where a type is written. No type is written in a block,
/// which is an expression, `Pair<{ !0usize & 3 }>`; in a macro's call, `f16!()` or `named!(f16)`, which the
/// macro reads; nor in an array's length, all that follows the `;` and an expression too,
/// `[u8; (1 != 2) as usize]`, but `found` is told of a length of one token `leaf` picks, `[u8; _]`. A token
/// `leaf` picks is a name, not a type, after a `'`, a lifetime's, `&'f16 u8`; after a `::`, a later segment of a
/// path, `core::primitive::f16`; and before a `:`, a function pointer's parameter's, `fn(f16: u8)`. Where
/// kernite cannot tell, as for a macro that writes the token into a type, it takes it for no type's. What it
/// writes is for Rust, so a fragment that stays one type in a bracket is in parentheses there, as [`for_rust`] has it.
fn walk(
    mut tokens: &[TokenTree],
    leaf: fn(&TokenTree) -> bool,
    found: &mut dyn FnMut(&TokenTree, Place) -> Option<TokenStream>,
) -> TokenStream {
    let all = tokens;
    let mut out = TokenStream::new();
    while let [t, rest @ ..] = tokens {
        let before = all.get(..all.len() - tokens.len()).unwrap_or_default();
        tokens = rest;
        match (t, rest) {
            (TokenTree::Ident(_), [bang, args @ TokenTree::Group(_), after @ ..]) if is_punct(bang, '!') => {
                out.extend([t.clone(), bang.clone(), args.clone()]);
                tokens = after;
            }
            (quote, [name, after @ ..]) if is_punct(quote, '\'') => {
                out.extend([quote.clone(), name.clone()]);
                tokens = after;
            }
            (semi, length) if is_punct(semi, ';') => {
                let blank = match length {
                    [one] if leaf(one) => found(one, Place::Length),
                    _ => None,
                };
                out.extend([semi.clone()]);
                out.extend(blank.unwrap_or_else(|| length.iter().cloned().collect()));
                tokens = &[];
            }
            (name, _) if leaf(name) && !is_name(before, rest) => {
                out.extend(found(name, Place::Type).unwrap_or_else(|| [name.clone()].into_iter().collect()));
            }
            (TokenTree::Group(g), _) if g.delimiter() != Delimiter::Brace => {
                let mut inner = Group::new(g.delimiter(), walk(&contents(g), leaf, found));
                inner.set_span(g.span());
                out.extend([TokenTree::Group(inner)]);
            }
            _ => out.extend([t.clone()]),
        }
    }
    out
}

/// Whether the token between `before` and `after` is a name, not a type: a later segment of a path, after a
/// `::`; or a function pointer's parameter's, before a `:`. A `::` is a `:` joined to the next, so the `:`
/// of a parameter's name before a path, `fn(f16: ::core::u8)`, is not the first of one.
fn is_name(before: &[TokenTree], after: &[TokenTree]) -> bool {
    let sep = |a: &TokenTree, b: &TokenTree| {
        matches!(a, TokenTree::Punct(p) if p.as_char() == ':' && p.spacing() == Spacing::Joint) && is_punct(b, ':')
    };
    matches!(before, [.., a, b] if sep(a, b))
        || matches!(after, [colon, next @ ..] if is_punct(colon, ':') && !matches!(next, [b, ..] if sep(colon, b)))
}

/// Whether `tokens` hold the placeholder `_`, at any depth, as `*const [_; 3]` does: Rust refuses it in a
/// signature, once, at the `_`. It is looked for where [`walk`] says a type is written. A `_` is no
/// placeholder in a block, which is an expression where `_` is a pattern, `Pair<{ let _ = 1; 3 }>`; in an
/// array's length, unless it is the whole length, `[u8; _]`, for `[u8; if let _ = 1 { 3 } else { 4 }]` has a
/// pattern; as the name of a function pointer's parameter, `fn(_: u8)`, or of a lifetime, `&'_ u8`; or among a
/// macro's arguments, `len!(_)`, which are the macro's. Where kernite cannot tell, as for a macro that writes
/// a `_` into a type, it takes the `_` for no placeholder. Rust then says what it says of the `_`, which a
/// kernel's author can mend, where a kernel that builds and does not run cannot be.
fn placeholder(tokens: &[TokenTree]) -> bool {
    let mut blank = false;
    walk(tokens, |t| is_ident(t, "_"), &mut |_, _| {
        blank = true;
        None
    });
    blank
}

/// Whether `f` gives back a type with a `_` in it, which Rust refuses in a signature.
pub(crate) fn gives_placeholder(f: &Fn, top: &[TokenTree]) -> bool {
    f.ret.as_ref().is_some_and(|r| placeholder(within(top, &r.tokens)))
}

/// What Rust is asked of a type of a kernel's, and where the type is.
enum Ask {
    /// Where the type a pointer of one word, `*const T` or `*mut T`, points to begins and where it
    /// ends: whether `T` has a size.
    Pointer(Span, Span),
    /// A number by its name or its path, and the tokens that write it: whether they are that number,
    /// the primitive `::core::primitive::u64`, and not another type: one the module gives the name,
    /// or one the path finds through a `core` or `std` the module gives another meaning. Rust is
    /// asked of the type alone, never of a value of it.
    Number(&'static str, Vec<TokenTree>),
}

/// The number `ty` is, if `ty` is one by its name or its path, `u64` or `core::primitive::u64`: `u8` to
/// `u64`, `i8` to `i64`, `usize`, `isize`, `f32`, `f64` or `bool`, which the language reads by their names.
pub(crate) fn number_name(ty: &[TokenTree]) -> Option<&'static str> {
    let read = lang::read_type(&lex(ty).0.ok()?);
    let known = lang::IntTy::parse(&read).map(lang::IntTy::name);
    known.or_else(|| ["bool", "f32", "f64"].into_iter().find(|n| *n == read))
}

/// For each of `f`'s parameters, and last for its result: what Rust is asked of its type, if it is
/// asked. It is asked of a number by its name or its path, as [`number_name`] reads it, and not of
/// `f16` and `bf16`, which Rust has not; of the elements of an array or a slice by reference,
/// `&[T; N]` or `&[T]`, as of a number; and of a pointer of one word, `*const T` or `*mut T`. The
/// language says which: a pointer to a slice, a `str` or a `dyn` has two words, and is its error
/// alone, which Rust is not asked of, for it would say the same twice. Nor is it asked of a pointer
/// to a type with a `_` in it, `*const _` or `*const [_; 3]`, which is Rust's error alone, at the
/// `_`: the type to ask of is one Rust cannot infer, and it would say that too.
fn asked(f: &Fn, top: &[TokenTree]) -> Vec<Option<Ask>> {
    let ask = |ty: &[TokenTree]| {
        if let Some(name) = number_name(ty) {
            return Some(Ask::Number(name, ty.to_vec()));
        }
        if let Some((_, elem, _)) = array_ref(ty) {
            return number_name(&elem).map(|name| Ask::Number(name, elem));
        }
        let (_, to) = pointer(ty)?;
        let one = lex(ty).0.is_ok_and(|trees| lang::pointer(&lang::read_type(&trees)) == Ok(true));
        (one && !placeholder(to)).then(|| Ask::Pointer(span_of(to.first()), span_of(to.last())))
    };
    let params = written_params(f, top).into_iter().map(|p| ask(&p.ty));
    params.chain([f.ret.as_ref().and_then(|r| ask(within(top, &r.tokens)))]).collect()
}

/// What asks Rust of each type `f` takes or gives back: [`numbers`] asks of the numbers, and one closure, which Rust
/// checks and never runs, takes each pointer as the kernel writes it, the type it points to as `pointee` names it, and
/// after it an array of none of that type, `[T; 0]`. The elements of an array must have a size, so Rust says of a type
/// that has none that its size cannot be known, whatever the type is and however it is called: an alias for a slice or
/// a trait object, a struct that ends in one. Rust says nothing more of a type it refuses by itself,
/// `NeedsCopy<String>`: the pointer's, before the array, has said it, and the same error twice is one. A closure's
/// parameter is Rust's to say an error in, at the type, as a function's is. Nothing if `f` has none to ask of.
pub(crate) fn asks(f: &Fn, top: &[TokenTree], pointee: fn(&[TokenTree]) -> TokenStream) -> TokenStream {
    let asked = asked(f, top);
    let mut out = numbers(&asked);
    let result = f.ret.as_ref().map(|r| within(top, &r.tokens).to_vec());
    let types = written_params(f, top).into_iter().map(|p| p.ty).chain(result);
    let mut params = TokenStream::new();
    for (ty, ask) in types.zip(&asked) {
        let Some(Ask::Pointer(first, _)) = ask else { continue };
        let Some((kind, to)) = pointer(&ty) else { continue };
        // Rust says of the type of a closure's parameter that holds a part with no size, `(dyn Debug, u8)`, that it has
        // none, at the whole parameter, from its `_` to the last token of its type, so the `_` is at the type's first
        // token: one of the macro's would put the error at the attribute.
        params.extend([TokenTree::Ident(Ident::new("_", span_of(ty.first())))]);
        params.extend(parse(&format!(":*{kind}")));
        params.extend(pointee(to));
        params.extend(parse(","));
        // Rust says that an element has no size at the array, `[[u8]; 0]`, or at the whole parameter, from its `_` to
        // the array, if the part with no size is a trait object, `[dyn Debug; 0]`; and a span that joins a token of the
        // macro's to the type's is the macro's: it puts a note beside the error, "in this attribute macro expansion".
        // So the `_` and the array are at the type's first token, as one token's span is the array's.
        let mut elements = pointee(&unparenthesized(to));
        elements.extend(parse(";0"));
        let mut array = Group::new(Delimiter::Bracket, elements);
        array.set_span(*first);
        params.extend(respan(parse("_:"), *first));
        params.extend([TokenTree::Group(array)]);
        params.extend(parse(","));
    }
    if !params.is_empty() {
        out.extend(parse("let _ = |"));
        out.extend(params);
        out.extend(parse("| {};"));
    }
    out
}

/// The block of a `const` operand of a naked kernel's own `naked_asm!`, which asks Rust of each type
/// the kernel takes or gives back, as [`asks`] says. A naked function has no body to write the
/// questions in. Nothing if the kernel has none to ask of.
pub(crate) fn probe(f: &Fn, top: &[TokenTree]) -> Option<TokenStream> {
    let mut block = asks(f, top, written);
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
        if let Some(Ask::Number(name, ty)) = ask {
            inner.extend(number(name, ty));
        }
    }
    unnamed(inner)
}

/// `inner` in an item of no name, `const _: () = { .. };`, where Rust checks it and runs nothing. No `inner`, no item.
pub(crate) fn unnamed(inner: TokenStream) -> TokenStream {
    if inner.is_empty() {
        return inner;
    }
    let mut out = respan(parse("const _: () ="), Span::mixed_site());
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, inner))]);
    out.extend(respan(parse(";"), Span::mixed_site()));
    out
}

/// `let _ = |__kernite_asked: u64| { let _: ::core::primitive::u64 = __kernite_asked; };`, which
/// asks Rust whether the type the kernel writes, `u64` or `core::primitive::u64`, `ty`, is the number
/// `name`, as a module's `u64` is not when it gives the name another type, nor its `core` when it gives
/// that another meaning. A closure that Rust checks and never calls takes a value of the type as
/// written; so Rust says of a type that is not the number that it is not, once, at the type, and in the
/// order a kernel that compiles says it: it expected the number and found the type. It asks of the type
/// and not of any value of it, which a type that has no size has none of. Each token but the type is
/// the macro's, located where the type begins.
fn number(name: &str, ty: &[TokenTree]) -> TokenStream {
    let at = span_of(ty.first());
    let mut out = placed("let _ = | __kernite_asked :", at);
    out.extend(ty.iter().cloned());
    out.extend(placed(&format!("| {{ let _: {} = __kernite_asked; }};", primitive(name)), at));
    out
}

/// The tokens `s` writes, each located at `at` and of the macro's hygiene.
fn placed(s: &str, at: Span) -> TokenStream {
    respan(parse(s), at.resolved_at(Span::mixed_site()))
}

/// `unsafe { call }`, which gives back the result of an instance of `f` as `f` gives it back: `call` is the call. A
/// pointer is cast back, `.cast()`, from the `*const ()` an instance gives. A cast says an error in the type again, as
/// any generic does, at the span of the cast: so the call and the cast are located across the type, the span Rust says
/// the error at in the signature when it says the whole type, and then the two are one error. For a type that holds the
/// part Rust refuses, `(u8, [CStr; 3])`, Rust says the part in the signature, `[CStr; 3]`, and the cast says the whole
/// type: the two are two. Rust is asked of a number and of the size of a pointer's type by [`asks`].
pub(crate) fn result(f: &Fn, top: &[TokenTree], call: TokenStream) -> TokenStream {
    let mut out = call;
    if let Some(Ask::Pointer(first, last)) = asked(f, top).pop().flatten() {
        out = out.into_iter().map(|t| located(t, first)).collect();
        out.extend(parse("."));
        out.extend([TokenTree::Ident(Ident::new("cast", first))]);
        out.extend([located(Group::new(Delimiter::Parenthesis, TokenStream::new()).into(), last)]);
    }
    let mut block = parse("unsafe");
    block.extend([TokenTree::Group(Group::new(Delimiter::Brace, out))]);
    block
}

/// `token`, located at `at`.
fn located(mut token: TokenTree, at: Span) -> TokenTree {
    token.set_span(at);
    token
}

/// `tokens`, which end an item the macros write for one of the user's, with its last token located at `at`, a token of
/// the user's. Rust says of an item, as of `missing_docs`, where the item's span is, which runs from its first token to
/// its last: the user's if both are, so Rust tells at the item as it does of any the user wrote; and the macro's
/// otherwise, which is the module's attribute, where nothing can be mended.
pub(crate) fn ending_at(tokens: TokenStream, at: Span) -> TokenStream {
    let mut tokens: Vec<TokenTree> = tokens.into_iter().collect();
    if let Some(last) = tokens.last_mut() {
        last.set_span(at);
    }
    tokens.into_iter().collect()
}

/// `code` in braces, the body of an item the macros write for one of the user's, located at `at`, a token of the
/// user's: it ends the item, as [`ending_at`] says.
pub(crate) fn body_at(code: TokenStream, at: Span) -> TokenTree {
    located(TokenTree::Group(Group::new(Delimiter::Brace, code)), at)
}

/// `f` with a body that never runs, for a kernel that does not compile. It asks Rust of each type `f`
/// takes or gives back, as [`asks`] says: of a number by a closure of the type as written, which it
/// never calls, so that a result reads as it does in a kernel that compiles, "expected `f64`, found
/// `f32`", and not the other way round, as a literal of the number would read. It ends with
/// `unreachable!()`.
///
/// If `f` gives back a type with a `_` in it, which Rust refuses in a signature, once, at the `_`,
/// the stub casts `f` itself, `self::f as *const ()`, which asks Rust for the signature it is in
/// the middle of checking. Rust gives up on the body, as it does after any error, and does not say
/// as well that a body that never returns leaves the type of the `_` to infer.
///
/// Its body has the `places` of the kernel's code that name an item, as [`named`](crate::expand) says them.
pub(crate) fn stub(f: &Fn, top: &[TokenTree], places: TokenStream) -> TokenStream {
    let mut out = attrs(f, top);
    out.extend(signature(f, top));
    let mut body = asks(f, top, written);
    body.extend(unnamed(places));
    if gives_placeholder(f, top)
        && let Some(name) = name_after(&f.head, top)
    {
        body.extend(parse(&format!("let _ = self::{name} as *const ();")));
    }
    body.extend(parse(&never(f, top)));
    body.extend(parse("::core::unreachable!()"));
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, body))]);
    out
}

/// What a body of `f`'s parameters does when it never runs, for a kernel that does not compile: it
/// uses each parameter, by its name, which Rust would find unused.
pub(crate) fn never(f: &Fn, top: &[TokenTree]) -> String {
    written_params(f, top).iter().map(|p| format!("let _ = {};", p.name)).collect()
}

/// `f`'s attributes as written, but `#[instances(..)]`, `#[cooperative]` and its directives,
/// `#[maxntid(..)]` and the like, which Rust does not know.
pub(crate) fn attrs(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let instances = f.instances.as_ref().map_or(0..0, |i| i.tokens.clone());
    let cooperative = f.cooperative.as_ref().map_or(0..0, |c| c.tokens.clone());
    let kernite = |i: &usize| {
        [&instances, &cooperative].into_iter().chain(f.directives.iter().map(|d| &d.tokens)).any(|r| r.contains(i))
    };
    f.attrs.clone().filter(|i| !kernite(i)).filter_map(|i| top.get(i).cloned()).collect()
}

/// What the code that stands for an item, an error said of it or a method it gives, keeps of the
/// item's attributes, `attrs`, `#` and brackets each: `cfg`, which says whether the item is there,
/// alone or in a `cfg_attr(..)`; and, if `offered`, `deprecated` and `doc(hidden)`, which say how it is
/// offered. Nothing else: a lint or a symbol is the item's own.
pub(crate) fn carried(attrs: impl IntoIterator<Item = TokenTree>, offered: bool) -> TokenStream {
    filtered(attrs, &mut |meta| match &meta[..] {
        [doc, TokenTree::Group(list)] if offered && named(doc, "doc") => flatten(list.stream())
            .iter()
            .any(|t| named(t, "hidden"))
            .then(|| parse("doc(hidden)").into_iter().collect()),
        [name, ..] if named(name, "cfg") || (offered && named(name, "deprecated")) => Some(meta),
        _ => None,
    })
}

/// The attributes that set the level of a lint.
pub(crate) const LEVELS: [&str; 5] = ["allow", "warn", "deny", "forbid", "expect"];

/// What an item's attributes, `attrs`, `#` and brackets each, say of the level of a lint, of those among `kinds`, which
/// are some of [`LEVELS`]: each of them, alone or in a `cfg_attr(..)`.
pub(crate) fn levels(attrs: impl IntoIterator<Item = TokenTree>, kinds: &[&str]) -> TokenStream {
    filtered(attrs, &mut |meta| {
        let [name, ..] = &meta[..] else { return None };
        kinds.iter().any(|kind| named(name, kind)).then_some(meta)
    })
}

/// What an item's attributes, `attrs`, say of the level of a lint, as [`levels`] says it, for a naked function, which
/// Rust refuses `expect` on: an `expect` is an `allow` there.
pub(crate) fn naked_levels(attrs: impl IntoIterator<Item = TokenTree>) -> TokenStream {
    filtered(levels(attrs, &LEVELS), &mut |mut meta| {
        if let Some(first) = meta.first_mut()
            && named(first, "expect")
        {
            *first = Ident::new("allow", first.span()).into();
        }
        Some(meta)
    })
}

/// What an item's attributes, `attrs`, `#` and brackets each, say of its being deprecated: each `deprecated`, alone or
/// in a `cfg_attr(..)`.
pub(crate) fn deprecated(attrs: impl IntoIterator<Item = TokenTree>) -> TokenStream {
    filtered(attrs, &mut |meta| matches!(&meta[..], [name, ..] if named(name, "deprecated")).then_some(meta))
}

/// What an item that is no item of Rust's takes of its attributes, `attrs`, `#` and brackets each, and, of the rest,
/// each one's name, a raw one plain, and where it is. A static of GPU kernels, a helper and an `extern "ptx-func"`
/// function are no symbol of the host's, nor an item of Rust's, so each takes what says how Rust has it, and no other
/// attribute: a doc, `cfg`, `deprecated`, the level of a lint, `allow`, `warn`, `deny`, `forbid` or `expect`, and a
/// `rustfmt` attribute, each alone or in a `cfg_attr(..)`, whatever its condition. The others are for what the item is
/// not: `no_mangle`, `export_name`, `link_section` and `used` for a symbol, `cold` and `track_caller` for a function of
/// Rust's, and the rest of Rust's for an item of its own. `unsafe(no_mangle)` is `no_mangle`.
pub(crate) fn taken(attrs: impl IntoIterator<Item = TokenTree>) -> (TokenStream, Vec<(Span, String)>) {
    let mut refused = vec![];
    let taken = filtered(attrs, &mut |meta| {
        let [first, ..] = &meta[..] else { return None };
        let at = first.span();
        let takes = ["doc", "cfg", "deprecated", "rustfmt"];
        if takes.iter().chain(&LEVELS).any(|name| named(first, name)) {
            return Some(meta);
        }
        let attr = match &meta[..] {
            [wrapper, TokenTree::Group(inner)] if named(wrapper, "unsafe") => flatten(inner.stream()),
            _ => meta,
        };
        // The name, which may be a path, `a::b`, up to its arguments or its value.
        let path: Vec<&TokenTree> =
            attr.iter().take_while(|t| matches!(t, TokenTree::Ident(_)) || is_punct(t, ':')).collect();
        let written = path.iter().map(|t| lang::unraw(&t.to_string()).to_string()).collect();
        refused.push((path.first().map_or(at, |t| t.span()), written));
        None
    });
    (taken, refused)
}

/// The error at each of `attrs` that the item named `name` does not take, as [`taken`] says, under `cfg`, the item's
/// own: where `cfg` leaves the item out, it says nothing. It names the item as written, `r#type` as well, and says what
/// it is, `is`, and what takes the attributes it takes, `takers`.
pub(crate) fn refused_attrs(
    attrs: impl IntoIterator<Item = TokenTree>,
    name: &TokenTree,
    (is, takers): (&str, &str),
    cfg: &TokenStream,
) -> TokenStream {
    let said = |(at, attr): (Span, String)| {
        let message = format!(
            "`{name}` is {is}, which `{attr}` is not for: {takers} takes a doc, `cfg`, `deprecated`, the level of a \
             lint, and a `rustfmt` attribute, alone or in a `cfg_attr(..)`"
        );
        error_under(cfg, at, &message)
    };
    taken(attrs).1.into_iter().map(said).collect()
}

/// The attributes `attrs`, `#` and brackets each, as `leaf` leaves them: `leaf` is given the tokens in an attribute's
/// brackets, and gives what it leaves of them, or `None` if nothing. An attribute that is a `cfg_attr(..)` is left as
/// its condition and what `leaf` leaves of each attribute in its list, or nothing if that is nothing: a `cfg_attr(..)`
/// of a condition alone is an error.
pub(crate) fn filtered(
    attrs: impl IntoIterator<Item = TokenTree>,
    leaf: &mut impl FnMut(Vec<TokenTree>) -> Option<Vec<TokenTree>>,
) -> TokenStream {
    let attrs: Vec<TokenTree> = attrs.into_iter().collect();
    let mut out = TokenStream::new();
    for attr in attrs.chunks(2) {
        let [hash, TokenTree::Group(group)] = attr else { continue };
        let Some(meta) = keep(flatten(group.stream()), leaf) else { continue };
        let mut kept = Group::new(Delimiter::Bracket, meta.into_iter().collect());
        kept.set_span(group.span());
        out.extend([hash.clone(), kept.into()]);
    }
    out
}

/// What [`filtered`] leaves of an attribute, `meta` the tokens in its brackets; `None` if nothing.
fn keep(
    meta: Vec<TokenTree>,
    leaf: &mut impl FnMut(Vec<TokenTree>) -> Option<Vec<TokenTree>>,
) -> Option<Vec<TokenTree>> {
    match &meta[..] {
        [cfg_attr, TokenTree::Group(list)] if named(cfg_attr, "cfg_attr") => {
            let tokens = flatten(list.stream());
            let mut parts = tokens.split(|t| is_punct(t, ','));
            let mut kept = parts.next().unwrap_or_default().to_vec();
            let condition = kept.len();
            for part in parts.filter_map(|part| keep(part.to_vec(), leaf)) {
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
        _ => leaf(meta),
    }
}

/// Whether `t` is the name `name`, written raw or not: Rust reads `r#name` as `name`, an attribute's, `r#cfg`, or a
/// type's, `r#u64`. A keyword is asked as written, by [`is_ident`]: `r#let` is a name, and not `let`.
pub(crate) fn named(t: &TokenTree, name: &str) -> bool {
    matches!(t, TokenTree::Ident(i) if lang::unraw(&i.to_string()) == name)
}

/// `f`'s signature as written, from its visibility to its result, but for its parameters, as
/// [`value_params`] has them, and its result, which is [`given_back`]'s.
pub(crate) fn signature(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out: TokenStream = top.get(f.head.start..f.params_at).unwrap_or_default().iter().cloned().collect();
    let mut params = Group::new(Delimiter::Parenthesis, value_params(f, top));
    params.set_span(span_of(top.get(f.params_at)));
    out.extend([TokenTree::Group(params)]);
    if let Some(r) = &f.ret {
        // The arrow, then the result, which ends where the kernel's does.
        out.extend(top.get(f.params_at + 1..r.tokens.start).unwrap_or_default().iter().cloned());
        let ret = within(top, &r.tokens);
        out.extend(ending_at(given_back(ret), span_of(ret.last())));
    }
    out
}

/// `f`'s parameters as [`written_params`] has them, a comma after each, but for each type, which is
/// [`value`]'s.
pub(crate) fn value_params(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out = TokenStream::new();
    for p in written_params(f, top) {
        out.extend(p.head);
        out.extend(value(&p.ty));
        out.extend(parse(","));
    }
    out
}

/// The tokens of `stream`, with the invisible groups a macro's fragments arrive in opened, but for those that stay: the
/// group [`stays`] says is one operand or one type, as Rust reads `$n * 2` with `$n` as `1 + 2` as `(1 + 2) * 2`, and
/// opened it would be `1 + 2 * 2`; and the group [`type_behind`] says ends the type behind a `&`, which [`for_rust`]
/// parts from the `&`. A group that stays is as it was: what is in it is opened where it is read. But a group of that
/// second kind that holds the parentheses of a trait object's bounds, `(dyn Debug + Send)`, is those parentheses
/// written again as the macro's own: [`for_rust`] would put parentheses round the user's pair, which a `&` wants there,
/// and Rust would call the user's unnecessary. Any other pair it holds, `(u32)`, `(dyn Debug)` or `([u8])`, is left to
/// [`for_rust`] with the rest, and Rust lints a needless pair of the user's behind a `&` as it does in a plain
/// function.
pub(crate) fn flatten(stream: TokenStream) -> Vec<TokenTree> {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    let types: Vec<Range<usize>> = (0..tokens.len()).filter_map(|at| type_behind(&tokens, at)).collect();
    let mut out = vec![];
    for (i, t) in tokens.iter().enumerate() {
        match t {
            TokenTree::Group(g) if g.delimiter() == Delimiter::None => {
                let inside = flatten(g.stream());
                let ends_type = types.iter().any(|ty| ty.end == i + 1);
                let (before, after) = (tokens.get(..i).unwrap_or_default(), tokens.get(i + 1..).unwrap_or_default());
                match inside.as_slice() {
                    [TokenTree::Group(b)] if ends_type && bounded(b).is_some() => {
                        out.push(own(b.delimiter(), b.stream(), place(&tokens, &types, i, b.span())));
                    }
                    [_, ..] if ends_type => out.push(t.clone()),
                    _ if stays(&inside, before, after) => out.push(t.clone()),
                    _ => out.extend(inside),
                }
            }
            t => out.push(t.clone()),
        }
    }
    out
}

/// Whether an invisible group that holds `inside`, between `before` and `after`, stays one. An operator in it
/// could be taken apart by a token next to it, as `* 2` takes `1 + 2` apart, and as `*const` takes the `+` of
/// `dyn Debug + Send` for its own, which Rust refuses as ambiguous: so it stays if it holds an operator outside
/// any bracket, `1 + 2`, `-1`, `a as u8`, `0..4` or `dyn Debug + Send`, and a token that takes it apart is next to
/// it. Before it is one of `+ - * / % ^ & | < >` and `!`; `==`, `!=`, `<=` or `>=`; `const` or `mut`, of a pointer,
/// `*const T`, or a reference, `&mut T`; or the name of a lifetime, `&'a T`. After it is one of `+ - * / % ^ & | < >`,
/// or `==`, `!=`, `.` or `as`. The `>` of an arrow, `->`, is a result's, and no operator's, but for the `+` of a trait
/// object's bounds, which Rust calls ambiguous after an arrow, `fn() -> dyn Debug + Send`, as after a `*const`: a group
/// that holds one stays after an arrow, whatever is after it. A lone `=` is a `let`'s or an assignment's. A group that
/// holds `a.b`, `f(x)` or `a::b`, or that no such token is next to, is read as it would be without the group, and is
/// opened.
fn stays(inside: &[TokenTree], before: &[TokenTree], after: &[TokenTree]) -> bool {
    let is_any = |t: &TokenTree, chars: &str| chars.chars().any(|c| is_punct(t, c));
    let loose = inside.iter().any(|t| is_any(t, "+-*/%^&|<>!=") || is_ident(t, "as"))
        || inside.windows(2).any(|w| matches!(w, [a, b] if is_punct(a, '.') && is_punct(b, '.')));
    // `==`, `!=`, `<=` and `>=` are an `=` that an operator is joint to.
    let compares = |op: &TokenTree, eq: &TokenTree| {
        is_punct(eq, '=') && is_any(op, "=!<>") && matches!(op, TokenTree::Punct(p) if p.spacing() == Spacing::Joint)
    };
    let arrow = matches!(before, [.., dash, head] if is_punct(dash, '-') && is_punct(head, '>'));
    let bounds = || split_top(inside, |t| is_punct(t, '+')).count() > 1;
    let left = before.last().is_some_and(|t| is_any(t, "+-*/%^&|<>!")) && (!arrow || bounds())
        || before.last().is_some_and(|t| is_ident(t, "const") || is_ident(t, "mut"))
        || matches!(before, [.., quote, _] if is_punct(quote, '\''))
        || matches!(before, [.., op, eq] if compares(op, eq));
    let right = after.first().is_some_and(|t| is_any(t, "+-*/%^&|<>.") || is_ident(t, "as"))
        || matches!(after, [op, eq, ..] if compares(op, eq));
    loose && (left || right)
}

/// The places in `tokens` of the type behind the `&` at `at`, `$t` of `&$t`, if the `&` starts one: from after the name
/// of its lifetime, `&'a $t`, and its `mut`, to its last token. A `&` starts a type where it is the first token, or
/// where a punct but `&` is before it, or `const` or `mut`, as in `*const &$t`. Any other token before it is taken for
/// an operand, as the `a` of `a & b` is, and so is a `&`, as in `a && b`: where a `&` behind another is a reference,
/// `&&$t` or `&'a &$t`, it is in the type of the first, which ends where its own would. The type ends at the first `,`,
/// `;`, `=`, `as`, `where` or body that is not among the arguments of a type's, or at the `>` that closes the arguments
/// it is among, as in `Option<&$t>` and `<&$t as Trait>::Out`. A brace is a body, and so is the group of no brackets
/// that a `$b:block` arrives in, which holds one: [`is_body`]. So `&dyn $d`, `&mut $t`, `&*const $t` and `&fn() -> $t`
/// end in `$t`, a group or an ident, and `&$t<u8>` starts with one; `&Wrap<$t>` and `&[$t]` start and end in tokens of
/// the macro's own, and Rust finds the `&` of those alone.
fn type_behind(tokens: &[TokenTree], at: usize) -> Option<Range<usize>> {
    let starts_type = match at.checked_sub(1).and_then(|before| tokens.get(before)) {
        None => true,
        Some(TokenTree::Punct(p)) => p.as_char() != '&',
        Some(before) => is_ident(before, "const") || is_ident(before, "mut"),
    };
    if !tokens.get(at).is_some_and(|t| is_punct(t, '&')) || !starts_type {
        return None;
    }
    let stops = |t: &TokenTree| {
        [',', ';', '>', '='].into_iter().any(|c| is_punct(t, c))
            || is_ident(t, "as")
            || is_ident(t, "where")
            || is_body(t)
    };
    // The first part is the type, with the name of its lifetime and its `mut`.
    let last = at + split_top(tokens.get(at + 1..)?, stops).next()?.len();
    let mut first = at + 1;
    if tokens.get(first).is_some_and(|t| is_punct(t, '\'')) {
        first += 2;
    }
    if tokens.get(first).is_some_and(|t| is_ident(t, "mut")) {
        first += 1;
    }
    (first <= last).then_some(first..last + 1)
}

/// Whether `t` is a body: a pair of braces, or the group of no brackets that holds one, which is how a `$b:block`
/// arrives, the macro's call giving the body.
fn is_body(t: &TokenTree) -> bool {
    match t {
        TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => true,
        TokenTree::Group(g) if g.delimiter() == Delimiter::None => {
            matches!(&g.stream().into_iter().collect::<Vec<_>>()[..], [inside] if is_body(inside))
        }
        _ => false,
    }
}

/// The first and the last token of `t`: itself, but for a group that [`flatten`] keeps, where they are those
/// of what is in it.
fn ends(t: &TokenTree) -> (TokenTree, TokenTree) {
    match t {
        TokenTree::Group(g) if g.delimiter() == Delimiter::None => {
            let inside = flatten(g.stream());
            match (inside.first(), inside.last()) {
                (Some(first), Some(last)) => (ends(first).0, ends(last).1),
                // `flatten` opens a group with nothing in it, so there is none.
                _ => (t.clone(), t.clone()),
            }
        }
        t => (t.clone(), t.clone()),
    }
}

/// `tokens`, as [`flatten`] gives them, for Rust: a group that stays one is written in parentheses. Rust keeps a
/// fragment's group with no brackets as one operand only where a macro hands it on as it came; a macro that takes the
/// tokens apart, as [`flatten`] does, writes the group again, and Rust reads it as no group at all. The parentheses are
/// the macro's own, as Rust lints none of what the macro adds, and would lint a type in them, `Foo<(Vec<u8>)>`; and
/// they stand where the fragment starts, for Rust to say an error of the whole fragment there. Where the group is the
/// whole type behind a `&`, `&$t`, they stand at the end of the token before it, the `&`, its `mut` or the name of its
/// lifetime: [`place`] says why. A bracket among the tokens is handed on as it came, with each fragment in it one
/// operand.
pub(crate) fn for_rust(tokens: Vec<TokenTree>) -> Vec<TokenTree> {
    let types: Vec<Range<usize>> = (0..tokens.len()).filter_map(|at| type_behind(&tokens, at)).collect();
    tokens
        .iter()
        .enumerate()
        .map(|(at, t)| match t {
            TokenTree::Group(g) if g.delimiter() == Delimiter::None => {
                own(Delimiter::Parenthesis, g.stream(), place(&tokens, &types, at, ends(t).0.span()))
            }
            t => t.clone(),
        })
        .collect()
}

/// Where the parentheses of the macro's own that stand for the group at `at` in `tokens` are: at `start`, where the
/// fragment starts, but where the group is the whole type behind a `&`, as `types` has them, at the end of the token
/// before it: the `&`, its `mut` or the name of its lifetime. Rust says that a `&` has no lifetime at the start of the
/// span from the `&` to the last token of its type, and its help, to return an owned value, takes out the span from the
/// `&` to the first. Where the type is a fragment's, the `&` is the macro's and the type the call's, two hygienes, and
/// Rust puts the type where the macro first wrote the fragment, at the `$t` of the first of the kernels that take it.
/// So it says the error of each of the others there, and its help takes out all the code between the two. The
/// parentheses are of a hygiene no fragment has, and they have no width: each of those spans is the `&`'s alone, or the
/// `&mut`'s, and Rust says what it says of the `&` at the `&`.
fn place(tokens: &[TokenTree], types: &[Range<usize>], at: usize, start: Span) -> Span {
    match at.checked_sub(1).and_then(|before| tokens.get(before)) {
        Some(before) if types.contains(&(at..at + 1)) => before.span().end(),
        _ => start,
    }
}

/// `tokens`, as [`flatten`] gives them, but for each type behind a `&` that starts or ends in an ident the macro was
/// given, a `$t:ident`: it is in a group of no brackets, as the tokens of a `$t:ty` are, for [`for_rust`] to write in
/// parentheses. An ident that ends it is written again with the place it has in the call and the hygiene of the macro,
/// which Rust does not move to where the macro first wrote it: no parentheses part it from a `&` in the group, the
/// second of `&&$t`. `end` is where the tokens end: the bracket they are in, closing.
pub(crate) fn grouped(tokens: Vec<TokenTree>, end: Span) -> Vec<TokenTree> {
    let given = |at: usize| matches!(tokens.get(at), Some(TokenTree::Ident(_))) && is_given(&tokens, at, end);
    let mut out = Vec::with_capacity(tokens.len());
    let mut at = 0;
    while let Some(t) = tokens.get(at) {
        let Some(ty) = type_behind(&tokens, at).filter(|ty| given(ty.start) || given(ty.end - 1)) else {
            out.push(t.clone());
            at += 1;
            continue;
        };
        out.extend(tokens.get(at..ty.start).unwrap_or_default().iter().cloned());
        let mut inside = tokens.get(ty.clone()).unwrap_or_default().to_vec();
        if let Some(TokenTree::Ident(name)) = inside.last_mut().filter(|_| given(ty.end - 1)) {
            name.set_span(Span::mixed_site().located_at(name.span()));
        }
        out.push(TokenTree::Group(Group::new(Delimiter::None, inside.into_iter().collect())));
        at = ty.end;
    }
    out
}

/// What is in the bracket `group`, as Rust has it: [`flatten`] opens what it does, [`grouped`] gives a group to each
/// type behind a `&` that starts or ends in an ident the macro was given, and [`for_rust`] writes the groups.
fn contents(group: &Group) -> Vec<TokenTree> {
    for_rust(grouped(flatten(group.stream()), group.span_close()))
}

/// Whether the ident at `at` in `tokens`, the first or the last of the type behind a `&`, is one the macro was given, a
/// `$t:ident`, and not one written in place, as the `u8` of `&u8` is. One it was given has the place of the call: in
/// another file, or before or after the body of the macro. One written in place is after the token before it, in their
/// file, and before the token after it, where that is after the token before it. Where it is not, the call wrote both,
/// as a `$r:ty` of `&u8` does, and the token after them is the macro's. The token after is `end`, the bracket the
/// tokens are in, closing, where the ident is the last token, and where a body follows it, `-> &$t $b`: a body the call
/// gives, a `$b:block`, has the place of the call, whoever wrote the ident. A `$t:tt` that Rust has moved to the place
/// of its `$t`, for the second kernel that takes it, is in place too. Where the `&` is not in the body of a macro, Rust
/// says an error at an ident written again, and not at the `&`, so one written in place is left as it is.
fn is_given(tokens: &[TokenTree], at: usize, end: Span) -> bool {
    let precedes = |first: Span, second: Span| {
        let place = |span: Span| (span.line(), span.column());
        first.file() == second.file() && place(first.end()) <= place(second)
    };
    // A `&` is before the first token of a type, so `at` is not the first.
    let (Some(before), Some(ident)) = (at.checked_sub(1).and_then(|before| tokens.get(before)), tokens.get(at)) else {
        return false;
    };
    let (before, ident) = (before.span(), ident.span());
    let after = tokens.get(at + 1).filter(|t| !is_body(t)).map_or(end, TokenTree::span);
    let in_place = precedes(before, ident) && (precedes(ident, after) || !precedes(before, after));
    !in_place
}

/// `stream` in a group of `delimiter` that is the macro's own, standing where `at` is. Rust lints none of what the
/// macro adds, and the hygiene of the group is the macro's: no fragment's, and no user's.
fn own(delimiter: Delimiter, stream: TokenStream, at: Span) -> TokenTree {
    let mut group = Group::new(delimiter, stream);
    group.set_span(Span::mixed_site().located_at(at));
    group.into()
}

/// What is in the parentheses `g`, as [`for_rust`] has it, if it is the bounds of a trait object: one type with a `+`
/// of its own, with `dyn` or without, `(dyn Debug + Send)`, and not a tuple, which has commas of its own. A `&` or a
/// pointer before it needs them, as `&dyn Debug + Send` is ambiguous, and Rust lints none of them as unnecessary there.
fn bounded(g: &Group) -> Option<Vec<TokenTree>> {
    if g.delimiter() != Delimiter::Parenthesis {
        return None;
    }
    let inside = contents(g);
    let parts = |sep| split_top(&inside, move |t| is_punct(t, sep)).count();
    (parts(',') == 1 && parts('+') > 1).then_some(inside)
}

/// `tokens` as the language reads them, each span numbered by its place in the spans beside
/// them. The spans come back with an error too, to place it: a bracket that nests too deep.
pub(crate) fn lex(tokens: &[TokenTree]) -> (Result<Vec<Tree>, lang::Error>, Vec<Span>) {
    let mut spans = vec![];
    let trees = run(tokens, None, 0, &mut spans);
    (trees, spans)
}

/// Whether the source has a space between what ends at `end` and what starts at `start`: it has, unless the
/// one ends on the line and column where the other starts. A macro's body and its call are two places, so a
/// fragment is taken to stand apart from the body round it, unless a line and a column of the one are those
/// of the other, which two files can have by chance.
fn apart(end: Span, start: Span) -> bool {
    end.line() != start.line() || end.column() != start.column()
}

/// `tokens` as the language reads them, `depth` brackets down, each span numbered by its place in `spans`.
/// `before` is where what comes before the first ends, a bracket that opens, if anything does.
fn run(
    tokens: &[TokenTree],
    mut before: Option<Span>,
    depth: usize,
    spans: &mut Vec<Span>,
) -> Result<Vec<Tree>, lang::Error> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut joint = false;
    for t in tokens {
        // `proc_macro` says that a punct is joint to the punct after it, which it touches, so the two have no
        // space between them, even where their spans lie in two places, as every span of a stream that `quote!`
        // writes does. A punct joint to a token that is no punct says nothing of it: it is the last of a
        // fragment, joint to what follows it at the call.
        let (first, last) = ends(t);
        let glued = joint && matches!(first, TokenTree::Punct(_));
        let space = !glued && before.is_some_and(|end| apart(end, first.span()));
        out.push(tree(t, space, depth, spans)?);
        before = Some(last.span().end());
        joint = matches!(&last, TokenTree::Punct(p) if p.spacing() == Spacing::Joint);
    }
    Ok(out)
}

/// `t` as the language reads it, with `space` before it, `depth` brackets down, each span numbered by its
/// place in `spans`.
fn tree(t: &TokenTree, space: bool, depth: usize, spans: &mut Vec<Span>) -> Result<Tree, lang::Error> {
    let span = lang::Span(spans.len() as u32);
    Ok(match t {
        TokenTree::Ident(i) => {
            spans.push(i.span());
            Tree::Ident { name: i.to_string(), space, span }
        }
        TokenTree::Punct(p) => {
            spans.push(p.span());
            Tree::Punct { ch: p.as_char(), space, span }
        }
        TokenTree::Literal(l) => {
            spans.push(l.span());
            Tree::Lit { text: l.to_string(), space, span }
        }
        TokenTree::Group(g) => {
            // A group with no brackets is where its first token is: its own span is that of the name of the
            // fragment in the macro that has it, which is not where the fragment was written.
            let none = g.delimiter() == Delimiter::None;
            spans.push(if none { ends(t).0.span() } else { g.span_open() });
            lang::nested(depth, span)?;
            let delim = match g.delimiter() {
                Delimiter::Parenthesis => Delim::Paren,
                Delimiter::Bracket => Delim::Bracket,
                Delimiter::Brace => Delim::Brace,
                Delimiter::None => Delim::None,
            };
            let inside = flatten(g.stream());
            let open = g.span_open().end();
            let trees = run(&inside, (!none).then_some(open), depth + 1, spans)?;
            let space_close = !none && apart(inside.last().map_or(open, |t| ends(t).1.span().end()), g.span_close());
            Tree::Group { delim, trees, space, space_close, span }
        }
    })
}

/// The macro an [`error`] calls.
const ERROR: &str = "::core::compile_error!";

/// `::core::compile_error! { message }`, reported where `span` is.
pub(crate) fn error(span: Span, message: &str) -> TokenStream {
    // Not by `parse`, which gives an error for text that is no tokens.
    let mut out: TokenStream = ERROR.parse().unwrap_or_default();
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, TokenTree::Literal(Literal::string(message)).into()))]);
    respan(out, span)
}

/// [`error`] under `cfg`, what [`carried`] keeps of an item's attributes: Rust says nothing of an item
/// it leaves out, and so does the error of one that does not compile.
pub(crate) fn error_under(cfg: &TokenStream, span: Span, message: &str) -> TokenStream {
    let mut out = cfg.clone();
    out.extend(error(span, message));
    out
}

/// What has Rust hold the path of `m`, a constant of a module of Rust's, to the value kernite took for it: an item
/// under `guard` that Rust works out, and that fails if the path gives a float of other bits, as it does where a name
/// in the path, `std` or `f32`, means another thing in the module. It fails at the place of the path, `at` its first
/// token. The path's own tokens keep the places the source has them, so that Rust says what it says of any path, at
/// the name that is wrong, if it finds no constant there, one of another type, or one that is unstable. Each path is an
/// item of its own: a statement is not worked out beside one that is an error.
pub(crate) fn takes(guard: TokenStream, m: &lang::Mention, at: &dyn std::ops::Fn(lang::Span) -> Span) -> TokenStream {
    let (float, bits) = match m.value {
        Some(Value::F32(x)) => ("f32", u64::from(x.to_bits())),
        Some(Value::F64(x)) => ("f64", x.to_bits()),
        // kernite takes a float alone for the path of a constant of a module of Rust's.
        _ => return TokenStream::new(),
    };
    let place = at(m.span);
    let path = (m.span.0..).zip(parse(&m.name)).map(|(i, mut t)| {
        t.set_span(at(lang::Span(i)));
        t
    });
    // The name of the constant, after the path of its module.
    let name = m.name.rsplit("::").next().unwrap_or_default();
    let message = format!(
        "`{}` is not Rust's constant, which kernite takes, with the bits {bits:#x}: a name in the path means another \
         thing in this module; write `::core::{float}::consts::{name}`",
        m.name
    );
    let paren = |inside: TokenStream| {
        let mut group = Group::new(Delimiter::Parenthesis, inside);
        group.set_span(place);
        TokenTree::Group(group)
    };
    let mut args = respan(parse(&format!("::core::primitive::{float}::to_bits")), place);
    args.extend([paren(path.collect())]);
    args.extend(respan(parse(&format!("== {bits:#x}, {message:?}")), place));
    let mut out = guard;
    out.extend(respan(parse("const _: () = ::core::assert!"), place));
    out.extend([paren(args)]);
    out.extend(respan(parse(";"), place));
    out
}

/// Whether `tokens` begin with the tokens of [`ERROR`], whatever follows them.
pub(crate) fn is_error(tokens: &[TokenTree]) -> bool {
    let path: Vec<String> = parse(ERROR).into_iter().map(|t| t.to_string()).collect();
    tokens.iter().take(path.len()).map(TokenTree::to_string).eq(path)
}

/// The crate that has `__kernels!` and `__Room`, which the Rust the macro writes calls by the crate's
/// name, `::kernite_macros`: a dependency of that name, not renamed in `Cargo.toml`.
pub(crate) const MACROS: &str = "kernite-macros";

/// Whether the crate being compiled has a dependency named `name`, `kernite-cpu`, `kernite-cuda` or
/// `kernite-macros`, which is how the Rust the macro writes names it. One renamed in `Cargo.toml` has
/// another name.
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

/// What a crate that has no dependency named `name` is told of `what`, which needs one.
pub(crate) fn needs_dependency(what: &str, name: &str) -> String {
    format!("{what} needs a dependency named `{name}`, which this crate does not have")
}

/// What stands for a type of the crate `name`'s in a crate that has no dependency named so, and what says at `span`
/// that `what` needs one, under `cfg`. The type is [`REFUSED`], a name Rust cannot find. The error is an item, for the
/// caller to write after the item the type is of. Rust then says nothing of the type, nor of the code that uses it.
pub(crate) fn missing(cfg: &TokenStream, span: Span, what: &str, name: &str) -> (TokenStream, TokenStream) {
    (placed(REFUSED, span), error_under(cfg, span, &needs_dependency(what, name)))
}

/// The type `ty`, a kernel's, as it is written.
pub(crate) fn written(ty: &[TokenTree]) -> TokenStream {
    ty.iter().cloned().collect()
}

/// The type `ty`, a kernel's, as the host names it: `f16` and `bf16`, which Rust has not, are
/// kernite-cuda's. Any other is as the kernel writes it, and means what it means in its module.
pub(crate) fn host(ty: &[TokenTree]) -> TokenStream {
    half(ty).unwrap_or_else(|| ty.iter().cloned().collect())
}

/// The type `to` that a kernel's pointer points to, as a type that is not behind a pointer has it: as written, but for
/// a trait object of more than one bound in parentheses, `(dyn Debug + Send)`, which it has without them. A pointer
/// needs them, as `*const dyn Debug + Send` is ambiguous, and Rust does not lint them there; an argument or an array's
/// element does not, and Rust would say there that they are unnecessary. Rust lints the parentheses of any other type
/// alike there and behind the pointer. A fragment in them that stays one type is in the macro's own parentheses, as
/// [`for_rust`] has it, for Rust reads a group the macro writes again as no group: `Fn() -> $t + Send`, with `$t` as
/// `dyn Debug + Sync`, would be `Fn() -> dyn Debug + Sync + Send`.
fn unparenthesized(to: &[TokenTree]) -> Vec<TokenTree> {
    match to {
        [TokenTree::Group(g)] => bounded(g).unwrap_or_else(|| to.to_vec()),
        _ => to.to_vec(),
    }
}

/// The type of a launcher's parameter for a kernel's pointer `ty` to `to`: `::kernite_cuda::Ptr<..>` of
/// [`unparenthesized`]. Rust says of a type that holds a part with no size, `(dyn Debug, u8)`, that it has none, at the
/// whole type, from its first token to its last, where [`asks`] says it too, and the same error twice at one place is
/// one. So the type begins and ends where the kernel's does: a first or a last token of the macro's would put the error
/// at the attribute. Both tokens of the `::` that begins it are the kernel's first, for Rust reads the two as one.
pub(crate) fn ptr_type(ty: &[TokenTree], to: &[TokenTree]) -> TokenStream {
    let mut out: Vec<TokenTree> =
        parse("::kernite_cuda::Ptr<").into_iter().chain(host(&unparenthesized(to))).chain(parse(">")).collect();
    out.iter_mut().take(2).for_each(|t| t.set_span(span_of(ty.first())));
    if let Some(close) = out.last_mut() {
        close.set_span(span_of(ty.last()));
    }
    out.into_iter().collect()
}

/// `ty` as kernite-cuda names it, if it is `f16` or `bf16`, written raw or not, which Rust has not.
pub(crate) fn half(ty: &[TokenTree]) -> Option<TokenStream> {
    match ty {
        [t] if named(t, "f16") || named(t, "bf16") => Some(parse(&primitive(lang::unraw(&t.to_string())))),
        _ => None,
    }
}

/// The number `ty` is, if it is one by its name or its path, as the Rust the macros write for a kernel
/// names it: the primitive by its path, `::core::primitive::u64`, located where `ty` begins. A module may
/// give the name another type, and Rust says so, once, at the type, where [`asks`] asks it. What the macros write
/// besides, an instance, a dispatcher or a launcher, takes the number and not the name, so it says nothing of the type:
/// a value of one with no size, or that a launcher cannot pass on, would be an error at the module's attribute, where
/// nothing can be mended.
pub(crate) fn number_type(ty: &[TokenTree]) -> Option<TokenStream> {
    number_name(ty).map(|name| placed(&primitive(name), span_of(ty.first())))
}

/// `ty`, a type a kernel takes, as the Rust the macros write has it: [`number_type`] if it is a
/// number's name, as written if it is a pointer or an array of numbers by reference, else
/// [`refused`], for a kernel takes no other type. It is for the signatures [`signature`] writes
/// and an x86 kernel's instances and launcher. A GPU launcher's parameters are
/// [`number_type`]'s, [`half`]'s or [`refused`].
pub(crate) fn value(ty: &[TokenTree]) -> TokenStream {
    if pointer(ty).is_some() || array_ref(ty).is_some_and(|(_, elem, _)| number_name(&elem).is_some()) {
        return ty.iter().cloned().collect();
    }
    number_type(ty).unwrap_or_else(|| refused(ty))
}

/// `ty`, a type a kernel gives back, as the Rust the macros write has it: as written if it is `()` or
/// `!`, which a kernel gives back and does not take, else [`value`].
pub(crate) fn given_back(ty: &[TokenTree]) -> TokenStream {
    if unit(ty) || matches!(ty, [t] if is_punct(t, '!')) {
        return ty.iter().cloned().collect();
    }
    value(ty)
}

/// Whether `ty` is `()`.
fn unit(ty: &[TokenTree]) -> bool {
    matches!(ty, [TokenTree::Group(g)] if g.delimiter() == Delimiter::Parenthesis && g.stream().is_empty())
}

/// What a launcher of `f` gives back: `launched`, the `Result` of the launch, but for a kernel that gives back
/// a type. kernite refuses that, at the type, and the launcher never launches; it gives back [`refused`]'s
/// name, with the type as its argument and `launched` beside it. Rust reads the type in the launcher's
/// signature as it reads a plain kernel's, and says what it says of any function's result, once, at the type;
/// an import that only the type names is used. Rust says nothing of a call of the launcher, which a crate
/// writes as it writes any: it takes the name for a type it has reported. A `&` in a launcher's result takes a
/// lifetime from what the launcher takes, a cooperative kernel's grid or a GPU kernel's `self`, where a plain
/// kernel's has none to take.
pub(crate) fn launcher_result(f: &Fn, top: &[TokenTree], launched: TokenStream) -> TokenStream {
    match f.ret.as_ref().map(|r| within(top, &r.tokens)) {
        Some(ty) if !unit(ty) => refused_beside(ty, launched),
        _ => launched,
    }
}

/// What shows Rust the types in the signature of `f`, a helper or an `extern "ptx-func"` function, of which the macros
/// write nothing else: a helper is compiled where a kernel calls it, and the function is a `.func` of the GPU's
/// module. Each compile-time argument's type, each parameter's and the result's is written as [`refused`] writes the
/// type of a kernel that does not compile, `let _: ..;`, under `guard`. So an import that only they name is used, and
/// Rust says of them only that a lifetime is not declared, that a module is where a type is written, and what a macro
/// writes. Most of the types are the language's, `gpr` or `local<[u32; 4], 16>`, and Rust cannot find them: it is
/// silent about a name it cannot find only where a macro has written an error as an item of the module, and `guard`
/// holds only there; or the lines are `__Room`'s to write, beside an error it writes, and `guard` is none.
pub(crate) fn seen(f: &Fn, top: &[TokenTree], guard: &TokenStream) -> TokenStream {
    let written = written_generics(f, top);
    let inside = written.get(1..written.len().saturating_sub(1)).unwrap_or_default();
    let generics = split_top(inside, |t| is_punct(t, ','))
        .filter(|g| !g.is_empty())
        .map(|g| g.get(3..).unwrap_or_default().to_vec());
    let params = written_params(f, top).into_iter().map(|p| p.ty);
    let result = f.ret.iter().map(|r| within(top, &r.tokens).to_vec());
    let mut out = TokenStream::new();
    for ty in generics.chain(params).chain(result) {
        out.extend(guard.clone());
        out.extend(parse("let _:"));
        out.extend(refused(&ty));
        out.extend(parse(";"));
    }
    out
}

/// The name the macros write for a type kernite refuses: one Rust cannot find.
const REFUSED: &str = "self::__kernite_refused";

/// The type `ty`, which kernite refuses, as the Rust the macros write has it: a name Rust cannot find,
/// located where `ty` begins, that takes `ty` as its argument, but for what [`unstable`] says. Written as it
/// is, Rust would say of `ty`, beside kernite's error, what it says wherever the type is named: that an
/// `f128` is unstable, say, or that a `[u64]` has no size. Where a macro has failed in a module, as
/// kernite's error is, Rust says nothing of a name it cannot find, nor of what the name takes, and takes a
/// type so named for one it has reported: so it says nothing of a call against it. It still finds the names
/// in the argument, so an import that only `ty` uses is used. And it says of the argument what it says of
/// any function's signature: that a result's `&` has no lifetime to take, that a lifetime is not declared,
/// and that a result is `_`.
pub(crate) fn refused(ty: &[TokenTree]) -> TokenStream {
    refused_beside(ty, TokenStream::new())
}

/// [`refused`], with `beside` among the arguments if it is not empty: Rust finds the names in it too.
fn refused_beside(ty: &[TokenTree], beside: TokenStream) -> TokenStream {
    let at = span_of(ty.first());
    let mut out = placed(&format!("{REFUSED}<"), at);
    out.extend(unstable(ty));
    if !beside.is_empty() {
        out.extend(placed(",", at));
        out.extend(beside);
    }
    out.extend(placed(">", at));
    out
}

/// `tokens`, a type kernite refuses, as written, but for each `f16` and `f128` where [`walk`] says a type is written:
/// wherever Rust finds one it says that the type is unstable, even in the argument of a name it cannot find. An `f16`
/// or an `f128` is written `self::f16` or `self::f128`, with the `r#` if it has one, which Rust reads as the name
/// without it, for a module may give either name a type: Rust finds it there, and its import is used, and where there
/// is none Rust finds nothing and says nothing. Where the module has a module so named, Rust says that it is no type,
/// as it does of any function's signature. A later segment of a path, `core::primitive::f16`, is a name in the path
/// before it, and stays as written. An `f16` or an `f128` that kernite does not see as a type, for a macro writes it,
/// `half!()`, or it is in an expression of the type, as in an array's length, `[u8; size_of::<f16>()]`, Rust still says
/// is unstable.
fn unstable(tokens: &[TokenTree]) -> TokenStream {
    let float = |t: &TokenTree| named(t, "f16") || named(t, "f128");
    walk(tokens, float, &mut |t, place| match place {
        Place::Type => Some(placed(&format!("self::{t}"), t.span())),
        Place::Length => None,
    })
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

/// The tokens `s` writes, each of the span [`SITE`] says. Text that is no tokens is an [`error`], kernite's own
/// mistake: kernite-macros writes none such.
pub(crate) fn parse(s: &str) -> TokenStream {
    let tokens: TokenStream = s.parse().unwrap_or_else(|why| {
        error(
            Span::call_site(),
            &format!("kernite-macros wrote `{s}`, which is no tokens: {why}; this is kernite's own mistake"),
        )
    });
    match SITE.get() {
        Some(span) => respan(tokens, span),
        None => tokens,
    }
}

/// The tokens `s` writes, as [`parse`] gives them, but for each name after a `$`: a local the
/// macro binds for itself, of the span `Span::mixed_site()`. No local the user wrote finds it, nor does it find theirs,
/// so a parameter of a kernel may be named as the macro's launcher names its own. A const or a static in scope is no
/// local: Rust reads a binding of a const's name as that const, and refuses one of a static's. So each name is one of
/// kernite's, `__kernite_grid` say, and a const or a static of the user's named `grid` is no matter to it. Where the
/// Rust never `runs`, each is `_`, for a compile-time argument may be named so.
pub(crate) fn hygienic(s: &str, runs: bool) -> TokenStream {
    fn own(tokens: TokenStream, runs: bool) -> TokenStream {
        let mut tokens = tokens.into_iter();
        let mut out = TokenStream::new();
        while let Some(t) = tokens.next() {
            out.extend([match t {
                TokenTree::Punct(p)
                    if p.as_char() == '$'
                        && let Some(name) = tokens.next() =>
                {
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

/// The raw identifier `name`, located at `at` and resolved as the macro's own tokens are, so that Rust finds the item
/// the macro wrote of that name by it, at `at`.
pub(crate) fn ident_at(name: &str, at: Span) -> TokenStream {
    respan(parse(&format!("r#{name}")), at.resolved_at(site()))
}

/// `tokens`, each of the span `span`.
pub(crate) fn respan(tokens: TokenStream, span: Span) -> TokenStream {
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
