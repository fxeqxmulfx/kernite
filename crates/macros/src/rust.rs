//! Rust's tokens: the language's trees made of them, the parts of a module or a function as
//! written, and tokens the macro makes, each at the span that says where Rust resolves it.

use std::cell::Cell;

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
/// kernite cannot tell, as for a macro that writes the token into a type, it takes it for no type's.
fn walk(
    mut tokens: &[TokenTree],
    leaf: fn(&TokenTree) -> bool,
    found: &mut dyn FnMut(&TokenTree, Place) -> Option<TokenStream>,
) -> TokenStream {
    let all = tokens;
    let mut out = TokenStream::new();
    while let [t, rest @ ..] = tokens {
        let before = &all[..all.len() - tokens.len()];
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
                let mut inner = Group::new(g.delimiter(), walk(&flatten(g.stream()), leaf, found));
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
    f.ret.as_ref().is_some_and(|r| placeholder(&top[r.tokens.clone()]))
}

/// The name of the trait the macros write into a body to ask Rust whether a pointer has a size.
/// Every type implements it, so what asks is the coercion of a pointer to a pointer to the trait,
/// which Rust makes of a pointer to a type with a size alone.
const THIN: &str = "__KerniteThin";

/// What Rust is asked of a type of a kernel's, and where the type is.
#[derive(Clone, Copy)]
enum Ask {
    /// Where the type a pointer of one word, `*const T` or `*mut T`, points to begins and where it
    /// ends: whether `T` has a size.
    Pointer(Span, Span),
    /// A number by its name, and where the name is: whether it is that number, the primitive
    /// `::core::primitive::u64`, and not another type the module gives the name. Rust is asked of
    /// the type alone, never of a value of it.
    Number(&'static str, Span),
}

/// The number `ty` is, and where its name is, if `ty` is one by its name: `u8` to `u64`, `i8` to
/// `i64`, `usize`, `isize`, `f32`, `f64` or `bool`, which the language reads by their names, written raw or not.
fn number_name(ty: &[TokenTree]) -> Option<(&'static str, Span)> {
    let [name] = ty else { return None };
    let written = name.to_string();
    let text = lang::unraw(&written);
    let known = lang::IntTy::parse(text).map(lang::IntTy::name);
    known.or_else(|| ["bool", "f32", "f64"].into_iter().find(|n| *n == text)).map(|n| (n, name.span()))
}

/// For each of `f`'s parameters, and last for its result: what Rust is asked of its type, if it is
/// asked. It is asked of a number by its name, as [`number_name`] reads it, and not of `f16` and
/// `bf16`, which Rust has not; and of a pointer of one word, `*const T` or `*mut T`. The language
/// says which: a pointer to a slice, a `str` or a `dyn` has two words, and is its error alone,
/// which Rust is not asked of, for it would say the same twice. Nor is it asked of a pointer to a
/// type with a `_` in it, `*const _` or `*const [_; 3]`, which is Rust's error alone, at the `_`:
/// the type to ask of is one Rust cannot infer, and it would say that too.
fn asked(f: &Fn, top: &[TokenTree]) -> Vec<Option<Ask>> {
    let ask = |ty: &[TokenTree]| {
        if let Some((name, at)) = number_name(ty) {
            return Some(Ask::Number(name, at));
        }
        let (_, to) = pointer(ty)?;
        let one = lex(ty).0.is_ok_and(|trees| lang::pointer(&lang::read_type(&trees)) == Ok(true));
        (one && !placeholder(to)).then(|| Ask::Pointer(to[0].span(), to[to.len() - 1].span()))
    };
    let params = written_params(f, top).into_iter().map(|(param, colon)| ask(&param[colon + 1..]));
    params.chain([f.ret.as_ref().and_then(|r| ask(&top[r.tokens.clone()]))]).collect()
}

/// What asks Rust of each type `f` takes, in a body that has the parameters by their names:
/// [`numbers`] asks of the numbers and [`pointers`] of the pointers. A result that is a pointer is
/// asked of where it is given back. Nothing if `f` has none to ask of.
pub(crate) fn checks(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let names: Vec<TokenStream> =
        written_params(f, top).iter().map(|(param, colon)| param[colon - 1].clone().into()).collect();
    let asked = asked(f, top);
    let mut out = numbers(&asked);
    out.extend(pointers(&asked, &names));
    out
}

/// What asks Rust of each type `f` takes or gives back, in a body that has none of them: [`numbers`]
/// asks of the numbers, and one closure, which Rust checks and never runs, takes the pointers as the
/// kernel writes them, the type each points to as `pointee` names it, for [`pointers`] to ask of
/// them. A closure's parameter is Rust's to say an error in, at the type, as a function's is, and the
/// same error twice at one place is one. Nothing if `f` has none to ask of.
pub(crate) fn asks(f: &Fn, top: &[TokenTree], pointee: fn(&[TokenTree]) -> TokenStream) -> TokenStream {
    let asked = asked(f, top);
    let mut out = numbers(&asked);
    let result = f.ret.as_ref().map(|r| top[r.tokens.clone()].to_vec());
    let types = written_params(f, top).into_iter().map(|(param, colon)| param[colon + 1..].to_vec()).chain(result);
    let (mut params, mut values) = (TokenStream::new(), vec![]);
    for (i, (ty, ask)) in types.zip(&asked).enumerate() {
        let local = TokenTree::Ident(Ident::new(&format!("__{i}"), Span::mixed_site()));
        values.push(match ask {
            Some(Ask::Pointer(..)) => {
                let (kind, to) = pointer(&ty).expect("a pointer is asked of");
                params.extend([local.clone()]);
                params.extend(parse(&format!(":*{kind}")));
                params.extend(pointee(to));
                params.extend(parse(","));
                local.into()
            }
            _ => TokenStream::new(),
        });
    }
    let questions = pointers(&asked, &values);
    if !questions.is_empty() {
        out.extend(parse("let _ = |"));
        out.extend(params);
        out.extend(parse("|"));
        out.extend([TokenTree::Group(Group::new(Delimiter::Brace, questions))]);
        out.extend(parse(";"));
    }
    out
}

/// The block of a `const` operand of a naked kernel's own `naked_asm!`, which asks Rust of each type
/// the kernel takes or gives back, as [`asks`] says. A naked function has no body to write the
/// questions in, and no name of the kernel's finds the closure's locals. Nothing if the kernel has
/// none to ask of.
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

/// What asks Rust of each pointer of one word `asked` has, by the values that `values` gives of them:
/// a trait that every type implements, and a coercion of each pointer to a pointer to the trait,
/// which Rust makes of a pointer to a type with a size alone: of one to a type with no size it says
/// that the size cannot be known. A coercion is not a call, so Rust says nothing of a type it
/// refuses by itself, again: a call says it at the call, once more. Nothing if there is nothing to
/// ask.
fn pointers(asked: &[Option<Ask>], values: &[TokenStream]) -> TokenStream {
    let mut questions = TokenStream::new();
    for (value, ask) in values.iter().zip(asked) {
        if let Some(Ask::Pointer(to, _)) = *ask {
            questions.extend(thin(value.clone(), to));
        }
    }
    if questions.is_empty() {
        return questions;
    }
    // The trait and the coercions are of one span, for Rust to find the one by the other.
    let mut out = respan(parse(&format!("trait {THIN} {{}} impl<T: ?Sized> {THIN} for T {{}}")), Span::mixed_site());
    out.extend(questions);
    out
}

/// The tokens `s` writes, each located at `at` and of the macro's hygiene.
fn placed(s: &str, at: Span) -> TokenStream {
    respan(parse(s), at.resolved_at(Span::mixed_site()))
}

/// `let _: *const (dyn __KerniteThin + '_) = value;` with each token located at `to`, where the type
/// `value` points to begins, for Rust to say there that the type has no size. `value` is resolved as
/// it is written; the trait is the macro's, which no name of the kernel's finds.
fn thin(value: TokenStream, to: Span) -> TokenStream {
    let mut out = placed(&format!("let _: *const (dyn {THIN} + '_) ="), to);
    out.extend(value.into_iter().map(|mut t| {
        t.set_span(to.resolved_at(t.span()));
        t
    }));
    out.extend(placed(";", to));
    out
}

/// `unsafe { call }`, which gives back the result of an instance of `f` as `f` gives it back: `call`
/// is the call. A pointer is cast back, `.cast()`, for Rust to say there that the type it points to
/// has no size. A cast says an error in the type again, as any generic does, at the span of the cast:
/// so the call and the cast are located across the type, the span Rust says the error at in the
/// signature when it says the whole type, and then the two are one error. For a type that holds the
/// part Rust refuses, `(u8, [CStr; 3])`, Rust says the part in the signature, `[CStr; 3]`, and the
/// cast says the whole type: the two are two. Rust is asked of a number by [`checks`].
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
pub(crate) fn stub(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out = attrs(f, top);
    out.extend(signature(f, top));
    let mut body = asks(f, top, written);
    if gives_placeholder(f, top) {
        body.extend(parse(&format!("let _ = self::{} as *const ();", top[f.head.end + 1])));
    }
    body.extend(parse(&never(f, top)));
    body.extend(parse("::core::unreachable!()"));
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

/// Whether `t` is the name `name`, written raw or not: Rust reads `r#name` as `name`, an attribute's, `r#cfg`, or a
/// type's, `r#u64`. A keyword is asked as written, by [`is_ident`]: `r#let` is a name, and not `let`.
pub(crate) fn named(t: &TokenTree, name: &str) -> bool {
    matches!(t, TokenTree::Ident(i) if lang::unraw(&i.to_string()) == name)
}

/// `f`'s signature as written, from its visibility to its result, but for its parameters, as
/// [`value_params`] has them, and its result, which is [`given_back`]'s.
pub(crate) fn signature(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out: TokenStream = top[f.head.start..f.params_at].iter().cloned().collect();
    let mut params = Group::new(Delimiter::Parenthesis, value_params(f, top));
    params.set_span(top[f.params_at].span());
    out.extend([TokenTree::Group(params)]);
    if let Some(r) = &f.ret {
        // The arrow, then the result.
        out.extend(top[f.params_at + 1..r.tokens.start].iter().cloned());
        out.extend(given_back(&top[r.tokens.clone()]));
    }
    out
}

/// `f`'s parameters as [`written_params`] has them, a comma after each, but for each type, which is
/// [`value`]'s.
pub(crate) fn value_params(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out = TokenStream::new();
    for (param, colon) in written_params(f, top) {
        out.extend(param[..=colon].iter().cloned());
        out.extend(value(&param[colon + 1..]));
        out.extend(parse(","));
    }
    out
}

/// The tokens of `stream`, with the invisible groups a macro's fragments arrive in opened, but for those that
/// [`stays`] says stay one operand or one type: Rust reads `$n * 2` with `$n` as `1 + 2` as `(1 + 2) * 2`, and
/// opened it would be `1 + 2 * 2`. A group that stays is as it was: what is in it is opened where it is read.
pub(crate) fn flatten(stream: TokenStream) -> Vec<TokenTree> {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    let mut out = vec![];
    for (i, t) in tokens.iter().enumerate() {
        match t {
            TokenTree::Group(g) if g.delimiter() == Delimiter::None => {
                let inside = flatten(g.stream());
                if stays(&inside, &tokens[..i], &tokens[i + 1..]) {
                    out.push(t.clone());
                } else {
                    out.extend(inside);
                }
            }
            t => out.push(t.clone()),
        }
    }
    out
}

/// Whether an invisible group that holds `inside`, between `before` and `after`, stays one. An operator in it
/// could be taken apart by one next to it, as `* 2` takes `1 + 2` apart: so it stays if it holds an operator
/// outside any bracket, `1 + 2`, `-1`, `a as u8` or `0..4`, and one is next to it. Before it is one of
/// `+ - * / % ^ & | < >` and `!`, or `==`, `!=`, `<=` or `>=`; after it, one of `+ - * / % ^ & | < >`, or `==`,
/// `!=`, `.` or `as`. The `>` of an arrow, `->`, is a result's, and no operator's, and a lone `=` is a `let`'s or an
/// assignment's. A group that holds `a.b`, `f(x)` or `a::b`, or that no operator is next to, is read as it would be
/// without the group, and is opened.
fn stays(inside: &[TokenTree], before: &[TokenTree], after: &[TokenTree]) -> bool {
    let is_any = |t: &TokenTree, chars: &str| chars.chars().any(|c| is_punct(t, c));
    let loose = inside.iter().any(|t| is_any(t, "+-*/%^&|<>!=") || is_ident(t, "as"))
        || inside.windows(2).any(|dots| is_punct(&dots[0], '.') && is_punct(&dots[1], '.'));
    // `==`, `!=`, `<=` and `>=` are an `=` that an operator is joint to.
    let compares = |pair: &[TokenTree]| match pair {
        [op, eq] => {
            is_punct(eq, '=')
                && is_any(op, "=!<>")
                && matches!(op, TokenTree::Punct(p) if p.spacing() == Spacing::Joint)
        }
        _ => false,
    };
    let arrow = matches!(before, [.., dash, head] if is_punct(dash, '-') && is_punct(head, '>'));
    let left = before.last().is_some_and(|t| is_any(t, "+-*/%^&|<>!")) && !arrow
        || before.len() >= 2 && compares(&before[before.len() - 2..]);
    let right = after.first().is_some_and(|t| is_any(t, "+-*/%^&|<>.") || is_ident(t, "as"))
        || after.len() >= 2 && compares(&after[..2]);
    loose && (left || right)
}

/// The first and the last token of `t`: itself, but for a group that [`flatten`] keeps, where they are those
/// of what is in it.
fn ends(t: &TokenTree) -> (TokenTree, TokenTree) {
    match t {
        TokenTree::Group(g) if g.delimiter() == Delimiter::None => {
            let inside = flatten(g.stream());
            (ends(&inside[0]).0, ends(&inside[inside.len() - 1]).1)
        }
        t => (t.clone(), t.clone()),
    }
}

/// `tokens`, as [`flatten`] gives them, for Rust: a group that stays one is written in parentheses. Rust keeps a
/// fragment's group with no brackets as one operand only where a macro hands it on as it came; a macro that takes
/// the tokens apart, as [`flatten`] does, writes the group again, and Rust reads it as no group at all. The
/// parentheses are the macro's own, as Rust lints none of what the macro adds, and would lint a type in them,
/// `Foo<(Vec<u8>)>`; and they stand where the fragment starts, for Rust to say an error of the whole fragment
/// there. A bracket among the tokens is handed on as it came, with each fragment in it one operand.
pub(crate) fn for_rust(tokens: Vec<TokenTree>) -> Vec<TokenTree> {
    tokens
        .into_iter()
        .map(|t| match t {
            TokenTree::Group(ref g) if g.delimiter() == Delimiter::None => {
                let mut group = Group::new(Delimiter::Parenthesis, g.stream());
                group.set_span(Span::mixed_site().located_at(ends(&t).0.span()));
                group.into()
            }
            t => t,
        })
        .collect()
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
    let mut out = parse(ERROR);
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

/// `ty` as kernite-cuda names it, if it is `f16` or `bf16`, written raw or not, which Rust has not.
pub(crate) fn half(ty: &[TokenTree]) -> Option<TokenStream> {
    match ty {
        [t] if named(t, "f16") || named(t, "bf16") => Some(parse(&primitive(lang::unraw(&t.to_string())))),
        _ => None,
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

/// `ty`, a type a kernel takes, as the Rust the macros write has it: [`number_type`] if it is a
/// number's name, as written if it is a pointer, else [`refused`], for a kernel takes no other type.
/// It is for the signatures [`signature`] writes and an x86 kernel's instances and launcher. A GPU
/// launcher's parameters are [`number_type`]'s, [`half`]'s or [`refused`].
pub(crate) fn value(ty: &[TokenTree]) -> TokenStream {
    if pointer(ty).is_some() {
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
/// writes as it writes any: it takes the name for a type it has reported. A GPU launcher is a method, so a `&`
/// in its result takes the lifetime of `self`, where a plain kernel's has none to take.
pub(crate) fn launcher_result(f: &Fn, top: &[TokenTree], launched: TokenStream) -> TokenStream {
    match f.ret.as_ref().map(|r| &top[r.tokens.clone()]) {
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
    let generics = written_generics(f, top)
        .windows(4)
        .filter(|w| is_ident(&w[0], "const") && is_punct(&w[2], ':'))
        .map(|w| vec![w[3].clone()]);
    let params = written_params(f, top).into_iter().map(|(param, colon)| param[colon + 1..].to_vec());
    let result = f.ret.iter().map(|r| top[r.tokens.clone()].to_vec());
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
    let at = ty[0].span();
    let mut out = placed(&format!("{REFUSED}<"), at);
    out.extend(unstable(ty));
    if !beside.is_empty() {
        out.extend(placed(",", at));
        out.extend(beside);
    }
    out.extend(placed(">", at));
    out
}

/// `tokens`, a type kernite refuses, as written, but for each `f16`, `f128` and `!` where [`walk`] says a type is
/// written: wherever Rust finds one it says that the type is unstable, even in the argument of a name it cannot
/// find. A `!` is the name [`refused`] writes. An `f16` or an `f128` is written `self::f16` or `self::f128`, with the
/// `r#` if it has one, which Rust reads as the name without it, for a module may give either name a type: Rust finds
/// it there, and its import is used, and where there is none Rust finds nothing and says nothing. Where the module
/// has a module so named, Rust says that it is no type, as it does of any function's signature. A later segment of
/// a path, `core::primitive::f16`, is a name in the path before it, and stays as written. An `f16` or an `f128` that
/// kernite does not see as a type, for a macro writes it, `half!()`, or it is in an expression of the type, as in an
/// array's length, `[u8; size_of::<f16>()]`, Rust still says is unstable.
fn unstable(tokens: &[TokenTree]) -> TokenStream {
    let float = |t: &TokenTree| named(t, "f16") || named(t, "f128") || is_punct(t, '!');
    walk(tokens, float, &mut |t, place| match place {
        Place::Type if is_punct(t, '!') => Some(placed(REFUSED, t.span())),
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
