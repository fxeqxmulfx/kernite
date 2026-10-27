//! Rust's tokens: the language's trees made of them, the parts of a module or a function as
//! written, and tokens the macro makes, each at the span that says where Rust resolves it.

use std::cell::Cell;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

use kgen::lang::{self, Delim, Fn, Tree, Value};

thread_local! {
    /// The span of each token the macro writes, which says where Rust resolves its names: none
    /// for the macro's own call site; or, for a module given consts, the path
    /// `#[kernels(consts = path)]` names them by, where the module is, since `__kernels!` is
    /// called from inside the consts' macro.
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

/// `const { match .. }` on `f`'s compile-time arguments, by their names as written, `r#in` as
/// well: what `arms` gives for each instance, by its arguments. Another is no instance, and Rust
/// refuses to compile the call.
pub(crate) fn pick(f: &Fn, arms: &[(Vec<Value>, String)], top: &[TokenTree]) -> String {
    let tuple = |xs: Vec<String>| if xs.len() == 1 { xs.join("") } else { format!("({})", xs.join(", ")) };
    let on = tuple(const_names(&top[f.head.end + 2..f.params_at]).map(TokenTree::to_string).collect());
    let mut out = format!("const {{ match {on} {{ ");
    let mut listed = vec![];
    for (args, then) in arms {
        out += &format!("{} => {then}, ", tuple(args.iter().map(Value::to_string).collect()));
        listed.push(format!("`{}`", lang::instance(f, args).unwrap_or_default()));
    }
    let message = format!("`{}` is compiled for {} alone, as its `#[instances(..)]` lists", f.name, listed.join(", "));
    out + &format!("_ => ::core::panic!({message:?}) }} }}")
}

/// The name after each `const` of `tokens`, as written, `r#type` as well: a const's, or those of
/// a function's compile-time arguments.
pub(crate) fn const_names(tokens: &[TokenTree]) -> impl Iterator<Item = &TokenTree> {
    tokens.windows(2).filter(|w| is_ident(&w[0], "const")).map(|w| &w[1])
}

/// `f`'s parameters as written, each its tokens, `x: *mut f32`, and the place of its colon.
pub(crate) fn written_params(f: &Fn, top: &[TokenTree]) -> Vec<(Vec<TokenTree>, usize)> {
    let TokenTree::Group(group) = &top[f.params_at] else { unreachable!("parameters are in parentheses") };
    let tokens = flatten(group.stream());
    let params = tokens.split(|t| is_punct(t, ',')).filter(|p| !p.is_empty());
    params.map(|p| (p.to_vec(), p.iter().position(|t| is_punct(t, ':')).expect("a parameter has a type"))).collect()
}

/// `f` with a body that never runs, for a kernel that does not compile.
pub(crate) fn stub(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out = attrs(f, top);
    out.extend(parse("#[allow(unused_variables)]"));
    out.extend(signature(f, top));
    out.extend(parse("{ ::core::unreachable!() }"));
    out
}

/// `f`'s attributes as written, but `#[instances(..)]` and `#[cooperative]`, which Rust does not
/// know.
pub(crate) fn attrs(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let instances = f.instances.as_ref().map_or(0..0, |i| i.tokens.clone());
    let cooperative = f.cooperative.as_ref().map_or(0..0, |c| c.tokens.clone());
    f.attrs.clone().filter(|i| !instances.contains(i) && !cooperative.contains(i)).map(|i| top[i].clone()).collect()
}

/// `f`'s signature as written, from its visibility to its result.
pub(crate) fn signature(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let end = f.ret.as_ref().map_or(f.params_at + 1, |r| r.tokens.end);
    top[f.head.start..end].iter().cloned().collect()
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

/// `t` as the language reads it, each span numbered by its place in `spans`.
pub(crate) fn tree(t: &TokenTree, spans: &mut Vec<Span>) -> Tree {
    let span = lang::Span(spans.len() as u32);
    match t {
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
            let delim = match g.delimiter() {
                Delimiter::Parenthesis => Delim::Paren,
                Delimiter::Bracket => Delim::Bracket,
                Delimiter::Brace | Delimiter::None => Delim::Brace,
            };
            Tree::Group { delim, trees: flatten(g.stream()).iter().map(|t| tree(t, spans)).collect(), span }
        }
    }
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

/// The type `ty` as the host names it: `f16` and `bf16`, which Rust has not, are kgen-cuda's.
pub(crate) fn host(ty: &[TokenTree]) -> TokenStream {
    match ty {
        [t] if is_ident(t, "f16") || is_ident(t, "bf16") => parse(&format!("::kgen_cuda::{t}")),
        _ => ty.iter().cloned().collect(),
    }
}

/// The tokens `s` writes, each of the span [`SITE`] says.
pub(crate) fn parse(s: &str) -> TokenStream {
    let tokens: TokenStream = s.parse().expect("kgen-macros writes tokens that parse");
    match SITE.get() {
        Some(span) => respan(tokens, span),
        None => tokens,
    }
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
