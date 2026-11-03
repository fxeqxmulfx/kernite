//! Rust's tokens: the language's trees made of them, the parts of a module or a function as
//! written, and tokens the macro makes, each at the span that says where Rust resolves it.

use std::cell::Cell;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

use kernite::lang::{self, Delim, Fn, Tree, Value};

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
    let names: Vec<String> = const_names(&top[f.head.end + 2..f.params_at]).map(TokenTree::to_string).collect();
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

/// The name after each `const` of `tokens`, as written, `r#type` as well: a const's, or those of
/// a function's compile-time arguments.
pub(crate) fn const_names(tokens: &[TokenTree]) -> impl Iterator<Item = &TokenTree> {
    tokens.windows(2).filter(|w| is_ident(&w[0], "const")).map(|w| &w[1])
}

/// An error at each of `f`'s compile-time arguments named as one of `own`, the locals the Rust of
/// `method` binds for itself. A compile-time argument is no local, and no span of stable Rust
/// keeps it apart from one of its name, as it does a parameter's.
pub(crate) fn clashes(f: &Fn, own: &[&str], method: &str, top: &[TokenTree]) -> TokenStream {
    let mut errors = TokenStream::new();
    for name in const_names(&top[f.head.end + 2..f.params_at]) {
        let written = name.to_string();
        let name_of = written.trim_start_matches("r#");
        if own.contains(&name_of) {
            errors.extend(error(name.span(), &format!("`{name_of}` names a parameter of {method} as well")));
        }
    }
    errors
}

/// `f`'s parameters as written, each its tokens, `x: *mut f32`, and the place of its colon. A comma
/// ends a parameter unless it is among the arguments of a type's, `x: *mut Pair<u8, 2>`.
pub(crate) fn written_params(f: &Fn, top: &[TokenTree]) -> Vec<(Vec<TokenTree>, usize)> {
    let TokenTree::Group(group) = &top[f.params_at] else { unreachable!("parameters are in parentheses") };
    let tokens = flatten(group.stream());
    let mut open = 0usize;
    let params = tokens
        .split(|t| {
            if is_punct(t, '<') {
                open += 1;
            } else if is_punct(t, '>') {
                // The `>` of an arrow, `->`, has no `<` to close.
                open = open.saturating_sub(1);
            }
            open == 0 && is_punct(t, ',')
        })
        .filter(|p| !p.is_empty());
    params.map(|p| (p.to_vec(), p.iter().position(|t| is_punct(t, ':')).expect("a parameter has a type"))).collect()
}

/// `f` with a body that never runs, for a kernel that does not compile.
pub(crate) fn stub(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out = attrs(f, top);
    out.extend(signature(f, top));
    out.extend(parse(&format!("{{ {} }}", never(f, top))));
    out
}

/// What a function of `f`'s parameters does when it never runs, for a kernel that does not
/// compile: it uses each parameter, by its name as written, which Rust would find unused.
pub(crate) fn never(f: &Fn, top: &[TokenTree]) -> String {
    let uses: String =
        written_params(f, top).iter().map(|(param, colon)| format!("let _ = {};", param[colon - 1])).collect();
    uses + "::core::unreachable!()"
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

/// The type `ty` as the host names it: `f16` and `bf16`, which Rust has not, are kernite-cuda's.
pub(crate) fn host(ty: &[TokenTree]) -> TokenStream {
    match ty {
        [t] if is_ident(t, "f16") || is_ident(t, "bf16") => parse(&format!("::kernite_cuda::{t}")),
        _ => ty.iter().cloned().collect(),
    }
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
