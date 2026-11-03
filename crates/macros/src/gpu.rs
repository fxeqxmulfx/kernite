//! What GPU kernels become in Rust: `PTX`, the module they are, and `Kernels`, which loads it
//! on a GPU, with a method to launch each kernel and one to give each static's memory.

use proc_macro::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};

use kernite::DeviceStatic;
use kernite::lang::{self, Fn, Value};

use crate::rust::{
    attrs, clashes, error, flatten, host, hygienic, is_ident, is_punct, never, parse, pick, site, visibility,
    written_params,
};

/// Where the module's path goes in its PTX and in its kernels' names. A macro is not told
/// the module it is in, so `kernite_cuda::__named!` writes `module_path!()` there once Rust
/// knows it, as C++ mangles a namespace's names.
pub(crate) const HOLE: &str = "\0";

/// What a kernel's instance is: its compile-time arguments, none for a kernel that takes
/// none, and its index among the module's kernels.
type Launched = (Vec<Value>, usize);

/// `PTX`, the module the GPU kernels are, and `Kernels`, which loads it on a GPU and has a
/// method to launch each kernel by its name: of each instance, for a kernel of compile-time
/// arguments; one that never runs for a kernel that did not compile, launched `None`. Each of
/// `symbols` is a kernel's name in `ptx`, by its index. Each static of `memory` has a method
/// of its name too, which gives its memory on the GPU, or never runs, if it did not compile;
/// and each cooperative kernel has one that says how many blocks its grid has at most, unless
/// a kernel or a static has that name, which is an error where `at` says.
pub(crate) fn gpu(
    launchers: &[(&Fn, Option<&[Launched]>)],
    symbols: &[String],
    memory: &[(&lang::Static, Option<&DeviceStatic>)],
    ptx: &str,
    top: &[TokenTree],
    at: &dyn std::ops::Fn(lang::Span) -> Span,
) -> TokenStream {
    let names: Vec<String> = symbols.iter().map(|s| named(s)).collect();
    let globals: Vec<String> = memory.iter().filter_map(|(_, m)| m.map(|m| named(&m.symbol))).collect();
    let mut out = parse(&format!(
        "/// The module's GPU kernels, in PTX, for the CUDA driver to compile for the GPU it finds.\n\
         /// Each is named as C++ names it in the namespaces of the workspace, the crate and the\n\
         /// module, with its compile-time arguments, for the tools that read PTX to show:\n\
         /// `kernite::my_app::gpu::tile<(unsigned long)4>`.\n\
         pub const PTX: &str = {};\n\
         /// The module's GPU kernels, loaded on a GPU: each is launched by the method of its name,\n\
         /// and the memory of each static is the `kernite_cuda::Global` the method of its name gives.\n\
         pub struct Kernels {{\n    module: ::kernite_cuda::Module,\n}}\n\
         impl Kernels",
        named(ptx)
    ));
    let mut methods = parse(&format!(
        "/// The kernels, compiled for `cuda`'s GPU by its driver.\n\
         pub fn load(cuda: &::kernite_cuda::Cuda) -> ::core::result::Result<Kernels, ::kernite_cuda::Error> {{\n\
             const NAMES: &[&str] = &[{}];\n\
             const GLOBALS: &[&str] = &[{}];\n\
             ::core::result::Result::Ok(Kernels {{ module: ::kernite_cuda::Module::load(cuda, PTX, NAMES, GLOBALS)? }})\n\
         }}",
        names.join(", "),
        globals.join(", ")
    ));
    let mut errors = TokenStream::new();
    let taken: Vec<&str> =
        launchers.iter().map(|(f, _)| f.name.as_str()).chain(memory.iter().map(|(s, _)| s.name.as_str())).collect();
    for (f, launched) in launchers {
        methods.extend(launcher(f, *launched, top));
        let Some(c) = &f.cooperative else { continue };
        let name = blocks_name(f);
        if taken.contains(&name.as_str()) {
            let message = format!(
                "`#[cooperative]` gives `Kernels` a method `{name}`, which says how many blocks a grid of `{}` has \
                 at most, and a kernel or a static of the module is named `{name}` as well",
                f.name
            );
            errors.extend(error(at(c.span), &message));
            continue;
        }
        methods.extend(blocks(f, *launched, top));
    }
    for (k, (s, made)) in memory.iter().enumerate() {
        // Its index among the variables the module has, those of the statics that compiled.
        let i = memory[..k].iter().filter(|(_, m)| m.is_some()).count();
        methods.extend(accessor(s, made.map(|m| (m, i)), top));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, methods))]);
    out.extend(errors);
    out
}

/// The method that gives static `s`'s memory on the GPU, `made`, by its index among the
/// module's variables, as the `kernite_cuda::Global` of the type of Rust's its elements are
/// read as; or one that never runs, for a static that did not compile.
fn accessor(s: &lang::Static, made: Option<(&DeviceStatic, usize)>, top: &[TokenTree]) -> TokenStream {
    let mut out: TokenStream = s.attrs.clone().map(|i| top[i].clone()).collect();
    out.extend(top[s.vis.clone()].iter().cloned());
    out.extend(parse("fn"));
    // Its name as written, after `static`, where it is but of the macro's span: Rust lints the case
    // of a name the user wrote, and `COUNTS` is the name of a method, not of a static.
    let mut name = top[s.vis.end + 1].clone();
    name.set_span(name.span().resolved_at(site()));
    out.extend([name]);
    out.extend(parse(&match made {
        Some((m, i)) => {
            let ty = host(&[TokenTree::Ident(Ident::new(&m.host, site()))]);
            format!("(&self) -> ::kernite_cuda::Global<'_, {ty}> {{ self.module.global({i}) }}")
        }
        None => {
            "<T: ::kernite_cuda::Plain>(&self) -> ::kernite_cuda::Global<'_, T> { ::core::unreachable!() }".to_string()
        }
    }));
    out
}

/// `text`, with the module's path written where each [`HOLE`] is, as the constant it is.
fn named(text: &str) -> String {
    let pieces: Vec<String> = text.split(HOLE).map(|p| format!("{p:?}")).collect();
    format!("::kernite_cuda::__named!(::core::module_path!(), [{}])", pieces.join(", "))
}

/// The method that launches `f` over a grid, each of its instances `launched`: its parameters
/// those of `f`, a pointer as the address on the GPU it is, `::kernite_cuda::Ptr`; and its
/// compile-time arguments those of `f`, which pick the instance. A cooperative kernel is
/// launched cooperatively, every block of its grid at once. The launcher binds `grid` for
/// itself, which a parameter of `f` may be named as well; a compile-time argument may not, which
/// is an error at its name, and the launcher never runs, as one of a kernel that did not compile.
fn launcher(f: &Fn, launched: Option<&[Launched]>, top: &[TokenTree]) -> TokenStream {
    let clashes = clashes(f, &["grid"], "the launcher", top);
    let launched = if clashes.is_empty() { launched } else { None };
    let mut out = attrs(f, top);
    if f.cooperative.is_some() {
        out.extend(parse(&format!(
            "#[doc = \"\"]\n#[doc = \" Every block of the grid runs at once, as `grid.sync()` needs, so a grid of more \
             blocks than [`{0}`](Self::{0}) says the GPU runs is an error.\"]",
            blocks_name(f)
        )));
    }
    if !out.to_string().contains("# Safety") {
        out.extend(parse(
            "#[doc = \"\"]\n#[doc = \" # Safety\"]\n#[doc = \"\"]\n\
             #[doc = \" The kernel trusts the pointers it is given: each must be to a buffer as long as the \
             kernel reads and writes through it.\"]",
        ));
    }
    // Clippy says "too many arguments" of a function of more than seven parameters wherever its
    // tokens come from, and the launcher takes `self` and the grid as well: so it allows that, and
    // a crate of kernels may not forbid `clippy::too_many_arguments`.
    out.extend(parse("#[allow(clippy::too_many_arguments)]"));
    // Its visibility, then `unsafe fn` and its name and generics as written.
    out.extend(visibility(f, top).iter().cloned());
    out.extend(parse("unsafe fn"));
    out.extend(top[f.head.end + 1..f.params_at].iter().cloned());
    let mut params = hygienic("&self, $grid: ::kernite_cuda::Grid, ", launched.is_some());
    let mut args = vec![];
    for (param, colon) in written_params(f, top) {
        // Its name as written, before the colon.
        args.push(format!("::kernite_cuda::arg(&{})", param[colon - 1]));
        params.extend(param[..=colon].iter().cloned());
        match &param[colon + 1..] {
            [star, kind, pointee @ ..] if is_punct(star, '*') && (is_ident(kind, "const") || is_ident(kind, "mut")) => {
                params.extend(parse("::kernite_cuda::Ptr<"));
                params.extend(host(pointee));
                params.extend(parse(">"));
            }
            ty => params.extend(host(ty)),
        }
        params.extend(parse(","));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, params))]);
    out.extend(parse("-> ::core::result::Result<(), ::kernite_cuda::Error>"));
    let how = if f.cooperative.is_some() { "launch_cooperative" } else { "launch" };
    out.extend(body(f, launched, top, &never(f, top), |index| {
        format!("unsafe {{ self.module.{how}({index}, $grid, &[{}]) }}", args.join(", "))
    }));
    out.extend(clashes);
    out
}

/// The method that says how many blocks a grid of `f`, a cooperative kernel, has at most, for each
/// of its instances `launched`: as many as the GPU runs at once, of so many threads and so much of
/// the launch's shared memory. It keeps what `f` says of whether it is there, and nothing else:
/// `cfg`, `deprecated` and `doc(hidden)`, each alone or in a `cfg_attr(..)`. It has docs of its
/// own, not those of `f`. The launcher alone keeps what `f` says of lints, of its code or of its
/// symbol, as `allow`, `target_feature` and `export_name` do. The method's name and generics are
/// where `f` has its own, but of the macro's span: Rust lints neither, and says a lint of a
/// compile-time argument once, at the launcher. The method binds `threads` and `shared` for itself,
/// which a compile-time argument of `f` may not be named: an error at its name, and the method
/// never runs, as one of a kernel that did not compile.
fn blocks(f: &Fn, launched: Option<&[Launched]>, top: &[TokenTree]) -> TokenStream {
    let clashes = clashes(f, &["threads", "shared"], &format!("`{}`", blocks_name(f)), top);
    let launched = if clashes.is_empty() { launched } else { None };
    let mut out = parse(&format!(
        "/// How many blocks a grid of [`{0}`](Self::{0}) has at most, of `threads` threads each and\n\
         /// `shared` bytes of the launch's shared memory: as many as the GPU runs at once, for it\n\
         /// runs them all together.",
        f.name
    ));
    for attr in attrs(f, top).into_iter().collect::<Vec<_>>().chunks(2) {
        let [hash, TokenTree::Group(group)] = attr else { unreachable!("an attribute is `#` and brackets") };
        let Some(meta) = carried(flatten(group.stream())) else { continue };
        let mut kept = Group::new(Delimiter::Bracket, meta.into_iter().collect());
        kept.set_span(group.span());
        out.extend([hash.clone(), kept.into()]);
    }
    // Its visibility, then `fn`, its name and the generics of `f`: the name and the generics
    // where `f` has its own, for an error or a reader to find, but of the macro's span.
    out.extend(visibility(f, top).iter().cloned());
    out.extend(parse("fn"));
    let name = top[f.head.end + 1].span().resolved_at(site());
    out.extend([TokenTree::Ident(Ident::new(&blocks_name(f), name))]);
    out.extend(top[f.head.end + 2..f.params_at].iter().map(|t| {
        let mut t = t.clone();
        t.set_span(t.span().resolved_at(site()));
        t
    }));
    out.extend(hygienic(
        "(&self, $threads: u32, $shared: u32) -> ::core::result::Result<u32, ::kernite_cuda::Error>",
        launched.is_some(),
    ));
    out.extend(body(f, launched, top, "::core::unreachable!()", |index| {
        format!("self.module.resident({index}, $threads, $shared)")
    }));
    out.extend(clashes);
    out
}

/// What the method of `blocks` keeps of an attribute of a kernel's, `meta` the tokens in its
/// brackets; `None` if nothing. Of a `cfg_attr(..)`, it keeps the condition and what it keeps of
/// the list, or nothing if that is nothing: a `cfg_attr(..)` of a condition alone is an error.
fn carried(meta: Vec<TokenTree>) -> Option<Vec<TokenTree>> {
    // Rust reads an attribute's name written raw, `r#cfg`, as the name.
    let named = |t: &TokenTree, name: &str| is_ident(t, name) || is_ident(t, &format!("r#{name}"));
    match &meta[..] {
        [doc, TokenTree::Group(list)] if named(doc, "doc") => flatten(list.stream())
            .iter()
            .any(|t| named(t, "hidden"))
            .then(|| parse("doc(hidden)").into_iter().collect()),
        [cfg_attr, TokenTree::Group(list)] if named(cfg_attr, "cfg_attr") => {
            let tokens = flatten(list.stream());
            let mut parts = tokens.split(|t| is_punct(t, ','));
            let mut kept = parts.next().expect("a split has a first part").to_vec();
            let condition = kept.len();
            for part in parts.filter_map(|part| carried(part.to_vec())) {
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
        [name, ..] if ["cfg", "deprecated"].iter().any(|n| named(name, n)) => Some(meta),
        _ => None,
    }
}

/// The name of the method that says how many blocks a grid of `f` has at most.
fn blocks_name(f: &Fn) -> String {
    format!("{}_blocks", f.name)
}

/// The body of a method of `f`, which does `then` with the index of the instance among
/// `launched` its compile-time arguments pick, as an expression; or `never`, for a kernel that
/// did not compile.
fn body(
    f: &Fn,
    launched: Option<&[Launched]>,
    top: &[TokenTree],
    never: &str,
    then: impl std::ops::Fn(&str) -> String,
) -> TokenStream {
    let body = match launched {
        Some([(args, i)]) if args.is_empty() => then(&i.to_string()),
        Some(instances) => {
            let arms: Vec<(Vec<Value>, String)> = instances.iter().map(|(a, i)| (a.clone(), i.to_string())).collect();
            then(&pick(f, &arms, top))
        }
        None => never.to_string(),
    };
    [TokenTree::Group(Group::new(Delimiter::Brace, hygienic(&body, launched.is_some())))].into_iter().collect()
}
