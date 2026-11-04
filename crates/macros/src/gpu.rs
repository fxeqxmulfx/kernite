//! What GPU kernels become in Rust: `PTX`, the module they are, and `Kernels`, which loads it
//! on a GPU, with a method to launch each kernel and one to give each static's memory.

use proc_macro::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};

use kernite::lang::{self, Fn, Value};
use kernite::{DeviceStatic, Launcher};

use crate::cond::Cond;
use crate::rust::{
    asks, attrs, carried, clashes, depends_on, error_under, half, host, hygienic, missing, never, number_type, parse,
    pick, pointer, primitive, quiet, refused, site, visibility, written_params,
};

/// The crate the Rust of a module of GPU kernels names.
const CUDA: &str = "kernite-cuda";

/// Where the module's path goes in its PTX and in its kernels' names. A macro is not told
/// the module it is in, so `kernite_cuda::__named!` writes `module_path!()` there once Rust
/// knows it, as C++ mangles a namespace's names.
pub(crate) const HOLE: &str = "\0";

/// What a kernel's instance is: its compile-time arguments, none for a kernel that takes
/// none, and its index among the names the driver is asked for, as an expression of Rust.
type Launched = (Vec<Value>, String);

/// `PTX`, the module the GPU kernels are, and `Kernels`, which loads it on a GPU and has a
/// method to launch each kernel by its name: of each instance, for a kernel of compile-time
/// arguments; one that never runs for a kernel that did not compile, a [`Launcher::Never`], each
/// instance by the index of its name in `symbols`. Each static of `memory` has a method
/// of its name too, which gives its memory on the GPU, or never runs, if it did not compile;
/// and each cooperative kernel has one that says how many blocks its grid has at most, unless
/// a kernel or a static has that name, which is an error where `at` says. `ptx` is the module's
/// text in parts. Each part, each name in `symbols` and each static of `memory` is there where its
/// condition holds, for Rust puts the module together where it knows which `cfg` holds; the index
/// of a name is its place among those that are there. A crate without `kernite-cuda` has `PTX`
/// empty and a `Kernels` that is the error, said once, at the module: nothing else is said of the
/// code that loads it or launches a kernel. Both are there when any of the module's GPU items is,
/// `on_gpu` their conditions: a feature that brings `kernite-cuda` and keeps them lets the crate
/// build without the feature.
pub(crate) fn gpu(
    launchers: &[(&Fn, &Launcher)],
    symbols: &[(Cond, &str)],
    memory: &[(&lang::Static, Option<&DeviceStatic>, Cond)],
    ptx: &[(Cond, &str)],
    top: &[TokenTree],
    at: &dyn std::ops::Fn(lang::Span) -> Span,
    on_gpu: &[Cond],
) -> TokenStream {
    if !depends_on(CUDA) {
        let Some(gate) = Cond::any(on_gpu.iter().cloned()).guard() else { return TokenStream::new() };
        let mut out = gate.clone();
        out.extend(parse("pub const PTX: &::core::primitive::str = \"\";"));
        out.extend(gate);
        out.extend(parse("pub type Kernels ="));
        out.extend(missing(site(), "a module of GPU kernels", CUDA));
        out.extend(parse(";"));
        return out;
    }
    // The module's variables: the statics that compiled, in order.
    let variables: Vec<(&Cond, &DeviceStatic)> = memory.iter().filter_map(|(_, m, c)| m.map(|m| (c, m))).collect();
    let mut out = parse(
        "/// The module's GPU kernels, functions and statics that `cfg` keeps, in PTX, for the CUDA\n\
         /// driver to compile for the GPU it finds. Each is named as C++ names it in the\n\
         /// namespaces of the workspace, the crate and the module, with its compile-time\n\
         /// arguments, for the tools that read PTX to show:\n\
         /// `kernite::my_app::gpu::tile<(unsigned long)4>`.\n\
         pub const PTX: &::core::primitive::str =",
    );
    out.extend(named(ptx));
    out.extend(parse(
        ";\n\
         /// The module's GPU kernels that `cfg` keeps, loaded on a GPU: each is launched by the\n\
         /// method of its name, and the memory of each static is the `kernite_cuda::Global` the\n\
         /// method of its name gives.\n\
         pub struct Kernels {\n    module: ::kernite_cuda::Module,\n}\n\
         impl Kernels",
    ));
    // The names the driver finds the kernels and the statics by, those `cfg` keeps.
    let mut load = parse("const NAMES: &[&::core::primitive::str] = &");
    load.extend([names(symbols.iter().map(|(c, s)| (c, *s)))]);
    load.extend(parse("; const GLOBALS: &[&::core::primitive::str] = &"));
    load.extend([names(variables.iter().map(|(c, m)| (*c, m.symbol.as_str())))]);
    load.extend(parse(
        ";\n::core::result::Result::Ok(Kernels { module: ::kernite_cuda::Module::load(cuda, PTX, NAMES, GLOBALS)? })",
    ));
    let mut methods = parse(
        "/// The kernels, compiled for `cuda`'s GPU by its driver.\n\
         pub fn load(cuda: &::kernite_cuda::Cuda) -> ::core::result::Result<Kernels, ::kernite_cuda::Error>",
    );
    methods.extend([TokenTree::Group(Group::new(Delimiter::Brace, load))]);
    // The index of the name of symbol `i` among the names that are there.
    let index = |i: usize| Cond::count(symbols[..i].iter().map(|(c, _)| c));
    let mut errors = TokenStream::new();
    let taken: Vec<&str> =
        launchers.iter().map(|(f, _)| f.name.as_str()).chain(memory.iter().map(|(s, ..)| s.name.as_str())).collect();
    for (f, how) in launchers {
        let launched: Option<Vec<Launched>> = match how {
            Launcher::Never => None,
            Launcher::Of(l) => Some(l.iter().map(|(args, i)| (args.clone(), index(*i))).collect()),
        };
        methods.extend(launcher(f, launched.as_deref(), top));
        let Some(c) = &f.cooperative else { continue };
        let name = blocks_name(f);
        if taken.contains(&name.as_str()) {
            let message = format!(
                "`#[cooperative]` gives `Kernels` a method `{name}`, which says how many blocks a grid of `{}` has \
                 at most, and a kernel or a static of the module is named `{name}` as well",
                f.name
            );
            // The method would be under the kernel's `cfg`, and so is the error of its name.
            errors.extend(error_under(&carried(attrs(f, top), false), at(c.span), &message));
            continue;
        }
        methods.extend(blocks(f, launched.as_deref(), top));
    }
    for (k, (s, made, _)) in memory.iter().enumerate() {
        // Its index among the variables there are, those of the statics that compiled.
        let i = Cond::count(memory[..k].iter().filter(|(_, m, _)| m.is_some()).map(|(.., c)| c));
        methods.extend(accessor(s, made.map(|m| (m, i)), top));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, methods))]);
    out.extend(errors);
    out
}

/// The method that gives static `s`'s memory on the GPU, `made`, by its index among the
/// module's variables, an expression, as the `kernite_cuda::Global` of the type of Rust's its
/// elements are read as; or one that never runs, for a static that did not compile.
fn accessor(s: &lang::Static, made: Option<(&DeviceStatic, String)>, top: &[TokenTree]) -> TokenStream {
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
            let ty = primitive(&m.host);
            format!("(&self) -> ::kernite_cuda::Global<'_, {ty}> {{ self.module.global({i}) }}")
        }
        None => {
            "<T: ::kernite_cuda::Plain>(&self) -> ::kernite_cuda::Global<'_, T> { ::core::unreachable!() }".to_string()
        }
    }));
    out
}

/// The constant that is the `parts` of a text joined, each under its condition, with the module's
/// path written where each [`HOLE`] is.
fn named(parts: &[(Cond, &str)]) -> TokenStream {
    let mut list = TokenStream::new();
    for (cond, text) in parts {
        let Some(guard) = cond.guard() else { continue };
        let pieces: Vec<String> = text.split(HOLE).map(|piece| format!("{piece:?}")).collect();
        list.extend(guard);
        list.extend(parse(&format!("[{}],", pieces.join(", "))));
    }
    let mut args = parse("::core::module_path!(),");
    args.extend([TokenTree::Group(Group::new(Delimiter::Bracket, list))]);
    let mut out = parse("::kernite_cuda::__named!");
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, args))]);
    out
}

/// The names of `symbols` as the constants `named` makes them, in brackets, each there where its
/// condition holds.
fn names<'a>(symbols: impl Iterator<Item = (&'a Cond, &'a str)>) -> TokenTree {
    let mut list = TokenStream::new();
    for (cond, symbol) in symbols {
        let Some(guard) = cond.guard() else { continue };
        list.extend(guard);
        list.extend(named(&[(Cond::Always, symbol)]));
        list.extend(parse(","));
    }
    TokenTree::Group(Group::new(Delimiter::Bracket, list))
}

/// The method that launches `f` over a grid, each of its instances `launched`: its parameters
/// those of `f`, a pointer as the address on the GPU it is, `::kernite_cuda::Ptr`, and a number as
/// the number and not the name the kernel writes; and its compile-time arguments those of `f`,
/// which pick the instance. A cooperative kernel is launched cooperatively, every block of its
/// grid at once. The launcher binds `grid` for itself, which a parameter of `f` may be named as
/// well; a compile-time argument may not, which is an error at its name, and the launcher never
/// runs, as one of a kernel that did not compile.
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
        // Its name, before the colon.
        let name = &param[colon - 1];
        params.extend(param[..=colon].iter().cloned());
        match pointer(&param[colon + 1..]) {
            Some((_, to)) => {
                // Its address, by a cast. A call of `arg` is generic over the type of the pointer, so Rust
                // would say an error in the type it points to again, at the call.
                args.push(format!("(&raw const {name}) as *const ::core::ffi::c_void"));
                params.extend(parse("::kernite_cuda::Ptr<"));
                params.extend(host(to));
                // A type that ends in `>` is joint with the next token, and Rust, which reads the two as
                // one, `>>`, and breaks it again, would say an error in the type at the attribute.
                params.extend(parse(",>"));
            }
            None => {
                args.push(format!("::kernite_cuda::arg(&{name})"));
                let ty = &param[colon + 1..];
                params.extend(number_type(ty).or_else(|| half(ty)).unwrap_or_else(|| refused(ty)));
            }
        }
        params.extend(parse(","));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, params))]);
    out.extend(parse("-> ::core::result::Result<(), ::kernite_cuda::Error>"));
    let how = if f.cooperative.is_some() { "launch_cooperative" } else { "launch" };
    let never = format!("{}::core::unreachable!()", never(f, top));
    out.extend(body(f, launched, top, asks(f, top, host), &never, |index| {
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
    out.extend(carried(attrs(f, top), true));
    // Its visibility, then `fn`, its name and the generics of `f`: the name and the generics
    // where `f` has its own, for an error or a reader to find, but of the macro's span.
    out.extend(visibility(f, top).iter().cloned());
    out.extend(parse("fn"));
    let name = top[f.head.end + 1].span().resolved_at(site());
    out.extend([TokenTree::Ident(Ident::new(&blocks_name(f), name))]);
    out.extend(quiet(&top[f.head.end + 2..f.params_at]));
    out.extend(hygienic(
        "(&self, $threads: ::core::primitive::u32, $shared: ::core::primitive::u32) \
         -> ::core::result::Result<::core::primitive::u32, ::kernite_cuda::Error>",
        launched.is_some(),
    ));
    out.extend(body(f, launched, top, TokenStream::new(), "::core::unreachable!()", |index| {
        format!("self.module.resident({index}, $threads, $shared)")
    }));
    out.extend(clashes);
    out
}

/// The name of the method that says how many blocks a grid of `f` has at most.
fn blocks_name(f: &Fn) -> String {
    format!("{}_blocks", f.name)
}

/// The body of a method of `f`, which does `then` with the index of the instance among
/// `launched` its compile-time arguments pick, as an expression; or `never`, for a kernel that
/// did not compile. The `checks` come first.
fn body(
    f: &Fn,
    launched: Option<&[Launched]>,
    top: &[TokenTree],
    mut checks: TokenStream,
    never: &str,
    then: impl std::ops::Fn(&str) -> String,
) -> TokenStream {
    let body = match launched {
        Some([(args, i)]) if args.is_empty() => then(i),
        Some(instances) => then(&pick(f, instances, top)),
        None => never.to_string(),
    };
    checks.extend(hygienic(&body, launched.is_some()));
    [TokenTree::Group(Group::new(Delimiter::Brace, checks))].into_iter().collect()
}
