//! What GPU kernels become in Rust: `PTX`, the module they are, and `Kernels`, which loads it
//! on a GPU, with a method to launch each kernel and one to give each static's memory.

use proc_macro::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};

use kgen::DeviceStatic;
use kgen::lang::{self, Fn, Value};

use crate::rust::{attrs, error, host, is_ident, is_punct, parse, pick, site, visibility, written_params};

/// Where the module's path goes in its PTX and in its kernels' names. A macro is not told
/// the module it is in, so `kgen_cuda::__named!` writes `module_path!()` there once Rust
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
    launchers: &[(&Fn, Option<Vec<Launched>>)],
    symbols: &[String],
    memory: &[(&lang::Static, Option<DeviceStatic>)],
    ptx: &str,
    top: &[TokenTree],
    at: &dyn std::ops::Fn(lang::Span) -> Span,
) -> TokenStream {
    let names: Vec<String> = symbols.iter().map(|s| named(s)).collect();
    let globals: Vec<String> = memory.iter().filter_map(|(_, m)| Some(named(&m.as_ref()?.symbol))).collect();
    let mut out = parse(&format!(
        "/// The module's GPU kernels, in PTX, for the CUDA driver to compile for the GPU it finds.\n\
         /// Each is named as C++ names it in the namespaces of the workspace, the crate and the\n\
         /// module, with its compile-time arguments, for the tools that read PTX to show:\n\
         /// `kgen::my_app::gpu::tile<(unsigned long)4>`.\n\
         #[allow(dead_code)]\n\
         pub const PTX: &str = {};\n\
         /// The module's GPU kernels, loaded on a GPU: each is launched by the method of its name,\n\
         /// and the memory of each static is the `kgen_cuda::Global` the method of its name gives.\n\
         #[allow(dead_code)]\n\
         pub struct Kernels {{\n    module: ::kgen_cuda::Module,\n}}\n\
         #[allow(dead_code)]\n\
         impl Kernels",
        named(ptx)
    ));
    let mut methods = parse(&format!(
        "/// The kernels, compiled for `cuda`'s GPU by its driver.\n\
         pub fn load(cuda: &::kgen_cuda::Cuda) -> ::core::result::Result<Kernels, ::kgen_cuda::Error> {{\n\
             const NAMES: &[&str] = &[{}];\n\
             const GLOBALS: &[&str] = &[{}];\n\
             ::core::result::Result::Ok(Kernels {{ module: ::kgen_cuda::Module::load(cuda, PTX, NAMES, GLOBALS)? }})\n\
         }}",
        names.join(", "),
        globals.join(", ")
    ));
    let mut errors = TokenStream::new();
    let taken: Vec<&str> =
        launchers.iter().map(|(f, _)| f.name.as_str()).chain(memory.iter().map(|(s, _)| s.name.as_str())).collect();
    for (f, launched) in launchers {
        methods.extend(launcher(f, launched.as_deref(), top));
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
        methods.extend(blocks(f, launched.as_deref(), top));
    }
    for (k, (s, made)) in memory.iter().enumerate() {
        // Its index among the variables the module has, those of the statics that compiled.
        let i = memory[..k].iter().filter(|(_, m)| m.is_some()).count();
        methods.extend(accessor(s, made.as_ref().map(|m| (m, i)), top));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, methods))]);
    out.extend(errors);
    out
}

/// The method that gives static `s`'s memory on the GPU, `made`, by its index among the
/// module's variables, as the `kgen_cuda::Global` of the type of Rust's its elements are read
/// as; or one that never runs, for a static that did not compile.
fn accessor(s: &lang::Static, made: Option<(&DeviceStatic, usize)>, top: &[TokenTree]) -> TokenStream {
    let mut out: TokenStream = s.attrs.clone().map(|i| top[i].clone()).collect();
    out.extend(parse("#[allow(non_snake_case)]"));
    out.extend(top[s.vis.clone()].iter().cloned());
    out.extend(parse("fn"));
    // Its name as written, after `static`.
    out.extend([top[s.vis.end + 1].clone()]);
    out.extend(parse(&match made {
        Some((m, i)) => {
            let ty = host(&[TokenTree::Ident(Ident::new(&m.host, site()))]);
            format!("(&self) -> ::kgen_cuda::Global<'_, {ty}> {{ self.module.global({i}) }}")
        }
        None => "<T: ::kgen_cuda::Plain>(&self) -> ::kgen_cuda::Global<'_, T> { ::core::unreachable!() }".to_string(),
    }));
    out
}

/// `text`, with the module's path written where each [`HOLE`] is, as the constant it is.
fn named(text: &str) -> String {
    let pieces: Vec<String> = text.split(HOLE).map(|p| format!("{p:?}")).collect();
    format!("::kgen_cuda::__named!(::core::module_path!(), [{}])", pieces.join(", "))
}

/// The method that launches `f` over a grid, each of its instances `launched`: its parameters
/// those of `f`, a pointer as the address on the GPU it is, `::kgen_cuda::Ptr`; and its
/// compile-time arguments those of `f`, which pick the instance. A cooperative kernel is
/// launched cooperatively, every block of its grid at once.
fn launcher(f: &Fn, launched: Option<&[Launched]>, top: &[TokenTree]) -> TokenStream {
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
    out.extend(parse(if launched.is_some() {
        "#[allow(clippy::too_many_arguments)]"
    } else {
        "#[allow(clippy::too_many_arguments, unused_variables)]"
    }));
    // Its visibility, then `unsafe fn` and its name and generics as written.
    out.extend(visibility(f, top).iter().cloned());
    out.extend(parse("unsafe fn"));
    out.extend(top[f.head.end + 1..f.params_at].iter().cloned());
    let mut params = parse("&self, grid: ::kgen_cuda::Grid, ");
    let mut args = vec![];
    for (param, colon) in written_params(f, top) {
        // Its name as written, before the colon.
        args.push(format!("::kgen_cuda::arg(&{})", param[colon - 1]));
        params.extend(param[..=colon].iter().cloned());
        match &param[colon + 1..] {
            [star, kind, pointee @ ..] if is_punct(star, '*') && (is_ident(kind, "const") || is_ident(kind, "mut")) => {
                params.extend(parse("::kgen_cuda::Ptr<"));
                params.extend(host(pointee));
                params.extend(parse(">"));
            }
            ty => params.extend(host(ty)),
        }
        params.extend(parse(","));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, params))]);
    out.extend(parse("-> ::core::result::Result<(), ::kgen_cuda::Error>"));
    let how = if f.cooperative.is_some() { "launch_cooperative" } else { "launch" };
    out.extend(body(f, launched, top, |index| {
        format!("unsafe {{ self.module.{how}({index}, grid, &[{}]) }}", args.join(", "))
    }));
    out
}

/// The method that says how many blocks a grid of `f`, a cooperative kernel, has at most, for
/// each of its instances `launched`: as many as the GPU runs at once, of so many threads and
/// so much of the launch's shared memory.
fn blocks(f: &Fn, launched: Option<&[Launched]>, top: &[TokenTree]) -> TokenStream {
    let mut out = parse(&format!(
        "/// How many blocks a grid of [`{0}`](Self::{0}) has at most, of `threads` threads each and\n\
         /// `shared` bytes of the launch's shared memory: as many as the GPU runs at once, for it\n\
         /// runs them all together.",
        f.name
    ));
    if launched.is_none() {
        out.extend(parse("#[allow(unused_variables)]"));
    }
    // Its visibility, then `fn`, its name and the generics of `f` as written: its name is
    // where `f`'s is, for an error or a reader to find.
    out.extend(visibility(f, top).iter().cloned());
    out.extend(parse("fn"));
    out.extend([TokenTree::Ident(Ident::new(&blocks_name(f), top[f.head.end + 1].span()))]);
    out.extend(top[f.head.end + 2..f.params_at].iter().cloned());
    out.extend(parse("(&self, threads: u32, shared: u32) -> ::core::result::Result<u32, ::kgen_cuda::Error>"));
    out.extend(body(f, launched, top, |index| format!("self.module.resident({index}, threads, shared)")));
    out
}

/// The name of the method that says how many blocks a grid of `f` has at most.
fn blocks_name(f: &Fn) -> String {
    format!("{}_blocks", f.name)
}

/// The body of a method of `f`, which does `then` with the index of the instance among
/// `launched` its compile-time arguments pick; or never runs, for a kernel that did not
/// compile.
fn body(
    f: &Fn,
    launched: Option<&[Launched]>,
    top: &[TokenTree],
    then: impl std::ops::Fn(&str) -> String,
) -> TokenStream {
    let body = match launched {
        Some([(args, i)]) if args.is_empty() => then(&i.to_string()),
        Some(instances) => {
            let arms: Vec<(Vec<Value>, String)> = instances.iter().map(|(a, i)| (a.clone(), i.to_string())).collect();
            format!("let index = {};\n{}", pick(f, &arms, top), then("index"))
        }
        None => "::core::unreachable!()".to_string(),
    };
    [TokenTree::Group(Group::new(Delimiter::Brace, parse(&body)))].into_iter().collect()
}
