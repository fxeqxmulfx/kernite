//! What GPU kernels become in Rust: `PTX`, the module they are, and `Kernels`, which loads it
//! on a GPU, with a method to launch each kernel and one to give each static's memory.

use proc_macro::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};

use kernite::DeviceStatic;
use kernite::lang::{self, Fn, Value};

use crate::rust::{
    attrs, carried, cfg_any, checks, clashes, depends_on, error_under, host, hygienic, missing, never, number_type,
    parse, pick, pointer, primitive, quiet, site, visibility, written_params,
};

/// The crate the Rust of a module of GPU kernels names.
const CUDA: &str = "kernite-cuda";

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
/// a kernel or a static has that name, which is an error where `at` says. A crate without
/// `kernite-cuda` has `PTX` empty and a `Kernels` that is the error, said once, at the module:
/// nothing else is said of the code that loads it or launches a kernel. Both are there when any of
/// the module's GPU items is, `on_gpu` their conditions, each as `condition` has it: a feature that
/// brings `kernite-cuda` and keeps them lets the crate build without the feature.
pub(crate) fn gpu(
    launchers: &[(&Fn, Option<&[Launched]>)],
    symbols: &[String],
    memory: &[(&lang::Static, Option<&DeviceStatic>)],
    ptx: &str,
    top: &[TokenTree],
    at: &dyn std::ops::Fn(lang::Span) -> Span,
    on_gpu: &[Option<TokenStream>],
) -> TokenStream {
    if !depends_on(CUDA) {
        let gate = cfg_any(on_gpu);
        let mut out = gate.clone();
        out.extend(parse("pub const PTX: &::core::primitive::str = \"\";"));
        out.extend(gate);
        out.extend(parse("pub type Kernels ="));
        out.extend(missing(site(), "a module of GPU kernels", CUDA));
        out.extend(parse(";"));
        return out;
    }
    let names: Vec<String> = symbols.iter().map(|s| named(s)).collect();
    let globals: Vec<String> = memory.iter().filter_map(|(_, m)| m.map(|m| named(&m.symbol))).collect();
    let mut out = parse(&format!(
        "/// The module's GPU kernels, in PTX, for the CUDA driver to compile for the GPU it finds.\n\
         /// Each is named as C++ names it in the namespaces of the workspace, the crate and the\n\
         /// module, with its compile-time arguments, for the tools that read PTX to show:\n\
         /// `kernite::my_app::gpu::tile<(unsigned long)4>`.\n\
         pub const PTX: &::core::primitive::str = {};\n\
         /// The module's GPU kernels, loaded on a GPU: each is launched by the method of its name,\n\
         /// and the memory of each static is the `kernite_cuda::Global` the method of its name gives.\n\
         pub struct Kernels {{\n    module: ::kernite_cuda::Module,\n}}\n\
         impl Kernels",
        named(ptx)
    ));
    let mut methods = parse(&format!(
        "/// The kernels, compiled for `cuda`'s GPU by its driver.\n\
         pub fn load(cuda: &::kernite_cuda::Cuda) -> ::core::result::Result<Kernels, ::kernite_cuda::Error> {{\n\
             const NAMES: &[&::core::primitive::str] = &[{}];\n\
             const GLOBALS: &[&::core::primitive::str] = &[{}];\n\
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
            // The method would be under the kernel's `cfg`, and so is the error of its name.
            errors.extend(error_under(&carried(attrs(f, top), false), at(c.span), &message));
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
            let ty = primitive(&m.host);
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
        // Its name as written, before the colon.
        args.push(format!("::kernite_cuda::arg(&{})", param[colon - 1]));
        params.extend(param[..=colon].iter().cloned());
        match pointer(&param[colon + 1..]) {
            Some((_, to)) => {
                params.extend(parse("::kernite_cuda::Ptr<"));
                params.extend(host(to));
                params.extend(parse(">"));
            }
            None => {
                let ty = &param[colon + 1..];
                params.extend(number_type(ty).unwrap_or_else(|| host(ty)));
            }
        }
        params.extend(parse(","));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, params))]);
    out.extend(parse("-> ::core::result::Result<(), ::kernite_cuda::Error>"));
    let how = if f.cooperative.is_some() { "launch_cooperative" } else { "launch" };
    let never = format!("{}::core::unreachable!()", never(f, top));
    out.extend(body(f, launched, top, checks(f, top, "::kernite_cuda::Ptr<T>"), &never, |index| {
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
        Some([(args, i)]) if args.is_empty() => then(&i.to_string()),
        Some(instances) => {
            let arms: Vec<(Vec<Value>, String)> = instances.iter().map(|(a, i)| (a.clone(), i.to_string())).collect();
            then(&pick(f, &arms, top))
        }
        None => never.to_string(),
    };
    checks.extend(hygienic(&body, launched.is_some()));
    [TokenTree::Group(Group::new(Delimiter::Brace, checks))].into_iter().collect()
}
