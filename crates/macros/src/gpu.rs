//! What GPU kernels become in Rust: `PTX`, the module they are, and `Kernels`, which loads it
//! on a GPU, with a method to launch each kernel and one to give each static's memory.

use proc_macro::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};

use kernite::lang::{self, Fn, Value};
use kernite::{Compiled, DeviceStatic, Failed, Launcher};

use crate::cond::Cond;
use crate::rust::{
    self, LEVELS, asks, attrs, body_at, carried, clashes, depends_on, ending_at, error_under, half, host, hygienic,
    launcher_result, levels, missing, name_at, never, number_type, parse, pick, pointer, primitive, ptr_type, quiet,
    refused, site, unnamed, visibility, written_params,
};

/// The crate the Rust of a module of GPU kernels names.
const CUDA: &str = "kernite-cuda";

/// Where the module's path goes in its PTX and in its kernels' names. A macro is not told
/// the module it is in, so `kernite_cuda::__named!` writes `module_path!()` there once Rust
/// knows it, as C++ mangles a namespace's names. It is a NUL, which the PTX holds nowhere else:
/// kernite refuses one in a note, the one place where a kernel's source could put one.
pub(crate) const HOLE: &str = "\0";

/// What a kernel's instance is: its compile-time arguments, none for a kernel that takes
/// none, and its index among the names the driver is asked for, as an expression of Rust.
type Launched = (Vec<Value>, String);

/// A static of the GPU's as kernite compiled it; or why it is none, and what stands for it.
pub(crate) type Laid = Result<Compiled<DeviceStatic>, Failed<DeviceStatic>>;

/// `PTX`, the module the GPU kernels are, and `Kernels`, which loads it on a GPU and has a
/// method to launch each kernel by its name: of each instance, for a kernel of compile-time
/// arguments; one that never runs for a kernel that did not compile, a [`Launcher::Never`], each
/// instance by the index of its name in `symbols`. Each static of `memory` has a method
/// of its name too, which gives its memory on the GPU, or never runs, if it did not compile;
/// and each cooperative kernel has one that says how many blocks its grid has at most, unless
/// a kernel or a static has that name, which is an error where `at` says. A compile-time argument
/// named as a parameter of a launcher or of a `_blocks` method is an error too, at its name. Each
/// is written after the `impl`, as an item of the module. `ptx` is the module's
/// text in parts. Each part, each name in `symbols` and each static of `memory` is there where its
/// condition holds, for Rust puts the module together where it knows which `cfg` holds; the index
/// of a name is its place among those that are there. A crate without `kernite-cuda` has `PTX`
/// empty and a `Kernels` that is [`missing`]'s type, with the error said once, at the module: nothing
/// else is said of the code that loads it or launches a kernel. Both are there when any of the
/// module's GPU items is, `on_gpu` their conditions: a feature that brings `kernite-cuda` and keeps
/// them lets the crate build without the feature.
///
/// Each kernel of `launchers` has its `places`, those of its code that name an item of the module, in the body of its
/// launcher, where the level the kernel sets holds; so has each static of `memory`, those of its type and of what it
/// holds, in the body of its method. In a crate without `kernite-cuda`, which has no method, a static has its places in
/// an item of no name with its levels, where its condition holds.
pub(crate) fn gpu(
    launchers: &[(&Fn, &Launcher, TokenStream)],
    symbols: &[(Cond, &str)],
    memory: &[(&lang::Static, &Laid, TokenStream, Cond)],
    ptx: &[(Cond, &str)],
    top: &[TokenTree],
    at: &dyn std::ops::Fn(lang::Span) -> Span,
    on_gpu: &[Cond],
) -> TokenStream {
    if !depends_on(CUDA) {
        let Some(gate) = Cond::any(on_gpu.iter().cloned()).guard() else { return TokenStream::new() };
        let (kernels, error) = missing(&gate, site(), "a module of GPU kernels", CUDA);
        let mut out = gate.clone();
        out.extend(parse("pub const PTX: &::core::primitive::str = \"\";"));
        out.extend(gate);
        out.extend(parse("pub type Kernels ="));
        out.extend(kernels);
        out.extend(parse(";"));
        out.extend(error);
        // No method holds the places of a static, so they are in an item of no name with its levels, where its
        // condition holds.
        for (s, _, places, cond) in memory {
            if !places.is_empty()
                && let Some(guard) = cond.guard()
            {
                out.extend(guard);
                out.extend(levels(top[s.attrs.clone()].iter().cloned(), &LEVELS));
                out.extend(unnamed(places.clone()));
            }
        }
        return out;
    }
    // The module's variables: the statics that compiled, in order.
    let variables: Vec<(&Cond, &DeviceStatic)> =
        memory.iter().filter_map(|(_, m, _, c)| m.as_ref().ok().map(|m| (c, &m.code))).collect();
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
    load.extend([listed(symbols.iter().map(|(c, s)| (c, named(&[(Cond::Always, s)]))))]);
    load.extend(parse(
        "; const GLOBALS: &[(&::core::primitive::str, ::core::option::Option<(::core::primitive::usize, \
         ::core::primitive::usize)>)] = &",
    ));
    load.extend([listed(variables.iter().map(|(c, m)| (*c, variable(m))))]);
    load.extend(parse(
        ";\n::core::result::Result::Ok(Kernels {\n\
         module: ::kernite_cuda::Module::load(__kernite_cuda, PTX, NAMES, GLOBALS)?,\n})",
    ));
    let mut methods = parse(
        "/// The kernels, compiled for the given `Cuda`'s GPU by its driver.\n\
         pub fn load(__kernite_cuda: &::kernite_cuda::Cuda) -> ::core::result::Result<Kernels, ::kernite_cuda::Error>",
    );
    methods.extend([TokenTree::Group(Group::new(Delimiter::Brace, load))]);
    // The index of the name of symbol `i` among the names that are there.
    let index = |i: usize| Cond::count(symbols[..i].iter().map(|(c, _)| c));
    // The errors that follow the `impl`, as items of the module: those of a name that two methods have, and those of
    // a compile-time argument named as a parameter that a method binds for itself.
    let (mut errors, mut clashed) = (TokenStream::new(), TokenStream::new());
    let taken: Vec<&str> =
        launchers.iter().map(|(f, ..)| f.name.as_str()).chain(memory.iter().map(|(s, ..)| s.name.as_str())).collect();
    for (f, how, places) in launchers {
        let launched: Option<Vec<Launched>> = match how {
            Launcher::Never => None,
            Launcher::Of(l) => Some(l.iter().map(|(args, i)| (args.clone(), index(*i))).collect()),
        };
        // The methods would be under the kernel's `cfg`, and so are their errors.
        let cfg = carried(attrs(f, top), false);
        let grid = clashes(f, &cfg, &["__kernite_grid"], "the launcher", top);
        methods.extend(launcher(f, launched.as_deref().filter(|_| grid.is_empty()), top, places.clone()));
        clashed.extend(grid);
        let Some(c) = &f.cooperative else { continue };
        let name = blocks_name(f);
        if taken.contains(&name.as_str()) {
            let message = format!(
                "`#[cooperative]` gives `Kernels` a method `{name}`, which says how many blocks a grid of `{}` has \
                 at most, and a kernel or a static of the module is named `{name}` as well",
                f.name
            );
            errors.extend(error_under(&cfg, at(c.span), &message));
            continue;
        }
        let threads = clashes(f, &cfg, &["__kernite_threads", "__kernite_shared"], &format!("`{name}`"), top);
        methods.extend(blocks(f, launched.as_deref().filter(|_| threads.is_empty()), top));
        clashed.extend(threads);
    }
    for (k, (s, laid, places, _)) in memory.iter().enumerate() {
        // Its index among the variables there are, those of the statics that compiled.
        let i = Cond::count(memory[..k].iter().filter(|(_, m, ..)| m.is_ok()).map(|(.., c)| c));
        methods.extend(accessor(s, laid, i, top, places.clone()));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, methods))]);
    out.extend(clashed);
    out.extend(errors);
    out
}

/// The method that gives static `s`'s memory on the GPU, `laid`, by its index `i` among the module's variables, an
/// expression, as the `kernite_cuda::Global` of the type of Rust's its elements are read as; or one that never runs,
/// for a static that did not compile: of the type its elements are read as, if only what the static holds is wrong, and
/// of any type if not. It has the attributes of the static that [`taken`](rust::taken) says it takes. Its body has the
/// `places` of the static's type and of what it holds that name an item, where the level the static sets holds, as in a
/// static of Rust's, `expect` too. It ends at the static's name, for Rust to tell of the method at the static, as it
/// tells of any item with no doc that a module has: see [`ending_at`].
fn accessor(s: &lang::Static, laid: &Laid, i: String, top: &[TokenTree], places: TokenStream) -> TokenStream {
    let mut out = rust::taken(top[s.attrs.clone()].iter().cloned()).0;
    out.extend(top[s.vis.clone()].iter().cloned());
    out.extend(parse("fn"));
    // Its name as written, after `static`, where it is but of the macro's span: Rust lints the case
    // of a name the user wrote, and `COUNTS` is the name of a method, not of a static.
    let mut name = top[s.vis.end + 1].clone();
    let at = name.span();
    name.set_span(at.resolved_at(site()));
    out.extend([name]);
    let body = match laid {
        Ok(_) => format!("self.module.global({i})"),
        Err(_) => "::core::unreachable!()".to_string(),
    };
    let signature = match read(laid) {
        Some(m) => format!("(&self) -> ::kernite_cuda::Global<'_, {}>", primitive(&m.host)),
        None => "<T: ::kernite_cuda::Plain>(&self) -> ::kernite_cuda::Global<'_, T>".to_string(),
    };
    out.extend(ending_at(parse(&signature), at));
    let mut code = unnamed(places);
    code.extend(parse(&body));
    out.extend([body_at(code, at)]);
    out
}

/// The static, `laid`, whose elements the method that gives its memory reads as the type of Rust's they are: itself, or
/// what stands for it if only what it holds is wrong. None for a static that has none: its method is of any type.
fn read(laid: &Laid) -> Option<&DeviceStatic> {
    laid.as_ref().map_or_else(|failed| failed.stub.as_ref(), |m| Some(&m.code))
}

/// What Rust reads for a use of static `item`, `laid`, by a kernel that names it at `span`: the path of its method,
/// `Kernels::COUNTS`, which has the attributes of the static, so Rust tells of the use as it tells of any use of the
/// method, a deprecated one at that place. None where there is nothing to read: a crate without `kernite-cuda` has no
/// `Kernels`, and the method of a static that has no type is of any type, which Rust cannot work out from a path.
pub(crate) fn used(item: &lang::Item, laid: &Laid, top: &[TokenTree], span: Span) -> Option<TokenStream> {
    if !depends_on(CUDA) || read(laid).is_none() {
        return None;
    }
    let mut path = parse("Kernels::");
    path.extend([name_at(item, top, span)]);
    Some(path)
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

/// What `Module::load` is given of memory `m`: its name, as the constant `named` makes it, and the bytes of memory the
/// module has no room for and their alignment, `Some((bytes, align))`, or `None`.
fn variable(m: &DeviceStatic) -> TokenStream {
    let mut tuple = named(&[(Cond::Always, &m.symbol)]);
    tuple.extend(parse(&match m.outside {
        Some((bytes, align)) => format!(", ::core::option::Option::Some(({bytes}, {align}))"),
        None => ", ::core::option::Option::None".to_string(),
    }));
    TokenTree::Group(Group::new(Delimiter::Parenthesis, tuple)).into()
}

/// `elements` in brackets, each there where its condition holds.
fn listed<'a>(elements: impl Iterator<Item = (&'a Cond, TokenStream)>) -> TokenTree {
    let mut list = TokenStream::new();
    for (cond, element) in elements {
        let Some(guard) = cond.guard() else { continue };
        list.extend(guard);
        list.extend(element);
        list.extend(parse(","));
    }
    TokenTree::Group(Group::new(Delimiter::Bracket, list))
}

/// The method that launches `f` over a grid, each of its instances `launched`: its parameters
/// those of `f`, a pointer as the address on the GPU it is, `::kernite_cuda::Ptr`, and a number as
/// the number and not the name the kernel writes; and its compile-time arguments those of `f`,
/// which pick the instance. A cooperative kernel is launched cooperatively, every block of its
/// grid at once. The launcher binds `__kernite_grid` for itself, which a parameter of `f` may be named as well; a
/// compile-time argument may not, and `launched` is `None` for the launcher that never runs, as one of a kernel that
/// did not compile. Its result is [`launcher_result`]'s. Its body has the `places` of `f`'s code that name an item,
/// where the level `f` sets holds, as in a function of Rust's, `expect` too.
fn launcher(f: &Fn, launched: Option<&[Launched]>, top: &[TokenTree], places: TokenStream) -> TokenStream {
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
             #[doc = \" The kernel trusts the pointers it is given: each must be to a buffer or to unified memory \
             of the GPU the kernels are loaded on, as long as the kernel reads and writes through it. From the \
             launch until the kernel is done, the host must hold no slice of memory the kernel writes, and no \
             slice it may write of memory the kernel reads; and where `Cuda::concurrent_managed_access` is false, \
             the GPU not reaching managed memory while the host does, the host must hold no slice of managed \
             memory at all.\"]",
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
    let mut params = hygienic("&self, $__kernite_grid: ::kernite_cuda::Grid, ", launched.is_some());
    let mut args = vec![];
    for (param, colon) in written_params(f, top) {
        // Its name, before the colon.
        let name = &param[colon - 1];
        params.extend(param[..=colon].iter().cloned());
        let ty = &param[colon + 1..];
        match pointer(ty) {
            Some((_, to)) => {
                // Its address, by a cast. A call of `arg` is generic over the type of the pointer, so Rust
                // would say an error in the type it points to again, at the call.
                args.push(format!("(&raw const {name}) as *const ::core::ffi::c_void"));
                params.extend(ptr_type(ty, to));
            }
            None => {
                args.push(format!("::kernite_cuda::arg(&{name})"));
                params.extend(number_type(ty).or_else(|| half(ty)).unwrap_or_else(|| refused(ty)));
            }
        }
        params.extend(parse(","));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, params))]);
    out.extend(parse("->"));
    out.extend(launcher_result(f, top, parse("::core::result::Result<(), ::kernite_cuda::Error>")));
    let how = if f.cooperative.is_some() { "launch_cooperative" } else { "launch" };
    let never = format!("{}::core::unreachable!()", never(f, top));
    let mut checks = asks(f, top, host);
    checks.extend(unnamed(places));
    out.extend(body(f, launched, top, checks, &never, |index| {
        format!("unsafe {{ self.module.{how}({index}, $__kernite_grid, &[{}]) }}", args.join(", "))
    }));
    out
}

/// The method that says how many blocks a grid of `f`, a cooperative kernel, has at most, for each
/// of its instances `launched`: as many as the GPU runs at once, of so many threads and so much of
/// the launch's shared memory. It keeps what `f` says of whether it is there, and nothing else:
/// `cfg`, `deprecated` and `doc(hidden)`, each alone or in a `cfg_attr(..)`. It has docs of its
/// own, not those of `f`. The launcher alone keeps what `f` says of lints, of its code or of its
/// symbol, as `allow`, `target_feature` and `export_name` do. The method's name and generics are
/// where `f` has its own, but of the macro's span: Rust lints neither, and says a lint of a
/// compile-time argument once, at the launcher. The method binds `__kernite_threads` and `__kernite_shared` for itself,
/// which a compile-time argument of `f` may not be named, and `launched` is `None` for the method that never runs, as
/// one of a kernel that did not compile.
fn blocks(f: &Fn, launched: Option<&[Launched]>, top: &[TokenTree]) -> TokenStream {
    let mut out = parse(&format!(
        "/// How many blocks a grid of [`{0}`](Self::{0}) has at most, given the threads of each block\n\
         /// and the bytes of the launch's shared memory: as many as the GPU runs at once, for it runs\n\
         /// them all together.",
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
        "(&self, $__kernite_threads: ::core::primitive::u32, $__kernite_shared: ::core::primitive::u32) \
         -> ::core::result::Result<::core::primitive::u32, ::kernite_cuda::Error>",
        launched.is_some(),
    ));
    out.extend(body(f, launched, top, TokenStream::new(), "::core::unreachable!()", |index| {
        format!("self.module.resident({index}, $__kernite_threads, $__kernite_shared)")
    }));
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
