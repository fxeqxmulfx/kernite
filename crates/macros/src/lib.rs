//! `#[kernels]` on a module: each function in it that names an ABI is a kernel, or a function
//! kernels call, written in the language of `kgen-lang` and compiled for the machine the ABI
//! belongs to, in place.
//!
//! ```ignore
//! #[kgen_macros::kernels]
//! mod cpu {
//!     /// The sum of the eight floats at `xs`.
//!     pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
//!         let v: ymm;
//!         vmovups(v, [xs]);
//!         ret_sum(v);
//!     }
//!
//!     fn ret_sum(v: ymm) {
//!         let high: ymm;
//!         vextractf128(high.xmm, v, 1);
//!         ..
//!     }
//! }
//! ```
//!
//! - An `unsafe extern "sysv64" fn` is compiled for x86-64 and becomes an `#[unsafe(naked)]`
//!   function of the signature it declares, its body the kernel's `naked_asm!`; constants it
//!   reads are statics beside it. `LISTINGS` holds each one's listing, by name, for snapshots.
//! - An `unsafe extern "ptx-kernel" fn` is compiled to PTX, for NVIDIA GPUs. The module's are
//!   one PTX module, `PTX`, and `Kernels::load(&cuda)` has the driver compile it for the GPU
//!   it finds; then each kernel is launched by the method of its name, which takes a
//!   `kgen_cuda::Grid` and the kernel's parameters, a pointer as the `kgen_cuda::Ptr` a
//!   buffer on the GPU gives, and an `f16` or a `bf16` as kgen-cuda's, Rust having neither.
//!   A crate with GPU kernels depends on `kgen-cuda` for these.
//! - An `extern "ptx-func" fn` is a `.func` in `PTX`, which the module's GPU kernels and
//!   functions call, itself among them, as `call(r, gcd, (a, b))` does. It takes and
//!   returns PTX's types, `u32` or `f16x2`, or never returns, `-> !`, and leaves nothing
//!   behind for Rust, which does not call it.
//! - A `static` is memory of the module's on the GPU, declared once in `PTX`, which each of
//!   its GPU kernels names as it names memory it declares: `static COUNTS: global<[u32; 16]>;`
//!   in global memory, or `static WEIGHTS: constant<[f32; 4]> = [1.0, 0.5, 0.25, 2.0];` in
//!   constant memory, which the kernels only read. It holds what it says until something
//!   writes it, or zeros, and keeps what is written from one launch to the next. `Kernels` has
//!   a method of its name that gives it on the GPU, a `kgen_cuda::Global` of the type of
//!   Rust's its elements are read as, `u32` for a `b32`, for the host to read and write:
//!   `kernels.COUNTS().read()?`.
//! - A function without an ABI is a helper, inlined where a kernel calls it, and leaves
//!   nothing behind. Constants and every other item but a `static` stay as written, for Rust
//!   to use as well.
//!   A constant a kernel reads counts as used, as it would if Rust had seen the read: only one
//!   that nothing reads is `dead_code`. Rust has no `bf16`, and its `f16` is unstable, so a
//!   constant of either goes in the kernel that reads it, where `const` is the language's alone.
//!
//! ```ignore
//! #[kgen_macros::kernels]
//! mod gpu {
//!     /// `x[i] *= a` for each `i < n`.
//!     pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32, n: u32) {
//!         ..
//!     }
//! }
//!
//! let cuda = kgen_cuda::Cuda::new()?;
//! let xs = cuda.upload(&[1.0, 2.0, 3.0])?;
//! let kernels = gpu::Kernels::load(&cuda)?;
//! unsafe { kernels.scale(kgen_cuda::Grid::new(1, 32), xs.ptr(), 2.0, 3)? };
//! ```
//!
//! A kernel with const generics, integers or `bool`s, is compiled for each instance its
//! `#[instances(..)]` lists, each a kernel of its own, of values known now, the module's
//! constants among them: `#[instances(256, 1024)]` on `fn block_sums<const N: u32>`, or
//! `#[instances((3, true), (-2, false))]` for two. On x86 the kernel is a generic function,
//! `cpu::dot::<4>(x, y, n)`, and `LISTINGS` holds each instance's listing by its name,
//! `dot::<4>`; on a GPU its launcher is generic, `kernels.block_sums::<1024>(grid, x, sums)`.
//! An instance it does not list is a compile error where Rust compiles the call, in
//! `cargo build`; `cargo check` evaluates no generic function's constants, and misses it.
//!
//! A GPU kernel whose blocks wait for each other, at `grid.sync()`, is `#[cooperative]`: its
//! launcher launches it cooperatively, every block of the grid at once, so the grid has at
//! most as many blocks as the GPU runs at once, which the method of its name and `_blocks`
//! says, of so many threads a block and so many bytes of the launch's shared memory:
//! `kernels.normalize_blocks(256, 0)?`. A grid of more is an error, and so is a GPU before
//! Pascal's, `sm_60`, which launches none cooperatively.
//!
//! A GPU kernel is named in PTX as C++ names a function in namespaces, those of the
//! workspace, the crate and its modules, mangled under Itanium's ABI:
//! `_ZN4kgen6my_app3gpu5scaleE`, which `cu++filt`, Nsight and `cuda-gdb` show as
//! `kgen::my_app::gpu::scale`, an instance as a template's,
//! `kgen::my_app::gpu::block_sums<(unsigned int)1024>`, and a static as a variable of the
//! namespaces, `kgen::my_app::gpu::COUNTS`. The workspace is named as the
//! directory of its `Cargo.toml` is, or as the manifest names it, which a clone keeps
//! wherever it is; `name = ""` leaves it out. A package of no workspace is named as the
//! package is.
//!
//! ```toml
//! [workspace.metadata.kgen]
//! name = "kgen"
//! ```
//!
//! A kernel is `unsafe`: it trusts the pointers it is given. One that does not compile is a
//! `compile_error!` where the trouble is, beside a stub of its signature or its launcher, so
//! that its callers are not errors as well.

extern crate proc_macro;

use std::path::{Path, PathBuf};

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

use kgen::lang::{self, Delim, Fn, ItemKind, Tree, Value};
use kgen::{Abi, Static};

/// See the crate documentation.
#[proc_macro_attribute]
pub fn kernels(attr: TokenStream, item: TokenStream) -> TokenStream {
    if let Some(t) = attr.into_iter().next() {
        return error(t.span(), "`#[kernels]` takes no arguments");
    }
    let tokens: Vec<TokenTree> = item.into_iter().collect();
    match tokens.split_last() {
        Some((TokenTree::Group(body), head))
            if body.delimiter() == Delimiter::Brace && head.iter().any(|t| is_ident(t, "mod")) =>
        {
            let mut out: TokenStream = head.iter().cloned().collect();
            let mut inner = Group::new(Delimiter::Brace, expand(body.stream()));
            inner.set_span(body.span());
            out.extend([TokenTree::Group(inner)]);
            out
        }
        _ => {
            let span = tokens.first().map_or_else(Span::call_site, TokenTree::span);
            error(span, "`#[kernels]` goes on a module written in braces, `mod name { .. }`, where the kernels are")
        }
    }
}

/// Where the module's path goes in its PTX and in its kernels' names. A macro is not told
/// the module it is in, so `kgen_cuda::__named!` writes `module_path!()` there once Rust
/// knows it, as C++ mangles a namespace's names.
const HOLE: &str = "\0";

/// What a kernel's instance is: its compile-time arguments, none for a kernel that takes
/// none, and its index among the module's kernels.
type Launched = (Vec<Value>, usize);

/// What the module's braces hold once its kernels are compiled.
fn expand(body: TokenStream) -> TokenStream {
    let top = flatten(body);
    let mut spans = vec![];
    let trees: Vec<Tree> = top.iter().map(|t| tree(t, &mut spans)).collect();
    let at = |s: lang::Span| spans.get(s.0 as usize).copied().unwrap_or_else(Span::call_site);
    let module = match lang::module(&trees) {
        Ok(module) => module,
        Err(e) => return error(at(e.span), &e.message),
    };
    // The GPU kernels' namespaces: the workspace's, then the module's path.
    let (workspace, manifest) = workspace();
    let names: Vec<&str> = [workspace.as_str()].into_iter().filter(|n| !n.is_empty()).collect();
    let scope = kgen::ptx_scope(&names) + HOLE;
    let mut out = TokenStream::new();
    let (mut listings, mut entries, mut launchers, mut read) = (vec![], vec![], vec![], vec![]);
    // The first function for a GPU, or static, where an error of the module as a whole goes;
    // the symbol of each kernel it launches, by index; and each static, compiled or not.
    let (mut first, mut symbols, mut memory) = (None, vec![], vec![]);
    let mut statics = kgen::ptx_statics(&module, &scope).into_iter();
    for item in &module.items {
        let f = match &item.kind {
            ItemKind::Const(_) | ItemKind::Other => {
                out.extend(top[item.tokens.clone()].iter().cloned());
                continue;
            }
            ItemKind::Static(s) => {
                first.get_or_insert(s.span);
                match statics.next().expect("each static is compiled") {
                    Ok(made) => {
                        read.extend(made.consts);
                        memory.push((s, Some(made.code)));
                    }
                    Err(e) => {
                        out.extend(error(at(e.span), &e.message));
                        memory.push((s, None));
                    }
                }
                continue;
            }
            ItemKind::Fn(f) => f,
        };
        let instances = lang::instances(&module, f);
        let cooperative = kgen::cooperative(f);
        let Some((name, abi_span)) = &f.abi else {
            // A helper is checked where it is called; one that does not parse is an error
            // whether it is called or not, as one that lists instances is, or says it is
            // cooperative.
            let errors = [f.body.as_ref().err(), instances.as_ref().err(), cooperative.as_ref().err()];
            for e in errors.into_iter().flatten() {
                out.extend(error(at(e.span), &e.message));
            }
            continue;
        };
        let Some(abi) = Abi::of(name) else {
            let message = format!(
                "kgen compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \"ptx-kernel\"` ones, for \
                 NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"{name}\"` is none of \
                 these"
            );
            out.extend(error(at(*abi_span), &message));
            out.extend(stub(f, &top));
            continue;
        };
        // A function the kernels call is called by nothing else, so it promises Rust nothing.
        let trusted = abi == Abi::PtxFunc || top[f.head.clone()].iter().any(|t| is_ident(t, "unsafe"));
        if !trusted {
            out.extend(error(at(f.span), "a kernel trusts the pointers it is given, so it is an `unsafe fn`"));
        }
        if abi != Abi::Sysv64 {
            first.get_or_insert(f.span);
        }
        // A `#[cooperative]` that is no GPU kernel's is said once, not for each instance.
        let instances = match cooperative.and(instances) {
            Ok(instances) => instances,
            Err(e) => {
                out.extend(error(at(e.span), &e.message));
                match abi {
                    Abi::Sysv64 => out.extend(stub(f, &top)),
                    Abi::PtxKernel => launchers.push((f, None)),
                    Abi::PtxFunc => {}
                }
                continue;
            }
        };
        // Each instance compiled, or the errors of those that do not compile.
        let mut errors = vec![];
        match abi {
            Abi::Sysv64 => {
                let mut compiled = vec![];
                for args in instances {
                    match kgen::x86(&module, f, &args) {
                        Ok(kernel) => {
                            read.extend(kernel.consts);
                            compiled.push((args, kernel.code));
                        }
                        Err(e) => errors.push(e),
                    }
                }
                if trusted && errors.is_empty() {
                    out.extend(naked(f, &compiled, &top));
                    for (args, listing) in &compiled {
                        listings.push((lang::instance(f, args).unwrap_or_default(), listing.text()));
                    }
                } else {
                    out.extend(stub(f, &top));
                }
            }
            Abi::PtxKernel | Abi::PtxFunc => {
                let mut compiled = vec![];
                for args in instances {
                    match kgen::ptx(&module, f, &args, &scope) {
                        Ok(kernel) => {
                            read.extend(kernel.consts);
                            compiled.push((args, kernel.code));
                        }
                        Err(e) => errors.push(e),
                    }
                }
                let ok = trusted && errors.is_empty();
                let mut launched = vec![];
                for (args, entry) in compiled {
                    if ok && abi == Abi::PtxKernel {
                        launched.push((args, symbols.len()));
                        symbols.push(entry.program.name.clone());
                    }
                    entries.push(entry);
                }
                if abi == Abi::PtxKernel {
                    launchers.push((f, ok.then_some(launched)));
                }
            }
        }
        for e in errors {
            out.extend(error(at(e.span), &e.message));
        }
    }
    out.extend(used(&module, &read));
    if !listings.is_empty() {
        let mut s = String::from(
            "/// Each x86 kernel's listing, by name, an instance's as `tile::<4>`: where its variables\n\
             /// live, and its code with its notes, for snapshots.\n#[allow(dead_code)]\n\
             pub const LISTINGS: &[(&str, &str)] = &[",
        );
        for (name, text) in &listings {
            s += &format!("({name:?}, {text:?}), ");
        }
        out.extend(parse(&(s + "];")));
    }
    if let Some(span) = first {
        let made: Vec<Static> = memory.iter().filter_map(|(_, m)| m.clone()).collect();
        let ptx = kgen::ptx_module(&entries, &made).unwrap_or_else(|e| {
            out.extend(error(at(span), &e));
            String::new()
        });
        out.extend(gpu(&launchers, &symbols, &memory, &ptx, &top, &at));
        // The workspace's name is in the PTX, so it is built again when it changes.
        if let Some(manifest) = manifest.as_deref().and_then(Path::to_str) {
            out.extend(parse(&format!("const _: &[u8] = ::core::include_bytes!({manifest:?});")));
        }
    }
    out
}

/// The workspace the crate is built in, by name, the first namespace its GPU kernels are in:
/// as its manifest has it, `name` in `[workspace.metadata.kgen]`, or the name of the directory
/// the manifest is in, with that manifest. A package of no workspace is one of its own, named
/// as the package is.
fn workspace() -> (String, Option<PathBuf>) {
    let package = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let Some(dir) = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from) else {
        return (package, None);
    };
    for dir in dir.ancestors() {
        let manifest = dir.join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&manifest) else { continue };
        if !text.lines().filter_map(table).any(|t| t == "workspace" || t.starts_with("workspace.")) {
            continue;
        }
        let named = metadata_name(&text).or_else(|| dir.file_name()?.to_str().map(String::from));
        return (named.unwrap_or(package), Some(manifest));
    }
    (package, None)
}

/// The table a line of a manifest opens, `[name]`, if it opens one.
fn table(line: &str) -> Option<&str> {
    line.trim().strip_prefix('[')?.split(']').next().map(str::trim)
}

/// `name = ".."` in `[workspace.metadata.kgen]` of `manifest`, if it is there.
fn metadata_name(manifest: &str) -> Option<String> {
    let mut inside = false;
    for line in manifest.lines() {
        if let Some(t) = table(line) {
            inside = t == "workspace.metadata.kgen";
            continue;
        }
        let Some((key, value)) = line.split_once('=').filter(|_| inside) else { continue };
        if key.trim() == "name" {
            let value = value.trim();
            let quote = value.chars().next().filter(|&c| c == '"' || c == '\'')?;
            return value[1..].split(quote).next().map(String::from);
        }
    }
    None
}

/// A `const _` that names each constant of `module` in `read`. The kernels read them, but Rust
/// never sees a kernel's body run, so it would find them unused.
fn used(module: &lang::Module, read: &[String]) -> TokenStream {
    let lets: String = module
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Const(c) if read.contains(&c.name) => Some(format!("let _ = {};\n", c.name)),
            _ => None,
        })
        .collect();
    if lets.is_empty() {
        return TokenStream::new();
    }
    parse(&format!("const _: () = {{\n{lets}}};"))
}

/// `PTX`, the module the GPU kernels are, and `Kernels`, which loads it on a GPU and has a
/// method to launch each kernel by its name: of each instance, for a kernel of compile-time
/// arguments; one that never runs for a kernel that did not compile, launched `None`. Each of
/// `symbols` is a kernel's name in `ptx`, by its index. Each static of `memory` has a method
/// of its name too, which gives its memory on the GPU, or never runs, if it did not compile;
/// and each cooperative kernel has one that says how many blocks its grid has at most, unless
/// a kernel or a static has that name, which is an error where `at` says.
fn gpu(
    launchers: &[(&Fn, Option<Vec<Launched>>)],
    symbols: &[String],
    memory: &[(&lang::Static, Option<Static>)],
    ptx: &str,
    top: &[TokenTree],
    at: &dyn std::ops::Fn(lang::Span) -> Span,
) -> TokenStream {
    let names: Vec<String> = symbols.iter().map(|s| named(s)).collect();
    let globals: Vec<String> = memory.iter().filter_map(|(_, m)| Some(named(&m.as_ref()?.memory.name))).collect();
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
fn accessor(s: &lang::Static, made: Option<(&Static, usize)>, top: &[TokenTree]) -> TokenStream {
    let mut out: TokenStream = s.attrs.clone().map(|i| top[i].clone()).collect();
    out.extend(parse("#[allow(non_snake_case)]"));
    out.extend(top[s.vis.clone()].iter().cloned());
    out.extend(parse("fn"));
    // Its name as written, after `static`.
    out.extend([top[s.vis.end + 1].clone()]);
    out.extend(parse(&match made {
        Some((m, i)) => {
            let ty = match m.elem.host().expect("a static is of a type the host reads") {
                ty @ ("f16" | "bf16") => format!("::kgen_cuda::{ty}"),
                ty => ty.to_string(),
            };
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
    let TokenTree::Group(group) = &top[f.params_at] else { unreachable!("parameters are in parentheses") };
    let tokens = flatten(group.stream());
    for param in tokens.split(|t| is_punct(t, ',')).filter(|p| !p.is_empty()) {
        let colon = param.iter().position(|t| is_punct(t, ':')).expect("a parameter has a type");
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
    let args: Vec<String> = f.params.iter().map(|p| format!("::kgen_cuda::arg(&{})", p.name)).collect();
    let how = if f.cooperative.is_some() { "launch_cooperative" } else { "launch" };
    out.extend(body(f, launched, |index| {
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
    out.extend(body(f, launched, |index| format!("self.module.resident({index}, threads, shared)")));
    out
}

/// The name of the method that says how many blocks a grid of `f` has at most.
fn blocks_name(f: &Fn) -> String {
    format!("{}_blocks", f.name.trim_start_matches("r#"))
}

/// `f`'s visibility as written, `pub` or `pub(..)`, or none.
fn visibility<'a>(f: &Fn, top: &'a [TokenTree]) -> &'a [TokenTree] {
    let head = &top[f.head.clone()];
    let vis = match head {
        [pub_, TokenTree::Group(g), ..] if is_ident(pub_, "pub") && g.delimiter() == Delimiter::Parenthesis => 2,
        [pub_, ..] if is_ident(pub_, "pub") => 1,
        _ => 0,
    };
    &head[..vis]
}

/// The body of a method of `f`, which does `then` with the index of the instance among
/// `launched` its compile-time arguments pick; or never runs, for a kernel that did not
/// compile.
fn body(f: &Fn, launched: Option<&[Launched]>, then: impl std::ops::Fn(&str) -> String) -> TokenStream {
    let body = match launched {
        Some([(args, i)]) if args.is_empty() => then(&i.to_string()),
        Some(instances) => {
            let arms: Vec<(Vec<Value>, String)> = instances.iter().map(|(a, i)| (a.clone(), i.to_string())).collect();
            format!("let index = {};\n{}", pick(f, &arms), then("index"))
        }
        None => "::core::unreachable!()".to_string(),
    };
    [TokenTree::Group(Group::new(Delimiter::Brace, parse(&body)))].into_iter().collect()
}

/// `const { match .. }` on `f`'s compile-time arguments: what `arms` gives for each instance, by
/// its arguments. Another is no instance, and Rust refuses to compile the call.
fn pick(f: &Fn, arms: &[(Vec<Value>, String)]) -> String {
    let tuple = |xs: Vec<String>| if xs.len() == 1 { xs.join("") } else { format!("({})", xs.join(", ")) };
    let on = tuple(f.generics.iter().map(|g| g.name.clone()).collect());
    let mut out = format!("const {{ match {on} {{ ");
    let mut listed = vec![];
    for (args, then) in arms {
        out += &format!("{} => {then}, ", tuple(args.iter().map(Value::to_string).collect()));
        listed.push(format!("`{}`", lang::instance(f, args).unwrap_or_default()));
    }
    let message = format!("`{}` is compiled for {} alone, as its `#[instances(..)]` lists", f.name, listed.join(", "));
    out + &format!("_ => ::core::panic!({message:?}) }} }}")
}

/// `f` as an `#[unsafe(naked)]` function whose body is the listing of its one instance, and
/// the statics its constants are, in a module beside it. For a function of compile-time
/// arguments, each instance is such a function in that module, and `f` is one that calls the
/// instance its arguments pick.
fn naked(f: &Fn, instances: &[(Vec<Value>, kgen::Listing)], top: &[TokenTree]) -> TokenStream {
    let module = format!("__kgen_{}", f.name);
    let mut out = attrs(f, top);
    let mut items = String::new();
    let mut statics = vec![];
    if let [(args, listing)] = instances
        && args.is_empty()
    {
        out.extend(parse("#[unsafe(naked)]"));
        out.extend(signature(f, top));
        out.extend([TokenTree::Group(Group::new(Delimiter::Brace, parse(&asm(listing, &format!("{module}::C")))))]);
        statics.push(("C".to_string(), listing));
    } else {
        // The function's type, and the names of its parameters, which it passes on.
        let params: TokenStream = [top[f.params_at].clone()].into_iter().collect();
        let ret: TokenStream = f.ret.as_ref().map_or_else(TokenStream::new, |r| {
            parse("->").into_iter().chain(top[r.tokens.clone()].iter().cloned()).collect()
        });
        let TokenTree::Group(group) = &top[f.params_at] else { unreachable!("parameters are in parentheses") };
        let tokens = flatten(group.stream());
        let tys: Vec<String> = tokens
            .split(|t| is_punct(t, ','))
            .filter(|p| !p.is_empty())
            .map(|p| p.iter().skip_while(|t| !is_punct(t, ':')).skip(1).cloned().collect::<TokenStream>().to_string())
            .collect();
        let ty = format!("unsafe extern \"sysv64\" fn({}) {ret}", tys.join(", "));
        let names: Vec<&str> = f.params.iter().map(|p| p.name.as_str()).collect();
        let mut arms = vec![];
        for (k, (args, listing)) in instances.iter().enumerate() {
            items += &format!(
                "#[unsafe(naked)]\npub unsafe extern \"sysv64\" fn I{k}{params} {ret} {{ {} }}\n",
                asm(listing, &format!("I{k}C"))
            );
            statics.push((format!("I{k}C"), listing));
            arms.push((args.clone(), format!("{module}::I{k} as {ty}")));
        }
        out.extend(parse("#[inline(always)]"));
        out.extend(signature(f, top));
        let body = format!("unsafe {{ ({})({}) }}", pick(f, &arms), names.join(", "));
        out.extend([TokenTree::Group(Group::new(Delimiter::Brace, parse(&body)))]);
    }
    let mut aligns: Vec<u32> = statics.iter().flat_map(|(_, l)| l.consts.iter().map(|c| c.align)).collect();
    aligns.sort();
    aligns.dedup();
    for a in &aligns {
        items += &format!("#[repr(C, align({a}))]\npub struct A{a}<T>(pub T);\n");
    }
    for (prefix, listing) in &statics {
        for (i, c) in listing.consts.iter().enumerate() {
            let bytes: Vec<String> = c.bytes.iter().map(u8::to_string).collect();
            items += &format!(
                "pub static {prefix}{i}: A{a}<[u8; {n}]> = A{a}([{bytes}]);\n",
                a = c.align,
                n = c.bytes.len(),
                bytes = bytes.join(", ")
            );
        }
    }
    if !items.is_empty() {
        out.extend(parse(&format!(
            "#[doc(hidden)]\n#[allow(dead_code, non_snake_case)]\nmod {module} {{\n{items}}}\n"
        )));
    }
    out
}

/// The body of a naked function whose code is `listing`'s, its constants the statics
/// `{statics}0`, `{statics}1` and on.
fn asm(listing: &kgen::Listing, statics: &str) -> String {
    let mut asm = String::from("::core::arch::naked_asm!(");
    for line in listing.code() {
        asm += &format!("{line:?}, ");
    }
    for i in 0..listing.consts.len() {
        asm += &format!("c{i} = sym {statics}{i}, ");
    }
    asm + ")"
}

/// `f` with a body that never runs, for a kernel that does not compile.
fn stub(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out = attrs(f, top);
    out.extend(parse("#[allow(unused_variables)]"));
    out.extend(signature(f, top));
    out.extend(parse("{ ::core::unreachable!() }"));
    out
}

/// `f`'s attributes as written, but `#[instances(..)]` and `#[cooperative]`, which Rust does not
/// know.
fn attrs(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let instances = f.instances.as_ref().map_or(0..0, |i| i.tokens.clone());
    let cooperative = f.cooperative.as_ref().map_or(0..0, |c| c.tokens.clone());
    f.attrs.clone().filter(|i| !instances.contains(i) && !cooperative.contains(i)).map(|i| top[i].clone()).collect()
}

/// `f`'s signature as written, from its visibility to its result.
fn signature(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let end = f.ret.as_ref().map_or(f.params_at + 1, |r| r.tokens.end);
    top[f.head.start..end].iter().cloned().collect()
}

/// The tokens of `stream`, with the invisible groups a macro's fragments arrive in opened.
fn flatten(stream: TokenStream) -> Vec<TokenTree> {
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
fn tree(t: &TokenTree, spans: &mut Vec<Span>) -> Tree {
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
fn error(span: Span, message: &str) -> TokenStream {
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
fn host(ty: &[TokenTree]) -> TokenStream {
    match ty {
        [t] if is_ident(t, "f16") || is_ident(t, "bf16") => parse(&format!("::kgen_cuda::{t}")),
        _ => ty.iter().cloned().collect(),
    }
}

fn parse(s: &str) -> TokenStream {
    s.parse().expect("kgen-macros writes tokens that parse")
}

fn is_ident(t: &TokenTree, name: &str) -> bool {
    matches!(t, TokenTree::Ident(i) if i.to_string() == name)
}

fn is_punct(t: &TokenTree, ch: char) -> bool {
    matches!(t, TokenTree::Punct(p) if p.as_char() == ch)
}
