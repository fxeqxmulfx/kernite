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
//! - A function without an ABI is a helper, inlined where a kernel calls it, and leaves
//!   nothing behind. Constants and every other item stay as written, for Rust to use as well.
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
//! A kernel is `unsafe`: it trusts the pointers it is given. One that does not compile is a
//! `compile_error!` where the trouble is, beside a stub of its signature or its launcher, so
//! that its callers are not errors as well.

extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

use kgen::Abi;
use kgen::lang::{self, Delim, Fn, ItemKind, Tree};

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
    let mut out = TokenStream::new();
    let (mut listings, mut entries, mut launchers, mut read) = (vec![], vec![], vec![], vec![]);
    // The first function for a GPU, where an error of the module as a whole goes; and how many
    // kernels it launches.
    let (mut first, mut launched) = (None, 0);
    for item in &module.items {
        let f = match &item.kind {
            ItemKind::Const(_) | ItemKind::Other => {
                out.extend(top[item.tokens.clone()].iter().cloned());
                continue;
            }
            ItemKind::Fn(f) => f,
        };
        let Some((name, abi_span)) = &f.abi else {
            // A helper is checked where it is called; one that does not parse is an error
            // whether it is called or not.
            if let Err(e) = &f.body {
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
        match abi {
            Abi::Sysv64 => match kgen::x86(&module, f) {
                Ok(kernel) => {
                    if trusted {
                        out.extend(naked(f, &kernel.code, &top));
                        listings.push((f.name.clone(), kernel.code.text()));
                    } else {
                        out.extend(stub(f, &top));
                    }
                    read.extend(kernel.consts);
                }
                Err(e) => {
                    out.extend(error(at(e.span), &e.message));
                    out.extend(stub(f, &top));
                }
            },
            Abi::PtxKernel => match kgen::ptx(&module, f) {
                Ok(kernel) => {
                    if trusted {
                        launchers.push((f, Some(launched)));
                        launched += 1;
                        entries.push(kernel.code);
                    } else {
                        launchers.push((f, None));
                    }
                    read.extend(kernel.consts);
                }
                Err(e) => {
                    out.extend(error(at(e.span), &e.message));
                    launchers.push((f, None));
                }
            },
            Abi::PtxFunc => match kgen::ptx(&module, f) {
                Ok(func) => {
                    entries.push(func.code);
                    read.extend(func.consts);
                }
                Err(e) => out.extend(error(at(e.span), &e.message)),
            },
        }
    }
    out.extend(used(&module, &read));
    if !listings.is_empty() {
        let mut s = String::from(
            "/// Each x86 kernel's listing, by name: where its variables live, and its code with\n\
             /// its notes, for snapshots.\n#[allow(dead_code)]\npub const LISTINGS: &[(&str, &str)] = &[",
        );
        for (name, text) in &listings {
            s += &format!("({name:?}, {text:?}), ");
        }
        out.extend(parse(&(s + "];")));
    }
    if let Some(span) = first {
        let ptx = kgen::ptx_module(&entries).unwrap_or_else(|e| {
            out.extend(error(at(span), &e));
            String::new()
        });
        out.extend(gpu(&launchers, &ptx, &top));
    }
    out
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
/// method to launch each kernel by its name: one that never runs for a kernel that did not
/// compile, kernel `None`.
fn gpu(launchers: &[(&Fn, Option<usize>)], ptx: &str, top: &[TokenTree]) -> TokenStream {
    let names: Vec<String> =
        launchers.iter().filter(|(_, index)| index.is_some()).map(|(f, _)| format!("{:?}", f.name)).collect();
    let mut out = parse(&format!(
        "/// The module's GPU kernels, in PTX, for the CUDA driver to compile for the GPU it finds.\n\
         #[allow(dead_code)]\n\
         pub const PTX: &str = {ptx:?};\n\
         /// The module's GPU kernels, loaded on a GPU: each is launched by the method of its name.\n\
         #[allow(dead_code)]\n\
         pub struct Kernels {{\n    module: ::kgen_cuda::Module,\n}}\n\
         #[allow(dead_code)]\n\
         impl Kernels",
    ));
    let mut methods = parse(&format!(
        "/// The kernels, compiled for `cuda`'s GPU by its driver.\n\
         pub fn load(cuda: &::kgen_cuda::Cuda) -> ::core::result::Result<Kernels, ::kgen_cuda::Error> {{\n\
             ::core::result::Result::Ok(Kernels {{ module: ::kgen_cuda::Module::load(cuda, PTX, &[{}])? }})\n\
         }}",
        names.join(", ")
    ));
    for (f, index) in launchers {
        methods.extend(launcher(f, *index, top));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, methods))]);
    out
}

/// The method that launches `f`, kernel `index` of the module, over a grid: its parameters
/// those of `f`, a pointer as the address on the GPU it is, `::kgen_cuda::Ptr`.
fn launcher(f: &Fn, index: Option<usize>, top: &[TokenTree]) -> TokenStream {
    let mut out: TokenStream = top[f.attrs.clone()].iter().cloned().collect();
    if !out.to_string().contains("# Safety") {
        out.extend(parse(
            "#[doc = \"\"]\n#[doc = \" # Safety\"]\n#[doc = \"\"]\n\
             #[doc = \" The kernel trusts the pointers it is given: each must be to a buffer as long as the \
             kernel reads and writes through it.\"]",
        ));
    }
    out.extend(parse(if index.is_some() {
        "#[allow(clippy::too_many_arguments)]"
    } else {
        "#[allow(clippy::too_many_arguments, unused_variables)]"
    }));
    // Its visibility, `pub` or `pub(..)`, then `unsafe fn` and its name as written.
    let head = &top[f.head.clone()];
    let vis = match head {
        [pub_, TokenTree::Group(g), ..] if is_ident(pub_, "pub") && g.delimiter() == Delimiter::Parenthesis => 2,
        [pub_, ..] if is_ident(pub_, "pub") => 1,
        _ => 0,
    };
    out.extend(head[..vis].iter().cloned());
    out.extend(parse("unsafe fn"));
    out.extend([top[f.head.end + 1].clone()]);
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
    let body = match index {
        Some(i) => {
            let args: Vec<String> = f.params.iter().map(|p| format!("::kgen_cuda::arg(&{})", p.name)).collect();
            format!("unsafe {{ self.module.launch({i}, grid, &[{}]) }}", args.join(", "))
        }
        None => "::core::unreachable!()".to_string(),
    };
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, parse(&body)))]);
    out
}

/// `f` as an `#[unsafe(naked)]` function whose body is `listing`, and the statics its
/// constants are.
fn naked(f: &Fn, listing: &kgen::Listing, top: &[TokenTree]) -> TokenStream {
    let consts = format!("__kgen_{}", f.name);
    let mut out: TokenStream = top[f.attrs.clone()].iter().cloned().collect();
    out.extend(parse("#[unsafe(naked)]"));
    out.extend(signature(f, top));
    let mut asm = String::from("::core::arch::naked_asm!(");
    for line in listing.code() {
        asm += &format!("{line:?}, ");
    }
    for i in 0..listing.consts.len() {
        asm += &format!("c{i} = sym {consts}::C{i}, ");
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, parse(&(asm + ")"))))]);
    if !listing.consts.is_empty() {
        let mut aligns: Vec<u32> = listing.consts.iter().map(|c| c.align).collect();
        aligns.sort();
        aligns.dedup();
        let mut s = format!("#[doc(hidden)]\n#[allow(dead_code, non_snake_case)]\nmod {consts} {{\n");
        for a in &aligns {
            s += &format!("    #[repr(C, align({a}))]\n    pub struct A{a}<T>(pub T);\n");
        }
        for (i, c) in listing.consts.iter().enumerate() {
            let bytes: Vec<String> = c.bytes.iter().map(u8::to_string).collect();
            s += &format!(
                "    pub static C{i}: A{a}<[u8; {n}]> = A{a}([{bytes}]);\n",
                a = c.align,
                n = c.bytes.len(),
                bytes = bytes.join(", ")
            );
        }
        out.extend(parse(&(s + "}\n")));
    }
    out
}

/// `f` with a body that never runs, for a kernel that does not compile.
fn stub(f: &Fn, top: &[TokenTree]) -> TokenStream {
    let mut out: TokenStream = top[f.attrs.clone()].iter().cloned().collect();
    out.extend(parse("#[allow(unused_variables)]"));
    out.extend(signature(f, top));
    out.extend(parse("{ ::core::unreachable!() }"));
    out
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
