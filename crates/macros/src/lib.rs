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
//!   The module's kernels call each other, themselves among them, as `call(r, gcd, (a, b))`
//!   does, or at the address `lea(fp, [gcd])` takes; all but a `#[cooperative]` one, which
//!   its launcher runs. One that never returns, `-> !`, ends each way out in `ud2()` or in a
//!   call of one that does not.
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
//! - A `static` is memory of the module's, which each of its kernels names as it names memory
//!   it declares: `static COUNTS: global<[u32; 16]>;` in global memory, or
//!   `static WEIGHTS: constant<[f32; 4]> = [1.0, 0.5, 0.25, 2.0];` in constant memory, which
//!   the kernels only read. It holds what it says until something writes it, or zeros, and
//!   keeps what is written from one call or launch to the next. It is its kernels' memory:
//!   - In a module of GPU kernels it is on the GPU, declared once in `PTX`. `Kernels` has a
//!     method of its name that gives it there, a `kgen_cuda::Global` of the type of Rust's its
//!     elements are read as, `u32` for a `b32`, for the host to read and write:
//!     `kernels.COUNTS().read()?`. A module of statics alone is the GPU's.
//!   - In a module of x86 kernels it is in the process's memory, which the kernels reach at a
//!     displacement from `rip`, `[COUNTS + 8]`: a `static` of Rust's, a `kgen_cpu::Global` of
//!     its elements, for the host to read and write: `cpu::COUNTS.read()`. A crate with such a
//!     module depends on `kgen-cpu` for it.
//!   - A module of both kinds of kernel and a static is an error: it is two modules, one of
//!     each kind of kernel, with the statics each names.
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
//! A module of kernels takes the consts it shares with others from a module `#[consts]` is
//! on, which it names by its path, as Rust reads one where the module is:
//! `#[kernels(consts = crate::model::sizes)]`, or `consts = sizes` for one beside it. Its
//! kernels read them as the module's own, and Rust sees them there by a `use` the macro
//! writes; a const, function or static of the module's own of one of their names is an error.
//! A module of consts holds `pub` consts alone, of what kernels read: numbers, floats,
//! `bool`s, arrays of them, and expressions of these. It is of the same crate, for the macro
//! that hands its consts on is the crate's alone.
//!
//! ```
//! #[kgen_macros::consts]
//! pub mod sizes {
//!     /// How far a value moves.
//!     pub const STEP: u64 = 3;
//! }
//!
//! #[kgen_macros::kernels(consts = crate::sizes)]
//! pub mod cpu {
//!     /// `x` a step on.
//!     pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
//!         add(x, STEP);
//!         ret(x);
//!     }
//!
//!     /// Two steps, as Rust counts them.
//!     pub const TWO: u64 = 2 * STEP;
//! }
//!
//! fn main() {
//!     assert_eq!(unsafe { cpu::step(1) }, 4);
//!     assert_eq!(cpu::TWO, 6);
//! }
//! ```
//!
//! ```compile_fail
//! #[kgen_macros::consts]
//! pub mod sizes {
//!     pub const STEP: u64 = 3;
//! }
//!
//! #[kgen_macros::kernels(consts = crate::sizes)]
//! pub mod cpu {
//!     // `STEP` names one of the consts the module takes from `crate::sizes` as well.
//!     pub const STEP: u64 = 4;
//! }
//!
//! fn main() {}
//! ```
//!
//! A GPU kernel whose blocks wait for each other, at `grid.sync()`, is `#[cooperative]`: its
//! launcher launches it cooperatively, every block of the grid at once, so the grid has at
//! most as many blocks as the GPU runs at once, which the method of its name and `_blocks`
//! says, of so many threads a block and so many bytes of the launch's shared memory:
//! `kernels.normalize_blocks(256, 0)?`. A grid of more is an error, and so is a GPU before
//! Pascal's, `sm_60`, which launches none cooperatively.
//!
//! An x86 kernel whose threads wait for each other is `#[cooperative]` too, and runs on the
//! threads of a grid, each given its place there, `grid.thread` of `grid.threads`: it becomes
//! the function of its name that takes a `kgen_cpu::Grid` and the kernel's parameters, runs
//! the kernel on every thread at once, and returns when all have ended:
//! `cpu::normalize(kgen_cpu::Grid::new(4), x.as_mut_ptr(), sums.as_mut_ptr(), n)?`. A thread
//! spins at `grid.sync()` until the last comes, so a grid has at most as many threads as the
//! machine runs at once, `kgen_cpu::most()`, and more is an error. A crate with such kernels
//! depends on `kgen-cpu` for these.
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
//! A kernel is `unsafe`: it trusts the pointers it is given, and that nothing else touches the
//! memory of the module's it names while it runs, as an x86 kernel's safety contract says, by
//! name, of the statics it and the functions it calls name. One that does not compile is a
//! `compile_error!` where the trouble is, beside a stub of its signature or its launcher, so
//! that its callers are not errors as well.

extern crate proc_macro;

mod expand;
mod gpu;
mod rust;
mod workspace;
mod x86;

use proc_macro::{Delimiter, Group, Span, TokenStream, TokenTree};

use kgen::lang::{self, ItemKind, Tree};

use crate::expand::expand;
use crate::rust::{
    Site, attrs_len, braced, error, flatten, inner_attribute, is_ident, is_path, is_punct, module, parse, tree,
};

/// See the crate documentation.
#[proc_macro_attribute]
pub fn kernels(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr: Vec<TokenTree> = attr.into_iter().collect();
    let tokens: Vec<TokenTree> = item.into_iter().collect();
    let Some((braces, head)) = braced(&tokens) else {
        let span = tokens.first().map_or_else(Span::call_site, TokenTree::span);
        return error(
            span,
            "`#[kernels]` goes on a module written in braces, `mod name { .. }`, where the kernels are",
        );
    };
    if attr.is_empty() {
        return module(head, braces, expand(None, braces.stream()));
    }
    let path = match &attr[..] {
        [c, eq, path @ ..] if is_ident(c, "consts") && is_punct(eq, '=') && is_path(path) => path,
        _ => {
            let message = "`#[kernels]` takes the module of consts its kernels read, as `consts = crate::model::sizes`, \
                           a module `#[kgen_macros::consts]` is on; or nothing";
            return error(attr[0].span(), message);
        }
    };
    // The macro the consts' module has calls `__kernels!` back with the consts, their module's
    // path, and this module.
    let mut out: TokenStream = path.iter().cloned().collect();
    out.extend(parse("::__kgen_consts!"));
    let mut given: TokenStream =
        [TokenTree::Group(Group::new(Delimiter::Parenthesis, path.iter().cloned().collect()))].into_iter().collect();
    given.extend(tokens);
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, given))]);
    out
}

/// See the crate documentation.
#[proc_macro_attribute]
pub fn consts(attr: TokenStream, item: TokenStream) -> TokenStream {
    if let Some(t) = attr.into_iter().next() {
        return error(t.span(), "`#[consts]` takes no arguments");
    }
    let tokens: Vec<TokenTree> = item.into_iter().collect();
    let Some((braces, head)) = braced(&tokens) else {
        let span = tokens.first().map_or_else(Span::call_site, TokenTree::span);
        return error(
            span,
            "`#[consts]` goes on a module written in braces, `mod name { .. }`, of the consts kernels read",
        );
    };
    let top = flatten(braces.stream());
    let mut spans = vec![];
    let trees: Vec<Tree> = top.iter().map(|t| tree(t, &mut spans)).collect();
    let at = |s: lang::Span| spans.get(s.0 as usize).copied().unwrap_or_else(Span::call_site);
    // The module as it is, for Rust; and the consts kgen reads, for the kernels.
    let mut inner: TokenStream = top.iter().cloned().collect();
    let mut given = TokenStream::new();
    match lang::module(&trees) {
        Err(e) => inner.extend(error(at(e.span), &e.message)),
        Ok(module) => {
            for item in &module.items {
                let tokens = &top[item.tokens.clone()];
                let named = &tokens[attrs_len(tokens)..];
                match &item.kind {
                    ItemKind::Const(_) if !named.first().is_some_and(|t| is_ident(t, "pub")) => {
                        let message = "a const of a module of consts is `pub`, for the modules of kernels that take \
                                       it to see";
                        inner.extend(error(named[0].span(), message));
                    }
                    ItemKind::Const(c) => match lang::constant(&module, c) {
                        Ok(_) => given.extend(tokens.iter().cloned()),
                        Err(e) => inner.extend(error(at(e.span), &e.message)),
                    },
                    // The module's own attribute, `#![..]`, is Rust's.
                    ItemKind::Other if inner_attribute(tokens) => {}
                    _ => {
                        let message = "a module of consts holds the consts kernels read, as `pub const E: usize = \
                                       16;`, and nothing else";
                        inner.extend(error(named.first().unwrap_or(&tokens[0]).span(), message));
                    }
                }
            }
        }
    }
    // The macro `#[kernels(consts = ..)]` calls, by the module's path, which calls `__kernels!`
    // back with the consts that compile first.
    let mut call = parse("::kgen_macros::__kernels!");
    let mut args: TokenStream = [TokenTree::Group(Group::new(Delimiter::Bracket, given))].into_iter().collect();
    args.extend(parse("$($module)*"));
    call.extend([TokenTree::Group(Group::new(Delimiter::Brace, args))]);
    let mut rule = parse("($($module:tt)*) =>");
    rule.extend([TokenTree::Group(Group::new(Delimiter::Brace, call))]);
    rule.extend(parse(";"));
    inner.extend(parse("#[doc(hidden)] #[allow(unused_macros)] macro_rules! __kgen_consts"));
    inner.extend([TokenTree::Group(Group::new(Delimiter::Brace, rule))]);
    inner.extend(parse("#[doc(hidden)] #[allow(unused_imports)] pub(crate) use __kgen_consts;"));
    module(head, braces, inner)
}

/// What `#[kernels(consts = path)]` on a module becomes once the macro of the module of consts
/// at `path` has handed it their tokens: `[consts] (path) module`.
#[doc(hidden)]
#[proc_macro]
pub fn __kernels(input: TokenStream) -> TokenStream {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    let [TokenTree::Group(consts), TokenTree::Group(path), rest @ ..] = &tokens[..] else {
        return error(Span::call_site(), "`__kernels!` is how `#[kernels(consts = ..)]` is given its consts");
    };
    let Some((braces, head)) = braced(rest) else {
        return error(Span::call_site(), "`__kernels!` is how `#[kernels(consts = ..)]` is given its consts");
    };
    let path: Vec<TokenTree> = path.stream().into_iter().collect();
    // What the macro writes resolves where the module is, not inside the consts' macro.
    let site = Site::at(path.first().map_or_else(Span::call_site, TokenTree::span));
    let inner = expand(Some((consts.stream(), &path)), braces.stream());
    drop(site);
    module(head, braces, inner)
}
