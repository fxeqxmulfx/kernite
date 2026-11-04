//! `#[kernels]` on a module: each function in it that names an ABI is a kernel, or a function
//! kernels call, written in the language of `kernite-lang` and compiled for the machine the ABI
//! belongs to, in place.
//!
//! ```ignore
//! #[kernite_macros::kernels]
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
//!   `kernite_cuda::Grid` and the kernel's parameters, a pointer as the `kernite_cuda::Ptr` a
//!   buffer on the GPU gives, and an `f16` or a `bf16` as kernite-cuda's, Rust having neither.
//!   A crate with a module of GPU kernels needs a dependency named `kernite-cuda` for these, not
//!   renamed in `Cargo.toml`, and the error is at the module when it has none and `cfg` keeps one
//!   of its GPU items: a feature can bring `kernite-cuda` and keep them, and the crate builds
//!   without it.
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
//!     method of its name that gives it there, a `kernite_cuda::Global` of the type of Rust's its
//!     elements are read as, `u32` for a `b32`, for the host to read and write:
//!     `kernels.COUNTS().read()?`. A module of statics alone is the GPU's.
//!   - In a module of x86 kernels it is in the process's memory, which the kernels reach at a
//!     displacement from `rip`, `[COUNTS + 8]`: a `static` of Rust's, a `kernite_cpu::Global` of
//!     its elements, for the host to read and write: `cpu::COUNTS.read()`. A crate with a static
//!     in such a module needs a dependency named `kernite-cpu` for it, not renamed in
//!     `Cargo.toml`, and the error is at the static when it has none.
//!   - A module of both kinds of kernel and a static is an error: it is two modules, one of
//!     each kind of kernel, with the statics each names.
//! - A function without an ABI is a helper, inlined where a kernel calls it, and leaves
//!   nothing behind. Constants and every other item but a `static` stay as written, for Rust
//!   to use as well.
//!   A constant a kernel reads counts as used, as it would if Rust had seen the read, under the
//!   `cfg` of each item that reads it, a helper's under that of the kernel that calls it and that
//!   of the helper: only one that nothing reads, or that only what `cfg` leaves out reads, is
//!   `dead_code`. Rust has no `bf16`, and its `f16` is unstable, so a constant of either goes in
//!   the kernel that reads it, where `const` is the language's alone.
//! - A kernel `cfg` keeps names only what `cfg` keeps. What it names is written for Rust at the
//!   name, for Rust to say what it says of any name it cannot find, at the name: a constant, a
//!   static of the process's, an x86 kernel that takes no compile-time arguments. Of the rest, a
//!   helper, an instance of an x86 kernel, an `extern "ptx-func"` function and a static of the
//!   GPU's, which Rust has no item for, it is an error of kernite's at the name, under
//!   `cfg(all(<the kernel's>, not(<the item's>)))`. Both are under the kernel's `cfg`, and, for a
//!   name in a helper's body, the helper's and that of a place that calls the helper. The naked
//!   function of an x86 kernel names the same items, and where `cfg` leaves one out it names
//!   `const 0`, for Rust to say a name once, at the name.
//!
//! ```ignore
//! #[kernite_macros::kernels]
//! mod gpu {
//!     /// `x[i] *= a` for each `i < n`.
//!     pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32, n: u32) {
//!         ..
//!     }
//! }
//!
//! let cuda = kernite_cuda::Cuda::new()?;
//! let xs = cuda.upload(&[1.0, 2.0, 3.0])?;
//! let kernels = gpu::Kernels::load(&cuda)?;
//! unsafe { kernels.scale(kernite_cuda::Grid::new(1, 32), xs.ptr(), 2.0, 3)? };
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
//! A kernel's parameters keep the names they are written with beside those the Rust the
//! macro writes binds for itself: `grid`, the launcher's, and `threads` and `shared`, those
//! of a cooperative GPU kernel's `_blocks` method. A compile-time argument may not be named
//! as one of them, for Rust cannot keep a const apart from a parameter of its name: that is
//! an error at its name.
//!
//! A parameter named `_` is given and not read, in a kernel of every kind. Rust cannot pass `_` on,
//! so the macros name it, with a name of their own that no parameter or compile-time argument of
//! the kernel has. The function Rust calls the kernel by takes it under that name, and its docs
//! show it.
//!
//! A type in a kernel's parameters or return means what it means in any function of the module,
//! `super::W` and `self::W` among them, and it may name the kernel's compile-time arguments, as in
//! any generic function: `*mut [u8; N]` in `fn arr<const N: usize>`. An error in one is Rust's, said
//! once, at the type, and so is a lint, at the level the kernel's `allow`, `warn`, `deny`, `forbid` or
//! `expect` gives it. So is a pointer to a type with no size, `*const CStr` say, which has two words
//! where a kernel's has one: the macros ask Rust of each pointer a kernel takes or gives back, and
//! it says that the size cannot be known, at the type, whatever the type is called and holds.
//! So is a number's name that the module gives another type, `pub type u64 = u32;` and a `use` of
//! it: kernite reads `u64` as 64 bits where Rust would pass a `u32`, so the macros ask Rust of each
//! number a kernel takes or gives back whether it is that number, and it says that it is not, at
//! the type, once. A type with no size, `pub type u8 = [u8];`, has Rust say as well that its size
//! cannot be known, at the type, and nothing at the module's attribute, where nothing could be
//! mended: the Rust the macros write for the kernel itself takes the number and not the name.
//! A generic is a value: `<T>` is an error at the `T`, and so is `impl` in a parameter's type or
//! the result's, wherever in it, `x: impl Copy`, `Option<impl Copy>` or `-> *mut impl Copy`, at the
//! `impl`: in a parameter it makes the function generic over a type, and in the result it hides
//! one. kernite says both in the same words.
//! The Rust the macros write names the numbers and `str` by their paths, `::core::primitive::u64`,
//! so a module may give any of those names another type and build, so long as none of its kernels
//! takes or gives that name as a number.
//!
//! The Rust the macro writes needs no `allow` of a lint of Rust's, so a crate may `forbid` one,
//! `dead_code` say, and build its kernels: Rust lints the names and the code a kernel is
//! written with, and none that the macro adds. Clippy's `too_many_arguments` is the one lint a
//! crate of kernels may not forbid: Clippy says it of a launcher of more than seven parameters,
//! whoever wrote it, and each launcher allows it.
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
//! #[kernite_macros::consts]
//! pub mod sizes {
//!     /// How far a value moves.
//!     pub const STEP: u64 = 3;
//! }
//!
//! #[kernite_macros::kernels(consts = crate::sizes)]
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
//! #[kernite_macros::consts]
//! pub mod sizes {
//!     pub const STEP: u64 = 3;
//! }
//!
//! #[kernite_macros::kernels(consts = crate::sizes)]
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
//! A GPU kernel says how it is launched with `#[maxntid(16, 16)]`, the most threads a block has,
//! `#[reqntid(8, 8, 4)]`, the threads it has, `#[minnctapersm(2)]`, the blocks an SM should hold at
//! once, and `#[maxnreg(32)]`, the most registers a thread may use: whole numbers known now, the
//! module's consts among them, and the compile-time arguments of an instance. They are PTX's
//! directives of the same names, which `ptxas` holds the kernel's code to, and they are not Rust's:
//! the kernel's launcher is the same, and a launch over a block the directives refuse is an error
//! that names the directive and the threads of the block.
//!
//! An x86 kernel whose threads wait for each other is `#[cooperative]` too, and runs on the
//! threads of a grid, each given its place there, `grid.thread` of `grid.threads`: it becomes
//! the function of its name that takes a `kernite_cpu::Grid` and the kernel's parameters, runs
//! the kernel on every thread at once, and returns when all have ended:
//! `cpu::normalize(kernite_cpu::Grid::new(4), x.as_mut_ptr(), sums.as_mut_ptr(), n)?`. A thread
//! spins at `grid.sync()` until the last comes, so a grid has at most as many threads as the
//! machine runs at once, `kernite_cpu::most()`, and more is an error. A crate with such a kernel
//! needs a dependency named `kernite-cpu` for it, not renamed in `Cargo.toml`, and the error is at
//! the kernel when it has none.
//!
//! A GPU kernel is named in PTX as C++ names a function in namespaces, those of the
//! workspace, the crate and its modules, mangled under Itanium's ABI:
//! `_ZN7kernite6my_app3gpu5scaleE`, which `cu++filt`, Nsight and `cuda-gdb` show as
//! `kernite::my_app::gpu::scale`, an instance as a template's,
//! `kernite::my_app::gpu::block_sums<(unsigned int)1024>`, and a static as a variable of the
//! namespaces, `kernite::my_app::gpu::COUNTS`. The workspace's manifest is the one Cargo
//! takes for the package. kernite reads a relative `CARGO_HOME` from the directory rustc runs
//! in, and Cargo from the one it was run in, which may be another. The workspace is named as
//! the directory of its manifest is, or as the manifest names it, which a clone keeps
//! wherever it is. `name = ""` leaves the workspace out of the names. A `name` that is not a
//! string is an error at the module. The error names the manifest. A package with no
//! workspace's manifest is one of its own, named as the package is.
//!
//! ```toml
//! [workspace.metadata.kernite]
//! name = "kernite"
//! ```
//!
//! A kernel is `unsafe`: it trusts the pointers it is given, and that nothing else touches the
//! memory of the module's it names while it runs, as an x86 kernel's safety contract says, by
//! name, of the statics it and the functions it calls name, whether it calls them by name or
//! at an address. One that does not compile is a `compile_error!` where the trouble is, beside
//! a stub of its signature or its launcher, so that its callers are not errors as well. Both
//! are under the kernel's `cfg`, as a helper's errors and a static's are under theirs, and so
//! are the function of each of its instances and the module of its constants: what `cfg` leaves
//! out says nothing.

extern crate proc_macro;

mod expand;
mod gpu;
mod rust;
mod workspace;
mod x86;

use proc_macro::{Delimiter, Group, Span, TokenStream, TokenTree};

use kernite::ConstItem;
use kernite::lang;

use crate::expand::expand;
use crate::rust::{
    Site, attrs_len, braced, error, flatten, inner_attribute, is_ident, is_path, is_punct, lex, module, parse,
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
                           a module `#[kernite_macros::consts]` is on; or nothing";
            return error(attr[0].span(), message);
        }
    };
    // The macro the consts' module has calls `__kernels!` back with the consts, their module's
    // path, and this module.
    let mut out: TokenStream = path.iter().cloned().collect();
    out.extend(parse("::__kernite_consts!"));
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
    let (trees, spans) = lex(&top);
    let at = |s: lang::Span| spans.get(s.0 as usize).copied().unwrap_or_else(Span::call_site);
    // The module as it is, for Rust; and the consts kernite reads, for the kernels.
    let mut inner: TokenStream = top.iter().cloned().collect();
    let mut given = TokenStream::new();
    match trees.and_then(|trees| lang::module(&trees, 0)) {
        Err(e) => inner.extend(error(at(e.span), &e.message)),
        Ok(module) => {
            for (item, verdict) in module.items.iter().zip(kernite::consts(&module)) {
                let tokens = &top[item.tokens.clone()];
                let named = &tokens[attrs_len(tokens)..];
                match verdict {
                    ConstItem::Given | ConstItem::Wrong(_) if !named.first().is_some_and(|t| is_ident(t, "pub")) => {
                        let message = "a const of a module of consts is `pub`, for the modules of kernels that take \
                                       it to see";
                        inner.extend(error(named[0].span(), message));
                    }
                    ConstItem::Given => given.extend(tokens.iter().cloned()),
                    ConstItem::Wrong(e) => inner.extend(error(at(e.span), &e.message)),
                    // The module's own attribute, `#![..]`, is Rust's.
                    ConstItem::NotAConst if inner_attribute(tokens) => {}
                    ConstItem::NotAConst => {
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
    let mut call = parse("::kernite_macros::__kernels!");
    let mut args: TokenStream = [TokenTree::Group(Group::new(Delimiter::Bracket, given))].into_iter().collect();
    args.extend(parse("$($module)*"));
    call.extend([TokenTree::Group(Group::new(Delimiter::Brace, args))]);
    let mut rule = parse("($($module:tt)*) =>");
    rule.extend([TokenTree::Group(Group::new(Delimiter::Brace, call))]);
    rule.extend(parse(";"));
    inner.extend(parse("#[doc(hidden)] macro_rules! __kernite_consts"));
    inner.extend([TokenTree::Group(Group::new(Delimiter::Brace, rule))]);
    inner.extend(parse("#[doc(hidden)] pub(crate) use __kernite_consts;"));
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
    // What the macro writes is of the hygiene of the group `#[kernels]` made of the path, which is
    // the attribute's: it resolves where the module is, not inside the consts' macro, and Rust,
    // which takes it for the macro's, lints none of it. It is where the path begins, for an error
    // to point at.
    let at = path.span();
    let path: Vec<TokenTree> = path.stream().into_iter().collect();
    let site = Site::at(path.first().map_or(at, |t| t.span().resolved_at(at)));
    let inner = expand(Some((consts.stream(), &path)), braces.stream());
    drop(site);
    module(head, braces, inner)
}
