//! Kernels written in place, as functions whose bodies are instructions over variables,
//! compiled to assembly that changes only when they do.
//!
//! The crates are layers, each depending only on the ones inside it
//! (`tests/architecture.rs` holds them to it):
//!
//! | layer | crate | what it knows |
//! |---|---|---|
//! | entities | `kernite-core` | programs over variables and over registers |
//! | language | `kernite-lang` | kernels in Rust's syntax: parsing, evaluation, the `Target` port |
//! | entity service | `kernite-compile` | the allocator: liveness, register allocation, spilling, the frame's layout; the `Machine` and `Spill` ports; tested on a toy machine |
//! | use cases | `kernite-build` | what kernite does with a module: ABIs, its rules, compiling each function; the ports a machine implements |
//! | adapters | `kernite-x86`, `kernite-ptx` | x86-64: SysV, Intel syntax; PTX for NVIDIA GPUs |
//! | composition | `kernite`, this crate | which machine and syntax each kernel compiles with |
//! | delivery | `kernite-macros` | kernels as functions, `#[unsafe(naked)]`, or launched on a grid |
//! | runtime | `kernite-cuda` | the CUDA driver a GPU kernel's launcher calls; it depends on no other crate |
//! | runtime | `kernite-cpu` | the threads a cooperative x86 kernel runs on and the CPUs they run on, and the memory of a module of x86 kernels; it depends on no other crate |
//!
//! None of them has a dependency outside the workspace, and none iterates a hash map, so a
//! kernel's text is a function of its source alone.
//!
//! ```
//! let src = r#"
//!     pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
//!         let v: ymm;
//!         vmovups(v, [xs]);
//!         let high: ymm;
//!         vextractf128(high.xmm, v, 1);
//!         let sum: ymm;
//!         vaddps(sum.xmm, v.xmm, high.xmm);
//!         let pair: ymm;
//!         vmovhlps(pair.xmm, sum.xmm, sum.xmm);
//!         vaddps(sum.xmm, sum.xmm, pair.xmm);
//!         vmovshdup(pair.xmm, sum.xmm);
//!         vaddss(sum.xmm, sum.xmm, pair.xmm);
//!         ret(sum.xmm);
//!     }
//! "#;
//! let listing = kernite::x86_str(src, "sum8")?;
//! print!("{}", listing.text());
//! # Ok::<(), String>(())
//! ```

pub use kernite_build::{
    Abi, Assembly, Call, Compiled, ConstItem, DeviceFnOut, DeviceOut, DeviceStatic, Given, HostStatic, ItemOut, Kept,
    Launcher, ModuleOut, Naked, NativeInstance, NativeOut, Owner, When, abi, consts, cooperative, function, instance,
    owner,
};
pub use kernite_lang as lang;

use kernite_build::{all, compile, one, prepare, statics};
use kernite_lang::{Fn, Module, Value};
use kernite_ptx::{Entry, Ptx, Static};
use kernite_x86::X86;
use kernite_x86::intel::Listing;
use kernite_x86::lang::Static as X86Static;

/// Kernel `f` of `module`, for x86-64 under SysV, for the compile-time arguments `args`:
/// those of one of its [`instances`](kernite_lang::instances), none for a kernel that takes none.
/// What goes wrong is placed where it can be mended: an instruction where it is called, a
/// register that ran out where its variable is declared. For an instance, it says which. A
/// kernel that is [`cooperative`] runs on the threads of a grid, and may wait for them,
/// `grid.sync()`. It may call the module's functions `extern "sysv64"`, and each instance of
/// one, but the cooperative, which its listing names in [`Listing::funcs`]. It names the
/// module's memory, its statics, as [`x86_statics`] has them, which its listing names in
/// [`Listing::statics`].
pub fn x86(module: &Module, f: &Fn, args: &[Value]) -> Result<Compiled<Listing>, kernite_lang::Error> {
    compile(module, &prepare(module, &X86), &X86, f, args)
}

/// The statics of `module`, its memory in the process's, in order, for its x86 kernels to name
/// and Rust to lay out: each of Rust's numbers, holding what it says or zeros. Or why one is
/// none, where it can be mended: at its type, at what it holds, or at its name, if a listing
/// names a constant or a function so, `c0` or `f0`.
pub fn x86_statics(module: &Module) -> Vec<Result<Compiled<X86Static>, kernite_lang::Error>> {
    statics(module, &X86)
}

/// `module`, a module of kernels, compiled: each function `extern "sysv64"` for x86-64, as
/// [`x86`] has it, and each the GPU runs for PTX, as [`ptx`] has it, into one module the GPU
/// loads, with the statics each machine lays out as [`x86_statics`] and [`ptx_statics`] have
/// them. `scope` is the namespaces its GPU kernels are in, as [`ptx_scope`] writes them, and
/// `given` the consts it takes from a module of consts. What is wrong is said where it can be
/// mended, and each item of the module is what [`ModuleOut`] says, for whatever writes the
/// module out for Rust.
pub fn module(module: Module, given: Option<Given>, scope: &str) -> ModuleOut {
    kernite_build::module(module, given, &X86, &Ptx::new(scope))
}

/// Kernel `name` of the module in `src`, for x86-64, or its instance `tile::<4>`; an error
/// reads `line:column: what`, a static's that does not compile among them.
pub fn x86_str(src: &str, name: &str) -> Result<Listing, String> {
    let module = kernite_lang::parse(src).map_err(|e| e.at(src))?;
    one(&module, &X86, name).map_err(|e| e.at(src))?.map(|k| k.code).ok_or_else(|| format!("no kernel `{name}`"))
}

/// Kernel `f` of `module`, in PTX, for the compile-time arguments `args`, those of one of its
/// [`instances`](kernite_lang::instances): its `.entry`, or for a function `extern "ptx-func"` its
/// `.func`, for [`kernite_ptx::module`] to put in a module. It may call the module's functions and
/// take their addresses and its kernels'. Nothing allocates its registers, which `ptxas`
/// does, but what allocation would find wrong that does not depend on the registers is found
/// here: a variable read before it is written, a jump to no label, control running off the
/// end. For an instance, it says which.
///
/// It is named in PTX, as the functions it names are, by its [`symbol`](kernite_ptx::symbol) in
/// `scope`, the namespaces [`ptx_scope`] writes: C++'s name for it, which the tools that read
/// PTX show. With no scope, `""`, one that takes no compile-time arguments is its own name.
/// It names the module's memory, its statics, as [`ptx_statics`] has them in `scope`. A kernel
/// that is [`cooperative`] may wait for its grid, `grid.sync()`.
pub fn ptx(module: &Module, f: &Fn, args: &[Value], scope: &str) -> Result<Compiled<Entry>, kernite_lang::Error> {
    let machine = Ptx::new(scope);
    compile(module, &prepare(module, &machine), &machine, f, args)
}

/// `path`, the namespaces a module's PTX kernels are in for [`ptx`], each in the one before:
/// their names as C++ mangles them, `<length><name>` each.
pub fn ptx_scope(path: &[&str]) -> String {
    kernite_ptx::scope(path)
}

/// The statics of `module`, its memory in PTX, in order: each named by its
/// [`symbol`](kernite_ptx::symbol) in `scope`, as [`ptx`] names a kernel, for its kernels to name
/// and [`kernite_ptx::module`] to declare; or why it is none, where it can be mended: at its
/// type, or at what it holds. One that does not fit beside those before it, in the constant
/// memory a module has or the global, is none: the module keeps them all.
pub fn ptx_statics(module: &Module, scope: &str) -> Vec<Result<Compiled<Static>, kernite_lang::Error>> {
    statics(module, &Ptx::new(scope))
}

/// Why each of `types`, those of the statics a module's PTX keeps, in order, as
/// [`DeviceStatic::ty`] has them, has no room in the module beside those before it that have, as
/// [`ptx_statics`] says of a module that keeps them all; `None` for each that has. Each is the type
/// of a static that was laid out.
pub fn ptx_room(types: &[&str]) -> Vec<Option<String>> {
    let statics: Vec<Static> =
        types.iter().map(|ty| Static::new("", "", ty).expect("the type of a static that was laid out")).collect();
    kernite_ptx::room(&statics)
}

/// Every kernel `extern "ptx-kernel"` and function `extern "ptx-func"` of the module in
/// `src`, each instance of each, in one PTX module, in no scope, with its statics; an error
/// reads `line:column: what`. A helper's `#[instances(..)]` is one, and its `#[cooperative]`.
pub fn ptx_str(src: &str) -> Result<String, String> {
    let module = kernite_lang::parse(src).map_err(|e| e.at(src))?;
    all(&module, &Ptx::new("")).map_err(|e| e.at(src))?
}
