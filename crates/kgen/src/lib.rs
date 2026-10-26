//! Kernels written in place, as functions whose bodies are instructions over variables,
//! compiled to assembly that changes only when they do.
//!
//! The crates are layers, each depending only on the ones inside it
//! (`tests/architecture.rs` holds them to it):
//!
//! | layer | crate | what it knows |
//! |---|---|---|
//! | entities | `kgen-core` | programs over variables and over registers; the `Machine` port |
//! | language | `kgen-lang` | kernels in Rust's syntax: parsing, evaluation, the `Target` port |
//! | use cases | `kgen-compile` | liveness, register allocation, the frame's layout |
//! | adapters | `kgen-x86`, `kgen-ptx` | x86-64: SysV, Intel syntax; PTX for NVIDIA GPUs |
//! | composition | `kgen`, this crate | which machine and syntax each kernel compiles with |
//! | delivery | `kgen-macros` | kernels as functions, `#[unsafe(naked)]` or launched on a GPU |
//! | runtime | `kgen-cuda` | the CUDA driver the launchers call; it depends on no other crate |
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
//! let listing = kgen::x86_str(src, "sum8")?;
//! print!("{}", listing.text());
//! # Ok::<(), String>(())
//! ```

pub use kgen_compile::Error;
pub use kgen_core::{Class, Program, Signature};
pub use kgen_lang as lang;
pub use kgen_ptx::{Entry, Static};
pub use kgen_x86::Op;
pub use kgen_x86::intel::{Line, Listing};

use kgen_lang::{Fn, ItemKind, Module, Span, Value};

/// The ABIs a kernel names, or a function kernels call, each of which kgen compiles for a
/// machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abi {
    /// `extern "sysv64"`: x86-64 under the System V calling convention.
    Sysv64,
    /// `extern "ptx-kernel"`: an NVIDIA GPU, in PTX.
    PtxKernel,
    /// `extern "ptx-func"`: a function the PTX kernels of its module call, `.func`, of PTX's
    /// types.
    PtxFunc,
}

impl Abi {
    /// The ABI Rust writes as `name`, if kgen compiles for it.
    pub fn of(name: &str) -> Option<Abi> {
        match name {
            "sysv64" => Some(Abi::Sysv64),
            "ptx-kernel" => Some(Abi::PtxKernel),
            "ptx-func" => Some(Abi::PtxFunc),
            _ => None,
        }
    }
}

/// `program` for x86-64 under SysV, in Intel syntax.
pub fn render(program: &Program<Op>) -> Result<Listing, Error> {
    let compiled = kgen_compile::compile(program, &kgen_x86::SysV)?;
    Ok(kgen_x86::intel::print(&compiled))
}

/// A kernel compiled, or a static, and the constants of its module it read. Rust never sees
/// those reads, so whatever writes the module out for Rust has to say they happened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compiled<T> {
    /// What it compiled to: its listing, its `.entry`, or the memory a static is.
    pub code: T,
    /// The constants, by name, in the module's order.
    pub consts: Vec<String>,
}

/// Kernel `f` of `module`, for x86-64 under SysV, for the compile-time arguments `args`:
/// those of one of its [`instances`](kgen_lang::instances), none for a kernel that takes none.
/// What goes wrong is placed where it can be mended: an instruction where it is called, a
/// register that ran out where its variable is declared. For an instance, it says which.
pub fn x86(module: &Module, f: &Fn, args: &[Value]) -> Result<Compiled<Listing>, kgen_lang::Error> {
    let name = kgen_lang::instance(f, args)?;
    let mut kernel = kgen_x86::lang::Kernel::new(&name);
    let run = kgen_lang::kernel(module, f, args, &mut kernel).map_err(|e| within(&name, args, e))?;
    let code = render(&kernel.finish()).map_err(|e| {
        let e = kgen_lang::Error { span: blame(&e, &run.decls).unwrap_or(f.span), message: e.to_string() };
        within(&name, args, e)
    })?;
    Ok(Compiled { code, consts: run.consts })
}

/// Kernel `name` of the module in `src`, for x86-64, or its instance `tile::<4>`; an error
/// reads `line:column: what`.
pub fn x86_str(src: &str, name: &str) -> Result<Listing, String> {
    let module = kgen_lang::parse(src).map_err(|e| e.at(src))?;
    let (f, args) = instance(&module, name).map_err(|e| e.at(src))?.ok_or_else(|| format!("no kernel `{name}`"))?;
    x86(&module, f, &args).map(|k| k.code).map_err(|e| e.at(src))
}

/// Kernel `f` of `module`, in PTX, for the compile-time arguments `args`, those of one of its
/// [`instances`](kgen_lang::instances): its `.entry`, or for a function `extern "ptx-func"` its
/// `.func`, for [`ptx_module`] to put in a module. It may call the module's functions and
/// take their addresses and its kernels'. Nothing allocates its registers, which `ptxas`
/// does, but what allocation would find wrong that does not depend on the registers is found
/// here: a variable read before it is written, a jump to no label, control running off the
/// end. For an instance, it says which.
///
/// It is named in PTX, as the functions it names are, by its [`symbol`](kgen_ptx::symbol) in
/// `scope`, the namespaces [`ptx_scope`] writes: C++'s name for it, which the tools that read
/// PTX show. With no scope, `""`, one that takes no compile-time arguments is its own name.
/// It names the module's memory, its statics, as [`ptx_statics`] has them in `scope`.
pub fn ptx(module: &Module, f: &Fn, args: &[Value], scope: &str) -> Result<Compiled<Entry>, kgen_lang::Error> {
    let name = kgen_lang::instance(f, args)?;
    let symbol = kgen_ptx::symbol(scope, &f.name, args);
    let mut kernel = match abi(f) {
        Some(Abi::PtxFunc) => kgen_ptx::lang::Kernel::func(&name, symbol),
        _ => kgen_ptx::lang::Kernel::new(&name, symbol),
    };
    // One that does not compile says why where it is declared.
    for s in ptx_statics(module, scope).into_iter().flatten() {
        kernel.module_memory(&s.code);
    }
    for item in &module.items {
        let ItemKind::Fn(g) = &item.kind else { continue };
        let proto = match abi(g) {
            Some(Abi::PtxKernel) => None,
            // One of no prototype says why where it is compiled, as one whose instances are
            // wrong does.
            Some(Abi::PtxFunc) => match prototype(g) {
                Some(proto) => Some(proto),
                None => continue,
            },
            _ => continue,
        };
        for args in kgen_lang::instances(module, g).unwrap_or_default() {
            if let Ok(name) = kgen_lang::instance(g, &args) {
                kernel.callee(name, kgen_ptx::symbol(scope, &g.name, &args), proto.clone());
            }
        }
    }
    let run = kgen_lang::kernel(module, f, args, &mut kernel).map_err(|e| within(&name, args, e))?;
    let code = kernel.finish();
    kgen_compile::check(&code.program).map_err(|e| {
        let e = kgen_lang::Error { span: blame(&e, &run.decls).unwrap_or(f.span), message: e.to_string() };
        within(&name, args, e)
    })?;
    Ok(Compiled { code, consts: run.consts })
}

/// `path`, the namespaces a module's PTX kernels are in for [`ptx`], each in the one before:
/// their names as C++ mangles them, `<length><name>` each.
pub fn ptx_scope(path: &[&str]) -> String {
    kgen_ptx::scope(path)
}

/// The statics of `module`, its memory in PTX, in order: each named by its
/// [`symbol`](kgen_ptx::symbol) in `scope`, as [`ptx`] names a kernel, for its kernels to name
/// and [`ptx_module`] to declare; or why it is none, where it can be mended: at its type, at
/// what it holds, or at its name, if another item of the module's has it. One that does not
/// fit beside those before it, in the constant memory a module has or the global, is none.
pub fn ptx_statics(module: &Module, scope: &str) -> Vec<Result<Compiled<Static>, kgen_lang::Error>> {
    let mut out: Vec<Result<Compiled<Static>, kgen_lang::Error>> = vec![];
    for item in &module.items {
        let ItemKind::Static(s) = &item.kind else { continue };
        let made = kgen_lang::memory(module, s).and_then(|memory| {
            let at = |span: Span| move |message: String| kgen_lang::Error { span, message };
            let symbol = kgen_ptx::symbol(scope, &s.name, &[]);
            let mut made = Static::new(&s.name, &symbol, &memory.ty).map_err(at(s.ty.span()))?;
            if let (Some(value), Some(e)) = (&memory.value, &s.value) {
                made.init(value).map_err(at(e.span()))?;
            }
            let before: Vec<Static> = out.iter().flatten().map(|s| s.code.clone()).collect();
            kgen_ptx::fits(&before, &made).map_err(at(s.ty.span()))?;
            Ok(Compiled { code: made, consts: memory.consts })
        });
        out.push(made);
    }
    out
}

/// The PTX module of `entries`, for the CUDA driver to load: of the PTX version and for the
/// GPU their instructions need, with `statics`, the module's memory. It fails if one needs a
/// GPU older than another needs.
pub fn ptx_module(entries: &[Entry], statics: &[Static]) -> Result<String, String> {
    kgen_ptx::module(entries, statics)
}

/// Every kernel `extern "ptx-kernel"` and function `extern "ptx-func"` of the module in
/// `src`, each instance of each, in one PTX module, in no scope, with its statics; an error
/// reads `line:column: what`. A helper's `#[instances(..)]` is one.
pub fn ptx_str(src: &str) -> Result<String, String> {
    let module = kgen_lang::parse(src).map_err(|e| e.at(src))?;
    let mut statics = vec![];
    for s in ptx_statics(&module, "") {
        statics.push(s.map_err(|e| e.at(src))?.code);
    }
    let mut entries = vec![];
    for item in &module.items {
        let ItemKind::Fn(f) = &item.kind else { continue };
        if !matches!(abi(f), None | Some(Abi::PtxKernel | Abi::PtxFunc)) {
            continue;
        }
        for args in kgen_lang::instances(&module, f).map_err(|e| e.at(src))? {
            entries.push(ptx(&module, f, &args, "").map_err(|e| e.at(src))?.code);
        }
    }
    ptx_module(&entries, &statics)
}

/// The ABI `f` names, if kgen compiles for it.
pub fn abi(f: &Fn) -> Option<Abi> {
    f.abi.as_ref().and_then(|(abi, _)| Abi::of(abi))
}

/// The prototype of `f`, a function `extern "ptx-func"`, as the kernels that call it see it;
/// `None` if its types are none PTX passes.
fn prototype(f: &Fn) -> Option<kgen_ptx::Prototype> {
    let params: Option<Vec<&str>> = f
        .params
        .iter()
        .map(|p| match &p.ty {
            kgen_lang::Type::Name(ty, _) => Some(ty.as_str()),
            _ => None,
        })
        .collect();
    kgen_ptx::Prototype::of(&params?, f.ret.as_ref().map(|r| r.ty.as_str())).ok()
}

/// The function `name` of `module`.
pub fn function<'a>(module: &'a Module, name: &str) -> Option<&'a Fn> {
    module.items.iter().find_map(|item| match &item.kind {
        ItemKind::Fn(f) if f.name == name => Some(f),
        _ => None,
    })
}

/// The function of `module` that `name` names, with the compile-time arguments of its instance
/// that `name` is, as `tile::<4>` is; none for a function that takes none.
pub fn instance<'a>(module: &'a Module, name: &str) -> Result<Option<(&'a Fn, Vec<Value>)>, kgen_lang::Error> {
    let Some(f) = function(module, name.split("::<").next().unwrap_or(name)) else { return Ok(None) };
    for args in kgen_lang::instances(module, f)? {
        if kgen_lang::instance(f, &args)? == name {
            return Ok(Some((f, args)));
        }
    }
    Ok(None)
}

/// `e`, which happened in `name`, the instance of a function for `args`, saying so.
fn within(name: &str, args: &[Value], e: kgen_lang::Error) -> kgen_lang::Error {
    match args {
        [] => e,
        _ => kgen_lang::Error { message: format!("in `{name}`: {}", e.message), ..e },
    }
}

/// Where the variable an allocation error names was declared.
fn blame(e: &Error, decls: &[(String, Span)]) -> Option<Span> {
    let var = match e {
        Error::ReadBeforeWrite { var } | Error::FixedTwice { var } | Error::OutOfRegisters { var, .. } => var,
        Error::FixedClash { a, .. } => a,
        Error::LabelTwice { .. } | Error::NoSuchLabel { .. } | Error::FallsOffEnd => return None,
    };
    decls.iter().find(|(name, _)| name == var).map(|&(_, span)| span)
}
