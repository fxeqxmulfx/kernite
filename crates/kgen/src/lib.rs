//! Kernels written in place, as functions whose bodies are instructions over variables,
//! compiled to assembly that changes only when they do.
//!
//! The crates are layers, each depending only on the ones inside it
//! (`tests/architecture.rs` holds them to it):
//!
//! | layer | crate | what it knows |
//! |---|---|---|
//! | entities | `kgen-core` | programs over variables and over registers; the `Machine` and `Spill` ports |
//! | language | `kgen-lang` | kernels in Rust's syntax: parsing, evaluation, the `Target` port |
//! | use cases | `kgen-compile` | liveness, register allocation, spilling, the frame's layout |
//! | adapters | `kgen-x86`, `kgen-ptx` | x86-64: SysV, Intel syntax; PTX for NVIDIA GPUs |
//! | composition | `kgen`, this crate | which machine and syntax each kernel compiles with |
//! | delivery | `kgen-macros` | kernels as functions, `#[unsafe(naked)]`, or launched on a grid |
//! | runtime | `kgen-cuda` | the CUDA driver a GPU kernel's launcher calls; it depends on no other crate |
//! | runtime | `kgen-cpu` | the threads a cooperative x86 kernel runs on, and the memory of a module of x86 kernels; it depends on no other crate |
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
pub use kgen_x86::lang::Static as X86Static;

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
/// register that ran out where its variable is declared. For an instance, it says which. A
/// kernel that is [`cooperative`] runs on the threads of a grid, and may wait for them,
/// `grid.sync()`. It may call the module's functions `extern "sysv64"`, and each instance of
/// one, but the cooperative, which its listing names in [`Listing::funcs`]. It names the
/// module's memory, its statics, as [`x86_statics`] has them, which its listing names in
/// [`Listing::statics`].
pub fn x86(module: &Module, f: &Fn, args: &[Value]) -> Result<Compiled<Listing>, kgen_lang::Error> {
    let name = kgen_lang::instance(f, args)?;
    let mut kernel = kgen_x86::lang::Kernel::new(&name);
    if cooperative(f)? {
        kernel.cooperative();
    }
    // One that does not compile says why where it is declared.
    for s in x86_statics(module).into_iter().flatten() {
        kernel.module_memory(&s.code);
    }
    for item in &module.items {
        use kgen_x86::lang::Callee;
        let ItemKind::Fn(g) = &item.kind else { continue };
        if abi(g) != Some(Abi::Sysv64) {
            continue;
        }
        // One of no prototype, or whose instances are wrong, says why where it is compiled; a
        // kernel that calls it, that it does not compile.
        let callee = match cooperative(g) {
            Ok(false) => x86_prototype(g).map_or(Callee::Broken, Callee::Calls),
            Ok(true) => Callee::Cooperative,
            Err(_) => Callee::Broken,
        };
        for args in kgen_lang::instances(module, g).unwrap_or_default() {
            if let Ok(name) = kgen_lang::instance(g, &args) {
                kernel.callee(name, callee.clone());
            }
        }
    }
    let run = kgen_lang::kernel(module, f, args, &mut kernel).map_err(|e| within(&name, args, e))?;
    let code = render(&kernel.finish()).map_err(|e| {
        let e = kgen_lang::Error { span: blame(&e, &run.decls).unwrap_or(f.span), message: e.to_string() };
        within(&name, args, e)
    })?;
    Ok(Compiled { code, consts: run.consts })
}

/// The statics of `module`, its memory in the process's, in order, for its x86 kernels to name
/// and Rust to lay out: each of Rust's numbers, holding what it says or zeros. Or why one is
/// none, where it can be mended: at its type, at what it holds, or at its name, if another item
/// of the module's has it or a listing names a constant or a function so, `c0` or `f0`.
pub fn x86_statics(module: &Module) -> Vec<Result<Compiled<X86Static>, kgen_lang::Error>> {
    let mut out = vec![];
    for item in &module.items {
        let ItemKind::Static(s) = &item.kind else { continue };
        let made = kgen_lang::memory(module, s).and_then(|memory| {
            let at = |span: Span| move |message: String| kgen_lang::Error { span, message };
            if kgen_x86::intel::taken(&s.name) {
                return Err(at(s.span)(format!(
                    "a kernel's listing names a constant `{{c0}}` and a function `{{f0}}`, and a static by its \
                     name, so one is not named `{}`",
                    s.name
                )));
            }
            let mut made = X86Static::new(&s.name, &memory.ty).map_err(at(s.ty.span()))?;
            if let (Some(value), Some(e)) = (&memory.value, &s.value) {
                made.init(value).map_err(at(e.span()))?;
            }
            Ok(Compiled { code: made, consts: memory.consts })
        });
        out.push(made);
    }
    out
}

/// Kernel `name` of the module in `src`, for x86-64, or its instance `tile::<4>`; an error
/// reads `line:column: what`, a static's that does not compile among them.
pub fn x86_str(src: &str, name: &str) -> Result<Listing, String> {
    let module = kgen_lang::parse(src).map_err(|e| e.at(src))?;
    for s in x86_statics(&module) {
        s.map_err(|e| e.at(src))?;
    }
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
/// It names the module's memory, its statics, as [`ptx_statics`] has them in `scope`. A kernel
/// that is [`cooperative`] may wait for its grid, `grid.sync()`.
pub fn ptx(module: &Module, f: &Fn, args: &[Value], scope: &str) -> Result<Compiled<Entry>, kgen_lang::Error> {
    let name = kgen_lang::instance(f, args)?;
    let symbol = kgen_ptx::symbol(scope, &f.name, args);
    let mut kernel = match abi(f) {
        Some(Abi::PtxFunc) => kgen_ptx::lang::Kernel::func(&name, symbol),
        _ => kgen_ptx::lang::Kernel::new(&name, symbol),
    };
    if cooperative(f)? {
        kernel.cooperative();
    }
    // One that does not compile says why where it is declared.
    for s in ptx_statics(module, scope).into_iter().flatten() {
        kernel.module_memory(&s.code);
    }
    for item in &module.items {
        use kgen_ptx::lang::Callee;
        let ItemKind::Fn(g) = &item.kind else { continue };
        // One of no prototype says why where it is compiled, as one whose instances are wrong
        // does; a kernel that calls it, that it does not compile.
        let callee = match abi(g) {
            Some(Abi::PtxKernel) => Callee::Kernel,
            Some(Abi::PtxFunc) => prototype(g).map_or(Callee::Broken, Callee::Calls),
            _ => continue,
        };
        for args in kgen_lang::instances(module, g).unwrap_or_default() {
            if let Ok(name) = kgen_lang::instance(g, &args) {
                kernel.callee(name, kgen_ptx::symbol(scope, &g.name, &args), callee.clone());
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
/// reads `line:column: what`. A helper's `#[instances(..)]` is one, and its `#[cooperative]`.
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
        cooperative(f).map_err(|e| e.at(src))?;
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

/// Whose memory a module's statics are: its kernels'.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    /// The process's, for x86 kernels, as [`x86_statics`] has them.
    Process,
    /// The GPU's, for PTX kernels, as [`ptx_statics`] has them.
    Gpu,
}

/// Whose memory the statics of `module` are, its kernels': the process's when its functions of
/// an ABI kgen compiles for are all `extern "sysv64"`, and else the GPU's, those of a module of
/// statics and no kernels among them. A module of both kinds of kernel and a static is two
/// modules, one of each, which the error at its first static says; its statics are then the
/// GPU's, which its x86 kernels do not reach.
pub fn owner(module: &Module) -> Result<Owner, kgen_lang::Error> {
    let abis: Vec<Abi> = module
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fn(f) => abi(f),
            _ => None,
        })
        .collect();
    if !abis.contains(&Abi::Sysv64) {
        return Ok(Owner::Gpu);
    }
    if abis.iter().all(|&a| a == Abi::Sysv64) {
        return Ok(Owner::Process);
    }
    match module.items.iter().find_map(|item| match &item.kind {
        ItemKind::Static(s) => Some(s),
        _ => None,
    }) {
        Some(s) => Err(kgen_lang::Error {
            span: s.span,
            message: "a static is memory of the module's kernels: the GPU's, for PTX kernels, or the process's, \
                      for x86 ones; and this module has both, so it is two modules, one of each kind of kernel and \
                      the statics they name"
                .to_string(),
        }),
        None => Ok(Owner::Gpu),
    }
}

/// Whether `f` is a kernel whose parts wait for each other, `#[cooperative]`, which every
/// launch runs all at once: a GPU kernel's blocks, or an x86 kernel's threads. Or why it may
/// not say so, if it says so and is no kernel. One of an ABI kgen knows nothing of is none,
/// whatever it says.
pub fn cooperative(f: &Fn) -> Result<bool, kgen_lang::Error> {
    let Some(c) = &f.cooperative else { return Ok(false) };
    let what = match (abi(f), &f.abi) {
        (Some(Abi::PtxKernel | Abi::Sysv64), _) => return Ok(true),
        (Some(Abi::PtxFunc), _) => "a function the kernels call, which runs as the kernel that calls it is launched",
        (None, Some(_)) => return Ok(false),
        (None, None) => "a helper, inlined where it is called",
    };
    let message = format!(
        "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \"ptx-kernel\"` or \
         `extern \"sysv64\"`; `{}` is {what}",
        f.name
    );
    Err(kgen_lang::Error { span: c.span, message })
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

/// The prototype of `f`, a function `extern "sysv64"`, as the kernels that call it see it;
/// `None` if SysV passes its types in no registers.
fn x86_prototype(f: &Fn) -> Option<kgen_x86::lang::Prototype> {
    let params: Option<Vec<(&str, &str)>> = f
        .params
        .iter()
        .map(|p| match &p.ty {
            kgen_lang::Type::Name(ty, _) => Some((p.name.as_str(), ty.as_str())),
            _ => None,
        })
        .collect();
    kgen_x86::lang::Prototype::of(&params?, f.ret.as_ref().map(|r| r.ty.as_str())).ok()
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
        Error::ReadBeforeWrite { var }
        | Error::FixedTwice { var }
        | Error::FixedChanged { var, .. }
        | Error::OutOfRegisters { var, .. } => var,
        Error::FixedClash { a, .. } => a,
        Error::LabelTwice { .. } | Error::NoSuchLabel { .. } | Error::FallsOffEnd => return None,
    };
    decls.iter().find(|(name, _)| name == var).map(|&(_, span)| span)
}
