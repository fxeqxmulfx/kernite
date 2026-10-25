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
pub use kgen_ptx::Entry;
pub use kgen_x86::Op;
pub use kgen_x86::intel::{Line, Listing};

use kgen_lang::{Fn, ItemKind, Module, Span};

/// The ABIs a kernel names, each of which kgen compiles for a machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abi {
    /// `extern "sysv64"`: x86-64 under the System V calling convention.
    Sysv64,
    /// `extern "ptx-kernel"`: an NVIDIA GPU, in PTX.
    PtxKernel,
}

impl Abi {
    /// The ABI Rust writes as `name`, if kgen compiles for it.
    pub fn of(name: &str) -> Option<Abi> {
        match name {
            "sysv64" => Some(Abi::Sysv64),
            "ptx-kernel" => Some(Abi::PtxKernel),
            _ => None,
        }
    }
}

/// `program` for x86-64 under SysV, in Intel syntax.
pub fn render(program: &Program<Op>) -> Result<Listing, Error> {
    let compiled = kgen_compile::compile(program, &kgen_x86::SysV)?;
    Ok(kgen_x86::intel::print(&compiled))
}

/// Kernel `f` of `module`, for x86-64 under SysV. What goes wrong is placed where it can be
/// mended: an instruction where it is called, a register that ran out where its variable is
/// declared.
pub fn x86(module: &Module, f: &Fn) -> Result<Listing, kgen_lang::Error> {
    let mut kernel = kgen_x86::lang::Kernel::new(&f.name);
    let decls = kgen_lang::kernel(module, f, &mut kernel)?;
    render(&kernel.finish())
        .map_err(|e| kgen_lang::Error { span: blame(&e, &decls).unwrap_or(f.span), message: e.to_string() })
}

/// Kernel `name` of the module in `src`, for x86-64; an error reads `line:column: what`.
pub fn x86_str(src: &str, name: &str) -> Result<Listing, String> {
    let module = kgen_lang::parse(src).map_err(|e| e.at(src))?;
    let f = function(&module, name).ok_or_else(|| format!("no kernel `{name}`"))?;
    x86(&module, f).map_err(|e| e.at(src))
}

/// Kernel `f` of `module`, in PTX: its `.entry`, for [`ptx_module`] to put in a module.
/// Nothing allocates its registers, which `ptxas` does, but what allocation would find wrong
/// that does not depend on the registers is found here: a variable read before it is written,
/// a jump to no label, control running off the end.
pub fn ptx(module: &Module, f: &Fn) -> Result<Entry, kgen_lang::Error> {
    let mut kernel = kgen_ptx::lang::Kernel::new(&f.name);
    let decls = kgen_lang::kernel(module, f, &mut kernel)?;
    let entry = kernel.finish();
    kgen_compile::check(&entry.program)
        .map_err(|e| kgen_lang::Error { span: blame(&e, &decls).unwrap_or(f.span), message: e.to_string() })?;
    Ok(entry)
}

/// The PTX module of `entries`, for the CUDA driver to load.
pub fn ptx_module(entries: &[Entry]) -> String {
    kgen_ptx::module(entries)
}

/// Every `extern "ptx-kernel"` kernel of the module in `src`, in one PTX module; an error
/// reads `line:column: what`.
pub fn ptx_str(src: &str) -> Result<String, String> {
    let module = kgen_lang::parse(src).map_err(|e| e.at(src))?;
    let mut entries = vec![];
    for item in &module.items {
        if let ItemKind::Fn(f) = &item.kind
            && f.abi.as_ref().and_then(|(abi, _)| Abi::of(abi)) == Some(Abi::PtxKernel)
        {
            entries.push(ptx(&module, f).map_err(|e| e.at(src))?);
        }
    }
    Ok(ptx_module(&entries))
}

/// The function `name` of `module`.
pub fn function<'a>(module: &'a Module, name: &str) -> Option<&'a Fn> {
    module.items.iter().find_map(|item| match &item.kind {
        ItemKind::Fn(f) if f.name == name => Some(f),
        _ => None,
    })
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
