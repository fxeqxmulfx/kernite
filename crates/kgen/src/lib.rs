//! Kernels written as instructions over variables, compiled to assembly that changes only
//! when they do.
//!
//! The crates are layers, each depending only on the ones inside it
//! (`tests/architecture.rs` holds them to it):
//!
//! | layer | crate | what it knows |
//! |---|---|---|
//! | entities | `kgen-core` | programs over variables and over registers; the `Machine` port |
//! | use cases | `kgen-compile` | liveness, register allocation, the frame's layout |
//! | adapter | `kgen-x86` | x86-64: the instruction builder, SysV, Intel syntax |
//! | kernels | `kgen-kernels` | the kernels, written in `kgen-x86`'s builder |
//! | composition | `kgen`, this crate | which machine and syntax the use cases run with |
//! | delivery | `kgen-macros` | a kernel as a `#[unsafe(naked)]` function |
//!
//! None of them has a dependency outside the workspace, and none iterates a hash map, so a
//! kernel's text is a function of its source alone. The snapshots in `snapshots/` hold every
//! listing; a change to any layer that moves an instruction shows there, in review.

pub use kgen_core::{Class, Signature};
pub use kgen_kernels::Kernel;
pub use kgen_x86::intel::{Line, Listing};

/// Why a kernel could not be rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// No such kernel, or parameters it does not take.
    Kernel(String),
    /// It did not compile.
    Compile(kgen_compile::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Kernel(e) => f.write_str(e),
            Error::Compile(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

/// `kernel` for x86-64 under SysV, in Intel syntax.
pub fn render(kernel: &Kernel) -> Result<Listing, Error> {
    let program = kernel.program();
    let compiled = kgen_compile::compile(&program, &kgen_x86::SysV).map_err(Error::Compile)?;
    Ok(kgen_x86::intel::print(&compiled))
}

/// The kernel `name` with `params`, as [`render`] renders it.
pub fn render_named(name: &str, params: &[(String, i64)]) -> Result<Listing, Error> {
    render(&Kernel::parse(name, params).map_err(Error::Kernel)?)
}
