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
//! | composition | `kgen`, this crate | which machine and syntax the use cases run with |
//! | delivery | `kgen-macros` | a kernel as a `#[unsafe(naked)]` function |
//!
//! None of them has a dependency outside the workspace, and none iterates a hash map, so a
//! kernel's text is a function of its source alone. The kernels themselves live in the
//! crates that use kgen, written in `kgen-x86`'s builder:
//!
//! ```
//! use kgen_x86::Asm;
//!
//! // `fn(xs: *const f32) -> f32`: the sum of the eight floats at `xs`.
//! let mut a = Asm::new("sum8");
//! let xs = a.param("xs");
//! let v = a.ymm("v");
//! a.vmovups(v, xs.at(0));
//! let high = a.ymm("high");
//! a.vextractf128(high.xmm(), v, 1);
//! let sum = a.ymm("sum");
//! a.vaddps(sum.xmm(), v.xmm(), high.xmm());
//! let pair = a.ymm("pair");
//! a.vmovhlps(pair.xmm(), sum.xmm(), sum.xmm());
//! a.vaddps(sum.xmm(), sum.xmm(), pair.xmm());
//! a.vmovshdup(pair.xmm(), sum.xmm());
//! a.vaddss(sum.xmm(), sum.xmm(), pair.xmm());
//! a.ret_f32(sum.xmm());
//! let listing = kgen::render(&a.finish())?;
//! print!("{}", listing.text());
//! # Ok::<(), kgen::Error>(())
//! ```

pub use kgen_compile::Error;
pub use kgen_core::{Class, Program, Signature};
pub use kgen_x86::Op;
pub use kgen_x86::intel::{Line, Listing};

/// `program` for x86-64 under SysV, in Intel syntax.
pub fn render(program: &Program<Op>) -> Result<Listing, Error> {
    let compiled = kgen_compile::compile(program, &kgen_x86::SysV)?;
    Ok(kgen_x86::intel::print(&compiled))
}
