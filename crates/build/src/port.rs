//! The ports a machine implements, which the use cases call: [`Backend`] for every machine,
//! and [`Native`] or [`Device`] for where its kernels run.

use kgen_lang::{Fn, Memory, Target, Value};

use crate::{DeviceStatic, HostStatic, Naked};

/// A machine kgen compiles a module's functions for: its kernels while their bodies are
/// written, its statics, how its kernels see the functions they call, and its code.
pub trait Backend {
    /// A kernel whose body the language is writing.
    type Kernel: Target;
    /// A static, laid out in the machine's memory.
    type Static: Clone;
    /// How a kernel sees a function it may call.
    type Callee: Clone;
    /// A kernel's code, once its body is written.
    type Code;

    /// Static `name`, as `memory` has it worked out, laid out after `before`, the statics
    /// before it that were laid out; or why not.
    fn lay_out(&self, name: &str, memory: &Memory, before: &[Self::Static]) -> Result<Self::Static, StaticError>;
    /// How this machine's kernels see `g` when they call it; `None` if they cannot name it.
    fn callee(&self, g: &Fn) -> Option<Self::Callee>;
    /// The kernel `name`, instance `args` of `f`, told whether it is `cooperative`, each of
    /// `statics` and each of `callees`.
    fn kernel(
        &self,
        f: &Fn,
        name: &str,
        args: &[Value],
        cooperative: bool,
        statics: &[Self::Static],
        callees: &[Called<Self::Callee>],
    ) -> Self::Kernel;
    /// `kernel`'s code, its body written: its registers allocated, or its code checked; or why
    /// not.
    fn finish(&self, kernel: Self::Kernel) -> Result<Self::Code, Unfinished>;
}

/// A machine whose kernels Rust calls in the process, as naked functions.
pub trait Native: Backend {
    /// `code` as `naked_asm!` takes it.
    fn naked(&self, code: &Self::Code) -> Naked;
    /// `s` as Rust lays it out in the process.
    fn host_static(&self, s: &Self::Static) -> HostStatic;
}

/// A machine whose kernels a driver loads as one module.
pub trait Device: Backend {
    /// The module of `code`, each kernel and function, with `statics`; or why it is none.
    fn module(&self, code: &[Self::Code], statics: &[Self::Static]) -> Result<String, String>;
    /// The name the driver finds `code` by.
    fn symbol(&self, code: &Self::Code) -> String;
    /// `s` as the host reads it.
    fn device_static(&self, s: &Self::Static) -> DeviceStatic;
}

/// A function a kernel may call: one of its instances, by `name`, `tile::<4>`.
#[derive(Debug, PartialEq)]
pub struct Called<C> {
    /// The instance's name, `tile::<4>`.
    pub name: String,
    /// The function's own name, `tile`.
    pub function: String,
    /// The instance's compile-time arguments, `[4]`; none for a function that takes none.
    pub args: Vec<Value>,
    /// How the kernel sees the function.
    pub callee: C,
}

/// Why a static is not laid out: what is wrong, and with which part of it, where the error is
/// placed.
#[derive(Debug, PartialEq)]
pub enum StaticError {
    /// What is wrong with its name.
    Name(String),
    /// What is wrong with its type.
    Type(String),
    /// What is wrong with what it holds.
    Value(String),
}

/// Why a kernel's code is not finished: what is wrong, and the variable to blame, if any.
#[derive(Debug, PartialEq)]
pub struct Unfinished {
    /// The variable to blame, by name: the error is placed where it is declared.
    pub var: Option<String>,
    /// What is wrong.
    pub message: String,
}
