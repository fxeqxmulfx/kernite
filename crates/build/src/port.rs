//! The ports a machine implements, which the use cases call: [`Backend`] for every machine,
//! and [`Native`] or [`Device`] for where its kernels run.

use kernite_lang::{Fn, Memory, Span, Target, Value};

use crate::{Assembly, DeviceStatic, Done, HostStatic, Naked, Requirement};

/// A machine kernite compiles a module's functions for: its kernels while their bodies are
/// written, its statics, how its kernels see the functions they call, and its code.
pub trait Backend {
    /// A kernel whose body the language is writing.
    type Kernel: Target;
    /// A static, laid out in the machine's memory.
    type Static: Clone;
    /// How a kernel sees a function it may call.
    type Callee: Clone;
    /// A kernel's code, once its body is written.
    type Code: Clone;

    /// Static `name`, as `memory` has it worked out, laid out; or why not.
    fn lay_out(&self, name: &str, memory: &Memory) -> Result<Self::Static, StaticError>;
    /// Why each of `statics`, which were laid out, has no room in the module's memory beside those
    /// before it that have, in order; `None` for each that has. A machine with room for them all, as
    /// the process has, says none.
    fn room(&self, statics: &[Self::Static]) -> Vec<Option<String>> {
        vec![None; statics.len()]
    }
    /// How this machine's kernels see `g` when they call an instance of it whose parameters are of the types `params`,
    /// worked out, `&[u64; 2]` for `&[u64; N]`; `None` if they cannot name it.
    fn callee(&self, g: &Fn, params: &[String]) -> Option<Self::Callee>;
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
    /// The functions `kernel`'s body calls, each by the name of its instance, `tile::<4>`: those whose code its own may
    /// depend on. A machine whose code does not says none.
    fn calls(&self, _: &Self::Kernel) -> Vec<String> {
        vec![]
    }
    /// `kernel`'s code, its body written: its registers allocated, or its code checked; or why
    /// not. `done` has what compiling each instance of the module gave so far, by its name: the code of each function
    /// it calls among them, and of each they call in turn, for a function is finished after those it calls. It has no
    /// code of a function that does not compile, and none of one that waits for this one: the function itself, if it
    /// calls itself, and each that calls it.
    fn finish(&self, kernel: Self::Kernel, done: &[Done<Self::Code>]) -> Result<Self::Code, Unfinished>;
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
    /// The module of `code`, each kernel and function, with `statics`, put together from what the
    /// program keeps of them: for every set of them kept, the text the module of that set alone
    /// has, or why it is none.
    fn module(&self, code: &[Self::Code], statics: &[Self::Static]) -> Assembly;
    /// The name the driver finds `code` by.
    fn symbol(&self, code: &Self::Code) -> String;
    /// What a launch of `code` must hold for it to be safe, if it is the code of a kernel that is no `unsafe fn`;
    /// none for one that is.
    fn requires(&self, code: &Self::Code) -> Vec<Requirement>;
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

/// Why a kernel's code is not finished: what is wrong, and where.
#[derive(Debug, PartialEq)]
pub struct Unfinished {
    /// Where the error is placed, if the machine knows: where the variable to blame is
    /// declared, or the instruction to blame is; else at the function.
    pub at: Option<Span>,
    /// What is wrong.
    pub message: String,
}

impl Unfinished {
    /// `message`, blaming the variable the machine's own key `var` names, if any: placed where
    /// `decls`, each key with where it is declared, says.
    pub fn blaming<V: PartialEq>(var: Option<V>, decls: &[(V, Span)], message: String) -> Unfinished {
        let at = var.and_then(|v| decls.iter().find(|(d, _)| *d == v)).map(|&(_, at)| at);
        Unfinished { at, message }
    }
}
