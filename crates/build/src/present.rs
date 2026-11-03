//! kgen-build's refusals in words: what is wrong, where it can be mended.

use kgen_lang::{Error, Span, Value};

/// What a function is that may not be cooperative.
pub(crate) enum Of {
    /// A function the kernels call, `extern "ptx-func"`.
    Func,
    /// A helper, a function of no ABI, inlined where it is called.
    Helper,
}

/// What kgen-build refuses.
pub(crate) enum Refusal<'a> {
    /// A module of both kinds of kernel and a static: two modules.
    TwoKinds,
    /// `#[cooperative]` on function `name`, which is no kernel but `of`.
    Cooperative { name: &'a str, of: Of },
    /// A function of ABI `abi`, which kgen compiles nothing for.
    UnknownAbi { abi: &'a str },
    /// A kernel that is not an `unsafe fn`.
    NotUnsafe,
    /// An item of the module's own named `name`, as a const it takes from module `from` is.
    GivenTwice { name: &'a str, from: &'a str },
}

/// `r`, in words, at `span`.
pub(crate) fn said(r: Refusal<'_>, span: Span) -> Error {
    let message = match r {
        Refusal::TwoKinds => {
            "a static is memory of the module's kernels: the GPU's, for PTX kernels, or the process's, \
             for x86 ones; and this module has both, so it is two modules, one of each kind of kernel and \
             the statics they name"
                .to_string()
        }
        Refusal::Cooperative { name, of } => {
            let what = match of {
                Of::Func => "a function the kernels call, which runs as the kernel that calls it is launched",
                Of::Helper => "a helper, inlined where it is called",
            };
            format!(
                "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \"ptx-kernel\"` \
                 or `extern \"sysv64\"`; `{name}` is {what}"
            )
        }
        Refusal::UnknownAbi { abi } => format!(
            "kgen compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \"ptx-kernel\"` ones, for NVIDIA \
             GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"{abi}\"` is none of these"
        ),
        Refusal::NotUnsafe => "a kernel trusts the pointers it is given, so it is an `unsafe fn`".to_string(),
        Refusal::GivenTwice { name, from } => {
            format!("`{name}` names one of the consts the module takes from `{from}` as well")
        }
    };
    Error { span, message }
}

/// `e`, which happened in instance `name` of a function, for `args`, saying so if it has any.
pub(crate) fn within(name: &str, args: &[Value], e: kgen_lang::Error) -> kgen_lang::Error {
    match args {
        [] => e,
        _ => kgen_lang::Error { message: format!("in `{name}`: {}", e.message), ..e },
    }
}
