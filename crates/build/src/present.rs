//! kgen-build's refusals in words: what is wrong, where it can be mended.

use kgen_lang::{Error, Span};

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
    };
    Error { span, message }
}
