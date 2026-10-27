//! PTX as kgen-build sees it: a machine whose kernels a driver loads as one module.

use kgen_build::{Abi, Backend, Called, Device, DeviceStatic, StaticError, Unfinished, abi};
use kgen_lang::{Fn, Memory, Value};

use crate::lang::{Callee, Kernel};
use crate::{Entry, Prototype, Static, fits};

/// PTX for NVIDIA GPUs: kernels and the functions they call, `.entry` and `.func`, and the
/// module's memory, in one module the driver loads, each named by its [`symbol`](crate::symbol)
/// in a scope.
pub struct Ptx {
    /// The namespaces the module's kernels are in, as [`scope`](crate::scope) writes them.
    scope: String,
}

impl Ptx {
    /// The machine that names each kernel, function and static by its symbol in `scope`, the
    /// namespaces [`scope`](crate::scope) writes, taken as it is given: `""` for none.
    pub fn new(scope: &str) -> Ptx {
        Ptx { scope: scope.to_string() }
    }
}

impl Backend for Ptx {
    type Kernel = Kernel;
    type Static = Static;
    type Callee = Callee;
    type Code = Entry;

    fn lay_out(&self, name: &str, memory: &Memory, before: &[Static]) -> Result<Static, StaticError> {
        let symbol = crate::symbol(&self.scope, name, &[]);
        let mut made = Static::new(name, &symbol, &memory.ty).map_err(StaticError::Type)?;
        if let Some(value) = &memory.value {
            made.init(value).map_err(StaticError::Value)?;
        }
        fits(before, &made).map_err(StaticError::Type)?;
        Ok(made)
    }

    fn callee(&self, g: &Fn) -> Option<Callee> {
        match abi(g) {
            Some(Abi::PtxKernel) => Some(Callee::Kernel),
            // One of no prototype says why where it is compiled; a kernel that calls it, that it
            // does not compile.
            Some(Abi::PtxFunc) => Some(prototype(g).map_or(Callee::Broken, Callee::Calls)),
            _ => None,
        }
    }

    fn kernel(
        &self,
        f: &Fn,
        name: &str,
        args: &[Value],
        cooperative: bool,
        statics: &[Static],
        callees: &[Called<Callee>],
    ) -> Kernel {
        let symbol = crate::symbol(&self.scope, &f.name, args);
        let mut kernel = match abi(f) {
            Some(Abi::PtxFunc) => Kernel::func(name, symbol),
            _ => Kernel::new(name, symbol),
        };
        if cooperative {
            kernel.cooperative();
        }
        for s in statics {
            kernel.module_memory(s);
        }
        for c in callees {
            kernel.callee(&c.name, crate::symbol(&self.scope, &c.function, &c.args), c.callee.clone());
        }
        kernel
    }

    fn finish(&self, kernel: Kernel) -> Result<Entry, Unfinished> {
        let code = kernel.finish();
        kgen_compile::check(&code.program)
            .map_err(|e| Unfinished { var: e.var().map(Into::into), message: e.to_string() })?;
        Ok(code)
    }
}

impl Device for Ptx {
    fn module(&self, code: &[Entry], statics: &[Static]) -> Result<String, String> {
        crate::module(code, statics)
    }

    fn symbol(&self, code: &Entry) -> String {
        code.program.name.clone()
    }

    fn device_static(&self, s: &Static) -> DeviceStatic {
        let host = s.elem.host().expect("a static is of a type the host reads");
        DeviceStatic { symbol: s.memory.name.clone(), host: host.into() }
    }
}

/// The prototype of `f`, a function `extern "ptx-func"`, as the kernels that call it see it;
/// `None` if PTX does not pass one of its types.
fn prototype(f: &Fn) -> Option<Prototype> {
    let params: Option<Vec<&str>> = f
        .params
        .iter()
        .map(|p| match &p.ty {
            kgen_lang::Type::Name(ty, _) => Some(ty.as_str()),
            _ => None,
        })
        .collect();
    Prototype::of(&params?, f.ret.as_ref().map(|r| r.ty.as_str())).ok()
}
