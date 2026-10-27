//! x86-64 under SysV as kgen-build sees it: a machine whose kernels Rust calls in the process.

use kgen_build::{Abi, Backend, Bytes, Called, HostStatic, Naked, Native, StaticError, Unfinished, abi, cooperative};
use kgen_lang::{Fn, Memory, Value};

use crate::SysV;
use crate::intel::{self, Listing};
use crate::lang::{Callee, Kernel, Prototype, Static};

/// x86-64 under SysV: kernels in Intel syntax, as naked functions Rust calls, and the module's
/// memory, which Rust lays out in the process.
pub struct X86;

impl Backend for X86 {
    type Kernel = Kernel;
    type Static = Static;
    type Callee = Callee;
    type Code = Listing;

    fn lay_out(&self, name: &str, memory: &Memory, _: &[Static]) -> Result<Static, StaticError> {
        if intel::taken(name) {
            return Err(StaticError::Name(format!(
                "a kernel's listing names a constant `{{c0}}` and a function `{{f0}}`, and a static by its name, so \
                 one is not named `{name}`"
            )));
        }
        let mut made = Static::new(name, &memory.ty).map_err(StaticError::Type)?;
        if let Some(value) = &memory.value {
            made.init(value).map_err(StaticError::Value)?;
        }
        Ok(made)
    }

    fn callee(&self, g: &Fn) -> Option<Callee> {
        if abi(g) != Some(Abi::Sysv64) {
            return None;
        }
        if cooperative(g) == Ok(true) {
            return Some(Callee::Cooperative);
        }
        // One of no prototype says why where it is compiled; a kernel that calls it, that it
        // does not compile.
        Some(prototype(g).map_or(Callee::Broken, Callee::Calls))
    }

    fn kernel(
        &self,
        _: &Fn,
        name: &str,
        _: &[Value],
        cooperative: bool,
        statics: &[Static],
        callees: &[Called<Callee>],
    ) -> Kernel {
        let mut kernel = Kernel::new(name);
        if cooperative {
            kernel.cooperative();
        }
        for s in statics {
            kernel.module_memory(s);
        }
        for c in callees {
            kernel.callee(&c.name, c.callee.clone());
        }
        kernel
    }

    fn finish(&self, kernel: Kernel) -> Result<Listing, Unfinished> {
        let compiled = kgen_compile::compile(&kernel.finish(), &SysV)
            .map_err(|e| Unfinished { var: e.var().map(Into::into), message: e.to_string() })?;
        Ok(intel::print(&compiled))
    }
}

impl Native for X86 {
    fn naked(&self, code: &Listing) -> Naked {
        Naked {
            code: code.code().map(Into::into).collect(),
            text: code.text(),
            needs: code.features.iter().map(|f| f.name().into()).collect(),
            consts: code.consts.iter().map(|c| Bytes { bytes: c.bytes.clone(), align: c.align }).collect(),
            funcs: code.funcs.clone(),
            statics: code.statics.clone(),
        }
    }

    fn host_static(&self, s: &Static) -> HostStatic {
        HostStatic { elem: s.elem.into(), size: s.size, len: s.len, align: s.align, init: s.init.clone() }
    }
}

/// The prototype of `f`, a function `extern "sysv64"`, as the kernels that call it see it;
/// `None` if SysV passes its types in no registers.
fn prototype(f: &Fn) -> Option<Prototype> {
    let params: Option<Vec<(&str, &str)>> = f
        .params
        .iter()
        .map(|p| match &p.ty {
            kgen_lang::Type::Name(ty, _) => Some((p.name.as_str(), ty.as_str())),
            _ => None,
        })
        .collect();
    Prototype::of(&params?, f.ret.as_ref().map(|r| r.ty.as_str())).ok()
}
