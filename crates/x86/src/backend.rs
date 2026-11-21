//! x86-64 under SysV as kernite-build sees it: a machine whose kernels Rust calls in the process.

use std::collections::BTreeSet;

use kernite_build::{
    Abi, Backend, Bytes, Called, Done, HostStatic, Naked, Native, StaticError, Unfinished, abi, cooperative,
};
use kernite_core::Clobber;
use kernite_lang::{Fn, Memory, Value};

use crate::SysV;
use crate::docs;
use crate::forms::Feature;
use crate::intel::{self, Listing};
use crate::lang::{Callee, Kernel, Prototype, Static};
use crate::sysv::{CALL_CLOBBERS, VECS};

/// x86-64 under SysV: kernels in Intel syntax, as naked functions Rust calls, and the module's
/// memory, which Rust lays out in the process.
pub struct X86;

impl Backend for X86 {
    type Kernel = Kernel;
    type Static = Static;
    type Callee = Callee;
    type Code = Listing;

    fn lay_out(&self, name: &str, memory: &Memory) -> Result<Static, StaticError> {
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

    fn doc(&self, name: &str) -> Option<String> {
        docs::of(name)
    }

    fn callee(&self, g: &Fn, params: &[String]) -> Option<Callee> {
        if abi(g) != Some(Abi::Sysv64) {
            return None;
        }
        if cooperative(g) == Ok(true) {
            return Some(Callee::Cooperative);
        }
        // One of no prototype says why where it is compiled; a kernel that calls it, that it
        // does not compile.
        let calls = if g.unsafe_fn { Callee::Unsafe } else { Callee::Calls };
        Some(prototype(g, params).map_or(Callee::Broken, calls))
    }

    fn kernel(
        &self,
        f: &Fn,
        name: &str,
        _: &[Value],
        cooperative: bool,
        statics: &[Static],
        callees: &[Called<Callee>],
    ) -> Kernel {
        let mut kernel = Kernel::new(name);
        if !f.unsafe_fn {
            kernel.safe();
        }
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

    fn calls(&self, kernel: &Kernel) -> Vec<String> {
        kernel.calls().map(Into::into).collect()
    }

    fn finish(&self, mut kernel: Kernel, done: &[Done<Listing>]) -> Result<Listing, Unfinished> {
        let decls = std::mem::take(&mut kernel.decls);
        let reach = std::mem::take(&mut kernel.reach);
        let any = kernel.reaches_any();
        if let Some((at, message)) = kernel.unowned().or_else(|| kernel.unordered()).or_else(|| kernel.unproven()) {
            return Err(Unfinished { at, message });
        }
        let program = kernel.finish(|called| changed_by(called, done), |name| may_run_sse(name, done));
        let allocated = kernite_compile::compile(&program, &SysV)
            .map_err(|e| Unfinished::blaming(e.var(), &decls, e.to_string()))?;
        let mut listing = intel::print(&allocated, reach).map_err(|message| Unfinished { at: None, message })?;
        if any {
            listing.changes = CALL_CLOBBERS.to_vec();
        }
        listing.features = needed_by(&listing, done);
        Ok(listing)
    }
}

/// The CPU features past x86-64's own that `listing` needs: those of its instructions, and of each function it may call
/// in turn, as the listings of `done` say, for a kernel runs only where each function it calls does. A function with no
/// listing adds none.
fn needed_by(listing: &Listing, done: &[Done<Listing>]) -> Vec<Feature> {
    let (mut seen, mut features) = (vec![], listing.features.clone());
    let mut todo = listing.funcs.iter().chain(&listing.reach).collect::<Vec<_>>();
    while let Some(name) = todo.pop() {
        if seen.contains(&name) {
            continue;
        }
        seen.push(name);
        if let Some(callee) = done.iter().find(|(done, _)| done == name).and_then(|(_, l)| l.as_ref().ok()) {
            features.extend(&callee.code.features);
            todo.extend(callee.code.funcs.iter().chain(&callee.code.reach));
        }
    }
    features.sort_by_key(|f| f.name());
    features.dedup();
    features
}

/// What a call of the functions `called`, each by what the source calls it, does to registers, as the listings of
/// `done` say: it changes those that each changes, and each function it calls in turn, counted once however often it is
/// called, itself among them; and clears the upper half of each vector register if one of them runs `vzeroupper`. None
/// if a function has no listing, for what it does is not known.
fn changed_by(called: &[String], done: &[Done<Listing>]) -> Option<Vec<Clobber>> {
    let (mut seen, mut changes, mut clears) = (vec![], BTreeSet::new(), false);
    let mut todo = called.iter().collect::<Vec<_>>();
    while let Some(name) = todo.pop() {
        if seen.contains(&name) {
            continue;
        }
        seen.push(name);
        let listing = &done.iter().find(|(done, _)| done == name)?.1.as_ref().ok()?.code;
        changes.extend(&listing.changes);
        clears |= listing.clears_upper;
        todo.extend(listing.funcs.iter().chain(&listing.reach));
    }
    let upper = VECS.iter().filter(|_| clears);
    Some(changes.into_iter().copied().map(Clobber::All).chain(upper.copied().map(Clobber::Upper)).collect())
}

/// Whether a call of the function `name` may run SSE code, as its listing in `done` says; it may where it has none, for
/// what it does is not known.
fn may_run_sse(name: &str, done: &[Done<Listing>]) -> bool {
    let listing = done.iter().find(|(done, _)| done == name).and_then(|(_, listing)| listing.as_ref().ok());
    listing.is_none_or(|listing| listing.code.reaches_sse)
}

impl Native for X86 {
    fn naked(&self, code: &Listing) -> Naked {
        Naked {
            code: code.code().map(Into::into).collect(),
            text: code.to_string(),
            needs: code.features.iter().map(|f| f.name().into()).collect(),
            consts: code.consts.iter().map(|c| Bytes { bytes: c.bytes.clone(), align: c.align }).collect(),
            funcs: code.funcs.clone(),
            reach: code.reach.clone(),
            statics: code.statics.clone(),
        }
    }

    fn host_static(&self, s: &Static) -> HostStatic {
        HostStatic { elem: s.elem.into(), size: s.size, len: s.len, align: s.align, init: s.init.clone() }
    }
}

/// The prototype of `f`, a function `extern "sysv64"`, whose parameters are of the types `params`, as the kernels
/// that call it see it; `None` if SysV does not pass one of its types whole in a register.
fn prototype(f: &Fn, params: &[String]) -> Option<Prototype> {
    let params: Vec<(&str, &str)> =
        f.params.iter().map(|p| p.name.as_str()).zip(params.iter().map(String::as_str)).collect();
    Prototype::of(&params, f.ret.as_ref().map(|r| r.ty.as_str())).ok()
}
