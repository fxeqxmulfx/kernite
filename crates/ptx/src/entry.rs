//! A kernel or a function written out, and the module of them the driver loads.

use kgen_core::{Item, Program};

use crate::call::{Returns, formal, passed};
use crate::needs::{Needs, clash};
use crate::statics::{Memory, Static, fits};
use crate::ty::Ty;

/// A PTX instruction as it is written, guard, operands and all; the operands of the
/// [`kgen_core::Inst`] it is the opcode of say what it reads and writes, for the checks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Op(pub String);

/// A kernel's parameter: the type it is passed as, and the register it is loaded into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    /// The type of its register; the `.param` is of [`Param::passed`].
    pub ty: Ty,
    /// The variable it is loaded into, by [`kgen_core::VReg::id`].
    pub reg: usize,
    /// Whether it is a pointer, to be made a global address.
    pub pointer: bool,
}

impl Param {
    /// The type it is passed as, and loaded by `ld.param`, as [`passed`] has it.
    pub fn passed(&self) -> Ty {
        passed(self.ty)
    }
}

/// A kernel written in PTX, an `.entry` the host launches; or a function kernels call, a
/// `.func`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// What the source calls it, for messages: `saxpy`, or `tile::<4>` for an instance.
    pub name: String,
    /// The kernel as a program, for the checks: its name is its symbol in PTX, and each
    /// instruction's opcode is its PTX.
    pub program: Program<Op>,
    /// Its parameters, in order.
    pub params: Vec<Param>,
    /// Each variable's register, its name unique in the entry, and its type, by
    /// [`kgen_core::VReg::id`].
    pub regs: Vec<(String, Ty)>,
    /// The memory it declares, in order.
    pub memory: Vec<Memory>,
    /// What it needs, above [`Needs::FLOOR`], and the first instruction or type that needs
    /// it, for when another kernel of the module needs what cannot go with it.
    pub needs: Vec<(Needs, String)>,
    /// For a function, what it returns; `None` for a kernel.
    pub func: Option<Returns>,
    /// The functions it names, each once, by their symbols: those it calls, and those whose
    /// addresses it takes, which a module declares before it if they come after.
    pub calls: Vec<String>,
}

impl Entry {
    /// What it needs altogether.
    pub fn needs(&self) -> Needs {
        self.needs.iter().fold(Needs::default(), |n, (m, _)| n.and(*m))
    }

    /// What it is called and passed: `.visible .entry` and its parameters, or `.func`, what
    /// it returns, and its parameters.
    fn head(&self) -> String {
        let name = &self.program.name;
        let mut out = match self.func {
            None => format!(".visible .entry {name}("),
            Some(Returns::Value(t)) => format!(".func (.param .{} {name}_retval) {name}(", formal(t).name()),
            Some(_) => format!(".func {name}("),
        };
        for (i, p) in self.params.iter().enumerate() {
            out += if i == 0 { "\n" } else { ",\n" };
            out += &format!("\t.param .{} {name}_param_{i}", self.passed(p).name());
        }
        out += "\n)";
        if self.func == Some(Returns::Never) {
            out += " .noreturn";
        }
        out
    }

    /// The type its parameter `p` is passed as: for a function, its [`formal`] type.
    fn passed(&self, p: &Param) -> Ty {
        match self.func {
            Some(_) => formal(p.ty),
            None => p.passed(),
        }
    }

    /// Its declaration, for a module that names it before it is defined.
    pub fn declaration(&self) -> String {
        self.head() + ";\n"
    }

    /// The `.entry` or the `.func`: its parameters, its registers, the loads of its
    /// parameters into them, and its body.
    pub fn text(&self) -> String {
        let name = &self.program.name;
        let mut out = self.head() + "\n{\n";
        for ty in Ty::ALL.into_iter().filter(|t| t.decl() == *t) {
            let regs: Vec<String> =
                self.regs.iter().filter(|(_, t)| t.decl() == ty).map(|(reg, _)| format!("%{reg}")).collect();
            for some in regs.chunks(8) {
                out += &format!("\t.reg .{} {};\n", ty.name(), some.join(", "));
            }
        }
        for m in &self.memory {
            if let Some(bytes) = m.bytes {
                out += &format!("\t.{} .align {} .b8 {}[{bytes}];\n", m.space.word(), m.align, m.name);
            }
        }
        out += "\n";
        for (i, p) in self.params.iter().enumerate() {
            let reg = &self.regs[p.reg].0;
            out += &format!("\tld.param.{} %{reg}, [{name}_param_{i}];\n", self.passed(p).name());
            if p.pointer {
                out += &format!("\tcvta.to.global.u64 %{reg}, %{reg};\n");
            }
        }
        for item in &self.program.body {
            match item {
                Item::Label(l) => out += &format!("$L{}:\n", l.0),
                Item::Note(note) => out += &format!("\t// {}\n", note.replace('\n', " ")),
                Item::Inst(inst) => out += &format!("\t{}\n", inst.op.0),
            }
        }
        out + "}\n"
    }
}

/// The PTX module of `entries`, kernels and functions, of the version and for the GPU the
/// instructions they use need, and no newer: [`Needs::FLOOR`] if they need nothing more. The
/// driver compiles it for the GPU it runs on, which may be any as new as its `.target` or
/// newer, unless an instruction is gone from newer ones. Each is where it is among
/// `entries`, after a declaration of each it names that comes later; `statics`, the module's
/// memory, before them all, in order.
///
/// It fails if one kernel is for GPUs older than another needs, in the version of PTX the
/// module would be of: PTX has no module for both. It fails if a static does not [`fit`]
/// beside those before it, or has a name in PTX that something else of the module has.
///
/// [`fit`]: fits
pub fn module(entries: &[Entry], statics: &[Static]) -> Result<String, String> {
    let needs = entries.iter().fold(Needs::default(), |n, e| n.and(e.needs()));
    if !needs.possible() {
        let named: Vec<(Needs, String)> = entries
            .iter()
            .flat_map(|e| e.needs.iter().map(move |(n, why)| (*n, format!("`{why}` in `{}`", e.name))))
            .collect();
        return Err(clash(&named));
    }
    let mut out =
        format!("// Written by kgen.\n\n.version {}\n.target {}\n.address_size 64\n", needs.version(), needs.target());
    // The names at the module's scope, and those of a kernel's or a function's parameters,
    // which a static's would hide there: PTX's own `WARP_SZ`, and what each name is.
    let mut names = vec![("WARP_SZ".to_string(), "PTX's own `WARP_SZ`".to_string())];
    for e in entries {
        let (symbol, name) = (&e.program.name, &e.name);
        names.push((symbol.clone(), format!("`{name}`")));
        names.extend((0..e.params.len()).map(|i| (format!("{symbol}_param_{i}"), format!("a parameter of `{name}`"))));
        if e.func.is_some() {
            names.push((format!("{symbol}_retval"), format!("what `{name}` returns")));
        }
        let launched = e.memory.iter().filter(|m| m.bytes.is_none());
        names.extend(launched.map(|m| (m.name.clone(), format!("the shared memory of `{name}`"))));
    }
    for (i, s) in statics.iter().enumerate() {
        fits(&statics[..i], s).map_err(|m| format!("`{}`: {m}", s.name))?;
        if let Some((_, what)) = names.iter().find(|(n, _)| *n == s.memory.name) {
            return Err(format!("`{}` is `{}` in PTX, as {what} is", s.name, s.memory.name));
        }
        names.push((s.memory.name.clone(), format!("`{}`", s.name)));
    }
    if !statics.is_empty() {
        out += "\n";
    }
    for s in statics {
        out += &s.declaration();
    }
    let launched: Vec<&Memory> = entries.iter().flat_map(|e| &e.memory).filter(|m| m.bytes.is_none()).collect();
    if !launched.is_empty() {
        out += "\n";
    }
    for m in launched {
        out += &format!(".extern .shared .align {} .b8 {}[];\n", m.align, m.name);
    }
    let mut declared: Vec<&str> = vec![];
    for e in entries {
        for name in &e.calls {
            if *name != e.program.name && !declared.contains(&name.as_str()) {
                if let Some(later) = entries.iter().find(|l| l.program.name == *name) {
                    out += "\n";
                    out += &later.declaration();
                }
                declared.push(name);
            }
        }
        out += "\n";
        out += &e.text();
        declared.push(&e.program.name);
    }
    Ok(out)
}
