//! A kernel or a function written out, and the module of them the driver loads.

use kernite_build::{Assembly, Kept, When};
use kernite_core::{Item, Program};

use crate::call::{Returns, formal, passed};
use crate::needs::{Needs, clash};
use crate::statics::{Memory, Static, fits};
use crate::ty::Ty;

/// A PTX instruction as it is written, guard, operands and all; the operands of the
/// [`kernite_core::Inst`] it is the opcode of say what it reads and writes, for the checks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Op(pub String);

/// A kernel's parameter: the type it is passed as, and the register it is loaded into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    /// The type of its register; the `.param` is of [`Param::passed`].
    pub ty: Ty,
    /// The variable it is loaded into, by [`kernite_core::VReg::id`].
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
    /// [`kernite_core::VReg::id`].
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
    /// For a kernel, its tuning directives as PTX writes them, `.maxntid 16, 16`, between its
    /// parameters and its body.
    pub directives: Vec<String>,
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

    /// The `.entry` or the `.func`: its parameters, its directives, its registers, the loads of
    /// its parameters into them, and its body.
    pub fn text(&self) -> String {
        let name = &self.program.name;
        let mut out = self.head() + "\n";
        for directive in &self.directives {
            out += &format!("{directive}\n");
        }
        out += "{\n";
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

/// That the code at `i` is kept.
fn code(i: usize) -> When {
    When::Kept(Kept::Code(i))
}

/// The module's header: a part for each pair of what some of `entries` need most of PTX and of the
/// GPU, each the largest of theirs. The part of the pair that what is kept needs is there, and no
/// other: each thing kept is within the pair, and something kept needs each of its values, but a 0,
/// which nothing needs. A pair that no entries reach has no part.
fn headers(entries: &[Entry]) -> Vec<(When, String)> {
    let needs: Vec<Needs> = entries.iter().map(Entry::needs).collect();
    let values = |of: fn(&Needs) -> u32| {
        let mut all: Vec<u32> = needs.iter().map(of).chain([0]).collect();
        all.sort_unstable();
        all.dedup();
        all
    };
    let mut parts = vec![];
    for ptx in values(|n| n.ptx) {
        for sm in values(|n| n.sm) {
            let (within, beyond): (Vec<usize>, Vec<usize>) =
                (0..needs.len()).partition(|&i| needs[i].ptx <= ptx && needs[i].sm <= sm);
            let reach = |value: u32, of: fn(&Needs) -> u32| -> Vec<When> {
                within.iter().filter(|&&i| of(&needs[i]) == value).map(|&i| code(i)).collect()
            };
            let (by_ptx, by_sm) = (reach(ptx, |n| n.ptx), reach(sm, |n| n.sm));
            if (ptx > 0 && by_ptx.is_empty()) || (sm > 0 && by_sm.is_empty()) {
                continue;
            }
            let mut when = vec![When::Not(Box::new(When::Any(beyond.into_iter().map(code).collect())))];
            when.extend((ptx > 0).then_some(When::Any(by_ptx)));
            when.extend((sm > 0).then_some(When::Any(by_sm)));
            let at = Needs { ptx, sm, ..Needs::default() };
            let (version, target) = (at.version(), at.target());
            let text = format!("// Written by kernite.\n\n.version {version}\n.target {target}\n.address_size 64\n");
            parts.push((When::All(when), text));
        }
    }
    parts
}

/// Why the module is none where what is kept has no module: a reason for each set of at most four
/// of `entries` that no module has, though one has any smaller set of them. It is said where all of
/// the set are kept, and names the set alone. Four are enough: [`Needs::and`] takes the largest
/// PTX, the newest GPU, the oldest `below` and the first `since` each on its own, so each is one
/// entry's, and those, at most four, have no module either. Entries added to a set never make a
/// module of it.
fn clashes(entries: &[Entry]) -> Vec<(When, String)> {
    let needs: Vec<Needs> = entries.iter().map(Entry::needs).collect();
    let join = |set: &[usize]| set.iter().fold(Needs::default(), |n, &i| n.and(needs[i]));
    let all: Vec<usize> = (0..entries.len()).collect();
    if join(&all).possible() {
        return vec![];
    }
    // A kernel that needs nothing is of no set that no module has.
    let asking: Vec<usize> = all.into_iter().filter(|&i| needs[i] != Needs::default()).collect();
    let mut sets: Vec<Vec<usize>> = vec![];
    for size in 1..=4 {
        choose(&asking, size, &mut vec![], &mut |set| {
            if !sets.iter().any(|found| found.iter().all(|i| set.contains(i))) && !join(set).possible() {
                sets.push(set.to_vec());
            }
        });
    }
    let named = |i: usize| {
        let (name, records) = (&entries[i].name, &entries[i].needs);
        records.iter().map(move |(n, why)| (*n, format!("`{why}` in `{name}`")))
    };
    let reason = |set: &[usize]| clash(&set.iter().flat_map(|&i| named(i)).collect::<Vec<_>>());
    sets.into_iter().map(|set| (When::All(set.iter().copied().map(code).collect()), reason(&set))).collect()
}

/// Each choice of `size` of `from` in order, after the `set` chosen so far, given to `each`.
fn choose(from: &[usize], size: usize, set: &mut Vec<usize>, each: &mut dyn FnMut(&[usize])) {
    if set.len() == size {
        return each(set);
    }
    for (at, &i) in from.iter().enumerate() {
        set.push(i);
        choose(&from[at + 1..], size, set, each);
        set.pop();
    }
}

/// The PTX module of `entries`, kernels and functions, with `statics`, the module's memory, put
/// together from what the program keeps of them: whichever are kept, it is the module of those
/// alone. That is of the version and for the GPU the instructions of what is kept need, and no
/// newer: [`Needs::FLOOR`] if they need nothing more. The driver compiles it for the GPU it runs
/// on, which may be any as new as its `.target` or newer, unless an instruction is gone from newer
/// ones. Each is where it is among `entries`, after a declaration of each it names that comes
/// later; `statics` before them all, in order.
///
/// It is none if something kept is for GPUs older than something else kept needs, in the version of
/// PTX the module would be of: PTX has no module for both. It is none if a static does not [`fit`]
/// beside those before it, kept or not, or has a name in PTX that something else kept has.
///
/// [`fit`]: fits
pub fn module(entries: &[Entry], statics: &[Static]) -> Assembly {
    let kept = |j: usize| When::Kept(Kept::Static(j));
    let (mut parts, mut refusals) = (headers(entries), clashes(entries));
    // The names at the module's scope, and those of a kernel's or a function's parameters,
    // which a static's would hide there: PTX's own `WARP_SZ`, and what each name is, and whose.
    let mut names = vec![("WARP_SZ".to_string(), "PTX's own `WARP_SZ`".to_string(), When::always())];
    for (i, e) in entries.iter().enumerate() {
        let (symbol, name) = (&e.program.name, &e.name);
        names.push((symbol.clone(), format!("`{name}`"), code(i)));
        let params = (0..e.params.len()).map(|p| (format!("{symbol}_param_{p}"), format!("a parameter of `{name}`")));
        let returned = e.func.is_some().then(|| (format!("{symbol}_retval"), format!("what `{name}` returns")));
        let launched = e.memory.iter().filter(|m| m.bytes.is_none());
        let shared = launched.map(|m| (m.name.clone(), format!("the shared memory of `{name}`")));
        names.extend(params.chain(returned).chain(shared).map(|(n, what)| (n, what, code(i))));
    }
    for (j, s) in statics.iter().enumerate() {
        if let Err(m) = fits(&statics[..j], s) {
            refusals.push((When::always(), format!("`{}`: {m}", s.name)));
        }
        for (_, what, whose) in names.iter().filter(|(n, ..)| *n == s.memory.name) {
            let message = format!("`{}` is `{}` in PTX, as {what} is", s.name, s.memory.name);
            refusals.push((When::All(vec![kept(j), whose.clone()]), message));
        }
        names.push((s.memory.name.clone(), format!("`{}`", s.name), kept(j)));
    }
    if !statics.is_empty() {
        parts.push((When::Any((0..statics.len()).map(kept).collect()), "\n".into()));
    }
    parts.extend(statics.iter().enumerate().map(|(j, s)| (kept(j), s.declaration())));
    let launched: Vec<(usize, &Memory)> = (entries.iter().enumerate())
        .flat_map(|(i, e)| e.memory.iter().filter(|m| m.bytes.is_none()).map(move |m| (i, m)))
        .collect();
    if !launched.is_empty() {
        parts.push((When::Any(launched.iter().map(|&(i, _)| code(i)).collect()), "\n".into()));
    }
    for (i, m) in launched {
        parts.push((code(i), format!(".extern .shared .align {} .b8 {}[];\n", m.align, m.name)));
    }
    // A function that a kernel or a function names, if it comes after, is declared before the first
    // of those kept that name it.
    for (i, e) in entries.iter().enumerate() {
        for name in &e.calls {
            let Some(at) = entries.iter().position(|l| l.program.name == *name).filter(|&at| at > i) else { continue };
            let before = (0..i).filter(|&j| entries[j].calls.contains(name)).map(code).collect();
            let when = When::All(vec![code(i), code(at), When::Not(Box::new(When::Any(before)))]);
            parts.push((when, "\n".to_string() + &entries[at].declaration()));
        }
        parts.push((code(i), "\n".to_string() + &e.text()));
    }
    Assembly { parts, refusals }
}
