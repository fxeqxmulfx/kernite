//! PTX as the language writes it: [`Kernel`] is the [`Target`] that checks each call against
//! the forms of [`crate::forms`], and writes the instruction out.
//!
//! The types are PTX's: `pred`; `b8` to `b128`, bits of no type in particular; `u8` to `u64`
//! and `s8` to `s64`; `f32` and `f64`; and the floats of 16 bits, `f16` and `bf16`, alone or
//! two together, `f16x2` and `bf16x2`, and `tf32`. An instruction is called by its PTX name,
//! words joined with dots, with the operands PTX writes: `add.rn.f32(acc, acc, x)`,
//! `setp.lt.u32(p, i, n)`, `ld.global.v4.f32((a, b, c, d), [p + 16])`.
//!
//! - A float `add`, `sub`, `mul`, `mad`, `fma`, `div`, `sqrt` or `rcp` says how it rounds,
//!   `.rn`, so that ptxas neither fuses nor approximates it: what it computes is what it says.
//! - An address is a variable and a displacement, `[p + 16]`; a generic or a global one is
//!   in 64 bits. A kernel's pointers arrive as global addresses, for `ld.global` and
//!   `st.global`.
//! - Memory the kernel declares is a variable too: `let tile: shared<[f32; 256], 16>` is 256
//!   `f32`s in shared memory at an address that is a multiple of 16, `shared<[f32]>` as many
//!   as the launch gives, and `local<[u32; 8]>` a thread's own. Its name is its address, for
//!   an instruction of its state space, as far as its end: `ld.shared.f32(v, [tile + 16])`.
//!   `mov.u32(a, tile)` puts the address in a variable, and `cvta.shared.u64(g, tile)` the
//!   generic one.
//! - `mov` and `cvt` read the special registers: `tid.x`, `laneid`, `clock64`.
//! - Operands taken together are a tuple, `(a, b)`, or variables declared together; two
//!   results written at once are `p | q`; `_` is a result left unwritten, where PTX lets one
//!   be; `!p` is a predicate read inverted, and `-a` a source negated, where PTX negates one.
//! - A video instruction takes a part of a 32-bit variable, as PTX selects it: `a.b0` is its
//!   lowest byte, `a.h1` its high half, and `d.b310` three bytes of a result.
//! - A texture or a surface is its handle, the `u64` a kernel takes for a CUDA texture or
//!   surface object, in brackets with the coordinates in it, an array's layer first:
//!   `tex._2d.v4.f32.f32((r, g, b, a), [t, (x, y)])`, `sust.b.a1d.b32.trap([s, (i, x)], v)`,
//!   `suq.width.b32(w, [s])`.
//! - `if p { .. }` runs each instruction inside where `p`, a `pred`, holds, and `if !p` where
//!   it does not; a guarded `bra(label)` is a branch. `brx.idx(i, (a, b, c))` jumps to the
//!   label `i` picks of those taken together, through a table it declares before it.
//! - A kernel returns nothing: `ret()`, which the end of the body adds if control reaches it.
//!   A function kernels call, [`Kernel::func`], takes and returns PTX's types, `u32` and
//!   `f16x2` but not `pred`: `ret(v)` gives back what it returns, and one that never returns,
//!   `-> !`, ends each way out in `exit()`, `trap()` or a call of one that never returns.
//! - `call(r, f, (a, b))` calls the function `f`, of the prototype [`Kernel::callee`] gives,
//!   with `a` and `b`, and writes what it returns to `r`, or to nothing, `_`; `call(f, a)`
//!   passes one, and `call(f)` none. An argument is a variable of the parameter's type, as
//!   `ptxas` checks one, or a value. `mov.u64(fp, f)` puts the address of `f` in `fp`, which
//!   `call(r, fp, (a, b))` calls as the prototype the operands' types make, and
//!   `call(r, fp, (a, b), (f, g))` as one of the functions taken together last, checked
//!   against each.
//!
//! What an instruction needs of PTX and of the GPU, its [`Needs`], the kernel needs: an
//! instruction that no GPU has with what the kernel needs already is an error where it is.

use std::collections::BTreeSet;

use kgen_core::Width::{W8, W16, W32, W64, W128};
use kgen_core::{
    Access, Arg, Base, Class, Flow, Inst, Item, Label, Mem, Operand, PReg, Program, RegInfo, Signature, VReg, Width,
};
use kgen_lang::{Fault, Guard, Operand as Given, Target};

use crate::forms::{self, Bound, Check, Count, Imm, Part, Role, Space, Spec, TyRef};
use crate::{Entry, Kind, Memory, Needs, Op, Param, Prototype, Returns, Ty, formal};

/// A variable: a register of the type it was declared, or a part of one, or memory the
/// kernel declares in a state space, which stands for its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Var {
    /// A register.
    Reg {
        /// Which.
        reg: VReg,
        /// Its type.
        ty: Ty,
    },
    /// A part of a 32-bit register, as a video instruction selects it: `r.b0`.
    Part {
        /// Which register.
        reg: VReg,
        /// Its type.
        ty: Ty,
        /// Which part.
        part: Part,
    },
    /// Memory, by its place among the kernel's.
    Mem(usize),
}

/// Shared memory a kernel declares, at most, beside what the launch gives it: 48 KB.
const SHARED: u64 = 48 << 10;

/// Local memory a thread has, at most: 512 KB.
const LOCAL: u64 = 512 << 10;

/// A kernel being written in the language, for PTX.
pub struct Kernel {
    /// What the source calls it, for messages: `saxpy`, or `tile::<4>` for an instance. Its
    /// program is called by its symbol, as PTX names it.
    name: String,
    p: Program<Op>,
    params: Vec<Param>,
    regs: Vec<(String, Ty)>,
    taken: BTreeSet<String>,
    memory: Vec<Memory>,
    /// Each memory's variable, and its type as declared: `shared<[f32; 256], 16>`.
    declared: Vec<(String, String)>,
    /// The names memory has taken, with the kernel's own and its parameters'.
    named: BTreeSet<String>,
    /// The bytes of shared and of local memory declared so far, each aligned as declared.
    shared: u64,
    local: u64,
    partial: Vec<Partial>,
    needs: Vec<(Needs, String)>,
    /// The tables declared so far: of labels `brx.idx` jumps through, of call targets, and
    /// prototypes.
    tables: usize,
    /// For a function, what it returns; `None` for a kernel.
    func: Option<Returns>,
    /// The functions it may name: each one it calls, and the kernels, whose addresses it
    /// takes.
    callees: Vec<Callee>,
    /// Those it named, each once, by their symbols.
    calls: Vec<String>,
}

/// A function or a kernel that a kernel may name: by what the source calls it, as an operand
/// names it, and by its symbol in PTX; and for a function, which it calls, its prototype,
/// `None` for a kernel, whose address it takes.
struct Callee {
    name: String,
    symbol: String,
    proto: Option<Prototype>,
}

/// A write under a guard, which leaves what was there where the guard does not hold. It is
/// taken for a write while nothing reads the variable but under the same guard: a read under
/// another, or after the guard changes, may find what was there before, so then the write
/// reads it too; unless a later write under the read's guard is what the read finds. A write
/// under the opposite guard, the other side of `if p { .. } else { .. }`, makes the two a
/// write on every path.
struct Partial {
    var: VReg,
    /// The guard, while it holds as it did; `None` once it is written again, or once control
    /// can come from elsewhere.
    side: Option<(VReg, bool)>,
    /// The instruction, by its place in the body, and the operand among its arguments.
    at: usize,
    arg: usize,
}

/// An instruction's operands written out: their text, what they read and write, what they
/// need and the special registers and kernels among them that need it, the labels of a table
/// it jumps through, and the functions it names; and as they are written, how many are addresses so far, the bytes the
/// instruction reaches at once at each, and whether they are those in brackets, where a
/// register is of the type the form says, though `+wide` lets the others be wider.
#[derive(Default)]
struct Written {
    texts: Vec<String>,
    args: Vec<Arg<VReg>>,
    needs: Needs,
    named: Vec<String>,
    table: Vec<u32>,
    funcs: Vec<String>,
    addrs: usize,
    reach: u64,
    inside: bool,
}

/// Why operands are not those of a form.
enum Miss {
    /// There are too few or too many: it takes from the first to the second.
    Count(usize, usize),
    /// One is wrong: how many before it were right, whether it has the shape the form wants
    /// there, and what is wrong.
    At(usize, bool, Fault),
}

impl Kernel {
    /// An empty kernel called `name` in messages, and `symbol` in PTX: what
    /// [`symbol`](crate::symbol) writes for it.
    pub fn new(name: impl Into<String>, symbol: impl Into<String>) -> Self {
        let symbol = symbol.into();
        Kernel {
            name: name.into(),
            named: BTreeSet::from([symbol.clone()]),
            p: Program {
                name: symbol,
                signature: Signature { params: vec![], result: None },
                regs: vec![],
                params: vec![],
                results: vec![],
                body: vec![],
                slots: vec![],
                consts: vec![],
            },
            params: vec![],
            regs: vec![],
            taken: BTreeSet::new(),
            memory: vec![],
            declared: vec![],
            shared: 0,
            local: 0,
            partial: vec![],
            needs: vec![],
            tables: 0,
            func: None,
            callees: vec![],
            calls: vec![],
        }
    }

    /// An empty function called `name` in messages, and `symbol` in PTX, which kernels call: a
    /// `.func`, which returns nothing until [`Target::result`] says what it returns.
    pub fn func(name: impl Into<String>, symbol: impl Into<String>) -> Self {
        let mut f = Kernel::new(name, symbol);
        f.named.insert(format!("{}_retval", f.p.name));
        f.func = Some(Returns::Nothing);
        f
    }

    /// Lets the kernel name `name`, which is `symbol` in PTX: a function it calls, of the
    /// prototype `proto`; or with `None` a kernel, whose address it takes for the GPU to launch
    /// it.
    pub fn callee(&mut self, name: impl Into<String>, symbol: impl Into<String>, proto: Option<Prototype>) {
        let symbol = symbol.into();
        self.named.insert(symbol.clone());
        self.callees.push(Callee { name: name.into(), symbol, proto });
    }

    /// The kernel as written.
    pub fn finish(self) -> Entry {
        Entry {
            name: self.name,
            program: self.p,
            params: self.params,
            regs: self.regs,
            memory: self.memory,
            needs: self.needs,
            func: self.func,
            calls: self.calls,
        }
    }

    /// A new variable, its register named after it: with `_` after a special register's
    /// name, and then a number if another has the name.
    fn new_var(&mut self, name: &str, ty: Ty) -> VReg {
        let class = match ty {
            Ty::Pred => Class::Pred,
            Ty::F32 => Class::Float,
            _ => Class::Gpr,
        };
        let id = self.p.regs.len() as u32;
        self.p.regs.push(RegInfo { class, name: name.to_string() });
        let base = if forms::reserved(name) { format!("{name}_") } else { name.to_string() };
        let (mut reg, mut k) = (base.clone(), 1);
        while self.taken.contains(&reg) {
            reg = format!("{base}_{k}");
            k += 1;
        }
        self.taken.insert(reg.clone());
        self.regs.push((reg, ty));
        VReg { id, class }
    }

    fn reg(&self, reg: VReg) -> &str {
        &self.regs[reg.id as usize].0
    }

    /// What an instruction under `guard` starts with: `@%p `, `@!%p `, or nothing.
    fn guarded(&self, guard: Option<(VReg, bool)>) -> String {
        match guard {
            Some((pred, negated)) => format!("@{}%{} ", if negated { "!" } else { "" }, self.reg(pred)),
            None => String::new(),
        }
    }

    /// Memory declared as `let name: ty`, where `ty` is `space<args>`: `shared<[f32; 256], 16>`
    /// with `args` `[f32; 256], 16>`.
    fn new_memory(&mut self, name: &str, ty: &str, space: Space, args: &str) -> Result<Var, String> {
        let word = space.word();
        let shape = || {
            format!(
                "`{word}<..>` is an array of PTX's types and what its address is a multiple of, `{word}<[f32; 256], \
                 16>`, or one, `{word}<f32>`; not `{ty}`"
            )
        };
        let args = args.strip_suffix('>').ok_or_else(shape)?;
        let (array, align) = match args.rsplit_once(", ") {
            Some((array, align)) => (array, Some(align)),
            None => (args, None),
        };
        if array.contains(", ") {
            return Err(shape());
        }
        let (elem, len) = match array.strip_prefix('[').and_then(|a| a.strip_suffix(']')) {
            Some(inner) => match inner.split_once("; ") {
                Some((elem, len)) => (elem, Some(len.parse::<u64>().map_err(|_| shape())?)),
                None => (inner, None),
            },
            None => (array, Some(1)),
        };
        let elem = match Ty::parse(elem) {
            Some(Ty::Pred) => {
                return Err("memory holds no `pred`, which is a register's alone: a `u8` holds a bool".into());
            }
            None => return Err(format!("memory holds {HELD}; not `{elem}`")),
            Some(t) => t,
        };
        let size = u64::from(elem.bits()) / 8;
        let align = match align {
            None => size,
            Some(a) => match a.parse::<u64>() {
                Ok(a) if a.is_power_of_two() && a >= size => a,
                Ok(a) if a.is_power_of_two() => {
                    return Err(format!("an array of `{}` is aligned to {size} at least, not {a}", elem.name()));
                }
                _ => return Err(format!("what an address is a multiple of is a power of two, not {a}")),
            },
        };
        let bytes = match len {
            Some(0) => return Err(format!("`{ty}` holds nothing")),
            Some(n) => Some(n.checked_mul(size).ok_or_else(|| format!("`{ty}` is more bytes than there are"))?),
            None if space == Space::Shared => {
                if let Some(other) = self.memory.iter().position(|m| m.bytes.is_none()) {
                    return Err(format!(
                        "`{name}` would start where `{}` does, after the memory with a size: a kernel has one \
                         `shared<[T]>`, which the launch gives",
                        self.declared[other].0
                    ));
                }
                None
            }
            None => {
                return Err(format!(
                    "`{ty}` has no size, which only shared memory may leave to the launch: `local<[{}; 64]>`",
                    elem.name()
                ));
            }
        };
        if let Some(n) = bytes {
            let (had, most) = match space {
                Space::Shared => (&mut self.shared, SHARED),
                _ => (&mut self.local, LOCAL),
            };
            let all = had.next_multiple_of(align).saturating_add(n);
            if all > most {
                return Err(match space {
                    Space::Shared => format!(
                        "this makes the kernel's shared memory {all} bytes; it declares {} KB at most, and has more \
                         as `shared<[{}]>`, which the launch gives",
                        most >> 10,
                        elem.name()
                    ),
                    _ => format!("this makes a thread's local memory {all} bytes, and a thread has {} KB", most >> 10),
                });
            }
            *had = all;
        }
        // PTX predefines `WARP_SZ`. Memory without a size is the module's, after the kernel,
        // named as C++ names a function's static.
        let base = match bytes {
            None => crate::local(&self.p.name, name),
            Some(_) if name == "WARP_SZ" => format!("{name}_"),
            Some(_) => name.to_string(),
        };
        let (mut text, mut k) = (base.clone(), 1);
        while self.named.contains(&text) {
            text = format!("{base}_{k}");
            k += 1;
        }
        self.named.insert(text.clone());
        self.memory.push(Memory { name: text, space, align, bytes });
        self.declared.push((name.to_string(), ty.to_string()));
        Ok(Var::Mem(self.memory.len() - 1))
    }

    /// What the kernel needs so far.
    fn needed(&self) -> Needs {
        self.needs.iter().fold(Needs::default(), |n, (m, _)| n.and(*m))
    }

    /// Notes that the kernel needs `needs` for `what`, an instruction; or says why no module
    /// has it as well as what the kernel needs already.
    fn require(&mut self, needs: Needs, what: &str) -> Result<(), String> {
        let had = self.needed();
        let all = had.and(needs);
        if !all.possible() {
            let mut named: Vec<(Needs, String)> =
                self.needs.iter().map(|(n, w)| (*n, format!("`{w}` before it"))).collect();
            named.push((needs, format!("`{what}`")));
            return Err(crate::clash(&named));
        }
        if all != had {
            self.needs.push((needs, what.to_string()));
        }
        Ok(())
    }

    /// Adds the instruction, under the predicate `guard` if there is one, held or not; keeping
    /// account of the writes under guards.
    fn push(&mut self, line: String, args: Vec<Arg<VReg>>, guard: Option<(VReg, bool)>, flow: Flow) {
        let at = self.p.body.len();
        let side = guard;
        let reads: Vec<VReg> = args.iter().flat_map(Arg::reads).chain(guard.map(|(pred, _)| pred)).collect();
        // Where the guard holds, what a write under it left is what a read under it finds,
        // and nothing written before.
        let last: Vec<(VReg, usize)> = reads
            .iter()
            .filter_map(|&r| {
                Some((r, self.partial.iter().rposition(|w| w.var == r && w.side == side && side.is_some())?))
            })
            .collect();
        let body = &mut self.p.body;
        let mut k = 0;
        self.partial.retain(|w| {
            k += 1;
            let covered = last.iter().any(|&(r, j)| r == w.var && k <= j + 1);
            let stale = !covered && reads.contains(&w.var) && (w.side.is_none() || w.side != side);
            if stale && let Item::Inst(inst) = &mut body[w.at] {
                inst.args[w.arg].access = Access::ReadWrite;
            }
            !stale
        });
        for (i, a) in args.iter().enumerate() {
            let Some(var) = a.writes() else { continue };
            match side {
                Some((pred, negated))
                    if !self.partial.iter().any(|w| w.var == var && w.side == Some((pred, !negated))) =>
                {
                    self.partial.push(Partial { var, side, at, arg: i });
                }
                _ => self.partial.retain(|w| w.var != var),
            }
            for w in &mut self.partial {
                if w.side.is_some_and(|(p, _)| p == var) {
                    w.side = None;
                }
            }
        }
        let implicit = guard.map(|(pred, _)| Arg { operand: Operand::Reg(pred, W32), access: Access::Read });
        self.p.body.push(Item::Inst(Inst { op: Op(line), args, implicit: implicit.into_iter().collect(), flow }));
    }

    /// `args` as the operands of `b`, the form of the instruction `name`.
    fn operands(&self, b: &Bound, name: &str, args: &[Given<Var>]) -> Result<Written, Miss> {
        let specs: Vec<&Spec> = b.form.operands.iter().filter(|s| s.when.is_none_or(|w| b.has(w))).collect();
        let least = specs.iter().filter(|s| !s.optional).count();
        if !(least..=specs.len()).contains(&args.len()) {
            return Err(Miss::Count(least, specs.len()));
        }
        let mut w = Written { reach: reach(b, args), ..Written::default() };
        for (i, (spec, a)) in specs.iter().zip(args).enumerate() {
            let text =
                self.operand(b, spec, a, name, &mut w).map_err(|m| Miss::At(i, shaped(b, spec, a), Fault::at(i, m)))?;
            w.texts.push(text);
            w.needs = w.needs.and(spec.needs);
        }
        // PTX leaves undefined an index past the labels, where ptxas 13.4 jumps to the first.
        if let Some(i) = specs.iter().position(|s| s.role == Role::Labels).and_then(|l| l.checked_sub(1))
            && let Given::Int(x) = args[i]
            && !(0..w.table.len() as i128).contains(&x)
        {
            let m = match w.table.len() {
                1 => format!("`{name}` takes 0 here, the index of its one label; not {x}"),
                n => format!("`{name}` takes 0 to {} here, the index of one of its {n} labels; {x} is not one", n - 1),
            };
            return Err(Miss::At(i, true, Fault::at(i, m)));
        }
        // A policy's primary size is part of its total, as the ISA has it, and ptxas 13.4
        // refuses more known now.
        if b.form.op == "createpolicy"
            && let [.., Given::Int(primary), Given::Int(total)] = args
            && read_as(Ty::U32, *primary) > read_as(Ty::U32, *total)
        {
            let i = args.len() - 2;
            let m = format!("`{name}` takes a primary size here no more than the total, {total}; {primary} is more");
            return Err(Miss::At(i, true, Fault::at(i, m)));
        }
        if b.form.checks.contains(&Check::Neg) {
            let negs: Vec<usize> = (0..args.len()).filter(|&i| specs[i].neg).collect();
            let negated = |i: usize| matches!(args[i], Given::Neg(_));
            if let &[x, y, z] = negs.as_slice()
                && negated(x) != negated(y)
                && negated(z)
            {
                let m = format!("`{name}` negates `a * b`, as `-a` or `-b` does, or `c`; not both");
                return Err(Miss::At(z, true, Fault::at(z, m)));
            }
        }
        Ok(w)
    }

    /// `a` as the operand `spec` of `b`, written out, with what it reads and writes added to
    /// `w`; or what is wrong with it.
    fn operand(&self, b: &Bound, spec: &Spec, a: &Given<Var>, name: &str, w: &mut Written) -> Result<String, String> {
        match (spec.role, a) {
            (Role::Label, Given::Label(l)) => {
                w.args.push(Arg { operand: Operand::Label(Label(*l)), access: Access::Read });
                Ok(format!("$L{l}"))
            }
            // The table is declared before the instruction, under the name it has here.
            (Role::Labels, Given::Vector(labels)) => {
                for l in labels {
                    let Given::Label(l) = l else {
                        return Err(format!(
                            "`{name}` jumps to one of labels taken together; this holds {}",
                            self.given(l)
                        ));
                    };
                    w.table.push(*l);
                    w.args.push(Arg { operand: Operand::Label(Label(*l)), access: Access::Read });
                }
                Ok(format!("$T{}", self.tables))
            }
            (Role::Handle, Given::List(items)) => self.handle(b, spec, items, name, w),
            (Role::Handle, &Given::Mem(kgen_lang::Mem { base: Some(v), index: None, disp: 0 })) => {
                self.handle(b, spec, &[Given::Var(v)], name, w)
            }
            (Role::Addr(access), Given::Mem(m)) => self.address(b, m, access, name, w),
            (Role::Addr(_), Given::Data(_)) => Err(format!(
                "`{name}` takes an address that starts at a variable, as `[p + 4]`; this is memory holding a value \
                 known now"
            )),
            (Role::Func, Given::Func(f)) => self.address_of(f, w),
            (Role::Imm(imm), Given::Int(x)) => known(imm, *x, name),
            (Role::Dst | Role::Src, _) => self.value(b, spec, a, name, w),
            (_, Given::Data(_)) => Err(format!("`{name}` takes a value as it is: `1.5`, not `[1.5]`")),
            (_, a) => Err(format!("`{name}` wants {}; this is {}", describe(b, spec), self.given(a))),
        }
    }

    /// `a` as the register operand `spec`: one, a few together, or two written at once.
    fn value(&self, b: &Bound, spec: &Spec, a: &Given<Var>, name: &str, w: &mut Written) -> Result<String, String> {
        let Some((second, optional)) = &spec.second else {
            return self.several(b, spec, a, name, w);
        };
        match a {
            Given::Pair(x, y) => {
                let x = self.several(b, spec, x, name, w)?;
                let y = self.several(b, second, y, name, w)?;
                w.needs = w.needs.and(second.needs);
                if x == "_" && y == "_" {
                    return Err(format!("`{name}` writes one of the two at least: `p | _` or `_ | q`"));
                }
                Ok(format!("{x}|{y}"))
            }
            _ if *optional => self.several(b, spec, a, name, w),
            _ => Err(format!("`{name}` writes two results at once, `p | q`; this is {}", self.given(a))),
        }
    }

    /// `a` as `spec`, as many registers as it takes together: braced, as PTX writes a fragment
    /// even of one, `{%r1}`, which may be given as one, `d`, or as one taken together, `(d,)`.
    fn several(&self, b: &Bound, spec: &Spec, a: &Given<Var>, name: &str, w: &mut Written) -> Result<String, String> {
        let n = lanes(b, spec);
        let braced = matches!(spec.count, Count::Fixed(_));
        if n == 1 && !braced {
            // ptxas leaves a register of a vector unwritten, but not the one of a name with
            // neither `.v2` nor `.v4`.
            if spec.count == Count::Vec && matches!(a, Given::Special(s) if s == "_") {
                return Err(format!(
                    "`{name}` writes its one result; `_` stands for one of a vector's, as `.v2` and `.v4` write"
                ));
            }
            // Nor does it store a special register but of a vector.
            if let Given::Special(s) = a
                && let Some(r) = forms::sreg(s).filter(|_| spec.count == Count::Vec && spec.sreg)
            {
                return Err(format!(
                    "`{name}` stores a special register as one of a vector's, as `.v2` and `.v4` store; `mov.{}` \
                     puts `{s}` in a variable",
                    r.ty.name()
                ));
            }
            return self.one(b, spec, a, name, w);
        }
        let parts = match a {
            Given::Vector(parts) if parts.len() == n => parts.as_slice(),
            Given::Vector(_) => return Err(format!("`{name}` wants {}; this is {}", describe(b, spec), self.given(a))),
            _ if n == 1 => std::slice::from_ref(a),
            _ => return Err(format!("`{name}` wants {}; this is {}", describe(b, spec), self.given(a))),
        };
        let mut texts = vec![];
        for (k, part) in parts.iter().enumerate() {
            texts.push(match spec.lead {
                Some(t) if k == 0 => self.one(b, &Spec { ty: Some(TyRef::Fixed(t)), ..spec.clone() }, part, name, w)?,
                _ => self.one(b, spec, part, name, w)?,
            });
        }
        if texts.iter().all(|t| t == "_") {
            return Err(format!("`{name}` writes one of them at least"));
        }
        if spec.one_var && parts.iter().any(is_value) && !parts.iter().any(anchors) {
            let part = parts.iter().find_map(|p| match p {
                Given::Special(s) => s.split_once('.'),
                _ => None,
            });
            return Err(match part {
                Some((v, x)) => format!(
                    "`{name}` takes values known now among these only beside a variable: one of the kernel's, or a \
                     special register but a part of a vector, as `{v}.{x}` is of `{v}`"
                ),
                None => format!("`{name}` takes values known now among these only beside a variable"),
            });
        }
        Ok(format!("{{{}}}", texts.join(", ")))
    }

    /// `a` as one register of `spec`: a variable, or what else `spec` lets it be.
    fn one(&self, b: &Bound, spec: &Spec, a: &Given<Var>, name: &str, w: &mut Written) -> Result<String, String> {
        let t = b.ty(spec.ty.expect("a register's operand has a type"));
        let write = spec.role == Role::Dst;
        let var = |reg: VReg, ty: Ty, w: &mut Written| {
            let access = if write { Access::Write } else { Access::Read };
            w.args.push(Arg { operand: Operand::Reg(reg, width(ty)), access });
            format!("%{}", self.reg(reg))
        };
        match a {
            Given::Var(Var::Reg { ty, .. } | Var::Part { ty, .. }) if spec.memory => Err(format!(
                "`{name}` takes the address of memory the kernel declares here, as `tile`; this is a `{}` variable",
                ty.name()
            )),
            Given::Var(v @ (Var::Reg { reg, ty } | Var::Part { reg, ty, .. })) => {
                let part = match v {
                    Var::Part { part, .. } => Some(*part),
                    _ => None,
                };
                match (spec.part, part) {
                    (None, Some(p)) => {
                        return Err(format!("`{name}` takes a whole variable here; this is a part of one, `.{p}`"));
                    }
                    (Some(sel), Some(p)) if !sel.takes(p) => {
                        return Err(format!("`{name}` takes {}; `.{p}` is not one", sel.what()));
                    }
                    _ => {}
                }
                let fits = if b.form.wide && !w.inside { ty.fits_wide(t) } else { ty.fits(t) };
                if !fits {
                    return Err(format!("`{name}` wants a `{}` variable; this is a `{}`", t.name(), ty.name()));
                }
                let text = var(*reg, *ty, w);
                Ok(match part {
                    Some(p) => format!("{text}.{p}"),
                    None => text,
                })
            }
            Given::Var(Var::Mem(i)) => {
                let (m, (v, _)) = (&self.memory[*i], &self.declared[*i]);
                if !spec.memory || write {
                    return Err(format!(
                        "`{name}` takes no memory as an operand: `mov.u32(a, {v})` puts the address of `{v}` in \
                         `a`, and `[{v}]` is what it holds"
                    ));
                }
                if b.space() != Space::Generic && b.space() != m.space {
                    return Err(format!(
                        "`{name}` takes the address of {} memory; `{v}` is {} memory",
                        b.space().word(),
                        m.space.word()
                    ));
                }
                Ok(m.name.clone())
            }
            Given::Special(s) if s == "_" => match spec.sink {
                true => Ok("_".into()),
                false => Err(format!("`{name}` has no result here that it may leave unwritten, `_`")),
            },
            Given::Special(s) => match forms::sreg(s) {
                None => match forms::beyond(s) {
                    Some(why) => Err(format!("`{s}` is a special register of PTX's that kgen does not read: {why}")),
                    None => Err(format!("`{s}` is not defined")),
                },
                Some(_) if write => {
                    Err(format!("`{name}` writes a variable; `{s}` is a special register, which it reads"))
                }
                Some(r) if !spec.sreg => Err(format!("`{name}` reads no special register; `mov.{}` does", r.ty.name())),
                Some(r) if !r.read_as(t) => {
                    Err(format!("`{name}` reads a `{}`; `{s}` is a `{}`", t.name(), r.ty.name()))
                }
                Some(r) => {
                    w.needs = w.needs.and(r.needs);
                    if r.needs != Needs::default() {
                        w.named.push(format!("%{}", r.name));
                    }
                    Ok(format!("%{s}"))
                }
            },
            Given::Not(p) if spec.not && t == Ty::Pred => match **p {
                Given::Var(Var::Reg { reg, ty: Ty::Pred }) => Ok(format!("!{}", var(reg, Ty::Pred, w))),
                _ => Err(format!("`{name}` reads a predicate inverted, `!p`; this is {} inverted", self.given(p))),
            },
            Given::Not(_) => Err(format!("`{name}` takes no operand inverted")),
            Given::Neg(v) if spec.neg => match **v {
                Given::Var(Var::Reg { .. } | Var::Part { .. }) => Ok(format!("-{}", self.one(b, spec, v, name, w)?)),
                _ => Err(format!("`{name}` negates a variable, `-a`; this is {} negated", self.given(v))),
            },
            Given::Neg(_) => Err(format!("`{name}` takes no operand negated")),
            Given::Data(_) => Err(format!("`{name}` takes a value as it is: `1.5`, not `[1.5]`")),
            Given::Int(_) | Given::F32(_) | Given::F64(_) | Given::F16(_) | Given::Bf16(_)
                if write || spec.reg_only =>
            {
                Err(format!("`{name}` wants {}; this is {}", describe(b, spec), self.given(a)))
            }
            Given::Int(x) if spec.fraction => match x {
                1 => int(t, 1, name),
                _ => Err(fraction(name, spec.step, x)),
            },
            Given::Int(x) => {
                let v = read_as(t, *x);
                if let Some((lo, hi)) = spec.range
                    && !(i128::from(lo)..=i128::from(hi)).contains(&v)
                {
                    return Err(format!("`{name}` takes {lo} to {hi} here; {x} is not one of them"));
                }
                if let Some(step) = spec.step
                    && v % i128::from(step) != 0
                {
                    return Err(format!("`{name}` takes a multiple of {step} here; {x} is not one"));
                }
                if let Some(mask) = spec.mask
                    && v & !i128::from(mask) != 0
                {
                    let x = if v < 0 { x.to_string() } else { format!("{v:#x}") };
                    return Err(format!("`{name}` takes no bits here but those of {mask:#x}; {x} has others"));
                }
                int(t, *x, name)
            }
            Given::F32(x)
                if spec.fraction
                    && !(*x > 0.0 && *x <= 1.0 && spec.step.is_none_or(|n| (x * n as f32).fract() == 0.0)) =>
            {
                Err(fraction(name, spec.step, x))
            }
            Given::F32(x) => match t {
                Ty::F32 | Ty::B32 => Ok(format!("0f{:08X}", x.to_bits())),
                Ty::F64 => Err(format!("`{name}` wants an `f64`; {x} is an `f32`: `{x}f64`")),
                _ if t.kind() == Kind::Float => Err(half(t, name)),
                _ => Err(format!("`{name}` wants {}; this is {}", describe(b, spec), self.given(a))),
            },
            Given::F64(x) => match t {
                Ty::F64 | Ty::B64 => Ok(format!("0d{:016X}", x.to_bits())),
                Ty::F32 => Err(format!("`{name}` wants an `f32`; {x}f64 is an `f64`")),
                _ if t.kind() == Kind::Float => Err(half(t, name)),
                _ => Err(format!("`{name}` wants {}; this is {}", describe(b, spec), self.given(a))),
            },
            Given::F16(x) | Given::Bf16(x) => match t {
                Ty::B16 => Ok(format!("0x{x:04X}")),
                _ if t.kind() == Kind::Float => Err(half(t, name)),
                _ => Err(format!("`{name}` wants {}; this is {}", describe(b, spec), self.given(a))),
            },
            a => Err(format!("`{name}` wants {}; this is {}", describe(b, spec), self.given(a))),
        }
    }

    /// `items`, in brackets, as the texture or surface `spec` is: its handle, and the
    /// coordinates in it where it takes them.
    fn handle(
        &self,
        b: &Bound,
        spec: &Spec,
        items: &[Given<Var>],
        name: &str,
        w: &mut Written,
    ) -> Result<String, String> {
        if items.len() != spec.inner.len() {
            let these = match items.len() {
                1 => "1 operand".into(),
                n => format!("{n} operands"),
            };
            return Err(format!("`{name}` wants {}; this is {these} in brackets", describe(b, spec)));
        }
        w.inside = true;
        let texts: Result<Vec<String>, String> =
            spec.inner.iter().zip(items).map(|(s, a)| self.several(b, s, a, name, w)).collect();
        w.inside = false;
        Ok(format!("[{}]", texts?.join(", ")))
    }

    /// The address `m`, in the state space `b` names in its place.
    fn address(
        &self,
        b: &Bound,
        m: &kgen_lang::Mem<Var>,
        access: Access,
        name: &str,
        w: &mut Written,
    ) -> Result<String, String> {
        if m.index.is_some() {
            return Err(format!(
                "`{name}` takes an address of a variable and a displacement; work out `p + 4 * i` first, as \
                 `mad.wide.u32(a, i, 4, p)` does"
            ));
        }
        let space = b.spaces()[w.addrs];
        w.addrs += 1;
        let wide = space.wide() && !b.form.checks.contains(&Check::Addr32);
        let (base, ty) = match m.base {
            // ptxas takes an address known now of local memory, `[256]`, as PTX's ISA has it;
            // but ptxas lays out local memory, for the kernel's variables and for its own, so
            // what lies at one is no variable's.
            None => {
                let bits = if wide { "a 64-bit" } else { "a" };
                return Err(format!("`{name}` takes an address that starts at {bits} variable"));
            }
            Some(Var::Mem(i)) => return self.symbol(i, m.disp, space, w.reach, access, name),
            Some(Var::Part { part, .. }) => {
                return Err(format!("`{name}` takes an address in a whole variable; this is a part of one, `.{part}`"));
            }
            Some(Var::Reg { reg, ty }) => (reg, ty),
        };
        if ty.kind() == Kind::Float || ty.kind() == Kind::Pred {
            return Err(format!("`{name}` takes an address in an integer variable; this is a `{}`", ty.name()));
        }
        if wide && ty.bits() != 64 {
            return Err(format!("`{name}` takes an address in a 64-bit variable; this is a `{}`", ty.name()));
        }
        if !wide && !matches!(ty.bits(), 32 | 64) {
            return Err(format!("`{name}` takes an address in 32 bits or 64; this is a `{}`", ty.name()));
        }
        let disp = i32::try_from(m.disp).map_err(|_| format!("{} bytes are too far for an address", m.disp))?;
        let mem = Mem { base: Base::Reg(base), index: None, disp, width: W32 };
        w.args.push(Arg { operand: Operand::Mem(mem), access });
        Ok(match disp {
            0 => format!("[%{}]", self.reg(base)),
            d => format!("[%{}+{d}]", self.reg(base)),
        })
    }

    /// The address `disp` bytes into memory `i`, where `name` reaches `size` bytes at once in
    /// `space`. PTX leaves undefined what reaches memory by its name but in its state space,
    /// and an address that is no multiple of the bytes reached at once; and what lies past the
    /// memory is not its.
    fn symbol(
        &self,
        i: usize,
        disp: i64,
        space: Space,
        size: u64,
        access: Access,
        name: &str,
    ) -> Result<String, String> {
        let (m, (v, ty)) = (&self.memory[i], &self.declared[i]);
        let verb = match access {
            Access::Write => "writes",
            Access::ReadWrite => "reads and writes",
            Access::Read | Access::Ignored => "reads",
        };
        let word = m.space.word();
        match space {
            s if s == m.space => {}
            Space::Generic => {
                return Err(format!(
                    "`{name}` {verb} at a generic address; `{v}` is {word} memory, which an instruction reaches by \
                     its name only as `.{word}`"
                ));
            }
            s => return Err(format!("`{name}` {verb} {} memory; `{v}` is {word} memory", s.word())),
        }
        let at = match disp {
            0 => format!("[{v}]"),
            d if d < 0 => format!("[{v} - {}]", d.unsigned_abs()),
            d => format!("[{v} + {d}]"),
        };
        if disp < 0 {
            return Err(format!("`{name}` {verb} at `{at}`, before `{v}` starts"));
        }
        if let Some(n) = m.bytes
            && disp.unsigned_abs() + size > n
        {
            return Err(format!("`{name}` {verb} {size} bytes at `{at}`, past the end of `{v}`, {n} bytes"));
        }
        if m.align < size {
            let stem = ty.strip_suffix('>').unwrap_or(ty);
            let stem = match stem.rsplit_once(", ") {
                Some((stem, align)) if align.parse::<u64>().is_ok() => stem,
                _ => stem,
            };
            return Err(format!(
                "`{name}` {verb} {size} bytes at once, at an address that is a multiple of {size}; `{v}` is at a \
                 multiple of {}: `{stem}, {size}>`",
                m.align
            ));
        }
        if !disp.unsigned_abs().is_multiple_of(size) {
            return Err(format!(
                "`{name}` {verb} {size} bytes at once, at an address that is a multiple of {size}; `{at}` is not one"
            ));
        }
        Ok(match disp {
            0 => format!("[{}]", m.name),
            d => format!("[{}+{d}]", m.name),
        })
    }

    /// The address of the function or the kernel `f`, which `mov` puts in a variable.
    fn address_of(&self, f: &str, w: &mut Written) -> Result<String, String> {
        let callee = self.callee_of(f)?;
        if callee.proto.is_none() {
            // A kernel's address is for the GPU to launch it, as CUDA's dynamic parallelism
            // does from Kepler's sm_35 on.
            w.needs = w.needs.and(Needs { ptx: 31, sm: 35, ..Needs::default() });
            w.named.push(f.to_string());
        }
        w.funcs.push(callee.symbol.clone());
        Ok(callee.symbol.clone())
    }

    /// The function or the kernel `f`, which the kernel may name; or why it may name none
    /// such.
    fn callee_of(&self, f: &str) -> Result<&Callee, String> {
        match self.callees.iter().find(|c| c.name == f) {
            Some(callee) => Ok(callee),
            None => Err(format!("`{f}` is no function of PTX's: a kernel calls those `extern \"ptx-func\"`")),
        }
    }

    /// The prototype of `f`, a function the kernel calls by its operand `at`, and its symbol;
    /// or why it calls no such function.
    fn called(&self, f: &str, at: usize) -> Result<(&Prototype, &str), Fault> {
        let callee = self.callee_of(f).map_err(|m| Fault::at(at, m))?;
        match &callee.proto {
            Some(proto) => Ok((proto, &callee.symbol)),
            None => Err(Fault::at(at, format!("`{f}` is a kernel, which the host launches; `call` calls a function"))),
        }
    }

    /// `call`: of a function of the module by its name, `call(r, f, (a, b))`, or of the one
    /// at the address in a variable, `call(r, fp, (a, b))`, as the prototype its operands'
    /// types make, or as one of the functions taken together last, `call(fp, a, (f, g))`.
    fn call(&mut self, name: &str, args: &[Given<Var>], guard: Option<(VReg, bool)>) -> Result<(), Fault> {
        if name != "call" && name != "call.uni" {
            return Err(Fault::new(format!(
                "PTX's `call` is `call`, or `call.uni` where the threads of a warp all call the same function; not \
                 `{name}`"
            )));
        }
        let shape = "`call` is `call(f, (a, b))`, or `call(r, f, (a, b))` for a function that returns a value, \
                     `f` a function of the module or a variable that holds the address of one";
        let targets: Option<Vec<&str>> = match args.last() {
            Some(Given::Vector(fs)) if !fs.is_empty() && fs.iter().all(|f| matches!(f, Given::Func(_))) => {
                Some(fs.iter().filter_map(|f| if let Given::Func(f) = f { Some(f.as_str()) } else { None }).collect())
            }
            _ => None,
        };
        let rest = &args[..args.len() - usize::from(targets.is_some())];
        let at = rest.iter().position(|a| matches!(a, Given::Func(_)));
        let (result, callee, passed) = match (at, rest.len()) {
            (Some(0), 1 | 2) => (None, 0, rest.get(1)),
            (Some(1), 2 | 3) => (Some(0), 1, rest.get(2)),
            (Some(i), _) if i < 2 => return Err(Fault::new(shape)),
            (Some(i), _) => return Err(Fault::at(i, shape)),
            (None, 1 | 2) => (None, 0, rest.get(1)),
            (None, 3) => (Some(0), 1, rest.get(2)),
            (None, _) => return Err(Fault::new(shape)),
        };
        let passed: Vec<&Given<Var>> = match passed {
            None => vec![],
            Some(Given::Vector(xs)) => xs.iter().collect(),
            Some(a) => vec![a],
        };
        let at_args = callee + 1;
        let mut reads = vec![];
        let mut line = String::new();
        let (target, texts, flow) = match (&rest[callee], &targets) {
            (Given::Func(f), None) => {
                let (proto, symbol) = self.called(f, callee)?;
                let texts = self.passing(f, proto, &passed, result.map(|r| &rest[r]), &mut reads, at_args)?;
                let flow = match (proto.returns, guard) {
                    (Returns::Never, None) => Flow::Return,
                    _ => Flow::Next,
                };
                let symbol = symbol.to_string();
                self.named_fn(&symbol);
                (symbol, texts, flow)
            }
            (Given::Func(f), Some(_)) => {
                return Err(Fault::at(
                    callee,
                    format!(
                        "`call` names the function it calls, `{f}`, or takes its address from a variable, of one \
                         of the functions taken together last; not both"
                    ),
                ));
            }
            (fp, targets) => {
                let fp = match fp {
                    Given::Var(Var::Reg { reg, ty })
                        if matches!(ty.kind(), Kind::Bit | Kind::Unsigned | Kind::Signed)
                            && matches!(ty.bits(), 32 | 64) =>
                    {
                        reads.push(Arg { operand: Operand::Reg(*reg, width(*ty)), access: Access::Read });
                        format!("%{}", self.reg(*reg))
                    }
                    a => {
                        return Err(Fault::at(
                            callee,
                            format!(
                                "`call` calls a function by its name, or at the address in a 32- or 64-bit \
                                 integer variable; this is {}",
                                self.given(a)
                            ),
                        ));
                    }
                };
                // As `ptxas` has a call through a variable.
                self.require(Needs { ptx: 21, sm: 20, ..Needs::default() }, &format!("{name} {fp}"))
                    .map_err(Fault::new)?;
                match targets {
                    Some(targets) => {
                        let mut texts: Option<Vec<String>> = None;
                        let mut never = true;
                        let mut symbols = vec![];
                        for t in targets {
                            let (proto, symbol) = self.called(t, args.len() - 1)?;
                            symbols.push(symbol.to_string());
                            never &= proto.returns == Returns::Never;
                            let mut some = vec![];
                            let these =
                                self.passing(t, proto, &passed, result.map(|r| &rest[r]), &mut some, at_args)?;
                            match &texts {
                                Some(first) if *first != these => {
                                    return Err(Fault::at(
                                        at_args,
                                        format!(
                                            "`{t}` takes other types than `{}` does, which write a value known \
                                             now another way: put it in a variable first",
                                            targets[0]
                                        ),
                                    ));
                                }
                                Some(_) => {}
                                None => {
                                    reads.append(&mut some);
                                    texts = Some(these);
                                }
                            }
                        }
                        for symbol in &symbols {
                            self.named_fn(symbol);
                        }
                        line += &format!("$C{}: .calltargets {};\n\t", self.tables, symbols.join(", "));
                        let flow = if never && guard.is_none() { Flow::Return } else { Flow::Next };
                        let mut texts = texts.expect("a call has a target");
                        texts.push(format!("$C{}", self.tables));
                        self.tables += 1;
                        (fp, texts, flow)
                    }
                    None => {
                        let mut texts = self.prototype(&passed, result.map(|r| &rest[r]), &mut reads, at_args)?;
                        let proto = texts.pop().expect("the prototype comes last");
                        line += &format!("$P{}: .callprototype {proto};\n\t", self.tables);
                        texts.push(format!("$P{}", self.tables));
                        self.tables += 1;
                        (fp, texts, Flow::Next)
                    }
                }
            }
        };
        line += &self.guarded(guard);
        // What it returns, the function, its arguments, and the table it is one of.
        let [r, a, table @ ..] = texts.as_slice() else { unreachable!("a call writes its result and its arguments") };
        line += name;
        line += " ";
        if !r.is_empty() {
            line += &format!("({r}), ");
        }
        line += &target;
        if !a.is_empty() || !r.is_empty() {
            line += &format!(", ({a})");
        }
        for t in table {
            line += &format!(", {t}");
        }
        line += ";";
        self.push(line, reads, guard, flow);
        Ok(())
    }

    /// Notes that the kernel names the function of the symbol `f`.
    fn named_fn(&mut self, f: &str) {
        if !self.calls.iter().any(|c| c == f) {
            self.calls.push(f.to_string());
        }
    }

    /// The result and the arguments of a call of `f`, of `proto`, written out: the variable it
    /// writes, `_` or nothing, and the arguments joined, with what they read and write added to
    /// `args`. The arguments are the operand `at`.
    fn passing(
        &self,
        f: &str,
        proto: &Prototype,
        passed: &[&Given<Var>],
        result: Option<&Given<Var>>,
        args: &mut Vec<Arg<VReg>>,
        at: usize,
    ) -> Result<Vec<String>, Fault> {
        let r = match (proto.returns, result) {
            (Returns::Value(t), None) => {
                return Err(Fault::new(format!(
                    "`{f}` returns a `{}`: `call(r, {f}, ..)` writes it to `r`, or `call(_, {f}, ..)` to nothing",
                    t.name()
                )));
            }
            (Returns::Value(t), Some(r)) => match r {
                Given::Var(Var::Reg { reg, ty }) if ty.fits(t) => {
                    args.push(Arg { operand: Operand::Reg(*reg, width(*ty)), access: Access::Write });
                    format!("%{}", self.reg(*reg))
                }
                Given::Special(s) if s == "_" => "_".into(),
                Given::Var(Var::Reg { ty, .. }) => {
                    return Err(Fault::at(
                        0,
                        format!("`{f}` returns a `{}`; this is a `{}` variable", t.name(), ty.name()),
                    ));
                }
                r => {
                    return Err(Fault::at(
                        0,
                        format!("`{f}` returns a `{}`, to a variable or `_`; this is {}", t.name(), self.given(r)),
                    ));
                }
            },
            (Returns::Nothing, Some(_)) => {
                return Err(Fault::at(0, format!("`{f}` returns nothing: `call({f}, ..)` calls it")));
            }
            (Returns::Never, Some(_)) => {
                return Err(Fault::at(0, format!("`{f}` never returns: `call({f}, ..)` calls it")));
            }
            (_, None) => String::new(),
        };
        if passed.len() != proto.params.len() {
            let takes = match proto.params.len() {
                0 => "no arguments".into(),
                1 => "1 argument".into(),
                n => format!("{n} arguments"),
            };
            return Err(Fault::at(at, format!("`{f}` takes {takes}, not {}", passed.len())));
        }
        let mut texts = vec![];
        for (k, (&t, a)) in proto.params.iter().zip(passed).enumerate() {
            let wants = match proto.params.len() {
                1 => format!("`{f}` takes a `{}`", t.name()),
                _ => format!("`{f}` takes a `{}` as argument {}", t.name(), k + 1),
            };
            texts.push(self.pass(t, &wants, a, args).map_err(|m| Fault::at(at, m))?);
        }
        Ok(vec![r, texts.join(", ")])
    }

    /// The result and the arguments of a call through a prototype, written out as
    /// [`Kernel::passing`] writes them, and last the prototype their types make.
    fn prototype(
        &self,
        passed: &[&Given<Var>],
        result: Option<&Given<Var>>,
        args: &mut Vec<Arg<VReg>>,
        at: usize,
    ) -> Result<Vec<String>, Fault> {
        let var = |a: &Given<Var>, at: usize, what: &str| match a {
            Given::Var(Var::Reg { ty: Ty::Pred, .. }) => Err(Fault::at(at, NO_PRED)),
            Given::Var(Var::Reg { reg, ty }) => Ok((*reg, *ty)),
            a => Err(Fault::at(
                at,
                format!(
                    "a call through a variable, as the prototype its operands' types make, {what} variables: this \
                     is {}; taking the functions it may call last, `call(r, fp, (a, b), (f, g))`, lets it be more",
                    self.given(a)
                ),
            )),
        };
        let (r, ret) = match result {
            Some(r) => {
                let (reg, ty) = var(r, 0, "writes")?;
                args.push(Arg { operand: Operand::Reg(reg, width(ty)), access: Access::Write });
                (format!("%{}", self.reg(reg)), format!("(.param .{} _) ", formal(ty).name()))
            }
            None => (String::new(), String::new()),
        };
        let (mut texts, mut params) = (vec![], vec![]);
        for a in passed {
            let (reg, ty) = var(a, at, "passes")?;
            args.push(Arg { operand: Operand::Reg(reg, width(ty)), access: Access::Read });
            texts.push(format!("%{}", self.reg(reg)));
            params.push(format!(".param .{} _", formal(ty).name()));
        }
        let proto = match (r.is_empty(), params.is_empty()) {
            (true, true) => "_ ".into(),
            _ => format!("{ret}_ ({})", params.join(", ")),
        };
        Ok(vec![r, texts.join(", "), proto])
    }

    /// `a` as a value of type `t` passed to a function or given back from one, written out: a
    /// variable of a type that fits it, as `ptxas` checks a call, or a value known now, as the
    /// type writes it; with what it reads added to `args`. `wants` says what it is to be.
    fn pass(&self, t: Ty, wants: &str, a: &Given<Var>, args: &mut Vec<Arg<VReg>>) -> Result<String, String> {
        let not = |what: String| format!("{wants}; this is {what}");
        match a {
            Given::Var(Var::Reg { reg, ty }) if ty.fits(t) => {
                args.push(Arg { operand: Operand::Reg(*reg, width(*ty)), access: Access::Read });
                Ok(format!("%{}", self.reg(*reg)))
            }
            Given::Var(Var::Reg { ty, .. }) => Err(not(format!("a `{}` variable", ty.name()))),
            Given::Int(x) => match t {
                Ty::F16 => Err(not(format!("the integer {x}: `{x}.0f16` is an `f16`"))),
                Ty::Bf16 => Err(not(format!("the integer {x}: `{x}.0bf16` is a `bf16`"))),
                _ => int(t, *x, "").map_err(|_| not(format!("the integer {x}"))),
            },
            Given::F32(x) => match t {
                Ty::F32 | Ty::B32 => Ok(format!("0f{:08X}", x.to_bits())),
                Ty::F64 => Err(not(format!("the f32 {x}: `{x}f64` is an `f64`"))),
                _ => Err(not(format!("the f32 {x}"))),
            },
            Given::F64(x) => match t {
                Ty::F64 | Ty::B64 => Ok(format!("0d{:016X}", x.to_bits())),
                _ => Err(not(format!("the f64 {x}"))),
            },
            Given::F16(x) if matches!(t, Ty::F16 | Ty::B16) => Ok(format!("0x{x:04X}")),
            Given::Bf16(x) if matches!(t, Ty::Bf16 | Ty::B16) => Ok(format!("0x{x:04X}")),
            Given::Func(f) => Err(not(format!("the function `{f}`: `mov.u64(a, {f})` puts its address in `a`"))),
            a => Err(not(self.given(a))),
        }
    }

    /// `ret(v)`, of a function that returns a value of type `t`: `v`, which it stores where the
    /// caller finds it, and then returns.
    fn give_back(&mut self, name: &str, t: Ty, args: &[Given<Var>], guard: Option<(VReg, bool)>) -> Result<(), Fault> {
        let (f, symbol) = (self.name.clone(), self.p.name.clone());
        let [a] = args else {
            return Err(Fault::new(format!("`{f}` returns a `{}`, which `{name}(v)` gives back", t.name())));
        };
        let mut reads = vec![];
        let text =
            self.pass(t, &format!("`{f}` returns a `{}`", t.name()), a, &mut reads).map_err(|m| Fault::at(0, m))?;
        let on = self.guarded(guard);
        let line = format!("{on}st.param.{} [{symbol}_retval], {text};\n\t{on}{name};", formal(t).name());
        let flow = if guard.is_none() { Flow::Return } else { Flow::Next };
        self.push(line, reads, guard, flow);
        Ok(())
    }

    /// What `a` is, for an error.
    fn given(&self, a: &Given<Var>) -> String {
        match a {
            Given::Var(Var::Reg { ty, .. }) => format!("a `{}` variable", ty.name()),
            Given::Var(Var::Part { ty, part, .. }) => format!("a part of a `{}` variable, `.{part}`", ty.name()),
            Given::Var(Var::Mem(i)) => format!("`{}`, {} memory", self.declared[*i].0, self.memory[*i].space.word()),
            Given::Special(name) if name == "_" => "`_`".into(),
            Given::Special(name) => format!("`{name}`"),
            Given::Int(x) => format!("the integer {x}"),
            Given::F32(x) => format!("the float {x}"),
            Given::F64(x) => format!("the f64 {x}"),
            Given::F16(x) => format!("the f16 {}", kgen_lang::Value::F16(*x)),
            Given::Bf16(x) => format!("the bf16 {}", kgen_lang::Value::Bf16(*x)),
            Given::Mem(_) => "an address".into(),
            Given::Data(_) => "memory holding data".into(),
            Given::Label(_) => "a label".into(),
            Given::Func(f) => format!("the function `{f}`"),
            Given::Vector(vs) => format!("{} operands taken together", vs.len()),
            Given::Pair(..) => "two operands joined by `|`".into(),
            Given::Not(_) => "an operand inverted".into(),
            Given::Neg(_) => "an operand negated".into(),
            Given::List(items) => format!("{} operands in brackets", items.len()),
        }
    }
}

/// How many bytes `b` reaches at once at each address, of `args`: what it loads, stores or
/// works on, as `ld.v4.b32` 16 and `atom.f16x2` 4, and a fragment `wmma` loads or stores,
/// whose size each row is aligned to; what `cp.async` copies, as its third operand says; an
/// mbarrier's 8; a row of a matrix `ldmatrix` loads, 16; or 1, for what reaches no bytes in
/// particular.
fn reach(b: &Bound, args: &[Given<Var>]) -> u64 {
    match b.form.op {
        "ld" | "st" | "atom" | "red" | "wmma" => {
            let data = b.form.operands.iter().find(|s| matches!(s.role, Role::Dst | Role::Src) && s.when.is_none());
            data.map_or(1, |s| u64::from(b.ty(s.ty.expect("data has a type")).bits()) / 8 * lanes(b, s) as u64)
        }
        "mbarrier" => 8,
        "ldmatrix" => 16,
        "cp" if b.has("mbarrier") => 8,
        "cp" => match args.get(2) {
            Some(&Given::Int(n @ (4 | 8 | 16))) => n as u64,
            _ => 1,
        },
        _ => 1,
    }
}

/// How many registers `spec` of `b` is.
fn lanes(b: &Bound, spec: &Spec) -> usize {
    match spec.count {
        Count::One => 1,
        Count::Fixed(n) => n,
        Count::Vec => b.lanes(),
    }
}

/// Whether `a` is what `spec` is, but for what it holds: a few together where it takes a
/// few, an address where it takes one, memory where it takes memory.
fn shaped(b: &Bound, spec: &Spec, a: &Given<Var>) -> bool {
    match (spec.role, a) {
        (Role::Addr(_), Given::Mem(_))
        | (Role::Handle, Given::List(_) | Given::Mem(_))
        | (Role::Label, Given::Label(_))
        | (Role::Labels, Given::Vector(_))
        | (Role::Func, Given::Func(_))
        | (Role::Imm(_), Given::Int(_)) => true,
        (Role::Dst | Role::Src, Given::Var(v)) => matches!(v, Var::Mem(_)) == spec.memory,
        (Role::Dst | Role::Src, Given::Vector(parts)) => lanes(b, spec) == parts.len(),
        (Role::Dst | Role::Src, Given::Pair(..)) => spec.second.is_some(),
        (Role::Dst | Role::Src, Given::Mem(_) | Given::Label(_)) => false,
        (Role::Dst | Role::Src, _) => lanes(b, spec) == 1,
        _ => false,
    }
}

/// The width of a register of type `t`, for what reads and writes it.
fn width(t: Ty) -> Width {
    match t.decl().bits() {
        8 => W8,
        16 => W16,
        64 => W64,
        128 => W128,
        _ => W32,
    }
}

/// Whether `a` is a value known now, as `7` or `1.5`, and not a variable.
fn is_value(a: &Given<Var>) -> bool {
    matches!(a, Given::Int(_) | Given::F32(_) | Given::F64(_) | Given::F16(_) | Given::Bf16(_))
}

/// Whether `a` is a variable that values known now may be beside, in registers taken
/// together: ptxas 13.4 takes `{%r1, 8}` and `{%laneid, 8}`, but not `{%tid.x, 8}`, though
/// it takes `{%tid.x, %r1}`.
fn anchors(a: &Given<Var>) -> bool {
    match a {
        Given::Var(Var::Reg { .. } | Var::Part { .. }) => true,
        Given::Special(s) => s != "_" && !s.contains('.'),
        _ => false,
    }
}

/// The integer `x` as an operand of type `t`.
fn int(t: Ty, x: i128, name: &str) -> Result<String, String> {
    match t {
        // An integer is a predicate as PTX's ISA says, 0 false and the rest true; written as
        // 0 or 1, as ptxas 13.4 reads `set`'s and `setp`'s `c` in its low 32 bits alone.
        Ty::Pred => Ok(if x == 0 { "0" } else { "1" }.into()),
        Ty::F32 if (x as f32) as i128 == x => Ok(format!("0f{:08X}", (x as f32).to_bits())),
        Ty::F64 if (x as f64) as i128 == x => Ok(format!("0d{:016X}", (x as f64).to_bits())),
        Ty::F32 | Ty::F64 => Err(format!("`{name}` wants an `{}`; {x} is not one", t.name())),
        _ if t.kind() == Kind::Float => Err(half(t, name)),
        Ty::B128 => Err(format!("`{name}` takes a `b128` only from a variable")),
        _ => {
            let bits = t.bits();
            if !(-(1i128 << (bits - 1))..(1i128 << bits)).contains(&x) {
                return Err(format!("`{name}` wants {bits} bits; {x} does not fit them"));
            }
            // Past 2^63 as the 64 bits are, which ptxas reads the same.
            Ok((x as i64).to_string())
        }
    }
}

/// `x` as an integer of type `t` reads it, where it fits its bits: to an `s32`, 2^32 - 1 is
/// -1, and to a `u32`, -1 is 2^32 - 1.
fn read_as(t: Ty, x: i128) -> i128 {
    let bits = t.bits();
    match t.kind() {
        Kind::Unsigned if (-(1i128 << (bits - 1))..0).contains(&x) => x + (1i128 << bits),
        Kind::Signed if (1i128 << (bits - 1)..1i128 << bits).contains(&x) => x - (1i128 << bits),
        _ => x,
    }
}

/// Why `x` is no fraction for `name`, which takes a multiple of one over `step`, if any.
fn fraction(name: &str, step: Option<i64>, x: impl std::fmt::Display) -> String {
    let of = step.map(|n| format!(", a multiple of 1/{n}")).unwrap_or_default();
    format!("`{name}` takes a fraction here{of}, more than 0 and at most 1; {x} is not one")
}

/// Why `name` takes no value known now as its operand of type `t`, a float PTX keeps in bits.
fn half(t: Ty, name: &str) -> String {
    let fill = match t {
        Ty::F16 => "mov.b16(h, 1.5f16)",
        Ty::Bf16 => "mov.b16(h, 1.5bf16)",
        Ty::F16x2 => "mov.b32(h, (1.5f16, 2.0f16))",
        Ty::Bf16x2 => "mov.b32(h, (1.5bf16, 2.0bf16))",
        _ => "mov.b32(t, 1.5)",
    };
    format!("`{name}` takes a `{}` only from a variable: `{fill}` puts one in it", t.name())
}

/// The integer `x` as an operand that must be known now, of what `imm` lets it be.
fn known(imm: Imm, x: i128, name: &str) -> Result<String, String> {
    let fits = match imm {
        Imm::Bits(bits) => (0..1i128 << bits).contains(&x),
        Imm::Of(_) => imm.values().iter().any(|&v| i128::from(v) == x),
        Imm::Pow2(p) => x > 0 && x & (x - 1) == 0 && x <= 1 << p,
    };
    if fits {
        return Ok(x.to_string());
    }
    Err(match imm {
        Imm::Bits(bits) => {
            format!("`{name}` takes {bits} bits here, 0 to {}; {x} is not one of them", (1i64 << bits) - 1)
        }
        Imm::Of(_) => match imm.values().as_slice() {
            [n] => format!("`{name}` takes {n} here, the one value PTX has for it; not {x}"),
            [some @ .., last] => {
                let some: Vec<String> = some.iter().map(i64::to_string).collect();
                format!("`{name}` takes {} or {last} here; {x} is not one of them", some.join(", "))
            }
            [] => unreachable!("the table gives one value at least"),
        },
        Imm::Pow2(p) => format!("`{name}` takes a power of two here, 2^{p} at most; {x} is not one"),
    })
}

/// What `spec` of `b` wants, for an error.
fn describe(b: &Bound, spec: &Spec) -> String {
    let t = spec.ty.map(|t| b.ty(t).name()).unwrap_or_default();
    let one = match spec.role {
        Role::Addr(_) => return "an address, as in `[p + 4]`".into(),
        Role::Label => return "a label".into(),
        Role::Labels => return "labels taken together, as `(a, b)`, for an index to pick one".into(),
        Role::Func => return "a function of the module, whose address it takes".into(),
        Role::Handle => {
            let what = if b.form.op.starts_with("su") { "surface" } else { "texture" };
            let Some(at) = spec.inner.get(1) else { return format!("a {what}'s handle in brackets, `[t]`") };
            let names = if at.lead.is_some() { ["i", "x", "y", "z"] } else { ["x", "y", "z", "w"] };
            let at = match lanes(b, at) {
                1 => names[0].to_string(),
                n => format!("({})", names[..n].join(", ")),
            };
            return format!("a {what}'s handle and the coordinates in it, in brackets: `[t, {at}]`");
        }
        Role::Imm(_) => return "an integer known now".into(),
        Role::Dst if spec.sink => format!("a `{t}` variable to write, or `_`"),
        Role::Dst => format!("a `{t}` variable to write"),
        Role::Src if spec.memory => "memory the kernel declares, whose address it takes".into(),
        Role::Src if spec.reg_only => format!("a `{t}` variable"),
        Role::Src if spec.sreg => format!("a `{t}` variable, a value or a special register"),
        Role::Src => format!("a `{t}` variable or a value"),
    };
    match (lanes(b, spec), spec.lead) {
        (1, _) => one,
        (n, Some(lead)) => {
            let first =
                describe(b, &Spec { ty: Some(TyRef::Fixed(lead)), lead: None, count: Count::One, ..spec.clone() });
            match n - 1 {
                1 => format!("{n} operands taken together, {first} and then {one}"),
                rest => format!("{n} operands taken together, {first} and then {rest} each {one}"),
            }
        }
        (n, None) if spec.one_var => {
            format!("{n} operands taken together, each {one}, and values only beside a variable")
        }
        (n, None) => format!("{n} operands taken together, each {one}"),
    }
}

/// How many operands the forms take, `1 or 2 operands`, from each one's fewest and most.
fn counts(ranges: &[(usize, usize)]) -> String {
    let all: BTreeSet<usize> = ranges.iter().flat_map(|&(a, b)| a..=b).collect();
    let words: Vec<String> = all.iter().map(usize::to_string).collect();
    match words.as_slice() {
        [one] if one == "0" => "no operands".into(),
        [one] if one == "1" => "1 operand".into(),
        [one] => format!("{one} operands"),
        [first @ .., last] => format!("{} or {last} operands", first.join(", ")),
        [] => unreachable!("an instruction has forms"),
    }
}

/// Why a function takes no predicate.
pub(crate) const NO_PRED: &str =
    "PTX passes no `pred` to a function, nor back: `selp.u32(x, 1, 0, p)` makes it a `u32`";

/// The types a variable may be, for an error about one that is none; and those memory holds.
const TYPES: &str = "`pred`, `b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, `bf16`, `bf16x2`, \
                     `tf32`, `f32` and `f64`";
const HELD: &str = "`b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, `bf16`, `bf16x2`, `tf32`, `f32` and \
                    `f64`";

impl Target for Kernel {
    type Var = Var;

    fn param(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        let pointer = ty.starts_with("*const ") || ty.starts_with("*mut ");
        let t = match ty {
            _ if self.func.is_some() => {
                let t = crate::passable(ty)?;
                if t.needs() != Needs::default() {
                    self.require(t.needs(), &format!("{name}: {}", t.name()))?;
                }
                t
            }
            _ if pointer => Ty::U64,
            "u8" => Ty::U8,
            "i8" => Ty::S8,
            "u16" => Ty::U16,
            "i16" => Ty::S16,
            "u32" => Ty::U32,
            "i32" => Ty::S32,
            "u64" | "usize" => Ty::U64,
            "i64" | "isize" => Ty::S64,
            "f16" => Ty::F16,
            "bf16" => Ty::Bf16,
            "f32" => Ty::F32,
            "f64" => Ty::F64,
            _ => {
                return Err(format!(
                    "a GPU kernel takes pointers, integers and the floats `f16`, `bf16`, `f32` and `f64`, not `{ty}`"
                ));
            }
        };
        let reg = self.new_var(name, t);
        self.p.params.push((reg, PReg { class: reg.class, hw: 0 }));
        self.p.signature.params.push(reg.class);
        self.named.insert(format!("{}_param_{}", self.p.name, self.params.len()));
        self.params.push(Param { ty: t, reg: reg.id as usize, pointer });
        Ok(Var::Reg { reg, ty: t })
    }

    fn result(&mut self, ty: &str) -> Result<(), String> {
        if self.func.is_none() {
            return match ty {
                "()" => Ok(()),
                _ => Err(format!("a GPU kernel returns nothing, not `{ty}`: it writes what it works out to memory")),
            };
        }
        let returns = crate::returns(ty)?;
        match returns {
            Returns::Value(t) if t.needs() != Needs::default() => self.require(t.needs(), &format!("-> {ty}"))?,
            // As `ptxas` has `.noreturn`.
            Returns::Never => self.require(Needs { ptx: 64, sm: 30, ..Needs::default() }, "-> !")?,
            _ => {}
        }
        self.func = Some(returns);
        Ok(())
    }

    fn var(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        if let Some(args) = ty.strip_prefix("shared<") {
            return self.new_memory(name, ty, Space::Shared, args);
        }
        if let Some(args) = ty.strip_prefix("local<") {
            return self.new_memory(name, ty, Space::Local, args);
        }
        let Some(t) = Ty::parse(ty) else {
            return Err(format!("PTX variables are {TYPES}, not `{ty}`"));
        };
        if t.needs() != Needs::default() {
            self.require(t.needs(), &format!("let {name}: {}", t.name()))?;
        }
        Ok(Var::Reg { reg: self.new_var(name, t), ty: t })
    }

    fn view(&mut self, v: Var, view: &str) -> Result<Var, String> {
        let ty = match v {
            Var::Reg { ty, .. } => ty,
            // As a helper's parameter, it is the part still.
            Var::Part { ty, part, .. } => {
                return match Ty::parse(view) {
                    Some(t) if ty.fits(t) => Ok(v),
                    _ => Err(format!("a part of a `{}` variable, `.{part}`, is no `{view}`", ty.name())),
                };
            }
            Var::Mem(i) => {
                let (name, ty) = &self.declared[i];
                return match view == ty {
                    true => Ok(v),
                    false => Err(format!("`{name}` is memory, `{ty}`; it is no `{view}`")),
                };
            }
        };
        match Ty::parse(view) {
            Some(t) if ty.fits(t) => Ok(v),
            Some(t) => Err(format!("a `{}` variable is no `{}`", ty.name(), t.name())),
            None if view.starts_with("shared<") || view.starts_with("local<") => {
                Err(format!("a `{}` variable is no memory, `{view}`", ty.name()))
            }
            None => Err(format!("PTX variables are {TYPES}, not `{view}`")),
        }
    }

    /// A part of a 32-bit variable, `r.b0`, before the variable seen as a type: `r.b32` is
    /// its bytes 3 and 2, as a video instruction writes them.
    fn field(&mut self, v: Var, field: &str) -> Result<Var, String> {
        match (v, Part::parse(field)) {
            (Var::Reg { reg, ty }, Some(part)) if ty.bits() == 32 => Ok(Var::Part { reg, ty, part }),
            (Var::Reg { ty, .. }, Some(part)) if Ty::parse(field).is_none() => Err(format!(
                "a video instruction takes a part of a 32-bit variable, as `.{part}`; this is a `{}`",
                ty.name()
            )),
            (Var::Part { part, .. }, Some(_)) => Err(format!("this is a part of a variable already, `.{part}`")),
            _ => self.view(v, field),
        }
    }

    fn label(&mut self, label: u32) {
        for w in &mut self.partial {
            w.side = None;
        }
        self.p.body.push(Item::Label(Label(label)));
    }

    fn note(&mut self, text: &str) {
        self.p.body.push(Item::Note(text.to_string()));
    }

    fn inst(&mut self, name: &str, args: &[Given<Var>], guard: Option<Guard<Var>>) -> Result<(), Fault> {
        let guard = match guard.map(|g| (g.pred, g.negated)) {
            None => None,
            Some((Var::Reg { reg, ty: Ty::Pred }, negated)) => Some((reg, negated)),
            Some((Var::Reg { ty, .. }, _)) => {
                return Err(Fault::new(format!("an instruction is guarded by a `pred`; this is a `{}`", ty.name())));
            }
            Some((v, _)) => {
                let what = self.given(&Given::Var(v));
                return Err(Fault::new(format!("an instruction is guarded by a `pred`; this is {what}")));
            }
        };
        let op = name.split('.').next().unwrap_or_default();
        if op == "call" {
            return self.call(name, args, guard);
        }
        let bound = forms::find(name).map_err(Fault::new)?;
        match (op, self.func) {
            ("ret", Some(Returns::Value(t))) => return self.give_back(name, t, args, guard),
            ("ret", Some(Returns::Never)) => {
                let f = &self.name;
                return Err(Fault::new(format!(
                    "`{f}` never returns, `-> !`: it ends each way out in `exit()`, `trap()` or a call of a \
                     function that never returns"
                )));
            }
            _ => {}
        }
        let (mut ranges, mut best, mut found) = (vec![], None::<(usize, bool, Fault)>, None);
        for b in &bound {
            match self.operands(b, name, args) {
                Ok(w) => {
                    found = Some((b, w));
                    break;
                }
                Err(Miss::Count(least, most)) => ranges.push((least, most)),
                Err(Miss::At(at, shape, fault)) => {
                    if best.as_ref().is_none_or(|(a, s, _)| (at, shape) > (*a, *s)) {
                        best = Some((at, shape, fault));
                    }
                }
            }
        }
        let Some((b, w)) = found else {
            return Err(match best {
                Some((.., fault)) => fault,
                None => Fault::new(format!("`{name}` takes {}, not {}", counts(&ranges), args.len())),
            });
        };
        let mut what = name.to_string();
        for s in &w.named {
            what += &format!(" {s}");
        }
        self.require(b.needs.and(w.needs), &what).map_err(Fault::new)?;
        for f in &w.funcs {
            self.named_fn(f);
        }
        let label = args.iter().find_map(|a| match a {
            Given::Label(l) => Some(Label(*l)),
            _ => None,
        });
        let flow = match (b.form.op, label, guard) {
            ("bra", Some(l), None) => Flow::Jump(l),
            ("bra", Some(l), Some(_)) => Flow::Branch(l),
            ("ret" | "exit" | "trap", _, None) => Flow::Return,
            _ if !w.table.is_empty() => {
                Flow::Table { to: w.table.iter().map(|&l| Label(l)).collect(), or_next: guard.is_some() }
            }
            _ => Flow::Next,
        };
        let mut line = String::new();
        if !w.table.is_empty() {
            let labels: Vec<String> = w.table.iter().map(|l| format!("$L{l}")).collect();
            line += &format!("$T{}: .branchtargets {};\n\t", self.tables, labels.join(", "));
            self.tables += 1;
        }
        line += &self.guarded(guard);
        line += name;
        if !w.texts.is_empty() {
            line += " ";
            line += &w.texts.join(", ");
        }
        line += ";";
        self.push(line, w.args, guard, flow);
        Ok(())
    }

    fn end(&mut self) -> Result<(), String> {
        let open = match self.p.body.iter().rev().find(|item| !matches!(item, Item::Note(_))) {
            Some(Item::Inst(inst)) => {
                !matches!(inst.flow, Flow::Return | Flow::Jump(_) | Flow::Table { or_next: false, .. })
            }
            _ => true,
        };
        if open {
            let f = &self.name;
            match self.func {
                Some(Returns::Value(t)) => {
                    return Err(format!(
                        "control reaches the end of `{f}`, which returns a `{}`: `ret(v)` gives it back",
                        t.name()
                    ));
                }
                Some(Returns::Never) => {
                    return Err(format!(
                        "control reaches the end of `{f}`, which never returns, `-> !`: end each way out in \
                         `exit()`, `trap()` or a call of a function that never returns"
                    ));
                }
                _ => self.push("ret;".into(), vec![], None, Flow::Return),
            }
        }
        Ok(())
    }
}
