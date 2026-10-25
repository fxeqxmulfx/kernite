//! PTX for NVIDIA GPUs, as `kgen` writes it: the adapter between `kgen-lang` and the virtual
//! instruction set that the CUDA driver compiles for whichever GPU it finds.
//!
//! - [`lang::Kernel`] is PTX for `kgen-lang`: variables of PTX's register types, and
//!   instructions by their PTX names, `ld.global.f32(x, [p + 4])`, each checked against the
//!   forms of [`forms`], the instructions PTX has for the GPUs from Fermi, `sm_20`, to
//!   Ampere, `sm_86`; and `call`, against the [`Prototype`] of the function it calls.
//! - [`Entry`] is a kernel or a function written out: the program `kgen-compile` checks, and
//!   the `.entry` or the `.func` it prints as. [`module`] puts them in a module for the
//!   driver to load, of the PTX version and for the GPU their instructions need.
//!
//! PTX's registers have no number, and `ptxas` allocates them, so nothing here allocates: a
//! variable is a register named after it. Memory a kernel declares, [`Memory`], is named
//! after its variable too.

pub mod forms;
pub mod lang;

use kgen_core::{Item, Program};

use crate::forms::Space;

/// A register type of PTX, which is what a variable is declared as, and the types an
/// instruction's name ends in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ty {
    /// One bit, set by `setp`, guarding what `if p { .. }` holds.
    Pred,
    /// 8 bits of no type in particular, for `ld`, `st`, `cvt` and `mov`'s parts.
    B8,
    /// 16 bits of no type in particular.
    B16,
    /// 32 bits of no type in particular: they fit any instruction of 32 bits.
    B32,
    /// 64 bits of no type in particular.
    B64,
    /// 128 bits, for `ld`, `st` and `mov`.
    B128,
    /// An unsigned integer of 8 bits.
    U8,
    /// An unsigned integer of 16 bits.
    U16,
    /// An unsigned integer of 32 bits.
    U32,
    /// An unsigned integer of 64 bits, an address among them.
    U64,
    /// A signed integer of 8 bits.
    S8,
    /// A signed integer of 16 bits.
    S16,
    /// A signed integer of 32 bits.
    S32,
    /// A signed integer of 64 bits.
    S64,
    /// A float of 16 bits, IEEE's half.
    F16,
    /// Two `f16` in 32 bits, the first in the low half.
    F16x2,
    /// A float of 32 bits.
    F32,
    /// A float of 64 bits.
    F64,
    /// A float of 16 bits with `f32`'s exponent; PTX keeps it in a `b16` register.
    Bf16,
    /// Two `bf16` in 32 bits, in a `b32` register.
    Bf16x2,
    /// `f32` with the precision of `f16`, as the tensor cores read it, in a `b32` register.
    Tf32,
}

/// What a type is, for which types it fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A predicate.
    Pred,
    /// Bits of no type in particular.
    Bit,
    /// An unsigned integer.
    Unsigned,
    /// A signed integer.
    Signed,
    /// A float, or floats packed together.
    Float,
}

impl Ty {
    /// Every type, in the order an entry declares its registers.
    pub const ALL: [Ty; 21] = [
        Ty::Pred,
        Ty::B8,
        Ty::U8,
        Ty::S8,
        Ty::B16,
        Ty::U16,
        Ty::S16,
        Ty::F16,
        Ty::Bf16,
        Ty::B32,
        Ty::U32,
        Ty::S32,
        Ty::F32,
        Ty::F16x2,
        Ty::Bf16x2,
        Ty::Tf32,
        Ty::B64,
        Ty::U64,
        Ty::S64,
        Ty::F64,
        Ty::B128,
    ];

    /// The type PTX writes `name`, if it is one.
    pub fn parse(name: &str) -> Option<Ty> {
        Ty::ALL.into_iter().find(|t| t.name() == name)
    }

    /// Its name, as PTX writes it after a dot.
    pub fn name(self) -> &'static str {
        match self {
            Ty::Pred => "pred",
            Ty::B8 => "b8",
            Ty::B16 => "b16",
            Ty::B32 => "b32",
            Ty::B64 => "b64",
            Ty::B128 => "b128",
            Ty::U8 => "u8",
            Ty::U16 => "u16",
            Ty::U32 => "u32",
            Ty::U64 => "u64",
            Ty::S8 => "s8",
            Ty::S16 => "s16",
            Ty::S32 => "s32",
            Ty::S64 => "s64",
            Ty::F16 => "f16",
            Ty::F16x2 => "f16x2",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            Ty::Bf16 => "bf16",
            Ty::Bf16x2 => "bf16x2",
            Ty::Tf32 => "tf32",
        }
    }

    /// The type its register is declared as: PTX has no `bf16`, `bf16x2` or `tf32`
    /// registers, and keeps them in bits of their size.
    pub fn decl(self) -> Ty {
        match self {
            Ty::Bf16 => Ty::B16,
            Ty::Bf16x2 | Ty::Tf32 => Ty::B32,
            t => t,
        }
    }

    /// What declaring a register of it needs: PTX has `b128` registers from 8.3 on, for Volta,
    /// and `f16x2` ones from 4.2, for `sm_53`.
    pub fn needs(self) -> Needs {
        match self {
            Ty::B128 => Needs { ptx: 83, sm: 70, ..Needs::default() },
            Ty::F16x2 => Needs { ptx: 42, sm: 53, ..Needs::default() },
            _ => Needs::default(),
        }
    }

    /// Bits: 1 for a predicate.
    pub fn bits(self) -> u32 {
        match self {
            Ty::Pred => 1,
            Ty::B8 | Ty::U8 | Ty::S8 => 8,
            Ty::B16 | Ty::U16 | Ty::S16 | Ty::F16 | Ty::Bf16 => 16,
            Ty::B32 | Ty::U32 | Ty::S32 | Ty::F32 | Ty::F16x2 | Ty::Bf16x2 | Ty::Tf32 => 32,
            Ty::B64 | Ty::U64 | Ty::S64 | Ty::F64 => 64,
            Ty::B128 => 128,
        }
    }

    /// What it is.
    pub fn kind(self) -> Kind {
        match self {
            Ty::Pred => Kind::Pred,
            Ty::B8 | Ty::B16 | Ty::B32 | Ty::B64 | Ty::B128 => Kind::Bit,
            Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64 => Kind::Unsigned,
            Ty::S8 | Ty::S16 | Ty::S32 | Ty::S64 => Kind::Signed,
            Ty::F16 | Ty::F16x2 | Ty::F32 | Ty::F64 | Ty::Bf16 | Ty::Bf16x2 | Ty::Tf32 => Kind::Float,
        }
    }

    /// Whether it is an integer, signed or not.
    pub fn int(self) -> bool {
        matches!(self.kind(), Kind::Unsigned | Kind::Signed)
    }

    /// Whether a register of this type is an operand an instruction of type `want` takes, as
    /// PTX checks most instructions: one of the same size that is the same type, or untyped
    /// bits, or an integer where an integer is wanted. ptxas 13.4 takes an `f16x2` for a
    /// 32-bit integer too, and for a predicate, which PTX's ISA has not.
    pub fn fits(self, want: Ty) -> bool {
        if self == want {
            return true;
        }
        if self.bits() != want.bits() || self == Ty::Pred || want == Ty::Pred {
            return false;
        }
        match (self.kind(), want.kind()) {
            (Kind::Bit, _) | (_, Kind::Bit) => true,
            (a, b) => a != Kind::Float && b != Kind::Float,
        }
    }

    /// Whether it fits as PTX checks `ld`, `st` and `cvt`: a register may be wider than the
    /// instruction's type, and bits fit any type of no more bits, an integer bits and
    /// integers, a float bits and itself. Not for `bf16`, `bf16x2` and `tf32`, which take
    /// [`Ty::fits`].
    pub fn fits_wide(self, want: Ty) -> bool {
        if matches!(want, Ty::Bf16 | Ty::Bf16x2 | Ty::Tf32) || matches!(self, Ty::Bf16 | Ty::Bf16x2 | Ty::Tf32) {
            return self.fits(want);
        }
        if self == Ty::Pred || want == Ty::Pred {
            return self == want;
        }
        if self.bits() < want.bits() {
            return false;
        }
        match (self.kind(), want.kind()) {
            (Kind::Bit, _) => true,
            (Kind::Unsigned | Kind::Signed, k) => k != Kind::Float,
            (Kind::Float, Kind::Bit) => true,
            (Kind::Float, Kind::Float) => self == want,
            _ => false,
        }
    }
}

/// What a kernel needs of the PTX it is written in and of the GPU it runs on, as `ptxas` has
/// it; a module is of [`Needs::FLOOR`] at the least.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Needs {
    /// The version of PTX, as ten times its major and its minor: 78 for 7.8. 0 for any.
    pub ptx: u32,
    /// The GPU, by its `sm_` number: 80 for Ampere's first. 0 for any.
    pub sm: u32,
    /// A GPU older than this, by its `sm_` number, in a module of PTX [`Needs::since`] or
    /// newer: `shfl` without `.sync` is for none from Volta's on, `sm_70`, from PTX 6.4. 0 for
    /// any.
    pub below: u32,
    /// The version of PTX from which [`Needs::below`] holds, as [`Needs::ptx`] writes it.
    pub since: u32,
}

/// The GPUs kgen writes for, by their `sm_` numbers, and the version of PTX that came with
/// each, which a module for it is of at least.
pub const GPUS: [(u32, u32); 17] = [
    (20, 20),
    (21, 20),
    (30, 30),
    (32, 40),
    (35, 31),
    (37, 41),
    (50, 40),
    (52, 41),
    (53, 42),
    (60, 50),
    (61, 50),
    (62, 50),
    (70, 60),
    (72, 61),
    (75, 63),
    (80, 70),
    (86, 71),
];

impl Needs {
    /// What every module is written for: PTX 2.3, the first with `.address_size 64`, which
    /// every module has, and Fermi, `sm_20`, the oldest GPU the driver compiles PTX for.
    pub const FLOOR: Needs = Needs { ptx: 23, sm: 20, below: 0, since: 0 };

    /// What both need, and the narrower of the GPUs they are for.
    pub fn and(self, other: Needs) -> Needs {
        let least = |a: u32, b: u32| match (a, b) {
            (0, x) | (x, 0) => x,
            (a, b) => a.min(b),
        };
        Needs {
            ptx: self.ptx.max(other.ptx),
            sm: self.sm.max(other.sm),
            below: least(self.below, other.below),
            since: least(self.since, other.since),
        }
    }

    /// Whether some module has it all: one for a GPU older than [`Needs::below`], or of a
    /// version older than [`Needs::since`].
    pub fn possible(self) -> bool {
        self.below == 0 || self.sm.max(Needs::FLOOR.sm) < self.below || self.written() < self.since
    }

    /// The version of PTX a module of it is of, as [`Needs::ptx`] writes it: what it needs,
    /// and what its GPU does.
    pub fn written(self) -> u32 {
        let sm = self.sm.max(Needs::FLOOR.sm);
        let came = GPUS.iter().find(|&&(g, _)| g == sm).map_or(0, |&(_, v)| v);
        self.ptx.max(came).max(Needs::FLOOR.ptx)
    }

    /// The version of PTX as a module's `.version` writes it: `7.8`.
    pub fn version(self) -> String {
        dotted(self.written())
    }

    /// The GPU as a module's `.target` writes it: `sm_80`.
    pub fn target(self) -> String {
        format!("sm_{}", self.sm.max(Needs::FLOOR.sm))
    }
}

/// A version of PTX as [`Needs::ptx`] writes it, as PTX does: `7.8`.
fn dotted(v: u32) -> String {
    format!("{}.{}", v / 10, v % 10)
}

/// Why no module has all of `needs`, each what an instruction needs and that instruction as
/// it is to be named: one is for GPUs older than another needs, in the version of PTX a
/// module for that GPU, or for a third, is of.
pub fn clash(needs: &[(Needs, String)]) -> String {
    let all = needs.iter().fold(Needs::default(), |n, (m, _)| n.and(*m));
    let (old, o) = needs.iter().find(|(n, _)| n.below == all.below).expect("one is for older GPUs");
    let first = format!("{o} is for GPUs older than sm_{} from PTX {} on", old.below, dotted(old.since));
    let gpu = needs.iter().max_by_key(|(n, _)| n.sm).expect("one needs a newer GPU");
    let ptx = needs.iter().max_by_key(|(n, _)| n.ptx).expect("one needs the version");
    let ((new, g), (v, w)) = (gpu, ptx);
    let came = Needs { sm: new.sm, ..Needs::default() }.written();
    if came >= all.since {
        format!("{first}, and {g} needs sm_{}, which PTX has from {} on: no module has both", new.sm, dotted(came))
    } else if std::ptr::eq(gpu, ptx) {
        format!("{first}, and {g} needs sm_{} and PTX {}: no module has both", new.sm, dotted(v.ptx))
    } else {
        format!("{first}; {g} needs sm_{}, and {w} PTX {}: no module has them all", new.sm, dotted(v.ptx))
    }
}

/// What a function takes and returns, as the kernels that call it see it: PTX's
/// prototype of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prototype {
    /// The type of each parameter, in order.
    pub params: Vec<Ty>,
    /// What it returns.
    pub returns: Returns,
}

/// What a function returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Returns {
    /// Nothing, `()`.
    Nothing,
    /// A value of this type, which `ret(v)` gives back.
    Value(Ty),
    /// It never returns, `-> !`: PTX's `.noreturn`.
    Never,
}

impl Prototype {
    /// The prototype of a function whose parameters are of the types `params` and which
    /// returns `result`, as Rust writes them: PTX's types, and `!` for one that never
    /// returns; or why it is none PTX has.
    pub fn of(params: &[&str], result: Option<&str>) -> Result<Prototype, String> {
        let params = params.iter().map(|t| passable(t)).collect::<Result<_, _>>()?;
        let returns = result.map_or(Ok(Returns::Nothing), returns)?;
        Ok(Prototype { params, returns })
    }
}

/// The type of a function's parameter written `ty`, if it is one: one of PTX's, but a
/// predicate, which PTX passes to no function.
pub fn passable(ty: &str) -> Result<Ty, String> {
    match Ty::parse(ty) {
        Some(Ty::Pred) => Err(lang::NO_PRED.into()),
        Some(t) => Ok(t),
        None => {
            let ptx = match ty {
                _ if ty.starts_with("*const ") || ty.starts_with("*mut ") => "an address is a `u64`",
                "i8" => "PTX writes it `s8`",
                "i16" => "PTX writes it `s16`",
                "i32" => "PTX writes it `s32`",
                "i64" | "isize" => "PTX writes it `s64`",
                "usize" => "PTX writes it `u64`",
                _ => "as its variables are: `u32`, `f32`, `b64`",
            };
            Err(format!("a function takes and returns PTX's types, {ptx}; not `{ty}`"))
        }
    }
}

/// What a function returns, as Rust writes it: `()`, one of PTX's types, or `!` for never.
fn returns(ty: &str) -> Result<Returns, String> {
    match ty {
        "()" => Ok(Returns::Nothing),
        "!" => Ok(Returns::Never),
        _ if ty.starts_with('(') => {
            Err("a function returns one value at most, as PTX's do from Fermi on, not a tuple".into())
        }
        _ => passable(ty).map(Returns::Value),
    }
}

/// The type a value of type `t` is passed as, and loaded and stored by `ld.param` and
/// `st.param`: its register's, but bits for a float of 16 bits or two, which `ld` and `st`
/// have no type for.
pub fn passed(t: Ty) -> Ty {
    match t {
        Ty::F16 => Ty::B16,
        Ty::F16x2 => Ty::B32,
        t => t.decl(),
    }
}

/// The type of a function's `.param` for a value of type `t`: bits for one of 8 or 16 bits,
/// which `ptxas` passes to a function only as bits, and otherwise as [`passed`] has it.
pub fn formal(t: Ty) -> Ty {
    match t.bits() {
        8 => Ty::B8,
        16 => Ty::B16,
        _ => passed(t),
    }
}

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

/// Memory a kernel declares: bytes in shared or local memory, at an address that is a
/// multiple of `align`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Memory {
    /// Its name in the PTX: unique in the entry, and in the module if it has no size.
    pub name: String,
    /// [`Space::Shared`] or [`Space::Local`].
    pub space: Space,
    /// What its address is a multiple of, in bytes: a power of two.
    pub align: u64,
    /// How many bytes it is; `None` for the shared memory the launch says the size of.
    pub bytes: Option<u64>,
}

/// A kernel written in PTX, an `.entry` the host launches; or a function kernels call, a
/// `.func`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The kernel as a program, for the checks: each instruction's opcode is its PTX.
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
    /// The functions it names, each once: those it calls, and those whose addresses it takes,
    /// which a module declares before it if they come after.
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
/// `entries`, after a declaration of each it names that comes later.
///
/// It fails if one kernel is for GPUs older than another needs, in the version of PTX the
/// module would be of: PTX has no module for both.
pub fn module(entries: &[Entry]) -> Result<String, String> {
    let needs = entries.iter().fold(Needs::default(), |n, e| n.and(e.needs()));
    if !needs.possible() {
        let named: Vec<(Needs, String)> = entries
            .iter()
            .flat_map(|e| e.needs.iter().map(move |(n, why)| (*n, format!("`{why}` in `{}`", e.program.name))))
            .collect();
        return Err(clash(&named));
    }
    let mut out =
        format!("// Written by kgen.\n\n.version {}\n.target {}\n.address_size 64\n", needs.version(), needs.target());
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
