//! PTX for NVIDIA GPUs, as `kgen` writes it: the adapter between `kgen-lang` and the virtual
//! instruction set that the CUDA driver compiles for whichever GPU it finds.
//!
//! - [`lang::Kernel`] is PTX for `kgen-lang`: variables of PTX's register types, and
//!   instructions by their PTX names, `ld.global.f32(x, [p + 4])`, each checked against the
//!   forms of [`forms`], the instructions PTX has for the GPUs from Fermi, `sm_20`, to
//!   Ampere, `sm_86`; and `call`, against the [`Prototype`] of the function it calls.
//! - [`Entry`] is a kernel or a function written out: the program `kgen-compile` checks, and
//!   the `.entry` or the `.func` it prints as. [`module`] puts them in a module for the
//!   driver to load, of the PTX version and for the GPU their instructions need, with the
//!   memory of the module's its kernels name, [`Static`].
//!
//! PTX's registers have no number, and `ptxas` allocates them, so nothing here allocates: a
//! variable is a register named after it. Memory a kernel declares, [`Memory`], is named
//! after its variable too.

pub mod forms;
pub mod lang;

use kgen_core::{Item, Program};
use kgen_lang::{IntTy, Value};

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

    /// The type of Rust's the host reads one of as, for memory the host reads: `u32` for a
    /// `b32` or a `u32`, `i32` for an `s32`, and kgen-cuda's `f16` and `bf16`; `None` for one
    /// it has none of.
    pub fn host(self) -> Option<&'static str> {
        Some(match self {
            Ty::B8 | Ty::U8 => "u8",
            Ty::S8 => "i8",
            Ty::B16 | Ty::U16 => "u16",
            Ty::S16 => "i16",
            Ty::B32 | Ty::U32 => "u32",
            Ty::S32 => "i32",
            Ty::B64 | Ty::U64 => "u64",
            Ty::S64 => "i64",
            Ty::F16 => "f16",
            Ty::Bf16 => "bf16",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            _ => return None,
        })
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

/// Memory a kernel declares, bytes in shared or local memory, or the module's, a [`Static`],
/// in global or constant memory: at an address that is a multiple of `align`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Memory {
    /// Its name in the PTX: unique in the entry, and in the module if it has no size or is
    /// the module's.
    pub name: String,
    /// [`Space::Shared`] or [`Space::Local`], or for the module's, [`Space::Global`] or
    /// [`Space::Const`].
    pub space: Space,
    /// What its address is a multiple of, in bytes: a power of two.
    pub align: u64,
    /// How many bytes it is; `None` for the shared memory the launch says the size of.
    pub bytes: Option<u64>,
}

/// Memory of the module's, which a `static` of the source declares: bytes in global or
/// constant memory that each of its kernels names, declared once, at the module's scope, which
/// the host reads and writes by its symbol, and which keeps what is written from one launch
/// to the next. Constant memory the kernels only read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Static {
    /// What the source calls it, which a kernel names it by: `COUNTS`.
    pub name: String,
    /// Its type, as the source declares it: `global<[u32; 16]>`.
    pub ty: String,
    /// Its memory, named by its symbol, with its size.
    pub memory: Memory,
    /// What each of its elements is.
    pub elem: Ty,
    /// What it holds until something writes it, as bytes; zeros past their end.
    pub init: Vec<u8>,
}

/// The constant memory a module has: 64 KB.
const CONSTANT: u64 = 64 << 10;

impl Static {
    /// The static `name`, of the type `ty` as the source declares it, `global<[u32; 16]>`, or
    /// `constant<[f32; 4], 16>` at a multiple of 16, named `symbol` in PTX: holding zeros until
    /// [`init`](Static::init) says otherwise. Or why it is none.
    pub fn new(name: &str, symbol: &str, ty: &str) -> Result<Static, String> {
        let (space, word) = match ty.split_once('<').map(|(word, _)| word) {
            Some("global") => (Space::Global, "global"),
            Some("constant") => (Space::Const, "constant"),
            _ => {
                return Err(format!(
                    "a static of the module is memory: in global memory, `global<[u32; 16]>`, or in constant memory, \
                     which its kernels only read, `constant<[f32; 4]>`; not `{ty}`"
                ));
            }
        };
        let (elem, bytes, align) = lang::shape(ty, word)?;
        if elem.host().is_none() {
            return Err(format!(
                "a static holds numbers the host reads as Rust's: `b8` to `b64`, `u8` to `u64`, `s8` to `s64`, \
                 `f16`, `bf16`, `f32` and `f64`; not `{}`",
                elem.name()
            ));
        }
        let Some(bytes) = bytes else {
            return Err(format!(
                "`{ty}` has no size, which only shared memory may leave to the launch: `{word}<[{}; 64]>`",
                elem.name()
            ));
        };
        let memory = Memory { name: symbol.to_string(), space, align, bytes: Some(bytes) };
        Ok(Static { name: name.to_string(), ty: ty.to_string(), memory, elem, init: vec![] })
    }

    /// Makes it hold `value` until something writes it: an array of as many elements as it
    /// has, each of its element's type, `[1.5, 2.0]`; or one, for memory of one that is no
    /// array, `global<f32>`. Or why it cannot.
    pub fn init(&mut self, value: &Value) -> Result<(), String> {
        let (ty, elem) = (&self.ty, self.elem.name());
        let len = self.memory.bytes.unwrap_or(0) / (u64::from(self.elem.bits()) / 8);
        let array = ty.split_once('<').is_some_and(|(_, args)| args.starts_with('['));
        let items = match (value, array) {
            (Value::Array(xs), true) if xs.len() as u64 == len => xs.as_slice(),
            (Value::Array(xs), true) => return Err(format!("`{ty}` holds {len} elements; this has {}", xs.len())),
            (x, true) => return Err(format!("`{ty}` holds an array of `{elem}`; this is {}", x.kind())),
            (x, false) => std::slice::from_ref(x),
        };
        let mut init = vec![];
        for (i, x) in items.iter().enumerate() {
            init.extend(datum(self.elem, x, &if array { format!("element {i}") } else { "this".into() })?);
        }
        self.init = init;
        Ok(())
    }

    /// Its declaration, at the module's scope, `.global .align 4 .b8 COUNTS[64];`: with what it
    /// holds until something writes it, but the zeros that ends in, which PTX fills in.
    pub fn declaration(&self) -> String {
        let m = &self.memory;
        let mut out = format!(".{} .align {} .b8 {}[{}]", m.space.word(), m.align, m.name, m.bytes.unwrap_or(0));
        let end = self.init.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        if end > 0 {
            let bytes: Vec<String> = self.init[..end].iter().map(u8::to_string).collect();
            out += &format!(" = {{{}}}", bytes.join(", "));
        }
        out + ";\n"
    }
}

/// `x` as the bytes of a `t` in memory, the lowest first; or why it is none, of `x` as `what`
/// names it, `this` or `element 2`. An integer fits the bits of a `u` type as a number of no
/// sign, of an `s` type as a signed one, and of a `b` type as either; a float is one of its own
/// type, or bits of its size.
fn datum(t: Ty, x: &Value, what: &str) -> Result<Vec<u8>, String> {
    let (name, bits) = (t.name(), t.bits());
    // A type's name as it is read: an `s8`, an `f32`, an `i64`, a `u8`.
    let a = |name: &str| if name.starts_with(['f', 'i', 's']) { "an" } else { "a" };
    match *x {
        Value::Int(v, ty) if t.int() || t.kind() == Kind::Bit => {
            let fits = |ty: IntTy| match t.kind() {
                Kind::Unsigned => !ty.signed(),
                Kind::Signed => ty.signed(),
                _ => true,
            };
            if let Some(ty) = ty
                && (ty.bits() != bits || !fits(ty))
            {
                return Err(format!("{what} is {} `{}`, not {} `{name}`", a(ty.name()), ty.name(), a(name)));
            }
            let (lo, hi) = match t.kind() {
                Kind::Unsigned => (0, (1i128 << bits) - 1),
                Kind::Signed => (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1),
                _ => (-(1i128 << (bits - 1)), (1i128 << bits) - 1),
            };
            if !(lo..=hi).contains(&v) {
                return Err(format!("{what} is {v}, out of range for `{name}`"));
            }
            Ok(v.to_le_bytes()[..bits as usize / 8].to_vec())
        }
        Value::Int(v, _) if t.kind() == Kind::Float => {
            Err(format!("{what} is an integer, and {} `{name}` a float: `{v}.0`", a(name)))
        }
        Value::F32(v) if matches!(t, Ty::F32 | Ty::B32) => Ok(v.to_le_bytes().to_vec()),
        Value::F64(v) if matches!(t, Ty::F64 | Ty::B64) => Ok(v.to_le_bytes().to_vec()),
        Value::F16(v) if matches!(t, Ty::F16 | Ty::B16) => Ok(v.to_le_bytes().to_vec()),
        Value::Bf16(v) if matches!(t, Ty::Bf16 | Ty::B16) => Ok(v.to_le_bytes().to_vec()),
        ref x => Err(format!("{what} is {}, not {} `{name}`", x.kind(), a(name))),
    }
}

/// Why `s` does not fit beside the statics `before` it, if it does not: in the 64 KB of
/// constant memory a module has, which `ptxas` lays out in the order the module declares it,
/// each at the multiple its address is; or in the global memory `ptxas` lays out with it.
pub fn fits(before: &[Static], s: &Static) -> Result<(), String> {
    let space = s.memory.space;
    let mine = before.iter().chain([s]).filter(|b| b.memory.space == space).map(|b| &b.memory);
    if space == Space::Const {
        let all = mine.fold(0u64, |end, m| end.next_multiple_of(m.align).saturating_add(m.bytes.unwrap_or(0)));
        if all > CONSTANT {
            return Err(format!("this makes the module's constant memory {all} bytes, and a module has 64 KB"));
        }
        return Ok(());
    }
    // ptxas 13.4 writes the size of the section it lays a module's global memory out in,
    // `.nv.global`, or `.nv.global.init` for what holds more than zeros, in 32 bits, and cuts
    // one of 4 GB or more to them without a word: the memory past is no memory's, and the
    // driver's copies run past what it allocated. It cuts a symbol's own size to 61 bits, and
    // writes 0 for 2^63; only at 2^64 does it refuse. The bytes of both sections at once,
    // each memory's padding counted, are under 4 GB.
    let all = mine.fold(0u64, |sum, m| sum.saturating_add(m.bytes.unwrap_or(0)).saturating_add(m.align - 1));
    if all >= 1 << 32 {
        return Err(format!(
            "this makes the module's global memory as much as {all} bytes, each static aligned, and `ptxas` \
             writes its size in 32 bits: a module has less than 4 GB"
        ));
    }
    Ok(())
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

/// `name` in the characters PTX takes in a name: ASCII's letters, digits and `_` as they are;
/// one beyond ASCII as Rust's legacy mangling writes it, `$u3c3$` for `σ`, which is no other
/// name's, since no name of Rust's holds a `$`; and any other `_`, as Cargo writes a crate's
/// `-`. `ptxas` takes `$` anywhere in a name, and nothing beyond ASCII.
pub fn ident(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => out.push(c),
            _ if c.is_ascii() => out.push('_'),
            _ => out += &format!("$u{:x}$", u32::from(c)),
        }
    }
    out
}

/// `path`, namespaces each in the one before, as C++ mangles their names for [`symbol`]:
/// `<length><name>` each, `["kgen", "my_app"]` as `4kgen6my_app`. Each name is as [`ident`]
/// writes it, and one that starts with a digit has `_` before it.
pub fn scope(path: &[&str]) -> String {
    let mut out = String::new();
    for name in path {
        let mut name = ident(name);
        if name.starts_with(|c: char| c.is_ascii_digit()) {
            name.insert(0, '_');
        }
        out += &format!("{}{name}", name.len());
    }
    out
}

/// The symbol of `name`, a kernel or a function, in `scope`, which [`scope`] writes, for the
/// compile-time arguments `args`: as C++ mangles the name of a template's instance there,
/// under Itanium's ABI, which the tools that read PTX demangle. `cu++filt`, as Nsight and
/// `cuda-gdb` do, shows `_ZN4kgen6my_app3gpu5saxpyE` as `kgen::my_app::gpu::saxpy`, and
/// `_ZN4kgen3gpu4tileILm4EEE` as `kgen::gpu::tile<(unsigned long)4>`. Without a scope and
/// without arguments, the symbol is `name`, as [`ident`] writes it.
///
/// `args` are integers and `bool`s, as a kernel's compile-time arguments are: an integer of
/// its type, C++'s of its size, and one of no type an `i32`, as Rust has it.
pub fn symbol(scope: &str, name: &str, args: &[Value]) -> String {
    if scope.is_empty() && args.is_empty() {
        return ident(name);
    }
    let mut out = match scope {
        "" => "_Z".to_string(),
        _ => format!("_ZN{scope}"),
    };
    out += &self::scope(&[name]);
    if !args.is_empty() {
        out += "I";
        for a in args {
            out += &match a {
                Value::Bool(b) => format!("Lb{}E", u8::from(*b)),
                Value::Int(x, ty) => {
                    let ty = match ty.unwrap_or(IntTy::I32) {
                        IntTy::I8 => 'a',
                        IntTy::I16 => 's',
                        IntTy::I32 => 'i',
                        IntTy::I64 | IntTy::Isize => 'l',
                        IntTy::U8 => 'h',
                        IntTy::U16 => 't',
                        IntTy::U32 => 'j',
                        IntTy::U64 | IntTy::Usize => 'm',
                    };
                    let sign = if *x < 0 { "n" } else { "" };
                    format!("L{ty}{sign}{}E", x.unsigned_abs())
                }
                other => unreachable!("a kernel's compile-time argument is an integer or a bool, not {}", other.kind()),
            };
        }
        out += "E";
    }
    if !scope.is_empty() {
        out += "E";
    }
    out
}

/// The symbol of `name`, memory of the function whose symbol is `function` that is the
/// module's rather than the function's, as C++ names a function's static and `nvcc` a
/// kernel's `__shared__` memory: `_ZZ`, the function, `E`, the name. `cu++filt` shows
/// `_ZZN4kgen3gpu7reverseEE3buf` as `kgen::gpu::reverse::buf`, and `_ZZ7reverseE3buf` as
/// `reverse::buf`.
fn local(function: &str, name: &str) -> String {
    // A symbol that is not mangled is a name as `ident` writes it already.
    let function = match function.strip_prefix("_Z") {
        Some(mangled) => mangled.to_string(),
        None => format!("{}{function}", function.len()),
    };
    format!("_ZZ{function}E{}", scope(&[name]))
}
