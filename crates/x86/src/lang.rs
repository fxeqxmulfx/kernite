//! x86-64 as the language writes it: [`Kernel`] is the [`Target`] that checks each call
//! against the forms x86 has for its instruction, and builds the program the builder would.
//!
//! The types are `gpr`, `ymm` and `xmm`, and the views `v.ymm`, `v.xmm`, `g.r32`, whose
//! writes clear the rest of the register as the machine's do. A kernel's parameters arrive
//! as SysV passes them: pointers and 64-bit integers in general registers, `f32` and `f64`
//! in vector ones, counted apart. It hands back a float in `xmm0` or an integer in `rax` with
//! `ret(v)`, or nothing with `ret()`, which the end of the body adds if control reaches it.
//! Besides the instructions, `zero(v)` and `ones(v)` set a variable without reading it.
//!
//! Stack memory is declared as PTX's local memory is: `let buf: local<[f32; 64], 32>;` is 256
//! bytes of the kernel's frame at a multiple of 32, and `local<u64>` one `u64`, at a multiple of
//! its 8 bytes. Brackets name what it holds, `vmovaps(v, [buf + 32])`, `vmovss(x, [buf + i * 4])`,
//! and `lea(p, [buf])` takes its address. Until written, it holds whatever was there before.
//! kgen checks an address in it where the index is 0: what it reaches lies inside, the address
//! reaching as far as one past the end for `lea`; and for `vmovaps` and `vmovdqa`, which fault
//! elsewhere, it is a multiple of the bytes they move. The index is the kernel's to keep inside,
//! and at such a multiple.
//!
//! Memory of the module's, a [`Static`] the kernel is let name by [`Kernel::module_memory`], is
//! declared as PTX's is, beside the kernels, of Rust's numbers: `static COUNTS: global<[u32; 16]>;`
//! in the process's memory, which keeps what the kernels write from call to call, and
//! `static WEIGHTS: constant<[f32; 4]> = [..];`, which they only read. A kernel reaches it at a
//! displacement from `rip`, with no register beside it, `mov(t.r32, [COUNTS + 8])`; for an
//! index, `lea(p, [COUNTS])` puts its address in `p`, and `[p + i * 4]` is what it holds there.
//! kgen checks what an address reaches as it does stack memory's, and refuses a write to
//! constant memory: a write through an address `lea` took is the kernel's to keep from.
//!
//! A `#[cooperative]` kernel runs on the threads of a grid, each the same code, as a GPU kernel
//! runs on blocks. A thread reads where it is as memory, 8 bytes each: `grid.thread`, its index
//! from 0, and `grid.threads`, how many there are, as in `mov(t, grid.thread)` or
//! `cmp(i, grid.threads)`. `grid.sync()` waits until every thread has come to it, so that what
//! each wrote before, every one reads after. It is code of the kernel's own, which calls
//! nothing: a thread counts itself in with `lock xadd`, and the last to come lets the others
//! go, who spin with `pause` until it does, so each thread needs a core. Every thread comes to
//! each `grid.sync()`, or the others wait for ever: kgen refuses one under an `if`, and a jump
//! around one is the kernel's to keep from.
//!
//! The launcher gives each thread the address of its place in the grid, in the register after
//! the kernel's parameters, as SysV passes the next integer, so a cooperative kernel takes five
//! integers at most; and its threads return nothing, but write what they compute to memory.
//! The place is three words: the thread's index, how many threads there are, and the address
//! of the grid's barrier, which is two: how many threads have come to it, and how many times
//! all have.
//!
//! A kernel calls the functions of its module `extern "sysv64"`, itself among them, but those
//! `#[cooperative]`, which run on a grid; [`Kernel::callee`] gives each, or each instance,
//! `tile::<4>`. `call(r, f, (a, b))` calls `f` with `a` and `b` and writes what it returns to
//! `r`, or to nothing, `_`; `call(f, a)` passes one, and `call(f)` none. An argument is a
//! variable of the parameter's file, a gpr's `r32` passing its 32 bits with the rest clear, or a
//! value known now; the result is written to a variable of its file. kgen moves each to the
//! register SysV passes it in, and the result from the one it comes back in; a value live
//! across the call stays in a register the callee keeps, `rbx` or `r12` to `r15`, or else in
//! the stack frame, as every vector does. `lea(fp, [f])` puts the address of `f` in `fp`, which
//! `call(r, fp, (a, b))` calls, passing each in the file its variable is of, and
//! `call(r, fp, (a, b), (f, g))` as one of the functions taken together last, checked against
//! each. A function that never returns, `-> !`, ends each way out in `ud2()`, which faults, or
//! in a call of one that never returns.

use kgen_core::Access::{self, Ignored, Read, ReadWrite, Write};
use kgen_core::Width::{self, W8, W16, W32, W64, W128, W256};
use kgen_core::{
    Arg, Base, Class, Const, ConstId, Flow, FuncId, Imm, Inst, Item, Label, Mem, Operand, Program, RegInfo, Signature,
    Slot, SlotId, Spill, StaticId, VReg,
};
use kgen_lang::{Fault, Guard, Operand as Given, Target, Value};

use crate::bytes;
use crate::op::Op;
use crate::sysv::{CALL_CLOBBERS, FLOAT_PARAMS, FLOAT_RESULT, INT_PARAMS, INT_RESULT, PAGE, SysV};

/// The most bytes of stack memory a kernel has: as far past the stack pointer as an address
/// reaches, whose displacement is 32 bits, signed.
const FRAME: u64 = i32::MAX as u64;

/// The most bytes of a static: an address reaches one at a displacement from `rip`, of 32 bits,
/// signed.
const FAR: u64 = i32::MAX as u64;

/// Where a thread's place in the grid has each of its words: its index, how many threads
/// there are, and the barrier's address. kgen-cpu lays it out so.
const THREAD: i32 = 0;
const THREADS: i32 = 8;
const BARRIER: i32 = 16;

/// Where the barrier has each of its words: how many threads have come to it, and how many
/// times all have, its generation.
const ARRIVED: i32 = 0;
const GENERATION: i32 = 8;

/// Rust's numbers, which stack memory holds, and the bytes of each.
const HELD: &[(&str, u64)] = &[
    ("u8", 1),
    ("i8", 1),
    ("u16", 2),
    ("i16", 2),
    ("u32", 4),
    ("i32", 4),
    ("f32", 4),
    ("u64", 8),
    ("i64", 8),
    ("usize", 8),
    ("isize", 8),
    ("f64", 8),
];

/// A variable and the view the language has of it, or memory, of the kernel's stack or of the
/// module's, which stands for its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(Place);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// A register, seen as the view says.
    Reg(VReg, View),
    /// Memory of the stack frame, by its slot.
    Stack(SlotId),
    /// Memory of the module's, a static, by its index among those the kernel names.
    Static(StaticId),
}

impl Var {
    /// Its register and the view of it, unless it is memory.
    fn reg(self) -> Option<(VReg, View)> {
        match self.0 {
            Place::Reg(reg, view) => Some((reg, view)),
            Place::Stack(_) | Place::Static(_) => None,
        }
    }
}

/// The register and the view of an operand that is a variable, unless it is memory.
fn reg_of(a: &Given<Var>) -> Option<(VReg, View)> {
    match a {
        Given::Var(v) => v.reg(),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    /// A general register, all 64 bits.
    R64,
    /// Its low 32 bits.
    R32,
    /// A vector register, whole.
    Ymm,
    /// Its low 128 bits.
    Xmm,
}

impl View {
    fn width(self) -> Width {
        match self {
            View::R64 => W64,
            View::R32 => W32,
            View::Ymm => W256,
            View::Xmm => W128,
        }
    }

    fn name(self) -> &'static str {
        match self {
            View::R64 => "a gpr",
            View::R32 => "a gpr's r32",
            View::Ymm => "a ymm",
            View::Xmm => "an xmm",
        }
    }
}

/// What an operand of a form may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A vector as wide as the form's: ymm or xmm, as the first of them says.
    V,
    /// That, or memory as wide.
    Vm,
    /// Memory as wide as the form's vectors.
    Mv,
    /// A vector, whole.
    Y,
    /// A vector's low 128 bits.
    X,
    /// That, or memory of this width.
    Xm(Width),
    /// Memory of this width.
    M(Width),
    /// A general register at this width.
    G(Width),
    /// An integer of 32 signed bits.
    I,
    /// Any integer of 64 bits.
    Q,
    /// An integer of so many bits, unsigned, which are all x86 reads of it: a gpr is shifted by
    /// the low 6 bits of a count, a vector's lanes by a byte, which empties them past their
    /// bits, and the half of a ymm is bit 0.
    N(u32),
    /// A control byte, of so many bits as x86 reads, which a listing shows in hex: `vcmpps`
    /// reads its predicate in the low 5, and `vroundps` and `vroundss` the rounding in the low
    /// 4.
    C(u32),
    /// A control byte of a bit for each 32-bit lane of the form's vectors, which a listing shows
    /// in hex: 8 of a ymm, and 4 of an xmm, as `vpblendd` reads no more.
    Lanes,
}

use Kind::{C, G, I, Lanes, M, Mv, N, Q, V, Vm, X, Xm, Y};

type Form = &'static [(Kind, Access)];

/// Instructions that take the same forms.
struct Group {
    ops: &'static [(&'static str, Op)],
    forms: &'static [Form],
}

/// Every instruction the language calls by name, but the jumps and `ret`.
const GROUPS: &[Group] = &[
    Group {
        ops: &[
            ("vaddps", Op::Vaddps),
            ("vsubps", Op::Vsubps),
            ("vmulps", Op::Vmulps),
            ("vdivps", Op::Vdivps),
            ("vmaxps", Op::Vmaxps),
            ("vminps", Op::Vminps),
            ("vandps", Op::Vandps),
            ("vandnps", Op::Vandnps),
            ("vorps", Op::Vorps),
            ("vxorps", Op::Vxorps),
            ("vpaddb", Op::Vpaddb),
            ("vpaddw", Op::Vpaddw),
            ("vpaddd", Op::Vpaddd),
            ("vpsubd", Op::Vpsubd),
            ("vpmulld", Op::Vpmulld),
            ("vpand", Op::Vpand),
            ("vpor", Op::Vpor),
            ("vpxor", Op::Vpxor),
            ("vpcmpeqb", Op::Vpcmpeqb),
            ("vpcmpeqd", Op::Vpcmpeqd),
            ("vpcmpgtd", Op::Vpcmpgtd),
            ("vpshufb", Op::Vpshufb),
            ("vpsignb", Op::Vpsignb),
            ("vpmaddubsw", Op::Vpmaddubsw),
            ("vpmaddwd", Op::Vpmaddwd),
            ("vpunpckldq", Op::Vpunpckldq),
            ("vpsllvd", Op::Vpsllvd),
            ("vpsrlvd", Op::Vpsrlvd),
        ],
        forms: &[&[(V, Write), (V, Read), (Vm, Read)]],
    },
    Group {
        ops: &[("vfmadd231ps", Op::Vfmadd231ps), ("vfnmadd231ps", Op::Vfnmadd231ps), ("vfmadd213ps", Op::Vfmadd213ps)],
        forms: &[&[(V, ReadWrite), (V, Read), (Vm, Read)]],
    },
    Group {
        ops: &[
            ("vsqrtps", Op::Vsqrtps),
            ("vcvtdq2ps", Op::Vcvtdq2ps),
            ("vcvtps2dq", Op::Vcvtps2dq),
            ("vcvttps2dq", Op::Vcvttps2dq),
            ("vmovshdup", Op::Vmovshdup),
            ("vmovsldup", Op::Vmovsldup),
            ("vpabsb", Op::Vpabsb),
        ],
        forms: &[&[(V, Write), (Vm, Read)]],
    },
    Group { ops: &[("vcmpps", Op::Vcmpps)], forms: &[&[(V, Write), (V, Read), (Vm, Read), (C(5), Read)]] },
    Group { ops: &[("vshufps", Op::Vshufps)], forms: &[&[(V, Write), (V, Read), (Vm, Read), (C(8), Read)]] },
    Group { ops: &[("vpblendd", Op::Vpblendd)], forms: &[&[(V, Write), (V, Read), (Vm, Read), (Lanes, Read)]] },
    Group { ops: &[("vroundps", Op::Vroundps)], forms: &[&[(V, Write), (Vm, Read), (C(4), Read)]] },
    Group { ops: &[("vblendvps", Op::Vblendvps)], forms: &[&[(V, Write), (V, Read), (Vm, Read), (V, Read)]] },
    Group {
        ops: &[
            ("vpslld", Op::Vpslld),
            ("vpsrld", Op::Vpsrld),
            ("vpsrad", Op::Vpsrad),
            ("vpsllw", Op::Vpsllw),
            ("vpsrlw", Op::Vpsrlw),
        ],
        forms: &[&[(V, Write), (V, Read), (N(8), Read)]],
    },
    Group {
        ops: &[("vmovups", Op::Vmovups), ("vmovaps", Op::Vmovaps), ("vmovdqu", Op::Vmovdqu), ("vmovdqa", Op::Vmovdqa)],
        forms: &[&[(V, Write), (Mv, Read)], &[(Mv, Write), (V, Read)], &[(V, Write), (V, Read)]],
    },
    Group {
        ops: &[("vbroadcastss", Op::Vbroadcastss), ("vpbroadcastd", Op::Vpbroadcastd)],
        forms: &[&[(V, Write), (M(W32), Read)], &[(V, Write), (X, Read)]],
    },
    Group { ops: &[("vpbroadcastb", Op::Vpbroadcastb)], forms: &[&[(V, Write), (M(W8), Read)]] },
    Group { ops: &[("vpbroadcastw", Op::Vpbroadcastw)], forms: &[&[(V, Write), (M(W16), Read)]] },
    Group {
        ops: &[("vpbroadcastq", Op::Vpbroadcastq), ("vbroadcastsd", Op::Vbroadcastsd)],
        forms: &[&[(V, Write), (M(W64), Read)], &[(V, Write), (X, Read)]],
    },
    Group { ops: &[("vbroadcastf128", Op::Vbroadcastf128)], forms: &[&[(Y, Write), (M(W128), Read)]] },
    Group {
        ops: &[
            ("vaddss", Op::Vaddss),
            ("vsubss", Op::Vsubss),
            ("vmulss", Op::Vmulss),
            ("vdivss", Op::Vdivss),
            ("vmaxss", Op::Vmaxss),
            ("vminss", Op::Vminss),
            ("vsqrtss", Op::Vsqrtss),
        ],
        forms: &[&[(X, Write), (X, Read), (Xm(W32), Read)]],
    },
    Group {
        ops: &[("vfmadd231ss", Op::Vfmadd231ss), ("vfnmadd231ss", Op::Vfnmadd231ss), ("vfmadd213ss", Op::Vfmadd213ss)],
        forms: &[&[(X, ReadWrite), (X, Read), (Xm(W32), Read)]],
    },
    Group { ops: &[("vroundss", Op::Vroundss)], forms: &[&[(X, Write), (X, Read), (Xm(W32), Read), (C(4), Read)]] },
    Group {
        ops: &[("vmovss", Op::Vmovss)],
        forms: &[&[(X, Write), (M(W32), Read)], &[(M(W32), Write), (X, Read)], &[(X, Write), (X, Read), (X, Read)]],
    },
    Group {
        ops: &[("vmovd", Op::Vmovd)],
        forms: &[
            &[(X, Write), (M(W32), Read)],
            &[(M(W32), Write), (X, Read)],
            &[(X, Write), (G(W32), Read)],
            &[(G(W32), Write), (X, Read)],
        ],
    },
    Group {
        ops: &[("vmovq", Op::Vmovq)],
        forms: &[
            &[(X, Write), (M(W64), Read)],
            &[(M(W64), Write), (X, Read)],
            &[(X, Write), (G(W64), Read)],
            &[(G(W64), Write), (X, Read)],
        ],
    },
    Group { ops: &[("vcomiss", Op::Vcomiss), ("vucomiss", Op::Vucomiss)], forms: &[&[(X, Read), (Xm(W32), Read)]] },
    Group { ops: &[("vextractf128", Op::Vextractf128)], forms: &[&[(X, Write), (Y, Read), (N(1), Read)]] },
    Group {
        ops: &[("vinsertf128", Op::Vinsertf128), ("vinserti128", Op::Vinserti128)],
        forms: &[&[(Y, Write), (Y, Read), (Xm(W128), Read), (N(1), Read)]],
    },
    Group { ops: &[("vpermps", Op::Vpermps)], forms: &[&[(Y, Write), (Y, Read), (Vm, Read)]] },
    Group { ops: &[("vpermq", Op::Vpermq)], forms: &[&[(Y, Write), (Vm, Read), (C(8), Read)]] },
    Group { ops: &[("vmovhlps", Op::Vmovhlps)], forms: &[&[(X, Write), (X, Read), (X, Read)]] },
    Group { ops: &[("vmovhps", Op::Vmovhps)], forms: &[&[(X, Write), (X, Read), (M(W64), Read)]] },
    Group { ops: &[("vcvtph2ps", Op::Vcvtph2ps)], forms: &[&[(Y, Write), (Xm(W128), Read)]] },
    Group {
        ops: &[("vpmovzxbd", Op::Vpmovzxbd)],
        forms: &[&[(Y, Write), (M(W64), Read)], &[(X, Write), (M(W32), Read)], &[(Y, Write), (X, Read)]],
    },
    Group { ops: &[("vpmovsxbd", Op::Vpmovsxbd)], forms: &[&[(Y, Write), (M(W64), Read)], &[(Y, Write), (X, Read)]] },
    Group { ops: &[("vpmovzxwd", Op::Vpmovzxwd)], forms: &[&[(Y, Write), (M(W128), Read)]] },
    Group {
        ops: &[("mov", Op::Mov)],
        forms: &[
            &[(G(W64), Write), (G(W64), Read)],
            &[(G(W32), Write), (G(W32), Read)],
            &[(G(W64), Write), (Q, Read)],
            &[(G(W64), Write), (M(W64), Read)],
            &[(G(W32), Write), (M(W32), Read)],
            &[(M(W64), Write), (G(W64), Read)],
            &[(M(W32), Write), (G(W32), Read)],
        ],
    },
    Group {
        ops: &[("movzx", Op::Movzx)],
        forms: &[&[(G(W32), Write), (M(W8), Read)], &[(G(W32), Write), (M(W16), Read)]],
    },
    Group { ops: &[("lea", Op::Lea)], forms: &[&[(G(W64), Write), (M(W64), Read)]] },
    Group {
        ops: &[("add", Op::Add), ("sub", Op::Sub), ("and", Op::And), ("or", Op::Or), ("xor", Op::Xor)],
        forms: &[
            &[(G(W64), ReadWrite), (I, Read)],
            &[(G(W64), ReadWrite), (G(W64), Read)],
            &[(G(W64), ReadWrite), (M(W64), Read)],
        ],
    },
    Group {
        ops: &[("imul", Op::Imul)],
        forms: &[&[(G(W64), ReadWrite), (G(W64), Read)], &[(G(W64), Write), (G(W64), Read), (I, Read)]],
    },
    Group {
        ops: &[("shl", Op::Shl), ("shr", Op::Shr), ("sar", Op::Sar)],
        forms: &[&[(G(W64), ReadWrite), (N(6), Read)]],
    },
    Group { ops: &[("inc", Op::Inc), ("dec", Op::Dec), ("neg", Op::Neg)], forms: &[&[(G(W64), ReadWrite)]] },
    Group {
        ops: &[("cmp", Op::Cmp)],
        forms: &[&[(G(W64), Read), (G(W64), Read)], &[(G(W64), Read), (I, Read)], &[(G(W64), Read), (M(W64), Read)]],
    },
    Group { ops: &[("test", Op::Test)], forms: &[&[(G(W64), Read), (G(W64), Read)], &[(G(W64), Read), (I, Read)]] },
    Group {
        ops: &[
            ("prefetcht0", Op::Prefetcht0),
            ("prefetcht1", Op::Prefetcht1),
            ("prefetcht2", Op::Prefetcht2),
            ("prefetchnta", Op::Prefetchnta),
        ],
        forms: &[&[(M(W8), Read)]],
    },
];

/// The jumps, each to a label: `jmp` always, the rest as the flags say.
const JUMPS: &[(&str, Op)] = &[
    ("jmp", Op::Jmp),
    ("jz", Op::Jz),
    ("jnz", Op::Jnz),
    ("ja", Op::Ja),
    ("jae", Op::Jae),
    ("jb", Op::Jb),
    ("jbe", Op::Jbe),
    ("jg", Op::Jg),
    ("jge", Op::Jge),
    ("jl", Op::Jl),
    ("jle", Op::Jle),
    ("js", Op::Js),
    ("jns", Op::Jns),
];

/// The names the language calls x86 instructions by, `ret` and the jumps among them.
pub fn mnemonics() -> impl Iterator<Item = &'static str> {
    let groups = GROUPS.iter().flat_map(|g| g.ops.iter().map(|(name, _)| *name));
    groups.chain(JUMPS.iter().map(|(name, _)| *name)).chain(["ret", "call", "ud2", "zero", "ones"])
}

/// A type SysV passes in a register: an integer or a pointer in a general one, a float in the
/// low lanes of a vector one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    Int,
    F32,
    F64,
}

impl Ty {
    /// The type of a parameter written `ty`, `*const f32`; or why SysV passes none such whole.
    fn param(ty: &str) -> Result<Ty, String> {
        match ty {
            "f32" => Ok(Ty::F32),
            "f64" => Ok(Ty::F64),
            _ if ty.starts_with("*const ") || ty.starts_with("*mut ") => Ok(Ty::Int),
            "usize" | "isize" | "u64" | "i64" => Ok(Ty::Int),
            "u32" | "i32" | "u16" | "i16" | "u8" | "i8" | "bool" => {
                Err(format!("SysV leaves the high bits of a `{ty}` as they were; take a usize or a u64"))
            }
            _ => Err(format!("a kernel takes pointers, 64-bit integers and floats, not `{ty}`")),
        }
    }

    fn class(self) -> Class {
        if self == Ty::Int { Class::Gpr } else { Class::Vec }
    }

    fn noun(self) -> &'static str {
        match self {
            Ty::Int => "an integer",
            Ty::F32 => "an `f32`",
            Ty::F64 => "an `f64`",
        }
    }
}

/// What a function gives back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Returns {
    /// Nothing, `()`.
    Nothing,
    /// A value: an integer in `rax`, a float in `xmm0`.
    Value(Ty),
    /// It never returns, `-> !`.
    Never,
}

impl Returns {
    /// What a function whose result is written `ty` gives back, an integer narrower than 64 bits
    /// as all of `rax`; or why SysV gives back none such in a register.
    fn of(ty: &str) -> Result<Returns, String> {
        match ty {
            "()" => Ok(Returns::Nothing),
            "!" => Ok(Returns::Never),
            "f32" => Ok(Returns::Value(Ty::F32)),
            "f64" => Ok(Returns::Value(Ty::F64)),
            _ if ty.starts_with("*const ") || ty.starts_with("*mut ") => Ok(Returns::Value(Ty::Int)),
            "usize" | "isize" | "u64" | "i64" | "u32" | "i32" | "u16" | "i16" | "u8" | "i8" | "bool" => {
                Ok(Returns::Value(Ty::Int))
            }
            _ => Err(format!("a kernel returns a float, an integer or a pointer, or never, `!`; not `{ty}`")),
        }
    }
}

/// What a kernel makes of a function of its module `extern "sysv64"`, which it may name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Callee {
    /// One it calls, which takes and gives back what the prototype says.
    Calls(Prototype),
    /// A cooperative kernel, whose threads run on a grid its launcher starts.
    Cooperative,
    /// One that does not compile, which says why where it is declared.
    Broken,
}

/// What a function of the module takes and gives, as a kernel that calls it passes them: each
/// parameter by its name and its type, and what it returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prototype {
    params: Vec<(String, Ty)>,
    returns: Returns,
}

impl Prototype {
    /// The prototype of a function of the parameters `params`, each a name and a type as
    /// written, `("x", "*const f32")`, and of the result `ret`, `None` for none; or why SysV
    /// passes them in no registers.
    pub fn of(params: &[(&str, &str)], ret: Option<&str>) -> Result<Prototype, String> {
        let params: Vec<(String, Ty)> =
            params.iter().map(|&(name, ty)| Ty::param(ty).map(|t| (name.to_string(), t))).collect::<Result<_, _>>()?;
        let ints = params.iter().filter(|(_, t)| *t == Ty::Int).count();
        if ints > INT_PARAMS.len() {
            return Err(SEVENTH.into());
        }
        if params.len() - ints > FLOAT_PARAMS.len() {
            return Err(NINTH.into());
        }
        Ok(Prototype { params, returns: Returns::of(ret.unwrap_or("()"))? })
    }
}

/// Memory of the module's, which a `static` of the source declares: elements of one of Rust's
/// numbers, in the process's memory, which each kernel of the module names, which the host
/// reads and writes, and which keeps what is written from one call to the next. Constant
/// memory the kernels only read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Static {
    /// What the source calls it, which a kernel and its listing name it by: `COUNTS`.
    pub name: String,
    /// Its type, as the source declares it: `global<[u32; 16]>`.
    pub ty: String,
    /// Its elements' type, one of Rust's numbers: `u32`.
    pub elem: &'static str,
    /// The bytes of each element.
    pub size: u64,
    /// How many elements it has.
    pub len: u64,
    /// What its address is a multiple of: a power of two up to a page, its element's bytes
    /// unless it says more.
    pub align: u64,
    /// Whether it is constant memory, `constant<..>`, which the kernels only read.
    pub constant: bool,
    /// What it holds until something writes it, as bytes, the lowest first; zeros past their
    /// end.
    pub init: Vec<u8>,
}

impl Static {
    /// The static `name`, of the type `ty` as the source declares it, `global<[u32; 16]>`, or
    /// `constant<[f32; 8], 32>` at a multiple of 32: holding zeros until
    /// [`init`](Static::init) says otherwise. Or why it is none.
    pub fn new(name: &str, ty: &str) -> Result<Static, String> {
        let (word, constant) = match ty.split_once('<').map(|(word, _)| word) {
            Some("global") => ("global", false),
            Some("constant") => ("constant", true),
            _ => {
                return Err(format!(
                    "a static of the module is memory: `global<[u32; 16]>`, which its kernels read and write, or \
                     `constant<[f32; 4]>`, which they only read; not `{ty}`"
                ));
            }
        };
        let (elem, size, len, align) = shape(ty, word)?;
        let Some(len) = len else {
            return Err(format!("`{ty}` has no size: a static has one, as in `{word}<[{elem}; 64]>`"));
        };
        if len.checked_mul(size).is_none_or(|bytes| bytes > FAR) {
            return Err(format!(
                "`{ty}` is more than {FAR} bytes: a kernel reaches memory of the module's at a displacement from \
                 `rip`, of 32 bits, signed"
            ));
        }
        Ok(Static { name: name.to_string(), ty: ty.to_string(), elem, size, len, align, constant, init: vec![] })
    }

    /// Makes it hold `value` until something writes it: an array of as many elements as it
    /// has, each of its element's type, `[1.5, 2.0]`; or one, for memory of one that is no
    /// array, `global<f32>`. Or why it cannot.
    pub fn init(&mut self, value: &Value) -> Result<(), String> {
        let (ty, elem) = (&self.ty, self.elem);
        let array = ty.split_once('<').is_some_and(|(_, args)| args.starts_with('['));
        let items = match (value, array) {
            (Value::Array(xs), true) if xs.len() as u64 == self.len => xs.as_slice(),
            (Value::Array(xs), true) => {
                return Err(format!("`{ty}` holds {} elements; this has {}", self.len, xs.len()));
            }
            (x, true) => return Err(format!("`{ty}` holds an array of `{elem}`; this is {}", x.kind())),
            (x, false) => std::slice::from_ref(x),
        };
        let mut init = vec![];
        for (i, x) in items.iter().enumerate() {
            let what = if array { format!("element {i}") } else { "this".into() };
            init.extend(datum(elem, self.size, x, &what)?);
        }
        self.init = init;
        Ok(())
    }

    /// How many bytes it is.
    pub fn bytes(&self) -> u64 {
        self.len * self.size
    }
}

/// `x` as the bytes of a `t`, one of Rust's numbers, of `size` bytes, in memory, the lowest
/// first; or why it is none, of `x` as `what` names it, `this` or `element 2`.
fn datum(t: &str, size: u64, x: &Value, what: &str) -> Result<Vec<u8>, String> {
    // A type's name as it is read: an `i8`, an `f32`, a `u64`.
    let a = |name: &str| if name.starts_with(['f', 'i']) { "an" } else { "a" };
    let bits = size * 8;
    match *x {
        Value::Int(v, ty) if !t.starts_with('f') => {
            if let Some(ty) = ty
                && ty.name() != t
            {
                return Err(format!("{what} is {} `{}`, not {} `{t}`", a(ty.name()), ty.name(), a(t)));
            }
            let (lo, hi) = match t.starts_with('i') {
                true => (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1),
                false => (0, (1i128 << bits) - 1),
            };
            if !(lo..=hi).contains(&v) {
                return Err(format!("{what} is {v}, out of range for `{t}`"));
            }
            Ok(v.to_le_bytes()[..size as usize].to_vec())
        }
        Value::Int(v, _) => Err(format!("{what} is an integer, and {} `{t}` a float: `{v}.0`", a(t))),
        Value::F32(v) if t == "f32" => Ok(v.to_le_bytes().to_vec()),
        Value::F64(v) if t == "f64" => Ok(v.to_le_bytes().to_vec()),
        ref x => Err(format!("{what} is {}, not {} `{t}`", x.kind(), a(t))),
    }
}

/// Why a function takes no more integers, or no more floats.
const SEVENTH: &str = "SysV passes a seventh integer on the stack, which kgen does not read";
const NINTH: &str = "SysV passes a ninth float on the stack, which kgen does not read";

/// How a function that never returns ends.
const NEVER: &str = "it ends each way out in `ud2()` or in a call of a function that never returns";

/// A kernel being written in the language, for x86-64 under SysV.
pub struct Kernel {
    p: Program<Op>,
    ints: usize,
    floats: usize,
    /// What it gives back.
    returns: Returns,
    /// The functions it may call, each by what the source calls it, with its prototype; `None`
    /// for a cooperative kernel, which it may not.
    callees: Vec<(String, Callee)>,
    /// Each slot's type as declared, `local<[f32; 64], 32>`, by [`SlotId`].
    declared: Vec<String>,
    /// The memory of the module's it may name: each static of the module that compiles.
    module: Vec<Static>,
    /// The memory of the module's it names, by [`StaticId`], as the program names them.
    statics: Vec<Static>,
    /// Whether it runs on the threads of a grid, `#[cooperative]`.
    cooperative: bool,
    /// The variable a thread's place in the grid arrives in, once there is one.
    grid: Option<VReg>,
    /// The next label of the kernel's own code, `grid.sync()`'s. They count down from the
    /// last, and the evaluator's up from 0, so that the two never meet.
    labels: u32,
}

impl Kernel {
    /// An empty kernel called `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Kernel {
            p: Program {
                name: name.into(),
                signature: Signature { params: vec![], result: None },
                regs: vec![],
                params: vec![],
                results: vec![],
                pinned: vec![],
                body: vec![],
                slots: vec![],
                consts: vec![],
                funcs: vec![],
                statics: vec![],
            },
            ints: 0,
            floats: 0,
            returns: Returns::Nothing,
            callees: vec![],
            declared: vec![],
            module: vec![],
            statics: vec![],
            cooperative: false,
            grid: None,
            labels: u32::MAX,
        }
    }

    /// Lets the kernel name `name`, a function of its module `extern "sysv64"`, or an instance
    /// of one, `tile::<4>`, which it calls or may not, as `callee` says.
    pub fn callee(&mut self, name: impl Into<String>, callee: Callee) {
        self.callees.push((name.into(), callee));
    }

    /// Lets the kernel name `s`, memory of the module's, by its name in the source, as it names
    /// stack memory: at a displacement from `rip`, as far as its end.
    pub fn module_memory(&mut self, s: &Static) {
        self.module.push(s.clone());
    }

    /// The name of memory `v`, of the stack or of the module's, and its type as declared; `None`
    /// for a register.
    fn declared(&self, v: Var) -> Option<(&str, &str)> {
        match v.0 {
            Place::Reg(..) => None,
            Place::Stack(s) => Some((&self.p.slots[s.0 as usize].name, &self.declared[s.0 as usize])),
            Place::Static(s) => {
                let s = &self.statics[s.0 as usize];
                Some((&s.name, &s.ty))
            }
        }
    }

    /// The prototype of `f`, a function the kernel names by its operand `at`; or why it may not
    /// call it.
    fn called(&self, f: &str, at: usize) -> Result<Prototype, Fault> {
        match self.callees.iter().find(|(name, _)| name == f) {
            Some((_, Callee::Calls(proto))) => Ok(proto.clone()),
            Some((_, Callee::Cooperative)) => Err(Fault::at(
                at,
                format!(
                    "`{f}` is `#[cooperative]`: its threads run on a grid, which its launcher starts, so no kernel \
                     calls it"
                ),
            )),
            Some((_, Callee::Broken)) => Err(Fault::at(at, format!("`{f}` does not compile, so no kernel calls it"))),
            None => Err(Fault::at(
                at,
                format!("`{f}` is no x86 function: an x86 kernel calls those of its module `extern \"sysv64\"`"),
            )),
        }
    }

    /// The function `f`, as the program names it: each once.
    fn func(&mut self, f: &str) -> FuncId {
        let at = match self.p.funcs.iter().position(|g| g == f) {
            Some(at) => at,
            None => {
                self.p.funcs.push(f.to_string());
                self.p.funcs.len() - 1
            }
        };
        FuncId(at as u32)
    }

    /// `call`: of a function of the module by its name, `call(r, f, (a, b))`, or of the one at
    /// the address in a gpr, `call(r, fp, (a, b))`, which passes each in the file its variable
    /// is of, or as one of the functions taken together last, `call(r, fp, (a, b), (f, g))`.
    /// Each argument is moved to a variable of its own in the register SysV passes it in, and
    /// the result back from one in the register it comes back in, for the allocator to keep
    /// there while the call needs it; the call changes every register a callee may.
    fn call(&mut self, args: &[Given<Var>]) -> Result<(), Fault> {
        let shape = "`call` is `call(f, (a, b))`, or `call(r, f, (a, b))` for a function that returns a value, `f` \
                     a function of the module or a gpr that holds the address of one";
        let targets: Option<Vec<&str>> = match args.last() {
            Some(Given::Vector(fs)) if !fs.is_empty() && fs.iter().all(|f| matches!(f, Given::Func(_))) => {
                Some(fs.iter().filter_map(|f| if let Given::Func(f) = f { Some(f.as_str()) } else { None }).collect())
            }
            _ => None,
        };
        let rest = &args[..args.len() - usize::from(targets.is_some())];
        let (result, callee, passed) = match (rest.iter().position(|a| matches!(a, Given::Func(_))), rest.len()) {
            (Some(0), 1 | 2) | (None, 1 | 2) => (None, 0, rest.get(1)),
            (Some(1), 2 | 3) | (None, 3) => (Some(&rest[0]), 1, rest.get(2)),
            (Some(i), _) if i >= 2 => return Err(Fault::at(i, shape)),
            _ => return Err(Fault::new(shape)),
        };
        let passed: Vec<&Given<Var>> = match passed {
            None => vec![],
            Some(Given::Vector(xs)) => xs.iter().collect(),
            Some(a) => vec![a],
        };
        let at = callee + 1;
        // What the call passes and gives back, whether it returns, and the names of its callee's
        // parameters, which name the variables that pass values known now.
        let (target, (tys, kept), never, names) = match (&rest[callee], &targets) {
            (Given::Func(f), None) => {
                let proto = self.called(f, callee)?;
                let passing = self.passing(f, &proto, &passed, result, at)?;
                let names = proto.params.into_iter().map(|(name, _)| name).collect();
                (Operand::Func(self.func(f)), passing, proto.returns == Returns::Never, names)
            }
            (Given::Func(f), Some(_)) => {
                return Err(Fault::at(
                    callee,
                    format!(
                        "`call` names the function it calls, `{f}`, or takes its address from a gpr, of one of the \
                         functions taken together last; not both"
                    ),
                ));
            }
            (fp, targets) => {
                let fp = match reg_of(fp) {
                    Some((reg, View::R64)) if reg.class == Class::Gpr => reg,
                    _ => {
                        return Err(Fault::at(
                            callee,
                            format!(
                                "`call` calls a function by its name, or at the address in a gpr; this is {}",
                                given(fp)
                            ),
                        ));
                    }
                };
                match targets {
                    // Each of the functions must take what the call passes. They take it the same
                    // way: a variable passes as it is, whatever its callee makes of it, and a value
                    // known now is of one type.
                    Some(targets) => {
                        let (mut passing, mut never) = (None, true);
                        for t in targets {
                            let proto = self.called(t, args.len() - 1)?;
                            never &= proto.returns == Returns::Never;
                            passing = Some(self.passing(t, &proto, &passed, result, at)?);
                        }
                        (Operand::Reg(fp, W64), passing.expect("a call has a target"), never, vec![])
                    }
                    None => (Operand::Reg(fp, W64), self.shaped(&passed, result, at)?, false, vec![]),
                }
            }
        };
        // Each argument to the register it goes in, in order; then the call; then the result.
        let mut implicit = vec![];
        let (mut ints, mut floats) = (0, 0);
        for (k, (a, ty)) in passed.iter().zip(&tys).enumerate() {
            let (count, regs) =
                if *ty == Ty::Int { (&mut ints, &INT_PARAMS[..]) } else { (&mut floats, &FLOAT_PARAMS[..]) };
            let Some(&to) = regs.get(*count) else {
                let why = if *ty == Ty::Int { SEVENTH } else { NINTH };
                return Err(Fault::at(at, format!("this passes too many: {why}")));
            };
            *count += 1;
            let name = match reg_of(a) {
                Some((reg, _)) => self.p.regs[reg.id as usize].name.clone(),
                None => names.get(k).cloned().unwrap_or_else(|| format!("argument {}", k + 1)),
            };
            let pinned = self.new_var(to.class, &name);
            self.p.pinned.push((pinned, to));
            self.pass(pinned, a, *ty);
            implicit.push(Arg { operand: Operand::Reg(pinned, SysV.width(to.class)), access: Read });
        }
        let back = match (result.and_then(reg_of), kept) {
            (Some((reg, view)), Some(ty)) => {
                let from = if ty == Ty::Int { INT_RESULT } else { FLOAT_RESULT };
                let pinned = self.new_var(from.class, &self.p.regs[reg.id as usize].name.clone());
                self.p.pinned.push((pinned, from));
                implicit.push(Arg { operand: Operand::Reg(pinned, SysV.width(from.class)), access: Write });
                Some((reg, view, pinned))
            }
            _ => None,
        };
        let flow = if never { Flow::Return } else { Flow::Next };
        let call = Inst {
            op: Op::Call,
            args: vec![Arg { operand: target, access: Read }],
            implicit,
            flow,
            clobbers: CALL_CLOBBERS.to_vec(),
        };
        self.p.body.push(Item::Inst(call));
        if let Some((reg, view, pinned)) = back {
            self.p.body.push(Item::Inst(match view {
                View::R32 => move_r32(reg, pinned),
                _ => SysV.copy(reg, pinned),
            }));
        }
        Ok(())
    }

    /// Moves `a`, an argument of type `ty`, to `to`, the variable SysV passes it in: a variable
    /// whole, or a gpr's `r32` with the rest clear; an integer known now; or a float known now,
    /// from memory.
    fn pass(&mut self, to: VReg, a: &Given<Var>, ty: Ty) {
        let inst = match (a, ty) {
            (Given::Var(v), _) => match v.reg().expect("an argument is a variable of a register") {
                (reg, View::R32) => move_r32(to, reg),
                (reg, _) => SysV.copy(to, reg),
            },
            (Given::Int(x), _) => {
                let value = Arg { operand: Operand::Imm(Imm { value: *x as i64, hex: false }), access: Read };
                let to = Arg { operand: Operand::Reg(to, W64), access: Write };
                Inst { op: Op::Mov, args: vec![to, value], implicit: vec![], flow: Flow::Next, clobbers: vec![] }
            }
            (Given::F32(x), _) => self.load_float(to, &x.to_le_bytes(), Op::Vmovss, W32),
            (Given::F64(x), _) => self.load_float(to, &x.to_le_bytes(), Op::Vmovq, W64),
            _ => unreachable!("an argument is a variable or a value known now"),
        };
        self.p.body.push(Item::Inst(inst));
    }

    /// `op(to.xmm, [bytes])`: a float known now, as a constant.
    fn load_float(&mut self, to: VReg, bytes: &[u8], op: Op, width: Width) -> Inst<Op, VReg> {
        let at = Mem { base: Base::Const(self.constant(bytes)), index: None, disp: 0, width };
        let args = vec![
            Arg { operand: Operand::Reg(to, W128), access: Write },
            Arg { operand: Operand::Mem(at), access: Read },
        ];
        Inst { op, args, implicit: vec![], flow: Flow::Next, clobbers: vec![] }
    }

    /// How a call of `f`, of `proto`, passes `passed`, the operand `at`, and gives back to
    /// `result`: each argument's type, and the result's, unless nothing keeps it; or what is
    /// wrong.
    fn passing(
        &self,
        f: &str,
        proto: &Prototype,
        passed: &[&Given<Var>],
        result: Option<&Given<Var>>,
        at: usize,
    ) -> Result<(Vec<Ty>, Option<Ty>), Fault> {
        let kept = match (proto.returns, result) {
            (Returns::Value(t), None) => {
                return Err(Fault::new(format!(
                    "`{f}` returns {}: `call(r, {f}, ..)` writes it to `r`, or `call(_, {f}, ..)` to nothing",
                    t.noun()
                )));
            }
            (Returns::Value(_), Some(Given::Special(s))) if s == "_" => None,
            (Returns::Value(t), Some(r)) => match reg_of(r) {
                Some((reg, _)) if reg.class == t.class() => Some(t),
                _ => {
                    let into = if t == Ty::Int { "a gpr" } else { "a vector" };
                    return Err(Fault::at(
                        0,
                        format!("`{f}` returns {}, to {into} or `_`; this is {}", t.noun(), given(r)),
                    ));
                }
            },
            (Returns::Nothing, Some(_)) => {
                return Err(Fault::at(0, format!("`{f}` returns nothing: `call({f}, ..)` calls it")));
            }
            (Returns::Never, Some(_)) => {
                return Err(Fault::at(0, format!("`{f}` never returns: `call({f}, ..)` calls it")));
            }
            (_, None) => None,
        };
        if passed.len() != proto.params.len() {
            let takes = match proto.params.len() {
                0 => "no arguments".into(),
                1 => "1 argument".into(),
                n => format!("{n} arguments"),
            };
            return Err(Fault::at(at, format!("`{f}` takes {takes}, not {}", passed.len())));
        }
        let mut tys = vec![];
        for (k, ((name, t), a)) in proto.params.iter().zip(passed).enumerate() {
            let fits = match (t, a) {
                (_, Given::Var(v)) => v.reg().is_some_and(|(reg, _)| reg.class == t.class()),
                (Ty::Int, Given::Int(x)) => (i128::from(i64::MIN)..=i128::from(u64::MAX)).contains(x),
                (Ty::F32, Given::F32(_)) | (Ty::F64, Given::F64(_)) => true,
                _ => false,
            };
            if !fits {
                let what = match (t, a) {
                    (_, Given::Func(g)) => format!("the function `{g}`: `lea(p, [{g}])` puts its address in `p`"),
                    (Ty::F64, Given::F32(x)) => format!("the f32 {x}: `{x:?}f64` is an `f64`"),
                    (Ty::F32, Given::Int(x)) => format!("the integer {x}: `{x}.0` is an `f32`"),
                    (Ty::F64, Given::Int(x)) => format!("the integer {x}: `{x}.0f64` is an `f64`"),
                    (_, Given::F32(x)) => format!("the f32 {x}"),
                    (_, Given::F64(x)) => format!("the f64 {x}"),
                    (_, a) => given(a),
                };
                let which = if proto.params.len() == 1 { String::new() } else { format!(" as argument {}", k + 1) };
                return Err(Fault::at(at, format!("`{f}` takes {} `{name}`{which}; this is {what}", t.noun())));
            }
            tys.push(*t);
        }
        Ok((tys, kept))
    }

    /// How a call through a variable, of no functions taken together, passes `passed`, the
    /// operand `at`, and gives back to `result`: each in the file its variable is of, so each
    /// is a variable; or what is wrong.
    fn shaped(
        &self,
        passed: &[&Given<Var>],
        result: Option<&Given<Var>>,
        at: usize,
    ) -> Result<(Vec<Ty>, Option<Ty>), Fault> {
        let ty = |a: &Given<Var>, at: usize, what: &str| match reg_of(a) {
            Some((reg, _)) if reg.class == Class::Gpr => Ok(Ty::Int),
            Some(_) => Ok(Ty::F64),
            None => Err(Fault::at(
                at,
                format!(
                    "a call through a gpr {what} variables, each in the file it is of: this is {}; taking the \
                     functions it may call last, `call(r, fp, (a, b), (f, g))`, lets it be more",
                    given(a)
                ),
            )),
        };
        let kept = match result {
            Some(Given::Special(s)) if s == "_" => None,
            Some(r) => Some(ty(r, 0, "writes")?),
            None => None,
        };
        let tys = passed.iter().map(|a| ty(a, at, "passes")).collect::<Result<_, _>>()?;
        Ok((tys, kept))
    }

    /// Makes the kernel one that runs on the threads of a grid, `#[cooperative]`, before its
    /// parameters are declared: each thread is given its place in the grid after them, and
    /// may wait for the others at `grid.sync()`.
    pub fn cooperative(&mut self) {
        self.cooperative = true;
    }

    /// Stack memory declared as `let name: ty`, `ty` being `local<[f32; 64], 32>`.
    fn new_memory(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        let (elem, size, len, align) = shape(ty, "local")?;
        let Some(len) = len else {
            return Err(format!("`{ty}` has no size: stack memory has one, as in `local<[{elem}; 64]>`"));
        };
        // The frame as `frame::layout` will lay it out, this last.
        let (mut end, mut most) = (0, align);
        for slot in &self.p.slots {
            end = u64::next_multiple_of(end, slot.align.into()) + u64::from(slot.size);
            most = most.max(slot.align.into());
        }
        let bytes = len.checked_mul(size);
        let all = bytes
            .and_then(|b| end.next_multiple_of(align).checked_add(b))
            .and_then(|e| e.checked_next_multiple_of(most));
        let (Some(bytes), Some(..=FRAME)) = (bytes, all) else {
            return Err(format!(
                "this makes the kernel's stack memory more than {FRAME} bytes, the most an address reaches past the \
                 stack pointer"
            ));
        };
        let narrow = |n: u64| u32::try_from(n).expect("a frame of 2^31 bytes at most");
        self.p.slots.push(Slot { name: name.to_string(), size: narrow(bytes), align: narrow(align), spill: false });
        self.declared.push(ty.to_string());
        Ok(Var(Place::Stack(SlotId(self.p.slots.len() as u32 - 1))))
    }

    /// The kernel as written. A cooperative one takes its place in the grid, read or not, as
    /// the launcher passes it.
    pub fn finish(mut self) -> Program<Op> {
        if self.cooperative {
            self.grid();
        }
        self.p
    }

    /// The variable a cooperative kernel's thread is given its place in the grid in: a
    /// parameter after the others, in the register SysV passes the next integer in, which
    /// [`Target::param`] leaves free.
    fn grid(&mut self) -> VReg {
        if let Some(g) = self.grid {
            return g;
        }
        let at = INT_PARAMS[self.ints];
        let g = self.new_var(Class::Gpr, "grid");
        self.p.params.push((g, at));
        self.p.signature.params.push(Class::Gpr);
        self.grid = Some(g);
        g
    }

    /// A label of the kernel's own code.
    fn own_label(&mut self) -> Label {
        self.labels -= 1;
        Label(self.labels + 1)
    }

    /// The memory that `name`, which is no variable, is if it is the grid's: `grid.thread` or
    /// `grid.threads`, in the place a cooperative kernel's thread is given. `None` for a name
    /// that is not the grid's.
    fn grid_value(&mut self, name: &str) -> Result<Option<Given<Var>>, String> {
        let disp = match name {
            "grid.thread" => THREAD,
            "grid.threads" => THREADS,
            _ if name.starts_with("grid.") => {
                return Err(format!(
                    "a thread reads `grid.thread`, its index in the grid, and `grid.threads`, how many there are; \
                     not `{name}`"
                ));
            }
            _ => return Ok(None),
        };
        if !self.cooperative {
            return Err(format!(
                "`{name}` is of the grid a kernel runs on when it is `#[cooperative]`, and `{}` is not",
                self.p.name
            ));
        }
        let base = Var(Place::Reg(self.grid(), View::R64));
        Ok(Some(Given::Mem(kgen_lang::Mem { base: Some(base), index: None, disp: disp.into() })))
    }

    /// `grid.sync()`, which waits until every thread of the grid has come to it: code of the
    /// kernel's own, for it calls nothing. A thread reads the barrier's generation, then counts
    /// itself in with `lock xadd`, which first makes what it wrote seen by every thread. The last
    /// to come resets the count for the next time, then moves the generation on, which lets the
    /// others go; they wait for it with `pause`, as Intel asks of a loop that spins. x86 makes a
    /// thread's stores seen in the order it makes them, so one that sees the new generation sees
    /// the count reset, and what every thread wrote before it came.
    fn sync(&mut self, name: &str, args: &[Given<Var>], guard: bool) -> Result<(), Fault> {
        if name != "grid.sync" {
            return Err(Fault::new(format!("kgen's barrier for the grid is `grid.sync()`; not `{name}`")));
        }
        if !args.is_empty() {
            return Err(Fault::new(format!("`{name}` takes no operands, not {}", args.len())));
        }
        if !self.cooperative {
            return Err(Fault::new(format!(
                "`grid.sync()` waits for every thread of the grid, which runs all at once only if `{}` is \
                 `#[cooperative]`",
                self.p.name
            )));
        }
        if guard {
            return Err(Fault::new(
                "`grid.sync()` is for every thread of the grid, under no `if`: the grid waits for ever for one that \
                 does not come to it",
            ));
        }
        let grid = self.grid();
        let (wait, done) = (self.own_label(), self.own_label());
        let barrier = self.new_var(Class::Gpr, "barrier");
        let generation = self.new_var(Class::Gpr, "generation");
        let arrived = self.new_var(Class::Gpr, "arrived");
        let reg = |v: VReg, access: Access| Arg { operand: Operand::Reg(v, W64), access };
        let mem = |base: VReg, disp: i32, access: Access| Arg {
            operand: Operand::Mem(Mem { base: Base::Reg(base), index: None, disp, width: W64 }),
            access,
        };
        let int = |value: i64| Arg { operand: Operand::Imm(Imm { value, hex: false }), access: Read };
        let to = |l: Label| vec![Arg { operand: Operand::Label(l), access: Read }];
        self.p.body.push(Item::Note("grid.sync()".into()));
        self.push(Op::Mov, vec![reg(barrier, Write), mem(grid, BARRIER, Read)], vec![], Flow::Next);
        self.push(Op::Mov, vec![reg(generation, Write), mem(barrier, GENERATION, Read)], vec![], Flow::Next);
        self.push(Op::Mov, vec![reg(arrived, Write), int(1)], vec![], Flow::Next);
        self.push(Op::LockXadd, vec![mem(barrier, ARRIVED, ReadWrite), reg(arrived, ReadWrite)], vec![], Flow::Next);
        self.push(Op::Inc, vec![reg(arrived, ReadWrite)], vec![], Flow::Next);
        self.push(Op::Cmp, vec![reg(arrived, Read), mem(grid, THREADS, Read)], vec![], Flow::Next);
        self.push(Op::Jnz, to(wait), vec![], Flow::Branch(wait));
        self.p.body.push(Item::Note("the last to come lets the others go".into()));
        self.push(Op::Mov, vec![mem(barrier, ARRIVED, Write), int(0)], vec![], Flow::Next);
        self.push(Op::Inc, vec![reg(generation, ReadWrite)], vec![], Flow::Next);
        self.push(Op::Mov, vec![mem(barrier, GENERATION, Write), reg(generation, Read)], vec![], Flow::Next);
        self.push(Op::Jmp, to(done), vec![], Flow::Jump(done));
        self.p.body.push(Item::Label(wait));
        self.p.body.push(Item::Note("the others wait for it".into()));
        self.push(Op::Pause, vec![], vec![], Flow::Next);
        self.push(Op::Cmp, vec![reg(generation, Read), mem(barrier, GENERATION, Read)], vec![], Flow::Next);
        self.push(Op::Jz, to(wait), vec![], Flow::Branch(wait));
        self.p.body.push(Item::Label(done));
        Ok(())
    }

    fn new_var(&mut self, class: Class, name: &str) -> VReg {
        let id = self.p.regs.len() as u32;
        self.p.regs.push(RegInfo { class, name: name.to_string() });
        VReg { id, class }
    }

    fn push(&mut self, op: Op, args: Vec<Arg<VReg>>, implicit: Vec<Arg<VReg>>, flow: Flow) {
        self.p.body.push(Item::Inst(Inst { op, args, implicit, flow, clobbers: vec![] }));
    }

    /// Read-only `bytes`, aligned to their size up to 32, stored once however often asked for.
    fn constant(&mut self, bytes: &[u8]) -> ConstId {
        let align = (bytes.len() as u32).next_power_of_two().min(32);
        let at = match self.p.consts.iter().position(|c| c.bytes == bytes) {
            Some(at) => {
                self.p.consts[at].align = self.p.consts[at].align.max(align);
                at
            }
            None => {
                self.p.consts.push(Const { bytes: bytes.to_vec(), align });
                self.p.consts.len() - 1
            }
        };
        ConstId(at as u32)
    }

    fn ret(&mut self, args: &[Given<Var>]) -> Result<(), Fault> {
        let given = match args {
            [a] => reg_of(a).map(|(reg, _)| reg),
            _ => None,
        };
        let (reg, width, at) = match (self.returns, args, given) {
            (Returns::Nothing, [], _) => {
                self.push(Op::Ret, vec![], vec![], Flow::Return);
                return Ok(());
            }
            (Returns::Value(Ty::Int), _, Some(r)) if r.class == Class::Gpr => (r, W64, INT_RESULT),
            (Returns::Value(_), _, Some(r)) if r.class == Class::Vec => (r, W128, FLOAT_RESULT),
            (Returns::Nothing, ..) => return Err(Fault::new("this kernel returns nothing: `ret()`")),
            (Returns::Never, ..) => {
                return Err(Fault::new(format!("`{}` never returns, `-> !`: {NEVER}", self.p.name)));
            }
            (Returns::Value(Ty::Int), ..) => {
                return Err(Fault::new("this kernel returns an integer: `ret(g)`, `g` a gpr"));
            }
            (Returns::Value(_), ..) => return Err(Fault::new("this kernel returns a float: `ret(v)`, `v` a vector")),
        };
        if !self.p.results.contains(&(reg, at)) {
            self.p.results.push((reg, at));
        }
        let implicit = vec![Arg { operand: Operand::Reg(reg, width), access: Read }];
        self.push(Op::Ret, vec![], implicit, Flow::Return);
        Ok(())
    }

    /// `zero(v)` and `ones(v)`: all bits clear or set, reading nothing.
    fn fill(&mut self, ones: bool, args: &[Given<Var>]) -> Result<(), Fault> {
        let name = if ones { "ones" } else { "zero" };
        let Some((v, view)) = (match args {
            [a] => reg_of(a),
            _ => None,
        }) else {
            return Err(Fault::new(format!("`{name}` takes the variable it sets")));
        };
        let reg = |w: Width, access: Access| Arg { operand: Operand::Reg(v, w), access };
        match (v.class, ones) {
            // A VEX write to the low half clears the rest, so zeros cost the shorter encoding.
            (Class::Vec, false) => self.push(
                Op::Vxorps,
                vec![reg(W128, Write), reg(W128, Ignored), reg(W128, Ignored)],
                vec![],
                Flow::Next,
            ),
            (Class::Vec, true) => {
                let w = view.width();
                self.push(Op::Vpcmpeqd, vec![reg(w, Write), reg(w, Ignored), reg(w, Ignored)], vec![], Flow::Next)
            }
            (_, false) => self.push(Op::Xor, vec![reg(W32, Write), reg(W32, Ignored)], vec![], Flow::Next),
            (_, true) => {
                let all = Arg { operand: Operand::Imm(Imm { value: -1, hex: false }), access: Read };
                self.push(Op::Mov, vec![reg(W64, Write), all], vec![], Flow::Next)
            }
        }
        Ok(())
    }

    /// The width of the form's vectors: its first vector's that has a variable, else the
    /// width its fixed vectors name.
    fn width(form: Form, args: &[Given<Var>]) -> Option<Width> {
        let given = form.iter().zip(args).find_map(|((kind, _), a)| match kind {
            V | Vm => reg_of(a).filter(|(reg, _)| reg.class == Class::Vec).map(|(_, view)| view.width()),
            _ => None,
        });
        given.or_else(|| {
            form.iter().find_map(|(kind, _)| match kind {
                Y => Some(W256),
                X => Some(W128),
                _ => None,
            })
        })
    }

    /// Whether `args` fit `form`, or the first that does not and why.
    fn fits(form: Form, args: &[Given<Var>]) -> Result<(), Fault> {
        if form.len() != args.len() {
            return Err(Fault::new(format!("takes {} operands, not {}", form.len(), args.len())));
        }
        let width = Self::width(form, args);
        for (i, (&(kind, _), a)) in form.iter().zip(args).enumerate() {
            let vector = |want: Option<View>| match reg_of(a) {
                Some((reg, view)) if reg.class == Class::Vec && want.is_none_or(|w| w == view) => Ok(()),
                _ => Err(Fault::at(i, format!("wants {}; this is {}", describe(kind, width), given(a)))),
            };
            let memory = matches!(a, Given::Mem(_) | Given::Data(_) | Given::Code(_));
            let int = |range: std::ops::RangeInclusive<i128>| match a {
                Given::Int(x) if range.contains(x) => Ok(()),
                _ => Err(Fault::at(i, format!("wants {}; this is {}", describe(kind, width), given(a)))),
            };
            match kind {
                V => vector(width.map(view_of))?,
                Vm if memory => {}
                Vm => vector(width.map(view_of))?,
                Y => vector(Some(View::Ymm))?,
                X => vector(Some(View::Xmm))?,
                Xm(_) if memory => {}
                Xm(_) => vector(Some(View::Xmm))?,
                Mv | M(_) if memory => {}
                G(w) => match reg_of(a) {
                    Some((reg, view)) if reg.class == Class::Gpr && view.width() == w => {}
                    _ => return Err(Fault::at(i, format!("wants {}; this is {}", describe(kind, width), given(a)))),
                },
                I => int(i32::MIN.into()..=i32::MAX.into())?,
                Q => int(i64::MIN.into()..=u64::MAX.into())?,
                N(bits) | C(bits) => int(0..=(1 << bits) - 1)?,
                Lanes => int(0..=if width == Some(W256) { 255 } else { 15 })?,
                Mv | M(_) => return Err(Fault::at(i, format!("wants memory; this is {}", given(a)))),
            }
        }
        Ok(())
    }

    /// The operands of `form` built from `args`, which fit it, for the instruction `name`.
    fn build(&mut self, form: Form, args: &[Given<Var>], name: &str) -> Result<Vec<Arg<VReg>>, Fault> {
        let width = Self::width(form, args);
        let mut out = vec![];
        for (i, (&(kind, access), a)) in form.iter().zip(args).enumerate() {
            let operand = match a {
                Given::Var(v) => {
                    let (reg, view) = v.reg().expect("stack memory is refused as an operand");
                    Operand::Reg(reg, view.width())
                }
                // Past 2^63 as the 64 bits are, as `mov` takes them.
                Given::Int(x) => Operand::Imm(Imm { value: *x as i64, hex: matches!(kind, C(_) | Lanes) }),
                Given::Mem(_) | Given::Data(_) | Given::Code(_) => {
                    let w = match kind {
                        Xm(w) | M(w) => w,
                        _ => width.expect("a form with vector memory has a vector"),
                    };
                    Operand::Mem(self.memory(a, w, access, name).map_err(|m| Fault::at(i, m))?)
                }
                _ => unreachable!("the form fits"),
            };
            out.push(Arg { operand, access });
        }
        Ok(out)
    }

    /// The address `a`, where the instruction `name` reaches `width` with `access`.
    fn memory(&mut self, a: &Given<Var>, width: Width, access: Access, name: &str) -> Result<Mem<VReg>, String> {
        let m = match a {
            Given::Data(bytes) => {
                if width.bytes() as usize > bytes.len() {
                    return Err(format!("this reads {} bytes of a constant of {}", width.bytes(), bytes.len()));
                }
                return Ok(Mem { base: Base::Const(self.constant(bytes)), index: None, disp: 0, width });
            }
            Given::Code(f) => {
                self.called(f, 0).map_err(|e| e.message)?;
                return Ok(Mem { base: Base::Func(self.func(f)), index: None, disp: 0, width });
            }
            Given::Mem(m) => m,
            _ => unreachable!("memory"),
        };
        // Memory stands for its address, which comes first, added as it is.
        let (mut base, mut index) = (m.base, m.index);
        if let (Some(b), Some((i, 1))) = (base, index)
            && b.reg().is_some()
            && i.reg().is_none()
        {
            (base, index) = (Some(i), Some((b, 1)));
        }
        if let Some((i, k)) = index
            && let Some((held, _)) = self.declared(i)
        {
            return Err(match base.and_then(|b| self.declared(b)) {
                Some((other, _)) => format!("an address names one memory at most; `{other}` and `{held}` are two"),
                None => {
                    let added = if matches!(i.0, Place::Static(_)) { "8" } else { "i * 4" };
                    format!(
                        "an address adds memory as it is, as in `[{held} + {added}]`; this multiplies `{held}` by {k}"
                    )
                }
            });
        }
        let held = base.filter(|b| b.reg().is_none());
        if let (Some(Var(Place::Static(s))), Some((i, k))) = (held, index) {
            let (held, i) = (&self.statics[s.0 as usize].name, i.reg().expect("an index is a register").0);
            let i = &self.p.regs[i.id as usize].name;
            let scaled = if k == 1 { i.clone() } else { format!("{i} * {k}") };
            return Err(format!(
                "x86 reaches memory of the module's at a displacement from `rip`, with no register beside it: \
                 `lea(p, [{held}])` puts the address of `{held}` in `p`, for `[p + {scaled}]`"
            ));
        }
        let whole = |v: Var| match v.0 {
            Place::Reg(reg, View::R64) => Ok(reg),
            Place::Reg(_, view) => Err(format!("an address is made of gprs seen whole; this is {}", view.name())),
            Place::Stack(_) | Place::Static(_) => unreachable!("memory is the base"),
        };
        let base = if held.is_some() { None } else { base.map(whole).transpose()? };
        let index = match index {
            Some((v, k)) => match u8::try_from(k) {
                Ok(scale @ (1 | 2 | 4 | 8)) => Some((whole(v)?, scale)),
                _ => return Err(format!("x86 scales an index by 1, 2, 4 or 8, not {k}")),
            },
            None => None,
        };
        match held.map(|v| v.0) {
            Some(Place::Stack(s)) => {
                let slot = &self.p.slots[s.0 as usize];
                let ty = &self.declared[s.0 as usize];
                let memory = Region { name: &slot.name, bytes: slot.size.into(), align: slot.align.into(), ty };
                self.within(memory, index, m.disp, width, access, name)?;
                let disp = i32::try_from(m.disp).expect("inside a frame of 2^31 bytes at most");
                return Ok(Mem { base: Base::Slot(s), index, disp, width });
            }
            Some(Place::Static(s)) => {
                let st = &self.statics[s.0 as usize];
                if st.constant && name != "lea" && matches!(access, Write | ReadWrite) {
                    return Err(format!(
                        "`{name}` {} `{}`, which is constant memory: the kernels only read it",
                        verb(access),
                        st.name
                    ));
                }
                let memory = Region { name: &st.name, bytes: st.bytes(), align: st.align, ty: &st.ty };
                self.within(memory, None, m.disp, width, access, name)?;
                let disp = i32::try_from(m.disp).expect("inside a static of 2^31 bytes at most");
                return Ok(Mem { base: Base::Static(s), index: None, disp, width });
            }
            _ => {}
        }
        let disp = i32::try_from(m.disp).map_err(|_| format!("{} bytes are too far for an address", m.disp))?;
        let (base, index) = match (base, index) {
            (Some(b), index) => (b, index),
            (None, Some((r, 1))) => (r, None),
            (None, _) => return Err("x86 wants a base beside a scaled index".into()),
        };
        Ok(Mem { base: Base::Reg(base), index, disp, width })
    }

    /// Whether the instruction `name` stays inside `memory`, of the stack or of the module's,
    /// reaching `width` with `access` at `disp` bytes into it and `index` past that; else why
    /// not. The address is checked where the index is 0, the rest being the kernel's to keep
    /// inside: it lies in the memory, or as far as one past its end for `lea`, which reaches
    /// nothing; and for `vmovaps` and `vmovdqa`, which fault elsewhere, at a multiple of the
    /// bytes they move.
    fn within(
        &self,
        memory: Region,
        index: Option<(VReg, u8)>,
        disp: i64,
        width: Width,
        access: Access,
        name: &str,
    ) -> Result<(), String> {
        let (v, n, ty) = (memory.name, memory.bytes, memory.ty);
        let mut at = format!("[{v}");
        let mut when = String::new();
        if let Some((i, k)) = index {
            let i = &self.p.regs[i.id as usize].name;
            at += &if k == 1 { format!(" + {i}") } else { format!(" + {i} * {k}") };
            when = format!(", when `{i}` is 0");
        }
        match disp {
            0 => {}
            d if d < 0 => at += &format!(" - {}", d.unsigned_abs()),
            d => at += &format!(" + {d}"),
        }
        at.push(']');
        if name == "lea" {
            return match disp {
                d if d < 0 => Err(format!("`lea` takes the address `{at}`, before `{v}` starts{when}")),
                d if d.unsigned_abs() > n => Err(format!(
                    "`lea` takes the address `{at}`, more than one past the end of `{v}`, {}{when}",
                    bytes(n)
                )),
                _ => Ok(()),
            };
        }
        let verb = verb(access);
        let size = u64::from(width.bytes());
        if disp < 0 {
            return Err(format!("`{name}` {verb} at `{at}`, before `{v}` starts{when}"));
        }
        if disp.unsigned_abs() + size > n {
            return Err(format!(
                "`{name}` {verb} {} at `{at}`, past the end of `{v}`, {}{when}",
                bytes(size),
                bytes(n)
            ));
        }
        let anywhere = match name {
            "vmovaps" => "vmovups",
            "vmovdqa" => "vmovdqu",
            _ => return Ok(()),
        };
        if memory.align < size {
            let stem = ty.strip_suffix('>').unwrap_or(ty);
            let stem = match stem.rsplit_once(", ") {
                Some((stem, align)) if align.parse::<u64>().is_ok() => stem,
                _ => stem,
            };
            return Err(format!(
                "`{name}` {verb} {size} bytes at once, at an address that is a multiple of {size}; `{v}` is at a \
                 multiple of {}: `{stem}, {size}>`, or `{anywhere}` at any",
                memory.align
            ));
        }
        if !disp.unsigned_abs().is_multiple_of(size) {
            return Err(format!(
                "`{name}` {verb} {size} bytes at once, at an address that is a multiple of {size}; `{at}` is not \
                 one{when}"
            ));
        }
        Ok(())
    }
}

/// Memory a kernel names, of its stack or of the module's, as [`Kernel::within`] checks an
/// address in it.
struct Region<'a> {
    /// What the source calls it.
    name: &'a str,
    /// How many bytes it is.
    bytes: u64,
    /// What its address is a multiple of.
    align: u64,
    /// Its type, as declared: `local<[f32; 64], 32>`.
    ty: &'a str,
}

/// What an instruction does to memory it reaches with `access`, in words.
fn verb(access: Access) -> &'static str {
    match access {
        Write => "writes",
        ReadWrite => "reads and writes",
        Read | Ignored => "reads",
    }
}

/// What memory of the type `ty`, `word<args>`, holds, stack memory `local<..>` or the module's,
/// `global<..>` or `constant<..>`: its elements' type and their bytes; how many, `None` for an
/// array of no length, `[f32]`; and what its address is a multiple of, a power of two up to a
/// page, its element's bytes unless it says more: `local<[f32; 64], 32>`.
fn shape(ty: &str, word: &str) -> Result<(&'static str, u64, Option<u64>, u64), String> {
    let wrong = || {
        format!(
            "`{word}<..>` is an array of Rust's numbers and what its address is a multiple of, \
             `{word}<[f32; 64], 32>`, or one, `{word}<u64>`; not `{ty}`"
        )
    };
    // What the memory is, in words.
    let what = if word == "local" { "stack memory" } else { "a static" };
    let args = ty.strip_prefix(word).and_then(|t| t.strip_prefix('<')).and_then(|t| t.strip_suffix('>'));
    let args = args.ok_or_else(wrong)?;
    let (array, align) = match args.rsplit_once(", ") {
        Some((array, align)) => (array, Some(align)),
        None => (args, None),
    };
    if array.contains(", ") {
        return Err(wrong());
    }
    let (elem, len) = match array.strip_prefix('[').and_then(|a| a.strip_suffix(']')) {
        Some(inner) => match inner.split_once("; ") {
            Some((elem, len)) => (elem, Some(len.parse::<u64>().map_err(|_| wrong())?)),
            None => (inner, None),
        },
        None => (array, Some(1)),
    };
    let Some(&(elem, size)) = HELD.iter().find(|(e, _)| *e == elem) else {
        return Err(format!(
            "{what} holds Rust's numbers, `u8` to `u64`, `i8` to `i64`, `usize`, `isize`, `f32` and `f64`; \
             not `{elem}`"
        ));
    };
    let align = match align {
        None => size,
        Some(a) => match a.parse::<u64>() {
            Ok(a) if a.is_power_of_two() && a < size => {
                return Err(format!("an array of `{elem}` is aligned to {size} at least, not {a}"));
            }
            Ok(a) if a.is_power_of_two() && a > u64::from(PAGE) => {
                return Err(format!("{what} is aligned to a page at most, {PAGE}; not {a}"));
            }
            Ok(a) if a.is_power_of_two() => a,
            _ => return Err(format!("what an address is a multiple of is a power of two, not {a}")),
        },
    };
    if len == Some(0) {
        return Err(format!("`{ty}` holds nothing"));
    }
    Ok((elem, size, len, align))
}

fn view_of(w: Width) -> View {
    if w == W256 { View::Ymm } else { View::Xmm }
}

fn describe(kind: Kind, width: Option<Width>) -> String {
    let vector = match width {
        Some(W256) => "a ymm",
        Some(_) => "an xmm",
        None => "a vector",
    };
    match kind {
        V => vector.to_string(),
        Vm => format!("{vector} or memory"),
        Mv => "memory".into(),
        Y => "a ymm".into(),
        X => "an xmm".into(),
        Xm(w) => format!("an xmm or {} bytes of memory", w.bytes()),
        M(w) => format!("{} bytes of memory", w.bytes()),
        G(W32) => "a gpr's r32".into(),
        G(_) => "a gpr".into(),
        I => "an integer of 32 bits".into(),
        Q => "an integer".into(),
        N(bits) => format!("an integer, 0 to {}", (1 << bits) - 1),
        C(bits) => format!("a control byte, 0 to {}", (1 << bits) - 1),
        Lanes => match width {
            Some(W256) => "a control byte, 0 to 255".into(),
            Some(_) => "a control byte, 0 to 15".into(),
            None => "a control byte of a bit for each lane".into(),
        },
    }
}

fn given(a: &Given<Var>) -> String {
    match a {
        Given::Var(v) => match v.0 {
            Place::Reg(_, view) => view.name().into(),
            Place::Stack(_) => "stack memory".into(),
            Place::Static(_) => "memory of the module's".into(),
        },
        Given::Special(name) => format!("`{name}`"),
        Given::Int(x) => format!("the integer {x}"),
        Given::F32(x) => format!("the float {x}"),
        Given::F64(x) => format!("the float {x}"),
        Given::F16(x) => format!("the f16 {}", Value::F16(*x)),
        Given::Bf16(x) => format!("the bf16 {}", Value::Bf16(*x)),
        Given::Mem(_) => "memory".into(),
        Given::Data(_) => "memory holding data".into(),
        Given::Label(_) => "a label".into(),
        Given::Vector(vs) => format!("{} operands taken together", vs.len()),
        Given::Pair(..) => "two operands joined by `|`".into(),
        Given::Not(_) => "an operand inverted".into(),
        Given::Neg(_) => "an operand negated".into(),
        Given::List(_) => "operands in brackets".into(),
        Given::Func(f) => format!("the function `{f}`"),
        Given::Code(f) => format!("the code of `{f}`"),
    }
}

impl Target for Kernel {
    type Var = Var;

    fn param(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        let (class, at, view) = if Ty::param(ty)? == Ty::Int {
            if self.cooperative && self.ints == INT_PARAMS.len() - 1 {
                return Err("a cooperative kernel takes five integers at most: SysV passes six in registers, and the \
                            sixth is the thread's place in the grid"
                    .into());
            }
            let at = *INT_PARAMS.get(self.ints).ok_or(SEVENTH)?;
            self.ints += 1;
            (Class::Gpr, at, View::R64)
        } else {
            let at = *FLOAT_PARAMS.get(self.floats).ok_or(NINTH)?;
            self.floats += 1;
            (Class::Vec, at, View::Xmm)
        };
        let reg = self.new_var(class, name);
        self.p.params.push((reg, at));
        self.p.signature.params.push(class);
        Ok(Var(Place::Reg(reg, view)))
    }

    fn result(&mut self, ty: &str) -> Result<(), String> {
        let returns = Returns::of(ty)?;
        if self.cooperative && returns != Returns::Nothing {
            return Err("a cooperative kernel's threads return nothing: they write what they compute to memory".into());
        }
        if let Returns::Value(t) = returns {
            self.p.signature.result = Some(t.class());
        }
        self.returns = returns;
        Ok(())
    }

    fn var(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        if ty.starts_with("local<") {
            return self.new_memory(name, ty);
        }
        if let Some(rest) = ty.strip_prefix("shared<") {
            return Err(format!("`shared<..>` is a GPU's memory; an x86 kernel has stack memory, `local<{rest}`"));
        }
        if ty.starts_with("global<") || ty.starts_with("constant<") {
            return Err(format!(
                "global and constant memory are the module's, which a static declares, as `static {name}: {ty};` \
                 beside the kernels: a kernel declares stack memory, `local<..>`"
            ));
        }
        let (class, view) = match ty {
            "gpr" => (Class::Gpr, View::R64),
            "ymm" => (Class::Vec, View::Ymm),
            "xmm" => (Class::Vec, View::Xmm),
            _ => {
                return Err(format!("x86 variables are `gpr`, `ymm`, `xmm` and stack memory, `local<..>`; not `{ty}`"));
            }
        };
        Ok(Var(Place::Reg(self.new_var(class, name), view)))
    }

    fn global(&mut self, name: &str) -> Result<Var, String> {
        let Some(s) = self.module.iter().find(|s| s.name == name) else {
            return Err(format!("`{name}` does not compile, so no kernel names it"));
        };
        let at = match self.p.statics.iter().position(|n| n == name) {
            Some(at) => at,
            None => {
                self.statics.push(s.clone());
                self.p.statics.push(name.to_string());
                self.p.statics.len() - 1
            }
        };
        Ok(Var(Place::Static(StaticId(at as u32))))
    }

    fn view(&mut self, v: Var, view: &str) -> Result<Var, String> {
        // As a helper's parameter, memory is the memory it was declared.
        if let Some((name, ty)) = self.declared(v) {
            return match view == ty {
                true => Ok(v),
                false => Err(format!("`{name}` is memory, `{ty}`; it is no `{view}`")),
            };
        }
        let Place::Reg(reg, from) = v.0 else { unreachable!("a variable that is no memory is a register") };
        let to = match (reg.class, view) {
            (Class::Vec, "ymm") => View::Ymm,
            (Class::Vec, "xmm") => View::Xmm,
            (Class::Vec, "vec") => from,
            (Class::Gpr, "gpr") => View::R64,
            (Class::Gpr, "r32") => View::R32,
            _ if ["local<", "global<", "constant<"].iter().any(|w| view.starts_with(w)) => {
                return Err(format!("{} is no memory, `{view}`", from.name()));
            }
            (Class::Vec, _) => return Err(format!("a vector is seen as `ymm`, `xmm` or `vec`, not `{view}`")),
            _ => return Err(format!("a gpr is seen as `gpr` or `r32`, not `{view}`")),
        };
        Ok(Var(Place::Reg(reg, to)))
    }

    fn label(&mut self, label: u32) {
        self.p.body.push(Item::Label(Label(label)));
    }

    fn note(&mut self, text: &str) {
        self.p.body.push(Item::Note(text.to_string()));
    }

    fn inst(&mut self, name: &str, args: &[Given<Var>], guard: Option<Guard<Var>>) -> Result<(), Fault> {
        if name == "grid" || name.starts_with("grid.") {
            return self.sync(name, args, guard.is_some());
        }
        if guard.is_some() {
            return Err(Fault::new("x86 runs an instruction whatever a predicate says; jump around it"));
        }
        // The grid's values are memory, which an instruction reads, by its name as written.
        let mut args = args.to_vec();
        let mut grid = vec![];
        for (i, a) in args.iter_mut().enumerate() {
            let Given::Special(s) = a else { continue };
            if let Some(m) = self.grid_value(s).map_err(|m| Fault::at(i, m))? {
                grid.push((i, std::mem::take(s)));
                *a = m;
            }
        }
        let args = args.as_slice();
        if name == "call" {
            return self.call(args);
        }
        for (i, a) in args.iter().enumerate() {
            match a {
                Given::Var(v) if v.reg().is_none() => {
                    let (held, _) = self.declared(*v).expect("a variable that is no register is memory");
                    return Err(Fault::at(
                        i,
                        format!(
                            "`{name}` takes no memory as an operand: `lea(p, [{held}])` puts the address of `{held}` \
                             in `p`, and `[{held}]` is what it holds"
                        ),
                    ));
                }
                Given::Special(name) => return Err(Fault::at(i, format!("`{name}` is not defined"))),
                Given::F32(x) => return Err(Fault::at(i, format!("x86 takes floats from memory: `[{x:?}]`"))),
                Given::F64(x) => return Err(Fault::at(i, format!("x86 takes floats from memory: `[{x:?}f64]`"))),
                Given::F16(x) => {
                    return Err(Fault::at(i, format!("x86 takes floats from memory: `[{}f16]`", Value::F16(*x))));
                }
                Given::Bf16(x) => {
                    return Err(Fault::at(i, format!("x86 takes floats from memory: `[{}bf16]`", Value::Bf16(*x))));
                }
                Given::Vector(vs) => {
                    return Err(Fault::at(
                        i,
                        format!(
                            "x86 takes a variable here, and these are {} together: name one, as in `a[0]`",
                            vs.len()
                        ),
                    ));
                }
                Given::Pair(..) => return Err(Fault::at(i, "x86 has no instruction that writes `a | b`, two at once")),
                Given::Not(_) | Given::Neg(_) => {
                    return Err(Fault::at(i, "x86 reads no operand inverted or negated; `!p` and `-a` are PTX's"));
                }
                Given::List(_) => return Err(Fault::at(i, "memory is one address in brackets, as in `[x + 4]`")),
                Given::Label(_) if !JUMPS.iter().any(|(n, _)| *n == name) => {
                    return Err(Fault::at(i, "a label is for a jump"));
                }
                Given::Func(f) => {
                    return Err(Fault::at(
                        i,
                        format!(
                            "`{name}` takes no function; `call(r, {f}, (a, b))` calls `{f}`, and `lea(p, [{f}])` puts \
                             its address in `p`"
                        ),
                    ));
                }
                Given::Code(f) if name != "lea" => {
                    return Err(Fault::at(
                        i,
                        format!("`{name}` takes no function's code: `lea(p, [{f}])` puts the address of `{f}` in `p`"),
                    ));
                }
                _ => {}
            }
        }
        match name {
            "ret" => return self.ret(args),
            "ud2" if args.is_empty() => {
                self.push(Op::Ud2, vec![], vec![], Flow::Return);
                return Ok(());
            }
            "ud2" => return Err(Fault::new(format!("`ud2` takes no operands, not {}", args.len()))),
            "zero" => return self.fill(false, args),
            "ones" => return self.fill(true, args),
            _ => {}
        }
        if let Some(&(_, op)) = JUMPS.iter().find(|(n, _)| *n == name) {
            let [Given::Label(l)] = args else {
                return Err(Fault::new(format!("`{name}` takes a label")));
            };
            let flow = if op == Op::Jmp { Flow::Jump(Label(*l)) } else { Flow::Branch(Label(*l)) };
            self.push(op, vec![Arg { operand: Operand::Label(Label(*l)), access: Read }], vec![], flow);
            return Ok(());
        }
        let Some((group, op)) =
            GROUPS.iter().find_map(|g| g.ops.iter().find(|(n, _)| *n == name).map(|&(_, op)| (g, op)))
        else {
            return Err(Fault::new(format!("x86 has no instruction `{name}` that kgen knows")));
        };
        // The form that fits, or the complaint of the one that fit furthest.
        let mut best: Option<Fault> = None;
        for &form in group.forms {
            match Self::fits(form, args) {
                Ok(()) => {
                    let built = self.build(form, args, name)?;
                    for (i, s) in &grid {
                        let wrong = if op == Op::Lea {
                            "`lea` takes the address of the kernel's memory"
                        } else if built[*i].access != Read {
                            "the launcher writes it"
                        } else {
                            continue;
                        };
                        return Err(Fault::at(*i, format!("`{s}` is read, as in `mov(t, {s})`; {wrong}")));
                    }
                    self.push(op, built, vec![], Flow::Next);
                    return Ok(());
                }
                Err(f) => {
                    if best.as_ref().is_none_or(|b| f.arg > b.arg) {
                        best = Some(f);
                    }
                }
            }
        }
        let mut f = best.expect("a group has forms");
        f.message = format!("`{name}` {}", f.message);
        Err(f)
    }

    fn end(&mut self) -> Result<(), String> {
        let open = match self.p.body.iter().rev().find(|item| !matches!(item, Item::Note(_))) {
            Some(Item::Inst(inst)) => !matches!(inst.flow, Flow::Return | Flow::Jump(_)),
            _ => true,
        };
        match (open, self.returns) {
            (false, _) => Ok(()),
            (true, Returns::Nothing) => {
                self.push(Op::Ret, vec![], vec![], Flow::Return);
                Ok(())
            }
            (true, Returns::Value(_)) => Err("control reaches the end of the body, and the kernel returns a value; \
                                              end each way out with `ret(v)`"
                .into()),
            (true, Returns::Never) => Err(format!(
                "control reaches the end of the body, and `{}` never returns, `-> !`: {NEVER}",
                self.p.name
            )),
        }
    }
}

/// `mov(to.r32, from.r32)`, which clears the rest of `to`.
fn move_r32(to: VReg, from: VReg) -> Inst<Op, VReg> {
    let args = vec![
        Arg { operand: Operand::Reg(to, W32), access: Write },
        Arg { operand: Operand::Reg(from, W32), access: Read },
    ];
    Inst { op: Op::Mov, args, implicit: vec![], flow: Flow::Next, clobbers: vec![] }
}
