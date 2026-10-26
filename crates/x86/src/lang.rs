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

use kgen_core::Access::{self, Ignored, Read, ReadWrite, Write};
use kgen_core::Width::{self, W8, W16, W32, W64, W128, W256};
use kgen_core::{
    Arg, Base, Class, Const, ConstId, Flow, Imm, Inst, Item, Label, Mem, Operand, Program, RegInfo, Signature, Slot,
    SlotId, VReg,
};
use kgen_lang::{Fault, Guard, Operand as Given, Target, Value};

use crate::bytes;
use crate::op::Op;
use crate::sysv::{FLOAT_PARAMS, FLOAT_RESULT, INT_PARAMS, INT_RESULT, PAGE};

/// The most bytes of stack memory a kernel has: as far past the stack pointer as an address
/// reaches, whose displacement is 32 bits, signed.
const FRAME: u64 = i32::MAX as u64;

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

/// A variable and the view the language has of it, or memory of the kernel's stack, which
/// stands for its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(Place);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// A register, seen as the view says.
    Reg(VReg, View),
    /// Memory of the stack frame, by its slot.
    Stack(SlotId),
}

impl Var {
    /// Its register and the view of it, unless it is memory.
    fn reg(self) -> Option<(VReg, View)> {
        match self.0 {
            Place::Reg(reg, view) => Some((reg, view)),
            Place::Stack(_) => None,
        }
    }

    /// Its slot, if it is memory.
    fn slot(self) -> Option<SlotId> {
        match self.0 {
            Place::Reg(..) => None,
            Place::Stack(s) => Some(s),
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
    groups.chain(JUMPS.iter().map(|(name, _)| *name)).chain(["ret", "zero", "ones"])
}

/// A kernel being written in the language, for x86-64 under SysV.
pub struct Kernel {
    p: Program<Op>,
    ints: usize,
    floats: usize,
    result: Option<Class>,
    /// Each slot's type as declared, `local<[f32; 64], 32>`, by [`SlotId`].
    declared: Vec<String>,
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
                body: vec![],
                slots: vec![],
                consts: vec![],
            },
            ints: 0,
            floats: 0,
            result: None,
            declared: vec![],
            cooperative: false,
            grid: None,
            labels: u32::MAX,
        }
    }

    /// Makes the kernel one that runs on the threads of a grid, `#[cooperative]`, before its
    /// parameters are declared: each thread is given its place in the grid after them, and
    /// may wait for the others at `grid.sync()`.
    pub fn cooperative(&mut self) {
        self.cooperative = true;
    }

    /// Stack memory declared as `let name: ty`, `ty` being `local<[f32; 64], 32>`.
    fn new_memory(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        let (elem, size, len, align) = shape(ty)?;
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
        self.p.body.push(Item::Inst(Inst { op, args, implicit, flow }));
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
        let (reg, width, at) = match (self.result, args, given) {
            (None, [], _) => {
                self.push(Op::Ret, vec![], vec![], Flow::Return);
                return Ok(());
            }
            (Some(Class::Vec), _, Some(r)) if r.class == Class::Vec => (r, W128, FLOAT_RESULT),
            (Some(Class::Gpr), _, Some(r)) if r.class == Class::Gpr => (r, W64, INT_RESULT),
            (None, ..) => return Err(Fault::new("this kernel returns nothing: `ret()`")),
            (Some(Class::Vec), ..) => return Err(Fault::new("this kernel returns a float: `ret(v)`, `v` a vector")),
            (Some(_), ..) => return Err(Fault::new("this kernel returns an integer: `ret(g)`, `g` a gpr")),
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
            let memory = matches!(a, Given::Mem(_) | Given::Data(_));
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
                Given::Mem(_) | Given::Data(_) => {
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
            Given::Mem(m) => m,
            _ => unreachable!("memory"),
        };
        // Stack memory stands for its address, which comes first, added as it is.
        let (mut base, mut index) = (m.base, m.index);
        if let (Some(b), Some((i, 1))) = (base, index)
            && b.slot().is_none()
            && i.slot().is_some()
        {
            (base, index) = (Some(i), Some((b, 1)));
        }
        if let Some((i, k)) = index
            && let Some(s) = i.slot()
        {
            let held = &self.p.slots[s.0 as usize].name;
            return Err(match base.and_then(Var::slot) {
                Some(b) => {
                    format!(
                        "an address names one memory at most; `{}` and `{held}` are two",
                        self.p.slots[b.0 as usize].name
                    )
                }
                None => format!(
                    "an address adds memory as it is, as in `[{held} + i * 4]`; this multiplies `{held}` by {k}"
                ),
            });
        }
        let whole = |v: Var| match v.0 {
            Place::Reg(reg, View::R64) => Ok(reg),
            Place::Reg(_, view) => Err(format!("an address is made of gprs seen whole; this is {}", view.name())),
            Place::Stack(_) => unreachable!("stack memory is the base"),
        };
        let slot = base.and_then(Var::slot);
        let base = if slot.is_some() { None } else { base.map(whole).transpose()? };
        let index = match index {
            Some((v, k)) => match u8::try_from(k) {
                Ok(scale @ (1 | 2 | 4 | 8)) => Some((whole(v)?, scale)),
                _ => return Err(format!("x86 scales an index by 1, 2, 4 or 8, not {k}")),
            },
            None => None,
        };
        if let Some(s) = slot {
            self.within(s, index, m.disp, width, access, name)?;
            let disp = i32::try_from(m.disp).expect("inside a frame of 2^31 bytes at most");
            return Ok(Mem { base: Base::Slot(s), index, disp, width });
        }
        let disp = i32::try_from(m.disp).map_err(|_| format!("{} bytes are too far for an address", m.disp))?;
        let (base, index) = match (base, index) {
            (Some(b), index) => (b, index),
            (None, Some((r, 1))) => (r, None),
            (None, _) => return Err("x86 wants a base beside a scaled index".into()),
        };
        Ok(Mem { base: Base::Reg(base), index, disp, width })
    }

    /// Whether the instruction `name` stays inside stack memory `s`, reaching `width` with
    /// `access` at `disp` bytes into it and `index` past that; else why not. The address is
    /// checked where the index is 0, the rest being the kernel's to keep inside: it lies in the
    /// memory, or as far as one past its end for `lea`, which reaches nothing; and for
    /// `vmovaps` and `vmovdqa`, which fault elsewhere, at a multiple of the bytes they move.
    fn within(
        &self,
        s: SlotId,
        index: Option<(VReg, u8)>,
        disp: i64,
        width: Width,
        access: Access,
        name: &str,
    ) -> Result<(), String> {
        let (slot, ty) = (&self.p.slots[s.0 as usize], &self.declared[s.0 as usize]);
        let v = &slot.name;
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
        let n = u64::from(slot.size);
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
        let verb = match access {
            Write => "writes",
            ReadWrite => "reads and writes",
            Read | Ignored => "reads",
        };
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
        if u64::from(slot.align) < size {
            let stem = ty.strip_suffix('>').unwrap_or(ty);
            let stem = match stem.rsplit_once(", ") {
                Some((stem, align)) if align.parse::<u64>().is_ok() => stem,
                _ => stem,
            };
            return Err(format!(
                "`{name}` {verb} {size} bytes at once, at an address that is a multiple of {size}; `{v}` is at a \
                 multiple of {}: `{stem}, {size}>`, or `{anywhere}` at any",
                slot.align
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

/// What stack memory of the type `ty`, `local<args>`, holds: its elements' type and their
/// bytes; how many, `None` for an array of no length, `[f32]`; and what its address is a
/// multiple of, a power of two up to a page, its element's bytes unless it says more:
/// `local<[f32; 64], 32>`.
fn shape(ty: &str) -> Result<(&'static str, u64, Option<u64>, u64), String> {
    let wrong = || {
        format!(
            "`local<..>` is an array of Rust's numbers and what its address is a multiple of, \
             `local<[f32; 64], 32>`, or one, `local<u64>`; not `{ty}`"
        )
    };
    let args = ty.strip_prefix("local<").and_then(|t| t.strip_suffix('>')).ok_or_else(wrong)?;
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
            "stack memory holds Rust's numbers, `u8` to `u64`, `i8` to `i64`, `usize`, `isize`, `f32` and \
             `f64`; not `{elem}`"
        ));
    };
    let align = match align {
        None => size,
        Some(a) => match a.parse::<u64>() {
            Ok(a) if a.is_power_of_two() && a < size => {
                return Err(format!("an array of `{elem}` is aligned to {size} at least, not {a}"));
            }
            Ok(a) if a.is_power_of_two() && a > u64::from(PAGE) => {
                return Err(format!("stack memory is aligned to a page at most, {PAGE}; not {a}"));
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
        Given::Var(v) => match v.reg() {
            Some((_, view)) => view.name().into(),
            None => "stack memory".into(),
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
    }
}

impl Target for Kernel {
    type Var = Var;

    fn param(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        let (class, at, view) = if ty == "f32" || ty == "f64" {
            let at = *FLOAT_PARAMS
                .get(self.floats)
                .ok_or("SysV passes a ninth float on the stack, which kgen does not read")?;
            self.floats += 1;
            (Class::Vec, at, View::Xmm)
        } else if ty.starts_with("*const ") || ty.starts_with("*mut ") || ["usize", "isize", "u64", "i64"].contains(&ty)
        {
            if self.cooperative && self.ints == INT_PARAMS.len() - 1 {
                return Err("a cooperative kernel takes five integers at most: SysV passes six in registers, and the \
                            sixth is the thread's place in the grid"
                    .into());
            }
            let at = *INT_PARAMS
                .get(self.ints)
                .ok_or("SysV passes a seventh integer on the stack, which kgen does not read")?;
            self.ints += 1;
            (Class::Gpr, at, View::R64)
        } else if ["u32", "i32", "u16", "i16", "u8", "i8", "bool"].contains(&ty) {
            return Err(format!("SysV leaves the high bits of a `{ty}` as they were; take a usize or a u64"));
        } else {
            return Err(format!("a kernel takes pointers, 64-bit integers and floats, not `{ty}`"));
        };
        let reg = self.new_var(class, name);
        self.p.params.push((reg, at));
        self.p.signature.params.push(class);
        Ok(Var(Place::Reg(reg, view)))
    }

    fn result(&mut self, ty: &str) -> Result<(), String> {
        let ints = ["usize", "isize", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8", "bool"];
        let class = match ty {
            "()" => return Ok(()),
            _ if self.cooperative => {
                return Err(
                    "a cooperative kernel's threads return nothing: they write what they compute to memory".into()
                );
            }
            "f32" | "f64" => Class::Vec,
            _ if ty.starts_with("*const ") || ty.starts_with("*mut ") || ints.contains(&ty) => Class::Gpr,
            _ => return Err(format!("a kernel returns a float, an integer or a pointer, not `{ty}`")),
        };
        self.result = Some(class);
        self.p.signature.result = Some(class);
        Ok(())
    }

    fn var(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        if ty.starts_with("local<") {
            return self.new_memory(name, ty);
        }
        for word in ["shared", "global", "constant"] {
            if let Some(rest) = ty.strip_prefix(word).and_then(|t| t.strip_prefix('<')) {
                return Err(format!("`{word}<..>` is a GPU's memory; an x86 kernel has stack memory, `local<{rest}`"));
            }
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

    fn view(&mut self, v: Var, view: &str) -> Result<Var, String> {
        let (reg, from) = match v.0 {
            Place::Reg(reg, from) => (reg, from),
            // As a helper's parameter, it is the memory it was declared.
            Place::Stack(s) => {
                let (name, ty) = (&self.p.slots[s.0 as usize].name, &self.declared[s.0 as usize]);
                return match view == ty {
                    true => Ok(v),
                    false => Err(format!("`{name}` is memory, `{ty}`; it is no `{view}`")),
                };
            }
        };
        let to = match (reg.class, view) {
            (Class::Vec, "ymm") => View::Ymm,
            (Class::Vec, "xmm") => View::Xmm,
            (Class::Vec, "vec") => from,
            (Class::Gpr, "gpr") => View::R64,
            (Class::Gpr, "r32") => View::R32,
            _ if view.starts_with("local<") => return Err(format!("{} is no memory, `{view}`", from.name())),
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
        for (i, a) in args.iter().enumerate() {
            match a {
                Given::Var(Var(Place::Stack(s))) => {
                    let held = &self.p.slots[s.0 as usize].name;
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
                        format!("an x86 kernel calls no function; `{f}` names an ABI, so it is one"),
                    ));
                }
                _ => {}
            }
        }
        match name {
            "ret" => return self.ret(args),
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
        match (open, self.result) {
            (false, _) => Ok(()),
            (true, None) => {
                self.push(Op::Ret, vec![], vec![], Flow::Return);
                Ok(())
            }
            (true, Some(_)) => Err("control reaches the end of the body, and the kernel returns a value; \
                                    end each way out with `ret(v)`"
                .into()),
        }
    }
}
