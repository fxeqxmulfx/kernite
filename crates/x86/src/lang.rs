//! x86-64 as the language writes it: [`Kernel`] is the [`Target`] that checks each call
//! against the forms x86 has for its instruction, and builds the program the builder would.
//!
//! The types are `gpr`, `ymm` and `xmm`, and the views `v.ymm`, `v.xmm`, `g.r32`, whose
//! writes clear the rest of the register as the machine's do. A kernel's parameters arrive
//! as SysV passes them: pointers and 64-bit integers in general registers, `f32` and `f64`
//! in vector ones, counted apart. It hands back a float in `xmm0` or an integer in `rax` with
//! `ret(v)`, or nothing with `ret()`, which the end of the body adds if control reaches it.
//! Besides the instructions, `zero(v)` and `ones(v)` set a variable without reading it.

use kgen_core::Access::{self, Ignored, Read, ReadWrite, Write};
use kgen_core::Width::{self, W8, W16, W32, W64, W128, W256};
use kgen_core::{
    Arg, Base, Class, Const, ConstId, Flow, Imm, Inst, Item, Label, Mem, Operand, Program, RegInfo, Signature, VReg,
};
use kgen_lang::{Fault, Guard, Operand as Given, Target, Value};

use crate::op::Op;
use crate::sysv::{FLOAT_PARAMS, FLOAT_RESULT, INT_PARAMS, INT_RESULT};

/// A variable, and the view the language has of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var {
    reg: VReg,
    view: View,
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
    /// A control byte, 0 to 255, which a listing shows in hex.
    C,
}

use Kind::{C, G, I, M, Mv, Q, V, Vm, X, Xm, Y};

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
    Group {
        ops: &[("vcmpps", Op::Vcmpps), ("vshufps", Op::Vshufps), ("vpblendd", Op::Vpblendd)],
        forms: &[&[(V, Write), (V, Read), (Vm, Read), (C, Read)]],
    },
    Group { ops: &[("vroundps", Op::Vroundps)], forms: &[&[(V, Write), (Vm, Read), (C, Read)]] },
    Group { ops: &[("vblendvps", Op::Vblendvps)], forms: &[&[(V, Write), (V, Read), (Vm, Read), (V, Read)]] },
    Group {
        ops: &[
            ("vpslld", Op::Vpslld),
            ("vpsrld", Op::Vpsrld),
            ("vpsrad", Op::Vpsrad),
            ("vpsllw", Op::Vpsllw),
            ("vpsrlw", Op::Vpsrlw),
        ],
        forms: &[&[(V, Write), (V, Read), (I, Read)]],
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
    Group { ops: &[("vroundss", Op::Vroundss)], forms: &[&[(X, Write), (X, Read), (Xm(W32), Read), (C, Read)]] },
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
    Group { ops: &[("vextractf128", Op::Vextractf128)], forms: &[&[(X, Write), (Y, Read), (I, Read)]] },
    Group {
        ops: &[("vinsertf128", Op::Vinsertf128), ("vinserti128", Op::Vinserti128)],
        forms: &[&[(Y, Write), (Y, Read), (Xm(W128), Read), (I, Read)]],
    },
    Group { ops: &[("vpermps", Op::Vpermps)], forms: &[&[(Y, Write), (Y, Read), (Vm, Read)]] },
    Group { ops: &[("vpermq", Op::Vpermq)], forms: &[&[(Y, Write), (Vm, Read), (C, Read)]] },
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
    Group { ops: &[("shl", Op::Shl), ("shr", Op::Shr), ("sar", Op::Sar)], forms: &[&[(G(W64), ReadWrite), (I, Read)]] },
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
        }
    }

    /// The kernel as written.
    pub fn finish(self) -> Program<Op> {
        self.p
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
        let (reg, width, at) = match (self.result, args) {
            (None, []) => {
                self.push(Op::Ret, vec![], vec![], Flow::Return);
                return Ok(());
            }
            (Some(Class::Vec), [Given::Var(v)]) if v.reg.class == Class::Vec => (v.reg, W128, FLOAT_RESULT),
            (Some(Class::Gpr), [Given::Var(v)]) if v.reg.class == Class::Gpr => (v.reg, W64, INT_RESULT),
            (None, _) => return Err(Fault::new("this kernel returns nothing: `ret()`")),
            (Some(Class::Vec), _) => return Err(Fault::new("this kernel returns a float: `ret(v)`, `v` a vector")),
            (Some(_), _) => return Err(Fault::new("this kernel returns an integer: `ret(g)`, `g` a gpr")),
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
        let [Given::Var(v)] = args else {
            return Err(Fault::new(format!("`{name}` takes the variable it sets")));
        };
        let reg = |w: Width, access: Access| Arg { operand: Operand::Reg(v.reg, w), access };
        match (v.reg.class, ones) {
            // A VEX write to the low half clears the rest, so zeros cost the shorter encoding.
            (Class::Vec, false) => self.push(
                Op::Vxorps,
                vec![reg(W128, Write), reg(W128, Ignored), reg(W128, Ignored)],
                vec![],
                Flow::Next,
            ),
            (Class::Vec, true) => {
                let w = v.view.width();
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
        let given = form.iter().zip(args).find_map(|((kind, _), a)| match (kind, a) {
            (V | Vm, Given::Var(v)) if v.reg.class == Class::Vec => Some(v.view.width()),
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
            let vector = |want: Option<View>| match a {
                Given::Var(v) if v.reg.class == Class::Vec && want.is_none_or(|w| w == v.view) => Ok(()),
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
                G(w) => match a {
                    Given::Var(v) if v.reg.class == Class::Gpr && v.view.width() == w => {}
                    _ => return Err(Fault::at(i, format!("wants {}; this is {}", describe(kind, width), given(a)))),
                },
                I => int(i32::MIN.into()..=i32::MAX.into())?,
                Q => int(i64::MIN.into()..=u64::MAX.into())?,
                C => int(0..=255)?,
                Mv | M(_) => return Err(Fault::at(i, format!("wants memory; this is {}", given(a)))),
            }
        }
        Ok(())
    }

    /// The operands of `form` built from `args`, which fit it.
    fn build(&mut self, form: Form, args: &[Given<Var>]) -> Result<Vec<Arg<VReg>>, Fault> {
        let width = Self::width(form, args);
        let mut out = vec![];
        for (i, (&(kind, access), a)) in form.iter().zip(args).enumerate() {
            let operand = match a {
                Given::Var(v) => Operand::Reg(v.reg, v.view.width()),
                // Past 2^63 as the 64 bits are, as `mov` takes them.
                Given::Int(x) => Operand::Imm(Imm { value: *x as i64, hex: kind == C }),
                Given::Mem(_) | Given::Data(_) => {
                    let w = match kind {
                        Xm(w) | M(w) => w,
                        _ => width.expect("a form with vector memory has a vector"),
                    };
                    Operand::Mem(self.memory(a, w).map_err(|m| Fault::at(i, m))?)
                }
                _ => unreachable!("the form fits"),
            };
            out.push(Arg { operand, access });
        }
        Ok(out)
    }

    fn memory(&mut self, a: &Given<Var>, width: Width) -> Result<Mem<VReg>, String> {
        match a {
            Given::Data(bytes) => {
                if width.bytes() as usize > bytes.len() {
                    return Err(format!("this reads {} bytes of a constant of {}", width.bytes(), bytes.len()));
                }
                Ok(Mem { base: Base::Const(self.constant(bytes)), index: None, disp: 0, width })
            }
            Given::Mem(m) => {
                let whole = |v: &Var| match v.view {
                    View::R64 => Ok(v.reg),
                    _ => Err(format!("an address is made of gprs seen whole; this is {}", v.view.name())),
                };
                let base = m.base.as_ref().map(whole).transpose()?;
                let index = match m.index {
                    Some((v, k)) => match u8::try_from(k) {
                        Ok(scale @ (1 | 2 | 4 | 8)) => Some((whole(&v)?, scale)),
                        _ => return Err(format!("x86 scales an index by 1, 2, 4 or 8, not {k}")),
                    },
                    None => None,
                };
                let disp = i32::try_from(m.disp).map_err(|_| format!("{} bytes are too far for an address", m.disp))?;
                let (base, index) = match (base, index) {
                    (Some(b), index) => (b, index),
                    (None, Some((r, 1))) => (r, None),
                    (None, _) => return Err("x86 wants a base beside a scaled index".into()),
                };
                Ok(Mem { base: Base::Reg(base), index, disp, width })
            }
            _ => unreachable!("memory"),
        }
    }
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
        C => "a control byte, 0 to 255".into(),
    }
}

fn given(a: &Given<Var>) -> String {
    match a {
        Given::Var(v) => v.view.name().into(),
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
        Ok(Var { reg, view })
    }

    fn result(&mut self, ty: &str) -> Result<(), String> {
        let ints = ["usize", "isize", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8", "bool"];
        let class = match ty {
            "()" => return Ok(()),
            "f32" | "f64" => Class::Vec,
            _ if ty.starts_with("*const ") || ty.starts_with("*mut ") || ints.contains(&ty) => Class::Gpr,
            _ => return Err(format!("a kernel returns a float, an integer or a pointer, not `{ty}`")),
        };
        self.result = Some(class);
        self.p.signature.result = Some(class);
        Ok(())
    }

    fn var(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        let (class, view) = match ty {
            "gpr" => (Class::Gpr, View::R64),
            "ymm" => (Class::Vec, View::Ymm),
            "xmm" => (Class::Vec, View::Xmm),
            _ => return Err(format!("x86 variables are `gpr`, `ymm` and `xmm`, not `{ty}`")),
        };
        Ok(Var { reg: self.new_var(class, name), view })
    }

    fn view(&mut self, v: Var, view: &str) -> Result<Var, String> {
        let to = match (v.reg.class, view) {
            (Class::Vec, "ymm") => View::Ymm,
            (Class::Vec, "xmm") => View::Xmm,
            (Class::Vec, "vec") => v.view,
            (Class::Gpr, "gpr") => View::R64,
            (Class::Gpr, "r32") => View::R32,
            (Class::Vec, _) => return Err(format!("a vector is seen as `ymm`, `xmm` or `vec`, not `{view}`")),
            _ => return Err(format!("a gpr is seen as `gpr` or `r32`, not `{view}`")),
        };
        Ok(Var { reg: v.reg, view: to })
    }

    fn label(&mut self, label: u32) {
        self.p.body.push(Item::Label(Label(label)));
    }

    fn note(&mut self, text: &str) {
        self.p.body.push(Item::Note(text.to_string()));
    }

    fn inst(&mut self, name: &str, args: &[Given<Var>], guard: Option<Guard<Var>>) -> Result<(), Fault> {
        if guard.is_some() {
            return Err(Fault::new("x86 runs an instruction whatever a predicate says; jump around it"));
        }
        for (i, a) in args.iter().enumerate() {
            match a {
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
                    let built = self.build(form, args)?;
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
