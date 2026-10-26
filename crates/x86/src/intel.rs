//! Intel syntax, as `naked_asm!` reads it.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use kgen_core::{Access, Arg, Base, Class, Compiled, Const, Frame, Item, Label, Mem, Operand, PReg, Signature, Width};

use crate::op::Op;
use crate::sysv::{PAGE, R11, RBP, RSP};

/// A line of a listing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Line {
    /// An instruction or a label, as the assembler reads it.
    Code(String),
    /// A remark for people.
    Note(String),
}

/// An allocated kernel in Intel syntax.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listing {
    /// What it is called.
    pub name: String,
    /// What it takes and gives.
    pub signature: Signature,
    /// The code, prologue and epilogues included. Constant `i`'s address reads `{ci}`, for
    /// the operand of `naked_asm!` that names it.
    pub lines: Vec<Line>,
    /// The constants, by index.
    pub consts: Vec<Const>,
    /// The functions it names, by index, each by what the source calls it, `gcd` or
    /// `tile::<4>`. Function `i`'s address reads `{fi}`, for the operand of `naked_asm!` that
    /// names it.
    pub funcs: Vec<String>,
    /// The memory of the module's it names, its statics, each by what the source calls it.
    /// Static `COUNTS`'s address reads `{COUNTS}`, for the operand of `naked_asm!` that names it.
    pub statics: Vec<String>,
    /// Each variable's name and its register's.
    pub regs: Vec<(String, String)>,
    /// The frame.
    pub frame: Frame,
    /// Bytes the prologue moves the stack pointer down besides the frame's, so that a kernel
    /// that calls calls at a multiple of 16, as SysV asks.
    pub padding: u32,
}

impl Listing {
    /// The code alone, a line per instruction or label, as `naked_asm!` takes it.
    pub fn code(&self) -> impl Iterator<Item = &str> {
        self.lines.iter().filter_map(|line| match line {
            Line::Code(code) => Some(code.as_str()),
            Line::Note(_) => None,
        })
    }

    /// Everything, for people and for snapshot files: the signature, the frame and where each
    /// of its slots is, a variable that spilled among them, where each variable lives, the code
    /// with its notes, the functions it names, and the constants' bytes.
    pub fn text(&self) -> String {
        let mut out = String::new();
        let class = |c: &Class| match c {
            Class::Gpr => "gpr",
            Class::Vec => "vec",
            Class::Pred => "pred",
            Class::Float => "float",
        };
        let params: Vec<_> = self.signature.params.iter().map(class).collect();
        // Code with no `ret` never returns.
        let result = match self.code().any(|c| c == "ret") {
            true => self.signature.result.as_ref().map_or("()", class),
            false => "!",
        };
        writeln!(out, "# {}({}) -> {}", self.name, params.join(", "), result).unwrap();
        let saved: Vec<_> = self.frame.saved.iter().map(|&r| name(r, Width::W64)).collect();
        let padded = match self.padding {
            0 => String::new(),
            n => format!(", and {n} more so that it calls at a multiple of 16"),
        };
        writeln!(
            out,
            "# frame: {} bytes aligned to {}, saves [{}]{padded}",
            self.frame.size,
            self.frame.align,
            saved.join(", ")
        )
        .unwrap();
        for (slot, &at) in self.frame.slots.iter().zip(&self.frame.offsets) {
            let sp = name(RSP, Width::W64);
            let at = if at == 0 { sp } else { format!("{sp} + {at}") };
            let spilled = if slot.spill { ", spilled" } else { "" };
            writeln!(out, "# {}{spilled}: {} at {at}", slot.name, crate::bytes(slot.size.into())).unwrap();
        }
        // A name the kernel gives many variables, one per place it writes one, shows once
        // with every register those got.
        let mut names: Vec<(&str, Vec<&str>)> = vec![];
        for (v, r) in &self.regs {
            match names.iter_mut().find(|(n, _)| n == v) {
                Some((_, regs)) if regs.contains(&r.as_str()) => {}
                Some((_, regs)) => regs.push(r),
                None => names.push((v, vec![r])),
            }
        }
        for chunk in names.chunks(6) {
            let pairs: Vec<_> = chunk.iter().map(|(v, r)| format!("{v}={}", r.join(","))).collect();
            writeln!(out, "# {}", pairs.join(" ")).unwrap();
        }
        for line in &self.lines {
            match line {
                Line::Code(code) if code.ends_with(':') => writeln!(out, "{code}"),
                Line::Code(code) => writeln!(out, "    {code}"),
                Line::Note(note) => writeln!(out, "    # {note}"),
            }
            .unwrap();
        }
        for (i, f) in self.funcs.iter().enumerate() {
            writeln!(out, "# f{i}: {f}").unwrap();
        }
        for (i, c) in self.consts.iter().enumerate() {
            writeln!(out, "# c{i}, aligned to {}:", c.align).unwrap();
            for row in c.bytes.chunks(16) {
                let bytes: Vec<_> = row.iter().map(|b| format!("{b:02x}")).collect();
                writeln!(out, "#   {}", bytes.join(" ")).unwrap();
            }
        }
        out
    }
}

/// Whether a listing names a constant or a function by `name`, `{c0}` or `{f0}`, as it names a
/// static by its own, `{COUNTS}`: a static of that name would be taken for one of them.
pub fn taken(name: &str) -> bool {
    let digits = name.strip_prefix(['c', 'f']).unwrap_or("");
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// `c` in Intel syntax, with the frame set up on entry and taken down before every `ret`, and
/// a note before each store of a variable that spilled and each load of it back. A kernel that
/// calls has its stack pointer at a multiple of 16 for each call; a move of a register to
/// itself that changes nothing is left out.
///
/// Labels become local numbers, which `naked_asm!` asks for so that the same code can be
/// emitted twice. Numbers made of ones and zeros alone are skipped: Intel syntax reads
/// `10b` as binary ten, not as label 10 behind.
pub fn print(c: &Compiled<Op>) -> Listing {
    let mut numbers = BTreeMap::new();
    let mut places = BTreeMap::new();
    let mut next = 2;
    let mut number = || {
        while next.to_string().chars().all(|d| d == '0' || d == '1') {
            next += 1;
        }
        next += 1;
        next - 1
    };
    let calls = c.body.iter().any(|item| matches!(item, Item::Inst(inst) if inst.op == Op::Call));
    let pad = padding(&c.frame, calls);
    // The prologue's loop, if it has one, comes first.
    let looped = (probes(&c.frame, pad) > UNROLLED).then(&mut number);
    for (i, item) in c.body.iter().enumerate() {
        if let Item::Label(l) = item {
            numbers.insert(*l, number());
            places.insert(*l, i);
        }
    }
    let vector = c.body.iter().any(|item| match item {
        Item::Inst(inst) => inst.operands().any(|a| a.regs().any(|r| r.class == Class::Vec)),
        _ => false,
    });
    let mut lines = prologue(&c.frame, pad, looped);
    for (i, item) in c.body.iter().enumerate() {
        match item {
            Item::Label(l) => lines.push(Line::Code(format!("{}:", numbers[l]))),
            Item::Note(note) => lines.push(Line::Note(note.clone())),
            Item::Inst(inst) if inst.op == Op::Ret => {
                lines.extend(epilogue(&c.frame, pad, vector).into_iter().map(Line::Code));
            }
            // All 64 bits, or all 256, to the same register: but one of 32 bits clears the upper
            // half, and one of an xmm the upper lanes.
            Item::Inst(inst)
                if matches!(inst.op, Op::Mov | Op::Vmovaps | Op::Vmovups | Op::Vmovdqa | Op::Vmovdqu)
                    && matches!(
                        inst.args.as_slice(),
                        [Arg { operand: Operand::Reg(a, w), .. }, Arg { operand: Operand::Reg(b, v), .. }]
                            if a == b && w == v && matches!(w, Width::W64 | Width::W256)
                    ) => {}
            Item::Inst(inst) => {
                let spill = inst.args.iter().find_map(|a| match a.operand {
                    Operand::Mem(Mem { base: Base::Slot(s), .. }) if c.frame.slots[s.0 as usize].spill => {
                        Some((&c.frame.slots[s.0 as usize].name, a.access))
                    }
                    _ => None,
                });
                match spill {
                    Some((var, Access::Write)) => lines.push(Line::Note(format!("{var} spilled"))),
                    Some((var, _)) => lines.push(Line::Note(format!("{var} reloaded"))),
                    None => {}
                }
                let operands: Vec<String> = inst
                    .args
                    .iter()
                    .map(|a| match a.operand {
                        Operand::Reg(r, w) => name(r, w),
                        Operand::Mem(m) => memory(&m, c, inst.op != Op::Lea, inst.op == Op::Vpgatherdq),
                        Operand::Imm(imm) if imm.hex => format!("{:#04x}", imm.value),
                        Operand::Imm(imm) => imm.value.to_string(),
                        Operand::Label(l) => jump_target(l, i, &numbers, &places),
                        Operand::Func(f) => format!("{{f{}}}", f.0),
                    })
                    .collect();
                let mnemonic = inst.op.mnemonic();
                lines.push(Line::Code(if operands.is_empty() {
                    mnemonic.to_string()
                } else {
                    format!("{mnemonic} {}", operands.join(", "))
                }));
            }
        }
    }
    Listing {
        name: c.name.clone(),
        signature: c.signature.clone(),
        lines,
        consts: c.consts.clone(),
        funcs: c.funcs.clone(),
        statics: c.statics.clone(),
        regs: c
            .assignment
            .iter()
            .filter_map(|(info, r)| r.map(|r| (info.name.clone(), name(r, full(r.class)))))
            .collect(),
        frame: c.frame.clone(),
        padding: pad,
    }
}

fn jump_target(l: Label, from: usize, numbers: &BTreeMap<Label, u32>, places: &BTreeMap<Label, usize>) -> String {
    let way = if places[&l] < from { 'b' } else { 'f' };
    format!("{}{way}", numbers[&l])
}

/// Pages past this many are touched in a loop.
const UNROLLED: u32 = 3;

/// How many bytes the stack pointer is a multiple of, always: SysV calls a function at a
/// multiple of 16, and each push moves it by 8.
const ALIGNED: u32 = 8;

/// How many pages the prologue touches on the way down to `frame`'s bottom, a page apart: as
/// many as leave no more than a page untouched above the bottom, counting `pad` and the most
/// the alignment may add.
fn probes(frame: &Frame, pad: u32) -> u32 {
    match frame.size {
        0 => 0,
        size => (size + pad + frame.align.saturating_sub(ALIGNED)).saturating_sub(PAGE).div_ceil(PAGE),
    }
}

/// Bytes the prologue moves the stack pointer down besides `frame`, if the kernel `calls`, so
/// that it calls at a multiple of 16: on entry it is 8 past one, the return address below it,
/// and each push moves it 8 more, the frame's `rbp` among them. A frame aligned to 16 or more
/// needs none.
fn padding(frame: &Frame, calls: bool) -> u32 {
    if !calls || frame.align >= 16 {
        return 0;
    }
    let pushes = frame.saved.len() as u32 + u32::from(frame.size > 0);
    (16 - (8 + 8 * pushes + frame.size) % 16) % 16
}

/// Saves the callee-saved registers the body writes; then, if there are slots, keeps the
/// caller's stack pointer in `rbp`, and moves the stack pointer down past them and `pad`, and
/// to their alignment; else down past `pad` alone.
///
/// The stack pointer never moves more than a page below memory the kernel has touched, so that
/// a frame that runs past the end of the stack faults at the guard page there rather than step
/// over it, as Rust's own functions do: a frame of more, its alignment counted, is touched a
/// page at a time on the way down, past [`UNROLLED`] pages in a loop at label `looped`, which
/// keeps its end in `r11`. The alignment comes last, a page at most, unless the stack pointer
/// has it already.
fn prologue(frame: &Frame, pad: u32, looped: Option<u32>) -> Vec<Line> {
    let (sp, fp) = (name(RSP, Width::W64), name(RBP, Width::W64));
    let code = Line::Code;
    let mut lines: Vec<Line> = frame.saved.iter().map(|&r| code(format!("push {}", name(r, Width::W64)))).collect();
    if frame.size == 0 {
        if pad > 0 {
            lines.push(code(format!("sub {sp}, {pad}")));
        }
        return lines;
    }
    assert!(frame.align <= PAGE, "a frame aligned to {}: the prologue might step over a page", frame.align);
    lines.extend([code(format!("push {fp}")), code(format!("mov {fp}, {sp}"))]);
    let pages = probes(frame, pad);
    if pages > 0 {
        lines.push(Line::Note("the frame a page at a time, each touched, so that an overflow faults".into()));
        let probe = [code(format!("sub {sp}, {PAGE}")), code(format!("or qword ptr [{sp}], 0"))];
        match looped {
            None => (0..pages).for_each(|_| lines.extend(probe.clone())),
            Some(l) => {
                let end = name(R11, Width::W64);
                lines.extend([code(format!("lea {end}, [{sp} - {}]", pages * PAGE)), code(format!("{l}:"))]);
                lines.extend(probe);
                lines.extend([code(format!("cmp {sp}, {end}")), code(format!("jne {l}b"))]);
            }
        }
    }
    let rest = frame.size + pad - pages * PAGE;
    if rest > 0 {
        lines.push(code(format!("sub {sp}, {rest}")));
    }
    if frame.align > ALIGNED {
        lines.push(code(format!("and {sp}, -{}", frame.align)));
    }
    lines
}

/// Undoes [`prologue`], and clears the vector registers' upper halves if the body used any,
/// so the caller's SSE code does not pay for them.
fn epilogue(frame: &Frame, pad: u32, vector: bool) -> Vec<String> {
    let (sp, fp) = (name(RSP, Width::W64), name(RBP, Width::W64));
    let mut lines = vec![];
    if frame.size > 0 {
        lines.extend([format!("mov {sp}, {fp}"), format!("pop {fp}")]);
    } else if pad > 0 {
        lines.push(format!("add {sp}, {pad}"));
    }
    lines.extend(frame.saved.iter().rev().map(|&r| format!("pop {}", name(r, Width::W64))));
    if vector {
        lines.push("vzeroupper".to_string());
    }
    lines.push("ret".to_string());
    lines
}

fn full(class: Class) -> Width {
    match class {
        Class::Gpr => Width::W64,
        Class::Vec => Width::W256,
        Class::Pred | Class::Float => unreachable!("x86 has no {class:?} registers"),
    }
}

/// A register's name at a width.
fn name(r: PReg, width: Width) -> String {
    const GPR64: [&str; 16] =
        ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
    const GPR32: [&str; 16] = [
        "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d",
        "r15d",
    ];
    const GPR16: [&str; 16] =
        ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w"];
    const GPR8: [&str; 16] = [
        "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b",
        "r15b",
    ];
    let hw = r.hw as usize;
    match (r.class, width) {
        (Class::Gpr, Width::W64) => GPR64[hw].to_string(),
        (Class::Gpr, Width::W32) => GPR32[hw].to_string(),
        (Class::Gpr, Width::W16) => GPR16[hw].to_string(),
        (Class::Gpr, Width::W8) => GPR8[hw].to_string(),
        (Class::Vec, Width::W128) => format!("xmm{hw}"),
        (Class::Vec, Width::W256) => format!("ymm{hw}"),
        (class, width) => panic!("no {width:?} view of a {class:?} register"),
    }
}

/// A memory operand; `sized` puts the access's size in front, which every instruction but
/// `lea` wants.
fn memory(m: &Mem<PReg>, c: &Compiled<Op>, sized: bool, xmm_index: bool) -> String {
    let (mut text, disp) = match m.base {
        Base::Reg(r) => (name(r, Width::W64), i64::from(m.disp)),
        Base::Slot(s) => (name(RSP, Width::W64), i64::from(m.disp) + i64::from(c.frame.offsets[s.0 as usize])),
        Base::Const(k) => (format!("rip + {{c{}}}", k.0), i64::from(m.disp)),
        Base::Func(f) => (format!("rip + {{f{}}}", f.0), i64::from(m.disp)),
        Base::Static(s) => (format!("rip + {{{}}}", c.statics[s.0 as usize]), i64::from(m.disp)),
    };
    if let Some((r, scale)) = m.index {
        let width = if r.class == Class::Vec { if xmm_index { Width::W128 } else { Width::W256 } } else { Width::W64 };
        write!(text, " + {}*{scale}", name(r, width)).unwrap();
    }
    match disp {
        0 => {}
        d if d > 0 => write!(text, " + {d}").unwrap(),
        d => write!(text, " - {}", -d).unwrap(),
    }
    if !sized {
        return format!("[{text}]");
    }
    let size = match m.width {
        Width::W8 => "byte",
        Width::W16 => "word",
        Width::W32 => "dword",
        Width::W64 => "qword",
        Width::W128 => "xmmword",
        Width::W256 => "ymmword",
    };
    format!("{size} ptr [{text}]")
}
