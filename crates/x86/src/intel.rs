//! Intel syntax, as `naked_asm!` reads it.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use kgen_core::{Base, Class, Compiled, Const, Flow, Frame, Item, Label, Mem, Operand, PReg, Signature, Width};

use crate::op::Op;
use crate::sysv::{RBP, RSP};

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
    /// Each variable's name and its register's.
    pub regs: Vec<(String, String)>,
    /// The frame.
    pub frame: Frame,
}

impl Listing {
    /// The code alone, a line per instruction or label, as `naked_asm!` takes it.
    pub fn code(&self) -> impl Iterator<Item = &str> {
        self.lines.iter().filter_map(|line| match line {
            Line::Code(code) => Some(code.as_str()),
            Line::Note(_) => None,
        })
    }

    /// Everything, for people and for snapshot files: the signature, where each variable
    /// lives, the code with its notes, and the constants' bytes.
    pub fn text(&self) -> String {
        let mut out = String::new();
        let class = |c: &Class| match c {
            Class::Gpr => "gpr",
            Class::Vec => "vec",
        };
        let params: Vec<_> = self.signature.params.iter().map(class).collect();
        let result = self.signature.result.as_ref().map_or("()", class);
        writeln!(out, "# {}({}) -> {}", self.name, params.join(", "), result).unwrap();
        let saved: Vec<_> = self.frame.saved.iter().map(|&r| name(r, Width::W64)).collect();
        writeln!(
            out,
            "# frame: {} bytes aligned to {}, saves [{}]",
            self.frame.size,
            self.frame.align,
            saved.join(", ")
        )
        .unwrap();
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

/// `c` in Intel syntax, with the frame set up on entry and taken down before every `ret`.
///
/// Labels become local numbers, which `naked_asm!` asks for so that the same code can be
/// emitted twice. Numbers made of ones and zeros alone are skipped: Intel syntax reads
/// `10b` as binary ten, not as label 10 behind.
pub fn print(c: &Compiled<Op>) -> Listing {
    let mut numbers = BTreeMap::new();
    let mut places = BTreeMap::new();
    let mut next = 2;
    for (i, item) in c.body.iter().enumerate() {
        if let Item::Label(l) = item {
            while next.to_string().chars().all(|d| d == '0' || d == '1') {
                next += 1;
            }
            numbers.insert(*l, next);
            places.insert(*l, i);
            next += 1;
        }
    }
    let vector = c.body.iter().any(|item| match item {
        Item::Inst(inst) => inst.operands().any(|a| a.regs().any(|r| r.class == Class::Vec)),
        _ => false,
    });
    let mut lines: Vec<Line> = prologue(&c.frame).into_iter().map(Line::Code).collect();
    for (i, item) in c.body.iter().enumerate() {
        match item {
            Item::Label(l) => lines.push(Line::Code(format!("{}:", numbers[l]))),
            Item::Note(note) => lines.push(Line::Note(note.clone())),
            Item::Inst(inst) if inst.flow == Flow::Return => {
                lines.extend(epilogue(&c.frame, vector).into_iter().map(Line::Code));
            }
            Item::Inst(inst) => {
                let operands: Vec<String> = inst
                    .args
                    .iter()
                    .map(|a| match a.operand {
                        Operand::Reg(r, w) => name(r, w),
                        Operand::Mem(m) => memory(&m, &c.frame, inst.op != Op::Lea),
                        Operand::Imm(imm) if imm.hex => format!("{:#04x}", imm.value),
                        Operand::Imm(imm) => imm.value.to_string(),
                        Operand::Label(l) => jump_target(l, i, &numbers, &places),
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
        regs: c
            .assignment
            .iter()
            .filter_map(|(info, r)| r.map(|r| (info.name.clone(), name(r, full(r.class)))))
            .collect(),
        frame: c.frame.clone(),
    }
}

fn jump_target(l: Label, from: usize, numbers: &BTreeMap<Label, u32>, places: &BTreeMap<Label, usize>) -> String {
    let way = if places[&l] < from { 'b' } else { 'f' };
    format!("{}{way}", numbers[&l])
}

/// Saves the callee-saved registers the body writes, then, if there are slots, keeps the
/// caller's stack pointer in `rbp` and aligns the stack below it.
fn prologue(frame: &Frame) -> Vec<String> {
    let (sp, fp) = (name(RSP, Width::W64), name(RBP, Width::W64));
    let mut lines: Vec<String> = frame.saved.iter().map(|&r| format!("push {}", name(r, Width::W64))).collect();
    if frame.size > 0 {
        lines.extend([
            format!("push {fp}"),
            format!("mov {fp}, {sp}"),
            format!("and {sp}, -{}", frame.align),
            format!("sub {sp}, {}", frame.size),
        ]);
    }
    lines
}

/// Undoes [`prologue`], and clears the vector registers' upper halves if the body used any,
/// so the caller's SSE code does not pay for them.
fn epilogue(frame: &Frame, vector: bool) -> Vec<String> {
    let (sp, fp) = (name(RSP, Width::W64), name(RBP, Width::W64));
    let mut lines = vec![];
    if frame.size > 0 {
        lines.extend([format!("mov {sp}, {fp}"), format!("pop {fp}")]);
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
fn memory(m: &Mem<PReg>, frame: &Frame, sized: bool) -> String {
    let (mut text, disp) = match m.base {
        Base::Reg(r) => (name(r, Width::W64), i64::from(m.disp)),
        Base::Slot(s) => (name(RSP, Width::W64), i64::from(m.disp) + i64::from(frame.offsets[s.0 as usize])),
        Base::Const(k) => (format!("rip + {{c{}}}", k.0), i64::from(m.disp)),
    };
    if let Some((r, scale)) = m.index {
        write!(text, " + {}*{scale}", name(r, Width::W64)).unwrap();
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
