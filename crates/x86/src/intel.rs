//! Intel syntax, as `naked_asm!` reads it.

use std::collections::BTreeMap;
use std::fmt;

use kernite_compile::{Allocated, Frame};
use kernite_core::{Base, Class, Const, Item, Label, Mem, Operand, PReg, Signature, Width};

use crate::forms::Feature;
use crate::frame;
use crate::op::Op;
use crate::sysv::RSP;

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
    /// The CPU features past x86-64's own its instructions need, and those of each function it may call, by name, which
    /// nothing checks for when it runs.
    pub features: Vec<Feature>,
    /// The code, prologue and epilogues included. Constant `i`'s address reads `{ci}`, for
    /// the operand of `naked_asm!` that names it.
    pub lines: Vec<Line>,
    /// The constants, by index.
    pub consts: Vec<Const>,
    /// The functions it names, by index, each by what the source calls it, `gcd` or
    /// `tile::<4>`. Function `i`'s address reads `{fi}`, for the operand of `naked_asm!` that
    /// names it.
    pub funcs: Vec<String>,
    /// The functions a call at an address may reach, each once, by what the source calls it.
    /// The code names none of them, so `funcs` does not.
    pub reach: Vec<String>,
    /// The memory of the module's it names, its statics, each by what the source calls it.
    /// Static `COUNTS`'s address reads `{COUNTS}`, for the operand of `naked_asm!` that names it.
    pub statics: Vec<String>,
    /// Each variable's name and its register's.
    pub regs: Vec<(String, String)>,
    /// The frame.
    pub frame: Frame,
    /// Bytes of the frame besides its slots, so that a kernel that calls calls at a multiple of 16, as SysV asks, and a
    /// frame aligned to 16 lies at one.
    pub padding: u32,
    /// How many bytes of the frame lie below the stack pointer, in the red zone, which SysV lets a function use without
    /// moving the stack pointer: a kernel that calls nothing keeps up to 128 bytes of a frame aligned to 16 or less
    /// there, the frame's size and `padding` counted. A slot is at its offset in `frame` from as far below the stack
    /// pointer as that.
    pub below: u32,
    /// Whether control returns: whether the kernel has a `ret`, though a jump to a callee may stand for it. One with
    /// none never returns.
    pub returns: bool,
    /// The registers a call of it changes that a caller does not keep: those of SysV's caller-saved that its code
    /// writes, its frame's among them. The registers the functions it calls change are theirs to say, `funcs` and
    /// `reach`, but for a call at an address with no functions given, which may change all of them: then it has all.
    pub changes: Vec<PReg>,
    /// Whether its code runs `vzeroupper`, which clears the upper half of each vector register: a call of it leaves the
    /// lower halves of those it does not write as they were.
    pub clears_upper: bool,
    /// Whether a call of it may run SSE code, which pays for an upper half set: its code calls a function it does not
    /// know, or one that may, with no ymm passed or given back, for such a callee runs AVX instructions. It does not
    /// clear an upper half that its caller set before that call, so a kernel that calls it has `vzeroupper` before the
    /// call where it may have set one.
    pub reaches_sse: bool,
}

impl Listing {
    /// The code alone, a line per instruction or label, as `naked_asm!` takes it.
    pub fn code(&self) -> impl Iterator<Item = &str> {
        self.lines.iter().filter_map(|line| match line {
            Line::Code(code) => Some(code.as_str()),
            Line::Note(_) => None,
        })
    }
}

impl fmt::Display for Listing {
    /// Everything, for people and for snapshot files: the signature, the CPU features it
    /// needs, the frame and where each of its slots is, a variable that spilled among them,
    /// where each variable lives, the code with its notes, the functions it names, and the
    /// constants' bytes.
    fn fmt(&self, out: &mut fmt::Formatter) -> fmt::Result {
        let class = |c: &Class| match c {
            Class::Gpr => "gpr",
            Class::Vec => "vec",
            Class::Pred => "pred",
            Class::Float => "float",
        };
        let params: Vec<_> = self.signature.params.iter().map(class).collect();
        let result = match self.returns {
            true => self.signature.result.as_ref().map_or("()", class),
            false => "!",
        };
        writeln!(out, "# {}({}) -> {}", self.name, params.join(", "), result)?;
        if !self.features.is_empty() {
            let features: Vec<_> = self.features.iter().map(|f| f.name()).collect();
            writeln!(out, "# needs: {}", features.join(", "))?;
        }
        // A general-purpose register has a name at 64 bits, as the frame's do.
        let gpr64 = |r: PReg| name(r, Width::W64).unwrap_or_default();
        let saved: Vec<_> = self.frame.saved.iter().map(|&r| gpr64(r)).collect();
        let padded = match (self.padding, self.frame.align) {
            (0, _) => String::new(),
            (n, 16) => format!(", and {n} more so that it is aligned to 16"),
            (n, _) => format!(", and {n} more so that it calls at a multiple of 16"),
        };
        writeln!(
            out,
            "# frame: {} bytes aligned to {}, saves [{}]{padded}",
            self.frame.size,
            self.frame.align,
            saved.join(", ")
        )?;
        let from = -i64::from(self.below);
        for (slot, &at) in self.frame.slots.iter().zip(&self.frame.offsets) {
            let at = format!("{}{}", gpr64(RSP), displacement(from + i64::from(at)));
            let spilled = if slot.spill { ", spilled" } else { "" };
            writeln!(out, "# {}{spilled}: {} at {at}", slot.name, crate::bytes(slot.size.into()))?;
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
            writeln!(out, "# {}", pairs.join(" "))?;
        }
        for line in &self.lines {
            match line {
                Line::Code(code) if code.ends_with(':') => writeln!(out, "{code}")?,
                Line::Code(code) => writeln!(out, "    {code}")?,
                Line::Note(note) => writeln!(out, "    # {note}")?,
            }
        }
        for (i, f) in self.funcs.iter().enumerate() {
            writeln!(out, "# f{i}: {f}")?;
        }
        for (i, c) in self.consts.iter().enumerate() {
            writeln!(out, "# c{i}, aligned to {}:", c.align)?;
            for row in c.bytes.chunks(16) {
                let bytes: Vec<_> = row.iter().map(|b| format!("{b:02x}")).collect();
                writeln!(out, "#   {}", bytes.join(" "))?;
            }
        }
        Ok(())
    }
}

/// A displacement as an address has it after the register: ` + 8`, ` - 8`, or nothing for 0.
fn displacement(disp: i64) -> String {
    match disp {
        0 => String::new(),
        d if d > 0 => format!(" + {d}"),
        d => format!(" - {}", -d),
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
/// calls has its stack pointer at a multiple of 16 for each call. The part of a frame in the red zone is neither set
/// up nor taken down. Or why not: a frame past what an address reaches.
///
/// Labels become local numbers, in the order they are placed, which `naked_asm!` asks for so
/// that the same code can be emitted twice. Numbers made of ones and zeros alone are skipped:
/// Intel syntax reads `10b` as binary ten, not as label 10 behind.
///
/// `reach` is what a call at an address may reach, which `c` does not name: the listing keeps it.
pub fn print(c: &Allocated<Op>, reach: Vec<String>) -> Result<Listing, String> {
    let lowered = frame::lower(c)?;
    let from = -i64::from(lowered.below);
    let mut numbers = BTreeMap::new();
    let mut places = BTreeMap::new();
    let mut next = 2;
    for (i, item) in lowered.body.iter().enumerate() {
        if let Item::Label(l) = item {
            while next.to_string().chars().all(|d| d == '0' || d == '1') {
                next += 1;
            }
            numbers.insert(*l, next);
            places.insert(*l, i);
            next += 1;
        }
    }
    let mut lines = Vec::with_capacity(lowered.body.len());
    for (i, item) in lowered.body.iter().enumerate() {
        lines.push(match item {
            Item::Label(l) => Line::Code(format!("{}:", number(&numbers, *l)?)),
            Item::Note(note) => Line::Note(note.clone()),
            Item::Inst(inst) => {
                // A gather's vector of indices is among the operands it does not write out, at
                // its width.
                let vector_index = |m: &Mem<PReg>| {
                    inst.implicit.iter().find_map(|a| match (a.operand, m.index) {
                        (Operand::Reg(r, w), Some((i, _))) if r == i && r.class == Class::Vec => Some(w),
                        _ => None,
                    })
                };
                let mut operands: Vec<String> = Vec::with_capacity(inst.args.len());
                for a in &inst.args {
                    operands.push(match a.operand {
                        Operand::Reg(r, w) => name(r, w)?,
                        Operand::Mem(m) => memory(&m, c, from, inst.op != Op::Lea, vector_index(&m))?,
                        Operand::Imm(imm) if imm.hex => format!("{:#04x}", imm.value),
                        Operand::Imm(imm) => imm.value.to_string(),
                        Operand::Label(l) => jump_target(l, i, &numbers, &places)?,
                        Operand::Func(f) => format!("{{f{}}}", f.0),
                    });
                }
                let mnemonic = inst.op.mnemonic();
                Line::Code(if operands.is_empty() {
                    mnemonic.to_string()
                } else {
                    format!("{mnemonic} {}", operands.join(", "))
                })
            }
        });
    }
    let mut regs = vec![];
    for (info, r) in &c.assignment {
        if let Some(r) = *r {
            regs.push((info.name.clone(), name(r, full(r.class)?)?));
        }
    }
    Ok(Listing {
        name: c.name.clone(),
        signature: c.signature.clone(),
        features: lowered.features,
        lines,
        consts: c.consts.clone(),
        funcs: c.funcs.clone(),
        reach,
        statics: c.statics.clone(),
        regs,
        frame: c.frame.clone(),
        padding: lowered.padding,
        below: lowered.below,
        returns: lowered.returns,
        changes: lowered.changes,
        clears_upper: lowered.clears_upper,
        reaches_sse: lowered.reaches_sse,
    })
}

/// The number local label `l` has, or why none: a jump to a label never placed.
fn number(numbers: &BTreeMap<Label, u32>, l: Label) -> Result<u32, String> {
    numbers.get(&l).copied().ok_or_else(|| format!("a jump goes to label {}, which is never placed", l.0))
}

fn jump_target(
    l: Label,
    from: usize,
    numbers: &BTreeMap<Label, u32>,
    places: &BTreeMap<Label, usize>,
) -> Result<String, String> {
    let way = if places.get(&l).is_some_and(|&at| at < from) { 'b' } else { 'f' };
    Ok(format!("{}{way}", number(numbers, l)?))
}

/// The widest view of a register of `class`, or why it has none.
fn full(class: Class) -> Result<Width, String> {
    match class {
        Class::Gpr => Ok(Width::W64),
        Class::Vec => Ok(Width::W256),
        Class::Pred | Class::Float => Err(format!("x86 has no {class:?} registers")),
    }
}

/// A register's name at a width, or why it has none: the file has no such view of it.
fn name(r: PReg, width: Width) -> Result<String, String> {
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
    let hw = usize::from(r.hw);
    let names = match (r.class, width) {
        (Class::Gpr, Width::W64) => GPR64,
        (Class::Gpr, Width::W32) => GPR32,
        (Class::Gpr, Width::W16) => GPR16,
        (Class::Gpr, Width::W8) => GPR8,
        (Class::Vec, Width::W128) => return Ok(format!("xmm{hw}")),
        (Class::Vec, Width::W256) => return Ok(format!("ymm{hw}")),
        (class, width) => return Err(format!("no {width:?} view of a {class:?} register")),
    };
    names
        .get(hw)
        .map(ToString::to_string)
        .ok_or_else(|| format!("x86 has {} gprs, so none is numbered {hw}", names.len()))
}

/// What the program has numbered `at`, or why it has none: an address names what the program does not hold.
fn numbered<'a, T>(items: &'a [T], at: u32, what: &str) -> Result<&'a T, String> {
    items.get(at as usize).ok_or_else(|| format!("an address is of {what} {at}, and the program has {}", items.len()))
}

/// A memory operand; a slot's address is its offset in the frame from `from` past the stack pointer, which is as far
/// below it as the frame's `below`. `sized` puts the access's size in front, which every instruction but `lea` wants.
/// `vector_index` is the width of its vector of indices, if it is at one; an index that is a vector of no width given,
/// which whoever built the instruction left out, is an error.
fn memory(
    m: &Mem<PReg>,
    c: &Allocated<Op>,
    from: i64,
    sized: bool,
    vector_index: Option<Width>,
) -> Result<String, String> {
    let (mut text, disp) = match m.base {
        Base::None => (String::new(), i64::from(m.disp)),
        Base::Reg(r) => (name(r, Width::W64)?, i64::from(m.disp)),
        Base::Slot(s) => {
            let offset = *numbered(&c.frame.offsets, s.0, "slot")?;
            (name(RSP, Width::W64)?, i64::from(m.disp) + from + i64::from(offset))
        }
        Base::Const(k) => (format!("rip + {{c{}}}", k.0), i64::from(m.disp)),
        Base::Func(f) => (format!("rip + {{f{}}}", f.0), i64::from(m.disp)),
        Base::Static(s) => (format!("rip + {{{}}}", numbered(&c.statics, s.0, "static")?), i64::from(m.disp)),
    };
    if let Some((r, scale)) = m.index {
        let width = match (r.class, vector_index) {
            (Class::Vec, Some(width)) => width,
            (Class::Vec, None) => {
                return Err("a vector of indices among the operands an instruction reads unwritten".into());
            }
            _ => Width::W64,
        };
        let plus = if text.is_empty() { "" } else { " + " };
        text += &format!("{plus}{}*{scale}", name(r, width)?);
    }
    text += &displacement(disp);
    if !sized {
        return Ok(format!("[{text}]"));
    }
    let size = match m.width {
        Width::W8 => "byte",
        Width::W16 => "word",
        Width::W32 => "dword",
        Width::W64 => "qword",
        Width::W128 => "xmmword",
        Width::W256 => "ymmword",
    };
    Ok(format!("{size} ptr [{text}]"))
}
