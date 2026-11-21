//! Which bits of a vector register are the kernel's own. SysV passes a float in the low bits of a vector register, 32
//! of an `f32` and 64 of an `f64`, and leaves the rest as the caller had it; a call gives one back so, and leaves the
//! rest as the callee had it. A variable's width is how many of the low bits of its register are the kernel's own: a
//! float's, to begin with, and all 256 of any other; and then what each instruction that writes it makes of it.
//!
//! An instruction does one of three things with a bit of a vector it takes. It reads it, whatever it makes of it, and
//! the bit must be within the variable's width. Or it carries it to the same place of its result, as `vandps` and
//! `vblendps` do: the bit is as much the kernel's own in the result as it was, and the instruction does not read it. Or
//! it leaves it, as `vpshufd` leaves the lanes that its number does not name. [`check`] follows the widths over the
//! body's flow, and gives the first read of a bit that is not the kernel's own.

use kernite_compile::forward;
use kernite_core::Access::{Read, ReadWrite, Write};
use kernite_core::Width::W128;
use kernite_core::{Arg, Class, Inst, Item, Operand, Program, VReg, Width};

use crate::forms::{self, Kind, Spec};
use crate::op::Op;

/// The width of each variable, in bits, by [`VReg::id`].
type Widths = Vec<u16>;

/// What stops an instruction: it reads `reads` bits of `var`, and only `has` are the kernel's own.
struct Past {
    var: VReg,
    reads: u16,
    has: u16,
}

/// `len` bits of operand `from`, from its bit `src`, that an instruction carries to the first operand from its bit
/// `dst`: a lane of `lane` bits of them is the result's own where all of it was the source's.
struct Carry {
    from: usize,
    src: u16,
    dst: u16,
    len: u16,
    lane: u16,
}

/// What an instruction does with the vectors it takes, by their place among its operands, written out and not: how many
/// low bits it reads of each, and what it carries to its first operand.
struct Effect<'a> {
    inst: &'a Inst<Op, VReg>,
    /// How many bits it writes of its first operand, if that is a vector; else 0.
    size: u16,
    /// How many low bits it reads of each operand; 0 of one that is no vector it reads.
    reads: Vec<u16>,
    carries: Vec<Carry>,
}

fn bits(w: Width) -> u16 {
    w.bytes() as u16 * 8
}

impl<'a> Effect<'a> {
    /// What `inst` does with the vectors it takes. Its form says how many bits it reads of each, and, of a scalar
    /// instruction, that it carries the rest of its first source; [`by_name`](Self::by_name) has the rest.
    fn of(inst: &'a Inst<Op, VReg>) -> Self {
        let size = match inst.args.first() {
            Some(Arg { operand: Operand::Reg(v, w), access: Write | ReadWrite }) if v.class == Class::Vec => bits(*w),
            _ => 0,
        };
        let mut e = Effect { inst, size, reads: vec![0; inst.args.len() + inst.implicit.len()], carries: vec![] };
        let form = forms::form(inst.op, &inst.args, &inst.implicit);
        for (i, a) in inst.operands().enumerate() {
            if let Operand::Reg(_, w) = a.operand {
                let kind = form.and_then(|f| f.operands.get(i)).map(|s| s.kind);
                e.read(i, bits(if let Some(Kind::Vec(k)) = kind { k } else { w }));
            }
        }
        // A scalar instruction writes the low `n` bits of its result and carries the rest of its first source. The form
        // says how many bits it reads of that source: none, if it is an `xmm`.
        if let Some(&Spec { kind: Kind::Vec(n), access }) = form.and_then(|f| f.operands.first())
            && n < W128
            && access != Read
        {
            let (n, from) = (bits(n), usize::from(access == Write));
            if matches!(form.and_then(|f| f.operands.get(from)).map(|s| s.kind), Some(Kind::Vec(W128))) {
                e.read(from, 0);
            }
            e.carry(from, n, n, 128 - n, 1);
        }
        e.by_name();
        e
    }

    /// The vector that operand `i` is, if it is one that the instruction reads.
    fn vector(&self, i: usize) -> Option<VReg> {
        match self.inst.operands().nth(i)? {
            Arg { operand: Operand::Reg(v, _), access: Read | ReadWrite } if v.class == Class::Vec => Some(*v),
            _ => None,
        }
    }

    /// Has it read the low `bits` of operand `i`, if that is a vector, and no more.
    fn read(&mut self, i: usize, bits: u16) {
        if self.vector(i).is_some()
            && let Some(read) = self.reads.get_mut(i)
        {
            *read = bits;
        }
    }

    /// Has it carry `len` bits of operand `from`, if that is a vector, from its bit `src` to the result's bit `dst`, in
    /// lanes of `lane` bits.
    fn carry(&mut self, from: usize, src: u16, dst: u16, len: u16, lane: u16) {
        if self.size > 0 && self.vector(from).is_some() {
            self.carries.push(Carry { from, src, dst, len, lane });
        }
    }

    /// Has it carry `len` bits of operand `i` from its bit `src` to the same place of the result, and read none.
    fn keep(&mut self, i: usize, src: u16, len: u16) {
        if self.size > 0 {
            self.read(i, 0);
            self.carry(i, src, src, len, 1);
        }
    }

    /// What the instruction does that its form does not say, by its name and its number: the bits it carries, and the
    /// lanes it picks, which are all it reads of an operand.
    fn by_name(&mut self) {
        let (name, size, args) = (self.inst.op.name(), self.size, &self.inst.args);
        let imm =
            args.last().and_then(|a| if let Operand::Imm(i) = a.operand { Some(i.value as u16 & 255) } else { None });
        let n = imm.unwrap_or(0);
        // The highest of the `lanes` fields of `each` bits that the number has.
        let top = |lanes: u16, each: u16| (0..lanes).map(|j| n >> (each * j) & ((1 << each) - 1)).max().unwrap_or(0);
        // How many low bits of an operand it takes to reach lane `last` of `lane` bits in each half of a ymm.
        let halves = (size / 128).max(1);
        let upto = |last: u16, lane: u16| 128 * (halves - 1) + (last + 1) * lane;
        match name {
            // The bits of its sources, and a whole register moved.
            "vandps" | "vandpd" | "vandnps" | "vandnpd" | "vorps" | "vorpd" | "vxorps" | "vxorpd" | "vpand"
            | "vpandn" | "vpor" | "vpxor" | "vmovaps" | "vmovapd" | "vmovups" | "vmovupd" | "vmovdqa" | "vmovdqu" => {
                (1..args.len()).for_each(|i| self.keep(i, 0, size));
            }
            "vmovq" => self.keep(1, 0, 64),
            // The low bits that `vmovss` and `vmovsd` move; the rest is its first source's, as a scalar instruction
            // passes it on.
            "vmovss" | "vmovsd" if args.len() == 3 => self.keep(2, 0, if name == "vmovss" { 32 } else { 64 }),
            // A lane of one of two sources, by a number, or by the top bit of a lane of a third.
            "vblendps" | "vblendpd" | "vpblendd" | "vpblendw" => {
                let (lane, fields) = match name {
                    "vblendps" | "vpblendd" => (32, 8),
                    "vblendpd" => (64, 4),
                    _ => (16, 8),
                };
                (1..=2).for_each(|i| self.read(i, 0));
                for j in 0..size / lane {
                    let from = 1 + usize::from(n >> (j % fields) & 1);
                    self.carry(from, j * lane, j * lane, lane, 1);
                }
            }
            "vblendvps" | "vblendvpd" | "vpblendvb" => {
                let lane = match name {
                    "vblendvps" => 32,
                    "vblendvpd" => 64,
                    _ => 8,
                };
                (1..=3).for_each(|i| self.read(i, 0));
                self.carry(1, 0, 0, size, 1);
                self.carry(2, 0, 0, size, 1);
                self.carry(3, 0, 0, size, lane);
            }
            // What a merge keeps of its first source.
            "vmovhps" | "vmovhpd" if args.len() == 3 => self.keep(1, 0, 64),
            "vmovlps" | "vmovlpd" | "vmovhlps" if args.len() == 3 => self.keep(1, 64, 64),
            // An unpack interleaves the elements of the low halves of each 128 bits of its sources, or of the high
            // halves. What stays in place it carries: of a low one the first element of the first source, and of a high
            // one the last of the second, which is all it takes of that half where the elements are 64 bits. The rest
            // moves, and it reads it.
            "vunpcklpd" | "vpunpcklqdq" | "vmovlhps" => {
                (0..halves).for_each(|h| self.keep(1, 128 * h, 64));
                self.read(2, upto(0, 64));
            }
            "vunpcklps" | "vpunpcklbw" | "vpunpcklwd" | "vpunpckldq" => {
                (1..=2).for_each(|i| self.read(i, upto(0, 64)));
            }
            "vunpckhpd" | "vpunpckhqdq" => (0..halves).for_each(|h| self.keep(2, 128 * h + 64, 64)),
            "vunpckhps" | "vpunpckhbw" | "vpunpckhwd" | "vpunpckhdq" => {
                let element = match name {
                    "vpunpckhbw" => 8,
                    "vpunpckhwd" => 16,
                    _ => 32,
                };
                let last = 128 * halves - element;
                self.read(2, last);
                self.carry(2, last, last, element, 1);
            }
            // It reads as far as what it takes of each half: the low quadword of `vmovddup`, and the even lanes of
            // `vmovsldup`, `vpmuludq` and `vpmuldq`.
            "vmovddup" => self.read(1, upto(0, 64)),
            "vmovsldup" => self.read(1, upto(2, 32)),
            "vpmuludq" | "vpmuldq" => (1..=2).for_each(|i| self.read(i, upto(2, 32))),
            // A gather keeps the lanes of its result that its mask leaves out, which no number known now names: it
            // carries all of them, and reads none.
            "vgatherdps" | "vpgatherdd" | "vgatherqps" | "vpgatherqd" | "vgatherdpd" | "vpgatherdq" | "vgatherqpd"
            | "vpgatherqq" => self.keep(0, 0, size),
            "vinsertps" => {
                let (from, to, zeroed) = (n >> 6 & 3, n >> 4 & 3, n & 15);
                self.read(1, 0);
                self.read(2, if zeroed >> to & 1 == 0 { (from + 1) * 32 } else { 0 });
                for j in (0..4).filter(|&j| j != to && zeroed >> j & 1 == 0) {
                    self.carry(1, j * 32, j * 32, 32, 1);
                }
            }
            "vpinsrb" | "vpinsrw" | "vpinsrd" | "vpinsrq" => {
                let lane = match name {
                    "vpinsrb" => 8,
                    "vpinsrw" => 16,
                    "vpinsrd" => 32,
                    _ => 64,
                };
                self.read(1, 0);
                for j in (0..128 / lane).filter(|&j| j != n) {
                    self.carry(1, j * lane, j * lane, lane, 1);
                }
            }
            "vinsertf128" | "vinserti128" => {
                self.read(1, 0);
                self.carry(1, 128 * (1 - n), 128 * (1 - n), 128, 1);
            }
            // The lanes it picks, by a number.
            "vextractps" | "vpextrd" => self.read(1, (n + 1) * 32),
            "vpextrb" => self.read(1, (n + 1) * 8),
            "vpextrw" => self.read(1, (n + 1) * 16),
            "vpextrq" => self.read(1, (n + 1) * 64),
            "vextractf128" | "vextracti128" => self.read(1, (n + 1) * 128),
            "vpshufd" | "vpermilps" if imm.is_some() => self.read(1, upto(top(4, 2), 32)),
            "vpshuflw" | "vpshufhw" => {
                // It picks four words of one quadword of each half, and keeps the other.
                let (picks, keeps) = if name == "vpshuflw" { (0, 64) } else { (64, 0) };
                self.read(1, upto(top(4, 2), 16) + picks);
                (0..halves).for_each(|h| self.carry(1, 128 * h + keeps, 128 * h + keeps, 64, 1));
            }
            "vpermilpd" if imm.is_some() => {
                let last = (0..size / 64).map(|j| (j & !1) + (n >> j & 1)).max().unwrap_or(0);
                self.read(1, (last + 1) * 64);
            }
            "vshufps" => {
                self.read(1, upto((n & 3).max(n >> 2 & 3), 32));
                self.read(2, upto((n >> 4 & 3).max(n >> 6 & 3), 32));
            }
            "vshufpd" => {
                let high = if size == 256 { 2 } else { 0 };
                self.read(1, (high + (n >> high & 1) + 1) * 64);
                self.read(2, (high + (n >> (high + 1) & 1) + 1) * 64);
            }
            "vpermq" | "vpermpd" => self.read(1, (top(4, 2) + 1) * 64),
            "vperm2f128" | "vperm2i128" => {
                // Each field picks a half of a source, unless its bit 3 clears it: bit 1 says of which, and bit 0 of
                // which half, so how much of it is read.
                let fields = [n & 15, n >> 4 & 15].into_iter().filter(|f| f & 8 == 0);
                let most = |source: u16| {
                    fields.clone().filter(|f| f >> 1 & 1 == source).map(|f| ((f & 1) + 1) * 128).max().unwrap_or(0)
                };
                self.read(1, most(0));
                self.read(2, most(1));
            }
            // A dot product reads the lanes it multiplies, which bits 4 and up of its number pick, in each half.
            "vdpps" | "vdppd" => {
                let (lane, picks) = if name == "vdpps" { (32, n >> 4 & 15) } else { (64, n >> 4 & 3) };
                let reads = picks.checked_ilog2().map_or(0, |last| upto(last as u16, lane));
                (1..=2).for_each(|i| self.read(i, reads));
            }
            // `vmpsadbw` reads 11 bytes of its first source, from byte 0 or 4 as bit 2 of its number says, and the
            // group of 4 bytes of its second that bits 0 and 1 name; in the high half of a ymm, as bit 5, and bits 3
            // and 4, say.
            "vmpsadbw" => {
                let n = if size == 256 { n >> 3 } else { n };
                self.read(1, upto(4 * (n >> 2 & 1) + 10, 8));
                self.read(2, upto(n & 3, 32));
            }
            "vpalignr" => {
                // The low `n` bytes of the first source, which go to the top of the result, and the second from byte
                // `n` up, which go below them; from 16 on, the first from byte `n - 16` up, alone; from 32, none.
                let (first, second) = match n {
                    0 => (0, 128 * halves),
                    1..16 => (128 * (halves - 1) + 8 * n, 128 * halves),
                    16..32 => (128 * halves, 0),
                    _ => (0, 0),
                };
                self.read(1, first);
                self.read(2, second);
            }
            "vpslldq" => self.read(1, if n < 16 { 128 * (halves - 1) + 8 * (16 - n) } else { 0 }),
            "vpsrldq" => self.read(1, if n < 16 { 128 * halves } else { 0 }),
            _ => {}
        }
    }
}

/// What `inst` does to `now`, the width of each variable: the first read past one is the error, and it goes on as if it
/// had read within them.
fn step(_: usize, inst: &Inst<Op, VReg>, now: &mut Widths) -> Result<(), Past> {
    let e = Effect::of(inst);
    let past = e.reads.iter().enumerate().find_map(|(i, &reads)| {
        let var = e.vector(i).filter(|_| reads > 0)?;
        let has = *now.get(var.id as usize)?;
        (has < reads).then_some(Past { var, reads, has })
    });
    // The result is its own as far as every lane it carries is.
    let carried = e
        .carries
        .iter()
        .filter_map(|c| {
            let have = now.get(e.vector(c.from)?.id as usize)?.saturating_sub(c.src).min(c.len);
            (have < c.len).then(|| c.dst + have / c.lane * c.lane)
        })
        .min()
        .unwrap_or(256);
    for (i, a) in inst.operands().enumerate() {
        if let (Operand::Reg(v, w), Write | ReadWrite) = (a.operand, a.access)
            && v.class == Class::Vec
        {
            // The first operand is as wide as the carried lanes leave it, for VEX clears what it does not compute; a
            // call's result, as wide as its callee's type.
            if let Some(width) = now.get_mut(v.id as usize) {
                *width = match i {
                    0 => carried,
                    i if i < inst.args.len() => 256,
                    _ => bits(w),
                };
            }
        }
    }
    past.map_or(Ok(()), Err)
}

/// The first instruction of `p` that reads a bit of a vector variable that is not the kernel's own, if any: its item's
/// index, and what is wrong. A variable begins as wide as `start` says, or all of its register. A body that is not
/// whole, that control leaves or a jump has no label for, has none: allocating registers says what is wrong with it.
pub(crate) fn check(p: &Program<Op>, start: &[(VReg, Width)]) -> Option<(usize, String)> {
    let mut widths = vec![256; p.regs.len()];
    for &(v, w) in start {
        if let Some(width) = widths.get_mut(v.id as usize) {
            *width = bits(w);
        }
    }
    let meet = |a: &Widths, b: &Widths| a.iter().zip(b).map(|(&a, &b)| a.min(b)).collect();
    let said = forward(p, widths, meet, step).ok()?;
    let (i, inst, Past { var, reads, has }) = p.body.iter().zip(said).enumerate().find_map(|(i, (item, said))| {
        let Item::Inst(inst) = item else { return None };
        Some((i, inst, said?.err()?))
    })?;
    let x = p.var_name(var);
    let message = format!(
        "`{}` reads {reads} bits of `{x}`, of which only the low {has} are the kernel's: SysV leaves the rest of the \
         register of a float as the caller had it, or the callee, of one it gave back. Read the float alone, as a \
         scalar instruction does, or write all of `{x}` first",
        inst.op.name()
    );
    Some((i, message))
}
