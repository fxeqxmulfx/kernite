use std::collections::VecDeque;

use kgen_core::Access::{Ignored, Read, ReadWrite, Write};
use kgen_core::Width::{W8, W16, W32, W64, W128};
use kgen_core::{
    Arg, Base, Class, Const, ConstId, Flow, Inst, Item, Label, Operand, Program, RegInfo, Signature, Slot as SlotSpec,
    SlotId, VReg,
};

use crate::op::Op;
use crate::operand::{Addr, Gpr, Hint, Slot, Src, Vector, Xmm, Ymm, arg, control, gpr, mem, number, src, vec};
use crate::sysv::{FLOAT_RESULT, INT_PARAMS};

/// Instructions [`Asm::capture`] took out of a kernel, to [`Asm::splice`] back in.
pub struct Stream(VecDeque<Item<Op, VReg>>);

impl Stream {
    /// The instructions left.
    pub fn len(&self) -> usize {
        self.0.iter().filter(|i| matches!(i, Item::Inst(_))).count()
    }

    /// Whether every instruction is back in the kernel.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A kernel being written, one instruction per call.
///
/// Variables stand where registers would: [`Asm::gpr`] and [`Asm::ymm`] make new ones, and
/// any number of instructions may write the same one. The allocator gives each a register
/// for its whole life, or says which ones were alive together when there were not enough.
/// Every method takes the operands its instruction takes, so an instruction the machine
/// has no encoding for does not compile.
pub struct Asm {
    p: Program<Op>,
    labels: u32,
}

impl Asm {
    /// An empty kernel called `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Asm {
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
            labels: 0,
        }
    }

    /// The kernel as written.
    pub fn finish(self) -> Program<Op> {
        self.p
    }

    fn var(&mut self, class: Class, name: String) -> VReg {
        let id = self.p.regs.len() as u32;
        self.p.regs.push(RegInfo { class, name });
        VReg { id, class }
    }

    /// The next integer or pointer parameter, in the register SysV passes it in.
    ///
    /// # Panics
    ///
    /// Past the sixth, which SysV passes on the stack.
    pub fn param(&mut self, name: impl Into<String>) -> Gpr {
        let at = *INT_PARAMS.get(self.p.params.len()).expect("six integer parameters in registers");
        let v = self.var(Class::Gpr, name.into());
        self.p.params.push((v, at));
        self.p.signature.params.push(Class::Gpr);
        Gpr(v)
    }

    /// A new general-purpose variable.
    pub fn gpr(&mut self, name: impl Into<String>) -> Gpr {
        Gpr(self.var(Class::Gpr, name.into()))
    }

    /// A new vector variable.
    pub fn ymm(&mut self, name: impl Into<String>) -> Ymm {
        Ymm(self.var(Class::Vec, name.into()))
    }

    /// A new label, to [`Asm::place`] somewhere.
    pub fn label(&mut self) -> Label {
        self.labels += 1;
        Label(self.labels - 1)
    }

    /// Puts `label` here.
    pub fn place(&mut self, label: Label) {
        self.p.body.push(Item::Label(label));
    }

    /// A remark for the listing, which also names the place in allocation errors.
    pub fn note(&mut self, text: impl Into<String>) {
        self.p.body.push(Item::Note(text.into()));
    }

    /// `size` bytes of stack, aligned to `align`.
    pub fn slot(&mut self, size: u32, align: u32) -> Slot {
        assert!(align.is_power_of_two(), "alignment {align}");
        self.p.slots.push(SlotSpec { size, align });
        Slot(SlotId(self.p.slots.len() as u32 - 1))
    }

    /// Read-only `bytes`, aligned to `align`. The same bytes asked for twice are stored once.
    pub fn constant(&mut self, bytes: &[u8], align: u32) -> Addr {
        assert!(align.is_power_of_two(), "alignment {align}");
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
        Addr { base: Base::Const(ConstId(at as u32)), index: None, disp: 0 }
    }

    /// `values` as bytes, aligned to their size up to 32.
    pub fn u8s(&mut self, values: &[u8]) -> Addr {
        let align = (values.len() as u32).next_power_of_two().min(32);
        self.constant(values, align)
    }

    /// `values` as little-endian 16-bit words.
    pub fn u16s(&mut self, values: &[u16]) -> Addr {
        self.u8s(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
    }

    /// `values` as little-endian 32-bit words.
    pub fn u32s(&mut self, values: &[u32]) -> Addr {
        self.u8s(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
    }

    /// `values` as little-endian 64-bit words.
    pub fn u64s(&mut self, values: &[u64]) -> Addr {
        self.u8s(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
    }

    /// `values` as floats.
    pub fn f32s(&mut self, values: &[f32]) -> Addr {
        self.u32s(&values.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
    }

    /// What `write` writes, taken out of the kernel rather than left at its end, for
    /// [`Asm::splice`] to put back a few instructions at a time among others.
    ///
    /// A chain of instructions that wait on each other, written in one piece, fills the
    /// scheduler with instructions not yet ready while the work behind it waits to get in.
    /// Spread among independent work, it runs in the gaps. What the chain stores to memory, and
    /// what others load from there, is the kernel's to keep in order: the allocator sees only
    /// registers.
    ///
    /// # Panics
    ///
    /// If `write` places a label or jumps: a branch cannot be spread.
    pub fn capture(&mut self, write: impl FnOnce(&mut Self)) -> Stream {
        let from = self.p.body.len();
        write(self);
        let items: VecDeque<_> = self.p.body.drain(from..).collect();
        for item in &items {
            match item {
                Item::Label(_) => panic!("a label in a captured stream"),
                Item::Inst(i) => assert!(i.flow == Flow::Next, "{:?} in a captured stream", i.op),
                Item::Note(_) => {}
            }
        }
        Stream(items)
    }

    /// The next `n` instructions of `stream` here, or as many as it has left, with the notes
    /// before them.
    pub fn splice(&mut self, stream: &mut Stream, n: usize) {
        let mut left = n;
        while let Some(item) = stream.0.front() {
            if let Item::Inst(_) = item {
                if left == 0 {
                    break;
                }
                left -= 1;
            }
            self.p.body.extend(stream.0.pop_front());
        }
    }

    fn emit(&mut self, op: Op, args: Vec<Arg<VReg>>) {
        self.inst(op, args, vec![], Flow::Next);
    }

    fn inst(&mut self, op: Op, args: Vec<Arg<VReg>>, implicit: Vec<Arg<VReg>>, flow: Flow) {
        self.p.body.push(Item::Inst(Inst { op, args, implicit, flow }));
    }

    fn jump(&mut self, op: Op, to: Label, flow: Flow) {
        self.inst(op, vec![arg(Operand::Label(to), Read)], vec![], flow);
    }
}

/// General purpose.
impl Asm {
    /// `mov dst, src`.
    pub fn mov(&mut self, dst: Gpr, src: Gpr) {
        self.emit(Op::Mov, vec![gpr(dst, W64, Write), gpr(src, W64, Read)]);
    }

    /// `movzx dst, byte ptr [from]`: one byte, zero-extended through all of `dst`.
    pub fn movzx_byte(&mut self, dst: Gpr, from: Addr) {
        self.emit(Op::Movzx, vec![gpr(dst, W32, Write), mem(from, W8)]);
    }

    /// `lea dst, [of]`: the address itself.
    pub fn lea(&mut self, dst: Gpr, of: Addr) {
        self.emit(Op::Lea, vec![gpr(dst, W64, Write), mem(of, W64)]);
    }

    /// `add dst, imm`.
    pub fn add(&mut self, dst: Gpr, imm: i32) {
        self.emit(Op::Add, vec![gpr(dst, W64, ReadWrite), number(imm.into())]);
    }

    /// `sub dst, imm`.
    pub fn sub(&mut self, dst: Gpr, imm: i32) {
        self.emit(Op::Sub, vec![gpr(dst, W64, ReadWrite), number(imm.into())]);
    }

    /// `and dst, imm`.
    pub fn and(&mut self, dst: Gpr, imm: i32) {
        self.emit(Op::And, vec![gpr(dst, W64, ReadWrite), number(imm.into())]);
    }

    /// `xor dst, imm`.
    pub fn xor(&mut self, dst: Gpr, imm: i32) {
        self.emit(Op::Xor, vec![gpr(dst, W64, ReadWrite), number(imm.into())]);
    }

    /// `dec dst`, which sets the zero flag when `dst` reaches zero.
    pub fn dec(&mut self, dst: Gpr) {
        self.emit(Op::Dec, vec![gpr(dst, W64, ReadWrite)]);
    }

    /// `test a, b`: the flags of `a & b`.
    pub fn test(&mut self, a: Gpr, b: Gpr) {
        self.emit(Op::Test, vec![gpr(a, W64, Read), gpr(b, W64, Read)]);
    }

    /// `jz to`.
    pub fn jz(&mut self, to: Label) {
        self.jump(Op::Jz, to, Flow::Branch(to));
    }

    /// `jnz to`.
    pub fn jnz(&mut self, to: Label) {
        self.jump(Op::Jnz, to, Flow::Branch(to));
    }

    /// `jmp to`.
    pub fn jmp(&mut self, to: Label) {
        self.jump(Op::Jmp, to, Flow::Jump(to));
    }

    /// `ret`, handing nothing back.
    pub fn ret(&mut self) {
        self.inst(Op::Ret, vec![], vec![], Flow::Return);
    }

    /// `prefetcht0` to `prefetchnta` as `hint` says: asks for the line at `at`. A hint, not a
    /// load: it cannot fault, whatever the address.
    pub fn prefetch(&mut self, hint: Hint, at: Addr) {
        let op = match hint {
            Hint::T0 => Op::Prefetcht0,
            Hint::T1 => Op::Prefetcht1,
            Hint::T2 => Op::Prefetcht2,
            Hint::Nta => Op::Prefetchnta,
        };
        self.emit(op, vec![mem(at, W8)]);
    }

    /// `ret`, handing back the float in the low lane of `value`, which SysV wants in `xmm0`.
    pub fn ret_f32(&mut self, value: Xmm) {
        if !self.p.results.contains(&(value.0, FLOAT_RESULT)) {
            self.p.results.push((value.0, FLOAT_RESULT));
        }
        self.p.signature.result = Some(Class::Vec);
        self.inst(Op::Ret, vec![], vec![vec(value, Read)], Flow::Return);
    }
}

/// `dst = op(a, b)`, for every width.
macro_rules! three {
    ($($(#[$m:meta])* $name:ident = $op:ident;)*) => {$(
        $(#[$m])*
        pub fn $name<V: Vector>(&mut self, dst: V, a: V, b: impl Into<Src<V>>) {
            self.emit(Op::$op, vec![vec(dst, Write), vec(a, Read), src(b.into())]);
        }
    )*};
}

/// `dst = op(a, b, control)`.
macro_rules! three_control {
    ($($(#[$m:meta])* $name:ident = $op:ident;)*) => {$(
        $(#[$m])*
        pub fn $name<V: Vector>(&mut self, dst: V, a: V, b: impl Into<Src<V>>, control_byte: u8) {
            self.emit(Op::$op, vec![vec(dst, Write), vec(a, Read), src(b.into()), control(control_byte)]);
        }
    )*};
}

/// `dst = a` shifted by `count` bits, lane by lane.
macro_rules! shift {
    ($($(#[$m:meta])* $name:ident = $op:ident;)*) => {$(
        $(#[$m])*
        pub fn $name<V: Vector>(&mut self, dst: V, a: V, count: u8) {
            self.emit(Op::$op, vec![vec(dst, Write), vec(a, Read), number(count.into())]);
        }
    )*};
}

/// `dst = op(a)`.
macro_rules! two {
    ($($(#[$m:meta])* $name:ident = $op:ident;)*) => {$(
        $(#[$m])*
        pub fn $name<V: Vector>(&mut self, dst: V, a: impl Into<Src<V>>) {
            self.emit(Op::$op, vec![vec(dst, Write), src(a.into())]);
        }
    )*};
}

/// Vector.
impl Asm {
    three! {
        /// `vpaddb`: bytes added, wrapping.
        vpaddb = Vpaddb;
        /// `vpaddd`: 32-bit lanes added.
        vpaddd = Vpaddd;
        /// `vpaddw`: 16-bit lanes added.
        vpaddw = Vpaddw;
        /// `vpand`.
        vpand = Vpand;
        /// `vpor`.
        vpor = Vpor;
        /// `vpxor`.
        vpxor = Vpxor;
        /// `vpshufb`: bytes of `a` picked by the bytes of `b`, within each 128-bit lane.
        vpshufb = Vpshufb;
        /// `vpcmpeqb`: bytes all ones where equal.
        vpcmpeqb = Vpcmpeqb;
        /// `vpsignb`: bytes of `a`, negated where `b`'s are negative, zeroed where zero.
        vpsignb = Vpsignb;
        /// `vpmaddubsw`: unsigned bytes of `a` times signed bytes of `b`, pairs summed to
        /// saturated 16-bit lanes.
        vpmaddubsw = Vpmaddubsw;
        /// `vpmaddwd`: 16-bit lanes multiplied, pairs summed to 32-bit lanes.
        vpmaddwd = Vpmaddwd;
        /// `vmulps`.
        vmulps = Vmulps;
        /// `vaddps`.
        vaddps = Vaddps;
    }

    three_control! {
        /// `vshufps`: floats of `a` to the low two of each lane's four, of `b` to the high.
        vshufps = Vshufps;
        /// `vpblendd`: 32-bit lanes from `b` where the control byte's bit is set.
        vpblendd = Vpblendd;
    }

    shift! {
        /// `vpsrld`: 32-bit lanes shifted right, zeros in.
        vpsrld = Vpsrld;
        /// `vpslld`: 32-bit lanes shifted left.
        vpslld = Vpslld;
        /// `vpsllw`: 16-bit lanes shifted left.
        vpsllw = Vpsllw;
    }

    two! {
        /// `vcvtdq2ps`: 32-bit integers to floats, rounded as MXCSR says.
        vcvtdq2ps = Vcvtdq2ps;
        /// `vmovshdup`: each odd float copied down to the even lane below it.
        vmovshdup = Vmovshdup;
    }

    /// `vfmadd231ps acc, a, b`: `acc += a * b`, rounded once.
    pub fn vfmadd231ps<V: Vector>(&mut self, acc: V, a: V, b: impl Into<Src<V>>) {
        self.emit(Op::Vfmadd231ps, vec![vec(acc, ReadWrite), vec(a, Read), src(b.into())]);
    }

    /// `vpermq`: the four quadwords of `a` rearranged across lanes.
    pub fn vpermq(&mut self, dst: Ymm, a: impl Into<Src<Ymm>>, control_byte: u8) {
        self.emit(Op::Vpermq, vec![vec(dst, Write), src(a.into()), control(control_byte)]);
    }

    /// `vmovdqa`: an aligned load.
    pub fn vmovdqa<V: Vector>(&mut self, dst: V, from: Addr) {
        self.emit(Op::Vmovdqa, vec![vec(dst, Write), mem(from, V::WIDTH)]);
    }

    /// `vmovdqu`: a load at any alignment.
    pub fn vmovdqu<V: Vector>(&mut self, dst: V, from: Addr) {
        self.emit(Op::Vmovdqu, vec![vec(dst, Write), mem(from, V::WIDTH)]);
    }

    /// `vmovups`: floats loaded at any alignment.
    pub fn vmovups<V: Vector>(&mut self, dst: V, from: Addr) {
        self.emit(Op::Vmovups, vec![vec(dst, Write), mem(from, V::WIDTH)]);
    }

    /// `vpmovzxwd`: eight 16-bit words from memory, zero-extended to 32-bit lanes.
    pub fn vpmovzxwd(&mut self, dst: Ymm, from: Addr) {
        self.emit(Op::Vpmovzxwd, vec![vec(dst, Write), mem(from, W128)]);
    }

    /// `vmovss [to], v`: the low float stored.
    pub fn store_vmovss(&mut self, to: Addr, v: Xmm) {
        self.emit(Op::Vmovss, vec![arg(to.mem(W32), Write), vec(v, Read)]);
    }

    /// `vmovdqa [to], v`: an aligned store.
    pub fn store_vmovdqa<V: Vector>(&mut self, to: Addr, v: V) {
        self.emit(Op::Vmovdqa, vec![arg(to.mem(V::WIDTH), Write), vec(v, Read)]);
    }

    /// `vmovaps [to], v`: an aligned store of floats.
    pub fn store_vmovaps<V: Vector>(&mut self, to: Addr, v: V) {
        self.emit(Op::Vmovaps, vec![arg(to.mem(V::WIDTH), Write), vec(v, Read)]);
    }

    /// `vpbroadcastw`: a 16-bit word from memory to every lane.
    pub fn vpbroadcastw<V: Vector>(&mut self, dst: V, from: Addr) {
        self.emit(Op::Vpbroadcastw, vec![vec(dst, Write), mem(from, W16)]);
    }

    /// `vpbroadcastd`: a 32-bit word from memory to every lane.
    pub fn vpbroadcastd<V: Vector>(&mut self, dst: V, from: Addr) {
        self.emit(Op::Vpbroadcastd, vec![vec(dst, Write), mem(from, W32)]);
    }

    /// `vpbroadcastq`: a 64-bit word from memory to every lane.
    pub fn vpbroadcastq<V: Vector>(&mut self, dst: V, from: Addr) {
        self.emit(Op::Vpbroadcastq, vec![vec(dst, Write), mem(from, W64)]);
    }

    /// `vbroadcastss`: a float from memory to every lane.
    pub fn vbroadcastss<V: Vector>(&mut self, dst: V, from: Addr) {
        self.emit(Op::Vbroadcastss, vec![vec(dst, Write), mem(from, W32)]);
    }

    /// `vmovq`: 64 bits from memory to the low quadword, zeros above.
    pub fn vmovq(&mut self, dst: Xmm, from: Addr) {
        self.emit(Op::Vmovq, vec![vec(dst, Write), mem(from, W64)]);
    }

    /// `vmovhps`: 64 bits from memory over the high quadword of `a`, its low one kept.
    pub fn vmovhps(&mut self, dst: Xmm, a: Xmm, from: Addr) {
        self.emit(Op::Vmovhps, vec![vec(dst, Write), vec(a, Read), mem(from, W64)]);
    }

    /// `vcvtph2ps`: eight halves to eight floats, exactly.
    pub fn vcvtph2ps(&mut self, dst: Ymm, a: impl Into<Src<Xmm>>) {
        self.emit(Op::Vcvtph2ps, vec![vec(dst, Write), src(a.into())]);
    }

    /// `vextractf128`: one 128-bit lane of `a`.
    pub fn vextractf128(&mut self, dst: Xmm, a: Ymm, lane: u8) {
        self.emit(Op::Vextractf128, vec![vec(dst, Write), vec(a, Read), number(lane.into())]);
    }

    /// `vinserti128`: `a` with its 128-bit lane `lane` replaced by `b`.
    pub fn vinserti128(&mut self, dst: Ymm, a: Ymm, b: impl Into<Src<Xmm>>, lane: u8) {
        self.emit(Op::Vinserti128, vec![vec(dst, Write), vec(a, Read), src(b.into()), number(lane.into())]);
    }

    /// `vmovhlps`: the high two floats of `b` over the low two of `a`.
    pub fn vmovhlps(&mut self, dst: Xmm, a: Xmm, b: Xmm) {
        self.emit(Op::Vmovhlps, vec![vec(dst, Write), vec(a, Read), vec(b, Read)]);
    }

    /// `vaddss`: the low floats added, the rest of `a` kept.
    pub fn vaddss(&mut self, dst: Xmm, a: Xmm, b: Xmm) {
        self.emit(Op::Vaddss, vec![vec(dst, Write), vec(a, Read), vec(b, Read)]);
    }

    /// `vxorps x, x, x`: all zeros, depending on nothing.
    pub fn zero<V: Vector>(&mut self, dst: V) {
        let x = Xmm(dst.var());
        self.emit(Op::Vxorps, vec![vec(x, Write), vec(x, Ignored), vec(x, Ignored)]);
    }
}
