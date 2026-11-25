use kernite_core::{Access, Arg, Base, Class, Imm, Mem, Operand, Register, SlotId, VReg, Width};

/// A general-purpose variable, seen as 64 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gpr(pub(crate) VReg);

/// A vector variable, seen whole: 256 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ymm(pub(crate) VReg);

/// A vector variable's low 128 bits. The VEX instructions that write it clear the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xmm(pub(crate) VReg);

impl Ymm {
    /// Its low 128 bits.
    pub fn xmm(self) -> Xmm {
        Xmm(self.0)
    }
}

impl Xmm {
    /// The whole register.
    pub fn ymm(self) -> Ymm {
        Ymm(self.0)
    }
}

/// A vector variable at the width an instruction works at.
pub trait Vector: Copy {
    /// That width.
    const WIDTH: Width;

    /// The variable.
    fn var(self) -> VReg;
}

impl Vector for Ymm {
    const WIDTH: Width = Width::W256;

    fn var(self) -> VReg {
        self.0
    }
}

impl Vector for Xmm {
    const WIDTH: Width = Width::W128;

    fn var(self) -> VReg {
        self.0
    }
}

/// How close to the core a prefetch brings its line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hint {
    /// Every level: `prefetcht0`.
    T0,
    /// L2 and below: `prefetcht1`.
    T1,
    /// `prefetcht2`, which this generation of Intel cores treats as `T1`.
    T2,
    /// Close to the core and out of the way of the rest: `prefetchnta`.
    Nta,
}

impl Hint {
    /// Its suffix, `t0` to `nta`, as a kernel's name spells it.
    pub fn suffix(self) -> &'static str {
        match self {
            Hint::T0 => "t0",
            Hint::T1 => "t1",
            Hint::T2 => "t2",
            Hint::Nta => "nta",
        }
    }
}

/// A slot of the stack frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot(pub(crate) SlotId);

impl Slot {
    /// The address `disp` bytes into it.
    pub fn at(self, disp: i32) -> Addr {
        Addr { base: Base::Slot(self.0), index: None, disp }
    }
}

/// An address: a base, maybe an index register times a scale, and a displacement. How
/// much it covers, the instruction says. [`Gpr::times`], [`Ymm::times`] and [`Xmm::times`] make
/// one with an index and no base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Addr {
    pub(crate) base: Base<VReg>,
    pub(crate) index: Option<(VReg, u8)>,
    pub(crate) disp: i32,
}

impl Gpr {
    /// The address `disp` bytes past where it points.
    pub fn at(self, disp: i32) -> Addr {
        Addr { base: Base::Reg(self.0), index: None, disp }
    }

    /// The address of it times `scale`, with no base: `[g*4]`. x86 writes it with a 32-bit
    /// displacement, 0 unless [`Addr::plus`] adds one. A scale other than 1, 2, 4 or 8 is an
    /// error, as [`Addr::index`] says.
    pub fn times(self, scale: u8) -> Result<Addr, String> {
        Addr::NOTHING.index(self, scale)
    }
}

impl Ymm {
    /// The address of eight 32-bit offsets in it times `scale`, with no base: `[v*4]`, for an AVX2
    /// gather. x86 writes it with a 32-bit displacement, 0 unless [`Addr::plus`] adds one. A
    /// scale other than 1, 2, 4 or 8 is an error, as [`Addr::index`] says.
    pub fn times(self, scale: u8) -> Result<Addr, String> {
        Addr::NOTHING.index_ymm(self, scale)
    }
}

impl Xmm {
    /// The address of four 32-bit offsets in it times `scale`, with no base: `[v*8]`, for an AVX2
    /// quadword gather. x86 writes it with a 32-bit displacement, 0 unless [`Addr::plus`] adds one.
    /// A scale other than 1, 2, 4 or 8 is an error, as [`Addr::index`] says.
    pub fn times(self, scale: u8) -> Result<Addr, String> {
        Addr::NOTHING.index_xmm(self, scale)
    }
}

impl Addr {
    /// No base, no index and no displacement: what an address that is an index alone starts as.
    const NOTHING: Addr = Addr { base: Base::None, index: None, disp: 0 };

    /// `disp` bytes further on. A displacement that no longer fits the 32 bits x86 writes it in is an error:
    /// `a displacement past 32 bits`.
    pub fn plus(self, disp: i32) -> Result<Addr, String> {
        let disp = self.disp.checked_add(disp).ok_or("a displacement past 32 bits")?;
        Ok(Addr { disp, ..self })
    }

    /// Plus `by` times `scale`. A scale other than 1, 2, 4 or 8, a second index, and an index on
    /// a constant, whose address is relative to the instruction and takes none, are errors:
    /// `scale 3`, `one index to an address`, `a constant's address takes no index`.
    pub fn index(self, by: Gpr, scale: u8) -> Result<Addr, String> {
        self.indexed(by.0, scale)
    }

    /// Plus eight 32-bit offsets in a YMM register, for an AVX2 gather. It is an error where
    /// [`Addr::index`] is.
    pub fn index_ymm(self, by: Ymm, scale: u8) -> Result<Addr, String> {
        self.indexed(by.0, scale)
    }

    /// Plus four 32-bit offsets in an XMM register, for an AVX2 quadword gather. It is an error
    /// where [`Addr::index`] is.
    pub fn index_xmm(self, by: Xmm, scale: u8) -> Result<Addr, String> {
        self.indexed(by.0, scale)
    }

    /// `by` times `scale` as the one index of the address: the rules for an index are kept here.
    fn indexed(self, by: VReg, scale: u8) -> Result<Addr, String> {
        if !matches!(scale, 1 | 2 | 4 | 8) {
            return Err(format!("scale {scale}"));
        }
        if self.index.is_some() {
            return Err("one index to an address".into());
        }
        if matches!(self.base, Base::Const(_)) {
            return Err("a constant's address takes no index".into());
        }
        Ok(Addr { index: Some((by, scale)), ..self })
    }

    pub(crate) fn mem(self, width: Width) -> Operand<VReg> {
        Operand::Mem(Mem { base: self.base, index: self.index, disp: self.disp, width })
    }
}

/// An operand that is a vector variable or memory of the same width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src<V> {
    /// A variable.
    Reg(V),
    /// Memory.
    Mem(Addr),
}

impl From<Ymm> for Src<Ymm> {
    fn from(v: Ymm) -> Self {
        Src::Reg(v)
    }
}

impl From<Xmm> for Src<Xmm> {
    fn from(v: Xmm) -> Self {
        Src::Reg(v)
    }
}

impl<V: Vector> From<Addr> for Src<V> {
    fn from(a: Addr) -> Self {
        Src::Mem(a)
    }
}

pub(crate) fn arg<R>(operand: Operand<R>, access: Access) -> Arg<R> {
    Arg { operand, access }
}

pub(crate) fn vec<V: Vector>(v: V, access: Access) -> Arg<VReg> {
    arg(Operand::Reg(v.var(), V::WIDTH), access)
}

pub(crate) fn gpr(r: Gpr, width: Width, access: Access) -> Arg<VReg> {
    arg(Operand::Reg(r.0, width), access)
}

pub(crate) fn src<V: Vector>(s: Src<V>) -> Arg<VReg> {
    match s {
        Src::Reg(v) => vec(v, Access::Read),
        Src::Mem(a) => arg(a.mem(V::WIDTH), Access::Read),
    }
}

pub(crate) fn mem(a: Addr, width: Width) -> Arg<VReg> {
    arg(a.mem(width), Access::Read)
}

/// A control byte, which a listing shows in hex.
pub(crate) fn control(value: u8) -> Arg<VReg> {
    arg(Operand::Imm(Imm { value: value.into(), hex: true }), Access::Read)
}

/// A count or an addend, which a listing shows in decimal.
pub(crate) fn number<R>(value: i64) -> Arg<R> {
    arg(Operand::Imm(Imm { value, hex: false }), Access::Read)
}

/// Whether `a` is all of a ymm, whose upper half an instruction that reads or writes it may set.
pub(crate) fn ymm<R: Register>(a: &Arg<R>) -> bool {
    matches!(a.operand, Operand::Reg(r, Width::W256) if r.class() == Class::Vec)
}

/// What a constant of `len` bytes is aligned to: its size, rounded up to a power of 2, 32 at most. The size is cut to
/// 32 first: a `u32` holds no power of 2 above 2^31, and a size past 2^32 is no `u32`.
pub(crate) fn constant_align(len: usize) -> u32 {
    (len.min(32) as u32).next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::constant_align;

    #[test]
    fn a_constant_is_aligned_to_its_size_up_to_32_whatever_the_size() {
        let sizes =
            [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 31, 32, 33, 4096, 1 << 31, (1 << 31) + 1, (1 << 32) + 3, usize::MAX];
        let aligns = sizes.map(constant_align);
        assert_eq!(aligns, [1, 1, 2, 4, 4, 8, 8, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32]);
    }
}
