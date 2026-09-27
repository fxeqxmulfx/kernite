use kgen_core::{Access, Arg, Base, Imm, Mem, Operand, SlotId, VReg, Width};

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
/// much it covers, the instruction says.
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
}

impl Addr {
    /// `disp` bytes further on.
    pub fn plus(self, disp: i32) -> Addr {
        Addr { disp: self.disp + disp, ..self }
    }

    /// Plus `by` times `scale`.
    ///
    /// # Panics
    ///
    /// On a scale other than 1, 2, 4 or 8, a second index, or an index on a constant,
    /// whose address is relative to the instruction and takes none.
    pub fn index(self, by: Gpr, scale: u8) -> Addr {
        assert!(matches!(scale, 1 | 2 | 4 | 8), "scale {scale}");
        assert!(self.index.is_none(), "one index to an address");
        assert!(!matches!(self.base, Base::Const(_)), "a constant's address takes no index");
        Addr { index: Some((by.0, scale)), ..self }
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

pub(crate) fn arg(operand: Operand<VReg>, access: Access) -> Arg<VReg> {
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
pub(crate) fn number(value: i64) -> Arg<VReg> {
    arg(Operand::Imm(Imm { value, hex: false }), Access::Read)
}
