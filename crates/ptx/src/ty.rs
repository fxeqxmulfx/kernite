//! PTX's register types: what a variable is declared as, and which of them an instruction takes.

use crate::needs::Needs;

/// A register type of PTX, which is what a variable is declared as, and the types an
/// instruction's name ends in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ty {
    /// One bit, set by `setp`, guarding what `if p { .. }` holds.
    Pred,
    /// 8 bits of no type in particular, for `ld`, `st`, `cvt` and `mov`'s parts.
    B8,
    /// 16 bits of no type in particular.
    B16,
    /// 32 bits of no type in particular: they fit any instruction of 32 bits.
    B32,
    /// 64 bits of no type in particular.
    B64,
    /// 128 bits, for `ld`, `st` and `mov`.
    B128,
    /// An unsigned integer of 8 bits.
    U8,
    /// An unsigned integer of 16 bits.
    U16,
    /// An unsigned integer of 32 bits.
    U32,
    /// An unsigned integer of 64 bits, an address among them.
    U64,
    /// A signed integer of 8 bits.
    S8,
    /// A signed integer of 16 bits.
    S16,
    /// A signed integer of 32 bits.
    S32,
    /// A signed integer of 64 bits.
    S64,
    /// A float of 16 bits, IEEE's half.
    F16,
    /// Two `f16` in 32 bits, the first in the low half.
    F16x2,
    /// A float of 32 bits.
    F32,
    /// A float of 64 bits.
    F64,
    /// A float of 16 bits with `f32`'s exponent; PTX keeps it in a `b16` register.
    Bf16,
    /// Two `bf16` in 32 bits, in a `b32` register.
    Bf16x2,
    /// `f32` with the precision of `f16`, as the tensor cores read it, in a `b32` register.
    Tf32,
}

/// What a type is, for which types it fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A predicate.
    Pred,
    /// Bits of no type in particular.
    Bit,
    /// An unsigned integer.
    Unsigned,
    /// A signed integer.
    Signed,
    /// A float, or floats packed together.
    Float,
}

impl Ty {
    /// Every type, in the order an entry declares its registers.
    pub const ALL: [Ty; 21] = [
        Ty::Pred,
        Ty::B8,
        Ty::U8,
        Ty::S8,
        Ty::B16,
        Ty::U16,
        Ty::S16,
        Ty::F16,
        Ty::Bf16,
        Ty::B32,
        Ty::U32,
        Ty::S32,
        Ty::F32,
        Ty::F16x2,
        Ty::Bf16x2,
        Ty::Tf32,
        Ty::B64,
        Ty::U64,
        Ty::S64,
        Ty::F64,
        Ty::B128,
    ];

    /// The type PTX writes `name`, if it is one.
    pub fn parse(name: &str) -> Option<Ty> {
        Ty::ALL.into_iter().find(|t| t.name() == name)
    }

    /// Its name, as PTX writes it after a dot.
    pub fn name(self) -> &'static str {
        match self {
            Ty::Pred => "pred",
            Ty::B8 => "b8",
            Ty::B16 => "b16",
            Ty::B32 => "b32",
            Ty::B64 => "b64",
            Ty::B128 => "b128",
            Ty::U8 => "u8",
            Ty::U16 => "u16",
            Ty::U32 => "u32",
            Ty::U64 => "u64",
            Ty::S8 => "s8",
            Ty::S16 => "s16",
            Ty::S32 => "s32",
            Ty::S64 => "s64",
            Ty::F16 => "f16",
            Ty::F16x2 => "f16x2",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            Ty::Bf16 => "bf16",
            Ty::Bf16x2 => "bf16x2",
            Ty::Tf32 => "tf32",
        }
    }

    /// The type its register is declared as: PTX has no `bf16`, `bf16x2` or `tf32`
    /// registers, and keeps them in bits of their size.
    pub fn decl(self) -> Ty {
        match self {
            Ty::Bf16 => Ty::B16,
            Ty::Bf16x2 | Ty::Tf32 => Ty::B32,
            t => t,
        }
    }

    /// What declaring a register of it needs: PTX has `b128` registers from 8.3 on, for Volta,
    /// and `f16x2` ones from 4.2, for `sm_53`.
    pub fn needs(self) -> Needs {
        match self {
            Ty::B128 => Needs { ptx: 83, sm: 70, ..Needs::default() },
            Ty::F16x2 => Needs { ptx: 42, sm: 53, ..Needs::default() },
            _ => Needs::default(),
        }
    }

    /// Bits: 1 for a predicate.
    pub fn bits(self) -> u32 {
        match self {
            Ty::Pred => 1,
            Ty::B8 | Ty::U8 | Ty::S8 => 8,
            Ty::B16 | Ty::U16 | Ty::S16 | Ty::F16 | Ty::Bf16 => 16,
            Ty::B32 | Ty::U32 | Ty::S32 | Ty::F32 | Ty::F16x2 | Ty::Bf16x2 | Ty::Tf32 => 32,
            Ty::B64 | Ty::U64 | Ty::S64 | Ty::F64 => 64,
            Ty::B128 => 128,
        }
    }

    /// What it is.
    pub fn kind(self) -> Kind {
        match self {
            Ty::Pred => Kind::Pred,
            Ty::B8 | Ty::B16 | Ty::B32 | Ty::B64 | Ty::B128 => Kind::Bit,
            Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64 => Kind::Unsigned,
            Ty::S8 | Ty::S16 | Ty::S32 | Ty::S64 => Kind::Signed,
            Ty::F16 | Ty::F16x2 | Ty::F32 | Ty::F64 | Ty::Bf16 | Ty::Bf16x2 | Ty::Tf32 => Kind::Float,
        }
    }

    /// Whether it is an integer, signed or not.
    pub fn int(self) -> bool {
        matches!(self.kind(), Kind::Unsigned | Kind::Signed)
    }

    /// The number the host reads one of as, in memory it reads, by the language's name: `u32` for a
    /// `b32` or a `u32`, `i32` for an `s32`, `bf16` for a `bf16`; `None` for one it has none of.
    pub fn host(self) -> Option<&'static str> {
        Some(match self {
            Ty::B8 | Ty::U8 => "u8",
            Ty::S8 => "i8",
            Ty::B16 | Ty::U16 => "u16",
            Ty::S16 => "i16",
            Ty::B32 | Ty::U32 => "u32",
            Ty::S32 => "i32",
            Ty::B64 | Ty::U64 => "u64",
            Ty::S64 => "i64",
            Ty::F16 => "f16",
            Ty::Bf16 => "bf16",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            _ => return None,
        })
    }

    /// Whether a register of this type is an operand an instruction of type `want` takes, as
    /// PTX checks most instructions: one of the same size that is the same type, or untyped
    /// bits, or an integer where an integer is wanted. ptxas 13.4 takes an `f16x2` for a
    /// 32-bit integer too, and for a predicate, which PTX's ISA has not.
    pub fn fits(self, want: Ty) -> bool {
        if self == want {
            return true;
        }
        if self.bits() != want.bits() || self == Ty::Pred || want == Ty::Pred {
            return false;
        }
        match (self.kind(), want.kind()) {
            (Kind::Bit, _) | (_, Kind::Bit) => true,
            (a, b) => a != Kind::Float && b != Kind::Float,
        }
    }

    /// Whether it fits as PTX checks `ld`, `st` and `cvt`: a register may be wider than the
    /// instruction's type, and bits fit any type of no more bits, an integer bits and
    /// integers, a float bits and itself. Not for `bf16`, `bf16x2` and `tf32`, which take
    /// [`Ty::fits`].
    pub fn fits_wide(self, want: Ty) -> bool {
        if matches!(want, Ty::Bf16 | Ty::Bf16x2 | Ty::Tf32) || matches!(self, Ty::Bf16 | Ty::Bf16x2 | Ty::Tf32) {
            return self.fits(want);
        }
        if self == Ty::Pred || want == Ty::Pred {
            return self == want;
        }
        if self.bits() < want.bits() {
            return false;
        }
        match (self.kind(), want.kind()) {
            (Kind::Bit, _) => true,
            (Kind::Unsigned | Kind::Signed, k) => k != Kind::Float,
            (Kind::Float, Kind::Bit) => true,
            (Kind::Float, Kind::Float) => self == want,
            _ => false,
        }
    }
}
