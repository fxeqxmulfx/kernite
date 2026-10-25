//! The floats of 16 bits a kernel takes and a buffer holds, which Rust has not: `f16` is
//! unstable, and there is no `bf16`. Each is its bits, and converts to and from `f32` as the
//! GPU's `cvt` does, rounding to the nearest and to even on a tie.

use std::cmp::Ordering;
use std::fmt;

/// IEEE 754's `binary16`: a sign, 5 bits of exponent and 10 of fraction, as a kernel's `f16`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct f16(u16);

/// bfloat16: a sign, 8 bits of exponent and 7 of fraction, the top half of an `f32`, as a
/// kernel's `bf16`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct bf16(u16);

impl f16 {
    /// The float these bits are.
    pub const fn from_bits(bits: u16) -> f16 {
        f16(bits)
    }

    /// Its bits.
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// The `f16` nearest `x`, the even one of two as near: infinity past the largest, 65504,
    /// and a NaN for a NaN.
    pub const fn from_f32(x: f32) -> f16 {
        let bits = x.to_bits();
        let sign = (bits >> 16) as u16 & 0x8000;
        let exp = (bits >> 23 & 0xff) as i32;
        let frac = bits & 0x7f_ffff;
        if exp == 0xff {
            // Infinity, or a NaN that stays one, quiet.
            let nan = if frac == 0 { 0 } else { 0x200 | (frac >> 13) as u16 };
            return f16(sign | 0x7c00 | nan);
        }
        // The exponent as `f16` biases it.
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return f16(sign | 0x7c00);
        }
        // What is left of the significand, and the bits that go, of which the first is a half.
        let (kept, shift) = match e {
            ..-10 => return f16(sign),
            -10..=0 => ((frac | 0x80_0000) >> (14 - e), 14 - e),
            _ => ((e as u32) << 10 | frac >> 13, 13),
        };
        let full = if e <= 0 { frac | 0x80_0000 } else { frac };
        let (gone, half) = (full & ((1 << shift) - 1), 1 << (shift - 1));
        // A carry out of the fraction is into the exponent, as far as infinity.
        let up = gone > half || (gone == half && kept & 1 == 1);
        f16(sign | (kept + up as u32) as u16)
    }

    /// The `f32` it is, exactly.
    pub const fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let (exp, frac) = (h >> 10 & 0x1f, h & 0x3ff);
        let bits = match (exp, frac) {
            (0, 0) => sign,
            (0, _) => {
                // A subnormal, `frac` 2^-24: its leading bit becomes the implicit one.
                let top = 31 - frac.leading_zeros();
                sign | (top + 103) << 23 | (frac << (23 - top) & 0x7f_ffff)
            }
            (0x1f, _) => sign | 0x7f80_0000 | frac << 13,
            _ => sign | (exp + 112) << 23 | frac << 13,
        };
        f32::from_bits(bits)
    }
}

impl bf16 {
    /// The float these bits are.
    pub const fn from_bits(bits: u16) -> bf16 {
        bf16(bits)
    }

    /// Its bits.
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// The `bf16` nearest `x`, the even one of two as near: infinity past the largest, and a
    /// NaN for a NaN.
    pub const fn from_f32(x: f32) -> bf16 {
        let bits = x.to_bits();
        if x.is_nan() {
            return bf16((bits >> 16) as u16 | 0x40);
        }
        // Half of what goes, less one, and one more if what stays is odd: a tie goes to even.
        let round = 0x7fff + (bits >> 16 & 1);
        bf16(((bits + round) >> 16) as u16)
    }

    /// The `f32` it is, exactly.
    pub const fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

macro_rules! float {
    ($t:ident) => {
        impl From<$t> for f32 {
            fn from(x: $t) -> f32 {
                x.to_f32()
            }
        }

        /// As floats compare: a NaN equals nothing, and zero equals minus zero.
        impl PartialEq for $t {
            fn eq(&self, other: &$t) -> bool {
                self.to_f32() == other.to_f32()
            }
        }

        impl PartialOrd for $t {
            fn partial_cmp(&self, other: &$t) -> Option<Ordering> {
                self.to_f32().partial_cmp(&other.to_f32())
            }
        }

        impl fmt::Debug for $t {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Debug::fmt(&self.to_f32(), f)
            }
        }

        impl fmt::Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Display::fmt(&self.to_f32(), f)
            }
        }
    };
}

float!(f16);
float!(bf16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_f16_is_the_f32_it_comes_back_from() {
        for bits in 0..=u16::MAX {
            let h = f16::from_bits(bits);
            let back = f16::from_f32(h.to_f32()).to_bits();
            match h.to_f32().is_nan() {
                true => assert!(f16::from_bits(back).to_f32().is_nan(), "{bits:#06x}"),
                false => assert_eq!(back, bits, "{bits:#06x} as {}", h.to_f32()),
            }
        }
    }

    #[test]
    fn f16_has_the_values_ieee_gives_it() {
        let cases = [
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x7bff, 65504.0),
            (0x0400, 6.103_515_6e-5),
            (0x0001, 5.960_464_5e-8),
            (0x03ff, 6.097_555e-5),
            (0x7c00, f32::INFINITY),
            (0x8000, -0.0),
        ];
        for (bits, x) in cases {
            assert_eq!(f16::from_bits(bits).to_f32().to_bits(), f32::to_bits(x), "{bits:#06x}");
            assert_eq!(f16::from_f32(x).to_bits(), bits, "{x}");
        }
    }

    #[test]
    fn a_tie_rounds_to_even_and_past_the_largest_is_infinity() {
        // 1 + 2^-11 lies halfway between 1 and the next f16, 1 + 2^-10; 1 + 3 2^-11 between
        // that and 1 + 2^-9.
        assert_eq!(f16::from_f32(1.0 + f32::powi(2.0, -11)).to_bits(), 0x3c00);
        assert_eq!(f16::from_f32(1.0 + 3.0 * f32::powi(2.0, -11)).to_bits(), 0x3c02);
        assert_eq!(f16::from_f32(65519.0).to_bits(), 0x7bff);
        assert_eq!(f16::from_f32(65520.0).to_bits(), 0x7c00);
        // Half the smallest subnormal is a tie with zero, and goes to it.
        assert_eq!(f16::from_f32(f32::powi(2.0, -25)).to_bits(), 0);
        assert_eq!(f16::from_f32(f32::from_bits(f32::powi(2.0, -25).to_bits() + 1)).to_bits(), 1);
        assert_eq!(bf16::from_f32(f32::from_bits(0x3f80_8000)).to_bits(), 0x3f80);
        assert_eq!(bf16::from_f32(f32::from_bits(0x3f81_8000)).to_bits(), 0x3f82);
        assert_eq!(bf16::from_f32(f32::MAX).to_bits(), 0x7f80);
        assert!(bf16::from_f32(f32::NAN).to_f32().is_nan());
    }
}
