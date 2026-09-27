//! The kernels as `kgen-macros` delivers them, beside scalar references that spell out the
//! arithmetic they must reproduce to the bit.

pub mod bf16;
pub mod iq2_xxs;
pub mod iq3_xxs;

/// 32 activation quants, aligned as the kernels load them.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block(pub [i8; 32]);

/// A half's bits as the float they are, as `vcvtph2ps` converts them; NaNs aside, whose
/// payloads it may quieten.
pub fn half(bits: u16) -> f32 {
    let negative = bits & 0x8000 != 0;
    let (exp, man) = (u32::from(bits >> 10 & 0x1f), u32::from(bits & 0x3ff));
    let magnitude = match exp {
        // Subnormal: `man` times 2^-24, exact in a float.
        0 => man as f32 * f32::from_bits(0x3380_0000),
        31 => f32::from_bits(0x7f80_0000 | man << 13),
        _ => f32::from_bits((exp + 112) << 23 | man << 13),
    };
    if negative { -magnitude } else { magnitude }
}

/// Whether this CPU runs the kernels: they are AVX2, FMA and F16C code.
pub fn supported() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
}

#[cfg(test)]
mod tests {
    use super::half;

    #[test]
    fn halves_convert_exactly() {
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xc000), -2.0);
        assert_eq!(half(0x0001), 2f32.powi(-24));
        assert_eq!(half(0x7bff), 65504.0);
        assert_eq!(half(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(half(0x7c00), f32::INFINITY);
    }
}
