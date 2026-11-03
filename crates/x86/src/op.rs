use std::borrow::Cow;

/// An instruction, by the name the language calls it: x86's mnemonic, `lock.` before it for the
/// lock prefix, as in `lock.xadd`. [`crate::forms`] has the forms of each.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Op(pub(crate) &'static str);

impl Op {
    /// The name the language calls it: `vaddps`, `lock.xadd`.
    pub fn name(self) -> &'static str {
        self.0
    }

    /// The name the assembler knows it by, its prefix and all: `vaddps`, `lock xadd`.
    pub fn mnemonic(self) -> Cow<'static, str> {
        match self.0.strip_prefix("lock.") {
            Some(op) => format!("lock {op}").into(),
            None => self.0.into(),
        }
    }
}

macro_rules! ops {
    ($($name:ident = $text:literal,)*) => {
        /// The instructions kernite writes of its own: the builder's, and those around a kernel's
        /// body.
        #[allow(non_upper_case_globals)]
        impl Op {
            $(
                #[doc = concat!("`", $text, "`")]
                pub const $name: Op = Op($text);
            )*
        }

        /// Every instruction kernite writes of its own.
        #[cfg(test)]
        const OWN: &[Op] = &[$(Op::$name),*];
    };
}

ops! {
    Add = "add",
    And = "and",
    Call = "call",
    Cmp = "cmp",
    Dec = "dec",
    Inc = "inc",
    Jmp = "jmp",
    Jne = "jne",
    Jnz = "jnz",
    Jz = "jz",
    Lea = "lea",
    LockXadd = "lock.xadd",
    Mov = "mov",
    Movzx = "movzx",
    Or = "or",
    Pause = "pause",
    Pop = "pop",
    Prefetchnta = "prefetchnta",
    Prefetcht0 = "prefetcht0",
    Prefetcht1 = "prefetcht1",
    Prefetcht2 = "prefetcht2",
    Push = "push",
    Ret = "ret",
    Sub = "sub",
    Shl = "shl",
    Shr = "shr",
    Test = "test",
    Ud2 = "ud2",
    Xor = "xor",
    Vaddps = "vaddps",
    Vaddss = "vaddss",
    Vbroadcastf128 = "vbroadcastf128",
    Vbroadcastsd = "vbroadcastsd",
    Vbroadcastss = "vbroadcastss",
    Vcvtdq2ps = "vcvtdq2ps",
    Vcvtph2ps = "vcvtph2ps",
    Vextractf128 = "vextractf128",
    Vfmadd231ps = "vfmadd231ps",
    Vinserti128 = "vinserti128",
    Vldmxcsr = "vldmxcsr",
    Vmovd = "vmovd",
    Vmovdqa = "vmovdqa",
    Vmovdqu = "vmovdqu",
    Vmovaps = "vmovaps",
    Vmovhlps = "vmovhlps",
    Vmovhps = "vmovhps",
    Vmovq = "vmovq",
    Vmovshdup = "vmovshdup",
    Vmovsldup = "vmovsldup",
    Vmovss = "vmovss",
    Vmovups = "vmovups",
    Vmulps = "vmulps",
    Vpaddb = "vpaddb",
    Vpaddd = "vpaddd",
    Vpaddw = "vpaddw",
    Vpabsb = "vpabsb",
    Vpand = "vpand",
    Vpblendd = "vpblendd",
    Vpbroadcastb = "vpbroadcastb",
    Vpbroadcastd = "vpbroadcastd",
    Vpbroadcastq = "vpbroadcastq",
    Vpbroadcastw = "vpbroadcastw",
    Vpcmpeqb = "vpcmpeqb",
    Vpcmpeqd = "vpcmpeqd",
    Vpermps = "vpermps",
    Vpermq = "vpermq",
    Vpgatherdd = "vpgatherdd",
    Vpgatherdq = "vpgatherdq",
    Vpmaddubsw = "vpmaddubsw",
    Vpmaddwd = "vpmaddwd",
    Vpmovsxbd = "vpmovsxbd",
    Vpmovzxbd = "vpmovzxbd",
    Vpmovzxwd = "vpmovzxwd",
    Vpor = "vpor",
    Vpshufb = "vpshufb",
    Vpsignb = "vpsignb",
    Vpslld = "vpslld",
    Vpsllvd = "vpsllvd",
    Vpsllw = "vpsllw",
    Vpsrld = "vpsrld",
    Vpsrlvd = "vpsrlvd",
    Vpsrlw = "vpsrlw",
    Vpunpckldq = "vpunpckldq",
    Vpxor = "vpxor",
    Vshufps = "vshufps",
    Vstmxcsr = "vstmxcsr",
    Vxorps = "vxorps",
    Vzeroupper = "vzeroupper",
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms;

    /// What kernite writes of its own, x86 has, but what the table leaves out: `ret` and `call`,
    /// which the language writes as kernite's returns and calls are, and the frame's `push`, `pop`
    /// and `vzeroupper`.
    #[test]
    fn x86_has_every_instruction_kernite_writes_of_its_own() {
        let left_out = [Op::Ret, Op::Call, Op::Push, Op::Pop, Op::Vzeroupper];
        for op in OWN.iter().filter(|op| !left_out.contains(op)) {
            assert!(forms::of(op.name()).is_some(), "`{}`", op.name());
        }
        assert_eq!(Op::LockXadd.mnemonic(), "lock xadd");
    }
}
