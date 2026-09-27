macro_rules! ops {
    ($($name:ident = $text:literal,)*) => {
        /// The instructions kernels use, by mnemonic.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Op {
            $(
                #[doc = concat!("`", $text, "`")]
                $name,
            )*
        }

        impl Op {
            /// The name the assembler knows it by.
            pub fn mnemonic(self) -> &'static str {
                match self {
                    $(Op::$name => $text,)*
                }
            }
        }
    };
}

ops! {
    Add = "add",
    And = "and",
    Dec = "dec",
    Jmp = "jmp",
    Jnz = "jnz",
    Jz = "jz",
    Lea = "lea",
    Mov = "mov",
    Movzx = "movzx",
    Ret = "ret",
    Sub = "sub",
    Test = "test",
    Xor = "xor",
    Vaddps = "vaddps",
    Vaddss = "vaddss",
    Vbroadcastss = "vbroadcastss",
    Vcvtdq2ps = "vcvtdq2ps",
    Vcvtph2ps = "vcvtph2ps",
    Vextractf128 = "vextractf128",
    Vfmadd231ps = "vfmadd231ps",
    Vmovaps = "vmovaps",
    Vmovdqa = "vmovdqa",
    Vmovdqu = "vmovdqu",
    Vmovhlps = "vmovhlps",
    Vmovq = "vmovq",
    Vmovshdup = "vmovshdup",
    Vmulps = "vmulps",
    Vpaddd = "vpaddd",
    Vpaddw = "vpaddw",
    Vpand = "vpand",
    Vpblendd = "vpblendd",
    Vpbroadcastd = "vpbroadcastd",
    Vpbroadcastq = "vpbroadcastq",
    Vpbroadcastw = "vpbroadcastw",
    Vpcmpeqb = "vpcmpeqb",
    Vpermq = "vpermq",
    Vpmaddubsw = "vpmaddubsw",
    Vpmaddwd = "vpmaddwd",
    Vpor = "vpor",
    Vpshufb = "vpshufb",
    Vpsignb = "vpsignb",
    Vpslld = "vpslld",
    Vpsllw = "vpsllw",
    Vpsrld = "vpsrld",
    Vpxor = "vpxor",
    Vshufps = "vshufps",
    Vxorps = "vxorps",
}
