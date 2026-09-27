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
    Or = "or",
    Prefetchnta = "prefetchnta",
    Prefetcht0 = "prefetcht0",
    Prefetcht1 = "prefetcht1",
    Prefetcht2 = "prefetcht2",
    Ret = "ret",
    Sub = "sub",
    Shl = "shl",
    Shr = "shr",
    Test = "test",
    Xor = "xor",
    Vaddps = "vaddps",
    Vaddss = "vaddss",
    Vbroadcastss = "vbroadcastss",
    Vcvtdq2ps = "vcvtdq2ps",
    Vcvtph2ps = "vcvtph2ps",
    Vextractf128 = "vextractf128",
    Vfmadd231ps = "vfmadd231ps",
    Vinserti128 = "vinserti128",
    Vmovaps = "vmovaps",
    Vmovd = "vmovd",
    Vmovdqa = "vmovdqa",
    Vmovdqu = "vmovdqu",
    Vmovhlps = "vmovhlps",
    Vmovhps = "vmovhps",
    Vmovq = "vmovq",
    Vmovshdup = "vmovshdup",
    Vmovss = "vmovss",
    Vmovups = "vmovups",
    Vmulps = "vmulps",
    Vpaddb = "vpaddb",
    Vpaddd = "vpaddd",
    Vpaddw = "vpaddw",
    Vpabsb = "vpabsb",
    Vpand = "vpand",
    Vpblendd = "vpblendd",
    Vpbroadcastd = "vpbroadcastd",
    Vpbroadcastb = "vpbroadcastb",
    Vpbroadcastq = "vpbroadcastq",
    Vpbroadcastw = "vpbroadcastw",
    Vpcmpeqb = "vpcmpeqb",
    Vpcmpeqd = "vpcmpeqd",
    Vpermq = "vpermq",
    Vpgatherdd = "vpgatherdd",
    Vpmaddubsw = "vpmaddubsw",
    Vpmaddwd = "vpmaddwd",
    Vpmovzxbd = "vpmovzxbd",
    Vpmovsxbd = "vpmovsxbd",
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
    Vpxor = "vpxor",
    Vshufps = "vshufps",
    Vxorps = "vxorps",
}
