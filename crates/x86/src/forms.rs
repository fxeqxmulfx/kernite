//! The instructions of x86-64-v3 that kernite knows, each in every form `llvm-mc` takes for
//! it, which assembles Rust's `naked_asm!`, but those only AVX-512 encodes: one [`Form`] each,
//! with the CPU feature it needs. They are x86-64's own on integers, SSE's to SSE4.2's as AVX
//! writes them, `vaddps` and not `addps`, and those of AVX, AVX2, FMA, F16C, BMI1, BMI2,
//! LZCNT, POPCNT and MOVBE; not the stack's, which is the frame's, nor the strings', `loop`,
//! x87's, MMX's or the system's. Those with registers of their own are not yet among them, as
//! `mul` writes `rdx:rax`, `shl` shifts by `cl`, and `vzeroupper` clears every vector:
//! [`LATER`] has each such instruction.
//!
//! A form is a line of [`TABLE`]. [`of`] gives an instruction's forms, by the name the language
//! calls it, and [`choose`] the one that operands fit, or why none does.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use kernite_core::Access::{self, Ignored, Read, ReadWrite, Write};
use kernite_core::Width::{self, W8, W16, W32, W64, W128, W256};
use kernite_core::{Arg, Class, Operand, Register};

use crate::op::Op;

/// The forms, a line for each set of them.
///
/// A line names instructions, then after ` : ` their forms, apart by ` ; `, each its operands
/// in Intel's order, apart by `, `. In a name, `{a,b}` is each of `a` and `b`, `$cc` each of
/// x86's conditions, as `cmov$cc` is `cmova` to `cmovz`, and `$pred` each of the predicates
/// `vcmpps` names, as `vcmpltps` is `vcmpps` with the predicate 1; and `lock.` is the lock
/// prefix. `[8 16 32 64]` after the names gives the forms once at each width, which `N` in an
/// operand stands for; `@avx` is the CPU feature they need, past x86-64's own, and
/// `[128@avx 256@avx2]` a width's own.
///
/// An operand is a general register, `r8` to `r64`, which the language sees as `g.r8`, `g.r16`,
/// `g.r32` and `g`; a vector, `xmm` or `ymm`, `vN` the one of `N` bits; memory of so many bits,
/// `m8` to `m256`, and `aN` such memory at a multiple of its size, or `m`, an address the
/// instruction reaches nothing at; or memory at a vector of indices: `vm32x` elements of 32
/// bits at an xmm's, `vm64y` of 64 at a ymm's. `iN` is an integer for an operation of `N` bits,
/// as x86 encodes one: of `N` bits, of either sign, but at 64 of 32, sign-extended; and
/// `imm64` one of 64 bits, as `mov` puts in a register. `#0-31` is an integer from 0 to 31, and
/// `#0x1f` a control byte of no bits but those of 0x1f, which a listing shows in hex: x86 reads
/// no more of either. `lab` is a label. `a/b` is either; `=` before an operand marks one the
/// instruction writes, `+` one it reads and writes, and `~` one it reads nothing of; it reads
/// the rest. `()` is no operands.
pub const TABLE: &str = r#"
$cc = {a,ae,b,be,c,e,g,ge,l,le,na,nae,nb,nbe,nc,ne,ng,nge,nl,nle,no,np,ns,nz,o,p,pe,po,s,z}
$pred = {eq,lt,le,unord,neq,nlt,nle,ord,eq_uq,nge,ngt,false,neq_oq,ge,gt,true,eq_os,lt_oq,le_oq,unord_s,neq_us,nlt_uq,nle_uq,ord_s,eq_us,nge_uq,ngt_uq,false_os,neq_os,ge_oq,gt_oq,true_us}

# Moves. A write to a gpr's r16 or r8 keeps the rest of it, as one to its r32 does not.
mov [8 16 32] : =rN/mN, rN ; =rN, mN ; =rN/mN, iN
mov [64] : =rN/mN, rN ; =rN, mN ; =mN, iN ; =rN, imm64
movzx movsx : =r16, r8/m8 ; =r32/r64, r8/m8/r16/m16
movsxd : =r64, r32/m32
movbe [16 32 64] @movbe : =rN, mN ; =mN, rN
movnti [32 64] : =mN, rN
lea : =r16/r32/r64, m
xchg [8 16 32 64] : +rN/mN, +rN ; +rN, +mN
xadd [8 16 32 64] : +rN/mN, +rN
bswap [32 64] : +rN
cmov$cc [16 32 64] : +rN, rN/mN
set$cc : =r8/m8

# Arithmetic and logic.
add adc sub sbb and or xor [8 16 32 64] : +rN/mN, rN ; +rN, mN ; +rN/mN, iN
cmp [8 16 32 64] : rN/mN, rN ; rN, mN ; rN/mN, iN
test [8 16 32 64] : rN/mN, rN ; rN, mN ; rN/mN, iN
inc dec neg not [8 16 32 64] : +rN/mN
imul [16 32 64] : +rN, rN/mN ; =rN, rN/mN, iN ; +rN, iN
# A shift or a rotation by 1, or by a count of which x86 reads 5 bits, and 6 at 64.
shl sal shr sar rol ror rcl rcr [8 16 32] : +rN/mN ; +rN/mN, #0-31
shl sal shr sar rol ror rcl rcr [64] : +rN/mN ; +rN/mN, #0-63
# x86 leaves a 16-bit `shld` or `shrd` by more than 16 undefined.
shld shrd [16] : +rN/mN, rN, #0-16
shld shrd [32] : +rN/mN, rN, #0-31
shld shrd [64] : +rN/mN, rN, #0-63
# A bit at a register's offset may be past memory of the operand's size, in the bytes after.
bt [16] : rN/mN, rN ; rN/mN, #0-15
bt [32] : rN/mN, rN ; rN/mN, #0-31
bt [64] : rN/mN, rN ; rN/mN, #0-63
btc btr bts [16] : +rN/mN, rN ; +rN/mN, #0-15
btc btr bts [32] : +rN/mN, rN ; +rN/mN, #0-31
btc btr bts [64] : +rN/mN, rN ; +rN/mN, #0-63
# Of 0, `bsf` and `bsr` leave the result as it was, as AMD has it and Intel's CPUs do; Intel's
# manual leaves it undefined.
bsf bsr [16 32 64] : +rN, rN/mN
popcnt [16 32 64] @popcnt : =rN, rN/mN
lzcnt [16 32 64] @lzcnt : =rN, rN/mN
tzcnt [16 32 64] @bmi1 : =rN, rN/mN
crc32 @sse4.2 : +r32, r8/m8/r16/m16/r32/m32 ; +r64, r8/m8/r64/m64
andn [32 64] @bmi1 : =rN, rN, rN/mN
bextr [32 64] @bmi1 : =rN, rN/mN, rN
blsi blsmsk blsr [32 64] @bmi1 : =rN, rN/mN
bzhi [32 64] @bmi2 : =rN, rN/mN, rN
pdep pext [32 64] @bmi2 : =rN, rN, rN/mN
rorx [32] @bmi2 : =rN, rN/mN, #0-31
rorx [64] @bmi2 : =rN, rN/mN, #0-63
sarx shlx shrx [32 64] @bmi2 : =rN, rN/mN, rN

# The rest of the general instructions.
nop : () ; ~r16/r32/r64 ; ~m16/m32/m64
pause lfence mfence sfence clc stc cmc int3 ud2 : ()
clflush prefetcht0 prefetcht1 prefetcht2 prefetchnta : m8
lock.add lock.adc lock.sub lock.sbb lock.and lock.or lock.xor [8 16 32 64] : +mN, rN ; +mN, iN
lock.inc lock.dec lock.neg lock.not [8 16 32 64] : +mN
lock.btc lock.btr lock.bts [16] : +mN, rN ; +mN, #0-15
lock.btc lock.btr lock.bts [32] : +mN, rN ; +mN, #0-31
lock.btc lock.btr lock.bts [64] : +mN, rN ; +mN, #0-63
lock.xadd [8 16 32 64] : +mN, +rN
lock.xchg [8 16 32 64] : +mN, +rN ; +rN, +mN
jmp j$cc : lab

# Floats.
vaddps vaddpd vsubps vsubpd vmulps vmulpd vdivps vdivpd vmaxps vmaxpd vminps vminpd [128 256] @avx : =vN, vN, vN/mN
vandps vandpd vandnps vandnpd vorps vorpd vxorps vxorpd [128 256] @avx : =vN, vN, vN/mN
vaddsubps vaddsubpd vhaddps vhaddpd vhsubps vhsubpd [128 256] @avx : =vN, vN, vN/mN
vunpcklps vunpcklpd vunpckhps vunpckhpd [128 256] @avx : =vN, vN, vN/mN
vsqrtps vsqrtpd vrcpps vrsqrtps [128 256] @avx : =vN, vN/mN
vaddss vsubss vmulss vdivss vmaxss vminss vsqrtss vrcpss vrsqrtss @avx : =xmm, xmm, xmm/m32
vaddsd vsubsd vmulsd vdivsd vmaxsd vminsd vsqrtsd @avx : =xmm, xmm, xmm/m64
vcmpps vcmppd [128 256] @avx : =vN, vN, vN/mN, #0x1f
vcmp$predps vcmp$predpd [128 256] @avx : =vN, vN, vN/mN
vcmpss @avx : =xmm, xmm, xmm/m32, #0x1f
vcmpsd @avx : =xmm, xmm, xmm/m64, #0x1f
vcmp$predss @avx : =xmm, xmm, xmm/m32
vcmp$predsd @avx : =xmm, xmm, xmm/m64
vcomiss vucomiss @avx : xmm, xmm/m32
vcomisd vucomisd @avx : xmm, xmm/m64
vroundps vroundpd [128 256] @avx : =vN, vN/mN, #0x0f
vroundss @avx : =xmm, xmm, xmm/m32, #0x0f
vroundsd @avx : =xmm, xmm, xmm/m64, #0x0f
vdpps [128 256] @avx : =vN, vN, vN/mN, #0xff
vdppd @avx : =xmm, xmm, xmm/m128, #0x33
vblendps [128] @avx : =vN, vN, vN/mN, #0x0f
vblendps [256] @avx : =vN, vN, vN/mN, #0xff
vblendpd [128] @avx : =vN, vN, vN/mN, #0x03
vblendpd [256] @avx : =vN, vN, vN/mN, #0x0f
vblendvps vblendvpd [128 256] @avx : =vN, vN, vN/mN, vN
vshufps [128 256] @avx : =vN, vN, vN/mN, #0xff
vshufpd [128] @avx : =vN, vN, vN/mN, #0x03
vshufpd [256] @avx : =vN, vN, vN/mN, #0x0f
vpermilps [128 256] @avx : =vN, vN, vN/mN ; =vN, vN/mN, #0xff
vpermilpd [128] @avx : =vN, vN, vN/mN ; =vN, vN/mN, #0x03
vpermilpd [256] @avx : =vN, vN, vN/mN ; =vN, vN/mN, #0x0f
vperm2f128 @avx : =ymm, ymm, ymm/m256, #0xbb
vinsertps @avx : =xmm, xmm, xmm/m32, #0xff
vextractps @avx : =r32/r64/m32, xmm, #0-3
vmovhlps vmovlhps @avx : =xmm, xmm, xmm
vmovhps vmovhpd vmovlps vmovlpd @avx : =xmm, xmm, m64 ; =m64, xmm
vmovddup @avx : =xmm, xmm/m64 ; =ymm, ymm/m256
vmovshdup vmovsldup [128 256] @avx : =vN, vN/mN
vtestps vtestpd vptest [128 256] @avx : vN, vN/mN

# Integers in vectors.
vpaddb vpaddw vpaddd vpaddq vpaddsb vpaddsw vpaddusb vpaddusw [128@avx 256@avx2] : =vN, vN, vN/mN
vpsubb vpsubw vpsubd vpsubq vpsubsb vpsubsw vpsubusb vpsubusw [128@avx 256@avx2] : =vN, vN, vN/mN
vpmullw vpmulld vpmulhw vpmulhuw vpmulhrsw vpmuludq vpmuldq vpmaddwd vpmaddubsw [128@avx 256@avx2] : =vN, vN, vN/mN
vpavgb vpavgw vpsadbw vpsignb vpsignw vpsignd [128@avx 256@avx2] : =vN, vN, vN/mN
vpmaxsb vpmaxsw vpmaxsd vpmaxub vpmaxuw vpmaxud [128@avx 256@avx2] : =vN, vN, vN/mN
vpminsb vpminsw vpminsd vpminub vpminuw vpminud [128@avx 256@avx2] : =vN, vN, vN/mN
vpand vpandn vpor vpxor [128@avx 256@avx2] : =vN, vN, vN/mN
vpcmpeqb vpcmpeqw vpcmpeqd vpcmpeqq vpcmpgtb vpcmpgtw vpcmpgtd vpcmpgtq [128@avx 256@avx2] : =vN, vN, vN/mN
vphaddw vphaddd vphaddsw vphsubw vphsubd vphsubsw [128@avx 256@avx2] : =vN, vN, vN/mN
vpacksswb vpackssdw vpackuswb vpackusdw [128@avx 256@avx2] : =vN, vN, vN/mN
vpunpcklbw vpunpcklwd vpunpckldq vpunpcklqdq vpunpckhbw vpunpckhwd vpunpckhdq vpunpckhqdq [128@avx 256@avx2] : =vN, vN, vN/mN
vpshufb [128@avx 256@avx2] : =vN, vN, vN/mN
vpabsb vpabsw vpabsd [128@avx 256@avx2] : =vN, vN/mN
vphminposuw @avx : =xmm, xmm/m128
vpshufd vpshufhw vpshuflw [128@avx 256@avx2] : =vN, vN/mN, #0xff
vpalignr [128@avx 256@avx2] : =vN, vN, vN/mN, #0-255
vpblendw [128@avx 256@avx2] : =vN, vN, vN/mN, #0xff
vpblendvb [128@avx 256@avx2] : =vN, vN, vN/mN, vN
vpblendd [128] @avx2 : =vN, vN, vN/mN, #0x0f
vpblendd [256] @avx2 : =vN, vN, vN/mN, #0xff
vmpsadbw [128] @avx : =vN, vN, vN/mN, #0x07
vmpsadbw [256] @avx2 : =vN, vN, vN/mN, #0x3f
# A vector's lanes are shifted by a byte, which empties them past their bits.
vpsllw vpslld vpsllq vpsrlw vpsrld vpsrlq vpsraw vpsrad [128@avx 256@avx2] : =vN, vN, xmm/m128 ; =vN, vN, #0-255
vpslldq vpsrldq [128@avx 256@avx2] : =vN, vN, #0-255
vpsllvd vpsllvq vpsrlvd vpsrlvq vpsravd [128 256] @avx2 : =vN, vN, vN/mN
vperm2i128 @avx2 : =ymm, ymm, ymm/m256, #0xbb
vpermd vpermps @avx2 : =ymm, ymm, ymm/m256
vpermq vpermpd @avx2 : =ymm, ymm/m256, #0xff
vpmovsxbw vpmovzxbw vpmovsxwd vpmovzxwd vpmovsxdq vpmovzxdq @avx : =xmm, xmm/m64
vpmovsxbw vpmovzxbw vpmovsxwd vpmovzxwd vpmovsxdq vpmovzxdq @avx2 : =ymm, xmm/m128
vpmovsxbd vpmovzxbd vpmovsxwq vpmovzxwq @avx : =xmm, xmm/m32
vpmovsxbd vpmovzxbd vpmovsxwq vpmovzxwq @avx2 : =ymm, xmm/m64
vpmovsxbq vpmovzxbq @avx : =xmm, xmm/m16
vpmovsxbq vpmovzxbq @avx2 : =ymm, xmm/m32

# Conversions.
vcvtdq2ps vcvtps2dq vcvttps2dq [128 256] @avx : =vN, vN/mN
vcvtdq2pd vcvtps2pd @avx : =xmm, xmm/m64 ; =ymm, xmm/m128
vcvtpd2ps vcvtpd2dq vcvttpd2dq @avx : =xmm, xmm/m128 ; =xmm, ymm/m256
vcvtss2sd @avx : =xmm, xmm, xmm/m32
vcvtsd2ss @avx : =xmm, xmm, xmm/m64
vcvtsi2ss vcvtsi2sd @avx : =xmm, xmm, r32/m32/r64/m64
vcvtss2si vcvttss2si @avx : =r32/r64, xmm/m32
vcvtsd2si vcvttsd2si @avx : =r32/r64, xmm/m64
vcvtph2ps @f16c : =xmm, xmm/m64 ; =ymm, xmm/m128
vcvtps2ph @f16c : =xmm/m64, xmm, #0x07 ; =xmm/m128, ymm, #0x07

# Moves of vectors.
vmovups vmovupd vmovdqu [128 256] @avx : =vN, vN/mN ; =mN, vN
vmovaps vmovapd vmovdqa [128 256] @avx : =vN, vN/aN ; =aN, vN
vmovntps vmovntpd vmovntdq [128 256] @avx : =aN, vN
vmovntdqa [128@avx 256@avx2] : =vN, aN
vlddqu [128 256] @avx : =vN, mN
vmovss @avx : =xmm, m32 ; =m32, xmm ; =xmm, xmm, xmm
vmovsd @avx : =xmm, m64 ; =m64, xmm ; =xmm, xmm, xmm
vmovd @avx : =xmm, r32/m32/r64 ; =r32/m32/r64, xmm
vmovq @avx : =xmm, xmm/m64/r64 ; =m64/r64, xmm
vmovmskps vmovmskpd [128 256] @avx : =r32/r64, vN
vpmovmskb [128@avx 256@avx2] : =r32/r64, vN
vmaskmovps vmaskmovpd [128 256] @avx : =vN, vN, mN ; =mN, vN, vN
vpmaskmovd vpmaskmovq [128 256] @avx2 : =vN, vN, mN ; =mN, vN, vN
vbroadcastss @avx : =xmm/ymm, m32
vbroadcastss @avx2 : =xmm/ymm, xmm
vbroadcastsd @avx : =ymm, m64
vbroadcastsd @avx2 : =ymm, xmm
vbroadcastf128 @avx : =ymm, m128
vbroadcasti128 @avx2 : =ymm, m128
vpbroadcastb [128 256] @avx2 : =vN, xmm/m8
vpbroadcastw [128 256] @avx2 : =vN, xmm/m16
vpbroadcastd [128 256] @avx2 : =vN, xmm/m32
vpbroadcastq [128 256] @avx2 : =vN, xmm/m64
vinsertf128 @avx : =ymm, ymm, xmm/m128, #0-1
vinserti128 @avx2 : =ymm, ymm, xmm/m128, #0-1
vextractf128 @avx : =xmm/m128, ymm, #0-1
vextracti128 @avx2 : =xmm/m128, ymm, #0-1
vpextrb @avx : =r32/r64/m8, xmm, #0-15
vpextrw @avx : =r32/r64/m16, xmm, #0-7
vpextrd @avx : =r32/m32, xmm, #0-3
vpextrq @avx : =r64/m64, xmm, #0-1
vpinsrb @avx : =xmm, xmm, r32/r64/m8, #0-15
vpinsrw @avx : =xmm, xmm, r32/r64/m16, #0-7
vpinsrd @avx : =xmm, xmm, r32/m32, #0-3
vpinsrq @avx : =xmm, xmm, r64/m64, #0-1

# Fused multiplies and adds.
vf{madd,msub,nmadd,nmsub,maddsub,msubadd}{132,213,231}{ps,pd} [128 256] @fma : +vN, vN, vN/mN
vf{madd,msub,nmadd,nmsub}{132,213,231}ss @fma : +xmm, xmm, xmm/m32
vf{madd,msub,nmadd,nmsub}{132,213,231}sd @fma : +xmm, xmm, xmm/m64

# Gathers, which fault unless their result, indices and mask are three registers. Each clears
# its mask as it reads, and leaves the lanes of its result the mask leaves out.
vgatherdps vpgatherdd @avx2 : +xmm, vm32x, +xmm ; +ymm, vm32y, +ymm
vgatherqps vpgatherqd @avx2 : +xmm, vm32x, +xmm ; +xmm, vm32y, +xmm
vgatherdpd vpgatherdq @avx2 : +xmm, vm64x, +xmm ; +ymm, vm64x, +ymm
vgatherqpd vpgatherqq @avx2 : +xmm, vm64x, +xmm ; +ymm, vm64y, +ymm

# The control and status of floats, which a kernel that writes it restores before it returns.
vldmxcsr @avx : m32
vstmxcsr @avx : =m32
"#;

/// The instructions of x86-64-v3 kernite does not yet write, each with what it does that keeps
/// it out: most reach registers of their own, which wait for kernite to place a kernel's variables
/// in the registers such an instruction names, as it places a call's arguments. Of those in
/// [`TABLE`], the forms of [`LATER_FORMS`] and shifts by `cl`, [`BY_CL`], wait so too.
pub const LATER: &[(&str, &str)] = &[
    ("cbw", "widens `al` into `ax`"),
    ("cdq", "widens `eax` into `edx:eax`"),
    ("cdqe", "widens `eax` into `rax`"),
    ("cmpxchg", "compares with `rax`, which it writes"),
    ("cmpxchg16b", "compares with `rdx:rax`, which it writes, and stores `rcx:rbx`"),
    ("cmpxchg8b", "compares with `edx:eax`, which it writes, and stores `ecx:ebx`"),
    ("cpuid", "reads `eax` and `ecx`, and writes them, `ebx` and `edx`"),
    ("cqo", "widens `rax` into `rdx:rax`"),
    ("cwd", "widens `ax` into `dx:ax`"),
    ("cwde", "widens `ax` into `eax`"),
    ("div", "divides `rdx:rax`, into `rax` and `rdx`"),
    ("idiv", "divides `rdx:rax`, into `rax` and `rdx`"),
    ("lahf", "loads the flags into `ah`"),
    ("lock.cmpxchg", "compares with `rax`, which it writes"),
    ("mul", "multiplies `rax` into `rdx:rax`"),
    ("mulx", "multiplies by `rdx`"),
    ("rdtsc", "reads the time stamp into `edx:eax`"),
    ("sahf", "stores `ah` in the flags"),
    ("vmaskmovdqu", "stores at `rdi`"),
    ("vpcmpestri", "reads lengths in `eax` and `edx`, and writes `ecx`"),
    ("vpcmpestrm", "reads lengths in `eax` and `edx`, and writes `xmm0`"),
    ("vpcmpistri", "writes `ecx`"),
    ("vpcmpistrm", "writes `xmm0`"),
    ("vzeroall", "clears every vector"),
    ("vzeroupper", "clears the upper half of every vector"),
    ("xgetbv", "reads `ecx`, and writes `edx:eax`"),
];

/// The instructions of [`TABLE`] that also shift or rotate by `cl`, which kernite does not yet
/// write.
pub const BY_CL: &[&str] = &["rcl", "rcr", "rol", "ror", "sal", "sar", "shl", "shld", "shr", "shrd"];

/// Forms of the instructions of [`TABLE`] that kernite does not yet write, by their number of
/// operands, each with what it does that keeps it out, as [`LATER`] has it.
pub const LATER_FORMS: &[(&str, usize, &str)] =
    &[("imul", 1, "multiplies `rax` into `rdx:rax`"), ("shld", 2, "shifts by `cl`"), ("shrd", 2, "shifts by `cl`")];

/// A CPU feature past x86-64's own that an instruction needs, as Rust's `target_feature`
/// names it; each after those it builds on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Feature {
    /// `popcnt`.
    Popcnt,
    /// `lzcnt`.
    Lzcnt,
    /// `movbe`.
    Movbe,
    /// BMI1: `andn`, `bextr`, `blsi`, `blsmsk`, `blsr` and `tzcnt`.
    Bmi1,
    /// BMI2: `bzhi`, `pdep`, `pext`, `rorx`, `sarx`, `shlx` and `shrx`.
    Bmi2,
    /// SSE4.2, of which kernite writes `crc32`, the rest being AVX's.
    Sse42,
    /// AVX: vectors of 256 bits, and every vector instruction kernite writes.
    Avx,
    /// F16C: `vcvtph2ps` and `vcvtps2ph`.
    F16c,
    /// FMA: the fused multiplies and adds.
    Fma,
    /// AVX2: the integers in vectors of 256 bits, the gathers, and the rest it adds.
    Avx2,
}

impl Feature {
    /// Every feature, each after those it builds on.
    pub const ALL: [Feature; 10] = [
        Feature::Popcnt,
        Feature::Lzcnt,
        Feature::Movbe,
        Feature::Bmi1,
        Feature::Bmi2,
        Feature::Sse42,
        Feature::Avx,
        Feature::F16c,
        Feature::Fma,
        Feature::Avx2,
    ];

    /// Its name, as Rust's `target_feature` and `is_x86_feature_detected!` write it.
    pub fn name(self) -> &'static str {
        match self {
            Feature::Popcnt => "popcnt",
            Feature::Lzcnt => "lzcnt",
            Feature::Movbe => "movbe",
            Feature::Bmi1 => "bmi1",
            Feature::Bmi2 => "bmi2",
            Feature::Sse42 => "sse4.2",
            Feature::Avx => "avx",
            Feature::F16c => "f16c",
            Feature::Fma => "fma",
            Feature::Avx2 => "avx2",
        }
    }
}

/// What an operand of a form may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A general register at this width.
    Gpr(Width),
    /// A vector register at this width.
    Vec(Width),
    /// Memory of this width.
    Mem(Width),
    /// Memory of this width at an address that is a multiple of it.
    Aligned(Width),
    /// An address, which the instruction reaches nothing at: `lea`'s.
    Addr,
    /// Memory at a vector of indices: elements of the first width, at indices in a vector of
    /// the second.
    Vsib(Width, Width),
    /// An integer for an operation of this width: of so many bits, of either sign, but at 64
    /// of 32, sign-extended.
    Imm(Width),
    /// An integer of 64 bits, of either sign.
    Imm64,
    /// An integer in this range.
    Range(i64, i64),
    /// A control byte of no bits but these, which a listing shows in hex.
    Control(u8),
    /// A label.
    Label,
}

impl Kind {
    /// The bytes of memory it reaches, if it is memory that does: an element's at a vector of
    /// indices.
    pub fn size(self) -> Option<Width> {
        match self {
            Kind::Mem(w) | Kind::Aligned(w) | Kind::Vsib(w, _) => Some(w),
            _ => None,
        }
    }

    /// Whether `operand` of an instruction is of this kind, `implicit` being the operands it
    /// does not write out, among which is the vector of indices of memory at one, at its width.
    fn takes<R: Register>(self, operand: &Operand<R>, implicit: &[Arg<R>]) -> bool {
        let vector_index = |m: &kernite_core::Mem<R>| match m.index {
            Some((r, _)) if r.class() == Class::Vec => implicit.iter().find_map(|a| match a.operand {
                Operand::Reg(i, w) if i == r => Some(w),
                _ => None,
            }),
            _ => None,
        };
        let int = |x: i64| self.fits(Shape::Int(x.into()));
        match (self, operand) {
            (Kind::Gpr(w), Operand::Reg(r, v)) => r.class() == Class::Gpr && w == *v,
            (Kind::Vec(w), Operand::Reg(r, v)) => r.class() == Class::Vec && w == *v,
            (Kind::Mem(w) | Kind::Aligned(w), Operand::Mem(m)) => m.width == w && vector_index(m).is_none(),
            (Kind::Addr, Operand::Mem(m)) => vector_index(m).is_none(),
            (Kind::Vsib(elem, index), Operand::Mem(m)) => m.width == elem && vector_index(m) == Some(index),
            (Kind::Imm(_) | Kind::Imm64 | Kind::Range(..) | Kind::Control(_), Operand::Imm(imm)) => int(imm.value),
            (Kind::Label, Operand::Label(_)) => true,
            _ => false,
        }
    }

    /// Whether the language's operand `shape` fits it.
    pub fn fits(self, shape: Shape) -> bool {
        match (self, shape) {
            (Kind::Gpr(w), Shape::Gpr(v)) | (Kind::Vec(w), Shape::Vec(v)) => w == v,
            (Kind::Mem(w) | Kind::Aligned(w), Shape::Mem { size, vsib: None, .. }) => size.is_none_or(|s| s == w),
            (Kind::Addr, Shape::Mem { size: None, vsib: None, .. }) => true,
            (Kind::Vsib(elem, index), Shape::Mem { size, data: None, vsib: Some(i) }) => {
                i == index && size.is_none_or(|s| s == elem)
            }
            (Kind::Control(mask), Shape::Int(x)) => (0..=255).contains(&x) && x & !i128::from(mask) == 0,
            (kind, Shape::Int(x)) => kind.ints().is_some_and(|(lo, hi)| (lo..=hi).contains(&x)),
            (Kind::Label, Shape::Label) => true,
            _ => false,
        }
    }

    /// The least and the most integer it takes, if it is an integer: for an operation of `N`
    /// bits, the least of `N` bits, signed, and the most, unsigned, but at 64, 32 bits, signed;
    /// for a control byte, 0 and every bit it reads.
    pub fn ints(self) -> Option<(i128, i128)> {
        match self {
            Kind::Imm(W64) => Some((i32::MIN.into(), i32::MAX.into())),
            Kind::Imm(w) => {
                let bits = w.bytes() * 8;
                Some((-(1 << (bits - 1)), (1 << bits) - 1))
            }
            Kind::Imm64 => Some((i64::MIN.into(), u64::MAX.into())),
            Kind::Range(lo, hi) => Some((lo.into(), hi.into())),
            Kind::Control(mask) => Some((0, mask.into())),
            _ => None,
        }
    }

    /// What it is, as a message names it: `a gpr's r32`, `4 bytes of memory`.
    fn noun(self) -> String {
        match self {
            Kind::Gpr(W64) => "a gpr".into(),
            Kind::Gpr(w) => format!("a gpr's r{}", w.bytes() * 8),
            Kind::Vec(W256) => "a ymm".into(),
            Kind::Vec(_) => "an xmm".into(),
            Kind::Mem(w) | Kind::Aligned(w) => format!("{} of memory", crate::bytes(w.bytes().into())),
            Kind::Addr => "memory".into(),
            Kind::Vsib(elem, index) => {
                let at = if index == W256 { "a ymm" } else { "an xmm" };
                format!("{} of memory at {at} of indices", crate::bytes(elem.bytes().into()))
            }
            Kind::Imm(W64) => "an integer of 32 bits, signed".into(),
            Kind::Imm(w) => format!("an integer of {} bits", w.bytes() * 8),
            Kind::Imm64 => "an integer of 64 bits".into(),
            Kind::Range(lo, hi) => format!("an integer, {lo} to {hi}"),
            Kind::Control(mask) if (u16::from(mask) + 1).is_power_of_two() => format!("a control byte, 0 to {mask}"),
            Kind::Control(mask) => format!("a control byte of no bits but those of {mask:#04x}"),
            Kind::Label => "a label".into(),
        }
    }

    /// What it is of: a register, memory, an integer or a label.
    fn sort(self) -> u8 {
        match self {
            Kind::Gpr(_) | Kind::Vec(_) => 0,
            Kind::Mem(_) | Kind::Aligned(_) | Kind::Addr | Kind::Vsib(..) => 1,
            Kind::Imm(_) | Kind::Imm64 | Kind::Range(..) | Kind::Control(_) => 2,
            Kind::Label => 3,
        }
    }
}

/// An operand of a form: what it may be, and how the instruction reaches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    /// What it may be.
    pub kind: Kind,
    /// Whether the instruction reads it, writes it, or both.
    pub access: Access,
}

/// A form of an instruction: the operands it takes, and what it needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Form {
    /// The instruction.
    pub op: Op,
    /// Its operands, in Intel's order.
    pub operands: Vec<Spec>,
    /// The CPU feature it needs past x86-64's own, if any.
    pub feature: Option<Feature>,
    /// The line of [`TABLE`] it is of.
    pub line: &'static str,
}

/// What the language gives an instruction in a place, as its forms tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// A general register, at the width the language sees it.
    Gpr(Width),
    /// A vector register, at the width the language sees it.
    Vec(Width),
    /// Memory: how much of it, if the language names that, as `dword[x]` does; how many bytes
    /// it holds, if it holds data; and the width of the vector of indices it is at, if it is.
    Mem {
        /// How much of it, as the language names it.
        size: Option<Width>,
        /// The bytes of data it holds.
        data: Option<usize>,
        /// The width of the vector of indices it is at.
        vsib: Option<Width>,
    },
    /// An integer known now.
    Int(i128),
    /// A label.
    Label,
}

impl Shape {
    /// What it is of, as [`Kind::sort`] says.
    fn sort(self) -> u8 {
        match self {
            Shape::Gpr(_) | Shape::Vec(_) => 0,
            Shape::Mem { .. } => 1,
            Shape::Int(_) => 2,
            Shape::Label => 3,
        }
    }
}

/// Why no form of an instruction fits the operands it is given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Miss {
    /// It takes so many operands, each number once, and none of them as many as it is given.
    Count(Vec<usize>),
    /// The operand at `at` fits none of the forms that fit furthest, which take `wants` there.
    Wants {
        /// Where.
        at: usize,
        /// What those forms take there, each once.
        wants: Vec<Kind>,
    },
    /// Memory at `at`, of no size, fits forms that reach each of `sizes` of it.
    Size {
        /// Where.
        at: usize,
        /// The sizes, from the least.
        sizes: Vec<Width>,
    },
}

/// Of `forms`, the one the operands `shapes` fit; else why none does. Where memory of no size
/// fits forms that reach different sizes of it, the one that reaches as much as the data it
/// holds, if it holds data, or none.
pub fn choose(forms: &'static [Form], shapes: &[Shape]) -> Result<&'static Form, Miss> {
    let mut fit: Vec<&'static Form> = vec![];
    let mut furthest: Option<(usize, Vec<Kind>)> = None;
    for form in forms.iter().filter(|f| f.operands.len() == shapes.len()) {
        let Some(at) = form.operands.iter().zip(shapes).position(|(s, &shape)| !s.kind.fits(shape)) else {
            fit.push(form);
            continue;
        };
        let kind = form.operands[at].kind;
        match &mut furthest {
            Some((far, wants)) if *far == at => {
                if !wants.contains(&kind) {
                    wants.push(kind);
                }
            }
            Some((far, _)) if *far > at => {}
            _ => furthest = Some((at, vec![kind])),
        }
    }
    if fit.is_empty() {
        return Err(match furthest {
            Some((at, wants)) => Miss::Wants { at, wants },
            None => {
                let mut counts: Vec<usize> = forms.iter().map(|f| f.operands.len()).collect();
                counts.sort();
                counts.dedup();
                Miss::Count(counts)
            }
        });
    }
    for (at, &shape) in shapes.iter().enumerate() {
        let Shape::Mem { size: None, data, .. } = shape else { continue };
        let mut sizes: Vec<Width> = fit.iter().filter_map(|f| f.operands[at].kind.size()).collect();
        sizes.sort();
        sizes.dedup();
        if sizes.len() < 2 {
            continue;
        }
        match data.and_then(|n| sizes.iter().copied().find(|w| w.bytes() as usize == n)) {
            Some(w) => fit.retain(|f| f.operands[at].kind.size() == Some(w)),
            None => return Err(Miss::Size { at, sizes }),
        }
    }
    Ok(fit[0])
}

/// What `wants`, kinds of operand forms take in a place, are, for a message about `given`,
/// which none of them fits: those of them of what `given` is of, if any are, a register, memory
/// or an integer, else all.
pub fn wanted(wants: &[Kind], given: Shape) -> String {
    let alike: Vec<Kind> = wants.iter().copied().filter(|k| k.sort() == given.sort()).collect();
    let wants = if alike.is_empty() { wants.to_vec() } else { alike };
    let mut parts: Vec<String> = vec![];
    let mut gprs: Vec<Width> =
        wants.iter().filter_map(|k| if let Kind::Gpr(w) = k { Some(*w) } else { None }).collect();
    gprs.sort_by(|a, b| b.cmp(a));
    gprs.dedup();
    let names = |ws: &[Width]| list(&ws.iter().map(|w| format!("r{}", w.bytes() * 8)).collect::<Vec<_>>());
    // A gpr seen more ways than one: `whole or as r32` alone, `(whole or as r32)` beside other parts.
    let views = match gprs.as_slice() {
        [] => None,
        [w] => {
            parts.push(Kind::Gpr(*w).noun());
            None
        }
        [W64, rest @ ..] => {
            Some((format!("a gpr, whole or as {}", names(rest)), format!("whole or as {}", names(rest))))
        }
        views => Some((format!("a gpr's {}", names(views)), format!("as {}", names(views)))),
    };
    if views.is_some() {
        parts.push(String::new());
    }
    for w in [W256, W128] {
        if wants.contains(&Kind::Vec(w)) {
            parts.push(Kind::Vec(w).noun());
        }
    }
    let mut sizes: Vec<Width> =
        wants.iter().filter_map(|k| matches!(k, Kind::Mem(_) | Kind::Aligned(_)).then(|| k.size()).flatten()).collect();
    sizes.sort();
    sizes.dedup();
    match sizes.as_slice() {
        [] => {}
        [w] => parts.push(Kind::Mem(*w).noun()),
        sizes => {
            let n: Vec<String> = sizes.iter().map(|w| w.bytes().to_string()).collect();
            parts.push(format!("memory of {} bytes", list(&n)));
        }
    }
    for k in &wants {
        let noun = match k {
            Kind::Gpr(_) | Kind::Vec(_) | Kind::Mem(_) | Kind::Aligned(_) => continue,
            k => k.noun(),
        };
        if !parts.contains(&noun) {
            parts.push(noun);
        }
    }
    if let Some((alone, beside)) = views {
        parts[0] = if parts.len() == 1 { alone } else { format!("a gpr ({beside})") };
    }
    list(&parts)
}

/// `a`, `a or b`, `a, b or c`.
pub(crate) fn list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} or {last}", rest.join(", ")),
    }
}

/// Intel's word for memory of width `w`, which the language writes before brackets.
pub fn word(w: Width) -> &'static str {
    match w {
        W8 => "byte",
        W16 => "word",
        W32 => "dword",
        W64 => "qword",
        W128 => "xmmword",
        W256 => "ymmword",
    }
}

/// The width Intel's word `word` names: `dword` 32 bits.
pub fn width_of(word: &str) -> Option<Width> {
    [W8, W16, W32, W64, W128, W256].into_iter().find(|&w| self::word(w) == word)
}

/// The forms of the instruction the language calls `name`, if kernite knows it.
pub fn of(name: &str) -> Option<&'static [Form]> {
    forms().get(name).map(Vec::as_slice)
}

/// The names of the instructions kernite knows, in order.
pub fn names() -> impl Iterator<Item = &'static str> {
    forms().keys().copied()
}

/// The CPU feature an instruction kernite wrote needs past x86-64's own, if any: `None` for one
/// the table does not have, as `ret`.
///
/// # Panics
///
/// If the table has the instruction, but in no form its operands fit, which is a bug of
/// whoever built it.
pub fn feature<R: Register>(op: Op, args: &[Arg<R>], implicit: &[Arg<R>]) -> Option<Feature> {
    let forms = of(op.name())?;
    let form = forms.iter().find(|f| {
        f.operands.len() == args.len() && f.operands.iter().zip(args).all(|(s, a)| s.kind.takes(&a.operand, implicit))
    });
    form.unwrap_or_else(|| panic!("`{}` in no form x86 has: {args:?}", op.name())).feature
}

/// Every form, parsed from [`TABLE`] once, by name.
fn forms() -> &'static BTreeMap<&'static str, Vec<Form>> {
    static FORMS: OnceLock<BTreeMap<&'static str, Vec<Form>>> = OnceLock::new();
    FORMS.get_or_init(|| parse(TABLE))
}

fn parse(text: &'static str) -> BTreeMap<&'static str, Vec<Form>> {
    let mut macros: Vec<(&str, &str)> = vec![];
    let mut out: BTreeMap<&'static str, Vec<Form>> = BTreeMap::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(def) = line.strip_prefix('$') {
            let (name, value) = def.split_once(" = ").unwrap_or_else(|| panic!("`{line}` defines `$name = {{..}}`"));
            macros.push((name, value));
            continue;
        }
        let (head, body) = line.split_once(" : ").unwrap_or_else(|| panic!("`{line}` has no ` : `"));
        let (names, widths, feature) = head_of(head, line);
        let names: Vec<&'static str> = names
            .iter()
            .flat_map(|n| {
                let n = macros.iter().fold(n.to_string(), |n, (m, value)| n.replace(&format!("${m}"), value));
                expand(&n)
            })
            .map(|n| &*Box::leak(n.into_boxed_str()))
            .collect();
        for &(n, own) in &widths {
            let feature = own.or(feature);
            for form in body.split(" ; ") {
                for operands in operands_of(form, n, line) {
                    for &name in &names {
                        let form = Form { op: Op(name), operands: operands.clone(), feature, line };
                        out.entry(name).or_default().push(form);
                    }
                }
            }
        }
    }
    out
}

/// The widths a line gives its forms at, each with its own feature if it has one.
type Widths = Vec<(u32, Option<Feature>)>;

/// A line's names, its widths, and its feature: one width, 0, if it gives none.
fn head_of(head: &'static str, line: &str) -> (Vec<&'static str>, Widths, Option<Feature>) {
    let (names, widths, rest) = match head.split_once('[') {
        Some((names, rest)) => {
            let (widths, rest) = rest.split_once(']').unwrap_or_else(|| panic!("`{line}` has no `]`"));
            let widths = widths
                .split_whitespace()
                .map(|w| match w.split_once('@') {
                    Some((w, f)) => (w, Some(feature_of(f, line))),
                    None => (w, None),
                })
                .map(|(w, f)| (w.parse::<u32>().unwrap_or_else(|_| panic!("`{line}`: no width `{w}`")), f))
                .collect();
            (names, widths, rest)
        }
        None => match head.split_once('@') {
            Some((names, _)) => (names, vec![(0, None)], &head[names.len()..]),
            None => (head, vec![(0, None)], ""),
        },
    };
    let feature = rest.trim().strip_prefix('@').map(|f| feature_of(f, line));
    (names.split_whitespace().collect(), widths, feature)
}

fn feature_of(name: &str, line: &str) -> Feature {
    Feature::ALL.into_iter().find(|f| f.name() == name).unwrap_or_else(|| panic!("`{line}`: no feature `{name}`"))
}

/// `name` with each of its `{a,b}` written out: every name it stands for.
fn expand(name: &str) -> Vec<String> {
    let Some(open) = name.find('{') else { return vec![name.to_string()] };
    let close = open + name[open..].find('}').unwrap_or_else(|| panic!("`{name}` has no `}}`"));
    let rest = expand(&name[close + 1..]);
    name[open + 1..close]
        .split(',')
        .flat_map(|alt| rest.iter().map(move |r| format!("{}{alt}{r}", &name[..open])))
        .collect()
}

/// The operands of each form `form` stands for, at width `n`: one for each of its operands'
/// alternatives.
fn operands_of(form: &str, n: u32, line: &str) -> Vec<Vec<Spec>> {
    if form == "()" {
        return vec![vec![]];
    }
    let mut out: Vec<Vec<Spec>> = vec![vec![]];
    for operand in form.split(", ") {
        let (access, alts) = match operand.as_bytes()[0] {
            b'=' => (Write, &operand[1..]),
            b'+' => (ReadWrite, &operand[1..]),
            b'~' => (Ignored, &operand[1..]),
            _ => (Read, operand),
        };
        let specs: Vec<Spec> = alts.split('/').map(|alt| Spec { kind: kind_of(alt, n, line), access }).collect();
        out = out.iter().flat_map(|head| specs.iter().map(move |s| [head.as_slice(), &[*s]].concat())).collect();
    }
    out
}

/// The kind the token `t` names at width `n`, which `N` stands for.
fn kind_of(t: &str, n: u32, line: &str) -> Kind {
    let t = match t.strip_suffix('N') {
        Some(stem) => {
            assert!(n != 0, "`{line}` has `N` and no widths");
            format!("{stem}{n}")
        }
        None => t.to_string(),
    };
    let width = |bits: &str| match bits {
        "8" => W8,
        "16" => W16,
        "32" => W32,
        "64" => W64,
        "128" => W128,
        "256" => W256,
        _ => panic!("`{line}`: no width `{bits}` in `{t}`"),
    };
    if let Some(c) = t.strip_prefix("#0x") {
        return Kind::Control(u8::from_str_radix(c, 16).unwrap_or_else(|_| panic!("`{line}`: no byte `{t}`")));
    }
    if let Some(r) = t.strip_prefix('#') {
        let (lo, hi) = r.split_once('-').unwrap_or_else(|| panic!("`{line}`: no range `{t}`"));
        let parse = |x: &str| x.parse::<i64>().unwrap_or_else(|_| panic!("`{line}`: no range `{t}`"));
        return Kind::Range(parse(lo), parse(hi));
    }
    match t.as_str() {
        "xmm" => return Kind::Vec(W128),
        "ymm" => return Kind::Vec(W256),
        "m" => return Kind::Addr,
        "imm64" => return Kind::Imm64,
        "lab" => return Kind::Label,
        "vm32x" => return Kind::Vsib(W32, W128),
        "vm32y" => return Kind::Vsib(W32, W256),
        "vm64x" => return Kind::Vsib(W64, W128),
        "vm64y" => return Kind::Vsib(W64, W256),
        _ => {}
    }
    let (letter, bits) = t.split_at(1);
    match letter {
        "r" => Kind::Gpr(width(bits)),
        "v" => Kind::Vec(width(bits)),
        "m" => Kind::Mem(width(bits)),
        "a" => Kind::Aligned(width(bits)),
        "i" => Kind::Imm(width(bits)),
        _ => panic!("`{line}`: no operand `{t}`"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_parses_and_names_each_instruction_once_per_form() {
        let all: usize = forms().values().map(Vec::len).sum();
        assert!(all > 2500, "{all} forms");
        for (name, forms) in forms() {
            for (i, a) in forms.iter().enumerate() {
                for b in &forms[..i] {
                    let kinds = |f: &Form| f.operands.iter().map(|s| s.kind).collect::<Vec<_>>();
                    assert_ne!(kinds(a), kinds(b), "`{name}` twice: `{}` and `{}`", a.line, b.line);
                }
            }
        }
        assert_eq!(of("cmovnle").map(<[Form]>::len), Some(6));
        assert_eq!(of("vcmpngt_uqps").map(<[Form]>::len), Some(4));
        assert_eq!(of("vfnmsub231pd").map(<[Form]>::len), Some(4));
        assert!(of("lock.xadd").is_some() && of("mul").is_none());
    }

    #[test]
    fn a_name_expands_to_every_name_it_stands_for() {
        assert_eq!(expand("vf{madd,msub}{132,231}ps"), ["vfmadd132ps", "vfmadd231ps", "vfmsub132ps", "vfmsub231ps"]);
        assert_eq!(expand("pause"), ["pause"]);
    }

    /// Forms that operands may fit at once differ only in how much memory they reach, which
    /// [`choose`] asks for, or in an immediate, which prints the same.
    #[test]
    fn forms_that_fit_the_same_operands_print_them_the_same() {
        let overlap = |a: Kind, b: Kind| match (a, b) {
            (Kind::Gpr(x), Kind::Gpr(y)) | (Kind::Vec(x), Kind::Vec(y)) => x == y,
            (Kind::Mem(_) | Kind::Aligned(_) | Kind::Addr, Kind::Mem(_) | Kind::Aligned(_) | Kind::Addr) => true,
            (Kind::Vsib(_, x), Kind::Vsib(_, y)) => x == y,
            (a, b) => a.sort() == 2 && b.sort() == 2 || a == b,
        };
        for (name, forms) in forms() {
            for (i, a) in forms.iter().enumerate() {
                for b in &forms[..i] {
                    let both = a.operands.len() == b.operands.len()
                        && a.operands.iter().zip(&b.operands).all(|(x, y)| overlap(x.kind, y.kind));
                    if !both {
                        continue;
                    }
                    let differ = a.operands.iter().zip(&b.operands).any(|(x, y)| x.kind.size() != y.kind.size());
                    assert!(
                        differ
                            || (a.feature == b.feature
                                && a.operands.iter().zip(&b.operands).all(|(x, y)| x.access == y.access)),
                        "`{name}`: `{}` and `{}` fit the same operands",
                        a.line,
                        b.line
                    );
                }
            }
        }
    }

    #[test]
    fn memory_of_no_size_is_the_datas_or_named() {
        let movzx = of("movzx").unwrap();
        let mem = |data| Shape::Mem { size: None, data, vsib: None };
        let r32 = Shape::Gpr(W32);
        assert_eq!(choose(movzx, &[r32, mem(None)]), Err(Miss::Size { at: 1, sizes: vec![W8, W16] }));
        let f = choose(movzx, &[r32, mem(Some(2))]).unwrap();
        assert_eq!(f.operands[1].kind, Kind::Mem(W16));
        let sized = Shape::Mem { size: Some(W8), data: None, vsib: None };
        assert_eq!(choose(movzx, &[r32, sized]).unwrap().operands[1].kind, Kind::Mem(W8));
    }

    #[test]
    fn a_miss_names_what_the_forms_that_fit_furthest_want() {
        let vaddps = of("vaddps").unwrap();
        let (y, x) = (Shape::Vec(W256), Shape::Vec(W128));
        let miss = choose(vaddps, &[y, x, y]).unwrap_err();
        assert_eq!(miss, Miss::Wants { at: 1, wants: vec![Kind::Vec(W256)] });
        assert_eq!(choose(vaddps, &[y, y]).unwrap_err(), Miss::Count(vec![3]));
        let Miss::Wants { wants, .. } = choose(of("add").unwrap(), &[Shape::Vec(W128), Shape::Int(1)]).unwrap_err()
        else {
            panic!()
        };
        assert_eq!(wanted(&wants, Shape::Vec(W128)), "a gpr, whole or as r32, r16 or r8");
        assert_eq!(wanted(&wants, Shape::Int(1)), "a gpr (whole or as r32, r16 or r8) or memory of 1, 2, 4 or 8 bytes");
        let Miss::Wants { wants, .. } = choose(of("vmovd").unwrap(), &[Shape::Gpr(W16), Shape::Vec(W128)]).unwrap_err()
        else {
            panic!()
        };
        assert_eq!(wanted(&wants, Shape::Gpr(W16)), "a gpr (whole or as r32) or an xmm");
        let Miss::Wants { wants, .. } = choose(of("movzx").unwrap(), &[Shape::Gpr(W32), Shape::Int(1)]).unwrap_err()
        else {
            panic!()
        };
        assert_eq!(wanted(&wants, Shape::Int(1)), "a gpr (as r16 or r8) or memory of 1 or 2 bytes");
        let shl = of("shl").unwrap();
        let Miss::Wants { at: 1, wants } = choose(shl, &[Shape::Gpr(W64), Shape::Int(64)]).unwrap_err() else {
            panic!()
        };
        assert_eq!(wanted(&wants, Shape::Int(64)), "an integer, 0 to 63");
        assert_eq!(Kind::Control(0xbb).noun(), "a control byte of no bits but those of 0xbb");
    }
}
