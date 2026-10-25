//! The instructions of PTX that kgen knows: those the GPUs from Fermi, `sm_20`, to Ampere,
//! `sm_86`, run, but `call`, written as the PTX ISA writes them, one [`Form`] each, and the
//! special registers they read.
//!
//! A form is a line of [`TABLE`]: the words of an instruction's name, some to choose from
//! and some to leave out, its operands, and the version of PTX and the GPU it and each of
//! its words need. [`find`] gives the forms a name is of, or why it is of none.

use std::fmt;
use std::sync::OnceLock;

use kgen_core::Access;

use crate::{GPUS, Kind, Needs, Ty};

/// The forms, a line each.
///
/// A form's name is PTX's, its words in PTX's order: `.w` is a word it has, `{.a|.b}` one it
/// may have, `(.a|.b)` one of these it has, and `$m` the elements a line `$m = ..` stands for.
/// A word, or a choice of words, that is a type is one of the instruction's types, which its
/// operands name by place: `T` is the first, then `U`, `V` and `W`. `@7.8` after a word or a
/// form is the version of PTX it needs, as ptxas takes it, which every form says: `@1.0` is
/// any. `@sm80` is the GPU, for one newer than Fermi, `sm_20`, and `@<sm70/6.4` a GPU older
/// than that in a module of PTX 6.4 or newer. Each address is in the state space the name
/// gives in its place, the first in the first, as `cp.async.ca.shared.global` writes shared
/// memory and reads global; one in none, a generic address, needs PTX 2.0 besides. `~round`,
/// `~half` and `~cmp` say what leaving an element out leaves unsaid.
///
/// After ` : ` are the operands. One is a type, `T` or `u32`, with `=` before it for one the
/// instruction writes, `%` for one that must be a variable, `_` for one `_` may stand for,
/// `!` for a predicate it may read inverted, `-` for one it may negate, `$` for one a special
/// register may be, and `&` for memory the kernel declares, which stands for its address;
/// and after it `*2` for so many taken together, `u32+T*3` for four of which the first is a
/// `u32`, `*v` for as many as `.v2` or `.v4` say, of which `_` stands only for a register of a
/// vector, and `{.bh}` for one that may be a part of a variable, as a video instruction
/// selects it, or `(.bh)` one that must be: `.bh` a byte or a half, `r.b0` or `r.h1`; `.hxy`
/// two of the halves of two sources together, `r.h32`, and `.bxyzw` four of their bytes,
/// `r.b7654`; `.hmask` the halves of a result it writes, `r.h10`, and `.bmask` its bytes,
/// `r.b310`. `a|b` is two written at once, and `a[|b]` one or two. `u32[0,15]` is one from 0
/// to 15, if it is one known now, `u32/32` a multiple of 32, and `f32(0,1]` a fraction, more
/// than 0 and at most 1. `[R]`, `[W]` and `[X]` are memory read, written, or both, and
/// `[%u64,T*2]` a texture or a surface by its handle, in brackets with the coordinates in it;
/// `lab` is a label, and `labs` labels taken together; `#` is an integer known now, `#8` one
/// of 8 bits, `#=4|8|16` one of those, and `#^23` a power of two to 2^23. `{w}` before an
/// operand has it only with the word `w`, `@4.3` after it is what it needs where it is
/// there, and `?` lets it be left out; `$m`, of no type, is the operands a line `$m : ..`
/// stands for.
/// Last, `+wide` lets a register outside brackets be wider than the instruction's type, as
/// `ld`, `st` and `cvt` do; `+vec` holds a vector to 128 bits; `+cvt` holds a conversion to
/// PTX's rules for rounding and saturating, and gives `.ftz` the version it needs there;
/// `+b128sys` gives `.sys` with `.b128` the version it needs; `+neg` negates the product of
/// the first two operands it may negate, as negating one of them does, or the third, not
/// both, as `vmad`; and `+tex` has a texture return four, `.v4`, but two of `.f16x2`, `.v2`.
pub const TABLE: &str = r#"
$int = (.u16|.u32|.u64|.s16|.s32|.s64)
$bits = (.b16|.b32|.b64)
$num = (.b16|.b32|.b64|.u16|.u32|.u64|.s16|.s32|.s64|.f32|.f64)
$rnd = (.rn|.rz|.rm|.rp)~round
$cmpb = (.eq|.ne)~cmp
$cmps = (.lt|.eq|.ne|.le|.gt|.ge)~cmp
$cmpu = (.lt|.eq|.ne|.le|.gt|.ge|.lo|.ls|.hi|.hs)~cmp
$cmpf = (.lt|.eq|.ne|.le|.gt|.ge|.equ|.neu|.ltu|.leu|.gtu|.geu|.num|.nan)~cmp
$bool = (.and|.or|.xor)
$cc = (.u32|.s32|.u64@4.3|.s64@4.3)
$space = (.const@3.1|.global|.local|.shared|.shared::cta@7.8|.param@7.7@sm70|.param::entry@8.3@sm70)
$ldss = (.const|.local|.param|.param::entry@8.3|.shared|.shared::cta@7.8)
$shared = (.shared|.shared::cta@7.8)
$cop = {.ca@2.0|.cg@2.0|.cs@2.0|.lu@2.0|.cv@2.0}
$scop = {.wb@2.0|.cg@2.0|.cs@2.0|.wt@2.0}
$l1 = {.L1::evict_normal@7.4@sm70|.L1::evict_unchanged@7.4@sm70|.L1::evict_first@7.4@sm70|.L1::evict_last@7.4@sm70|.L1::no_allocate@7.4@sm70}
$hint = {.L2::cache_hint@7.4@sm80}
$size = {.L2::64B@7.4@sm75|.L2::128B@7.4@sm75|.L2::256B@7.4@sm80}
$vec = {.v2|.v4}
$ty = (.b8|.b16|.b32|.b64|.b128@8.3@sm70|.u8|.u16|.u32|.u64|.s8|.s16|.s32|.s64|.f32|.f64)
$ty64 = (.b8|.b16|.b32|.b64|.u8|.u16|.u32|.u64|.s8|.s16|.s32|.s64|.f32|.f64)
$scope = (.cta|.gpu|.sys)
$sem = {.relaxed@6.0@sm70|.acquire@6.0@sm70|.release@6.0@sm70|.acq_rel@6.0@sm70}
$rsem = {.relaxed@6.0@sm70|.release@6.0@sm70}
$ascope = {.cta@5.0@sm60|.gpu@5.0@sm60|.sys@5.0@sm60}
$cvty = (.u8|.u16|.u32|.u64|.s8|.s16|.s32|.s64|.f16|.f32|.f64)
$prio = (.L2::evict_last|.L2::evict_normal|.L2::evict_first|.L2::evict_unchanged)
$prio2 = {.L2::evict_first|.L2::evict_unchanged}
$cta = {.cta@7.8}
$shfl = (.up|.down|.bfly|.idx)
$gsh = {.global|.shared|.shared::cta@7.8}
$fshape = (.m16n16k16|.m8n32k16|.m32n8k16)
$lay = (.row|.col)
$vty = (.u32|.s32)(.u32|.s32)(.u32|.s32)
$vop = (.add|.min|.max)
$bty = (.b8|.b16|.b32|.b64)
$sucop = {.ca|.cg|.cs|.cv}
$clamp = (.trap|.clamp|.zero)
$tdt = (.u32|.s32|.f16@4.2@sm53|.f32|.f16x2@4.2@sm53)
$texd : {v4}=_T*4[|=pred@7.1@sm60], {v2}=T*2[|=pred@7.1@sm60]

# Integers.
add$int : =T, T, T @1.0
add.sat.s32 : =T, T, T @1.0
sub$int : =T, T, T @1.0
sub.sat.s32 : =T, T, T @1.0
mul(.hi|.lo)~half$int : =T, T, T @1.0
mul.wide~half.u16 : =u32, T, T @1.0
mul.wide~half.u32 : =u64, T, T @1.0
mul.wide~half.s16 : =s32, T, T @1.0
mul.wide~half.s32 : =s64, T, T @1.0
mad(.hi|.lo)~half$int : =T, T, T, T @1.0
mad.hi.sat.s32 : =T, T, T, T @1.0
mad.wide~half.u16 : =u32, T, T, u32 @1.0
mad.wide~half.u32 : =u64, T, T, u64 @1.0
mad.wide~half.s16 : =s32, T, T, s32 @1.0
mad.wide~half.s32 : =s64, T, T, s64 @1.0
clmad(.hi|.lo).u64 : =T, T, T, T @9.3 @sm80
mul24(.hi|.lo)(.u32|.s32) : =T, T, T @1.0
mad24(.hi|.lo)(.u32|.s32) : =T, T, T, T @1.0
mad24.hi.sat.s32 : =T, T, T, T @1.0
sad$int : =T, T, T, T @1.0
div$int : =T, T, T @1.0
rem$int : =T, T, T @1.0
abs(.s16|.s32|.s64) : =T, T @1.0
neg(.s16|.s32|.s64) : =T, T @1.0
min$int : =T, T, T @1.0
max$int : =T, T, T @1.0
popc(.b32|.b64) : =u32, T @2.0
clz(.b32|.b64) : =u32, T @2.0
bfind{.shiftamt}(.u32|.u64|.s32|.s64) : =u32, T @2.0
fns.b32 : =T, T, T, s32 @6.0 @sm30
brev(.b32|.b64) : =T, T @2.0
bfe(.u32|.u64|.s32|.s64) : =T, T, u32, u32 @2.0
bfi(.b32|.b64) : =T, T, T, u32, u32 @2.0
szext(.clamp|.wrap)(.u32|.s32) : =T, T, u32 @7.6 @sm70
bmsk(.clamp|.wrap).b32 : =T, u32, u32 @7.6 @sm70
dp4a.u32.u32 : =u32, T, U, u32 @5.0 @sm61
dp4a.u32.s32 : =s32, T, U, s32 @5.0 @sm61
dp4a.s32.u32 : =s32, T, U, s32 @5.0 @sm61
dp4a.s32.s32 : =s32, T, U, s32 @5.0 @sm61
dp2a(.lo|.hi).u32.u32 : =u32, T, U, u32 @5.0 @sm61
dp2a(.lo|.hi).u32.s32 : =s32, T, U, s32 @5.0 @sm61
dp2a(.lo|.hi).s32.u32 : =s32, T, U, s32 @5.0 @sm61
dp2a(.lo|.hi).s32.s32 : =s32, T, U, s32 @5.0 @sm61
add.cc$cc : =T, T, T @1.2
addc(.u32|.s32|.u64|.s64) : =T, T, T @1.2
addc.cc$cc : =T, T, T @1.2
sub.cc$cc : =T, T, T @1.3
subc(.u32|.s32|.u64|.s64) : =T, T, T @1.3
subc.cc$cc : =T, T, T @1.3
mad(.hi|.lo).cc$cc : =T, T, T, T @3.0
madc(.hi|.lo)(.u32|.s32|.u64|.s64) : =T, T, T, T @3.0
madc(.hi|.lo).cc$cc : =T, T, T, T @3.0

# Floats.
testp(.finite|.infinite|.number|.notanumber|.normal|.subnormal)(.f32|.f64) : =pred, T @2.0
copysign(.f32|.f64) : =T, T, T @2.0
add(.rn|.rz|.rm@2.0|.rp@2.0)~round{.ftz@1.4}{.sat}.f32 : =T, T, T @1.0
add(.rn|.rz|.rm@1.2|.rp@1.2)~round.f64 : =T, T, T @1.0
sub(.rn|.rz|.rm@2.0|.rp@2.0)~round{.ftz@1.4}{.sat}.f32 : =T, T, T @1.0
sub(.rn|.rz|.rm@1.2|.rp@1.2)~round.f64 : =T, T, T @1.0
mul(.rn|.rz|.rm@2.0|.rp@2.0)~round{.ftz@1.4}{.sat}.f32 : =T, T, T @1.0
mul(.rn|.rz|.rm@1.2|.rp@1.2)~round.f64 : =T, T, T @1.0
fma$rnd{.ftz}{.sat}.f32 : =T, T, T, T @2.0
fma$rnd.f64 : =T, T, T, T @1.4
mad$rnd{.ftz}{.sat}.f32 : =T, T, T, T @2.0
mad(.rn|.rz|.rm@1.2|.rp@1.2)~round.f64 : =T, T, T, T @1.0
div.approx{.ftz}.f32 : =T, T, T @1.4
div.full{.ftz}.f32 : =T, T, T @1.4
div$rnd{.ftz}.f32 : =T, T, T @2.0
div(.rn|.rz@2.0|.rm@2.0|.rp@2.0)~round.f64 : =T, T, T @1.0
abs{.ftz@1.4}.f32 : =T, T @1.0
abs.f64 : =T, T @1.0
neg{.ftz@1.4}.f32 : =T, T @1.0
neg.f64 : =T, T @1.0
min{.ftz@1.4}{.NaN@7.0@sm80}{.xorsign.abs@7.2@sm86}.f32 : =T, T, T @1.0
min.f64 : =T, T, T @1.0
max{.ftz@1.4}{.NaN@7.0@sm80}{.xorsign.abs@7.2@sm86}.f32 : =T, T, T @1.0
max.f64 : =T, T, T @1.0
rcp.approx{.ftz}.f32 : =T, T @1.4
rcp$rnd{.ftz}.f32 : =T, T @2.0
rcp(.rn|.rz@2.0|.rm@2.0|.rp@2.0)~round.f64 : =T, T @1.4
rcp.approx.ftz.f64 : =T, T @2.1
sqrt.approx{.ftz}.f32 : =T, T @1.4
sqrt$rnd{.ftz}.f32 : =T, T @2.0
sqrt(.rn|.rz@2.0|.rm@2.0|.rp@2.0)~round.f64 : =T, T @1.0
rsqrt.approx{.ftz}.f32 : =T, T @1.4
rsqrt.approx.f64 : =T, T @1.4
rsqrt.approx.ftz.f64 : =T, T @4.0
sin.approx{.ftz}.f32 : =T, T @1.4
cos.approx{.ftz}.f32 : =T, T @1.4
lg2.approx{.ftz}.f32 : =T, T @1.4
ex2.approx{.ftz}.f32 : =T, T @1.4
tanh.approx.f32 : =T, T @7.0 @sm75

# Floats of 16 bits.
add.rn~round{.ftz}{.sat}(.f16|.f16x2) : =T, T, T @4.2 @sm53
sub.rn~round{.ftz}{.sat}(.f16|.f16x2) : =T, T, T @4.2 @sm53
mul.rn~round{.ftz}{.sat}(.f16|.f16x2) : =T, T, T @4.2 @sm53
fma.rn~round{.ftz}{.sat}(.f16|.f16x2) : =T, T, T, T @4.2 @sm53
fma.rn~round{.ftz}.relu(.f16|.f16x2) : =T, T, T, T @7.0 @sm80
fma.rn~round{.relu}(.bf16|.bf16x2) : =T, T, T, T @7.0 @sm80
neg{.ftz}(.f16|.f16x2) : =T, T @6.0 @sm53
neg(.bf16|.bf16x2) : =T, T @7.0 @sm80
abs{.ftz}(.f16|.f16x2) : =T, T @6.5 @sm53
abs(.bf16|.bf16x2) : =T, T @7.0 @sm80
min{.ftz}{.NaN}{.xorsign.abs@7.2@sm86}(.f16|.f16x2) : =T, T, T @7.0 @sm80
min{.NaN}{.xorsign.abs@7.2@sm86}(.bf16|.bf16x2) : =T, T, T @7.0 @sm80
max{.ftz}{.NaN}{.xorsign.abs@7.2@sm86}(.f16|.f16x2) : =T, T, T @7.0 @sm80
max{.NaN}{.xorsign.abs@7.2@sm86}(.bf16|.bf16x2) : =T, T, T @7.0 @sm80
tanh.approx(.f16|.f16x2) : =T, T @7.0 @sm75
ex2.approx(.f16|.f16x2) : =T, T @7.0 @sm75

# Comparisons and selection.
setp$cmpb$bits : =_pred[|=_pred], T, T @1.0
setp$cmpb$bool$bits : =_pred[|=_pred], T, T, !pred @1.0
setp$cmpu(.u16|.u32|.u64) : =_pred[|=_pred], T, T @1.0
setp$cmpu$bool(.u16|.u32|.u64) : =_pred[|=_pred], T, T, !pred @1.0
setp$cmps(.s16|.s32|.s64) : =_pred[|=_pred], T, T @1.0
setp$cmps$bool(.s16|.s32|.s64) : =_pred[|=_pred], T, T, !pred @1.0
setp$cmpf{.ftz@1.4}.f32 : =_pred[|=_pred], T, T @1.0
setp$cmpf$bool{.ftz@1.4}.f32 : =_pred[|=_pred], T, T, !pred @1.0
setp$cmpf.f64 : =_pred[|=_pred], T, T @1.0
setp$cmpf$bool.f64 : =_pred[|=_pred], T, T, !pred @1.0
setp$cmpf{.ftz}.f16 : =pred, T, T @4.2 @sm53
setp$cmpf$bool{.ftz}.f16 : =pred, T, T, !pred @4.2 @sm53
setp$cmpf{.ftz}.f16x2 : =pred|=pred, T, T @4.2 @sm53
setp$cmpf$bool{.ftz}.f16x2 : =pred|=pred, T, T, !pred @4.2 @sm53
set$cmpb(.u32|.s32|.f32)$bits : =T, U, U @1.0
set$cmpb$bool(.u32|.s32|.f32)$bits : =T, U, U, !pred @1.0
set$cmpu(.u32|.s32|.f32)(.u16|.u32|.u64) : =T, U, U @1.0
set$cmpu$bool(.u32|.s32|.f32)(.u16|.u32|.u64) : =T, U, U, !pred @1.0
set$cmps(.u32|.s32|.f32)(.s16|.s32|.s64) : =T, U, U @1.0
set$cmps$bool(.u32|.s32|.f32)(.s16|.s32|.s64) : =T, U, U, !pred @1.0
set$cmpf{.ftz@1.4}(.u32|.s32|.f32).f32 : =T, U, U @1.0
set$cmpf$bool{.ftz@1.4}(.u32|.s32|.f32).f32 : =T, U, U, !pred @1.0
set$cmpf(.u32|.s32|.f32).f64 : =T, U, U @1.0
set$cmpf$bool(.u32|.s32|.f32).f64 : =T, U, U, !pred @1.0
set$cmpb.f16$bits : =T, U, U @4.2 @sm53
set$cmpb$bool.f16$bits : =T, U, U, !pred @4.2 @sm53
set$cmpu.f16(.u16|.u32|.u64) : =T, U, U @4.2 @sm53
set$cmpu$bool.f16(.u16|.u32|.u64) : =T, U, U, !pred @4.2 @sm53
set$cmps.f16(.s16|.s32|.s64) : =T, U, U @4.2 @sm53
set$cmps$bool.f16(.s16|.s32|.s64) : =T, U, U, !pred @4.2 @sm53
set$cmpf{.ftz}.f16(.f16|.f32) : =T, U, U @4.2 @sm53
set$cmpf$bool{.ftz}.f16(.f16|.f32) : =T, U, U, !pred @4.2 @sm53
set$cmpf.f16.f64 : =T, U, U @4.2 @sm53
set$cmpf$bool.f16.f64 : =T, U, U, !pred @4.2 @sm53
set$cmpf{.ftz}(.u16|.s16|.u32|.s32).f16 : =T, U, U @6.5 @sm53
set$cmpf$bool{.ftz}(.u16|.s16|.u32|.s32).f16 : =T, U, U, !pred @6.5 @sm53
set$cmpf{.ftz}(.f16x2|.u32@6.5|.s32@6.5).f16x2 : =T, U, U @4.2 @sm53
set$cmpf$bool{.ftz}(.f16x2|.u32@6.5|.s32@6.5).f16x2 : =T, U, U, !pred @4.2 @sm53
selp$num : =T, T, T, pred @1.0
slct$num.s32 : =T, T, T, U @1.0
slct{.ftz@1.4}$num.f32 : =T, T, T, U @1.0

# Logic and shifts.
and(.pred|.b16|.b32|.b64) : =T, T, T @1.0
or(.pred|.b16|.b32|.b64) : =T, T, T @1.0
xor(.pred|.b16|.b32|.b64) : =T, T, T @1.0
not(.pred|.b16|.b32|.b64) : =T, T @1.0
cnot$bits : =T, T @1.0
lop3.b32 : =T, T, T, T, #8 @4.3 @sm50
lop3(.or|.and).b32 : =_T|=pred, T, T, T, #8, pred @8.2 @sm70
shf(.l|.r)(.clamp|.wrap).b32 : =T, T, T, u32 @3.1 @sm32
shl$bits : =T, T, u32 @1.0
shr(.b16|.b32|.b64|.u16|.u32|.u64|.s16|.s32|.s64) : =T, T, u32 @1.0

# Video: arithmetic on bytes and halves of 32-bit variables. A source may be a part of one,
# `a.b0`, which it extends as its type says. A scalar one works the last operand in, `.add`,
# or writes a part of its result, `d.h1`, and the rest from the last operand. A SIMD one
# selects what it works on from the halves or bytes of its two sources together, `a.h32`
# being two of the second, and adds the last operand, `.add`, or writes those of its result
# that `d.h1` says and the rest from the last; and takes no value known now.
vadd$vty{.sat} : =T, U{.bh}, V{.bh} @2.0
vadd$vty{.sat}$vop : =T, U{.bh}, V{.bh}, T @2.0
vadd$vty{.sat} : =T(.bh), U{.bh}, V{.bh}, T @2.0
vsub$vty{.sat} : =T, U{.bh}, V{.bh} @2.0
vsub$vty{.sat}$vop : =T, U{.bh}, V{.bh}, T @2.0
vsub$vty{.sat} : =T(.bh), U{.bh}, V{.bh}, T @2.0
vabsdiff$vty{.sat} : =T, U{.bh}, V{.bh} @2.0
vabsdiff$vty{.sat}$vop : =T, U{.bh}, V{.bh}, T @2.0
vabsdiff$vty{.sat} : =T(.bh), U{.bh}, V{.bh}, T @2.0
vmin$vty{.sat} : =T, U{.bh}, V{.bh} @2.0
vmin$vty{.sat}$vop : =T, U{.bh}, V{.bh}, T @2.0
vmin$vty{.sat} : =T(.bh), U{.bh}, V{.bh}, T @2.0
vmax$vty{.sat} : =T, U{.bh}, V{.bh} @2.0
vmax$vty{.sat}$vop : =T, U{.bh}, V{.bh}, T @2.0
vmax$vty{.sat} : =T(.bh), U{.bh}, V{.bh}, T @2.0
vshl(.u32|.s32)(.u32|.s32).u32{.sat}(.clamp|.wrap) : =T, U{.bh}, V{.bh} @2.0
vshl(.u32|.s32)(.u32|.s32).u32{.sat}(.clamp|.wrap)$vop : =T, U{.bh}, V{.bh}, T @2.0
vshl(.u32|.s32)(.u32|.s32).u32{.sat}(.clamp|.wrap) : =T(.bh), U{.bh}, V{.bh}, T @2.0
vshr(.u32|.s32)(.u32|.s32).u32{.sat}(.clamp|.wrap) : =T, U{.bh}, V{.bh} @2.0
vshr(.u32|.s32)(.u32|.s32).u32{.sat}(.clamp|.wrap)$vop : =T, U{.bh}, V{.bh}, T @2.0
vshr(.u32|.s32)(.u32|.s32).u32{.sat}(.clamp|.wrap) : =T(.bh), U{.bh}, V{.bh}, T @2.0
vmad$vty{.sat}{.shr7|.shr15} : =T, -U{.bh}, -V{.bh}, -T @2.0 +neg
vmad$vty.po{.sat}{.shr7|.shr15} : =T, U{.bh}, V{.bh}, T @2.0
vset(.u32|.s32)(.u32|.s32)$cmps : =u32, T{.bh}, U{.bh} @2.0
vset(.u32|.s32)(.u32|.s32)$cmps$vop : =u32, T{.bh}, U{.bh}, u32 @2.0
vset(.u32|.s32)(.u32|.s32)$cmps : =u32(.bh), T{.bh}, U{.bh}, u32 @2.0
vadd2$vty{.sat|.add} : =T{.hmask}, %U{.hxy}, %V{.hxy}, %T @3.0 @sm30
vsub2$vty{.sat|.add} : =T{.hmask}, %U{.hxy}, %V{.hxy}, %T @3.0 @sm30
vavrg2$vty{.sat|.add} : =T{.hmask}, %U{.hxy}, %V{.hxy}, %T @3.0 @sm30
vabsdiff2$vty{.sat|.add} : =T{.hmask}, %U{.hxy}, %V{.hxy}, %T @3.0 @sm30
vmin2$vty{.sat|.add} : =T{.hmask}, %U{.hxy}, %V{.hxy}, %T @3.0 @sm30
vmax2$vty{.sat|.add} : =T{.hmask}, %U{.hxy}, %V{.hxy}, %T @3.0 @sm30
vset2(.u32|.s32)(.u32|.s32)$cmps{.add} : =u32{.hmask}, %T{.hxy}, %U{.hxy}, %u32 @3.0 @sm30
vadd4$vty{.sat|.add} : =T{.bmask}, %U{.bxyzw}, %V{.bxyzw}, %T @3.0 @sm30
vsub4$vty{.sat|.add} : =T{.bmask}, %U{.bxyzw}, %V{.bxyzw}, %T @3.0 @sm30
vavrg4$vty{.sat|.add} : =T{.bmask}, %U{.bxyzw}, %V{.bxyzw}, %T @3.0 @sm30
vabsdiff4$vty{.sat|.add} : =T{.bmask}, %U{.bxyzw}, %V{.bxyzw}, %T @3.0 @sm30
vmin4$vty{.sat|.add} : =T{.bmask}, %U{.bxyzw}, %V{.bxyzw}, %T @3.0 @sm30
vmax4$vty{.sat|.add} : =T{.bmask}, %U{.bxyzw}, %V{.bxyzw}, %T @3.0 @sm30
vset4(.u32|.s32)(.u32|.s32)$cmps{.add} : =u32{.bmask}, %T{.bxyzw}, %U{.bxyzw}, %u32 @3.0 @sm30

# Moving data, and converting it.
mov(.pred|.b16|.b32|.b64|.u16|.u32|.u64|.s16|.s32|.s64|.f32|.f64) : =T, $T @1.0
mov(.u32|.u64) : =T, &T @1.0
mov.b128 : =T, T @8.3 @sm70
mov.b16 : =T, b8*2 @1.0
mov.b32 : =T, b16*2 @1.0
mov.b32 : =T, b8*4 @1.0
mov.b64 : =T, b32*2 @1.0
mov.b64 : =T, b16*4 @1.0
mov.b128 : =T, b64*2 @8.3 @sm70
mov.b128 : =T, b32*4 @8.3 @sm70
mov.b16 : =_b8*2, %T @1.0
mov.b32 : =_b16*2, %T @1.0
mov.b32 : =_b8*4, %T @1.0
mov.b64 : =_b32*2, %T @1.0
mov.b64 : =_b16*4, %T @1.0
mov.b128 : =_b64*2, %T @8.3 @sm70
mov.b128 : =_b32*4, %T @8.3 @sm70
shfl$shfl.b32 : =T[|=pred], T, u32, u32 @3.0 @sm30 @<sm70/6.4
shfl.sync$shfl.b32 : =T[|=pred], T, u32, u32, u32 @6.0 @sm30
prmt.b32{.f4e|.b4e|.rc8|.ecl|.ecr|.rc16} : =T, T, T, T @2.0
ld{.weak@6.0@sm70}{.global}$cop$hint$size$vec$ty : =T*v, [R], {L2::cache_hint}b64 @1.0 +wide +vec
ld{.weak@6.0@sm70}{.global}$l1$hint$size$vec$ty : =T*v, [R], {L2::cache_hint}b64 @1.0 +wide +vec
ld{.weak@6.0@sm70}$ldss$cop$vec$ty : =T*v, [R] @1.0 +wide +vec
ld.volatile{.global}$size$vec$ty : =T*v, [R] @1.1 +wide +vec
ld.volatile(.local@9.1|.shared|.shared::cta@7.8)$vec$ty : =T*v, [R] @1.1 +wide +vec
ld(.relaxed|.acquire)$scope{.global}$l1$hint$size$vec$ty : =T*v, [R], {L2::cache_hint}b64 @6.0 @sm70 +wide +vec +b128sys
ld(.relaxed|.acquire)$scope$shared$vec$ty : =T*v, [R] @6.0 @sm70 +wide +vec +b128sys
ld.mmio.relaxed.sys{.global}$ty64 : =T, [R] @8.2 @sm70 +wide
ld.mmio.acquire.sys{.global}$ty64 : =T, [R] @9.3 @sm70 +wide
ld.global{.ca|.cg|.cs}.nc$hint$size$vec$ty : =T*v, [R], {L2::cache_hint}b64 @3.1 @sm32 +wide +vec
ld.global.nc$l1$hint$size$vec$ty : =T*v, [R], {L2::cache_hint}b64 @3.1 @sm32 +wide +vec
ldu{.global}$vec$ty : =T*v, [R] @2.0 +wide +vec
st{.weak@6.0@sm70}{.global}$scop$hint$vec$ty : [W], T*v, {L2::cache_hint}b64 @1.0 +wide +vec
st{.weak@6.0@sm70}{.global}$l1$hint$vec$ty : [W], T*v, {L2::cache_hint}b64 @1.0 +wide +vec
st{.weak@6.0@sm70}(.local|.shared|.shared::cta@7.8)$scop$vec$ty : [W], T*v @1.0 +wide +vec
st.volatile{.global|.local@9.1|.shared|.shared::cta@7.8}$vec$ty : [W], T*v @1.1 +wide +vec
st(.relaxed|.release)$scope{.global}$l1$hint$vec$ty : [W], T*v, {L2::cache_hint}b64 @6.0 @sm70 +wide +vec +b128sys
st(.relaxed|.release)$scope$shared$vec$ty : [W], T*v @6.0 @sm70 +wide +vec +b128sys
st.mmio.relaxed.sys{.global}$ty64 : [W], T @8.2 @sm70 +wide
st.mmio.release.sys{.global}$ty64 : [W], T @9.3 @sm70 +wide
prefetch{.global|.local}(.L1|.L2) : [R] @2.0
prefetch.global(.L2::evict_last|.L2::evict_normal) : [R] @7.4 @sm80
prefetchu.L1 : [R] @2.0
applypriority{.global}.L2::evict_normal : [R], #=128 @7.4 @sm80
discard{.global}.L2 : [W], #=128 @7.4 @sm80
createpolicy.range{.global}$prio$prio2.b64 : =T, [R], u32, u32 @7.4 @sm80
createpolicy.fractional$prio$prio2.b64 : =T, f32(0,1]? @7.4 @sm80
createpolicy.cvt.L2.b64 : =T, b64 @7.4 @sm80
isspacep$space : =pred, %u64 @2.0
cvta$space.u64 : =T, T @2.0
cvta(.local|.shared|.shared::cta@7.8).u64 : =T, &T @2.0
cvta.to$space.u64 : =T, T @2.0
cvt{.rni|.rzi|.rmi|.rpi|.rn|.rz|.rm|.rp}{.ftz@1.4}{.sat}$cvty$cvty : =T, $U @1.0 +wide +cvt
cvt(.rn|.rz)~round{.relu@7.0@sm80}{.satfinite@8.1@sm70}(.f16|.bf16@7.0@sm80).f32 : =T, U @1.0 +wide
cvt(.rn|.rz)~round{.relu}{.satfinite@8.1}(.f16x2|.bf16x2).f32 : =T, U, U @7.0 @sm80
cvt.rna{.satfinite@8.1}.tf32.f32 : =T, U @7.0 @sm80
cvt.f32.bf16 : =T, U @7.1 @sm80
cvt.pack.sat(.u16|.s16).s32 : =u32, s32, s32 @6.5 @sm72
cvt.pack.sat(.u8|.s8).s32.b32 : =u32, s32, s32, b32 @6.5 @sm72
cvt.pack.sat(.u4|.s4|.u2|.s2).s32.b32 : =u32, s32, s32, b32 @6.5 @sm75

# Textures, each by its handle, the `u64` of a CUDA texture object, which holds its sampler
# too, in brackets with the coordinates in it, as for surfaces, though some are floats. A
# texture returns four, but two of `f16x2`; ptxas 13.4 takes `_` for one of four, and not
# of two, saying that `mov` may not discard its result. The level of detail of `.level`, or
# the gradients of `.grad`, follow the brackets, then an offset of each coordinate, and a
# depth to compare with, for `f32` coordinates, which without the offset is a line of its
# own. There are no `.texref` and `.samplerref` variables, and so no sampler apart from its
# texture, nor its `force_unnormalized_coords`.
tex{.base}.1d(.v4|.v2)$tdt.s32 : $texd, [%u64,U*1], s32*1[-8,7]@4.3@sm30? @3.1 +tex
tex{.base}.1d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*1], s32*1[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex{.base}.1d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*1], f32@4.3@sm30 @3.1 +tex
tex{.base}.2d(.v4|.v2)$tdt.s32 : $texd, [%u64,U*2], s32*2[-8,7]@4.3@sm30? @3.1 +tex
tex{.base}.2d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*2], s32*2[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex{.base}.2d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*2], f32@4.3@sm30 @3.1 +tex
tex{.base}.a1d(.v4|.v2)$tdt.s32 : $texd, [%u64,u32+U*1], s32*1[-8,7]@4.3@sm30? @3.1 +tex
tex{.base}.a1d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*1], s32*1[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex{.base}.a1d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*1], f32@4.3@sm30 @3.1 +tex
tex{.base}.a2d(.v4|.v2)$tdt.s32 : $texd, [%u64,u32+U*3], s32*2[-8,7]@4.3@sm30? @3.1 +tex
tex{.base}.a2d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*3], s32*2[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex{.base}.a2d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*3], f32@4.3@sm30 @3.1 +tex
tex{.base}.3d(.v4|.v2)$tdt(.s32|.f32) : $texd, [%u64,U*4], s32*4[-8,7]@4.3@sm30? @3.1 +tex
tex{.base}.cube(.v4|.v2)$tdt.f32 : $texd, [%u64,U*4], f32@4.3@sm30? @3.1 +tex
tex{.base}.acube(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*3], f32@4.3@sm30? @3.1 +tex
tex{.base}.2dms(.v4|.v2)$tdt.s32 : $texd, [%u64,U*4], s32*2[-8,7]@4.3@sm30? @3.2 @sm30 +tex
tex{.base}.a2dms(.v4|.v2)$tdt.s32 : $texd, [%u64,U*4], s32*2[-8,7]@4.3@sm30? @3.2 @sm30 +tex
tex.level.1d(.v4|.v2)$tdt.s32 : $texd, [%u64,U*1], U, s32*1[-8,7]@4.3@sm30? @3.1 +tex
tex.level.1d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*1], U, s32*1[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex.level.1d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*1], U, f32@4.3@sm30 @3.1 +tex
tex.level.2d(.v4|.v2)$tdt.s32 : $texd, [%u64,U*2], U, s32*2[-8,7]@4.3@sm30? @3.1 +tex
tex.level.2d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*2], U, s32*2[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex.level.2d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*2], U, f32@4.3@sm30 @3.1 +tex
tex.level.a1d(.v4|.v2)$tdt.s32 : $texd, [%u64,u32+U*1], U, s32*1[-8,7]@4.3@sm30? @3.1 +tex
tex.level.a1d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*1], U, s32*1[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex.level.a1d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*1], U, f32@4.3@sm30 @3.1 +tex
tex.level.a2d(.v4|.v2)$tdt.s32 : $texd, [%u64,u32+U*3], U, s32*2[-8,7]@4.3@sm30? @3.1 +tex
tex.level.a2d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*3], U, s32*2[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex.level.a2d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*3], U, f32@4.3@sm30 @3.1 +tex
tex.level.3d(.v4|.v2)$tdt(.s32|.f32) : $texd, [%u64,U*4], U, s32*4[-8,7]@4.3@sm30? @3.1 +tex
tex.level.cube(.v4|.v2)$tdt.f32 : $texd, [%u64,U*4], U, f32@4.3@sm30? @3.1 +tex
tex.level.acube(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*3], U, f32@4.3@sm30? @3.1 +tex
tex.grad.1d(.v4|.v2)$tdt.s32 : $texd, [%u64,U*1], f32*1, f32*1, s32*1[-8,7]@4.3@sm30? @3.1 +tex
tex.grad.1d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*1], f32*1, f32*1, s32*1[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex.grad.1d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*1], f32*1, f32*1, f32@4.3@sm30 @3.1 +tex
tex.grad.2d(.v4|.v2)$tdt.s32 : $texd, [%u64,U*2], f32*2, f32*2, s32*2[-8,7]@4.3@sm30? @3.1 +tex
tex.grad.2d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*2], f32*2, f32*2, s32*2[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex.grad.2d(.v4|.v2)$tdt.f32 : $texd, [%u64,U*2], f32*2, f32*2, f32@4.3@sm30 @3.1 +tex
tex.grad.a1d(.v4|.v2)$tdt.s32 : $texd, [%u64,u32+U*1], f32*1, f32*1, s32*1[-8,7]@4.3@sm30? @3.1 +tex
tex.grad.a1d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*1], f32*1, f32*1, s32*1[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex.grad.a1d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*1], f32*1, f32*1, f32@4.3@sm30 @3.1 +tex
tex.grad.a2d(.v4|.v2)$tdt.s32 : $texd, [%u64,u32+U*3], f32*2, f32*2, s32*2[-8,7]@4.3@sm30? @3.1 +tex
tex.grad.a2d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*3], f32*2, f32*2, s32*2[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1 +tex
tex.grad.a2d(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*3], f32*2, f32*2, f32@4.3@sm30 @3.1 +tex
tex.grad.3d(.v4|.v2)$tdt(.s32|.f32) : $texd, [%u64,U*4], f32*4, f32*4, s32*4[-8,7]@4.3@sm30? @3.1 +tex
tex.grad.cube(.v4|.v2)$tdt.f32 : $texd, [%u64,U*4], f32*4, f32*4, f32@4.3@sm30? @4.3 +tex
tex.grad.acube(.v4|.v2)$tdt.f32 : $texd, [%u64,u32+U*3], f32*4, f32*4, f32@4.3@sm30? @4.3 +tex
tld4(.r|.g|.b|.a).2d.v4(.u32|.s32|.f32).f32 : =_T*4[|=pred@7.1@sm60], [%u64,U*2], s32*2[-8,7]@4.3@sm30?, f32@4.3@sm30? @3.1
tld4(.r|.g|.b|.a).2d.v4(.u32|.s32|.f32).f32 : =_T*4[|=pred@7.1@sm60], [%u64,U*2], f32@4.3@sm30 @3.1
tld4(.r|.g|.b|.a).a2d.v4(.u32|.s32|.f32).f32 : =_T*4[|=pred@7.1@sm60], [%u64,u32+U*3], s32*2[-8,7]@4.3@sm30?, f32@4.3@sm30? @4.3 @sm30
tld4(.r|.g|.b|.a).a2d.v4(.u32|.s32|.f32).f32 : =_T*4[|=pred@7.1@sm60], [%u64,u32+U*3], f32@4.3@sm30 @4.3 @sm30
tld4(.r|.g|.b|.a).cube.v4(.u32|.s32|.f32).f32 : =_T*4[|=pred@7.1@sm60], [%u64,U*4], f32@4.3@sm30? @4.3 @sm30
tld4(.r|.g|.b|.a).acube.v4(.u32|.s32|.f32).f32 : =_T*4[|=pred@7.1@sm60], [%u64,u32+U*3], f32@4.3@sm30? @4.3 @sm30
txq(.width|.height|.depth|.channel_data_type|.channel_order|.normalized_coords).b32 : =T, [%u64] @3.1
txq(.array_size|.num_mipmap_levels|.num_samples@sm30).b32 : =T, [%u64] @4.1
txq(.filter_mode|.addr_mode_0|.addr_mode_1|.addr_mode_2).b32 : =T, [%u64] @3.1
txq.level(.width|.height|.depth).b32 : =T, [%u64], u32 @4.3 @sm30
istypep(.texref|.samplerref|.surfref) : =pred, %u64 @4.0 @sm30

# Surfaces, each by its handle, the `u64` of a CUDA surface object, in brackets with the
# coordinates in it: an array's layer first, then four where there would be three, the last
# ignored; the lowest in bytes, but in samples for `sust.p` and `sured.p`. `sured` has no
# `.xor`, on which ptxas 13.4 crashes, a segfault, rather than refusing it.
suld.b.1d$sucop$vec$bty$clamp : =_T*v, [%u64,s32*1] @3.1 +wide +vec
suld.b.2d$sucop$vec$bty$clamp : =_T*v, [%u64,s32*2] @3.1 +wide +vec
suld.b.3d$sucop$vec$bty$clamp : =_T*v, [%u64,s32*4] @3.1 +wide +vec
suld.b.a1d$sucop$vec$bty$clamp : =_T*v, [%u64,u32+s32*1] @3.1 +wide +vec
suld.b.a2d$sucop$vec$bty$clamp : =_T*v, [%u64,u32+s32*3] @3.1 +wide +vec
sust.b.1d$scop$vec$bty$clamp : [%u64,s32*1], T*v @3.1 +wide +vec
sust.b.2d$scop$vec$bty$clamp : [%u64,s32*2], T*v @3.1 +wide +vec
sust.b.3d$scop$vec$bty$clamp : [%u64,s32*4], T*v @3.1 +wide +vec
sust.b.a1d$scop$vec$bty$clamp : [%u64,u32+s32*1], T*v @3.1 +wide +vec
sust.b.a2d$scop$vec$bty$clamp : [%u64,u32+s32*3], T*v @3.1 +wide +vec
sust.p.1d$vec.b32$clamp : [%u64,s32*1], T*v @3.1 +wide
sust.p.2d$vec.b32$clamp : [%u64,s32*2], T*v @3.1 +wide
sust.p.3d$vec.b32$clamp : [%u64,s32*4], T*v @3.1 +wide
sured.b.add.1d(.u32|.s32|.u64)$clamp : [%u64,s32*1], T @3.1 +wide
sured.b.add.2d(.u32|.s32|.u64)$clamp : [%u64,s32*2], T @3.1 +wide
sured.b.add.3d(.u32|.s32|.u64)$clamp : [%u64,s32*4], T @3.1 +wide
sured.b(.min|.max).1d(.u32|.s32|.u64@8.1@sm50|.s64@8.1@sm50)$clamp : [%u64,s32*1], T @3.1 +wide
sured.b(.min|.max).2d(.u32|.s32|.u64@8.1@sm50|.s64@8.1@sm50)$clamp : [%u64,s32*2], T @3.1 +wide
sured.b(.min|.max).3d(.u32|.s32|.u64@8.1@sm50|.s64@8.1@sm50)$clamp : [%u64,s32*4], T @3.1 +wide
sured.b(.and|.or).1d.b32$clamp : [%u64,s32*1], T @3.1 +wide
sured.b(.and|.or).2d.b32$clamp : [%u64,s32*2], T @3.1 +wide
sured.b(.and|.or).3d.b32$clamp : [%u64,s32*4], T @3.1 +wide
sured.p(.add|.min|.max|.and|.or).1d.b32$clamp : [%u64,s32*1], T @3.1 +wide
sured.p(.add|.min|.max|.and|.or).2d.b32$clamp : [%u64,s32*2], T @3.1 +wide
sured.p(.add|.min|.max|.and|.or).3d.b32$clamp : [%u64,s32*4], T @3.1 +wide
sured.p(.min|.max).1d.b64$clamp : [%u64,s32*1], T @8.1 @sm50 +wide
sured.p(.min|.max).2d.b64$clamp : [%u64,s32*2], T @8.1 @sm50 +wide
sured.p(.min|.max).3d.b64$clamp : [%u64,s32*4], T @8.1 @sm50 +wide
suq(.width|.height|.depth|.channel_data_type|.channel_order|.array_size@4.1|.memory_layout@4.2).b32 : =T, [%u64] @3.1

# Control.
bra{.uni} : lab @1.0
brx.idx{.uni} : %u32, labs @6.0 @sm30
ret{.uni} @1.0
exit @1.0

# Barriers, fences and atomics.
barrier$cta.sync{.aligned} : u32[0,15], u32/32? @6.0 @sm30
barrier$cta.arrive{.aligned} : u32[0,15], u32/32 @6.0 @sm30
barrier$cta.red.popc{.aligned}.u32 : =T, u32[0,15], !pred @6.0 @sm30
barrier$cta.red.popc{.aligned}.u32 : =T, u32[0,15], u32/32, !pred @6.0 @sm30
barrier$cta.red(.and|.or){.aligned}.pred : =T, u32[0,15], !pred @6.0 @sm30
barrier$cta.red(.and|.or){.aligned}.pred : =T, u32[0,15], u32/32, !pred @6.0 @sm30
bar$cta.sync : u32[0,15], u32/32? @1.0
bar$cta.arrive : u32[0,15], u32/32 @2.0
bar$cta.red.popc.u32 : =T, u32[0,15], !pred @2.0
bar$cta.red.popc.u32 : =T, u32[0,15], u32/32, !pred @2.0
bar$cta.red(.and|.or).pred : =T, u32[0,15], !pred @2.0
bar$cta.red(.and|.or).pred : =T, u32[0,15], u32/32, !pred @2.0
bar.warp.sync : u32 @6.0 @sm30
membar(.cta|.gl|.sys@2.0) @1.4
membar.proxy.alias @7.5 @sm60
fence{.sc|.acq_rel}$scope @6.0 @sm70
fence.proxy.alias @7.5 @sm70
atom$sem$ascope{.global}(.and|.or|.xor)$hint(.b32|.b64@3.1@sm32) : =T, [X], T, {L2::cache_hint}b64 @1.0
atom$sem$ascope$shared(.and|.or|.xor)(.b32|.b64@3.1@sm32) : =T, [X], T @1.2
atom$sem$ascope{.global}(.add|.inc|.dec|.min|.max)$hint.u32 : =T, [X], T, {L2::cache_hint}b64 @1.0
atom$sem$ascope$shared(.add|.inc|.dec|.min|.max).u32 : =T, [X], T @1.2
atom$sem$ascope{.global}(.add|.min|.max)$hint.s32 : =T, [X], T, {L2::cache_hint}b64 @1.0
atom$sem$ascope{.global}.add$hint.u64 : =T, [X], T, {L2::cache_hint}b64 @1.2
atom$sem$ascope$shared(.add|.min|.max).s32 : =T, [X], T @1.2
atom$sem$ascope$shared.add.u64 : =T, [X], T @2.0
atom$sem$ascope{.global}(.min|.max)$hint(.u64|.s64) : =T, [X], T, {L2::cache_hint}b64 @3.1 @sm32
atom$sem$ascope$shared(.min|.max)(.u64|.s64) : =T, [X], T @3.1 @sm32
atom$sem$ascope{.global}.add$hint(.f32|.f64@5.0@sm60) : =T, [X], T, {L2::cache_hint}b64 @2.0
atom$sem$ascope$shared.add(.f32|.f64@5.0@sm60) : =T, [X], T @2.0
atom$sem$ascope{.global}.exch$hint(.b32|.b64@1.2) : =T, [X], T, {L2::cache_hint}b64 @1.0
atom$sem$ascope$shared.exch(.b32|.b64@2.0) : =T, [X], T @1.2
atom$sem$ascope{.global}.cas(.b32|.b64@1.2) : =T, [X], T, T @1.0
atom$sem$ascope$shared.cas(.b32|.b64@2.0) : =T, [X], T, T @1.2
atom$sem$ascope{.global}.cas.b16 : =T, [X], T, T @6.3 @sm70
atom$sem$ascope$shared.cas.b16 : =T, [X], T, T @6.3 @sm70
atom$sem$ascope{.global}.add.noftz$hint.f16 : =T, [X], T, {L2::cache_hint}b64 @6.3 @sm70
atom$sem$ascope$shared.add.noftz.f16 : =T, [X], T @6.3 @sm70
atom$sem$ascope{.global}.add.noftz$hint.f16x2 : =T, [X], T, {L2::cache_hint}b64 @6.2 @sm60
atom$sem$ascope$shared.add.noftz.f16x2 : =T, [X], T @6.2 @sm60
red$rsem$ascope{.global}(.and|.or|.xor)$hint(.b32|.b64@3.1@sm32) : [X], T, {L2::cache_hint}b64 @1.2
red$rsem$ascope$shared(.and|.or|.xor)(.b32|.b64@3.1@sm32) : [X], T @1.2
red$rsem$ascope{.global}(.add|.inc|.dec|.min|.max)$hint.u32 : [X], T, {L2::cache_hint}b64 @1.2
red$rsem$ascope$shared(.add|.inc|.dec|.min|.max).u32 : [X], T @1.2
red$rsem$ascope{.global}(.add|.min|.max)$hint.s32 : [X], T, {L2::cache_hint}b64 @1.2
red$rsem$ascope{.global}.add$hint.u64 : [X], T, {L2::cache_hint}b64 @1.2
red$rsem$ascope$shared(.add|.min|.max).s32 : [X], T @1.2
red$rsem$ascope$shared.add.u64 : [X], T @2.0
red$rsem$ascope{.global}(.min|.max)$hint(.u64|.s64) : [X], T, {L2::cache_hint}b64 @3.1 @sm32
red$rsem$ascope$shared(.min|.max)(.u64|.s64) : [X], T @3.1 @sm32
red$rsem$ascope{.global}.add$hint(.f32|.f64@5.0@sm60) : [X], T, {L2::cache_hint}b64 @2.0
red$rsem$ascope$shared.add(.f32|.f64@5.0@sm60) : [X], T @2.0
red$rsem$ascope{.global}.add.noftz$hint.f16 : [X], T, {L2::cache_hint}b64 @6.3 @sm70
red$rsem$ascope$shared.add.noftz.f16 : [X], T @6.3 @sm70
red$rsem$ascope{.global}.add.noftz$hint.f16x2 : [X], T, {L2::cache_hint}b64 @6.2 @sm60
red$rsem$ascope$shared.add.noftz.f16x2 : [X], T @6.2 @sm60

# Copies that go on while the thread does, and the barriers in shared memory that count
# what arrives. ptxas takes `cp.async.ca.shared::cta` from 7.0, though the ISA says 7.8.
cp.async.ca(.shared|.shared::cta).global$hint$size : [W], [R], #=4|8|16, {L2::cache_hint}b64 @7.0 @sm80
cp.async.ca(.shared|.shared::cta).global$hint$size : [W], [R], #=4, u32[0,4], {L2::cache_hint}b64 @7.0 @sm80
cp.async.ca(.shared|.shared::cta).global$hint$size : [W], [R], #=8, u32[0,8], {L2::cache_hint}b64 @7.0 @sm80
cp.async.ca(.shared|.shared::cta).global$hint$size : [W], [R], #=16, u32[0,16], {L2::cache_hint}b64 @7.0 @sm80
cp.async.ca(.shared|.shared::cta).global$hint$size : [W], [R], #=4|8|16, !pred, {L2::cache_hint}b64 @7.5 @sm80
cp.async.cg(.shared|.shared::cta).global$hint$size : [W], [R], #=16, {L2::cache_hint}b64 @7.0 @sm80
cp.async.cg(.shared|.shared::cta).global$hint$size : [W], [R], #=16, u32[0,16], {L2::cache_hint}b64 @7.0 @sm80
cp.async.cg(.shared|.shared::cta).global$hint$size : [W], [R], #=16, !pred, {L2::cache_hint}b64 @7.5 @sm80
cp.async.commit_group @7.0 @sm80
cp.async.wait_group : # @7.0 @sm80
cp.async.wait_all @7.0 @sm80
cp.async.mbarrier.arrive{.noinc}{.shared|.shared::cta@7.8}.b64 : [X] @7.0 @sm80
mbarrier.init{.shared|.shared::cta@7.8}.b64 : [W], u32[1,1048575] @7.0 @sm80
mbarrier.inval{.shared|.shared::cta@7.8}.b64 : [W] @7.0 @sm80
mbarrier.arrive{.release.cta@8.0}{.shared|.shared::cta@7.8}.b64 : =u64, [X] @7.0 @sm80
mbarrier.arrive{.release.cta@8.0}{.shared|.shared::cta@7.8}.b64 : =_u64, [X] @7.1 @sm80
mbarrier.arrive.noComplete{.release.cta@8.0}{.shared|.shared::cta@7.8}.b64 : =u64, [X], u32[1,1048575] @7.0 @sm80
mbarrier.arrive.noComplete{.release.cta@8.0}{.shared|.shared::cta@7.8}.b64 : =_u64, [X], u32[1,1048575] @7.1 @sm80
mbarrier.arrive_drop{.release.cta@8.0}{.shared|.shared::cta@7.8}.b64 : =_u64, [X] @7.0 @sm80
mbarrier.arrive_drop.noComplete{.release.cta@8.0}{.shared|.shared::cta@7.8}.b64 : =_u64, [X], u32[1,1048575] @7.0 @sm80
mbarrier.test_wait{.acquire.cta@8.0}{.shared|.shared::cta@7.8}.b64 : =pred, [R], u64 @7.0 @sm80
mbarrier.test_wait.parity{.acquire.cta@8.0}{.shared|.shared::cta@7.8}.b64 : =pred, [R], u32[0,1] @7.1 @sm80
mbarrier.pending_count.b64 : =u32, u64 @7.0 @sm80

# The threads of a warp together.
vote(.all|.any|.uni).pred : =T, !pred @1.2 @<sm70/6.4
vote.ballot.b32 : =T, !pred @2.0 @<sm70/6.4
vote.sync(.all|.any|.uni).pred : =T, !pred, u32 @6.0 @sm30
vote.sync.ballot.b32 : =T, !pred, u32 @6.0 @sm30
match.any.sync(.b32|.b64) : =b32, T, u32 @6.0 @sm70
match.all.sync(.b32|.b64) : =b32[|=_pred], T, u32 @6.0 @sm70
activemask.b32 : =T @6.2 @sm30
redux.sync(.add|.min|.max)(.u32|.s32) : =T, T, u32 @7.0 @sm80
redux.sync(.and|.or|.xor).b32 : =T, T, u32 @7.0 @sm80

# Matrices a warp loads and multiplies together, each thread holding a fragment of each: so
# many registers, braced even when one. `mma` takes `_` for what it leaves unwritten, and
# `movmatrix` doesn't: on `_` there ptxas 13.4 crashes, a segfault, rather than refusing it.
# A sparse one's selector is the threads of four that hold the metadata, as the ISA has it
# for the shape: ptxas takes 0 to 3 for some it has 0 or 1 for.
ldmatrix.sync.aligned.m8n8.x1{.trans}{.shared|.shared::cta@7.8}.b16 : =b32*1, [R] @6.5 @sm75
ldmatrix.sync.aligned.m8n8.x2{.trans}{.shared|.shared::cta@7.8}.b16 : =_b32*2, [R] @6.5 @sm75
ldmatrix.sync.aligned.m8n8.x4{.trans}{.shared|.shared::cta@7.8}.b16 : =_b32*4, [R] @6.5 @sm75
movmatrix.sync.aligned.m8n8.trans.b16 : =b32, %b32 @7.8 @sm75
mma.sync.aligned.m8n8k4(.row|.col)(.row|.col).f16.f16.f16.f16 : =_f16x2*4, %f16x2*2, %f16x2*2, %f16x2*4 @6.4 @sm70
mma.sync.aligned.m8n8k4(.row|.col)(.row|.col).f32.f16.f16.f16 : =_f32*8, %f16x2*2, %f16x2*2, %f16x2*4 @6.4 @sm70
mma.sync.aligned.m8n8k4(.row|.col)(.row|.col).f32.f16.f16.f32 : =_f32*8, %f16x2*2, %f16x2*2, f32*8 @6.4 @sm70
mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 : =_f16x2*2, %f16x2*2, %f16x2*1, %f16x2*2 @6.5 @sm75
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 : =_f32*4, %f16x2*2, %f16x2*1, f32*4 @6.5 @sm75
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 : =_f16x2*2, %f16x2*4, %f16x2*2, %f16x2*2 @7.0 @sm80
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 : =_f32*4, %f16x2*4, %f16x2*2, f32*4 @7.0 @sm80
mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 : =_f32*4, %tf32*2, %tf32*1, f32*4 @7.0 @sm80
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 : =_f32*4, %tf32*4, %tf32*2, f32*4 @7.0 @sm80
mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 : =_f32*4, %bf16x2*2, %bf16x2*1, f32*4 @7.0 @sm80
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 : =_f32*4, %bf16x2*4, %bf16x2*2, f32*4 @7.0 @sm80
mma.sync.aligned.m8n8k4.row.col{.rn|.rz|.rm|.rp}.f64.f64.f64.f64 : =_f64*2, f64*1, f64*1, f64*2 @7.0 @sm80
mma.sync.aligned.m8n8k16.row.col{.satfinite}.s32(.u8|.s8)(.u8|.s8).s32 : =_s32*2, %u32*1, %u32*1, s32*2 @6.5 @sm75
mma.sync.aligned.m8n8k32.row.col{.satfinite}.s32(.u4|.s4)(.u4|.s4).s32 : =_s32*2, %u32*1, %u32*1, s32*2 @6.5 @sm75
mma.sync.aligned.m16n8k16.row.col{.satfinite}.s32(.u8|.s8)(.u8|.s8).s32 : =_s32*4, %u32*2, %u32*1, s32*4 @7.0 @sm80
mma.sync.aligned.m16n8k32.row.col{.satfinite}.s32(.u8|.s8)(.u8|.s8).s32 : =_s32*4, %u32*4, %u32*2, s32*4 @7.0 @sm80
mma.sync.aligned.m16n8k32.row.col{.satfinite}.s32(.u4|.s4)(.u4|.s4).s32 : =_s32*4, %u32*2, %u32*1, s32*4 @7.0 @sm80
mma.sync.aligned.m16n8k64.row.col{.satfinite}.s32(.u4|.s4)(.u4|.s4).s32 : =_s32*4, %u32*4, %u32*2, s32*4 @7.0 @sm80
mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32(.xor|.and@7.1@sm80).popc : =_s32*2, %b32*1, %b32*1, s32*2 @7.0 @sm75
mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32(.xor|.and@7.1).popc : =_s32*4, %b32*2, %b32*1, s32*4 @7.0 @sm80
mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32(.xor|.and@7.1).popc : =_s32*4, %b32*4, %b32*2, s32*4 @7.0 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 : =_f16x2*2, %f16x2*2, %f16x2*2, %f16x2*2, %b32, #=0|1|2|3 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 : =_f32*4, %f16x2*2, %f16x2*2, f32*4, %b32, #=0|1|2|3 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 : =_f16x2*2, %f16x2*4, %f16x2*4, %f16x2*2, %b32, #=0|1 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 : =_f32*4, %f16x2*4, %f16x2*4, f32*4, %b32, #=0|1 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 : =_f32*4, %bf16x2*2, %bf16x2*2, f32*4, %b32, #=0|1|2|3 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 : =_f32*4, %bf16x2*4, %bf16x2*4, f32*4, %b32, #=0|1 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 : =_f32*4, %tf32*2, %tf32*2, f32*4, %b32, #=0|1|2|3 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 : =_f32*4, %tf32*4, %tf32*4, f32*4, %b32, #=0|1 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k32.row.col{.satfinite}.s32(.u8|.s8)(.u8|.s8).s32 : =_s32*4, %u32*2, %u32*2, s32*4, %b32, #=0|1 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k64.row.col{.satfinite}.s32(.u8|.s8)(.u8|.s8).s32 : =_s32*4, %u32*4, %u32*4, s32*4, %b32, #=0 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k64.row.col{.satfinite}.s32(.u4|.s4)(.u4|.s4).s32 : =_s32*4, %u32*2, %u32*2, s32*4, %b32, #=0|1 @7.1 @sm80
mma(.sp|.sp::ordered_metadata@8.5).sync.aligned.m16n8k128.row.col{.satfinite}.s32(.u4|.s4)(.u4|.s4).s32 : =_s32*4, %u32*4, %u32*4, s32*4, %b32, #=0 @7.1 @sm80

# The same, each fragment loaded and stored by the warp together, its stride in elements.
# ptxas requires `.aligned` from PTX 6.3 and refuses it before, so kgen, which writes it,
# writes wmma from 6.3, though the ISA has 6.0 without it; and no `.satfinite` on a float
# one, which 6.5 removes.
wmma.load(.a|.b).sync.aligned$lay$fshape$gsh.f16 : =_f16x2*8, [R], u32? @6.3 @sm70
wmma.load.c.sync.aligned$lay$fshape$gsh.f16 : =_f16x2*4, [R], u32? @6.3 @sm70
wmma.load.c.sync.aligned$lay$fshape$gsh.f32 : =_f32*8, [R], u32? @6.3 @sm70
wmma.load(.a|.b).sync.aligned$lay.m16n16k16$gsh(.u8|.s8) : =_u32*2, [R], u32? @6.3 @sm72
wmma.load.a.sync.aligned$lay.m8n32k16$gsh(.u8|.s8) : =u32*1, [R], u32? @6.3 @sm72
wmma.load.b.sync.aligned$lay.m8n32k16$gsh(.u8|.s8) : =_u32*4, [R], u32? @6.3 @sm72
wmma.load.a.sync.aligned$lay.m32n8k16$gsh(.u8|.s8) : =_u32*4, [R], u32? @6.3 @sm72
wmma.load.b.sync.aligned$lay.m32n8k16$gsh(.u8|.s8) : =u32*1, [R], u32? @6.3 @sm72
wmma.load.c.sync.aligned$lay$fshape$gsh.s32 : =_s32*8, [R], u32? @6.3 @sm72
wmma.load.a.sync.aligned.row.m8n8k32$gsh(.u4|.s4) : =u32*1, [R], u32? @6.3 @sm75
wmma.load.b.sync.aligned.col.m8n8k32$gsh(.u4|.s4) : =u32*1, [R], u32? @6.3 @sm75
wmma.load.a.sync.aligned.row.m8n8k128$gsh.b1 : =u32*1, [R], u32? @6.3 @sm75
wmma.load.b.sync.aligned.col.m8n8k128$gsh.b1 : =u32*1, [R], u32? @6.3 @sm75
wmma.load.c.sync.aligned$lay(.m8n8k32|.m8n8k128)$gsh.s32 : =_s32*2, [R], u32? @6.3 @sm75
wmma.load(.a|.b).sync.aligned$lay.m16n16k16$gsh.bf16 : =_bf16x2*4, [R], u32? @7.0 @sm80
wmma.load.a.sync.aligned$lay.m8n32k16$gsh.bf16 : =_bf16x2*2, [R], u32? @7.0 @sm80
wmma.load.b.sync.aligned$lay.m8n32k16$gsh.bf16 : =_bf16x2*8, [R], u32? @7.0 @sm80
wmma.load.a.sync.aligned$lay.m32n8k16$gsh.bf16 : =_bf16x2*8, [R], u32? @7.0 @sm80
wmma.load.b.sync.aligned$lay.m32n8k16$gsh.bf16 : =_bf16x2*2, [R], u32? @7.0 @sm80
wmma.load(.a|.b).sync.aligned$lay.m16n16k8$gsh.tf32 : =_tf32*4, [R], u32? @7.0 @sm80
wmma.load.c.sync.aligned$lay.m16n16k8$gsh.f32 : =_f32*8, [R], u32? @7.0 @sm80
wmma.load(.a|.b).sync.aligned$lay.m8n8k4$gsh.f64 : =f64*1, [R], u32? @7.0 @sm80
wmma.load.c.sync.aligned$lay.m8n8k4$gsh.f64 : =_f64*2, [R], u32? @7.0 @sm80
wmma.store.d.sync.aligned$lay$fshape$gsh.f16 : [W], %f16x2*4, u32? @6.3 @sm70
wmma.store.d.sync.aligned$lay$fshape$gsh.f32 : [W], f32*8, u32? @6.3 @sm70
wmma.store.d.sync.aligned$lay$fshape$gsh.s32 : [W], s32*8, u32? @6.3 @sm72
wmma.store.d.sync.aligned$lay(.m8n8k32|.m8n8k128)$gsh.s32 : [W], s32*2, u32? @6.3 @sm75
wmma.store.d.sync.aligned$lay.m16n16k8$gsh.f32 : [W], f32*8, u32? @7.0 @sm80
wmma.store.d.sync.aligned$lay.m8n8k4$gsh.f64 : [W], f64*2, u32? @7.0 @sm80
wmma.mma.sync.aligned$lay$lay$fshape.f16.f16 : =f16x2*4, %f16x2*8, %f16x2*8, %f16x2*4 @6.3 @sm70
wmma.mma.sync.aligned$lay$lay$fshape.f32.f16 : =f32*8, %f16x2*8, %f16x2*8, %f16x2*4 @6.3 @sm70
wmma.mma.sync.aligned$lay$lay$fshape.f16.f32 : =f16x2*4, %f16x2*8, %f16x2*8, f32*8 @6.3 @sm70
wmma.mma.sync.aligned$lay$lay$fshape.f32.f32 : =f32*8, %f16x2*8, %f16x2*8, f32*8 @6.3 @sm70
wmma.mma.sync.aligned$lay$lay.m16n16k16.s32(.u8.u8|.s8.s8).s32{.satfinite} : =s32*8, u32*2, u32*2, s32*8 @6.3 @sm72
wmma.mma.sync.aligned$lay$lay.m8n32k16.s32(.u8.u8|.s8.s8).s32{.satfinite} : =s32*8, u32*1, u32*4, s32*8 @6.3 @sm72
wmma.mma.sync.aligned$lay$lay.m32n8k16.s32(.u8.u8|.s8.s8).s32{.satfinite} : =s32*8, u32*4, u32*1, s32*8 @6.3 @sm72
wmma.mma.sync.aligned.row.col.m8n8k32.s32(.u4.u4|.s4.s4).s32{.satfinite} : =s32*2, %u32*1, %u32*1, s32*2 @6.3 @sm75
wmma.mma(.xor|.and@7.1@sm80).popc.sync.aligned.row.col.m8n8k128.s32.b1.b1.s32 : =s32*2, %u32*1, %u32*1, s32*2 @6.3 @sm75
wmma.mma.sync.aligned$lay$lay.m16n16k16.f32.bf16.bf16.f32 : =f32*8, %bf16x2*4, %bf16x2*4, f32*8 @7.0 @sm80
wmma.mma.sync.aligned$lay$lay.m8n32k16.f32.bf16.bf16.f32 : =f32*8, %bf16x2*2, %bf16x2*8, f32*8 @7.0 @sm80
wmma.mma.sync.aligned$lay$lay.m32n8k16.f32.bf16.bf16.f32 : =f32*8, %bf16x2*8, %bf16x2*2, f32*8 @7.0 @sm80
wmma.mma.sync.aligned$lay$lay.m16n16k8.f32.tf32.tf32.f32 : =f32*8, %tf32*4, %tf32*4, f32*8 @7.0 @sm80
wmma.mma.sync.aligned$lay$lay.m8n8k4{.rn|.rz|.rm|.rp}.f64.f64.f64.f64 : =f64*2, f64*1, f64*1, f64*2 @7.0 @sm80

# The stack, and the rest.
stacksave(.u32|.u64) : =T @7.3 @sm52
stackrestore(.u32|.u64) : %T @7.3 @sm52
alloca(.u32|.u64) : =T, T, #^23? @7.3 @sm52
brkpt @1.0
nanosleep.u32 : T @6.2 @sm70
pmevent : #4 @1.4
pmevent.mask : #16 @3.0
trap @1.0
"#;

/// An instruction's form.
#[derive(Clone, Debug)]
pub struct Form {
    /// The first word of its name: `add`.
    pub op: &'static str,
    /// The words after it, in the order PTX writes them.
    pub elems: Vec<Elem>,
    /// Its operands.
    pub operands: Vec<Spec>,
    /// What it needs, whatever words it has.
    pub needs: Needs,
    /// Whether a register may be wider than the instruction's type, as `ld`, `st` and `cvt`
    /// let it be.
    pub wide: bool,
    /// What the words must be together, beyond what each is alone.
    pub checks: Vec<Check>,
    /// The line of [`TABLE`] it is.
    pub line: &'static str,
}

/// An element of a name: a word, or one of a few.
#[derive(Clone, Debug)]
pub struct Elem {
    /// The words it may be.
    pub alts: Vec<Alt>,
    /// Whether the name may leave it out.
    pub optional: bool,
    /// Which of the instruction's types it is, if its words are types: 0 for the first.
    pub slot: Option<usize>,
    /// What leaving it out leaves unsaid: `round`, `half` or `cmp`.
    pub why: Option<&'static str>,
}

/// One of the words an element may be: a word, or a few, as `.xorsign.abs` is.
#[derive(Clone, Debug)]
pub struct Alt {
    /// The words, without their dots.
    pub words: Vec<&'static str>,
    /// What they need.
    pub needs: Needs,
}

/// What an operand is to the instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A register it writes.
    Dst,
    /// A register it reads, or a value known now.
    Src,
    /// Memory at an address, and what it does there.
    Addr(Access),
    /// A label it jumps to.
    Label,
    /// Labels taken together, of which a register it reads picks the one it jumps to:
    /// `brx.idx`'s table.
    Labels,
    /// A texture or a surface, by its handle, in brackets with the coordinates in it where it
    /// takes them: `[t, (x, y)]`. What the brackets hold is [`Spec::inner`].
    Handle,
    /// An integer known now.
    Imm(Imm),
}

/// What an operand that is an integer known now may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Imm {
    /// One of so many bits, unsigned: `lop3`'s table is 8.
    Bits(u32),
    /// One of these, as the table writes them: `4|8|16`, the bytes `cp.async` copies; `128`,
    /// the only one `applypriority` has for now.
    Of(&'static str),
    /// A power of two, to 2 to this power: `alloca`'s alignment.
    Pow2(u32),
}

impl Imm {
    /// The values one of [`Imm::Of`] may be, and none for another.
    pub fn values(self) -> Vec<i64> {
        match self {
            Imm::Of(values) => values.split('|').map(|v| v.parse().expect("the table's values are numbers")).collect(),
            _ => vec![],
        }
    }
}

/// An operand's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TyRef {
    /// The type the name gives at this place among its types: 0 for the first.
    Slot(usize),
    /// This one.
    Fixed(Ty),
}

/// How many registers an operand is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Count {
    /// One.
    One,
    /// So many, taken together: `{a, b}`.
    Fixed(usize),
    /// As many as `.v2` or `.v4` say, or one.
    Vec,
}

/// The parts of a register an operand may be, as a video instruction selects them after a
/// dot: of a source, the bytes or halves it works on; of a result, those it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sel {
    /// `.bh`: a byte or a half, `.b0` to `.b3`, `.h0` or `.h1`.
    One,
    /// `.hxy`: two of the four halves of the two sources together, as `.h10` or `.h32`.
    Halves,
    /// `.bxyzw`: four of their eight bytes, as `.b3210` or `.b7654`.
    Bytes,
    /// `.hmask`: the halves of the result it writes, `.h0`, `.h1` or `.h10`.
    HalfMask,
    /// `.bmask`: the bytes, from the highest down, each once at most: `.b3210`, `.b31`.
    ByteMask,
}

impl Sel {
    /// Whether an operand of these parts may be `part`.
    pub fn takes(self, part: Part) -> bool {
        let w = part.which();
        match self {
            Sel::One => w.len() == 1 && w[0] <= if part.halves { 1 } else { 3 },
            Sel::Halves => part.halves && w.len() == 2,
            Sel::Bytes => !part.halves && w.len() == 4,
            Sel::HalfMask => part.halves && matches!(w, [0] | [1] | [1, 0]),
            Sel::ByteMask => !part.halves && w.iter().all(|&x| x <= 3) && w.windows(2).all(|p| p[0] > p[1]),
        }
    }

    /// What they are, for an error.
    pub fn what(self) -> &'static str {
        match self {
            Sel::One => "a byte or a half of a variable here, `.b0` to `.b3`, `.h0` or `.h1`",
            Sel::Halves => "two of the four halves of the two sources here, as `.h10` or `.h32`",
            Sel::Bytes => "four of the eight bytes of the two sources here, as `.b3210` or `.b7654`",
            Sel::HalfMask => "the halves of the result it writes here, `.h0`, `.h1` or `.h10`",
            Sel::ByteMask => {
                "the bytes of the result it writes here, from `.b3` down to `.b0`, each once at most, as `.b31`"
            }
        }
    }
}

/// A part of a register, as a video instruction selects it after a dot: bytes, `b`, or halves,
/// `h`, numbered from the lowest, as `b0` and `h10` are; of two sources together, those of the
/// second after the first's, as `b7654` is the bytes of the second.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    /// Whether it is halves, not bytes.
    pub halves: bool,
    /// Which, as written, and how many.
    which: [u8; 4],
    len: u8,
}

impl Part {
    /// The part `field` names, if a video instruction has it: four bytes at most, each 0 to
    /// 7, or two halves, each 0 to 3.
    pub fn parse(field: &str) -> Option<Part> {
        let (halves, digits) = match field.as_bytes() {
            [b'b', d @ ..] => (false, d),
            [b'h', d @ ..] => (true, d),
            _ => return None,
        };
        let (most, top) = if halves { (2, b'3') } else { (4, b'7') };
        if digits.is_empty() || digits.len() > most || !digits.iter().all(|d| (b'0'..=top).contains(d)) {
            return None;
        }
        let mut which = [0; 4];
        for (w, d) in which.iter_mut().zip(digits) {
            *w = d - b'0';
        }
        Some(Part { halves, which, len: digits.len() as u8 })
    }

    /// Which bytes or halves, as written.
    pub fn which(&self) -> &[u8] {
        &self.which[..usize::from(self.len)]
    }
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(if self.halves { "h" } else { "b" })?;
        self.which().iter().try_for_each(|w| write!(f, "{w}"))
    }
}

/// An operand of a form.
#[derive(Clone, Debug)]
pub struct Spec {
    /// What it is.
    pub role: Role,
    /// Its type, for a register or a value.
    pub ty: Option<TyRef>,
    /// How many.
    pub count: Count,
    /// The type of the first of so many, where it is not the type of the rest: the layer of
    /// an array, among the coordinates in a texture.
    pub lead: Option<Ty>,
    /// Whether it must be a variable, not a value known now.
    pub reg_only: bool,
    /// Whether `_` may stand for it.
    pub sink: bool,
    /// Whether it may be read inverted, `!p`.
    pub not: bool,
    /// Whether it may be negated, `-a`.
    pub neg: bool,
    /// Whether it may be a special register, `tid.x`.
    pub sreg: bool,
    /// Whether it is memory the kernel declares, which stands for its address there.
    pub memory: bool,
    /// What a value known now is from and to, if anything: 0 to 15 for a barrier.
    pub range: Option<(i64, i64)>,
    /// What a value known now is a multiple of, and more than 0: a barrier's threads are
    /// warps.
    pub step: Option<i64>,
    /// Whether a value known now is a fraction, more than 0 and at most 1.
    pub fraction: bool,
    /// The parts of a register it may be, and whether it must be one of them.
    pub part: Option<(Sel, bool)>,
    /// The word the name has when it is there: the policy of `.L2::cache_hint`.
    pub when: Option<&'static str>,
    /// Whether it may be left out, at the end.
    pub optional: bool,
    /// The second of two written at once, `a|b`, and whether it may be left out.
    pub second: Option<(Box<Spec>, bool)>,
    /// What it needs where it is there, beyond what the form needs: an offset's PTX 4.3.
    pub needs: Needs,
    /// The operands in brackets of a [`Role::Handle`].
    pub inner: Vec<Spec>,
}

/// A check of a name's words together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    /// A vector's registers together are 128 bits or fewer.
    Vec,
    /// A conversion rounds where it must and only there, and saturates only where it can.
    Cvt,
    /// `.sys` with `.b128` needs PTX 8.4.
    B128Sys,
    /// Of the operands that may be negated, the product of the first two, which negating one
    /// of them negates, and the third are not both negated.
    Neg,
    /// A texture returns four, `.v4`, but two of `.f16x2`, `.v2`.
    Tex,
}

/// Where an address points: the state space the name gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// Any, by a generic address.
    Generic,
    /// Global memory.
    Global,
    /// The CTA's shared memory.
    Shared,
    /// The thread's own.
    Local,
    /// Constant memory.
    Const,
    /// Parameters.
    Param,
}

impl Space {
    /// Whether its addresses are 64 bits: generic and global ones are, under
    /// `.address_size 64`; the others fit 32.
    pub fn wide(self) -> bool {
        matches!(self, Space::Generic | Space::Global)
    }

    /// The state space a word of a name is, `shared` or `shared::cta`, if it is one.
    pub fn of(word: &str) -> Option<Space> {
        match word.split("::").next().unwrap_or_default() {
            "global" => Some(Space::Global),
            "shared" => Some(Space::Shared),
            "local" => Some(Space::Local),
            "const" => Some(Space::Const),
            "param" => Some(Space::Param),
            _ => None,
        }
    }

    /// The word for it, as in `ld.shared`; nothing for a generic address.
    pub fn word(self) -> &'static str {
        match self {
            Space::Generic => "",
            Space::Global => "global",
            Space::Shared => "shared",
            Space::Local => "local",
            Space::Const => "const",
            Space::Param => "param",
        }
    }
}

/// A form that a name is of, with the words it took.
#[derive(Clone, Debug)]
pub struct Bound {
    /// The form.
    pub form: &'static Form,
    /// Which word each element took, by its place among [`Elem::alts`]; `None` for one the
    /// name leaves out.
    pub choice: Vec<Option<usize>>,
    /// The types the name gives, by slot.
    pub types: Vec<Ty>,
    /// What the form and its words need.
    pub needs: Needs,
}

impl Bound {
    /// The words the name took, after its first.
    pub fn words(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.form
            .elems
            .iter()
            .zip(&self.choice)
            .flat_map(|(e, c)| c.map(|k| e.alts[k].words.as_slice()))
            .flatten()
            .copied()
    }

    /// Whether the name has `word`.
    pub fn has(&self, word: &str) -> bool {
        self.words().any(|w| w == word)
    }

    /// The type `t` is.
    pub fn ty(&self, t: TyRef) -> Ty {
        match t {
            TyRef::Slot(i) => self.types[i],
            TyRef::Fixed(t) => t,
        }
    }

    /// How many registers a vector of [`Count::Vec`] is: 2 for `.v2`, 4 for `.v4`, and 1 for
    /// a name without either.
    pub fn lanes(&self) -> usize {
        if self.has("v4") {
            4
        } else if self.has("v2") {
            2
        } else {
            1
        }
    }

    /// The first state space the name gives, where the memory it takes the address of is:
    /// `cvta.shared`'s.
    pub fn space(&self) -> Space {
        self.words().find_map(Space::of).unwrap_or(Space::Generic)
    }

    /// Where each of its addresses points, in order: into the state space the name gives in
    /// the same place, or the last it gives, as `cp.async.ca.shared.global` writes shared
    /// memory and reads global; anywhere, by a generic address, if it gives none.
    pub fn spaces(&self) -> Vec<Space> {
        let given: Vec<Space> = self.words().filter_map(Space::of).collect();
        let addrs = self.form.operands.iter().filter(|s| matches!(s.role, Role::Addr(_))).count();
        (0..addrs).map(|i| given.get(i).or(given.last()).copied().unwrap_or(Space::Generic)).collect()
    }
}

/// Every form, parsed from [`TABLE`] once.
pub fn forms() -> &'static [Form] {
    static FORMS: OnceLock<Vec<Form>> = OnceLock::new();
    FORMS.get_or_init(|| parse(TABLE))
}

/// The instructions of PTX's for GPUs newer than kgen writes for, and the oldest each is for.
const NEWER: &[(&str, &str)] = &[
    ("elect", "sm_90"),
    ("getctarank", "sm_90"),
    ("griddepcontrol", "sm_90"),
    ("mapa", "sm_90"),
    ("multimem", "sm_90"),
    ("stmatrix", "sm_90"),
    ("tensormap", "sm_90"),
    ("setmaxnreg", "sm_90a"),
    ("wgmma", "sm_90a"),
    ("clusterlaunchcontrol", "sm_100"),
    ("fabric", "sm_100"),
    ("tcgen05", "sm_100a"),
    ("spcompress", "sm_107a"),
    ("spdecompress", "sm_107a"),
];

/// What of PTX's kgen leaves out, and why: an instruction, by its name and no word, or a word
/// of one kgen knows.
const LEFT_OUT: &[(&str, &str, &str)] = &[
    ("call", "", "calls a `.func`, which kgen does not write: a function without an ABI is inlined where it is called"),
    (
        "txq",
        "force_unnormalized_coords",
        "asks of a sampler apart from its texture, in the independent mode, which kgen does not write",
    ),
];

/// The forms the instruction `name` is of, its words as PTX writes them, in the order of
/// [`TABLE`]; or what is wrong with its name.
pub fn find(name: &str) -> Result<Vec<Bound>, String> {
    let mut words = name.split('.');
    let op = words.next().unwrap_or_default();
    let words: Vec<&str> = words.collect();
    let of: Vec<&'static Form> = forms().iter().filter(|f| f.op == op).collect();
    if of.is_empty() {
        if let Some((.., why)) = LEFT_OUT.iter().find(|&&(o, w, _)| o == op && w.is_empty()) {
            return Err(format!("PTX's `{op}` {why}"));
        }
        return Err(match NEWER.iter().find(|(o, _)| *o == op) {
            Some((_, gpu)) => format!("PTX's `{op}` is for {gpu} and newer, and kgen writes PTX for sm_20 to sm_86"),
            None => format!("PTX has no instruction `{op}` that kgen knows"),
        });
    }
    let mut bound = vec![];
    for &form in &of {
        let mut choice = vec![];
        if bind(&form.elems, &words, 0, &mut choice) {
            bound.push(make(form, choice));
        }
    }
    if bound.is_empty() {
        return Err(unknown(name, op, &words, &of));
    }
    let mut first = None;
    let mut kept = vec![];
    for mut b in bound {
        match check(&b, name) {
            Ok(n) => {
                b.needs = b.needs.and(n);
                kept.push(b);
            }
            Err(m) => {
                first.get_or_insert(m);
            }
        }
    }
    match (kept.is_empty(), first) {
        (true, Some(m)) => Err(m),
        _ => Ok(kept),
    }
}

/// Matches `words` from `at` with the elements after those `choice` took.
fn bind(elems: &[Elem], words: &[&str], at: usize, choice: &mut Vec<Option<usize>>) -> bool {
    let Some(e) = elems.get(choice.len()) else { return at == words.len() };
    for (k, alt) in e.alts.iter().enumerate() {
        let n = alt.words.len();
        if words.get(at..at + n) == Some(&alt.words[..]) {
            choice.push(Some(k));
            if bind(elems, words, at + n, choice) {
                return true;
            }
            choice.pop();
        }
    }
    if e.optional {
        choice.push(None);
        if bind(elems, words, at, choice) {
            return true;
        }
        choice.pop();
    }
    false
}

fn make(form: &'static Form, choice: Vec<Option<usize>>) -> Bound {
    let mut needs = form.needs;
    let mut types = vec![];
    for (e, c) in form.elems.iter().zip(&choice) {
        if let Some(k) = *c {
            needs = needs.and(e.alts[k].needs);
            if e.slot.is_some() {
                types.push(Ty::parse(e.alts[k].words[0]).expect("a slot's words are types"));
            }
        }
    }
    let mut b = Bound { form, choice, types, needs };
    // An address in no state space is generic, which PTX has from 2.0 on.
    if b.spaces().contains(&Space::Generic) {
        b.needs = b.needs.and(Needs { ptx: 20, ..Needs::default() });
    }
    b
}

/// Why a name of `op` is of no form, from the form it comes nearest.
fn unknown(name: &str, op: &str, words: &[&str], of: &[&'static Form]) -> String {
    let known = |w: &str| of.iter().any(|f| f.elems.iter().any(|e| e.alts.iter().any(|a| a.words.contains(&w))));
    if let Some(w) = words.iter().find(|w| !known(w)) {
        if let Some((.., why)) = LEFT_OUT.iter().find(|&&(o, x, _)| o == op && x == *w) {
            return format!("PTX's `{op}.{w}` {why}");
        }
        return format!("PTX's `{op}` has no `.{w}` that kgen knows");
    }
    let mut best: Option<Near> = None;
    for f in of {
        let n = near(f, words);
        let better = match &best {
            None => true,
            Some(b) => n.distance < b.distance || (n.distance == b.distance && n.why().is_some() && b.why().is_none()),
        };
        if better {
            best = Some(n);
        }
    }
    let n = best.expect("an instruction kgen knows has forms");
    let canon = n.canon.join(".");
    if n.distance == 0 {
        return format!("PTX writes `{name}` as `{canon}`, its words in that order");
    }
    match (n.missing.as_slice(), n.extra.as_slice()) {
        ([e], []) => match n.why() {
            Some("round") => {
                format!("`{name}` leaves ptxas to choose how it rounds, and to fuse it; say so: `{canon}`")
            }
            Some("half") => {
                format!(
                    "`{name}` keeps the low half of the product, the high or both: `{op}.lo`, `{op}.hi`, `{op}.wide`"
                )
            }
            Some("cmp") => format!("`{name}` names one comparison, as in `{canon}`"),
            _ => {
                let alts: Vec<String> = e.alts.iter().map(|a| format!("`.{}`", a.words.join("."))).collect();
                let what = if alts.len() == 1 { alts[0].clone() } else { format!("one of {}", alts.join(", ")) };
                format!("`{name}` leaves out {what}: `{canon}`")
            }
        },
        ([], [w]) => format!("PTX's `{op}` takes no `.{w}` with the rest of `{name}`: `{canon}`"),
        _ => format!("`{name}` is no form of `{op}` that kgen knows; the nearest is `{canon}`"),
    }
}

/// How near a name's words come to a form: what it would be with the fewest words added or
/// taken away.
struct Near {
    distance: usize,
    canon: Vec<&'static str>,
    missing: Vec<&'static Elem>,
    extra: Vec<String>,
}

impl Near {
    fn why(&self) -> Option<&'static str> {
        match self.missing.as_slice() {
            [e] if self.extra.is_empty() => e.why,
            _ => None,
        }
    }
}

fn near(form: &'static Form, words: &[&str]) -> Near {
    let mut left: Vec<&str> = words.to_vec();
    let (mut canon, mut missing, mut short) = (vec![form.op], vec![], 0);
    for e in &form.elems {
        // A type is the next the name has, where it may be: each says what it is of in its
        // place, as `tld4.r.2d.v4.f32.s32` writes `f32`s of `s32` coordinates.
        let next = left.iter().find(|w| Ty::parse(w).is_some());
        let typed = e.slot.and(next).and_then(|w| e.alts.iter().find(|a| a.words == [*w]));
        let mut best: Option<(usize, &Alt)> = typed.map(|a| (1, a));
        if typed.is_none() {
            for a in &e.alts {
                let k = a.words.iter().filter(|w| left.contains(w)).count();
                if best.is_none_or(|(b, _)| k > b) {
                    best = Some((k, a));
                }
            }
        }
        let Some((k, a)) = best else { continue };
        if k == 0 && e.optional {
            continue;
        }
        for w in &a.words {
            match left.iter().position(|x| x == w) {
                Some(p) => {
                    left.remove(p);
                }
                None => short += 1,
            }
        }
        if k < a.words.len() {
            missing.push(e);
        }
        canon.extend(&a.words);
    }
    Near { distance: left.len() + short, canon, missing, extra: left.into_iter().map(String::from).collect() }
}

/// What `b`'s checks need, or why its words cannot go together.
fn check(b: &Bound, name: &str) -> Result<Needs, String> {
    let mut needs = Needs::default();
    for c in &b.form.checks {
        match c {
            Check::Vec => {
                let bits = b.lanes() as u32 * b.types[0].bits();
                if bits > 128 {
                    return Err(format!(
                        "`{name}` moves {bits} bits at once; a vector of PTX's holds 128 at most, as `.v4.b32` or \
                         `.v2.b64`"
                    ));
                }
            }
            Check::B128Sys => {
                if b.has("sys") && b.types[0] == Ty::B128 {
                    needs = needs.and(Needs { ptx: 84, ..Needs::default() });
                }
            }
            Check::Cvt => needs = needs.and(cvt(b, name)?),
            Check::Tex => {
                let t = b.types[0].name();
                match (b.has("v2"), b.types[0] == Ty::F16x2) {
                    (true, false) => {
                        return Err(format!("`{name}` returns four of a `{t}`, `.v4`; two only of an `f16x2`, `.v2`"));
                    }
                    (false, true) => {
                        return Err(format!("`{name}` returns two of an `f16x2`, `.v2`; four only of another type"));
                    }
                    _ => {}
                }
            }
            // Of the operands, which the target checks.
            Check::Neg => {}
        }
    }
    Ok(needs)
}

/// PTX's rules for `cvt` between integers and floats: it rounds where it loses precision,
/// to an integer from a float, and only there or where it rounds a float to an integral
/// value; it flushes subnormals only of an `f32`; and it saturates only where the result can
/// fall outside what its type holds. What it needs beyond its words: ptxas takes `.ftz` from
/// PTX 1.4 on, but with a few conversions from 1.5.
fn cvt(b: &Bound, name: &str) -> Result<Needs, String> {
    let (d, a) = (b.types[0], b.types[1]);
    let float = |t: Ty| t.kind() == Kind::Float;
    let irnd = ["rni", "rzi", "rmi", "rpi"].into_iter().find(|w| b.has(w));
    let frnd = ["rn", "rz", "rm", "rp"].into_iter().find(|w| b.has(w));
    let plain = format!("cvt.{}.{}", d.name(), a.name());
    match (float(d), float(a)) {
        (false, false) if irnd.is_some() || frnd.is_some() => {
            return Err(format!("`{name}` converts an integer to an integer, which does not round: `{plain}`"));
        }
        (false, true) if irnd.is_none() => {
            return Err(format!(
                "`{name}` converts a float to an integer, so it says how it rounds to one: `cvt.rni.{}.{}` to \
                 the nearest, `cvt.rzi` toward zero",
                d.name(),
                a.name()
            ));
        }
        (true, false) if frnd.is_none() => {
            return Err(format!(
                "`{name}` converts an integer to a float, so it says how it rounds: `cvt.rn.{}.{}`",
                d.name(),
                a.name()
            ));
        }
        (true, true) if d.bits() < a.bits() && frnd.is_none() => {
            return Err(format!(
                "`{name}` loses precision, so it says how it rounds: `cvt.rn.{}.{}`",
                d.name(),
                a.name()
            ));
        }
        (true, true) if d.bits() == a.bits() && frnd.is_some() => {
            return Err(format!(
                "`{name}` loses no precision, so it does not round, but it may round to an integral value: \
                 `cvt.rni.{}.{}`",
                d.name(),
                a.name()
            ));
        }
        (true, true) if d.bits() > a.bits() && (frnd.is_some() || irnd.is_some()) => {
            return Err(format!("`{name}` loses no precision, so it does not round: `{plain}`"));
        }
        (true, false) | (false, false) if irnd.is_some() => {
            return Err(format!("`{name}` rounds to an integer only from a float"));
        }
        (false, true) if frnd.is_some() => {
            return Err(format!("`{name}` rounds a float to an integer: `.rni`, `.rzi`, `.rmi` or `.rpi`"));
        }
        _ => {}
    }
    if b.has("ftz") && d != Ty::F32 && a != Ty::F32 {
        return Err(format!("`.ftz` flushes the subnormals of an `f32`, and `{name}` has none"));
    }
    if b.has("sat") && !float(d) && !float(a) && holds(d, a) {
        return Err(format!("`{name}` has nothing to saturate: a `{}` holds every `{}`", d.name(), a.name()));
    }
    let later = [(Ty::F64, Ty::F32), (Ty::F32, Ty::S8), (Ty::F32, Ty::U8), (Ty::S64, Ty::F32), (Ty::U64, Ty::F32)];
    Ok(match b.has("ftz") && later.contains(&(d, a)) {
        true => Needs { ptx: 15, ..Needs::default() },
        false => Needs::default(),
    })
}

/// Whether every value of the integer type `a` is one of `d`.
fn holds(d: Ty, a: Ty) -> bool {
    match (d.kind(), a.kind()) {
        (x, y) if x == y => d.bits() >= a.bits(),
        (Kind::Signed, Kind::Unsigned) => d.bits() > a.bits(),
        _ => false,
    }
}

/// A special register: its name as an instruction's operand writes it, `tid.x`, its type,
/// and what it needs.
#[derive(Clone, Debug)]
pub struct Sreg {
    /// Its name, without PTX's `%`.
    pub name: String,
    /// Its type.
    pub ty: Ty,
    /// What reading it needs.
    pub needs: Needs,
    /// The fewest bits `mov` and `cvt` may read of it, which PTX allows of the registers that
    /// were once narrower: 16 for `tid.x`.
    pub fewest: u32,
}

impl Sreg {
    /// Whether an instruction of type `t` reads it: one of the type it is, by PTX's rules,
    /// or an integer of fewer bits, down to [`Sreg::fewest`].
    pub fn read_as(&self, t: Ty) -> bool {
        self.ty.fits(t)
            || (t.kind() != Kind::Float && t.kind() != Kind::Pred && (self.fewest..self.ty.bits()).contains(&t.bits()))
    }
}

/// Every special register kgen knows.
pub fn sregs() -> &'static [Sreg] {
    static SREGS: OnceLock<Vec<Sreg>> = OnceLock::new();
    SREGS.get_or_init(|| {
        let mut out = vec![];
        let mut add = |name: String, ty: Ty, ptx: u32, sm: u32, fewest: u32| {
            out.push(Sreg { name, ty, needs: Needs { ptx, sm, ..Needs::default() }, fewest });
        };
        for v in ["tid", "ntid", "ctaid", "nctaid"] {
            for c in ["x", "y", "z"] {
                add(format!("{v}.{c}"), Ty::U32, 10, 0, 16);
            }
        }
        add("gridid".into(), Ty::U64, 10, 0, 16);
        add("clock".into(), Ty::U32, 10, 0, 32);
        for i in 0..32 {
            add(format!("envreg{i}"), Ty::B32, 10, 0, 32);
        }
        for n in ["laneid", "warpid", "smid"] {
            add(n.into(), Ty::U32, 13, 0, 32);
        }
        for n in ["nwarpid", "nsmid", "lanemask_eq", "lanemask_le", "lanemask_lt", "lanemask_ge", "lanemask_gt"] {
            add(n.into(), Ty::U32, 20, 0, 32);
        }
        add("clock64".into(), Ty::U64, 20, 0, 64);
        add("clock_hi".into(), Ty::U32, 50, 0, 32);
        add("globaltimer".into(), Ty::U64, 31, 30, 64);
        add("globaltimer_lo".into(), Ty::U32, 31, 30, 32);
        add("globaltimer_hi".into(), Ty::U32, 31, 30, 32);
        for i in 0..8 {
            add(format!("pm{i}"), Ty::U32, if i < 4 { 13 } else { 30 }, 0, 32);
            add(format!("pm{i}_64"), Ty::U64, 40, 50, 64);
        }
        add("total_smem_size".into(), Ty::U32, 41, 0, 32);
        add("dynamic_smem_size".into(), Ty::U32, 41, 0, 32);
        for n in ["begin", "end", "cap", "0", "1"] {
            add(format!("reserved_smem_offset_{n}"), Ty::B32, 76, 80, 32);
        }
        add("current_graph_exec".into(), Ty::U64, 80, 50, 64);
        out
    })
}

/// The special register `name`, if it is one.
pub fn sreg(name: &str) -> Option<&'static Sreg> {
    sregs().iter().find(|s| s.name == name)
}

/// The special registers of PTX's that kgen reads none of, and why.
const BEYOND: &[(&str, &str)] = &[
    ("is_explicit_cluster", HOPPER),
    ("clusterid", HOPPER),
    ("nclusterid", HOPPER),
    ("cluster_ctaid", HOPPER),
    ("cluster_nctaid", HOPPER),
    ("cluster_ctarank", HOPPER),
    ("cluster_nctarank", HOPPER),
    ("aggr_smem_size", HOPPER),
    ("perctamemoryoffset", PER_CTA),
    ("perctamemorysize", PER_CTA),
];
const HOPPER: &str = "it needs sm_90, and kgen writes PTX for sm_20 to sm_86";
const PER_CTA: &str = "a kernel has it only with the directive `.minperctamemory`, which kgen does not write";

/// Why kgen reads no special register `name`, `clusterid.x`, if it is one of PTX's.
pub fn beyond(name: &str) -> Option<&'static str> {
    let first = name.split('.').next().unwrap_or(name);
    BEYOND.iter().find(|(n, _)| *n == first).map(|&(_, why)| why)
}

/// Whether a register named `name` would be one of PTX's special registers, or of their
/// vectors: `%tid` is, and so are those kgen reads none of.
pub fn reserved(name: &str) -> bool {
    BEYOND.iter().any(|(n, _)| *n == name) || sregs().iter().any(|s| s.name.split('.').next() == Some(name))
}

fn parse(text: &'static str) -> Vec<Form> {
    let mut macros: Vec<(&'static str, Vec<Elem>)> = vec![];
    let mut operands: Vec<(&'static str, &'static str)> = vec![];
    let mut forms = vec![];
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(def) = line.strip_prefix('$') {
            if let Some((name, ops)) = def.split_once(" : ") {
                // `$U` is an operand a special register may be.
                assert!(Ty::parse(name).is_none() && !["T", "U", "V", "W"].contains(&name), "`{line}` is a type");
                operands.push((name, ops));
                continue;
            }
            let (name, elem) = def.split_once(" = ").unwrap_or_else(|| panic!("`{line}` defines `$name = ..`"));
            let (op, elems) = name_of(elem, &macros);
            assert!(op.is_empty(), "`{line}` is elements, not a name");
            macros.push((name, elems));
            continue;
        }
        forms.push(form(line, &macros, &operands));
    }
    forms
}

fn form(
    line: &'static str,
    macros: &[(&'static str, Vec<Elem>)],
    operand_macros: &[(&'static str, &'static str)],
) -> Form {
    let (name, rest) = match line.split_once(" : ") {
        Some((name, rest)) => (name, rest),
        None => line.split_once(' ').unwrap_or((line, "")),
    };
    let (op, mut elems) = name_of(name, macros);
    let mut slot = 0;
    for e in &mut elems {
        if e.alts.iter().all(|a| a.words.len() == 1 && Ty::parse(a.words[0]).is_some()) {
            assert!(!e.optional, "`{line}`: a type is not to be left out");
            e.slot = Some(slot);
            slot += 1;
        }
    }
    let mut tokens = vec![];
    for token in rest.split_whitespace() {
        let m = token.trim_end_matches(',').strip_prefix('$');
        match operand_macros.iter().find(|(name, _)| Some(*name) == m) {
            Some((_, ops)) => tokens.extend(ops.split_whitespace()),
            None => tokens.push(token),
        }
    }
    let (mut operands, mut needs, mut wide, mut checks) = (vec![], Needs::default(), false, vec![]);
    for token in tokens {
        if let Some(r) = token.strip_prefix('@') {
            needs = needs.and(req(r, line));
        } else if let Some(c) = token.strip_prefix('+') {
            match c {
                "wide" => wide = true,
                "vec" => checks.push(Check::Vec),
                "cvt" => checks.push(Check::Cvt),
                "b128sys" => checks.push(Check::B128Sys),
                "neg" => checks.push(Check::Neg),
                "tex" => checks.push(Check::Tex),
                _ => panic!("`{line}`: no check `+{c}`"),
            }
        } else {
            operands.push(spec(token.trim_end_matches(','), line));
        }
    }
    for s in &operands {
        if let Some(TyRef::Slot(i)) = s.ty {
            assert!(i < slot, "`{line}` has no type {i}");
        }
    }
    assert!(needs.ptx != 0, "`{line}` says the version of PTX it needs, `@1.0` for any");
    Form { op, elems, operands, needs, wide, checks, line }
}

/// The first word of `s` and the elements after it.
fn name_of(s: &'static str, macros: &[(&'static str, Vec<Elem>)]) -> (&'static str, Vec<Elem>) {
    let end = s.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(s.len());
    let (op, mut rest) = s.split_at(end);
    let mut elems: Vec<Elem> = vec![];
    while let Some(c) = rest.chars().next() {
        match c {
            '.' => {
                let (alt, r) = alt_of(rest, true, s);
                elems.push(Elem { alts: vec![alt], optional: false, slot: None, why: None });
                rest = r;
            }
            '{' | '(' => {
                let close = if c == '{' { '}' } else { ')' };
                let end = rest.find(close).unwrap_or_else(|| panic!("`{s}`: `{c}` is not closed"));
                let alts = rest[1..end]
                    .split('|')
                    .map(|a| {
                        let (alt, r) = alt_of(a, false, s);
                        assert!(r.is_empty(), "`{s}`: `{a}` is one choice");
                        alt
                    })
                    .collect();
                elems.push(Elem { alts, optional: c == '{', slot: None, why: None });
                rest = &rest[end + 1..];
            }
            '$' => {
                let end =
                    rest[1..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(rest.len(), |e| e + 1);
                let m = &rest[1..end];
                let (_, more) = macros.iter().find(|(n, _)| *n == m).unwrap_or_else(|| panic!("`{s}`: no `${m}`"));
                elems.extend(more.iter().cloned());
                rest = &rest[end..];
            }
            '~' => {
                let end = rest[1..].find(|c: char| !c.is_ascii_alphabetic()).map_or(rest.len(), |e| e + 1);
                elems.last_mut().unwrap_or_else(|| panic!("`{s}`: `~` follows an element")).why = Some(&rest[1..end]);
                rest = &rest[end..];
            }
            _ => panic!("`{s}`: `{rest}` is no element"),
        }
    }
    (op, elems)
}

/// One choice, `.a.b@7.0@sm80`, and what follows it: its first word only, if `one`.
fn alt_of(s: &'static str, one: bool, line: &str) -> (Alt, &'static str) {
    let mut rest = s;
    let mut words = vec![];
    while let Some(r) = rest.strip_prefix('.') {
        let end = r.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':')).unwrap_or(r.len());
        words.push(&r[..end]);
        rest = &r[end..];
        if one {
            break;
        }
    }
    assert!(!words.is_empty(), "`{line}`: `{s}` has no word");
    let mut needs = Needs::default();
    while let Some(r) = rest.strip_prefix('@') {
        let digits = |from: usize| from + r[from..].find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len() - from);
        let version = |from: usize| match digits(from) {
            major if r[major..].starts_with('.') => digits(major + 1),
            major => major,
        };
        let end = match r.strip_prefix('<').unwrap_or(r).strip_prefix("sm") {
            Some(_) => match digits(r.find("sm").expect("it starts so") + 2) {
                sm if r[sm..].starts_with('/') => version(sm + 1),
                sm => sm,
            },
            None => version(0),
        };
        needs = needs.and(req(&r[..end], line));
        rest = &r[end..];
    }
    (Alt { words, needs }, rest)
}

/// A requirement after `@`: `7.8`, `sm80` or `<sm70/6.4`.
fn req(r: &str, line: &str) -> Needs {
    let num = |s: &str| s.parse::<u32>().unwrap_or_else(|_| panic!("`{line}`: `@{r}` is no requirement"));
    let gpu = |s: &str| {
        let sm = num(s);
        assert!(GPUS.iter().any(|&(g, _)| g == sm), "`{line}`: `@{r}` is no GPU kgen writes for");
        sm
    };
    let version = |s: &str| {
        let (major, minor) = s.split_once('.').unwrap_or_else(|| panic!("`{line}`: `@{r}` is no version"));
        num(major) * 10 + num(minor)
    };
    if let Some(sm) = r.strip_prefix("<sm") {
        let (sm, since) = sm.split_once('/').unwrap_or_else(|| panic!("`{line}`: `@{r}` says from which version"));
        Needs { below: gpu(sm), since: version(since), ..Needs::default() }
    } else if let Some(sm) = r.strip_prefix("sm") {
        Needs { sm: gpu(sm), ..Needs::default() }
    } else {
        Needs { ptx: version(r), ..Needs::default() }
    }
}

/// An operand, as [`TABLE`] writes it.
fn spec(s: &'static str, line: &str) -> Spec {
    let (when, s) = match s.strip_prefix('{') {
        Some(r) => {
            let (w, r) = r.split_once('}').unwrap_or_else(|| panic!("`{line}`: `{s}` is not closed"));
            (Some(w), r)
        }
        None => (None, s),
    };
    let (optional, s) = match s.strip_suffix('?') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let mut first = if let Some((a, b)) = s.split_once("[|") {
        let mut first = one(a, line);
        first.second = Some((Box::new(one(b.strip_suffix(']').unwrap_or(b), line)), true));
        first
    } else if let Some((a, b)) = s.split_once('|').filter(|_| !s.starts_with(['[', '#'])) {
        let mut first = one(a, line);
        first.second = Some((Box::new(one(b, line)), false));
        first
    } else {
        one(s, line)
    };
    first.when = when;
    first.optional = optional;
    first
}

fn one(s: &'static str, line: &str) -> Spec {
    let (s, needs) = match s.split_once('@') {
        Some((s, r)) => (s, r.split('@').fold(Needs::default(), |n, r| n.and(req(r, line)))),
        None => (s, Needs::default()),
    };
    let mut spec = Spec {
        role: Role::Src,
        ty: None,
        count: Count::One,
        lead: None,
        reg_only: false,
        sink: false,
        not: false,
        neg: false,
        sreg: false,
        memory: false,
        range: None,
        step: None,
        fraction: false,
        part: None,
        when: None,
        optional: false,
        second: None,
        needs,
        inner: vec![],
    };
    let num = |n: &str| n.parse::<i64>().unwrap_or_else(|_| panic!("`{line}`: `{s}` has no number `{n}`"));
    match s {
        "[R]" => spec.role = Role::Addr(Access::Read),
        "[W]" => spec.role = Role::Addr(Access::Write),
        "[X]" => spec.role = Role::Addr(Access::ReadWrite),
        "lab" => spec.role = Role::Label,
        "labs" => spec.role = Role::Labels,
        "#" => spec.role = Role::Imm(Imm::Bits(32)),
        _ if s.starts_with('[') => {
            let inner = s[1..].strip_suffix(']').unwrap_or_else(|| panic!("`{line}`: `{s}` is not closed"));
            spec.role = Role::Handle;
            spec.inner = inner.split(',').map(|o| one(o, line)).collect();
        }
        _ if s.starts_with('#') => {
            let r = &s[1..];
            spec.role = Role::Imm(if let Some(n) = r.strip_prefix('=') {
                n.split('|').for_each(|v| _ = num(v));
                Imm::Of(n)
            } else if let Some(n) = r.strip_prefix('^') {
                Imm::Pow2(num(n) as u32)
            } else {
                Imm::Bits(num(r) as u32)
            });
        }
        _ => {
            let mut t = s;
            while let Some(c) = t.chars().next().filter(|c| "=%_!-$&".contains(*c)) {
                match c {
                    '=' => spec.role = Role::Dst,
                    '%' => spec.reg_only = true,
                    '_' => spec.sink = true,
                    '!' => spec.not = true,
                    '-' => spec.neg = true,
                    '&' => spec.memory = true,
                    _ => spec.sreg = true,
                }
                t = &t[1..];
            }
            if let Some((lead, r)) = t.split_once('+') {
                spec.lead = Some(Ty::parse(lead).unwrap_or_else(|| panic!("`{line}`: `{lead}` is no type")));
                t = r;
            }
            if let Some((ty, r)) = t.split_once("{.").or(t.split_once("(.")) {
                let must = t[ty.len()..].starts_with('(');
                let sel = match r.strip_suffix(if must { ')' } else { '}' }) {
                    Some("bh") => Sel::One,
                    Some("hxy") => Sel::Halves,
                    Some("bxyzw") => Sel::Bytes,
                    Some("hmask") => Sel::HalfMask,
                    Some("bmask") => Sel::ByteMask,
                    _ => panic!("`{line}`: `{s}` has no parts that kgen knows"),
                };
                spec.part = Some((sel, must));
                t = ty;
            }
            if let Some((ty, r)) = t.split_once('[') {
                let (lo, hi) =
                    r.strip_suffix(']').and_then(|r| r.split_once(',')).unwrap_or_else(|| panic!("`{line}`: `{s}`"));
                spec.range = Some((num(lo), num(hi)));
                t = ty;
            }
            if let Some((ty, n)) = t.split_once('/') {
                spec.step = Some(num(n));
                t = ty;
            }
            if let Some(ty) = t.strip_suffix("(0,1]") {
                spec.fraction = true;
                t = ty;
            }
            let (ty, count) = match t.split_once('*') {
                Some((ty, "v")) => (ty, Count::Vec),
                Some((ty, n)) => (ty, Count::Fixed(n.parse().unwrap_or_else(|_| panic!("`{line}`: `{s}`")))),
                None => (t, Count::One),
            };
            let count = match (count, spec.lead) {
                (_, None) => count,
                (Count::Fixed(n), Some(_)) => Count::Fixed(n + 1),
                _ => panic!("`{line}`: `{s}` leads so many taken together, `u32+T*3`"),
            };
            spec.ty = Some(match ty {
                "T" => TyRef::Slot(0),
                "U" => TyRef::Slot(1),
                "V" => TyRef::Slot(2),
                "W" => TyRef::Slot(3),
                _ => TyRef::Fixed(Ty::parse(ty).unwrap_or_else(|| panic!("`{line}`: `{ty}` is no type"))),
            });
            spec.count = count;
        }
    }
    spec
}
