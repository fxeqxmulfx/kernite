//! The instructions of PTX that kgen knows: those the GPUs from Pascal, `sm_61`, to Ampere,
//! `sm_86`, run, written as the PTX ISA writes them, one [`Form`] each, and the special
//! registers they read.
//!
//! A form is a line of [`TABLE`]: the words of an instruction's name, some to choose from
//! and some to leave out, its operands, and the version of PTX and the GPU it and each of
//! its words need. [`find`] gives the forms a name is of, or why it is of none.

use std::sync::OnceLock;

use kgen_core::Access;

use crate::{Kind, Needs, Ty};

/// The forms, a line each.
///
/// A form's name is PTX's, its words in PTX's order: `.w` is a word it has, `{.a|.b}` one it
/// may have, `(.a|.b)` one of these it has, and `$m` the elements a line `$m = ..` stands for.
/// A word, or a choice of words, that is a type is one of the instruction's types, which its
/// operands name by place: `T` is the first, then `U`, `V` and `W`. `@7.8` after a word or a
/// form is the version of PTX it needs, `@sm80` the GPU, and `@<sm70` a GPU older than that.
/// `~round`, `~half` and `~cmp` say what leaving an element out leaves unsaid.
///
/// After ` : ` are the operands. One is a type, `T` or `u32`, with `=` before it for one the
/// instruction writes, `%` for one that must be a variable, `_` for one `_` may stand for,
/// `!` for a predicate it may read inverted, and `$` for one a special register may be; and
/// after it `*2` for so many taken together, `*v` for as many as `.v2` or `.v4` say. `a|b` is
/// two written at once, and `a[|b]` one or two. `u32<16` is a value below 16, if it is one
/// known now, `u32/32` a multiple of 32, and `f32(0,1]` a fraction, more than 0 and at most 1.
/// `[R]`, `[W]` and `[X]` are memory read, written,
/// or both; `lab` is a label; `#` is an integer known now, `#8` one of 8 bits, `#=128` that
/// one, and `#^23` a power of two to 2^23. `{w}` before an operand has it only with the word
/// `w`, and `?` after it lets it be left out. Last, `+wide` lets a register be wider than the
/// instruction's type, as `ld`, `st`
/// and `cvt` do; `+vec` holds a vector to 128 bits; `+cvt` holds a conversion to PTX's rules
/// for rounding and saturating; and `+b128sys` gives `.sys` with `.b128` the version it needs.
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
$cc = (.u32|.s32|.u64|.s64)
$space = (.const|.global|.local|.shared|.shared::cta@7.8|.param@7.7@sm70|.param::entry@8.3@sm70)
$ldss = (.const|.local|.param|.param::entry@8.3|.shared|.shared::cta@7.8)
$shared = (.shared|.shared::cta@7.8)
$cop = {.ca|.cg|.cs|.lu|.cv}
$scop = {.wb|.cg|.cs|.wt}
$l1 = {.L1::evict_normal@7.4@sm70|.L1::evict_unchanged@7.4@sm70|.L1::evict_first@7.4@sm70|.L1::evict_last@7.4@sm70|.L1::no_allocate@7.4@sm70}
$hint = {.L2::cache_hint@7.4@sm80}
$size = {.L2::64B@7.4@sm75|.L2::128B@7.4@sm75|.L2::256B@7.4@sm80}
$vec = {.v2|.v4}
$ty = (.b8|.b16|.b32|.b64|.b128@8.3@sm70|.u8|.u16|.u32|.u64|.s8|.s16|.s32|.s64|.f32|.f64)
$ty64 = (.b8|.b16|.b32|.b64|.u8|.u16|.u32|.u64|.s8|.s16|.s32|.s64|.f32|.f64)
$scope = (.cta|.gpu|.sys)
$sem = {.relaxed@6.0@sm70|.acquire@6.0@sm70|.release@6.0@sm70|.acq_rel@6.0@sm70}
$rsem = {.relaxed@6.0@sm70|.release@6.0@sm70}
$ascope = {.cta|.gpu|.sys}
$cvty = (.u8|.u16|.u32|.u64|.s8|.s16|.s32|.s64|.f16|.f32|.f64)
$prio = (.L2::evict_last|.L2::evict_normal|.L2::evict_first|.L2::evict_unchanged)
$prio2 = {.L2::evict_first|.L2::evict_unchanged}
$cta = {.cta@7.8}
$shfl = (.up|.down|.bfly|.idx)

# Integers.
add$int : =T, T, T
add.sat.s32 : =T, T, T
sub$int : =T, T, T
sub.sat.s32 : =T, T, T
mul(.hi|.lo)~half$int : =T, T, T
mul.wide~half.u16 : =u32, T, T
mul.wide~half.u32 : =u64, T, T
mul.wide~half.s16 : =s32, T, T
mul.wide~half.s32 : =s64, T, T
mad(.hi|.lo)~half$int : =T, T, T, T
mad.hi.sat.s32 : =T, T, T, T
mad.wide~half.u16 : =u32, T, T, u32
mad.wide~half.u32 : =u64, T, T, u64
mad.wide~half.s16 : =s32, T, T, s32
mad.wide~half.s32 : =s64, T, T, s64
clmad(.hi|.lo).u64 : =T, T, T, T @9.3 @sm80
mul24(.hi|.lo)(.u32|.s32) : =T, T, T
mad24(.hi|.lo)(.u32|.s32) : =T, T, T, T
mad24.hi.sat.s32 : =T, T, T, T
sad$int : =T, T, T, T
div$int : =T, T, T
rem$int : =T, T, T
abs(.s16|.s32|.s64) : =T, T
neg(.s16|.s32|.s64) : =T, T
min$int : =T, T, T
max$int : =T, T, T
popc(.b32|.b64) : =u32, T
clz(.b32|.b64) : =u32, T
bfind{.shiftamt}(.u32|.u64|.s32|.s64) : =u32, T
fns.b32 : =T, T, T, s32
brev(.b32|.b64) : =T, T
bfe(.u32|.u64|.s32|.s64) : =T, T, u32, u32
bfi(.b32|.b64) : =T, T, T, u32, u32
szext(.clamp|.wrap)(.u32|.s32) : =T, T, u32 @7.6 @sm70
bmsk(.clamp|.wrap).b32 : =T, u32, u32 @7.6 @sm70
dp4a.u32.u32 : =u32, T, U, u32
dp4a.u32.s32 : =s32, T, U, s32
dp4a.s32.u32 : =s32, T, U, s32
dp4a.s32.s32 : =s32, T, U, s32
dp2a(.lo|.hi).u32.u32 : =u32, T, U, u32
dp2a(.lo|.hi).u32.s32 : =s32, T, U, s32
dp2a(.lo|.hi).s32.u32 : =s32, T, U, s32
dp2a(.lo|.hi).s32.s32 : =s32, T, U, s32
add.cc$cc : =T, T, T
addc{.cc}$cc : =T, T, T
sub.cc$cc : =T, T, T
subc{.cc}$cc : =T, T, T
mad(.hi|.lo).cc$cc : =T, T, T, T
madc(.hi|.lo){.cc}$cc : =T, T, T, T

# Floats.
testp(.finite|.infinite|.number|.notanumber|.normal|.subnormal)(.f32|.f64) : =pred, T
copysign(.f32|.f64) : =T, T, T
add$rnd{.ftz}{.sat}.f32 : =T, T, T
add$rnd.f64 : =T, T, T
sub$rnd{.ftz}{.sat}.f32 : =T, T, T
sub$rnd.f64 : =T, T, T
mul$rnd{.ftz}{.sat}.f32 : =T, T, T
mul$rnd.f64 : =T, T, T
fma$rnd{.ftz}{.sat}.f32 : =T, T, T, T
fma$rnd.f64 : =T, T, T, T
mad$rnd{.ftz}{.sat}.f32 : =T, T, T, T
mad$rnd.f64 : =T, T, T, T
div.approx{.ftz}.f32 : =T, T, T
div.full{.ftz}.f32 : =T, T, T
div$rnd{.ftz}.f32 : =T, T, T
div$rnd.f64 : =T, T, T
abs{.ftz}.f32 : =T, T
abs.f64 : =T, T
neg{.ftz}.f32 : =T, T
neg.f64 : =T, T
min{.ftz}{.NaN@7.0@sm80}{.xorsign.abs@7.2@sm86}.f32 : =T, T, T
min.f64 : =T, T, T
max{.ftz}{.NaN@7.0@sm80}{.xorsign.abs@7.2@sm86}.f32 : =T, T, T
max.f64 : =T, T, T
rcp.approx{.ftz}.f32 : =T, T
rcp$rnd{.ftz}.f32 : =T, T
rcp$rnd.f64 : =T, T
rcp.approx.ftz.f64 : =T, T
sqrt.approx{.ftz}.f32 : =T, T
sqrt$rnd{.ftz}.f32 : =T, T
sqrt$rnd.f64 : =T, T
rsqrt.approx{.ftz}(.f32|.f64) : =T, T
sin.approx{.ftz}.f32 : =T, T
cos.approx{.ftz}.f32 : =T, T
lg2.approx{.ftz}.f32 : =T, T
ex2.approx{.ftz}.f32 : =T, T
tanh.approx.f32 : =T, T @7.0 @sm75

# Floats of 16 bits.
add.rn~round{.ftz}{.sat}(.f16|.f16x2) : =T, T, T
sub.rn~round{.ftz}{.sat}(.f16|.f16x2) : =T, T, T
mul.rn~round{.ftz}{.sat}(.f16|.f16x2) : =T, T, T
fma.rn~round{.ftz}{.sat}(.f16|.f16x2) : =T, T, T, T
fma.rn~round{.ftz}.relu(.f16|.f16x2) : =T, T, T, T @7.0 @sm80
fma.rn~round{.relu}(.bf16|.bf16x2) : =T, T, T, T @7.0 @sm80
neg{.ftz}(.f16|.f16x2) : =T, T
neg(.bf16|.bf16x2) : =T, T @7.0 @sm80
abs{.ftz}(.f16|.f16x2) : =T, T
abs(.bf16|.bf16x2) : =T, T @7.0 @sm80
min{.ftz}{.NaN}{.xorsign.abs@7.2@sm86}(.f16|.f16x2) : =T, T, T @7.0 @sm80
min{.NaN}{.xorsign.abs@7.2@sm86}(.bf16|.bf16x2) : =T, T, T @7.0 @sm80
max{.ftz}{.NaN}{.xorsign.abs@7.2@sm86}(.f16|.f16x2) : =T, T, T @7.0 @sm80
max{.NaN}{.xorsign.abs@7.2@sm86}(.bf16|.bf16x2) : =T, T, T @7.0 @sm80
tanh.approx(.f16|.f16x2) : =T, T @7.0 @sm75
ex2.approx(.f16|.f16x2) : =T, T @7.0 @sm75

# Comparisons and selection.
setp$cmpb$bits : =_pred[|=_pred], T, T
setp$cmpb$bool$bits : =_pred[|=_pred], T, T, !pred
setp$cmpu(.u16|.u32|.u64) : =_pred[|=_pred], T, T
setp$cmpu$bool(.u16|.u32|.u64) : =_pred[|=_pred], T, T, !pred
setp$cmps(.s16|.s32|.s64) : =_pred[|=_pred], T, T
setp$cmps$bool(.s16|.s32|.s64) : =_pred[|=_pred], T, T, !pred
setp$cmpf{.ftz}.f32 : =_pred[|=_pred], T, T
setp$cmpf$bool{.ftz}.f32 : =_pred[|=_pred], T, T, !pred
setp$cmpf.f64 : =_pred[|=_pred], T, T
setp$cmpf$bool.f64 : =_pred[|=_pred], T, T, !pred
setp$cmpf{.ftz}.f16 : =pred, T, T
setp$cmpf$bool{.ftz}.f16 : =pred, T, T, !pred
setp$cmpf{.ftz}.f16x2 : =pred|=pred, T, T
setp$cmpf$bool{.ftz}.f16x2 : =pred|=pred, T, T, !pred
set$cmpb(.u32|.s32|.f32)$bits : =T, U, U
set$cmpb$bool(.u32|.s32|.f32)$bits : =T, U, U, !pred
set$cmpu(.u32|.s32|.f32)(.u16|.u32|.u64) : =T, U, U
set$cmpu$bool(.u32|.s32|.f32)(.u16|.u32|.u64) : =T, U, U, !pred
set$cmps(.u32|.s32|.f32)(.s16|.s32|.s64) : =T, U, U
set$cmps$bool(.u32|.s32|.f32)(.s16|.s32|.s64) : =T, U, U, !pred
set$cmpf{.ftz}(.u32|.s32|.f32).f32 : =T, U, U
set$cmpf$bool{.ftz}(.u32|.s32|.f32).f32 : =T, U, U, !pred
set$cmpf(.u32|.s32|.f32).f64 : =T, U, U
set$cmpf$bool(.u32|.s32|.f32).f64 : =T, U, U, !pred
set$cmpb.f16$bits : =T, U, U
set$cmpb$bool.f16$bits : =T, U, U, !pred
set$cmpu.f16(.u16|.u32|.u64) : =T, U, U
set$cmpu$bool.f16(.u16|.u32|.u64) : =T, U, U, !pred
set$cmps.f16(.s16|.s32|.s64) : =T, U, U
set$cmps$bool.f16(.s16|.s32|.s64) : =T, U, U, !pred
set$cmpf{.ftz}.f16(.f16|.f32) : =T, U, U
set$cmpf$bool{.ftz}.f16(.f16|.f32) : =T, U, U, !pred
set$cmpf.f16.f64 : =T, U, U
set$cmpf$bool.f16.f64 : =T, U, U, !pred
set$cmpf{.ftz}(.u16|.s16|.u32|.s32).f16 : =T, U, U @6.5
set$cmpf$bool{.ftz}(.u16|.s16|.u32|.s32).f16 : =T, U, U, !pred @6.5
set$cmpf{.ftz}(.f16x2|.u32@6.5|.s32@6.5).f16x2 : =T, U, U
set$cmpf$bool{.ftz}(.f16x2|.u32@6.5|.s32@6.5).f16x2 : =T, U, U, !pred
selp$num : =T, T, T, pred
slct$num.s32 : =T, T, T, U
slct{.ftz}$num.f32 : =T, T, T, U

# Logic and shifts.
and(.pred|.b16|.b32|.b64) : =T, T, T
or(.pred|.b16|.b32|.b64) : =T, T, T
xor(.pred|.b16|.b32|.b64) : =T, T, T
not(.pred|.b16|.b32|.b64) : =T, T
cnot$bits : =T, T
lop3.b32 : =T, T, T, T, #8
lop3(.or|.and).b32 : =_T|=pred, T, T, T, #8, pred @8.2 @sm70
shf(.l|.r)(.clamp|.wrap).b32 : =T, T, T, u32
shl$bits : =T, T, u32
shr(.b16|.b32|.b64|.u16|.u32|.u64|.s16|.s32|.s64) : =T, T, u32

# Moving data, and converting it.
mov(.pred|.b16|.b32|.b64|.u16|.u32|.u64|.s16|.s32|.s64|.f32|.f64) : =T, $T
mov.b128 : =T, T @8.3 @sm70
mov.b16 : =T, b8*2
mov.b32 : =T, b16*2
mov.b32 : =T, b8*4
mov.b64 : =T, b32*2
mov.b64 : =T, b16*4
mov.b128 : =T, b64*2 @8.3 @sm70
mov.b128 : =T, b32*4 @8.3 @sm70
mov.b16 : =_b8*2, %T
mov.b32 : =_b16*2, %T
mov.b32 : =_b8*4, %T
mov.b64 : =_b32*2, %T
mov.b64 : =_b16*4, %T
mov.b128 : =_b64*2, %T @8.3 @sm70
mov.b128 : =_b32*4, %T @8.3 @sm70
shfl$shfl.b32 : =T[|=pred], T, u32, u32 @<sm70
shfl.sync$shfl.b32 : =T[|=pred], T, u32, u32, u32
prmt.b32{.f4e|.b4e|.rc8|.ecl|.ecr|.rc16} : =T, T, T, T
ld{.weak@6.0@sm70}{.global}$cop$hint$size$vec$ty : =T*v, [R], {L2::cache_hint}b64 +wide +vec
ld{.weak@6.0@sm70}{.global}$l1$hint$size$vec$ty : =T*v, [R], {L2::cache_hint}b64 +wide +vec
ld{.weak@6.0@sm70}$ldss$cop$vec$ty : =T*v, [R] +wide +vec
ld.volatile{.global}$size$vec$ty : =T*v, [R] +wide +vec
ld.volatile(.local@9.1|.shared|.shared::cta@7.8)$vec$ty : =T*v, [R] +wide +vec
ld(.relaxed|.acquire)$scope{.global}$l1$hint$size$vec$ty : =T*v, [R], {L2::cache_hint}b64 @6.0 @sm70 +wide +vec +b128sys
ld(.relaxed|.acquire)$scope$shared$vec$ty : =T*v, [R] @6.0 @sm70 +wide +vec +b128sys
ld.mmio.relaxed.sys{.global}$ty64 : =T, [R] @8.2 @sm70 +wide
ld.mmio.acquire.sys{.global}$ty64 : =T, [R] @9.3 @sm75 +wide
ld.global{.ca|.cg|.cs}.nc$hint$size$vec$ty : =T*v, [R], {L2::cache_hint}b64 +wide +vec
ld.global.nc$l1$hint$size$vec$ty : =T*v, [R], {L2::cache_hint}b64 +wide +vec
ldu{.global}$vec$ty : =T*v, [R] +wide +vec
st{.weak@6.0@sm70}{.global}$scop$hint$vec$ty : [W], T*v, {L2::cache_hint}b64 +wide +vec
st{.weak@6.0@sm70}{.global}$l1$hint$vec$ty : [W], T*v, {L2::cache_hint}b64 +wide +vec
st{.weak@6.0@sm70}(.local|.shared|.shared::cta@7.8)$scop$vec$ty : [W], T*v +wide +vec
st.volatile{.global|.local@9.1|.shared|.shared::cta@7.8}$vec$ty : [W], T*v +wide +vec
st(.relaxed|.release)$scope{.global}$l1$hint$vec$ty : [W], T*v, {L2::cache_hint}b64 @6.0 @sm70 +wide +vec +b128sys
st(.relaxed|.release)$scope$shared$vec$ty : [W], T*v @6.0 @sm70 +wide +vec +b128sys
st.mmio.relaxed.sys{.global}$ty64 : [W], T @8.2 @sm70 +wide
st.mmio.release.sys{.global}$ty64 : [W], T @9.3 @sm75 +wide
prefetch{.global|.local}(.L1|.L2) : [R]
prefetch.global(.L2::evict_last|.L2::evict_normal) : [R] @7.4 @sm80
prefetchu.L1 : [R]
applypriority{.global}.L2::evict_normal : [R], #=128 @7.4 @sm80
discard{.global}.L2 : [W], #=128 @7.4 @sm80
createpolicy.range{.global}$prio$prio2.b64 : =T, [R], u32, u32 @7.4 @sm80
createpolicy.fractional$prio$prio2.b64 : =T, f32(0,1]? @7.4 @sm80
createpolicy.cvt.L2.b64 : =T, b64 @7.4 @sm80
isspacep$space : =pred, %u64
cvta$space.u64 : =T, T
cvta.to$space.u64 : =T, T
cvt{.rni|.rzi|.rmi|.rpi|.rn|.rz|.rm|.rp}{.ftz}{.sat}$cvty$cvty : =T, $U +wide +cvt
cvt(.rn|.rz)~round{.relu@7.0@sm80}{.satfinite@8.1@sm70}(.f16|.bf16@7.0@sm80).f32 : =T, U +wide
cvt(.rn|.rz)~round{.relu}{.satfinite@8.1}(.f16x2|.bf16x2).f32 : =T, U, U @7.0 @sm80
cvt.rna{.satfinite@8.1}.tf32.f32 : =T, U @7.0 @sm80
cvt.f32.bf16 : =T, U @7.1 @sm80
cvt.pack.sat(.u16|.s16).s32 : =u32, s32, s32 @6.5 @sm72
cvt.pack.sat(.u8|.s8).s32.b32 : =u32, s32, s32, b32 @6.5 @sm72
cvt.pack.sat(.u4|.s4|.u2|.s2).s32.b32 : =u32, s32, s32, b32 @6.5 @sm75

# Control.
bra{.uni} : lab
ret{.uni}
exit

# Barriers, fences and atomics.
barrier$cta.sync{.aligned} : u32<16, u32/32?
barrier$cta.arrive{.aligned} : u32<16, u32/32
barrier$cta.red.popc{.aligned}.u32 : =T, u32<16, !pred
barrier$cta.red.popc{.aligned}.u32 : =T, u32<16, u32/32, !pred
barrier$cta.red(.and|.or){.aligned}.pred : =T, u32<16, !pred
barrier$cta.red(.and|.or){.aligned}.pred : =T, u32<16, u32/32, !pred
bar$cta.sync : u32<16, u32/32?
bar$cta.arrive : u32<16, u32/32
bar$cta.red.popc.u32 : =T, u32<16, !pred
bar$cta.red.popc.u32 : =T, u32<16, u32/32, !pred
bar$cta.red(.and|.or).pred : =T, u32<16, !pred
bar$cta.red(.and|.or).pred : =T, u32<16, u32/32, !pred
bar.warp.sync : u32
membar(.cta|.gl|.sys)
membar.proxy.alias @7.5 @sm60
fence{.sc|.acq_rel}$scope @6.0 @sm70
fence.proxy.alias @7.5 @sm70
atom$sem$ascope{.global}(.and|.or|.xor)$hint(.b32|.b64) : =T, [X], T, {L2::cache_hint}b64
atom$sem$ascope$shared(.and|.or|.xor)(.b32|.b64) : =T, [X], T
atom$sem$ascope{.global}(.add|.inc|.dec|.min|.max)$hint.u32 : =T, [X], T, {L2::cache_hint}b64
atom$sem$ascope$shared(.add|.inc|.dec|.min|.max).u32 : =T, [X], T
atom$sem$ascope{.global}(.add|.min|.max)$hint(.s32|.u64) : =T, [X], T, {L2::cache_hint}b64
atom$sem$ascope$shared(.add|.min|.max)(.s32|.u64) : =T, [X], T
atom$sem$ascope{.global}(.min|.max)$hint.s64 : =T, [X], T, {L2::cache_hint}b64
atom$sem$ascope$shared(.min|.max).s64 : =T, [X], T
atom$sem$ascope{.global}.add$hint(.f32|.f64) : =T, [X], T, {L2::cache_hint}b64
atom$sem$ascope$shared.add(.f32|.f64) : =T, [X], T
atom$sem$ascope{.global}.exch$hint(.b32|.b64) : =T, [X], T, {L2::cache_hint}b64
atom$sem$ascope$shared.exch(.b32|.b64) : =T, [X], T
atom$sem$ascope{.global}.cas(.b32|.b64) : =T, [X], T, T
atom$sem$ascope$shared.cas(.b32|.b64) : =T, [X], T, T
atom$sem$ascope{.global}.cas.b16 : =T, [X], T, T @6.3 @sm70
atom$sem$ascope$shared.cas.b16 : =T, [X], T, T @6.3 @sm70
atom$sem$ascope{.global}.add.noftz$hint.f16 : =T, [X], T, {L2::cache_hint}b64 @6.3 @sm70
atom$sem$ascope$shared.add.noftz.f16 : =T, [X], T @6.3 @sm70
atom$sem$ascope{.global}.add.noftz$hint.f16x2 : =T, [X], T, {L2::cache_hint}b64 @6.2 @sm60
atom$sem$ascope$shared.add.noftz.f16x2 : =T, [X], T @6.2 @sm60
red$rsem$ascope{.global}(.and|.or|.xor)$hint(.b32|.b64) : [X], T, {L2::cache_hint}b64
red$rsem$ascope$shared(.and|.or|.xor)(.b32|.b64) : [X], T
red$rsem$ascope{.global}(.add|.inc|.dec|.min|.max)$hint.u32 : [X], T, {L2::cache_hint}b64
red$rsem$ascope$shared(.add|.inc|.dec|.min|.max).u32 : [X], T
red$rsem$ascope{.global}(.add|.min|.max)$hint(.s32|.u64) : [X], T, {L2::cache_hint}b64
red$rsem$ascope$shared(.add|.min|.max)(.s32|.u64) : [X], T
red$rsem$ascope{.global}(.min|.max)$hint.s64 : [X], T, {L2::cache_hint}b64
red$rsem$ascope$shared(.min|.max).s64 : [X], T
red$rsem$ascope{.global}.add$hint(.f32|.f64) : [X], T, {L2::cache_hint}b64
red$rsem$ascope$shared.add(.f32|.f64) : [X], T
red$rsem$ascope{.global}.add.noftz$hint.f16 : [X], T, {L2::cache_hint}b64 @6.3 @sm70
red$rsem$ascope$shared.add.noftz.f16 : [X], T @6.3 @sm70
red$rsem$ascope{.global}.add.noftz$hint.f16x2 : [X], T, {L2::cache_hint}b64 @6.2 @sm60
red$rsem$ascope$shared.add.noftz.f16x2 : [X], T @6.2 @sm60

# The threads of a warp together.
vote(.all|.any|.uni).pred : =T, !pred @<sm70
vote.ballot.b32 : =T, !pred @<sm70
vote.sync(.all|.any|.uni).pred : =T, !pred, u32
vote.sync.ballot.b32 : =T, !pred, u32
match.any.sync(.b32|.b64) : =b32, T, u32 @sm70
match.all.sync(.b32|.b64) : =b32[|=_pred], T, u32 @sm70
activemask.b32 : =T @6.2
redux.sync(.add|.min|.max)(.u32|.s32) : =T, T, u32 @7.0 @sm80
redux.sync(.and|.or|.xor).b32 : =T, T, u32 @7.0 @sm80

# The stack, and the rest.
stacksave(.u32|.u64) : =T @7.3
stackrestore(.u32|.u64) : %T @7.3
alloca(.u32|.u64) : =T, T, #^23? @7.3
brkpt
nanosleep.u32 : T @6.3 @sm70
pmevent : #4
pmevent.mask : #16
trap
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
    /// An integer known now.
    Imm(Imm),
}

/// What an operand that is an integer known now may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Imm {
    /// One of so many bits, unsigned: `lop3`'s table is 8.
    Bits(u32),
    /// This one, the only one PTX has for now: `applypriority`'s 128 bytes.
    Exactly(i64),
    /// A power of two, to 2 to this power: `alloca`'s alignment.
    Pow2(u32),
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

/// An operand of a form.
#[derive(Clone, Debug)]
pub struct Spec {
    /// What it is.
    pub role: Role,
    /// Its type, for a register or a value.
    pub ty: Option<TyRef>,
    /// How many.
    pub count: Count,
    /// Whether it must be a variable, not a value known now.
    pub reg_only: bool,
    /// Whether `_` may stand for it.
    pub sink: bool,
    /// Whether it may be read inverted, `!p`.
    pub not: bool,
    /// Whether it may be a special register, `tid.x`.
    pub sreg: bool,
    /// What a value known now is below, if anything: 16 for a barrier.
    pub below: Option<i64>,
    /// What a value known now is a multiple of, and more than 0: a barrier's threads are
    /// warps.
    pub step: Option<i64>,
    /// Whether a value known now is a fraction, more than 0 and at most 1.
    pub fraction: bool,
    /// The word the name has when it is there: the policy of `.L2::cache_hint`.
    pub when: Option<&'static str>,
    /// Whether it may be left out, at the end.
    pub optional: bool,
    /// The second of two written at once, `a|b`, and whether it may be left out.
    pub second: Option<(Box<Spec>, bool)>,
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

    /// Where its addresses point.
    pub fn space(&self) -> Space {
        for w in self.words() {
            match w.split("::").next().unwrap_or_default() {
                "global" => return Space::Global,
                "shared" => return Space::Shared,
                "local" => return Space::Local,
                "const" => return Space::Const,
                "param" => return Space::Param,
                _ => {}
            }
        }
        Space::Generic
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

/// The forms the instruction `name` is of, its words as PTX writes them, in the order of
/// [`TABLE`]; or what is wrong with its name.
pub fn find(name: &str) -> Result<Vec<Bound>, String> {
    let mut words = name.split('.');
    let op = words.next().unwrap_or_default();
    let words: Vec<&str> = words.collect();
    let of: Vec<&'static Form> = forms().iter().filter(|f| f.op == op).collect();
    if of.is_empty() {
        return Err(match NEWER.iter().find(|(o, _)| *o == op) {
            Some((_, gpu)) => format!("PTX's `{op}` is for {gpu} and newer, and kgen writes PTX for sm_61 to sm_86"),
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
    Bound { form, choice, types, needs }
}

/// Why a name of `op` is of no form, from the form it comes nearest.
fn unknown(name: &str, op: &str, words: &[&str], of: &[&'static Form]) -> String {
    let known = |w: &str| of.iter().any(|f| f.elems.iter().any(|e| e.alts.iter().any(|a| a.words.contains(&w))));
    if let Some(w) = words.iter().find(|w| !known(w)) {
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
        let mut best: Option<(usize, &Alt)> = None;
        for a in &e.alts {
            let k = a.words.iter().filter(|w| left.contains(w)).count();
            if best.is_none_or(|(b, _)| k > b) {
                best = Some((k, a));
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
            Check::Cvt => cvt(b, name)?,
        }
    }
    Ok(needs)
}

/// PTX's rules for `cvt` between integers and floats: it rounds where it loses precision,
/// to an integer from a float, and only there or where it rounds a float to an integral
/// value; it flushes subnormals only of an `f32`; and it saturates only where the result can
/// fall outside what its type holds.
fn cvt(b: &Bound, name: &str) -> Result<(), String> {
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
    Ok(())
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
            out.push(Sreg { name, ty, needs: Needs { ptx, sm, below: 0 }, fewest });
        };
        for v in ["tid", "ntid", "ctaid", "nctaid"] {
            for c in ["x", "y", "z"] {
                add(format!("{v}.{c}"), Ty::U32, 0, 0, 16);
            }
        }
        for n in
            ["laneid", "warpid", "nwarpid", "smid", "nsmid", "clock", "clock_hi", "globaltimer_lo", "globaltimer_hi"]
        {
            add(n.into(), Ty::U32, 0, 0, 32);
        }
        for n in ["lanemask_eq", "lanemask_le", "lanemask_lt", "lanemask_ge", "lanemask_gt"] {
            add(n.into(), Ty::U32, 0, 0, 32);
        }
        add("gridid".into(), Ty::U64, 0, 0, 16);
        add("clock64".into(), Ty::U64, 0, 0, 64);
        add("globaltimer".into(), Ty::U64, 0, 0, 64);
        for i in 0..8 {
            add(format!("pm{i}"), Ty::U32, 0, 0, 32);
            add(format!("pm{i}_64"), Ty::U64, 0, 0, 64);
        }
        for i in 0..32 {
            add(format!("envreg{i}"), Ty::B32, 0, 0, 32);
        }
        for n in ["begin", "end", "cap", "0", "1"] {
            add(format!("reserved_smem_offset_{n}"), Ty::B32, 76, 80, 32);
        }
        add("total_smem_size".into(), Ty::U32, 0, 0, 32);
        add("dynamic_smem_size".into(), Ty::U32, 0, 0, 32);
        add("current_graph_exec".into(), Ty::U64, 80, 0, 64);
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
const HOPPER: &str = "it needs sm_90, and kgen writes PTX for sm_61 to sm_86";
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
    let mut forms = vec![];
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(def) = line.strip_prefix('$') {
            let (name, elem) = def.split_once(" = ").unwrap_or_else(|| panic!("`{line}` defines `$name = ..`"));
            let (op, elems) = name_of(elem, &macros);
            assert!(op.is_empty(), "`{line}` is elements, not a name");
            macros.push((name, elems));
            continue;
        }
        forms.push(form(line, &macros));
    }
    forms
}

fn form(line: &'static str, macros: &[(&'static str, Vec<Elem>)]) -> Form {
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
    let (mut operands, mut needs, mut wide, mut checks) = (vec![], Needs::default(), false, vec![]);
    for token in rest.split_whitespace() {
        if let Some(r) = token.strip_prefix('@') {
            needs = needs.and(req(r, line));
        } else if let Some(c) = token.strip_prefix('+') {
            match c {
                "wide" => wide = true,
                "vec" => checks.push(Check::Vec),
                "cvt" => checks.push(Check::Cvt),
                "b128sys" => checks.push(Check::B128Sys),
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
        let end = match r.strip_prefix('<').unwrap_or(r).strip_prefix("sm") {
            Some(_) => digits(r.find("sm").expect("it starts so") + 2),
            None => match digits(0) {
                major if r[major..].starts_with('.') => digits(major + 1),
                major => major,
            },
        };
        needs = needs.and(req(&r[..end], line));
        rest = &r[end..];
    }
    (Alt { words, needs }, rest)
}

/// A requirement after `@`: `7.8`, `sm80` or `<sm70`.
fn req(r: &str, line: &str) -> Needs {
    let num = |s: &str| s.parse::<u32>().unwrap_or_else(|_| panic!("`{line}`: `@{r}` is no requirement"));
    if let Some(sm) = r.strip_prefix("<sm") {
        Needs { below: num(sm), ..Needs::default() }
    } else if let Some(sm) = r.strip_prefix("sm") {
        Needs { sm: num(sm), ..Needs::default() }
    } else {
        let (major, minor) = r.split_once('.').unwrap_or_else(|| panic!("`{line}`: `@{r}` is no version"));
        Needs { ptx: num(major) * 10 + num(minor), ..Needs::default() }
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
    } else if let Some((a, b)) = s.split_once('|').filter(|_| !s.starts_with('[')) {
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

fn one(s: &str, line: &str) -> Spec {
    let mut spec = Spec {
        role: Role::Src,
        ty: None,
        count: Count::One,
        reg_only: false,
        sink: false,
        not: false,
        sreg: false,
        below: None,
        step: None,
        fraction: false,
        when: None,
        optional: false,
        second: None,
    };
    let num = |n: &str| n.parse::<i64>().unwrap_or_else(|_| panic!("`{line}`: `{s}` has no number `{n}`"));
    match s {
        "[R]" => spec.role = Role::Addr(Access::Read),
        "[W]" => spec.role = Role::Addr(Access::Write),
        "[X]" => spec.role = Role::Addr(Access::ReadWrite),
        "lab" => spec.role = Role::Label,
        "#" => spec.role = Role::Imm(Imm::Bits(32)),
        _ if s.starts_with('#') => {
            let r = &s[1..];
            spec.role = Role::Imm(if let Some(n) = r.strip_prefix('=') {
                Imm::Exactly(num(n))
            } else if let Some(n) = r.strip_prefix('^') {
                Imm::Pow2(num(n) as u32)
            } else {
                Imm::Bits(num(r) as u32)
            });
        }
        _ => {
            let mut t = s;
            while let Some(c) = t.chars().next().filter(|c| "=%_!$".contains(*c)) {
                match c {
                    '=' => spec.role = Role::Dst,
                    '%' => spec.reg_only = true,
                    '_' => spec.sink = true,
                    '!' => spec.not = true,
                    _ => spec.sreg = true,
                }
                t = &t[1..];
            }
            if let Some((ty, n)) = t.split_once('<') {
                spec.below = Some(num(n));
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
