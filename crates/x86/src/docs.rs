//! The docs of x86's instructions, which an editor shows where a kernel names one. A doc says what the instruction
//! does, in a sentence of kernite's own; then each form it takes as a kernel writes it, `vaddps(=ymm, ymm, ymm/m256)`,
//! with each operand it writes marked; the CPU feature it needs; and where Intel's SDM says the rest. [`of`] writes the
//! doc of each instruction of [`TABLE`](crate::forms::TABLE); a test holds each to a doc, and the forms of each doc to
//! the table's.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use kernite_core::Access::{self, Ignored, ReadWrite, Write};
use kernite_core::Width::W128;

use crate::forms::{self, Feature, Form, Kind, Spec};

/// What each instruction does: a line for each set of names, then ` : `, then the sentence, which goes on in the lines
/// after it that start with a space. A family that one sentence has the pattern of, `cmov$cc` with its conditions, the
/// comparisons with their predicates, and the fused multiplies and adds, has none here: [`sentence`] works each out.
const SENTENCES: &str = r#"
mov : Copies the second operand to the first.
movzx : Copies the second operand, of 8 or 16 bits, to the first, and fills the rest of the first with zeros.
movsx : Copies the second operand, of 8 or 16 bits, to the first, and fills the rest of the first with copies of its
    sign bit.
movsxd : Copies the second operand, of 32 bits, to the first, of 64, and fills the rest of the first with copies of
    its sign bit.
movbe : Copies the second operand to the first with its bytes in the opposite order.
movnti : Stores a gpr in memory without keeping the cache line; `sfence` orders the store with the others.
lea : Writes to the first operand the address that the second names, and reads no memory.
xchg : Swaps the two operands. With memory it is atomic, as `lock.xchg` is.
xadd : Adds the second operand to the first, and writes the old first to the second.
bswap : Reverses the order of the bytes of its operand.
jmp : Jumps to the label.
add : Adds the second operand to the first, and sets the flags.
adc : Adds the second operand and the carry flag to the first, and sets the flags.
sub : Subtracts the second operand from the first, and sets the flags.
sbb : Subtracts the second operand and the carry flag from the first, and sets the flags.
and : Writes the bits that both operands have set to the first, and sets the flags.
or : Writes the bits that either operand has set to the first, and sets the flags.
xor : Writes the bits that one operand has set, and not both, to the first, and sets the flags.
cmp : Subtracts the second operand from the first and sets the flags, but writes nothing.
test : Ands the two operands and sets the flags, but writes nothing.
inc : Adds 1 to its operand, and sets the flags but the carry flag.
dec : Subtracts 1 from its operand, and sets the flags but the carry flag.
neg : Negates its operand as a signed number, and sets the flags.
not : Flips each bit of its operand; it sets no flag.
imul : Multiplies two signed numbers, the first operand and the second, or the second and an integer, and writes the
    low half of the product to the first. It sets the carry and overflow flags if the product does not fit.
shl sal : Shifts its operand left, by 1 or by a count, and fills it with zeros; the carry flag is the last bit
    shifted out.
shr : Shifts its operand right, by 1 or by a count, and fills it with zeros; the carry flag is the last bit shifted
    out.
sar : Shifts its operand right, by 1 or by a count, and fills it with copies of its sign bit; the carry flag is the
    last bit shifted out.
rol : Rotates its operand left, by 1 or by a count: the bits that go out at the left come in at the right.
ror : Rotates its operand right, by 1 or by a count: the bits that go out at the right come in at the left.
rcl : Rotates its operand left, by 1 or by a count, through the carry flag, which is one more bit of it.
rcr : Rotates its operand right, by 1 or by a count, through the carry flag, which is one more bit of it.
shld : Shifts the first operand left by a count, and fills it from the high bits of the second.
shrd : Shifts the first operand right by a count, and fills it from the low bits of the second.
bt : Copies the bit of the first operand at the index of the second to the carry flag. Memory and an index in a
    register reach past the operand, into the bytes before it or after it: the index is signed.
btc : Copies the bit at the index to the carry flag as `bt` does, and flips it.
btr : Copies the bit at the index to the carry flag as `bt` does, and clears it.
bts : Copies the bit at the index to the carry flag as `bt` does, and sets it.
bsf : Writes the index of the lowest set bit of the second operand to the first. Of 0 it sets the zero flag and
    leaves the first as it was, as AMD documents and Intel's CPUs do; Intel's manual leaves it undefined. Of 0 into
    an r32, AMD's CPUs keep the rest of its gpr and Intel's clear it.
bsr : Writes the index of the highest set bit of the second operand to the first. Of 0 it sets the zero flag and
    leaves the first as it was, as AMD documents and Intel's CPUs do; Intel's manual leaves it undefined. Of 0 into
    an r32, AMD's CPUs keep the rest of its gpr and Intel's clear it.
popcnt : Counts the set bits of the second operand into the first.
lzcnt : Counts the zero bits of the second operand above its highest set bit into the first: its size, of 0.
tzcnt : Counts the zero bits of the second operand below its lowest set bit into the first: its size, of 0.
crc32 : Folds the second operand into the CRC-32C checksum in the first: the polynomial 0x11EDC6F41, with its bits
    in reflected order, as iSCSI has it.
andn : Writes to the first the bits of the third operand that the second does not have set.
bextr : Writes to the first the bits of the second operand that the third says: its bits 0 to 7 are the first bit
    and 8 to 15 the count.
blsi : Writes to the first only the lowest set bit of the second operand.
blsmsk : Writes to the first a mask of the bits of the second operand up to its lowest set bit, and that bit.
blsr : Writes to the first the second operand with its lowest set bit cleared.
bzhi : Writes to the first the second operand with its bits cleared from the index that the third says.
pdep : Writes to the first the low bits of the second operand, one at each set bit of the third, the mask, and zeros
    elsewhere.
pext : Writes to the low bits of the first the bits of the second operand at the set bits of the third, the mask,
    and zeros above.
rorx : Rotates the second operand right by a count into the first, and sets no flag.
sarx : Shifts the second operand right by the third into the first, and fills it with copies of its sign bit; it
    sets no flag.
shlx : Shifts the second operand left by the third into the first, and fills it with zeros; it sets no flag.
shrx : Shifts the second operand right by the third into the first, and fills it with zeros; it sets no flag.
nop : Does nothing. It may name a gpr or memory, which it does not read: that only makes the instruction longer.
pause : Hints that the CPU waits in a loop for another to write memory, so that it spends less power and slows the
    other thread of its core less.
lfence : Waits until each earlier instruction is done, and starts no later instruction before.
mfence : Orders each earlier load and store before each later one.
sfence : Orders each earlier store, the non-temporal ones too, before each later one.
clc : Clears the carry flag.
stc : Sets the carry flag.
cmc : Flips the carry flag.
int3 : Raises a breakpoint trap, which a debugger takes and which stops a program that has none.
ud2 : Raises an invalid-instruction fault, which stops the program.
clflush : Writes the cache line at the address back to memory, and drops it from every cache.
prefetcht0 : Hints that the cache line at the address is read soon, and asks for it in each level of the cache.
prefetcht1 : Hints that the cache line at the address is read soon, and asks for it in the second level of the cache
    and those outside it.
prefetcht2 : Hints that the cache line at the address is read soon, and asks for it in the third level of the cache
    and those outside it.
prefetchnta : Hints that the cache line at the address is read soon and once, and asks for it so as to push the
    least out of the cache.
lock.add : Adds the second operand to memory in one atomic step, so that no access of another CPU comes between its
    read and its write.
lock.adc : Adds the second operand and the carry flag to memory in one atomic step, so that no access of another CPU
    comes between its read and its write.
lock.sub : Subtracts the second operand from memory in one atomic step, so that no access of another CPU comes
    between its read and its write.
lock.sbb : Subtracts the second operand and the carry flag from memory in one atomic step, so that no access of
    another CPU comes between its read and its write.
lock.and : Ands the second operand into memory in one atomic step, so that no access of another CPU comes between
    its read and its write.
lock.or : Ors the second operand into memory in one atomic step, so that no access of another CPU comes between its
    read and its write.
lock.xor : Xors the second operand into memory in one atomic step, so that no access of another CPU comes between
    its read and its write.
lock.inc : Adds 1 to memory in one atomic step, so that no access of another CPU comes between its read and its
    write.
lock.dec : Subtracts 1 from memory in one atomic step, so that no access of another CPU comes between its read and
    its write.
lock.neg : Negates memory as a signed number in one atomic step, so that no access of another CPU comes between its
    read and its write.
lock.not : Flips each bit of memory in one atomic step, so that no access of another CPU comes between its read and
    its write.
lock.btc : Copies the bit of memory at the index to the carry flag, and flips it, in one atomic step, so that no
    access of another CPU comes between its read and its write.
lock.btr : Copies the bit of memory at the index to the carry flag, and clears it, in one atomic step, so that no
    access of another CPU comes between its read and its write.
lock.bts : Copies the bit of memory at the index to the carry flag, and sets it, in one atomic step, so that no
    access of another CPU comes between its read and its write.
lock.xadd : Adds the second operand to memory, and writes the old memory to the second, in one atomic step, so that
    no access of another CPU comes between its read and its write.
lock.xchg : Swaps the two operands in one atomic step, so that no access of another CPU comes between its read and
    its write.
vaddps : Adds the floats of 32 bits of the last two operands, element by element, into the first.
vaddpd : Adds the floats of 64 bits of the last two operands, element by element, into the first.
vaddss : Adds the low float of 32 bits of the last two operands into the low float of 32 bits of the first, and
    copies the rest of the first from the second operand.
vaddsd : Adds the low float of 64 bits of the last two operands into the low float of 64 bits of the first, and
    copies the rest of the first from the second operand.
vsubps : Subtracts the floats of 32 bits of the third operand from those of the second, element by element, into the
    first.
vsubpd : Subtracts the floats of 64 bits of the third operand from those of the second, element by element, into the
    first.
vsubss : Subtracts the low float of 32 bits of the third operand from that of the second into the low float of 32
    bits of the first, and copies the rest of the first from the second operand.
vsubsd : Subtracts the low float of 64 bits of the third operand from that of the second into the low float of 64
    bits of the first, and copies the rest of the first from the second operand.
vmulps : Multiplies the floats of 32 bits of the last two operands, element by element, into the first.
vmulpd : Multiplies the floats of 64 bits of the last two operands, element by element, into the first.
vmulss : Multiplies the low float of 32 bits of the last two operands into the low float of 32 bits of the first,
    and copies the rest of the first from the second operand.
vmulsd : Multiplies the low float of 64 bits of the last two operands into the low float of 64 bits of the first,
    and copies the rest of the first from the second operand.
vdivps : Divides the floats of 32 bits of the second operand by those of the third, element by element, into the
    first.
vdivpd : Divides the floats of 64 bits of the second operand by those of the third, element by element, into the
    first.
vdivss : Divides the low float of 32 bits of the second operand by that of the third into the low float of 32 bits
    of the first, and copies the rest of the first from the second operand.
vdivsd : Divides the low float of 64 bits of the second operand by that of the third into the low float of 64 bits
    of the first, and copies the rest of the first from the second operand.
vmaxps : Writes the greater of each two floats of 32 bits of the last two operands to the first; where they are
    equal, or one is a NaN, the third operand's.
vmaxpd : Writes the greater of each two floats of 64 bits of the last two operands to the first; where they are
    equal, or one is a NaN, the third operand's.
vmaxss : Writes the greater of the low float of 32 bits of the last two operands to the low float of 32 bits of the
    first, or the third operand's where they are equal or one is a NaN, and copies the rest of the first from the
    second operand.
vmaxsd : Writes the greater of the low float of 64 bits of the last two operands to the low float of 64 bits of the
    first, or the third operand's where they are equal or one is a NaN, and copies the rest of the first from the
    second operand.
vminps : Writes the lesser of each two floats of 32 bits of the last two operands to the first; where they are
    equal, or one is a NaN, the third operand's.
vminpd : Writes the lesser of each two floats of 64 bits of the last two operands to the first; where they are
    equal, or one is a NaN, the third operand's.
vminss : Writes the lesser of the low float of 32 bits of the last two operands to the low float of 32 bits of the
    first, or the third operand's where they are equal or one is a NaN, and copies the rest of the first from the
    second operand.
vminsd : Writes the lesser of the low float of 64 bits of the last two operands to the low float of 64 bits of the
    first, or the third operand's where they are equal or one is a NaN, and copies the rest of the first from the
    second operand.
vsqrtps : Writes the square roots of the floats of 32 bits of the second operand to the first.
vsqrtpd : Writes the square roots of the floats of 64 bits of the second operand to the first.
vsqrtss : Writes the square root of the low float of 32 bits of the third operand to the low float of 32 bits of the
    first, and copies the rest of the first from the second operand.
vsqrtsd : Writes the square root of the low float of 64 bits of the third operand to the low float of 64 bits of the
    first, and copies the rest of the first from the second operand.
vrcpps : Writes approximations of the reciprocals of the floats of 32 bits of the second operand to the first, to a
    relative error of at most 1.5 / 4096.
vrcpss : Writes an approximation of the reciprocal of the low float of 32 bits of the third operand to the low float
    of 32 bits of the first, to a relative error of at most 1.5 / 4096, and copies the rest of the first from the
    second operand.
vrsqrtps : Writes approximations of the reciprocals of the square roots of the floats of 32 bits of the second
    operand to the first, to a relative error of at most 1.5 / 4096.
vrsqrtss : Writes an approximation of the reciprocal of the square root of the low float of 32 bits of the third
    operand to the low float of 32 bits of the first, to a relative error of at most 1.5 / 4096, and copies the rest
    of the first from the second operand.
vandps : Ands the bits of the last two operands into the first, as bits of floats of 32 bits.
vandnps : Writes to the first the bits of the third operand that the second does not have set.
vorps : Ors the bits of the last two operands into the first, as bits of floats of 32 bits.
vxorps : Xors the bits of the last two operands into the first, as bits of floats of 32 bits.
vaddsubps : Subtracts the floats of 32 bits of the third operand from those of the second at the even places, and
    adds them at the odd, into the first.
vhaddps : Adds each pair of adjacent floats of 32 bits: the pairs of the second operand fill the low half of each
    128-bit lane of the first, and those of the third operand the high half.
vhsubps : Subtracts the second of each pair of adjacent floats of 32 bits from the first: the pairs of the second
    operand fill the low half of each 128-bit lane of the first, and those of the third operand the high half.
vunpcklps : Interleaves the low floats of 32 bits of each 128-bit lane of the second and third operands into the
    first.
vunpckhps : Interleaves the high floats of 32 bits of each 128-bit lane of the second and third operands into the
    first.
vcmpps : Compares the floats of 32 bits of the second and third operands as the predicate in the fourth says, 0 to
    31, and writes all ones to each element of the first where it holds, else zeros.
vroundps : Rounds each of the floats of 32 bits of the second operand to a whole number, as the third says: its low
    two bits choose the way (0 to the nearest, 1 down, 2 up, 3 toward zero), its bit 2 the way of MXCSR instead, and
    its bit 3 keeps the precision exception from being raised.
vblendps : Copies each of the floats of 32 bits of the first from the third operand where its bit of the fourth is
    set, else from the second.
vblendvps : Copies each of the floats of 32 bits of the first from the third operand where the sign bit of the same
    element of the fourth is set, else from the second.
vtestps : Sets the zero flag if no sign bit of the floats of 32 bits is set in both operands, and the carry flag if
    each sign bit set in the second operand is set in the first.
vandpd : Ands the bits of the last two operands into the first, as bits of floats of 64 bits.
vandnpd : Writes to the first the bits of the third operand that the second does not have set.
vorpd : Ors the bits of the last two operands into the first, as bits of floats of 64 bits.
vxorpd : Xors the bits of the last two operands into the first, as bits of floats of 64 bits.
vaddsubpd : Subtracts the floats of 64 bits of the third operand from those of the second at the even places, and
    adds them at the odd, into the first.
vhaddpd : Adds each pair of adjacent floats of 64 bits: the pairs of the second operand fill the low half of each
    128-bit lane of the first, and those of the third operand the high half.
vhsubpd : Subtracts the second of each pair of adjacent floats of 64 bits from the first: the pairs of the second
    operand fill the low half of each 128-bit lane of the first, and those of the third operand the high half.
vunpcklpd : Interleaves the low floats of 64 bits of each 128-bit lane of the second and third operands into the
    first.
vunpckhpd : Interleaves the high floats of 64 bits of each 128-bit lane of the second and third operands into the
    first.
vcmppd : Compares the floats of 64 bits of the second and third operands as the predicate in the fourth says, 0 to
    31, and writes all ones to each element of the first where it holds, else zeros.
vroundpd : Rounds each of the floats of 64 bits of the second operand to a whole number, as the third says: its low
    two bits choose the way (0 to the nearest, 1 down, 2 up, 3 toward zero), its bit 2 the way of MXCSR instead, and
    its bit 3 keeps the precision exception from being raised.
vblendpd : Copies each of the floats of 64 bits of the first from the third operand where its bit of the fourth is
    set, else from the second.
vblendvpd : Copies each of the floats of 64 bits of the first from the third operand where the sign bit of the same
    element of the fourth is set, else from the second.
vtestpd : Sets the zero flag if no sign bit of the floats of 64 bits is set in both operands, and the carry flag if
    each sign bit set in the second operand is set in the first.
vcmpss : Compares the low float of 32 bits of the second and third operands as the predicate in the fourth says, 0
    to 31, and writes all ones to the low float of 32 bits of the first if it holds, else zeros, and copies the rest
    of the first from the second operand.
vroundss : Rounds the low float of 32 bits of the third operand to a whole number, as the fourth says: its low two
    bits choose the way (0 to the nearest, 1 down, 2 up, 3 toward zero), its bit 2 the way of MXCSR instead, and its
    bit 3 keeps the precision exception from being raised. It writes it to the low float of 32 bits of the first,
    and copies the rest of the first from the second operand.
vcmpsd : Compares the low float of 64 bits of the second and third operands as the predicate in the fourth says, 0
    to 31, and writes all ones to the low float of 64 bits of the first if it holds, else zeros, and copies the rest
    of the first from the second operand.
vroundsd : Rounds the low float of 64 bits of the third operand to a whole number, as the fourth says: its low two
    bits choose the way (0 to the nearest, 1 down, 2 up, 3 toward zero), its bit 2 the way of MXCSR instead, and its
    bit 3 keeps the precision exception from being raised. It writes it to the low float of 64 bits of the first,
    and copies the rest of the first from the second operand.
vcomiss : Compares the low floats of 32 bits of its operands and sets the zero, parity and carry flags: all three if
    one is a NaN, the carry flag alone if the first is less, the zero flag alone if they are equal, and none if the
    first is greater. It clears the overflow, sign and adjust flags. A NaN of either kind raises the
    invalid-operation exception.
vucomiss : Compares the low floats of 32 bits of its operands and sets the zero, parity and carry flags: all three
    if one is a NaN, the carry flag alone if the first is less, the zero flag alone if they are equal, and none if
    the first is greater. It clears the overflow, sign and adjust flags. Only a signaling NaN raises the
    invalid-operation exception.
vcomisd : Compares the low floats of 64 bits of its operands and sets the zero, parity and carry flags: all three if
    one is a NaN, the carry flag alone if the first is less, the zero flag alone if they are equal, and none if the
    first is greater. It clears the overflow, sign and adjust flags. A NaN of either kind raises the
    invalid-operation exception.
vucomisd : Compares the low floats of 64 bits of its operands and sets the zero, parity and carry flags: all three
    if one is a NaN, the carry flag alone if the first is less, the zero flag alone if they are equal, and none if
    the first is greater. It clears the overflow, sign and adjust flags. Only a signaling NaN raises the
    invalid-operation exception.
vdpps : Multiplies the floats of 32 bits of the second and third operands, those of each 128-bit lane that bits 4 to
    7 of the fourth choose, adds the products, and writes the sum to the elements of the lane of the first that bits
    0 to 3 choose, and zeros to the others.
vdppd : Multiplies the floats of 64 bits of the second and third operands that bits 4 and 5 of the fourth choose,
    adds the products, and writes the sum to the elements of the first that bits 0 and 1 choose, and zeros to the
    others.
vshufps : Shuffles floats of 32 bits within each 128-bit lane: the low two of the first come from the second
    operand, the high two from the third, each by two bits of the fourth.
vshufpd : Shuffles floats of 64 bits within each 128-bit lane: the low one of the first comes from the second
    operand, the high one from the third, each by one bit of the fourth.
vpermilps : Writes each float of 32 bits of the first from one in the same 128-bit lane of the second operand, which
    a control chooses: two bits of the same element of the third operand, or of the integer in the last place.
vpermilpd : Writes each float of 64 bits of the first from one in the same 128-bit lane of the second operand, which
    a control chooses: a bit of the same element of the third operand, or of the integer in the last place.
vperm2f128 : Writes each 128-bit half of the first from a half of the second or third operand, as the integer in the
    last place says, or zeros it.
vinsertps : Copies the second operand to the first, with one float of 32 bits of it replaced by one of the third
    operand: bits 6 and 7 of the integer in the last place say which of the third operand's, if it is a register,
    bits 4 and 5 which it replaces, and each of bits 0 to 3 that is set zeros the element of its place after.
vextractps : Writes the float of 32 bits, as bits, at the index in the last place of the second operand to a gpr or
    memory.
vmovhlps : Copies the high 64 bits of the third operand to the low 64 bits of the first, and the high 64 bits of the
    second to its high.
vmovlhps : Copies the low 64 bits of the third operand to the high 64 bits of the first, and the low 64 bits of the
    second to its low.
vmovhps vmovhpd : Copies 64 bits between memory and the high half of an xmm: a load writes it to the high half of
    the first operand, whose low half is the second operand's, and a store writes the high half.
vmovlps vmovlpd : Copies 64 bits between memory and the low half of an xmm: a load writes it to the low half of the
    first operand, whose high half is the second operand's, and a store writes the low half.
vmovddup : Copies the low float of 64 bits of each 128-bit lane of the second operand into both halves of the lane
    of the first.
vmovshdup : Copies each odd float of 32 bits of the second operand into the even place before it too.
vmovsldup : Copies each even float of 32 bits of the second operand into the odd place after it too.
vptest : Sets the zero flag if no bit is set in both operands, and the carry flag if each bit set in the second
    operand is set in the first.
vpaddb : Adds the integers of 8 bits of the last two operands, element by element, into the first.
vpsubb : Subtracts the integers of 8 bits of the third operand from those of the second, element by element, into
    the first.
vpaddw : Adds the integers of 16 bits of the last two operands, element by element, into the first.
vpsubw : Subtracts the integers of 16 bits of the third operand from those of the second, element by element, into
    the first.
vpaddd : Adds the integers of 32 bits of the last two operands, element by element, into the first.
vpsubd : Subtracts the integers of 32 bits of the third operand from those of the second, element by element, into
    the first.
vpaddq : Adds the integers of 64 bits of the last two operands, element by element, into the first.
vpsubq : Subtracts the integers of 64 bits of the third operand from those of the second, element by element, into
    the first.
vpaddsb : Adds the signed integers of 8 bits of the last two operands, element by element, into the first; a sum
    past the range of 8 bits stays at its end.
vpaddusb : Adds the unsigned integers of 8 bits of the last two operands, element by element, into the first; a sum
    past the range of 8 bits stays at its end.
vpsubsb : Subtracts the signed integers of 8 bits of the third operand from those of the second, element by element,
    into the first; a difference past the range of 8 bits stays at its end.
vpsubusb : Subtracts the unsigned integers of 8 bits of the third operand from those of the second, element by
    element, into the first; a difference below 0 stays at 0.
vpaddsw : Adds the signed integers of 16 bits of the last two operands, element by element, into the first; a sum
    past the range of 16 bits stays at its end.
vpaddusw : Adds the unsigned integers of 16 bits of the last two operands, element by element, into the first; a sum
    past the range of 16 bits stays at its end.
vpsubsw : Subtracts the signed integers of 16 bits of the third operand from those of the second, element by
    element, into the first; a difference past the range of 16 bits stays at its end.
vpsubusw : Subtracts the unsigned integers of 16 bits of the third operand from those of the second, element by
    element, into the first; a difference below 0 stays at 0.
vpmullw : Multiplies the integers of 16 bits of the last two operands, element by element, into the first, which
    keeps the low 16 bits of each product.
vpmulld : Multiplies the integers of 32 bits of the last two operands, element by element, into the first, which
    keeps the low 32 bits of each product.
vpmulhw : Multiplies the signed integers of 16 bits of the last two operands, element by element, into the first,
    which keeps the high 16 bits of each product.
vpmulhuw : Multiplies the unsigned integers of 16 bits of the last two operands, element by element, into the first,
    which keeps the high 16 bits of each product.
vpmulhrsw : Multiplies the signed integers of 16 bits of the last two operands, element by element, and writes each
    product to the first shifted right by 15, rounded to the nearest.
vpmuludq : Multiplies the low unsigned integer of 32 bits of each 64 bits of the last two operands into an unsigned
    integer of 64 bits in the first.
vpmuldq : Multiplies the low signed integer of 32 bits of each 64 bits of the last two operands into a signed
    integer of 64 bits in the first.
vpmaddwd : Multiplies the signed integers of 16 bits of the last two operands, element by element, and adds each
    pair of adjacent products into an integer of 32 bits of the first.
vpmaddubsw : Multiplies the unsigned bytes of the second operand by the signed bytes of the third, and adds each
    pair of adjacent products into a signed integer of 16 bits of the first, which stays at the end of its range.
vpavgb vpavgw : Averages the unsigned integers of the last two operands, element by element, into the first: the sum
    plus 1, shifted right by 1.
vpsadbw : Adds the differences, without their signs, of the unsigned bytes of the last two operands, each 8 bytes to
    an integer of 64 bits of the first.
vpsignb vpsignw vpsignd : Negates each integer of the second operand where the same element of the third is
    negative, zeros it where that is 0, and keeps it where that is positive, into the first.
vpmaxsb : Writes the greater of each two signed integers of 8 bits of the last two operands to the first.
vpminsb : Writes the lesser of each two signed integers of 8 bits of the last two operands to the first.
vpmaxsw : Writes the greater of each two signed integers of 16 bits of the last two operands to the first.
vpminsw : Writes the lesser of each two signed integers of 16 bits of the last two operands to the first.
vpmaxsd : Writes the greater of each two signed integers of 32 bits of the last two operands to the first.
vpminsd : Writes the lesser of each two signed integers of 32 bits of the last two operands to the first.
vpmaxub : Writes the greater of each two unsigned integers of 8 bits of the last two operands to the first.
vpminub : Writes the lesser of each two unsigned integers of 8 bits of the last two operands to the first.
vpmaxuw : Writes the greater of each two unsigned integers of 16 bits of the last two operands to the first.
vpminuw : Writes the lesser of each two unsigned integers of 16 bits of the last two operands to the first.
vpmaxud : Writes the greater of each two unsigned integers of 32 bits of the last two operands to the first.
vpminud : Writes the lesser of each two unsigned integers of 32 bits of the last two operands to the first.
vpand : Ands the bits of the last two operands into the first.
vpandn : Writes to the first the bits of the third operand that the second does not have set.
vpor : Ors the bits of the last two operands into the first.
vpxor : Xors the bits of the last two operands into the first.
vpcmpeqb : Compares the integers of 8 bits of the last two operands, element by element, and writes all ones to each
    element of the first where they are equal, else zeros.
vpcmpgtb : Compares the signed integers of 8 bits of the last two operands, element by element, and writes all ones
    to each element of the first where the second operand's is greater, else zeros.
vpcmpeqw : Compares the integers of 16 bits of the last two operands, element by element, and writes all ones to
    each element of the first where they are equal, else zeros.
vpcmpgtw : Compares the signed integers of 16 bits of the last two operands, element by element, and writes all ones
    to each element of the first where the second operand's is greater, else zeros.
vpcmpeqd : Compares the integers of 32 bits of the last two operands, element by element, and writes all ones to
    each element of the first where they are equal, else zeros.
vpcmpgtd : Compares the signed integers of 32 bits of the last two operands, element by element, and writes all ones
    to each element of the first where the second operand's is greater, else zeros.
vpcmpeqq : Compares the integers of 64 bits of the last two operands, element by element, and writes all ones to
    each element of the first where they are equal, else zeros.
vpcmpgtq : Compares the signed integers of 64 bits of the last two operands, element by element, and writes all ones
    to each element of the first where the second operand's is greater, else zeros.
vphaddw vphaddd : Adds each pair of adjacent integers: the pairs of the second operand fill the low half of each
    128-bit lane of the first, and those of the third operand the high half.
vphaddsw : Adds each pair of adjacent signed integers of 16 bits: the pairs of the second operand fill the low half
    of each 128-bit lane of the first, and those of the third operand the high half; a sum past the range stays at
    its end.
vphsubw vphsubd : Subtracts the second of each pair of adjacent integers from the first: the pairs of the second
    operand fill the low half of each 128-bit lane of the first, and those of the third operand the high half.
vphsubsw : Subtracts the second of each pair of adjacent signed integers of 16 bits from the first: the pairs of the
    second operand fill the low half of each 128-bit lane of the first, and those of the third operand the high
    half; a difference past the range stays at its end.
vpacksswb : Narrows the signed integers of 16 bits of the second and third operands to signed integers of 8 bits,
    which stay at the end of the range, into the first: those of the second operand first, in each 128-bit lane.
vpackssdw : Narrows the signed integers of 32 bits of the second and third operands to signed integers of 16 bits,
    which stay at the end of the range, into the first: those of the second operand first, in each 128-bit lane.
vpackuswb : Narrows the signed integers of 16 bits of the second and third operands to unsigned integers of 8 bits,
    which stay at the end of the range, into the first: those of the second operand first, in each 128-bit lane.
vpackusdw : Narrows the signed integers of 32 bits of the second and third operands to unsigned integers of 16 bits,
    which stay at the end of the range, into the first: those of the second operand first, in each 128-bit lane.
vpunpcklbw : Interleaves the low integers of 8 bits of each 128-bit lane of the second and third operands into the
    first.
vpunpckhbw : Interleaves the high integers of 8 bits of each 128-bit lane of the second and third operands into the
    first.
vpunpcklwd : Interleaves the low integers of 16 bits of each 128-bit lane of the second and third operands into the
    first.
vpunpckhwd : Interleaves the high integers of 16 bits of each 128-bit lane of the second and third operands into the
    first.
vpunpckldq : Interleaves the low integers of 32 bits of each 128-bit lane of the second and third operands into the
    first.
vpunpckhdq : Interleaves the high integers of 32 bits of each 128-bit lane of the second and third operands into the
    first.
vpunpcklqdq : Interleaves the low integers of 64 bits of each 128-bit lane of the second and third operands into the
    first.
vpunpckhqdq : Interleaves the high integers of 64 bits of each 128-bit lane of the second and third operands into
    the first.
vpshufb : Shuffles the bytes of the second operand within each 128-bit lane by the bytes of the third, which are
    indices into the lane, into the first; a byte of the third operand with its top bit set gives a zero.
vpabsb : Writes the absolute values of the signed integers of 8 bits of the second operand to the first.
vpabsw : Writes the absolute values of the signed integers of 16 bits of the second operand to the first.
vpabsd : Writes the absolute values of the signed integers of 32 bits of the second operand to the first.
vphminposuw : Finds the least of the eight unsigned integers of 16 bits of the second operand, and writes it to the
    low 16 bits of the first and its index to the 3 bits after, and zeros to the rest.
vpshufd : Shuffles the integers of 32 bits of the second operand within each 128-bit lane, by the integer in the
    last place: two bits for each element of the first.
vpshufhw : Shuffles the four high integers of 16 bits of each 128-bit lane of the second operand, by the integer in
    the last place: two bits for each; the low four are copied.
vpshuflw : Shuffles the four low integers of 16 bits of each 128-bit lane of the second operand, by the integer in
    the last place: two bits for each; the high four are copied.
vpalignr : Joins each 128-bit lane of the second operand above that of the third, shifts the 256 bits right by the
    integer in the last place, in bytes, and writes the low 128 bits to the lane of the first.
vpblendw : Copies each integer of 16 bits of the first from the third operand where its bit of the integer in the
    last place is set, else from the second; each 128-bit lane has the same bits.
vpblendvb : Copies each byte of the first from the third operand where the top bit of the same byte of the fourth is
    set, else from the second.
vpblendd : Copies each integer of 32 bits of the first from the third operand where its bit of the integer in the
    last place is set, else from the second.
vmpsadbw : Adds the differences, without their signs, of the unsigned bytes of blocks of four of the second operand
    and of the third, at the offsets that the integer in the last place says, into integers of 16 bits of the first.
vpsllw : Shifts each integer of 16 bits of the second operand left, by the count in the low 64 bits of the third or
    by the integer in the last place, into the first, and fills it with zeros; a count past 15 gives 0.
vpsrlw : Shifts each integer of 16 bits of the second operand right, by the count in the low 64 bits of the third or
    by the integer in the last place, into the first, and fills it with zeros; a count past 15 gives 0.
vpslld : Shifts each integer of 32 bits of the second operand left, by the count in the low 64 bits of the third or
    by the integer in the last place, into the first, and fills it with zeros; a count past 31 gives 0.
vpsrld : Shifts each integer of 32 bits of the second operand right, by the count in the low 64 bits of the third or
    by the integer in the last place, into the first, and fills it with zeros; a count past 31 gives 0.
vpsllq : Shifts each integer of 64 bits of the second operand left, by the count in the low 64 bits of the third or
    by the integer in the last place, into the first, and fills it with zeros; a count past 63 gives 0.
vpsrlq : Shifts each integer of 64 bits of the second operand right, by the count in the low 64 bits of the third or
    by the integer in the last place, into the first, and fills it with zeros; a count past 63 gives 0.
vpsraw : Shifts each integer of 16 bits of the second operand right, by the count in the low 64 bits of the third or
    by the integer in the last place, into the first, and fills it with copies of its sign bit; a count past 15 is
    15.
vpsrad : Shifts each integer of 32 bits of the second operand right, by the count in the low 64 bits of the third or
    by the integer in the last place, into the first, and fills it with copies of its sign bit; a count past 31 is
    31.
vpslldq : Shifts each 128-bit lane of the second operand left by the integer in the last place, in bytes, into the
    first, and fills it with zeros.
vpsrldq : Shifts each 128-bit lane of the second operand right by the integer in the last place, in bytes, into the
    first, and fills it with zeros.
vpsllvd : Shifts each integer of 32 bits of the second operand left by the same element of the third, into the
    first, and fills it with zeros; a count past 31 gives 0.
vpsrlvd : Shifts each integer of 32 bits of the second operand right by the same element of the third, into the
    first, and fills it with zeros; a count past 31 gives 0.
vpsllvq : Shifts each integer of 64 bits of the second operand left by the same element of the third, into the
    first, and fills it with zeros; a count past 63 gives 0.
vpsrlvq : Shifts each integer of 64 bits of the second operand right by the same element of the third, into the
    first, and fills it with zeros; a count past 63 gives 0.
vpsravd : Shifts each integer of 32 bits of the second operand right by the same element of the third, into the
    first, and fills it with copies of its sign bit; a count past 31 is 31.
vperm2i128 : Writes each 128-bit half of the first from a half of the second or third operand, as the integer in the
    last place says, or zeros it.
vpermd : Writes each integer of 32 bits of the first from the one of the third operand that the same element of the
    second, an index, says.
vpermps : Writes each float of 32 bits of the first from the one of the third operand that the same element of the
    second, an index, says.
vpermq : Writes each integer of 64 bits of the first from one of the four of the second operand, as the integer in
    the last place says, two bits for each.
vpermpd : Writes each float of 64 bits of the first from one of the four of the second operand, as the integer in
    the last place says, two bits for each.
vpmovsxbw : Widens the low signed integers of 8 bits of the second operand to integers of 16 bits, with copies of
    the sign bit, into the first.
vpmovzxbw : Widens the low unsigned integers of 8 bits of the second operand to integers of 16 bits, with zeros,
    into the first.
vpmovsxbd : Widens the low signed integers of 8 bits of the second operand to integers of 32 bits, with copies of
    the sign bit, into the first.
vpmovzxbd : Widens the low unsigned integers of 8 bits of the second operand to integers of 32 bits, with zeros,
    into the first.
vpmovsxbq : Widens the low signed integers of 8 bits of the second operand to integers of 64 bits, with copies of
    the sign bit, into the first.
vpmovzxbq : Widens the low unsigned integers of 8 bits of the second operand to integers of 64 bits, with zeros,
    into the first.
vpmovsxwd : Widens the low signed integers of 16 bits of the second operand to integers of 32 bits, with copies of
    the sign bit, into the first.
vpmovzxwd : Widens the low unsigned integers of 16 bits of the second operand to integers of 32 bits, with zeros,
    into the first.
vpmovsxwq : Widens the low signed integers of 16 bits of the second operand to integers of 64 bits, with copies of
    the sign bit, into the first.
vpmovzxwq : Widens the low unsigned integers of 16 bits of the second operand to integers of 64 bits, with zeros,
    into the first.
vpmovsxdq : Widens the low signed integers of 32 bits of the second operand to integers of 64 bits, with copies of
    the sign bit, into the first.
vpmovzxdq : Widens the low unsigned integers of 32 bits of the second operand to integers of 64 bits, with zeros,
    into the first.
vcvtdq2ps : Converts the integers of 32 bits of the second operand to floats of 32 bits in the first, rounded as
    MXCSR says.
vcvtps2dq : Converts the floats of 32 bits of the second operand to integers of 32 bits in the first, rounded as
    MXCSR says; one that does not fit is 0x80000000.
vcvttps2dq : Converts the floats of 32 bits of the second operand to integers of 32 bits in the first, toward zero;
    one that does not fit is 0x80000000.
vcvtdq2pd : Converts the low integers of 32 bits of the second operand to floats of 64 bits in the first.
vcvtps2pd : Converts the low floats of 32 bits of the second operand to floats of 64 bits in the first.
vcvtpd2ps : Converts the floats of 64 bits of the second operand to floats of 32 bits in the low part of the first,
    rounded as MXCSR says, and zeros the rest.
vcvtpd2dq : Converts the floats of 64 bits of the second operand to integers of 32 bits in the low part of the
    first, rounded as MXCSR says, and zeros the rest; one that does not fit is 0x80000000.
vcvttpd2dq : Converts the floats of 64 bits of the second operand to integers of 32 bits in the low part of the
    first, toward zero, and zeros the rest; one that does not fit is 0x80000000.
vcvtss2sd : Converts the low float of 32 bits of the third operand to a float of 64 bits in the low part of the
    first, and copies the rest of the first from the second operand.
vcvtsd2ss : Converts the low float of 64 bits of the third operand to a float of 32 bits in the low part of the
    first, rounded as MXCSR says, and copies the rest of the first from the second operand.
vcvtsi2ss : Converts the integer of 32 or 64 bits of the third operand to a float of 32 bits in the low part of the
    first, rounded as MXCSR says, and copies the rest of the first from the second operand.
vcvtsi2sd : Converts the integer of 32 or 64 bits of the third operand to a float of 64 bits in the low part of the
    first, rounded as MXCSR says, and copies the rest of the first from the second operand.
vcvtss2si : Converts the low float of 32 bits of the second operand to an integer of 32 or 64 bits in the first,
    rounded as MXCSR says; one that does not fit is the least number.
vcvttss2si : Converts the low float of 32 bits of the second operand to an integer of 32 or 64 bits in the first,
    toward zero; one that does not fit is the least number.
vcvtsd2si : Converts the low float of 64 bits of the second operand to an integer of 32 or 64 bits in the first,
    rounded as MXCSR says; one that does not fit is the least number.
vcvttsd2si : Converts the low float of 64 bits of the second operand to an integer of 32 or 64 bits in the first,
    toward zero; one that does not fit is the least number.
vcvtph2ps : Converts the floats of 16 bits of the second operand to floats of 32 bits in the first.
vcvtps2ph : Converts the floats of 32 bits of the second operand to floats of 16 bits in the first, rounded as the
    integer in the last place says: its low two bits choose the way (0 to the nearest, 1 down, 2 up, 3 toward zero),
    its bit 2 the way of MXCSR instead.
vmovups vmovupd vmovdqu : Copies 128 or 256 bits between vectors, or between a vector and memory, at any address.
vmovaps vmovapd vmovdqa : Copies 128 or 256 bits between vectors, or between a vector and memory, at an address that
    is a multiple of their size: another address faults.
vmovntps vmovntpd vmovntdq : Stores 128 or 256 bits in memory at an address that is a multiple of their size,
    without keeping the cache line; `sfence` orders the store with the others.
vmovntdqa : Loads 128 or 256 bits from memory at an address that is a multiple of their size, and reads memory of
    the write-combining kind past the cache.
vlddqu : Loads 128 or 256 bits from memory at any address, as a load of `vmovdqu` does; Intel's manual says it may be
    faster where the load crosses a cache line.
vmovss : Copies a float of 32 bits: memory to an xmm, which zeros its rest; an xmm to memory; or an xmm to an xmm,
    where the first takes the low float of the third operand and the rest of the second.
vmovsd : Copies a float of 64 bits: memory to an xmm, which zeros its rest; an xmm to memory; or an xmm to an xmm,
    where the first takes the low float of the third operand and the rest of the second.
vmovd : Copies 32 bits between the low part of an xmm and a gpr or memory; into an xmm it zeros the rest.
vmovq : Copies 64 bits between the low part of an xmm and an xmm, a gpr or memory; into an xmm it zeros the rest.
vmovmskps vmovmskpd : Writes the sign bit of each float of the second operand to the low bits of a gpr, one for each
    element, and zeros to the rest.
vpmovmskb : Writes the top bit of each byte of the second operand to the low bits of a gpr, one for each byte, and
    zeros to the rest.
vmaskmovps vmaskmovpd : Copies the elements of memory that the sign bits of a mask choose, the second operand: a
    load writes zeros to the elements not chosen, a store leaves their memory as it was, and neither touches their
    memory, so an element not chosen cannot fault.
vpmaskmovd vpmaskmovq : Copies the integers of memory that the sign bits of a mask choose, the second operand: a
    load writes zeros to the elements not chosen, a store leaves their memory as it was, and neither touches their
    memory, so an element not chosen cannot fault.
vbroadcastss : Copies a float of 32 bits, from memory or the low part of an xmm, to each element of the first.
vbroadcastsd : Copies a float of 64 bits, from memory or the low part of an xmm, to each element of the first.
vbroadcastf128 vbroadcasti128 : Copies 128 bits of memory to both halves of the first.
vpbroadcastb : Copies an integer of 8 bits, from memory or the low part of an xmm, to each element of the first.
vpbroadcastw : Copies an integer of 16 bits, from memory or the low part of an xmm, to each element of the first.
vpbroadcastd : Copies an integer of 32 bits, from memory or the low part of an xmm, to each element of the first.
vpbroadcastq : Copies an integer of 64 bits, from memory or the low part of an xmm, to each element of the first.
vinsertf128 vinserti128 : Writes the second operand to the first with the 128-bit half that the integer in the last
    place chooses, 0 or 1, replaced by the third.
vextractf128 vextracti128 : Writes the 128-bit half of the second operand that the integer in the last place
    chooses, 0 or 1, to an xmm or memory.
vpextrb : Writes the integer of 8 bits of the second operand at the index in the last place to a gpr or memory; a
    gpr's rest is zeros.
vpinsrb : Copies the second operand to the first with the integer of 8 bits at the index in the last place replaced
    by the third, a gpr's or memory's.
vpextrw : Writes the integer of 16 bits of the second operand at the index in the last place to a gpr or memory; a
    gpr's rest is zeros.
vpinsrw : Copies the second operand to the first with the integer of 16 bits at the index in the last place replaced
    by the third, a gpr's or memory's.
vpextrd : Writes the integer of 32 bits of the second operand at the index in the last place to a gpr or memory; a
    gpr's rest is zeros.
vpinsrd : Copies the second operand to the first with the integer of 32 bits at the index in the last place replaced
    by the third, a gpr's or memory's.
vpextrq : Writes the integer of 64 bits of the second operand at the index in the last place to a gpr or memory; a
    gpr's rest is zeros.
vpinsrq : Copies the second operand to the first with the integer of 64 bits at the index in the last place replaced
    by the third, a gpr's or memory's.
vgatherdps : Loads floats of 32 bits from memory at the base plus each index of the vector, integers of 32 bits,
    where the sign bit of the mask is set, into the first, and clears the mask; the other elements stay as they
    were.
vgatherqps : Loads floats of 32 bits from memory at the base plus each index of the vector, integers of 64 bits,
    where the sign bit of the mask is set, into the first, and clears the mask; the other elements stay as they
    were.
vgatherdpd : Loads floats of 64 bits from memory at the base plus each index of the vector, integers of 32 bits,
    where the sign bit of the mask is set, into the first, and clears the mask; the other elements stay as they
    were.
vgatherqpd : Loads floats of 64 bits from memory at the base plus each index of the vector, integers of 64 bits,
    where the sign bit of the mask is set, into the first, and clears the mask; the other elements stay as they
    were.
vpgatherdd : Loads integers of 32 bits from memory at the base plus each index of the vector, integers of 32 bits,
    where the sign bit of the mask is set, into the first, and clears the mask; the other elements stay as they
    were.
vpgatherqd : Loads integers of 32 bits from memory at the base plus each index of the vector, integers of 64 bits,
    where the sign bit of the mask is set, into the first, and clears the mask; the other elements stay as they
    were.
vpgatherdq : Loads integers of 64 bits from memory at the base plus each index of the vector, integers of 32 bits,
    where the sign bit of the mask is set, into the first, and clears the mask; the other elements stay as they
    were.
vpgatherqq : Loads integers of 64 bits from memory at the base plus each index of the vector, integers of 64 bits,
    where the sign bit of the mask is set, into the first, and clears the mask; the other elements stay as they
    were.
vldmxcsr : Loads MXCSR, the control and status of floats, from memory: the way they round, which exceptions they
    raise, and what they have raised.
vstmxcsr : Stores MXCSR, the control and status of floats, to memory.
"#;

/// The conditions of `cmov`, `set` and `j`, each with what Intel's SDM calls it, and the flags that make it hold.
static CONDS: [(&str, &str, &str); 30] = [
    ("a", "above", "the carry and zero flags are clear"),
    ("ae", "above or equal", "the carry flag is clear"),
    ("b", "below", "the carry flag is set"),
    ("be", "below or equal", "the carry flag or the zero flag is set"),
    ("c", "carry", "the carry flag is set"),
    ("e", "equal", "the zero flag is set"),
    ("g", "greater", "the zero flag is clear and the sign flag equals the overflow flag"),
    ("ge", "greater or equal", "the sign flag equals the overflow flag"),
    ("l", "less", "the sign flag differs from the overflow flag"),
    ("le", "less or equal", "the zero flag is set or the sign flag differs from the overflow flag"),
    ("na", "not above", "the carry flag or the zero flag is set"),
    ("nae", "not above or equal", "the carry flag is set"),
    ("nb", "not below", "the carry flag is clear"),
    ("nbe", "not below or equal", "the carry and zero flags are clear"),
    ("nc", "not carry", "the carry flag is clear"),
    ("ne", "not equal", "the zero flag is clear"),
    ("ng", "not greater", "the zero flag is set or the sign flag differs from the overflow flag"),
    ("nge", "not greater or equal", "the sign flag differs from the overflow flag"),
    ("nl", "not less", "the sign flag equals the overflow flag"),
    ("nle", "not less or equal", "the zero flag is clear and the sign flag equals the overflow flag"),
    ("no", "not overflow", "the overflow flag is clear"),
    ("np", "not parity", "the parity flag is clear"),
    ("ns", "not sign", "the sign flag is clear"),
    ("nz", "not zero", "the zero flag is clear"),
    ("o", "overflow", "the overflow flag is set"),
    ("p", "parity", "the parity flag is set"),
    ("pe", "parity even", "the parity flag is set"),
    ("po", "parity odd", "the parity flag is clear"),
    ("s", "sign", "the sign flag is set"),
    ("z", "zero", "the zero flag is set"),
];

/// The predicates of `vcmpps` and its kin, in the order of their numbers, 0 to 31: each as the table writes it in
/// `vcmp<predicate>ps`, and as Intel's SDM does.
static PREDS: [(&str, &str); 32] = [
    ("eq", "EQ_OQ"),
    ("lt", "LT_OS"),
    ("le", "LE_OS"),
    ("unord", "UNORD_Q"),
    ("neq", "NEQ_UQ"),
    ("nlt", "NLT_US"),
    ("nle", "NLE_US"),
    ("ord", "ORD_Q"),
    ("eq_uq", "EQ_UQ"),
    ("nge", "NGE_US"),
    ("ngt", "NGT_US"),
    ("false", "FALSE_OQ"),
    ("neq_oq", "NEQ_OQ"),
    ("ge", "GE_OS"),
    ("gt", "GT_OS"),
    ("true", "TRUE_UQ"),
    ("eq_os", "EQ_OS"),
    ("lt_oq", "LT_OQ"),
    ("le_oq", "LE_OQ"),
    ("unord_s", "UNORD_S"),
    ("neq_us", "NEQ_US"),
    ("nlt_uq", "NLT_UQ"),
    ("nle_uq", "NLE_UQ"),
    ("ord_s", "ORD_S"),
    ("eq_us", "EQ_US"),
    ("nge_uq", "NGE_UQ"),
    ("ngt_uq", "NGT_UQ"),
    ("false_os", "FALSE_OS"),
    ("neq_os", "NEQ_OS"),
    ("ge_oq", "GE_OQ"),
    ("gt_oq", "GT_OQ"),
    ("true_us", "TRUE_US"),
];

/// When each relation of a predicate holds of the elements of the second operand and the third.
static RELATIONS: [(&str, &str); 12] = [
    ("EQ", "they are equal"),
    ("LT", "the second operand's is less than the third's"),
    ("LE", "the second operand's is less than or equal to the third's"),
    ("UNORD", "either is a NaN"),
    ("NEQ", "they are not equal"),
    ("NLT", "the second operand's is not less than the third's"),
    ("NLE", "the second operand's is not less than or equal to the third's"),
    ("ORD", "neither is a NaN"),
    ("NGE", "the second operand's is not greater than or equal to the third's"),
    ("NGT", "the second operand's is not greater than the third's"),
    ("GE", "the second operand's is greater than or equal to the third's"),
    ("GT", "the second operand's is greater than the third's"),
];

/// The sentences by name, parsed from [`SENTENCES`] once. Text that does not parse has none: a test of the crate parses
/// it, and says which line is wrong.
fn sentences() -> &'static BTreeMap<&'static str, String> {
    static SENTENCES_BY_NAME: OnceLock<BTreeMap<&'static str, String>> = OnceLock::new();
    SENTENCES_BY_NAME.get_or_init(|| parse(SENTENCES).unwrap_or_default())
}

/// The sentences of `text`, by name, or the first line that is wrong, and why.
fn parse(text: &'static str) -> Result<BTreeMap<&'static str, String>, String> {
    let mut entries: Vec<(&'static str, String)> = vec![];
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        match (line.starts_with(' '), entries.last_mut()) {
            (true, Some((_, sentence))) => {
                sentence.push(' ');
                sentence.push_str(line.trim());
            }
            (true, None) => return Err(format!("`{}` goes on from no sentence", line.trim())),
            (false, _) => {
                let (names, sentence) = line.split_once(" : ").ok_or_else(|| format!("`{line}` has no ` : `"))?;
                entries.push((names, sentence.to_string()));
            }
        }
    }
    let mut out = BTreeMap::new();
    for (names, sentence) in entries {
        for name in names.split_whitespace() {
            if out.insert(name, sentence.clone()).is_some() {
                return Err(format!("`{name}` has two sentences"));
            }
        }
    }
    Ok(out)
}

/// The stem and the condition of `name`, if it is `cmov`, `set` or `j` with a condition.
fn condition(name: &str) -> Option<(&'static str, &'static (&'static str, &'static str, &'static str))> {
    ["cmov", "set", "j"].into_iter().find_map(|stem| {
        let cc = name.strip_prefix(stem)?;
        Some((stem, CONDS.iter().find(|(c, ..)| *c == cc)?))
    })
}

/// The predicate Intel names, and the suffix, of `name`, if it is `vcmp`, a predicate, and `ps`, `pd`, `ss` or `sd`.
fn comparison(name: &str) -> Option<(&'static str, &str)> {
    let rest = name.strip_prefix("vcmp")?;
    let (predicate, suffix) = rest.split_at_checked(rest.len().checked_sub(2)?)?;
    Some((PREDS.iter().find(|(p, _)| *p == predicate)?.1, suffix))
}

/// What the comparison with the predicate `intel` does to the elements `on` of its operands, written to `onto`.
fn compared(intel: &str, suffix: &str) -> Option<String> {
    let (relation, ordering) = intel.split_once('_')?;
    let size = if suffix.ends_with('s') { 32 } else { 64 };
    let (on, onto, rest) = match suffix.starts_with('p') {
        true => (format!("floats of {size} bits"), "each element of the first", ""),
        false => (
            format!("low floats of {size} bits"),
            "the low element of the first",
            ", and copies the rest of the first from the second operand",
        ),
    };
    let mut text = match relation {
        "FALSE" => format!("Writes zeros to {onto}, whatever the operands are{rest}."),
        "TRUE" => format!("Writes all ones to {onto}, whatever the operands are{rest}."),
        _ => {
            let when = RELATIONS.iter().find(|(r, _)| *r == relation)?.1;
            format!(
                "Compares the {on} of the second and third operands, and writes all ones to {onto} where {when}, \
                 else zeros{rest}."
            )
        }
    };
    if !["FALSE", "TRUE", "ORD", "UNORD"].contains(&relation) {
        let nan = if ordering.starts_with('U') { "true" } else { "false" };
        text += &format!(" If either is a NaN, the result is {nan}.");
    }
    text += match ordering.ends_with('S') {
        true => " A NaN of either kind raises the invalid-operation exception.",
        false => " Only a signaling NaN raises the invalid-operation exception.",
    };
    let number = PREDS.iter().position(|(_, i)| *i == intel)?;
    Some(format!("{text} It is `vcmp{suffix}` with the predicate {number}, {intel}."))
}

/// What the fused multiply and add `name` does, if it is one: `vfmadd231ps` writes `second * third + first`.
fn fused(name: &str) -> Option<String> {
    let rest = name.strip_prefix("vf")?;
    let (kind, rest) = ["maddsub", "msubadd", "nmadd", "nmsub", "madd", "msub"]
        .into_iter()
        .find_map(|k| Some((k, rest.strip_prefix(k)?)))?;
    let (order, suffix) = rest.split_at_checked(3)?;
    let (x, y, z) = match order {
        "132" => ("first", "third", "second"),
        "213" => ("second", "first", "third"),
        "231" => ("second", "third", "first"),
        _ => return None,
    };
    let formula = match kind {
        "madd" => format!("{x} * {y} + {z}"),
        "msub" => format!("{x} * {y} - {z}"),
        "nmadd" => format!("{z} - {x} * {y}"),
        "nmsub" => format!("-({x} * {y}) - {z}"),
        "maddsub" => format!("{x} * {y} - {z} at the even places and {x} * {y} + {z} at the odd"),
        _ => format!("{x} * {y} + {z} at the even places and {x} * {y} - {z} at the odd"),
    };
    let size = if suffix.ends_with('s') { 32 } else { 64 };
    Some(match suffix.starts_with('p') {
        true => format!(
            "Writes {formula} to the first operand, element by element, as floats of {size} bits, rounded once."
        ),
        false => format!(
            "Writes {formula} to the low float of {size} bits of the first operand, rounded once, and keeps the rest \
             of the first."
        ),
    })
}

/// What the instruction `name` does, in a sentence: the one [`SENTENCES`] has, or one worked out for the member of a
/// family.
fn sentence(name: &str) -> Option<String> {
    if let Some(sentence) = sentences().get(name) {
        return Some(sentence.clone());
    }
    if let Some((stem, &(_, meaning, flags))) = condition(name) {
        return Some(match stem {
            "cmov" => format!(
                "Copies the second operand to the first if {meaning}, that is, if {flags}. In any case it reads a \
                 memory operand, and clears the rest of the gpr of an r32."
            ),
            "set" => format!("Writes 1 to its operand if {meaning}, that is, if {flags}, and 0 if not."),
            _ => format!("Jumps to the label if {meaning}, that is, if {flags}."),
        });
    }
    match comparison(name) {
        Some((intel, suffix)) => compared(intel, suffix),
        None => fused(name),
    }
}

/// What to look up in volume 2 of Intel's SDM for `name`: its mnemonic in capitals, `VADDPS`, which the SDM gives in
/// the entry of the instruction, `ADDPS`; a family's mnemonic, `CMOVcc` or `VCMPPS`; and a `lock.` instruction's prefix
/// and its own, `LOCK, BTS`.
fn sdm(name: &str) -> String {
    if let Some((stem, _)) = condition(name) {
        return format!("{}cc", stem.to_uppercase());
    }
    if let Some((_, suffix)) = comparison(name) {
        return format!("VCMP{}", suffix.to_uppercase());
    }
    match name.strip_prefix("lock.") {
        Some(rest) => format!("LOCK, {}", rest.to_uppercase()),
        None => name.to_uppercase(),
    }
}

/// What a mark before an operand says.
fn mark(access: Access) -> &'static str {
    match access {
        Write => "=",
        ReadWrite => "+",
        Ignored => "~",
        Access::Read => "",
    }
}

/// What an operand that is not plain says: one a reader of the doc may not know the table's word for.
fn meaning(kind: Kind) -> Option<&'static str> {
    Some(match kind {
        Kind::Vec(w) if w < W128 => "`vN` is the low N bits of an xmm.",
        Kind::Aligned(_) => "`aN` is `mN` at an address that is a multiple of its size.",
        Kind::Addr => "`m` is an address, which the instruction reaches no memory at.",
        Kind::Vsib(..) => {
            "`vm32x` is elements of 32 bits at the indices in an xmm, and `vm64y` of 64 at those in a ymm."
        }
        Kind::Imm(_) => "`iN` is an integer for an operation of N bits, which at 64 is of 32, sign-extended.",
        Kind::Imm64 => "`imm64` is an integer of 64 bits.",
        Kind::Control(_) => "`#0xM` is an integer with no bit set but those of the mask 0xM.",
        _ => return None,
    })
}

/// The forms of the instruction `name`, as a kernel writes them, each with the CPU feature it needs: a line for each
/// form of each line of the table that `forms` are of, at each width the line gives; the operands a form lets
/// alternatives of are apart by `/`.
fn lines(name: &str, forms: &[Form]) -> Option<Vec<(String, Option<Feature>)>> {
    let mut lines = vec![];
    let mut seen: Vec<&str> = vec![];
    for form in forms {
        if seen.contains(&form.line) {
            continue;
        }
        seen.push(form.line);
        let (_, widths, feature, body) = forms::pieces(form.line).ok()?;
        for (n, own) in widths {
            for text in body.split(" ; ") {
                let all = forms::operands_of(text, n, form.line).ok()?;
                let operands: Vec<String> = (0..all.first()?.len())
                    .map(|at| {
                        let mut specs: Vec<Spec> = vec![];
                        for spec in all.iter().filter_map(|operands| operands.get(at)) {
                            if !specs.contains(spec) {
                                specs.push(*spec);
                            }
                        }
                        let kinds: Vec<String> = specs.iter().map(|s| s.kind.text()).collect();
                        format!("{}{}", specs.first().map_or("", |s| mark(s.access)), kinds.join("/"))
                    })
                    .collect();
                lines.push((format!("{name}({})", operands.join(", ")), own.or(feature)));
            }
        }
    }
    Some(lines)
}

/// The doc of the instruction `name`, as the language calls it, in Markdown: what it does, its forms with the operands
/// it writes marked, the CPU feature it needs, and what to look up in Intel's SDM. None for a name that is no
/// instruction kernite writes.
pub fn of(name: &str) -> Option<String> {
    let forms = forms::of(name)?;
    let lines = lines(name, forms)?;
    let mut features: Vec<Option<Feature>> = vec![];
    for &(_, feature) in &lines {
        if !features.contains(&feature) {
            features.push(feature);
        }
    }
    let specs: Vec<&Spec> = forms.iter().flat_map(|form| &form.operands).collect();
    let marks = [(Write, "=", "writes"), (ReadWrite, "+", "reads and writes"), (Ignored, "~", "reads nothing of")];
    let marked: Vec<String> = marks
        .into_iter()
        .filter(|(access, ..)| specs.iter().any(|s| s.access == *access))
        .enumerate()
        .map(|(i, (_, sign, verb))| {
            format!("`{sign}` {} {verb}", if i == 0 { "marks an operand it" } else { "one it" })
        })
        .collect();
    let mut notes: Vec<String> = vec![];
    if !marked.is_empty() {
        notes.push(format!("{}.", marked.join(", ")));
    }
    for note in specs.iter().filter_map(|s| meaning(s.kind)) {
        if !notes.iter().any(|n| n == note) {
            notes.push(note.to_string());
        }
    }
    let several = features.len() > 1;
    match features.as_slice() {
        [Some(feature)] => notes.push(format!("Needs the CPU feature `{}`.", feature.name())),
        [_, _, ..] => notes.push("`@feature` after a form is the CPU feature it needs.".into()),
        _ => {}
    }
    let shown: Vec<String> = lines
        .iter()
        .map(|(line, feature)| match feature {
            Some(feature) if several => format!("{line}  @{}", feature.name()),
            _ => line.clone(),
        })
        .collect();
    let mut doc = format!("{}\n\n```text\n{}\n```\n", sentence(name)?, shown.join("\n"));
    if !notes.is_empty() {
        doc += &format!("\n{}\n", notes.join(" "));
    }
    doc += &format!("\nIntel's SDM, volume 2: {}.", sdm(name));
    Some(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

    #[test]
    fn the_sentences_parse_and_each_is_of_a_name_kernite_writes() -> Result {
        let all = parse(SENTENCES)?;
        assert!(all.len() > 340, "{} sentences", all.len());
        for (name, sentence) in &all {
            assert!(forms::of(name).is_some(), "`{name}` is no instruction of the table");
            assert!(sentence.ends_with('.') && !sentence.contains("  "), "`{name}`: {sentence}");
        }
        assert_eq!(parse("a : b.\n    c.\n"), Ok(BTreeMap::from([("a", "b. c.".to_string())])));
        assert_eq!(parse("a b : c.\na : d.\n"), Err("`a` has two sentences".into()));
        assert_eq!(parse("a b.\n"), Err("`a b.` has no ` : `".into()));
        assert_eq!(parse("  a.\n"), Err("`a.` goes on from no sentence".into()));
        Ok(())
    }

    #[test]
    fn each_instruction_of_the_table_has_a_doc() -> Result {
        for name in forms::names() {
            let doc = of(name).ok_or_else(|| format!("`{name}` has no doc"))?;
            assert!(doc.contains(&format!("{name}(")) && doc.contains("Intel's SDM, volume 2: "), "`{name}`: {doc}");
        }
        for name in ["vaddpz", "ret", "mul", "label", ""] {
            assert_eq!(of(name), None, "`{name}`");
        }
        Ok(())
    }

    /// The forms in a doc are the table's: each line of its block, read as the table reads a form, gives the forms of
    /// the instruction in order, each with the feature the doc says it needs.
    #[test]
    fn the_forms_of_each_doc_are_the_forms_of_the_table() -> Result {
        for name in forms::names() {
            let doc = of(name).ok_or("of gave None")?;
            let block = doc.split("```text\n").nth(1).and_then(|rest| rest.split("\n```").next()).ok_or("no forms")?;
            let alone =
                Feature::ALL.into_iter().find(|f| doc.contains(&format!("Needs the CPU feature `{}`.", f.name())));
            let (mut read, mut needs) = (vec![], vec![]);
            for line in block.lines() {
                let (form, feature) = match line.split_once("  @") {
                    Some((form, feature)) => (form, Feature::ALL.into_iter().find(|f| f.name() == feature)),
                    None => (line, alone),
                };
                let operands =
                    form.strip_prefix(name).and_then(|l| l.strip_prefix('(')?.strip_suffix(')')).ok_or(line)?;
                let specs = forms::operands_of(if operands.is_empty() { "()" } else { operands }, 0, line)?;
                needs.extend(specs.iter().map(|_| feature));
                read.extend(specs);
            }
            let table = forms::of(name).ok_or("of gave None")?;
            assert_eq!(read, table.iter().map(|f| f.operands.clone()).collect::<Vec<_>>(), "`{name}`");
            assert_eq!(needs, table.iter().map(|f| f.feature).collect::<Vec<_>>(), "`{name}`");
        }
        Ok(())
    }

    /// A doc, whole: a sentence, the forms with a mark on each operand written, the feature, and Intel's name.
    #[test]
    fn a_doc_says_what_it_does_its_forms_its_feature_and_where_intel_says_the_rest() {
        let vaddps = "Adds the floats of 32 bits of the last two operands, element by element, into the first.\n\n\
                      ```text\nvaddps(=xmm, xmm, xmm/m128)\nvaddps(=ymm, ymm, ymm/m256)\n```\n\n\
                      `=` marks an operand it writes. Needs the CPU feature `avx`.\n\n\
                      Intel's SDM, volume 2: VADDPS.";
        assert_eq!(of("vaddps").as_deref(), Some(vaddps));
        // A form with no mark, a feature that is x86-64's own, and an instruction of two words.
        let cmp = of("cmp").unwrap_or_default();
        assert!(cmp.contains("cmp(r8/m8, r8)\ncmp(r8, m8)\ncmp(r8/m8, i8)\n"), "{cmp}");
        assert!(!cmp.contains("marks") && !cmp.contains("CPU feature"), "{cmp}");
        let bts = of("lock.bts").unwrap_or_default();
        assert!(bts.contains("lock.bts(+m16, r16)\nlock.bts(+m16, #0-15)\n"), "{bts}");
        assert!(bts.ends_with("Intel's SDM, volume 2: LOCK, BTS."), "{bts}");
        // Forms that need different features say so each, and what an operand is that is not plain is told.
        let vpaddb = of("vpaddb").unwrap_or_default();
        assert!(vpaddb.contains("vpaddb(=xmm, xmm, xmm/m128)  @avx\nvpaddb(=ymm, ymm, ymm/m256)  @avx2\n"), "{vpaddb}");
        assert!(vpaddb.contains("`@feature` after a form is the CPU feature it needs."), "{vpaddb}");
        let vgather = of("vgatherdps").unwrap_or_default();
        assert!(vgather.contains("`vm32x` is elements of 32 bits at the indices in an xmm"), "{vgather}");
        assert!(vgather.contains("`+` marks an operand it reads and writes."), "{vgather}");
        // The note of a control byte holds of any mask, as each form has its own.
        let round = of("vroundps").unwrap_or_default();
        assert!(round.contains("vroundps(=xmm, xmm/m128, #0x0f)"), "{round}");
        assert!(round.contains("`#0xM` is an integer with no bit set but those of the mask 0xM."), "{round}");
        // An operand it neither reads nor writes.
        let nop = of("nop").unwrap_or_default();
        assert!(nop.contains("nop()\nnop(~r16/r32/r64)\nnop(~m16/m32/m64)\n"), "{nop}");
        assert!(nop.contains("`~` marks an operand it reads nothing of."), "{nop}");
    }

    /// A family is documented as its members are, each with the words its own condition, predicate or order gives.
    #[test]
    fn a_family_of_names_has_the_sentence_its_member_gives() {
        let first = |name: &str| of(name).unwrap_or_default().lines().next().unwrap_or_default().to_string();
        assert_eq!(
            first("cmovnle"),
            "Copies the second operand to the first if not less or equal, that is, if the zero flag is clear and the \
             sign flag equals the overflow flag. In any case it reads a memory operand, and clears the rest of the gpr \
             of an r32."
        );
        assert_eq!(first("setz"), "Writes 1 to its operand if zero, that is, if the zero flag is set, and 0 if not.");
        assert_eq!(
            first("jl"),
            "Jumps to the label if less, that is, if the sign flag differs from the overflow flag."
        );
        assert_eq!(
            first("vcmpngt_uqps"),
            "Compares the floats of 32 bits of the second and third operands, and writes all ones to each element of \
             the first where the second operand's is not greater than the third's, else zeros. If either is a NaN, \
             the result is true. Only a signaling NaN raises the invalid-operation exception. It is `vcmpps` with the \
             predicate 26, NGT_UQ."
        );
        assert!(
            first("vcmpfalse_ossd")
                .starts_with("Writes zeros to the low element of the first, whatever the operands are, and copies")
        );
        assert_eq!(
            first("vfnmadd132pd"),
            "Writes second - first * third to the first operand, element by element, as floats of 64 bits, rounded \
             once."
        );
        assert!(
            first("vfmaddsub213ps")
                .contains("second * first - third at the even places and second * first + third at the odd")
        );
        assert!(first("vfmsub231ss").ends_with(
            "the low float of 32 bits of the first operand, rounded once, and keeps the rest of the first."
        ));
        assert!(of("vcmpltps").unwrap_or_default().ends_with("Intel's SDM, volume 2: VCMPPS."));
        assert!(of("cmovz").unwrap_or_default().ends_with("Intel's SDM, volume 2: CMOVcc."));
    }
}
