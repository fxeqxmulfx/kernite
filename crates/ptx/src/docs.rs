//! The docs of PTX's instructions, which an editor shows where a kernel names one. A doc says what the instruction
//! does, in a sentence of kernite's own; what each word of its name means, in a phrase; each form the name is of, as
//! a kernel writes it, `ld.global.f32(=f32, [u64])`, with each operand it writes marked; the version of PTX and the
//! GPU it needs; and where PTX's ISA says the rest. [`of`] writes the doc of the instruction a name is of; a test
//! holds each instruction of [`TABLE`](crate::forms::TABLE), and each word of it, to a doc.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use kernite_core::Access;

use crate::forms::{self, Bound, Imm, Role, Sel, Space, Spec};
use crate::{Kind, Needs, Ty};

/// What each instruction does: a line for each set of names, then ` : `, then the sentence, which goes on in the lines
/// after it that start with a space. A name is an instruction, `ld`, or the first words of one that has a sentence of
/// its own for them, `cp.async.commit_group`: [`sentence`] takes the longest of these that a name starts with. What a
/// word of a name says is [`PHRASES`]', and a sentence leaves it to them. Operands are counted from the first, which
/// is the one an instruction writes, but for those that write none.
const SENTENCES: &str = r#"
add : Adds the third operand to the second, and writes the sum to the first.
sub : Subtracts the third operand from the second, and writes the difference to the first.
mul : Multiplies the second operand by the third, and writes the product to the first. An integer product has twice
    the bits of an operand, and the word says which of them the first has.
mad : Multiplies the second operand by the third, adds the fourth, and writes the sum to the first. An integer product
    is cut as the word says before the add; a float one is not rounded before it.
clmad : Multiplies the second operand by the third without carries, as polynomials with coefficients mod 2, XORs the
    fourth operand into a half of the product of 128 bits that the word says, and writes that to the first.
mul24 : Multiplies the low 24 bits of the second operand by the low 24 bits of the third, which makes 48 bits, and
    writes 32 of them to the first.
mad24 : Multiplies the low 24 bits of the second operand by the low 24 bits of the third, which makes 48 bits, adds the
    fourth operand to 32 of them, and writes the sum to the first.
sad : Adds the distance between the second operand and the third to the fourth, and writes the sum to the first.
div : Divides the second operand by the third, and writes the quotient to the first. An integer quotient is rounded
    toward zero, and an integer division by 0 writes a value that PTX leaves unspecified.
rem : Divides the second operand by the third, and writes the remainder to the first: it has the sign of the dividend,
    the second operand. A division by 0 writes a value that PTX leaves unspecified.
abs : Writes the absolute value of the second operand to the first. The most negative integer is its own absolute
    value.
neg : Writes the second operand negated to the first. The most negative integer is its own negation.
min : Writes the lesser of the second operand and the third to the first. If one float is a NaN, it writes the other.
max : Writes the greater of the second operand and the third to the first. If one float is a NaN, it writes the other.
popc : Counts the bits of the second operand that are set, and writes the count to the first, a `u32`.
clz : Counts the zero bits of the second operand above its highest bit that is set, and writes the count to the first,
    a `u32`: the number of its bits, if none is set.
bfind : Writes to the first operand, a `u32`, the index of the highest bit of the second that is set, or, for a signed
    type, that differs from its sign bit; 0xffffffff if there is none.
fns : Finds a set bit of the second operand: from the bit that the third operand names, it counts as many set bits as
    the fourth operand says, up if it is more than 0 and down if it is less, and writes the index of the last it
    counted to the first, or 0xffffffff if there are too few. A fourth operand of 0 gives the start bit, if it is set.
brev : Reverses the order of the bits of the second operand, and writes them to the first.
bfe : Writes to the first operand a field of the second, moved to the bottom: it starts at the bit that the third
    operand names, and is as many bits as the fourth says. A signed type fills the rest of the first with copies of the
    top bit of the field, and an unsigned one with zeros.
bfi : Writes to the first operand the third, with a field of it replaced by the low bits of the second: the field starts
    at the bit that the fourth operand names, and is as many bits as the fifth says.
szext : Keeps the low bits of the second operand, as many as the third says, and fills the rest of the first with copies
    of the top bit of them, for a signed type, or with zeros, for an unsigned one. A width of 0 gives 0.
bmsk : Writes to the first operand a mask: the bits from the bit that the second operand names, as many as the third
    says, are set, but none past bit 31.
dp4a : Multiplies each of the four bytes of the second operand by the byte of the third at the same place, taking them
    as the first type in the name and the second, and writes the sum of the four products and the fourth operand to
    the first.
dp2a : Multiplies the low half of the second operand by the first of two bytes of the third, which the word picks, and
    the high half by the second, taking them as the first type in the name and the second, and writes the sum of the two
    products and the fourth operand to the first.
addc : Adds the third operand and the carry flag to the second, and writes the sum to the first.
subc : Subtracts the third operand and the carry flag from the second, and writes the difference to the first.
madc : Multiplies the second operand by the third, adds the fourth and the carry flag to the half of the product that
    the word says, and writes the sum to the first.
testp : Writes true to the first operand, a predicate, if the second, a float, is of the kind the word says, and false
    if not.
copysign : Writes the magnitude of the third operand with the sign of the second to the first.
fma : Multiplies the second operand by the third, adds the fourth, and writes the sum to the first, rounded once: the
    product is not rounded before it is added.
rcp : Writes the reciprocal of the second operand, 1 divided by it, to the first.
sqrt : Writes the square root of the second operand to the first.
rsqrt : Writes the reciprocal of the square root of the second operand to the first.
sin : Writes the sine of the second operand, an angle in radians, to the first.
cos : Writes the cosine of the second operand, an angle in radians, to the first.
lg2 : Writes the base 2 logarithm of the second operand to the first.
ex2 : Writes 2 raised to the power of the second operand to the first.
tanh : Writes the hyperbolic tangent of the second operand to the first.
setp : Compares the second operand with the third as the word says, and writes true or false to the first, a predicate.
    A comparison of floats with a NaN is false, but for `.nan` and the words that end in `u`. A fourth operand, a
    predicate, with a boolean word combines the result with it; and the first as `p | q` has `q` get the opposite of
    the comparison, combined in the same way.
set : Compares the second operand with the third as the word says, and writes the result to the first as a number of
    its type: all ones for an integer and 1.0 for a float, if it holds, and 0 if not. A fourth operand, a predicate,
    with a boolean word combines the result with it, as `setp` does.
selp : Writes the second operand to the first if the fourth, a predicate, is true, and the third if it is false.
slct : Writes the second operand to the first if the fourth is 0 or more, and the third if it is less or a NaN; -0.0
    counts as 0.
and : Writes the bits that both operands have set to the first.
or : Writes the bits that either operand has set to the first.
xor : Writes the bits that one operand has set, and not both, to the first.
not : Flips each bit of the second operand, and writes it to the first.
cnot : Writes 1 to the first operand if the second is 0, and 0 if not.
lop3 : Writes to the first operand any function of the bits of the second operand, the third and the fourth. For each
    bit, the table, the fifth operand, of 8 bits, gives the result at the index `4a + 2b + c` of the three bits a, b and
    c.
shf : Shifts the 64 bits that the third operand and the second make, the third above, by the fourth operand, and writes
    32 of them to the first.
shl : Shifts the second operand left by the third, and fills it with zeros. A count past the bits of the operand gives
    0.
shr : Shifts the second operand right by the third, and fills it with zeros, or, for a signed type, with copies of its
    sign bit. A count past the bits of the operand fills all of them.
vadd : Adds the second operand and the third, and writes the sum to the first. The types in the name are those of the
    first operand, the second and the third.
vsub : Subtracts the third operand from the second, and writes the difference to the first. The types in the name are
    those of the first operand, the second and the third.
vabsdiff : Writes the distance between the second operand and the third to the first. The types in the name are those of
    the first operand, the second and the third.
vmin : Writes the lesser of the second operand and the third to the first. The types in the name are those of the first
    operand, the second and the third.
vmax : Writes the greater of the second operand and the third to the first. The types in the name are those of the first
    operand, the second and the third.
vshl : Shifts the second operand left by the third, and writes it to the first. The types in the name are those of the
    first operand, the second and the third.
vshr : Shifts the second operand right by the third, and writes it to the first. The types in the name are those of the
    first operand, the second and the third.
vmad : Multiplies the second operand by the third, adds the fourth, and writes the sum to the first. The types in the
    name are those of the first operand, the second and the third.
vset : Compares the second operand with the third as the word says, and writes 1 to the first if it holds, and 0 if not.
    The types in the name are those of the second operand and the third.
vadd2 : Adds each half of the second operand to the half of the third at the same place, and writes the sums to the
    first. The types in the name are those of the first operand, the second and the third, and say if the halves are
    signed.
vsub2 : Subtracts each half of the third operand from the half of the second at the same place, and writes the
    differences to the first. The types in the name are those of the first operand, the second and the third, and say if
    the halves are signed.
vavrg2 : Writes the average of each pair of halves of the second operand and the third, rounded away from zero, to the
    first. The types in the name are those of the first operand, the second and the third, and say if the halves are
    signed.
vabsdiff2 : Writes the distance between each pair of halves of the second operand and the third to the first. The types
    in the name are those of the first operand, the second and the third, and say if the halves are signed.
vmin2 : Writes the lesser of each pair of halves of the second operand and the third to the first. The types in the name
    are those of the first operand, the second and the third, and say if the halves are signed.
vmax2 : Writes the greater of each pair of halves of the second operand and the third to the first. The types in the
    name are those of the first operand, the second and the third, and say if the halves are signed.
vset2 : Compares each pair of halves of the second operand and the third as the word says, and writes 1 to the half of
    the first where it holds, and 0 where not. The types in the name are those of the second operand and the third, and
    say if the halves are signed.
vadd4 : Adds each byte of the second operand to the byte of the third at the same place, and writes the sums to the
    first. The types in the name are those of the first operand, the second and the third, and say if the bytes are
    signed.
vsub4 : Subtracts each byte of the third operand from the byte of the second at the same place, and writes the
    differences to the first. The types in the name are those of the first operand, the second and the third, and say if
    the bytes are signed.
vavrg4 : Writes the average of each pair of bytes of the second operand and the third, rounded away from zero, to the
    first. The types in the name are those of the first operand, the second and the third, and say if the bytes are
    signed.
vabsdiff4 : Writes the distance between each pair of bytes of the second operand and the third to the first. The types
    in the name are those of the first operand, the second and the third, and say if the bytes are signed.
vmin4 : Writes the lesser of each pair of bytes of the second operand and the third to the first. The types in the name
    are those of the first operand, the second and the third, and say if the bytes are signed.
vmax4 : Writes the greater of each pair of bytes of the second operand and the third to the first. The types in the name
    are those of the first operand, the second and the third, and say if the bytes are signed.
vset4 : Compares each pair of bytes of the second operand and the third as the word says, and writes 1 to the byte of
    the first where it holds, and 0 where not. The types in the name are those of the second operand and the third, and
    say if the bytes are signed.
mov : Copies the second operand to the first. A tuple of registers is packed into a register, `mov.b64(d, (lo, hi))`,
    and a register is unpacked into a tuple, `mov.b64((lo, hi), s)`. The address of memory the kernel declares, or of
    a function, may be copied to an integer.
shfl : Reads the second operand in another lane of the warp, which the word and the third operand pick, and writes it
    to the first. A lane that is out of range, as the fourth operand says, reads its own value. A predicate beside the
    first says whether the lane was in range. The fourth operand packs the clamp, in its bits 0 to 4, and the segment
    mask, in its bits 8 to 12; the last is the mask of lanes that take part.
prmt : Picks four bytes of the eight that the second operand and the third make, the second the low four, and writes
    them to the first. Without a word, each nibble of the low 16 bits of the fourth operand picks one, and a nibble
    with its top bit set gives instead the top bit of the byte it picks, copied over a byte; a word picks them by n,
    the fourth operand, 0 to 3.
ld : Reads memory at the address in the second operand, and writes it to the first.
ldu : Reads memory at a global address that is the same in every thread of the warp, and writes it to the first.
st : Writes the second operand to memory at the address in the first.
prefetch : Hints that the line of memory at the address will be read soon, and brings it to the cache that the word
    names. It writes nothing, and changes no value.
prefetchu : Hints that the line of memory at the generic address will be read soon, and brings it to the uniform cache.
applypriority : Sets the eviction priority of the 128 bytes at the global address, in the L2 cache, to the one that the
    word names.
discard : Drops the 128 bytes at the global address from the L2 cache, without writing them back to memory. What
    the memory holds there afterwards is undefined.
createpolicy.range : Writes to the first operand a cache eviction policy of 64 bits for a range of global memory that
    starts at the address in the second operand: the first bytes of the range, as many as the third operand says, have
    the first priority in the name, and the rest of it, up to as many bytes as the fourth says, the second,
    `.L2::evict_unchanged` if the name has none.
createpolicy.fractional : Writes to the first operand a cache eviction policy of 64 bits that gives the first priority
    in the name to the fraction of the accesses that the second operand says, 1 if it is left out, and the second
    priority, `.L2::evict_unchanged` if the name has none, to the rest.
createpolicy.cvt : Writes to the first operand a cache eviction policy of 64 bits made from the access property of 64
    bits in the second.
isspacep : Writes true to the first operand, a predicate, if the generic address in the second is in the state space the
    word names, and false if not.
cvta : Converts the address in the second operand between the state space the word names and the generic space, and
    writes it to the first: from the state space to a generic address, or, with `.to`, from a generic address to it.
cvt : Converts the second operand to the type of the first, and writes it to the first: the two types end the name, the
    first operand's, then the second's. A form with two sources, for `.f16x2` and `.bf16x2`, converts both, and packs
    the results into the two halves of the first, the second operand's in the upper half.
cvt.pack : Converts the second operand and the third to the first type in the name, clamping each, and packs the two
    into the low bits of the first, the second operand's above the third's. A last operand, a `b32`, is shifted up past
    them, and stays in the first above them.
tex : Samples a texture, by its handle, at the coordinates in the brackets, and writes the texel to the first operands.
    The first type in the name is the texels', the second the coordinates'. The operands after the brackets are, in
    order, the level of detail or the gradients, if the name has them, an offset for each coordinate, from -8 to 7, and
    a depth to compare with, where the form shows them. A predicate beside the first operands says whether the texels
    are in memory that is mapped.
tld4 : Fetches the four texels that a bilinear sample would read, one component of each, and writes them to the first
    operands. The first type in the name is the texels', the second the coordinates'. A predicate beside them says
    whether the texels are in memory that is mapped.
txq : Writes to the first operand a property of the texture that the handle names, as the word says.
istypep : Writes true to the first operand, a predicate, if the handle in the second is of the type the word says, and
    false if not.
suld : Loads a sample from a surface, by its handle, at the coordinates in the brackets, and writes it to the first
    operands.
sust : Stores the second operands to a surface, by its handle, at the coordinates in the brackets.
sured : Combines the second operand into the sample of a surface, by its handle, at the coordinates in the brackets, as
    the word says, as one step that no other thread's access splits. It writes no register.
suq : Writes to the first operand a property of the surface that the handle names, as the word says.
bra : Jumps to the label.
brx : Jumps to the label, of those in the second operand, that the index in the first picks.
ret : Returns from the function to its caller; in a kernel it ends the thread.
exit : Ends the thread.
barrier bar : Arrives at a barrier, numbered 0 to 15, that the threads of a CTA share, and waits there or not, as the
    word says. The operand after the number of the barrier, if there is one, is the count of the threads that take part,
    a multiple of 32 to 1024; without it all the threads of the CTA take part.
bar.warp.sync : Waits until the threads of the warp that the mask of lanes names have arrived, and orders their memory
    accesses among them.
membar : Orders the memory accesses of the thread before it with those after it, as the threads of the scope the word
    names see them.
fence : Orders the memory accesses of the thread before it with those after it, as the threads of the scope see them.
membar.proxy fence.proxy : Orders the accesses of the thread, before it and after it, to one memory by different
    virtual addresses, which are aliases of each other.
atom : Reads the memory at the address in the second operand, changes it as the word says with the operands after the
    address, and writes the value it read to the first operand, all as one step that no other thread's access splits.
red : Changes the memory at the address in the first operand as the word says with the operands after the address, as
    one step that no other thread's access splits. It writes no register.
cp.async : Copies 4, 8 or 16 bytes, as the third operand says, from the global address in the second operand to the
    shared address in the first, while the thread goes on. A fourth operand, a count of bytes to read, fills the rest
    with zeros; a predicate there, when true, reads none, and fills all with zeros.
cp.async.commit_group : Ends the group of the copies that the thread has begun with `cp.async` since its last group,
    so that `cp.async.wait_group` can wait for it.
cp.async.wait_group : Waits until at most as many of the thread's groups of copies as the operand says, a number known
    now, are still going on.
cp.async.wait_all : Ends the group of the copies that the thread has begun, and waits until all its copies are done.
cp.async.mbarrier.arrive : Makes the mbarrier at the address count an arrival when the copies that the thread has begun
    are done.
mbarrier.init : Sets up the mbarrier at the address in the first operand, 8 bytes of shared memory, to expect as many
    arrivals as the second says, 1 to 1048575.
mbarrier.inval : Invalidates the mbarrier at the address, so that its memory may serve for something else.
mbarrier.arrive : Counts an arrival at the mbarrier at the address in the second operand, and writes the state of its
    current phase to the first, which `mbarrier.test_wait` takes, or drops it with `_`.
mbarrier.arrive_drop : Counts an arrival at the mbarrier at the address in the second operand, and takes the arrivals it
    counts out of those that the mbarrier expects in the phases after this one. It writes the state of the current phase
    to the first operand, or drops it with `_`.
mbarrier.test_wait : Tests whether a phase of the mbarrier at the address in the second operand is complete, and writes
    true or false to the first, a predicate; it does not wait. The last operand names the phase: it is the state
    that `mbarrier.arrive` wrote, or, with `.parity`, the parity of the phase.
mbarrier.pending_count : Writes to the first operand, a `u32`, the count of arrivals that the phase still waited for
    just before the arrival that wrote the state in the second. That arrival must be of `mbarrier.arrive.noComplete` or
    `mbarrier.arrive_drop.noComplete`; PTX leaves it undefined if it is not.
vote : Votes on the predicate of the threads of the warp that the mask of lanes names, and writes the result to the
    first operand, as the word says.
match : Compares the second operand of the threads of the warp that the mask of lanes names, and writes to the first
    operand as the word says.
activemask : Writes to the first operand, a `b32`, the mask of the lanes of the warp whose threads run this instruction
    together.
redux : Reduces the second operand of the threads of the warp that the mask of lanes names, as the word says, and writes
    the result to the first operand of each of them.
ldmatrix : Loads matrices of 8 by 8 elements of 16 bits from shared memory, for the warp together: the threads 0 to 7
    give the addresses of the rows of the first matrix, the threads 8 to 15 of the second, and so on, and each thread
    gets a register from each matrix, with two of its elements.
movmatrix : Writes to the first operand of each thread of the warp the transpose of the matrix of 8 by 8 elements of 16
    bits that the second operands of the warp together hold.
mma : Multiplies the matrix a by the matrix b, adds the matrix c, and writes the matrix d to the first operands, for the
    warp together: each thread holds a part of each matrix in its registers. The shape in the name gives their sizes,
    the layouts in it are those of a and b, and the four types are those of d, a, b and c. With `.sp`, a is sparse: it
    holds half its elements, the operand after c, a `b32`, is the metadata that says which, and the last, a number
    known now, picks the threads that hold it.
wmma.load : Loads a matrix, a, b or c as the name says, from memory for the warp together: each thread gets a part of it
    in its registers. The address is of its first element, and the last operand, if there is one, is the distance
    between its rows, or its columns, in elements.
wmma.store : Stores the matrix d from the registers of the warp together to memory: the address is of its first
    element, and the last operand, if there is one, is the distance between its rows, or its columns, in elements.
wmma.mma : Multiplies the matrix a by the matrix b, adds the matrix c, and writes the matrix d to the first operands,
    for the warp together: each thread holds a part of each matrix in its registers. The layouts in the name are those
    of a and b, the types those of d first and c last, and the shape gives the sizes.
stacksave : Writes the stack pointer to the first operand.
stackrestore : Sets the stack pointer to the operand, which `stacksave` wrote: the space that `alloca` took since is
    freed.
alloca : Takes as many bytes of the stack as the second operand says, aligned to the third, if there is one, a power of
    two known now, and writes their address to the first; `stackrestore` frees them.
brkpt : Stops the kernel at a breakpoint, for a debugger.
nanosleep : Suspends the thread for about as many nanoseconds as the operand says: it may wake at any time within twice
    that, and no sleep lasts more than a millisecond.
pmevent : Triggers performance monitor events, as the operand says: the event that it names, 0 to 15, or, with
    `.mask`, each one whose bit it has set.
trap : Ends the kernel with an error, which the host sees as one of the launch.
"#;

/// What each word of a name means, in a phrase: a line for each set of words, then ` : `, then the phrase, which goes
/// on as [`SENTENCES`] do, and has no full stop. A word is by itself, `rn`, or after the instruction it is of,
/// `min.xorsign.abs`, which has it mean what it says there: `lo` and `hi` are halves of a product, and the unsigned
/// `<` and `>` of a comparison. The words that PTX's names have of a type, `f32`, or of a shape, `m16n8k16`, [`rule`]
/// works out.
const PHRASES: &str = r#"
1d : of one dimension, with one coordinate
2d : of two dimensions, with two coordinates
3d : of three dimensions, with three coordinates
2dms : of two dimensions, with several samples to a texel
a1d : an array of those of one dimension: the first coordinate is the index of the layer
a2d : an array of those of two dimensions: the first coordinate is the index of the layer
a2dms : an array of those of two dimensions with several samples to a texel: the first coordinate is the index of the
    layer
cube : a cube map: its coordinates give a direction from its center
acube : an array of cube maps: the first coordinate is the index of the cube
base : the base level of detail
tex.level : takes the level of detail in the operand after the brackets
txq.level : asks of the level of detail that the last operand gives
grad : takes the gradients of the coordinates, two operands after the brackets, to pick the level of detail
tld4.r : the first component of each texel, the red
tld4.g : the second component of each texel, the green
tld4.b : the third component of each texel, the blue
tld4.a : the fourth component of each texel, the alpha
width : the width, in elements
height : the height, in elements
depth : the depth, in elements
array_size : the number of layers of an array
num_mipmap_levels : the number of levels of detail
num_samples : the number of samples to a texel
channel_data_type : the type of the data of the channels, as a number
channel_order : the order of the channels, as a number
normalized_coords : whether the coordinates are normalized, 1 or 0
filter_mode : the filter mode of the sampler, as a number
addr_mode_0 : the addressing mode of the sampler for the first coordinate, as a number
addr_mode_1 : the addressing mode of the sampler for the second coordinate, as a number
addr_mode_2 : the addressing mode of the sampler for the third coordinate, as a number
memory_layout : the layout of the memory of the surface, as a number
texref : a texture
samplerref : a sampler
surfref : a surface
b : unformatted: the bits as they are, of the size of the type
p : formatted: converted to the format of the surface
trap : a coordinate out of range aborts the kernel
clamp : a coordinate out of range is moved to the nearest one in range
zero : a read out of range gives zeros, and a write or a reduction out of range does nothing
ca : caches at all levels, for data that will be used again
cg : caches at the global level, in L2 and not in L1
cs : caches as streaming, for data that will be used once, and evicts it first
lu : last use: the line will not be used again, so the GPU may drop it
cv : treats a cached copy as stale, and fetches the line again
nc : non-coherent: reads through the read-only cache, from memory that nothing writes meanwhile
wb : writes back, at all the levels that keep it coherent
wt : writes through the L2 cache to system memory
L1 : the L1 cache
L2 : the L2 cache
L1::evict_normal : keeps the line in L1 at normal priority
L1::evict_unchanged : leaves the eviction priority of the line in L1 as it is
L1::evict_first : evicts the line from L1 first
L1::evict_last : evicts the line from L1 last
L1::no_allocate : does not allocate a line in L1
L2::evict_normal : keeps the line in L2 at normal priority
L2::evict_unchanged : leaves the eviction priority of the line in L2 as it is
L2::evict_first : evicts the line from L2 first
L2::evict_last : evicts the line from L2 last
L2::cache_hint : takes a cache policy of 64 bits in the last operand, which `createpolicy` writes
L2::64B : hints to fetch 64 bytes more into L2
L2::128B : hints to fetch 128 bytes more into L2
L2::256B : hints to fetch 256 bytes more into L2
weak : a weak access: it orders nothing with the accesses of other threads, which is the default
volatile : a volatile access: the compiler does not remove or merge it, and keeps it in order with other volatile ones,
    but the GPU may merge and reorder the accesses they make
mmio : an access to memory-mapped I/O: the GPU makes every one, merges none with another, and reads none from a cache
relaxed : a relaxed access: atomic, but it orders no other access
acquire : an acquire: no later access of the thread moves before it
release : a release: no earlier access of the thread moves after it
acq_rel : both an acquire and a release
sc : sequentially consistent: the `.sc` fences of the threads in scope fall in one order
cta : the threads of the CTA
gpu : the threads of the GPU
sys : the threads of the whole system, the host and every GPU
gl : the threads of the GPU
alias : accesses to one memory by different virtual addresses
acquire.cta : an acquire for the threads of the CTA
release.cta : a release for the threads of the CTA
const : constant memory, which is read only
global : global memory, which every thread of the GPU reaches
local : the thread's own memory
shared : the memory that the threads of a CTA share
shared::cta : the same as `.shared`, with the CTA said
param : the parameters of the kernel
param::entry : the parameters of a kernel
to : from a generic address to the state space's
ld.v2 ldu.v2 suld.v2 : reads two values at once, into registers taken together
ld.v4 ldu.v4 suld.v4 : reads four values at once, into registers taken together
st.v2 sust.v2 : writes two values at once, from registers taken together
st.v4 sust.v4 : writes four values at once, from registers taken together
tex.v2 : two values: for the `f16x2` types, the four components of a texel, two in each
tex.v4 tld4.v4 : four values: the four components of a texel, or one component of each of four texels
rn : rounds to the nearest, to the even one if two are as near
rz : rounds toward zero
rm : rounds toward minus infinity
rp : rounds toward plus infinity
rna : rounds to the nearest, away from zero if two are as near
rni : rounds to the nearest integer, to the even one if two are as near
rzi : rounds to an integer toward zero
rmi : rounds to an integer toward minus infinity
rpi : rounds to an integer toward plus infinity
ftz : flushes a subnormal input or result to a zero
sat : clamps the result to the range of its type; a float to 0.0 to 1.0
satfinite : clamps a result out of the range of its type to the nearest finite value: a float is never an infinity,
    an integer never wraps
relu : a negative result is 0
NaN : if either operand is a NaN, so is the result
xorsign.abs : compares the magnitudes, and gives the result the sign of the XOR of the signs
approx : an approximation, with the error that PTX's ISA gives it
full : a fast approximation over the full range of inputs, within 2 ulps
finite : a number: not an infinity, not a NaN
infinite : an infinity
number : not a NaN
notanumber : a NaN
normal : a normal number: not zero, not subnormal, not an infinity, not a NaN
subnormal : a subnormal number: not zero, and smaller than any normal number
mul.lo mad.lo madc.lo : the low half of the product, as many bits as an operand has
mul.hi mad.hi madc.hi : the high half of the product, as many bits as an operand has
wide : the whole product, in a register twice as wide as the operands
clmad.lo : the low 64 bits of the product
clmad.hi : the high 64 bits of the product
mul24.lo mad24.lo : the low 32 bits of the 48
mul24.hi mad24.hi : the 32 bits of the 48 from bit 16 up
dp2a.lo : the low two bytes of the third operand
dp2a.hi : the high two bytes of the third operand
cc : writes the carry out to the carry flag, which `addc`, `subc` and `madc` read; for a subtraction, the borrow
shiftamt : writes the shift left that brings the bit to the top, instead of its index
szext.clamp : a width past 32 is taken as 32
szext.wrap : the width is taken modulo 32
bmsk.clamp : a start or a length past 32 is taken as 32
bmsk.wrap : the start and the length are taken modulo 32
shf.l : shifts left, and writes the high 32 bits
shf.r : shifts right, and writes the low 32 bits
shf.clamp : a count past 32 is taken as 32
shf.wrap : the count is taken modulo 32
vshl.clamp vshr.clamp : a count past 32 is taken as 32
vshl.wrap vshr.wrap : the count is taken modulo 32
vadd.add vsub.add vabsdiff.add vmin.add vmax.add vshl.add vshr.add vset.add : adds the result to the fourth operand
vadd.min vsub.min vabsdiff.min vmin.min vmax.min vshl.min vshr.min vset.min : keeps the lesser of the result and the
    fourth operand
vadd.max vsub.max vabsdiff.max vmin.max vmax.max vshl.max vshr.max vset.max : keeps the greater of the result and the
    fourth operand
vadd2.add vsub2.add vavrg2.add vabsdiff2.add vmin2.add vmax2.add vset2.add : adds the results of the halves and the
    fourth operand, and writes the sum to the whole of the first
vadd4.add vsub4.add vavrg4.add vabsdiff4.add vmin4.add vmax4.add vset4.add : adds the results of the bytes and the
    fourth operand, and writes the sum to the whole of the first
vmad.po : adds 1 to the sum
vmad.shr7 : shifts the sum right by 7 bits
vmad.shr15 : shifts the sum right by 15 bits
eq : holds if the second operand is equal to the third
ne : holds if the second operand is not equal to the third
lt : holds if the second operand is less than the third
le : holds if the second operand is less than or equal to the third
gt : holds if the second operand is greater than the third
ge : holds if the second operand is greater than or equal to the third
setp.lo set.lo : holds if the second operand is lower than the third, as unsigned integers
setp.ls set.ls : holds if the second operand is lower than the third or the same, as unsigned integers
setp.hi set.hi : holds if the second operand is higher than the third, as unsigned integers
setp.hs set.hs : holds if the second operand is higher than the third or the same, as unsigned integers
equ : holds if the second operand is equal to the third, or either is a NaN
neu : holds if the second operand is not equal to the third, or either is a NaN
ltu : holds if the second operand is less than the third, or either is a NaN
leu : holds if the second operand is less than or equal to the third, or either is a NaN
gtu : holds if the second operand is greater than the third, or either is a NaN
geu : holds if the second operand is greater than or equal to the third, or either is a NaN
num : holds if neither operand is a NaN
nan : holds if either operand is a NaN
setp.and set.and : combines the result with the fourth operand, a predicate, by AND
setp.or set.or : combines the result with the fourth operand, a predicate, by OR
setp.xor set.xor : combines the result with the fourth operand, a predicate, by XOR
lop3.and : also writes true to the predicate beside the first if the result is not 0 and the last operand is true
lop3.or : also writes true to the predicate beside the first if the result is not 0 or the last operand is true
f4e : the four bytes that start at byte n, going up
b4e : the four bytes that start at byte n, going down, around the eight
rc8 : byte n, four times
rc16 : the low half of the second operand if n is even, the high half if it is odd, twice
ecl : the bytes of the second operand, but those below byte n are byte n
ecr : the bytes of the second operand, but those above byte n are byte n
sync : the threads that take part wait for each other, and all run this instruction
aligned : every thread of the warp runs this instruction, as PTX requires
barrier.aligned : every thread of the CTA runs this instruction, as PTX requires
barrier.sync bar.sync : waits until all the threads that take part have arrived
barrier.arrive bar.arrive : says that the thread has arrived, and goes on without waiting
barrier.red bar.red : waits as `.sync` does, and reduces a predicate over the threads
barrier.popc bar.popc : counts the threads whose predicate is true
barrier.and bar.and : true if the predicate is true in every thread
barrier.or bar.or : true if the predicate is true in at least one thread
up : from the lane as many below this one as the third operand says
down : from the lane as many above this one as the third operand says
bfly : from the lane whose index is this one's XOR the third operand
shfl.idx : from the lane that the third operand names
brx.idx : the index picks the label
uni : the jump is the same for every thread of the warp
ret.uni : every thread of the warp returns together
vote.uni : true if the predicate is the same in all the threads
vote.all : true if the predicate is true in all the threads
vote.any : true if the predicate is true in at least one thread
ballot : a `b32` with a bit set for each lane whose thread has the predicate true
match.any : the mask of the threads that hold the same value as this one
match.all : the mask of all the threads if they all hold the same value, and 0 if not, and a predicate that says which
atom.add red.add : adds the operand to the value in memory
atom.min red.min : keeps the lesser of the value in memory and the operand
atom.max red.max : keeps the greater of the value in memory and the operand
atom.and red.and : ANDs the operand into the value in memory
atom.or red.or : ORs the operand into the value in memory
atom.xor red.xor : XORs the operand into the value in memory
atom.inc red.inc : adds 1 to the value in memory, or sets it to 0 if it is at least the operand
atom.dec red.dec : subtracts 1 from the value in memory, or sets it to the operand if it is 0 or more than the operand
atom.exch : replaces the value in memory by the operand
atom.cas : replaces the value in memory by the last operand, if it equals the operand before
noftz : does not flush a subnormal to zero
redux.add : the sum of the operands
redux.min : the least of the operands
redux.max : the greatest of the operands
redux.and : the bits that all the operands have set
redux.or : the bits that any operand has set
redux.xor : the bits that an odd number of operands have set
sured.add : adds the operand to the sample
sured.min : keeps the lesser of the sample and the operand
sured.max : keeps the greater of the sample and the operand
sured.and : ANDs the operand into the sample
sured.or : ORs the operand into the sample
noinc : counts the arrival without raising the pending count of the mbarrier first: it is one the mbarrier expects
noComplete : counts as many arrivals as the last operand says, which do not complete the phase; PTX leaves it undefined
    if they do
parity : the phase is the one whose parity, 0 or 1, the last operand says
mask : a mask of 16 bits instead: each event whose bit is set is triggered
trans : the matrix is transposed
row : stored by rows
col : stored by columns
sp : the matrix a is sparse
sp::ordered_metadata : as `.sp`, with the metadata's indices in order, rising from the lowest bits up
mma.and wmma.and : each product is a bit of a ANDed with a bit of b
mma.xor wmma.xor : each product is a bit of a XORed with a bit of b
mma.popc wmma.popc : each sum counts the bits that are set
wmma.a : the matrix a, the left factor
wmma.b : the matrix b, the right factor
wmma.c : the matrix c, which the product is added to
wmma.d : the matrix d, the result
wmma.u8.u8 : a and b are unsigned integers of 8 bits
wmma.s8.s8 : a and b are signed integers of 8 bits
wmma.u4.u4 : a and b are unsigned integers of 4 bits
wmma.s4.s4 : a and b are signed integers of 4 bits
b1 : one bit
u4 : an unsigned integer of 4 bits
s4 : a signed integer of 4 bits
u2 : an unsigned integer of 2 bits
s2 : a signed integer of 2 bits
"#;

/// The texts of [`SENTENCES`] by name, parsed once. Text that does not parse has none: a test of the crate parses it,
/// and says which line is wrong.
fn sentences() -> &'static BTreeMap<&'static str, String> {
    static BY_NAME: OnceLock<BTreeMap<&'static str, String>> = OnceLock::new();
    BY_NAME.get_or_init(|| kernite_build::texts(SENTENCES).unwrap_or_default())
}

/// The texts of [`PHRASES`] by word, parsed once, as [`sentences`] are.
fn phrases() -> &'static BTreeMap<&'static str, String> {
    static BY_WORD: OnceLock<BTreeMap<&'static str, String>> = OnceLock::new();
    BY_WORD.get_or_init(|| kernite_build::texts(PHRASES).unwrap_or_default())
}

/// What the instruction `name` does, in a sentence, and the first words of `name` that the sentence is of: the longest
/// that [`SENTENCES`] has a sentence of.
fn sentence(name: &str) -> Option<(&str, &'static str)> {
    let mut key = name;
    loop {
        if let Some(text) = sentences().get(key) {
            return Some((key, text.as_str()));
        }
        key = key.rsplit_once('.')?.0;
    }
}

/// What a value of type `t` is.
fn of_type(t: Ty) -> String {
    let bits = t.bits();
    match (t, t.kind()) {
        (Ty::F16, _) => "a float of 16 bits, IEEE's half".into(),
        (Ty::F16x2, _) => "two `f16` in 32 bits, the first in the low half".into(),
        (Ty::Bf16, _) => "a float of 16 bits with the exponent of an `f32`".into(),
        (Ty::Bf16x2, _) => "two `bf16` in 32 bits, the first in the low half".into(),
        (Ty::Tf32, _) => "a float with the exponent of an `f32` and the precision of an `f16`, in 32 bits".into(),
        (_, Kind::Pred) => "a predicate, true or false".into(),
        (_, Kind::Bit) => format!("{bits} bits, whatever they mean"),
        (_, Kind::Unsigned) => format!("an unsigned integer of {bits} bits"),
        (_, Kind::Signed) => format!("a signed integer of {bits} bits"),
        (_, Kind::Float) => format!("a float of {bits} bits"),
    }
}

/// What a word of a name that is a type, a count of matrices or a shape means, which no table says word by word.
fn rule(word: &str) -> Option<String> {
    if let Some(t) = Ty::parse(word) {
        return Some(of_type(t));
    }
    if let Some(n) = word.strip_prefix('x').and_then(|n| n.parse::<u32>().ok()) {
        return Some(match n {
            1 => "one matrix".into(),
            2 => "two matrices".into(),
            n => format!("{n} matrices"),
        });
    }
    let (m, rest) = word.strip_prefix('m')?.split_once('n')?;
    let number = |text: &str| text.parse::<u32>().ok();
    match rest.split_once('k') {
        None => Some(format!("matrices of {} by {} elements", number(m)?, number(rest)?)),
        Some((n, k)) => {
            let (m, n, k) = (number(m)?, number(n)?, number(k)?);
            Some(format!("a is {m} by {k}, b is {k} by {n}, and c and d are {m} by {n}"))
        }
    }
}

/// What the word `word` means in a name of the instruction `op`: its phrase for `op`, or its phrase, or the one the
/// rule gives.
fn means(op: &str, word: &str) -> Option<String> {
    let all = phrases();
    match all.get(format!("{op}.{word}").as_str()).or_else(|| all.get(word)) {
        Some(phrase) => Some(phrase.clone()),
        None => rule(word),
    }
}

/// The words the name `b` is of, after its first, each with the number of words before it: an element's word, or its
/// words joined by dots, as `xorsign.abs` is.
fn words(b: &Bound) -> Vec<(usize, String)> {
    let mut before = 0;
    let mut out = vec![];
    for (elem, choice) in b.form.elems.iter().zip(&b.choice) {
        if let Some(alt) = choice.and_then(|k| elem.alts.get(k)) {
            out.push((before, alt.words.join(".")));
            before += alt.words.len();
        }
    }
    out
}

/// What the forms of a doc leave to say: the signs they put before an operand, and a note for each kind of operand that
/// a reader may not know the table's word for.
#[derive(Default)]
struct Notes {
    /// The signs, `=`, in the order they came.
    signs: Vec<&'static str>,
    /// The notes, each once.
    texts: Vec<String>,
}

impl Notes {
    /// Says that an operand has `sign` before it.
    fn sign(&mut self, sign: &'static str) {
        if !self.signs.contains(&sign) {
            self.signs.push(sign);
        }
    }

    /// Adds `text`, if the notes lack it.
    fn say(&mut self, text: &str) {
        if !self.texts.iter().any(|t| t == text) {
            self.texts.push(text.to_string());
        }
    }

    /// What the signs say, as one note.
    fn signed(&self) -> Option<&'static str> {
        match (self.signs.contains(&"="), self.signs.contains(&"+")) {
            (true, true) => Some("`=` marks an operand it writes, `+` one it reads and writes."),
            (true, false) => Some("`=` marks an operand it writes."),
            (false, true) => Some("`+` marks an operand it reads and writes."),
            (false, false) => None,
        }
    }
}

/// What a mark before an address says.
fn mark(access: Access) -> &'static str {
    match access {
        Access::Write => "=",
        Access::ReadWrite => "+",
        Access::Read | Access::Ignored => "",
    }
}

/// How a kernel writes the part of a variable that `sel` says, `.bh`, and what that is; `written` if the instruction
/// writes the variable.
fn part(sel: Sel, written: bool) -> (&'static str, &'static str) {
    match (sel, written) {
        (Sel::One, false) => (
            ".bh",
            "`.bh` after the second or the third operand is the byte or half of it that is read, `.b0` to `.b3`, \
             `.h0` or `.h1`; without it, all 32 bits are.",
        ),
        (Sel::One, true) => (
            ".bh",
            "`.bh` after the first operand is the byte or half that the result goes to, `.b0` to `.b3`, `.h0` or \
             `.h1`, and the rest of the first is the fourth operand's; without it, the result is all of the first, \
             and the fourth is read for nothing.",
        ),
        (Sel::Halves, _) => (
            ".hxy",
            "`.hxy` is two of the four halves of the two sources, as `.h10` or `.h32`; without it, the second \
             operand's own, `.h10`, and the third's, `.h32`.",
        ),
        (Sel::Bytes, _) => (
            ".bxyzw",
            "`.bxyzw` is four of the eight bytes of the two sources, as `.b3210` or `.b7654`; without it, the second \
             operand's own, `.b3210`, and the third's, `.b7654`.",
        ),
        (Sel::HalfMask, _) => (
            ".hmask",
            "`.hmask` after the first operand is the halves of the result it writes, `.h0`, `.h1` or `.h10`, which is \
             the default; the fourth operand gives the other halves, and is read for nothing else but `.add`.",
        ),
        (Sel::ByteMask, _) => (
            ".bmask",
            "`.bmask` after the first operand is the bytes of the result it writes, from `.b3` down to `.b0`, each \
             once at most, as `.b31`, and all four by default; the fourth operand gives the other bytes, and is read \
             for nothing else but `.add`.",
        ),
    }
}

/// What a number known now of `spec` may be, as the table writes it: `[0,15]` for a range, `[0,1024 by 32]` for the
/// multiples of 32 in it, `(0,1]` for a fraction, and `&0x1f1f` for the bits it may have; and what each says.
fn limits(spec: &Spec, notes: &mut Notes) -> String {
    let mut text = match (spec.range, spec.step, spec.fraction) {
        (Some((from, to)), Some(step), _) => format!("[{from},{to} by {step}]"),
        (Some((from, to)), None, _) => format!("[{from},{to}]"),
        (None, Some(step), true) => format!("(0,1 by 1/{step}]"),
        (None, None, true) => "(0,1]".to_string(),
        (None, _, false) => String::new(),
    };
    if spec.range.is_some() {
        notes.say(
            "`[a,b]` after a type limits a number known now to the numbers from a to b, and `[a,b by n]` to the \
             multiples of n among them; a variable is not checked.",
        );
    }
    if spec.fraction {
        notes.say(
            "`(0,1]` limits a number known now to more than 0 and at most 1, and `(0,1 by 1/n]` to the multiples of \
             1/n among them.",
        );
    }
    if let Some(mask) = spec.mask {
        text += &format!("&{mask:#x}");
        notes.say("`&0x1f1f` limits a number known now to the bits of the mask.");
    }
    text
}

/// How a kernel writes one register of `spec`, whose type is `ty`: the type, or what stands for it, with the other
/// things it may be, apart by `/`.
fn element(spec: &Spec, ty: &str, notes: &mut Notes) -> String {
    let mut base = ty.to_string();
    if spec.lanes {
        notes.say(
            "`lanes` is a mask of lanes: a `u32` whose bits are the lanes of the warp that take part, the thread that \
             runs the instruction among them.",
        );
        base = "lanes".into();
    }
    if let Some(sel) = spec.part {
        let (word, note) = part(sel, spec.role == Role::Dst);
        base += word;
        notes.say(note);
    }
    if !spec.lanes {
        base += &limits(spec, notes);
    }
    let mut alts = vec![base.clone()];
    if spec.not && ty == Ty::Pred.name() {
        alts.push(format!("!{base}"));
    }
    if spec.neg {
        alts.push(format!("-{base}"));
        notes.say("`-T` is a variable of type T negated, as `-a`.");
    }
    if spec.sreg && ty != Ty::Pred.name() {
        alts.push("sreg".into());
        notes.say("`sreg` is a special register, as `tid.x`.");
    }
    if spec.memory {
        alts.push("memory".into());
        notes.say("`memory` is memory the kernel declares, which stands for its address.");
    }
    if spec.sink {
        alts.push("_".into());
        notes.say("`_` may stand for an operand whose value the kernel does not want.");
    }
    alts.join("/")
}

/// How a kernel writes the registers of `spec`: one, or a few taken together, `f32*4`.
fn registers(b: &Bound, spec: &Spec, notes: &mut Notes) -> String {
    let ty = spec.ty.and_then(|t| b.ty(t)).map_or("", Ty::name);
    let rest = element(spec, ty, notes);
    let mut parts: Vec<String> = spec.lead.map(|t| element(spec, t.name(), notes)).into_iter().collect();
    parts.extend(std::iter::repeat_n(rest, b.count(spec)));
    let [first, others @ ..] = parts.as_slice() else { return String::new() };
    if others.is_empty() {
        return first.clone();
    }
    notes.say("Registers taken together are a tuple in a kernel, `(a, b)`; `T*n` is n of type T.");
    if others.iter().all(|p| p == first) {
        let one = if first.contains('/') { format!("({first})") } else { first.clone() };
        format!("{one}*{}", parts.len())
    } else {
        format!("({})", parts.join(", "))
    }
}

/// The address a kernel writes, whose variable is of 64 bits if `wide`, and what that is.
fn address(wide: bool, notes: &mut Notes) -> String {
    let (ty, bits) = if wide { ("u64", "64 bits") } else { ("u32", "32 bits or 64") };
    notes.say(&format!(
        "`[{ty}]` is an address in a variable of {bits}, `[p]`, with a number of bytes known now added, `[p + 4]`; \
         memory the kernel declares gives one too, `[m + 4]`."
    ));
    format!("[{ty}]")
}

/// How a kernel writes the integer known now of `imm`, and what that is.
fn immediate(imm: Imm, notes: &mut Notes) -> String {
    notes.say(
        "`#` marks an integer known now: `#8` is that one, `#0-255` any from 0 to 255, `#4|8|16` one of those, and \
         `#2^0-2^23` a power of two from 2^0 to 2^23.",
    );
    match imm {
        Imm::Bits(n) => format!("#0-{}", 1u64.checked_shl(n).map_or(u64::MAX, |v| v - 1)),
        Imm::Of(values) => format!("#{values}"),
        Imm::Pow2(p) => format!("#2^0-2^{p}"),
    }
}

/// How a kernel writes the operand `spec` of `b`: its type, with a sign before it if the instruction writes it;
/// `spaces` are the state spaces of the addresses still to come.
fn operand(b: &Bound, spec: &Spec, spaces: &mut std::vec::IntoIter<Space>, notes: &mut Notes) -> String {
    let (sign, body) = match spec.role {
        Role::Dst => ("=", registers(b, spec, notes)),
        Role::Src => ("", registers(b, spec, notes)),
        Role::Addr(access) => (mark(access), address(spaces.next().is_some_and(|s| b.wide(s)), notes)),
        Role::Label => ("", "label".to_string()),
        Role::Labels => {
            notes.say("`labels` are labels taken together, `(a, b)`, of which the index picks one.");
            ("", "labels".to_string())
        }
        Role::Func => {
            notes.say("`function` is a function of the module or a kernel, whose address it takes.");
            ("", "function".to_string())
        }
        Role::Handle => {
            notes.say(
                "A texture or a surface is `[u64, ..]` in brackets: its handle, then the coordinates it is read at, \
                 as `[t, (x, y)]` in a kernel.",
            );
            let inner: Vec<String> = spec.inner.iter().map(|s| operand(b, s, spaces, notes)).collect();
            ("", format!("[{}]", inner.join(", ")))
        }
        Role::Imm(imm) => ("", immediate(imm, notes)),
    };
    if !sign.is_empty() {
        notes.sign(sign);
    }
    let must = matches!(spec.role, Role::Dst | Role::Src) && spec.reg_only;
    if must {
        notes.say("`%` marks an operand that must be a variable, not a number known now.");
    }
    let mut text = format!("{sign}{}{body}", if must { "%" } else { "" });
    if let Some((second, optional)) = &spec.second {
        notes.say("`a | b` is two operands written at once, `p | q`; `[| b]` may be left out.");
        let second = operand(b, second, spaces, notes);
        text += &if *optional { format!(" [| {second}]") } else { format!(" | {second}") };
    }
    if spec.optional {
        notes.say("`?` after an operand says it may be left out.");
        text += "?";
    }
    text
}

/// How a kernel writes the operands of `b`, which are those that the name has, as a form's `when` says.
fn operands(b: &Bound, notes: &mut Notes) -> String {
    let mut spaces = b.spaces().into_iter();
    let shown: Vec<String> = b
        .form
        .operands
        .iter()
        .filter(|s| s.when.is_none_or(|w| b.has(w)))
        .map(|s| operand(b, s, &mut spaces, notes))
        .collect();
    shown.join(", ")
}

/// The PTX and the GPU `needs` is, as the doc says them: `7.0 sm_80`, which a module's `.version` and `.target` write.
fn at(needs: Needs) -> String {
    format!("{} {}", needs.version(), needs.target())
}

/// The chapter of PTX's ISA that has the instruction `b` is of.
fn chapter(b: &Bound) -> Option<&'static str> {
    let half = b.types.iter().any(|t| matches!(t, Ty::F16 | Ty::F16x2 | Ty::Bf16 | Ty::Bf16x2));
    let float = b.types.first().is_some_and(|t| t.kind() == Kind::Float);
    Some(match b.form.op {
        "set" | "setp" if half => "Half Precision Comparison Instructions",
        "add" | "sub" | "mul" | "fma" | "abs" | "neg" | "min" | "max" | "tanh" | "ex2" if half => {
            "Half Precision Floating-Point Instructions"
        }
        "add" | "sub" | "mul" | "mad" | "div" | "abs" | "neg" | "min" | "max" if float => "Floating-Point Instructions",
        "testp" | "copysign" | "fma" | "rcp" | "sqrt" | "rsqrt" | "sin" | "cos" | "lg2" | "ex2" | "tanh" => {
            "Floating-Point Instructions"
        }
        "addc" | "subc" | "madc" => "Extended-Precision Integer Arithmetic Instructions",
        "add" | "sub" | "mad" if b.has("cc") => "Extended-Precision Integer Arithmetic Instructions",
        "add" | "sub" | "mul" | "mad" | "clmad" | "mul24" | "mad24" | "sad" | "div" | "rem" | "abs" | "neg" | "min"
        | "max" | "popc" | "clz" | "bfind" | "fns" | "brev" | "bfe" | "bfi" | "szext" | "bmsk" | "dp4a" | "dp2a" => {
            "Integer Arithmetic Instructions"
        }
        "setp" | "set" | "selp" | "slct" => "Comparison and Selection Instructions",
        "and" | "or" | "xor" | "not" | "cnot" | "lop3" | "shf" | "shl" | "shr" => "Logic and Shift Instructions",
        "mov" | "shfl" | "prmt" | "ld" | "ldu" | "st" | "prefetch" | "prefetchu" | "applypriority" | "discard"
        | "createpolicy" | "isspacep" | "cvta" | "cvt" | "cp" => "Data Movement and Conversion Instructions",
        "tex" | "tld4" | "txq" | "istypep" => "Texture Instructions",
        "suld" | "sust" | "sured" | "suq" => "Surface Instructions",
        "bra" | "brx" | "ret" | "exit" => "Control Flow Instructions",
        "barrier" | "bar" | "membar" | "fence" | "atom" | "red" | "redux" | "vote" | "match" | "activemask"
        | "mbarrier" => "Parallel Synchronization and Communication Instructions",
        "ldmatrix" | "movmatrix" | "mma" | "wmma" => "Warp Level Matrix Multiply-Accumulate Instructions",
        "vadd" | "vsub" | "vabsdiff" | "vmin" | "vmax" | "vshl" | "vshr" | "vmad" | "vset" | "vadd2" | "vsub2"
        | "vavrg2" | "vabsdiff2" | "vmin2" | "vmax2" | "vset2" | "vadd4" | "vsub4" | "vavrg4" | "vabsdiff4"
        | "vmin4" | "vmax4" | "vset4" => "Video Instructions",
        "stacksave" | "stackrestore" | "alloca" => "Stack Manipulation Instructions",
        "brkpt" | "nanosleep" | "pmevent" | "trap" => "Miscellaneous Instructions",
        _ => return None,
    })
}

/// The doc of the instruction `name`, as the language calls it, in Markdown: what it does, what each word of its name
/// means, its forms with the operands it writes marked, the version of PTX and the GPU it needs, and where PTX's ISA
/// says the rest. None for a name that is no instruction kernite writes.
pub fn of(name: &str) -> Option<String> {
    let bound = forms::find(name).ok()?;
    let first = bound.first()?;
    let (key, sentence) = sentence(name)?;
    let mut notes = Notes::default();
    let mut lines: Vec<(String, String)> = vec![];
    for b in &bound {
        let line = (format!("{name}({})", operands(b, &mut notes)), at(b.needs));
        if !lines.contains(&line) {
            lines.push(line);
        }
    }
    // The first form is the plain one: it says what the name needs, and a form that needs other says so after it.
    let base = at(first.needs);
    let shown: Vec<String> =
        lines.iter().map(|(line, n)| if *n == base { line.clone() } else { format!("{line}  @{n}") }).collect();
    let mut text: Vec<String> = notes.signed().map(String::from).into_iter().chain(notes.texts).collect();
    if bound.iter().any(|b| b.form.wide && b.types.iter().any(|t| t.int() || t.kind() == Kind::Bit)) {
        text.push(
            "For an integer or bits type, a variable of more bits is taken too: its low bits are read, and a value \
             written to it is extended, by its sign for a signed type and by zeros for the rest."
                .into(),
        );
    }
    let more = lines.iter().any(|(_, n)| *n != base);
    match (base != at(Needs::FLOOR), more) {
        (true, false) => text.push(format!("Needs PTX {} and `{}`.", first.needs.version(), first.needs.target())),
        (true, true) => text.push(format!(
            "Needs PTX {} and `{}`, but a form with `@` after it needs the PTX and the GPU that follow.",
            first.needs.version(),
            first.needs.target()
        )),
        (false, true) => text.push("A form with `@` after it needs the PTX and the GPU that follow.".into()),
        (false, false) => {}
    }
    let skip = key.matches('.').count();
    let mut listed: Vec<String> = vec![];
    for (_, word) in words(first).into_iter().filter(|(before, _)| *before >= skip) {
        let line = format!("- `.{word}`: {}", means(first.form.op, &word)?);
        if !listed.contains(&line) {
            listed.push(line);
        }
    }
    let mut doc = format!("{sentence}\n\n");
    if !listed.is_empty() {
        doc += &format!("{}\n\n", listed.join("\n"));
    }
    doc += &format!("```text\n{}\n```\n\n", shown.join("\n"));
    if !text.is_empty() {
        doc += &format!("{}\n\n", text.join(" "));
    }
    // PTX's ISA has a section of `shfl` and of `vote` without `.sync`, deprecated, before the one of each with it.
    let section = if matches!(key, "shfl" | "vote") { format!("{key}.sync") } else { key.to_string() };
    doc += &format!("PTX's ISA, {}: {section}.", chapter(first)?);
    Some(doc)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::forms::{Alt, Form};

    type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

    /// The instructions of the table.
    fn ops() -> BTreeSet<&'static str> {
        forms::forms().iter().map(|f| f.op).collect()
    }

    /// The first word of each element of `form`, and none for an element that may be left out, unless `all`.
    fn base(form: &'static Form, all: bool) -> Vec<Option<&'static Alt>> {
        form.elems.iter().map(|e| if e.optional && !all { None } else { e.alts.first() }).collect()
    }

    /// `picked` with `alt` in the place of the element `at`.
    fn with(picked: &[Option<&'static Alt>], at: usize, alt: &'static Alt) -> Vec<Option<&'static Alt>> {
        let mut out = picked.to_vec();
        if let Some(place) = out.get_mut(at) {
            *place = Some(alt);
        }
        out
    }

    /// The name of `form` that has these words: none for an element left out.
    fn written(form: &Form, picked: &[Option<&Alt>]) -> String {
        let words: Vec<String> = picked.iter().flatten().map(|a| a.words.join(".")).collect();
        [vec![form.op.to_string()], words].concat().join(".")
    }

    /// Names that the forms of the table are of: for each form, its first word of each element, and each word of each
    /// element in turn in its place, with the elements that may be left out there and without them. Some are of no
    /// form: a texture of `.v2` wants `.f16x2`, which the first word of its type is not.
    fn names() -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for form in forms::forms() {
            for all in [false, true] {
                let base = base(form, all);
                out.insert(written(form, &base));
                for (at, elem) in form.elems.iter().enumerate() {
                    out.extend(elem.alts.iter().map(|alt| written(form, &with(&base, at, alt))));
                }
            }
        }
        out
    }

    /// Whether a form takes `name`; if it does, the name has a doc, and each word of the name that its sentence leaves
    /// to the phrases has one. The words of the name are put in `covered`, each with its instruction, and the phrases
    /// they have in `said`.
    fn check(name: &str, covered: &mut BTreeSet<(&'static str, String)>, said: &mut BTreeSet<String>) -> Result<bool> {
        let Ok(bound) = forms::find(name) else { return Ok(false) };
        let doc = of(name).ok_or_else(|| format!("`{name}` has no doc"))?;
        assert!(doc.contains("\nPTX's ISA, "), "`{name}`: {doc}");
        // Each line of the forms is the name, its operands in brackets, and, if it needs other, what.
        let block = doc.split("```text\n").nth(1).and_then(|rest| rest.split("\n```").next()).ok_or("no forms")?;
        let forms: BTreeSet<_> = bound.iter().map(|b| (operands(b, &mut Notes::default()), at(b.needs))).collect();
        assert_eq!(block.lines().count(), forms.len(), "`{name}` has not each form of the table once:\n{doc}");
        for line in block.lines() {
            let form = line.split_once("  @").map_or(line, |(form, _)| form);
            assert!(form.starts_with(&format!("{name}(")) && form.ends_with(')'), "`{name}`: {line}");
        }
        let (key, _) = sentence(name).ok_or("no sentence")?;
        for b in &bound {
            for (before, word) in words(b) {
                if before >= key.matches('.').count() {
                    assert!(means(b.form.op, &word).is_some(), "`{name}`: `.{word}` means nothing");
                    said.insert(format!("{}.{word}", b.form.op));
                    said.insert(word.clone());
                }
                covered.insert((b.form.op, word));
            }
        }
        Ok(true)
    }

    #[test]
    fn the_sentences_parse_and_each_is_of_a_name_kernite_writes() -> Result {
        let all = kernite_build::texts(SENTENCES)?;
        let (ops, names) = (ops(), names());
        for (key, sentence) in &all {
            let op = key.split('.').next().unwrap_or_default();
            assert!(ops.contains(op), "`{key}` is no instruction of the table");
            assert!(names.iter().any(|n| n == key || n.starts_with(&format!("{key}."))), "`{key}` starts no name");
            assert!(sentence.ends_with('.') && !sentence.contains("  "), "`{key}`: {sentence}");
        }
        for op in ops {
            assert!(all.contains_key(op) || all.keys().any(|k| k.starts_with(&format!("{op}."))), "`{op}` has none");
        }
        Ok(())
    }

    /// Each phrase is of a word the table has, by itself or after an instruction that has it.
    #[test]
    fn the_phrases_parse_and_each_is_of_a_word_the_table_has() -> Result {
        let all = kernite_build::texts(PHRASES)?;
        let ops = ops();
        let words = |op: Option<&str>| -> BTreeSet<String> {
            let forms = forms::forms().iter().filter(|f| op.is_none_or(|o| f.op == o));
            forms.flat_map(|f| f.elems.iter().flat_map(|e| e.alts.iter().map(|a| a.words.join(".")))).collect()
        };
        for (key, phrase) in &all {
            assert!(!phrase.is_empty() && !phrase.ends_with('.') && !phrase.contains("  "), "`{key}`: {phrase}");
            let known = match key.split_once('.') {
                Some((op, word)) if ops.contains(op) => words(Some(op)).contains(word),
                _ => words(None).contains(*key),
            };
            assert!(known, "`{key}` is no word of the table");
        }
        Ok(())
    }

    /// Each word of the table is in a name that has a doc, in which it has a phrase, if the sentence leaves it to them;
    /// and each phrase is the one of some word of such a name.
    #[test]
    fn each_instruction_of_the_table_has_a_doc_and_each_word_of_it_a_phrase() -> Result {
        let (mut covered, mut said, mut seen) = (BTreeSet::new(), BTreeSet::new(), 0);
        for name in names() {
            seen += usize::from(check(&name, &mut covered, &mut said)?);
        }
        assert!(seen > 4000, "{seen} names");
        // A word that no name with it and the first words of the rest is of a form takes needs one other word beside.
        for form in forms::forms() {
            for (at, elem) in form.elems.iter().enumerate() {
                for alt in &elem.alts {
                    let word = alt.words.join(".");
                    'search: for all in [false, true] {
                        for (other, others) in form.elems.iter().enumerate() {
                            for second in &others.alts {
                                if covered.contains(&(form.op, word.clone())) {
                                    break 'search;
                                }
                                if other != at {
                                    let picked = with(&with(&base(form, all), at, alt), other, second);
                                    check(&written(form, &picked), &mut covered, &mut said)?;
                                }
                            }
                        }
                    }
                    assert!(covered.contains(&(form.op, word.clone())), "`{}.{word}` is in no name of a form", form.op);
                }
            }
        }
        for key in kernite_build::texts(PHRASES)?.keys() {
            assert!(said.contains(*key), "`{key}` is the phrase of no word of a doc");
        }
        Ok(())
    }

    #[test]
    fn a_name_that_is_no_instruction_has_no_doc() {
        for name in [
            "ld.globl.f32",
            "ld.global.f33",
            "vaddps",
            "wgmma",
            "shfl.bfly.b32",
            "vote.all.pred",
            "vote.ballot.b32",
            "",
            "label",
            "ld.global.",
        ] {
            assert_eq!(of(name), None, "`{name}`");
        }
        for name in ["shfl.sync.bfly.b32", "vote.sync.all.pred", "vote.sync.ballot.b32"] {
            assert!(of(name).is_some(), "`{name}`");
        }
    }

    /// A doc, whole: a sentence, what each word of its name is, the forms with a mark on each operand written, and
    /// where PTX's ISA says the rest.
    #[test]
    fn a_doc_says_what_it_does_what_its_words_are_its_forms_and_where_ptx_says_the_rest() {
        let ld = "Reads memory at the address in the second operand, and writes it to the first.\n\n\
                  - `.global`: global memory, which every thread of the GPU reaches\n\
                  - `.f32`: a float of 32 bits\n\n\
                  ```text\nld.global.f32(=f32/_, [u64])\n```\n\n\
                  `=` marks an operand it writes. `_` may stand for an operand whose value the kernel does not want. \
                  `[u64]` is an address in a variable of 64 bits, `[p]`, with a number of bytes known now added, \
                  `[p + 4]`; memory the kernel declares gives one too, `[m + 4]`.\n\n\
                  PTX's ISA, Data Movement and Conversion Instructions: ld.";
        assert_eq!(of("ld.global.f32").as_deref(), Some(ld));
        let fma = of("fma.rn.ftz.f32").unwrap_or_default();
        assert!(fma.contains("- `.rn`: rounds to the nearest, to the even one if two are as near\n"), "{fma}");
        assert!(fma.contains("- `.ftz`: flushes a subnormal input or result to a zero\n"), "{fma}");
        assert!(fma.contains("\nfma.rn.ftz.f32(=f32, f32, f32, f32)\n"), "{fma}");
        assert!(fma.ends_with("PTX's ISA, Floating-Point Instructions: fma."), "{fma}");
        // The operands of a pair, a number known now with its limits, a mask, and the PTX and the GPU it needs.
        let shfl = of("shfl.sync.bfly.b32").unwrap_or_default();
        assert!(shfl.contains("\nshfl.sync.bfly.b32(=b32 [| =pred/_], b32, u32[0,31], u32&0x1f1f, lanes)\n"), "{shfl}");
        assert!(shfl.contains("- `.bfly`: from the lane whose index is this one's XOR the third operand\n"), "{shfl}");
        assert!(shfl.contains("`&0x1f1f` limits a number known now to the bits of the mask."), "{shfl}");
        assert!(shfl.contains("Needs PTX 6.0 and `sm_30`.\n"), "{shfl}");
        // The section of PTX's ISA of `shfl.sync`, not that of the deprecated `shfl` before it; and so of `vote.sync`.
        assert!(shfl.ends_with("PTX's ISA, Data Movement and Conversion Instructions: shfl.sync."), "{shfl}");
        let vote = of("vote.sync.ballot.b32").unwrap_or_default();
        assert!(vote.ends_with("Parallel Synchronization and Communication Instructions: vote.sync."), "{vote}");
        // An address that is read and written.
        let atom = of("atom.global.add.u32").unwrap_or_default();
        assert!(atom.contains("\natom.global.add.u32(=u32/_, +[u64], u32)\n"), "{atom}");
        assert!(atom.contains("`=` marks an operand it writes, `+` one it reads and writes."), "{atom}");
        // Registers taken together, a shape that the rule works out, and a type that is two of another.
        let mma = of("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").unwrap_or_default();
        let form = "\nmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(=(f32/_)*4, %f16x2*4, %f16x2*2, f32*4)\n";
        assert!(mma.contains(form), "{mma}");
        assert!(mma.contains("- `.m16n8k16`: a is 16 by 16, b is 16 by 8, and c and d are 16 by 8\n"), "{mma}");
        assert!(mma.contains("`T*n` is n of type T."), "{mma}");
        assert!(mma.contains("Needs PTX 7.0 and `sm_80`.\n"), "{mma}");
        assert!(mma.ends_with("PTX's ISA, Warp Level Matrix Multiply-Accumulate Instructions: mma."), "{mma}");
        // A part of a variable that a video instruction names, and what it is without one.
        let vadd = of("vadd.u32.u32.u32").unwrap_or_default();
        assert!(vadd.contains("vadd.u32.u32.u32(=u32.bh, u32.bh, u32.bh, u32)"), "{vadd}");
        assert!(vadd.contains("`.h0` or `.h1`; without it, all 32 bits are."), "{vadd}");
        assert!(vadd.contains("the result is all of the first, and the fourth is read for nothing."), "{vadd}");
        let vadd2 = of("vadd2.u32.u32.u32").unwrap_or_default();
        assert!(vadd2.contains("without it, the second operand's own, `.h10`, and the third's, `.h32`."), "{vadd2}");
        let vadd4 = of("vadd4.u32.u32.u32").unwrap_or_default();
        assert!(
            vadd4.contains("without it, the second operand's own, `.b3210`, and the third's, `.b7654`."),
            "{vadd4}"
        );
    }

    /// Forms that need other than the first say so, after the form; forms that need the least of PTX and the GPU, and
    /// instructions with no operand to write, say no more than that.
    #[test]
    fn a_doc_says_what_each_form_needs_and_what_no_form_does() {
        let cp = of("cp.async.ca.shared.global").unwrap_or_default();
        assert!(cp.contains("\ncp.async.ca.shared.global(=[u32], [u32], #4|8|16)\n"), "{cp}");
        assert!(cp.contains("(=[u32], [u32], #4|8|16, %pred/!pred)  @7.5 sm_80\n"), "{cp}");
        assert!(
            cp.contains("Needs PTX 7.0 and `sm_80`, but a form with `@` after it needs the PTX and the GPU"),
            "{cp}"
        );
        let add = of("add.s32").unwrap_or_default();
        assert!(!add.contains("Needs") && !add.contains('@'), "{add}");
        let ret = of("ret").unwrap_or_default();
        assert!(ret.contains("\nret()\n") && !ret.contains("marks") && !ret.contains("Needs"), "{ret}");
        assert!(ret.ends_with("PTX's ISA, Control Flow Instructions: ret."), "{ret}");
    }

    /// A word has the phrase of the instruction it is in, if it has one, and its own if not.
    #[test]
    fn a_word_means_what_the_instruction_it_is_in_says() {
        let has = |name: &str, line: &str| {
            let doc = of(name).unwrap_or_default();
            assert!(doc.contains(&format!("\n{line}\n")), "{doc}");
        };
        has("mul.hi.u32", "- `.hi`: the high half of the product, as many bits as an operand has");
        has("setp.hi.u32", "- `.hi`: holds if the second operand is higher than the third, as unsigned integers");
        has(
            "min.xorsign.abs.f32",
            "- `.xorsign.abs`: compares the magnitudes, and gives the result the sign of the XOR of the signs",
        );
        has("ld.global.v4.f32", "- `.v4`: reads four values at once, into registers taken together");
        has("barrier.sync.aligned", "- `.aligned`: every thread of the CTA runs this instruction, as PTX requires");
        has(
            "tex.2d.v4.f32.f32",
            "- `.v4`: four values: the four components of a texel, or one component of each of four texels",
        );
    }

    /// A name that starts with the name of a sentence of its own has that sentence, and the words after it have their
    /// phrases.
    #[test]
    fn a_name_has_the_sentence_of_the_longest_name_it_starts_with() {
        let first = |name: &str| of(name).unwrap_or_default().lines().next().unwrap_or_default().to_string();
        assert!(first("cp.async.ca.shared.global").starts_with("Copies 4, 8 or 16 bytes"));
        assert!(first("cp.async.commit_group").starts_with("Ends the group of the copies"));
        assert!(first("cp.async.wait_all").starts_with("Ends the group of the copies"));
        assert!(first("createpolicy.cvt.L2.b64").starts_with("Writes to the first operand a cache eviction policy"));
        assert!(
            of("membar.proxy.alias")
                .unwrap_or_default()
                .ends_with("Parallel Synchronization and Communication Instructions: membar.proxy.")
        );
    }

    /// A register of more bits than the type is taken for an integer or bits type, and a float must be as it is.
    #[test]
    fn a_doc_says_a_variable_of_more_bits_is_taken_for_an_integer_type_alone() {
        let wide = "a variable of more bits is taken too";
        assert!(of("ld.global.u8").unwrap_or_default().contains(wide));
        assert!(!of("ld.global.f32").unwrap_or_default().contains(wide));
    }
}
