//! A part of memory, on the GPU here, if there is one: a buffer's, and each kind of unified memory's, from an element
//! for a count. A kernel and a copy take it as they take the memory itself, but for the elements outside it, which they
//! do not reach: each test gives the memory words that differ one from another, and compares all of it after, bit for
//! bit. `twice`, written here, writes the double of each of the first `n` words of its first memory to its second, with
//! the address and the count that a launcher is lent for a part: its first element, and how many it has.

use kernite_cuda::{Cuda, Grid, Memory, MemoryMut, Module, arg};

mod common;
use common::{kinds, lent};
mod gpu;
use gpu::cuda;

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// `twice(from, to, n)`: `to[i] = 2 * from[i]` for each `i < n`, a thread each.
const PTX: &str = "
.version 7.5
.target sm_61
.address_size 64

.visible .entry twice(.param .u64 from_param, .param .u64 to_param, .param .u32 n_param)
{
	.reg .pred %p;
	.reg .u32 %n, %i, %block, %size, %thread, %v;
	.reg .u64 %from, %to, %at_from, %at_to;

	ld.param.u64 %from, [from_param];
	ld.param.u64 %to, [to_param];
	ld.param.u32 %n, [n_param];
	mov.u32 %block, %ctaid.x;
	mov.u32 %size, %ntid.x;
	mov.u32 %thread, %tid.x;
	mad.lo.u32 %i, %block, %size, %thread;
	setp.ge.u32 %p, %i, %n;
	@%p bra done;
	mad.wide.u32 %at_from, %i, 4, %from;
	mad.wide.u32 %at_to, %i, 4, %to;
	ld.global.u32 %v, [%at_from];
	shl.b32 %v, %v, 1;
	st.global.u32 [%at_to], %v;
done:
	ret;
}
";

/// How many words each memory of the tests has.
const WORDS: usize = 16;

/// What the memory a part is of holds before a kernel or a copy writes some of it: a word of its own for each element,
/// so that a word moved to another element, or left out, is seen.
fn canaries() -> Vec<u32> {
    (0..WORDS as u32).map(|i| 0xCA00_0000 + 0x101 * i).collect()
}

/// What `twice` reads: words that its doubling wraps for some of, and that none of `canaries` is a double of.
fn sources() -> Vec<u32> {
    (0..WORDS as u32).map(|i| 0x7000_0000u32.wrapping_add(0x0123_4567 * i)).collect()
}

/// What memory that held `canaries` holds after `twice` wrote the doubles of `from` to its elements from `at`.
fn after(from: &[u32], at: usize) -> Vec<u32> {
    let mut words = canaries();
    for (word, from) in words.iter_mut().skip(at).zip(from) {
        *word = from << 1;
    }
    words
}

/// The kernel `twice`, as a module of the GPU of `cuda`.
fn load(cuda: &Cuda) -> Result<Module> {
    Ok(Module::load(cuda, PTX, &["twice"], &[])?)
}

/// `twice` over `from` and `to`, as the launcher of a kernel is lent them: each at the address of its first element,
/// and the kernel over as many elements as `from` has. `to` has as many, or the test is wrong.
fn twice(module: &Module, from: &impl Memory<u32>, to: &mut impl MemoryMut<u32>) -> Result {
    let ((at_from, len), (at_to, room)) = (from.lend(module)?, to.lend(module)?);
    assert_eq!(len, room, "the two memories differ in length");
    let n = u32::try_from(len)?;
    // SAFETY: `twice` is given the addresses of two memories of this GPU's, that the test holds, of `n` elements each,
    // and writes the first `n` words of the second alone.
    unsafe { module.launch(0, Grid::new(1, 32), &[arg(&at_from), arg(&at_to), arg(&n)]) }?;
    Ok(())
}

#[test]
fn a_kernel_reads_a_part_of_a_buffer_and_writes_a_part_of_one_and_no_other_element_changes() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (from, mut to) = (cuda.upload(&sources())?, cuda.upload(&canaries())?);
    // Elements 3 to 9 of the first, to elements 5 to 11 of the second.
    twice(&module, &from.part(3..10).ok_or("part gave None")?, &mut to.part_mut(5..12).ok_or("part_mut gave None")?)?;
    assert_eq!(to.read()?, after(&sources()[3..10], 5), "elements 3 to 9 of the first, to 5 to 11");
    // Elements 3 to the last, to the first elements of a second buffer: the part reaches the end.
    let mut other = cuda.upload(&canaries())?;
    twice(&module, &from.part(3..).ok_or("part gave None")?, &mut other.part_mut(..13).ok_or("part_mut gave None")?)?;
    assert_eq!(other.read()?, after(&sources()[3..], 0), "elements 3 to 15");
    // What the kernel read, it did not change.
    assert_eq!(from.read()?, sources());
    // A part is written whole by a kernel over all of its elements, and no more.
    let mut all = cuda.upload(&canaries())?;
    twice(&module, &from.part(..).ok_or("part gave None")?, &mut all.part_mut(..).ok_or("part_mut gave None")?)?;
    assert_eq!(all.read()?, sources().iter().map(|word| word << 1).collect::<Vec<_>>(), "the whole buffer");
    // A part of no elements is lent with none, and the kernel does nothing.
    twice(&module, &from.part(16..).ok_or("part gave None")?, &mut all.part_mut(7..7).ok_or("part_mut gave None")?)?;
    assert_eq!(all.read()?, sources().iter().map(|word| word << 1).collect::<Vec<_>>(), "no element");
    Ok(())
}

/// Each kind of unified memory a kernel is lent, a part of it, to read and to write: page-locked and managed memory, a
/// slice registered for good, and a slice lent for good, where the GPU makes it. The other memory is a buffer.
#[test]
fn a_kernel_reads_and_writes_a_part_of_each_kind_of_unified_memory_and_no_other_element_changes() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let lent_for_good = lent(&cuda, WORDS).map(|memory| ("a slice lent for good", memory));
    for (what, mut memory) in kinds(&cuda, WORDS)?.into_iter().chain(lent_for_good) {
        // A part to read: elements 3 to 9 of the memory, to elements 5 to 11 of a buffer.
        memory.host_mut()?.copy_from_slice(&sources());
        let mut buffer = cuda.upload(&canaries())?;
        let part = memory.part(3..10).ok_or("part gave None")?;
        twice(&module, &part, &mut buffer.part_mut(5..12).ok_or("part_mut gave None")?)?;
        assert_eq!(buffer.read()?, after(&sources()[3..10], 5), "a part of {what} to read");
        assert_eq!(memory.host()?, sources(), "{what}, which the kernel read");
        // A part to write: elements 3 to 9 of the memory, from elements 5 to 11 of a buffer.
        memory.host_mut()?.copy_from_slice(&canaries());
        let buffer = cuda.upload(&sources())?;
        twice(
            &module,
            &buffer.part(5..12).ok_or("part gave None")?,
            &mut memory.part_mut(3..10).ok_or("part_mut gave None")?,
        )?;
        assert_eq!(memory.host()?, after(&sources()[5..12], 3), "a part of {what} to write");
        // A part to the end of the memory.
        memory.host_mut()?.copy_from_slice(&canaries());
        twice(
            &module,
            &buffer.part(..13).ok_or("part gave None")?,
            &mut memory.part_mut(3..).ok_or("part_mut gave None")?,
        )?;
        assert_eq!(memory.host()?, after(&sources()[..13], 3), "a part of {what} to its end");
    }
    Ok(())
}

/// A part ends at the end of its memory, or before it, and starts no later than it ends: a range past either end, or
/// back to front, is `None`, as `at` gives `None` for an element past the end. One that ends at the end, or takes no
/// element, is not.
#[test]
#[expect(clippy::reversed_empty_ranges, reason = "a range back to front is what the test refuses")]
fn a_part_past_the_end_is_none_and_one_at_the_end_is_not() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let mut buffer = cuda.alloc::<u32>(WORDS)?;
    assert!(buffer.part(5..3).is_none() && buffer.part(0..17).is_none() && buffer.part_mut(17..).is_none());
    assert!(
        buffer.part(..=16).is_none() && buffer.part(usize::MAX..).is_none() && buffer.part(..=usize::MAX).is_none()
    );
    for range in [0..16, 16..16, 15..16, 3..3] {
        assert_eq!(buffer.part(range.clone()).map(|part| part.read().map(|words| words.len())), Some(Ok(range.len())));
        assert!(buffer.part_mut(range).is_some());
    }
    let lent_for_good = lent(&cuda, WORDS).map(|memory| ("a slice lent for good", memory));
    for (what, mut memory) in kinds(&cuda, WORDS)?.into_iter().chain(lent_for_good) {
        assert!(
            memory.part(5..3).is_none() && memory.part(0..17).is_none() && memory.part_mut(17..).is_none(),
            "{what}"
        );
        assert!(memory.part(..=16).is_none() && memory.part(usize::MAX..).is_none(), "{what}");
        assert!(memory.part(0..16).is_some() && memory.part(16..).is_some() && memory.part_mut(..).is_some(), "{what}");
    }
    Ok(())
}

/// The host writes a part of a buffer, and reads one, and `read_into` fills a slice of as many elements as the part
/// has: the elements of the part, and no other element of the buffer, are written or read, and a slice of another
/// length is an error, in the words of the whole buffer's, that says it is a part.
#[test]
fn the_host_writes_and_reads_a_part_of_a_buffer_and_no_other_element_changes() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let mut buffer = cuda.upload(&canaries())?;
    let words = [0x11u32, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77];
    buffer.part_mut(3..10).ok_or("part_mut gave None")?.write(&words)?;
    let mut want = canaries();
    want[3..10].copy_from_slice(&words);
    assert_eq!(buffer.read()?, want, "a part written");
    assert_eq!(buffer.part(3..10).ok_or("part gave None")?.read()?, words, "a part read");
    assert_eq!(buffer.part(2..4).ok_or("part gave None")?.read()?, [canaries()[2], words[0]], "a part across an end");
    let mut out = [0u32; 7];
    buffer.part(8..15).ok_or("part gave None")?.read_into(&mut out)?;
    assert_eq!(out, want[8..15], "a part read into a slice");
    // The part is the memory of the host's calls for as long as it lasts, and the buffer is whole again after.
    let mut part = buffer.part_mut(0..4).ok_or("part_mut gave None")?;
    part.write(&[1, 2, 3, 4])?;
    assert_eq!(part.read()?, [1, 2, 3, 4], "a part mutable, read");
    want[..4].copy_from_slice(&[1, 2, 3, 4]);
    assert_eq!(buffer.read()?, want, "the elements of a part, and no other, were written");
    // A slice of another length is refused, and nothing is written or read.
    let said = |what: &str| format!("a part of a buffer of 4 elements {what}");
    let mut part = buffer.part_mut(0..4).ok_or("part_mut gave None")?;
    let e = part.write(&[9; 3]).err().ok_or("write gave Ok")?;
    assert_eq!(e.to_string(), format!("cuMemcpyHtoD: {}", said("written with 3")));
    let e = part.write(&[9; 5]).err().ok_or("write gave Ok")?;
    assert_eq!(e.to_string(), format!("cuMemcpyHtoD: {}", said("written with 5")));
    let mut small = [7u32; 3];
    let e = part.read_into(&mut small).err().ok_or("read_into gave Ok")?;
    assert_eq!(e.to_string(), format!("cuMemcpyDtoH: {}", said("read into 3")));
    assert_eq!(small, [7; 3], "a refused read wrote");
    assert_eq!(part.read()?, [1, 2, 3, 4], "a refused write wrote");
    Ok(())
}
