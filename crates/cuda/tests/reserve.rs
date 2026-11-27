//! A buffer that reserves addresses for more elements than it has, on the GPU here, if there is one: it grows and
//! shrinks at the same addresses, moves the granules past a split to the host's memory and back, and a kernel, a copy
//! and `read` take it as any buffer. Two kernels: `add_one`,
//! written here, adds 1 to each of the first `n` words of its memory, and `waits`, of `gpu`, spins until the host
//! writes a flag, giving up only in a hang, and then writes whether it saw the flag. A kernel that sees a flag the host
//! writes after a call returned shows that the call did not wait for it, however long either took, as other programs'
//! kernels on the GPU change the times.
//!
//! The tests run one at a time: a kernel that a flag holds keeps back the work of every test on the null stream. What
//! the driver is asked for, a granule at a time, is the fake driver's to show, in the tests of `kernite-cuda` itself:
//! other programs take and give back the GPU's memory, so its free bytes do not show it.

use std::sync::MutexGuard;

use kernite_cuda::{Cuda, Grid, Module, Ptr, arg};

mod common;
use common::NO_VMM;
mod gpu;
use gpu::{SEEN, cuda, hold, load, poke};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// `add_one(x, n)`: `x[i] += 1` for each `i < n`, a thread each.
const PTX: &str = "
.version 7.5
.target sm_61
.address_size 64

.visible .entry add_one(.param .u64 x_param, .param .u32 n_param)
{
	.reg .pred %p;
	.reg .u32 %n, %i, %block, %size, %thread, %v;
	.reg .u64 %x, %at;

	ld.param.u64 %x, [x_param];
	ld.param.u32 %n, [n_param];
	mov.u32 %block, %ctaid.x;
	mov.u32 %size, %ntid.x;
	mov.u32 %thread, %tid.x;
	mad.lo.u32 %i, %block, %size, %thread;
	setp.ge.u32 %p, %i, %n;
	@%p bra done;
	mad.wide.u32 %at, %i, 4, %x;
	ld.global.u32 %v, [%at];
	add.u32 %v, %v, 1;
	st.global.u32 [%at], %v;
done:
	ret;
}
";
/// The one kernel of `PTX`.
const ADD_ONE: usize = 0;

/// What the driver backs with memory at a time, in `u32`s: 2 MiB, on the GPU here.
const WORDS: usize = (2 << 20) / 4;

/// The elements of the buffers that grow far: 40 MB of words, which is 20 granules, and more of a smaller granule.
const FAR: usize = 10_000_000;

/// What the reservations are for: 1 << 28 words, which is 1 GiB of addresses and none of the GPU's memory.
const CAPACITY: usize = 1 << 28;

/// GUIDE.md's words for a GPU that cannot map the host's memory at addresses it reserves.
const NO_HOST: &str = "cuMemCreate: the GPU cannot map the host's memory at addresses it reserves: its \
                       CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED is 0";

/// What a test of the GPU is given: the lock, the GPU and the kernel `add_one`. A test binds the lock first, so that it
/// is dropped last, after the memory of the test is freed.
type Setup = (MutexGuard<'static, ()>, Cuda, Module);

/// What `Setup` has; or `None` where there is no GPU, or the GPU cannot map memory at addresses it reserves, after it
/// says so.
fn setup() -> Result<Option<Setup>> {
    let Some((cuda, alone)) = cuda() else { return Ok(None) };
    if let Err(e) = cuda.reserve::<u32>(0) {
        assert_eq!(e.to_string(), NO_VMM);
        eprintln!("the GPU cannot: skipped. {e}");
        return Ok(None);
    }
    let module = Module::load(&cuda, PTX, &["add_one"], &[])?;
    Ok(Some((alone, cuda, module)))
}

/// Pseudo-random words, the same for the same seed.
fn words(n: usize, seed: u32) -> Vec<u32> {
    let mut s = seed;
    Vec::from_fn(n, |_| {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    })
}

/// Launches `add_one` over the `n` words at `x`.
fn add_one(module: &Module, x: Ptr<u32>, n: u32) -> Result {
    // SAFETY: `add_one` takes a pointer and a `u32`, and writes only the first `n` words there, which the caller's
    // buffer has. The caller holds no slice of it, and no other kernel runs.
    unsafe { module.launch(ADD_ONE, Grid::new(n.div_ceil(256), 256), &[arg(&x), arg(&n)]) }?;
    Ok(())
}

#[test]
fn a_buffer_grows_at_its_addresses_and_holds_what_it_had_and_zeros_past_it() -> Result {
    let Some((_alone, cuda, _)) = setup()? else { return Ok(()) };
    let mut cache = cuda.reserve::<f32>(CAPACITY)?;
    // It has no element, and an address all the same, which no growth changes.
    assert_eq!((cache.len(), cache.is_empty(), cache.read()?), (0, true, vec![]));
    let at = cache.ptr().addr();
    // The place past the last element has an address too, as in any buffer, and no place past that has.
    assert_eq!(cache.at(0).map(|p| p.addr()), Some(at));
    assert!(cache.at(1).is_none());
    cache.grow(0)?;
    cache.grow(1000)?;
    assert_eq!((cache.len(), cache.ptr().addr()), (1000, at));
    assert_eq!(cache.read()?, vec![0.0; 1000]);
    let written: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
    cache.write(&written)?;
    cache.grow(FAR)?;
    assert_eq!((cache.len(), cache.ptr().addr()), (FAR, at));
    let all = cache.read()?;
    assert_eq!(all[..1000], written[..], "what it held changed");
    assert!(all[1000..].iter().all(|x| *x == 0.0), "the elements it gained are not zero");
    // Pointers to its elements are good as far as it goes.
    assert_eq!(cache.at(FAR - 1).map(|p| p.addr()), Some(at + 4 * (FAR as u64 - 1)));
    assert_eq!(cache.at(FAR).map(|p| p.addr()), Some(at + 4 * FAR as u64));
    assert!(cache.at(FAR + 1).is_none());
    Ok(())
}

#[test]
fn a_kernel_a_copy_and_a_read_take_a_buffer_that_grew_as_they_take_any_buffer() -> Result {
    let Some((_alone, cuda, module)) = setup()? else { return Ok(()) };
    let mut cache = cuda.reserve::<u32>(CAPACITY)?;
    cache.grow(1000)?;
    let words = words(1000, 7);
    cache.write(&words)?;
    cache.grow(FAR)?;
    // A kernel over each element, those it had and those it gained, in all the granules.
    add_one(&module, cache.ptr(), FAR as u32)?;
    let mut sums: Vec<u32> = words.iter().map(|w| w.wrapping_add(1)).collect();
    sums.resize(FAR, 1);
    assert_eq!(cache.read()?, sums);
    // The part from an element on, as any buffer gives it.
    let tail = cache.at(FAR - 1000).ok_or("at gave None")?;
    add_one(&module, tail, 1000)?;
    let mut again = cache.read()?;
    assert!(again[FAR - 1000..].iter().all(|w| *w == 2), "the kernel did not reach the part");
    assert_eq!(again[..FAR - 1000], sums[..FAR - 1000]);
    // Copies, from the buffer and to it.
    let mut copy = cuda.alloc::<u32>(FAR)?;
    copy.copy_from(&cache)?;
    assert_eq!(copy.read()?, again);
    again.reverse();
    copy.write(&again)?;
    cache.copy_from(&copy)?;
    assert_eq!(cache.read()?, again);
    // A buffer of the same kind as another: the copy goes from one reservation to another.
    let mut other = cuda.reserve::<u32>(CAPACITY)?;
    other.grow(FAR)?;
    other.copy_from(&cache)?;
    assert_eq!(other.read()?, again);
    assert_ne!(other.ptr().addr(), cache.ptr().addr());
    Ok(())
}

#[test]
fn a_buffer_backs_a_granule_at_a_time_and_is_refused_a_growth_past_its_reservation() -> Result {
    let Some((_alone, cuda, module)) = setup()? else { return Ok(()) };
    let mut cache = cuda.reserve::<u32>(CAPACITY)?;
    // One element has a granule of memory, and the GPU may use all of it.
    cache.grow(1)?;
    cache.grow(WORDS)?;
    add_one(&module, cache.ptr(), WORDS as u32)?;
    assert!(cache.read()?.iter().all(|w| *w == 1), "the GPU did not reach the whole granule");
    // The element past the granule has the next, and the buffer is as long as it was told.
    cache.grow(WORDS + 1)?;
    assert_eq!(cache.len(), WORDS + 1);
    assert_eq!(cache.read()?[WORDS], 0);
    // The reservation is past: the error says what was reserved and what was asked, and the buffer is as it was.
    let words = format!("a buffer reserved for {CAPACITY} elements grown to {}", CAPACITY + 1);
    assert_eq!(cache.grow(CAPACITY + 1).err().ok_or("grow gave Ok")?.to_string(), format!("cuMemMap: {words}"));
    assert_eq!(cache.len(), WORDS + 1);
    // The reservation's own size is not past it, one more is, and so is a size below what the buffer has.
    let mut small = cuda.reserve::<u32>(WORDS + 5)?;
    small.grow(WORDS + 5)?;
    assert_eq!(small.len(), WORDS + 5);
    let past = small.grow(WORDS + 6).err().ok_or("grow gave Ok")?;
    assert_eq!(
        past.to_string(),
        format!("cuMemMap: a buffer reserved for {} elements grown to {}", WORDS + 5, WORDS + 6)
    );
    let less = small.grow(10).err().ok_or("grow gave Ok")?;
    assert_eq!(less.to_string(), format!("cuMemMap: a buffer of {} elements grown to 10", WORDS + 5));
    assert_eq!(small.len(), WORDS + 5);
    let mut none = cuda.reserve::<u32>(0)?;
    let one = none.grow(1).err().ok_or("grow gave Ok")?;
    assert_eq!(one.to_string(), "cuMemMap: a buffer reserved for 0 elements grown to 1");
    Ok(())
}

#[test]
fn a_buffer_that_alloc_or_upload_made_neither_grows_nor_shrinks() -> Result {
    let Some((_alone, cuda, _)) = setup()? else { return Ok(()) };
    let (mut allocated, mut uploaded) = (cuda.alloc::<u32>(4)?, cuda.upload(&[1u32, 2, 3, 4])?);
    for buffer in [&mut allocated, &mut uploaded] {
        let words = "a buffer of 4 elements was not made by reserve, so it does not";
        let grown = buffer.grow(8).err().ok_or("grow gave Ok")?;
        assert_eq!(grown.to_string(), format!("cuMemMap: {words} grow"));
        // SAFETY: nothing is queued or running, so no work reaches what it would give back.
        let shrunk = unsafe { buffer.shrink(2) }.err().ok_or("shrink gave Ok")?;
        assert_eq!(shrunk.to_string(), format!("cuMemUnmap: {words} shrink"));
        assert_eq!(buffer.len(), 4);
    }
    assert_eq!((allocated.read()?, uploaded.read()?), (vec![0; 4], vec![1, 2, 3, 4]));
    Ok(())
}

#[test]
fn a_shrink_leaves_what_the_buffer_keeps_as_it_was_and_the_elements_it_grows_into_again_are_zero() -> Result {
    let Some((_alone, cuda, _)) = setup()? else { return Ok(()) };
    let mut cache = cuda.reserve::<u32>(CAPACITY)?;
    cache.grow(FAR)?;
    let (at, words) = (cache.ptr().addr(), words(FAR, 3));
    cache.write(&words)?;
    // The buffer keeps the first granule and ten words of the second, and gives back the 18 granules after them.
    let kept = WORDS + 10;
    // SAFETY: no work is queued or running: `write` came after the zeros of the growth, and returned when its words
    // were on the GPU; nothing else reaches the buffer.
    unsafe { cache.shrink(kept) }?;
    assert_eq!((cache.len(), cache.ptr().addr()), (kept, at));
    assert_eq!(cache.read()?, words[..kept], "what the buffer keeps changed");
    // Down to 1000 elements, the second granule goes back as well: no element of it is left.
    // SAFETY: as above.
    unsafe { cache.shrink(1000) }?;
    assert_eq!((cache.len(), cache.ptr().addr()), (1000, at));
    assert_eq!(cache.read()?, words[..1000], "what the buffer keeps changed");
    // The elements it grows into again are zero, those of the granule it kept too, which held words.
    cache.grow(FAR)?;
    let all = cache.read()?;
    assert_eq!(all[..1000], words[..1000]);
    assert!(all[1000..].iter().all(|w| *w == 0), "the elements it grew into again are not zero");
    assert_eq!(cache.ptr().addr(), at);
    // And it does it again, the whole way down and up.
    // SAFETY: as above.
    unsafe { cache.shrink(0) }?;
    assert_eq!((cache.len(), cache.read()?), (0, vec![]));
    cache.grow(FAR)?;
    assert!(cache.read()?.iter().all(|w| *w == 0));
    // A shrink past what it has says so, and the buffer is as it was.
    // SAFETY: as above.
    let past = unsafe { cache.shrink(FAR + 1) }.err().ok_or("shrink gave Ok")?;
    assert_eq!(past.to_string(), format!("cuMemUnmap: a buffer of {FAR} elements shrunk to {}", FAR + 1));
    assert_eq!(cache.len(), FAR);
    Ok(())
}

#[test]
fn a_shrink_does_not_wait_for_a_kernel_that_works_on_the_elements_the_buffer_keeps() -> Result {
    let Some((_alone, cuda, _)) = setup()? else { return Ok(()) };
    let module = load(&cuda)?;
    let mut cache = cuda.reserve::<u32>(CAPACITY)?;
    cache.grow(FAR)?;
    // The growth queued the zeros of the elements it gained, 40 MB, and the shrink below gives back granules of them:
    // it does not wait for them, so the sync does.
    cuda.sync()?;
    let mut flag = cuda.pinned::<u32>(1)?;
    // `waits` spins on the flag, and writes to the first element of the buffer what it saw: the kernel alone writes it.
    hold(&module, flag.ptr(), cache.ptr())?;
    // SAFETY: the kernel reaches the first word, which the buffer keeps, and nothing else reaches the elements given
    // back: the zeros of the growth were waited for.
    unsafe { cache.shrink(1000) }?;
    // The kernel still spins, as the flag is not written: a shrink that waited for it would have returned only when
    // the kernel gave up, in a hang, and the kernel would have written `NOT_SEEN`.
    poke(&mut flag, 1);
    cuda.sync()?;
    assert_eq!(cache.read()?[0], SEEN, "the kernel did not see the flag: the shrink waited for it");
    Ok(())
}

#[test]
fn a_growth_returns_before_the_gpu_has_written_its_zeros() -> Result {
    let Some((_alone, cuda, _)) = setup()? else { return Ok(()) };
    let module = load(&cuda)?;
    let mut cache = cuda.reserve::<u32>(CAPACITY)?;
    let (mut flag, out) = (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    // `waits` holds the null stream until the host writes the flag, and the zeros of the growth are queued behind it.
    hold(&module, flag.ptr(), out.ptr())?;
    cache.grow(FAR)?;
    // The host has not written the flag, so the kernel still spins: a growth that waited for the work before its zeros
    // would return only when the kernel gives up, in a hang, and writes `NOT_SEEN`.
    poke(&mut flag, 1);
    cuda.sync()?;
    assert_eq!(out.host()?, [SEEN], "the kernel did not see the flag: the growth waited for the kernel before it");
    assert!(cache.read()?.iter().all(|w| *w == 0), "the zeros of the growth were not written");
    Ok(())
}

#[test]
fn two_buffers_reserved_have_addresses_of_their_own_and_hold_what_each_was_given() -> Result {
    let Some((_alone, cuda, _)) = setup()? else { return Ok(()) };
    // 4 MiB of addresses each: two granules.
    let (mut a, mut b) = (cuda.reserve::<u32>(2 * WORDS)?, cuda.reserve::<u32>(2 * WORDS)?);
    let (a_at, b_at, span) = (a.ptr().addr(), b.ptr().addr(), 4 << 20);
    assert!(a_at + span <= b_at || b_at + span <= a_at, "the reservations overlap: {a_at:#x} and {b_at:#x}");
    a.grow(2 * WORDS)?;
    b.grow(WORDS + 5)?;
    let (a_words, b_words) = (words(2 * WORDS, 1), words(WORDS + 5, 2));
    a.write(&a_words)?;
    b.write(&b_words)?;
    assert_eq!((a.read()?, b.read()?), (a_words, b_words));
    Ok(())
}

#[test]
fn a_split_moves_granules_with_what_they_hold_and_a_kernel_and_the_host_reach_those_in_the_hosts_memory() -> Result {
    let Some((_alone, cuda, module)) = setup()? else { return Ok(()) };
    let mut cache = cuda.reserve::<u32>(4 * WORDS)?;
    cache.grow(3 * WORDS)?;
    let (at, words) = (cache.ptr().addr(), words(3 * WORDS, 5));
    cache.write(&words)?;
    // The split is that of whole granules: the third granule goes to the host's memory, with what it holds.
    // SAFETY: no work is queued or running: `write` came after the zeros of the growth, and returned when its words
    // were on the GPU; nothing else reaches the buffer.
    if let Err(e) = unsafe { cache.move_split(WORDS + 1) } {
        assert_eq!(e.to_string(), NO_HOST);
        eprintln!("the GPU cannot: skipped. {e}");
        return Ok(());
    }
    assert_eq!((cache.split(), cache.ptr().addr()), (2 * WORDS, at));
    assert_eq!(cache.read()?, words, "what the buffer holds changed as it moved");
    // A kernel reaches the granule in the host's memory, and the host does too, at the same addresses.
    add_one(&module, cache.ptr(), 3 * WORDS as u32)?;
    cuda.sync()?;
    let sums: Vec<u32> = words.iter().map(|w| w.wrapping_add(1)).collect();
    assert_eq!(cache.read()?, sums);
    let host = std::ptr::with_exposed_provenance::<u32>((at + 8 * WORDS as u64) as usize);
    // SAFETY: the granule from element 2 * WORDS on is page-locked memory of the host, which the host may read and
    // write at the buffer's addresses, and no work reaches it: the sync waited for the kernel.
    let held = unsafe { std::slice::from_raw_parts(host, WORDS) };
    assert_eq!(held, &sums[2 * WORDS..], "the host does not read what the kernel wrote");
    // A growth past the split backs a granule in the host's memory too, and zeros it.
    cache.grow(4 * WORDS)?;
    let all = cache.read()?;
    assert_eq!((&all[..3 * WORDS], all[3 * WORDS..].iter().all(|w| *w == 0)), (&sums[..], true));
    // And back to the GPU, each granule with what it holds.
    // SAFETY: no work is queued or running: `read` waited for the zeros of the growth, and nothing else reaches the
    // buffer.
    unsafe { cache.move_split(4 * WORDS) }?;
    assert_eq!((cache.split(), cache.read()?), (4 * WORDS, all));
    Ok(())
}
