//! Copies and zeroings, on a GPU, if there is one: each goes to the current stream of its GPU and returns before the
//! GPU has done it. The kernels of `gpu` show it. A call that only queues its work is made after a kernel that waits
//! for a flag, which the host writes once the call has returned: the kernel cannot be done when the call returns,
//! however slow the host is.

use std::time::{Duration, Instant};

use kernite_cuda::{Buffer, Cuda, Error, Module, Ptr, Stream, Unified, arg};

mod common;
use common::managed_or_skip;
mod gpu;
use gpu::{COPIES, SEEN, SETS, TIMED_OUT, WAITS, cuda, hold, launch, load, peek, poke, run};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// How long `sets` keeps a stream busy, and a kernel waits for a word where the copy that writes it can run no sooner:
/// a tenth of a second, which is more than the host takes to make the next call. A call that read memory then, and not
/// after the kernel before it, would read it before the kernel wrote it.
const A_TENTH: u64 = 100_000_000;

/// How long `waits` waits for a word, in the tests where the word is written by a copy beside it.
const A_SECOND: u64 = 1_000_000_000;

/// 64 MiB of words.
const WORDS: usize = 16 << 20;

/// `len` words from a generator, from `seed`: the bits of floats of every kind, NaNs too, which a copy must not change.
fn words(len: usize, seed: u32) -> Vec<u32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x
        })
        .collect()
}

/// Calls `work` on `stream`, right after a kernel that a flag holds and that writes the page-locked word `out` at its
/// end, and says how long the call took. The call is to return before the kernel is done, and the kernel cannot be
/// done: the host writes the flag once the call has returned, so a slow host does not fail it. `what` is the call, for
/// the messages. The stream is synced when this returns.
fn queued_behind_a_kernel(
    cuda: &Cuda,
    module: &Module,
    stream: &Stream,
    what: &str,
    work: impl FnOnce() -> Result<(), Error>,
) -> Result<Duration> {
    // The words are made before the kernel is queued: a call that makes memory may wait of itself.
    let (flag, out) = (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    let (took, seen) = run(stream, || -> Result<(Duration, u32)> {
        hold(module, flag.ptr(), out.ptr())?;
        let started = Instant::now();
        work()?;
        Ok((started.elapsed(), peek(&out)))
    })?;
    assert_eq!(seen, 0, "{what} waited for the kernel before it, which was not done when it returned");
    poke(&flag, 1);
    stream.sync()?;
    assert_eq!(peek(&out), SEEN, "the kernel before {what} did not see the flag");
    Ok(took)
}

/// A kind of memory the host reaches that a copy takes, for the tests that go through each: managed, page-locked, and
/// registered for good, `len` elements of each. A kind the GPU does not take is left out, after a line that says so. A
/// slice that `lend` lent is not one: a copy refuses it, and `lent` makes one.
fn kinds(cuda: &Cuda, len: usize) -> Result<Vec<(&'static str, Unified<'static, u32>)>> {
    let mut kinds = vec![("page-locked memory", cuda.pinned::<u32>(len)?)];
    if let Some(managed) = managed_or_skip(cuda.managed::<u32>(len)) {
        kinds.push(("managed memory", managed));
    }
    match cuda.register(Vec::leak(vec![0u32; len])) {
        Ok(registered) => kinds.push(("a slice registered for good", registered)),
        Err(e) => eprintln!("a GPU that page-locks no slice: none is registered. {e}"),
    }
    Ok(kinds)
}

/// `len` elements of a slice lent for good, if the GPU reaches pageable memory.
fn lent(cuda: &Cuda, len: usize) -> Option<Unified<'static, u32>> {
    let lent = cuda.lend(Vec::leak(vec![0u32; len]));
    lent.inspect_err(|e| eprintln!("a GPU that reaches no pageable memory: no slice is lent. {e}")).ok()
}

#[test]
fn a_copy_of_64_mib_to_and_from_each_kind_of_memory_is_queued_and_bit_for_bit() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let stream = cuda.stream()?;
    let (first, second) = (words(WORDS, 3), words(WORDS, 4));
    let mut buffer = cuda.alloc::<u32>(WORDS)?;
    for (what, mut memory) in kinds(&cuda, WORDS)? {
        memory.host_mut()?.copy_from_slice(&first);
        let from = format!("a copy from {what}");
        let took = queued_behind_a_kernel(&cuda, &module, &stream, &from, || cuda.copy(&mut buffer, &memory))?;
        eprintln!("a copy of 64 MiB from {what} returned after {took:?}");
        assert!(buffer.read()? == first, "the buffer does not hold what was copied from {what}");
        buffer.write(&second)?;
        let to = format!("a copy to {what}");
        let took = queued_behind_a_kernel(&cuda, &module, &stream, &to, || cuda.copy(&mut memory, &buffer))?;
        eprintln!("a copy of 64 MiB to {what} returned after {took:?}");
        assert!(memory.host()? == second, "{what} does not hold what was copied to it");
    }
    // From a buffer to a buffer.
    let mut other = cuda.alloc::<u32>(WORDS)?;
    buffer.write(&first)?;
    let between = "a copy between buffers";
    let took = queued_behind_a_kernel(&cuda, &module, &stream, between, || cuda.copy(&mut other, &buffer))?;
    eprintln!("a copy of 64 MiB between buffers returned after {took:?}");
    assert!(other.read()? == first, "the second buffer does not hold what was copied to it");
    Ok(())
}

/// `copy`, made on the current stream, which is the stream of a `run` where `on` has one and the null stream where not,
/// after a kernel that writes 1 to the word `from` a tenth of a second after it starts, and before one that copies the
/// word `to`, which `copy` writes, to a word of page-locked memory at once. That word is what this gives: 1 if the copy
/// read `from` after the first kernel wrote it and wrote `to` before the second kernel read it, 0 if not.
fn in_order(
    cuda: &Cuda,
    module: &Module,
    on: Option<&Stream>,
    from: Ptr<u32>,
    to: Ptr<u32>,
    copy: impl FnOnce() -> Result<(), Error>,
) -> Result<u32> {
    let out = cuda.pinned::<u32>(1)?;
    let work = || -> Result {
        launch(module, SETS, &[arg(&from), arg(&A_TENTH)])?;
        copy()?;
        launch(module, COPIES, &[arg(&to), arg(&out.ptr())])
    };
    match on {
        Some(stream) => run(stream, work)?,
        None => work()?,
    }
    cuda.sync()?;
    Ok(peek(&out))
}

/// Says that `found`, which `in_order` gave for `copy_is`, a copy on `stream_is`, is 1, and that `moved`, the four
/// words the copy wrote, are the one the first kernel wrote and three zeros.
fn assert_in_order(found: u32, moved: &[u32], copy_is: &str, stream_is: &str) {
    assert_eq!(found, 1, "{copy_is}, on {stream_is}, did not run after the kernel before it and before the one after");
    assert_eq!(moved, [1, 0, 0, 0], "the words that {copy_is}, on {stream_is}, wrote");
}

#[test]
fn a_copy_runs_after_the_kernel_before_it_and_before_the_kernel_after_it_for_each_kind_of_memory() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let stream = cuda.stream()?;
    // On the null stream and on a stream of its own: from each kind of memory to a buffer, from a buffer to each kind,
    // and from a buffer to a buffer. The first kernel writes the first of the four words that the copy moves.
    for (on, stream_is) in [(None, "the null stream"), (Some(&stream), "a stream")] {
        for (what, mut memory) in kinds(&cuda, 4)? {
            let mut buffer = cuda.alloc::<u32>(4)?;
            let found = in_order(&cuda, &module, on, memory.ptr(), buffer.ptr(), || cuda.copy(&mut buffer, &memory))?;
            assert_in_order(found, &buffer.read()?, &format!("a copy from {what}"), stream_is);

            memory.host_mut()?.fill(0);
            let source = cuda.alloc::<u32>(4)?;
            let found = in_order(&cuda, &module, on, source.ptr(), memory.ptr(), || cuda.copy(&mut memory, &source))?;
            assert_in_order(found, memory.host()?, &format!("a copy to {what}"), stream_is);
        }
        let (from, mut to) = (cuda.alloc::<u32>(4)?, cuda.alloc::<u32>(4)?);
        let found = in_order(&cuda, &module, on, from.ptr(), to.ptr(), || cuda.copy(&mut to, &from))?;
        assert_in_order(found, &to.read()?, "a copy between buffers", stream_is);
    }
    Ok(())
}

/// What a kernel finds that waits up to `wait` nanoseconds for the word `flag`, on stream `a`, while `write` is called
/// after it: `SEEN` if it saw the word not 0, and `TIMED_OUT` if not.
fn what_a_kernel_sees(
    cuda: &Cuda,
    module: &Module,
    a: &Stream,
    flag: Ptr<u32>,
    wait: u64,
    write: impl FnOnce() -> Result<(), Error>,
) -> Result<u32> {
    let out = cuda.pinned::<u32>(1)?;
    run(a, || launch(module, WAITS, &[arg(&flag), arg(&out.ptr()), arg(&wait)]))?;
    write()?;
    cuda.sync()?;
    Ok(peek(&out))
}

#[test]
fn a_copy_on_one_stream_runs_beside_a_kernel_on_another_that_waits_for_what_it_writes() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    let staged = cuda.pinned::<u32>(1)?;
    poke(&staged, 1);
    let mut flag = cuda.upload(&[0u32])?;
    let at = flag.ptr();
    // The null stream waits for the kernel on `a`, as `a` waits for it, so a copy that goes there cannot run before the
    // kernel ends, however long it waits for the element: a tenth of a second shows it as a second does.
    let seen = what_a_kernel_sees(&cuda, &module, &a, at, A_TENTH, || cuda.copy(&mut flag, &staged))?;
    assert_eq!(seen, TIMED_OUT, "a copy on the null stream ran beside a kernel on a stream");
    // A stream does not wait for another, so the copy on `b` writes the element while the kernel spins on `a`: at once,
    // and the second is spent only where it did not.
    flag.write(&[0])?;
    let seen = what_a_kernel_sees(&cuda, &module, &a, at, A_SECOND, || run(&b, || cuda.copy(&mut flag, &staged)))?;
    assert_eq!(seen, SEEN, "the copy on a stream did not run beside the kernel on another");
    Ok(())
}

#[test]
fn memory_of_another_length_or_gpu_is_not_copied_and_the_error_says_which() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let (mut long, mut short) = (cuda.upload(&[1u32, 2, 3, 4])?, cuda.upload(&[9u32, 9])?);
    let e = cuda.copy(&mut short, &long).err().ok_or("copy gave Ok")?;
    assert_eq!(e.to_string(), "cuMemcpyAsync: memory of 2 elements copied from memory of 4");
    let e = cuda.copy(&mut long, &short).err().ok_or("copy gave Ok")?;
    assert_eq!(e.to_string(), "cuMemcpyAsync: memory of 4 elements copied from memory of 2");
    cuda.sync()?;
    assert_eq!((long.read()?, short.read()?), (vec![1, 2, 3, 4], vec![9, 9]), "a copy that was refused copied");
    // No elements are nothing to copy, whatever the memory is.
    let (mut none, also_none) = (cuda.alloc::<u32>(0)?, cuda.alloc::<u32>(0)?);
    cuda.copy(&mut none, &also_none)?;
    cuda.sync()?;

    let count = Cuda::count()?;
    if count < 2 {
        eprintln!("two GPUs: skipped, the driver shows {count}; a copy to memory of another GPU is not shown");
        return Ok(());
    }
    let other = Cuda::on(1)?;
    let (mut there, mut here) = (other.upload(&[5u32, 6, 7, 8])?, cuda.upload(&[1u32, 2, 3, 4])?);
    let said = "cuMemcpyAsync: the memory is on another GPU than the copy is made on";
    let e = cuda.copy(&mut here, &there).err().ok_or("copy gave Ok")?;
    assert_eq!(e.to_string(), said, "from memory of another GPU");
    let e = cuda.copy(&mut there, &here).err().ok_or("copy gave Ok")?;
    assert_eq!(e.to_string(), said, "to memory of another GPU");
    cuda.sync()?;
    other.sync()?;
    assert_eq!((there.read()?, here.read()?), (vec![5, 6, 7, 8], vec![1, 2, 3, 4]), "a copy that was refused copied");
    Ok(())
}

/// On a stream, a kernel that a flag holds back, then one that writes 1 to the first of four words at `first`, and
/// then a zeroing by `zero`, which returns while the first is held. Once the flag is written and the stream synced,
/// the words the caller reads are the zeroing's, if it ran after the kernel that wrote the first, and wrote all four.
fn zeros_after_a_kernel(
    cuda: &Cuda,
    module: &Module,
    what: &str,
    first: Ptr<u32>,
    zero: impl FnOnce() -> Result<(), Error>,
) -> Result {
    let (stream, flag, out) = (cuda.stream()?, cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    run(&stream, || -> Result {
        hold(module, flag.ptr(), out.ptr())?;
        launch(module, SETS, &[arg(&first), arg(&0u64)])?;
        zero()?;
        assert_eq!(peek(&out), 0, "zeroing {what} waited for the kernel before it, which the flag held");
        Ok(())
    })?;
    poke(&flag, 1);
    stream.sync()?;
    assert_eq!(peek(&out), SEEN, "the kernel before the zeroing did not see the flag");
    Ok(())
}

/// A module of no kernels, with four words of memory, 7 each.
const CELLS: &str = "
.version 7.5
.target sm_61
.address_size 64

.visible .global .align 4 .u32 cells[4] = {7, 7, 7, 7};
";

#[test]
fn a_zeroing_in_a_run_runs_after_the_kernel_there_that_writes_it_and_returns_before_it_is_done() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let zeros = [0u32; 4];

    let mut buffer = cuda.upload(&[7u32; 4])?;
    zeros_after_a_kernel(&cuda, &module, "a buffer", buffer.ptr(), || buffer.zero())?;
    assert_eq!(buffer.read()?, zeros, "a buffer is not zeros after the kernel that wrote it and the zeroing");

    // A zeroing is queued for each kind of memory, and for a slice that `lend` lent, whose copies are refused.
    let lent_slice = lent(&cuda, 4).map(|lent| ("a slice lent for good", lent));
    for (what, mut memory) in kinds(&cuda, 4)?.into_iter().chain(lent_slice) {
        memory.host_mut()?.fill(7);
        zeros_after_a_kernel(&cuda, &module, what, memory.ptr(), || memory.zero())?;
        assert_eq!(memory.host()?, zeros, "{what} is not zeros after the kernel that wrote it and the zeroing");
    }

    let cells = Module::load(&cuda, CELLS, &[], &[("cells", None)])?;
    let global = cells.global::<u32>(0)?;
    assert_eq!(global.read()?, [7; 4]);
    zeros_after_a_kernel(&cuda, &module, "a module's memory", global.ptr(), || global.zero())?;
    assert_eq!(global.read()?, zeros, "a module's memory is not zeros after the kernel that wrote it and the zeroing");
    Ok(())
}

#[test]
fn a_zeroing_on_one_stream_runs_beside_a_kernel_on_another_which_the_host_sees_in_page_locked_memory() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    let (flag, out) = (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    let mut target = cuda.pinned::<u32>(1)?;
    // The kernel on `a` waits for the flag, which the host writes after it has seen the zero. The zeroing on `b` runs
    // beside it, so the host sees the zero while the kernel is still spinning on `a`, and `out` is not written.
    poke(&target, 7);
    run(&a, || hold(&module, flag.ptr(), out.ptr()))?;
    run(&b, || target.zero())?;
    let started = Instant::now();
    while peek(&target) != 0 && started.elapsed() < Duration::from_secs(2) {
        std::hint::spin_loop();
    }
    assert_eq!(peek(&target), 0, "the zeroing on a stream did not run beside the kernel on another");
    assert_eq!(peek(&out), 0, "the kernel on the other stream was done before the host saw the zero");
    poke(&flag, 1);
    cuda.sync()?;
    assert_eq!(peek(&out), SEEN);
    Ok(())
}

/// What the host's calls are made on: a kernel that waits for `flag` and then writes `out`, and behind it a copy of
/// the page-locked words, 7 each, to the buffer, which held 9 each, a zeroing of the buffer and a zeroing of the
/// page-locked words, all queued on a stream, and returned from while the kernel is held. A call that waited for the
/// kernel alone would find the buffer at 9, or at 7, and the page-locked words at 7: not zeros.
struct Queued {
    flag: Unified<'static, u32>,
    out: Unified<'static, u32>,
    buffer: Buffer<u32>,
    staged: Unified<'static, u32>,
}

/// Queues the work of `Queued` on `stream`.
fn queue_behind_a_kernel(cuda: &Cuda, module: &Module, stream: &Stream) -> Result<Queued> {
    // The memory is made and filled before the kernel is queued: a call that makes memory may wait of itself.
    let (flag, out) = (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    let (mut buffer, mut staged) = (cuda.upload(&[9u32; 4])?, cuda.pinned::<u32>(4)?);
    staged.host_mut()?.fill(7);
    run(stream, || -> Result {
        hold(module, flag.ptr(), out.ptr())?;
        cuda.copy(&mut buffer, &staged)?;
        buffer.zero()?;
        staged.zero()?;
        assert_eq!(peek(&out), 0, "a call that queues work waited for the kernel before it, which the flag held");
        Ok(())
    })?;
    Ok(Queued { flag, out, buffer, staged })
}

/// Writes the flag of `queued`, from a stream that the work of `queued`'s does not wait for, a tenth of a second from
/// now: while the host's call that is made next waits for the kernel that the flag holds.
fn release(module: &Module, queued: &Queued, on: &Stream) -> Result {
    run(on, || launch(module, SETS, &[arg(&queued.flag.ptr()), arg(&A_TENTH)]))
}

/// A call of the host on the memory that a copy and a zeroing are queued on, and the words it read, if it reads any.
type HostCall = fn(&mut Buffer<u32>, &mut Unified<'static, u32>, &Cuda) -> Result<Vec<u32>>;

/// The host's calls that reach memory, and wait for a kernel before they do: `read`, `write`, `host`, `host_mut` and
/// `Cuda::sync`, each with the words it reads where the copy and the zeroings are done, and the words the buffer holds
/// after it and a sync. `write` reads none, and the buffer holds its words: a call that returned before the zeroing of
/// the buffer was done would find them zeros after the sync.
const HOST_CALLS: [(&str, HostCall, &[u32], [u32; 4]); 5] = [
    ("read", |buffer, _, _| Ok(buffer.read()?), &[0; 4], [0; 4]),
    (
        "write",
        |buffer, _, _| {
            buffer.write(&[1; 4])?;
            Ok(vec![])
        },
        &[],
        [1; 4],
    ),
    ("host", |_, staged, _| Ok(staged.host()?.to_vec()), &[0; 4], [0; 4]),
    ("host_mut", |_, staged, _| Ok(staged.host_mut()?.to_vec()), &[0; 4], [0; 4]),
    (
        "sync",
        |_, _, cuda| {
            cuda.sync()?;
            Ok(vec![])
        },
        &[],
        [0; 4],
    ),
];

#[test]
fn the_host_waits_for_a_copy_and_a_zeroing_as_it_waits_for_a_kernel() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (stream, other) = (cuda.stream()?, cuda.stream()?);
    for (name, call, read, left) in HOST_CALLS {
        let queued = queue_behind_a_kernel(&cuda, &module, &stream)?;
        release(&module, &queued, &other)?;
        let Queued { out, mut buffer, mut staged, .. } = queued;
        let got = call(&mut buffer, &mut staged, &cuda)?;
        assert_eq!(peek(&out), SEEN, "{name} returned before the kernel before the copy and the zeroing was done");
        assert_eq!(got, read, "the words {name} read were not those that the copy and the zeroing leave");
        cuda.sync()?;
        assert_eq!(buffer.read()?, left, "the buffer after {name} and a sync");
        assert_eq!(staged.host()?, [0; 4], "the page-locked words after {name} and a sync");
    }
    // Dropping the memory waits for them too, whichever of the two it is.
    for dropped in ["a buffer", "page-locked memory"] {
        let queued = queue_behind_a_kernel(&cuda, &module, &stream)?;
        release(&module, &queued, &other)?;
        let Queued { out, buffer, staged, .. } = queued;
        if dropped == "a buffer" {
            drop(buffer);
        } else {
            drop(staged);
        }
        let said = "returned before the kernel before the copy and the zeroing was done";
        assert_eq!(peek(&out), SEEN, "dropping {dropped} {said}");
    }
    Ok(())
}

/// A slice that `lend` lent, for good, where a copy takes memory, as the source and as the target: it is refused, and
/// nothing is copied, since the driver reads pageable memory as the call is made, before the work that the stream was
/// given earlier has run, and writes it after the call waits for the stream.
#[test]
fn a_slice_that_lend_lent_is_not_copied_from_or_to_and_the_error_says_what_to_do() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let Some(mut lent) = lent(&cuda, 4) else { return Ok(()) };
    let mut buffer = cuda.upload(&[5u32; 4])?;
    let said = "cuMemcpyAsync: a slice that `lend` lent is not copied from or to, as the driver reads it when the call \
                is made, and writes it after the call waits for the stream: `register` page-locks the slice where it \
                is, and its copies are queued in order";
    let e = cuda.copy(&mut buffer, &lent).err().ok_or("copy gave Ok")?;
    assert_eq!(e.to_string(), said, "from a slice lent");
    let e = cuda.copy(&mut lent, &buffer).err().ok_or("copy gave Ok")?;
    assert_eq!(e.to_string(), said, "to a slice lent");
    cuda.sync()?;
    assert_eq!((buffer.read()?, lent.host()?), (vec![5; 4], [0; 4].as_slice()), "a copy that was refused copied");
    // The slice is as it was: the host writes it, and reads what it wrote.
    lent.host_mut()?.fill(2);
    assert_eq!(lent.host()?, [2; 4]);
    Ok(())
}

#[test]
fn a_copy_from_a_view_of_a_buffer_copies_the_elements_the_view_covers() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let buffer = cuda.upload(&[1u32, 2, 3, 4])?;
    let view = buffer.tail(1).ok_or("tail gave None")?;
    let mut rest = cuda.alloc::<u32>(3)?;
    cuda.copy(&mut rest, &view)?;
    assert_eq!(rest.read()?, [2, 3, 4]);
    // A view is the elements it covers and no more, whatever the buffer it is of holds.
    let mut all = cuda.alloc::<u32>(4)?;
    let e = cuda.copy(&mut all, &view).err().ok_or("copy gave Ok")?;
    assert_eq!(e.to_string(), "cuMemcpyAsync: memory of 4 elements copied from memory of 3");
    Ok(())
}
