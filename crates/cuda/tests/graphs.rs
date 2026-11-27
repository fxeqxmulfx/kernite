//! Graphs, on a GPU, if there is one: the work of a stream, captured, and run as often as it is launched. Two kernels
//! of `gpu` change the words they are given, `adds_one` and `doubles`, so that the words show what ran, how often and
//! in what order: `(x + 1) * 2` is not `x * 2 + 1`. `waits` and `sets` show which stream a launch went to, and that the
//! host was not kept waiting, as in `streams.rs`. What a graph saves of the launches one by one is a probe's to say,
//! not a test's. The tests go one at a time, as `gpu::cuda` sees to.

use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;

use kernite_cuda::{Buffer, Cuda, Error, Event, Graph, Grid, Module, Ptr, Stream, Unified, arg};

mod common;
use common::{NO_PAGEABLE, NO_VMM, managed_or_skip, or_skip, register_or_skip};
mod gpu;
use gpu::{
    ADDS_ONE, AT_ONCE, COPIES, DOUBLES, HANG, LOOKS, NOT_SEEN, SEEN, SETS, WAITS, cuda, hold, load, peek, poke,
    release_while, run, until, watch,
};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// The words that the tests start from.
const START: [u32; 4] = [0, 1, 2, 3];

/// A buffer of the words `values`.
fn filled(cuda: &Cuda, values: &[u32]) -> Result<Buffer<u32>> {
    let mut buffer = cuda.alloc::<u32>(values.len())?;
    buffer.write(values)?;
    Ok(buffer)
}

/// A word of page-locked memory, zero, which a kernel and the host both reach.
fn word(cuda: &Cuda) -> Result<Unified<'static, u32>> {
    Ok(cuda.pinned::<u32>(1)?)
}

/// Launches `kernel`, `ADDS_ONE` or `DOUBLES`, over the first `n` words at `words`, one thread each, on the current
/// stream.
fn change(module: &Module, kernel: usize, words: Ptr<u32>, n: u32) -> Result<(), Error> {
    // SAFETY: the kernel is given what its `.param`s are: the address of `n` words of the GPU's memory that the test
    // holds, and `n` as a `u32`. It writes those words alone, and no test holds a slice of them: the host reads them
    // with `Buffer::read`, once the work is done.
    unsafe { module.launch(kernel, Grid::new(1, n), &[arg(&words), arg(&n)]) }
}

/// The work that most tests capture: each of the four words at `words` adds one, then doubles.
fn adds_then_doubles(module: &Module, words: Ptr<u32>) -> Result<(), Error> {
    change(module, ADDS_ONE, words, 4)?;
    change(module, DOUBLES, words, 4)
}

/// What the words of `START` are after `times` runs of `adds_then_doubles`.
fn after(times: u32) -> [u32; 4] {
    (0..times).fold(START, |words, _| words.map(|x| (x + 1) * 2))
}

/// Launches `kernel`, `SETS`, `WAITS` or `COPIES`, over one thread, on the current stream.
fn launch(module: &Module, kernel: usize, args: &[*const c_void]) -> Result<(), Error> {
    // SAFETY: the kernel is given what its `.param`s are: addresses of words of page-locked memory that the test
    // holds, and numbers, as `u64`; and it writes only the word that an address it is given is of.
    unsafe { module.launch(kernel, Grid::new(1, 1), args) }
}

/// Launches `graph` on the current stream.
fn go(graph: &Graph) -> Result<(), Error> {
    // SAFETY: each graph of these tests reaches memory and a module that its test holds until the work is done, which
    // the test waits for before it drops them. The work of another stream that may run beside it shares one word with
    // it at most, as `gpu::run` says, and no test holds a slice of what a kernel of a graph writes.
    unsafe { graph.launch() }
}

#[test]
fn a_graph_runs_the_captured_work_as_often_as_it_is_launched_and_the_capture_runs_none_of_it() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let mut words = filled(&cuda, &START)?;
    // The stream is dropped when the capture is over: the graph does not hold it.
    let graph = cuda.stream()?.capture(|| adds_then_doubles(&module, words.ptr()))?;
    cuda.sync()?;
    assert_eq!(words.read()?, START, "the capture ran the work");
    for times in 1..=3 {
        go(&graph)?;
        cuda.sync()?;
        assert_eq!(words.read()?, after(times), "after {times} launches");
    }
    // The graph holds the address of the words and not what they hold: the host writes new values, and the next launch
    // uses them.
    words.write(&[10, 20, 30, 40])?;
    go(&graph)?;
    cuda.sync()?;
    assert_eq!(words.read()?, [22, 42, 62, 82], "a launch did not work on the words the host wrote");
    Ok(())
}

#[test]
fn a_graph_holds_the_arguments_of_its_work_as_they_were_at_the_capture_numbers_too() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let words = filled(&cuda, &START)?;
    let at = words.ptr();
    // `adds_one` adds one to the first `count` words, one thread each, of the four that the grid has.
    let mut count = 4u32;
    let graph = cuda.stream()?.capture(|| {
        // SAFETY: `adds_one` is given what its `.param`s are, the address of four words of the GPU's memory that the
        // test holds and a `u32`, and it writes those words alone, which no test holds a slice of.
        unsafe { module.launch(ADDS_ONE, Grid::new(1, 4), &[arg(&at), arg(&count)]) }
    })?;
    // The count changes after the capture: a launch of the graph that read it now would work on one word.
    count = 1;
    go(&graph)?;
    cuda.sync()?;
    assert_eq!(words.read()?, [1, 2, 3, 4], "the graph did not work on the count that it had at the capture");
    // The same kernel launched one by one is given the count that the host has now, and works on one word.
    change(&module, ADDS_ONE, at, count)?;
    cuda.sync()?;
    assert_eq!(words.read()?, [2, 2, 3, 4], "a launch one by one did not work on the count it was given");
    Ok(())
}

#[test]
fn a_graph_launches_on_the_current_stream_after_the_work_before_it_and_before_the_work_after_it() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let mut words = filled(&cuda, &START)?;
    let at = words.ptr();
    let doubles = a.capture(|| change(&module, DOUBLES, at, 4))?;
    // Adds one, doubles with the graph, adds one: `((x + 1) * 2) + 1`, on a stream and on the null stream.
    let sequence = || -> Result<(), Error> {
        change(&module, ADDS_ONE, at, 4)?;
        go(&doubles)?;
        change(&module, ADDS_ONE, at, 4)
    };
    let want = START.map(|x| (x + 1) * 2 + 1);
    run(&a, sequence)?;
    cuda.sync()?;
    assert_eq!(words.read()?, want, "on a stream, the graph did not run between the kernels given before and after it");
    words.write(&START)?;
    sequence()?;
    cuda.sync()?;
    assert_eq!(words.read()?, want, "on the null stream, the graph did not run between them");
    // A launch of a graph runs after the launch before it, with no wait for the first to be done.
    let twice = a.capture(|| adds_then_doubles(&module, at))?;
    words.write(&START)?;
    run(&a, || -> Result<(), Error> {
        go(&twice)?;
        go(&twice)
    })?;
    cuda.sync()?;
    assert_eq!(words.read()?, after(2), "the second launch of a graph did not run after the first");
    Ok(())
}

#[test]
fn a_graph_launches_on_the_stream_run_made_current_and_on_no_other() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    let (mut flag, mut out, mut launched) = (word(&cuda)?, word(&cuda)?, word(&cuda)?);
    let sets = a.capture(|| launch(&module, SETS, &[arg(&flag.ptr()), arg(&AT_ONCE)]))?;
    // `waits` is queued on `a`, and the graph, whose `sets` writes the flag at once, is launched where `place` puts it:
    // `waits` sees the flag only if the graph ran beside it, on a stream that is neither `a`, where it is queued behind
    // `waits`, nor the null stream, which `a` blocks. It looks for the flag once the host has launched the graph:
    // `HANG` times at most where it must see it, and `LOOKS` where it must not.
    let mut race = |expected: u32, place: &dyn Fn() -> Result<(), Error>, what: &str| -> Result {
        let looks = if expected == SEEN { HANG } else { LOOKS };
        poke(&mut flag, 0);
        poke(&mut out, 0);
        poke(&mut launched, 0);
        run(&a, || watch(&module, launched.ptr(), flag.ptr(), out.ptr(), looks))?;
        place()?;
        poke(&mut launched, 1);
        cuda.sync()?;
        assert_eq!(peek(&mut out), expected, "{what}");
        Ok(())
    };
    // The stream a graph was captured on is no part of it: this one was captured on `a`, and `b` runs it.
    race(SEEN, &|| run(&b, || go(&sets)), "launched in a run of another stream")?;
    race(NOT_SEEN, &|| go(&sets), "launched with no run, on the null stream")?;
    race(NOT_SEEN, &|| run(&a, || go(&sets)), "launched in a run of the stream that `waits` is queued on")?;
    Ok(())
}

#[test]
fn a_capture_waits_for_nothing_and_the_work_given_before_it_runs_as_it_did_and_is_no_part_of_the_graph() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let (mut flag, mut out) = (word(&cuda)?, word(&cuda)?);
    let words = filled(&cuda, &START)?;
    let at = words.ptr();
    // `waits` holds the stream until the host writes the flag, and a kernel that adds one is queued behind it.
    run(&a, || -> Result {
        hold(&module, flag.ptr(), out.ptr())?;
        Ok(change(&module, ADDS_ONE, at, 4)?)
    })?;
    // The capture returns while both are queued. Had it waited for the stream, the host would not have written the flag
    // until `waits` gave up, in a hang, and `waits` would have written `NOT_SEEN`.
    let graph = a.capture(|| change(&module, DOUBLES, at, 4))?;
    assert_eq!(peek(&mut out), 0, "the capture waited for the work given before it");
    poke(&mut flag, 1);
    a.sync()?;
    assert_eq!(peek(&mut out), SEEN, "`waits` did not see the flag: the capture waited for it");
    assert_eq!(
        words.read()?,
        START.map(|x| x + 1),
        "the work given before the capture did not run, or the capture ran"
    );
    for times in 1..=2 {
        run(&a, || go(&graph))?;
        a.sync()?;
        assert_eq!(words.read()?, START.map(|x| (x + 1) << times), "after {times} launches");
    }
    Ok(())
}

#[test]
fn a_copy_and_a_zeroing_are_captured_and_run_at_each_launch_on_what_the_memory_holds_then() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let a = cuda.stream()?;
    let (mut from, mut to, mut cleared) = (filled(&cuda, &START)?, filled(&cuda, &[9; 4])?, filled(&cuda, &[7; 4])?);
    let graph = a.capture(|| {
        cuda.copy(&mut to, &from)?;
        cleared.zero()
    })?;
    cuda.sync()?;
    assert_eq!((to.read()?, cleared.read()?), (vec![9; 4], vec![7; 4]), "the capture ran a copy or a zeroing");
    go(&graph)?;
    cuda.sync()?;
    assert_eq!((to.read()?, cleared.read()?), (START.to_vec(), vec![0; 4]), "the graph did not copy and zero");
    // The memory is written again, and the next launch works on that.
    from.write(&[5, 6, 7, 8])?;
    cleared.write(&[1; 4])?;
    go(&graph)?;
    cuda.sync()?;
    assert_eq!((to.read()?, cleared.read()?), (vec![5, 6, 7, 8], vec![0; 4]), "the second launch");
    Ok(())
}

/// What the driver says of a call that waits for the GPU, or for the host, in a capture.
const UNSUPPORTED: &str =
    "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED (900): operation not permitted when stream is capturing";
/// Of a call that copies through the null stream, which a stream that is capturing blocks.
const IMPLICIT: &str = "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT (906): \
                        operation would make the legacy stream depend on a capturing blocking stream";
/// Of a wait for work that is not in the capture.
const ISOLATION: &str =
    "CUDA_ERROR_STREAM_CAPTURE_ISOLATION (905): dependency created on uncaptured work in another stream";
/// Of a query of an event that the capture recorded.
const CAPTURED: &str =
    "CUDA_ERROR_CAPTURED_EVENT (907): operation not permitted on an event last recorded in a capturing stream";
/// What `capture` says where the closure ignored a refusal: the capture could not be recorded, and it is over.
const INVALIDATED: &str = "cuStreamEndCapture: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (901): \
                           operation failed due to a previous error during capture";

/// A call that the driver cannot capture.
#[derive(Clone, Copy)]
enum Call {
    Read,
    Write,
    StreamSync,
    CudaSync,
    Alloc,
    WaitForAnEventRecordedBefore,
    DoneOfAnEventRecordedIn,
    SyncOfAnEventRecordedIn,
}

/// The calls that cannot be captured, each with the call of the driver that the error names, and what it says.
const REFUSED: [(Call, &str, &str); 8] = [
    (Call::Read, "cuMemcpyDtoH", IMPLICIT),
    (Call::Write, "cuMemcpyHtoD", IMPLICIT),
    (Call::StreamSync, "cuStreamSynchronize", UNSUPPORTED),
    (Call::CudaSync, "cuCtxSynchronize", UNSUPPORTED),
    (Call::Alloc, "cuMemAlloc", UNSUPPORTED),
    (Call::WaitForAnEventRecordedBefore, "cuStreamWaitEvent", ISOLATION),
    (Call::DoneOfAnEventRecordedIn, "cuEventQuery", CAPTURED),
    (Call::SyncOfAnEventRecordedIn, "cuEventSynchronize", CAPTURED),
];

/// What a call is made on: the GPU, a stream, a buffer, and an event recorded before the capture and one recorded in
/// it.
struct Rig<'a> {
    cuda: &'a Cuda,
    stream: &'a Stream,
    buffer: Buffer<u32>,
    before: Event,
    inside: Event,
}

impl Rig<'_> {
    /// Makes `call`, and gives the error of the driver where it refuses it.
    fn make(&mut self, call: Call) -> Result<(), Error> {
        match call {
            Call::Read => self.buffer.read().map(drop),
            Call::Write => self.buffer.write(&START),
            Call::StreamSync => self.stream.sync(),
            Call::CudaSync => self.cuda.sync(),
            Call::Alloc => self.cuda.alloc::<u32>(1).map(drop),
            Call::WaitForAnEventRecordedBefore => self.before.wait(),
            Call::DoneOfAnEventRecordedIn => {
                self.inside.record()?;
                self.inside.done().map(drop)
            }
            Call::SyncOfAnEventRecordedIn => {
                self.inside.record()?;
                self.inside.sync()
            }
        }
    }
}

#[test]
fn a_call_that_waits_or_makes_memory_or_waits_for_work_outside_is_refused_and_the_capture_is_over() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let words = filled(&cuda, &START)?;
    let at = words.ptr();
    let mut rig =
        Rig { cuda: &cuda, stream: &a, buffer: filled(&cuda, &START)?, before: cuda.event()?, inside: cuda.event()? };
    run(&a, || rig.before.record())?;
    // Each call is refused while the stream captures. The error of `capture` is the call's where the closure returns
    // it, and the end of the capture's where the closure ignores it and returns `Ok`.
    for (call, name, said) in REFUSED {
        for ignored in [false, true] {
            let made = a.capture(|| {
                let refused = rig.make(call);
                if ignored { Ok(()) } else { refused }
            });
            let want = if ignored { INVALIDATED.to_string() } else { format!("{name}: {said}") };
            let e = made.err().ok_or("capture gave Ok")?;
            assert_eq!(e.to_string(), want, "{name}, ignored: {ignored}");
            // The capture is over: the stream takes work, and waits, and captures again.
            run(&a, || change(&module, ADDS_ONE, at, 4))?;
            a.sync()?;
            a.capture(|| change(&module, ADDS_ONE, at, 4))?;
        }
    }
    assert_eq!(words.read()?, START.map(|x| x + 2 * REFUSED.len() as u32), "the stream took no work after a capture");
    Ok(())
}

#[test]
fn a_call_of_another_thread_that_makes_or_frees_memory_is_not_refused_and_the_capture_goes_on() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let words = filled(&cuda, &START)?;
    let at = words.ptr();
    let (made, seen) = mpsc::channel();
    let (freed, gone) = mpsc::channel();
    let other = thread::spawn(move || -> Result {
        let cuda = Cuda::new()?;
        let b = cuda.stream()?;
        // SAFETY: the work of `b` is the zeroing of the memory, which no other work reaches.
        let memory = unsafe { b.run(|| cuda.alloc::<u32>(1)) }?;
        made.send(())?;
        // This thread captures nothing, and the driver refuses another thread's capture's own calls alone.
        drop(memory);
        freed.send(())?;
        Ok(())
    });
    let (mut made_it, mut freed_it) = (None, None);
    let graph = a.capture(|| {
        change(&module, ADDS_ONE, at, 4)?;
        // The other thread has made its memory, and freed it, when these receive.
        made_it = Some(seen.recv());
        freed_it = Some(gone.recv());
        Ok(())
    })?;
    other.join().map_err(|_| "the other thread panicked")??;
    assert_eq!(made_it, Some(Ok(())), "the other thread made no memory while this one captured");
    assert_eq!(freed_it, Some(Ok(())), "the other thread freed no memory while this one captured");
    go(&graph)?;
    cuda.sync()?;
    assert_eq!(words.read()?, START.map(|x| x + 1), "the capture did not hold what its own thread enqueued");
    Ok(())
}

/// A capture on `a`, during which another thread calls `call`, which waits for the stream that captures: the driver
/// refuses it with `said`, the capture is lost, and `a` takes work again.
fn lost_to(a: &Stream, name: &str, said: &str, call: impl Fn(&Cuda) -> Result<(), Error> + Sync) -> Result {
    let mut other = None;
    let made = a.capture(|| {
        other = Some(thread::scope(|scope| {
            scope
                .spawn(|| {
                    Cuda::new().map_err(|e| e.to_string()).and_then(|cuda| call(&cuda).map_err(|e| e.to_string()))
                })
                .join()
        }));
        Ok(())
    });
    assert_eq!(made.err().ok_or("capture gave Ok")?.to_string(), INVALIDATED, "{name}");
    let refused = other.ok_or("the closure did not run")?.map_err(|_| "the other thread panicked")?;
    assert_eq!(refused, Err(format!("{name}: {said}")), "the call of the other thread");
    a.sync()?;
    Ok(())
}

#[test]
fn a_call_of_another_thread_that_waits_for_the_gpu_or_uses_the_null_stream_loses_the_capture() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let a = cuda.stream()?;
    lost_to(&a, "cuCtxSynchronize", UNSUPPORTED, |cuda| cuda.sync())?;
    // Memory is zeroed on the current stream, which is the null stream in a thread that made no other current.
    lost_to(&a, "cuMemsetD8Async", IMPLICIT, |cuda| cuda.alloc::<u32>(1).map(drop))
}

/// What a drop gives back, other than a module: a buffer, or unified memory.
enum Memory<'a> {
    Buffer(Buffer<u32>),
    Unified(Unified<'a, u32>),
}

impl Memory<'_> {
    /// The address of its first word.
    fn at(&self) -> Ptr<u32> {
        match self {
            Memory::Buffer(buffer) => buffer.ptr(),
            Memory::Unified(unified) => unified.ptr(),
        }
    }
}

/// What each kind of memory is called.
const KINDS: [&str; 6] =
    ["a buffer", "managed memory", "page-locked memory", "a registered slice", "a lent slice", "a reserved buffer"];

/// The memory of `kind`, of words that are 0; or `None` where the GPU cannot make it, after it says so, and the test
/// goes on without it. A slice of `host` is lent or registered.
fn made<'a>(cuda: &Cuda, kind: &str, host: &'a mut [u32]) -> Result<Option<Memory<'a>>> {
    Ok(match kind {
        "a buffer" => Some(Memory::Buffer(cuda.alloc(4)?)),
        "managed memory" => managed_or_skip(cuda.managed(4)).map(Memory::Unified),
        "page-locked memory" => Some(Memory::Unified(cuda.pinned(4)?)),
        "a registered slice" => register_or_skip(cuda, host)?.map(Memory::Unified),
        "a lent slice" => or_skip(cuda.lend(host), NO_PAGEABLE).map(Memory::Unified),
        _ => match or_skip(cuda.reserve::<u32>(4), NO_VMM) {
            Some(mut buffer) => {
                buffer.grow(4)?;
                Some(Memory::Buffer(buffer))
            }
            None => None,
        },
    })
}

/// Launches on the null stream, the current stream of a thread that has made no other current, `waits` of `module`,
/// which holds the GPU until the host writes the page-locked word `flag`, and then writes `SEEN` to the page-locked
/// word `out`; and `copies` behind it, which writes the word at `at` over itself, and so reaches what a drop gives
/// back.
fn holds(module: &Module, flag: &Unified<u32>, at: Ptr<u32>, out: &Unified<u32>) -> Result<(), Error> {
    launch(module, WAITS, &[arg(&flag.ptr()), arg(&out.ptr()), arg(&HANG)])?;
    launch(module, COPIES, &[arg(&at), arg(&at)])
}

/// What the tests of a drop in a capture have: the GPU, the kernels, a stream to capture on, and words that the work of
/// that stream changes, to show that it works again after a capture is lost.
struct Dropping {
    cuda: Cuda,
    module: Module,
    a: Stream,
    words: Buffer<u32>,
    /// How many times the work on `words` ran.
    runs: u32,
}

impl Dropping {
    fn new(cuda: Cuda) -> Result<Dropping> {
        let (module, a, words) = (load(&cuda)?, cuda.stream()?, filled(&cuda, &START)?);
        Ok(Dropping { cuda, module, a, words, runs: 0 })
    }

    /// Drops `memory` in the closure of a capture on the stream, while the kernels that `holds` launched with `flag`
    /// and `out` still hold the GPU, which reach what `memory` gives back. The host writes the flag while it is in the
    /// drop, and the drop waits for the kernels, though the driver refuses the wait while the capture is on: the
    /// page-locked word `out` is the kernel's right after the drop. The capture is lost, and its error is the end's.
    /// The work the closure enqueues after the drop does not run. The stream takes work after it, and captures again.
    fn drops<M>(&mut self, name: &str, flag: &mut Unified<u32>, out: &mut Unified<u32>, memory: M) -> Result {
        let (mut released, mut after, at) = (None, None, self.words.ptr());
        let captured = self.a.capture(|| {
            released = Some(release_while(flag, || drop(memory)));
            after = Some(peek(out));
            // What the closure enqueues after the drop is recorded, and does not run.
            change(&self.module, ADDS_ONE, at, 4)
        });
        assert_eq!(captured.err().ok_or("capture gave Ok")?.to_string(), INVALIDATED, "{name}: the capture");
        released.ok_or("the closure did not run")??;
        assert_eq!(after, Some(SEEN), "{name}: the word right after the drop, which waits for the kernel");
        self.cuda.sync()?;
        assert_eq!(self.words.read()?, START.map(|x| x + self.runs), "{name}: the words after the recorded work");
        run(&self.a, || change(&self.module, ADDS_ONE, at, 4))?;
        let graph = self.a.capture(|| change(&self.module, ADDS_ONE, at, 4))?;
        go(&graph)?;
        self.cuda.sync()?;
        self.runs += 2;
        assert_eq!(self.words.read()?, START.map(|x| x + self.runs), "{name}: the stream after the drop");
        Ok(())
    }

    /// Makes `kind` of memory in another thread, which holds the GPU with kernels that reach it, and drops it while
    /// this thread captures on the stream. The drop waits for the GPU as a whole, which the driver refuses while any
    /// capture is on: its first call is refused, and loses this thread's capture, which the closure of the capture
    /// sees as a launch refused, and ends. The drop tries again until the capture is over, which `capture` ends as soon
    /// as its closure does. Then it waits for the kernels, which the other thread lets go while it is in the drop: the
    /// page-locked word the first writes is the kernel's right after the drop.
    fn drops_in_another_thread(&self, kind: &str) -> Result {
        let (ready, ready_rx) = mpsc::channel();
        let (go, go_rx) = mpsc::channel::<()>();
        let dropped = &AtomicBool::new(false);
        thread::scope(|scope| -> Result {
            let other = scope.spawn(move || -> Result<Option<u32>> {
                let cuda = Cuda::new()?;
                let (module, mut flag, mut out, mut host) = (load(&cuda)?, word(&cuda)?, word(&cuda)?, vec![0u32; 4]);
                let Some(memory) = made(&cuda, kind, &mut host)? else {
                    ready.send(false)?;
                    return Ok(None);
                };
                holds(&module, &flag, memory.at(), &out)?;
                ready.send(true)?;
                go_rx.recv()?;
                let released = release_while(&mut flag, || drop(memory));
                dropped.store(true, Ordering::Release);
                released?;
                Ok(Some(peek(&mut out)))
            });
            if !ready_rx.recv()? {
                other.join().map_err(|_| "the other thread panicked")??;
                return Ok(());
            }
            let (mut sent, mut lost, at) = (None, None, self.words.ptr());
            let captured = self.a.capture(|| {
                sent = Some(go.send(()));
                // The launches are recorded until the other thread's first call is refused, and refused from then on. A
                // drop that returned with the capture still on has not lost it, which the end then says.
                lost = Some(until("the other thread's call to lose the capture", || {
                    Ok(change(&self.module, ADDS_ONE, at, 4).is_err() || dropped.load(Ordering::Acquire))
                }));
                Ok(())
            });
            assert_eq!(captured.err().ok_or("capture gave Ok")?.to_string(), INVALIDATED, "{kind}: the capture");
            sent.ok_or("the closure did not run")??;
            lost.ok_or("the closure did not run")??;
            let after = other.join().map_err(|_| "the other thread panicked")??;
            assert_eq!(after, Some(SEEN), "{kind}: the word right after the drop of the other thread");
            Ok(())
        })
    }
}

#[test]
fn a_drop_in_a_capture_waits_for_the_kernel_for_each_kind_of_memory_and_for_a_module_and_loses_the_capture() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let mut rig = Dropping::new(cuda)?;
    // That each drop gives its memory back is the fake driver's to show, in the tests of `kernite-cuda` itself: other
    // programs take and give back the GPU's memory, so its free bytes do not show it.
    for kind in KINDS {
        let mut host = vec![0u32; 4];
        let Some(memory) = made(&rig.cuda, kind, &mut host)? else { continue };
        let (mut flag, mut out) = (word(&rig.cuda)?, word(&rig.cuda)?);
        holds(&rig.module, &flag, memory.at(), &out)?;
        rig.drops(kind, &mut flag, &mut out, memory)?;
    }
    // The module's kernels hold the GPU, and the drop unloads the module under them.
    let (own, spare) = (load(&rig.cuda)?, word(&rig.cuda)?);
    let (mut flag, mut out) = (word(&rig.cuda)?, word(&rig.cuda)?);
    holds(&own, &flag, spare.ptr(), &out)?;
    rig.drops("a module", &mut flag, &mut out, own)
}

#[test]
fn a_drop_in_another_thread_waits_out_the_capture_of_this_one_and_then_the_kernel_and_loses_the_capture() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let rig = Dropping::new(cuda)?;
    for kind in ["a lent slice", "a reserved buffer"] {
        rig.drops_in_another_thread(kind)?;
    }
    Ok(())
}

#[test]
fn the_error_of_the_closure_is_the_error_of_the_capture_and_a_panic_goes_on_and_ends_the_capture() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let words = filled(&cuda, &START)?;
    let at = words.ptr();
    // An error of the closure's own, which the driver did not make: two buffers of other lengths are not copied.
    let (mut one, two) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(2)?);
    let own = cuda.copy(&mut one, &two).err().ok_or("copy gave Ok")?.to_string();
    let made = a.capture(|| {
        change(&module, ADDS_ONE, at, 4)?;
        cuda.copy(&mut one, &two)
    });
    assert_eq!(made.err().ok_or("capture gave Ok")?.to_string(), own, "the error of the capture is not the closure's");
    // The capture is over: the stream takes work, waits, and captures again, and none of the work recorded ran.
    let doubles = || -> Result {
        run(&a, || change(&module, DOUBLES, at, 4))?;
        a.sync()?;
        a.capture(|| change(&module, DOUBLES, at, 4))?;
        Ok(())
    };
    doubles()?;
    assert_eq!(words.read()?, START.map(|x| x * 2), "the work of the capture that failed ran, or the stream took none");
    // A panic goes on, through `capture`, and ends the capture too.
    let panicked = panic::catch_unwind(AssertUnwindSafe(|| {
        a.capture(|| -> Result<(), Error> {
            change(&module, ADDS_ONE, at, 4)?;
            panic::resume_unwind(Box::new("a panic"))
        })
    }));
    assert!(panicked.is_err(), "the panic did not go on");
    doubles()?;
    assert_eq!(
        words.read()?,
        START.map(|x| x * 4),
        "the work of the capture that panicked ran, or the stream took none"
    );
    Ok(())
}

#[test]
fn a_capture_on_a_stream_that_is_capturing_is_refused_and_the_capture_that_is_on_goes_on() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let words = filled(&cuda, &START)?;
    let at = words.ptr();
    let mut refused = None;
    let graph = a.capture(|| {
        change(&module, ADDS_ONE, at, 4)?;
        refused = a.capture(|| Ok(())).err().map(|e| e.to_string());
        change(&module, DOUBLES, at, 4)
    })?;
    let said =
        "cuStreamBeginCapture: CUDA_ERROR_ILLEGAL_STATE (401): the operation cannot be performed in the present state";
    assert_eq!(refused.as_deref(), Some(said), "the capture in the capture");
    go(&graph)?;
    cuda.sync()?;
    assert_eq!(words.read()?, after(1), "the capture that was on did not go on, or the refused one ended it");
    Ok(())
}

#[test]
fn a_graph_that_is_dropped_while_it_runs_runs_to_its_end_and_the_drop_does_not_wait() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let (mut flag, mut out) = (word(&cuda)?, word(&cuda)?);
    let waits = [arg(&flag.ptr()), arg(&out.ptr()), arg(&HANG)];
    let graph = a.capture(|| launch(&module, WAITS, &waits))?;
    run(&a, || go(&graph))?;
    // The graph's `waits` holds until the host writes the flag, which it does after the drop: the drop did not wait.
    drop(graph);
    poke(&mut flag, 1);
    a.sync()?;
    assert_eq!(peek(&mut out), SEEN, "the work of a graph that was dropped did not run to its end, or the drop waited");
    Ok(())
}

#[test]
fn a_graph_of_1100_launches_does_the_work_of_the_1100_launches_one_by_one() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let words = filled(&cuda, &[0])?;
    let at = words.ptr();
    const LAUNCHES: usize = 1100;
    let one_by_one = || (0..LAUNCHES).try_for_each(|_| change(&module, ADDS_ONE, at, 1));
    let graph = a.capture(one_by_one)?;
    run(&a, one_by_one)?;
    a.sync()?;
    assert_eq!(words.read()?, [LAUNCHES as u32], "the launches one by one did not each add one");
    run(&a, || go(&graph))?;
    a.sync()?;
    assert_eq!(words.read()?, [2 * LAUNCHES as u32], "the graph did not add one for each launch it holds");
    Ok(())
}
