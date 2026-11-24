//! Streams, on a GPU: which work runs beside which, what the host waits for, and where a launch goes. Two kernels of
//! `gpu` share a flag: `waits` spins until it sees the flag, or a time has gone by, and `sets` writes it. A `waits`
//! that sees the flag ran beside the `sets` that wrote it, and one that does not was kept from it, however long either
//! took, as other programs' kernels on the GPU change the times. A test of its own, as a program of its own, and one
//! test at a time.

use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

use kernite_cuda::{Buffer, Cuda, Error, Module, Ptr, Stream, arg};

mod gpu;
use gpu::{A_SECOND, A_TENTH, AT_ONCE, SEEN, SETS, TIMED_OUT, WAITS, cuda, hold, launch, load, peek, poke, run};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// How long `sets` waits before it writes the flag, where the host's call must wait for it: almost half a second.
const A_LITTLE_WHILE: u64 = 400_000_000;
/// And where another stream's kernel must be running still when the first one is done: a second.
const A_LONG_WHILE: u64 = 1_000_000_000;

/// That `waits` wrote `expected` after `around_waits` launched it and `around_sets` launched `sets`, each given the
/// launch to put where it likes: on a stream by `run`, or where it goes. `sets` is launched after `waits`, and writes
/// the flag at once. Where `waits` must see the flag it waits `A_SECOND` for it, and where it must not, `A_TENTH`.
fn race(
    cuda: &Cuda,
    module: &Module,
    expected: u32,
    around_waits: impl FnOnce(&dyn Fn() -> Result) -> Result,
    around_sets: impl FnOnce(&dyn Fn() -> Result) -> Result,
    what: &str,
) -> Result {
    let nanos = if expected == SEEN { A_SECOND } else { A_TENTH };
    let (flag, out) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?);
    let (flag_at, out_at) = (flag.ptr(), out.ptr());
    around_waits(&|| launch(module, WAITS, &[arg(&flag_at), arg(&out_at), arg(&nanos)]))?;
    around_sets(&|| launch(module, SETS, &[arg(&flag_at), arg(&AT_ONCE)]))?;
    cuda.sync()?;
    assert_eq!(out.read()?, [expected], "{what}");
    Ok(())
}

/// A launch put on `stream`.
fn on(stream: &Stream) -> impl Fn(&dyn Fn() -> Result) -> Result {
    move |launch| run(stream, launch)
}

/// A launch left where it goes with no stream made current: the null stream.
fn on_null(launch: &dyn Fn() -> Result) -> Result {
    launch()
}

#[test]
fn the_work_of_two_streams_runs_at_once_and_a_stream_and_the_null_stream_wait_for_each_other() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    // `waits` on `a` sees the flag that `sets` writes on `b`: the two ran beside each other.
    race(&cuda, &module, SEEN, on(&a), on(&b), "two streams")?;
    // On the null stream `sets` waits for `a`'s `waits`, which waits for `sets`: the time goes by.
    race(&cuda, &module, TIMED_OUT, on(&a), on_null, "a stream, then the null stream")?;
    // On the stream of `waits`, `sets` is queued behind it.
    race(&cuda, &module, TIMED_OUT, on(&a), on(&a), "one stream")?;
    // On `b`, `sets` waits for the null stream's `waits`.
    race(&cuda, &module, TIMED_OUT, on_null, on(&b), "the null stream, then a stream")?;
    Ok(())
}

#[test]
fn a_launch_goes_to_the_stream_run_made_current_and_afterwards_to_the_stream_before() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    // `sets` is on `b` in each: `waits` sees the flag only if it went to a stream that is neither `b`, where it is
    // queued before `sets`, nor the null stream, which `b` waits for.
    let waits = |expected: u32, what: &str, around_waits: &dyn Fn(&dyn Fn() -> Result) -> Result| {
        race(&cuda, &module, expected, around_waits, on(&b), what)
    };
    waits(SEEN, "in a run", &on(&a))?;
    waits(TIMED_OUT, "with no run", &on_null)?;
    assert_eq!(run(&a, || 7), 7, "run gives back what the closure gives");
    // After a run in the run: the stream of the run, not that of the inner one, and not the null stream.
    let after_inner = |launch: &dyn Fn() -> Result| {
        run(&a, || {
            run(&b, || ());
            launch()
        })
    };
    waits(SEEN, "after a run of another stream in a run", &after_inner)?;
    let after_itself = |launch: &dyn Fn() -> Result| {
        run(&a, || {
            run(&a, || ());
            launch()
        })
    };
    waits(SEEN, "after a run of the same stream in a run", &after_itself)?;
    let inner = |launch: &dyn Fn() -> Result| run(&a, || run(&b, launch));
    waits(TIMED_OUT, "in the run in a run", &inner)?;
    // A closure that panics gives the stream back too.
    let after_panic = |launch: &dyn Fn() -> Result| {
        run(&a, || {
            let panicked =
                panic::catch_unwind(AssertUnwindSafe(|| run(&b, || panic::resume_unwind(Box::new("a panic")))));
            assert!(panicked.is_err(), "the closure did not panic");
            launch()
        })
    };
    waits(SEEN, "after a run that panicked in a run", &after_panic)?;
    let after_run = |launch: &dyn Fn() -> Result| {
        run(&a, || ());
        launch()
    };
    waits(TIMED_OUT, "after a run, with no run", &after_run)?;
    Ok(())
}

#[test]
fn a_launch_through_another_cuda_of_the_gpu_goes_to_the_stream_of_the_gpu() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let other = Cuda::new()?;
    let module = load(&other)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    // The streams are made by `cuda`, and the kernels are launched with the module that `other` loaded: the GPU is one.
    race(&other, &module, SEEN, on(&a), on(&b), "a launch through another `Cuda` of the GPU")?;
    Ok(())
}

/// Launches `sets` on stream `a`, to write the word at `at` after a little while.
fn queue_sets(module: &Module, a: &Stream, at: Ptr<u32>) -> Result {
    run(a, || launch(module, SETS, &[arg(&at), arg(&A_LITTLE_WHILE)]))
}

#[test]
fn the_host_waits_for_the_work_of_a_stream_where_it_waits_for_the_null_streams() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    // What the calls are made on, made before each kernel is queued: a call that makes memory may wait of itself. The
    // kernel writes a word of the memory that a call reaches, or a page-locked word `done` where the call frees memory
    // or reaches none, which the host reads where it is, with `peek`, as `host` waits for the GPU first: that the
    // kernel has written the word when the call returns says that the call waited. Memory the host reaches is in
    // `unified.rs`.
    let (mut buffer, dropped, done) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?, cuda.pinned::<u32>(1)?);
    queue_sets(&module, &a, buffer.ptr())?;
    assert_eq!(buffer.read()?, [1], "Buffer::read did not see what the kernel of a stream wrote");
    queue_sets(&module, &a, buffer.ptr())?;
    buffer.write(&[5])?;
    assert_eq!(buffer.read()?, [5], "Buffer::write came before the kernel of a stream, which wrote over it");
    queue_sets(&module, &a, done.ptr())?;
    cuda.sync()?;
    assert_eq!(peek(&done), 1, "Cuda::sync returned before the kernel of a stream was done");
    poke(&done, 0);
    queue_sets(&module, &a, done.ptr())?;
    drop(dropped);
    assert_eq!(peek(&done), 1, "the drop of a buffer returned before the kernel of a stream was done");
    // A module is unloaded when it is dropped, after the kernels it launched on a stream are done.
    poke(&done, 0);
    queue_sets(&module, &a, done.ptr())?;
    drop(module);
    assert_eq!(peek(&done), 1, "the drop of a module returned before the kernel of a stream was done");
    Ok(())
}

/// `zero` and `copy_from` run after the kernel that a stream was given before them, as the stream blocks the null
/// stream they go to, which the values they leave show. `sets` writes the 1 a little while after it is queued; `zero`
/// clears it if it runs after, and `copy_from` copies it if it does.
#[test]
fn zero_and_copy_from_run_after_the_work_a_stream_was_given_before_them() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let (mut flag, mut copy) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?);
    queue_sets(&module, &a, flag.ptr())?;
    flag.zero()?;
    assert_eq!(flag.read()?, [0], "zero ran before the kernel of a stream queued before it");
    queue_sets(&module, &a, flag.ptr())?;
    copy.copy_from(&flag)?;
    assert_eq!(copy.read()?, [1], "copy_from ran before the kernel of a stream queued before it");
    Ok(())
}

/// A call that writes the first buffer, with a second to copy from if it likes.
type Call = fn(&mut Buffer<u32>, &Buffer<u32>) -> Result<(), Error>;

/// The two calls that write a buffer: `zero` and `copy_from`. Both go to the null stream.
const CALLS: [(&str, Call); 2] = [("zero", |to, _| to.zero()), ("copy_from", |to, from| to.copy_from(from))];

/// `zero` and `copy_from` run before the work a stream is given after them, as the stream's work waits for what the
/// null stream, which they go to, was given before it. The null stream is busy for a tenth of a second, so that the
/// call waits in the queue behind that kernel; then a stream is given a `sets` that writes at once. It must wait for
/// the call, or it would write its 1 first, and the call would clear it, or copy the 0 over it.
#[test]
fn zero_and_copy_from_run_before_the_work_a_stream_was_given_after_them() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let (mut flag, mut zeros, busy) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?);
    zeros.zero()?;
    let (flag_at, busy_at) = (flag.ptr(), busy.ptr());
    for (name, call) in CALLS {
        launch(&module, SETS, &[arg(&busy_at), arg(&A_TENTH)])?;
        call(&mut flag, &zeros)?;
        run(&a, || launch(&module, SETS, &[arg(&flag_at), arg(&AT_ONCE)]))?;
        assert_eq!(flag.read()?, [1], "{name} ran after the kernel of a stream queued after it");
    }
    Ok(())
}

/// `zero` and `copy_from` return before the GPU has done their work: they are queued behind the kernel that keeps the
/// null stream busy until the host writes a flag, and the page-locked word that the kernel writes at its end is still 0
/// when each call returns, however slow the host is; `Cuda::sync` waits for the kernel and for the call. The GPU does
/// what the call asked once the kernel is done, which the values in the tests above show.
#[test]
fn zero_and_copy_from_return_before_the_gpu_has_done_their_work() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (mut to, from) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?);
    for (name, call) in CALLS {
        // Words for each call, made before the kernel is queued: a call that makes memory may wait of itself.
        let (flag, done) = (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
        hold(&module, flag.ptr(), done.ptr())?;
        let started = Instant::now();
        call(&mut to, &from)?;
        let took = started.elapsed();
        let seen = peek(&done);
        eprintln!("{name} returned after {took:?}, and the kernel before it, which a flag held, had set {seen}");
        assert_eq!(seen, 0, "{name} waited for the kernel before it, which was not done when it returned");
        poke(&flag, 1);
        cuda.sync()?;
        assert_eq!(peek(&done), SEEN, "the kernel before {name} did not see the flag, or is not done after sync");
    }
    Ok(())
}

#[test]
fn a_stream_waits_for_its_own_work_alone() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    let (early, late) = (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    let (early_at, late_at) = (early.ptr(), late.ptr());
    run(&a, || launch(&module, SETS, &[arg(&early_at), arg(&A_LITTLE_WHILE)]))?;
    run(&b, || launch(&module, SETS, &[arg(&late_at), arg(&A_LONG_WHILE)]))?;
    a.sync()?;
    assert_eq!(peek(&early), 1, "the kernel of `a` is not done when `a.sync()` returns");
    assert_eq!(peek(&late), 0, "`a.sync()` waited for the kernel of `b`");
    b.sync()?;
    assert_eq!(peek(&late), 1, "the kernel of `b` is not done when `b.sync()` returns");
    Ok(())
}

#[test]
fn the_work_of_a_stream_runs_on_when_the_stream_is_dropped() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, flag) = (cuda.stream()?, cuda.pinned::<u32>(1)?);
    let at = flag.ptr();
    run(&a, || launch(&module, SETS, &[arg(&at), arg(&A_LITTLE_WHILE)]))?;
    drop(a);
    cuda.sync()?;
    assert_eq!(peek(&flag), 1, "the kernel of the stream dropped did not run");
    Ok(())
}
