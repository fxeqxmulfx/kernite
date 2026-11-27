//! Streams, on a GPU: which work runs beside which, what the host waits for, and where a launch goes. Two kernels of
//! `gpu` share a flag: `waits` looks for the flag, and `sets` writes it. A `waits` that sees the flag ran beside the
//! `sets` that wrote it, and one that does not was kept from it: it looks from when the host has launched `sets`, so
//! a slow host does not end its looks early. A call that waits for the GPU is made while a kernel that a flag holds
//! is queued, and the flag is written while the host is in the call. A test of its own, as a program of its own, and
//! one test at a time.

use std::panic::{self, AssertUnwindSafe};

use kernite_cuda::{Buffer, Cuda, Error, Module, Ptr, Stream, Unified, arg};

mod gpu;
use gpu::{
    AT_ONCE, HANG, LOOKS, NOT_SEEN, SEEN, SETS, cuda, hold, launch, load, pause, peek, poke, release_while, run, watch,
};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// That `waits` wrote `expected` after `around_waits` launched it and `around_sets` launched `sets`, each given the
/// launch to put where it likes: on a stream by `run`, or where it goes. `sets` is launched after `waits`, and writes
/// the flag at once. `waits` looks for the flag once the host has launched `sets`: `HANG` times at most where it must
/// see it, and `LOOKS` where it must not.
fn race(
    cuda: &Cuda,
    module: &Module,
    expected: u32,
    around_waits: impl FnOnce(&dyn Fn() -> Result) -> Result,
    around_sets: impl FnOnce(&dyn Fn() -> Result) -> Result,
    what: &str,
) -> Result {
    let looks = if expected == SEEN { HANG } else { LOOKS };
    let (flag, out, mut go) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?, cuda.pinned::<u32>(1)?);
    let (flag_at, out_at, go_at) = (flag.ptr(), out.ptr(), go.ptr());
    around_waits(&|| watch(module, go_at, flag_at, out_at, looks))?;
    around_sets(&|| launch(module, SETS, &[arg(&flag_at), arg(&AT_ONCE)]))?;
    poke(&mut go, 1);
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
    // On the null stream `sets` waits for `a`'s `waits`, which looks for the flag in vain.
    race(&cuda, &module, NOT_SEEN, on(&a), on_null, "a stream, then the null stream")?;
    // On the stream of `waits`, `sets` is queued behind it.
    race(&cuda, &module, NOT_SEEN, on(&a), on(&a), "one stream")?;
    // On `b`, `sets` waits for the null stream's `waits`.
    race(&cuda, &module, NOT_SEEN, on_null, on(&b), "the null stream, then a stream")?;
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
    waits(NOT_SEEN, "with no run", &on_null)?;
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
    waits(NOT_SEEN, "in the run in a run", &inner)?;
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
    waits(NOT_SEEN, "after a run, with no run", &after_run)?;
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

/// Queues on stream `a` a kernel that the page-locked flag this gives holds back, then a `pause`, and then `sets`,
/// which writes 1 to the word at `at`. The kernel writes what it saw to the flag, `SEEN` once the host has written it.
fn held_sets(cuda: &Cuda, module: &Module, a: &Stream, at: Ptr<u32>) -> Result<Unified<'static, u32>> {
    // The flag is made before the kernel is queued: a call that makes memory may wait of itself.
    let flag = cuda.pinned::<u32>(1)?;
    run(a, || -> Result {
        hold(module, flag.ptr(), flag.ptr())?;
        pause(module)?;
        launch(module, SETS, &[arg(&at), arg(&AT_ONCE)])
    })?;
    Ok(flag)
}

#[test]
fn the_host_waits_for_the_work_of_a_stream_where_it_waits_for_the_null_streams() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    // What the calls are made on, made before each kernel is queued: a call that makes memory may wait of itself. The
    // flag is written while the host is in the call, which returns only if it waits for the kernel that the flag holds.
    // `sets` writes a word of the memory that a call reaches, or a page-locked word `done` where the call frees memory
    // or reaches none, which the host reads where it is, with `peek`, as `host` waits for the GPU first: that `sets`
    // has written the word when the call returns says that the call waited for it too. Memory the host reaches is in
    // `unified.rs`.
    let (mut buffer, dropped, mut done) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?, cuda.pinned::<u32>(1)?);
    let mut flag = held_sets(&cuda, &module, &a, buffer.ptr())?;
    let read = release_while(&mut flag, || buffer.read())??;
    assert_eq!(read, [1], "Buffer::read did not see what the kernel of a stream wrote");
    let mut flag = held_sets(&cuda, &module, &a, buffer.ptr())?;
    release_while(&mut flag, || buffer.write(&[5]))??;
    assert_eq!(buffer.read()?, [5], "Buffer::write came before the kernel of a stream, which wrote over it");
    let mut flag = held_sets(&cuda, &module, &a, done.ptr())?;
    release_while(&mut flag, || cuda.sync())??;
    assert_eq!(peek(&mut done), 1, "Cuda::sync returned before the kernel of a stream was done");
    poke(&mut done, 0);
    let mut flag = held_sets(&cuda, &module, &a, done.ptr())?;
    release_while(&mut flag, || drop(dropped))?;
    assert_eq!(peek(&mut done), 1, "the drop of a buffer returned before the kernel of a stream was done");
    // A module is unloaded when it is dropped, after the kernels it launched on a stream are done.
    poke(&mut done, 0);
    let mut flag = held_sets(&cuda, &module, &a, done.ptr())?;
    release_while(&mut flag, || drop(module))?;
    assert_eq!(peek(&mut done), 1, "the drop of a module returned before the kernel of a stream was done");
    assert_eq!(peek(&mut flag), SEEN, "the kernel the flag held did not see it");
    Ok(())
}

/// `zero` and `copy_from` run after the kernel that a stream was given before them, as the stream blocks the null
/// stream they go to, which the values they leave show. `sets` writes the 1 once the host has written a flag, after
/// the call, and a pause; `zero` clears it if it runs after, and `copy_from` copies it if it does. A call that did not
/// wait would run while the kernel is held, or in the pause.
#[test]
fn zero_and_copy_from_run_after_the_work_a_stream_was_given_before_them() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let (mut flag, mut copy) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?);
    let mut go = held_sets(&cuda, &module, &a, flag.ptr())?;
    flag.zero()?;
    poke(&mut go, 1);
    assert_eq!(flag.read()?, [0], "zero ran before the kernel of a stream queued before it");
    let mut go = held_sets(&cuda, &module, &a, flag.ptr())?;
    copy.copy_from(&flag)?;
    poke(&mut go, 1);
    assert_eq!(copy.read()?, [1], "copy_from ran before the kernel of a stream queued before it");
    Ok(())
}

/// A call that writes the first buffer, with a second to copy from if it likes.
type Call = fn(&mut Buffer<u32>, &Buffer<u32>) -> Result<(), Error>;

/// The two calls that write a buffer: `zero` and `copy_from`. Both go to the null stream.
const CALLS: [(&str, Call); 2] = [("zero", |to, _| to.zero()), ("copy_from", |to, from| to.copy_from(from))];

/// `zero` and `copy_from` run before the work a stream is given after them, as the stream's work waits for what the
/// null stream, which they go to, was given before it. The null stream is held by a kernel until the host writes a
/// flag, and then by a pause, so that the call waits in the queue behind them; then a stream is given a `sets` that
/// writes at once. It must wait for the call, or it would write its 1 while the kernel is held, and the call would
/// clear it, or copy the 0 over it.
#[test]
fn zero_and_copy_from_run_before_the_work_a_stream_was_given_after_them() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let a = cuda.stream()?;
    let (mut flag, zeros) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?);
    let flag_at = flag.ptr();
    for (name, call) in CALLS {
        let mut go = cuda.pinned::<u32>(1)?;
        hold(&module, go.ptr(), go.ptr())?;
        pause(&module)?;
        call(&mut flag, &zeros)?;
        run(&a, || launch(&module, SETS, &[arg(&flag_at), arg(&AT_ONCE)]))?;
        poke(&mut go, 1);
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
        let (mut flag, mut done) = (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
        hold(&module, flag.ptr(), done.ptr())?;
        call(&mut to, &from)?;
        assert_eq!(peek(&mut done), 0, "{name} waited for the kernel before it, which was not done when it returned");
        poke(&mut flag, 1);
        cuda.sync()?;
        assert_eq!(peek(&mut done), SEEN, "the kernel before {name} did not see the flag, or is not done after sync");
    }
    Ok(())
}

#[test]
fn a_stream_waits_for_its_own_work_alone() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    // A kernel on each stream, each held by a flag of its own, and each writing what it saw to a word of its own.
    let (mut early, mut late) = (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    let (mut early_out, mut late_out) = (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    run(&a, || hold(&module, early.ptr(), early_out.ptr()))?;
    run(&b, || hold(&module, late.ptr(), late_out.ptr()))?;
    // `a.sync()` returns once the kernel of `a` is done, with that of `b` still held: one that waited for it too would
    // return only when it gave up, in a hang, and wrote `NOT_SEEN`.
    release_while(&mut early, || a.sync())??;
    assert_eq!(peek(&mut early_out), SEEN, "the kernel of `a` is not done when `a.sync()` returns");
    assert_eq!(peek(&mut late_out), 0, "`a.sync()` waited for the kernel of `b`");
    release_while(&mut late, || b.sync())??;
    assert_eq!(peek(&mut late_out), SEEN, "the kernel of `b` is not done when `b.sync()` returns");
    Ok(())
}

#[test]
fn the_work_of_a_stream_runs_on_when_the_stream_is_dropped() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, mut set) = (cuda.stream()?, cuda.pinned::<u32>(1)?);
    // The stream is dropped while a kernel holds it, with `sets` queued behind: both run once the host writes the flag.
    let mut flag = held_sets(&cuda, &module, &a, set.ptr())?;
    drop(a);
    poke(&mut flag, 1);
    cuda.sync()?;
    assert_eq!(peek(&mut set), 1, "the kernel queued on the stream dropped did not run");
    Ok(())
}
