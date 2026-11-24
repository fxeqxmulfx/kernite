//! Events, on a GPU, if there is one: when the work before an event is done, which stream waits for which event, and
//! the time between two events. Three kernels of `gpu`: `waits` spins until the host writes a flag, or a time has gone
//! by, and then writes whether it saw the flag; `sets` writes a word after a time; `copies` copies a word. A kernel
//! that waits for the host holds back what is queued after it on its stream, however long the host takes to write the
//! flag, so what a test asserts of the work before an event does not depend on how long a call took, as other programs'
//! kernels on the GPU change the times. The tests go one at a time, as `gpu::cuda` sees to.

use std::thread;
use std::time::Duration;

use kernite_cuda::{Cuda, Module, Unified, arg};

mod gpu;
use gpu::{COPIES, SEEN, SETS, cuda, hold, launch, load, peek, poke, run};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// A word of page-locked memory, zero, which a kernel and the host both reach.
fn word(cuda: &Cuda) -> Result<Unified<'static, u32>> {
    Ok(cuda.pinned::<u32>(1)?)
}

/// Keeps the current stream busy for `nanos` nanoseconds, with `sets`, which writes 1 to `sink` at its end.
fn spins(module: &Module, sink: &Unified<'_, u32>, nanos: u64) -> Result {
    launch(module, SETS, &[arg(&sink.ptr()), arg(&nanos)])
}

/// Launches `copies` on the current stream: it writes the word `from` to `to`.
fn copies(module: &Module, from: &Unified<'_, u32>, to: &Unified<'_, u32>) -> Result {
    launch(module, COPIES, &[arg(&from.ptr()), arg(&to.ptr())])
}

#[test]
fn an_event_is_not_done_while_the_work_before_it_runs_and_sync_waits_for_all_of_it() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (mut flag, mut out, mut sink) = (word(&cuda)?, word(&cuda)?, word(&cuda)?);
    let event = cuda.event()?;
    // An event that was never recorded stands for no work.
    assert!(event.done()?, "an event never recorded is not done");
    event.sync()?;
    // The kernel waits for the flag, and then spins for 50 ms: the event is recorded after both.
    hold(&module, flag.ptr(), out.ptr())?;
    spins(&module, &sink, 50_000_000)?;
    event.record()?;
    assert!(!event.done()?, "the event is done, with the kernel before it held by the flag");
    poke(&mut flag, 1);
    event.sync()?;
    assert!(event.done()?, "the event is not done after sync");
    assert_eq!(peek(&mut out), SEEN, "the kernel did not see the flag");
    assert_eq!(peek(&mut sink), 1, "sync returned before the kernel of 50 ms, which the event stands for, was done");
    Ok(())
}

#[test]
fn an_event_stands_for_the_work_before_it_on_its_stream_and_on_the_null_stream_for_that_of_every_stream() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    let (mut flag, mut out) = (word(&cuda)?, word(&cuda)?);
    let (on_a, on_b) = (cuda.event()?, cuda.event()?);
    // The kernel on `a` is held by the flag. An event recorded on `b`, which has no work, is done; one recorded on `a`
    // is not.
    run(&a, || -> Result {
        hold(&module, flag.ptr(), out.ptr())?;
        on_a.record()?;
        Ok(())
    })?;
    run(&b, || on_b.record())?;
    assert!(!on_a.done()?, "the event recorded on the stream that has the kernel is done");
    // `sync` waits for the work the event stands for, and for no more: it returns while the kernel on `a` is held. The
    // GPU takes a moment to reach a record, so what is done is asked after `sync`.
    on_b.sync()?;
    assert!(on_b.done()?, "an event recorded on a stream that has no work is not done after sync");
    assert!(!on_a.done()?, "sync of an event waited for the work of another stream");
    // A record again makes it stand for the work of the stream it is recorded on then, and for that alone.
    run(&b, || on_a.record())?;
    on_a.sync()?;
    assert!(on_a.done()?, "an event recorded again on a stream with no work is not done after sync");
    run(&a, || on_a.record())?;
    assert!(!on_a.done()?, "an event recorded again on the stream with the kernel is done");
    // The null stream is blocking: an event recorded on it stands for the work that every stream was given before it,
    // and for none that a stream is given after it. A second kernel on `a`, held by a flag of its own, is given after
    // the record.
    let (mut later_flag, mut later_out) = (word(&cuda)?, word(&cuda)?);
    let on_null = cuda.event()?;
    on_null.record()?;
    assert!(!on_null.done()?, "an event recorded on the null stream is done, with a stream's kernel before it");
    run(&a, || hold(&module, later_flag.ptr(), later_out.ptr()))?;
    poke(&mut flag, 1);
    on_null.sync()?;
    assert!(on_a.done()? && on_null.done()?);
    assert_eq!(peek(&mut out), SEEN);
    assert_eq!(
        peek(&mut later_out),
        0,
        "sync of an event on the null stream waited for work a stream was given after it"
    );
    poke(&mut later_flag, 1);
    a.sync()?;
    assert_eq!(peek(&mut later_out), SEEN, "the kernel given to `a` after the record did not see its flag");
    Ok(())
}

#[test]
fn a_stream_that_waits_for_an_event_runs_after_the_work_it_stands_for_and_one_that_does_not_wait_does_not() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b, c) = (cuda.stream()?, cuda.stream()?, cuda.stream()?);
    let event = cuda.event()?;
    // `waits` on `a` writes 1 to `mark` once the host writes `go`. `copies` copies `mark` to `seen`, on `b`, and to
    // `other`, on `c`, which does not wait for the event.
    let (mut go, mut mark, mut seen, mut other) = (word(&cuda)?, word(&cuda)?, word(&cuda)?, word(&cuda)?);
    for waited in [false, true] {
        poke(&mut go, 0);
        poke(&mut mark, 0);
        poke(&mut seen, 7);
        poke(&mut other, 7);
        run(&a, || -> Result {
            hold(&module, go.ptr(), mark.ptr())?;
            event.record()?;
            Ok(())
        })?;
        run(&b, || -> Result {
            if waited {
                event.wait()?;
            }
            copies(&module, &mark, &seen)
        })?;
        // A wait holds the work of the stream that waits, and no other stream's: `c` copies at once, while `waits`
        // still spins, whether or not `b` waits.
        run(&c, || copies(&module, &mark, &other))?;
        c.sync()?;
        assert_eq!(peek(&mut other), 0, "the wait of a stream held back the work of another");
        if waited {
            // `copies` is held by the event, which the kernel on `a` holds back until the host writes `go`.
            poke(&mut go, 1);
            b.sync()?;
            assert_eq!(
                peek(&mut seen),
                SEEN,
                "the stream that waited for the event ran before the kernel it stands for"
            );
        } else {
            // Nothing holds `copies` back, and it is done while `waits` still spins: the two streams run at once.
            b.sync()?;
            assert_eq!(peek(&mut seen), 0, "the stream that did not wait ran after the kernel on the other stream");
            poke(&mut go, 1);
        }
        a.sync()?;
        assert_eq!(peek(&mut mark), SEEN, "the kernel did not see the flag");
    }
    Ok(())
}

#[test]
fn a_record_after_a_wait_does_not_change_what_the_stream_waits_for() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b, c) = (cuda.stream()?, cuda.stream()?, cuda.stream()?);
    let (mut go, mark, mut seen) = (word(&cuda)?, word(&cuda)?, word(&cuda)?);
    let event = cuda.event()?;
    poke(&mut seen, 7);
    // The event stands for the kernel on `a`, which `go` holds, and `b` waits for it before it copies `mark`.
    run(&a, || -> Result {
        hold(&module, go.ptr(), mark.ptr())?;
        event.record()?;
        Ok(())
    })?;
    run(&b, || -> Result {
        event.wait()?;
        copies(&module, &mark, &seen)
    })?;
    // The event is recorded again, on `c`, which has no work: it stands for nothing that is not done, so `sync`
    // returns.
    run(&c, || event.record())?;
    event.sync()?;
    // A copy that the record let go would have run by now, as nothing holds it: a pause of 50 ms is far more than it
    // takes. It has not run, as `b` still waits for the work the event stood for when `b` waited.
    thread::sleep(Duration::from_millis(50));
    assert_eq!(
        peek(&mut seen),
        7,
        "a record after the wait of `b` let its copy go, before the kernel the wait was for"
    );
    poke(&mut go, 1);
    b.sync()?;
    assert_eq!(peek(&mut seen), SEEN, "the stream did not wait for the work the event stood for when it waited");
    a.sync()?;
    Ok(())
}

#[test]
fn a_wait_for_an_event_never_recorded_waits_for_nothing_and_a_dropped_event_is_still_waited_for() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    let (mut go, mark, mut seen) = (word(&cuda)?, word(&cuda)?, word(&cuda)?);
    // The kernel on `a` is held by `go`. `b` waits for an event never recorded, and its kernel runs at once.
    run(&a, || hold(&module, go.ptr(), mark.ptr()))?;
    poke(&mut seen, 7);
    let never = cuda.event()?;
    run(&b, || -> Result {
        never.wait()?;
        copies(&module, &mark, &seen)
    })?;
    b.sync()?;
    assert_eq!(peek(&mut seen), 0, "the stream that waited for an event never recorded was held back");
    // An event recorded after the kernel on `a`, waited for by `b`, and dropped at once: dropping it does not wait, and
    // `b` still waits for the kernel it stood for.
    let event = cuda.event()?;
    poke(&mut seen, 7);
    run(&a, || event.record())?;
    run(&b, || -> Result {
        event.wait()?;
        copies(&module, &mark, &seen)
    })?;
    drop(event);
    // A copy that the drop let go would have run by now, as nothing holds it: a pause of 50 ms is far more than it
    // takes, and the kernel that sees `go` right after would otherwise race it. It has not run, as `b` still waits for
    // the kernel the event stood for.
    thread::sleep(Duration::from_millis(50));
    assert_eq!(
        peek(&mut seen),
        7,
        "the drop of the event let the copy of `b` go, before the kernel the event stood for"
    );
    poke(&mut go, 1);
    b.sync()?;
    assert_eq!(peek(&mut seen), SEEN, "the stream did not wait for the work of an event that was dropped");
    a.sync()?;
    Ok(())
}

/// What `duration_since` says where the GPU reached the event it is asked about before the one it is timed since.
const REACHED_BEFORE: &str =
    "cuEventElapsedTime: the GPU reached this event before the one it is timed since: ask the later of the two";

#[test]
fn two_events_say_the_time_between_them_and_it_is_an_error_where_it_has_none_to_say() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (start, stop) = (cuda.timed_event()?, cuda.timed_event()?);
    let never = cuda.timed_event()?;
    let sink = word(&cuda)?;
    start.record()?;
    spins(&module, &sink, 10_000_000)?;
    stop.record()?;
    stop.sync()?;
    // The kernel spun for 10 ms between the records, so the GPU's clock says at least as much.
    let time = stop.duration_since(&start)?;
    eprintln!("the time between two events around a kernel of 10 ms: {time:?}");
    assert!(time >= Duration::from_millis(10), "{time:?} is less than the kernel's 10 ms");
    assert_eq!(start.duration_since(&start)?, Duration::ZERO);
    // The other way round has a time below zero, which a `Duration` is not.
    assert_eq!(start.duration_since(&stop).err().ok_or("duration_since gave Ok")?.to_string(), REACHED_BEFORE);
    // An event never recorded has no time, and the driver says so.
    let said = "cuEventElapsedTime: CUDA_ERROR_INVALID_HANDLE (400): invalid resource handle";
    assert_eq!(stop.duration_since(&never).err().ok_or("duration_since gave Ok")?.to_string(), said);
    assert_eq!(never.duration_since(&start).err().ok_or("duration_since gave Ok")?.to_string(), said);
    // An event that is not done has none either, until the work before it is.
    let (mut flag, out) = (word(&cuda)?, word(&cuda)?);
    let (first, later) = (cuda.timed_event()?, cuda.timed_event()?);
    first.record()?;
    hold(&module, flag.ptr(), out.ptr())?;
    later.record()?;
    let said = "cuEventElapsedTime: CUDA_ERROR_NOT_READY (600): device not ready";
    assert_eq!(later.duration_since(&first).err().ok_or("duration_since gave Ok")?.to_string(), said);
    poke(&mut flag, 1);
    later.sync()?;
    assert!(later.duration_since(&first)? > Duration::ZERO);
    Ok(())
}

#[test]
fn the_time_between_two_events_is_from_when_the_gpu_reached_one_to_when_it_reached_the_other() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    let (mut flag, out) = (word(&cuda)?, word(&cuda)?);
    let (earlier, later) = (cuda.timed_event()?, cuda.timed_event()?);
    // The host records `earlier` first, on `a`, after a kernel that the flag holds; then `later`, on `b`, which has no
    // work. The GPU reaches `later` first, and `earlier` only when the host has written the flag.
    run(&a, || -> Result {
        hold(&module, flag.ptr(), out.ptr())?;
        earlier.record()?;
        Ok(())
    })?;
    run(&b, || later.record())?;
    later.sync()?;
    assert!(!earlier.done()?, "the GPU reached `earlier`, with the kernel before it held by the flag");
    thread::sleep(Duration::from_millis(50));
    poke(&mut flag, 1);
    earlier.sync()?;
    // The time of `later` since `earlier` is below zero, though the host recorded `earlier` first: the GPU reached
    // `later` first. The time of `earlier` since `later` is what went by while the kernel was held, 50 ms and a little
    // more, by the GPU's clock.
    let said = later.duration_since(&earlier).err().ok_or("duration_since gave Ok")?;
    assert_eq!(said.to_string(), REACHED_BEFORE);
    let time = earlier.duration_since(&later)?;
    eprintln!("the GPU reached `earlier` {time:?} after `later`, which the host recorded after it");
    assert!(time >= Duration::from_millis(40), "{time:?} is less than the 50 ms the kernel was held");
    Ok(())
}
