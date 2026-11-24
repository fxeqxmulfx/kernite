//! Streams, on a GPU: which work runs beside which, what the host waits for, and where a launch goes. Two kernels,
//! written by hand, share a flag: `waits` spins until it sees the flag, or a time has gone by, and `sets` writes it.
//! A `waits` that sees the flag ran beside the `sets` that wrote it, and one that does not was kept from it, however
//! long either took, as other programs' kernels on the GPU change the times. A test of its own, as a program of its
//! own, and one test at a time: the null stream is the GPU's and not a thread's, so a kernel that waits on a stream
//! keeps back what another test puts on the null stream.

use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use kernite_cuda::{Buffer, Cuda, Error, Grid, Module, Ptr, Stream, Unified, arg};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

static ALONE: Mutex<()> = Mutex::new(());

/// The kernels, written by hand. `%globaltimer` is the GPU's clock, in nanoseconds.
///
/// `waits(flag, out, nanos)` spins until `*flag` is not 0, or until `nanos` nanoseconds have gone by; then it writes 1
/// to `*out` if it saw the flag, and 2 if the time went by first. `sets(flag, nanos)` writes 1 to `*flag` after `nanos`
/// nanoseconds.
const PTX: &str = "
.version 7.5
.target sm_61
.address_size 64

.visible .entry waits(.param .u64 flag_param, .param .u64 out_param, .param .u64 nanos_param)
{
	.reg .u64 %flag, %out, %nanos, %start, %now, %gone;
	.reg .u32 %word;
	.reg .pred %set, %early;

	ld.param.u64 %flag, [flag_param];
	ld.param.u64 %out, [out_param];
	ld.param.u64 %nanos, [nanos_param];
	cvta.to.global.u64 %flag, %flag;
	cvta.to.global.u64 %out, %out;
	mov.u64 %start, %globaltimer;
SPIN:
	ld.volatile.global.u32 %word, [%flag];
	setp.ne.u32 %set, %word, 0;
	@%set bra SEEN;
	mov.u64 %now, %globaltimer;
	sub.u64 %gone, %now, %start;
	setp.lt.u64 %early, %gone, %nanos;
	@%early bra SPIN;
	mov.u32 %word, 2;
	st.volatile.global.u32 [%out], %word;
	ret;
SEEN:
	mov.u32 %word, 1;
	st.volatile.global.u32 [%out], %word;
	ret;
}

.visible .entry sets(.param .u64 flag_param, .param .u64 nanos_param)
{
	.reg .u64 %flag, %nanos, %start, %now, %gone;
	.reg .u32 %one;
	.reg .pred %early;

	ld.param.u64 %flag, [flag_param];
	ld.param.u64 %nanos, [nanos_param];
	cvta.to.global.u64 %flag, %flag;
	mov.u64 %start, %globaltimer;
LATER:
	mov.u64 %now, %globaltimer;
	sub.u64 %gone, %now, %start;
	setp.lt.u64 %early, %gone, %nanos;
	@%early bra LATER;
	mov.u32 %one, 1;
	st.volatile.global.u32 [%flag], %one;
	ret;
}
";

/// The kernels of `PTX`, in the order `Module::load` is given their names.
const KERNELS: &[&str] = &["waits", "sets"];
const WAITS: usize = 0;
const SETS: usize = 1;

/// What `waits` writes: that it saw the flag, or that the time went by first.
const SEEN: u32 = 1;
const TIMED_OUT: u32 = 2;

/// How long `waits` waits for the flag, in nanoseconds, where it must see it: a second, much more than `sets` takes to
/// run beside it. It returns as soon as it sees the flag, so the second is not spent.
const A_SECOND: u64 = 1_000_000_000;
/// And where it must not: a `sets` that ran beside it would write the flag at once, so a tenth of a second shows the
/// same, and it is spent.
const A_TENTH: u64 = 100_000_000;
/// `sets` at once.
const AT_ONCE: u64 = 0;
/// How long `sets` waits before it writes the flag, where the host's call must wait for it: almost half a second.
const A_LITTLE_WHILE: u64 = 400_000_000;
/// And where another stream's kernel must be running still when the first one is done: a second.
const A_LONG_WHILE: u64 = 1_000_000_000;
/// The least a call that waits for `sets` takes, with `A_LITTLE_WHILE`: the kernel is launched before the call is
/// timed.
const AT_LEAST: Duration = Duration::from_millis(300);

/// The GPU, with the lock that keeps the other tests off the null stream; or a line that says the test is skipped.
fn cuda() -> Option<(Cuda, MutexGuard<'static, ()>)> {
    let alone = ALONE.lock().unwrap_or_else(PoisonError::into_inner);
    Cuda::new().map(|cuda| (cuda, alone)).map_err(|e| eprintln!("no GPU: skipped. {e}")).ok()
}

/// `work`, with `stream` the current stream of its GPU.
fn run<R>(stream: &Stream, work: impl FnOnce() -> R) -> R {
    // SAFETY: the kernels of this file share one word between streams at most, the flag that `sets` writes and `waits`
    // reads, each by one volatile access that the GPU does whole: that is what the tests look for. `waits` writes its
    // own `out`, and the kernels reach no other memory in common.
    unsafe { stream.run(work) }
}

/// The kernel `kernel` of `module`, launched over one thread on the current stream of its GPU, with `args`.
fn launch(module: &Module, kernel: usize, args: &[*const c_void]) -> Result {
    // SAFETY: each kernel is given what its `.param`s are: addresses of memory of the GPU that the test holds, and
    // numbers, as `u64`; and it writes only the word that an address it is given is of.
    unsafe { module.launch(kernel, Grid::new(1, 1), args) }?;
    Ok(())
}

/// What the page-locked word `at` holds now. It is read where it is, and not by `host`, which waits for the GPU first.
fn peek(at: &Unified<'_, u32>) -> u32 {
    // SAFETY: `at` is a word of page-locked memory that lasts as long as `at`, which the GPU reaches at the address the
    // host has for it; the kernel writes it by one store of the whole word.
    unsafe { ptr::read_volatile(ptr::with_exposed_provenance::<u32>(at.ptr().addr() as usize)) }
}

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
    let module = Module::load(&cuda, PTX, KERNELS, &[])?;
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
    let module = Module::load(&cuda, PTX, KERNELS, &[])?;
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
    let module = Module::load(&other, PTX, KERNELS, &[])?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    // The streams are made by `cuda`, and the kernels are launched with the module that `other` loaded: the GPU is one.
    race(&other, &module, SEEN, on(&a), on(&b), "a launch through another `Cuda` of the GPU")?;
    Ok(())
}

/// Launches `sets` on stream `a`, to write the word at `at` after a little while; and says when, for a call that waits.
fn queue_sets(module: &Module, a: &Stream, at: Ptr<u32>) -> Result<Instant> {
    run(a, || launch(module, SETS, &[arg(&at), arg(&A_LITTLE_WHILE)]))?;
    Ok(Instant::now())
}

/// That the call made at `started` took as long as the kernel queued before it has, so that it waited for the kernel.
fn waited(started: Instant, what: &str) {
    let took = started.elapsed();
    assert!(took >= AT_LEAST, "{what} took {took:?}, and did not wait for the kernel of a stream");
}

#[test]
fn the_host_waits_for_the_work_of_a_stream_where_it_waits_for_the_null_streams() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = Module::load(&cuda, PTX, KERNELS, &[])?;
    let a = cuda.stream()?;
    // What the calls are made on, made before each kernel is queued: a call that makes memory may wait of itself. The
    // kernel writes a word of the memory that each call reaches. Memory the host reaches is in `unified.rs`.
    let (mut buffer, dropped) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?);
    let started = queue_sets(&module, &a, buffer.ptr())?;
    let seen = buffer.read()?;
    waited(started, "Buffer::read");
    assert_eq!(seen, [1], "Buffer::read did not see what the kernel of a stream wrote");
    let started = queue_sets(&module, &a, buffer.ptr())?;
    buffer.write(&[5])?;
    waited(started, "Buffer::write");
    assert_eq!(buffer.read()?, [5], "Buffer::write came before the kernel of a stream, which wrote over it");
    let started = queue_sets(&module, &a, buffer.ptr())?;
    cuda.sync()?;
    waited(started, "Cuda::sync");
    let started = queue_sets(&module, &a, dropped.ptr())?;
    drop(dropped);
    waited(started, "the drop of a buffer");
    // A kernel that wrote freed memory would have failed.
    cuda.sync()?;
    // A module is unloaded when it is dropped, after the kernels it launched on a stream are done.
    buffer.zero()?;
    let started = queue_sets(&module, &a, buffer.ptr())?;
    drop(module);
    waited(started, "the drop of a module");
    assert_eq!(buffer.read()?, [1], "the kernel of a stream did not run to its end when its module was dropped");
    Ok(())
}

/// `zero` and `copy_from` are queued and not waited for: they run after the kernel that a stream was given before them,
/// as the stream blocks the null stream they go to, which the values they leave show. `sets` writes the 1 a little
/// while after it is queued; `zero` clears it if it runs after, and `copy_from` copies it if it does.
#[test]
fn zero_and_copy_from_run_after_the_work_a_stream_was_given_before_them() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = Module::load(&cuda, PTX, KERNELS, &[])?;
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

/// A call that writes the first buffer, with a second to copy from if it likes: `zero` or `copy_from`.
type Call = fn(&mut Buffer<u32>, &Buffer<u32>) -> Result<(), Error>;

/// `zero` and `copy_from` run before the work a stream is given after them, as the stream's work waits for what the
/// null stream, which they go to, was given before it, and they are queued and not waited for. The null stream is busy
/// for a tenth of a second, so that the call waits in the queue behind that kernel; then a stream is given a `sets`
/// that writes at once. It must wait for the call, or it would write its 1 first, and the call would clear it, or copy
/// the 0 over it.
#[test]
fn zero_and_copy_from_run_before_the_work_a_stream_was_given_after_them() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = Module::load(&cuda, PTX, KERNELS, &[])?;
    let a = cuda.stream()?;
    let (mut flag, mut zeros, busy) = (cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?, cuda.alloc::<u32>(1)?);
    zeros.zero()?;
    let (flag_at, busy_at) = (flag.ptr(), busy.ptr());
    let calls: [(&str, Call); 2] =
        [("zero", |flag, _| flag.zero()), ("copy_from", |flag, zeros| flag.copy_from(zeros))];
    for (name, call) in calls {
        launch(&module, SETS, &[arg(&busy_at), arg(&A_TENTH)])?;
        call(&mut flag, &zeros)?;
        run(&a, || launch(&module, SETS, &[arg(&flag_at), arg(&AT_ONCE)]))?;
        assert_eq!(flag.read()?, [1], "{name} ran after the kernel of a stream queued after it");
    }
    Ok(())
}

#[test]
fn a_stream_waits_for_its_own_work_alone() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = Module::load(&cuda, PTX, KERNELS, &[])?;
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
    let module = Module::load(&cuda, PTX, KERNELS, &[])?;
    let (a, flag) = (cuda.stream()?, cuda.pinned::<u32>(1)?);
    let at = flag.ptr();
    run(&a, || launch(&module, SETS, &[arg(&at), arg(&A_LITTLE_WHILE)]))?;
    drop(a);
    cuda.sync()?;
    assert_eq!(peek(&flag), 1, "the kernel of the stream dropped did not run");
    Ok(())
}
