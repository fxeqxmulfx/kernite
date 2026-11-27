//! What the tests that keep a stream busy have in common: six kernels, written by hand. `waits` holds work back until
//! the host writes a flag, or looks for a word that other work writes, a number of times. `pauses` holds work back for
//! a number of looks. `sets` writes a word, at once or after a time by the GPU's clock. `copies` copies a word, to show
//! when the work after it runs. `adds_one` and `doubles` change the words they are given, each its own way, to show in
//! what order they ran. Then the way a kernel of the tests takes its start from that clock; the lock that keeps the
//! other tests of a file off the null stream, which is the GPU's and not a thread's; the host's reads and writes of the
//! words the kernels share; and the way the host lets a kernel go while it is in a call that waits for it.
//!
//! No test holds a time. A kernel that holds work back waits for the host, and gives up only in a hang. A kernel that
//! must not see a word looks for it a number of times, from when the host has done what the test asks of it: what
//! runs beside it writes the word long before it is done, and what is queued behind it cannot.
#![allow(dead_code, reason = "a test file takes what it needs of this module, and rustc calls the rest dead")]

use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use kernite_cuda::{Cuda, Grid, Module, Ptr, Stream, Unified, arg};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// Where a kernel that waits by the clock takes its start. `%globaltimer` is the GPU's clock, in nanoseconds, and on
/// the GPU here, an RTX 3050 under NVIDIA's open kernel module 620.70.06 and CUDA 13.5, a kernel's first reads of it
/// are now and then wrong: the time with its top bits wrong, until the clock next moves. A start taken at the first
/// read then ends a wait of a second at once, or never. So `clock_start` waits for the clock to move, and returns the
/// first read after: the time, to within a tick of the clock, which is a microsecond. `clock.rs` counts the starts it
/// gives that are wrong, and those it takes at the first read.
const CLOCK: &str = "
.func (.reg .u64 %start) clock_start()
{
	.reg .u64 %now;
	.reg .pred %same;

	mov.u64 %start, %globaltimer;
MOVES:
	mov.u64 %now, %globaltimer;
	setp.eq.u64 %same, %now, %start;
	@%same bra MOVES;
	mov.u64 %start, %now;
	ret;
}
";

/// The PTX of a module of the tests, for `sm_61` and later: its header, `clock_start`, and `kernels`, which call it by
/// `call (%start), clock_start, ();`.
pub fn ptx(kernels: &str) -> String {
    ptx_for("sm_61", kernels)
}

/// The same for `target` and later, for kernels whose instructions need it: `ld.acquire` and `st.release` need `sm_70`.
pub fn ptx_for(target: &str, kernels: &str) -> String {
    format!(".version 7.5\n.target {target}\n.address_size 64\n{CLOCK}{kernels}")
}

/// The kernels.
///
/// `waits(flag, out, looks)` looks at `*flag` until it is not 0, `looks` times at most; then it writes 1 to `*out` if
/// it saw it, and 2 if not. `pauses(looks)` looks `looks` times at a word of the module's own, which nothing writes.
/// `sets(flag, nanos)` writes 1 to `*flag` after `nanos` nanoseconds. `copies(from, to)` writes the word `*from` to
/// `*to`. `adds_one` and `doubles` are below.
const PTX: &str = "
.visible .entry waits(.param .u64 flag_param, .param .u64 out_param, .param .u64 looks_param)
{
	.reg .u64 %flag, %out, %looks;
	.reg .u32 %word;
	.reg .pred %clear, %over;

	ld.param.u64 %flag, [flag_param];
	ld.param.u64 %out, [out_param];
	ld.param.u64 %looks, [looks_param];
	cvta.to.global.u64 %flag, %flag;
	cvta.to.global.u64 %out, %out;
LOOK:
	setp.eq.u64 %over, %looks, 0;
	@%over bra NOT_SEEN;
	sub.u64 %looks, %looks, 1;
	ld.volatile.global.u32 %word, [%flag];
	setp.eq.u32 %clear, %word, 0;
	@%clear bra LOOK;
	mov.u32 %word, 1;
	st.volatile.global.u32 [%out], %word;
	ret;
NOT_SEEN:
	mov.u32 %word, 2;
	st.volatile.global.u32 [%out], %word;
	ret;
}

.global .u32 never;

.visible .entry pauses(.param .u64 looks_param)
{
	.reg .u64 %looks;
	.reg .u32 %word;
	.reg .pred %over;

	ld.param.u64 %looks, [looks_param];
LOOK:
	setp.eq.u64 %over, %looks, 0;
	@%over bra DONE;
	sub.u64 %looks, %looks, 1;
	ld.volatile.global.u32 %word, [never];
	bra LOOK;
DONE:
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
	call (%start), clock_start, ();
LATER:
	mov.u64 %now, %globaltimer;
	sub.u64 %gone, %now, %start;
	setp.lt.u64 %early, %gone, %nanos;
	@%early bra LATER;
	mov.u32 %one, 1;
	st.volatile.global.u32 [%flag], %one;
	ret;
}

.visible .entry copies(.param .u64 from_param, .param .u64 to_param)
{
	.reg .u64 %from, %to;
	.reg .u32 %word;

	ld.param.u64 %from, [from_param];
	ld.param.u64 %to, [to_param];
	cvta.to.global.u64 %from, %from;
	cvta.to.global.u64 %to, %to;
	ld.volatile.global.u32 %word, [%from];
	st.volatile.global.u32 [%to], %word;
	ret;
}

.visible .entry adds_one(.param .u64 words_param, .param .u32 n_param)
{
	.reg .u64 %words, %at;
	.reg .u32 %n, %i, %word;
	.reg .pred %past;

	ld.param.u64 %words, [words_param];
	ld.param.u32 %n, [n_param];
	mov.u32 %i, %tid.x;
	setp.ge.u32 %past, %i, %n;
	@%past bra DONE;
	cvta.to.global.u64 %words, %words;
	mad.wide.u32 %at, %i, 4, %words;
	ld.global.u32 %word, [%at];
	add.u32 %word, %word, 1;
	st.global.u32 [%at], %word;
DONE:
	ret;
}

.visible .entry doubles(.param .u64 words_param, .param .u32 n_param)
{
	.reg .u64 %words, %at;
	.reg .u32 %n, %i, %word;
	.reg .pred %past;

	ld.param.u64 %words, [words_param];
	ld.param.u32 %n, [n_param];
	mov.u32 %i, %tid.x;
	setp.ge.u32 %past, %i, %n;
	@%past bra DONE;
	cvta.to.global.u64 %words, %words;
	mad.wide.u32 %at, %i, 4, %words;
	ld.global.u32 %word, [%at];
	add.u32 %word, %word, %word;
	st.global.u32 [%at], %word;
DONE:
	ret;
}
";

/// The kernels of `PTX`, in the order `Module::load` is given their names.
const KERNELS: &[&str] = &["waits", "sets", "copies", "adds_one", "doubles", "pauses"];
pub const WAITS: usize = 0;
pub const SETS: usize = 1;
pub const COPIES: usize = 2;
/// `adds_one(words, n)` adds 1 to each of the first `n` words, and `doubles(words, n)` doubles them, one thread each,
/// for an `n` of 32 or less, one block's threads: `(x + 1) * 2` is not `x * 2 + 1`, so the order they ran in shows.
pub const ADDS_ONE: usize = 3;
pub const DOUBLES: usize = 4;
pub const PAUSES: usize = 5;

/// What `waits` writes: that it saw the flag, or that it looked as many times as it was told to and did not.
pub const SEEN: u32 = 1;
pub const NOT_SEEN: u32 = 2;

/// How many times `waits` looks at a flag that the host or other work is to write, before it gives up: about 50 s on
/// page-locked memory and 8 s on the GPU's own, here. It returns as soon as it sees the flag, so the time is spent only
/// in a hang.
pub const HANG: u64 = 1 << 26;

/// And at a word that work beside it must not write: about 12 ms on page-locked memory and 2 ms on the GPU's own, here.
/// What runs beside it writes the word in microseconds, so these looks show it as a hang's would, and they are spent.
pub const LOOKS: u64 = 1 << 14;

/// `sets` at once.
pub const AT_ONCE: u64 = 0;

/// How long the host waits for what must come before it says that it never came: only a hang takes this long.
pub const HOST_HANG: Duration = Duration::from_secs(60);

/// How much CPU a thread spends in a call that waits for a kernel before `release_while` writes the flag that holds the
/// kernel back: far more than a call takes that does not wait. The driver spins while it waits for the GPU, here, with
/// one context and twelve CPUs: the thread runs, and its CPU time grows. It does so while no other thread of the test
/// is in the driver: a call of another may hold a lock of the driver that this thread sleeps on, with no CPU spent, so
/// the tests that take `release_while` go one at a time, as `cuda` sees to.
pub const IN_THE_CALL: Duration = Duration::from_millis(5);

/// Held by each test of a file that takes it, from its start to its end.
static ALONE: Mutex<()> = Mutex::new(());

/// The GPU, with the lock that keeps the other tests of the file off the null stream: it is the GPU's and not a
/// thread's, so a kernel that waits on a stream keeps back what another test puts on it. Or a line that says the test
/// is skipped.
pub fn cuda() -> Option<(Cuda, MutexGuard<'static, ()>)> {
    let alone = ALONE.lock().unwrap_or_else(PoisonError::into_inner);
    Cuda::new().map(|cuda| (cuda, alone)).map_err(|e| eprintln!("no GPU: skipped. {e}")).ok()
}

/// The kernels, as a module of the GPU of `cuda`.
pub fn load(cuda: &Cuda) -> Result<Module> {
    Ok(Module::load(cuda, &ptx(PTX), KERNELS, &[])?)
}

/// `work`, with `stream` the current stream of its GPU.
pub fn run<R>(stream: &Stream, work: impl FnOnce() -> R) -> R {
    // SAFETY: the work of two streams shares one word at most: the flag that `sets` or a copy writes and `waits` reads,
    // or the word that `waits` writes and `copies` reads. A kernel reads and writes it by one `ld.volatile` or
    // `st.volatile` of the whole word. A copy or a zeroing may write it by stores of any width, and the tests do not
    // need to know which: a copy writes 1 over 0 and a zeroing 0 over 7, so the word changes in its low byte alone, and
    // a read sees the old word or the new. The words that the work writes are its own, and it reaches no other memory
    // in common. The work of a stream apart is done before the memory it reaches is dropped: each test waits for it.
    unsafe { stream.run(work) }
}

/// The kernel `kernel` of `module`, launched over one thread on the current stream of its GPU, with `args`.
pub fn launch(module: &Module, kernel: usize, args: &[*const c_void]) -> Result {
    // SAFETY: each kernel is given what its `.param`s are: addresses of memory of the GPU that the test holds, and
    // numbers, as `u64`; and it writes only the word that an address it is given is of.
    unsafe { module.launch(kernel, Grid::new(1, 1), args) }?;
    Ok(())
}

/// Launches `waits` on the current stream of its GPU, to hold back what is queued after it on that stream until the
/// host writes `flag`, with `poke` or `release_while`. The kernel writes `SEEN` to `out` if it saw the flag, and
/// `NOT_SEEN` if it did not in `HANG` looks.
pub fn hold(module: &Module, flag: Ptr<u32>, out: Ptr<u32>) -> Result {
    launch(module, WAITS, &[arg(&flag), arg(&out), arg(&HANG)])
}

/// Launches `pauses` on the current stream of its GPU, for `LOOKS`: work that runs beside it, which was queued before
/// it started, is done before it is, and work queued behind it is not.
pub fn pause(module: &Module) -> Result {
    launch(module, PAUSES, &[arg(&LOOKS)])
}

/// Launches on the current stream of its GPU a `waits` that `hold` holds until the host writes `go`, and that then
/// looks at `flag` `looks` times at most, and writes to `out` whether it saw it. Its looks start once the host has done
/// what it does before it writes `go`, however slow the host is to do it.
pub fn watch(module: &Module, go: Ptr<u32>, flag: Ptr<u32>, out: Ptr<u32>, looks: u64) -> Result {
    hold(module, go, out)?;
    launch(module, WAITS, &[arg(&flag), arg(&out), arg(&looks)])
}

/// What the page-locked word `at` holds now. It is read where it is, by `atomics`, and not by `host`, which waits for
/// the GPU first. A kernel writes it by one `st.volatile` of the whole word, and a copy or a zeroing may write it by
/// stores of any width: it changes in its low byte alone, 1 over 0 and 0 over 7, so a read sees the old word or the
/// new.
pub fn peek(at: &mut Unified<'_, u32>) -> u32 {
    at.atomics().first().map_or(0, |word| word.load(Ordering::Acquire))
}

/// Writes `value` to the page-locked word `at`, where the kernel that spins on it reads it by one load of the whole
/// word.
pub fn poke(at: &mut Unified<'_, u32>, value: u32) {
    if let Some(word) = at.atomics().first() {
        word.store(value, Ordering::Release);
    }
}

/// Calls `call`, which is to wait for a kernel that the page-locked word `flag` holds back, and has another thread
/// write the flag while this one is in the call: once this thread has spent `IN_THE_CALL` of CPU in it. What the call
/// gives; or, where the call returned before the flag was written, that it did not wait for the kernel.
pub fn release_while<R>(flag: &mut Unified<'_, u32>, call: impl FnOnce() -> R) -> Result<R> {
    let word = flag.atomics().first().ok_or("the flag has no word")?;
    let (caller, returned) = (me()?, AtomicBool::new(false));
    let start = cpu(&caller)?;
    thread::scope(|scope| {
        let releaser = scope.spawn(|| {
            let waited = until("a call that waits for a kernel to spend its CPU on the wait", || {
                Ok(returned.load(Ordering::Acquire) || cpu(&caller)? >= start + IN_THE_CALL)
            });
            // The flag is written whether or not the call returned, so that the kernel ends.
            word.store(1, Ordering::Release);
            waited
        });
        let got = call();
        // This thread spent less than `IN_THE_CALL` on its CPU if the call did not wait, and the flag is still 0.
        let early = word.load(Ordering::Acquire) == 0;
        returned.store(true, Ordering::Release);
        releaser.join().map_err(|_| "the thread that writes the flag panicked")??;
        if early {
            return Err("the call returned before the flag was written: it did not wait for the kernel it holds".into());
        }
        Ok(got)
    })
}

/// The directory in `/proc` of the calling thread, `/proc/<pid>/task/<tid>`, which `/proc/thread-self` links to.
pub fn me() -> Result<PathBuf> {
    Ok(Path::new("/proc").join(fs::read_link("/proc/thread-self")?))
}

/// How long the thread whose directory in `/proc` is `task` has run on a CPU, as Linux counts it.
pub fn cpu(task: &Path) -> Result<Duration> {
    let stat = fs::read_to_string(task.join("schedstat"))?;
    Ok(Duration::from_nanos(stat.split(' ').next().ok_or("schedstat is empty")?.parse()?))
}

/// Waits until `now` says it has come, and says that `what` did not if it has not in `HOST_HANG`, as in a hang.
pub fn until(what: &str, mut now: impl FnMut() -> Result<bool>) -> Result {
    let hang = Instant::now() + HOST_HANG;
    while !now()? {
        if Instant::now() > hang {
            return Err(format!("{what}: not in {HOST_HANG:?}, a hang").into());
        }
        thread::yield_now();
    }
    Ok(())
}
