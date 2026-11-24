//! What the tests that keep a stream busy have in common: five kernels, written by hand, two of which hold work back
//! until the host writes a flag or the GPU's clock says a time has gone by, one that copies a word, to show when the
//! work after it runs, and two that change the words they are given, each its own way, to show in what order they ran;
//! the way every kernel of the tests takes its start from that clock; the lock that keeps the other tests of a file off
//! the null stream, which is the GPU's and not a thread's; and the host's reads and writes of the words the kernels
//! share.
#![allow(dead_code, reason = "a test file takes what it needs of this module, and rustc calls the rest dead")]

use std::ffi::c_void;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard, PoisonError};

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
/// `waits(flag, out, nanos)` spins until `*flag` is not 0, or until `nanos` nanoseconds have gone by; then it writes 1
/// to `*out` if it saw the flag, and 2 if the time went by first. `sets(flag, nanos)` writes 1 to `*flag` after `nanos`
/// nanoseconds. `copies(from, to)` writes the word `*from` to `*to`. `adds_one` and `doubles` are below.
const PTX: &str = "
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
	call (%start), clock_start, ();
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
const KERNELS: &[&str] = &["waits", "sets", "copies", "adds_one", "doubles"];
pub const WAITS: usize = 0;
pub const SETS: usize = 1;
pub const COPIES: usize = 2;
/// `adds_one(words, n)` adds 1 to each of the first `n` words, and `doubles(words, n)` doubles them, one thread each,
/// for an `n` of 32 or less, one block's threads: `(x + 1) * 2` is not `x * 2 + 1`, so the order they ran in shows.
pub const ADDS_ONE: usize = 3;
pub const DOUBLES: usize = 4;

/// What `waits` writes: that it saw the flag, or that the time went by first.
pub const SEEN: u32 = 1;
pub const TIMED_OUT: u32 = 2;

/// How long `hold` gives the host to write the flag: five seconds, much more than the host takes. The kernel returns
/// as soon as it sees the flag, so the time is spent only where a call kept the host from writing it.
pub const FOR_THE_HOST: u64 = 5_000_000_000;

/// How long `waits` waits for the flag, in nanoseconds, where it must see it: a second, much more than `sets` takes to
/// run beside it. It returns as soon as it sees the flag, so the second is not spent.
pub const A_SECOND: u64 = 1_000_000_000;
/// And where it must not: a `sets` that ran beside it would write the flag at once, so a tenth of a second shows the
/// same, and it is spent.
pub const A_TENTH: u64 = 100_000_000;
/// `sets` at once.
pub const AT_ONCE: u64 = 0;

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
    // in common.
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
/// host writes `flag`, with `poke`, or `FOR_THE_HOST` goes by. The kernel writes `SEEN` to `out` if it saw the flag,
/// and `TIMED_OUT` if the time went by first.
pub fn hold(module: &Module, flag: Ptr<u32>, out: Ptr<u32>) -> Result {
    launch(module, WAITS, &[arg(&flag), arg(&out), arg(&FOR_THE_HOST)])
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
