//! What the tests that keep a stream busy have in common: three kernels, written by hand, two of which hold work back
//! until the host writes a flag or the GPU's clock says a time has gone by, and one that copies a word, to show when
//! the work after it runs; the lock that keeps the other tests of a file off the null stream, which is the GPU's and
//! not a thread's; and the host's reads and writes of the words the kernels share.
#![allow(dead_code, reason = "a test file takes what it needs of this module, and rustc calls the rest dead")]

use std::ffi::c_void;
use std::ptr;
use std::sync::{Mutex, MutexGuard, PoisonError};

use kernite_cuda::{Cuda, Grid, Module, Ptr, Stream, Unified, arg};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// The kernels. `%globaltimer` is the GPU's clock, in nanoseconds.
///
/// `waits(flag, out, nanos)` spins until `*flag` is not 0, or until `nanos` nanoseconds have gone by; then it writes 1
/// to `*out` if it saw the flag, and 2 if the time went by first. `sets(flag, nanos)` writes 1 to `*flag` after `nanos`
/// nanoseconds. `copies(from, to)` writes the word `*from` to `*to`.
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
";

/// The kernels of `PTX`, in the order `Module::load` is given their names.
const KERNELS: &[&str] = &["waits", "sets", "copies"];
pub const WAITS: usize = 0;
pub const SETS: usize = 1;
pub const COPIES: usize = 2;

/// What `waits` writes: that it saw the flag, or that the time went by first.
pub const SEEN: u32 = 1;
pub const TIMED_OUT: u32 = 2;

/// How long `hold` gives the host to write the flag: five seconds, much more than the host takes. The kernel returns
/// as soon as it sees the flag, so the time is spent only where a call kept the host from writing it.
const FOR_THE_HOST: u64 = 5_000_000_000;

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
    Ok(Module::load(cuda, PTX, KERNELS, &[])?)
}

/// `work`, with `stream` the current stream of its GPU.
pub fn run<R>(stream: &Stream, work: impl FnOnce() -> R) -> R {
    // SAFETY: the kernels share a word between streams at most: the flag that `sets` writes and `waits` reads, and the
    // word that `waits` writes and `copies` reads, each by one volatile access that the GPU does whole: that is what
    // the tests look for. The words that the kernels write are their own, and they reach no other memory in common.
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

/// What the page-locked word `at` holds now. It is read where it is, and not by `host`, which waits for the GPU first.
pub fn peek(at: &Unified<'_, u32>) -> u32 {
    // SAFETY: `at` is a word of page-locked memory that lasts as long as `at`, which the GPU reaches at the address the
    // host has for it; the kernel writes it by one store of the whole word.
    unsafe { ptr::read_volatile(ptr::with_exposed_provenance::<u32>(at.ptr().addr() as usize)) }
}

/// Writes `value` to the page-locked word `at`, where the kernel that spins on it reads it.
pub fn poke(at: &Unified<'_, u32>, value: u32) {
    // SAFETY: `at` is a word of page-locked memory that lasts as long as `at`, which the GPU reaches at the address the
    // host has for it; the kernel reads it by one load of the whole word.
    unsafe { ptr::write_volatile(ptr::with_exposed_provenance_mut::<u32>(at.ptr().addr() as usize), value) }
}
