//! The host reaches unified memory while kernels run, on a GPU, if there is one: `atomics`, which is safe, and the
//! `unsafe` `host_now` and `host_now_mut`, which wait for nothing, where `host` and `host_mut` wait for every kernel.
//! A kernel and the host hand over a job and its answer, as an engine's decode hands the work of a layer to the CPU:
//! the kernel writes the job, then a number by `st.release.sys`, and spins for the answer by `ld.acquire.sys` and
//! `nanosleep`, giving up only in a hang; the host loads the number, reads the job, writes the answer and stores an
//! ack. It is so in each kind of unified memory: page-locked, managed, a slice lent and a slice registered. A kernel
//! that loads a word again and again, while the host stores two values to it by turns, loads one of the two each time.
//! The tests go one at a time, as `gpu::cuda` sees to.

use std::sync::atomic::Ordering;

use kernite_cuda::{Cuda, Module, Plain, Unified, arg};

mod common;
mod gpu;
use common::{NO_PAGEABLE, managed_or_skip, or_skip, register_or_skip};
use gpu::{COPIES, HANG, SEEN, cuda, hold, launch, peek, poke, ptx_for, release_while, run, until};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// The kernels. `hands_off`: one thread writes the 1024 floats `i * 0.5` to `job`, then `SEQ` to the word after them by
/// `st.release.sys`; it looks at the word `ack` by `ld.acquire.sys`, with a `nanosleep` between, until it is not 0,
/// `looks` times at most; and it writes `SEEN` to `out`, by `st.release.sys`, if it saw the ack, and `NOT_SEEN` if not.
/// `reads_again(words, first, second, n, looks)`: one thread loads the word `words[0]` by `ld.global.cv.u32`, which is
/// to fetch it again each time, `n` times and on until it has loaded both `first` and `second`, `looks` times at most;
/// a load that is neither ends it at once, and it writes that load to `words[3]`. Then it writes to `words[2]` which of
/// the two it loaded, bit 1 for `first` and bit 2 for `second`, and 1 to `words[1]` by `st.release.sys`.
const PTX: &str = "
.visible .entry hands_off(.param .u64 job_param, .param .u64 ack_param, .param .u64 out_param, .param .u64 looks_param)
{
	.reg .u64 %job, %ack, %out, %looks, %at;
	.reg .u32 %i, %word;
	.reg .f32 %x;
	.reg .pred %more, %set, %over;

	ld.param.u64 %job, [job_param];
	ld.param.u64 %ack, [ack_param];
	ld.param.u64 %out, [out_param];
	ld.param.u64 %looks, [looks_param];
	cvta.to.global.u64 %job, %job;
	cvta.to.global.u64 %ack, %ack;
	cvta.to.global.u64 %out, %out;
	mov.u32 %i, 0;
WRITE:
	cvt.rn.f32.u32 %x, %i;
	mul.rn.f32 %x, %x, 0f3F000000;
	mul.wide.u32 %at, %i, 4;
	add.u64 %at, %job, %at;
	st.global.f32 [%at], %x;
	add.u32 %i, %i, 1;
	setp.lt.u32 %more, %i, 1024;
	@%more bra WRITE;
	mov.u32 %word, 7;
	st.release.sys.global.u32 [%job+4096], %word;
WAIT:
	setp.eq.u64 %over, %looks, 0;
	@%over bra NOT_SEEN;
	sub.u64 %looks, %looks, 1;
	ld.acquire.sys.global.u32 %word, [%ack];
	setp.ne.u32 %set, %word, 0;
	@%set bra SAW;
	nanosleep.u32 200;
	bra WAIT;
SAW:
	mov.u32 %word, 1;
	st.release.sys.global.u32 [%out], %word;
	ret;
NOT_SEEN:
	mov.u32 %word, 2;
	st.release.sys.global.u32 [%out], %word;
	ret;
}

.visible .entry reads_again(
	.param .u64 words_param,
	.param .u32 first_param,
	.param .u32 second_param,
	.param .u32 n_param,
	.param .u64 looks_param
)
{
	.reg .u64 %words, %looks;
	.reg .u32 %first, %second, %n, %i, %value, %bit, %loaded, %done;
	.reg .pred %over, %is_first, %is_second, %either, %more;

	ld.param.u64 %words, [words_param];
	ld.param.u32 %first, [first_param];
	ld.param.u32 %second, [second_param];
	ld.param.u32 %n, [n_param];
	ld.param.u64 %looks, [looks_param];
	cvta.to.global.u64 %words, %words;
	mov.u32 %i, 0;
	mov.u32 %loaded, 0;
LOAD:
	setp.eq.u64 %over, %looks, 0;
	@%over bra DONE;
	sub.u64 %looks, %looks, 1;
	ld.global.cv.u32 %value, [%words];
	setp.eq.u32 %is_first, %value, %first;
	setp.eq.u32 %is_second, %value, %second;
	or.pred %either, %is_first, %is_second;
	@!%either bra STRAY;
	selp.u32 %bit, 1, 2, %is_first;
	or.b32 %loaded, %loaded, %bit;
	add.u32 %i, %i, 1;
	setp.lt.u32 %more, %i, %n;
	@%more bra LOAD;
	setp.ne.u32 %more, %loaded, 3;
	@%more bra LOAD;
	bra DONE;
STRAY:
	st.global.u32 [%words+12], %value;
DONE:
	st.global.u32 [%words+8], %loaded;
	mov.u32 %done, 1;
	st.release.sys.global.u32 [%words+4], %done;
	ret;
}
";

/// The kernels of `PTX`, and their numbers in the module that `load` makes.
const HANDS_OFF: usize = 0;
const READS_AGAIN: usize = 1;

/// How many floats `hands_off` writes, and the number it writes after them.
const FLOATS: usize = 1024;
const SEQ: u32 = 7;

/// How many loads `reads_again` makes at least, and the two words that the host stores by turns while it does. The
/// second has no bit the first has, so a load of a mix of the two is neither, and neither has a byte of 0, so no mix of
/// them is 0.
const LOADS: u32 = 20_000;
const FIRST: u32 = 0x0123_4567;
const SECOND: u32 = !FIRST;

/// What `reads_again` writes where it loaded both words.
const BOTH: u32 = 0b11;

/// The module: `hands_off` and, for the later kernel and for the tests that hold one back, the kernels of `gpu`, which
/// the module of `PTX` has its own `clock_start` beside. Or a line that says the GPU is too old for `ld.acquire`.
fn load(cuda: &Cuda) -> Result<Option<(Module, Module)>> {
    if cuda.sm() < 70 {
        eprintln!("{}, sm_{}, has no `ld.acquire` and `st.release`: skipped", cuda.name(), cuda.sm());
        return Ok(None);
    }
    let hands = Module::load(cuda, &ptx_for("sm_70", PTX), &["hands_off", "reads_again"], &[])?;
    Ok(Some((hands, gpu::load(cuda)?)))
}

/// The kinds of unified memory that a job is handed over in.
#[derive(Clone, Copy)]
enum Kind {
    PageLocked,
    Managed,
    /// A slice of the host's, lent to the GPU.
    Lent,
    /// A slice of the host's, page-locked until the memory is dropped.
    Registered,
}

/// Whether the GPU makes memory of `kind`. Where it cannot, a line says so, and the test is skipped, as for a machine
/// with no GPU.
fn makes(cuda: &Cuda, kind: Kind) -> Result<bool> {
    Ok(match kind {
        Kind::PageLocked => true,
        Kind::Managed => managed_or_skip(cuda.managed::<u32>(1)).is_some(),
        Kind::Lent => or_skip(cuda.lend(&mut [0u32; 1]), NO_PAGEABLE).is_some(),
        Kind::Registered => register_or_skip(cuda, &mut [0u32; 1])?.is_some(),
    })
}

/// `len` zeros of `T`, of `kind`. The slice that is lent or registered is leaked, as the other kinds are memory of the
/// GPU that lasts until it is dropped: the test is short, and the memory is a few words.
fn memory<T: Plain>(cuda: &Cuda, kind: Kind, len: usize) -> Result<Unified<'static, T>> {
    let slice = || Vec::leak(vec![T::default(); len]);
    Ok(match kind {
        Kind::PageLocked => cuda.pinned(len)?,
        Kind::Managed => cuda.managed(len)?,
        Kind::Lent => cuda.lend(slice())?,
        Kind::Registered => cuda.register(slice())?,
    })
}

/// The bits of `floats`: what the host compares, so that no value is a NaN that is not equal to itself.
fn bits(floats: &[f32]) -> Vec<u32> {
    floats.iter().map(|x| x.to_bits()).collect()
}

/// `hands_off` launched on the null stream with its job, ack and out. It looks for the ack until a hang.
fn start(hands: &Module, job: &Unified<'_, f32>, ack: &Unified<'_, u32>, out: &Unified<'_, u32>) -> Result {
    launch(hands, HANDS_OFF, &[arg(&job.ptr()), arg(&ack.ptr()), arg(&out.ptr()), arg(&HANG)])
}

/// The kernel and the host hand over a job and its answer, in memory of `kind`: the host sees the number by `atomics`,
/// reads the floats the kernel wrote by `host_now`, writes their sum by `host_now_mut` into other memory and stores the
/// ack, and the kernel sees the ack. A kernel launched after reads the sum.
fn hand_over(kind: Kind) -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    if !makes(&cuda, kind)? {
        return Ok(());
    }
    let Some((hands, kernels)) = load(&cuda)? else { return Ok(()) };
    let mut job = memory::<f32>(&cuda, kind, FLOATS + 1)?;
    let (mut ack, out) = (memory::<u32>(&cuda, kind, 1)?, memory::<u32>(&cuda, kind, 1)?);
    let (mut sum, after) = (memory::<f32>(&cuda, kind, 1)?, memory::<f32>(&cuda, kind, 1)?);
    // Memory that the GPU zeroes, which is managed memory, is zero when the work after the zeros runs.
    cuda.sync()?;
    start(&hands, &job, &ack, &out)?;
    // The kernel runs now, and waits for the ack. The host waits for the number by loads of an atomic, which is no
    // wait for the GPU: `host()` here would wait for the kernel, which waits for the ack, until a hang.
    let number = job.atomics().last().ok_or("a job has no word for the number")?;
    until("the kernel to write the number", || Ok(number.load(Ordering::Acquire) == SEQ))?;
    // SAFETY: the kernel wrote the floats before it stored the number by `st.release.sys`, which the host loaded by
    // `Acquire`, and it writes nothing to `job` after the number: it reads `ack` and writes `out`, other memory.
    let (_, floats) = unsafe { job.host_now() }.split_last().ok_or("a job has no word for the number")?;
    let wanted: Vec<f32> = (0..FLOATS).map(|i| i as f32 * 0.5).collect();
    assert_eq!(bits(floats), bits(&wanted), "the floats the kernel wrote, which the host read at the number");
    let total: f32 = floats.iter().sum();
    // SAFETY: nothing reaches `sum` but the host: the kernel running is given `job`, `ack` and `out`, and the kernel
    // launched after reads it once this slice is let go.
    unsafe { sum.host_now_mut() }.fill(total);
    for flag in ack.atomics() {
        flag.store(1, Ordering::Release);
    }
    cuda.sync()?;
    assert_eq!(out.host()?, [SEEN], "the kernel did not see the ack");
    launch(&kernels, COPIES, &[arg(&sum.ptr()), arg(&after.ptr())])?;
    assert_eq!(bits(after.host()?), [total.to_bits()], "the kernel after read the sum the host wrote");
    Ok(())
}

#[test]
fn the_host_and_a_kernel_hand_over_a_job_and_its_answer_while_the_kernel_runs() -> Result {
    hand_over(Kind::PageLocked)
}

#[test]
fn managed_memory_is_handed_over_as_page_locked_memory_is() -> Result {
    hand_over(Kind::Managed)
}

#[test]
fn a_slice_lent_is_handed_over_as_page_locked_memory_is() -> Result {
    hand_over(Kind::Lent)
}

#[test]
fn a_slice_registered_is_handed_over_as_page_locked_memory_is() -> Result {
    hand_over(Kind::Registered)
}

/// The host stores `FIRST` and `SECOND` by turns to a word, through `atomics`, while `reads_again` loads the word again
/// and again, by plain loads of the word's size, `LOADS` times and on until it has loaded both. The two race over the
/// same four bytes, which PTX calls a uniform-size data-race, and each load is one of the two words: none is a mix of
/// them. A race of accesses that overlap in part is a mixed-size one, which the axioms of PTX's memory model do not
/// apply to.
#[test]
fn a_kernel_loads_one_of_two_words_that_the_host_stores_by_turns() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let Some((hands, _)) = load(&cuda)? else { return Ok(()) };
    // The word that the host stores; the word after it, where the kernel says that it is done; which of the two the
    // kernel loaded; and a load that was neither.
    let mut words = cuda.pinned::<u32>(4)?;
    let at = words.ptr();
    let [word, done, ..] = words.atomics() else { return Err("four words have no four atomics".into()) };
    word.store(FIRST, Ordering::Relaxed);
    launch(&hands, READS_AGAIN, &[arg(&at), arg(&FIRST), arg(&SECOND), arg(&LOADS), arg(&HANG)])?;
    // The second word is the first with its bits flipped, so each store flips the one before.
    let mut next = SECOND;
    until("the kernel to load both words", || {
        word.store(next, Ordering::Relaxed);
        next = !next;
        Ok(done.load(Ordering::Acquire) != 0)
    })?;
    let &[_, _, loaded, stray] = words.host()? else { return Err("four words are not four".into()) };
    assert_eq!(stray, 0, "a load saw neither word: a mix of them, say");
    assert_eq!(loaded, BOTH, "the loads did not race with the stores, and the kernel gave up in a hang");
    Ok(())
}

#[test]
fn host_waits_for_the_kernel_to_be_done_where_atomics_do_not() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let Some((hands, _)) = load(&cuda)? else { return Ok(()) };
    let (job, mut ack, mut out) = (cuda.pinned::<f32>(FLOATS + 1)?, cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    start(&hands, &job, &ack, &out)?;
    // `host()` waits for the kernel, which waits for an ack that the host gives while it is in `host()`: the kernel
    // sees it, and writes that, and the number is there too.
    let number = release_while(&mut ack, || job.host().map(|floats| floats[FLOATS].to_bits()))??;
    assert_eq!(peek(&mut out), SEEN, "host() returned before the kernel was done");
    assert_eq!(number, SEQ);
    Ok(())
}

#[test]
fn host_now_reads_what_one_stream_wrote_while_a_kernel_of_another_runs_and_host_waits_for_both() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = gpu::load(&cuda)?;
    let (a, b) = (cuda.stream()?, cuda.stream()?);
    let (mut from, written, mut flag, mut out) =
        (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    poke(&mut from, 7);
    // `copies` writes the word on `a`, and `waits` holds `b` until a flag is written.
    run(&a, || launch(&module, COPIES, &[arg(&from.ptr()), arg(&written.ptr())]))?;
    run(&b, || hold(&module, flag.ptr(), out.ptr()))?;
    a.sync()?;
    // SAFETY: `a`'s copy is all that writes `written`, and `a.sync()` waited for it; the kernel of `b` reaches `flag`
    // and `out`, and no part of `written`.
    assert_eq!(unsafe { written.host_now() }, [7], "host_now did not give what the copy of `a` wrote");
    assert_eq!(peek(&mut out), 0, "the kernel of `b` was done when host_now returned: it did not run beside it");
    // The host writes the flag while it is in `host()`, which waits for it, and for the kernel of `b` that saw it.
    let read = release_while(&mut flag, || written.host().map(<[u32]>::to_vec))??;
    assert_eq!(read, [7]);
    assert_eq!(peek(&mut out), SEEN, "host() returned before the kernel of `b` was done");
    Ok(())
}
