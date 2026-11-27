//! The GPU's clock, which the kernels of the tests wait by. On the GPU here, now and then the first reads of
//! `%globaltimer` in a kernel give the time with its top bits wrong, until the value next changes, and a kernel that
//! took its start from such a read would end a wait of a second at once. A kernel is launched many times, one at a
//! time. It reads the clock as it starts, takes its start as the kernels of `gpu` do, with `clock_start`, waits a
//! little by the clock from it, and says what it read. The test asserts that no start is the first read, as
//! `clock_start` waits for the clock to move, which is so of every launch; and that no start is far from the last
//! read, as a wrong one would be, which a run with a wrong first read puts to the test. It prints how many first
//! reads were wrong, and does not assert it, as how often the GPU gives one is not in the test's hands. It is a test
//! of its own, as a program of its own, so that no other kernel of the tests is on the GPU while it runs.

use kernite_cuda::{Grid, Module, arg};

mod gpu;
use gpu::{cuda, ptx};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// `starts(out, nanos)`: one thread reads `%globaltimer`, takes its start with `clock_start`, waits `nanos` nanoseconds
/// from it by the clock, and writes to the three words at `out` the first read, the start, and the last read of the
/// wait.
const PTX: &str = "
.visible .entry starts(.param .u64 out_param, .param .u64 nanos_param)
{
	.reg .u64 %out, %nanos, %first, %start, %now, %gone;
	.reg .pred %early;

	ld.param.u64 %out, [out_param];
	ld.param.u64 %nanos, [nanos_param];
	cvta.to.global.u64 %out, %out;
	mov.u64 %first, %globaltimer;
	call (%start), clock_start, ();
LATER:
	mov.u64 %now, %globaltimer;
	sub.u64 %gone, %now, %start;
	setp.lt.u64 %early, %gone, %nanos;
	@%early bra LATER;
	st.volatile.global.u64 [%out], %first;
	st.volatile.global.u64 [%out+8], %start;
	st.volatile.global.u64 [%out+16], %now;
	ret;
}
";

/// How many times the kernel is launched: a run of under two seconds. About one run in four has a wrong first read to
/// put to the assert on the starts; the assert on `at_first` has its launches in every run.
const LAUNCHES: usize = 50_000;

/// How long each kernel waits, by the clock from its start: about twenty ticks of the clock, each about a microsecond.
const NANOS: u64 = 20_000;

/// The most that the wait may take beyond `NANOS`, by the clock: more than eleven days, which no run takes, however
/// long the GPU is given to another program, and much less than a wrong read is off by, which is more than 10^17
/// nanoseconds.
const FAR: u64 = 1_000_000_000_000_000;

/// The words one launch writes.
const WORDS: usize = 3;

#[test]
fn a_start_is_a_read_after_the_clock_moved_and_never_a_wrong_read_though_a_first_read_sometimes_is() -> Result {
    let Some((cuda, _alone)) = cuda() else { return Ok(()) };
    let module = Module::load(&cuda, &ptx(PTX), &["starts"], &[])?;
    let reads = cuda.alloc::<u64>(WORDS * LAUNCHES)?;
    let base = reads.ptr().addr();
    for i in 0..LAUNCHES {
        // Each launch is of an idle GPU, as the first reads were wrong on it.
        let out = base + (WORDS * size_of::<u64>() * i) as u64;
        // SAFETY: `starts` takes a `u64` address and a `u64`: the address is of three words of `reads`, which this
        // launch alone writes, and `reads` lasts until the sync after the launch.
        unsafe { module.launch(0, Grid::new(1, 1), &[arg(&out), arg(&NANOS)]) }?;
        cuda.sync()?;
    }
    let reads = reads.read()?;
    let (reads, rest) = reads.as_chunks::<WORDS>();
    assert!(rest.is_empty(), "the words of the launches are not a whole number of launches");
    let (mut at_first, mut wrong_first, mut wrong_start) = (0, 0, 0);
    for (i, &[first, start, last]) in reads.iter().enumerate() {
        // The last read is right, as the kernel waits for the clock to move before it. A read is wrong if it is far
        // from it: a first read, or a start that the wait from it did not bring to the last read, `NANOS` on.
        let first_wrong = last.abs_diff(first) >= FAR;
        let start_wrong = !(NANOS..NANOS + FAR).contains(&last.wrapping_sub(start));
        at_first += usize::from(start == first);
        wrong_first += usize::from(first_wrong);
        wrong_start += usize::from(start_wrong);
        if first_wrong || start_wrong {
            eprintln!("launch {i}: first read {first:#x}, start {start:#x}, last read {last:#x}");
        }
    }
    eprintln!(
        "{LAUNCHES} launches: {wrong_first} first reads wrong, {wrong_start} starts wrong, {at_first} at the first"
    );
    assert_eq!(at_first, 0, "a kernel took its start from its first read of the clock, before the clock moved");
    assert_eq!(wrong_start, 0, "a kernel took a wrong read of the clock for its start");
    Ok(())
}
