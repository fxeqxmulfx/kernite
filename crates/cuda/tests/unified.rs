//! Memory the host and a kernel both reach, on the GPU here, if there is one: what each sees of
//! it, and when the host may touch it.
//!
//! The tests run beside each other, so one holds a slice of managed memory while a kernel of
//! another runs. A launch's contract allows that where the GPU reaches managed memory while the
//! host does, `Cuda::concurrent_managed_access`, as the GPU here does, and nowhere else: where the
//! GPU does not, `setup` makes the tests run one at a time.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use kernite_cuda::{Cuda, Grid, Module, Ptr, arg};

/// Two kernels, written by hand. `add_one`: `x[i] += 1` for each `i < n`, a thread each, wrapping.
/// `delay`: one thread waits `ns` nanoseconds on the GPU's global timer, then `x[0] += 100`.
const PTX: &str = "
.version 7.5
.target sm_61
.address_size 64

.visible .entry add_one(.param .u64 x_param, .param .u32 n_param)
{
	.reg .pred %p;
	.reg .u32 %n, %i, %block, %size, %thread, %v;
	.reg .u64 %x, %at;

	ld.param.u64 %x, [x_param];
	ld.param.u32 %n, [n_param];
	mov.u32 %block, %ctaid.x;
	mov.u32 %size, %ntid.x;
	mov.u32 %thread, %tid.x;
	mad.lo.u32 %i, %block, %size, %thread;
	setp.ge.u32 %p, %i, %n;
	@%p bra done;
	mad.wide.u32 %at, %i, 4, %x;
	ld.global.u32 %v, [%at];
	add.u32 %v, %v, 1;
	st.global.u32 [%at], %v;
done:
	ret;
}

.visible .entry delay(.param .u64 x_param, .param .u64 ns_param)
{
	.reg .pred %p;
	.reg .u32 %v;
	.reg .u64 %x, %ns, %start, %now, %waited;

	ld.param.u64 %x, [x_param];
	ld.param.u64 %ns, [ns_param];
	mov.u64 %start, %globaltimer;
again:
	mov.u64 %now, %globaltimer;
	sub.u64 %waited, %now, %start;
	setp.lt.u64 %p, %waited, %ns;
	@%p bra again;
	ld.global.u32 %v, [%x];
	add.u32 %v, %v, 100;
	st.global.u32 [%x], %v;
	ret;
}
";

/// How long `delay` waits: long enough that the host, which asks for the memory as soon as the
/// launch returns, is early by far.
const DELAY: Duration = Duration::from_millis(300);

/// Held by each test, from its start to its end, where the GPU cannot reach managed memory while
/// the host does: the host may then hold no slice of managed memory while any kernel runs, so no
/// other test may.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The lock where the GPU needs the tests to run one at a time, the GPU and the kernels; or `None`
/// where there is no GPU, after it says so. A test binds the lock first, so that it is dropped
/// last, after the memory of the test is freed.
fn setup() -> Option<(Option<MutexGuard<'static, ()>>, Cuda, Module)> {
    let cuda = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")).ok()?;
    let one = (!cuda.concurrent_managed_access()).then(|| ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner));
    let module = Module::load(&cuda, PTX, &["add_one", "delay"], &[]).unwrap();
    Some((one, cuda, module))
}

/// Pseudo-random words, the same for the same seed.
fn words(n: usize, seed: u32) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        })
        .collect()
}

/// Launches `add_one` over the `n` words at `x`.
fn add_one(module: &Module, x: Ptr<u32>, n: u32) {
    // SAFETY: `add_one` takes a pointer and a `u32`, and writes only the first `n` words there,
    // which the caller's memory has. The caller holds no slice of it. Where the GPU cannot reach
    // managed memory while the host does, `setup` has the test run alone, so no other test holds a
    // slice of managed memory either, as the header says.
    unsafe { module.launch(0, Grid::new(n.div_ceil(256), 256), &[arg(&x), arg(&n)]) }.unwrap();
}

/// Launches `delay` on the word at `x`, which waits for `DELAY` and adds 100 to the word.
fn delay(module: &Module, x: Ptr<u32>) {
    let ns = DELAY.as_nanos() as u64;
    // SAFETY: `delay` takes a pointer and a `u64`, and writes the one word at the pointer, which
    // the caller's memory has. The caller holds no slice of it. Where the GPU cannot reach managed
    // memory while the host does, `setup` has the test run alone, so no other test holds a slice
    // of managed memory either, as the header says.
    unsafe { module.launch(1, Grid::new(1, 1), &[arg(&x), arg(&ns)]) }.unwrap();
}

#[test]
fn a_managed_buffer_is_filled_by_the_host_added_to_by_a_kernel_and_read_by_the_host() {
    let Some((_one, cuda, module)) = setup() else { return };
    // Not a whole number of blocks, and words that wrap.
    let n = 1_000_003;
    let mut input = words(n, 1);
    input[0] = u32::MAX;
    let mut m = cuda.managed::<u32>(n).unwrap();
    assert_eq!(m.len(), n);
    assert!(m.host().unwrap().iter().all(|&w| w == 0), "managed memory is zero until written");
    m.host_mut().unwrap().copy_from_slice(&input);
    add_one(&module, m.ptr(), n as u32);
    cuda.sync().unwrap();
    let want: Vec<u32> = input.iter().map(|w| w.wrapping_add(1)).collect();
    assert_eq!(m.host().unwrap(), want);
    // The kernel and the host went to one memory: one address, and no copy.
    assert_eq!(m.host().unwrap().as_ptr() as u64, m.ptr().addr());
}

#[test]
fn the_host_waits_for_a_kernel_before_it_reads_or_writes_managed_memory() {
    let Some((_one, cuda, module)) = setup() else { return };
    let mut m = cuda.managed::<u32>(4).unwrap();
    m.host_mut().unwrap().copy_from_slice(&[5, 1, 2, 3]);
    // The kernel adds 100 to the first word after `DELAY`; the host asks as soon as it is launched.
    let started = Instant::now();
    delay(&module, m.ptr());
    assert_eq!(m.host().unwrap(), [105, 1, 2, 3], "a read without a sync of its own is of the finished kernel");
    assert!(started.elapsed() >= DELAY * 5 / 6, "the read came after {:?}", started.elapsed());
    // And a write: the kernel has finished, so its add is not written over, nor does it write over the host's.
    m.host_mut().unwrap().copy_from_slice(&[5, 6, 7, 8]);
    delay(&module, m.ptr());
    let started = Instant::now();
    let slice = m.host_mut().unwrap();
    assert_eq!(slice[0], 105, "{:?}", started.elapsed());
    slice[1] = 9;
    assert_eq!(m.host().unwrap(), [105, 9, 7, 8]);
}

#[test]
fn managed_memory_has_one_address_for_both_and_may_have_no_elements() {
    let Some((_one, cuda, _)) = setup() else { return };
    let m = cuda.managed::<u32>(4).unwrap();
    assert_eq!(m.at(3).addr(), m.ptr().addr() + 12);
    assert_eq!(m.at(4).addr(), m.ptr().addr() + 16);
    assert_eq!(format!("{m:?}"), format!("Unified(4 elements at {:#x})", m.ptr().addr()));
    let past = catch_unwind(AssertUnwindSafe(|| m.at(5)));
    assert!(past.is_err(), "an address past the end is refused");
    let none = cuda.managed::<u64>(0).unwrap();
    assert!(none.is_empty());
    assert_eq!(none.host().unwrap(), [] as [u64; 0]);
}

#[test]
fn dropping_managed_memory_waits_for_the_kernel_that_uses_it() {
    let Some((_one, cuda, module)) = setup() else { return };
    let m = cuda.managed::<u32>(1).unwrap();
    let started = Instant::now();
    delay(&module, m.ptr());
    drop(m);
    assert!(started.elapsed() >= DELAY * 5 / 6, "freed after {:?}", started.elapsed());
    // A kernel that wrote freed memory would have failed.
    cuda.sync().unwrap();
}
