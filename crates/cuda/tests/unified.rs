//! Memory the host and a kernel both reach, on the GPU here, if there is one: what each sees of
//! it, and when the host may touch it. Each of its three kinds, managed, page-locked and lent, is
//! held to the same three rules, below.
//!
//! The tests run beside each other, so one holds a slice of managed memory while a kernel of
//! another runs. A launch's contract allows that where the GPU reaches managed memory while the
//! host does, `Cuda::concurrent_managed_access`, as the GPU here does, and nowhere else: where the
//! GPU does not, `setup` makes the tests run one at a time, as those of `placement.rs` do.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use kernite_cuda::{Advice, Cuda, Grid, Location, Module, Ptr, Unified, arg};

mod common;
use common::{NO_ADVICE, NO_PREFETCH, PAGE, first_page, or_skip};

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

/// The README's words for a GPU that cannot be lent a slice.
const NO_PAGEABLE: &str = "cuDeviceGetAttribute: the GPU cannot reach pageable memory, a slice of the host's: \
                           its CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS is 0";

/// The README's words for a GPU that cannot make page-locked memory.
const NO_PAGE_LOCKED: &str = "cuMemHostAlloc: the GPU cannot reach page-locked memory at the host's addresses: \
                              its CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING is 0";

/// The host fills `m` with `input`, a kernel adds 1 to each word, and the host reads the sums, bit
/// for bit, from the one address the kernel was given, with no copy and no `sync` of its own.
fn fill_add_read(module: &Module, m: &mut Unified<u32>, input: &[u32]) {
    assert_eq!(m.len(), input.len());
    m.host_mut().unwrap().copy_from_slice(input);
    add_one(module, m.ptr(), input.len() as u32);
    let want: Vec<u32> = input.iter().map(|w| w.wrapping_add(1)).collect();
    assert_eq!(m.host().unwrap(), want);
    assert_eq!(m.host().unwrap().as_ptr() as u64, m.ptr().addr());
}

/// The host asks for `m`, of four words, as soon as a kernel that writes the first is launched, and
/// has what the kernel wrote, `DELAY` later: a read, and then a write.
fn host_waits(module: &Module, m: &mut Unified<u32>) {
    m.host_mut().unwrap().copy_from_slice(&[5, 1, 2, 3]);
    let started = Instant::now();
    delay(module, m.ptr());
    assert_eq!(m.host().unwrap(), [105, 1, 2, 3], "a read without a sync of its own is of the finished kernel");
    assert!(started.elapsed() >= DELAY * 5 / 6, "the read came after {:?}", started.elapsed());
    // The kernel has finished, so its add is not written over, nor does it write over the host's.
    m.host_mut().unwrap().copy_from_slice(&[5, 6, 7, 8]);
    delay(module, m.ptr());
    let started = Instant::now();
    let slice = m.host_mut().unwrap();
    assert_eq!(slice[0], 105, "{:?}", started.elapsed());
    slice[1] = 9;
    assert_eq!(m.host().unwrap(), [105, 9, 7, 8]);
}

/// Dropping `m` waits for a kernel launched on it.
fn drop_waits(module: &Module, m: Unified<u32>) {
    let started = Instant::now();
    delay(module, m.ptr());
    drop(m);
    assert!(started.elapsed() >= DELAY * 5 / 6, "dropped after {:?}", started.elapsed());
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
    fill_add_read(&module, &mut m, &input);
}

#[test]
fn managed_memory_made_with_no_zeros_is_filled_by_the_host_added_to_by_a_kernel_and_read_by_the_host() {
    let Some((_one, cuda, module)) = setup() else { return };
    let n = 1_000_003;
    let mut input = words(n, 5);
    input[0] = u32::MAX;
    // SAFETY: `fill_add_read` writes every word, with the host, before anything reads one.
    let mut m = unsafe { cuda.managed_uninit::<u32>(n) }.unwrap();
    assert_eq!(m.len(), n);
    fill_add_read(&module, &mut m, &input);
}

#[test]
fn the_host_waits_for_a_kernel_before_it_reads_or_writes_managed_memory() {
    let Some((_one, cuda, module)) = setup() else { return };
    host_waits(&module, &mut cuda.managed::<u32>(4).unwrap());
}

#[test]
fn dropping_managed_memory_waits_for_the_kernel_that_uses_it() {
    let Some((_one, cuda, module)) = setup() else { return };
    drop_waits(&module, cuda.managed::<u32>(1).unwrap());
    // A kernel that wrote freed memory would have failed.
    cuda.sync().unwrap();
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
fn page_locked_memory_is_filled_by_the_host_added_to_by_a_kernel_and_read_by_the_host() {
    let Some((_one, cuda, module)) = setup() else { return };
    let n = 1_000_003;
    let mut input = words(n, 2);
    input[0] = u32::MAX;
    let Some(mut m) = or_skip(cuda.pinned::<u32>(n), NO_PAGE_LOCKED) else { return };
    assert_eq!(m.len(), n);
    assert!(m.host().unwrap().iter().all(|&w| w == 0), "page-locked memory is zero until written");
    fill_add_read(&module, &mut m, &input);
}

#[test]
fn page_locked_memory_made_with_no_zeros_is_filled_by_the_host_added_to_by_a_kernel_and_read_by_the_host() {
    let Some((_one, cuda, module)) = setup() else { return };
    let n = 1_000_003;
    let mut input = words(n, 6);
    input[0] = u32::MAX;
    // SAFETY: `fill_add_read` writes every word, with the host, before anything reads one.
    let Some(mut m) = or_skip(unsafe { cuda.pinned_uninit::<u32>(n) }, NO_PAGE_LOCKED) else { return };
    assert_eq!(m.len(), n);
    fill_add_read(&module, &mut m, &input);
}

#[test]
fn the_host_waits_for_a_kernel_before_it_reads_or_writes_page_locked_memory() {
    let Some((_one, cuda, module)) = setup() else { return };
    let Some(mut m) = or_skip(cuda.pinned::<u32>(4), NO_PAGE_LOCKED) else { return };
    host_waits(&module, &mut m);
}

#[test]
fn dropping_page_locked_memory_waits_for_the_kernel_that_uses_it() {
    let Some((_one, cuda, module)) = setup() else { return };
    let Some(m) = or_skip(cuda.pinned::<u32>(1), NO_PAGE_LOCKED) else { return };
    drop_waits(&module, m);
    cuda.sync().unwrap();
}

#[test]
fn page_locked_memory_has_one_address_for_both_and_may_have_no_elements() {
    let Some((_one, cuda, _)) = setup() else { return };
    let Some(m) = or_skip(cuda.pinned::<u32>(4), NO_PAGE_LOCKED) else { return };
    assert_eq!(m.at(3).addr(), m.ptr().addr() + 12);
    assert_eq!(m.host().unwrap().as_ptr() as u64, m.ptr().addr());
    let none = cuda.pinned::<u64>(0).unwrap();
    assert!(none.is_empty());
    assert_eq!(none.host().unwrap(), [] as [u64; 0]);
}

#[test]
fn a_kernel_adds_to_the_memory_of_a_vec_and_the_vec_has_it_back() {
    let Some((_one, cuda, module)) = setup() else { return };
    let n = 1_000_003;
    let mut input = words(n, 3);
    input[0] = u32::MAX;
    let want: Vec<u32> = input.iter().map(|w| w.wrapping_add(1)).collect();
    let mut v = vec![0u32; n];
    let at = v.as_ptr() as u64;
    let Some(mut lent) = or_skip(cuda.lend(&mut v), NO_PAGEABLE) else { return };
    assert_eq!((lent.len(), lent.ptr().addr()), (n, at), "the kernel is given the Vec's own address");
    fill_add_read(&module, &mut lent, &input);
    drop(lent);
    assert_eq!(v, want, "the Vec holds what the kernel wrote");
}

#[test]
fn a_part_of_a_vec_or_a_box_is_lent_and_the_rest_of_it_is_not_touched() {
    let Some((_one, cuda, module)) = setup() else { return };
    let mut v = words(300, 4);
    let before = v.clone();
    let Some(lent) = or_skip(cuda.lend(&mut v[1..299]), NO_PAGEABLE) else { return };
    add_one(&module, lent.ptr(), 298);
    cuda.sync().unwrap();
    drop(lent);
    assert_eq!(v[0], before[0]);
    assert_eq!(v[299], before[299]);
    assert!((1..299).all(|i| v[i] == before[i].wrapping_add(1)));
    // A Box's array is lent as a Vec's is, and so is none at all. An array on a thread's stack
    // is not: the driver faults a kernel on one near the guard page below a stack.
    let mut a = Box::new([7u32; 16]);
    let lent = cuda.lend(&mut *a).unwrap();
    add_one(&module, lent.ptr(), 16);
    drop(lent);
    assert_eq!(*a, [8; 16]);
    let none = cuda.lend::<u32>(&mut []).unwrap();
    assert!(none.is_empty());
    assert_eq!(none.host().unwrap(), [] as [u32; 0]);
}

#[test]
fn the_host_waits_for_a_kernel_before_it_reads_or_writes_a_slice_lent() {
    let Some((_one, cuda, module)) = setup() else { return };
    let mut v = vec![0u32; 4];
    let Some(mut lent) = or_skip(cuda.lend(&mut v), NO_PAGEABLE) else { return };
    host_waits(&module, &mut lent);
}

#[test]
fn dropping_a_slice_lent_waits_for_the_kernel_and_the_host_has_the_slice_back() {
    let Some((_one, cuda, module)) = setup() else { return };
    let mut v = vec![5u32, 1, 2, 3];
    let Some(lent) = or_skip(cuda.lend(&mut v), NO_PAGEABLE) else { return };
    drop_waits(&module, lent);
    assert_eq!(v, [105, 1, 2, 3], "the kernel's write is in the Vec when the host has it back");
}

#[test]
fn page_locked_memory_is_not_moved_advised_or_placed_and_a_slice_lent_is_not_moved_or_placed() {
    let Some((_one, cuda, _)) = setup() else { return };
    let pinned = or_skip(cuda.pinned::<u32>(PAGE), NO_PAGE_LOCKED);
    let mut v = vec![0u32; PAGE];
    let lent = or_skip(cuda.lend(&mut v), NO_PAGEABLE);
    let said = |this, call, rule| format!("{call}: the driver {rule}: this is {this}");
    for (m, this) in [(pinned.as_ref(), "page-locked memory"), (lent.as_ref(), "a slice lent")] {
        let Some(m) = m else { continue };
        let prefetch = said(this, "cuMemPrefetchAsync", "prefetches managed memory alone");
        assert_eq!(m.prefetch(0..PAGE, Location::Gpu).unwrap_err().to_string(), prefetch);
        let placement = said(this, "cuMemRangeGetAttribute", "tells the place of managed memory alone");
        assert_eq!(m.placement(0..PAGE).unwrap_err().to_string(), placement);
    }
    if let Some(m) = pinned {
        let advise = said("page-locked memory", "cuMemAdvise", "advises managed and pageable memory alone");
        assert_eq!(m.advise(0..PAGE, Advice::ReadMostly).unwrap_err().to_string(), advise);
    }
}

#[test]
fn a_slice_lent_takes_each_advice_and_a_kernel_adds_to_it_after_each() {
    let Some((_one, cuda, module)) = setup() else { return };
    let n = 3 * PAGE;
    let input = words(n, 11);
    let mut v = input.clone();
    let Some(lent) = or_skip(cuda.lend(&mut v), NO_PAGEABLE) else { return };
    // The one page of the slice that is its own.
    let page = first_page(&lent)..first_page(&lent) + PAGE;
    let advice = [
        Advice::ReadMostly,
        Advice::NotReadMostly,
        Advice::PreferredLocation(Location::Gpu),
        Advice::PreferredLocation(Location::Host),
        Advice::NoPreferredLocation,
        Advice::AccessedBy(Location::Gpu),
        Advice::AccessedBy(Location::Host),
        Advice::NotAccessedBy(Location::Gpu),
        Advice::NotAccessedBy(Location::Host),
    ];
    let mut taken = 0u32;
    for advice in advice {
        let Some(()) = or_skip(lent.advise(page.clone(), advice), NO_ADVICE) else { continue };
        add_one(&module, lent.ptr(), n as u32);
        taken += 1;
    }
    drop(lent);
    let want: Vec<u32> = input.iter().map(|w| w.wrapping_add(taken)).collect();
    assert_eq!(v, want, "{taken} advice taken");
}

#[test]
fn a_kernel_adds_to_memory_prefetched_to_the_gpu_and_the_host_reads_it_prefetched_back() {
    let Some((_one, cuda, module)) = setup() else { return };
    let n = 1_000_003;
    let mut input = words(n, 5);
    input[0] = u32::MAX;
    let want: Vec<u32> = input.iter().map(|w| w.wrapping_add(1)).collect();
    let mut m = cuda.managed::<u32>(n).unwrap();
    m.host_mut().unwrap().copy_from_slice(&input);
    let Some(()) = or_skip(m.prefetch(0..n, Location::Gpu), NO_PREFETCH) else { return };
    add_one(&module, m.ptr(), n as u32);
    m.prefetch(0..n, Location::Host).unwrap();
    assert_eq!(m.host().unwrap(), want, "the sums, bit for bit, from the one address");
    assert_eq!(m.placement(0..n).unwrap().last_prefetch, Some(Location::Host));
}
