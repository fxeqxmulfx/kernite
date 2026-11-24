//! Memory the host and a kernel both reach, on the GPU here, if there is one: what each sees of
//! it, and when the host may touch it. Each of its four kinds, managed, page-locked, lent and
//! registered, is held to the same three rules, below.
//!
//! The tests run beside each other, so one holds a slice of managed memory while a kernel of
//! another runs, which a launch's contract allows: the kernel does not reach that memory.
//! `managed` makes none on a machine where a GPU cannot reach managed memory while the host
//! does, and the tests of managed memory skip there, with the words of GUIDE.md.

use std::ffi::{CString, c_char};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::time::Duration;

use kernite_cuda::{Advice, Cuda, Error, Grid, Location, Module, Ptr, Unified, arg};

mod common;
use common::{NO_ADVICE, PAGE, first_page, managed_or_skip, or_skip};
mod gpu;
use gpu::{peek, ptx};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// Two kernels, written by hand. `add_one`: `x[i] += 1` for each `i < n`, a thread each, wrapping.
/// `delay`: one thread waits `ns` nanoseconds on the GPU's global timer, then `x[0] += 100`.
const PTX: &str = "
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
	call (%start), clock_start, ();
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

/// What a test of the GPU is given: the GPU and the kernels.
type Setup = (Cuda, Module);

/// What `Setup` has; or `None` where there is no GPU, after it says so.
fn setup() -> Result<Option<Setup>> {
    let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return Ok(None) };
    let module = Module::load(&cuda, &ptx(PTX), &["add_one", "delay"], &[])?;
    Ok(Some((cuda, module)))
}

/// Pseudo-random words, the same for the same seed.
fn words(n: usize, seed: u32) -> Vec<u32> {
    let mut s = seed;
    Vec::from_fn(n, |_| {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    })
}

/// Launches `add_one` over the `n` words at `x`.
fn add_one(module: &Module, x: Ptr<u32>, n: u32) -> Result {
    // SAFETY: `add_one` takes a pointer and a `u32`, and writes only the first `n` words there,
    // which the caller's memory has. The caller holds no slice of it.
    unsafe { module.launch(0, Grid::new(n.div_ceil(256), 256), &[arg(&x), arg(&n)]) }?;
    Ok(())
}

/// Launches `delay` on the word at `x`, which waits for `DELAY` and adds 100 to the word.
fn delay(module: &Module, x: Ptr<u32>) -> Result {
    let ns = DELAY.as_nanos() as u64;
    // SAFETY: `delay` takes a pointer and a `u64`, and writes the one word at the pointer, which
    // the caller's memory has. The caller holds no slice of it.
    unsafe { module.launch(1, Grid::new(1, 1), &[arg(&x), arg(&ns)]) }?;
    Ok(())
}

/// GUIDE.md's words for a GPU that cannot be lent a slice.
const NO_PAGEABLE: &str = "cuDeviceGetAttribute: the GPU cannot reach pageable memory, a slice of the host's: \
                           its CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS is 0";

/// GUIDE.md's words for a GPU that cannot make page-locked memory.
const NO_PAGE_LOCKED: &str = "cuMemHostAlloc: the GPU cannot reach page-locked memory at the host's addresses: \
                              its CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING is 0";

/// GUIDE.md's words for a GPU that cannot page-lock memory the host has, and for one that cannot reach it at the host's
/// addresses.
const NO_REGISTER: [&str; 2] = [
    "cuMemHostRegister: the GPU cannot page-lock memory the host has: \
     its CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED is 0",
    "cuMemHostRegister: the GPU cannot reach page-locked memory the host has at the host's addresses: \
     its CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM is 0",
];

/// `cuda.register(slice)`; or `None` where the GPU cannot, after it says so in one of GUIDE.md's words, and the test is
/// skipped, as for a machine with no GPU.
fn register_or_skip<'a>(cuda: &Cuda, slice: &'a mut [u32]) -> Result<Option<Unified<'a, u32>>> {
    match cuda.register(slice) {
        Err(e) if NO_REGISTER.contains(&e.to_string().as_str()) => {
            eprintln!("the GPU cannot: skipped. {e}");
            Ok(None)
        }
        made => Ok(Some(made?)),
    }
}

/// The host fills `m` with `input`, a kernel adds 1 to each word, and the host reads the sums, bit
/// for bit, from the one address the kernel was given, with no copy and no `sync` of its own.
fn fill_add_read(module: &Module, m: &mut Unified<u32>, input: &[u32]) -> Result {
    assert_eq!(m.len(), input.len());
    m.host_mut()?.copy_from_slice(input);
    add_one(module, m.ptr(), input.len() as u32)?;
    let want: Vec<u32> = input.iter().map(|w| w.wrapping_add(1)).collect();
    assert_eq!(m.host()?, want);
    assert_eq!(m.host()?.as_ptr() as u64, m.ptr().addr());
    Ok(())
}

/// The host asks for `m`, of four words, as soon as a kernel that writes the first is launched, and
/// has what the kernel wrote, `DELAY` later: a read, and then a write.
fn host_waits(module: &Module, m: &mut Unified<u32>) -> Result {
    m.host_mut()?.copy_from_slice(&[5, 1, 2, 3]);
    delay(module, m.ptr())?;
    assert_eq!(m.host()?, [105, 1, 2, 3], "a read without a sync of its own is of the finished kernel");
    // The kernel has finished, so its add is not written over, nor does it write over the host's.
    m.host_mut()?.copy_from_slice(&[5, 6, 7, 8]);
    delay(module, m.ptr())?;
    let [first, second, ..] = m.host_mut()? else { return Err("four words".into()) };
    assert_eq!(*first, 105, "a write without a sync of its own is of the finished kernel");
    *second = 9;
    assert_eq!(m.host()?, [105, 9, 7, 8]);
    Ok(())
}

/// Dropping `m` waits for a kernel launched on it, and for the work given to the GPU before the drop: `add_one` adds 1
/// to a page-locked word of its own, after `delay` is done, and the word is 1 when the drop returns.
fn drop_waits(cuda: &Cuda, module: &Module, m: Unified<u32>) -> Result {
    // The word is made before the kernels are launched: a call that makes memory may wait of itself.
    let done = cuda.pinned::<u32>(1)?;
    delay(module, m.ptr())?;
    add_one(module, done.ptr(), 1)?;
    drop(m);
    assert_eq!(peek(&done), 1, "the drop returned before the kernels launched before it were done");
    Ok(())
}

#[test]
fn a_managed_buffer_is_filled_by_the_host_added_to_by_a_kernel_and_read_by_the_host() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    // Not a whole number of blocks, and words that wrap.
    let n = 1_000_003;
    let mut input = words(n, 1);
    input[0] = u32::MAX;
    let Some(mut m) = managed_or_skip(cuda.managed::<u32>(n)) else { return Ok(()) };
    assert_eq!(m.len(), n);
    assert!(m.host()?.iter().all(|&w| w == 0), "managed memory is zero until written");
    fill_add_read(&module, &mut m, &input)?;
    Ok(())
}

#[test]
fn managed_memory_made_with_no_zeros_is_filled_by_the_host_added_to_by_a_kernel_and_read_by_the_host() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let n = 1_000_003;
    let mut input = words(n, 5);
    input[0] = u32::MAX;
    // SAFETY: `fill_add_read` writes every word, with the host, before anything reads one.
    let Some(mut m) = managed_or_skip(unsafe { cuda.managed_uninit::<u32>(n) }) else { return Ok(()) };
    assert_eq!(m.len(), n);
    fill_add_read(&module, &mut m, &input)?;
    Ok(())
}

#[test]
fn the_host_waits_for_a_kernel_before_it_reads_or_writes_managed_memory() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let Some(mut m) = managed_or_skip(cuda.managed::<u32>(4)) else { return Ok(()) };
    host_waits(&module, &mut m)?;
    Ok(())
}

#[test]
fn dropping_managed_memory_waits_for_the_kernel_that_uses_it() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let Some(m) = managed_or_skip(cuda.managed::<u32>(1)) else { return Ok(()) };
    drop_waits(&cuda, &module, m)?;
    // A kernel that wrote freed memory would have failed.
    cuda.sync()?;
    Ok(())
}

#[test]
fn managed_memory_has_one_address_for_both_and_may_have_no_elements() -> Result {
    let Some((cuda, _)) = setup()? else { return Ok(()) };
    let Some(m) = managed_or_skip(cuda.managed::<u32>(4)) else { return Ok(()) };
    assert_eq!(m.at(3).ok_or("at gave None")?.addr(), m.ptr().addr() + 12);
    assert_eq!(m.at(4).ok_or("at gave None")?.addr(), m.ptr().addr() + 16);
    assert_eq!(format!("{m:?}"), format!("Unified(4 elements at {:#x})", m.ptr().addr()));
    assert!(m.at(5).is_none(), "an address past the end is none");
    let none = cuda.managed::<u64>(0)?;
    assert!(none.is_empty());
    assert_eq!(none.host()?, [] as [u64; 0]);
    Ok(())
}

#[test]
fn page_locked_memory_is_filled_by_the_host_added_to_by_a_kernel_and_read_by_the_host() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let n = 1_000_003;
    let mut input = words(n, 2);
    input[0] = u32::MAX;
    let Some(mut m) = or_skip(cuda.pinned::<u32>(n), NO_PAGE_LOCKED) else { return Ok(()) };
    assert_eq!(m.len(), n);
    assert!(m.host()?.iter().all(|&w| w == 0), "page-locked memory is zero until written");
    fill_add_read(&module, &mut m, &input)?;
    Ok(())
}

#[test]
fn page_locked_memory_made_with_no_zeros_is_filled_by_the_host_added_to_by_a_kernel_and_read_by_the_host() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let n = 1_000_003;
    let mut input = words(n, 6);
    input[0] = u32::MAX;
    // SAFETY: `fill_add_read` writes every word, with the host, before anything reads one.
    let Some(mut m) = or_skip(unsafe { cuda.pinned_uninit::<u32>(n) }, NO_PAGE_LOCKED) else { return Ok(()) };
    assert_eq!(m.len(), n);
    fill_add_read(&module, &mut m, &input)?;
    Ok(())
}

#[test]
fn the_host_waits_for_a_kernel_before_it_reads_or_writes_page_locked_memory() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let Some(mut m) = or_skip(cuda.pinned::<u32>(4), NO_PAGE_LOCKED) else { return Ok(()) };
    host_waits(&module, &mut m)?;
    Ok(())
}

#[test]
fn dropping_page_locked_memory_waits_for_the_kernel_that_uses_it() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let Some(m) = or_skip(cuda.pinned::<u32>(1), NO_PAGE_LOCKED) else { return Ok(()) };
    drop_waits(&cuda, &module, m)?;
    cuda.sync()?;
    Ok(())
}

#[test]
fn page_locked_memory_has_one_address_for_both_and_may_have_no_elements() -> Result {
    let Some((cuda, _)) = setup()? else { return Ok(()) };
    let Some(m) = or_skip(cuda.pinned::<u32>(4), NO_PAGE_LOCKED) else { return Ok(()) };
    assert_eq!(m.at(3).ok_or("at gave None")?.addr(), m.ptr().addr() + 12);
    assert_eq!(m.host()?.as_ptr() as u64, m.ptr().addr());
    let none = cuda.pinned::<u64>(0)?;
    assert!(none.is_empty());
    assert_eq!(none.host()?, [] as [u64; 0]);
    Ok(())
}

#[test]
fn a_kernel_adds_to_the_memory_of_a_vec_and_the_vec_has_it_back() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let n = 1_000_003;
    let mut input = words(n, 3);
    input[0] = u32::MAX;
    let want: Vec<u32> = input.iter().map(|w| w.wrapping_add(1)).collect();
    let mut v = vec![0u32; n];
    let at = v.as_ptr() as u64;
    let Some(mut lent) = or_skip(cuda.lend(&mut v), NO_PAGEABLE) else { return Ok(()) };
    assert_eq!((lent.len(), lent.ptr().addr()), (n, at), "the kernel is given the Vec's own address");
    fill_add_read(&module, &mut lent, &input)?;
    drop(lent);
    assert_eq!(v, want, "the Vec holds what the kernel wrote");
    Ok(())
}

#[test]
fn a_part_of_a_vec_or_a_box_is_lent_and_the_rest_of_it_is_not_touched() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let mut v = words(300, 4);
    let before = v.clone();
    let Some(lent) = or_skip(cuda.lend(&mut v[1..299]), NO_PAGEABLE) else { return Ok(()) };
    add_one(&module, lent.ptr(), 298)?;
    cuda.sync()?;
    drop(lent);
    assert_eq!(v[0], before[0]);
    assert_eq!(v[299], before[299]);
    assert!((1..299).all(|i| v[i] == before[i].wrapping_add(1)));
    // A Box's array is lent as a Vec's is, and so is none at all. An array on a thread's stack
    // is not: the driver faults a kernel on one near the guard page below a stack.
    let mut a = Box::new([7u32; 16]);
    let lent = cuda.lend(&mut *a)?;
    add_one(&module, lent.ptr(), 16)?;
    drop(lent);
    assert_eq!(*a, [8; 16]);
    let none = cuda.lend::<u32>(&mut [])?;
    assert!(none.is_empty());
    assert_eq!(none.host()?, [] as [u32; 0]);
    Ok(())
}

#[test]
fn the_host_waits_for_a_kernel_before_it_reads_or_writes_a_slice_lent() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let mut v = vec![0u32; 4];
    let Some(mut lent) = or_skip(cuda.lend(&mut v), NO_PAGEABLE) else { return Ok(()) };
    host_waits(&module, &mut lent)?;
    Ok(())
}

#[test]
fn dropping_a_slice_lent_waits_for_the_kernel_and_the_host_has_the_slice_back() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let mut v = vec![5u32, 1, 2, 3];
    let Some(lent) = or_skip(cuda.lend(&mut v), NO_PAGEABLE) else { return Ok(()) };
    drop_waits(&cuda, &module, lent)?;
    assert_eq!(v, [105, 1, 2, 3], "the kernel's write is in the Vec when the host has it back");
    Ok(())
}

/// `host_waits` and then `drop_waits` on `m`, with each kernel launched in the `run` of a stream: the host and the drop
/// wait for the kernel of a stream, as they do for one of the null stream, which the stream is blocking for, and the
/// host sees what the kernel wrote.
fn waits_for_a_stream(cuda: &Cuda, module: &Module, mut m: Unified<u32>) -> Result {
    let stream = cuda.stream()?;
    // SAFETY: each kernel writes the first word of `m` alone, or the word of `drop_waits` that is its own, which no
    // other kernel reaches: those of the other tests are on their own memory. The host holds no slice of `m` while one
    // runs, as `host_waits` and `drop_waits` hold none.
    unsafe {
        stream.run(|| host_waits(module, &mut m))?;
        stream.run(|| drop_waits(cuda, module, m))
    }
}

#[test]
fn the_host_and_the_drop_wait_for_a_kernel_of_a_stream_in_managed_memory() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let Some(m) = managed_or_skip(cuda.managed::<u32>(4)) else { return Ok(()) };
    waits_for_a_stream(&cuda, &module, m)?;
    // A kernel that wrote freed memory would have failed.
    cuda.sync()?;
    Ok(())
}

#[test]
fn the_host_and_the_drop_wait_for_a_kernel_of_a_stream_in_page_locked_memory() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let Some(m) = or_skip(cuda.pinned::<u32>(4), NO_PAGE_LOCKED) else { return Ok(()) };
    waits_for_a_stream(&cuda, &module, m)?;
    cuda.sync()?;
    Ok(())
}

#[test]
fn the_host_and_the_drop_wait_for_a_kernel_of_a_stream_in_a_slice_lent_and_the_host_has_the_slice_back() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let mut v = vec![0u32; 4];
    let Some(lent) = or_skip(cuda.lend(&mut v), NO_PAGEABLE) else { return Ok(()) };
    waits_for_a_stream(&cuda, &module, lent)?;
    // `host_waits` leaves [105, 9, 7, 8], and the kernel that `drop_waits` launches adds 100 to the first word.
    assert_eq!(v, [205, 9, 7, 8], "the kernels' writes are in the Vec when the host has it back");
    Ok(())
}

#[test]
fn the_host_and_the_drop_wait_for_a_kernel_of_a_stream_in_a_slice_registered_and_the_host_has_the_slice_back() -> Result
{
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let mut v = vec![0u32; 4];
    let Some(registered) = register_or_skip(&cuda, &mut v)? else { return Ok(()) };
    waits_for_a_stream(&cuda, &module, registered)?;
    // `host_waits` leaves [105, 9, 7, 8], and the kernel that `drop_waits` launches adds 100 to the first word.
    assert_eq!(v, [205, 9, 7, 8], "the kernels' writes are in the Vec when the host has it back");
    Ok(())
}

#[test]
fn a_kernel_adds_to_the_memory_of_a_vec_registered_and_the_vec_has_it_back() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let n = 1_000_003;
    let mut input = words(n, 6);
    input[0] = u32::MAX;
    let want: Vec<u32> = input.iter().map(|w| w.wrapping_add(1)).collect();
    let mut v = vec![0u32; n];
    let at = v.as_ptr() as u64;
    let Some(mut registered) = register_or_skip(&cuda, &mut v)? else { return Ok(()) };
    assert_eq!((registered.len(), registered.ptr().addr()), (n, at), "the kernel is given the Vec's own address");
    fill_add_read(&module, &mut registered, &input)?;
    drop(registered);
    assert_eq!(v, want, "the Vec holds what the kernel wrote");
    // Dropping it unlocked the Vec: the driver takes it once more, as it does not memory that is page-locked.
    let again = register_or_skip(&cuda, &mut v)?.ok_or("the GPU registered it before")?;
    assert_eq!(again.len(), n);
    Ok(())
}

#[test]
fn a_part_of_a_vec_or_a_box_is_registered_and_the_rest_of_it_is_not_touched() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let mut v = words(300, 7);
    let before = v.clone();
    let Some(registered) = register_or_skip(&cuda, &mut v[1..299])? else { return Ok(()) };
    add_one(&module, registered.ptr(), 298)?;
    cuda.sync()?;
    drop(registered);
    assert_eq!(v[0], before[0]);
    assert_eq!(v[299], before[299]);
    assert!((1..299).all(|i| v[i] == before[i].wrapping_add(1)));
    // A Box's array is registered as a Vec's is, and so is none at all.
    let mut a = Box::new([7u32; 16]);
    let registered = cuda.register(&mut *a)?;
    add_one(&module, registered.ptr(), 16)?;
    drop(registered);
    assert_eq!(*a, [8; 16]);
    let none = cuda.register::<u32>(&mut [])?;
    assert!(none.is_empty());
    assert_eq!(none.host()?, [] as [u32; 0]);
    Ok(())
}

#[test]
fn two_parts_of_a_vec_are_registered_at_once_and_each_is_unlocked_by_its_own_drop() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let mut v = words(300, 8);
    let before = v.clone();
    let (left, right) = v.split_at_mut(150);
    let Some(left) = register_or_skip(&cuda, left)? else { return Ok(()) };
    let right = cuda.register(right)?;
    add_one(&module, left.ptr(), 150)?;
    add_one(&module, right.ptr(), 150)?;
    drop(left);
    // The right half is registered still, and a kernel adds to it.
    add_one(&module, right.ptr(), 150)?;
    drop(right);
    assert!((0..150).all(|i| v[i] == before[i].wrapping_add(1)));
    assert!((150..300).all(|i| v[i] == before[i].wrapping_add(2)));
    // Both are the host's again, and the driver takes each once more.
    let (left, right) = v.split_at_mut(150);
    let _both = (cuda.register(left)?, cuda.register(right)?);
    Ok(())
}

#[test]
fn the_host_waits_for_a_kernel_before_it_reads_or_writes_a_slice_registered() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let mut v = vec![0u32; 4];
    let Some(mut registered) = register_or_skip(&cuda, &mut v)? else { return Ok(()) };
    host_waits(&module, &mut registered)?;
    Ok(())
}

#[test]
fn dropping_a_slice_registered_waits_for_the_kernel_and_the_host_has_the_slice_back() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let mut v = vec![5u32, 1, 2, 3];
    let Some(registered) = register_or_skip(&cuda, &mut v)? else { return Ok(()) };
    drop_waits(&cuda, &module, registered)?;
    assert_eq!(v, [105, 1, 2, 3], "the kernel's write is in the Vec when the host has it back");
    Ok(())
}

/// Holds that `made` is the driver's refusal in `words`, under the name `cuMemHostRegister`.
fn refused(made: Result<Unified<u32>, Error>, words: &str) -> Result {
    let e = made.err().ok_or("register gave Ok")?;
    assert_eq!(e.to_string(), format!("cuMemHostRegister: {words}"));
    Ok(())
}

#[test]
fn the_driver_refuses_memory_that_is_registered_already_or_that_it_made() -> Result {
    let Some((cuda, _)) = setup()? else { return Ok(()) };
    // A slice registered and then forgotten stays registered, and its memory may not be freed: it is leaked.
    let leaked = Vec::leak(words(300, 9));
    let Some(forgotten) = register_or_skip(&cuda, &mut *leaked)? else { return Ok(()) };
    std::mem::forget(forgotten);
    let already = "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED (712): \
                   part or all of the requested memory range is already mapped";
    refused(cuda.register(&mut *leaked), already)?;
    refused(cuda.register(&mut leaked[100..200]), already)?;
    refused(cuda.register(&mut leaked[299..]), already)?;
    // The driver made page-locked memory and managed memory, and takes neither.
    let invalid = "CUDA_ERROR_INVALID_VALUE (1): invalid argument";
    if let Some(mut pinned) = or_skip(cuda.pinned::<u32>(300), NO_PAGE_LOCKED) {
        refused(cuda.register(pinned.host_mut()?), invalid)?;
    }
    if let Some(mut managed) = managed_or_skip(cuda.managed::<u32>(300)) {
        refused(cuda.register(managed.host_mut()?), invalid)?;
    }
    Ok(())
}

unsafe extern "C" {
    fn mmap(address: *mut u8, bytes: usize, protection: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn statfs(path: *const c_char, buffer: *mut [i64; 15]) -> i32;
}

/// What `statfs` says is the type of the file system of ext4, as `f_type`.
const EXT4: i64 = 0xEF53;

/// A page of the file at `file`, mapped for reading and writing: shared, as MAP_SHARED (1) maps it, or private, as
/// MAP_PRIVATE (2) does. The mapping lasts as long as the program.
fn map(file: &File, flags: i32) -> Result<&'static mut [u32]> {
    // SAFETY: a mapping of our own, of the first page of the file, read and write (3), and `flags`.
    let at = unsafe { mmap(std::ptr::null_mut(), PAGE * 4, 3, flags, file.as_raw_fd(), 0) };
    assert_ne!(at as isize, -1, "mmap");
    // SAFETY: the mapping is `PAGE` words, aligned for `u32`, and the program's own for as long as it runs. Nothing
    // else holds a slice of it.
    Ok(unsafe { std::slice::from_raw_parts_mut(at.cast::<u32>(), PAGE) })
}

/// GUIDE.md says what the driver here does with a mapping of a file on ext4, and the test holds it there, and
/// skips where the file is on another file system: a shared mapping is refused, in the words the driver refuses
/// memory it made in, and a private one is taken, its page copied as it is locked, so that a kernel's write is in the
/// mapping and not in the file.
#[test]
fn a_shared_mapping_of_a_file_is_refused_and_a_private_one_is_taken_with_its_pages_copied() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("mapping.bin");
    let mut page = vec![0u8; PAGE * 4];
    page[..4].copy_from_slice(&7u32.to_le_bytes());
    std::fs::write(&path, &page)?;
    let (c_path, mut fs) = (CString::new(path.as_os_str().as_bytes())?, [0i64; 15]);
    // SAFETY: the path ends in a NUL, and `fs` is as big as a `struct statfs`, 15 words of 8 bytes.
    let said = unsafe { statfs(c_path.as_ptr(), &mut fs) };
    assert_eq!(said, 0, "statfs");
    if fs[0] != EXT4 {
        eprintln!("the file is not on ext4, where GUIDE.md says what the driver does: skipped");
        return Ok(());
    }
    let file = OpenOptions::new().read(true).write(true).open(&path)?;
    let private = map(&file, 2)?;
    let Some(registered) = register_or_skip(&cuda, &mut *private)? else { return Ok(()) };
    add_one(&module, registered.ptr(), 1)?;
    drop(registered);
    assert_eq!(private[0], 8, "the kernel's write is in the mapping, which the host reads");
    let in_file = std::fs::read(&path)?;
    assert_eq!(in_file[..4], 7u32.to_le_bytes(), "and the file has the old word");
    let word = u32::from_le_bytes([in_file[0], in_file[1], in_file[2], in_file[3]]);
    eprintln!(
        "a private mapping of a file on ext4 is registered: its first word is {}, and the file's is {word}",
        private[0]
    );
    let made = cuda.register(map(&file, 1)?);
    let said = made.as_ref().err().map_or("registered".to_string(), ToString::to_string);
    eprintln!("a shared mapping of a file on ext4: {said}");
    refused(made, "CUDA_ERROR_INVALID_VALUE (1): invalid argument")?;
    std::fs::remove_file(&path)?;
    Ok(())
}

#[test]
fn page_locked_memory_made_or_registered_is_not_moved_advised_or_placed_and_a_slice_lent_is_not_moved_or_placed()
-> Result {
    let Some((cuda, _)) = setup()? else { return Ok(()) };
    let pinned = or_skip(cuda.pinned::<u32>(PAGE), NO_PAGE_LOCKED);
    let (mut v, mut w) = (vec![0u32; PAGE], vec![0u32; PAGE]);
    let lent = or_skip(cuda.lend(&mut v), NO_PAGEABLE);
    let registered = register_or_skip(&cuda, &mut w)?;
    let said = |this, call, rule| format!("{call}: the driver {rule}: this is {this}");
    let locked = "page-locked memory";
    for (m, this) in [(pinned.as_ref(), locked), (registered.as_ref(), locked), (lent.as_ref(), "a slice lent")] {
        let Some(m) = m else { continue };
        let prefetch = said(this, "cuMemPrefetchAsync", "prefetches managed memory alone");
        assert_eq!(m.prefetch(0..PAGE, Location::Gpu).err().ok_or("prefetch gave Ok")?.to_string(), prefetch);
        let placement = said(this, "cuMemRangeGetAttribute", "tells the place of managed memory alone");
        assert_eq!(m.placement(0..PAGE).err().ok_or("placement gave Ok")?.to_string(), placement);
    }
    for m in [pinned, registered].into_iter().flatten() {
        let advise = said(locked, "cuMemAdvise", "advises managed and pageable memory alone");
        assert_eq!(m.advise(0..PAGE, Advice::ReadMostly).err().ok_or("advise gave Ok")?.to_string(), advise);
    }
    Ok(())
}

#[test]
fn a_slice_lent_takes_each_advice_and_a_kernel_adds_to_it_after_each() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let n = 3 * PAGE;
    let input = words(n, 11);
    let mut v = input.clone();
    let Some(lent) = or_skip(cuda.lend(&mut v), NO_PAGEABLE) else { return Ok(()) };
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
        add_one(&module, lent.ptr(), n as u32)?;
        taken += 1;
    }
    drop(lent);
    let want: Vec<u32> = input.iter().map(|w| w.wrapping_add(taken)).collect();
    assert_eq!(v, want, "{taken} advice taken");
    Ok(())
}

#[test]
fn a_kernel_adds_to_memory_prefetched_to_the_gpu_and_the_host_reads_it_prefetched_back() -> Result {
    let Some((cuda, module)) = setup()? else { return Ok(()) };
    let n = 1_000_003;
    let mut input = words(n, 5);
    input[0] = u32::MAX;
    let want: Vec<u32> = input.iter().map(|w| w.wrapping_add(1)).collect();
    let Some(mut m) = managed_or_skip(cuda.managed::<u32>(n)) else { return Ok(()) };
    m.host_mut()?.copy_from_slice(&input);
    m.prefetch(0..n, Location::Gpu)?;
    add_one(&module, m.ptr(), n as u32)?;
    m.prefetch(0..n, Location::Host)?;
    assert_eq!(m.host()?, want, "the sums, bit for bit, from the one address");
    assert_eq!(m.placement(0..n)?.last_prefetch, Some(Location::Host));
    Ok(())
}
