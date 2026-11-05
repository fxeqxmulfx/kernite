//! A lent element that has a guard page in its aligned 64 KiB is one the driver here faults a
//! kernel on, as GUIDE.md says. A test of its own, as a program of its own, as `sticky.rs` is: the
//! fault fails the GPU's context for every call after it. A driver that reaches such an element
//! fails this test, and then the sentence in GUIDE.md and the one in the docs of `Cuda::lend` go.

use kernite_cuda::{Cuda, Grid, Module, arg};

/// `bump`: `x[0] += 1`.
const BUMP: &str = "
.version 3.0
.target sm_30
.address_size 64

.visible .entry bump(.param .u64 x_param)
{
	.reg .u32 %v;
	.reg .u64 %x;

	ld.param.u64 %x, [x_param];
	ld.global.u32 %v, [%x];
	add.u32 %v, %v, 1;
	st.global.u32 [%x], %v;
	ret;
}
";

unsafe extern "C" {
    fn mmap(address: *mut u8, bytes: usize, protection: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn madvise(address: *mut u8, bytes: usize, advice: i32) -> i32;
}

#[test]
fn the_driver_faults_a_kernel_on_a_lent_element_with_a_guard_page_in_its_64_kib() {
    let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
    let module = Module::load(&cuda, BUMP, &["bump"], &[]).unwrap();
    const PAGE: usize = 4096;
    // SAFETY: a mapping of our own, 1 MiB, read and write (3), private and anonymous (0x22).
    let mapping = unsafe { mmap(std::ptr::null_mut(), 1 << 20, 3, 0x22, -1, 0) };
    assert_ne!(mapping as isize, -1, "mmap");
    // The element is on the sixth page of an aligned 64 KiB, and the guard page is the seventh.
    let chunk = (mapping as usize).next_multiple_of(16 * PAGE);
    let element = chunk + 5 * PAGE;
    // SAFETY: `element` is inside the mapping, which is ours and aligned for a `u32`, and anonymous
    // memory is zero, which is a `u32`. Nothing else holds a slice of it.
    let word = unsafe { std::slice::from_raw_parts_mut(element as *mut u32, 1) };
    word[0] = 7;
    // SAFETY: a page of the mapping. 102 is MADV_GUARD_INSTALL, which a Linux before 6.13 refuses.
    if unsafe { madvise((element + PAGE) as *mut u8, PAGE, 102) } != 0 {
        eprintln!("no guard pages: skipped");
        return;
    }
    let lent = match cuda.lend(word) {
        Ok(lent) => lent,
        Err(e) => return eprintln!("the GPU cannot: skipped. {e}"),
    };
    // SAFETY: `bump` takes a pointer and adds 1 to the word there, which `lent` has. The host holds
    // no slice of it. The launch starts, and the kernel fails on the GPU, which this test is for.
    unsafe { module.launch(0, Grid::new(1, 1), &[arg(&lent.ptr())]) }.unwrap();
    let failed = cuda.sync().unwrap_err();
    assert_eq!(failed.message, "CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered");
    assert_eq!(lent.host().unwrap_err(), failed, "every call after it fails the same");
    drop(lent);
    assert_eq!(word[0], 7, "the kernel wrote nothing");
}
