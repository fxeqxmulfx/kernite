//! A lent element that has a guard page in its aligned 64 KiB is one the driver here faults a kernel on, as GUIDE.md
//! says, and a registered one is not. A test of its own, as a program of its own, as `sticky.rs` is: the fault fails
//! the GPU's context for every call after it, so the registered element is tried first. A driver that reaches a lent
//! element fails this test, and then the sentence in GUIDE.md and the one in the docs of `Cuda::lend` go; and a driver
//! that faults on a registered one fails it, and then the sentence of GUIDE.md about `Cuda::register` goes.

use kernite_cuda::{Cuda, Grid, Module, Ptr, arg};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

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

const PAGE: usize = 4096;

/// The word on the sixth page of the `n`th aligned 64 KiB of the mapping at `mapping`, where the seventh page is a
/// guard page, with `7` in it; or `None` where this Linux has no guard pages, as one before 6.13.
fn guarded(mapping: usize, n: usize) -> Option<&'static mut u32> {
    let element = (mapping.next_multiple_of(16 * PAGE) + n * 16 * PAGE) + 5 * PAGE;
    // SAFETY: `element` is inside the mapping, which is ours and lasts as long as the program, and aligned for a `u32`,
    // and anonymous memory is zero, which is a `u32`. Nothing else holds a slice of it.
    let word = unsafe { &mut *(element as *mut u32) };
    *word = 7;
    // SAFETY: a page of the mapping. 102 is MADV_GUARD_INSTALL, which a Linux before 6.13 refuses.
    (unsafe { madvise((element + PAGE) as *mut u8, PAGE, 102) } == 0).then_some(word)
}

/// Launches `bump` on the word at `x`.
fn bump(module: &Module, x: Ptr<u32>) -> Result {
    // SAFETY: `bump` takes a pointer and adds 1 to the word there, which the caller's memory has. The caller holds no
    // slice of it. The launch starts, and the kernel may fail on the GPU, which this test is for.
    unsafe { module.launch(0, Grid::new(1, 1), &[arg(&x)]) }?;
    Ok(())
}

#[test]
fn the_driver_faults_a_kernel_on_a_lent_element_with_a_guard_page_in_its_64_kib_and_not_on_a_registered_one() -> Result
{
    let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return Ok(()) };
    let module = Module::load(&cuda, BUMP, &["bump"], &[])?;
    // SAFETY: a mapping of our own, 1 MiB, read and write (3), private and anonymous (0x22).
    let mapping = unsafe { mmap(std::ptr::null_mut(), 1 << 20, 3, 0x22, -1, 0) };
    assert_ne!(mapping as isize, -1, "mmap");
    let (Some(registered), Some(lent)) = (guarded(mapping as usize, 0), guarded(mapping as usize, 1)) else {
        eprintln!("no guard pages: skipped");
        return Ok(());
    };
    let ran = match cuda.register(std::slice::from_mut(registered)) {
        Ok(memory) => {
            bump(&module, memory.ptr())?;
            true
        }
        Err(e) => {
            eprintln!("the GPU cannot register: skipped. {e}");
            false
        }
    };
    // Dropping it waited for the kernel.
    assert!(!ran || *registered == 8, "the kernel did not add 1 to the registered element");
    let memory = match cuda.lend(std::slice::from_mut(lent)) {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("the GPU cannot: skipped. {e}");
            return Ok(());
        }
    };
    bump(&module, memory.ptr())?;
    let failed = cuda.sync().err().ok_or("sync gave Ok")?;
    assert_eq!(failed.message, "CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered");
    assert_eq!(memory.host().err().ok_or("host gave Ok")?, failed, "every call after it fails the same");
    drop(memory);
    assert_eq!(*lent, 7, "the kernel wrote nothing");
    Ok(())
}
