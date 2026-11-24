//! Dropping a buffer that reserved addresses waits for the kernels that use it, and then unmaps its memory: the driver
//! does not wait for a kernel before it unmaps what the kernel reaches, and the kernel then fails with
//! `CUDA_ERROR_ILLEGAL_ADDRESS (700)`, as does each call after it. A test of its own, as a program of its own, as
//! `sticky.rs` is: a drop that did not wait would fail the GPU's context for every test beside it.

use kernite_cuda::{Cuda, Grid, Module, arg};

mod gpu;
use gpu::{peek, ptx};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// `later(x, flag, nanos)`: one thread waits `nanos` nanoseconds on the GPU's global timer, then writes 7 to `*x`, and
/// then 1 to `*flag`.
const PTX: &str = "
.visible .entry later(.param .u64 x_param, .param .u64 flag_param, .param .u64 nanos_param)
{
	.reg .u64 %x, %flag, %nanos, %start, %now, %gone;
	.reg .u32 %seven, %one;
	.reg .pred %early;

	ld.param.u64 %x, [x_param];
	ld.param.u64 %flag, [flag_param];
	ld.param.u64 %nanos, [nanos_param];
	cvta.to.global.u64 %x, %x;
	cvta.to.global.u64 %flag, %flag;
	call (%start), clock_start, ();
LATER:
	mov.u64 %now, %globaltimer;
	sub.u64 %gone, %now, %start;
	setp.lt.u64 %early, %gone, %nanos;
	@%early bra LATER;
	mov.u32 %seven, 7;
	st.volatile.global.u32 [%x], %seven;
	mov.u32 %one, 1;
	st.volatile.global.u32 [%flag], %one;
	ret;
}
";

/// How long the kernel waits before it writes: almost half a second, while the host drops the buffer at once.
const A_LITTLE_WHILE: u64 = 400_000_000;

#[test]
fn dropping_a_buffer_waits_for_the_kernel_that_writes_it_and_then_unmaps_it() -> Result {
    let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return Ok(()) };
    let Ok(mut cache) = cuda.reserve::<u32>(1 << 28).inspect_err(|e| eprintln!("the GPU cannot: skipped. {e}")) else {
        return Ok(());
    };
    let module = Module::load(&cuda, &ptx(PTX), &["later"], &[])?;
    let mut flag = cuda.pinned::<u32>(1)?;
    cache.grow(10_000_000)?;
    let (x_at, flag_at) = (cache.ptr(), flag.ptr());
    // SAFETY: `later` takes two pointers and a `u64`: the first word of the buffer and the page-locked flag, which it
    // alone writes, and the buffer and the flag are held by this test until after the kernel is waited for.
    unsafe { module.launch(0, Grid::new(1, 1), &[arg(&x_at), arg(&flag_at), arg(&A_LITTLE_WHILE)]) }?;
    // The kernel is queued, and waits: the drop returns when it has written the flag, or the kernel has failed on the
    // memory that the drop unmapped under it, and the flag is 0.
    drop(cache);
    assert_eq!(peek(&mut flag), 1, "the drop returned before the kernel wrote to the buffer");
    // A kernel that fails leaves every call after it an error, `CUDA_ERROR_ILLEGAL_ADDRESS (700)`.
    cuda.sync()?;
    Ok(())
}
