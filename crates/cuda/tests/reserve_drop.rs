//! Dropping a buffer that reserved addresses waits for the kernels that use it, and then unmaps its memory: the driver
//! does not wait for a kernel before it unmaps what the kernel reaches, and the kernel then fails with
//! `CUDA_ERROR_ILLEGAL_ADDRESS (700)`, as does each call after it. A test of its own, as a program of its own, as
//! `sticky.rs` is: a drop that did not wait would fail the GPU's context for every test beside it.

use kernite_cuda::{Cuda, Grid, Module, arg};

mod gpu;
use gpu::{HANG, peek, ptx, release_while};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// `later(x, go, flag, looks)`: one thread looks at the word `*go` until it is not 0, `looks` times at most, then
/// writes 7 to `*x`, and then 1 to `*flag`.
const PTX: &str = "
.visible .entry later(.param .u64 x_param, .param .u64 go_param, .param .u64 flag_param, .param .u64 looks_param)
{
	.reg .u64 %x, %go, %flag, %looks;
	.reg .u32 %word, %seven, %one;
	.reg .pred %clear, %over;

	ld.param.u64 %x, [x_param];
	ld.param.u64 %go, [go_param];
	ld.param.u64 %flag, [flag_param];
	ld.param.u64 %looks, [looks_param];
	cvta.to.global.u64 %x, %x;
	cvta.to.global.u64 %go, %go;
	cvta.to.global.u64 %flag, %flag;
LOOK:
	setp.eq.u64 %over, %looks, 0;
	@%over bra WRITE;
	sub.u64 %looks, %looks, 1;
	ld.volatile.global.u32 %word, [%go];
	setp.eq.u32 %clear, %word, 0;
	@%clear bra LOOK;
WRITE:
	mov.u32 %seven, 7;
	st.volatile.global.u32 [%x], %seven;
	mov.u32 %one, 1;
	st.volatile.global.u32 [%flag], %one;
	ret;
}
";

#[test]
fn dropping_a_buffer_waits_for_the_kernel_that_writes_it_and_then_unmaps_it() -> Result {
    let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return Ok(()) };
    let Ok(mut cache) = cuda.reserve::<u32>(1 << 28).inspect_err(|e| eprintln!("the GPU cannot: skipped. {e}")) else {
        return Ok(());
    };
    let module = Module::load(&cuda, &ptx(PTX), &["later"], &[])?;
    let (mut go, mut flag) = (cuda.pinned::<u32>(1)?, cuda.pinned::<u32>(1)?);
    cache.grow(10_000_000)?;
    let (x_at, go_at, flag_at) = (cache.ptr(), go.ptr(), flag.ptr());
    // SAFETY: `later` takes three pointers and a `u64`: the first word of the buffer and the page-locked flag, which it
    // alone writes, and the page-locked word `go`, which it reads while the host writes it by an atomic. The buffer and
    // the words are held by this test until after the kernel is waited for.
    unsafe { module.launch(0, Grid::new(1, 1), &[arg(&x_at), arg(&go_at), arg(&flag_at), arg(&HANG)]) }?;
    // The kernel is queued, and waits for `go`, which the host writes while it is in the drop: the drop returns when
    // the kernel has written the flag, or the kernel has failed on the memory that the drop unmapped under it, and the
    // flag is 0.
    release_while(&mut go, || drop(cache))?;
    assert_eq!(peek(&mut flag), 1, "the drop returned before the kernel wrote to the buffer");
    // A kernel that fails leaves every call after it an error, `CUDA_ERROR_ILLEGAL_ADDRESS (700)`.
    cuda.sync()?;
    Ok(())
}
