//! The driver as kgen calls it, on the GPU here, if there is one.

use kgen_cuda::{Cuda, Grid, Module, arg};

/// `x[i] += 1` for each `i < n`, a thread each, written by hand.
const PTX: &str = "
.version 7.5
.target sm_75
.address_size 64

.visible .entry add_one(.param .u64 x_param, .param .u32 n_param)
{
	.reg .pred %p;
	.reg .u32 %n, %i, %block, %size, %thread;
	.reg .f32 %v;
	.reg .u64 %x, %at;

	ld.param.u64 %x, [x_param];
	cvta.to.global.u64 %x, %x;
	ld.param.u32 %n, [n_param];
	mov.u32 %block, %ctaid.x;
	mov.u32 %size, %ntid.x;
	mov.u32 %thread, %tid.x;
	mad.lo.u32 %i, %block, %size, %thread;
	setp.ge.u32 %p, %i, %n;
	@%p bra done;
	mad.wide.u32 %at, %i, 4, %x;
	ld.global.f32 %v, [%at];
	add.rn.f32 %v, %v, 0f3F800000;
	st.global.f32 [%at], %v;
done:
	ret;
}
";

/// The GPU, or `None` and why not.
fn cuda() -> Option<Cuda> {
    Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")).ok()
}

#[test]
fn a_kernel_runs_on_what_was_uploaded() {
    let Some(cuda) = cuda() else { return };
    let xs: Vec<f32> = (0..1000).map(|i| i as f32 / 3.0).collect();
    let buf = cuda.upload(&xs).unwrap();
    let module = Module::load(&cuda, PTX, &["add_one"]).unwrap();
    // Threads past the first 999, to see the bounds checked.
    let (x, n) = (buf.ptr(), 999u32);
    // SAFETY: `add_one` takes a pointer and a `u32`, and writes only the first `n` of the
    // 1000 the buffer has.
    unsafe { module.launch(0, Grid::new(4, 256), &[arg(&x), arg(&n)]) }.unwrap();
    cuda.sync().unwrap();
    let want: Vec<f32> = xs.iter().enumerate().map(|(i, x)| if i < 999 { x + 1.0 } else { *x }).collect();
    assert_eq!(buf.read().unwrap(), want);
}

#[test]
fn a_buffer_is_zero_until_written_and_reads_back_what_was() {
    let Some(cuda) = cuda() else { return };
    let mut buf = cuda.alloc::<u32>(5).unwrap();
    assert_eq!(buf.read().unwrap(), [0; 5]);
    buf.write(&[1, 2, 3, 4, u32::MAX]).unwrap();
    let mut out = [0; 5];
    buf.read_into(&mut out).unwrap();
    assert_eq!(out, [1, 2, 3, 4, u32::MAX]);
    buf.zero().unwrap();
    assert_eq!(buf.read().unwrap(), [0; 5]);
    assert_eq!(buf.at(2).addr(), buf.ptr().addr() + 8);
    assert!(cuda.alloc::<f32>(0).unwrap().is_empty());
}

#[test]
fn ptx_the_driver_will_not_compile_is_logged() {
    let Some(cuda) = cuda() else { return };
    let e = Module::load(&cuda, &PTX.replace("add.rn.f32", "add.rn.f99"), &["add_one"]).err().unwrap();
    assert_eq!(e.call, "cuModuleLoadDataEx");
    assert!(e.message.starts_with("CUDA_ERROR_INVALID_PTX"), "{e}");
    assert!(e.log.contains("f99"), "{e}");
}

#[test]
fn a_kernel_the_module_has_not_is_named() {
    let Some(cuda) = cuda() else { return };
    let e = Module::load(&cuda, PTX, &["add_one", "add_two"]).err().unwrap();
    assert_eq!(e.call, "cuModuleGetFunction");
    assert!(e.message.starts_with("CUDA_ERROR_NOT_FOUND") && e.message.ends_with("for `add_two`"), "{e}");
}
