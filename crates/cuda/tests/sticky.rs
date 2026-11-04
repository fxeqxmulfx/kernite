//! An error of a launch that is not about its grid is said as the driver says it, whatever the
//! grid. A test of its own, as a program of its own: a kernel that faults leaves the GPU's
//! context failed for every call after it, which would fail the driver's other tests if they
//! shared its process.

use kernite_cuda::{Grid, Module, arg};

/// `fault` writes at address 0, where the GPU has no memory.
const FAULT: &str = "
.version 3.0
.target sm_30
.address_size 64

.visible .entry fault(.param .u64 unused_param)
{
	.reg .u64 %zero;
	.reg .u32 %one;

	mov.u64 %zero, 0;
	mov.u32 %one, 1;
	st.global.u32 [%zero], %one;
	ret;
}
";

#[test]
fn an_error_of_a_launch_that_is_not_about_its_grid_is_said_as_the_driver_says_it() {
    let Ok(cuda) = kernite_cuda::Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
    let module = Module::load(&cuda, FAULT, &["fault"], &[]).unwrap();
    let out = cuda.alloc::<u32>(1).unwrap();
    let at = out.ptr();
    // SAFETY: `fault` takes a pointer and does not read it. The launch starts, and the kernel
    // fails on the GPU, which this test is for.
    unsafe { module.launch(0, Grid::new(1, 1), &[arg(&at)]) }.unwrap();
    let failed = cuda.sync().unwrap_err();
    assert!(failed.message.starts_with("CUDA_ERROR_ILLEGAL_ADDRESS"), "{failed}");
    // Every launch after it returns that error, and says no more of a grid the GPU does not take.
    let grid = |blocks, threads| Grid { blocks, threads, shared: 0 };
    let grids = [grid([1, 65536, 1], [1, 1, 1]), grid([1, 1, 1], [1025, 1, 1]), grid([0, 1, 1], [1, 1, 1])];
    for grid in grids {
        // SAFETY: as above.
        let e = unsafe { module.launch(0, grid, &[arg(&at)]) }.unwrap_err();
        assert_eq!(e.message, failed.message, "{grid:?}");
        if cuda.cooperative() {
            // SAFETY: as above.
            let e = unsafe { module.launch_cooperative(0, grid, &[arg(&at)]) }.unwrap_err();
            assert_eq!(e.message, failed.message, "{grid:?}, cooperatively");
        }
    }
}
