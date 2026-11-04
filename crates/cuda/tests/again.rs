//! A GPU taken again after it was let go. A test of its own, as a program of its own: the driver
//! destroys a GPU's primary context when the last thing holding it is dropped, and the thread
//! may be left with that context for its current one. The driver's other tests hold a GPU all
//! the while, and never see a context destroyed.

use std::sync::{Mutex, PoisonError};

use kernite_cuda::{Buffer, Cuda, Grid, Module, arg};

/// One test at a time: the `Cuda` that one holds keeps the context that another means to destroy.
static ALONE: Mutex<()> = Mutex::new(());

/// How many GPUs the driver shows, if it shows one at least; or a line that says the test is
/// skipped.
fn gpus() -> Option<usize> {
    match Cuda::count() {
        Ok(0) => eprintln!("no GPU: skipped, the driver shows none"),
        Ok(count) => return Some(count),
        Err(e) => eprintln!("no GPU: skipped. {e}"),
    }
    None
}

/// `x[0] = 7`, written by hand.
const SEVEN: &str = "
.version 7.5
.target sm_61
.address_size 64

.visible .entry seven(.param .u64 x_param)
{
	.reg .u64 %x;
	.reg .u32 %seven;

	ld.param.u64 %x, [x_param];
	cvta.to.global.u64 %x, %x;
	mov.u32 %seven, 7;
	st.global.u32 [%x], %seven;
	ret;
}
";

/// A buffer of one word, which is zero, and `seven` launched on it.
fn seven(cuda: &Cuda) -> (Module, Buffer<u32>) {
    let module = Module::load(cuda, SEVEN, &["seven"], &[]).unwrap();
    let x = cuda.alloc::<u32>(1).unwrap();
    assert_eq!(x.read().unwrap(), [0]);
    let at = x.ptr();
    // SAFETY: `seven` takes a pointer, and writes the one word the buffer has.
    unsafe { module.launch(0, Grid::new(1, 1), &[arg(&at)]) }.unwrap();
    (module, x)
}

#[test]
fn a_gpu_is_taken_again_after_the_last_cuda_of_it_was_dropped() {
    let Some(count) = gpus() else { return };
    let _alone = ALONE.lock().unwrap_or_else(PoisonError::into_inner);
    for index in 0..count {
        // The driver keeps the GPU's context while either of two is held.
        let (first, second) = (Cuda::on(index).unwrap(), Cuda::on(index).unwrap());
        drop(first);
        assert_eq!(seven(&second).1.read().unwrap(), [7], "GPU {index}, the other dropped");
        // The last one dropped destroys the context, which the thread has for its current one
        // still. A GPU taken again has a context of its own, with the handle of the one gone.
        drop(second);
        for time in 1..=3 {
            let cuda = Cuda::on(index).unwrap();
            assert_eq!(seven(&cuda).1.read().unwrap(), [7], "GPU {index}, taken again {time} times");
        }
    }
}

#[test]
fn a_buffer_and_a_module_keep_their_gpu_after_the_cuda_is_dropped() {
    let Some(count) = gpus() else { return };
    let _alone = ALONE.lock().unwrap_or_else(PoisonError::into_inner);
    for index in 0..count {
        let (module, x) = seven(&Cuda::on(index).unwrap());
        // Nothing but the module and the buffer holds the GPU now; they use it, and let it go.
        assert_eq!(x.read().unwrap(), [7], "GPU {index}");
        let at = x.ptr();
        // SAFETY: as in `seven`.
        unsafe { module.launch(0, Grid::new(1, 1), &[arg(&at)]) }.unwrap();
        drop(module);
        assert_eq!(x.read().unwrap(), [7], "GPU {index}, the module dropped");
    }
}
