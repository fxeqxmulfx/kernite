//! A launcher the macros write that is no `unsafe fn` takes a buffer that `cuda.reserve` made, as it takes any buffer:
//! it is lent the elements the buffer has, as many as it was grown to, and none of the addresses reserved past them.
//! The kernel `bump` adds 1 to each of the first `n` elements of `y`. The launcher refuses an `n` past the elements the
//! buffer has, as it refuses one past a buffer that `alloc` made, and runs the kernel over all of them when the buffer
//! has grown. Rust builds the crate without a word, under `#![deny(warnings)]`, and the test runs the kernel on the GPU
//! if there is one that can.

#![deny(warnings)]

#[kernite_macros::kernels]
pub mod gpu {
    /// `y[i] += 1` for each `i < n`: a kernel that kernite proves safe, so that its launcher is no `unsafe fn`.
    pub extern "ptx-kernel" fn bump(y: *mut u32, n: u32) {
        let (block, size, thread, i): (u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        mad.lo.u32(i, block, size, thread);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let at: u64;
        mad.wide.u32(at, i, 4, y);
        let value: u32;
        ld.global.u32(value, [at]);
        add.u32(value, value, 1);
        st.global.u32([at], value);
    }
}

#[test]
fn a_launcher_takes_a_buffer_that_grew_and_is_lent_the_elements_it_has() {
    let Ok(cuda) = kernite_cuda::Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
    let Ok(mut ys) = cuda.reserve::<u32>(1 << 20).inspect_err(|e| eprintln!("the GPU cannot: skipped. {e}")) else {
        return;
    };
    let kernels = gpu::Kernels::load(&cuda).unwrap();
    ys.grow(3).unwrap();
    ys.write(&[1, 2, 3]).unwrap();
    // The addresses past the 3 elements have no memory, and the launcher does not lend them: it refuses a count past
    // the elements, and no kernel runs.
    let refused = kernels.bump(kernite_cuda::Grid::new(1, 32), &mut ys, 4).unwrap_err();
    assert_eq!(
        refused.to_string(),
        "cuLaunchKernel: `bump` is safe on this launch only if `y.len() >= n`, so that each element it reaches of `y` \
         is inside it, and that does not hold: `y.len()` is 3, `n` is 4"
    );
    kernels.bump(kernite_cuda::Grid::new(1, 32), &mut ys, 3).unwrap();
    assert_eq!(ys.read().unwrap(), [2, 3, 4], "bump did not run");
    // Grown, the buffer is lent them all: the elements it had, and the zeros it gained.
    ys.grow(1000).unwrap();
    kernels.bump(kernite_cuda::Grid::new(32, 32), &mut ys, 1000).unwrap();
    let all = ys.read().unwrap();
    assert_eq!(all[..3], [3, 4, 5], "bump did not run over the elements the buffer had");
    assert!(all[3..].iter().all(|y| *y == 1), "bump did not run over the elements the buffer gained");
}
