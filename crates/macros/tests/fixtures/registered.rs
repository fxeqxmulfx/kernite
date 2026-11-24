//! A launcher the macros write that is no `unsafe fn` takes a slice the host registered for good, as it takes a
//! buffer: `cuda.register(Vec::leak(v))` is a `Unified<'static, u32>`, which a launcher takes by `&mut`. The kernel
//! `bump` adds 1 to each element, and the host reads the sums from the slice. Rust builds the crate without a word,
//! under `#![deny(warnings)]`, and the test runs the kernel on the GPU if there is one that can.

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
fn a_launcher_takes_a_slice_registered_for_good() {
    let Ok(cuda) = kernite_cuda::Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
    let kernels = gpu::Kernels::load(&cuda).unwrap();
    let Ok(mut ys) =
        cuda.register(Vec::leak(vec![1u32, 2, 3])).inspect_err(|e| eprintln!("the GPU cannot: skipped. {e}"))
    else {
        return;
    };
    kernels.bump(kernite_cuda::Grid::new(1, 32), &mut ys, 3).unwrap();
    assert_eq!(ys.host().unwrap(), [2, 3, 4], "bump did not run");
}
