//! A crate that forbids `unsafe_code` has GPU kernels that are no `unsafe fn`, and launches them as it calls any
//! function: the code the macro writes for a launcher is no `unsafe` of the crate's.

#![forbid(unsafe_code)]

use kernite_cuda::{Buffer, Error, Grid};

#[kernite_macros::kernels]
pub mod gpu {
    /// `y[i] = x[i] + y[i]` for each `i < n`.
    pub extern "ptx-kernel" fn plus(x: *const f32, y: *mut f32, n: u32) {
        let i: u32;
        global_index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let (at_x, at_y): (u64, u64);
        mad.wide.u32(at_x, i, 4, x);
        mad.wide.u32(at_y, i, 4, y);
        let (a, b): (f32, f32);
        ld.global.f32(a, [at_x]);
        ld.global.f32(b, [at_y]);
        add.rn.f32(b, a, b);
        st.global.f32([at_y], b);
    }

    /// This thread's index among all the grid's.
    fn global_index(i: u32) {
        let (block, size, thread): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        mad.lo.u32(i, block, size, thread);
    }
}

/// `plus` over the first `n` floats of `x` and of `y`, with no `unsafe`: the launch is checked, and an error says what
/// does not hold.
pub fn plus(kernels: &gpu::Kernels, x: &Buffer<f32>, y: &mut Buffer<f32>, n: u32) -> Result<(), Error> {
    kernels.plus(Grid::new(n.div_ceil(256), 256), x, y, n)
}
