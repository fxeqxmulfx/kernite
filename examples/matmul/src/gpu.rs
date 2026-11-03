//! The product on the GPU's cores: a kernel in PTX, in `f32`.

use kernite_cuda::{Buffer, Error, Grid};

/// The kernels.
#[kernite_macros::kernels]
pub mod kernels {
    /// `c = a b`, of `a`, `m` rows of `k` floats, and `b`, `k` rows of `n`, into `c`, `m` rows
    /// of `n`: a thread for each element of `c`, its chain of fused multiply-adds over `k` in
    /// order.
    pub unsafe extern "ptx-kernel" fn sgemm(c: *mut f32, a: *const f32, b: *const f32, m: u32, n: u32, k: u32) {
        let (block, size, thread, i, row, col): (u32, u32, u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        mad.lo.u32(i, block, size, thread);
        div.u32(row, i, n);
        rem.u32(col, i, n);
        let past: pred;
        setp.ge.u32(past, row, m);
        if past {
            ret();
        }
        note("along row `row` of `a`, down column `col` of `b`");
        let (x, y, down): (u64, u64, u64);
        mul.lo.u32(i, row, k);
        mad.wide.u32(x, i, 4, a);
        mad.wide.u32(y, col, 4, b);
        mul.wide.u32(down, n, 4);
        let acc: f32;
        mov.f32(acc, 0);
        let more: pred;
        setp.ne.u32(more, k, 0);
        if !more {
            bra(store);
        }
        let (u, v): (f32, f32);
        label(step);
        ld.global.f32(u, [x]);
        ld.global.f32(v, [y]);
        fma.rn.f32(acc, u, v, acc);
        add.u64(x, x, 4);
        add.u64(y, y, down);
        sub.u32(k, k, 1);
        setp.ne.u32(more, k, 0);
        if more {
            bra(step);
        }
        label(store);
        mad.lo.u32(i, row, n, col);
        let z: u64;
        mad.wide.u32(z, i, 4, c);
        st.global.f32([z], acc);
    }
}

/// Launches `sgemm` on `c`, `a` and `b`, of the sizes `shape` gives, `(m, n, k)`.
///
/// # Safety
///
/// The buffers last until the kernel ends: until a call that waits for the GPU, as a read
/// does.
pub unsafe fn launch(
    kernels: &kernels::Kernels,
    c: &Buffer<f32>,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    shape: crate::Shape,
) -> Result<(), Error> {
    let (m, n, k) = crate::gpu_operands((a.len(), b.len(), c.len()), shape);
    if m * n == 0 {
        return Ok(());
    }
    let blocks = (m * n).div_ceil(256);
    // SAFETY: `m` rows of `k` floats behind `a`, `k` rows of `n` behind `b` and `m` rows of `n`
    // behind `c`, which last as the caller says.
    unsafe { kernels.sgemm(Grid::new(blocks, 256), c.ptr(), a.ptr(), b.ptr(), m, n, k) }
}
