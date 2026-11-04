//! The product on the GPU from `f16`, for its tensor cores: a kernel in PTX, `f16` in and
//! `f32` out.

use kernite_cuda::{Buffer, Error, f16};

/// The kernels.
#[kernite_macros::kernels]
pub mod kernels {
    /// `c = a b`, of `a`, `m` rows of `k` halves, and `b`, `k` rows of `n`, into `c`, `m` rows
    /// of `n` floats: a thread for each element of `c`, its chain of fused multiply-adds over
    /// `k` in order, each half made a float.
    pub unsafe extern "ptx-kernel" fn hgemm(c: *mut f32, a: *const f16, b: *const f16, m: u32, n: u32, k: u32) {
        let (block, size, thread, i, row, col, w): (u32, u32, u32, u32, u32, u32, u32);
        note("the block's number counts along the grid's x, then y, as `gpu_grid` makes it");
        mov.u32(block, ctaid.y);
        mov.u32(w, nctaid.x);
        mul.lo.u32(block, block, w);
        mov.u32(w, ctaid.x);
        add.u32(block, block, w);
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
        mad.wide.u32(x, i, 2, a);
        mad.wide.u32(y, col, 2, b);
        mul.wide.u32(down, n, 2);
        let acc: f32;
        mov.f32(acc, 0);
        let more: pred;
        setp.ne.u32(more, k, 0);
        if !more {
            bra(store);
        }
        let (p, q): (f16, f16);
        let (u, v): (f32, f32);
        label(step);
        ld.global.b16(p, [x]);
        ld.global.b16(q, [y]);
        cvt.f32.f16(u, p);
        cvt.f32.f16(v, q);
        fma.rn.f32(acc, u, v, acc);
        add.u64(x, x, 2);
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

/// Launches `hgemm` on `c`, `a` and `b`, of the sizes `shape` gives, `(m, n, k)`.
///
/// # Safety
///
/// The buffers last until the kernel ends: until a call that waits for the GPU, as a read
/// does.
pub unsafe fn launch(
    kernels: &kernels::Kernels,
    c: &Buffer<f32>,
    a: &Buffer<f16>,
    b: &Buffer<f16>,
    shape: crate::Shape,
) -> Result<(), Error> {
    let (m, n, k) = crate::gpu_operands((a.len(), b.len(), c.len()), shape);
    if m * n == 0 {
        return Ok(());
    }
    let blocks = (m * n).div_ceil(256);
    // SAFETY: `m` rows of `k` halves behind `a`, `k` rows of `n` behind `b` and `m` rows of `n`
    // floats behind `c`, which last as the caller says.
    unsafe { kernels.hgemm(crate::gpu_grid(blocks, 256), c.ptr(), a.ptr(), b.ptr(), m, n, k) }
}
