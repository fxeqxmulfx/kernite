//! The product on the CPU: a kernel in x86, which needs AVX2 and FMA.

/// The kernels.
#[kernite_macros::kernels]
pub mod kernels {
    /// `c = a b`, of `a`, `m` rows of `k` floats, and `b`, `k` rows of `n`, into `c`, `m` rows
    /// of `n`: each element of `c` a chain of fused multiply-adds over `k` in order, eight
    /// columns to a vector. `n` is a multiple of 8.
    pub unsafe extern "sysv64" fn sgemm(c: *mut f32, a: *const f32, b: *const f32, m: usize, n: usize, k: usize) {
        test(m, m);
        jz(done);
        test(n, n);
        jz(done);
        label(row);
        let j: gpr;
        mov(j.r32, 0);
        label(column);
        let acc: ymm;
        zero(acc);
        test(k, k);
        jz(store);
        note("down column `j` of `b`, along row `a`");
        let (x, y, l): (gpr, gpr, gpr);
        mov(x, a);
        lea(y, [b + j * 4]);
        mov(l, k);
        label(step);
        let v: ymm;
        vbroadcastss(v, [x]);
        vfmadd231ps(acc, v, [y]);
        add(x, 4);
        lea(y, [y + n * 4]);
        dec(l);
        jnz(step);
        label(store);
        vmovups([c + j * 4], acc);
        add(j, 8);
        cmp(j, n);
        jb(column);
        lea(a, [a + k * 4]);
        lea(c, [c + n * 4]);
        dec(m);
        jnz(row);
        label(done);
    }
}

/// Whether this CPU runs the kernels.
pub fn supported() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// `c = a b` by the kernel, of the operands of `shape`. `n` must be a multiple of 8.
pub fn sgemm(c: &mut [f32], a: &[f32], b: &[f32], shape: crate::Shape) {
    let (m, n, k) = shape;
    assert!(supported(), "the kernels need AVX2 and FMA");
    crate::operands((a.len(), b.len(), c.len()), shape);
    assert!(n.is_multiple_of(8), "{n} columns are no whole number of vectors of 8");
    // SAFETY: `m` rows of `k` floats behind `a`, `k` rows of `n` behind `b` and `m` rows of `n`
    // behind `c`, `n` a multiple of 8, and the CPU has what the kernel needs.
    unsafe { kernels::sgemm(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), m, n, k) }
}
