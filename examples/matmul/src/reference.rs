//! What the kernels compute, in plain Rust: each element of `c = a b` a chain of fused
//! multiply-adds over `k` in order, from zero.

/// `a b` in `f32`, of the operands of `shape`.
pub fn sgemm(a: &[f32], b: &[f32], (m, n, k): crate::Shape) -> Vec<f32> {
    (0..m * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..k).fold(0.0, |acc, l| a[i * k + l].mul_add(b[l * n + j], acc))
        })
        .collect()
}
