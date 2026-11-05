# kernite-macros

`#[kernite_macros::kernels]` on a module compiles each function in it that names an ABI: kernels written where they are
called, in Rust's syntax read as instructions over variables, for x86-64 and for NVIDIA GPUs. A crate of kernels adds
this crate, with `kernite-cuda` for a GPU kernel and `kernite-cpu` for a cooperative x86 kernel or an x86 static.

```sh
cargo add kernite-macros
```

```rust
#[kernite_macros::kernels]
mod kernels {
    /// The sum of the eight floats at `xs`.
    pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
        let v: ymm;
        vmovups(v, [xs]);
        let high: ymm;
        vextractf128(high.xmm, v, 1);
        let sum: ymm;
        vaddps(sum.xmm, v.xmm, high.xmm);
        let pair: ymm;
        vmovhlps(pair.xmm, sum.xmm, sum.xmm);
        vaddps(sum.xmm, sum.xmm, pair.xmm);
        vmovshdup(pair.xmm, sum.xmm);
        vaddss(sum.xmm, sum.xmm, pair.xmm);
        ret(sum.xmm);
    }
}

fn main() {
    assert!(is_x86_feature_detected!("avx"), "sum8 needs AVX");
    let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // SAFETY: eight floats are there to read, and the CPU has AVX.
    let sum = unsafe { kernels::sum8(xs.as_ptr()) };
    assert_eq!(sum, 36.0);
}
```

`sum8` is an `#[unsafe(naked)]` function of the signature it declares, its body the kernel in Intel syntax. A kernel for
a GPU is written the same way, as an `extern "ptx-kernel"` function:
[kernite-cuda's README](https://crates.io/crates/kernite-cuda) has one.

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite-macros).
