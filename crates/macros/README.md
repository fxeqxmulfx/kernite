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
    pub extern "sysv64" fn sum8(xs: &[f32; 8]) -> f32 {
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

/// `sum8` needs AVX, so a function that has it calls it with no `unsafe`.
#[target_feature(enable = "avx")]
fn sum() -> f32 {
    kernels::sum8(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
}

fn main() {
    assert!(is_x86_feature_detected!("avx"), "sum8 needs AVX");
    // SAFETY: the CPU has AVX, as the line above found.
    assert_eq!(unsafe { sum() }, 36.0);
}
```

`sum8` is a function of the signature it declares, and Rust calls it with no `unsafe`: it takes an array by reference
and reaches no memory but that and its own frame, which kernite checks where it compiles the kernel, so there is
nothing for the caller to promise. It needs AVX, which Rust checks where it is called: only a function that has it,
`#[target_feature(enable = "avx")]`, calls it with no `unsafe`. Its body is the kernel in Intel syntax, in an
`#[unsafe(naked)]` function. A kernel that takes a pointer, or that kernite cannot prove safe in some other way, is an
`unsafe extern "sysv64" fn`. A kernel for a GPU is written the same way, as an `extern "ptx-kernel"` function, and
it is no `unsafe fn` where kernite proves that each thread reads and writes inside the memory it is lent, and writes
a part of its own: [kernite-cuda's README](https://crates.io/crates/kernite-cuda) has one.

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite-macros).
