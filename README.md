# kernite

Kernels written where they are called, in Rust's syntax read as instructions over variables: compiled for x86-64
to assembly that changes only when they do, and for NVIDIA GPUs to PTX.

A kernel's body is instructions, called by name, over variables declared with a type and no value; kernite allocates
the registers. Everything that is not an instruction happens while the kernel compiles: `for` unrolls, `if` on a bool
chooses, and `<const N: usize>` makes one generic.

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

`sum8` is an `#[unsafe(naked)]` function of the signature it declares, its body the kernel in Intel syntax. A kernel
for a GPU is written the same way, as an `extern "ptx-kernel"` function: [kernite-cuda's README](crates/cuda) has one.

## Crates

A crate of kernels adds `kernite-macros`, and `kernite-cuda` or `kernite-cpu` where its kernels need them. The rest are
the layers behind `kernite-macros`, each depending only on those below it.

- [`kernite-macros`](crates/macros): `#[kernite_macros::kernels]` on a module compiles each function that names an ABI.
- [`kernite-cuda`](crates/cuda): the CUDA driver, opened when a GPU is asked for: memory on it, modules, launches.
- [`kernite-cpu`](crates/cpu): the threads of a cooperative kernel's grid, and the memory of a module of x86 kernels.
- [`kernite`](crates/kernite): kernels compiled from text, `x86_str` and `ptx_str`, for tools and tests.
- [`kernite-x86`](crates/x86): x86-64-v3: its instructions, the System V calling convention, Intel syntax.
- [`kernite-ptx`](crates/ptx): PTX for NVIDIA GPUs: each instruction checked against the types its name gives.
- [`kernite-build`](crates/build): what kernite does with a module of kernels, whatever machine compiles it.
- [`kernite-compile`](crates/compile): the allocator: liveness, register allocation that spills, the frame's layout.
- [`kernite-lang`](crates/lang): the language: a Rust function's body read as instructions over variables.
- [`kernite-core`](crates/core): a kernel as instructions over registers.

## Commands

```sh
./make.py check    # formatting, clippy, every test, and the docs
./make.py fmt      # formats every crate
./make.py publish  # publishes each crate crates.io lacks at its version, one at a time
```

`make.py` runs under [uv](https://docs.astral.sh/uv/).

The language, the targets, the guarantees and the examples are in [GUIDE.md](GUIDE.md).
