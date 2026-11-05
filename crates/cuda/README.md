# kernite-cuda

kernite-cuda is the CUDA driver, opened when a GPU is asked for, so that a program built with GPU kernels still runs
where there is none. It holds memory on the GPU, and loads and launches the PTX that `kernite-macros` writes. A crate
with a GPU kernel adds it, beside `kernite-macros`.

```sh
cargo add kernite-macros kernite-cuda
```

```rust,no_run
#[kernite_macros::kernels]
mod gpu {
    /// `y[i] = a * x[i] + y[i]` for each `i < n`, rounded once.
    pub unsafe extern "ptx-kernel" fn saxpy(a: f32, x: *const f32, y: *mut f32, n: u32) {
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
        let (at_x, at_y): (u64, u64);
        mad.wide.u32(at_x, i, 4, x);
        mad.wide.u32(at_y, i, 4, y);
        let (xi, yi): (f32, f32);
        ld.global.f32(xi, [at_x]);
        ld.global.f32(yi, [at_y]);
        fma.rn.f32(yi, a, xi, yi);
        st.global.f32([at_y], yi);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda = kernite_cuda::Cuda::new()?;
    let kernels = gpu::Kernels::load(&cuda)?;
    let a = 0.1f32;
    let x: Vec<f32> = (0..1000).map(|i| i as f32 * 0.7).collect();
    let mut y = vec![1.0f32; 1000];
    let want: Vec<f32> = x.iter().zip(&y).map(|(x, y)| a.mul_add(*x, *y)).collect();
    let n = x.len() as u32;
    let (xs, ys) = (cuda.upload(&x)?, cuda.upload(&y)?);
    // SAFETY: `n` floats behind each pointer, in buffers that last until `read_into` has waited for the kernel.
    unsafe { kernels.saxpy(kernite_cuda::Grid::new(n.div_ceil(256), 256), a, xs.ptr(), ys.ptr(), n)? };
    ys.read_into(&mut y)?;
    assert_eq!(y, want);
    Ok(())
}
```

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite-cuda).
