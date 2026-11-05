# kernite-cpu

kernite-cpu runs a `#[cooperative]` x86 kernel on the threads of a `Grid`, each given its place there, on any CPUs or on
those the caller names. It also holds the memory of a module's statics, which the kernels and the host both reach. A
crate of kernels adds it, beside `kernite-macros`, for either.

```sh
cargo add kernite-macros kernite-cpu
```

```rust
#[kernite_macros::kernels]
mod kernels {
    /// `x[t] = t` for each thread `t` of the grid.
    #[cooperative]
    pub unsafe extern "sysv64" fn fill(x: *mut u64) {
        let t: gpr;
        mov(t, grid.thread);
        mov([x + t * 8], t);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A grid has at most as many threads as the process runs at once.
    let mut x = vec![0; kernite_cpu::most()?];
    // SAFETY: `x` has a word for each thread of the grid, and each writes its own and no other.
    unsafe { kernels::fill(kernite_cpu::Grid::new(x.len()), x.as_mut_ptr())? };
    assert_eq!(x, (0..x.len() as u64).collect::<Vec<_>>());
    Ok(())
}
```

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite-cpu).
