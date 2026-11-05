# kernite

kernite compiles kernels from text, for a tool or a test: `x86_str` gives x86-64 assembly in Intel syntax and `ptx_str`
gives PTX for NVIDIA GPUs. A kernel is Rust's syntax read as instructions over variables, and an error in one reads
`line:column: what`. A crate of kernels adds `kernite-macros` instead, which compiles the kernels where they are called.

```sh
cargo add kernite
```

```rust
fn main() -> Result<(), String> {
    let src = r#"
        pub unsafe extern "sysv64" fn sum4(x: *const u64) -> u64 {
            let sum: gpr;
            mov(sum, [x]);
            for i in 1..4 {
                add(sum, [x + i * 8]);
            }
            ret(sum);
        }

        pub unsafe extern "ptx-kernel" fn store(x: *mut u32) {
            st.global.u32([x], 1);
        }
    "#;

    // x86-64: `for` unrolls while the kernel compiles, and kernite chooses the registers.
    let listing = kernite::x86_str(src, "sum4")?;
    print!("{}", listing.text());
    assert_eq!(listing.code().nth(3), Some("add rax, qword ptr [rdi + 24]"));

    // PTX: each function the GPU runs, in one module.
    let ptx = kernite::ptx_str(src)?;
    print!("{ptx}");
    assert!(ptx.contains("st.global.u32 [%x], 1;"));

    // An error is at its `line:column`.
    let wrong = src.replace("add(", "adds(");
    let error = kernite::x86_str(&wrong, "sum4").unwrap_err();
    assert_eq!(error, "6:17: x86 has no instruction `adds` that kernite knows");
    Ok(())
}
```

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite).
