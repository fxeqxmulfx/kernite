# kernite-build

kernite-build is what kernite does with a module of kernels, whatever machine compiles them: it sorts the kernels the
process runs from those the GPU runs, checks the rules of the module, and has the machine compile each. A machine
implements `Backend`, with `Native` for the process or `Device` for a GPU, as `kernite_x86::X86` and `kernite_ptx::Ptx`
do. A crate of kernels adds `kernite-macros` instead; a crate that writes a machine adds this one.

```sh
cargo add kernite-build kernite-lang kernite-x86
```

```rust
use kernite_build::one;
use kernite_lang::parse;
use kernite_x86::X86;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let src = r#"
        #[instances(2, 4)]
        pub extern "sysv64" fn scale<const K: u64>(x: u64) -> u64 {
            let v: gpr;
            imul(v, x, K);
            ret(v);
        }

        pub extern "ptx-kernel" fn store(x: *mut u32) {
            st.global.u32([x], 1);
        }
    "#;
    let module = parse(src)?;

    // `one` finds an instance by its name and has the machine compile it.
    let listing = one(&module, &X86, "scale::<4>")?.ok_or("`scale` has no instance `scale::<4>`")?.code;
    assert_eq!(listing.code().collect::<Vec<_>>(), ["imul rax, rdi, 4", "ret"]);

    // The GPU runs `store`, so x86 has no kernel of that name.
    assert!(one(&module, &X86, "store")?.is_none());
    Ok(())
}
```

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite-build).
