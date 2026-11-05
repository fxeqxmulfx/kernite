# kernite-lang

kernite-lang is the language of kernels: Rust's syntax, read as instructions over variables. It reads a module from
text, works out each value known now, and says what is wrong at its `line:column`. A crate of kernels adds
`kernite-macros` instead; a tool that reads kernels itself adds this one, as does a machine, which implements `Target`.

```sh
cargo add kernite-lang
```

```rust
use kernite_lang::{Error, ItemKind, instance, instances, parse};

/// The name of each instance of each function in `src`.
fn names(src: &str) -> Result<Vec<String>, Error> {
    let module = parse(src)?;
    let mut names = vec![];
    for item in &module.items {
        let ItemKind::Fn(f) = &item.kind else { continue };
        for args in instances(&module, f)? {
            names.push(instance(f, &args)?);
        }
    }
    Ok(names)
}

fn main() -> Result<(), Error> {
    let src = r#"
        const BLOCK: u32 = 128;

        #[instances(BLOCK, BLOCK * 2)]
        pub unsafe extern "ptx-kernel" fn store<const N: u32>(x: *mut u32) {
            st.global.u32([x], N);
        }
    "#;
    assert_eq!(names(src)?, ["store::<128>", "store::<256>"]);

    // An error is at its `line:column`.
    let wrong = src.replace("BLOCK * 2", "SIZE * 2");
    assert_eq!(names(&wrong).unwrap_err().at(&wrong), "4:28: `SIZE` is not defined");
    Ok(())
}
```

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite-lang).
