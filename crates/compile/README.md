# kernite-compile

The allocator of kernite: it gives the variables of a program registers, spills to the stack the ones that do not fit,
and lays out the frame. It knows the machine only through two ports, `Machine` and `Spill`, so it runs for any
instruction set. A crate of kernels does not add it: it adds `kernite-macros`.

```sh
cargo add kernite-compile kernite-x86
```

```rust
use kernite_compile::compile;
use kernite_x86::{Asm, SysV};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Seventeen vectors alive at once, where x86-64 has sixteen vector registers.
    let mut asm = Asm::new("sum17");
    let v: Vec<_> = (0..17).map(|i| asm.ymm(format!("v{i}"))).collect();
    for &x in &v {
        asm.zero(x);
    }
    for &x in &v[1..] {
        asm.vaddps(v[0], v[0], x);
    }
    asm.ret();

    let kernel = compile(&asm.finish(), &SysV)?;
    let spilled: Vec<_> = kernel.frame.slots.iter().filter(|slot| slot.spill).map(|slot| &slot.name[..]).collect();
    assert_eq!(spilled, ["v16", "v15"]);
    assert_eq!((kernel.frame.size, kernel.frame.align), (64, 32));
    Ok(())
}
```

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite-compile).
