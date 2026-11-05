# kernite-x86

x86-64-v3 for kernite: its instructions, the System V calling convention, and the printer that writes an allocated
kernel in Intel syntax. `Asm` builds a program from Rust, one method per instruction, with variables in the place of
registers. A crate of kernels does not add it: it adds `kernite-macros`.

```sh
cargo add kernite-x86 kernite-compile
```

```rust
use kernite_compile::compile;
use kernite_x86::{Asm, SysV, intel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `fn sum8(xs: *const f32) -> f32`: the eight floats at `xs`, added.
    let mut asm = Asm::new("sum8");
    let xs = asm.param("xs");
    let (v, half) = (asm.ymm("v"), asm.ymm("half"));
    asm.vmovups(v, xs.at(0));
    asm.vextractf128(half.xmm(), v, 1);
    asm.vaddps(v.xmm(), v.xmm(), half.xmm());
    asm.vmovhlps(half.xmm(), v.xmm(), v.xmm());
    asm.vaddps(v.xmm(), v.xmm(), half.xmm());
    asm.vmovshdup(half.xmm(), v.xmm());
    asm.vaddss(v.xmm(), v.xmm(), half.xmm());
    asm.ret_f32(v.xmm());

    // The allocator gives the variables registers, and the printer writes the lines that `naked_asm!` takes.
    // `vec![]` lists the functions that a call at an address may reach: none.
    let listing = intel::print(&compile(&asm.finish(), &SysV)?, vec![])?;
    let want = [
        "vmovups ymm0, ymmword ptr [rdi]",
        "vextractf128 xmm1, ymm0, 1",
        "vaddps xmm0, xmm0, xmm1",
        "vmovhlps xmm1, xmm0, xmm0",
        "vaddps xmm0, xmm0, xmm1",
        "vmovshdup xmm1, xmm0",
        "vaddss xmm0, xmm0, xmm1",
        "vzeroupper",
        "ret",
    ];
    assert_eq!(listing.code().collect::<Vec<_>>(), want);
    Ok(())
}
```

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite-x86).
