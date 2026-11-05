# kernite-ptx

PTX for NVIDIA GPUs, as a `Target` of kernite-lang. It checks each instruction against the forms PTX has for its name
and types, and writes the module that the CUDA driver loads. A crate of kernels does not add it: it adds
`kernite-macros`.

```sh
cargo add kernite-ptx kernite-lang
```

```rust
use kernite_lang::{Mem, Operand, Target};
use kernite_ptx::lang::Kernel;
use kernite_ptx::module;

fn main() {
    // `pub unsafe extern "ptx-kernel" fn store(out: *mut f32)`, which writes 1.5 to `out`.
    let mut kernel = Kernel::new("store", "store");
    let out = kernel.param("out", "*mut f32").unwrap();
    let x = Operand::Var(kernel.var("x", "f32").unwrap());
    let at = Operand::Mem(Mem { base: Some(out), index: None, disp: 0 });

    // Each instruction is checked: `st.global.u32` does not store an `f32`.
    let fault = kernel.inst("st.global.u32", &[at.clone(), x.clone()], None).unwrap_err();
    assert_eq!(fault.message, "`st.global.u32` wants a `u32` variable; this is a `f32`");
    kernel.inst("mov.f32", &[x.clone(), Operand::F32(1.5)], None).unwrap();
    kernel.inst("st.global.f32", &[at, x], None).unwrap();
    kernel.end().unwrap();

    // The module is of the PTX version and for the GPU that its instructions need.
    let ptx = module(&[kernel.finish()], &[]).whole().unwrap();
    assert!(ptx.contains(".version 3.0\n.target sm_30\n"));
    assert!(ptx.contains("\tmov.f32 %x, 0f3FC00000;\n\tst.global.f32 [%out], %x;\n\tret;\n"));
}
```

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite-ptx).
