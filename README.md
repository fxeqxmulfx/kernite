# kgen

Kernels written as x86 instructions over variables, compiled to assembly that changes only when they do.

A kernel is Rust code that calls one builder method per instruction (`a.vpmaddubsw(p, p, x)`), on variables
instead of registers. `kgen` allocates the registers with no spills: if the variables that are alive together do
not fit, it fails and names them. It prints Intel syntax, and `kgen-macros` turns that into the body of a
`#[unsafe(naked)]` function.

This repository is the library alone; the kernels live in the crates that use it.

## A kernel

```rust
use kgen_x86::Asm;

// `fn(xs: *const f32) -> f32`: the sum of the eight floats at `xs`.
let mut a = Asm::new("sum8");
let xs = a.param("xs");
let v = a.ymm("v");
a.vmovups(v, xs.at(0));
let high = a.ymm("high");
a.vextractf128(high.xmm(), v, 1);
let sum = a.ymm("sum");
a.vaddps(sum.xmm(), v.xmm(), high.xmm());
let pair = a.ymm("pair");
a.vmovhlps(pair.xmm(), sum.xmm(), sum.xmm());
a.vaddps(sum.xmm(), sum.xmm(), pair.xmm());
a.vmovshdup(pair.xmm(), sum.xmm());
a.vaddss(sum.xmm(), sum.xmm(), pair.xmm());
a.ret_f32(sum.xmm());
let listing = kgen::render(&a.finish())?;
print!("{}", listing.text()); // where each variable lives, the code with its notes, the constants
```

## As a function

A procedural macro runs only code it depends on, so the attribute that names a kernel is declared beside the
kernels, by a proc-macro crate that depends on them and on `kgen-macros`:

```rust
#[proc_macro_attribute]
pub fn kernel(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // my_kernels::program(name: &str, params: &[(String, i64)]) -> Result<kgen::Program<kgen::Op>, String>
    kgen_macros::kernel(attr, item, my_kernels::program)
}
```

Its users declare the function, and the attribute gives it the kernel for a body:

```rust
#[my_kernels_macros::kernel(sum8)]
pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32;
```

The declaration is checked against the kernel's signature, and the kernel's constants become statics beside it.

## Layers

Each crate depends only on the layers inside it. `crates/kgen/tests/architecture.rs` checks this rule.

| layer | crate | what it knows |
|---|---|---|
| entities | `kgen-core` | programs over variables and over registers; the `Machine` port |
| use cases | `kgen-compile` | liveness, register allocation, the frame's layout; tested on a toy machine |
| adapter | `kgen-x86` | x86-64: the typed instruction builder, SysV, Intel syntax |
| composition | `kgen` | which machine and syntax the use cases run with |
| delivery | `kgen-macros` | a kernel as the body of a naked function, for the attribute beside the kernels |

A crate of kernels depends on `kgen-core` and `kgen-x86`; its proc-macro crate depends on it and `kgen-macros`.

## Guarantees

The output is deterministic:
- no crate depends on anything outside the workspace;
- clippy forbids hash maps and hash sets.

Checking the generated kernels is for the crates that hold them: a snapshot of every listing, so that any change
to any layer that moves an instruction shows in review, and runs against scalar references, bit for bit.

## Commands

```
./make.py check    # formatting, clippy and every test
./make.py fmt      # formats every crate
```
