# kgen

Kernels written as x86 instructions over variables, compiled to assembly that changes only when they do.

A kernel is Rust code that calls one builder method per instruction (`a.vpmaddubsw(p, p, x)`), on variables
instead of registers. `kgen` allocates the registers with no spills: if the variables that are alive together do
not fit, it fails and names them. It prints Intel syntax, and a proc-macro turns that into a `#[unsafe(naked)]`
function:

```rust
#[kgen_macros::kernel(iq2_xxs(ahead = 3))]
pub unsafe extern "sysv64" fn dot(row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, grid: *const u64) -> f32;
```

## Layers

Each crate depends only on the layers inside it. `crates/kgen/tests/architecture.rs` checks this rule.

| layer | crate | what it knows |
|---|---|---|
| entities | `kgen-core` | programs over variables and over registers; the `Machine` port |
| use cases | `kgen-compile` | liveness, register allocation, the frame's layout; tested on a toy machine |
| adapter | `kgen-x86` | x86-64: the typed instruction builder, SysV, Intel syntax |
| kernels | `kgen-kernels` | the kernels, written in the builder |
| composition | `kgen` | which machine and syntax the use cases run with |
| delivery | `kgen-macros` | a kernel as the body of a naked function |
| verification | `kgen-harness` | the delivered kernels against scalar references, bit for bit, and their speed |

## Guarantees

The output is deterministic:
- no crate depends on anything outside the workspace;
- clippy forbids hash maps and hash sets.

The generated code is checked in three ways:
- `snapshots/` holds every kernel's listing, so any change to any layer that moves an instruction shows in review;
- `kgen-harness` runs every kernel against a scalar reference that spells out the float operations in order;
- the results must match bit for bit.

## Commands

```
cargo test                                  # everything, the bit-exact runs included
KGEN_BLESS=1 cargo test -p kgen --test snapshots   # rewrite snapshots/ after a deliberate change
cargo build --release --example speed && taskset -c 0 target/release/examples/speed
```
