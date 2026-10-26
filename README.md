# kgen

Kernels written where they are called, in Rust's syntax read as instructions over variables: compiled for x86-64
to assembly that changes only when they do, and for NVIDIA GPUs to PTX.

## A kernel

`#[kgen_macros::kernels]` on a module compiles each function in it that names an ABI. Its body is instructions,
called by name, over variables declared with a type and no value; kgen allocates the registers.

```rust
#[kgen_macros::kernels]
mod kernels {
    /// The sum of the eight floats at `xs`.
    pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
        let v: ymm;
        vmovups(v, [xs]);
        let high: ymm;
        vextractf128(high.xmm, v, 1);
        let sum: ymm;
        vaddps(sum.xmm, v.xmm, high.xmm);
        let pair: ymm;
        vmovhlps(pair.xmm, sum.xmm, sum.xmm);
        vaddps(sum.xmm, sum.xmm, pair.xmm);
        vmovshdup(pair.xmm, sum.xmm);
        vaddss(sum.xmm, sum.xmm, pair.xmm);
        ret(sum.xmm);
    }
}

let sum = unsafe { kernels::sum8(xs.as_ptr()) };
```

`sum8` becomes an `#[unsafe(naked)]` function of the signature it declares, its body the kernel in Intel syntax,
and `kernels::LISTINGS` holds each kernel's listing for snapshots. Everything that is not an instruction happens
while the kernel compiles: `for` unrolls, `if` on a bool chooses, a function without an ABI is a helper inlined
where it is called, and `<const N: usize>` makes one generic. A generic kernel is compiled for each value its
`#[instances(1, 4)]` lists, each a kernel of its own, which Rust calls as it calls a generic function,
`kernels::dot::<4>(x, y, n)`; a value it does not list fails to build. [`kgen-lang`](crates/lang/src/lib.rs)
describes the language.

Stack memory a kernel declares is a variable, as a GPU kernel's local memory is: `let tables: local<[u32; 1024], 32>;`
is 4 KB of its frame at a multiple of 32, reached by its name, `mov(c.r32, [tables + b * 4])`, and `lea(p, [tables])`
takes its address. Until written, it holds whatever was there. kgen checks an address in it where the index is 0: what
it reaches lies inside, and `vmovaps` and `vmovdqa` reach it at a multiple of what they move; the index is the kernel's
to keep inside. A frame of more than a page is touched a page at a time on the way down, as Rust's functions are, so
that one past the end of the stack faults at the guard page there rather than reach past it.

When the variables alive together do not fit in the registers, kgen spills one to a slot of the frame: it stores it
after each instruction that writes it and loads it before each one that reads it. It spills, of the variable left
without a register and those holding the ones it could have, the one whose loads and stores are fewest for how long
it is alive, one in a loop counting ten times one outside, and spills again until all fit. The listing names each
spill, in its header and at each load and store. Sixteen accumulators of `dot` and the load beside them take 17 of the
16 vector registers, and two spill:

```
# dot::<16>(gpr, gpr, gpr) -> vec
# frame: 64 bytes aligned to 32, saves []
# acc15, spilled: 32 bytes at rsp
# acc14, spilled: 32 bytes at rsp + 32
...
    vmovups ymm14, ymmword ptr [rdi + 480]
    # acc15 reloaded
    vmovups ymm15, ymmword ptr [rsp]
    vfmadd231ps ymm15, ymm14, ymmword ptr [rsi + 480]
    # acc15 spilled
    vmovups ymmword ptr [rsp], ymm15
```

A spill moves a register with `mov` or `vmovups`, which leave the flags as they were, so it may go between a
comparison and its jump. Only when no spill makes room, as when one instruction names more variables than there are
registers, is it an error, which names the variables at the declaration of the one that found none.

## On a GPU

An `extern "ptx-kernel"` function is compiled to PTX. The module's kernels are one PTX module, `PTX`, which
`Kernels::load` hands to the CUDA driver; each kernel is then a method that launches it on a grid.
[`kgen-cuda`](crates/cuda/src/lib.rs) is the driver, opened when a GPU is asked for, so that a program built with
GPU kernels still runs where there is none.

```rust
#[kgen_macros::kernels]
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
        ..
    }
}

let cuda = kgen_cuda::Cuda::new()?;
let kernels = gpu::Kernels::load(&cuda)?;
let (xs, ys) = (cuda.upload(&x)?, cuda.upload(&y)?);
unsafe { kernels.saxpy(kgen_cuda::Grid::new(n.div_ceil(256), 256), 2.0, xs.ptr(), ys.ptr(), n)? };
ys.read_into(&mut y)?;
```

The PTX target knows every instruction of PTX's for the GPUs from Fermi, `sm_20`, to Ampere, `sm_86`, in each form
`ptxas` takes, [`forms`](crates/ptx/src/forms.rs), and checks each operand against them, and a `call`'s against what the
function it calls takes and returns. Where `ptxas` takes more than PTX's ISA has, and reads it as the ISA reads the
rest, kgen takes it too: `sust.p` with a cache operation, as `sust.b` has, or an `mma` fragment `(a, 8)`, a value beside
a variable. Where a value known now would be cut to fewer bits, by `ptxas` or by the instruction as the ISA has it, or
is one the ISA leaves undefined, kgen refuses it, and the table says so: `lop3`'s table is 8 bits, and `ptxas` takes a
bigger one and keeps its low 8; `shfl` reads its lane in the low 5 bits of `b`; and a barrier counts 1024 threads at
most, the most a CTA has. Where `ptxas` takes less, kgen does too: `st` stores a special register only as one of a
vector's. And kgen checks what `ptxas` leaves to the program: that an access stays in the memory it names, at a multiple
of the bytes it moves. Each form, and each of its words, says the oldest version of PTX and the oldest GPU `ptxas` takes
it at, back to PTX 1.0, which is at times older than the ISA says: `ptxas` takes `ld.mmio.acquire` for `sm_70`, which
the ISA has from `sm_75`. A module's `.version` and `.target` are the oldest that have every instruction its kernels
use: PTX 2.3 and `sm_20` at the least. Where `ptxas` is installed, a test hands it every form at what kgen says it
needs, at the newest version, and for each newer GPU to `sm_86`, which the driver compiles a module for when it runs
there, and `ptxas` must take it each time; and at the version before and on the GPU before, which it must refuse.

An `extern "ptx-func"` function is a `.func` that the module's kernels and functions call, itself among them. It
takes and returns PTX's types, and `call(g, gcd, (a, b))` calls it by its name, or `call(g, fp, (a, b))` at the
address `mov.u64(fp, gcd)` put in `fp`. Each call is checked against what the function takes and returns, and a call
of one that never returns, `-> !`, ends its caller's path there. A helper, a function without an ABI, is inlined
instead, and leaves no call behind.

An instruction's name is Rust's words: a word that starts with a digit takes a `_`, a `::` is `__`, and a keyword
takes `r#`, so `tex.2d` is `tex._2d`, `ld.global.L2::64B` is `ld.global.L2__64B`, and `cp.async` is `cp.r#async`. A
texture or a surface is its handle, the `u64` of a CUDA texture or surface object, in brackets with its coordinates:
`tex._2d.v4.f32.f32((r, g, b, a), [t, (x, y)])`.

Memory a kernel declares is a variable: `let tile: shared<[f32; 256], 16>;` is 256 floats of shared memory at a
multiple of 16, reached by its name, `ld.shared.f32(v, [tile + 16])`, and `shared<[f32]>` is as many as the launch
gives, `Grid::new(blocks, threads).with_shared(bytes)`. A kernel's `f16` and `bf16`, which Rust has not, are
kgen-cuda's [`f16` and `bf16`](crates/cuda/src/half.rs): their bits, converted to and from `f32` as `cvt` converts
them.

Memory of the module's is a `static` beside its kernels, declared once in `PTX`, which each kernel names as it names its
own, and which keeps what they write from one launch to the next: `static COUNTS: global<[u32; 16]>;` is 16 words of
global memory, `red.global.add.u32([COUNTS + 8], 1)`, and `static WEIGHTS: constant<[f32; 4]> = [1.0, 0.5, 0.25, 2.0];`
four floats of constant memory, which the kernels only read, `ld.r#const.f32(w, [WEIGHTS + 4])`, holding what it says
until something writes it. The host reads and writes each as `Kernels` gives it, a
[`Global`](crates/cuda/src/lib.rs) of Rust's type for its elements: `kernels.COUNTS().read()?`,
`kernels.WEIGHTS().write(&[2.0; 4])?`. kgen refuses what a module cannot hold: more than the 64 KB of constant memory it
has, or 4 GB of global memory, whose size `ptxas` writes in 32 bits.

A kernel whose blocks wait for each other is `#[cooperative]`, and its `grid.sync()` waits until every thread of the
grid has come to it, as CUDA's does. The driver launches it cooperatively, every block of the grid at once, so what a
thread keeps in its registers, or a block in its shared memory, lasts across the wait, where two kernels would write it
to memory and read it back. Its grid has at most as many blocks as the GPU runs at once, which `Kernels` says by a
method of the kernel's name: `kernels.normalize_blocks(256, 0)?` for blocks of 256 threads and no shared memory from
the launch. A grid of more is an error, which says how many. Such a kernel needs a GPU from Pascal's on, `sm_60`, and
kgen refuses `grid.sync()` where a thread of the grid might not come to it: under an `if`, in an `extern "ptx-func"`
function, or in a kernel that is not cooperative.

A kernel is named in PTX as C++ names a function in the namespaces of its workspace, its crate and its modules, so
that a profiler tells two crates' `saxpy` apart: `_ZN4kgen6my_app3gpu5saxpyE`, which `cu++filt`, Nsight and
`cuda-gdb` show as `kgen::my_app::gpu::saxpy`. An instance of a generic kernel is named as a template's,
`kgen::my_app::gpu::block_sums<(unsigned int)1024>`, and launched as Rust calls a generic method,
`kernels.block_sums::<1024>(grid, x, sums)`; the shared memory the launch gives is named as the kernel's static,
`kgen::my_app::gpu::reverse_runs::run`, and a static of the module's as a variable of its namespace,
`kgen::my_app::gpu::COUNTS`. The workspace is named as the directory of its `Cargo.toml` is, unless the manifest names
it, as kgen's does, so that a clone keeps the names wherever it is:

```toml
[workspace.metadata.kgen]
name = "kgen"
```

## Examples

[`examples/my-app`](examples/my-app/src/main.rs) uses kgen as a crate outside it would: `sum8`, `dot` for one
accumulator, for four and for sixteen, two of which spill, and `count_bytes`, by tables in its stack memory, for the CPU, and for the GPU `saxpy`, `block_sums` for blocks of 256 threads and of 1024
and `reverse_runs` in shared memory, `scale_f16`, written in place, `gcds`, which calls `gcd`, an `extern "ptx-func"`
function that calls itself, `histogram`, which counts in the module's global memory from launch to launch, and
`weigh`, by weights in its constant memory, with safe functions over them; and
`matmul` on the tensor cores, by `cp.async`, `ldmatrix` and `mma`, in a module of its own, since it needs an Ampere
GPU, `sm_80`; and in another, for Pascal's, `sm_60`, `normalize`, whose threads keep their floats in registers while
the grid waits for every block's sum, and `diffuse`, which takes any number of steps in one launch, the grid waiting
between them. Its tests hold each kernel to a scalar reference that does the same float operations in the same order,
bit for bit, or the tensor cores, which add in an order of their own, to products of integers, exact in any order;
and its listings to [`snapshots/`](examples/my-app/snapshots/).

```
cargo run -p my-app
KGEN_BLESS=1 cargo test -p my-app   # rewrites the snapshots after a deliberate change, for review
```

[`examples/microgpt`](examples/microgpt/src/main.rs) is Karpathy's
[microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95): a GPT of one layer that learns from
32,033 names to make up new ones. It runs on 25 kernels in x86 for the CPU, or on 25 in PTX for the GPU, and on a
plain Rust reference that does the same float operations in the same order. Its tests hold the three to each other
bit for bit, one operation at a time.

Its weights come from the same Mersenne Twister draws as Python's, and it trains on the same names in the same
order, so it prints what microgpt.py prints: the same twenty names, and the same loss at 998 of its 1000 steps. At
the other two the fourth decimal differs by one, where `f32` rounds differently from Python's `f64`. The tests hold
it to what Python drew and printed, in [`fixtures/`](examples/microgpt/fixtures/). It downloads the names on its
first run, as microgpt.py does.

```
cargo run --release -p microgpt              # the x86 kernels
cargo run --release -p microgpt -- gpu       # the PTX kernels, on the GPU
cargo run --release -p microgpt -- reference # plain Rust
```

## Layers

Each crate depends only on the layers inside it. `crates/kgen/tests/architecture.rs` checks this rule.

| layer | crate | what it knows |
|---|---|---|
| entities | `kgen-core` | programs over variables and over registers; the `Machine` and `Spill` ports |
| language | `kgen-lang` | kernels in Rust's syntax: parsing, evaluation, the `Target` port |
| use cases | `kgen-compile` | liveness, register allocation, spilling, the frame's layout; tested on a toy machine |
| adapters | `kgen-x86`, `kgen-ptx` | x86-64: its instructions, SysV, Intel syntax; PTX for NVIDIA GPUs |
| composition | `kgen` | which machine and syntax each kernel compiles with |
| delivery | `kgen-macros` | kernels as functions: `#[unsafe(naked)]`, or launchers of PTX |
| runtime | `kgen-cuda` | the CUDA driver: memory on the GPU, modules, launches; `f16` and `bf16`; depends on no other crate |

## Guarantees

The output is deterministic:
- no crate depends on anything outside the workspace;
- clippy forbids hash maps and hash sets.

Checking the generated kernels is for the crates that hold them: a snapshot of every listing, so that any change
to any layer that moves an instruction shows in review, and runs against scalar references, bit for bit, as in
the examples.

## Commands

```
./make.py check    # formatting, clippy, every test, and the docs
./make.py fmt      # formats every crate
```

kgen builds with Rust 1.99.1 and later, as the workspace's `rust-version` says. `make.py` runs under
[uv](https://docs.astral.sh/uv/).
