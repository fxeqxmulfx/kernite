# TMEM on Ampere and Pascal

A plan, written 2026-10-25; nothing of it is built yet.

Datacenter Blackwell gives each SM a tensor memory, TMEM, that its tensor cores accumulate into. This plan is for
kernels written against it, compiled by kgen for the GPUs that have it and for those that do not: Ampere, like
this machine's RTX 3050 (`sm_86`), and back to Pascal (`sm_61`), which has no tensor cores at all. There TMEM is
not emulated while the kernel runs but resolved while it compiles: kgen knows, before anything runs, which thread
holds each of its cells, and holds them in registers.

## TMEM

- `sm_100a` has 256 KB of it in each SM, beside the 256 KB of registers there: 128 lanes of 512 columns, each
  column 32 bits. Consumer Blackwell, `sm_120a`, has none; nor have Ampere and Pascal.
- A CTA takes columns with `tcgen05.alloc`, a power of two from 32 to 512, which waits while they are taken; gives
  them back with `tcgen05.dealloc`; and says it will take no more with `tcgen05.relinquish_alloc_permit`.
- Warp `w` reaches lanes `32·(w%4)` to `32·(w%4)+31`, and no others. In the `.32x32b` shape, `tcgen05.ld` and
  `tcgen05.st` give thread `t` of it lane `32·(w%4)+t`: each thread a run of columns of one lane.
- `tcgen05.mma` is issued by one thread. It reads A and B from shared memory through 64-bit descriptors, and
  accumulates D in TMEM: for M = 128, row `i` in lane `i`. It runs asynchronously; `tcgen05.commit` has an
  mbarrier arrive when the thread's MMAs so far are done.

## The idea

Emulated while the kernel runs, TMEM would be slow: it does not fit in shared memory, 256 KB against the 99 KB a
block may have on the 3050 and the 48 KB on Pascal, and in global memory every access is a trip through the
caches. That is worth having as a fake to test against (phase 3), and not as the way kernels run.

But on a machine without TMEM nothing of it need exist at run time, if kgen knows while compiling which thread
holds each cell and when it is written. In kgen it can:

- **Lanes are threads.** A lane is reached by the warps of its quarter alone, and in the `.32x32b` shape by one
  thread of each. A cell belongs to a thread, as a register does.
- **Columns are values.** The kernel declares its regions of TMEM, and kgen allocates them while compiling. An
  instruction names a column as an offset into a region: a constant, the counter of an unrolled `for`, a helper's
  `<const>`, all of which kgen has evaluated by the time it builds the instruction.
- **The schedule is in the text.** Which warp issues an MMA, which warps wait for it and on which barrier, is
  written in the kernel; with the warp known while compiling (below), kgen has each warp's program.

On Ampere, then, a thread's columns are registers kgen names, `tcgen05.ld` and `tcgen05.st` rename them, and
`tcgen05.mma` is `ldmatrix` and `mma.sync` over them, run by the warps that hold them; on Pascal it is the same,
with `fma.rn.f32` for `mma.sync`. What is asynchronous on Blackwell becomes the order of each warp's program, and
the overlap Blackwell gets from a tensor core that runs on its own comes instead from warps taking turns.

## What each operation becomes

| in the kernel | `sm_100a` | `sm_80`: Ampere, so the 3050 | `sm_61`: Pascal |
|---|---|---|---|
| a region, `let acc: tmem<[f32; 128]>;` | its columns of the one `tcgen05.alloc` kgen makes for all the kernel's regions, freed as the kernel ends | registers | registers |
| a column of it, `[acc + c]` | the allocation's address, plus the region's first column and `c`, at the warp's lanes | registers of the warp that reads it, one of the two that reach its lane, laid out as `mma.sync` leaves D | the same, each thread a block of rows by columns |
| `tcgen05.mma`, by an elected thread | as written, with descriptors kgen makes from the tiles' layouts | `ldmatrix` and `mma.sync.m16n8k16`, in each warp that holds cells of D, for those cells, where that warp waits for it (below) | the same, with `ld.shared` and `fma.rn.f32` along K, f16 and bf16 widened to f32, exactly |
| `tcgen05.commit`, to an mbarrier | as written | the warps that ran the MMA meet at a named barrier, and one of them arrives; where only they wait on it, the arrival and the wait both go | the same |
| `tcgen05.ld`, `tcgen05.st`, `.32x32b` | as written | registers renamed, and moved within the warp to a lane a thread and back, by shuffles or through shared memory: once a tile, not once a step of K | the same, through shared memory |
| `tcgen05.wait`, `tcgen05.fence` | as written | nothing: each warp's program is in order already | nothing |
| `cp.async`, and its arrival on an mbarrier | as written | as written | `ld.global` into registers and `st.shared` out of them, then the arrival |
| an mbarrier | as written | as written, with `test_wait` in a loop for `try_wait`, which is `sm_90`'s | a count of arrivals in shared memory: `atom.shared.add` arrives, a loop of `ld.volatile.shared` waits |
| `elect.sync` | as written | lane 0, as `elect` is `sm_90`'s | lane 0 |

TMA, the narrower formats and `cta_group::2` come later, in phase 5.

**Where an MMA runs without TMEM.** `mma.sync` keeps D in the registers of the warp that runs it, as Pascal's FMAs
do, so the warps that hold D run the MMA, and the warp that issued it on Blackwell does not. Its block (its waits
for full stages, its MMAs, its commits) moves into each warp that holds cells of D, to just before that warp waits
on the barrier the block last commits to. That is sound when the block and the wait are in the same pass of the
same loop, as they are when roles are `if`s on the warp inside one loop, as in the sketch below; when they are
not, kgen says where.

**Ping-pong, for nothing.** When two accumulators are held by two warpgroups, one each, the producer fills the
stages in the order of the tiles, so the warpgroup with the next tile starts its MMAs as the last tile's stages
come free. One warpgroup's epilogue then runs while the other's MMAs do: the schedule CUTLASS calls ping-pong,
here from nothing but the order the producer fills in.

## What does not carry over

- **Storage.** Blackwell has TMEM's 256 KB beside its registers; Ampere and Pascal have the registers alone, 256 KB
  an SM, for everything every thread holds. Accumulators can take about half: some 128×256 `f32`, such as two of
  128×128, each in a warpgroup at 128 registers a thread. A thread has at most 255, so one warpgroup cannot hold a
  128×256 accumulator, 256 a thread, at all; a kernel that keeps two of them, all 512 columns, does not fit, and
  kgen says which regions are alive together.
- **Speed.** The 3050's tensor cores are Ampere's, and `mma.sync` takes A and B from registers, so each warp loads
  its own from shared memory, where Blackwell's MMA reads it itself. Pascal has no tensor cores, and its MMAs are
  FMAs. The ceiling on each is the best matmul written for it by hand: a TMEM kernel should come to that, not to
  Blackwell.
- **Shared memory.** A block may have 99 KB of it on the 3050 and 48 KB on Pascal. A kernel sized for Blackwell
  fits neither unless it sizes its stages by the target, with `SM` (below).
- **Columns known only at run time** are an error, not a slow path: unroll the loop that picks one by its period,
  two for two accumulators.
- **Columns two warps read.** Warps `w` and `w+4` reach the same lanes. Each may hold columns of its own there,
  but a column the MMA writes for one and the other reads would have to cross between threads, through shared
  memory: an error until a kernel needs it.

## What kgen checks

For every target, while compiling:

- the kernel's TMEM fits: in 512 columns on `sm_100a`, in the registers elsewhere;
- its shared memory fits: 48 KB a block for `sm_61`; 99 KB for `sm_80`, the least that any device it runs on
  gives, and the launcher holds it to the device's own;
- each warp reaches only its own lanes: a kernel names a column, and kgen puts in the warp's lanes;
- `tcgen05.mma` and `tcgen05.commit` are issued by the thread `elect.sync` chose, the one choice kgen can see is a
  single thread: issued by every thread of a warp, an MMA runs 32 times;
- an MMA reads a tile in the layout its producer writes, since kgen makes both the descriptor and the producer's
  addresses from the tile's declaration;
- the registers. kgen-compile's liveness gives the most 32-bit values a thread has alive at once, in any warp's
  program; more than the warps leave a thread, `65536 / (32·warps)` down to a multiple of 8 and at most 255, is an
  error naming them, as an x86 kernel's is. ptxas would spill, and say so only when asked.

## In the language

The syntax is proposed; the decisions at the end are what to settle first.

- **The warp, known while compiling.** `#[warps(10)]` on a kernel, which takes `<const WARP: usize>` as a helper
  takes `<const N: usize>`. kgen evaluates the body once a warp; warps whose programs come out the same share one
  copy, and the kernel begins with one branch on `tid.x / 32` to its warp's. So `if WARP == 9 { .. }` chooses
  while compiling, as any `if` on a bool does. The entry says `.reqntid` for its warps, and its launcher checks
  that the grid has as many threads.
- **The target, known while compiling.** `SM` is the compute capability kgen is compiling for, 61, 80 or 100: a
  value, as `WARP` is, so that a kernel can fit each target, as in `const STAGES: usize = if SM >= 80 { 4 } else
  { 2 };`.
- **Shared memory and TMEM, as types.** `let tile: shared<[f16; 8192]>;` is memory in `.shared`, named by its
  address; `let acc: tmem<[f32; 128]>;` is 128 columns of TMEM. A tile an MMA reads names its layout,
  `shared<[[f16; 64]; 128], swizzle_128b>`, and kgen addresses its elements through that layout, for the
  producer's copies as for the MMA's descriptors. Where every access to a tile goes through its declaration, its
  layout is the target's to choose: on Pascal, K outermost, so that a step of K is a few vector loads. kgen lays
  out shared memory while compiling; past 48 KB it is dynamic, `.extern .shared`, and the launcher asks the driver
  for it.
- **Vector operands**, PTX's `{a, b, c, d}`: an array declared together, passed whole, `d` for `d0` to `d3`; or a
  tuple, `(a, b)`. Brackets are memory already.
- **Names Rust will not parse.** `#[kernels]` is an attribute, so Rust parses the module before kgen sees it, and
  refuses `cp.async`, `async` being a keyword; `.32x32b`, a word that starts with a digit; and every `::`, as in
  `cta_group::1`, `kind::f16` and `fence::before_thread_sync`. The same holds today of `ld.const`, which the PTX
  target takes and a `#[kernels]` module cannot write. Proposed: a keyword raw, `cp.r#async`; `_` before a leading
  digit, `._32x32b`; and `__` for `::`, as in `tcgen05.mma.cta_group__1.kind__f16`. The PTX target writes them
  back. Both lexers keep the `r#` in a name today; it should go.

A GEMM written for Blackwell, with a warp to load, a warp to issue MMAs and two warpgroups to take turns at the
results, would read:

```rust
const STAGES: usize = if SM >= 80 { 4 } else { 2 }; // Pascal gives a block 48 KB

/// D = A·Bᵀ, a 128×128 tile at a time.
#[warps(10)]
pub unsafe extern "ptx-kernel" fn gemm<const WARP: usize>(a: *const f16, b: *const f16, d: *mut f32, k: u32) {
    let a_stage: [shared<[[f16; 32]; 128], swizzle_64b>; STAGES]; // 32 of K a stage
    let b_stage: [shared<[[f16; 32]; 128], swizzle_64b>; STAGES];
    let acc: [tmem<[f32; 128]>; 2];
    let full: [shared<b64>; STAGES]; // a stage is loaded
    let empty: [shared<b64>; STAGES]; // a stage is read
    let done: [shared<b64>; 2]; // an accumulator is computed
    let free: [shared<b64>; 2]; // an accumulator is read
    ..
    label(tiles);
    for buf in 0..2 {
        if WARP == 8 {
            load(a, b, a_stage, b_stage, full, empty); // cp.async, a stage at a time
        }
        if WARP == 9 {
            issue(acc[buf], a_stage, b_stage, full, empty, done[buf], free[buf]); // the thread elect.sync chose
        }
        if WARP / 4 == buf {
            drain(acc[buf], done[buf], free[buf], d); // warps 0 to 3, or 4 to 7
        }
    }
    ..
    bra(tiles);
}
```

For `sm_100a` it is what it says. Elsewhere, warp 9's `issue` moves into warps `4·buf` to `4·buf+3`, before they
wait on `done[buf]`; `done` and `free` are then waited on only by the warps that arrive on them, and both go; and
warp 9 is left with nothing to do. Each warpgroup holds its accumulator at 128 registers a thread, inside the 200
that ten warps leave it; on Pascal its two stages, 32 KB, leave room in the 48 for what `tcgen05.ld` moves
through shared memory.

## Phases

Each ends with something running on the 3050, checked, and worth having if the next never comes.

### 1. The instructions, back to `sm_61`

What TMEM is lowered to, and useful alone: kernels written as they are by hand for Pascal and for Ampere.

- `kgen-ptx`. Each instruction knows the oldest target and PTX version that have it, and a module's header is the
  newest its kernels need, at the least `.version 7.5` and `.target sm_61`. No kernel in the examples needs more:
  their modules assemble and load as `sm_61` as they are, so they change in that line alone. Then Pascal's
  instructions, which every later GPU has too: `bar.arrive`, and `bar.sync` with a barrier and a count;
  `shfl.sync`; `dp4a`; `membar.cta`; `ld` and `st` of `.v2` and `.v4`; `cvt` between f16 and f32. And Ampere's:
  `ldmatrix` (`.x1`, `.x2`, `.x4`, `.trans`); `mma.sync.m16n8k16`, f16 or bf16 into f32; `cp.async` (`.ca` of 4, 8
  or 16 bytes, `.cg` of 16), with `commit_group`, `wait_group` and `mbarrier.arrive.noinc`; `mbarrier` `init`,
  `arrive`, `test_wait` and `inval`; `movmatrix`; `cvt` to and from pairs of f16 and of bf16. Types `b16` and
  `f16`, and pairs packed in 32 bits, `f16x2` and `bf16x2`. Shared memory declared in a kernel, as a tile with a
  layout if it has one, its elements addressed through it from a 32-bit base. Vector operands.
- `kgen-lang`. An `Operand` for a vector; types with parameters, as `shared<..>`; the spellings above, with `r#`
  taken off.
- `kgen-cuda`. A launch takes its dynamic shared memory. A kernel that needs more than 48 KB has the driver told,
  with `cuFuncSetAttribute`. A device says its compute capability and how much shared memory a block may have.
- `kgen-macros`. Each launcher passes its kernel's shared memory. A pointer to `f16` or `bf16` is a `Ptr<u16>` to
  Rust, whose own `f16` is unstable.

Tests. In `crates/kgen/tests/ptx.rs`, what each instruction comes to and what each wrong operand is told. ptxas
takes each module for the target its header names; CUDA 13's ptxas no longer knows `sm_61`, but it assembles
`.target sm_61` for `sm_75` and still refuses what `sm_61` lacks, which is what the test does now. A new example,
`examples/matmul`, holds two matmuls written by hand to a scalar reference: for `sm_61`, FMAs from shared memory,
each thread a block of D, bit for bit on any finite f16 inputs, the reference doing the same FMAs in the same
order; for
`sm_80`, `cp.async` into swizzled stages, `ldmatrix` and `mma.sync`, bit for bit on inputs whose sums are exact in
`f32`. Both run on the 3050, whose driver compiles `sm_61` PTX for it, and both are timed.

Done when both matmuls are exact on the 3050, their times written down: the baselines for what follows.

### 2. The warp, known while compiling

- `kgen-lang`. `#[warps(n)]` and `<const WARP: usize>` on a kernel; `kernel` takes the values of a kernel's
  const parameters, as a call to a helper gives them.
- `kgen`. A kernel is evaluated once a warp, and the warps whose programs are the same are merged.
- `kgen-ptx`. An entry of several programs, their labels kept apart, behind one branch on the warp; `.reqntid`.
- `kgen-compile`. The most values alive at once, which `kgen` holds to the warps' budget, blaming the
  declarations as x86's errors do.
- `kgen-macros`. `#[warps]` taken off; each launcher checks the grid's threads.

Tests. The text of a kernel with a producer warp and consumer warps meeting at named barriers, which every target
has; the budget's error. In `examples/matmul`, both matmuls again, warp-specialized, a producer warp filling the
stages for the others: at named barriers on `sm_61`, at mbarriers on `sm_80`. Exact as before, and timed against
phase 1's.

Done when both warp-specialized matmuls are exact on the 3050, and a kernel made to need too many registers is
refused with their names.

### 3. TMEM

- `kgen-lang`. `tmem<..>`.
- `kgen-ptx`. `tcgen05.mma` (`cta_group::1`, `kind::f16`: f16 or bf16 into f32, M = 128), `tcgen05.commit`,
  `tcgen05.ld` and `tcgen05.st` (`.32x32b`), `tcgen05.wait`, `tcgen05.fence`, and `elect.sync`, in three
  lowerings:
  - **native**, for `sm_100a` at `.version 8.6`: as written, with the one `tcgen05.alloc`, its
    `relinquish_alloc_permit`, and its `tcgen05.dealloc` behind a barrier every warp reaches on its way out;
  - **Ampere**, for `sm_80`: registers, `mma.sync`, and each MMA moved into the warps that hold its D;
  - **plain**, for `sm_80`: the fake the others are held to. TMEM is global memory, a buffer its launcher
    allocates, a CTA's 128 lanes of as many columns as the kernel declares; `tcgen05.ld` and `tcgen05.st` are
    loads and stores at the lane and the column; `tcgen05.mma` runs where it is issued, by the issuing warp, as
    loops of `fma.rn.f32` along K, in order, reading A and B through the tiles' layouts. It is slow, and a
    transcription of what the PTX ISA says the instructions do.
- `kgen-macros`. A module for each lowering: `PTX` for `sm_80`, `PTX_SM100A`, and `PTX_PLAIN`. `Kernels::load`
  takes the one for the device's compute capability, and a test can load the plain one.

Tests. In `crates/kgen/tests/ptx.rs`, what each instruction comes to in each lowering, and each error: a column
known only at run time, an MMA no elected thread issues, a column two warps read, too many registers. ptxas takes
the native module for `sm_100a` and the others for `sm_80`. In `examples/matmul`, the sketch's GEMM: in the plain
lowering, bit for bit with the reference on any finite f16 inputs; in the Ampere lowering, bit for bit with both on
inputs whose sums are exact in `f32`; both timed.

Done when the TMEM GEMM is exact on the 3050 in both lowerings, its native module assembles for `sm_100a`, and its
Ampere lowering runs within 10% of the faster of phases 1's and 2's `sm_80` matmuls.

### 4. Pascal

- `kgen-ptx`. A fourth lowering, for `sm_61`. TMEM's cells are registers, each thread a block of rows by columns
  of its warp's lanes. `tcgen05.mma` is `ld.shared` and `fma.rn.f32` along K, in order, a column of A by a row of
  B at a time, f16 widened by `cvt` and bf16 by a shift, both exactly; the tiles are kept with K outermost.
  `cp.async` is loads into registers and stores out of them, a stage's loads all issued before its stores. An
  mbarrier is a count of arrivals in shared memory, a phase being so many of them: `membar.cta` and
  `atom.shared.add` arrive, and a loop of `ld.volatile.shared` waits. `tcgen05.ld` and `tcgen05.st` move values
  within the warp through shared memory, a few columns at a time.
- `kgen-lang`. `SM`; and `if` as a value, as `STAGES` is written, where it is a statement today.
- `kgen-macros`. `PTX_SM61`, and `Kernels::load` by compute capability: for 10.0 the native module; from 8.0 on,
  Ampere's; from 6.1 to 7.5, Pascal's.

Tests. In `crates/kgen/tests/ptx.rs`, what each instruction comes to for `sm_61`, and a kernel refused for more
than 48 KB of shared memory. In `examples/matmul`, the TMEM GEMM in the Pascal lowering, on the 3050: bit for bit
with the plain lowering on any inputs, since for each cell both do the same FMAs in the same order; and timed
against the faster of phases 1's and 2's `sm_61` matmuls.

Done when the TMEM GEMM in the Pascal lowering matches the plain lowering bit for bit on the 3050, on any inputs,
and runs within 10% of those matmuls there. There is no Pascal GPU here, so its speed on one goes unmeasured.

### 5. The rest of tcgen05

- TMA, `cp.async.bulk.tensor`, its tensor maps made on the host and passed as `.param .align 64 .b8 [128]`;
  elsewhere, the producer's copies.
- Formats the 3050's tensor cores lack. `kind::f8f6f4` converted to f16, exactly, since each e4m3, e5m2, e2m3,
  e3m2 and e2m1 value is an f16; the block-scaled kinds with their scales applied in f32, a block of K at a time.
  `kind::tf32` and `kind::i8` as `mma.sync`'s own. On Pascal every format widens to f32 by bit operations, and
  `kind::i8` is `dp4a`.
- `cta_group::2`. Elsewhere each CTA computes its own half of M and loads all of B itself, there being no
  distributed shared memory before `sm_90`.
- A in TMEM, `tcgen05.cp`, the `.16x` shapes.

None of it runs natively here: the native lowering is assembled, never run, until there is an `sm_100` to run it
on.

## Not in this plan

Hopper (`sm_90`) as a target of its own, with `wgmma`, TMA, clusters and `setmaxnreg`; Volta's and Turing's tensor
cores, `mma.sync` `m8n8k4` and `m16n8k8`, those GPUs taking Pascal's lowering meanwhile; kernels across the CTAs
of a cluster; and running any of the native lowering.

## Decisions to make

1. **How a kernel names its warps.** `#[warps(n)]` and `<const WARP: usize>`, as above, recommended: it is how
   helpers take values already. Or roles as the arms of a `match` on the warp.
2. **How shared memory and TMEM are declared.** As the types `shared<..>` and `tmem<..>` of a `let`, recommended;
   or as items of the module.
3. **How PTX's names Rust refuses are written.** `r#`, `_` and `__`, recommended: mechanical both ways, with no
   table to keep. Or a table of names in the PTX target.
4. **32-bit addresses.** For `.shared` alone, recommended: there they are the norm, and a 64-bit one costs a
   register. Elsewhere a 64-bit base, as now.
5. **Where the plain lowering keeps TMEM.** In global memory, recommended: it holds the same kernel at any size.
   Shared memory would leave the launch as it is, but the sketch's two accumulators, 128 KB, do not fit in 99 KB.
6. **How the owners arrive for the issuer.** A named barrier and one arrival, recommended, which keeps the counts
   the kernel gave `mbarrier.init`. Or every owner warp arriving, with kgen changing those counts.
7. **Mbarriers where there are none.** On `sm_61`, a count of arrivals in shared memory, recommended: it keeps
   what every use means, though a waiting warp polls. Or named barriers, which wait in hardware, but only where
   kgen can match each wait to the arrivals it waits for, pass for pass, and a ring's first pass, its waits met
   before anything arrives, defeats that.
8. **Which module a device gets.** By its compute capability, in `Kernels::load`, recommended; or the caller's
   choice.

## What this rests on

Checked on this machine, 2026-10-25, except where it says otherwise.

**The machine.** An NVIDIA GeForce RTX 3050 Laptop GPU: compute capability 8.6, 16 SMs, 4 GB. An SM has 65536
registers and 100 KB of shared memory; a block gets 48 KB by default, and 99 KB, 101376 bytes, when it asks.
Static `.shared` stops at 48 KB: ptxas refuses 49168 bytes. CUDA 13.4's ptxas, V13.4.92, assembles for `sm_100a`,
`sm_103a`, `sm_110a`, `sm_120a` and `sm_121a`, so native output can be checked here, though not run; it no longer
knows `sm_61`.

**Pascal**, from NVIDIA's documentation, there being no Pascal GPU here: compute capability 6.1 gives a block
48 KB of shared memory and an SM 96 KB, 65536 registers an SM and 255 a thread, no tensor cores, and f16
arithmetic at a small fraction of f32's. NVIDIA's 580 drivers are the last for Pascal
([Phoronix](https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW)); `.version 7.5` needs one from CUDA
11.5 on.

**PTX for `sm_61`, as ptxas took it.** ptxas refuses `--gpu-name sm_61` (`Value 'sm_61' is not defined for option
'gpu-name'`), so these were assembled at `.version 7.5`, `.target sm_61`, for `sm_75` and for `sm_86`. ptxas still
holds such a module to `sm_61`: there it says `Instruction 'nanosleep' requires .target sm_70 or higher`.

```
.shared .align 16 .b8 tile[1024];
ld.global.v4.b32 {%r2, %r3, %r4, %r5}, [%rd0];
st.shared.v4.b32 [%r0], {%r2, %r3, %r4, %r5};
bar.arrive 1, 64;
bar.sync 2, 64;
ld.shared.v4.f32 {%f0, %f1, %f2, %f3}, [%r0];
mov.b32 {%h0, %h1}, %r2;
cvt.f32.f16 %f4, %h0;
fma.rn.f32 %f0, %f4, %f1, %f0;
shfl.sync.idx.b32 %r6, %r2, 3, 31, -1;
shfl.sync.bfly.b32 %r7, %r2, 1, 31, -1;
dp4a.s32.s32 %r8, %r2, %r3, %r4;
atom.shared.add.u32 %r9, [%r0+512], 1;
ld.volatile.shared.u32 %r10, [%r0+512];
membar.cta;
```

The driver, CUDA 13.4's, compiled `.target sm_61` PTX for the 3050 and ran it right: two warps meeting at
`bar.arrive 1, 64` and `bar.sync 1, 64`, `shfl.sync`, `dp4a`, an atomic in shared memory. The modules of
`examples/my-app` and `examples/microgpt`, their headers made `.target sm_61`, assemble with
`--warning-as-error` and load.

**PTX for `sm_80`**, at `.version 7.8`, and so for the 3050's `sm_86`:

```
cp.async.cg.shared.global [%r0], [%rd0], 16;
cp.async.mbarrier.arrive.noinc.shared.b64 [%r0+1024];
cp.async.commit_group;
cp.async.wait_group 0;
mbarrier.init.shared.b64 [%r0+1024], 32;
mbarrier.arrive.shared.b64 %rd1, [%r0+1024];
mbarrier.test_wait.shared.b64 %p0, [%r0+1024], %rd1;     // and .parity, each in a loop
bar.sync 1, 128;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r2, %r3, %r4, %r5}, [%r0];
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r6, %r7}, [%r0];
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0, %f1, %f2, %f3}, {%r2, %r3, %r4, %r5}, {%r6, %r7}, {%f0, %f1, %f2, %f3};
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ..;
shfl.sync.idx.b32 %r8, %r2, 3, 31, -1;
movmatrix.sync.aligned.m8n8.trans.b16 %r9, %r8;
cvt.rn.f16x2.f32 %r10, %f1, %f0;
st.global.v4.f32 [%rd0], {%f0, %f1, %f2, %f3};
```

with `.extern .shared .align 128 .b8 smem[];`, addressed as `mov.u32 %r0, smem;`, and `.reqntid 288` on the
entry. `movmatrix` needs `.version 7.8`, and `cp.async` `sm_80`; `mbarrier.try_wait` and `elect.sync` need
`sm_90`.

**PTX for `sm_100a`**, at `.version 8.6`, which 8.5 is not enough for:

```
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r0], 32;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
tcgen05.st.sync.aligned.32x32b.x2.b32 [%r1], {%r2, %r2};
tcgen05.wait::st.sync.aligned;
tcgen05.ld.sync.aligned.32x32b.x2.b32 {%r3, %r4}, [%r1];
tcgen05.wait::ld.sync.aligned;
elect.sync %r2|%p1, -1;
@%p1 tcgen05.mma.cta_group::1.kind::f16 [%r0], %rd0, %rd1, %r1, %p0;   // D; A's, B's and the instruction's descriptors; accumulate
@%p1 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r5];
tcgen05.fence::before_thread_sync;
tcgen05.fence::after_thread_sync;
mbarrier.try_wait.parity.shared::cta.b64 %p2, [%r5], %r1;
tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r1, 32;
```

For `sm_120a`, ptxas says `Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'`.

**Layouts**, from the PTX ISA. `.32x32b` gives thread `t` of warp `w` lane `32·(w%4)+t`, and `.xN` N columns of
it. For M = 128, row `i` of D is lane `i`. `mma.sync.m16n8k16` leaves thread `l` of the warp D's rows `l/4` and
`l/4+8`, at columns `2·(l%4)` and `2·(l%4)+1`.

**Rust's parser**, rustc for edition 2024, on a module behind `#[cfg(any())]`, which Rust parses, as it parses the
module an attribute is on, and never compiles. It refuses `ld.const`, `cp.async`, `.32x32b`, `cta_group::1` and
`fence::before_thread_sync`; it takes `ld.r#const`, `cp.r#async`, `._32x32b`, `._2d`, `cta_group__1.kind__f16`,
`fence__before_thread_sync`, `mbarrier__arrive__one.shared__cluster`, and tuples as operands. Stable rustc,
1.99.1, refuses `f16` as a type: `the type f16 is unstable`.

**The driver.** `cuFuncSetAttribute` with `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` (8) lets a kernel have
more than 48 KB; a device's attributes 75 and 76 are its compute capability, and 97 its shared memory a block when
asked for.
