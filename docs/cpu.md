# Closing the CPU's gaps with the GPU

Notes for the work on the branch `cpu-grid`, which starts at master e9426cf: what reading the code
found before it started, and the plan, in the agreed order: stack memory, then spilling, then a
CPU grid. The State section at the end says how far it has come.

## What is there already

- `kgen-core` has stack slots: `Slot { size, align }`, `SlotId`, `Base::Slot`, `Program::slots`,
  and `Frame { saved, offsets, size, align }`.
- `kgen-compile` lays them out. `frame::layout` (`crates/compile/src/frame.rs`) puts each slot
  after the one before, at its alignment. The frame's size is a multiple of the largest alignment.
- The x86 printer handles a frame. `prologue` and `epilogue` (`crates/x86/src/intel.rs`) push
  `rbp`, keep the caller's `rsp` in it, then `and rsp, -align` and `sub rsp, size`. A
  `Base::Slot` prints as `[rsp + offset + disp]`. The allocator never hands out `rsp` or `rbp`
  (`GPRS` in `crates/x86/src/sysv.rs`).
- The language is what's missing. The x86 target never makes a slot (`slots: vec![]`,
  `crates/x86/src/lang.rs:335`). Its `fn var` (`:629`) takes only `gpr`, `ymm` and `xmm`, and its
  `fn memory` (`:501`) builds only `Base::Reg` and `Base::Const`.

## 1. Stack memory

Copy PTX's `local<..>`. `let buf: local<[f32; 64], 32>;` is 256 bytes of the kernel's frame, at a
multiple of 32. Brackets name it, as in `vmovaps([buf + 32], v)`, `vmovss(x, [buf + i * 4])` and
`lea(p, [buf])`.

- PTX's version, to copy:
  - `shape` (`crates/ptx/src/lang.rs:1581`) parses the type.
  - `new_memory` (`:309`) declares it.
  - `symbol` (`:828`) refuses an access before the start, past the end, or needing more
    alignment than the memory has.
- What x86's target needs:
  - `Var` needs a memory variant, a slot.
  - `fn var` needs a `local<` arm that pushes a `Slot`.
  - `fn memory` needs to accept a slot as the base and make a `Base::Slot`.
  - `fn view` needs to give a helper's parameter the same type, as PTX's does.
- x86 allows an index beside a slot, `[rsp + i*4 + off]`. Then only the displacement can be
  checked against the bounds, which the docs should say.
- The aligned moves, `vmovaps` and `vmovdqa`, fault at an address that is not a multiple of 32
  (16 for an xmm). Refuse them unless the slot's alignment and the displacement guarantee one.
- **Stack probes.** A frame larger than a page can jump past the guard page. Then an overflow
  silently corrupts memory instead of faulting. kgen refuses silent corruption, so do one of
  these:
  - cap the frame at about a page, counting the alignment padding;
  - or touch each page on the way down, as Rust's own `probestack` does.

  Decided: touch each page.
- Until written, stack memory holds whatever was there, like PTX's local memory. To decide:
  document this, or zero the memory in the prologue. Decided: document it.
- Tests:
  - exact listings, and errors with line:column, in `crates/kgen/tests/x86.rs`;
  - a my-app kernel that uses it, with a `.s` snapshot, tested against a reference.

## 2. Spilling

kgen deliberately does not spill today; it says which variables hold the registers. These places
say so, and change with it:

- the doc of `allocate`, `crates/compile/src/alloc.rs:19`
- `OutOfRegisters`, `crates/compile/src/error.rs:41`
- `README.md:42`
- `dot`, `examples/my-app/src/main.rs:22`
- the tests `sixteen_accumulators_and_a_load_do_not_fit_and_the_error_says_where` and
  `up_to_fifteen_accumulators_fit` (`crates/kgen/tests/x86.rs:85`, `:91`)

The plan:

- **Choosing a spill.** When colouring leaves a variable without a register, pick a variable to
  spill. The choice is deterministic, so the same program still gives the same code.
- **The rewrite.**
  - Give the spilled variable a slot in `Program::slots`.
  - Store it after each write.
  - Load it into a short-lived variable before each read.
  - A `ReadWrite` operand gets both.
  - Then allocate again.
- **A port for loads and stores.** kgen-compile knows no opcodes, so the instruction set has to
  make the loads and stores. That needs a new port beside `Machine`
  (`crates/core/src/machine.rs`), such as "store a class to a slot" and "load a class from a slot".
  x86 implements it with `mov` and `vmovups`. The Toy machine in `crates/compile/src/tests.rs`
  implements it too, as a fake, not a mock.
- **Reporting.** No spill is silent: the listing says what spilled and where, with a note at each
  load and store and a line in its header.
- **To decide:** spill always, or keep the error unless the kernel asks for spilling (an
  attribute)? I'd spill always, with the listing naming each spill. Decided: spill always.

## 3. A CPU grid

The goal: run a kernel on N threads, with `grid.sync()` as a barrier, as `#[cooperative]` does on
the GPU.

- **Allowing it.**
  - `cooperative()` in `crates/kgen/src/lib.rs` refuses `#[cooperative]` on an `extern "sysv64"`
    kernel.
  - `an_x86_kernel_is_not_cooperative` (`crates/kgen/tests/x86.rs:297`) tests that refusal.
  - The macro's docs say `#[cooperative]` is for GPU kernels only.
- **The thread's inputs.** Each thread needs its index, the thread count and the barrier's
  address. They could come as hidden SysV parameters, using three of the six integer registers, or
  as one pointer to a context. The language could name them as PTX names `tid.x`. Today an x86
  kernel refuses such names: "`{name}` is not defined".
- **The barrier.** It is inline, since an x86 kernel calls nothing:
  - a counter and a generation;
  - a thread arrives with `lock xadd`;
  - the last thread to arrive resets the counter and bumps the generation;
  - the others spin on the generation with `pause`.

  `crates/x86/src/op.rs` has neither `lock xadd` nor `pause` yet.
- **Where `grid.sync()` may go.** The rules come from PTX's `grid_sync()` and `GRID_SYNC` in
  `crates/ptx/src/lang.rs`: never under an `if`, and only in a cooperative kernel or a helper
  inlined there.
- **How many threads.** A spinning thread needs a core to itself, just as every block of a GPU's
  grid has to be resident. The launcher caps N at `std::thread::available_parallelism` and says
  how many, as `{name}_blocks` does for a grid.
- **The launcher.** The macro makes it beside the naked function (`naked()`,
  `crates/macros/src/lib.rs:596`):
  - It uses `std::thread::scope`, spawning N - 1 threads and running the last on the caller's.
  - Raw pointers are wrapped so they can cross to other threads.
  - The crates take no dependencies, so it is std only. It goes in the generated code, or in a
    small runtime crate. A new crate needs a place in `tests/architecture.rs` and in the README's
    table.
- **Tests:**
  - the barrier's listing;
  - my-app's `normalize` and `diffuse`, run on the CPU grid and checked bit for bit against the
    GPU's references.

## State

- The branch `cpu-grid` starts at master e9426cf, which is pushed.
- Step 1, stack memory, is done:
  - `local<..>` in `crates/x86/src/lang.rs`: `new_memory`, `shape`, and `within`, which checks an
    address where the index is 0.
  - The probes in `prologue` (`crates/x86/src/intel.rs`): unrolled up to three pages, a loop
    past that. The stack pointer is a multiple of 8 already, so a frame aligned to 8 or less
    has no `and`, and its alignment adds nothing to the count.
  - my-app's `count_bytes`, for 4 tables and for 16, whose 16 KB the probe loop touches; its
    tests run both against a reference. A test runs a kernel with a 1 MB frame on a thread of
    64 KB in a process of its own, and sees Rust report the overflow at the guard page. Without
    the probes the same kernel dies of a plain segfault, past the guard page.
- Step 2, spilling, is done:
  - The port is `Spill`, beside `Machine` in `crates/core/src/machine.rs`: how wide a register
    of a file is, and the instructions that store one in a slot and load it back. `SysV` uses
    `mov` and `vmovups`, to slots aligned to their size. The Toy machine's are a fake; its tests
    run each program before allocation and after, on a small interpreter, and compare.
  - `allocate` (`crates/compile/src/alloc.rs`) colours as before. When a variable is left
    without a register, `spill::choose` takes, of it and the variables that hold the registers
    it could have, the one with the fewest loads and stores for how long it is live, one in a
    loop counting ten times. `spill::rewrite` gives that one a slot, a store after each write
    and a load before each read, each into a variable of its own; then it colours again. A
    parameter is stored on entry, and a result loaded into its register before each return.
  - Nothing spills twice, nor a variable spilling made, nor one an instruction writes as it
    jumps. When nothing else may, `OutOfRegisters` is the error still.
  - The listing's header names each slot that keeps a spill, `# acc15, spilled: 32 bytes at
    rsp`, and a note before each load and store says `acc15 reloaded` or `acc15 spilled`.
  - my-app's `dot::<16>` spills two accumulators and matches its reference bit for bit.
  - Not done: one slot for variables never live together, a load folded into the instruction
    that reads it, a loaded value kept for the next read.
- Next: step 3, the CPU grid.
