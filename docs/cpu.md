# Closing the CPU's gaps with the GPU

Notes for the work, in the agreed order: what reading the code found before each part started, and
the plan. Steps 1 to 3, stack memory, spilling and a CPU grid, were done on the branch `cpu-grid`,
from master e9426cf, and merged into master at 1a154ab. Steps 4 to 6, calls, memory of the
module's, and consts from outside a module, were done on `cpu-calls`, from there, and merged into
master. The State section at the end says how far it has come.

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

## 4. Calls

PTX's kernels call functions; an x86 kernel calls none.

- What PTX has (`call` in `crates/ptx/src/lang.rs`):
  - An `extern "ptx-func"` function is a `.func` that the module's kernels and functions call,
    itself among them.
  - `call(r, f, (a, b))`, `call(f, a)` and `call(f)` call it by its name. `mov.u64(fp, f)` takes
    its address; `call(r, fp, (a, b))` calls through it as the operands' types say, and
    `call(r, fp, (a, b), (f, g))` as one of the functions given, checked against each.
  - Each call is checked against what the function takes and returns. A call of one that never
    returns, `-> !`, ends its caller's path.
- What x86 has:
  - The language hands the target a function's name as `Given::Func`, which x86's `inst` refuses:
    "an x86 kernel calls no function".
  - The allocator gives each variable one register for its whole life, and pins each parameter
    and result to its register throughout (`Program::params`, `Program::results`). Nothing else
    names a register: no instruction kgen has for x86 reads or writes one of its own.
  - The prologue counts on the stack pointer being a multiple of 8 (`ALIGNED` in
    `crates/x86/src/intel.rs`). A call needs a multiple of 16.

The plan:

- **What a kernel calls.** Any `extern "sysv64"` function of its module that is not
  `#[cooperative]`, itself among them, and an instance as `f::<4>`. On x86 a function the kernels
  call is one Rust can call too, so it needs no ABI of its own, as PTX's `.func` does. A
  cooperative kernel runs on a grid: it is launched, not called.
- **The language** is PTX's: `call(r, f, (a, b))`, `call(f, (a, b))`, `call(f, a)`, `call(f)`.
  An argument is a variable of the class SysV passes the parameter in, a `gpr` for an integer or
  a pointer and an `xmm` for a float, or a value known now: an integer, or a float, which x86
  reads from memory. `r` is a `gpr` or an `xmm`, as the function returns. `lea(fp, [f])` puts `f`'s address in `fp`, for `call(r, fp, (a, b))` and
  `call(r, fp, (a, b), (f, g))`. A function `-> !` ends each way out in `ud2`, a new
  instruction, or in a call of one that never returns.
- **What SysV asks of a call.** The callee keeps to it:
  - the integers in `rdi`, `rsi`, `rdx`, `rcx`, `r8` and `r9`, the floats in `xmm0` to `xmm7`,
    and the result back in `rax` or `xmm0`;
  - the callee may change `rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8` to `r11` and every vector
    register, and keeps `rbx`, `rbp` and `r12` to `r15`;
  - the stack pointer is a multiple of 16 at the `call`.
- **The registers a call changes.** An instruction says which registers it changes besides its
  operands. A variable live across it takes none of them. So a `gpr` live across a call is kept
  in `rbx` or `r12` to `r15`. A vector live across a call spills, since the callee may change
  any vector register.
- **Arguments and results in their registers.**
  - The x86 target moves each argument to a variable of its own, which it pins to the register
    SysV passes it in. It moves the result from one pinned to `rax` or `xmm0`: `mov rdi, rbx`,
    `call {f0}`, `mov rbx, rax`.
  - Each lives from its move to the call, or from the call to its move, so pinning it throughout
    costs nothing.
  - A variable moved to or from one prefers its register, so that a move is often to the register
    the value is in already. Kernels that call nothing are allocated as before.
  - The listing leaves out a move to the register it is from, where that changes nothing: a
    64-bit `mov`, and a `vmovaps` or `vmovups` of a `ymm`. A move of a 32-bit register or of an
    `xmm` clears the rest, so it stays. This drops `diffuse`'s `mov rsi, rsi` too.
- **A parameter or a result in the way.** One pinned throughout clashes with a call that changes
  its register while it is live, and with an argument pinned there. Then it splits:
  - a parameter moves on entry to a variable of its own, of its name, which the allocator places
    as any other: `x=rdi,rbx` in the header;
  - a result moves into its register before each return.

  Each splits once at most, as each spills once at most. The move is a new instruction of the
  `Spill` port: `mov`, or `vmovaps`.
- **The stack.** SysV calls a function with the stack pointer 8 past a multiple of 16. A kernel
  that calls pads its frame by 8 where its pushes and its slots would leave it there, so that the
  stack pointer is a multiple of 16 at each call.
- **The macro.** A call is `call {f0}`, an operand of `naked_asm!`: `f0 = sym gcd`, or the
  instance's function in the hidden module, by a path from where the caller is. The listing ends
  with what each `{f0}` names. A function that does not compile leaves a stub of its name and of
  each of its instances, so that its callers still compile.
- **Not in it: dividing.** `div` reads and writes `rdx:rax`, registers of its own, as a call does,
  and the pinned variables would make it possible later. So the CPU's `gcd` is Stein's, by shifts
  and subtractions, where the GPU's is Euclid's, by remainders. Decided: later.
- **Tests:**
  - kgen-compile's Toy machine calls too. Its interpreter writes junk to every register a call
    changes, so a test fails if a value was kept in one across a call. Splits are run on the
    interpreter as well.
  - Exact listings in `crates/kgen/tests/x86.rs`:
    - a call and its moves;
    - a value kept in `rbx` across one, and a vector spilled across one;
    - the padding;
    - a function that calls itself;
    - a call through a variable;
    - a function that never returns.

    Each error is checked too, with line:column.
  - my-app's `gcds` for the CPU calls `gcd` for each pair. It keeps its pointers, its count and
    its index in the registers a callee keeps. It is checked against the reference the GPU's is
    held to.
  - my-app's `sum16` adds what `sum8` gives for each half, so that a float lives across a call.
    It is checked against `sum8`'s reference bit for bit.

## 5. Memory of the module's

PTX's kernels share memory of their module's; an x86 kernel takes memory by pointer alone.

- What PTX has:
  - `static COUNTS: global<[u32; 16]>;` and `static WEIGHTS: constant<[f32; 4]> = [..];` beside
    the kernels.
  - A kernel names a static as it names memory it declares, `[COUNTS + 8]`. A static keeps what
    the kernels write from launch to launch, and the kernels only read `constant` memory.
  - The host reads and writes each through `kernels.COUNTS()`, a `kgen_cuda::Global`.
- What x86 has:
  - The language asks the target for a static by `Target::global`, which x86 leaves at its
    default: "this machine has none" (`crates/lang/src/eval.rs`).
  - Its `var` refuses `global<..>` and `constant<..>` as a GPU's memory.
  - The macro compiles every static for PTX, and a static alone gives a module `PTX` and
    `Kernels` (`expand` in `crates/macros/src/lib.rs`).

The plan:

- **Whose memory.** A module's statics are its kernels': in the GPU's memory when they are PTX
  kernels, and in the process's when they are x86 ones. A module with kernels of both and a
  static is an error that says to split it. A module of statics and no kernels is the GPU's, as
  now. Decided: this, so that `global` and `constant` say the same on both, rather than a type
  of the CPU's own for its statics.
- **The types** are PTX's. The kernels read and write `global<..>`, and only read
  `constant<..>`, which kgen checks, as it does for PTX. The host writes both.
- **Reaching it.**
  - x86 reaches a static at a displacement from `rip`, with no register beside it:
    `vmovups(w, [WEIGHTS])`, `mov([COUNTS + 8], t)`.
  - `lea(p, [COUNTS])` puts its address in a register, for an index: `[p + i * 4]`. An index
    beside the static's name is an error that says so.
  - kgen checks the displacement against the static's bounds, as it does stack memory's, and the
    alignment `vmovaps` needs against the static's. It refuses a write to `constant` memory. A
    write through an address `lea` took is the kernel's to keep from.
- **The listing** writes the address as `[rip + {COUNTS} + 8]`, an operand of `naked_asm!` that
  the macro names by `sym`.
- **The host's side.**
  - The macro makes each static a Rust `static` of the module's: a `kgen_cpu::Global` of its
    elements.
  - Its methods are `kgen_cuda::Global`'s, `kernels::COUNTS.read()`, `.write(&[..])` and
    `.zero()`, but they return no `Result`, since nothing fails.
  - kgen-cpu has no `unsafe`, so the elements are atomics of their width. The host loads and
    stores them, and the kernels reach them as the plain memory they are.
  - A kernel's safety contract says that nothing else touches the module's memory it uses while
    it runs.
- **Tests:**
  - listings and errors in `crates/kgen/tests/x86.rs`;
  - my-app's `histogram` and `weigh` for the CPU, against the references the GPU's are held to,
    the counts kept from call to call.

## 6. Consts from outside a module

microgpt has its sizes in `examples/microgpt/src/model.rs`, and again in each of its kernel
modules, `cpu.rs` and `gpu.rs`, which assert that the copies agree. `#[kernels]` takes no
arguments (`crates/macros/src/lib.rs:122`). A macro sees only its own tokens, so it cannot read a
const of Rust's.

The plan:

- `#[kgen_macros::consts]` on a module of consts gives the module to Rust as it is. It adds a
  hidden macro that hands the consts' tokens on to the macro it is given.
- `#[kgen_macros::kernels(consts = crate::model::sizes)]` expands to a call of that macro, which
  calls the kernels macro back with the consts first.
  - The kernels read them as the module's own.
  - Rust's code in the module sees them by a `use` that the macro adds.
  - A name that both the module and the consts have is an error.
- Tried on rustc 1.99.1: a `macro_rules!` that a macro makes, re-exported by `pub(crate) use`, can
  be called by its path from any module of the crate, before it or after. So the consts are the
  crate's own, not another crate's.
- A module of consts holds what kgen reads: numbers, floats, bools, arrays of them, and
  expressions over them. microgpt's `Layer`s stay in `model.rs`, written in terms of them.
- microgpt then has each size and each of `exp`'s constants once, and its asserts that the copies
  agree go. That it builds and its tests pass is the test, with a doctest for the error of a name
  given twice.

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
- Step 3, the CPU grid, is done:
  - `#[cooperative]` goes on an `extern "sysv64"` kernel too (`cooperative()` in
    `crates/kgen/src/lib.rs`). The x86 target (`crates/x86/src/lang.rs`) gives such a kernel one
    hidden parameter after the others: the address of the thread's place, three words that
    `kgen-cpu` lays out, the thread's index, the thread count, and the barrier's address. So a
    cooperative kernel takes five integers at most, and returns nothing.
  - `grid.thread` and `grid.threads` are memory in the place, which a thread reads, as in
    `mov(t, grid.thread)`; kgen refuses a write, `lea`, another `grid.` name, and either in a
    kernel that is not cooperative.
  - `grid.sync()` is inline (`sync`). A thread reads the generation, counts itself in with
    `lock xadd`, and the last to come resets the count and moves the generation on; the others
    spin on it with `pause`. x86-TSO makes this enough: the locked add orders what each thread
    wrote before it, and the reset is stored before the new generation. kgen refuses it under
    an `if`, with operands, and in a kernel that is not cooperative; another `grid.` call is no
    barrier. `crates/x86/src/op.rs` has `lock xadd` and `pause`.
  - The runtime is a crate of its own, `kgen-cpu` (`crates/cpu`), std only and without
    `unsafe`. `launch` starts a thread for each of the grid's but the first, which is the
    caller's, behind a gate, so that none runs the kernel unless all have started. It refuses
    a grid of no threads, and of more than `most()`, `available_parallelism`.
  - The macro (`grid_launcher` in `crates/macros/src/lib.rs`) puts the naked functions in the
    hidden module, each taking the place last, and makes the kernel's name the launcher:
    `kernels::normalize(kgen_cpu::Grid::new(4), x, sums, n)?`. A pointer crosses to the threads
    as its address, `expose_provenance`, and each thread makes it a pointer again. A kernel that
    does not compile leaves a launcher that never runs.
  - Tests: the barrier's exact listing, in a kernel's body and in a helper's, and each error,
    in `crates/kgen/tests/x86.rs`; my-app's `normalize` and `diffuse` for the CPU, against the
    GPU's references bit for bit, on one thread to four, with their snapshots. Both pass on all
    12 threads of this machine as well.
  - The tests run on four threads at most, since `cargo test` runs tests beside each other and
    a spinning thread takes a core.
- Steps 1 to 3 are done, and merged into master at 1a154ab, which is pushed.
- Step 4, calls, is done, on the branch `cpu-calls`:
  - The allocator (`crates/compile/src/alloc.rs`). An instruction's `clobbers` are the registers
    it changes besides its operands, which no variable live across it takes. `Program::pinned`
    holds the variables that stay in one register throughout, which never split or spill: a
    call's arguments and its result. A parameter or a result that clashes with a call, or with
    an argument, splits once, into a variable of its own (`x=rdi,rbx` in the header). A variable
    moved to or from a pinned one prefers its register. The new errors are `FixedTwice`,
    `FixedClash` and `FixedChanged`; `OutOfRegisters` says when a call changed the registers.
  - The x86 target (`call` in `crates/x86/src/lang.rs`) has `call(r, f, (a, b))` and its shorter
    forms, `lea(fp, [f])`, `call(r, fp, (a, b))` of variables, `call(r, fp, (a, b), (f, g))`,
    `ud2`, and `-> !`. `crates/kgen/src/lib.rs` gives it the `Prototype` of each of the module's
    `extern "sysv64"` functions and instances, which each call is checked against; a cooperative
    one it may not call. The checks of a call through a gpr with the functions given are each
    function's: a variable passes the same whatever its callee makes of it, and a value known
    now is of one type, so they cannot disagree.
  - The listing (`crates/x86/src/intel.rs`) has `call {f0}`, `lea rax, [rip + {f0}]`, and what
    each names at its end, `# f0: gcd`. The header says the padding, `and 8 more so that it calls
    at a multiple of 16`, and `-> !` when no way out returns. A move to the register it is from
    is left out where it changes nothing.
  - The macro passes `f0 = sym gcd`, or `sym super::__kgen_tile::I1` for an instance, and leaves
    a stub for each instance of a function that does not compile.
  - Tests: the Toy machine's, its interpreter writing junk to each register a call changes; the
    exact listings and every error in `crates/kgen/tests/x86.rs`; my-app's `gcds`, Stein's `gcd`
    on 64 bits, and `sum16`, against Euclid's and `sum8`'s references, with their snapshots.
  - Decided: `div` later, so the CPU's `gcd` is Stein's.
  - Not done: a value moved to the register a call wants though it is there already (`mov rbx,
    rdi` then `mov rdi, rbx`), and a tail call, which is a call and a `ret`.
- Step 5, memory of the module's, is done, on `cpu-calls`:
  - Decided: a module's statics are its kernels'. `kgen::owner` (`crates/kgen/src/lib.rs`)
    says whose: the process's when its functions are all `extern "sysv64"`, else the GPU's. A
    module of both kinds of kernel and a static is an error at its first static.
  - `kgen-core` has `StaticId` and `Base::Static`, and a program names its statics,
    `Program::statics`. The listing writes an address in one as `[rip + {COUNTS} + 8]`
    (`crates/x86/src/intel.rs`), so `kgen::x86_statics` refuses a static named as a listing
    names a constant or a function, `c0` or `f0`.
  - The x86 target (`crates/x86/src/lang.rs`) has `Static`, a static as the host lays it out:
    its element, length, alignment, and the bytes it holds. `Target::global` gives a kernel the
    statics `kgen::x86` hands it, and `within` checks an address in one as it does one in stack
    memory: inside it, or one past its end for `lea`, at the multiple `vmovaps` needs, with no
    index beside the static's name, and no write to constant memory. A static is of Rust's
    numbers, 2 GB at most, which a displacement of 32 bits reaches, aligned to a page at most.
  - The language reads a negated literal as one number, as Rust does, so `-128i8` is in range
    (`value` in `crates/lang/src/eval.rs`).
  - `kgen_cpu::Global<T, N, A>` (`crates/cpu/src/global.rs`) is a static's memory: an atomic of
    each element's width, at a multiple of `A`, made in a `const fn` from what the static holds.
    The host reads, writes and zeroes it without `unsafe`, and nothing it does fails.
  - The macro makes each static a Rust `static` of its name, `::kgen_cpu::Global<u32, 16, 4>`,
    and passes `COUNTS = sym COUNTS` to each kernel whose listing names it. A kernel's safety
    contract names the statics that it, and the functions it calls on down, name; one that names
    none says nothing of them. In a module of both kinds of kernel, an x86 kernel that names a
    static is a stub beside the error.
  - Tests: every listing and error in `crates/kgen/tests/x86.rs`, `kgen::owner`'s among
    them; `Global`'s layout and the host's reads and writes in kgen-cpu; my-app's `histogram`
    and `weigh` for the CPU, against the GPU's references, with their snapshots, the counts kept
    from call to call, written by the host, and wrapping past `u32::MAX`. The safety contracts
    were checked by hand, in the expansion of my-app and of a crate whose kernels reach statics
    through calls, recursion among them.
  - Not done: an instruction that writes the memory it reads, as `add([COUNTS + 8], 1)`, which
    kgen's x86 forms have for no memory yet, so a kernel loads, adds and stores; and `movsx`.
- Step 6, consts from outside a module, is done, on `cpu-calls`:
  - `#[kgen_macros::consts]` (`consts` in `crates/macros/src/lib.rs`) gives its module to Rust as
    it is, and refuses what kernels could not read there: a const that is not `pub`, one whose
    value kgen cannot work out (`kgen_lang::constant`, which works out one constant of a module),
    and any item but a const. It adds `__kgen_consts!`, a `macro_rules!` re-exported
    `pub(crate)`, which calls `kgen_macros::__kernels!` with the consts' tokens and the tokens it
    is given.
  - `#[kernels(consts = path)]` becomes `path::__kgen_consts! { (path) module }`, and
    `__kernels!` reads the consts before the module's items, so that its kernels read them as
    its own. Rust's code in the module sees them by `use path::{E, ..};`, the path as Rust reads
    it where the module is: from `crate` or `::` as it is, `self` as `super`, and any other with
    `super::` before it. A const, function or static of the module's own of one of their names
    is an error at it.
  - A proc macro called from a `macro_rules!` has that macro's call site, whose local names are
    the macro's own, so a name the macro wrote missed the same name in the module's tokens:
    "cannot find value `x`". Each token the macro writes takes the span of the path instead,
    which is the module's (`SITE`), and a guard resets it, panicking or not.
  - A const named as a keyword, `r#type`, is named so where the macro marks it read, which was
    `let _ = type;` before, for a module's own consts as well.
  - The module's own attributes, `//!` among them, come first in what the macro writes, as Rust
    has them. The `use` went before them, and so did step 5's error of a module of both kinds of
    kernel and a static, which Rust refused as well.
  - microgpt's `model::consts` holds its sizes, where its embeddings are, and `exp`'s constants,
    each once. `model` re-exports them, `cpu.rs` and `gpu.rs` take them, and the reference reads
    `exp`'s constants there. It is `consts`, not `sizes`, since it holds more than sizes. The
    copies are gone, and the asserts that they agree; what is left asserts what the kernels take
    of the sizes, as `WTE == 0`.
  - Tests: the doctests of `kgen-macros`, one that runs and a `compile_fail` one for a name given
    twice; `a_constant_works_out_on_its_own` in `crates/kgen/tests/lang.rs`; and microgpt, which
    builds, whose tests pass, and whose listings are as they were. Every error, a relative path, a
    raw name, and a module given consts with a generic kernel, a cooperative one, a static, an
    `impl` and a GPU kernel were checked in crates of their own.
  - Not done: consts from another crate, whose macro is the crate's alone, as planned.
- Steps 4 to 6 are done, the last at cca66bf, and merged into master, which is not pushed.
