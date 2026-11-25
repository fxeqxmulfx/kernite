# Changelog

Every crate of kernite has the same version, and each release publishes all ten to crates.io. A release that may
break a crate built on the one before raises the minor version, as Cargo reads it: 0.6.0 may break what built with
0.5.0, and 0.3.2 may not break what built with 0.3.1. Unreleased is what master has that no release has yet, and a
release gives it its version.

## Unreleased

- kernite-cuda says how many bytes of a GPU's memory are free, and how many it has.
- kernite-cuda launches on a stream, and runs the work of two streams at once.
- kernite-cuda page-locks memory the host has, a `Vec`'s say, for kernels to reach until it is dropped.
- kernite-cuda makes a buffer on a GPU that grows and shrinks at the same addresses, `cuda.reserve::<T>(n)`.
- kernite-cuda orders the work of two streams with an event, and says the time between two events.
- kernite-cuda copies memory on a GPU, and zeros it, on the current stream without waiting.
- kernite-cuda writes the zeros of new memory on the current stream, not the null stream.
- kernite-cuda makes managed memory only where each GPU reaches it while the host does, which WSL2's does not.
- kernite-cuda records the work of a GPU stream as a graph, `stream.capture(..)`, and launches it, `unsafe`, as often
  as it is asked: 1100 launches as one.
- kernite-cuda's host reaches unified memory while kernels run, with no wait: as atomics, safe, `memory.atomics()`, and
  as a slice, `unsafe { memory.host_now() }`.
- kernite-cuda lends a part of a buffer or of unified memory, `xs.part(3..10)`, to a kernel or a copy.
- GUIDE.md names once the driver its facts of a driver hold on: NVIDIA's open kernel module 620.70.06, not 615.71.10.
- kernite-cuda's `Memory` names a slice registered for good too, and its `lend` and `reach` name only what they refuse.
- kernite builds the PTX of a module of megabytes, or of many kernels (GitHub issue 1).
- kernite finds a kernel's `# Safety` by its heading, as rustdoc does, and adds what it writes at the end of it.
- kernite-cpu's threads wait spinning for as long as their grid says, `Grid::on(&cpus).spin(d)`: a launch of twelve
  threads 5 ms after the last costs 2.5 microseconds, not 371.
- kernite-cpu runs a closure once for each block, on the threads of a grid, `launch_blocks(grid, n, kernel)`: a thread
  the system puts aside holds the launch back by the block it has, and by no other. A launch of twelve threads costs
  0.2 to 0.4 microseconds more for it, as each kept thread's queue now holds four launches at most.
- kernite writes x86's `cmpxchg` and `lock.cmpxchg` of each width, and places the variable they compare with in `rax`,
  as it places a call's arguments: `lock.cmpxchg([p], new, old)`.
- Breaks: a safe GPU kernel reaches each element whole, and kernite refuses `ld.global.u16` through a `*const u32`, as
  the host may store the whole element while the kernel runs; kernite-cuda's `Cuda::concurrent_managed_access` is gone,
  with the clause of an `unsafe` launch's contract it served; kernite-cuda's `Buffer::tail` and `View` are gone, as
  `xs.part(i..)` does what `tail(i)` did.

## 0.6.0 (2026-11-22)

- kernite leaves out a move into a register that holds the value already, the kernel's own too, in each spelling.
- The macros write the CPU features an x86 kernel needs into its `# Safety`.
- Breaks: kernite-compile's `Spill` says what an instruction copies, by `copies`.

## 0.5.0 (2026-11-22)

- kernite proves a kernel safe where it can, and Rust calls it with no `unsafe`: an x86 kernel of numbers, of arrays
  by reference or of slices, and a GPU kernel in which each thread works on an element of its own.
- An editor shows the doc of an x86 or PTX instruction where a kernel names it.
- kernite panics by no code of its own: an error goes on by `?`.
- x86 code is as rustc writes it: a frame in the red zone, an xmm where no ymm is needed, `vzeroupper` only where an
  upper half is set, `rbp` only for a frame aligned past 16, and a spill stored and loaded only where it must be.
- Rust tells of a missing doc at the item, and of a deprecated item where a kernel names it.
- Breaks: what could panic gives a `Result` or an `Option`, `Module::global` and `at` say; a kernel proved safe is no
  `unsafe fn`; the ports gain `doc`, `requires` and `Spill::kept`.

## 0.4.0 (2026-11-19)

- kernite needs Rust 1.100.
- A value names Rust's float constants by Rust's own paths, `core::f32::consts::PI`, and Rust checks their bits.
- PTX is as `ptxas` 13.5 takes it: `shfl` and `vote` without `.sync`, which PTX deprecated, are refused, and an
  `f16x2`, a `bf16` and a special register are taken where `ptxas` takes them.
- x86: a value lives across a call in any register the call keeps, a call that `ret` follows is a jump, and a kernel
  writes `vzeroupper` only where it may have set an upper half.
- An x86 static past `rip`'s reach is taken by its address, and one of 3 GiB builds as fast as one of 4 KiB. A GPU
  static of 6 GB is allocated when its module loads.
- Strings and names are read as Rust reads them: raw strings, each escape, Unicode's spaces, and names in NFC.
- Breaks: kernite-cuda's `Module::load` takes the memory a module has no room for; kernite-build's `Backend` lays a
  static out beside those before it; kernite-core's clobbers say how much of a register a call changes.

## 0.3.2 (2026-11-05)

- kernite-macros' tests name their made-up folders under /home again. The API is 0.3.1's.

## 0.3.1 (2026-11-05)

- Each crate has a README, its page on crates.io, with an example that runs. The language is in GUIDE.md. The API is
  0.3.0's.

## 0.3.0 (2026-11-05)

- A type in a message is written as Rust writes it.
- A raw name, `r#u64`, a macro's fragment, and a primitive written as a path, `core::primitive::u64`, are read as Rust
  reads them.
- Breaks: kernite-lang's `Delim` has `None`, and kernite-build's `ItemOut` has no `failed`.

## 0.2.0 (2026-11-05)

- kernite-cuda: a thread takes any GPU the driver shows, a buffer is copied to another GPU, and managed and
  page-locked memory are reached by the host and by a kernel at one address.
- kernite-cpu: a grid's threads run on the CPUs the caller gives, one on each, and each page of its memory is on its
  thread's NUMA node.
- Fixes of bugs found in the first tasks: types read as Rust writes them, `cfg` in a module, PTX's tuning directives,
  and a launch that says which limit of the GPU it is past.
- examples/matmul multiplies matrices at 93 to 99% of OpenBLAS's speed on the CPU, and at 100 to 126% of cuBLAS's on
  the GPU.
- Breaks: kernite-cpu's `Grid` has a lifetime and `launch` takes the kernel by value; kernite-cuda's `join` takes the
  parts of a text; kernite-build's port lays a static out and gives back an `Assembly`.

## 0.1.0 (2026-11-03)

The first release. crates.io has a crate named kgen, the project's name until then, so it is kernite.

- Kernels in Rust's syntax, written where they are called: `#[kernite_macros::kernels]` on a module compiles each
  function in it that names an ABI. Its body is instructions over variables, and kernite allocates the registers, and
  spills where they run out.
- What is not an instruction happens while the kernel compiles: `for` unrolls, `if` chooses, a helper is inlined,
  `#[instances]` compiles a generic kernel for each value, and modules share consts.
- x86-64: each instruction of x86-64-v3 in each form, in `#[unsafe(naked)]` functions in Intel syntax; SysV calls,
  stack memory, a module's memory, and a cooperative grid on threads.
- PTX: the instructions from Pascal to Ampere as `ptxas` takes them, the tensor cores' too, each with the oldest PTX
  and GPU it needs; shared, local, global and constant memory, calls, and cooperative launches. kernite-cuda loads and
  launches them through the CUDA driver.
- An error points at the tokens at fault.

### Before 0.1.0

- 2026-09-27: kgen starts as x86 kernels of quantized weights, instructions over variables compiled without spills.
- 2026-10-25: the kernels go, and kgen is a language of its own, for x86-64 and for NVIDIA GPUs.
- 2026-11-03: the crates are layers, each depending only on those inside it.
