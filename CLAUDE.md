# kgen

kgen compiles kernels written in Rust's syntax, instructions over variables in a module that
`#[kgen_macros::kernels]` is on, to x86-64 assembly, as `#[unsafe(naked)]` functions in Intel syntax,
and to PTX for NVIDIA GPUs. [README.md](README.md) describes the language, the targets and the
guarantees: read it before changing what kgen does.

## Goals

Every change serves these five.

1. **Zero cost.** A kernel runs as fast as the best code written by hand for it. kgen adds no
   check, copy or call at run time that the kernel does not ask for.
2. **Minimum code.** The least code that does the job: no rule written twice, no dead code, no
   type or function that only hands a value on. A change that deletes code beats one that adds
   it.
3. **The whole spec.** kgen takes all of each target's spec, x86-64-v3 and PTX's ISA, in every
   form, and refuses what the spec leaves undefined. Every rule README states has a test.
4. **Safe where it can be.** A kernel kgen can prove safe is called without `unsafe`, as in
   NVIDIA's cuda-oxide for SIMT, where each thread writes only its own part of the output. Where
   safety would cost speed, zero cost wins: the call stays `unsafe`, and says why.
5. **Clean architecture.** Each layer depends only on the layers inside it, and a port belongs
   to the code that calls it. Every step leaves the architecture cleaner than it found it.

## Commands

- `./make.py check`: formatting, clippy on every target with `-D warnings`, every test, and the docs
  without a warning. It passes before every commit. It takes about a minute here.
- `./make.py fmt` formats every crate.
- While working: `cargo test -p <crate>`, or `cargo test -p kgen --test x86` (or `ptx`, `lang`) for
  the language through a target.
- `KGEN_BLESS=1 cargo test -p my-app` (or `-p microgpt`) rewrites the snapshots of the listings, after
  a change to the generated code made on purpose, for the diff to be reviewed.
- Rust 1.99.1, edition 2024. `make.py` runs under [uv](https://docs.astral.sh/uv/), as every Python
  script does. A script kept as a file is Python, not shell.

## Layout

The crates are layers, each depending only on those inside it. README.md's "Layers" table says what
each knows. `crates/kgen/tests/architecture.rs` holds the rule of which may depend on which, of every
kind of dependency, and a new crate needs its line there. Change both when the crates change.

- `kgen-core`: programs over variables and over registers, and the allocator's ports `Machine` and `Spill`.
- `kgen-lang`: the language: tokens, the parser, values, the evaluator, and the `Target` port a machine
  implements.
- `kgen-compile`: liveness, register allocation, spilling, and the frame's slots.
- `kgen-x86`, `kgen-ptx`: each machine: its forms, the language's binding to it, its printer.
- `kgen`: what compiles each kernel with which machine. `kgen-macros`: the proc macros.
- `kgen-cuda`, `kgen-cpu`: what the generated code runs on: the CUDA driver, and an x86 grid's threads
  and module memory. Neither depends on another crate.
- `examples/my-app`, `examples/microgpt`: kgen used as a crate outside it would be. Their tests hold each
  kernel to a scalar reference, bit for bit, and each listing to `snapshots/`.

A rewrite to clean architecture is under way on the branch `clean-architecture`. Its plan is on the
board (below). Until it is merged, master's layout is the one above.

## Rules the code keeps

- The output is a function of the source alone. Clippy denies `HashMap` and `HashSet`, and nothing
  comes from outside the workspace: `Cargo.lock` has no `source =` line.
- The generated code changes only on purpose. Every listing in `examples/*/snapshots/` and
  `crates/kgen/tests/builder/` is compared byte for byte, and a refactor changes none of them.
- Errors are behaviour. Tests assert the text of many, and the `line:column` they point at; keep both.
  A new error says what is wrong where it can be mended, in the README's words.
- `unsafe_code` is forbidden everywhere but kgen-cuda, where each `unsafe` block says why it holds, and the
  examples, which call kernels.
- Every public item has a doc comment: `missing_docs` warns, and `./make.py check` denies warnings.
- PTX: take what `ptxas` takes beyond PTX's ISA where it reads it as the ISA reads the rest. Refuse every
  value `ptxas` or the instruction would cut, every value the ISA leaves undefined, and anything that looks
  like a `ptxas` bug. A toolchain bug, a `ptxas` crash say, gets a sentence by the form in
  `crates/ptx/src/forms.rs`, naming the tool's version and the input.

## Tests

- Fakes or the real thing, never mocks. kgen-compile's `Toy` machine is a fake. The language's tests run
  on the real targets, through `kgen::x86_str` and `kgen::ptx_str`, in `crates/kgen/tests/`; kgen-lang
  may not depend on a target, even for its tests.
- A test kernel really compiles: each variable written before it is read, each operand one the machine takes.
- Tests that need `ptxas` (`$PTXAS`, the path, or `/usr/local/*/bin`), `llvm-mc` (`$LLVM_MC`), GNU `as`
  (`$GNU_AS`) or a GPU skip where it is missing, and say so. This machine has them all: CUDA 13.4 and an
  RTX 3050, `sm_86`, with 4 GB.

## How to work here

- Change code in a git worktree of your own, under `.claude/worktrees/`, never in the main checkout,
  where the user and other sessions work. Reading the main checkout is fine.
- Commit after any progress: each piece that passes `./make.py check` gets its own commit. A message is
  a sentence of what kgen now does, in the README's plain words, then a body that says what changed and
  why, wrapped at 72 columns, as `git log` shows, ending with the `Co-Authored-By` line.
- Don't push, and don't merge into master: both are the user's call.
- Don't stop to ask. Choose the sensible default, say what you chose, and keep going. Ask only before
  something that cannot be undone or that leaves this machine.
- Write as the README does: plain, short words, a value "known now" rather than at compile time, one
  thing a sentence. Doc comments, error messages and commit messages alike.

## Planning and review

Work is planned on [agent-kanban](.claude/skills/agent-kanban/SKILL.md), a task board in
`.kanban/board.db`, which git ignores. Developers claim a task, do it on a branch of its own, and hand
it in with evidence for each of its tests. A reviewer approves it, or sends it back with what must
change. `agent-kanban guide` says how. Notes for the work in hand live beside the board in
`.kanban/notes/`, never in the repository.

The reviewer merges only code that is 10/10, with nothing left to change. Anything less goes back
for polishing, with every finding listed at once, so that one more round can finish it.
