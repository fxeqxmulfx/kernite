# kernite

kernite compiles kernels written in Rust's syntax, instructions over variables in a module that
`#[kernite_macros::kernels]` is on, to x86-64 assembly, as `#[unsafe(naked)]` functions in Intel syntax,
and to PTX for NVIDIA GPUs. [GUIDE.md](GUIDE.md) describes the language, the targets and the
guarantees: read it before changing what kernite does.

## Goals

Every change serves these five.

1. **Zero cost.** A kernel runs as fast as the best code written by hand for it. kernite adds no
   check, copy or call at run time that the kernel does not ask for.
2. **Minimum code.** The least code that does the job: no rule written twice, no dead code, no
   type or function that only hands a value on. A change that deletes code beats one that adds
   it.
3. **The whole spec.** kernite takes all of each target's spec, x86-64-v3 and PTX's ISA, in every
   form, and refuses what the spec leaves undefined. Every rule GUIDE.md states has a test.
4. **Safe where it can be.** A kernel kernite can prove safe is called without `unsafe`, as in
   NVIDIA's cuda-oxide for SIMT, where each thread writes only its own part of the output. Where
   safety would cost speed, zero cost wins: the call stays `unsafe`, and says why.
5. **Clean architecture.** Each layer depends only on the layers inside it, and a port belongs
   to the code that calls it. Every step leaves the architecture cleaner than it found it.

## Commands

- `./make.py check`: formatting, clippy on every target with `-D warnings`, every test, and the docs
  without a warning. It passes before every commit. It takes about a minute here.
- `./make.py fmt` formats every crate.
- `./make.py publish` publishes each crate that crates.io lacks at its version, one at a time, those
  it needs first. Publishing is the user's call alone.
- While working: `cargo test -p <crate>`, or `cargo test -p kernite --test x86` (or `ptx`, `lang`) for
  the language through a target.
- `KERNITE_BLESS=1 cargo test -p my-app` (or `-p microgpt`) rewrites the snapshots of the listings,
  after a change to the generated code made on purpose, for the diff to be reviewed.
- Rust 1.99.1, edition 2024. `make.py` runs under [uv](https://docs.astral.sh/uv/), as every Python
  script does. A script kept as a file is Python, not shell.

## Layout

The crates are layers, each depending only on those inside it. GUIDE.md's "Layers" table says what
each knows. `crates/kernite/tests/architecture.rs` holds the rule of which may depend on which, of
every kind of dependency, and a new crate needs its line there. Change both when the crates change.

- `kernite-core`: programs over variables and over registers.
- `kernite-lang`: the language: tokens, the parser, values, the evaluator, and the `Target` port a machine
  implements.
- `kernite-compile`: the allocator: liveness, register allocation, spilling, and the frame's slots; the
  ports `Machine` and `Spill` a machine implements for it. It also says which items every path of a
  program runs.
- `kernite-build`: what kernite does with a module of kernels, whatever machine compiles them: what
  its items are, the ports `Backend`, `Native` and `Device` a machine implements, and the use cases
  that call them. It depends on kernite-lang alone.
- `kernite-x86`, `kernite-ptx`: each machine, `X86` and `Ptx`: its forms, the language's binding to it, its
  printer.
- `kernite`: which machine each kernel compiles with, `X86` or `Ptx`; kernite-build's use cases do the
  compiling. `kernite-macros`: the proc macros.
- `kernite-cuda`, `kernite-cpu`: what the generated code runs on: the CUDA driver, and an x86 grid's
  threads and module memory. Neither depends on another crate.
- `examples/my-app`, `examples/microgpt`, `examples/matmul`: kernite used as a crate outside it
  would be. Their tests hold each kernel to a scalar reference, bit for bit, and each listing to
  `snapshots/`.
- `examples/readmes`: runs the example of each crate's `README.md` and of the repository's as a
  doctest, all together, built against the crates as a crate of a user's is. kernite-cpu's and
  kernite-cuda's cannot run in their own crates: they need kernite-macros, and `architecture.rs`
  lets neither depend on another crate.

## Rules the code keeps

- The output is a function of the source alone. Clippy denies `HashMap` and `HashSet`, and nothing
  comes from outside the workspace: `Cargo.lock` has no `source =` line.
- The generated code changes only on purpose. Every listing in `examples/*/snapshots/` and
  `crates/kernite/tests/builder/` is compared byte for byte, and a refactor changes none of them.
- Errors are behaviour. Tests assert the text of many, and the `line:column` they point at; keep both.
  A new error says what is wrong where it can be mended, in GUIDE.md's words.
- What rustc can do, rustc does. The macros write the Rust so that rustc checks, works out and says
  all it can of it, whether a type is a number say, and kernite does none of it again. kernite does
  itself only what rustc cannot: reading a kernel and compiling it, and reading the text that
  `x86_str` and `ptx_str` are given, which no rustc reads.
- `unsafe_code` is forbidden everywhere but in kernite-cuda; in kernite-cpu's `affinity` module, which calls the C
  library to read and set the CPUs a thread may run on, and in its test of where pages are, `tests/nodes.rs`; and in
  the examples, which call kernels. Each `unsafe` block says why it holds. kernite-cuda and kernite-cpu keep lints of
  their own for this, as a workspace's lints cannot be eased for a part of a crate.
- Every public item has a doc comment: `missing_docs` warns, and `./make.py check` denies warnings.
- PTX: take what `ptxas` takes beyond PTX's ISA where it reads it as the ISA reads the rest. Refuse every
  value `ptxas` or the instruction would cut, every value the ISA leaves undefined, and anything that looks
  like a `ptxas` bug. A toolchain bug, a `ptxas` crash say, gets a sentence by the form in
  `crates/ptx/src/forms.rs`, naming the tool's version and the input.

## Tests

- Fakes or the real thing, never mocks. kernite-compile's `Toy` machine is a fake. The language's tests
  run on the real targets, through `kernite::x86_str` and `kernite::ptx_str`, in `crates/kernite/tests/`;
  kernite-lang may not depend on a target, even for its tests.
- A test kernel really compiles: each variable written before it is read, each operand one the machine takes.
- Tests that need `ptxas` (`$PTXAS`, the path, or `/usr/local/*/bin`), `llvm-mc` (`$LLVM_MC`), GNU `as`
  (`$GNU_AS`), `strace` (`$STRACE`) or a GPU skip where it is missing, and say so. This machine has them all:
  CUDA 13.4 and an RTX 3050, `sm_86`, with 4 GB.

## How to work here

- Change code in a git worktree of your own, under `.claude/worktrees/`, never in the main checkout,
  where the user and other sessions work. Reading the main checkout is fine.
- Commit after any progress: each piece that passes `./make.py check` gets its own commit. A message is
  a sentence of what kernite now does, in GUIDE.md's plain words, then a body that says what changed
  and why, wrapped at 72 columns, as `git log` shows, ending with the `Co-Authored-By` line.
- Don't push, and don't merge into master: both are the user's call.
- Don't stop to ask. Choose the sensible default, say what you chose, and keep going. Ask only before
  something that cannot be undone or that leaves this machine.
- A crate's API may change where a task needs it. Make the least change that does the job, and keep
  no old form beside the new one.
- Write as GUIDE.md does: plain, short words, a value "known now" rather than at compile time, one
  thing a sentence. Doc comments, error messages and commit messages alike.

## Planning and review

Work is planned on [agent-kanban](.claude/skills/agent-kanban/SKILL.md), a task board in
`.kanban/board.db`, which git ignores. `agent-kanban guide` says how to use it.

- [DEVELOPER.md](DEVELOPER.md): a developer claims a task, does it on a branch of its own, and hands
  it in with evidence for each of its tests.
- [REVIEWER.md](REVIEWER.md): the reviewer merges it and approves it, or sends it back with what
  must change.

Developers are subagents the reviewer starts, each on Sonnet 5.5: the Agent tool's
`model: "sonnet"`. Their commits end `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
The reviewer reuses a developer where it can. A developer whose run has ended takes the next round
of its task, or a new task, by SendMessage, and keeps what it knows of the code. The reviewer
starts a new developer only when none is free, and gives a task to the one that knows its code.
At most three developers run at once; the next task starts when one ends its run.

A bug found in kernite, in a review, a test or a task, becomes a task of its own on the board,
tagged `kernite`, with tests that show it, and a developer resolves it. The reviewer files it and
reviews the fix, and does not fix it. A developer who meets one outside its task does not fix it
there: its hand-in gives the smallest kernel that shows it, and any way around it the task took.

Notes for the work in hand, its design, plan and gate, live beside the board in `.kanban/notes/`,
never in the repository.

The reviewer merges only code that is 10/10, with nothing left to change. Anything less goes back
for polishing, with every finding listed at once, so that one more round can finish it.
