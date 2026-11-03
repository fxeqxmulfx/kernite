# How a session does a task of the clean-architecture rewrite

You are one of several sessions working the board at once. A reviewer, in another session,
approves your work or sends it back, as [REVIEWER.md](REVIEWER.md) says. The bar is 10/10:
everything below is checked, and work with anything left to change comes back for polishing.
All paths are under `<main>`. Run `agent-kanban` there.

## 1. Take a task

Pick a name for this session, `dev-` and four letters of your choice, and use it in every call:

    agent-kanban agent register dev-abcd
    agent-kanban claim-next --agent dev-abcd --lease 14400

- `claim-next` takes the most urgent task that is free. Its reply's first line is `#<ID> [revN]
  Tnn: title` or `#<ID> Dn: title`.
- A `changes|...` line follows if the reviewer sent the task back: do what it says first.
- Then comes one line per acceptance test, `IDX|WHAT|INPUT|OUTPUT`.
- After about three hours, claim the task again (`agent-kanban claim <ID> --agent dev-abcd
  --lease 14400`) to keep it.
- If the reply is `idle, N open`, every task left waits for another: run `uv run
  .kanban/notes/ready.py --wait 1800`, then claim again. If it is `idle` alone, the board is
  empty: stop.

Read, in this order:

1. `.kanban/notes/design/design.md`: §4, the rules every task keeps, and your task's row and
   changes in §5;
2. the text your task's row points to, at those lines: it says what to do, in which files, by
   which tests. Its `file:line`s of code are START's, or the commit it names: design.md's second
   paragraph says how to read them;
3. the code you will change, in full, before you change it.

The map of the code as it was is `.kanban/notes/architecture-map.md`; its §9 lists the risks of
moving code. The reviewer's decisions are `.kanban/notes/reviewer-decisions.md`.

## 2. A design task (Dn)

A design task writes one file in `.kanban/notes/design/`, which its first test names, and
changes nothing else:

- no worktree;
- no change to the repository;
- no change to the board but your own task's claim and hand-in.

Read the code in `.claude/worktrees/clean-arch`, and only read it. Check every claim against the
code before you write it, and give its `file:line`. Then hand it in (§6), with the file's path as
evidence. Skip §3-§5.

## 3. Your worktree (Tnn)

If `.claude/worktrees/ka-<ID>` exists, the task came back from review: work there, on its branch,
as it is now (the reviewer may have rebased it). Otherwise make it:

    git -C <main> worktree add .claude/worktrees/ka-<ID> -b ka-<ID> clean-architecture

Change files only in your worktree. Never touch another worktree, the main checkout, the branches
`clean-architecture` or `master`, and never push.

## 4. Work

- Build and test with `CARGO_BUILD_JOBS=2`, and only what you need: `cargo check -p <crate>`,
  `cargo test -p <crate> [name]`, `cargo clippy -p <crate> --all-targets -- -D warnings`. Many
  sessions share twelve CPUs and 29 GB.
- Do not run `./make.py check`, the tests of `kernite-cuda`, or the examples' tests yourself: they
  use the GPU, which has 4 GB, and the gate runs them a few at a time.
- Keep to your task's text and design.md's changes to it. If the text is wrong about the code, do
  what the design means and keep every rule below. Write down what you did differently, and why,
  in your commit message and your hand-in. If it cannot be done as designed, say so: do not fake
  it.
- The rules every task keeps, which the gate checks:
  - the 45 listings stay byte for byte, and every message kernite gives stays byte for byte, at its
    position;
  - every test keeps its name. Tests may move, and their code may follow a new API. But every
    string literal of the start's test code stays in test code: a file in a `tests` directory,
    a file named `tests.rs`, or the body of a `#[cfg(test)]` item;
  - `examples/` stays byte for byte, and the Rust the macro writes for them, and for the gate's
    crates in `.kanban/notes/expand-fixtures/`, stays token for token, but for changes a review
    approved;
  - no `#[ignore]`, no package from outside, no `HashMap` or `HashSet`, and no `unsafe` outside
    kernite-cuda;
  - docs on every public item, and every crate on `[lints] workspace = true`;
  - tests use fakes or the real thing, never mocks.
- design.md §4 adds these rules:
  - a move is a move;
  - old paths keep working;
  - the staged copies design.md lists, and no others;
  - the stop rule for the conditional tasks;
  - edits placed where two tasks meet.
- Keep CLAUDE.md's five goals: zero cost, minimum code, the whole spec, safe where it can be,
  clean architecture. In this rewrite they mean:
  - each task leaves the architecture cleaner: no dependency that points outward, no rule
    copied into an adapter;
  - the generated code does not change, which the gate checks;
  - your task adds the least code it needs, with no type or function that only hands a value
    on, and leaves nothing dead. The gate prints the count: "code outside tests";
  - every new rule has a test.
- Write docs, comments and messages as README.md does: plain words, short sentences.
- Commit each step that builds and passes its tests, as CLAUDE.md says:
  - a first line saying what the change does;
  - a body saying why, wrapped at 72 columns;
  - the last line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## 5. The gate (Tnn)

When the task is done and committed, bring your branch onto the latest `clean-architecture`, then
run the gate:

    git -C .claude/worktrees/ka-<ID> rebase clean-architecture
    uv run <main>/.kanban/notes/verify.py <main>/.claude/worktrees/ka-<ID>

- On a Cargo.lock conflict in the rebase, take either side. Then run `cargo metadata --offline
  --format-version 1 > /dev/null`, which writes it again, and go on.
- The gate takes a few minutes the first time in a worktree, and may wait for a slot. It must end
  `VERIFY OK`.
- If it fails, read `target/verify.log` and `target/expanded-*.diff` in your worktree. Fix,
  commit, and run it again.
- Never change the gate, its baselines or `approved.json`. If you believe a check is wrong for
  your task, say so when you hand it in, with the evidence.

## 6. Hand it in

    agent-kanban submit-review <ID> --agent dev-abcd --pass 0 "EVIDENCE" --pass 1 "EVIDENCE" ...

- Give one `--pass` or `--fail` per acceptance test, by its index.
- Evidence is what shows the test passed:
  - the branch and commit (`ka-<ID>` at `abc1234`);
  - the gate's last lines (`VERIFY OK`, the counts);
  - the tests you ran, and their result;
  - for a design task, the file you wrote.
- Use `--fail`, with the reason, for anything not done. Never claim a pass you did not see.

Then take the next task (§1), until the board is empty. Your last message says:

- each task you did, its branch and commit;
- `VERIFY OK`, or what failed;
- what the reviewer should look at first: where you left the text, and why.
