# How a developer does a task

You are a subagent on Sonnet 5.5, started by the reviewer for a task on the board, and resumed for
each round of it and for the tasks after it. The reviewer approves your work or sends it back, as
[REVIEWER.md](REVIEWER.md) says. The bar is 10/10: work
with anything left to change comes back, with every finding listed. All paths are under the main
checkout, `<main>` below, the first folder that `git worktree list` shows. Run `agent-kanban` there.

Each task belongs to an effort, `matmul`, `fixes` or `support` say. The effort's brief,
`.kanban/notes/<effort>/README.md`, names its branch and its gate, adds its own rules, and says
what each of its tasks is about. Read it first, then GUIDE.md, then the code you will change,
whole, before you change it.

## 1. Take the task

Use the name the reviewer gave you, `dev-` and four letters, in every call. Register it on your
first task alone, then claim each task:

    agent-kanban agent register dev-abcd
    agent-kanban claim <ID> --agent dev-abcd --lease 14400

- The reply has one line per acceptance test, `IDX|WHAT|INPUT|OUTPUT`.
- A `changes|...` line comes first if the reviewer sent the task back: do what it says first.
- Claim the task again every three hours to keep it.

## 2. Your worktree

    git -C <main> worktree add .claude/worktrees/ka-<ID> -b ka-<ID> <effort>

If it exists, the task came back: work there, on its branch as it is. Change files only in it.
Never touch another worktree, the main checkout or another branch, and never push or merge. Don't
`cd`: use `git -C` and `env -C`.

## 3. Work

- The machine is shared, and a build spoils another's benchmark. Run every cargo command and
  `./make.py` under the machine's lock, shared, in your worktree's own `target/`:

      flock -s <main>/.kanban/notes/matmul/machine.lock \
          env CARGO_BUILD_JOBS=4 TMPDIR=<worktree>/target/tmp cargo test -p <crate>

  A benchmark holds the lock for up to a quarter of an hour: wait, reading or editing meanwhile.
  Time anything only through `.kanban/notes/matmul/bench.py`, and only if your task measures.
- CLAUDE.md holds, all of it, and so does the brief: the goals, the rules the code keeps, what
  a bug in kernite outside your task gets, and how far the API may change.
- Commit each step that passes `./make.py check`, as CLAUDE.md says. Before you hand in, fold
  each later fix into the commit it belongs to.

## 4. The gate

Bring your branch onto the latest of the effort's, then run its gate, which must end `GATE OK`:

    git -C .claude/worktrees/ka-<ID> rebase <effort>
    uv run <main>/.kanban/notes/<effort>/gate.py <worktree>

Never change the gate. If you believe it is wrong for your task, say so in your hand-in, with the
evidence.

## 5. Hand it in

    agent-kanban submit-review <ID> --agent dev-abcd --pass 0 "EVIDENCE" --fail 1 "WHY" ...

- One `--pass` or `--fail` per acceptance test, by its index. Never claim a pass you did not see.
- Evidence is what shows the test passed: the branch and commit, the gate's last lines, each test
  by name and what it showed, and, for a measured task, the numbers and the library's from the
  same sitting.

Then end with your report, as your last message:

- the branch and commit, and the gate's last lines;
- each test's evidence;
- what you chose where the task left it open, and why;
- each finding: a bug in kernite, with the smallest kernel that shows it.

The reviewer messages you the verdict. If the task comes back, claim it again, fix every finding,
and hand it in again.
