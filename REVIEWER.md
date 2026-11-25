# How the reviewer reviews a task

**The bar is 10/10** (the user's rule, 2026-10-27). Merge only code with nothing left to
change: every check below holds, and there is no name, doc line, test or line of code you would
still write differently. Anything less goes back with `request-changes` for polishing, with
every finding listed at once, each with its `file:line` and what to write instead, so that one
more round finishes it. A finding of taste is still a finding; say that it is one. A point
outside the task's lines that the task did not make worse is not a reason to hold it: put it
on the task that touches those lines next, or on a task of its own.

Developers are sessions of their own, and each bug in kernite is a task of its own, as CLAUDE.md
says. With each task, the reviewer writes in the effort's brief what its developer needs: the
files, the commands and the evidence to bring, as [DEVELOPER.md](DEVELOPER.md) has a developer
work. Only one developer whose task measures runs at a time, as another's build or benchmark
spoils its numbers. The reviewer checks a bug against the code before it files it, and a limit
GUIDE.md states is not a bug; it does not fix a bug itself.

Run the commands from the main checkout, where `review.py` means
`uv run .kanban/notes/<effort>/review.py`, of the task's effort. For each task in review, as it
comes:

1. `agent-kanban claim ID --agent reviewer`.
2. `review.py show ID`: the history and evidence, the commits, the files, and the test code
   touched. The developer's report comes by mail: `agent-kanban inbox --agent reviewer`.
3. Read the task's tests, the effort's brief, and the whole diff, `review.py diff ID [PATH..]`.
   Refuse when any of these fails:
   - **Does what the task says, and only that.** No dead code left behind, no unrelated change,
     no TODO in place of the work.
   - **Layers.** Every new `use` and every dependency points inward, as
     `crates/kernite/tests/architecture.rs` says. A port is owned by its caller.
   - **Behaviour.** A listing changes only where the task changes the generated code, and every
     message kernite gives keeps its text and its `line:column` unless the task changes it.
   - **Tests.** New behaviour has a new test, a fake or the real thing, never a mock. No test is
     weakened. Every rule GUIDE.md states has a test, and a test that skips says so.
   - **Words.** Docs, messages and commit messages in GUIDE.md's plain words, as CLAUDE.md says;
     every public item documented.
   - **Size.** Commits of one step each, that build and pass, with each later fix folded into
     the commit it belongs to.
   - **Goals,** CLAUDE.md's eight. Zero cost: a measured task's numbers are measured again by
     the reviewer, through `bench.py`, beside the library's in the same sitting. Minimum code:
     no line the task does not need, no function that only hands a value on, nothing left dead.
     The whole spec: no form refused that the target encodes, checked with its tools where in
     doubt. Safe where it can be. Clean architecture. An old form of an API kept beside the new
     one is a finding. The newest PTX: what the newest `ptxas` refuses is refused at every
     version and what it takes is taken, checked with it, and a module's `.version` is the
     oldest that has its instructions. Nothing deprecated: a form a target deprecates is refused.
     Fast and deterministic tests: no test sleeps, waits for time to pass, or holds how long a
     thing takes or how much CPU it uses; one that does is rewritten, not given more room.

   What only the hardware can settle, settle with a small program on this machine, not with a
   vendor's manual.
4. `review.py merge ID`: it rebases the branch onto the effort's if needed, runs the gate, and
   fast-forwards the effort's branch. A conflict or a failed gate is a request for changes, with
   the output.
5. `agent-kanban approve ID --agent reviewer --notes "merged as <sha>: <what was checked>"`,
   then `review.py clean ID`; or `agent-kanban request-changes ID --agent reviewer --notes
   "<each finding, where, what to write instead>"`. Either way, mail the developer the verdict,
   by `agent-kanban send`.

Approve only after the merge, so that the tasks it unblocks start from code that holds it.

Then merge what is ready into master, at once (the user's rule, 2026-11-24):
- In a worktree of your own, from master, merge each effort's branch that has work master
  lacks. Each is a merge commit of its own, whose message has a line for each task it brings,
  and which adds a short line for each change a user sees to CHANGELOG.md's Unreleased.
- Mend a conflict in the merge commit, and say how in its message.
- Run `./make.py check` on the top. If it fails, master waits, and what fails becomes a task.
- If it passes, fast-forward master to it by `git merge --ff-only` in the main checkout, when
  that is clean and on master; else leave master as it is, and say so. Then remove the worktree
  and its branch.

Pushing and publishing are the user's call.
