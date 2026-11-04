# How the reviewer reviews a task

**The bar is 10/10** (the user's rule, 2026-10-27). Merge only code with nothing left to
change: every check below holds, and there is no name, doc line, test or line of code you would
still write differently. Anything less goes back with `request-changes` for polishing, with
every finding listed at once, each with its `file:line` and what to write instead, so that one
more round finishes it. A finding of taste is still a finding; say that it is one. A point
outside the task's lines that the task did not make worse is not a reason to hold it: put it
on the task that touches those lines next, or on a task of its own.

Developers are subagents on Sonnet 5.5, reused where they can be, and each bug in kernite is a
task of its own, as CLAUDE.md says. With each task, the reviewer gives its developer, new or
resumed, a brief that names the task, the effort's brief, the files, the commands and the evidence
to bring, as [DEVELOPER.md](DEVELOPER.md) has a developer work. Only one developer whose task measures runs at a time, as another's build or
benchmark spoils its numbers. The reviewer checks a bug against the code before it files it, and
a limit README states is not a bug; it does not fix a bug itself.

Run the commands from the main checkout, where `review.py` means
`uv run .kanban/notes/<effort>/review.py`, of the task's effort. For each task in review, as it
comes:

1. `agent-kanban claim ID --agent reviewer`.
2. `review.py show ID`: the history and evidence, the commits, the files, and the test code
   touched.
3. Read the task's tests, the effort's brief, and the whole diff, `review.py diff ID [PATH..]`.
   Refuse when any of these fails:
   - **Does what the task says, and only that.** No dead code left behind, no unrelated change,
     no TODO in place of the work.
   - **Layers.** Every new `use` and every dependency points inward, as
     `crates/kernite/tests/architecture.rs` says. A port is owned by its caller.
   - **Behaviour.** A listing changes only where the task changes the generated code, and every
     message kernite gives keeps its text and its `line:column` unless the task changes it.
   - **Tests.** New behaviour has a new test, a fake or the real thing, never a mock. No test is
     weakened. Every rule README states has a test, and a test that skips says so.
   - **Words.** Docs, messages and commit messages in README's plain words, as CLAUDE.md says;
     every public item documented.
   - **Size.** Commits of one step each, that build and pass, with each later fix folded into
     the commit it belongs to.
   - **Goals,** CLAUDE.md's five. Zero cost: a measured task's numbers are measured again by the
     reviewer, through `bench.py`, beside the library's in the same sitting. Minimum code: no
     line the task does not need, no function that only hands a value on, nothing left dead.
     The whole spec: no form refused that the target encodes, checked with its tools where in
     doubt. Safe where it can be. Clean architecture. An old form of an API kept beside the new
     one is a finding.

   What only the hardware can settle, settle with a small program on this machine, not with a
   vendor's manual.
4. `review.py merge ID`: it rebases the branch onto the effort's if needed, runs the gate, and
   fast-forwards the effort's branch. A conflict or a failed gate is a request for changes, with
   the output.
5. `agent-kanban approve ID --agent reviewer --notes "merged as <sha>: <what was checked>"`,
   then `review.py clean ID`; or `agent-kanban request-changes ID --agent reviewer --notes
   "<each finding, where, what to write instead>"`. Either way, message the developer the
   verdict.

Approve only after the merge, so that the tasks it unblocks start from code that holds it.
Merging an effort's branch into master is the user's call.
