# How the reviewer reviews a task

**The bar is 10/10** (the user's rule, 2026-10-27). Merge only code with nothing left to
change: every check below holds, and there is no name, doc line, test or line of code you would
still write differently. Anything less goes back with `request-changes` for polishing, with
every finding listed at once, each with its `file:line` and what to write instead, so that one
more round finishes it. A finding of taste is still a finding; say that it is one. A point
outside the task's lines that the task did not make worse is not a reason to hold it: put it
on the task that touches those lines next (design.md §5), or on a task of its own.

This guide is for the clean-architecture rewrite. Run its commands from the main checkout. Its
notes are in `.kanban/notes/`: below, `design.md` means `.kanban/notes/design/design.md`, "the
map" means `.kanban/notes/architecture-map.md`, and `review.py` means
`uv run .kanban/notes/review.py`. How a developer does a task is [DEVELOPER.md](DEVELOPER.md).

For each task in review, in the order they arrive:

1. `agent-kanban claim ID --agent reviewer`
2. A design task (Dn): read the file it wrote, and check a sample of its claims against the code
   in `.claude/worktrees/clean-arch`. Refuse a file whose findings are not checked or not true,
   or that leaves part of its lens unread. Then weigh each finding, as design.md §6 says:
   - answer it in `.kanban/notes/design/revision-notes.md`: taken, and how, or refused, and
     why;
   - a finding taken: change design.md, and `agent-kanban edit` every unclaimed task it
     touches, before approving, since approving frees the tasks that wait (D3 frees T07);
   - a finding on a task already claimed: keep it for that task's review;
   - record board changes in `.kanban/notes/plan.md`.

   Then approve, and go to 6.
3. An implementation task (Tnn): `review.py show ID` shows the history and evidence, the
   commits, the files, and the test code touched.
4. Read design.md's row and changes for the task, the text it points to, and the whole diff
   (`review.py diff ID [PATH..]`). Refuse when any of these fails:
   - **Does what the text says, and only that.** Nothing half-moved but design.md §4's staged
     copies, no dead code left behind, no unrelated change, no TODO in place of the work.
   - **Layers.** Every new `use` and every dependency points inward, as RULE says. No adapter
     type reaches the macro. A port is owned by its caller. No use-case rule is copied into an
     adapter.
   - **Behaviour.** Anything that could change an order (of VRegs, labels, slots, names,
     declarations, `Target` calls, the messages that win) is justified by the text or the code.
     The map's §9 lists these risks.
   - **Tests.** Test code changes only to follow an API or to move. New behaviour has a new test
     (a fake or the real thing, never a mock). No test weakened: no assertion dropped or
     loosened.
   - **Moves.** A mechanical task's diff is moves, plus `use`, `mod`, visibility and paths:
     `git diff --color-moved=dimmed-zebra clean-architecture...HEAD` in its worktree.
   - **Words.** Docs and messages in README's plain words; every public item documented; commit
     messages as CLAUDE.md says, with the Co-Authored-By line.
   - **Size.** Reviewable: a big diff is a mechanical move and says so, or is about the size its
     text gives, in commits of one step each that build and pass.
   - **Goals** (CLAUDE.md's five, design.md §4 rule 8). Clean architecture: the Layers check
     above, and the task leaves the code cleaner than it found it. Zero cost: the gate holds
     the generated code.
     Minimum code: no line the task does not need, no type or function that only hands a value
     on, nothing left dead; the gate's "code outside tests" line shows what the task added, and
     growth its text does not call for is refused. The whole spec: kernite refuses no form of a
     target's spec that it took before, and refuses what it took against its own rules only as
     decision 3 fixes a defect; every new rule has a test. Safety: nothing to build in this
     rewrite, but no change may make a safe call harder to add later.

   For a big or risky task, also have two or three agents read the diff adversarially (one lens
   each: behaviour, layering, tests) and weigh what they find. They advise; the verdict is the
   reviewer's.
5. `review.py merge ID`: it rebases onto `clean-architecture` if needed, runs the gate (check 8
   holds the macro's Rust), and fast-forwards. A conflict or a failed gate is a request for
   changes, with the output.
6. `agent-kanban approve ID --agent reviewer --notes "merged as <sha>: <what was checked>"`, or
   `agent-kanban request-changes ID --agent reviewer --notes "<each problem, where, what to do>"`.
7. After approving an implementation task: `review.py clean ID`.

Approve an implementation task only after its merge, so the tasks it unblocks start from code
that holds it.
