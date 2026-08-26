---
name: task
description: Task lifecycle — plan and execute. Use on "/task plan", "/task execute", "plan a task", "continue T-NNN" — and proactively propose "/task plan" when the researcher is designing substantial, plannable work without naming it.
---

# /task

Task-file format and rules live in AGENTS.md (Tasks); this skill is only the procedure.

## plan

1. Read `docs/framing.md`. Grep `docs/tasks/BACKLOG.md` and `docs/decisions/` for the task's area; surface matching entries to the researcher. If a backlog entry is graduating into this task, consume and delete it.
2. Interview relentlessly until shared understanding: walk down each branch of the design, resolving dependencies between decisions batch by batch; questions come in batches of 3, each with a recommended answer; if a question can be answered by exploring the codebase, explore instead of asking. Scale to the task — a small task needs one batch, not seven. If the interview resolves something direction-steering (hard to reverse ∧ surprising without context ∧ real trade-off), propose a decision record on the spot.
3. Allocate T-NNN (max across git history and `docs/tasks/`, +1). Write the four-field file, status `planned`. Plan lines name their product type and the exact files or directories they will create or edit — nothing compressed, one line per run.
4. Show the file to the researcher for approval.
5. Append the spine line (`- date · T-NNN · [task] planned <slug> · docs/tasks/T-NNN-slug.md`), commit as `T-NNN: planned <slug>` staging explicit paths. Plan edits and backlog entries are [task] lines too — `docs/tasks/` is the [task] product's home.

## execute

1. Read the task file; find the first unfinished plan line. Set status `active` if not already.
2. Execute to the next commit boundary: one plan line by default; consecutive lines validated together as one coherent outcome may close as one commit. Route products per the AGENTS.md table; for [experiment], [analysis], [reference] operations, use their skills.
3. Close the outcome: products in their homes, one spine line per product appended with `>>`, one commit with explicit paths, message `T-NNN: <summary>`.
4. Show the researcher the spine line(s) and a diff summary. Stop and wait. Continue only on their word; "run through" grants autonomy only for the span named.

## retroactive

Work happened without a task (free-form session that produced something durable): allocate T-NNN, append the spine line(s), commit — a single-commit outcome needs no file. If the work spanned more than one commit or was paused, write the file at close — goal as it emerged, plan as what actually happened, status `done` — confirming name and goal with the researcher.

## close

(Fileless single-commit tasks are born closed — skip this.) Flip status; append the closing spine line (`done:` or `killed:` with the outcome). Sweep the conversation once: unrecorded pivotal resolutions → propose decision records; "later" items → backlog; framing contradictions → propose amendment. Write at most 2 memory entries (default 0). Commit.
