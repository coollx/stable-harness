# AGENTS.md — the rulebook

Project: {{PROJECT}} · Package: `{{PACKAGE}}/`
Research identity, claims, and vocabulary: `docs/framing.md`. This file is how we work; that file is what we work on.

## The objects

- **Product**: a durable artifact of work. Ten types, each with exactly one destination (table below). Work is classified by what it produces, never by what activity it was.
- **Operation**: one unit of work yielding one product, written `[type]`.
- **Task**: one or more operations toward one goal, identified by a number (T-NNN). Every durable product and every commit traces to a task. A task gets a one-page file only when a mechanical trigger fires (see Tasks); a single-commit task's entire record is its spine line. Tasks may be created before work (planned) or after it (retroactive) — paperwork follows work, never gates it.
- **Spine**: the append-only chronological log, one line per finished operation, one file per month: `docs/log/YYYY-MM.md`. The single source of history.

## Routing (closed table — nothing else may be created; changing a row is /meta)

| Product | Destination |
|---|---|
| [experiment] | `experiments/<family>/` — its runs, launcher, defaults.yaml, README |
| [analysis] | `analyses/<slug>/` |
| [code] | `{{PACKAGE}}/` (incl. configs/, tests/, `{{PACKAGE}}/analysis/`) |
| [asset] | its gitignored location (data/, checkpoints/, envs), announced by an [asset] spine line |
| [reference] | `refs/related-work/<Name>/`; cross-paper work in `refs/lit-review/` |
| [decision] | `docs/decisions/D-NNN-slug.md` |
| [task] | `docs/tasks/` — task files and BACKLOG.md entries |
| [framing] | `docs/framing.md` (rare) |
| [paper] | `paper/` — see Paper below; `paper/overleaf/` is an independent shared git clone, gitignored here and outside the commit gate |
| [meta] | the harness itself: this file, `.claude/`, hooks, skills |

When routing is ambiguous: ask the researcher. Never guess.

## Spine and commits

Line format (exactly this shape; the commit gate checks it):

    - 2026-08-06 · T-042 · [analysis] judge term carries +4pp, cost term inert · analyses/reward-ablation/

`- ` bullet (so the month file renders as a markdown list), then date · task id · [type] plain-language summary of at most 150 characters · the product's path (one path, never a file enumeration — the commit carries the file list). Append with shell `>>`, never editor rewrite. Lines are never edited; a correction is a new line repeating the old line's path and key terms; a log rewrite happens only at the researcher's explicit direction, never on the agent's initiative.

- Commits exist only at outcome boundaries. Default: one operation, one commit. Operations from the same task that were validated together as one coherent outcome may share one commit, carrying one spine line per durable product — merge only when they'd be reviewed as one diff; when in doubt, split. Message = `T-NNN: summary`. Bookkeeping (status flips, memory entries, incident notes) rides inside its outcome's commit; work-in-progress stays uncommitted until the outcome completes.
- Stage explicit paths only — never `git add -A` (parallel sessions share the tree).
- Never include AI co-author lines in commit messages.
- New numbers (T-NNN, D-NNN, run NNN) = max of git history and working tree, +1; never reused.

## Tasks

A task file exists exactly when one of three conditions fires — the work was planned before execution, the task reaches its second commit, or the task is paused. None fired → no file: the spine line is the task's entire record (a file that would only photocopy its spine line is illegal). The file appears at whichever trigger fires first, same number, nothing renamed.

Four fields, nothing else legal in the file (that is what keeps it one page):

    # T-042 reward-term ablation
    status: planned            # planned | active | done | killed
    context: why this task exists; what spawned it (≤3 sentences)
    goal: what done looks like (≤3 sentences)
    plan:                      # linear, fully unfolded, one atomic operation per line
      1. [code]       reward-variant switch — configs/rl/, {{PACKAGE}}/rl/reward.py
      2. [experiment] run 183: judge term removed
      ...

Plans are edited freely as reality diverges (each edit = one spine line saying why). Nothing is compressed ("runs 183-185" is illegal — one line per run). Multiple tasks may be active at once. Closing = status flip + `done:`/`killed:` spine line; fileless single-commit tasks are born closed, nothing to flip.

**BACKLOG.md**: deferred ideas, one line to one paragraph, future tense, optional trigger clause, provenance date. Outgrows a paragraph → becomes a planned task file. Consumed and deleted by `/task plan`; pruned opportunistically; deletion is expected. Backlog defers work; a decision closes a debate — that is the boundary.

## Experiments

- The run folder holds what exists BEFORE launch plus machine-written facts (`run.yaml`, `NNN_slug_plan.md`, `summary.json`); every generated byte goes to `outputs/` (gitignored, including redirected tool output dirs). Everything interpretive lives in `analyses/` — never in the run folder.
- One launcher per family: `experiments/<family>/launch.sh runs/NNN_slug` reads `defaults.yaml` overlaid with the run's `run.yaml`, tees everything to `outputs/run.log`, and writes `summary.json` (extraction, zero judgment) as its final step. Per-run scripts only when a run genuinely deviates. (The predecessor accumulated 183 near-identical per-run launch scripts.)
- `run.yaml` holds ONLY the keys that differ from defaults.yaml — no prose blocks, no mandatory metadata. Baselines are named in words in the analysis plan, not as a config key. Key-diff against defaults and a sibling before launch.
- `NNN_slug_plan.md` = context · hypothesis · setup (the delta, in words) · analysis plan (concrete: metrics, comparisons, what result means what) — and ends with the exact launch command. Frozen at launch; results never enter it. What freezing truly pins is the analysis rules before results exist; the hypothesis field is a note, not proof of foresight.
- No pilot runs: pipelines are resumable via one meaningfully-named volume argument (`--train-steps`, `--n-problems`; never generic `--size`) — omitted = everything; bounded = that much, next invocation continues.
- Smoke rule: any run expected over ~1 hour first runs the full workflow at <3 instances into `outputs/smoke/` — no run number, no ceremony, deletable.
- Long launches are handed to the researcher, never auto-executed; afterward, transient-failure recovery is the agent's job. A sweep launched as one unit is one run number. A run that fails mid-way is fixed and relaunched under its own number — no ceremony beyond a dated incident note.

## Analyses

- `analyses/<slug>/` is the only home for analysis of our own work, any scope. Run-bound analyses reuse the run's number+slug. `sources:` line names what it reads, when it reads anything.
- The report is an executed notebook (`<slug>_report.ipynb`) whenever the analysis computes; markdown when pure prose. The agent executes it headlessly and commits it WITH outputs; the researcher never runs cells. Verdict first.
- **No pasted numbers**: every number in a report is produced by a cell in the same notebook reading named persisted sources — tracked files (`summary.json`, `run.yaml`) preferred; intentionally-untracked paths (`outputs/` traces, local datasets) are legal when the sources line names them. The verdict cell may restate numbers computed below it. (This is what makes reports auditable — the predecessor's reports were unownable because number provenance was lost.)
- Ad-hoc code lives beside its report and may be one-time; code needed twice moves to `{{PACKAGE}}/analysis/` in that same commit. Code that runs DURING an experiment (scorers, judges) is pipeline [code], never analysis.

## Decisions

Every choice already lands in history through its operation's spine line — that is the default record. A decision file (`D-NNN`) is the promotion reserved for resolutions that steer the project's direction: a task-domain pivot, a method adopted or abandoned. Promotion filter: hard to reverse ∧ surprising without context ∧ a real trade-off existed. Technical details stay at spine level because they are recoverable from code and config diffs, and because the store's value is that everything in it matters — a decisions folder holding every choice is a folder nobody consults. Typical birth: a dedicated discussion session; the agent proposes recording at its end. Format: context / decision / rationale, one paragraph default. Body immutable; the only legal append is a dated one-line `## Amendments` entry (partial overrides: "2026-09-02 · overridden in T-051: ..."). Before work in a decided area: `grep -il <topic> docs/decisions/` and flag conflicts in one sentence, then proceed on the researcher's word.

## Framing and language

- `docs/framing.md` Terms section is the vocabulary allowlist: agents never coin terms or abbreviations; additions only with researcher approval. First use in any document: full term, identifier in parentheses.
- Plain language everywhere: define concepts at first use; no defensive disclaimers or self-correction narration; labels are A/B/C or numbers, never Greek letters; version names purely numeric.
- Discussion replies: answer the researcher point by point — for each point, first reword it as understood, then a one-sentence answer, then the development; never refer to a point by its number without restating it (the researcher brainstorms freely and will not re-read what they wrote).
- Markdown prose is written without hard line breaks: one line per paragraph or bullet, however long; let the editor wrap.
- Framing amendments: additive ones batch (spine line now, file edit after they survive); retractions edit the file immediately, same commit as the finding.

## Paper

- Before responding to any paper-related request — discussion, planning, drafting, figures, reviews, or any edit under `paper/` — read `paper/AGENTS.md`; where the two files conflict, it wins for paper work.
- The press-conference-principle skill governs every paper-facing selection decision — what to include, emphasize, or omit, including which experiments become paper claims — from ideation onward, not only while drafting.
- Paper source commits happen inside `paper/overleaf/` via `/paper:push` (no AI-attribution trailers, confirmation-gated); outer spine lines record `[paper]` outcomes, one per finished operation.

## Memory (`docs/memory/`, git-tracked; harness path symlinks here)

Two categories only: (a) researcher working-style feedback; (b) environment and tooling facts, including hard-won traps. Project knowledge — results, status, decisions, ideas — is banned (pointers allowed, content not; the predecessor's memory silently became an invisible shadow log). Growth: ≤2 new entries per task, default 0, at task close. `MEMORY.md` is generated from the files' `description:` lines — agents never hand-edit it (the researcher may edit anything; rules bind agents, not the researcher).

## Code standards

The bar: keep only code whose removal breaks the program — if 100 lines works, never 101. The words sha, hash, gate, stamp, marker, tag, or a tuning constant in a design are red flags for over-engineering.

- Flat modules; config-first — a CLI argument exists only if it varies between runs of one script; machine facts live in `configs/env/<machine>.yaml`, never in code; comments only for constraints the code cannot show.
- No abstraction before a second call site. No backward compatibility — remove obsolete paths instead of adding layers, fallbacks, or migrations.
- Simplest implementation that fully meets the current requirement; no speculative abstractions, configuration, or indirection. Grow in layers: the smallest end-to-end working version first; never trade a working artifact for unfinished complexity. Modular components, separated concerns.
- Prefer established libraries over reimplementation; lean on dependencies already in the project; never assume a library lacks a capability without checking its documentation and source.
- Make architectural decisions for the long term; never accept a stopgap that only works for now and is meant to be replaced later — planned rewrites don't happen, and the hack calcifies.
- Study how the projects under `refs/` solve a problem before designing one.
- No defensive coding: no handling, fallbacks, or validation for scenarios that cannot happen; trust internal code and framework guarantees.
- Any prompt written for a language model is code: minimal words, positive phrasing, concrete numbers; wording stays stable across a comparison sequence.
- Tests: few, end-to-end, real logic on real data samples; nothing a smoke pass covers; no dummy values, no per-function unit tests.
- Review: substantial changes get 2 Claude Opus + 2 codex reviewers in parallel; findings are TRIAGED, never obeyed — additions rejected unless they name a concrete failure in our usage, deletions prioritized, bug claims verified. Then a simplification pass on the outcome's diff (the /simplify skill where installed) — once, before its commit; never re-simplify already-committed code unless a task explicitly targets it.

## Agent duties

- **Human-on-the-loop**: execute to the next commit boundary (one plan line by default; tightly-coupled lines validated together may close as one), close it (products → homes, spine lines, commit), show the spine line(s) and a diff summary, stop. "Run through" only when explicitly granted, only for the granted span.
- **Steering**: when the researcher's activity matches a paved verb they didn't name — designing a plannable task (/task plan), sketching an experiment (/experiment plan), asking an analysis question (/analysis), discussing a paper (/ref add) — propose the command. Propose; never assume.
- **Documentation triggers** — propose, don't wait to be told: a discussion session just resolved something pivotal → decision record; "later / at some point" → backlog; a claim contradicting or extending framing.md → amendment; at task close, sweep the conversation once for all three.
- **Grounding**: claims about data or literature rest on files and sources that exist — no invented numbers, no unverified paper characterizations.
- **Verification**: never report an operation done without demonstrating it (run it, show the output). Infra serves research — never cite an implementation limit as a reason to defer a research idea.
- **Subagents**: every dispatched subagent run on Opus.
- Free-form and throwaway work (scratch/, discussion, probes) carries zero recording obligations until something is committed.

## /meta

All changes to the harness — this file, hooks, skills, the routing table — go through `/meta`, invoked ONLY by the researcher (the agent may recommend it, never run it). When the agent notices harness friction during work — a rule fighting the situation, a scenario the design doesn't cover — it says so in the moment and proposes the fix in one or two sentences; the researcher either invokes `/meta` or the observation becomes a `harness:` backlog line for a later `/meta` session. The agent never applies a harness change itself. Every /meta commit carries a `[meta]` spine line, so `grep '\[meta\]' docs/log/` is the harness's own history.
