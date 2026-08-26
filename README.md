# research-harness

A minimal workflow template for a research lab of one human researcher plus AI coding agents. It governs how work is recorded and where every artifact lives, so that months later any result, decision, or rationale is one grep away — while adding near-zero ceremony to daily work.

## Quick start

1. Create a new repository from this template (GitHub: "Use this template" — gives fresh history; do not plain-clone).
2. `./init.sh <package-name> [project-name]`
3. Open Claude Code in the repo and say what you want. The first real session's job is filling `docs/framing.md`.

## What you get

- **AGENTS.md** — the single rulebook agents load every session.
- **The spine** (`docs/log/YYYY-MM.md`) — append-only chronological log, one line per finished operation; a commit gate makes unlogged work impossible.
- **Tasks** (`docs/tasks/`) — one-page files: status, context, goal, unfolded plan. Everything durable traces to one.
- **Ten product types**, each with exactly one destination: experiment, analysis, code, asset, reference, decision, task (task files + backlog), framing, paper, meta.
- **Three hooks** — session-start context injection, the commit gate, a turn-end stray-file check. All git-tracked, self-healing.
- **Five commands** — /task, /experiment, /analysis, /ref, /meta — plan/execute verbs with the human on the loop; everything else is plain prose.

## Philosophy

Ceremony binds to durability: throwaway work records nothing. Classify by product, not activity. Append-only history; repair artifacts instead of marking them. Rules bind the agents, never the researcher. Minimal always — if 100 lines works, never 101. The workflow evolves only through `/meta`, invoked by the researcher; improvements worth keeping are ported back to this template.
