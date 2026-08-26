---
name: meta
description: Revise the harness itself — AGENTS.md, hooks, skills, the routing table. Use ONLY when the researcher explicitly invokes "/meta". The agent may recommend invoking it, never run it on its own initiative.
---

# /meta

The only path by which the workflow changes. HARD RULE: if this skill was triggered by anything other than the researcher explicitly invoking it, stop and say so.

1. Collect the candidates: `harness:` lines in `docs/tasks/BACKLOG.md`, plus friction observed in the current conversation. Present the list; the researcher picks what to address.
2. Diagnose each item in plain language: what rule or gap causes the friction, and what failure the current rule was protecting against (read its rationale in AGENTS.md before proposing to touch it).
3. Propose the minimal revision — prefer deleting or simplifying a rule over adding one; check the proposal against the red-flag words (sha, hash, gate, stamp, marker, tag, tuning constant).
4. Assess risk. For substantial changes — a routing-table row, a hook behavior, a new rule — dispatch adversarial reviewer agents to attack the proposal before applying it.
5. Apply only on the researcher's explicit approval: edit AGENTS.md / hooks / skills, append the `[meta]` spine line, commit as `T-NNN: <change>`. Delete the consumed `harness:` backlog lines in the same commit.
6. Every change is applied in both places: this project, and the research-harness template repository so the next project inherits it. The template edit always waits for the researcher's explicit approval — show the proposed template edit, ask, apply only on yes.
