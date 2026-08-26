---
name: analysis
description: Analysis of our own work — runs, datasets, system behavior — at any scope. Use on "/analysis <run-number>", "/analysis <question>", "analyze run NNN", or proactively propose it when the researcher asks an analysis question in prose.
---

# /analysis

Every analysis lives in `analyses/<slug>/`. Run-bound analyses reuse the run's number and slug (`analyses/183_ablate_judge/`); cross-run and no-run analyses use a descriptive slug. Analysis of literature is not this — it belongs under `refs/` (the /ref skill).

## entry

- **Run-bound** (`/analysis 183`): read that run's `*_plan.md` and follow its pre-registered analysis plan.
- **Anything else**: ALWAYS interview first — derive or brainstorm the analysis plan (batches of 3 questions, each with a recommended answer) and get the researcher's explicit confirmation before writing any code.

## procedure

1. Create `analyses/<slug>/`. First line of the report names `sources:` — the runs, datasets, or files it reads (omit only for pure reasoning).
2. The report is an executed notebook `<slug>_report.ipynb` whenever anything is computed; a markdown file when pure prose. Notebook shape: markdown cells sources / question / method / findings / verdict, code cells between; the verdict is stated first, answer before evidence.
3. Numbers follow the AGENTS.md no-pasted-numbers rule. If the underlying data was never persisted, stop and tell the researcher — never reconstruct numbers from memory or conversation.
4. Recurring queries import from `{{PACKAGE}}/analysis/`. Ad-hoc code lives beside the report and may be one-time; the moment code is needed a second time, move it to `{{PACKAGE}}/analysis/` in the same commit.
5. Execute headlessly (`jupyter nbconvert --to notebook --execute --inplace <file>`), commit the notebook WITH outputs. The researcher reads rendered results and never runs cells.
6. Report the verdict in chat. Spine line + commit.
7. Route what the verdict spawns: a direction-steering resolution → propose a decision record; future work → backlog line; a framing contradiction → propose an amendment.
