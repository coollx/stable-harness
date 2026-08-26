---
name: experiment
description: Run lifecycle — plan and execute. Use on "/experiment plan", "/experiment execute", "plan a run", "launch the experiment" — and proactively propose "/experiment plan" when the researcher is sketching an experiment without naming it.
---

# /experiment

Run-folder rules live in AGENTS.md (Experiments); this skill is only the procedure.

## plan

1. Interview if the intent is not settled — relentlessly, in batches of 3 questions each with a recommended answer, resolving dependent choices batch by batch; explore the codebase instead of asking when the code holds the answer. This mode produces configuration and the plan file, never experiment code — pipeline code is a [code] operation in the package.
2. Family: use the existing one, or create it per AGENTS.md (one-paragraph README, launch.sh, defaults.yaml).
3. Allocate the global run number NNN (max across git history and all `experiments/*/runs/`, +1).
4. Write `run.yaml` (overrides only). Show the researcher the key-diff against defaults and against one named sibling run; every differing key must be intended.
5. Write `NNN_slug_plan.md` per the AGENTS.md structure (context with task id · hypothesis · setup · analysis plan), ending with the exact launch command.
6. Spine line + commit.

## execute

1. If the run is expected over ~1 hour, run the smoke pass first: the full workflow at <3 instances into `outputs/smoke/` — no run number, no ceremony. A failed smoke means fix before launch.
2. Hand the researcher the launch command; never auto-launch long jobs. Short runs may be run directly.
3. Monitor `outputs/run.log` with failure-signature searches — silence is not success. Recover transient failures (standing authorization once the researcher launched); record each incident as a dated paragraph in the run's log. Long pipelines resume via their volume argument — never a pilot copy.
4. On completion, confirm `summary.json` landed and commit it with the run's spine line. A discarded run commits nothing.
