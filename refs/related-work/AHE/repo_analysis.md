# Repo Analysis: agentic-harness-engineering

**Path:** `refs/related-work/AHE/repo/` · **Source:** https://github.com/china-qijizhifeng/agentic-harness-engineering · **Analyzed:** 2026-09-14 (commit `8b2a55d`, "fix: broken config references, batch-mode path, and crash guards (#11)", 2026-08-02)

## Overview

- **Paper:** Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses (see `paper_analysis.md`)
- **Framework:** custom outer loop in one file (`evolve.py`, 4719 lines) driving three external pieces — NexAU v0.3.9 (the harness runtime), harbor-LJH (benchmark dispatcher), and `agent_debugger_core` (trace analysis, vendored at `agents/evolve_agent/skills/agent-debugger-cli/_source`). Python >= 3.13, uv, MIT.
- **RL Algorithm:** none. No gradients anywhere in the repository. The optimiser is a prompted language model editing files; "learning" is the accumulated diff on a git workspace.
- **Base Model:** `gpt-5.4` for all four roles (Code Agent, Agent Debugger, Evolve Agent, Explore Agent), configured through `configs/base.yaml` and `.env`; the code is model-agnostic through an OpenAI-compatible endpoint.
- **Key Innovation:** a harness laid out as seven orthogonal component directories in a git workspace, edited by an agent that must declare, per edit, which tasks it expects to fix and which it might break — and whose declaration is checked by set intersection against the next round's actual pass/fail deltas.

Caveat on completeness: the README states the Agent Debugger is only *partially* open-sourced "due to company strategy". The reference run's traces, per-iteration change manifests, and change-evaluation JSONs are not released — `experiments/` contains only the frozen evolved harness, not the campaign that produced it.

The repository is written for a Chinese-speaking team: `configs/base.yaml`, `scripts/*.sh`, and most `evolve.py` docstrings are in Chinese; the agent-facing prompts (`agents/evolve_agent/evolve_prompt.md`, the skills) are in English.

## File Structure Map

```
evolve.py                          4719 lines — the entire outer loop: phases 0 through 3.5, resume, best-of-N
trace_converter.py                  936 lines — raw harbor traces -> the file-per-message layout the debugger navigates
configs/
  base.yaml                         shared defaults: dataset, k, concurrency, explore/debugger settings
  experiments/exp-simple-code-gpt54.yaml    the paper's reference run (plus -wo-adb and -new-adb ablation configs, and a dry run)
agents/
  code_agent_simple/                THE SEED HARNESS (NexAU-0): 9-line systemprompt.md, one shell tool, empty memory
  evolve_agent/                     the optimiser: evolve_prompt.md, 20 tool descriptions, 7 middleware, 2 skills
  explore_agent/                    one-shot knowledge gatherer run in parallel with iteration 1
experiments/evolved_harness/        THE RELEASED ARTIFACT: the harness the 10-iteration campaign produced
skills/agentic-harness-engineering/ a portable Claude-Code skill packaging the AHE pattern (HARNESS.md v1.0 spec)
scripts/
  evolve.sh / evolve-resume.sh      tmux launchers (single run or --batch over all configs)
  build_templates.py                pre-builds one sandbox image per benchmark task
```

## Component Inventory

The 10-component protocol is aimed at RL codebases; five of its slots are simply empty here, which is itself the most informative fact about the repository. Present: entry points, environment/tool interaction, "reward" (read as: the scoring and attribution machinery), the outer loop, data pipeline, infrastructure, evaluation. Absent: special-token handling, token masking, SFT/cold-start.

### 1. Entry points

| Path | Role |
|---|---|
| `evolve.py:4148` `main()` | the loop; args `--config`, `--experiment`, `--start-iteration`, `--skip-eval` |
| `scripts/evolve.sh` | wraps `uv run python evolve.py --config <cfg>` in a detached tmux session; `--batch` launches every config in `configs/experiments/` |
| `scripts/build_templates.py` | builds the E2B sandbox template per task before any run |
| `agents/code_agent_simple/start.py` | runs the harness itself, standalone, outside the loop |

The loop is resumable at any iteration: `--start-iteration N` restores the workspace from iteration N's snapshot and truncates the score history (`rollback_experiment_metadata`, `evolve.py:2457`).

### 2. Environment / tool interaction

Rollouts never run locally. `_build_harbor_cmd` (`evolve.py:396`) shells out to the harbor dispatcher with `--agent nexau --env e2b -k 2 -c 64`; every trial gets a fresh E2B sandbox with a 3600-second lifetime, so shell side effects cannot leak between tasks. The loop then polls the job directory (`wait_for_job`, `evolve.py:355`) and parses results off disk. The Evolve Agent reaches the workspace through ordinary file tools (`read_file`, `write_file`, `replace`, `apply_patch`, `run_shell_command`) — 20 tool descriptions under `agents/evolve_agent/tool_descriptions/`, including `WebSearch` and `WebFetch`.

### 3. The seven editable layers (the substrate)

The seed, `agents/code_agent_simple/`, is the minimal starting point the paper describes:

```
code_agent.yaml              one tool, no middlewares:, no skills, no sub_agents
systemprompt.md              9 lines
tool_descriptions/run_shell_command.tool.yaml
tools/shell_tools/run_shell_command.py      263 lines
LongTermMEMORY.md            header only, "## Agent Added Memories" empty
ShortTermMEMORY.md           session-scoped, off-limits to the Evolve Agent
```

Registration is explicit, not by discovery: a new tool needs a `tools:` entry with `name`, `yaml_path`, and `binding`; middleware needs a `middlewares: - import: module:ClassName` line. Creating the file alone does nothing, which is why the evolve skill ships `scripts/validate_harness.py`.

`experiments/evolved_harness/` is what ten rounds produced. The diff against the seed is the single most useful artifact in the repository:

| Layer | Seed | Evolved |
|---|---|---|
| System prompt | 9 lines | 75 lines |
| Tool implementation | 263-line shell wrapper | 687-line shell wrapper |
| Middleware | none | `middleware.execution_risk_hints:ExecutionRiskHintsMiddleware`, 525 lines |
| Long-term memory | empty | 5 lessons |
| Skills | none | **none** |
| Sub-agents | none | **none** |
| Tool descriptions | 1 | 1 (rewritten) |

Two observations. **The loop used four of its seven layers.** No skill and no sub-agent survived to the released harness, despite the explore agent seeding skills at iteration 1 and the evolve prompt advertising both. **The released artifact is smaller than the paper's description** — the paper calls the evolved tool "a 1364-line shell" and the system prompt "79 lines of universal discipline", against 687 and 75 lines on disk. Whether the release is a cleaned variant or a different iteration is not stated anywhere in the repository.

The five surviving memory entries are worth reading in full (`experiments/evolved_harness/LongTermMEMORY.md`) because they are what ten rounds of greedy patching distilled into the deepest cheap layer. All five are variations on one theme — stop touching a passing state:

> "Once an evaluator-style end-to-end check passes, freeze the published state. Cleanup should be narrowly bounded; do not rerun live generators, reset webroots/repos, or rewrite git history after success unless new failing evidence demands it."

### 4. Scoring, attribution, and the verdict machinery (the "reward")

There is no reward function; there is a pass rate and a set-intersection audit.

`compute_pass_at_k_metrics` (`evolve.py:568`) and `compute_stats` (`evolve.py:599`) turn the harbor job directory into per-task rollout results; infrastructure exceptions are typed by `_extract_exception_type` and counted as failures.

`compute_iteration_diff` (`evolve.py:830`) produces the round-over-round transition matrix: `flipped`, `regressed`, `net`, `stable_pass`, `stable_fail`, `infra_recovered`, `infra_lost`, `exception_to_fail`, `fail_to_exception`, `exception_stable`, plus `rollout_improved`/`regressed`/`unchanged` when $k > 1$.

`compute_task_stability` (`evolve.py:783`, `min_iterations=3`) classifies each task across history as `stable_pass`, `stable_fail`, `unstable`, `possibly_unstable`, or `infra_only` — the loop's only defence against chasing noise.

`evaluate_changes` (`evolve.py:2239`) is the attribution. Verbatim:

```python
actually_fixed = [t for t in predicted if t in flipped_set]
still_failed   = [t for t in predicted if t not in flipped_set]
risk_realized  = [t for t in risks if t in regressed_set]
...
if n_risk_hit > 0 and n_fixed == 0:                 verdict = "HARMFUL"
elif n_risk_hit > 0 and n_fixed > 0:                verdict = "MIXED"
elif n_fixed == n_predicted and n_predicted > 0:    verdict = "EFFECTIVE"
elif n_fixed > 0:                                   verdict = "PARTIALLY_EFFECTIVE"
else:                                               verdict = "INEFFECTIVE"
...
attributed_tasks = all_predicted | all_risk
unattributed = [t for t in regressed_set if t not in attributed_tasks]
```

Its own comment concedes the limit: "whether regressions have tasks related to this change's files but not in predicted/risk — this requires knowing which file affects which task; we cannot determine this precisely, so we only use declared risk_tasks". Every regression the agent did not name lands in `unattributed_regressions` with no owner. In the paper's Appendix D this bucket holds 40 of the 45 observed regressions.

### 5. Retention is computed, displayed, and ignored

`build_evolution_query` (`evolve.py:2563`) assembles the markdown the Evolve Agent reads. Section 7, "Task Stability Analysis", contains:

```python
retention_rate = n_stable_pass / n_prev_pass_tested
lines.append(f"- Regressions: **{n_regressed}** (passed last iteration -> failed this iteration)")
lines.append(f"- Retention rate: **{retention_rate:.1%}** ({n_stable_pass}/{n_prev_pass_tested} previously passed and still passing)")
```

This is the anchor-set measurement of `docs/framing.md`, already implemented — and it terminates in a string. Nothing in the repository branches on `retention_rate`; no threshold, no gate, no rejection. It is one bullet in a prompt.

### 6. Rollback is a prompt instruction, not a gate

The paper's abstract says ineffective edits "are reverted at file granularity". In the code, the call site reads:

```python
# Phase 2.7: Change attribution evaluation (report only, rollback decided by evolve agent)
```

(`evolve.py:4377`). The evaluation is rendered into query section 8 under: "You must use this report to decide whether to rollback previous changes. HARMFUL and INEFFECTIVE changes should be prioritized for rollback." The deterministic `perform_auto_rollback(exp_dir, workspace_dir, best_iteration)` (`evolve.py:2418`) restores the *entire* workspace from the best iteration's snapshot and is invoked at exactly one site — `evolve.py:4169`, inside the resume path. It is never a per-round gate. **The only thing standing between a harmful edit and the next round is the same model that made it.**

`update_best_ever` (`evolve.py:2361`) writes `best_ever.json` by argmax over pass rate. Iteration $t+1$ still evolves from iteration $t$'s workspace regardless of score, so the best-ever record is bookkeeping surfaced in the prompt ("Best version snapshot: `runs/iteration_NNN/input/workspace/`", `evolve.py:2811`) — the agent may copy from it, and nothing makes it.

### 7. Best-of-N variant exploration (off by default)

`run_best_of_n_evolution` (`evolve.py:3690`) forks $N$ git worktrees, runs one Evolve Agent per worktree under a complementary strategy hint, evaluates all variants on the full benchmark, and adopts the winner. The two shipped hints in `configs/base.yaml` split exactly along layer depth: variant 1 "MUST focus on STRUCTURAL changes: middleware, tool implementations, or sub-agents", variant 2 "MUST focus on GUIDANCE changes: system prompt rules, skill packages, tool descriptions, or LongTermMEMORY". `select_variant_winner` (`evolve.py:3313`) is pure argmax on pass rate with tie-breaks on fewer exceptions then lower index, and it will adopt a winner that is worse than the pre-edit baseline after printing a warning. `run_multi_variant_adb` (`evolve.py:3418`) then feeds the cross-variant comparison into the next round, so losing variants still inform the next proposal.

### 8. Data pipeline / evidence corpus

`trace_converter.py` (936 lines) explodes each rollout into one file per message. `_build_adb_jobs` / `run_parallel_adb_ask` (`evolve.py:1216`, `1722`) fan out up to 16 concurrent `adb ask` processes, one per task (all its rollouts together), 600 s timeout, 3 retries with exponential backoff, capped at 90 tasks with debug tasks prioritised over summaries. `_write_debugger_analyse` (`evolve.py:1589`) assembles the per-task reports plus the aggregate `overview.md` the Evolve Agent opens first. `extract_verifier_failures` and `extract_agent_behavior_stats` (`evolve.py:967`, `1016`) pre-extract error types and behaviour counters so the agent does not re-read raw traces for basics.

### 9. Infrastructure

No GPUs, no distributed training. Concurrency is process-level: 64 concurrent sandboxes for rollouts, 16 concurrent debugger processes, $N$ concurrent evolve agents in best-of-N. Long runs live in tmux. Optional Feishu webhook notifications per iteration (`send_feishu_notification`, `evolve.py:61`).

### 10. Evaluation

Evolution runs on `terminal-bench@2.0` through harbor. `post_evolve` (`evolve.py:3999`, disabled in `base.yaml`) re-runs the frozen harness on `swebench-verified` — that is the transfer study, structurally outside the loop, exactly as the paper describes. `regenerate_scores_md` (`evolve.py:2059`) maintains a human-readable score table across iterations.

### Absent components

No special-token or action-token handling (the agent emits ordinary tool calls). No token masking, no loss, no gradients, no SFT or cold-start pipeline, no trajectory filtering for training. Nothing in the repository trains anything.

### Bonus: the portable spec

`skills/agentic-harness-engineering/` packages AHE as a Claude-Code skill independent of NexAU, built on a "HARNESS.md v1.0" specification whose seven components are named `system_rules`, `tool_descriptions`, `tool_implementations`, `middleware`, `skills`, `sub_agents`, `long_term_memory` — an enum enforced by `references/change-manifest-schema.json`. Three profiles map the spec onto other harnesses (`profiles/codex.yaml`, `hermes.yaml`, `openclaw.yaml`). Scripts: `init_harness.py`, `generate_manifest.py`, `verify_manifest.py`, `validate_harness.py`, `ci.sh`. Its `references/docs/rollback-strategy.md` states the deterministic rule the main loop does not implement — revert when all `expected_fixes` still fail, when any `at_risk_regressions` task regresses, when 2 or more unforeseen failures appear, or when any single P0 task regresses — and its closing best practice is the sharpest line in the repository: always pre-fill `at_risk_regressions`, because "an empty list means 'I am sure there is no risk' — which is almost never true."

## Relevance to this project

### Components we can directly reuse

- **The seven-layer workspace layout**, `agents/code_agent_simple/` plus `skills/agentic-harness-engineering/references/HARNESS.md`. Our layer list in `docs/framing.md` is this list plus model weights. This is a working action space for a harness agent, with registration rules and a validator, and the spec is deliberately harness-agnostic.
- **`compute_iteration_diff` and `compute_task_stability`** (`evolve.py:830`, `evolve.py:783`). The transition matrix and the stable/unstable classification are precisely the bookkeeping an anchor set needs, already handling the $k > 1$ and infrastructure-exception cases that make naive pass/fail diffs lie.
- **The retention computation** at `evolve.py:2628`. It already produces "previously passed and still passing" over the tested intersection. We need the number it computes; we need it wired to a gate instead of an f-string.
- **The change manifest schema** (`skills/agentic-harness-engineering/references/change-manifest-schema.json`) as the per-edit record format, and `evaluate_changes` (`evolve.py:2239`) as the audit that grades it.
- **`experiments/evolved_harness/`** as a concrete target artifact: what a greedily evolved harness actually looks like after ten rounds, at file granularity, with the seed beside it for diffing.

### Components we need to modify

- **The task supply.** `harbor.dataset` names one fixed benchmark and every round re-runs all of it. A stream means: draw the next batch, score it with the harness as it stood before the batch (which phase 1 already does), then re-check an anchor set grown from earlier batches. The loop's phase structure survives; the dataset handling does not.
- **The commit decision.** Today phase 2.7 renders a report and the Evolve Agent decides. We need a deterministic gate between proposal and commit — the logic is already written in prose in `rollback-strategy.md` and simply never implemented in `evolve.py`.
- **`update_best_ever`.** Argmax over a noisy re-evaluated benchmark is the reporting device that lets a peak-then-decline run publish its peak. For a stream we want the realised integral, not the maximum.
- **The proposer.** `run_evolve_agent` (`evolve.py:3008`) calls a prompted model. Our harness agent is trained, and it needs the critic's estimate as part of its objective; the call site is a clean seam but the objective is not there at all.
- **`select_variant_winner`** (`evolve.py:3313`). Best-of-N with argmax-on-current-batch is the greedy selection rule we argue against; the same fork-worktrees-and-evaluate scaffolding is exactly what group-sampled candidate edits need, with the selection rule replaced.

### Components that don't apply

- The Feishu notification path, the tmux launchers, and the batch mode — operational convenience for a Chinese-team overnight workflow.
- The E2B/harbor coupling: our environment choice is independent, and `_build_harbor_cmd` hardcodes their dispatcher's flags.
- The Agent Debugger internals are only partially released, so `agent_debugger_core` is a dependency to observe, not to build on.
- Everything the RL half of the protocol asks about — masking, advantage, rollout groups for training. There is none.

### Key code snippets worth studying

| Location | Why |
|---|---|
| `evolve.py:4276-4560` | the main loop body: phases 1, 2, 2.4, 2.7, 2.8, 2.5a, 3 in order — the whole architecture on two screens |
| `evolve.py:2239-2332` | `evaluate_changes`, the full attribution: intersection, five verdicts, unattributed bucket |
| `evolve.py:2563-2928` | `build_evolution_query`, every piece of evidence the optimiser sees, in the order it sees it |
| `evolve.py:2600-2660` | the retention-rate block — the measurement we need, with no consumer |
| `evolve.py:4377` and `evolve.py:2418` | the paper-versus-code gap on rollback, in two lines |
| `evolve.py:783-923` | `compute_task_stability` and `compute_iteration_diff`, reusable nearly as-is |
| `agents/evolve_agent/evolve_prompt.md` | the full optimiser contract: controllability rules, component-selection guidance, manifest schema, escalation anti-pattern |
| `experiments/evolved_harness/LongTermMEMORY.md` | five lines; what ten greedy rounds deposited in the deepest cheap layer |
| `skills/agentic-harness-engineering/references/docs/rollback-strategy.md` | the deterministic revert rule the loop never implements |
