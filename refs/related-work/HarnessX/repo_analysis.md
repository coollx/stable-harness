# Repo Analysis: HarnessX

**Path:** `refs/related-work/HarnessX/repo/` · **Source:** https://github.com/Darwin-Agent/HarnessX · **Analyzed:** 2026-09-14 (commit `bf5f199`, dated 2026-07-29)

## Overview

- **Paper:** *HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry* (see `paper_analysis.md`)
- **Framework:** custom. HarnessX is itself the harness framework — an async event pipeline with typed processors — plus thin bridges out to veRL and slime for model training. MIT licence, v0.1.0 Beta, Python 3.11+, 505 Python files and roughly 108k lines, 85 test files. Dependencies are `pydantic`, `anthropic`, `openai`, `litellm`, `httpx`, `structlog`, `tiktoken`, `hydra-core`/`omegaconf`, `fastapi`/`uvicorn` (a web lab UI ships in `frontend/`).
- **RL Algorithm:** Group Relative Policy Optimization (GRPO) for the task model, via `recipe/verl_harnessX/` (veRL multi-turn agentic rollout) and `recipe/slime/`. The harness-evolution loop itself is not gradient-based — it is a language model writing files behind a scripted accept/revert gate.
- **Base Model:** provider-agnostic through litellm. Recipe defaults (`recipe/gaia_evolver/defaults.py:8-10`) are `claude-sonnet-4-6` as the task agent and `claude-opus-4-6` as the meta-agent; the veRL recipe targets a local Qwen served through SGLang.
- **Key Innovation:** the harness is a serializable `HarnessConfig` — a flat list of typed processors bound to eight lifecycle hooks — so a language model can rewrite it as a YAML file plus a few Python modules, and a scripted gate can accept or revert the result.

**Headline finding for us: the release is a substantially smaller system than the paper.** Three of the paper's named mechanisms have no implementation at this commit. The string `AEGIS` appears zero times. There is no `Planner`, `Evolver`, or `Critic` stage (the one `Planner` hit is prose inside a tau-cubed-Bench playbook skill; all `Evolver` hits are the recipe folder names `gaia_evolver` / `tau2_evolver` / `tb2_evolver`; all `Critic` hits are the string `CRITICAL` in log lines). There is no seesaw constraint (`seesaw`: zero hits), no ensemble routing over harness variants (every `variant` hit is a processor-contract or SGLang naming variant, none the paper's harness variants), and no shared replay buffer or cross-harness GRPO grouping (`replay_buffer`, `cross-harness`: zero hits). The change-manifest field names from Table 9 (`predicted_impact`, `attribution_signature`) also do not appear; the released journal uses a different, simpler schema (below). What ships is a single `MetaAgent` with a prompt workspace and four skills, gated by an aggregate pass-rate comparison — structurally the shape of the paper's own single-agent comparison in Table 6, not the four-stage pipeline of Algorithm 1.

## File Structure Map

```
harnessx/
  cli.py                       entry point for both `hx` and `harnessx` console scripts
  core/
    harness.py                 HarnessConfig (:743): tool_registry, tracer, processors, workspace, sandbox, plugins
    processor.py               Processor / MultiHookProcessor; the eight hook dispatch table (:561-568)
    events.py                  the eight event dataclasses (TaskStart .. TaskEnd)
    runloop.py                 the agent loop that fires hooks and drives tool calls
    builder.py                 config -> live object instantiation from `_target_` dicts
    config_schema.py           YAML schema for a harness config
    contract_check.py          import-time validation that processors implement legal hook methods
    model_config.py            model routing / fallback; `model_config.agentic(harness_config)`
    trajectory.py, state.py    per-run records and shared slot state
  processors/                  the shipped processor library, one folder per behavioral dimension
    context/                   system_prompt.py, user_wrapper.py, env_context_injector.py
    control/                   14 modules: compaction, cost_guard, loop_detection, token_budget,
                               self_verify, sycophancy_detector, tool_failure_guard, rl_signal, ...
    memory/                    memory_extraction.py, memory_retrieval.py
    tools/                     skill_loader.py, tool_filter.py, tool_whitelist.py, model_schema_adapter.py
    evaluation/                evaluation.py, llm_judge.py
    multi_model/               model_router.py
    observability/             checkpoint.py, episode_metrics.py, otel_proc.py
  tools/                       base.py registry, builtin/, mcp.py, code_execution.py, spawn_subagent.py
  tracing/                     base.py, journal.py, otel_tracer.py, null_tracer.py
  sandbox/                     sandbox providers (local, container, harbor)
  meta_harness/               <- THE EVOLUTION LOOP
    agent.py                   build_meta_agent_harness_config (:98), compute_changeset (:439), MetaAgent (:493)
    journal.py                 JournalEntry schema (:80), build_context (:366), compute_attribution (:605)
    replay.py                  re-runs a recorded trajectory under a given harness
    validate_workflow.py       the pre-ship validation sequence
    workers/trajectory_digester.py   trace -> per-task structured summary
    processors/                six guards applied to the meta-agent itself: write_scope_gate,
                               read_scope_gate, leakage_guard, contract_autocheck,
                               step_deadline_reminder, tool_result_noise_filter
    workspace/SOUL.md          the meta-agent's own system prompt (155 lines)
    workspace/skills/          analyze/ journal/ reference/ validate/ SKILL.md
  rl/                          builder.py, config.py, task.py (RLTask, ProcessRewardModel)
  plugins/, providers/, api/, bundles/, config/, workspace/
benchmarks/                    gaia/ swebench/ tau2/ terminal_bench_2/ locomo/ — task loader,
                               harness, evaluator, run script per benchmark
recipe/
  gaia_evolver/run.py          the full evolution loop, 1300+ lines; _score_and_gate at :1174
  tau2_evolver/run.py          same shape, reward-based gate at :1164; ships guidance_retail.md
  tb2_evolver/run.py           Terminal-Bench 2 variant with a task-sample JSON
  verl_harnessX/               multi-turn agentic GRPO on veRL: agent_loop.py, reward.py,
                               dataset.py, config.yaml, run_train.sh
  slime/                       the same bridge for the slime trainer
examples/ extensions/ gateway/ frontend/ docs/ tests/ container/
```

## Component Inventory

### Primary contribution: the composition substrate and the evolution loop

**The eight hooks.** `harnessx/core/processor.py:561-568` is the whole dispatch table, and it matches Table 1 of the paper exactly: `TaskStartEvent -> on_task_start`, `StepStartEvent -> on_step_start`, `BeforeModelEvent -> on_before_model`, `ModelResponseEvent -> on_after_model`, `ToolCallEvent -> on_before_tool`, `ToolResultEvent -> on_after_tool`, `StepEndEvent -> on_step_end`, `TaskEndEvent -> on_task_end`. A processor subclasses `MultiHookProcessor` and overrides only the handlers it needs; `contract_check.py` rejects near-miss method names (`on_tool_call` for `on_before_tool`) at import time. Composition metadata is three class attributes: `_singleton_group` (mutual exclusion), `_order` (`PRE`/`NORMAL`/`POST` or an integer offset, phases defined at `processor.py:334`), and `_after` (`processor.py:610`, soft ordering against named singleton groups).

**The harness as data.** `HarnessConfig` (`harnessx/core/harness.py:743`) holds `tool_registry`, `tracer`, `processors` (a flat list of `_target_` dicts with keyword arguments), `workspace`, `workspace_template`, `init_workspace`, `step_snapshots`, `sandbox_provider`, `sandbox_hint_id`, `plugins`. It carries no model information; the agent is assembled as `agent = model_config.agentic(harness_config)`. This is what makes an edit a YAML diff rather than a code refactor, and `compute_changeset` (`harnessx/meta_harness/agent.py:439`) diffs two configs by tool-name set, processor-label set, per-processor keyword fingerprints, and template-file fingerprints (`_processor_kwargs_fingerprint` :381, `_template_fingerprints` :423).

**The proposer action space — this is the part that answers our open item.** The meta-agent is a single agent (`MetaAgent`, `agent.py:493`, `async def evolve` at :538) running against `workspace/SOUL.md` with four skills. `workspace/skills/analyze/SKILL.md` enumerates the action space as exactly four **levers**, and `journal.py:15` enforces the same four as a validated enum (`_VALID_LEVERS`, checked at `journal.py:155-157`):

| Lever (released code) | What the meta-agent may write | Paper's bucket name |
|---|---|---|
| `configuration` | keyword arguments on an already-registered processor | `config` |
| `control` | a new `MultiHookProcessor` module under `processors/` | `processor` |
| `action` | a new `@tool` module under `tools/` | `tools` |
| `instruction` | a Jinja template edit, a `SOUL.md` edit, or a new skill file | `prompt` |

Its required output is `output_dir/config.yaml` plus optional `tools/<name>.py`, `processors/<name>.py`, `templates/<name>.j2`, plus one journal entry. The analyze skill crosses these four levers with three lenses (failure, capability gap, success) and three intents (corrective, preservative-lock, preservative-transfer), and requires a three-variant retroactive check per candidate: A "If this fix had already been in place on the cited failing trajectories, would the task have succeeded?", B "If this pattern were removed from the current config, would the cited passing trajectories have failed?", C "If this pattern had been applied to the cited failing trajectories, would they have passed?" — with the stated rationale that "without Variant B/C, Success-lens evidence has no path into candidates.md, and the round ships fixes while quietly regressing what was already working." A failure cluster earns a configuration change only when the same shape recurs across at least two distinct tasks.

**The released gate.** `recipe/gaia_evolver/run.py:1174-1257`, `_score_and_gate`, is the entire acceptance mechanism:

```python
def _score_and_gate(*, round_pass_rate, round_cost, round_idx, round_config,
                    round_passed, best, tolerance, cost_weight,
                    pass_count_noise_threshold: int = 3):
    """Best-so-far gating kernel.

    Compares this round to the historical best (not the last-accepted round)
    so tolerance cannot drift the baseline downward over many rounds."""
    ...
    score = round_pass_rate - cost_weight * max(cost_delta_ratio, 0.0)
    best_score = best_rate
    if score < best_score - tolerance:
        count_delta = abs(round_passed - best_passed)
        if count_delta < pass_count_noise_threshold:
            return ("ACCEPTED", reason, best, None)   # noise-level regression
        return ("REVERTED", reason, best, best_cfg)
    if score > best_score:
        new_best = (...)
    return ("ACCEPTED", reason, new_best, None)
```

Three things follow. The gate is an **aggregate pass rate on the round's own fixed evaluation set**, not the paper's per-task no-regression constraint — a candidate that breaks two previously passing tasks and fixes two others is accepted with no record of the trade. Comparison is against the historical best rather than the last accepted round, explicitly so tolerance cannot ratchet the baseline down. And three tunable numbers decide what counts as a regression: `--regression-tolerance` (default 0.03), `--cost-weight` (default 0.0, so cost is inert unless switched on), and `--pass-count-noise-threshold` (default 3, from `defaults.py:23`). `recipe/tau2_evolver/run.py:1164` carries a parallel reward-based version.

**The attribution record.** `harnessx/meta_harness/journal.py` is the closest released analogue to the paper's change manifest, with a smaller schema (`journal.py:15-18`): `levers` (subset of the four above), `predicted_affected` (task ids the round's author expects to flip from fail to pass), `gating_outcome`, and `gating_attribution` — a dict mapping each task id to `flipped | still_F | regressed | still_T | absent`, computed by `compute_attribution` (:605). `build_context` (:366) turns the journal into proposer-facing reputation: per-lever precision as `precision_hits / precision_denom`, where the denominator adds each round's `regressed_unpredicted` count specifically so that (quoting :391) "a round that flipped 3 predicted tasks but broke 5 unpredicted ones" cannot register as 100% precision, under a time-decayed Beta posterior over a five-round window. This is **context handed to the proposer, not a gate** — nothing in the accept path reads it.

**Guards on the meta-agent itself.** Six processors in `meta_harness/processors/` constrain the editor: `write_scope_gate` and `read_scope_gate` bound which paths it may touch, `leakage_guard` blocks evaluation-set contamination, `contract_autocheck` validates that generated processors satisfy the hook contract, `step_deadline_reminder` and `tool_result_noise_filter` manage its context. `replay.py` re-runs a recorded trajectory under a given harness, which is what makes the analyze skill's retroactive variants checkable rather than purely rhetorical.

### The 10 protocol components

| # | Component | Status | Location and details |
|---|---|---|---|
| 1 | Entry points | present | Console scripts `hx` and `harnessx` -> `harnessx.cli:main` (`pyproject.toml:63-65`). Per-benchmark runners `benchmarks/{gaia,swebench,locomo}/run_*.py`, `benchmarks/tau2/test_tau2.py`. Evolution loops `recipe/{gaia,tau2,tb2}_evolver/run.py` (`main` at `gaia_evolver/run.py:414`) plus `run_meta.py`. Training `recipe/verl_harnessX/main.py` with `run_train.sh`. |
| 2 | Special token / action handling | effectively absent | No custom action tokens and no generation interception on emit. Tool calls ride provider-native function calling; `harnessx/providers/sglang.py:497` notes the internal `Message.tool_calls` field is named `input` and must be adapted per provider. `processors/tools/model_schema_adapter.py` reshapes tool schemas per model family. |
| 3 | Environment / tool interaction | present, central | `harnessx/tools/base.py` (registry ABC), `tools/inmemory.py` (in-process registry), `tools/builtin/`, `tools/mcp.py` (Model Context Protocol clients), `tools/code_execution.py`, `tools/spawn_subagent.py`. Execution isolation via `harnessx/sandbox/` providers (local, container, Harbor); `benchmarks/terminal_bench_2/dind_environment.py` runs Docker-in-Docker. Results re-enter the loop as `ToolResultEvent` through `on_after_tool`, where `tool_result_noise_filter` and `tool_failure_guard` may rewrite them. |
| 4 | Token masking | present, only in the veRL recipe | `recipe/verl_harnessX/agent_loop.py`: model-generated tokens are appended as `agent_data.response_mask += [1] * n_gen` (:301); tool-result tokens (:367) and forced continuation text (:383) are appended as `+= [0] * len(response_ids)`. So observation tokens are excluded from the loss in the standard way. Truncation bookkeeping at :242-252. |
| 5 | Reward function | present | `recipe/verl_harnessX/reward.py:20-22`: `ACCURACY_REWARD_WEIGHT = 0.8`, `FORMAT_REWARD_WEIGHT = 0.1`, `TOOL_CALL_REWARD_WEIGHT = 0.1`, combined at :337-340. Numeric answers match within a 5% tolerance (`_try_numeric_match`, :160). An LLM-judge accuracy path exists but is commented out (:51-95). Process rewards live at `harnessx/rl/task.py:68` (`ProcessRewardModel`), :116 (`NullPRM`), :141 (`EnhancedToolSuccessPRM`). Harness-level "reward" is the benchmark evaluator (`benchmarks/gaia/evaluator.py` etc.) feeding `_score_and_gate`. |
| 6 | RL training loop | present, but plain | `recipe/verl_harnessX/` inherits veRL's `ppo_trainer` defaults (`config.yaml:6`) with `multi_turn.enable: True` and `max_new_tokens_per_turn: 1024`. Standard GRPO grouping over rollouts of the same prompt. **The paper's equation 2 (cross-harness grouping by task identity across harness versions), the FIFO replay buffer, and the cached behavior log-probabilities are not implemented** — nothing in the recipe tags a trajectory with the harness version that produced it. `recipe/slime/` is the equivalent bridge for the slime trainer (`harness_rollout.py`, `spec.py`, `preflight.py`). |
| 7 | SFT / cold-start | absent | No supervised phase, no trajectory-filtering pipeline. Evolution starts from a hand-built benchmark harness (`benchmarks/<name>/harness.py`), which is round 0. |
| 8 | Data pipeline | present | Per-benchmark `task.py` loaders; `recipe/verl_harnessX/dataset.py` for training; `recipe/tb2_evolver/tasks_all_tb2.json` and `tasks_sample16_seed42_act15.json` pin the task samples. Round outputs are written as per-task trajectory markdown with YAML frontmatter (`_render_trajectory_frontmatter`, `gaia_evolver/run.py:1026`; `write_round_trajectories`, :1151) — these files, not a database, are what the meta-agent reads. |
| 9 | Multi-GPU / infrastructure | present via veRL | `recipe/verl_harnessX/run_train.sh` and `config.yaml` drive FSDP training with SGLang async rollout; the paper reports 8 H100s, batch 256, learning rate $1 \times 10^{-6}$, GRPO clip 0.2, Kullback-Leibler coefficient 0. Agent-side concurrency is a plain asyncio semaphore (`DEFAULT_CONCURRENCY = 4` in `gaia_evolver/defaults.py:16`; the paper used 10). |
| 10 | Evaluation | present | Five benchmark packages: `benchmarks/gaia/evaluator.py` (exact match with an external-judge option), `benchmarks/swebench/evaluate.py` and `evaluate_local.py`, `benchmarks/tau2/` (with `policy_hint.py`, `stop_guard.py`, `tool_filter.py`), `benchmarks/terminal_bench_2/` (Harbor sandbox), `benchmarks/locomo/` (long-term memory, with `judge.py` — a benchmark not reported in the paper). Cross-round reporting is `print_multiround_comparison` (`gaia_evolver/run.py:255`). Gating tests live at `tests/unit/test_gaia_gating.py` and `test_evolve_strict_validation.py`. |

### Explicitly absent relative to the paper

| Paper mechanism | Where described | Repo status at `bf5f199` |
|---|---|---|
| AEGIS, the named four-stage engine | Sec. 4, Algorithm 1 | String absent entirely. One `MetaAgent` with four skills. |
| Digester / Planner / Evolver / Critic as stages | Sec. 4.2-4.4 | Only `workers/trajectory_digester.py` survives as a stage-shaped module; the other three have no implementation. |
| Seesaw constraint (per-task no-regression) | Sec. 4.1, 4.4 | Absent. Replaced by an aggregate pass-rate comparison with tolerance 0.03 and a 3-task noise floor. |
| Variant isolation / ensemble routing over $K$ harnesses | Sec. 4.5, Table 5 | Absent — and this is the paper's headline fix for the -24.3 point collapse. |
| Change manifest (`predicted_impact`, `attribution_signature`) | Table 9, App. B.3 | Field names absent; the journal's `predicted_affected` + `gating_attribution` is a reduced form with no attribution signature. |
| Cross-harness GRPO grouping, shared FIFO buffer, cached behavior log-probabilities | Sec. 5.1, eq. 2-5 | Absent. `recipe/verl_harnessX` is plain multi-turn agentic GRPO. |

Two caveats on reading this table. The clone is `--depth 1` at a commit dated 2026-07-29, six days after arXiv v3, so later releases may close some gaps. And an absence in the release is not evidence that the reported experiments were not run with these mechanisms — it only means the released artifact does not reproduce them.

## Relevance to this project

### Components we can directly reuse

- `harnessx/core/processor.py` and `harnessx/core/events.py` — the eight-hook dispatch table, `_singleton_group`/`_order`/`_after`, and the import-time contract check. This is a working, tested answer to "what is a harness edit allowed to touch", with 85 test files exercising it.
- `harnessx/core/harness.py:743` (`HarnessConfig`) plus `harnessx/core/builder.py` — harness state as one serializable YAML object instantiated from `_target_` dicts, which is exactly the representation a state-level critic $\hat{\Phi}(h)$ would have to read.
- `harnessx/meta_harness/agent.py:439` (`compute_changeset`) — a structural diff between two harness configs over tool names, processor labels, per-processor keyword fingerprints, and template fingerprints. A ready-made edit-distance primitive.
- `harnessx/meta_harness/replay.py` — replaying a recorded trajectory under a chosen harness is the machinery a score-before-update protocol needs to re-check an anchor set cheaply.
- `harnessx/meta_harness/workspace/skills/analyze/SKILL.md` — the four-lever action space, already reduced to a validated enum in `journal.py`, as a starting enumeration for our proposer.
- `recipe/verl_harnessX/agent_loop.py:301,367,383` — the observation-token masking pattern, unremarkable but correct.

### Components we need to modify

- **The gate** (`recipe/gaia_evolver/run.py:1174`). It is aggregate, same-round, and fixed-set. Our objective needs the retention check to run against an anchor set grown from the stream and to be per-task, and it needs a forward term the gate does not have. The useful part to keep is the "compare to historical best, not last accepted" rule, which is a real guard against tolerance ratcheting the baseline down and which our own tolerance sweep will need.
- **The journal** (`harnessx/meta_harness/journal.py`). Its per-lever precision, discounted by unpredicted regressions, is a proposer statistic, not a property of harness state, so it is not a plasticity estimate. But the per-task `flipped | still_F | regressed | still_T | absent` classification with `regressed_unpredicted` folded into the denominator is precisely the label stream a critic trained by regression on realized returns would consume, and it is already computed from logs.
- **The action space.** Four levers cover our system prompt, skill, tool implementation, and middleware layers. Three of our layers are outside it entirely — sub-agent configuration (`tools/spawn_subagent.py` exists as a tool but is not an edit target), long-term memory content (`processors/memory/` is configurable but its contents are not written by the meta-agent), and model weights (moved only by the separate GRPO recipe). There is also no move operation: nothing in the four levers relocates content from one layer to another, so promotion is not expressible.
- **The RL bridge.** If we ever train the harness agent and the task model together, the paper's task-identity grouping would have to be built; the recipe does not have it.

### Components that don't apply

- `gateway/`, `frontend/`, `harnessx/api/` — a chat gateway (Feishu channels), a web lab UI, and a FastAPI server. Product surface, not research.
- `extensions/skills/{docx,pptx,xlsx,pdf}` — document-authoring skills, unrelated.
- `harnessx/plugins/` and its manifest machinery — a packaging format for third-party dimension bundles.
- The process-reward models in `harnessx/rl/task.py:68-141` — per-step shaping on tool success, orthogonal to our objective.
- `benchmarks/locomo/` — a long-term-memory benchmark not used in the paper.

### Key code snippets worth studying

| Location | Why |
|---|---|
| `harnessx/core/processor.py:334-368, 489-620, 718-750` | The hook contract in full: phases, dispatch table, near-miss rejection, default no-op handlers. |
| `harnessx/core/harness.py:743-820` | The complete harness state object. Read this before designing our own harness representation. |
| `recipe/gaia_evolver/run.py:1174-1257` | The released acceptance gate in 80 lines. The clearest available artifact of what a horizon $\gamma = 0$ commit rule actually looks like in production code. |
| `recipe/gaia_evolver/run.py:414-1000` | The round loop end to end: run the set, write trajectories, invoke the meta-agent, gate, revert or keep. |
| `harnessx/meta_harness/journal.py:366-470, 605-672` | Attribution computation and the decayed per-lever posterior, including the `regressed_unpredicted` denominator rule at :386-396. |
| `harnessx/meta_harness/workspace/SOUL.md` | 155 lines of prompt doing the work our objective is meant to do with a number: the Pareto rule ("Do not chase single-task wins that weaken the overall benchmark"), the generalization test ("Would this guidance help an agent solving a task it has never seen before?"), hard invariant 6 "No local-only optimization", and the exploration nudge "Regressions auto-revert, so big bets that fail cost one round's compute; timid bets that 'succeed' waste the round entirely". |
| `harnessx/meta_harness/workspace/skills/analyze/SKILL.md` | The four levers, three lenses, three intents, and the three retroactive variants. |
| `harnessx/meta_harness/processors/{write,read}_scope_gate.py`, `leakage_guard.py` | How they bound a self-editing agent's blast radius and keep the evaluation set out of the edit. |
