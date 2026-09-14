# Repo Analysis: hexo-ai/sia

**Path:** `refs/related-work/SIA/repo/` · **Source:** https://github.com/hexo-ai/sia · **Analyzed:** 2026-09-14 (commit `7fd04d0`, authored 2026-08-26) · **License:** MIT · **Package:** `sia-agent` 0.6.0 on PyPI

## Overview

- **Paper:** SIA: Self Improving AI with Harness & Weight Updates (see `paper_analysis.md`)
- **Framework:** custom orchestrator, roughly 9,500 lines of Python across 61 files; no training framework vendored
- **RL Algorithm:** none implemented in this repo — weight updates are delegated to `tinker-cookbook` (Thinking Machines Lab), pulled at runtime from a git nightly (`sia/config.py:71-72`) and executed remotely on Modal or SandboxFusion
- **Base Model:** configurable per run through JSON profiles (`sia/defaults/profiles/`); bundled targets include gpt-oss on Nebius and on Tinker, Qwen3 4B Instruct on Tinker, Kimi on Nebius, and any OpenRouter model. The meta and feedback roles default to Claude via the Claude Agent SDK
- **Key Innovation (as built):** a generation loop in which an LLM reads a full execution trajectory and writes the next generation's `target_agent.py` wholesale, with a task's `evaluate.py` as the only source of score

## File Structure Map

```
sia/
  cli.py                      argument parsing; the `run` and `web` sub-commands (186 lines)
  orchestrator.py             the whole loop: setup, per-generation execution, evaluation, feedback (967 lines)
  prompts.py                  meta and feedback prompt builders, plus an embedded RL integration guide (960 lines)
  context_manager.py          context.md: per-generation entries, metric deltas, LLM summaries, final best-generation pick (558 lines)
  agent_impls/                runner backends: claude.py (Claude Agent SDK), openhands.py, pydantic_ai.py, base.py
  agent_reference.py          resolves the reference agent handed to the meta-agent
  config.py                   all tunables as a dataclass with SIA_* env overrides
  layout.py                   run/gen/task directory names; the only place paths are built
  profiles.py, providers.py   (agent_impl, model, provider) bundles loaded from defaults/*.json
  results.py                  two small dataclasses replacing tuple returns (34 lines)
  run_setup.py, config_files.py, io_utils.py, api_keys.py, util.py, logging_setup.py
  web/                        FastAPI live dashboard over runs/ (server.py, runs.py, static/index.html)
  prepare_mlebench_dataset.py builds MLE-Bench task folders
  defaults/providers/*.json   endpoint + auth per provider (anthropic, gemini, nebius, openai, openrouter, tinker, together)
  defaults/profiles/*.json    per-role (model, provider, agent_impl) bundles
  tasks/<task>/               gpqa, lawbench, longcot-chess, spaceship-titanic
    data/public/              task.md, unlabeled test csv, evaluate.py, sample_submission.csv
    data/private/             ground-truth labels, read only by evaluate.py
    reference/                reference_target_agent.py, SAMPLE_TASK_DESCRIPTIONS.md
tests/                        30 test files; golden/ pins the exact meta and feedback prompt text
docs/                         architecture, configuration, evaluator_design, walkthrough, troubleshooting
```

## Component Inventory

### 1. Entry points

`pyproject.toml:46-47` exposes one console script, `sia = "sia.orchestrator:main"`. `sia/cli.py` defines two sub-commands, `run` (the loop) and `web` (the dashboard), with `sia --task ...` still accepted as `sia run ...`. The flags that matter:

| Flag | Default | Effect |
|---|---|---|
| `--task` / `--task_dir` | — | bundled task name, or an external task directory |
| `--max_gen` | 3 (`sia/config.py:25`) | number of generations |
| `--focus` | `harness` (`sia/cli.py:83-89`) | `harness` for scaffold edits, `weights` for RL tuning |
| `--sandbox` | `none` | `docker` isolates target-agent execution with no network |
| `--training_sandbox` | `modal` | where the generated `train.py` executes code |
| `--meta-agent-profile` / `--target-agent-profile` | `default-meta` / `default-target` | which (model, provider, impl) each role uses |

The loop body is `sia/orchestrator.py:920-956`: build the meta prompt, run the meta-agent once, then `for current_gen in range(1, max_gen + 1): run_generation(...)`.

### 2. Special token / action handling — **absent, and this is the load-bearing finding**

There are no action tokens and no interception of generation. The improvement "action" is a file the feedback agent writes into `gen_{n+1}/`: `target_agent.py` in harness mode, `train.py` in weights mode (`sia/orchestrator.py:662-663`).

The paper's lever selector is not in the code. `--focus` is a launch-time argument with `choices=["harness", "weights"]`, threaded unchanged from `args.focus` through `run_generation` into the prompt builders and held constant for every generation of the run (`sia/cli.py:83-89`, `sia/orchestrator.py:864, 890, 942`). The Feedback-Agent is never asked which lever to pull; a human picks the mode before launch and the loop stays in it. The paper's Sec. 5.1 — "the Feedback-Agent... dynamically selects, at each step, between two complementary actions" — has no counterpart here. The only feedback-agent control signal over the loop's length is a `COMPLETED` sentinel file in weights mode that stops the loop early (`sia/orchestrator.py:947-952`).

### 3. Environment / tool interaction

Two nested layers. The meta and feedback agents run through a pluggable impl (`sia/agent_impls/claude.py` wrapping the Claude Agent SDK; `openhands.py` and `pydantic_ai.py` for multi-provider), writing files into a working directory. The target agent is plain Python: `_run_target_agent` (`sia/orchestrator.py:355-400`) executes `gen_{n}/target_agent.py` as a subprocess inside a per-run virtualenv, passing `dataset_dir` and `working_dir` as the two positional arguments (`sia/prompts.py:660`), streaming stdout to a log. With `--sandbox docker`, `_run_target_agent_sandboxed` (`sia/orchestrator.py:312-352`) runs it in a container with a 2 GB memory and 2 CPU cap (`sia/config.py:46-47`) and no network. Each generation may ship its own `requirements.txt`, installed into the run venv before execution.

### 4. Token masking — absent

No training happens in-process, so there is no loss mask, no token-type handling, nothing to exclude from a gradient.

### 5. Reward function

The reward is the task's `evaluate.py`, run as a subprocess after every generation (`run_evaluation`, `sia/orchestrator.py:185-240`), resolved from `data/public/evaluate.py` and scoring `gen_{n}/submission.csv` against `data/private/`. For LawBench (`sia/tasks/lawbench/data/public/evaluate.py:19-45`) it merges predictions onto the private truth by id, fills missing ids with a sentinel, and returns accuracy plus a per-class breakdown. Deterministic, no model in the loop, no judgment — the same design our own launcher's `summary.json` step takes.

The label split is clean: `data/public/test.csv` carries `id,text` for 913 LawBench cases, `data/private/test.csv` carries `id,label` for the same 913 ids. The agent cannot read labels. What it can do — and does, every generation — is read its own accuracy on exactly those 913 cases and edit to raise it, with no development set in between. There is no held-out set that the improvement loop has not optimized against.

`docs/evaluator_design.md` is a short guide warning task authors that "the evaluator is the task specification: if it rewards the wrong behavior" the loop will chase it, and lists failure modes including "rewarding verbosity, code size, or tool usage instead of final quality."

### 6. RL training loop — not implemented; generated

In weights mode the meta-agent writes `train.py` instead of `target_agent.py`. `_build_weights_meta_prompt` (`sia/prompts.py:39-610`) embeds a nine-section "RL Integration Guide: Custom Task Tuning with Tinker-Cookbook" directly into the prompt: the `Env` / `EnvGroupBuilder` / `RLDataset` abstractions, a reward-shaping table (format reward in `Env.step()`, per-turn penalty, correctness reward in `compute_group_rewards()`), and a worked `tinker_cookbook.rl.train.Config` with `batch_size=16`, `group_size=8`, `learning_rate=1e-5`, `max_tokens=1024`. The guide's own advice is group-relative: "Don't Normalize: Tinker automatically centers rewards per group (`advantage = reward - group_mean`)" and "RL works best when the model can clearly distinguish a 'Winner' from a 'Loser' within the same 8-sample group."

So the released weight lever is one group-relative recipe, written by an LLM against a third-party library. The paper's six-objective menu (PPO with GAE, GRPO, entropic advantage weighting, REINFORCE with KL-to-base, best-of-$N$ cloning, DPO; Sec. 7.3) is not in the repo, and neither is the trajectory-conditioned rule for choosing among them. The paper's Modal-hosted "our RL training platform" (Sec. 4.3) appears here only as a sandbox for the generated script to execute code in.

### 7. SFT / cold-start pipeline — absent

No supervised phase, no trajectory filtering. Generation 1 is whatever the meta-agent writes.

### 8. Data pipeline

Per-task folders under `sia/tasks/`, each with `data/public/` (a `task.md` written for the agent to read, the unlabeled evaluation csv, `evaluate.py`, `sample_submission.csv`), `data/private/` (labels), and `reference/` (a `reference_target_agent.py` shown to the meta-agent as a shape to imitate, plus `SAMPLE_TASK_DESCRIPTIONS.md`). LawBench additionally ships `data/training_data/` with 5,332 labeled train rows for the weights path. `prepare_mlebench_dataset.py` generates task folders for MLE-Bench competitions. External tasks are accepted through `--task_dir` with no registration step.

### 9. Multi-GPU / infrastructure — absent locally

No FSDP, DeepSpeed, DDP, or tensor parallelism. Gradient work is remote: a Tinker API key is mandatory in weights mode (`sia/orchestrator.py:822-828`) and a Modal token is required when the training sandbox is Modal (`sia/orchestrator.py:831-833`).

### 10. Evaluation and history

`ContextManager` (`sia/context_manager.py`) maintains `runs/run_{id}/context.md` as the loop's memory. Per generation it extracts metrics, diffs them against the previous generation (`_format_metrics_comparison`, line 183), computes code-size deltas, and calls an LLM for a prose summary of what changed (`_generate_llm_summary`, line 66). The feedback prompt then instructs the agent to read that file first: "Don't repeat failed approaches from earlier generations. Build upon successful patterns that improved performance" (`tests/golden/feedback_prompt.txt`).

`finalize()` (`sia/context_manager.py:267-311`) scans all generations for the highest accuracy and writes a "Best Performance: Generation N" line. This is the only place a best generation is identified, and it happens **after** the loop ends. Nothing in the loop reverts a bad generation, re-runs a prior scaffold, or refuses to deploy an edit — generation $n+1$ replaces generation $n$ whichever way the score moved. There is no commit gate, no anchor set, and no retention check anywhere in the repository.

### Prompt discipline worth copying

`tests/golden/` pins the meta prompt, the feedback prompt, and three feedback-context variants as exact text files, checked by `tests/test_prompts_snapshot.py`, `test_context_golden.py`, and `test_feedback_context_golden.py`. Any prompt edit fails a test until the golden file is updated deliberately. This is the mechanism our own standard — a prompt is code, and "wording stays stable across a comparison sequence" — needs, and it costs about three small test files.

## Relevance to this project

### Components we can directly reuse

- **The task folder contract** (`sia/tasks/<task>/data/{public,private}` plus `evaluate.py`): unlabeled inputs and a human-readable `task.md` on one side, labels reachable only by the grader on the other, and the grader as a plain script that emits a metrics dict. It gives an agent-facing task a single shape and keeps scoring free of judgment. `sia/tasks/lawbench/data/public/evaluate.py` is 137 lines and readable end to end.
- **Golden-file prompt pinning** (`tests/golden/`, `tests/test_prompts_snapshot.py`): the cheapest enforcement of stable prompt wording across a comparison sequence.
- **Single-place path construction** (`sia/layout.py`, 196 lines): every run, generation, and task path is a method on a layout object; nothing else in the codebase joins path strings.
- **The per-generation history file** (`sia/context_manager.py`): a growing markdown log of what each edit changed and how the metric moved, handed back to the editor as context. The structure is directly usable as the input representation for a critic over harness state, and a run's `context.md` plus its `gen_*/target_agent.py` sequence is a harness-evolution log of exactly the kind our critic pretraining would consume.

### Components we need to modify

- **The lever selector does not exist and must be built, not adapted.** Threading a per-step decision through `run_generation` means replacing the `focus` parameter (`sia/orchestrator.py:648`) with a choice the feedback agent returns, which is a small change to this code and the whole of what the paper claims.
- **The scoring protocol is the opposite of ours.** Every generation is scored on the same fixed dataset after the edit. Our score-before-update protocol over a task stream requires a task queue this repo has no concept of — `run_generation` takes one `dataset_dir` for the life of the run.
- **Edits are whole-file rewrites.** The feedback agent regenerates `target_agent.py` in full each generation; there is no structured edit action, no layer targeting, and no diff. Our action space — add, remove, rewrite, merge, or move content across layers — has no representation here, because the harness is one Python file.
- **A gate would have to be added from scratch.** There is no accept/reject decision and no revert path anywhere in the loop.

### Components that don't apply

- The RL guide embedded in `sia/prompts.py:51-610` targets `tinker-cookbook`'s hosted API; we would not have an LLM write a training loop against a third-party service.
- The web dashboard (`sia/web/`) serves interactive run browsing, which our executed-notebook reports already cover.
- `prepare_mlebench_dataset.py` is MLE-Bench specific.

### Key code snippets worth studying

- `sia/orchestrator.py:636-760` — one generation start to finish: execute, evaluate, feed back. The whole loop is about 120 lines, which is a useful ceiling on how large ours should get.
- `sia/orchestrator.py:920-956` — the generation loop and the `COMPLETED` early-stop, plus `finalize()` picking the best generation after the fact.
- `sia/context_manager.py:183-311` — metric diffing, LLM summarization, and the best-generation scan; the clearest read on what the loop remembers and what it decides.
- `sia/prompts.py:747-830` and `tests/golden/feedback_prompt.txt` — the feedback prompt in both builder and pinned form. Two instructions are the paper's entire generalization story: "Focus on structural improvements to the agent scaffold... Make the agent more robust and generalizable. Don't optimize for this specific task" and "Make the agent work well across diverse task types (see sample task descriptions)." Nothing measures whether either happens.
- `sia/config.py:25-52` — every tunable in one dataclass with `SIA_*` environment overrides, including the context previews (agent code truncated to 3,000 characters, trajectories to 1,000, tool results to 500) that bound what the feedback agent actually sees.
