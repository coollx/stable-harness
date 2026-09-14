# Repo Analysis: meta-harness

**Path:** `refs/related-work/Meta-Harness/repo/` · **Source:** https://github.com/stanford-iris-lab/meta-harness · **Analyzed:** 2026-09-14 (commit `0cbc31e`, 2026-09-11)

A second, smaller upstream repository holds the single discovered TerminalBench-2 harness as a standalone runnable agent: https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact (one 53KB `agent.py`). It is not cloned here; its one interesting component is summarized under "Discovered artifact" below.

## Overview

- **Paper:** Meta-Harness: End-to-End Optimization of Model Harnesses (see `paper_analysis.md`)
- **Framework:** custom, thin. Two independent reference experiments, each a self-contained `uv` project; no shared library between them (the duplicated `claude_wrapper.py`, 715 lines in both, is copy-pasted). Terminal-Bench 2 depends on Harbor 0.18.0; text classification depends only on `litellm`, `datasets`, `tenacity`.
- **RL algorithm:** none. There is no training, no gradient, no learned component anywhere in the repo.
- **Base model:** frozen per domain — GPT-OSS-120B via OpenRouter for text classification (`config.yaml`), Claude Opus 4.6 for Terminal-Bench 2. The **proposer** is always Claude Code on Opus at maximum effort.
- **Key innovation:** the filesystem is the feedback channel. The proposer is launched with the tool allowlist `[Read, Glob, Grep, Agent, Write, Edit, Bash]` and pointed at a run directory holding every prior candidate's source, score, and trace; nothing is summarized for it.

**Release caveat, from the README:** "This is a cleaned up version of the code we used for the paper. It has not been tested beyond verifying that it runs."

## File Structure Map

```
README.md                         community projects, quick start, citation
ONBOARDING.md                     a conversation script for adapting Meta-Harness to a new domain (see below — the most transferable file here)
reference_examples/
  text_classification/            domain 1: memory-system search
    meta_harness.py        (629)  the outer loop: propose -> validate -> benchmark -> record; also the one-shot --test finalizer
    benchmark.py           (962)  runs candidates over datasets/models, writes val.json/test.json, computes the Pareto frontier
    inner_loop.py          (825)  the per-episode loop: online (predict then learn) or offline (learn with labels then eval)
    memory_system.py       (127)  the harness interface every candidate implements
    llm.py                 (448)  model client, retries, token accounting
    claude_wrapper.py      (715)  spawns the proposer as a `claude` CLI subprocess and logs the session
    config.yaml                   split sizes, inner-loop mode, model list, dataset list, baseline list
    .claude/skills/meta-harness/SKILL.md   the proposer's instructions (the real "method" of the search)
    agents/                       baselines (no_memory, fewshot_memory, fewshot_all) + the write target for candidates
    data/                         loaders, evaluators, and committed jsonl splits for 5 datasets
  terminal_bench_2/               domain 2: agent-scaffold search
    meta_harness.py        (964)  same loop shape, Harbor jobs instead of local benchmarks; adds a single-task smoke test
    agents/baseline_kira.py (1185) the Terminus-KIRA baseline agent, the parent every candidate copies
    prompt-templates/terminus-kira.txt
    .claude/skills/meta-harness-terminal-bench-2/SKILL.md
    scripts/run_eval.sh           validates the import path is a real Terminus2 subclass, then runs Harbor
experimental/harbor_meta_harness/ a pilot that runs the loop as a Harbor task, with mechanized leakage checks
    controller.py          (308)  trusted scorer: validates candidate source against forbidden strings, then scores on a suite
    tasks/select-harness/         the outer task: "improve /app/harness.py" with a bounded evaluate_harness tool
```

## Component Inventory

The repo-analysis protocol's ten components are written for reinforcement-learning training repos. Five have no counterpart here and are marked absent rather than forced.

### 1. Entry points

| Entry | Command | What it does |
|---|---|---|
| `reference_examples/text_classification/meta_harness.py:360` (`run_evolve`) | `uv run python meta_harness.py --iterations 20 --fresh --run-name X` | Phase 0 evaluates baselines, then N iterations of propose/validate/benchmark, writing everything under `logs/<run-name>/` |
| same file `:286` (`finalize_run`) | `uv run python meta_harness.py --run-name X --test` | The **only** path that touches the test split; freezes the run first |
| `reference_examples/terminal_bench_2/meta_harness.py` | `uv run python meta_harness.py --iterations 10 --trials 2 --fresh` | Same shape over Harbor jobs on 89 Terminal-Bench 2 tasks |
| `reference_examples/terminal_bench_2/scripts/run_eval.sh` | `... agents.baseline_kira:AgentHarness full 1 1 -i extract-elf` | Single-task smoke run; verifies the import path names a real `Terminus2` subclass before spending money |
| `reference_examples/text_classification/benchmark.py` | called by the outer loop, also runnable standalone | Scores one named harness across all datasets |

### 2. The harness interface (the search space)

This is the repo's equivalent of an action space, and it is the component most worth studying.

**Text classification** — `memory_system.py:61`, four abstract methods:

```python
class MemorySystem(ABC):
    def __init__(self, llm: LLMCallable): ...
    def predict(self, input: str) -> tuple[str, dict[str, Any]]: ...
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None: ...
    def get_state(self) -> str: ...
    def set_state(self, state: str) -> None: ...
```

Plus a non-abstract `get_context_length()` at `:111`, which is what the accuracy-versus-context Pareto frontier is computed against. `predict` must work cold, before any `learn_from_batch` call. A candidate is one file in `agents/`, auto-discovered — the proposer never edits `config.yaml`.

**Terminal-Bench 2** — the candidate is a class named `AgentHarness` subclassing Harbor's `Terminus2`, in one file under `agents/`. The skill file names the overridable seams explicitly: `_call_llm_with_tools` (the litellm call, tool schema, retries), `_parse_tool_calls` (add new tools), `_execute_commands` (tmux execution), `_run_agent_loop` (the episode loop), `_get_completion_confirmation_message`, `_get_prompt_template_path` (swap in a new system prompt file), `_summarize_context` (context-overflow handling), `_execute_image_read`. The stated search space is "arbitrary Python code."

In our vocabulary, one candidate file can span the system prompt, tool description, tool implementation, middleware, and long-term memory layers at once, with model weights frozen. Nothing in the code separates or orders those layers — the layer distinction exists only in what the proposer happens to write.

### 3. Proposer invocation (the "policy")

`text_classification/meta_harness.py:39` and `:164`:

```python
PROPOSER_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Agent", "Write", "Edit", "Bash"]
...
result = claude_wrapper.run(
    prompt=task_prompt, model="opus", allowed_tools=PROPOSER_ALLOWED_TOOLS,
    skills=[str(EVOLVE_DIR / ".claude/skills/meta-harness")],
    cwd=str(EVOLVE_DIR), log_dir=str(LOGS_DIR / "claude_sessions"),
    name=f"iter{iteration}", timeout_seconds=timeout, effort="max",
)
```

The task prompt handed to the proposer (`render_task_prompt`, `:136`) is four lines: the iteration number, the dataset count, the paths to `evolution_summary.jsonl` / `frontier_val.json` / `reports/`, and where to write `pending_eval.json`. Everything else the proposer needs, it finds by reading. `claude_wrapper.py` runs the `claude` CLI as a subprocess with local skills and plugins disabled, so the search does not silently inherit the developer's own setup — and it strips `ANTHROPIC_API_KEY` so the CLI uses subscription auth.

The proposer returns `pending_eval.json`, a list of candidates each carrying `name`, `file`, a **falsifiable `hypothesis` string**, an `axis` tag, `base_system`, and `components` tags. Those fields are recorded per candidate in `evolution_summary.jsonl` alongside the realized score — which makes the log a dataset of (harness diff, stated hypothesis, realized delta) triples.

### 4. Proposer skill files (prompts as code)

`text_classification/.claude/skills/meta-harness/SKILL.md` and the Terminal-Bench 2 equivalent are where the search's actual heuristics live. Both are short and worth reading whole. The notable content:

- **Forced productivity**: "You MUST implement 3 new memory systems every iteration. Do NOT write 'the frontier is optimal' or 'stop iterating', or abort early." The coding variant demands 1 candidate per iteration.
- **Anti-parameter-tuning**: an explicit list of what counts as a new mechanism (a new retrieval algorithm, prompt architecture, learning strategy, memory structure) against what does not ("If the logic in `predict()` and `learn_from_batch()` is identical to the base except for constants, it's a parameter variant. Rewrite"). Six exploitation axes are named A–F with a rule to rotate off an axis explored in the last three iterations.
- **Mandatory prototyping**: write a throwaway script in `/tmp/`, exercise the mechanism against real logged examples, try 2–3 variants, then implement.
- **Anti-overfitting**, hand-written and unmeasured: "No dataset-specific hints", "Never mention dataset names in system code, prompts, or comments", "General patterns are OK. Rules like 'prioritize recent errors' or 'balance label coverage' are fine." The coding variant adds the test "would this advice be useful to a human developer working on MANY unfamiliar tasks?" and "If in doubt, make it more general."
- **Copy-then-edit**: candidates start as a copy of a top performer, never an import from another candidate.

### 5. Scoring and selection

No reward function in the RL sense. The score is the metric itself: average validation accuracy across datasets for classification, `total_passes / total_trials` across 89 tasks x 2 trials for coding (`terminal_bench_2/meta_harness.py:256`).

Selection is Pareto dominance over (accuracy, context length) — `benchmark.py:274`:

```python
def compute_pareto_frontier(points):
    """Compute Pareto frontier for (name, accuracy, ctx_tokens)."""
```

No parent selection, no tournament, no elitism, no population cap. `update_frontier` (`terminal_bench_2/meta_harness.py:373`) keeps a per-task best and an overall best; the frontier is presented to the proposer as a file, not used to constrain what it may propose.

### 6. Validation before evaluation

`validate_candidates` (`text_classification/meta_harness.py:192`) runs one subprocess per candidate: `python -c "from text_classification.agents.<name> import *; print('OK')"` with a 30-second timeout. That is the entire admission check in the cheap domain. The expensive domain adds `validate_agent_class` (`:140`, checks the import path resolves to a class subclassing `Terminus2`) and `smoke_test` (`:481`, one task, 1800-second timeout) before committing to a ~$500 full evaluation.

### 7. Search-set / test-set separation

The mechanism is **temporal, not access control**, and this is worth being precise about because the paper leans on it.

- During evolution, only `val.json` files exist. `finalize_run` (`:286`) is the only writer of `test.json`.
- `finalize_run` writes `finalized.json` with `status: in_progress`, evaluates the Pareto frontier plus baselines on test, then flips to `complete`.
- `run_evolve` (`:398`) refuses to continue a run whose `finalized.json` is `complete`: "run '<name>' is finalized; use a new --run-name to continue evolution."
- The proposer has `Read`, `Grep`, and `Bash` over the working directory, so nothing *prevents* it from reading a test file. The guarantee is that no test file exists while it is running.

The `experimental/harbor_meta_harness` pilot goes further and is the only mechanized anti-leakage check in the repo: each task declares `[meta_harness].forbidden_references` in its `task.toml` (the task name, input and output filenames, other giveaway strings), and `controller.py:144` `validate_source` rejects any candidate harness whose source contains one before scoring it (`:272`). The suite config carries an `eval_budget` and an aggregate over `mean`, `min`, or `fraction_solved`.

### 8. Inner loop and the scoring order

`inner_loop.py` supports two modes (`:285`). **Online** (`:322`) is per-batch predict-then-learn: phase 1 predicts every example in the batch with the memory as it currently stands, phase 2 hands the batch results — `input`, `prediction`, `ground_truth`, `was_correct`, `metadata` — to `learn_from_batch`. At the shipped `batch_size: 1` this is exactly score-before-update (each task scored with the harness as it stood before that task arrived) *at the level of the harness's own memory*. **Offline** (`:108`) trains with ground truth visible for `num_epochs`, then evaluates.

The shipped `config.yaml` uses `mode: offline, num_epochs: 1`. Either way, the outer loop is unaffected: **every candidate harness is constructed fresh and run from a cold start on the same search set.** Nothing carries a harness forward across iterations and patches it in place; there is no sequence of harness states in this codebase at all.

### 9. Data pipeline

Text classification ships committed jsonl train/val/test splits for five datasets (`data/`: aegis2, crime_prediction, finer, symptom_diagnosis, uspto) plus HuggingFace loaders (`data/loaders.py`, 509 lines) and per-dataset evaluators (`data/evaluators.py`). Split sizes from `config.yaml`: default 200 train / 50 val / 100 test, with USPTO at 50/30/100, Symptom2Disease at 200/50/212, LawBench at 200/50/100. `seeds: [42]` — single seed. `concurrency: 16`.

Terminal-Bench 2 pulls the 89-task dataset through Harbor and runs sandboxes on Runloop or Modal.

### 10. Absent components

No special-token handling, no token masking, no reward model, no RL training loop, no supervised phase, no multi-GPU or distributed training. The repo is an orchestration layer around API calls; the only GPU anywhere would be behind the OpenRouter endpoint.

### Discovered artifact (second upstream repo)

`meta-harness-tbench2-artifact` is one 53KB `agent.py` reporting 76.4% on Terminal-Bench 2.0 (89 tasks x 5 trials, Opus 4.6; Easy 100.0 / Medium 81.1 / Hard 64.7). Its single discovered delta over Terminus-KIRA is `_gather_env_snapshot` (`agent.py:873`), called once at `:987` before the agent loop: one compound shell command collecting working directory, a truncated `/app` listing, available languages and versions, package managers, and memory, injected as an `[Environment Snapshot]` block into the initial prompt, guarded by a 15-second timeout that fails silently. About 80 lines. The README states the purpose plainly: "This saves 2-5 early exploration turns that the agent normally spends on `ls`, `which python3`, etc."

### `ONBOARDING.md`

Not code, but the most directly useful file in the repo for anyone building a comparable loop. It is a prompt that walks a coding assistant through producing a `domain_spec.md`, with required fields grouped as problem framing, harness definition, evaluation, baselines, offline experience, online experience. It carries a screening list of when Meta-Harness fits (long-horizon tasks, repeated episodes, frozen base model, a real metric, a search set large enough to expose failures and small enough to iterate on, recurring error patterns, existing traces, a plausible held-out set) and a standing instruction: "Be especially careful about evaluation leakage and hidden dependence on the final test set." The worked example dialogues include one that pushes back on a vague user, one that screens out a poor-fit domain, and one that forces a concrete budget.

## Relevance to this project

### Components we can directly reuse

- **The proposer-invocation pattern** — `reference_examples/text_classification/claude_wrapper.py` plus `meta_harness.py:164`. A coding agent launched as a subprocess with an explicit tool allowlist, local skills disabled, session logged to disk. This is the cleanest available reference for running a harness-editing proposer without inheriting the developer's environment, and it is the piece we would otherwise write badly.
- **The candidate record format** — `pending_eval.json` and `evolution_summary.jsonl` (`meta_harness.py:218`). One line per candidate carrying the stated hypothesis, the base it built on, component tags, and the realized delta. This is very close to the shape a critic's training data would need: a harness state, an edit, and a realized return. What it is missing is the return over anything later than the same evaluation.
- **The Pareto-frontier helper** — `benchmark.py:274`, twenty lines, no dependencies. Our objective has two terms as well, though the second is a different quantity.
- **The mechanized leakage check** — `experimental/harbor_meta_harness/controller.py:144`. Declaring per-task forbidden strings and rejecting candidate source that contains them is a concrete, cheap guard we could adopt for our own anchor-set hygiene.

### Components we need to modify

- **The objective.** `compute_pass_rates` and the average-validation-accuracy path score a candidate on one fixed set and nothing else. To express our objective we would add a term over later batches and a retention check against previously passed tasks; neither has a hook in this code, because there is no later batch and no notion of a previously passed task.
- **The harness lifecycle.** Every candidate is constructed fresh and run cold on the same set. Our setting needs a single harness carried forward and edited in place along an arriving stream, with each task scored against the harness as it stood before that task arrived. That is a different loop shape, not a parameter change — `run_evolve`'s structure would not survive it.
- **The anti-overfitting rules.** Both `SKILL.md` files encode our concern as prompt text ("If in doubt, make it more general"). That is the thing our claim proposes to replace with an estimated quantity. Keeping them as a baseline to beat is more useful than keeping them as a component.
- **The evaluation split.** The temporal separation trick (test files simply do not exist during search) is clean and cheap and we should keep it; but it protects a final report, not an optimization signal, and it says nothing about later tasks.

### Components that don't apply

- All the reinforcement-learning machinery the protocol asks about — masking, advantages, rollout groups, distributed training — is absent because the method has no learned component. If we train a proposer or a critic, none of this repo helps.
- The Harbor / Runloop / Modal sandbox plumbing in `terminal_bench_2/` is specific to Terminal-Bench 2 and to a ~$500-per-iteration budget.
- `data/loaders.py` and the classification datasets are domain-specific; the task stream we need is heterogeneous by construction.

### Key code snippets worth studying

- `reference_examples/text_classification/.claude/skills/meta-harness/SKILL.md` (whole file, ~140 lines) — the actual search heuristics, including the anti-parameter-tuning rules and the six rotation axes. The paper's Appendix D says iterating on this file "had a larger effect on search quality than changing iteration count or population size."
- `reference_examples/text_classification/meta_harness.py:286-357` (`finalize_run`) — the freeze-then-test-once protocol, in 70 readable lines.
- `reference_examples/text_classification/inner_loop.py:322-372` — the online predict-then-learn loop; the closest thing in the reference set to score-before-update, at the memory level rather than the harness level.
- `reference_examples/terminal_bench_2/meta_harness.py:431-528` — propose, validate the import path, smoke-test on one task, only then commit to the full benchmark. The cost-control ladder is worth copying regardless of objective.
- `experimental/harbor_meta_harness/controller.py:144-180, 264-295` — source-level leakage validation before scoring.
