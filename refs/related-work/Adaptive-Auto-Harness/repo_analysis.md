# Repo Analysis: a-evolve (branch `release/adaptive-auto-harness`)

**Path:** `refs/related-work/Adaptive-Auto-Harness/repo/` · **Source:** https://github.com/A-EVO-Lab/a-evolve (branch `release/adaptive-auto-harness`) · **Analyzed:** 2026-09-14 (commit `17bc9eb`, authored 2026-06-03) · **License:** MIT

## Overview

- **Paper:** Adaptive Auto-Harness: Sustained Self-Improvement for Agentic System Deployment on Open-Ended Task Streams (see `paper_analysis.md`)
- **Framework:** custom, prompting-only. Python 3.11+, ~36k lines of Python, hard dependencies are only `matplotlib` and `pyyaml`; model access is optional extras (`boto3` for AWS Bedrock, `openai`, `litellm`, `strands-agents`).
- **Training algorithm:** none. No reinforcement learning, no supervised fine-tuning, no gradients, no local model. Every model call is a provider-hosted inference call; the harness is the only thing that changes.
- **Base models:** Claude Sonnet 4.6 (solver and router), Claude Opus 4.6 (evolver), all at temperature 0; baseline rows additionally use Haiku 4.5, DeepSeek-V3.2, GLM-4.7, Kimi-K2.5.
- **Key innovation:** a git repository *is* the harness store — each branch is a complete regime-specific harness, built by a four-role evolver and selected per task at solve time by a router agent behind a pluggable adaptation interface.

## File Structure Map

```
solve_all_with_evolution.py            Single entry point. Batch loop: route → solve → reveal → evolve.
agent_evolve/
  config.py                            EvolveConfig dataclass; every knob, loaded from experiments/*/configs/*.yaml
  types.py                             Task, Trajectory, Feedback, Observation, BranchInfo, StrategyTree
  contract/workspace.py                AgentWorkspace: the harness on disk (prompts/, skills/, tools/, memory/, infra/)
  contract/manifest.yaml schema        Declares which layers are evolvable per benchmark
  engine/
    loop.py                            Generic solve→observe→snapshot→step→snapshot→reload loop
    observer.py                        Reveal gate: decides which labels the evolver may see this cycle
    versioning.py                      Git wrapper: init, commit+tag, branch create/checkout/merge, worktrees, rollback
    human_interface.py                 Stdin / Telegram channels exposed to the evolver as a tool
    history.py, trial.py               Cycle records; holdout trial runner
  algorithms/
    aevolve/{engine,prompts,tools,gating,egl}.py    Single-agent A-Evolve baseline (the paper's own baseline row)
    navigation/engine.py               NavigationEngine.navigate — the solve-time router
    navigation/adaptation.py           TreeRoutingAdaptation: select=navigate, materialize=git checkout
    navigation/templates/
      inline.py                        One-shot evolver (V1)
      orchestrated.py                  Plan-driven analyst→branches evolver (V2)
      structured_evolution.py          Four phases: analyze → research → build → verify (the paper's evolver)
      structured_navigation.py         Same four phases plus branch creation (the paper's Full System)
      _evolution_workspace.py          Cross-cycle state: task_board.md, research_log.jsonl, architecture.md
      _guardrails.py                   Post-evolution hard constraints, incl. the prompt size cap
      prompts/*.md                     The four role prompts, shared across benchmarks
  protocol/adaptation/                 The solve-time adaptation seam: base protocol + 4 operators + registry
  benchmarks/{polybench,ctf_dojo,futurex}/   Stream loading, chronological ordering, scoring
  agents/{polybench,ctf_dojo,futurex}/       Solvers and per-task sandboxes
experiments/<bench>/configs/*.yaml     One YAML per paper cell (baseline, full_evo, navigation, structured_*)
experiments/<bench>/seed/              The seed harness shipped before cycle 1
scripts/{poly,ctf_dojo,futurex}_hypothesis.sh   Named launchers: H0, H1, H4, H4_multi, H4_multi_nav
evaluations/analysis_poly/             Scripts regenerating the PolyBench figures from results.jsonl + the SQLite snapshot
```

## Component Inventory

### 1. Entry points

One real entry point: `solve_all_with_evolution.py:132 main()`. Everything is selected by config plus a handful of flags; the launcher scripts map paper cell names onto config files, e.g. `scripts/poly_hypothesis.sh:118` (`H0` → `baseline`), `:128` (`H1` → `full_evo`), `:150` (`H4_multi` → `structured_evo`), `:155` (`H4_multi_nav` → `structured_nav`).

The batch loop is `solve_all_with_evolution.py:386` onward, in exactly the paper's order: route every task in the batch (`:407–431`), group tasks by selection key and materialize one workspace per group (`:439–466`), submit to a thread pool (`solve_workers` = 24/8/10), return to `main` (`:506`), then run one evolution cycle unless suppressed (`:767`). Resumption is by `results.jsonl`: `_load_done_ids` (`:67`) skips completed task ids, so a killed run continues at the batch boundary.

### 2. Special token / action handling

Absent. There are no action tokens, no generation interception, no parser over special markers. The solver is an ordinary tool-calling loop; the evolver edits files through a sandboxed bash tool. The only text parsing of consequence is `_strip_preamble` (`structured_evolution.py:46`), which discards an LLM's conversational lead-in before a required `## Failure Patterns` heading, and `_parse_json_blocks` (`:116`).

### 3. Environment and tool interaction

The solver runs against a per-task Docker sandbox for CTF-Dojo and FutureX (`agents/*/sandbox.py`, `_sandbox_image.py`); PolyBench is pure reasoning with no container. Network policy is per-benchmark config (`sandbox_network: none|bridge`). Evolver-built tools are ordinary Python files in `tools/` with a YAML registry; the Verifier subprocess-executes each one and deletes the ones that fail (`_guardrails.py:75 verify_tools`).

FutureX's temporal-retrieval contract is the most interesting piece of environment work: `tools/htmldate_search.py` fetches Wikipedia through the revision API at the latest revision before the task's cutoff, filters DuckDuckGo results by extracted publication date, and passes live command output through a language-model temporal filter that returns `CLEAN` or replaces post-cutoff values with `[REDACTED]` (paper Appendix A.3).

### 4. Token masking

Absent — nothing is trained, so there is no loss to mask. The analogous mechanism is *information* masking, and it is real: see 6 below.

### 5. Reward / scoring

Scoring lives in the benchmark adapters behind one method, `BenchmarkAdapter.evaluate(task, trajectory) -> Feedback` (`benchmarks/base.py:31`). PolyBench (`benchmarks/polybench/polybench.py:135`) resolves the market's ground truth, simulates an order-book fill at two budget levels, and returns `score = 1.0 if is_correct else 0.0` (`:199`) with the confidence-weighted return carried in the feedback payload for ranking. CTF-Dojo verifies a submitted flag by SHA-256 comparison against the official flag; FutureX applies the benchmark's own pass criterion.

There is no reward shaping, no cost term, no penalty, and no aggregate objective anywhere: the batch mean of `feedback.score` is computed only for logging and for the plateau stop (`engine/loop.py:102`).

### 6. Evolution loop and the reveal gate

`engine/loop.py:68 run()` is the generic cycle: solve the batch, collect observations, commit a `pre-evo-N` tag, call `engine.step()`, commit an `evo-N` tag, reload the agent from disk, check for a plateau. The git tag pair per cycle means the repository holds a complete, replayable harness-evolution log.

The reveal gate is `engine/observer.py:43 _label_revealed`: a task with no resolution date is never revealed; a task with one is revealed exactly when its resolution timestamp is at or before the batch watermark. Two channels are reported each cycle — in-batch reveals, and tasks from *earlier* batches whose labels have only now resolved, appended to `revealed_supplement.jsonl`. This is score-before-update implemented honestly, with the delay the real world imposes rather than a simulated one.

A second, stronger information barrier is `evolver_trajectory_mode` (`config.py:66`, default `"index"`): the evolver prompt carries only a compact index of recent trajectories (task id, batch, cycle age, turn count, input preview, file paths), pulls full trajectories on demand from a read-only mount, and *never* sees success, score, feedback, or judge claims. The legacy `"inline"` mode that inlines ground truth is kept only for parity testing and is marked "unsafe for real evolution runs."

### 7. Solve-time adaptation (the paper's distinctive seam)

A two-method protocol in `protocol/adaptation/base.py:44`:

```python
def select(self, store_path: Path, task: Task) -> str:
    """Pick a selection key for ``task`` (read-only, no side effects)."""

def materialize(self, store_path: Path, key: str, workspace: AgentWorkspace) -> None:
    """Realize the harness for ``key`` by mutating ``workspace`` in place."""
```

The docstring states two invariants explicitly: **no feedback** ("Neither method receives reward/score/success. The selector conditions on the task only — never on outcomes"), and **capacity** (`materialize` must leave the workspace within the budget). Four operators are registered (`registry.py:102`):

| Name | Granularity | Implementation |
|---|---|---|
| `whole_store` | full harness | `base.py::WholeStoreAdaptation` — no selection, no materialization |
| `tree_routing` | one git branch | `navigation/adaptation.py:32` — the paper default |
| `retrieval` | per-task top-k items | `operators.py::RetrievalAdaptation` — dense embedding search over a skill/tool/memory catalog |
| `agentic_filter` | per-task LLM-chosen subset | `operators.py::AgenticFilterAdaptation` — retrieve 16 candidates, let a model keep 8 |

`tree_routing.select` (`navigation/adaptation.py:47`) builds a routing context from task id, category/event/challenge/year metadata, and the task text, and delegates to `NavigationEngine.navigate` (`navigation/engine.py:138`), which reads each viable branch's system prompt, skills, and tools out of git, asks the router for `{branch, confidence}` at 256 max tokens, and falls back to `main` whenever confidence is below `branch_confidence_threshold` (0.7 in every config). `materialize` (`:59`) is a `git checkout`; a failed checkout increments `failed_checkouts` on the branch and falls back to `main`.

The retrieval operators never touch the workspace: they record a per-task filter that the worker applies at harness-assembly time (`harness_filter.py`), and they rebuild their index from the *current* evolved store at the start of every batch (`operators.py::prepare`), because the evolver mutates skills and tools every cycle.

### 8. The four-role evolver

`structured_evolution.py:254 execute()` runs `_phase_analyze` (`:337`), `_phase_research` (`:464`, three parallel agents via `ThreadPoolExecutor`), and `_phase_build_verify` (`:743`, `_verify` at `:861`, three retries with `vc.rollback_to_tag(pre_build_tag)` between attempts at `:831` and `:847`). `structured_navigation.py` is the same four phases plus branch creation. Cross-cycle state is file-backed in `_evolution_workspace.py`: `task_board.md` with validation, `research_log.jsonl` with pass/fail verdicts per tested hypothesis, `architecture.md`, and an insights file.

The human hooks are `_hitl_credentials` (`:610`) and `_hitl_task_board` (`:724`), both gated by `structured_evolution.hitl_enabled` (false in every config except the dedicated human-steering run) and both routed through `engine/human_interface.py`.

### 9. Data pipeline

Streams are loaded in chronological order by the benchmark adapters. PolyBench reads a SQLite snapshot of Polymarket markets (`data/download_data.py`, `benchmarks/polybench/polybench.py:266 _load_from_db`), storing a release timestamp and a resolution timestamp per market. FutureX loads from FutureX-Past with per-task cutoff dates (`benchmarks/futurex/data_loader.py`). CTF-Dojo loads `pwncollege/ctf-archive` challenges ordered by competition year. Results stream to `results/<cell>/results.jsonl`, one JSON line per task; the figure scripts in `evaluations/analysis_poly/` read that file plus the SQLite snapshot and nothing else.

### 10. Infrastructure

No GPU code anywhere — no FSDP, DeepSpeed, DDP, tensor parallelism, or device management. Parallelism is thread pools: `solve_workers` (24/8/10) for task solving, three threads for the research phase. A batch deadline of `rounds × task_timeout × 1.5` bounds each batch, and tasks still running past it are recorded as `HUNG` rather than retried (`solve_all_with_evolution.py:565`).

### 11. Evaluation

`evaluations/analysis_poly/` regenerates the paper's PolyBench figures — learning curve, Pareto frontier, overview, opening comparison — from `results.jsonl`. CTF-Dojo and FutureX report their own official Pass@1 through their adapters. No aggregate results files, logs, or evolved workspaces ship with the repo; only the code that would produce them.

## What the code says that the paper does not

Four findings that only reading the code surfaces, all of them load-bearing for how we position this system.

**1. The commit gate is disabled in every paper run.** `aevolve/gating.py:20 GatingStrategy` splits tasks and rejects a mutation whose holdout average falls below a threshold — inherited from A-Evolve. Its defaults are `holdout_ratio=0.2, min_score_threshold=0.0` (`:27`), the config default is already `holdout_ratio: 0.0` (`config.py:28`, commented "V2 benchmarks pre-split"), and **every** config under `experiments/*/configs/` sets `holdout_ratio: 0.0` explicitly. `GatingStrategy` is referenced nowhere outside its own file. In the paper's runs, no harness mutation is ever validated against held-out tasks before being committed: the Verifier checks that the built artifacts *run*, not that the harness still *scores*.

**2. The harness tree only grows.** `config.py:79–80` declares `promotion_threshold: 0.15  # cross-region improvement to merge branch` and `staleness_window: 5  # prune branches unused for N cycles`. Neither name is read by any code in the repository, and `VersionControl.merge_branch` (`versioning.py:296`) has no caller. Branch creation exists; branch merging and branch retirement do not. Nothing removes content from any branch either.

**3. The entire defense against harness bloat is one constant.** `_guardrails.py:25 MAX_PROMPT_CHARS = 10_000`, applied by `cap_prompt_size` as a truncation of `prompts/system.md` after every evolution cycle. The paper's Figure 1 shows an ungoverned run's prompt reaching 68 KB; the released system caps the system prompt at 10 KB and isolates the rest into branches. Skills, tools, and memory have no size bound at all. The one visible attempt at judgment is a config comment rather than code: `experiments/polybench/configs/structured_navigation_evo.yaml` sets `evolve_memory: false` with the note that "memory was accreting ~30k of unbounded single-market learnings never loaded usefully" — the fix for memory bloat was to turn the memory layer off.

**4. The EGL trigger described in the paper is not the trigger in the code.** Paper Table 3 lists "EGL threshold 0.05 / EGL window 3" and defines EGL as "the Expected-Gain-from-Learning trigger that gates whether a cycle runs." In the repository, `aevolve/egl.py:1` defines EGL as "Evolutionary Generality Loss", $\mathrm{EGL} = 1000 \cdot (\text{new skills} / \text{tasks solved})$ (`:14`), with an `is_converged` helper (`:20`). Neither function is called anywhere. `egl_threshold` is declared at `config.py:72` and read by no code. What actually stops a run is `engine/loop.py:33 _is_score_converged`, a plain batch-score plateau check that borrows `egl_window` for its window length and hardcodes `epsilon=0.01`. Nothing gates whether a cycle runs; cycles run unconditionally unless `evo_trigger_threshold` is set negative to produce a no-evolution baseline (`solve_all_with_evolution.py:767`).

## Relevance to this project

### Components we can directly reuse

- **The three stream adapters** — `benchmarks/{polybench,ctf_dojo,futurex}/`. Heterogeneous, non-stationary, strictly chronological, with delayed labels already implemented: 5,075 / 261 / 503 tasks. This is the task-stream infrastructure we would otherwise build.
- **The reveal gate** — `engine/observer.py:43` plus the `revealed_supplement.jsonl` channel. A working score-before-update implementation that also handles labels arriving for *earlier* batches, which our setting explicitly allows ("labels may arrive late").
- **The adaptation seam** — `protocol/adaptation/base.py`. Two methods, four existing implementations, an explicit no-feedback invariant. A new selection policy is a sibling class and one registry line.
- **The git-backed workspace and its per-cycle tags** — `engine/versioning.py`, tags `pre-evo-N` / `evo-N` per cycle. Every run leaves a replayable sequence of harness states paired with the batch score that preceded each edit. That is the shape of the training data a state-level estimate of future stream performance would regress on, and it is produced for free by any run of this code.
- **The seed harnesses** — `seed_workspaces/`. 27, 24, and 114 prompt lines, zero skills, zero tools, zero memory entries (matching paper Table 4, verified by `wc -l`). A clean floor: everything above it is agent-built, so measured differences are attributable to the editing policy rather than to hand-engineering.

### Components we need to modify

- **The proposer's objective.** `structured_evolution.py` proposes from the batch just solved and commits whatever the Verifier can execute. Adding a future-stream term means a scoring step between Build and commit, which the four-phase structure has no slot for — the Verify phase tests artifacts, not harness quality.
- **The commit path.** There is no gate to modify; one has to be added. `GatingStrategy` is the nearest hook but it is dead code with a threshold of 0.0, and its holdout is drawn from the *current* split, not from previously passed tasks — it would have to be rewritten against an anchor set rather than adapted.
- **The tree lifecycle.** Merging and retirement are named in the config and absent from the code. Any move that relocates content across layers, or any promotion that replaces several narrow entries with one general entry, has no implementation to start from.
- **Layer coverage.** `contract/workspace.py` covers prompts, fragments, skills, tools, memory, and infrastructure. Middleware, sub-agent configuration, and model weights — the deeper layers in our setting — have no representation; `manifest.yaml::evolvable_layers` is the place they would be declared.

### Components that do not apply

- The `agents/` directory beyond the three paper benchmarks (`arc`, `swe`, `terminal`, `mcp`, `skillbench`) is other work carried along on this branch; `benchmarks/cl_bench.py` (1,931 lines) and `benchmarks/skill_bench.py` belong to sibling branches of the same lab and are unused by the paper's launchers.
- `algorithms/aevolve/` is the single-agent baseline, useful as a reference implementation of exactly the greedy behavior we argue against, not as a component.
- All plotting under `evaluations/analysis_poly/` is PolyBench-specific and paper-figure-specific.

### Key code worth studying

| What | Where |
|---|---|
| The adaptation protocol and its two invariants | `protocol/adaptation/base.py:1–60` |
| Router: branch summaries in, `{branch, confidence}` out, confidence fallback | `algorithms/navigation/engine.py:138–200` |
| Reveal gate and the two reveal channels | `engine/observer.py:43–110` |
| Per-cycle snapshot pair around the evolver step | `engine/loop.py:110–135` |
| Four-phase evolver with rollback-on-verify-failure | `algorithms/navigation/templates/structured_evolution.py:254–336, 743–890` |
| The batch loop: route, group, materialize, solve, evolve | `solve_all_with_evolution.py:386–520, 760–790` |
| The dead gate, the dead thresholds, the live truncation constant | `algorithms/aevolve/gating.py:20–68`, `config.py:72–80`, `algorithms/navigation/templates/_guardrails.py:25` |
