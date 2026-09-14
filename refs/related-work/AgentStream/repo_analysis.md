# Repo Analysis: AgentStream

**Path:** `refs/related-work/AgentStream/repo/` · **Source:** https://github.com/Jasper-Yan/AgentStream (mirrored at https://github.com/microsoft/Sico under `labs/AgentStream`) · **Analyzed:** 2026-09-14 (commit `dc635f7`, dated 2026-08-11)

## Overview

- **Paper:** AgentStream: How Well Do Self-Evolving LLM Agents Perform Under Streaming Tasks? (see `paper_analysis.md`)
- **Framework:** custom, prompting-only. The repo is the AgentStream streaming harness (six runner scripts plus a task-ordering module) laid on top of a bundled vendored copy of Microsoft's **Exgentic** agent-evaluation framework (`exgentic/`, 4.9 MB, MIT inside an Apache-2.0 outer repo). No training code of any kind.
- **RL algorithm:** none. Nothing in the repo computes a gradient, an advantage, or a reward. Self-evolution is entirely a language model rewriting natural-language state between episodes.
- **Base model:** none trained. Three proprietary APIs are called through LiteLLM — `openai/gpt-5.4`, Gemini 3.1 Pro, Claude Opus 4.7 — and the same string is used for both the acting model and the evolver model (`evolver_model=args.model` in every runner).
- **Key innovation:** a task-stream driver that keeps one persistent evolution state alive across a 300-task mixed-benchmark stream and writes a per-task metrics record (score, running cumulative accuracy, cost, steps, state size in characters, skill count), instantiated in three stream shapes — Isolated, Sequential, Interleaved.

## File Structure Map

```
repo/
├── README.md                          paper abstract, install (uv sync), per-method run instructions
├── LICENSE                            Apache-2.0
├── figs/evaluation_compare.png        the paper's Figure 1
└── exgentic/                          vendored Exgentic framework (MIT, pyproject name "exgentic")
    ├── pyproject.toml                 requires-python >=3.11; litellm>=1.65, mcp>=1.24, pydantic>=2.9
    ├── scripts/                       ← the AgentStream layer (the paper's own code)
    │   ├── utils/task_ordering.py     stream construction for all three scenarios (175 lines)
    │   ├── harness/run_experiment.py  Harness stream driver (386 lines) + .sh launcher
    │   ├── ace/run_experiment.py      same driver, ACE agent
    │   ├── a_mem/run_experiment.py    same driver, A-Mem agent
    │   ├── reasoning_bank/run_experiment.py
    │   ├── autoskill/run_experiment.py
    │   └── litellm/run_baseline.py    the no-state control (vanilla arm)
    └── src/exgentic/
        ├── core/                      agent.py, agent_instance.py, benchmark.py, session.py, evaluator.py
        ├── agents/
        │   ├── harness/               ← closest analogue to our object
        │   │   ├── harness_store.py   (320) prompt + memory + skill library, versioned
        │   │   ├── harness_instance.py(579) start() injects, close() evolves
        │   │   ├── evolver.py         (382) multi-turn tool-calling editor
        │   │   ├── retriever.py       (93)  MiniLM cosine over skill descriptions
        │   │   └── prompts/           evolver.py (edit tools + constraints), inject.py
        │   ├── ace/                   playbook_store.py, bulletpoint_analyzer.py, ace_instance.py (775)
        │   ├── a_mem/                 memory_note.py, memory_store.py, a_mem_instance.py (726)
        │   ├── reasoning_bank/        evaluator.py, induce_memory.py, rb_instance.py (552)
        │   ├── autoskill/             skill_extraction/maintenance/retrieval, autoskill_instance.py (636)
        │   └── litellm_tool_calling/  stateless baseline agent
        └── benchmarks/                appworld, bfcl, browsecompplus, hle, swebench, tau2 (+ gsm8k, hotpotqa)
```

## Component Inventory

### 1. Entry points

Six parallel runners, one per arm, with an identical command-line surface (`scripts/harness/run_experiment.py:370-386`):

| Flag | Values | Note |
|---|---|---|
| `--mode` | `isolated` \| `sequential` \| `interleaved` | the stream shape; required |
| `--seed` | int | required; permutes order only, not task selection |
| `--num-tasks` | int, default 50 | per benchmark |
| `--model` | default `openai/gpt-5.4` | also used as the evolver model |
| `--benchmarks` | comma-separated slugs | the paper uses `hle,bfcl,browsecompplus,appworld,swebench,tau2` |
| `--output-dir` | path | required |
| `--max-tokens`, `--reasoning-effort` | optional | how the paper sets Claude to high and the others to medium |

`scripts/harness/run_experiment.sh` is the paper's launcher: `SEED=44 NUM_TASKS=50 MODEL="openai/gpt-5.4" MODE="sequential"`, looping one benchmark at a time under `isolated` and passing all six at once otherwise. The vanilla control is `scripts/litellm/run_baseline.py`, the same driver with a stateless agent and no `--mode`.

`BENCHMARK_REGISTRY` (`run_experiment.py:29-106`) pins every benchmark configuration in code, not in a config file: `browsecompplus` with a faiss index and `eval_model_id: openai/gpt-5.4`; `swebench` on SWE-bench_Verified; `appworld` on `test_challenge` with tool shortlisting capped at 30 tools; `bfcl` on `multi_turn_base`; `tau2` on `telecom` with `user_simulator_model: openai/gpt-5.4`; `hle` with `judge_model` gpt-5.4.

### 2. Stream construction — `scripts/utils/task_ordering.py`

The whole of the paper's Section 3.2 is this one 175-line file. `get_unified_task_order(configs, num_tasks, seed, mode)` (line 32) does three things:

```python
_SELECTION_SEED = 42
per_bm_tasks = _select_tasks(benchmark_configs, num_tasks_per_benchmark, _SELECTION_SEED)
```

Task **selection** is hard-pinned to seed 42 regardless of `--seed`, so all runs at all seeds see the same 300 tasks; only the **order** varies (line 67 reshuffles per benchmark via `_derive_seed(master_seed, slug)`, an md5 of seed+slug). Interleaving preserves within-benchmark order exactly (`_interleave_preserving_order`, line 126): it keeps one deque per benchmark and at each step pops from a uniformly-chosen non-empty queue, so the relative order inside each benchmark is identical across Isolated, Sequential, and Interleaved. This is the mechanism behind the paper's claim that scenario comparisons are not confounded by ordering — and it is the piece most worth copying wholesale.

### 3. Evolution state — `agents/harness/harness_store.py`

A process-global registry of stores keyed by `store_id` (`harness_isolated_<bm>`, `harness_sequential_global`, `harness_interleaved_global`), which is exactly how one state is kept alive across a 300-task stream while agent and benchmark objects are re-instantiated per task. Three record types:

```python
HarnessSkill(name, description, body, last_used_session, created_session)
VersionEntry(version, session_id, session_count, ops_summary, timestamp, system_prompt, memory, skills)
LearningEvent(...)
```

`commit_version` (line 205) snapshots the **full** prompt, memory, and skill list at every version, so a run produces a complete harness-evolution log — the pretraining corpus our critic wants, already in the shape we would need. `snapshot`/`rollback` (189/198) exist only as crash protection around the evolver, not as a quality gate.

There is **no capacity cap, no pruning, and no eviction** anywhere in the store. Skills accumulate without bound, which is why the paper's Table 9 reports 463 skills after 300 tasks on Claude Opus 4.7.

### 4. The update rule — `harness_instance.py` and `harness/evolver.py`

`start()` (line 123) retrieves the top 3 skills by embedding similarity and assembles the system message from `store.system_prompt`, `store.memory`, and the retrieved skill bodies. `close()` (line 285) runs the evolver after the episode:

```python
ops_applied, ops_summary = run_evolver(
    store=store,
    task=self.task if self.task else "",
    injected_skill_names=self._injected_skill_names,
    trajectory=trajectory,
    llm_call=self._llm_call_simple,
    evolver_model=self.evolver_model,
    embedding_model=self.embedding_model,
)
```

**No score, no reward, and no gate is passed to `run_evolver`** (signature at `evolver.py:339-347`). Whatever the evolver writes is applied; the only rollback path is an exception. Scoring happens outside the agent entirely, in the benchmark's evaluator, and is never routed back. This is score-before-update with the horizon set to zero, confirmed at the level of a function signature rather than inferred from prose.

The same holds for the other four arms. ACE's `close()` (`ace/ace_instance.py:305-350`) runs reflection then a curator every `curator_frequency` sessions and then calls `store.record_learning(..., was_correct_before=False, was_correct_after=False, ...)` — the correctness fields are hardcoded `False` because no score is available. ReasoningBank (`rb_instance.py:267`) calls `_run_post_session_learning`, which asks `TrajectoryEvaluator` (`reasoning_bank/evaluator.py`) to prompt an LLM for `"success"` or `"failure"`; that self-judgment is the closest any arm comes to a reward, and it is generated by the same model that produced the trajectory.

The evolver's action space is a tool-calling loop (`agents/harness/prompts/evolver.py`): read tools `read_prompt`, `read_memory`, `list_skills`, `read_skill`; write tools `edit_prompt`, `edit_memory`, `add_skill`, `edit_skill`, `delete_skill`; constraints "At most 1 edit_prompt call per session. At most 1 edit_memory call per session. No limit on skill operations." The user message supplies only Task, Skills Injected, and Session Trajectory.

### 5. Retrieval — `agents/harness/retriever.py`

`sentence-transformers` `all-MiniLM-L6-v2` embeddings of skill **descriptions** (not bodies), plain cosine similarity, `top_k` default 5 with the harness instance overriding to `top_k_skills: int = 3`. No reranking, no recency weighting, no usage weighting — `last_used_session` is recorded but never read by the retriever. This retrieval gate is the entire mechanistic content of the paper's context-integrated / retrieval-based distinction.

### 6. Metrics — `run_experiment.py:88-155`

Per task, appended to `online_metrics.jsonl`:

```python
record = {
    "session_index": session_index, "seed": seed, "mode": mode,
    "agent": "harness", "model": model, "benchmark_slug": bm_slug,
    "task_id": task_id, "score": score,
    "cumulative_avg_score": sum(all_scores) / len(all_scores),
    "benchmark_cumulative_avg_score": (sum(bm_scores[bm_slug]) / len(bm_scores[bm_slug])),
    "steps": sr.steps, "action_count": sr.action_count,
    "agent_cost": sr.agent_cost, "input_tokens": input_tokens, "output_tokens": output_tokens,
    "memory_tokens": memory_tokens, "skill_count": skill_count,
    "execution_time": sr.execution_time, "status": ..., "timestamp": ...,
}
```

`cumulative_avg_score` is the y-axis of every plot in Appendix D. `memory_tokens` is a character-count proxy for harness size (`get_memory_tokens`, line 88):

```python
total_chars = len(store.system_prompt) + len(store.memory)
for skill in store.list_skills():
    total_chars += len(skill.description) + len(skill.body)
mem_tokens = total_chars // 4
```

Cost accounting comes from LiteLLM via `sr.agent_cost`; there is no separate cost model.

### 7. Execution and resumption

`max_workers=1` everywhere — tasks are strictly serial, as a stream must be. Isolated and Sequential iterate `group_by_benchmark(task_order)` and hand a whole benchmark's task list to `evaluate()`; Interleaved (line 304) loops one task at a time, re-instantiating benchmark and agent per task while the store persists through the class-level registry. Both paths checkpoint the store to `harness_<store_id>.json` after each unit and truncate `online_metrics.jsonl` back to the checkpointed session count on restart (lines 193-235), so a 300-task API run survives crashes. This resumption logic is worth reading before we write our own long-stream driver.

### 8. Components absent

- **Token masking, RL training loop, SFT pipeline, multi-GPU infrastructure** — all absent. Nothing is trained; there is no PyTorch dependency.
- **Reward function** — absent in the RL sense. Scores exist only as benchmark evaluator outputs consumed by the metrics writer; no code path feeds a score into an evolution decision.
- **Special action tokens** — absent; tool calls go through the provider's native function-calling API via LiteLLM.
- **Released results** — absent. The repo ships zero experiment outputs: `figs/` holds only `evaluation_compare.png`, and the only `.jsonl` files are HTTP recordings under `exgentic/tests/benchmarks/recordings/{swebench,appworld,tau2,browsecompplus}/`. No evolved stores, no `online_metrics.jsonl` from any paper run, no per-task scores. Every number in the paper would have to be regenerated against three paid APIs.
- **Aggregation and plotting code** — absent, despite an `analysis` extra in `pyproject.toml` declaring matplotlib, numpy, and pandas. Tables 1-13 and Figures 5-19 are produced by scripts that were not released.

## Relevance to this project

### Components we can directly reuse

- `exgentic/scripts/utils/task_ordering.py` — stream construction for all three scenarios, including the selection-seed / order-seed separation and the order-preserving interleave. Copy nearly as-is; it is the cleanest available definition of a heterogeneous non-stationary task stream with controlled ordering.
- The per-task metrics record and `get_memory_tokens` in `scripts/*/run_experiment.py` — gives us `cumulative_avg_score` (the decline readout) and a harness-size series in the same file, which is precisely what the equal-harness-size control in our falsifier needs.
- `scripts/litellm/run_baseline.py` — a no-state control measured on the identical stream. Our stream-performance integral needs exactly this reference arm.
- The six benchmark adapters under `src/exgentic/benchmarks/` with the paper's pinned splits, so our numbers are comparable to its Tables 1-6 without re-deriving a task set.
- `HarnessStore.commit_version` version history — a ready-made harness-evolution log format; `save_checkpoint`/`load_checkpoint` give us serialized harness states to feed a critic.

### Components we need to modify

- **The evolver call site** (`harness_instance.py:305`) is where our work goes. It currently takes no score; we need it to take the batch reward and a plasticity estimate, and to choose among several sampled candidate edits rather than applying one unconditionally.
- **The store needs an anchor set.** `HarnessStore` records `LearningEvent`s but never re-runs a task. Retention measurement requires keeping previously-passed tasks and re-checking them at commit, which is new code, not a modification.
- **The stream driver assumes single-candidate evolution.** Sampling a group of candidate edits, applying each, and scoring each on the next batch needs the store's `snapshot`/`rollback` promoted from an exception handler to the main control flow.
- **Retrieval is description-only cosine with a fixed top-3.** If we treat retrieved-skill selection as part of the harness, it needs to be either frozen as a control or exposed as an edit target.
- **The harness layers are three, not eight.** System prompt, long-term memory, and skills are covered; tool descriptions, tool implementations, middleware, and sub-agent configuration are not editable here, and model weights are out of reach behind proprietary APIs. Extending the evolver's tool list is the natural path.

### Components that don't apply

- All training infrastructure the protocol asks about (token masking, RL loop, SFT, multi-GPU) — this is a prompting-only system with no trainable parameters.
- The three-model capability sweep — our question is about the objective, not about which API tier benefits; one model held fixed is the right control.
- LiteLLM cost accounting — cost as an objective term is explicitly out of scope for us (`docs/framing.md`, Setting).
- The released repository as a source of *data* — there is none, so the critic-pretraining corpus has to be generated by running this code, not downloaded.

### Key code snippets worth studying

| Location | Why |
|---|---|
| `exgentic/scripts/utils/task_ordering.py:32-145` | the three stream shapes in one function; the selection-seed / order-seed split |
| `exgentic/src/exgentic/agents/harness/harness_instance.py:285-330` | `close()` — the update applied with no score in scope, i.e. the greedy case at code level |
| `exgentic/src/exgentic/agents/harness/harness_store.py:189-240` | `snapshot`, `rollback`, `commit_version`, `record_learning` — the hooks a retention gate would attach to |
| `exgentic/src/exgentic/agents/harness/prompts/evolver.py` | the edit action space and its per-session limits, verbatim |
| `exgentic/scripts/harness/run_experiment.py:88-155` | harness-size proxy and the per-task metrics record |
| `exgentic/scripts/harness/run_experiment.py:193-235, 300-352` | checkpoint-and-resume for a 300-task API stream, and the one-task-at-a-time interleaved loop |
| `exgentic/src/exgentic/agents/reasoning_bank/evaluator.py` | the self-generated success/failure label — the only reward-shaped signal in the repo |
| `exgentic/src/exgentic/core/agent_instance.py:14-82` | the `start`/`react`/`close` interface our own harness agent would implement to reuse the benchmarks |
