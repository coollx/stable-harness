# Repo Analysis: ReMe

**Path:** `refs/related-work/ReMe/repo/` · **Source:** https://github.com/agentscope-ai/ReMe · **Analyzed:** 2026-09-15 (tag `v0.2.0.6`, commit `554eec1c`, dated 2026-01-07)

## Overview

- **Paper:** *Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution* (see `paper_analysis.md`)
- **Framework:** custom, built on `flowllm` (`pyproject.toml:36`, `flowllm[reme]>=0.2.0.10`). Every unit of work is an operation class; pipelines are written as expressions in one YAML file, with `>>` for sequential and `|` for parallel composition. Served as an HTTP or Model Context Protocol service on port 8002. Apache 2.0 (`LICENSE`, 201 lines). 262 Python files, about 34.7k lines, 33 MB on disk.
- **RL Algorithm:** none. No gradients anywhere — no trainer, no optimizer, no reward model. The whole system is prompting plus a vector store plus two integer counters.
- **Base Model:** provider-agnostic. The benchmark drivers default to `qwen3-8b` (`cookbook/bfcl/bfcl_agent.py:63`); the quickstart runs the memory service on `qwen-max-2025-01-25` with embeddings from `text-embedding-v4` and a local vector-store backend (`docs/cookbook/bfcl/quickstart.md`).
- **Key Innovation:** two integers per stored experience — `freq` (times retrieved) and `utility` (times a retrieval coincided with a later task succeeding) — and a sweep that deletes any experience whose ratio falls below a threshold once it has been retrieved enough times.

**Version warning, read this first.** The default clone does not contain the paper. `main` at `16269c9a` (2026-09-15) is a rewritten product — a local-first personal knowledge base with a studio interface, a plugin system, and packaged skills — whose top level is `reme/`, `reme_studio/`, `plugins/`, `skills/`, `integrations/`. There is no `cookbook/` and no `reme_ai/` on `main`, so the BFCL-V3 driver, the AppWorld driver, the task-memory operations, and the utility-based deletion are all absent from it. The paper's code is the repository's single tag, `v0.2.0.6`. This clone is checked out at that tag (detached head) and every file:line reference below is against it. To reproduce: `git fetch --depth 1 origin tag v0.2.0.6 && git checkout v0.2.0.6`.

## File Structure Map

```
reme_ai/
  main.py                        `reme` console script (pyproject.toml:79); starts the service
  config/default.yaml            THE FILE. Every pipeline is one line here (see below)
  schema/memory.py               BaseMemory and TaskMemory; to_vector_node (:106-116)
  summary/task/                  <- EXPERIENCE ACQUISITION
    trajectory_preprocess_op.py  splits trajectories into success / failure by score (:49-54)
    success_extraction_op.py     + success_extraction_prompt.yaml
    failure_extraction_op.py     + failure_extraction_prompt.yaml
    comparative_extraction_op.py + comparative_extraction_prompt.yaml
    memory_validation_op.py      language-model judge on each candidate (:93-100)
    memory_deduplication_op.py   embedding similarity against store and batch (:54, :119)
  retrieve/task/                 <- EXPERIENCE REUSE
    build_query_op.py, merge_memory_op.py
    rerank_memory_op.py          + rerank_memory_prompt.yaml (:41-46)
    rewrite_memory_op.py         + rewrite_memory_prompt.yaml (:75)
  vector_store/                  <- EXPERIENCE REFINEMENT
    recall_vector_store_op.py    top-k similarity search
    update_memory_freq_op.py     freq += 1 on every retrieved item (:53)
    update_memory_utility_op.py  utility += 1, but only if update_utility is set (:45,:51,:62)
    delete_memory_op.py          THE GATE, 10 lines (:47-56)
    update_vector_store_op.py    write-back
  service/task_memory_service.py the memory service used by the drivers
  retrieve/personal/, summary/personal/, summary/tool/, summary/working/, mem_agent/, mem_tool/
                                 other memory types, not used by the paper
cookbook/
  bfcl/
    run_bfcl.py                  entry point; 4 ray actors, round-robin shard (:56), main (:89)
    bfcl_agent.py                THE DRIVER. execute (:591); the refinement block (:647-661)
    init_exp_pool.py             builds the initial pool; keeps best+worst per task (:66-82)
    init_task_memory_pool.py     posts trajectories to summary_task_memory
    split_into_trainval.py       random.shuffle then ratio split (:9)
    run_exp_statistic.py         best@k (:9-29) and pass@k (:33-41)
    local_file_to_library.py, bfcl_utils.py
  appworld/
    run_appworld.py, appworld_react_agent.py   same shape; execute (:170), gate (:208-213)
    prompt.py, run_exp_statistic.py
  frozenlake/, simple_demo/, tool_memory/, working_memory/
docs/
  cookbook/bfcl/quickstart.md    the four-step reproduction recipe
  library/paper_data/task/*.jsonl   THE RELEASED POOLS — six files, all freq=0 utility=0
  library/{bfcl_v3,appworld,research_plan,research_tips}.jsonl
```

## Component Inventory

### The four pipelines, verbatim from `reme_ai/config/default.yaml`

Everything the paper describes is these four lines. They are worth reading as a unit because they are the complete specification of the system's behavior.

```yaml
# :17
retrieve_task_memory:
  flow_content: BuildQueryOp() >> RecallVectorStoreOp() >> RerankMemoryOp(enable_llm_rerank=False, enable_score_filter=False, top_k=5) >> RewriteMemoryOp(enable_llm_rewrite=False)
# :26
summary_task_memory:
  flow_content: TrajectoryPreprocessOp(success_threshold=1.0) >> (SuccessExtractionOp()|FailureExtractionOp()|ComparativeExtractionOp(enable_soft_comparison=True)) >> MemoryValidationOp(validation_threshold=0.5) >> MemoryDeduplicationOp() >> UpdateVectorStoreOp()
# :117
record_task_memory:
  flow_content: UpdateMemoryFreqOp() >> UpdateMemoryUtilityOp() >> UpdateVectorStoreOp()
# :134
delete_task_memory:
  flow_content: DeleteMemoryOp() >> UpdateVectorStoreOp()
```

Two shipped defaults differ from the paper. `RerankMemoryOp(enable_llm_rerank=False)` and `RewriteMemoryOp(enable_llm_rewrite=False)` disable both optional reuse modules that Table 4 credits with +1.83 Avg@4 / +6.01 Pass@4; their own class defaults are `True` (`rerank_memory_op.py:41`, `rewrite_memory_op.py:75`), so the disabling is a deliberate choice in the shipped configuration. And `MemoryValidationOp(validation_threshold=0.5)` rejects candidates scoring below 0.5, where the paper's validation prompt (Table 13) says "Mark as invalid if score is below 0.3"; the code's own fallback default is also 0.5 (`memory_validation_op.py:97`).

### The gate: `reme_ai/vector_store/delete_memory_op.py:47-56`

This is the entire acceptance-and-removal mechanism of the system, and the only place where evidence from a later task changes the stored harness state:

```python
nodes: Iterable[VectorNode] = self.vector_store.iter_workspace_nodes(workspace_id=workspace_id)

deleted_memory_ids = []
for node in nodes:
    freq = node.metadata.get("freq", 0)
    utility = node.metadata.get("utility", 0)
    if freq >= freq_threshold:
        if utility * 1.0 / freq < utility_threshold:
            deleted_memory_ids.append(node.unique_id)
```

This is equation (1) of the paper with $\alpha = $ `freq_threshold` and $\beta = $ `utility_threshold`. The driver defaults are `freq_threshold: int = 5` and `utility_threshold: float = 0.5` (`cookbook/bfcl/bfcl_agent.py:73-74`), matching the paper. Note the comparison is strict `<` here while the paper writes $\le \beta$ — an entry sitting exactly at 0.5 survives in code and is deleted in the paper's notation.

The two counters that feed it:

- `update_memory_freq_op.py:53` — `metadata["freq"] = metadata.get("freq", 0) + 1`, applied unconditionally to every retrieved item.
- `update_memory_utility_op.py:45,51,62` — reads `update_utility` from the request, returns early if it is false (`:51`), otherwise `metadata["utility"] = metadata.get("utility", 0) + 1` on every item in the same list.

So `utility` is incremented for *all* five retrieved entries whenever the task they were retrieved for succeeded. There is no attribution step: no check that the agent read the entry, no counterfactual re-run without it. The credit is pure co-occurrence.

### The driver: `cookbook/bfcl/bfcl_agent.py`

`execute` (`:591`) is one loop over the actor's task shard, with an inner retry loop over `num_trials`.

**Retrieval happens before the task and only at turn 0** (`:600-602`):

```python
for i in range(self.max_interactions):
    if self.use_memory and i == 0:
        self.update_task_history_with_memory(run_id, task_index, previous_memories)
```

`update_task_history_with_memory` (`:121-136`) has two branches. On the first trial `previous_memories` is empty, so it POSTs `retrieve_task_memory` with `top_k: 5` (`get_memory`, `:166-181`), stores the returned list in `self.retrieved_memory_list[run_id][task_index]` (`:126`) for later crediting, and rewrites the first user message. On a retry it skips retrieval entirely and injects `previous_memories` — the entries just written from the failed attempt — instead. The injection is a prefix on the user message (`get_query_with_memory`, `:138-143`):

```python
"content": "Task:\n" + query + "\n\nSome Related Experience to help you to complete the task:\n" + memory,
```

**The update happens after the task is scored** (`:647-661`), and this block is the whole refinement phase:

```python
reward = self.get_reward(run_id, task_index)
if self.use_memory:
    if self.use_memory_addition:  # selectively add memories when succeed
        new_traj_list = [self.get_traj_from_task_history(task_id, self.history[run_id][task_index], reward)]
        previous_memories = self.add_memory(new_traj_list)
        if reward != 1:
            self.delete_memory_by_ids([mem["memory_id"] for mem in previous_memories])

    # update the freq & utility attributes of retrieved memories
    update_utility: bool = reward == 1
    self.update_memory_information(self.retrieved_memory_list[run_id][task_index], update_utility)

counter += 1
if self.use_memory_deletion and counter % self.delete_freq == 0:
    self.delete_memory()
```

Four mechanisms are visible here and each maps to a named part of the paper.

*Selective addition* is implemented as add-then-delete-if-failed: every trajectory is summarized into the store, and the resulting memory ids are immediately removed again when `reward != 1`. The net effect matches the paper, but the intermediate write means a concurrent actor could retrieve an entry that is about to be deleted.

*Failure-aware reflection* is the interaction between `previous_memories` surviving the deletion (it is a local Python list, not a store query) and the retry branch of `update_task_history_with_memory`. The lesson written from the failed attempt is injected into the next trial of the same task; if that trial succeeds, `add_memory` is called again on the successful trajectory and those entries are kept. The maximum number of retries is `num_trials` (`:67`), and the loop breaks on success at `:672-673`.

*Utility crediting* is `update_utility: bool = reward == 1` at `:656` — the single line that makes this repo different from every other self-improving harness in this index.

*Periodic sweep*: deletion runs every `delete_freq` tasks, defaulting to 10 (`:72`) and set to 5 in the committed `main()` (`run_bfcl.py:117`).

`cookbook/appworld/appworld_react_agent.py:197-213` is the same block with AppWorld's partial scoring: `after_score = self.get_reward(world)` (`:197`), `update_utility: bool = after_score == 1` (`:208`), and one difference that matters —

```python
if self.use_memory_deletion: # and counter % self.delete_freq == 0:
    self.delete_memory()
```

(`:212`). The periodic condition is commented out, so on AppWorld the deletion sweep runs after every single task, while on BFCL-V3 it runs every 5 or 10. This is an undocumented protocol difference between the two benchmarks in the paper's own main table.

### The 10 protocol components

| # | Component | Status | Location and details |
|---|---|---|---|
| 1 | Entry points | present | Console script `reme` → `reme_ai.main:main` (`pyproject.toml:79`) starts the HTTP or Model Context Protocol service. Benchmarks: `cookbook/bfcl/run_bfcl.py:125`, `cookbook/appworld/run_appworld.py`. Pool construction: `cookbook/bfcl/init_exp_pool.py:214`, `init_task_memory_pool.py:214`. Statistics: `run_exp_statistic.py` in each cookbook folder. |
| 2 | Special token / action handling | absent | No custom action tokens, no generation interception. BFCL uses provider-native function calling; AppWorld extracts fenced Python from the response (`appworld_react_agent.py:138-168`). |
| 3 | Environment / tool interaction | present, central | The memory system is an HTTP service the agent POSTs to: `get_memory` → `retrieve_task_memory` (`bfcl_agent.py:166-181`), `add_memory` → `summary_task_memory` (`:183-199`), `delete_memory_by_ids` → `vector_store` action `delete_ids` (`:201-210`), `update_memory_information` → `record_task_memory` (`:212-222`), `delete_memory` → `delete_task_memory` (`:224-233`). Service side: `reme_ai/service/task_memory_service.py:80,124,173`. Retrieved content re-enters as a prefix on the first user message, never as a system prompt or a tool result. |
| 4 | Token masking | absent | No training, so nothing to mask. |
| 5 | Reward function | present, external | Both rewards come from the benchmark, not from this repo. BFCL-V3 scoring is delegated to the Gorilla evaluator — `get_reward` (`bfcl_agent.py:406`) imports `multi_turn_runner` from `bfcl_eval` (`:40`) and calls it at `:525` against the possible-answer files, giving a binary score. AppWorld computes a partial score as `num_passes / (num_passes + num_failures)` from `world.evaluate()` (`appworld_react_agent.py:132-136`), taken both before and after the episode (`:180,197`) with an unused `uplift_score` recorded (`:198`). There is no harness-level reward and no critic. |
| 6 | RL training loop | absent | No gradients, no advantage, no rollout groups. The word "reward" appears only as the benchmark score. |
| 7 | SFT / cold-start | present in substance | The acquisition phase is the cold start: run the training split with `use_memory=False`, then `init_exp_pool.py` groups trajectories by task id and, for groups larger than two, keeps only the highest- and lowest-reward pair (`:66-82`) so the comparative extractor has a contrast. That pair selection is why the paper samples $N = 8$ trajectories and then only uses two. |
| 8 | Data pipeline | present | JSON Lines throughout. `split_into_trainval.py:9` is `random.shuffle(data)` followed by a ratio split. Stored memories are `TaskMemory` objects (`reme_ai/schema/memory.py`) whose vector-node content is the `when_to_use` field (`:106-116`), confirming the paper's usage-scenario indexing at code level; everything else lives in metadata, including `freq` and `utility`. |
| 9 | Multi-GPU / infrastructure | present, modest | Four ray actors by default (`run_bfcl.py:90`), each given `task_ids[i::max_workers]` (`:56`) — a round-robin shard. All four share one `memory_workspace_id` (`bfcl_v3`, `:99`), so the pool is global across workers while the task order each worker sees is a strided subsequence. Vector store backends: local file or Elasticsearch. |
| 10 | Evaluation | present | `cookbook/bfcl/run_exp_statistic.py` computes two families: `calculate_best_at_k` (`:9-29`), the mean over groups of size $k$ of the group maximum, and `calculate_pass_at_k` (`:33-41`), the fraction of groups whose maximum reaches 1.0. The paper's Pass@4 is `pass@4`; the paper's Avg@4 — defined on page 6 as "the average task success rate across four independent trials" — corresponds to `best@1` over four runs, not to `best@4`. Aggregation is a plain mean per file (`:121-129`); no per-position or stream-time breakdown is computed anywhere. |

### Answers to the six questions we came with

| Question | Answer | Evidence |
|---|---|---|
| What state persists across tasks? | One vector-store workspace of `TaskMemory` entries, each indexed by the embedding of its `when_to_use` text, each carrying `freq` and `utility` integers in metadata. Nothing else — no prompt files, no tool definitions, no configuration. | `reme_ai/schema/memory.py:106-116`; `reme_ai/vector_store/update_memory_freq_op.py:53` |
| What signal triggers or gates an update? | Addition passes three filters: the producing trajectory's own score must reach 1.0 (`success_threshold`), a language-model judge must return valid with score at least 0.5, and the candidate must not exceed 0.5 cosine similarity to anything already stored or in the same batch. Deletion is the `freq`/`utility` ratio. At driver level the score filter is add-then-delete-if-failed. | `summary/task/trajectory_preprocess_op.py:49-54`; `memory_validation_op.py:93-100`; `memory_deduplication_op.py:54,119`; `bfcl_agent.py:649-653` |
| Does any score from a later task feed back into the update? | **Yes, for deletion only.** `update_utility: bool = reward == 1` credits every entry retrieved for a task on that task's outcome, and the task is strictly later than the one that created the entry. Acceptance sees nothing but the producing batch. | `bfcl_agent.py:656-657`; `appworld_react_agent.py:208-209`; `delete_memory_op.py:47-56` |
| Is any earlier task ever re-evaluated? | **No.** Nothing re-runs a completed task. `grep -rniE '\b(rollback\|revert\|anchor set\|holdout\|regression\|re-?evaluat\|replay buffer\|catastrophic forget)\b'` over `reme_ai/` and `cookbook/` returns exactly one hit, and it is BFCL's own held-out-turn mechanic (`cookbook/bfcl/bfcl_utils.py:66`). No anchor set, no revert path, no regression check. The only re-run in the system is the within-task retry, which re-runs the *current* task. | negative grep; `bfcl_agent.py:597,672-673` |
| Are result files or logs shipped? | **No scores or run logs.** `cookbook/appworld/exp_result/*` and `cookbook/appworld/experiments/*` are gitignored (`.gitignore`), and no BFCL results are tracked. What ships is `docs/library/paper_data/task/*.jsonl`: six experience pools (appworld 8B/14B/32B at 207/218/184 entries, bfcl 8B/14B/32B at 96/110/99), plus `docs/library/{bfcl_v3,appworld}.jsonl` at 474 and 193 entries. Every entry in all six paper pools has `freq: 0` and `utility: 0` — verified by counting nonzero values, which is zero in every file. These are acquisition-phase snapshots taken before any reuse, not evolution logs. | `.gitignore`; `docs/library/paper_data/task/` |
| What stream or ordering construction exists? | **None.** Evaluation order is the benchmark file's order, strided across four concurrent actors sharing one pool; the train/evaluation split is a random shuffle. No non-stationarity, no shift, no ordering control, and no reporting against stream position. | `run_bfcl.py:56,90,99`; `split_into_trainval.py:9`; `run_exp_statistic.py:121-129` |

One consequence of the sharding is worth stating plainly: with four actors round-robin over the same pool and deleting from it on their own counters, the effective task stream each actor sees is neither the benchmark order nor a controlled one, and the pool state at any point depends on interleaving. Any attempt to reconstruct a stream-time curve from this setup would be measuring the scheduler as much as the method.

## Relevance to this project

### Components we can directly reuse

- `reme_ai/vector_store/{update_memory_freq_op.py,update_memory_utility_op.py,delete_memory_op.py}` — roughly sixty lines total, and the cheapest working implementation of a forward-looking signal on harness content that exists in this index. The schema (two integers in metadata, incremented by a service call after scoring, swept by a threshold rule) is directly liftable to our long-term-memory layer.
- `reme_ai/config/default.yaml:17,26,117,134` — the whole system expressed as four composable pipelines. Independent of the method, this is a good model for how to make a memory layer's behavior a configuration diff rather than a code change, which is what a critic reading harness files would need.
- `reme_ai/summary/task/*_prompt.yaml` — six prompts (success, failure, comparative extraction; validation; rerank; rewrite) with stable wording, already used across a published comparison. A ready proposer prompt set for the memory layer.
- `docs/library/paper_data/task/*.jsonl` — six seed pools. Useful as realistic memory-layer content for building and debugging our own pipeline, and as a concrete answer to what a distilled experience looks like in practice.
- `cookbook/bfcl/bfcl_agent.py:591-680` — the cleanest short example of a score-before-update loop we have seen: retrieve at turn 0, run, score, then update. Ninety lines, no framework.

### Components we need to modify

- **The crediting rule** (`bfcl_agent.py:656`, `update_memory_utility_op.py:62`). Co-occurrence crediting over five simultaneously retrieved entries is too noisy to serve as a return estimate, and it is one-sided: a failed task increments `freq` but never penalizes beyond the missing increment. We would need per-entry attribution (was it actually used) or a counterfactual scoring pass, and a signal defined on the harness state rather than on individual entries.
- **The gate** (`delete_memory_op.py:47-56`). It only removes, it only acts after five retrievals, and it never informs acceptance. Our commit rule needs the forward-looking term at the accept decision, which means the estimate has to exist before any retrieval has happened — that is exactly the gap a learned $\hat{\Phi}(h)$ fills and this repo shows is otherwise unfillable.
- **The retention check.** There is none to modify. An anchor set, its growth rule, and a per-task re-check at commit would all be new construction on top of this code.
- **The stream** (`run_bfcl.py:56`, `split_into_trainval.py:9`). Random shuffle plus round-robin sharding has to be replaced entirely by an ordered, non-stationary stream with a single writer to the pool.
- **The retry loop** (`bfcl_agent.py:597,602,651`). The failure-aware reflection is a horizon $\gamma = 0$ update at the memory layer and would have to be removed or reframed if this driver were reused as a baseline, since it scores the edit on the very task that produced it.

### Components that don't apply

- `reme_ai/summary/personal/`, `retrieve/personal/`, `mem_agent/`, `mem_tool/` — a separate personal-memory product line (observations, reflections, contradiction removal) unrelated to the paper.
- `reme_ai/summary/tool/` and `cookbook/tool_memory/` — tool-call memory belonging to a different paper (`README.md` points at arXiv:2608.03403).
- `reme_ai/summary/working/`, `retrieve/working/`, `cookbook/working_memory/` — context compaction and offload, an operational concern, not an edit to persistent state.
- `cookbook/frozenlake/`, `cookbook/simple_demo/` — demonstrations.
- Everything on the `main` branch — a different product, as described in the version warning.

### Key code snippets worth studying

| Location | Why |
|---|---|
| `reme_ai/vector_store/delete_memory_op.py:47-56` | Ten lines. The only removal rule in this index whose input is task outcomes measured after the edit. Read it beside `recipe/gaia_evolver/run.py:1174` in HarnessX to see the two shapes of a commit rule side by side. |
| `cookbook/bfcl/bfcl_agent.py:647-661` | The full refinement block. Selective addition, utility crediting, and the periodic sweep in fifteen lines, with the score-before-update ordering visible. |
| `cookbook/bfcl/bfcl_agent.py:121-143` | Retrieval-at-turn-0 and the retry branch that injects the failed attempt's own lesson — the two halves of the protocol, one correct by our standard and one deliberately not. |
| `cookbook/appworld/appworld_react_agent.py:208-213` | The commented-out periodic condition, an example of an undocumented protocol difference between two benchmarks in the same table. |
| `reme_ai/config/default.yaml:17,26` | The shipped defaults that disable the reranker and rewriter, and the validation threshold that disagrees with the paper's own prompt. Check these before trusting a reproduction. |
| `cookbook/bfcl/init_exp_pool.py:66-82` | Why $N = 8$ samples produce two usable trajectories: the best and worst of each group are kept so the comparative extractor has a contrast. |
| `cookbook/bfcl/run_exp_statistic.py:9-41` | How Avg@4 and Pass@4 are actually computed, and why `best@4` in the code is not the paper's Avg@4. |
