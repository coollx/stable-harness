# Repo Analysis: MemRL

**Path:** `refs/related-work/MemRL/repo/` · **Source:** https://github.com/MemTensor/MemRL · **Analyzed:** 2026-09-15 (commit `c1b322c`, dated 2026-07-18, "fix: align retrieval similarity thresholds")

## Overview

- **Paper:** *MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory* (see `paper_analysis.md`)
- **Framework:** custom, built on top of the MemOS memory service (`memos.mem_os.main.MOS`) with a Qdrant vector store and OpenAI-compatible model and embedding providers. MIT licence ("Copyright (c) 2026 jiaqian"), Python 3.10, 12 MB, 246 tracked files, roughly 29.7k lines of Python of which most is the repository's own code plus two vendored benchmarks under `3rdparty/` (BigCodeBench and LifelongAgentBench). Dependencies include `mem0ai`, `memoryos`, `alfworld`, `qdrant-client`, `datasets`, and `tensorboard`.
- **RL Algorithm:** none in the gradient sense. The only learned quantity is one float per memory item, maintained by an exponential moving average of the observed reward (`memrl/service/value_driven.py:181-221`). No weights are touched anywhere in the repository.
- **Base Model:** provider-agnostic via OpenAI-compatible endpoints; the configuration files carry the paper's per-benchmark backbones and require the user's own API key.
- **Key Innovation:** a `q_value` float stored in each memory item's metadata, updated from the task's pass/fail outcome, and consumed by a two-phase retriever that blends z-normalized similarity with z-normalized utility.

**Headline finding for us: the discount is zero everywhere, deliberately and redundantly, and the one code path that could have carried value across tasks does not exist.** The discount factor defaults to `0.0` in the dataclass (`memrl/service/value_driven.py:30`), in the configuration schema (`memrl/configs/config.py:207`), and in all four shipped configuration files (`configs/rl_bcb_config.yaml:108`, `configs/rl_hle_config.yaml:70`, `configs/rl_alf_config.yaml:71`, `configs/rl_llb_config.yaml:152`), the last of which comments it as "Discount factor (0 for single-step credit assignment)". Independently of the discount, the batch update path hard-codes the bootstrap term to `None` (`memrl/service/memory_service.py:820`), so even a non-zero discount would multiply zero. The Lifelong Agent Bench runner announces a multi-step variant — its own docstring says "'mdp': Section level `update_values_chain_mdp()` only (slow propagation)" (`memrl/run/llb_rl_runner.py:1091`) — and calls `self.memory_service.update_values_chain_mdp(...)` at `memrl/run/llb_rl_runner.py:1333`. **That method is defined nowhere in the repository** (`grep -rn "def update_values_chain_mdp" .` returns nothing). The runner even builds and maintains the data structure it would consume, a first-in-first-out map from each newly written memory to the memory identifiers retrieved when it was created, capped at 10,000 entries (`memrl/run/llb_rl_runner.py:1259-1268`, comment "Track memory references for chain MDP"). The plumbing for cross-task credit propagation is laid; the consumer is missing, so selecting `algorithm: mdp` crashes.

## File Structure Map

```
memrl/
  service/
    value_driven.py       THE METHOD. RLConfig defaults (:26-42), ValueAwareSelector.select (:82-152),
                          QUpdater.update (:181-221), MemoryCurator merge/attribution (:225-273)
    memory_service.py     2199 lines; the MemOS wrapper. update_value (:770-790), update_values (:793-845),
                          normalization (:1239-1252), two-phase retrieve_query (:1254-1500),
                          add_memories (:1508+), save/load_checkpoint_snapshot (:1667, :1760)
    updater.py            memory-write strategies: Vanilla (:272), Validation (:292), Adjustment (:316);
                          reflection prompt (:248-261); ADJUSTMENT NOTE append (:383)
    retrievers.py         retrieval helpers
    strategies.py         per-benchmark prompt/extraction strategies
    procedural_memory.py  the MemP-style procedural memory this repo extends
    builders.py, keyer.py object construction and intent-key derivation
  run/
    base_runner.py        shared runner scaffolding
    llb_rl_runner.py      1387 lines; Lifelong Agent Bench. _split_dataset (:307-338),
                          _sample_from_indices (:458-520), _evaluate (:998-1084),
                          main loop (:1128-1276), chain-MDP call site (:1316-1344),
                          per-section snapshot (:1347-1350)
    bcb_runner.py         750 lines; BigCodeBench. _run_phase(update_memory) (:412-489),
                          per-epoch train/val split of memory writing (:704-733)
    hle_runner.py         1277 lines; Humanity's Last Exam. cumulative-correct maps (:106-107, :1187-1229),
                          _prune_valid_memories (:413-426), _eval_ckpt_sequence (:428-444)
    alfworld_rl_runner.py ALFWorld
  configs/config.py       the full pydantic configuration schema; gamma (:207), q_min_threshold (:233),
                          weight_sim/weight_q (:234-235), llb z-score and q_floor switches (:143-154)
  agent/, envs/, providers/, trace/, utils/, cli/, bigcodebench_eval/, lifelongbench_eval/
configs/
  rl_llb_config.yaml      169 lines, the most documented config; rl block at :149-169
  rl_bcb_config.yaml, rl_alf_config.yaml, rl_hle_config.yaml
  alfworld/, bigcodebench/, envs/
run/
  run_llb.py, run_bcb.py, run_alfworld.py, run_hle.py     the four entry points
data/
  bigcodebench/bigcodebench_full.jsonl, splits/full_seed123.json
  llb/{os_interaction,db_bench}_*.json                     task data only, no results
3rdparty/
  bigcodebench-main/, LifelongAgentBench/                  vendored benchmarks
openspec/                 change proposals and specs for this codebase (design notes, not runtime)
scripts/                  test_sim_threshold.py, merge_llb_train_val.py
```

## Component Inventory

### Entry points

Four scripts, one per benchmark: `run/run_llb.py`, `run/run_bcb.py`, `run/run_alfworld.py`, `run/run_hle.py`. Each reads one YAML file from `configs/`; the README states that logs go to `logs/` and results to `results/`, both configurable via `experiment.output_dir` and both absent from git. The Lifelong Agent Bench runner takes an `algorithm` key with values `memp | rl | mdp | rlmdp | slow_rl` (`configs/rl_llb_config.yaml:93`); only the paths containing `rl` reach the working update, and the `mdp` paths reach the undefined method described above.

### The learned state (the closest thing here to a critic)

`memrl/service/value_driven.py:26-42` is the whole hyperparameter surface:

| Field | Default | Meaning |
|---|---|---|
| `epsilon` | 0.1 (0.01 in the Lifelong Agent Bench config) | exploration probability in the final top-k pick |
| `tau` | 0.35 | unknown-detection threshold on similarity, separate from `sim_threshold` |
| `alpha` | 0.1 (0.3 in the configs, matching Table 8) | step size of the utility update |
| `gamma` | 0.0 | discount, commented "discount factor (single-step default)" |
| `q_init_pos` / `q_init_neg` | 0 / 0 | initial utility for memories written from a success / a failure |
| `q_floor` | None | optional lower clamp on the utility |
| `success_reward` / `failure_reward` | +1.0 / −1.0 | the only rewards in the system |
| `sim_threshold` | 0.5 | the Phase A recall cutoff (the paper's $\delta$) |
| `topk` | 5 | candidate pool size |
| `novelty_threshold` | 0.85 | above this similarity, a new memory is merged instead of appended |
| `recency_boost` | 0.0 | additive bonus for recently used memories, off by default |
| `reward_merge_gain` | 0.1 | fraction of reward propagated to a merge target |
| `weight_sim` / `weight_q` | 0.5 / 0.5 | the Phase B blend (the paper's $\lambda = 0.5$) |

### The utility update — exact code

`memrl/service/value_driven.py:181-221`, reproduced verbatim for the lines that matter:

```python
    def update(self, memory_id: str, reward: float, next_max_q: Optional[float] = None) -> float:
        """Apply single-step Q-learning update on a memory item's metadata."""
        ...
        old_q = float(old_meta.get("q_value", self.cfg.q_init_pos))
        target = float(reward) + (self.cfg.gamma * float(next_max_q or 0.0))
        new_q = (1.0 - self.cfg.alpha) * old_q + self.cfg.alpha * target
        ...
        visits = int(old_meta.get("q_visits", 0)) + 1
        old_ma = float(old_meta.get("reward_ma", 0.0))
        reward_ma = (1.0 - self.cfg.alpha) * old_ma + self.cfg.alpha * float(reward)
        new_meta = old_meta | {
            "q_value": float(new_q), "q_visits": visits, "last_reward": float(reward),
            "reward_ma": reward_ma, "q_updated_at": _now_iso(), "last_used_at": _now_iso(),
        }
        # keep retrieval key unchanged; only update metadata
        text_mem.update(memory_id, {"id": memory_id, "memory": getattr(item, "memory", ""), "metadata": new_meta})
        return float(new_q)
```

With `gamma = 0.0` this is exactly $Q \leftarrow (1-\alpha) Q + \alpha r$, the paper's eq. 4, at $\alpha = 0.3$. The comment on the final line is the design in one sentence: the retrieval key is immutable and only the metadata moves, so learning never changes what a memory *says*, only how eagerly it is retrieved.

### What triggers an update, and what signal enters it

`memrl/service/memory_service.py:793-845` is the path the runners actually call. It takes a list of per-trajectory outcomes and a parallel list of the memory identifiers retrieved for each trajectory, maps each outcome to $\pm 1$ at `:814-818`, and then:

```python
            for mem_id in mem_ids:
                updates.append((mem_id, reward, None))          # memory_service.py:820
```

Three consequences, all load-bearing for us. First, **the bootstrap slot is hard-coded to `None`**, so no value from any other state — later task or otherwise — enters the update, independent of the discount setting. Second, **credit is uniform across the trajectory**: every memory retrieved anywhere in a multi-step episode receives the identical episode-level reward, with no per-step attribution. Third, **the trigger is the environment verifier's binary outcome on the current task only** — there is no other update path in the repository.

### Two-phase retrieval

`memrl/service/memory_service.py:1254-1500`. Phase A at `:1306-1318` scores the intent embedding against every stored query embedding, keeps candidates with `sim >= threshold`, sorts descending, and truncates to $k_1$. Per-candidate utility is resolved at `:1363-1422` through a cache, then the item metadata, then `q_init_pos` as the fallback; the optional recency boost and utility floor are applied here. `:1425-1427` drops candidates below `q_min_threshold`. `:1437-1465` computes the blend, `c["score"] = sim_z * w_sim + q_z * w_q`. `:1467-1493` takes the top $k_2$ with an epsilon-greedy branch and, for Lifelong Agent Bench, an optional `dedup_by_task_id`.

The normalization asymmetry at `:1239-1252` is worth noting: `_normalize_similarity` uses a fixed mean and standard deviation from configuration, while `_normalize_q` computes mean and standard deviation over the current call's candidate pool and clips to $\pm 3.0$. A memory's effective utility score therefore depends on which other candidates happened to be recalled with it. The docstring at `:1254-1262` claims "No two-stage retrieval" and is stale — the function below it is two-phase.

A parallel selector lives in `memrl/service/value_driven.py:82-152` (`ValueAwareSelector.select`), which sorts by utility descending with similarity as tiebreak rather than blending, applies `q_min_threshold` at `:126-128`, and picks epsilon-greedily at `:143-147` via `random.sample`. The runners use the `memory_service` path; this one is the simpler reference implementation.

### The gate

`q_min_threshold` is the only gate-shaped object in the system: `−0.8` in the configuration schema (`memrl/configs/config.py:233`) and in the BigCodeBench and Lifelong Agent Bench configs, `−10` in the ALFWorld and Humanity's Last Exam configs (effectively disabled). It is a **retrieval suppression filter, not a commit gate**: a memory whose utility falls below it stops being retrievable but stays in the store, keeps its text, and can rise again if it is ever selected by the epsilon-greedy branch. Nothing in the repository deletes a memory, reverts a write, or rolls back to a previous snapshot on a regression.

### Memory writing

`memrl/service/updater.py` (424 lines) implements three strategies, chosen by configuration:

- `VanillaUpdater` (`:272-290`) — always write the trajectory.
- `ValidationUpdater` (`:292-314`) — write only on success.
- `AdjustmentUpdater` (`:316-400`) — write on success; on failure either generate a reflection with the prompt at `:248-261` (which matches the paper's Appendix I verbatim) and append it as a new memory, or append an "`--- ADJUSTMENT NOTE ---`" block in place to the existing memory (`:383`) and multiply its confidence by `0.8` (`:391`).

Preparation runs in parallel and writes run serially. The confidence field is separate from the utility and is not consumed by retrieval ranking.

### Stream construction and what persists

For Lifelong Agent Bench, `memrl/run/llb_rl_runner.py:307-338` sorts the task keys deterministically and then does this:

```python
        else:
            # Copy all keys for each section instead of splitting
            self.section_splits = [list(all_keys) for _ in range(self.num_section)]
```

So a "section" is a full replay of the entire dataset in the same fixed order, and `num_sections` is the epoch count — the configuration comment says so directly: "num_sections = 10 means replaying the same dataset 10 times ('10 epochs')" (`configs/rl_llb_config.yaml:6-7`; the shipped value is `20` at `:111`). There is no shuffling between epochs, no distribution shift, and no held-out portion of the stream. The main loop (`:1128-1276`) walks sections, splits each into mini-batches of `batch_size: 5` (`configs/rl_llb_config.yaml:114`), runs the batch in a thread pool, then calls `update_values` (step 3) and `add_memories` (step 5) — so **scoring happens before any update, at mini-batch granularity**. Newly written memories are stamped with `lb_epoch`, `sample_index`, `task_id`, `success`, an initial `q_value` of `q_init_pos` or `q_init_neg`, and `q_visits: 0` (`:1236-1250`).

What persists across tasks is exactly: the memory texts, their intent embeddings, and their metadata (`q_value`, `q_visits`, `last_reward`, `reward_ma`, timestamps). Nothing else — no running statistics over the stream, no per-task history, no record of which tasks previously passed.

`memrl/run/bcb_runner.py` is the only runner with an explicit memory-writing switch: `_run_phase(..., update_memory: bool)` at `:412-489` buffers updates and flushes them through `_flush_memory_updates` (`:444`) at a buffer size of 25, and `run()` at `:673-733` calls the train phase with `update_memory=True` (`:713`) and the validation phase with `update_memory=False` (`:722`). That is a per-epoch frozen-memory evaluation on a held-out set, written to `epoch_summary.json`, `samples.jsonl`, `metrics.json`, and `run_config.json` under a per-epoch directory.

### Re-evaluation of earlier state

Two mechanisms, both in the Humanity's Last Exam runner, neither an anchor set:

- `_prune_valid_memories` (`memrl/run/hle_runner.py:413-426`) removes validation questions from the local memory indices before evaluating, "to avoid leakage" — it pops matching keys out of `dict_memory` and `query_embeddings`.
- `_eval_ckpt_sequence` (`:428-444`) loads the saved per-section snapshots one after another via `load_checkpoint_snapshot`, prunes, and evaluates each against the same fixed validation set. **This is the most reusable artifact in the repository**: it produces a performance curve as a function of the memory store's own age, offline, from snapshots. It is a diagnostic tool, run outside the training loop; no result from it feeds back into any update.

Snapshots themselves are written per section by `memory_service.save_checkpoint_snapshot` (`memrl/service/memory_service.py:1667`), which dumps the MemCube and the Qdrant collection to `<ck_dir>/snapshot/<ckpt_id>/`, and restored by `load_checkpoint_snapshot` (`:1760`).

Cumulative success is implemented in the Humanity's Last Exam runner as two monotone dictionaries, `train_cumulative_correct_map` and `valid_cumulative_correct_map` (`:106-107`, updated at `:1187-1229`): a question identifier is set to `True` the first time it is answered correctly and never set back to `False`. This confirms in code that the paper's cumulative success rate cannot decrease. `hle_runner.py:484` adds "You attempted this question before." to the prompt on a repeat, so repeats are not independent attempts.

The Lifelong Agent Bench runner does have a frozen-memory evaluation, `_evaluate` at `:998-1084`, which samples with `phase="eval"` and never writes memory — but it is gated on `valid_interval`, which the shipped configuration sets to `0` (`configs/rl_llb_config.yaml:118`), so it never fires in a default run.

### Components absent

- **No reinforcement-learning training loop, no reward model, no advantage, no token masking, no multi-GPU infrastructure.** The repository has no gradient step of any kind; components 4, 6, 7, and 9 of the analysis protocol do not exist here.
- **No forgetting-rate computation.** The paper's Section 5.4.2 metric (tasks lost over current failures, 0.041 for MemRL) is not computed by any shipped code; it must have been produced outside the repository.
- **No released results.** `git ls-files` returns no logs, no trajectories, no memory snapshots, and no result files. The only `.jsonl` tracked is `data/bigcodebench/bigcodebench_full.jsonl`, which is task data. Every number in the paper must be regenerated against paid model endpoints.
- **No revert, rollback, or memory deletion path.**

## Relevance to this project

### Components we can directly reuse

- **`memrl/run/hle_runner.py:428-444` (`_eval_ckpt_sequence`)** — replay a sequence of saved harness states against one fixed evaluation set and get a performance-versus-state curve, offline. This is the measurement shape our critic training needs (realized outcomes as a function of harness state), applied to the memory layer and already written.
- **`memrl/run/bcb_runner.py:704-733`** — the two-line pattern of running the same phase function twice per epoch with `update_memory=True` then `update_memory=False`, which is the cheapest correct way to separate "learning on this set" from "measuring on that set" without duplicating the runner.
- **`memrl/service/value_driven.py:181-221`** — the utility update itself is twenty usable lines, and the metadata schema around it (`q_value`, `q_visits`, `last_reward`, `reward_ma`, `q_updated_at`) is a sensible minimum for any per-item value we attach to a harness element.
- **`memrl/service/memory_service.py:1239-1252` and `:1437-1465`** — combining a relevance score and a learned value into one ranking requires normalizing them onto a comparable scale; this is a worked example, including the asymmetry to avoid (fixed normalization for one term, per-call normalization for the other).

### Components we need to modify

- **The update rule needs a real bootstrap term.** `memory_service.py:820` must stop passing `None`, and something must define what the "next state" of a memory-layer harness is. The repository's own `memid_pair` structure (`llb_rl_runner.py:1259-1268`) is the authors' guess at that: a map from each newly written memory to the memories that were retrieved when it was created, which would let credit flow backwards along a chain of derivations. They built it and never wrote the consumer. If we want a memory-layer instance of our horizon, this is the exact gap to fill.
- **`q_min_threshold` must become a commit-time gate rather than a retrieval filter.** Today a harmful memory is merely hidden; under our objective the decision to write it should be gated, and the gate should consult a retention measurement.
- **Stream construction must be replaced outright.** `llb_rl_runner.py:331-332` replays one fixed order of one fixed dataset; our regime needs heterogeneous, non-stationary ordering with a held-out anchor set grown from the stream.
- **Credit assignment within an episode.** Uniform trajectory-level reward across every retrieved memory (`memory_service.py:820`) means a good memory retrieved in a failing episode is punished. Any per-step attribution would be an improvement; the repository's `MemoryCurator.attribute_reward` (`value_driven.py:225-273`) has the shape of one but is only applied at merge time with `gain = reward_merge_gain * reward`.

### Components that don't apply

- The MemOS and Qdrant stack, the vendored BigCodeBench and LifelongAgentBench harnesses, and the four benchmark-specific runners are deployment scaffolding for their benchmarks, not reusable for ours.
- The three memory-write strategies in `updater.py` operate on the content of a single layer; our edit space spans layers and includes moves between them, which has no analogue here.
- `openspec/` is this codebase's own change-proposal directory, not a runtime component.
- The convergence machinery of the paper has no code: there is nothing to reuse from Theorem A.1 beyond the statement that the learned quantity is a one-step expectation.

### Key code snippets worth studying

| Location | Why |
|---|---|
| `memrl/service/value_driven.py:181-221` | the entire learning rule, and the comment that learning touches only metadata |
| `memrl/service/memory_service.py:810-828` | `next_max_q` hard-coded to `None` and uniform trajectory-level reward, in six lines |
| `memrl/run/llb_rl_runner.py:1091-1100` and `:1316-1344` | a documented, called, and undefined cross-task value-propagation method |
| `memrl/run/llb_rl_runner.py:1259-1268` | the memory-derivation graph they collect but never consume |
| `memrl/run/llb_rl_runner.py:327-332` | "Copy all keys for each section instead of splitting" — the whole stream design |
| `memrl/run/llb_rl_runner.py:1128-1276` | the score-then-update ordering at mini-batch granularity |
| `memrl/run/hle_runner.py:413-444` | leakage pruning plus snapshot-sequence replay against a fixed set |
| `memrl/run/hle_runner.py:1187-1229` | cumulative success implemented as a monotone dictionary |
| `memrl/service/memory_service.py:1425-1465` | the utility floor and the similarity-utility blend |
| `memrl/service/updater.py:248-261` and `:375-391` | the failure-reflection prompt and the in-place adjustment-note append |
