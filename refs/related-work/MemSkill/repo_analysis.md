# Repo Analysis: MemSkill

**Path:** `refs/related-work/MemSkill/repo/` · **Source:** https://github.com/ViktorAxelsen/MemSkill · **Analyzed:** 2026-09-15 (commit `9907c35`, "support for qwen3-embedding retriever", 2026-05-24)

## Overview

- **Paper:** MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents (see `paper_analysis.md`)
- **Framework:** custom PyTorch. No veRL, TRL, OpenRLHF, DeepSpeed, or Accelerate — the whole reinforcement-learning stack is about 1000 hand-written lines in `src/controller.py`. 31 Python files, 12,290 lines, 9.0 MB excluding git history. Apache 2.0.
- **RL algorithm:** proximal policy optimization with generalized advantage estimation (`src/controller.py:119-165`, `src/controller.py:573`), acting on an ordered Top-K subset sampled by the Gumbel-Top-K trick (`src/controller.py:507-512`).
- **Trained parameters:** three small multilayer perceptrons plus a value head — nothing else. The executor and designer language models are called through a hosted API and are never updated (`src/executor.py`, `src/designer.py`, `llm_utils.py`). The embedding model (Qwen3-Embedding-0.6B or Contriever) is frozen and runs locally on one GPU.
- **Base models:** set by `--model` and `--designer-model` command-line flags; the paper used LLaMA-3.3-70B-Instruct and Qwen3-Next-80B-A3B-Instruct through the NVIDIA NIM and Together APIs.
- **Key innovation in code:** `src/operation_bank.py` holds a mutable, embedding-indexed set of natural-language memory operations, and `src/designer.py` rewrites that set during training under a snapshot-and-rollback gate.

Naming note: the paper's "skill" is the code's `Operation`, and the "skill bank" is `OperationBank`. The paper's SKIP primitive is `noop` in code. Reading the repo requires this mapping throughout.

## File Structure Map

```
main.py                        entry point: arg parsing, data loading, split, train or eval-only (1092 lines)
train_locomo.sh                LoCoMo training recipe (the paper's main conversational run)
train_alfworld.sh              ALFWorld training recipe (Batch A / Batch B protocol)
eval_locomo.sh eval_hp.sh eval_longmemeval.sh eval_alfworld.sh
                               eval-only recipes; load a checkpoint, build memory, score
alfworld_replay.py             collects expert trajectories into an offline JSON corpus
src/
  trainer.py    (2202)         the closed loop: episode collection, PPO updates, the designer cycle,
                               snapshot gate, rollback, early stop; three trainer classes
  controller.py  (958)         PPOBuffer, PPOController, Top-K log-probability, new-skill logit bias
  designer.py   (1674)         CaseCollector, EvolutionSnapshot, EvolutionSnapshotManager, Designer
  operation_bank.py (396)      Operation, OperationBank: add / update / evict / embed / rank
  memory_bank.py               the per-trace memory store and its retriever
  executor.py                  builds the executor prompt from span + retrieved memories + skills
  interactive_designer.py      designer variant for the ALFWorld setting
  alfworld_env_runner.py       runs the agent in ALFWorld to obtain Batch B reward
  config.py      (366)         every default and every command-line flag
  data_processing/             locomo.py longmemeval.py hotpotqa.py alfworld.py base.py
  eval/                        metric implementations (F1, LLM judge)
prompts/
  operation_templates.py       the four seed skills (insert / update / delete / noop)
  designer_prompts.py          analysis, reflection, refinement prompts for the designer
  designer_prompts_interactive.py, prompt_pool.py
skills/
  conversational_skills/       9 markdown files: the final evolved LoCoMo bank
  embodied_task_skills/        7 markdown files: the final evolved ALFWorld bank
data/longmemeval_s_splits.json the only shipped data file
```

Absent: no `results/`, no `checkpoints/`, no `outputs/`, no `logs/`, no configuration YAML, no tests, no Docker file, no CI. Every path the scripts write to is created at run time and none is committed.

## Component Inventory

### 1. Entry points

`main.py:1042` loads and splits data, `main.py:1052` builds the trainer, `main.py:1077` calls `trainer.train(train_data)`, and `main.py:1069` takes the `--eval-only` branch that loads a checkpoint and runs inference instead. Trainer selection is by dataset: `BaseTrainer` (`src/trainer.py:46`) for conversational data, `OfflineTrainer` (`src/trainer.py:1851`), and `AlfworldPairTrainer` (`src/trainer.py:1870`).

Released recipes, verbatim from the shell scripts:

| Setting | `train_locomo.sh` | `train_alfworld.sh` | `eval_locomo.sh` |
|---|---|---|---|
| inner / outer epochs | 100 / 10 | 100 / 10 | (eval only) |
| batch size (episodes per update) | 4 | 16 | — |
| skills selected per span | `--action-top-k 3` | 3 | `--action-top-k 7` |
| retriever | `contriever` | `contriever` | `qwen3-embedding-0.6b` |
| reward metric | `f1` | `llm_judge` | `llm_judge` |
| evolution frequency | every outer epoch | every outer epoch | — |
| edits per evolution round | 3 | 3 | 2 |
| new-skill exploration window | `--new-action-bias-steps 25` | 25 | 25 |
| failure window / pool | 100 epochs / 2000 cases | same | same |
| span construction | `--session-mode full-session` | `fixed-length` | `fixed-length`, `--chunk-size 512 --chunk-overlap 64` |

### 2. Action handling (the analogue of special-token handling)

There are no generation control tokens. The action is a set of skill names. `src/controller.py:284` scores every candidate skill against the state; `src/controller.py:507-512` samples the ordered Top-K without replacement by adding Gumbel noise to logits and taking the largest $K$; `src/controller.py:365` computes the matching joint log-probability for the policy ratio. The candidate list is sorted by name in `OperationBank.get_candidate_operations` so that action indices stay stable across a batch even though the bank's size changes between cycles.

### 3. Environment and tool interaction

All executor, judge, and designer calls go through `llm_utils.py` against an OpenAI-compatible endpoint, with round-robin over several API keys (`--api-key KEY1 KEY2`) and `ThreadPoolExecutor` fan-out. Query answering is batched: `src/trainer.py:855-863` builds one prompt per query and submits them together. ALFWorld rollouts go through `src/alfworld_env_runner.py` in a `ProcessPoolExecutor` (`--alfworld-pair-b-workers 80`), capped at 50 environment steps.

### 4. Token masking

**Absent, and necessarily so** — no language model is fine-tuned, so there is no loss over generated tokens and nothing to mask. The only gradient flows through three small networks over frozen embeddings.

### 5. Reward function

One scalar per episode. `src/trainer.py:542` calls `_evaluate_qa` after every span of the trace has been processed, `src/trainer.py:551` assigns it as `episode_log['total_reward']`, and `src/trainer.py:558` pushes it into the buffer as the terminal reward:

```python
qa_performance, raw_performance = self._evaluate_qa(
    conversation_data, memory_bank, ...)
episode_log['final_qa_performance'] = qa_performance
episode_log['raw_performance'] = raw_performance
episode_log['total_reward'] = qa_performance
```

`_evaluate_qa` (`src/trainer.py:750-920`) retrieves memories per query from the memory bank just built, answers each query with the base model, and scores by F1 or by an LLM judge depending on `--reward-metric`. There is no format term, no cost term, and no process reward that affects the objective.

### 6. RL training loop

`BaseTrainer.train` (`src/trainer.py:1031`) is two nested loops. The inner loop (`src/trainer.py:1120`) collects `batch_size` episodes in parallel threads, each through `_run_single_episode` (`src/trainer.py:1642`), then runs PPO. The outer loop (`src/trainer.py:1305` onward) is the designer cycle.

Advantages come from `PPOBuffer.compute_returns_and_advantages` (`src/controller.py:119`) with `gamma=0.99`, `gae_lambda=0.95`, and an explicit episode-boundary reset:

```python
next_non_terminal = 1.0 - dones[t]
delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
```

The comment on that block — "This prevents advantage from bleeding across episode boundaries" — is the precise statement of the horizon: the discount reaches backward across spans of one trace and stops at the trace edge.

The new-skill exploration incentive is split across two files. `src/trainer.py:1129-1133` computes a linearly decaying scale, `bias_scale = max(0.0, 1.0 - (self.new_action_bias_step / bias_steps))`, and `src/trainer.py:1196-1201` retires it once the window is exhausted, also clearing `new_operation_names` so the bias can never persist. `src/controller.py:323-364` applies it: the minimum logit gain that lifts the new skills' total probability to `new_action_p_min`, clamped at `new_action_delta_max` and scaled by the decaying factor, `delta = torch.log(target / (safe_p + eps))`.

### 7. Supervised or cold-start phase

**Absent.** The cold start is four hand-written seed skills in `prompts/operation_templates.py:133` (`get_initial_operations`), loaded by `OperationBank._initialize_with_seeds` (`src/operation_bank.py:121-133`). There is no imitation phase and no trajectory filtering.

### 8. Data pipeline

`main.py:65-112` (`split_data`) fixes the splits. For LoCoMo they are literal indices:

```python
train_index = [0, 1, 2, 3, 4, 5]
val_index   = [6, 7]
test_index  = [8, 9]
```

Six conversations are the entire training set. `data/longmemeval_s_splits.json` is the only shipped split file. HotpotQA and ALFWorld pass whole files through with empty validation splits (`main.py:94-100`).

`src/data_processing/alfworld.py:63-108` (`sample_pair`) builds the ALFWorld training pair. With `--alfworld-pair-same-type-prob 1.0` as the released script sets, it draws one task type and then splits a single sample into two disjoint halves:

```python
selected = random.sample(gamefiles, batch_a_size + batch_b_size)
batch_a_files = selected[:batch_a_size]
batch_b_files = selected[batch_a_size:]
```

### 9. Multi-GPU and infrastructure

Single GPU by design — `export CUDA_VISIBLE_DEVICES=0` in every script; the GPU holds only the embedding model. Scale comes from concurrency, not from parallel training: `ThreadPoolExecutor` for episodes and for query answering, `ProcessPoolExecutor` for ALFWorld environments, multi-key API round-robin, and `--inference-workers` / `--inference-session-workers` for evaluation. Metrics go to Weights and Biases (`src/trainer.py:474`, `src/trainer.py:1053`); nothing is written to a tracked file.

### 10. Evaluation

`src/eval/` implements F1 and the LLM judge; `--reward-metric` selects which one is the training signal and which one gates a case as a failure (`--designer-f1-threshold`, default 0.5, `src/trainer.py:869`). Evaluation at test time reuses the training path with `--eval-only`, a loaded checkpoint, and different span and Top-K settings (see the table in component 1). Per-dataset evaluators live in `src/data_processing/`.

## The six questions, answered from the code

**1. What state persists across tasks?** Exactly two things: the skill bank and the controller. The skill bank is a `Dict[str, Operation]` on the trainer (`src/operation_bank.py:108-133`), capped at `max_ops = 30` (`src/config.py:24`) with lowest-average-reward eviction at capacity (`src/operation_bank.py:225-252`). The controller is three MLPs plus a value head. The hard-case pool is a third, semi-persistent piece of state: `CaseCollector.clear(reset_pool=False)` (`src/designer.py:193-197`) deliberately keeps the pool across evolution cycles, so failures survive until they age out.

What does **not** persist is the memory. `_run_single_episode` calls `_initialize_memory_bank` for every episode (`src/trainer.py:1662`), and that function's own docstring says "Initialize an EMPTY memory bank" (`src/trainer.py:1699-1704`). Whatever the executor wrote for one trace is discarded before the next trace starts. Despite the paper's framing, no memory content carries across tasks during training.

**2. What triggers an update, and what gates it?** The trigger is periodic, not performance-based: `src/trainer.py:1305`, `if self.designer is not None and (outer_epoch + 1) % self.config.designer_freq == 0`, with `designer_freq = 1` in both released scripts, so one evolution attempt per outer epoch of 100 inner epochs.

The gate is on the *previous* cycle, evaluated before the next edit is made. `src/trainer.py:1312-1345` computes the cycle score as the mean of the last 25% of `self.stage_rewards`, where each entry is one inner epoch's mean episode reward (appended at `src/trainer.py:1292`). `src/trainer.py:1351` submits it, and `EvolutionSnapshotManager.add_snapshot` (`src/designer.py:307-380`) sets `is_new_best` only on strict improvement, `if avg_reward > best_reward`. On failure it increments `consecutive_no_improvement` and records the rejected edit in `failed_evolution_attempts`, which is then fed back into the designer's prompt (`src/designer.py:687-691`) so it knows what has already been tried and failed.

Three consequences follow. Rollback (`src/trainer.py:1387-1396`):

```python
best_snapshot = self.snapshot_manager.get_best_snapshot()
if not is_new_best and best_snapshot is not None:
    self.operation_bank = OperationBank.from_dict(
        best_snapshot.operation_bank_dict, encoder=self.op_encoder)
```

Only the skill bank is restored. The controller weights and the optimizer state trained under the rejected bank are kept, so a revert is partial. Early stop (`src/trainer.py:1364-1381` calling `should_stop_evolving`, `src/designer.py:428-445`): training breaks out of the outer loop when `total_evolves >= max_evolves` or when the patience is exhausted, with defaults `max_designer_evolves = 6` and `designer_early_stop_patience = 3` (`src/config.py:86-87`). Neither is overridden on the command line, so the released LoCoMo recipe — which passes `--outer-epochs 10` — in fact stops at 6 evolution rounds at the latest. And case reuse (`src/trainer.py:1429-1439`): after a rollback the designer re-analyses the *best snapshot's saved* hard cases rather than the rejected cycle's fresh ones.

Edits are further constrained at application time (`src/designer.py:1483-1600`): only `add_new` and `refine_existing` are accepted, a new skill's `update_type` must be `insert` or `update` — "only 'insert' and 'update' are allowed" — the instruction template must be non-empty and must not contain `{session_text}` or `{retrieved_memories}`, and the change list is truncated to `max_changes`.

**3. Does any score from a later task feed back into the update?** No. Two separate loops confirm it. For the controller, the reward is the score of the same trace's own queries computed immediately after that trace's memory was built (`src/trainer.py:542-551`), and GAE resets at the episode boundary (`src/controller.py:157-160`), so no gradient ever crosses from one trace to another. For the designer, the gate compares the mean reward of the cycle's *own* freshly sampled episodes against the best previous cycle's value. Those episodes are sampled uniformly from the same six-conversation pool by `conv_idx = np.random.randint(len(train_data))` (`src/trainer.py:1656`), so the comparison is same-distribution and, in expectation, over the same handful of conversations that motivated the edit. There is no forward-looking term and no discount across cycles.

The nearest exception is the ALFWorld pair protocol, which does score the memory on instances it was not built from: `sample_pair` splits one draw into disjoint Batch A (memory construction) and Batch B (environment reward) file lists of the same task type (`src/data_processing/alfworld.py:80-83`), and `AlfworldPairTrainer` (`src/trainer.py:1870-2195`) builds memory from A and collects reward by running the agent on B. This is a genuine held-out score, but both halves are drawn fresh from the training pool each episode, and it still gates nothing about any earlier task.

**4. Is any earlier task ever re-evaluated?** No. A grep over every `*.py` and `*.sh` in the repo for `anchor`, `retention`, `forget`, `regress`, `revisit`, `re-eval`, `reeval`, `holdout`, `hold_out`, `held_out`, `catastrophic`, and `plasticity` returns **zero matches**. No fixed evaluation set is carried forward, and no episode is ever replayed on a later skill bank. The validation split that `split_data` constructs is never used: `grep val_data main.py` shows it created at `main.py:74`, possibly emptied at `main.py:1045` and `main.py:1048`, printed at `main.py:1049`, and never passed to `trainer.train(train_data)` at `main.py:1077` or to any gate. The one thing that lingers is the hard-case pool, and it only lingers negatively — `CaseCollector.add_case` returns early for correct answers (`src/designer.py:160-162`), so a case that a later skill bank fixes is never removed on success; it simply ages out of the window.

**5. Are result files or logs shipped?** No run artifacts of any kind. The scripts write to `./results/` and `./checkpoints/`, and neither directory exists in the clone. There are no training logs, no per-cycle scores, no trajectories, and no evolved-bank history. What *is* shipped is the endpoint: `skills/conversational_skills/` (9 files) and `skills/embodied_task_skills/` (7 files), the final evolved banks rendered as markdown with Description / Purpose / When to Use / How to Apply / Constraints sections, matching Figure 4 and Appendix C of the paper. Note that these markdown files are documentation only — no code path reads them; the runnable bank lives inside a checkpoint.

One finding worth acting on. The checkpoint format (`src/trainer.py:1740-1765`) serializes far more than weights:

```python
'controller_state_dict': self.controller.state_dict(),
'operation_bank': self.operation_bank.to_dict(),
'snapshot_manager': self.snapshot_manager.to_dict() if self.snapshot_manager else None,
'designer_state': self._get_designer_state(),
'stage_rewards': self.stage_rewards,
'training_logs': self.training_logs,
```

and `EvolutionSnapshotManager.to_dict` (`src/designer.py:782-790`) writes the full list of snapshots, each of which is `{stage_id, operation_bank_dict, avg_reward, evolution_result, analysis_cases}` (`src/designer.py:232-250`) — that is, every intermediate skill bank, the stabilized reward it achieved, the exact edit that produced it, and the serialized hard cases that motivated it. The README's March 2026 news entry announces released controller weights in a Hugging Face collection. If those files were written by this save path, they contain a short but complete harness-evolution history in exactly the (harness state, realized return, edit) shape our critic training needs. Downloading one and inspecting `snapshot_manager` is a cheap, high-value check.

**6. What stream or ordering construction exists?** None. Episodes are independent uniform draws from a fixed pool — `conv_idx = np.random.randint(len(train_data))` (`src/trainer.py:1656`) — with replacement, in parallel threads, with no curriculum, no arrival order, and no distribution shift over time. The only temporal structure in the whole system is within a trace: spans are processed sequentially and the memory bank grows across them (`src/trainer.py:527-537`). ALFWorld's `--alfworld-pair-same-type-prob 1.0` makes the sampling *less* heterogeneous, not more, by forcing both halves of every episode to come from one task template.

## Divergences between paper and code

| Item | Paper | Code |
|---|---|---|
| Hard-case difficulty | $d(q) = (1 - r(q)) \cdot c(q)$, linear in the failure count (eq. 4) | `severity * np.log1p(fail_count)` — logarithmic (`src/designer.py:954-963`) |
| Skill evolution stages | "two-stage skill evolution": analyze, then propose (Sec. 3.4) | Stage 1 runs up to `designer_reflection_cycles = 3` reflection rounds before Stage 2 (`src/designer.py:1272-1301`), and all three released scripts pass `--designer-reflection-cycles 3` |
| Exploration window | $T_{explore} = 50$ steps, $\tau_0 = 0.3$ (eq. 7) | default `new_action_bias_steps = 50` (`src/config.py:133`) but every released script passes `--new-action-bias-steps 25`; `new_action_p_min = 0.30` matches |
| Bank size | not stated | `max_ops = 30` with lowest-average-reward eviction (`src/config.py:24`, `src/operation_bank.py:225-252`) |
| Retriever | Qwen3-Embedding-0.6B, "also used as the memory retriever" (Sec. 4.1) | training scripts pass `--retriever contriever`; only the eval scripts use `qwen3-embedding-0.6b` |
| Evolution rounds | "periodically", with early stopping on patience | hard cap of 6 rounds by default, never overridden, so `--outer-epochs 10` cannot produce more than 6 (`src/config.py:86`, `src/trainer.py:1364-1381`) |
| Rollback | "roll back the skill bank to the previously best-performing snapshot" | exactly that, and only that — controller weights and optimizer state are not restored (`src/trainer.py:1387-1396`) |

None of these change the paper's conclusions, but the first three would change a reimplementation's numbers, and the last is the one that matters for reading MemSkill as a gate-and-revert system.

## Relevance to this project

### Components we can directly reuse

- **`src/designer.py:232-445` — snapshot, best-tracking, rollback, and early stop.** The cleanest implementation of a commit gate over a text harness layer in our reference set: a dataclass holding the serialized layer plus the score that layer earned, a manager that decides `is_new_best` on strict improvement, a rollback that restores the layer from the best snapshot, and a patience-based stop. The structure is criterion-agnostic; we can keep it whole and change only what `avg_reward` means.
- **`src/trainer.py:1129-1133` plus `src/controller.py:323-364` — the new-entry exploration incentive.** A minimal logit gain that lifts a newly added entry's selection mass to a target, clamped, then linearly decayed to zero over a fixed window and retired. This is the answer to a problem we will hit as soon as our harness agent adds an entry that a selector must then choose, and it costs no architectural change.
- **`src/designer.py:109-196` — the rolling failure pool.** Keyed by query identity, increments a per-case failure count on repeat, prunes by age window then by capacity, and survives across update cycles. A ready mechanism for deciding which past failures drive the next edit.
- **`src/designer.py:368-374` and `src/designer.py:687-700` — rejected-edit feedback.** Failed edits accumulate and are formatted into the next proposal prompt so the proposer sees what has already been tried and failed. Cheap, and a direct improvement over proposing blind.
- **`src/operation_bank.py:205-258` — capacity with reward-based eviction.** A concrete answer to harness bloat that is not a prompt-level audit: cap the layer, evict the lowest-average-reward used entry.
- **`skills/conversational_skills/*.md` — the edit format.** Description / Purpose / When to Use / How to Apply / Constraints is a good default schema for a skill-layer edit, and `src/designer.py:1566-1572` shows the validation worth keeping (non-empty template, no context placeholders).

### Components we need to modify

- **The gate criterion (`src/trainer.py:1312-1345`).** Replace the mean over the last quarter of the current cycle's own training episodes with a score on an anchor set of previously passed tasks. The plumbing stays; only the number that enters `add_snapshot` changes.
- **Rollback scope (`src/trainer.py:1387-1396`).** Restoring the skill bank while keeping the controller trained against the rejected bank leaves the two out of sync. Any revert we build should restore both, or deliberately decide not to and say why.
- **Episode sampling (`src/trainer.py:1656`).** Uniform sampling with replacement from a six-item pool has to become an ordered stream with a shifting distribution before any of our questions can even be asked of this code.
- **The failure pool's forgetting rule (`src/designer.py:137-158`).** Cases age out by epoch window, which is what makes retention unmeasurable: a task fixed three cycles ago leaves the pool and can never be regressed against. We want the opposite — solved tasks retained as anchors.
- **Reward horizon (`src/controller.py:157-160`).** The episode-boundary reset is correct for their objective and exactly what we need to remove for ours.

### Components that don't apply

- **Everything memory-specific**: `src/memory_bank.py`, the span segmenter, the retriever wiring. Their memory bank is per-trace scratch space, not our long-term memory layer, and their span construction solves a long-document problem we do not have.
- **The executor path** (`src/executor.py`): our harness edits are applied to files a model reads, not compiled into a one-shot extraction prompt.
- **Token masking, supervised warm start, distributed training**: absent here and irrelevant — the trained object is three small networks over frozen embeddings.
- **ALFWorld and LoCoMo data plumbing**: benchmark-specific.

### Key code snippets worth studying

| Location | Why |
|---|---|
| `src/trainer.py:1305-1500` | the entire evolution cycle in one block: stabilized score, gate, early stop, rollback, feedback, case preparation, retrying edits. Read this first. |
| `src/designer.py:307-380` | `add_snapshot` — the gate decision and the failed-attempt bookkeeping. |
| `src/designer.py:232-250`, `src/designer.py:782-790` | the serialized evolution history; the shape of the data a critic would train on. |
| `src/controller.py:323-364` | the new-entry logit gain, including the clamping that keeps it from dominating. |
| `src/controller.py:119-165` | GAE with the episode-boundary reset — the code-level statement of this system's horizon. |
| `src/trainer.py:1642-1712` | `_run_single_episode` and `_initialize_memory_bank`: the eight lines that make the memory per-trace. |
| `src/data_processing/alfworld.py:63-108` | the disjoint Batch A / Batch B draw — the only held-out scoring anywhere in the repo. |
| `src/designer.py:1483-1600` | `apply_evolution`: what an edit is allowed to be, and every rejection rule. |
