# Repo Analysis: MemEvolve

**Path:** `refs/related-work/MemEvolve/repo/` · **Source:** https://github.com/bingreeky/MemEvolve · **Analyzed:** 2026-09-15 (commit `6035d56`, dated 2026-05-05)

## Overview

- **Paper:** *MemEvolve: Meta-Evolution of Agent Memory Systems* (see `paper_analysis.md`)
- **Framework:** custom, built on a vendored copy of Flash-Searcher. Apache-2.0, Python 3.10, 146 Python files and roughly 33.6k lines across 288 tracked files. The clone is 174M on disk but 88M of that is a cached `sentence-transformers/all-MiniLM-L6-v2` checkpoint committed under `Flash-Searcher-main/storage/models/`; the code itself is small (EvolveLab 211K, MemEvolve 168K). A second vendored dependency, `mini-swe-agent`, is used only to auto-repair generated provider code during validation.
- **Training algorithm:** none. There is no gradient step anywhere. The outer loop is a language model reading trajectory logs and writing a new Python file; the inner loop is an ordinary memory provider ingesting trajectories.
- **Base model:** provider-agnostic through the Flash-Searcher model layer. `MemEvolve/config.py` defaults the analysis and generation models to `os.getenv("ANALYSIS_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))` (`MemEvolve/core/auto_evolver.py:70-71`); the README recommends `claude-sonnet-4.5` or `gpt-5` for the evolution models, while the paper's experiments used GPT-5-mini.
- **Key innovation in code:** a memory system is one Python file implementing three methods, registered by editing two source files. `MemEvolve/phases/memory_creator.py:86-286` literally writes a new provider file, appends an enum member and a `PROVIDER_MAPPING` entry to `EvolveLab/memory_types.py`, and appends a config block to `EvolveLab/config.py`. Architectural evolution is source-code editing with marker-comment insertion points.

**Headline findings for us.**

1. **The released selection protocol is stricter than the paper's.** The paper reports one 60-task fitness batch per candidate (40 fresh, 20 reused). The code runs *two* stages: a tournament in which every candidate is scored on **exactly the tasks the diagnosis read** (`auto_evolver.py:906-911`, `tasks=batch_indices`), and only then a final on a mixed set. The first stage is pure current-batch scoring.
2. **No evolution logs, round summaries, or result files are shipped.** The orchestrator writes `round_summary.json` per round (`auto_evolver.py:973-977`) and a global state file, but none are in the release. There is nothing here to pretrain a critic on.
3. **Memory content is wiped at the start of every outer round** (`auto_evolver.py:815-831`), so no memory state ever survives an architectural edit.
4. **The generation prompt forbids shrinking.** `MemEvolve/prompts/generation_prompt.yaml:23,42,145` require "Complexity must be >= template system (no simple fallbacks or trivial modifications)". The search has a growth ratchet and no size term in the accept rule.
5. **The TaskCraft runner the paper's main results depend on is not in the repo.** `MemEvolve/config.py:25-31` maps `taskcraft` to `run_flash_searcher_mm_taskcraft.py` and `coin_flip` to `run_coin_flip.py`; neither file exists. Only the GAIA, WebWalkerQA, and xBench runners ship. No benchmark data ships either (`Flash-Searcher-main/data/` contains one README and one sampling script).

## File Structure Map

```
README.md                          MemEvolve's own README (ICML'26 banner, setup, provider table, evolve commands)
LICENSE                            Apache 2.0
Flash-Searcher-main/               everything of interest lives here
  README.md                        the upstream Flash-Searcher README, unchanged
  evolve_cli.py                    719 lines; subcommands analyze / generate / create / validate / delete /
                                   run-all / status / auto-evolve / list
  run_flash_searcher_mm_gaia.py    GAIA runner: solve -> judge -> take_in_memory
  run_flash_searcher_webwalkerqa.py  same shape for WebWalkerQA
  run_flash_searcher_mm_xbench.py    same shape for xBench-DeepSearch
  lasj.py                          judge_equivalence, the language-model judge
  eval_utils.py, utils.py          run-directory creation, token counting
  base_agent.py                    agent base class
  FlashOAgents/agents.py           the agent loop; :600-613 builds a MemoryRequest and calls provide_memory
  EvolveLab/
    base_memory.py                 BaseMemoryProvider ABC — three abstract methods (:18, :31, :44)
    memory_types.py                MemoryType enum (:26-38) and PROVIDER_MAPPING (:48-63)
    config.py                      global ./storage paths and per-provider hyperparameters
    providers/                     13 provider implementations + a template (inventory below)
  MemEvolve/
    config.py                      dataset paths, runner map, and the evolution defaults (:52-56)
    core/auto_evolver.py           1006 lines; the multi-round orchestrator (the file that matters)
    core/memory_evolver.py         473 lines; one round's analyze -> generate -> create -> validate chain
    phases/phase_analyzer.py       337 lines; reads task logs, computes statistics, fills the analysis prompt
    phases/phase_generator.py      386 lines; turns the diagnosis into a design specification
    phases/memory_creator.py       468 lines; writes the provider file and registers it in two source files
    phases/phase_validator.py      836 lines; isolated-environment smoke test + auto-repair loop
    validators/swe_agent_validator.py  378 lines; mini-swe-agent wrapper used to fix broken generated code
    utils/run_provider.py          113 lines; subprocess launch of a benchmark runner
    utils/trajectory_tools.py      716 lines; per-task feedback vector and batch aggregation
    prompts/analysis_prompt.yaml   338 lines; the diagnosis prompt
    prompts/generation_prompt.yaml the redesign prompt
  mini-swe-agent/                  vendored, used only by the validator
  storage/models/                  88M cached sentence-transformers checkpoint
  data/                            gaia/README.md and webwalkerqa/sample_webwalkerqa.py only — no data
```

## Component Inventory

### 1. Entry points

| Entry point | Location | What it does |
|---|---|---|
| `evolve_cli.py auto-evolve <dataset>` | `evolve_cli.py:380-454`, parser at `:623-700` | Runs the full multi-round loop. Flags: `--num-rounds`, `--provider`, `--num-systems`, `--task-batch-x`, `--top-t`, `--extra-sample-y`, `--creativity`, `--use-pareto-selection`. |
| `evolve_cli.py analyze / generate / create / validate` | `evolve_cli.py:19,57,144,170` | The same four phases run by hand. |
| `run_flash_searcher_mm_gaia.py` and siblings | top level | Evaluate one memory provider on a task list. Flags include `--memory_provider`, `--task_indices`, `--concurrency`, `--sample_num`, `--max_steps`, `--disable_memory_evolution`. |

The README's recommended command is `python evolve_cli.py auto-evolve gaia --num-rounds 3 --provider agent_kb --num-systems 3 --task-batch-x 40 --top-t 2 --extra-sample-y 20 --creativity 0.5`, which reproduces the paper's configuration: an Agent-KB start, three rounds, three descendants.

The in-code defaults are four times smaller (`MemEvolve/config.py:52-56`):

```python
EVOLVE_TASK_BATCH_X = 20         # each round: run x tasks per provider
EVOLVE_TOP_T = 2                 # select top t systems after first evaluation
EVOLVE_EXTRA_SAMPLE_Y = 5        # reuse y tasks from first batch for finalists
EVOLVE_GENERATED_M = 3           # number of systems to generate per round (m)
EVOLVE_RANDOM_SEED = 42
```

### 2. The outer loop, step by step

`auto_evolver.py:788-1000` is the whole meta-evolution. Per round $r$:

**Step 0 — wipe memory** (`:813-838`). With `clear_storage_per_round: bool = True` (`:68`), the incumbent's storage directory is renamed away:

```python
storage_dir = Path(f"./storage/{current_provider}")
if storage_dir.exists():
    backup_name = f"{current_provider}_round{r:02d}_{timestamp}"
    shutil.move(str(storage_dir), str(backup_dir))
    print(f"[Fairness] Provider will start with empty storage")
```

Renamed, not deleted — but nothing ever reads the backup. This is the code form of the paper's "initialized as an empty memory at the beginning of iteration $k$".

**Step 1 — collect the diagnosis batch** (`:843-859`). `batch_indices = self._select_tasks(dataset_cursor, self.task_batch_x)`, where `_select_tasks` (`:784-786`) is just `list(range(start_idx, start_idx + count))` — a contiguous, monotone window over the dataset. The incumbent runs on it, and its memory fills up as it goes.

**Step 2 — diagnose and generate** (`:861-894`). `_run_memory_evolver` chains analyze, generate, create, validate; survivors are `validated_systems`.

**Step 3 — tournament on the same tasks** (`:895-917`):

```python
providers_to_eval = [current_provider] + validated_systems
...
eval_results = self._evaluate_providers_parallel(
    providers=providers_to_eval,
    tasks=batch_indices,          # <- identical to Step 1
    round_dir=round_dir / "eval_batch",
)
top_candidates = self._select_top(eval_results, k=self.top_t)
```

**Step 4 — finals on a mixed set** (`:920-945`):

```python
sampled_tasks = random.sample(batch_indices, min(self.extra_sample_y, len(batch_indices)))
new_tasks = self._select_tasks(dataset_cursor + self.task_batch_x, self.task_batch_x)
finalist_tasks = sampled_tasks + new_tasks
...
winner = self._select_top(finalist_results, k=1)[0] if finalist_results else current_provider
```

With the README's flags this is 20 already-seen tasks plus 40 fresh ones — the paper's 60-task batch.

**Step 5 — advance and commit** (`:947-996`). `dataset_cursor += self.task_batch_x * 2` (`:948`), a round summary is written to `round_dir / "round_summary.json"` (`:973-977`), and `current_provider = winner` (`:979`).

### 3. Selection and the accept rule

Default selection is not Pareto. `use_pareto_selection: bool = False` (`:67`), and `_select_top` (`:763-782`) falls through to a plain lexicographic sort:

```python
scored.append((provider, accuracy, total_tokens))
scored.sort(key=lambda x: (-x[1], x[2]))
```

Accuracy first, token count as tie-break; latency is not consulted. With the flag on, `_pareto_select_top` (`:623-757`) does non-dominated sorting over (accuracy up, total tokens down, execution time down) with a scalarized tie-break weighting 0.6 accuracy, 0.25 tokens, 0.15 time. A failed evaluation is not skipped but scored at the floor: `{"accuracy": 0, "tokens": {"total_tokens": 999999}}` (`:553`).

**There is no threshold.** Nothing compares the winner's score against any absolute bar, any previous round's score, or any held-out set. The only protection against a bad round is that the incumbent is entered into the tournament alongside its descendants (`:895`), so if every descendant scores worse the incumbent survives. That is a comparative accept, not a gate, and it has a defect: the incumbent enters Step 3 carrying the memory it built during Step 1 **on those same tasks**, while every generated candidate starts from empty storage — nothing between `:859` and `:906` resets it. The comparison is not like-for-like, and it favours the incumbent.

### 4. Validation is a compile gate, not a performance gate

`phase_validator.py` checks that generated code is runnable, not that it is good. A static check requires the three method definitions to be present (`:253`, `:260`, `:267`: `if "def provide_memory" not in code`, `"def take_in_memory"`, `"def initialize"`), then a smoke run over `SMOKE_TASK_LIMIT = 5` tasks (`MemEvolve/config.py:33-34`) inside an isolated copy of the tree (`:332-372`). Failures trigger up to `max_fix_attempts: int = 3` (`:43`, loop at `:170-197`) repair passes driven by the vendored mini-swe-agent (`validators/swe_agent_validator.py`). A system that survives is synced into the live tree (`:814`). Nothing in this phase looks at accuracy.

### 5. The diagnosis signal

`prompts/analysis_prompt.yaml` (338 lines) drives Step 2. It names **three** operations, not four: "the three core operations — **PROVIDE**, **TAKE-IN**, and **MANAGEMENT**" (`:7`), with per-operation sections for current behaviour (`:169-190`), implementation review (`:202-227`), effectiveness (`:235-265`), and recommendations (`:279-300`). Encode and store are never separated, matching the collapsed interface. Sampling is explicit: "Select **3-4 failed tasks** for in-depth analysis" and "Select **2-3 successful tasks** for comparison" (`:142-143`) — so the diagnosis reads at most seven trajectories out of the batch. The prompt ends with the literal token `READY_TO_GENERATE` (`:307`).

`prompts/generation_prompt.yaml` constrains the redesign. Three lines make growth mandatory: "Adopt 2-3 innovative points from the recommendations, or create your own innovations. Complexity must be >= template system." (`:23`); "Complexity must be >= template system (no simple fallbacks or trivial modifications)" (`:42`); "Innovation >= template complexity" (`:145`). There is no rule permitting simplification, and no size budget anywhere in the loop.

### 6. The per-task feedback vector

`utils/trajectory_tools.py:145-172` builds the per-task record that the aggregator averages:

```python
feedback = {
    "accuracy": 1 if is_correct else 0,
    "status": task_data.get("status", "unknown"),
    "score_raw": task_data.get("score"),
    "steps": {..., "memory_guidance": {"count": ..., "items": ..., "ratio": ...}},
    "tools": {"total_calls": ..., "unique_tools": ..., "error_like_calls": ...},
    "text": {"answer_len": ..., "question_len": ..., "avg_obs_len": ...},
}
```

`aggregate()` (`:175-235`) averages these into the `summary` dict that selection reads: `accuracy`, step counts including a `memory_guidance_ratio`, tool-call statistics, text lengths, and token totals. Note that latency is not in this record — `_pareto_select_top` recovers it separately via `_compute_avg_execution_time`. Note also `error_like_calls` and `memory_guidance_ratio`: these are exactly the kind of state-level features a critic over harness files might use, and they are computed here for free.

### 7. Environment and tool interaction

Evaluation is out-of-process. `utils/run_provider.py:48-63` builds a subprocess command:

```python
cmd = [
    "python", str(runner_path),
    "--infile", str(dataset_file),
    "--outfile", str(outfile),
    "--task_indices", task_arg,
    "--memory_provider", provider_name,
    "--concurrency", "1",
    "--direct_output_dir", str(output_dir),
]
```

`--concurrency 1` is hard-coded, so every evolution run is strictly sequential within a provider. Providers are run in parallel across processes with `max_workers: int = 3` (`auto_evolver.py:66`, pool at `:560-561`).

### 8. Score-before-update at the task level

All three runners share one shape. In `run_flash_searcher_mm_gaia.py`, the agent solves (`:294`), the judge runs (`:307-315`, `is_correct = (judgement_str == "correct")`), and only then is the trajectory ingested (`:319-338`):

```python
if memory_provider and enable_memory_evolution:
    trajectory_data = TrajectoryData(
        query=original_question,
        trajectory=agent_messages,
        result=result.get("agent_result"),
        metadata={"task_id": task_id, "status": "success",
                  "is_correct": is_correct, "full_query": question},
    )
    success, msg = memory_provider.take_in_memory(trajectory_data)
```

The error path does the same at `:369-388`. WebWalkerQA (`:295`/`:308`, `:352`/`:365`) and xBench (`:286`/`:297`, `:335`/`:346`) match. `--disable_memory_evolution` (`:668-669`) turns ingestion off entirely, which is how the no-memory baseline is produced.

The ordering is exactly score-before-update, and the verdict is handed to the memory as `is_correct`. But the call is **unconditional**: every task is ingested whatever the verdict. Whether the verdict is used at all is left to each provider — some filter on it, some do not.

On the read side, `FlashOAgents/agents.py:600-613` assembles a `MemoryRequest` from the task, the formatted current context, a phase status, and the step number, then calls `provide_memory`; the returned items are split into text guidance and tool definitions. The phase enum has BEGIN and IN values (`EvolveLab/memory_types.py:11-15`), so memory is consulted both at task start and mid-trajectory.

### 9. The twelve memory systems in the design space

`EvolveLab/memory_types.py:26-38` defines thirteen enum members and `:48-63` maps each to a class and module. The providers, by size:

| Provider file | Lines | Table 1 row | Store | Manage in code |
|---|---|---|---|---|
| `dilu_memory_provider.py` | 296 | IV. DILU | vector database | none |
| `agent_workflow_memory_provider.py` | 318 | V. Agent Workflow Memory | vector database | none |
| `generative_memory_provider.py` | 321 | III. Generative | vector database | none |
| `voyager_memory_provider.py` | 349 | I. Voyager | vector database | none |
| `dynamic_cheatsheet_provider.py` | 364 | VII. Cheatsheet | JSON | none |
| `skillweaver_provider.py` | 389 | VIII. SkillWeaver | tool library | skill pruning |
| `mobilee_provider.py` | 409 | VI. Mobile-E | vector database | none |
| `expel_provider.py` | 447 | II. ExpeL | vector database | none |
| `memp_memory_provider.py` | 507 | XI. Memp | JSON | failure-driven adjustment |
| `evolver_memory_provider.py` | 687 | XII. EvolveR | JSON | update and pruning |
| `agent_kb_provider.py` | 804 | X. Agent-KB | hybrid database | deduplication |
| `lightweight_memory_provider.py` | **1596** | — (evolved) | JSON under `./storage/lightweight_memory` | short-term eviction by cap |
| `cerebra_fusion_memory_provider.py` | **1601** | — (evolved) | JSON graph database | node/edge pruning, consolidation |
| `base_provider_template.py` | 174 | — | — | the skeleton new systems are grown from |

**Row IX of Table 1, G-Memory, has no provider file.** Eleven of the twelve taxonomy rows are implemented; the README's own table says "11 baseline memory systems ... and 2 evolved systems". So the paper's "twelve representative memory systems" is one more than the release contains.

The size jump is the clearest artifact of the growth ratchet: the two evolved systems are 1596 and 1601 lines, roughly double the largest hand-written provider and five times the median.

Provider configuration is global and path-hard-coded (`EvolveLab/config.py:10-112`): `STORAGE_BASE_DIR = "./storage"` with one fixed subdirectory per provider. The evolved Cerebra system's config block (`:54-70`) is the most elaborate in the file and shows what the evolution produced as tunable state: `"min_score": 0.22`, `"semantic_edge_threshold": 0.75`, `"max_neighbors_expand": 3`, `"enable_graph_expansion": True`, `"enable_tool_memory": True`, `"consolidation_interval": 50`, `"search_weights": {"text": 0.2, "semantic": 0.8}`. The Lightweight system's block (`:44-53`) is the other end: `max_strategic_memories: 30`, `max_operational_memories: 30`, `max_shortterm_items: 15`, `shortterm_provision_interval: 5`, `enable_longterm_provision: False`.

### 10. Components that are absent

- **No reinforcement learning, no token masking, no supervised phase, no multi-GPU code.** The repo trains nothing. The only model weights present are the frozen sentence-transformers encoder used for semantic retrieval.
- **No released evolution artifacts.** The orchestrator writes `round_summary.json`, per-round checkpoints, and a global state file at runtime, but the release contains none of them. The only JSON in the tree is mini-swe-agent's own test fixtures (`mini-swe-agent/tests/test_data/{github_issue.traj.json,local.traj.json,results.json}`) and the cached encoder's config files. The README's manual-mode example points at `./test/test_traj/lightweight_memory`, a directory that does not exist in the clone.
- **No intermediate architectures.** Figure 6 names seven systems along the evolutionary path (AKB Mem, Adaptive Decision, Meta Mem, Domain Mem, Riva, Omni-Mem, Cerebra). Only the two endpoints ship as code: Lightweight and Cerebra. Riva, the paper's other highlighted discovery (Figure 9), is not in the repo.
- **No benchmark data and no TaskCraft runner**, as noted above. The README gives download instructions for GAIA, WebWalkerQA, and xBench but says nothing about TaskCraft, which is the benchmark all of the paper's evolution runs used.
- **The authors flag the release as provisional**: "This is not the final release. A more robust version is still being organized." (README). The auto-evolve loop is checkpoint-and-resume precisely because it is expected to crash: "If auto-evolution hits an error, it will exit and save a checkpoint. Just fix the issue manually (we'll show you how), then re-run the same command."

## Answers to the six questions we came with

**1. What state persists across tasks at each loop level?**

*Inner loop (within a round):* the provider's storage directory under `./storage/<provider>` (`EvolveLab/config.py:10`), written by `take_in_memory` after each task. This persists across every task in a round, and — because nothing resets it between Step 1, Step 3, and Step 4 — it also persists across the three evaluation stages of a round.

*Outer loop (across rounds):* only the winning provider's **identity and source code**. `current_provider = winner` (`auto_evolver.py:979`) plus the generated provider file, its enum entry, and its config block. Memory content does not persist: Step 0 renames it away (`:815-831`). So an architecture is always judged on what it can build from scratch in one round, never on what it holds after many.

**2. What signal triggers or gates an update?**

*Inner loop:* nothing gates it. `take_in_memory` is called unconditionally after every task, success or failure (`run_flash_searcher_mm_gaia.py:319-338` for the success path, `:369-388` for the error path). The judge verdict is passed along in `metadata["is_correct"]` but the ingestion happens either way.

*Outer loop:* the trigger is a fixed schedule — one architectural edit per round, always. The gate is a ranking, not a threshold: `_select_top` sorts on `(-accuracy, total_tokens)` (`:781`) and takes the top one. The incumbent is in the ranking (`:895`), which gives an implicit revert, but there is no absolute bar, no significance test, and no check against anything outside the round.

**3. Does any score from a later task feed back into the update?**

No. The objective is the aggregate accuracy over the evaluation batch and nothing else. Step 3 scores candidates on `batch_indices`, the identical set the diagnosis read (`:906-911`) — pure current-batch scoring, the greedy case exactly. Step 4 adds 40 tasks drawn from the *next* contiguous window (`:922`, `dataset_cursor + self.task_batch_x`), which is a genuine one-step lookahead and better than the paper's own description suggests. But the horizon stops there: no candidate is ever scored on a batch from a later round, and `dataset_cursor += self.task_batch_x * 2` (`:948`) means round $r+1$ starts past everything round $r$ touched.

**4. Is any earlier task ever re-evaluated?**

Within a round, yes, and twice over: Step 3 re-runs the whole tournament on the Step 1 tasks, and Step 4 resamples `extra_sample_y` of them (`:921`). Across rounds, never. `_select_tasks` returns a contiguous forward window (`:784-786`) and the cursor only advances, so once a round is finished its tasks are gone. There is no anchor set, nothing is held out permanently, and no retention number is computed anywhere in the codebase — `trajectory_tools.py` has no concept of a previously-passed task.

**5. Are evolution logs, discovered architectures, or result files shipped?**

Architectures: partly — the two winners ship as source (`EvolveLab/providers/lightweight_memory_provider.py`, 1596 lines; `cerebra_fusion_memory_provider.py`, 1601 lines), with their configuration blocks. Logs and results: no. No `round_summary.json`, no checkpoints, no per-candidate scores, no trajectories, no benchmark outputs. The infrastructure to produce them exists (`auto_evolver.py:950-977`) and was clearly used, but nothing was released.

**6. How are the twelve memory systems laid out in the design space?**

One abstract base class with three abstract methods (`EvolveLab/base_memory.py:18,31,44`) — `provide_memory` (retrieve), `take_in_memory` (encode and store fused), `initialize`. The paper's fourth operation, manage, has no slot; where it exists it is a private method called from inside `take_in_memory` or on a counter. Systems are selected by a string flag (`--memory_provider`) resolved through `PROVIDER_MAPPING` (`memory_types.py:48-63`), and configured from one global dictionary (`EvolveLab/config.py:12-112`). Eleven of Table 1's twelve rows have an implementation; G-Memory does not. Three benchmark runners bind any provider to any of GAIA, WebWalkerQA, or xBench with judge-then-ingest ordering already correct.

## Relevance to this project

### Components we can directly reuse

- **`EvolveLab/` as a long-term-memory layer with eleven ready-made variants.** `base_memory.py` (58 lines), `memory_types.py` (60 lines), `config.py`, and `providers/` give a clean, small interface with a dozen behavioural alternatives already written. If we want a memory layer whose contents an editor can rewrite, this is a much shorter path than building one, and the Apache-2.0 licence permits it.
- **The judge-then-ingest ordering in the runners** (`run_flash_searcher_mm_gaia.py:307-338`). This is score-before-update implemented correctly, with the verdict threaded into the memory call. The pattern transfers unchanged.
- **`utils/trajectory_tools.py:130-235`.** The per-task feedback record and its aggregation are exactly the shape a per-step reward needs, and two of its fields (`memory_guidance_ratio`, `error_like_calls`) are candidate state features for a critic.
- **`phases/memory_creator.py:86-286`.** A worked example of a language model safely editing live source: write a file, insert an enum member and a mapping entry at marker comments, append a config block, with a matching `delete_memory_system` (`:287-430`) for rollback. Any harness agent that edits code needs this machinery.
- **`phases/phase_validator.py`.** Static presence check plus isolated-environment smoke test plus bounded auto-repair is a reasonable pre-gate for generated code, independent of any performance gate we add.

### Components we need to modify

- **The scoring protocol.** Delete Step 3, or at minimum score it on tasks the diagnosis did not read. As written (`auto_evolver.py:906-911`) it is the greedy case we are arguing against.
- **The storage reset.** `clear_storage_per_round=True` (`:68`) has to go if we want to study what a long-lived memory does to a long-lived architecture. Turning it off is a one-line change the authors already support (`:840` prints "Storage clearing disabled - providers keep previous knowledge"), but then the incumbent-versus-candidate comparison becomes unfair in the opposite direction and needs a redesign.
- **The task cursor.** `_select_tasks` (`:784-786`) must be replaced by a sampler that can draw from an anchor set of previously passed tasks, and `_select_top` must be replaced by an accept rule that checks the anchor set before promoting a winner.
- **The generation prompt.** `generation_prompt.yaml:23,42,145` forbids simplification. For our purposes the constraint should be inverted or at least made two-sided, and a size term added to the accept rule.
- **Concurrency.** `--concurrency 1` is hard-coded in `run_provider.py:57-58`; a stream experiment at any scale will need that parameterized.

### Components that don't apply

- The vendored `mini-swe-agent` and `validators/swe_agent_validator.py` — a code-repair loop for generated providers, orthogonal to our question.
- `FlashOAgents/` and the Flash-Searcher DAG planner — a specific agent scaffold; the memory interface is the portable part.
- The Pareto selector (`:623-757`) — cost and latency as objectives are out of scope for us per `docs/framing.md`, and it is off by default anyway.
- The 88M cached encoder under `storage/models/` — an artifact of committing a Hugging Face cache.

### Key code snippets worth studying

| What | Where |
|---|---|
| The entire outer loop, Step 0 through Step 5 | `Flash-Searcher-main/MemEvolve/core/auto_evolver.py:788-1000` |
| The two-stage selection and the identical-batch tournament | `auto_evolver.py:895-945` |
| Memory wipe between rounds | `auto_evolver.py:813-838` |
| The accept rule (default and Pareto) | `auto_evolver.py:623-782` |
| Score-before-update at the task level | `run_flash_searcher_mm_gaia.py:294-338` |
| The three-method memory interface | `EvolveLab/base_memory.py:10-58` |
| Per-task feedback vector and aggregation | `MemEvolve/utils/trajectory_tools.py:130-235` |
| Safe source-code registration of a generated component | `MemEvolve/phases/memory_creator.py:86-286` |
| The monotone-growth constraint | `MemEvolve/prompts/generation_prompt.yaml:23,42,145` |
| What an evolved memory system actually looks like | `EvolveLab/providers/cerebra_fusion_memory_provider.py` (1601 lines) and its config at `EvolveLab/config.py:54-70` |
