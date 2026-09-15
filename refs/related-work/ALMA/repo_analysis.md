# Repo Analysis: alma

**Path:** `refs/related-work/ALMA/repo/` · **Source:** https://github.com/zksha/alma · **Analyzed:** 2026-09-15 (commit `7f78ac8`, 2026-04-07, "Update README") · **License:** Apache 2.0 · **Size:** 7.5 MB excluding `.git`, of which 5.0 MB is `misc/` figures; about 17,700 lines of Python, roughly 11,000 of which are the vendored BALROG environment code.

## Overview

- **Paper:** Learning to Continually Learn via Meta-learning Agentic Memory Designs (see `paper_analysis.md`)
- **Framework:** custom and small. No agent framework, no orchestration library. Dependencies are `openai`, `rich`, `jsonschema`, `scipy`, `dotenv`, `PyYAML` (`requirements.txt`); the environments live in two Docker images built from `envs_docker/alfworld/` and `envs_docker/BALROG/`.
- **RL algorithm:** none. Nothing is trained, no gradients, no learned parameters. The search is a prompted proposer plus a scored archive.
- **Base models:** proposer GPT-5 (`training.sh`, `--meta_model gpt-5`), inner agent GPT-5-nano at low effort (`--execution_model gpt-5-nano/low`), memory-internal calls default to GPT-4o-mini and `text-embedding-3-small` (`evals/utils/hire_agent.py:24`, `:119`).
- **Key innovation as implemented:** a memory design is a single Python file implementing two async methods; the outer loop copies that file into a fixed slot inside a container, runs a collect-then-deploy evaluation, and files the score in a JSON archive that also drives parent selection.

**What ships and what does not.** The repository contains the outer loop, the archive, the four environment adapters, the container runner, five baseline memory designs, and the exact commands used for the paper (`training.sh`, `testing.sh`). It does **not** contain the discovered designs named in `testing.sh` (`53cee295` for ALFWorld, `70430b60` for TextWorld, `7e79483e` for Baba Is AI, `0892408e` for MiniHack): `memo_archive/` holds only `baseline/`. The README points to a Google Drive folder for "the learned memory designs and logs required to give the results presented in our work". One exception, apparently accidental and useful: `evals/memo_test/memo_test.py` is the staging slot that each candidate is copied into, and the checked-in copy is a complete discovered TextWorld design (`TextWorldMemory`, 756 lines, six sub-layers) left over from a run — see below.

## File Structure Map

```
run_main.py              (66)  entry point; --status train runs the search, anything else evaluates one stored design
training.sh                    the four paper search runs, one per domain
testing.sh                     the five paper test runs (four in-distribution, one ALFWorld distribution shift)
eval_in_container.py    (109)  builds the docker run command; copies env adapter + candidate design into the image
core/
  meta_agent.py         (323)  the outer loop: select -> reflect -> generate -> debug -> evaluate -> record
  meta_agent_prompt.py  (532)  benchmark descriptions and tool cheatsheets (Chroma, networkx, agent/embedding helpers)
  memo_manager.py       (203)  the archive: file storage, scores, parent links, visit counts, sampling
evals/                         everything below is bind-mounted into the container as /opt/evals
  launch.py             (260)  in-container entry: load candidate, run all tasks, aggregate score, write eval log
  workflows/agent_workflow.py (301)  collection and deployment loops, task ordering, retrieve/update call sites
  agents/memo_structure.py (46)  the search space: Sub_memo_layer and MemoStructure abstract classes
  agents/base.py                 model client and the global token tracker
  utils/hire_agent.py   (244)  Agent and Embedding helpers offered to model-written designs
  eval_envs/base_envs.py         Basic_Recorder, the only object a memory design sees
  memo_test/memo_test.py (756)  staging slot for the candidate under evaluation (currently holds a discovered design)
envs_archive/
  alfworld_envs.py      (220)  ALFWorld adapter; split by TASK_MAP over valid_seen / valid_unseen
  textworld_envs.py     (280)  BALROG TextWorld adapter; 50% learning split
  babaisai_envs.py      (376)  BALROG Baba Is AI adapter; 50% learning split
  minihack_envs.py      (298)  BALROG MiniHack adapter; 30% learning split
  prompts/ configs/              per-benchmark prompt modules and YAML configs copied into the container
memo_archive/baseline/         the five reference designs: no_mem, similarity (Trajectory Retrieval),
                               insights_traj_sim (ReasoningBank), pure_insights (Dynamic Cheatsheet), g_memory
envs_docker/                   two Dockerfiles plus a vendored copy of BALROG
```

## Component Inventory

### 1. Entry points

`run_main.py:47-65` exposes the whole system through one argument parser. `--status train` calls `MetaAgent.forward` (the search); any other status calls `run_single_memo` once on a stored design, which is how the paper's test numbers are produced. Defaults worth knowing: `--steps 10` (`run_main.py:52`), `--max_container_concurrent 5` (`:53`), `--train_size 500` (`:63`). Ten steps plus the initial from-scratch design is the paper's "11 learning steps".

`training.sh` is four commands, one per domain, all `--rollout_type batched --status train`. `testing.sh` is five commands: four `--status eval_in_distribution` in batched mode, plus one ALFWorld run with `--rollout_type sequential --status eval_out_of_distribution --update_task eval_in_distribution --update_size 70` — the only place in the repository where memory updates while it is being scored.

### 2. The memory interface (the search space)

`evals/agents/memo_structure.py` is 46 lines and is the entire contract. `MemoStructure` requires two async methods, `general_retrieve(recorder) -> Dict` (`:35`) and `general_update(recorder) -> None` (`:42`). `Sub_memo_layer` (`:7-26`) is an optional per-module base with abstract `retrieve`/`update`, a `layer_intro` string, and an optional `database` field. The only object crossing the boundary is `Basic_Recorder` (`evals/eval_envs/base_envs.py:19-24`): `init` (the initial observation and goal), `steps` (the trajectory), `reward` (the scalar score), `memory_retrieved`. A design therefore sees a full trajectory plus its outcome and nothing else — no model handle it did not create, no access to the agent loop, no ability to change prompts outside the retrieved block.

### 3. Proposer invocation

`core/meta_agent.py:146-202` is one search step for one parent. Order: bump the parent's visit count (`:160`), read the parent (`analyze_memo_structure`, `:164`), generate new code (`:168`), validate it (`:170-176`), evaluate it (`:185-197`), then record (`:199-201`). What the proposer receives is assembled in `read_memo_info` (`:45-62`): the parent's source code, its evaluation log, and — when the parent itself has a parent — an `improve_example` holding the grandparent's source, the suggestion that produced the parent, and `improve_score`. `core/meta_agent_prompt.py` supplies the per-benchmark task description plus cheatsheets for Chroma, networkx, and the provided agent and embedding helpers.

### 4. The archive and selection

`core/memo_manager.py` is the interesting file. The archive `memo_db` is a flat dictionary keyed by an eight-character identifier that is a hash of a timestamp and a fresh UUID (`:80-83`) — an identifier, not a content hash. Source files go to `memo_archive/<task>/memo_structure_<id>.py` (`:42-43`), evaluation logs to `evals/logs/<task>/<id>_<mode>.json` (`:140-146`), and the whole `memo_db` is dumped to `logs/<result_dir>_<task>_<rollout>_<steps>_<timestamp>.json` at the top of every step and again at the end with a `token_usage` entry appended (`core/meta_agent.py:277-279`, `:305-306`).

Per-entry fields: `reward`, `normalized_reward`, `visit_time`, `final_score` (`memo_manager.py:158-171`), `parent` and `improve_score` (`:148-152`), `suggestion` (`:154-156`).

```python
    def update_reward(self, memo_sha: str, reward: float, alpha = 0.5):
        def sigmoid(x, lam=1.0):
            return 1 / (1 + np.exp(-lam * x))
        self.memo_db[memo_sha]['reward'] = reward
        self.memo_db[memo_sha]['normalized_reward'] = sigmoid(reward - self.no_memo_reward)
        self.memo_db[memo_sha]['visit_time'] = 0
        penalty = np.log1p(self.memo_db[memo_sha]['visit_time'])
        self.memo_db[memo_sha]['final_score'] = self.memo_db[memo_sha]['normalized_reward'] - alpha*penalty
```

`no_memo_reward` is the paper's $f_0$; it starts at `0.0` (`:29`) and is overwritten once, before the search begins, by running the `no_mem` design on the same deployment half (`meta_agent.py:216-229`). `improve_score` is `child reward − parent reward` (`:151`), computed unconditionally.

Selection is `select_structure` (`:182-203`): a softmax over `final_score` at temperature `tau = 0.5`, then `np.random.choice(..., size=min(5, n), replace=False, p=probs)`. Two details. The function re-seeds the global numpy generator with `np.random.seed(seed)` on every call (`:183`), so the draw is deterministic given the archive contents. And the greedy ablation of the paper's Appendix C.2 is present only as three commented-out lines:

```python
        # for greedy search:
        # valid_items.sort(key=lambda x: x[1], reverse=True)
        # selected_keys = [valid_items[0][0]]
```

### 5. Validation before evaluation

`meta_agent.py:100-141`. A candidate gets up to `self.examine_trial = 3` attempts (`:41`, loop at `:113`); each attempt runs the design in `mode='test'` inside the container, and on failure the error text is fed back for a rewrite. After three failures the candidate raises and is dropped by the step's exception handler (`:141`, caught at `:299-300`). One copy-paste artifact: `:175` passes `batch_max_retrieve_concurrent = batch_max_update_concurrent`, so the retrieve concurrency argument of `examine_new_code` is ignored at that call site.

### 6. Fitness computation

Two files. `evals/workflows/agent_workflow.py:134-285` produces the records; `evals/launch.py:36-104` turns them into the number.

```python
        group_avgs = [np.mean(group) for group in rewards]
        overall_avg = np.mean(group_avgs)
        overall_se = np.std(group_avgs, ddof=1) / np.sqrt(len(group_avgs)) if len(group_avgs) > 1 else 0.0
```

The records are split into `chunk_size = 3` contiguous chunks (`launch.py:42-44`), which line up with the three replicate deployment passes because the deployment list is repeated three times contiguously (`agent_workflow.py:259-262` batched, `:236-238` sequential). So the reported standard error is across three passes, not across tasks, exactly as the paper states. A failed episode contributes `0.0` (`launch.py:48-51`, `:57-59`); any tasks beyond `3 * (record_len // 3)` are silently dropped by the integer division. The single number returned is `benchmark_overall_eval_score` (`:102`), which `meta_agent.py:200` writes into the archive as the design's reward.

The whole fitness is therefore: **the mean per-task score on the deployment half of one benchmark's learning set, with memory frozen, averaged over three replicate passes.** Nothing else enters. The collection half is never scored into the fitness — those episodes run with no memory at all and exist only to build the memory.

Scale of that set, from the released commands: ALFWorld uses `--train_size 30` (`training.sh`), and `alfworld_envs.py:161-164` takes the first 30 files of the `train` split, which `agent_workflow.py:154` halves into 15 collection and 15 deployment tasks, giving 45 scored episodes per candidate. Baba Is AI passes no `--train_size`, so the default 500 caps to the 50 percent learning split of its 52 tasks (26 tasks, 13/13). MiniHack likewise caps to its 30 percent split of the `distance` category. TextWorld passes `--train_size 1`: taken literally, `textworld_envs.py:167-168` yields a one-task learning set, and `agent_workflow.py:154` then gives `mid = 0`, an empty collection half and a single deployment task run three times. The paper describes an even split of the learning set into collection and deployment halves (Sec. 4.1, Appendix A.2); the released TextWorld command does not produce that.

### 7. Task sequences: ordering, length, held-out split

Ordering is fixed and identical for every candidate (`agent_workflow.py:152-154`):

```python
        random.seed(42)
        random.shuffle(task_list)
        mid = len(task_list)//2
```

The first half becomes the collection set, the second half the deployment set, and the two are disjoint. Because the seed is fixed inside the function, every candidate in every learning step sees the same tasks in the same order.

Splits per domain:

| Domain | Adapter | Learning / testing split | Out-of-distribution set |
|---|---|---|---|
| ALFWorld | `alfworld_envs.py:25-29`, `:157-176` | learning = first `train_size` files of the `train` split; testing = `valid_seen` | `valid_unseen`, used by the one `testing.sh` sequential run |
| TextWorld | `textworld_envs.py:151-172` | two categories (`treasure_hunter`, `the_cooking_game`), shuffled at seed 42, `train_n = int(n*0.5)` each | none — `ood_cats = []` (`:152`), so `test_out_of_domain` is empty |
| Baba Is AI | `babaisai_envs.py:120-145` | one category holding all tasks (`:121`), flat shuffle, `train_n = int(n*0.5)` | none — `ood_cats = []` (`:124`) |
| MiniHack | `minihack_envs.py:166-216` | four categories by name; only `distance` is split, `train_n = int(n*0.3)` (comment: "original 0.2, current 0.3") | `boxoban`, `corridor`, `quest` held out whole (`:174`); no released command evaluates them |

So the paper's distribution-shift experiment rests entirely on ALFWorld's `valid_unseen`. TextWorld and Baba Is AI ship with an empty out-of-distribution split; MiniHack defines one but `testing.sh` only ever asks it for `eval_in_distribution`.

The deployment sequences themselves differ by rollout type. **Batched** (`agent_workflow.py:258-282`) builds the deployment list three times over, runs all of it concurrently under a semaphore with one shared frozen memory instance (`launch.py:183-186` constructs a single memory object for batched, three for sequential), and never calls `general_update` during deployment. **Sequential** (`:216-257`) creates three independent memory snapshots and walks the full deployment list once per snapshot under `asyncio.Semaphore(1)`; within a snapshot each task retrieves first (`:62-74`) and updates after the episode (`:120-130`), which is score-before-update on an ordered sequence. The collection phase is the mirror image: all collection tasks are run concurrently with retrieval disabled, then their recorders are fed to `general_update` one at a time in completion order (`:186-215`), which is the paper's sequential memory construction.

### 8. Data pipeline and released logs

Two log products. Per evaluation, `launch.py:108-137` writes `{benchmark_eval_score, examples[{init_environment, memo_retrieved, steps, final_reward}], token_usage}`; `examples` is a stratified sample of at most three trajectories per reward bin over the bins `[0, 0.33), [0.33, 0.66), [0.66, 1.0]` (`:67-81`, `sample_size = 3` at `:28`), plus up to three error strings. Per search run, the archive JSON described in component 4. Together these are a harness-evolution log: for each of about 43 designs per domain, the code, the parent, the realized one-step delta, the proposer's suggestion, the visit count, the aggregate score, and a handful of full trajectories with the retrieved memory that produced them. Neither product is in the repository; both are in the Google Drive folder linked from the README.

### 9. Infrastructure and sandboxing

`eval_in_container.py:58-98`. Each evaluation is one `docker run --rm` against the `alfworld` or `balrog` image. The command copies the environment adapter, its prompt module and its YAML config into `/opt/evals/eval_envs/`, copies the candidate design to `/opt/evals/memo_test/memo_test.py`, and runs `python -u -m launch`. Concurrency across candidates is `asyncio.Semaphore(max_concurrent)` with default 5 (`meta_agent.py:276`, `run_main.py:53`).

One qualification of the paper's safety claim that "access confined to the sandbox" (Sec. 5): the container is started with bind mounts of the host's `evals/`, `memo_archive/` and `envs_archive/` directories and of the host `.env` file (`eval_in_container.py:91-94`), all writable. Model-written code running inside the container can therefore read the API key and write into the host archive. The checked-in `.env` contains the placeholder `OPENAI_API_KEY=your_openai_api_key_here`, so no credential is leaked by the repository itself.

### 10. Baselines

`memo_archive/baseline/` holds five files: `memo_structure_no_mem.py` (12 lines, returns nothing), `memo_structure_similarity.py` (52 lines, embedding nearest-trajectory — the paper's Trajectory Retrieval), `memo_structure_insights_traj_sim.py` (126 lines, per-trajectory success and failure insight extraction — ReasoningBank), `memo_structure_pure_insights.py` (193 lines, a single cumulative cheatsheet curated by a model — Dynamic Cheatsheet), `memo_structure_g_memory.py` (1,313 lines, hierarchical graph memory). Only three of the five are listed in `BASELINES` (`eval_in_container.py:17`), so the other two resolve to the per-task archive directory instead of `baseline/` and will not be found by identifier without being copied there first.

### 11. Absent components

No reinforcement learning, no token masking, no supervised warm start, no multi-GPU anything — there is no training of any kind in this repository, so five of the protocol's ten standard components do not exist here. Also absent, and more relevant to us: no acceptance test on a candidate's score, no revert path, no anchor set, no re-evaluation of any previously seen task, and no cost term in the objective despite the paper's cost figure (Fig. 6 is computed from the token tracker after the fact, `evals/agents/base.py` tracker, surfaced at `meta_agent.py:303`).

## The six questions, answered

| Question | Answer | Evidence |
|---|---|---|
| How is outer-loop fitness computed? | Mean per-task score over the deployment half of one benchmark's learning set, memory frozen, averaged over three replicate passes; that single scalar is the design's archive reward | `evals/launch.py:63-65`, `:101-104`; `core/meta_agent.py:199-200` |
| How is the archive stored and sampled? | Flat JSON dictionary with reward, normalized reward, visit count, final score, parent, one-step improvement, suggestion; sampling is a temperature-0.5 softmax over `normalized_reward − 0.5·log(1+visits)`, five designs without replacement | `core/memo_manager.py:148-171`, `:182-203`; dumped at `core/meta_agent.py:277-280` |
| What is the inner agent's memory interface? | Two async methods, `general_retrieve(recorder) -> Dict` and `general_update(recorder) -> None`, over a recorder holding the initial observation, the step list and the scalar reward | `evals/agents/memo_structure.py:28-46`; `evals/eval_envs/base_envs.py:19-24`; call sites `evals/workflows/agent_workflow.py:62-64`, `:120-122`, `:207-210` |
| Is any earlier task ever re-evaluated? | No. Collection and deployment halves are disjoint at a fixed seed, no collection task is ever scored, every candidate starts from an empty memory, and no task is re-run across learning steps | `evals/workflows/agent_workflow.py:152-154`, `:62-74`; `core/meta_agent.py:185-197` |
| Are the archive, discovered designs and logs shipped? | Not in git. `memo_archive/` holds only the five baselines; the four identifiers in `testing.sh` are absent; the README links a Google Drive folder. The log *writers* are in the repo, and one discovered TextWorld design survives in the staging slot | `memo_archive/baseline/`, `testing.sh`, `README.md`, `evals/memo_test/memo_test.py:679-756` |
| What do the four task sequences look like? | Shuffle at seed 42, first half collection, second half deployment; learning/testing split is 50/50 per category except MiniHack at 30/70; only ALFWorld has a genuine out-of-distribution set | `evals/workflows/agent_workflow.py:152-154`; `envs_archive/*_envs.py` as tabled in component 7 |

## Relevance to this project

### Components we can directly reuse

- **The archive sampling score**, `core/memo_manager.py:158-203`. Sigmoid-normalized performance against a measured no-memory baseline, minus a log visit-count penalty, softmaxed at a temperature. Eleven lines, no dependencies, and it gives an archive of harness states an exploration rule that never zeroes out any member.
- **The one-step edit record**, `core/memo_manager.py:148-152` plus `core/meta_agent.py:45-62`. Each entry is (parent code, natural-language suggestion, child code, realized score delta). That is the tuple shape a critic needs, and the released logs contain about 43 of them per domain across four domains.
- **The two-method memory contract**, `evals/agents/memo_structure.py`. If our long-term-memory layer needs an interface that a proposer can rewrite freely, this is a working minimal one: two methods, one dataclass, no framework.
- **Score-before-update on an ordered sequence**, `evals/workflows/agent_workflow.py:216-257` with `:62-64` and `:120-122`. Retrieve, act, score, then update, one task at a time under a semaphore of one, replicated over independent memory snapshots. This is our scoring protocol already implemented; only the reporting is missing.
- **The container runner**, `eval_in_container.py:58-98`, as a worked example of executing model-written harness code per candidate — with the bind-mount caveat noted above to fix rather than copy.

### Components we need to modify

- **The fitness** (`evals/launch.py:63-65`) aggregates one static set into one scalar. For us it would need to become a stream statistic: per-task scores in arrival order, an integral over the stream, and a separate anchor-set score. The record list already carries per-task rewards in order, so this is a reporting change, not an architectural one.
- **The task construction** (`agent_workflow.py:152-154`) splits one list into two disjoint halves. A retention measurement needs the opposite: a growing anchor set drawn from tasks already passed, re-scored at commit time.
- **The archive insert** (`meta_agent.py:199-201`) is unconditional. A gate would go exactly there — `improve_score` is already computed one line away in `memo_manager.py:151`.
- **The sequential path** currently exists only for the ALFWorld distribution-shift test and is reported as one average. Making it the default and logging score against task position is the smallest change that would let this codebase show a peak-then-decline curve if one exists.

### Components that don't apply

- The four environment adapters and the vendored BALROG copy: our task stream is not text-game episodes with binary success.
- `memo_archive/baseline/g_memory.py` and the other baseline designs: they are comparison points for memory research, not harness components we would adopt.
- Everything the protocol asks about training — masking, rollouts, advantages, distributed setup — has no counterpart here.

### Key code snippets worth studying

- `core/memo_manager.py:158-203` — the whole selection rule, including the commented-out greedy ablation at `:189-191`.
- `evals/workflows/agent_workflow.py:186-257` — collection-then-deployment in both rollout modes, side by side; the clearest statement of what this system does and does not measure.
- `evals/launch.py:36-104` — how three replicate passes become a mean and a standard error, and where per-task information is discarded.
- `core/meta_agent.py:146-202` — one search step end to end, including the point at which a design is accepted with no test.
- `evals/memo_test/memo_test.py:133-756` — a complete discovered memory design: six sub-layers (task parsing, semantic task index, strategy library, object affordances, failure memory, procedural path memory) composed into one retrieve pass and one update pass. It is the only discovered design in the repository and it matches the TextWorld column of the paper's Figure 3.
