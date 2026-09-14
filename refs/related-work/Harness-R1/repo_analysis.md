# Repo Analysis: Harness-R1

**Path:** `refs/related-work/Harness-R1/repo/` · **Source:** https://github.com/DeepExperience/Harness-R1 · **Analyzed:** 2026-09-14 (commit `411bb54`)

## Overview

- **Paper:** Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories (see `paper_analysis.md`)
- **Framework:** Relax (a trimmed vendored snapshot of `redai-infra/Relax`, Megatron-LM training backend plus SGLang rollout) for the RL stage; LLaMA-Factory configs for supervised fine-tuning; an AgentBench fork vendored at `code/life-harness/AgentBench` for the three task runtimes
- **RL Algorithm:** GRPO with $K = 8$ candidates per prompt, clip bounds 0.20/0.28, truncated importance sampling, no KL loss, no entropy bonus (`gspo`, `sapo`, `ppo`, `reinforce_plus_plus` are also reachable in Relax but unused)
- **Base Model:** Qwen3.5-9B for the harness engineer; the target agent is any OpenAI-compatible endpoint and is never trained by this code path
- **Key Innovation:** the reward is not a model — every candidate patch is compiled into sandboxed Python hooks, installed into a live task runtime, and scored by rerunning the frozen target on the identical task batch

Apache-2.0, 4.6 MB, no vendored weights, benchmark assets, or training data. `python scripts/check_release.py` runs 17 protocol/sandbox/demo tests and a `compileall` pass; it was executed during this analysis and passes clean. `python demo/run_demo.py` runs a full compile-install-intervene walkthrough offline with no endpoints, and was also executed.

## File Structure Map

```
Harness-R1/
├── scripts/                                       thin launch and check wrappers
│   ├── train_engineer_sft.sh                      cold-start SFT (LLaMA-Factory)
│   ├── train_engineer_rl.sh                       online GRPO (Relax/Megatron/SGLang)
│   ├── eval_{webshop,alfworld,dbbench}.sh         fixed-batch run-patch-rerun evaluation
│   ├── probe_openai_tool_calls.py                 pre-flight check for structured tool_calls
│   └── check_release.py                           test + compile gate for the release
├── configs/
│   ├── rl/mixed_codepatch.yaml                    reward metric, runtime paths, target endpoint
│   ├── sft/qwen35_9b_engineer_coldstart.yaml      engineer SFT (877 examples, 2 epochs, 1e-5)
│   ├── sft/qwen35_9b_agent_sft.example.yaml       target-agent SFT (the co-evolution arm)
│   └── eval/endpoints.env.example                 engineer/target endpoints and interpreters
├── code/Relax/examples/harness_r1/                the contribution's RL half
│   ├── reward_mixed_codepatch.py                  benchmark dispatcher (125 lines)
│   ├── reward_webshop_patch.py                    same-batch rerun reward (1,004 lines)
│   ├── reward_alfworld_patch.py                   same (946 lines)
│   ├── reward_dbbench_codepatch.py                same (896 lines)
│   ├── build_*_dataset.py                         failure-packet RL row builders
│   └── eval_*_patches.py                          offline patch evaluation drivers
├── code/life-harness/AgentBench/                  the contribution's runtime half
│   ├── scripts/harness_r1_patch.py                patch schema, validation, prompts (1,812 lines)
│   ├── scripts/harness_r1_trace_packet.py         failure-packet construction
│   ├── src/server/harness/code_runner.py          AST sandbox + hook executor (388 lines)
│   ├── src/server/harness/dsl.py                  typed overlay evaluator (disabled path)
│   ├── src/server/harness/{webshop,alfworld,dbbench}.py   the hand-built base runtime
│   └── src/server/tasks/{webshop,alfworld,dbbench}/task.py  where hooks actually fire
├── examples/
│   ├── webshop_patch.json                         one complete validated patch
│   └── heldout_generalization/                    27 patches, 3 editors × 3 seeds × 3 benchmarks
├── demo/run_demo.py                               offline compile-install-intervene walkthrough
└── tests/                                         5 files, 17 tests, protocol + sandbox + demo
```

## Component Inventory

### 1. Entry points

| Script | Role | Required inputs |
|---|---|---|
| `scripts/train_engineer_sft.sh` | cold-start SFT | a LLaMA-Factory dataset dir built by `build_llamafactory_mixed_sft_dataset.py` |
| `scripts/train_engineer_rl.sh` | online GRPO | `QWEN35_HF`, `HARNESS_R1_DATA` (grouped JSONL), `NUM_ROLLOUT`, `HARNESS_R1_CONFIG` |
| `scripts/eval_<bench>.sh` | evaluation | input JSONL, output dir; `GENERATE_ONLY=1` / `PATCH_SOURCE_ROOT=` split the generate and rerun phases |
| `demo/run_demo.py` | offline demo | nothing — replays stored states through a real compile and hook run |

The RL launcher is a 260-line bash wrapper around `python -m relax.entrypoints.train` that pins the resource split `{"actor":[1,4],"rollout":[1,2],"reference":[1,1],"actor_fwd":[1,1],"advantages":[1,0]}`, tensor parallel 4, and a `--fully-async` pipeline with `--max-staleness 4`. All hyperparameters are shell defaults overridable by environment variable; nothing is hard-coded in Python.

### 2. Special token and action handling

The engineer's output protocol is `prefill_think_patch`: the chat template prefills the opening `<think>\n`, the model closes it and emits exactly one `<patch>` block holding a JSON object. Parsing lives in `harness_r1_patch.py` — `extract_json_object` (line 262), `extract_think_patch_json_object` (322), `extract_prefilled_think_patch_json_object` (341). A parse failure is not an error; it returns `patch = None` and the sample is scored as no-patch with reward 0.

The declared action vocabulary (`ACTION_TYPES`, line 19) is six types:

```python
ACTION_TYPES = {
    "set_config",
    "edit_tool_hint",
    "add_or_edit_skill",
    "add_guard_rule",
    "add_recovery_rule",
    "add_code_hook",
}
```

**Only the last one is live in the paper's experiments.** `require_code_hook_only_patch` (line 357) rejects every other type, and `configs/rl/mixed_codepatch.yaml` sets `harness_r1_require_code_hook_only: true`. The five rejected types compile into the typed-condition evaluator in `dsl.py` (skills, guard rules with `eq`/`gt`/`contains`/`startswith` conditions over `state.*`, `action.*`, `task.*`, `predicates.*` paths, recovery rules, BM25 skill retrieval) — a complete second action space that the released system leaves switched off. `tests/test_patch_protocol.py::test_legacy_action_is_rejected_by_strict_protocol` pins this.

`normalize_patch` (line 373) caps a patch at **12 actions** and rewrites it into a canonical `harness-r1-patch-v1` object; `_compile_code_hook_for_validation` (417) compiles each hook at validation time so an uncompilable patch never reaches a rerun.

### 3. Environment and tool interaction

Two HTTP surfaces, both OpenAI-compatible: the engineer endpoint (`ENGINEER_BASE_URL` / `ENGINEER_MODEL`, with `ENGINEER_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}`) and the frozen target endpoint (`harness_r1_target_urls`). The reward function shells out to an AgentBench controller plus worker subprocesses on dynamically probed ports (`harness_r1_controller_port_base: 14000`, `worker_port_base: 24000`, `port_probe_span: 7000`, `avoid_ephemeral_ports: true`), each benchmark getting its own Python interpreter because WebShop and ALFWorld have incompatible dependency trees.

The trap the README flags in bold is real and load-bearing: the target server must return structured `message.tool_calls`; XML-shaped tool text inside `message.content` "silently zeroes rewards". `scripts/probe_openai_tool_calls.py` exists solely to check this before a long run.

### 4. Token masking

**Absent, and correctly so.** The engineer emits one single-turn response with no interleaved environment observations, so there is nothing to mask out of the loss. The advantage is sequence-level and shared by all response tokens; the only per-token weighting is `w_{k,t}`, the truncated importance weight correcting rollout-engine versus training-engine log-probability mismatch (`--use-tis`, clipped to $[0, 2]$).

### 5. Reward function — the heart of the repo

`reward_mixed_codepatch.py` dispatches on `sample.metadata["benchmark"]` to one of three near-identical ~950-line implementations. Each one: parses the patch, validates it, writes it to a cache dir, spawns the AgentBench controller and workers, reruns the frozen target on the identical task range, collects per-task rewards, and returns the delta. The scoring core (`reward_webshop_patch.py:924-953`):

```python
baseline_average_reward = _baseline_average_reward(metadata, batch_size=batch_size)
patched_average_reward = _average_reward(patched_rewards, batch_size=batch_size)
delta_average_reward = patched_average_reward - baseline_average_reward
...
delta_pass = patched_pass - baseline_pass
delta_pass_rate = delta_pass / max(1, batch_size)
delta_score = delta_average_reward if reward_metric == "delta_average_reward" else delta_pass_rate
result = {
    "score": delta_score + valid_bonus,
    ...
}
```

Configured behavior in `configs/rl/mixed_codepatch.yaml`:

| Key | Value | Effect |
|---|---|---|
| `harness_r1_reward_metric` | `delta_average_reward` | continuous shaped delta rather than pass-rate delta |
| `harness_r1_valid_bonus` | `0.0` | no format reward — a well-formed inert patch earns nothing |
| `harness_r1_reject_runtime_noop_patch` | `true` | a patch whose hooks cannot fire in this runtime scores as no-patch |
| `harness_r1_require_code_hook_only` | `true` | the five DSL action types are refused |
| `harness_r1_allow_partial_eval_reward` | `false` | an incomplete rerun scores 0, never a partial credit |

**Four distinct zero-reward paths**, all collapsing to `score: 0.0` via `_no_patch_reward`: `invalid_patch_treated_as_no_patch`, `runtime_noop_patch_treated_as_no_patch`, `eval_failed_treated_as_no_patch`, `unknown_benchmark`. Since zero is also the reward of a perfectly neutral patch, and a regressive patch goes negative, **an unparseable patch is scored strictly better than a harmful one** — under group normalization that makes malformed output the risk-free option.

There is **no retention term, no anchor set, no held-out term, no cost term, and no size penalty** anywhere in the reward. A repo-wide grep for `anchor|retention|regress|forget|held_out|plastic` outside vendored infrastructure returns only DBBench prompt strings, an unused `--max-revision-regression-traces` flag on the GPT-patch evaluation driver (set to 0 in the dataset builders), and the word "regression" inside engineer prompt text asking the model to reason about regression risk in its `<think>` block. Regression avoidance is requested in the prompt and never priced in the objective.

WebShop carries a pairing guard the other two lack: `validate_identity_metadata` raises `WebShopIdentityError` if the configured `harness_r1_webshop_goal_seed` disagrees with the baseline manifest, and the `webshop_batch_identity_v1` protocol compares per-task manifest digests between baseline and patched runs. The README states the reason plainly — "matching integer task indices are not proof of a paired comparison".

### 6. RL training loop

Relax's `Advantages` service (`code/Relax/relax/components/advantages.py:167`) routes `grpo`/`gspo`/`sapo` to `get_grpo_returns(rewards, kl)`, giving the group-normalized advantage of eq. 3. Reference settings, cross-checked between `scripts/train_engineer_rl.sh` and Appendix B.2 of the paper (they agree):

| Setting | Value |
|---|---|
| Candidates per prompt $K$ | 8 (`N_SAMPLES_PER_PROMPT`) |
| Rollout batch / global batch | 4 prompts / 32 sequences |
| Iterations per train update | 4 |
| Max policy staleness | 4 in the script, 8 in the paper table |
| Clip low / high | 0.2 / 0.28 |
| Truncated importance sampling | `--use-tis`, enabled |
| Entropy / KL coefficient | 0 / 0 (`USE_KL_LOSS` defaults off) |
| Learning rate / schedule | `1e-6`, constant, Adam(0.9, 0.98), weight decay 0.1 |
| Rollout temperature / top-p | 0.7 / 0.95 |
| Max prompt / response | 28,672 / 12,288 |
| Reward concurrency | 10 (the rerun is the bottleneck) |

`ROLLOUT_SHUFFLE=0` is required for pre-grouped mixed data, because rollout groups must not mix benchmarks — the engineer never faces a shifting task distribution inside one group. Dynamic sampling (`check_reward_nonzero_std`, which drops groups where all eight candidates scored identically) is available but off by default.

### 7. SFT / cold-start pipeline

`configs/sft/qwen35_9b_engineer_coldstart.yaml` is a LLaMA-Factory config: full fine-tuning, DeepSpeed ZeRO-3, Liger kernel, `template: qwen3_5`, `enable_thinking: true`, cutoff 32,768, 2 epochs, LR `1e-5` cosine with 0.03 warmup, bf16, seed 42, effective global batch 24. Its last line is a comment worth noting: `### no held-out loss set was used during cold-start SFT`. Rows are built by `build_llamafactory_mixed_sft_dataset.py` from GPT-5.5 teacher proposals filtered on the frozen target for executability, completeness, and non-negative same-batch delta — 877 survivors from disjoint task batches.

`configs/sft/qwen35_9b_agent_sft.example.yaml` is the other half of the co-evolution experiment: it fine-tunes the *target* on its own successful no-intervention trajectories (2,515 of them), producing the stronger frozen actor that the second engineer is then trained against.

### 8. Data pipeline

An RL row is a prompt plus immutable baseline metadata; the reward never recomputes the baseline, it reads it:

```json
{"prompt": [{"role": "system", ...}, {"role": "user", ...}], "label": "",
 "metadata": {"benchmark": "webshop", "batch_tag": "b0001", "start": 10, "end": 20,
              "batch_size": 10, "baseline_pass": 2, "baseline_rewards": {"10": 0.0},
              "target_model": "Qwen3.5-9B", "target_agent_name": "qwen35-9b-nothink"}}
```

This is why the README warns never to reuse baseline rewards produced by a different target model or serving protocol — the comparison's before-half is frozen in the dataset, not measured at reward time.

`harness_r1_trace_packet.py` builds the failure packet: load trajectories, keep cases below `--reward-threshold` (default 1.0), select up to `--max-traces` (default 20) by `round_robin` across batches or `lowest_reward`, trim observations to 1,200 characters and assistant turns to 600, and cap the packet at roughly 18,000 approximate tokens. Selection is deterministic and carries no learned component.

### 9. Multi-GPU and infrastructure

One node, 8× H800, for all three stages. Megatron-LM backend for the actor (TP 4, PP 1, CP 1, sequence parallel on, full recompute with uniform method and 1 layer, 40,960 tokens per GPU, CPU-offloaded precision-aware optimizer), SGLang for rollout (2 GPUs per engine, static memory fraction 0.60), Ray for orchestration with a `RUN_DIRECT=1` single-process path that bypasses Ray entirely. `code/Relax/relax/` is a trimmed vendored snapshot — bridges for Qwen-Omni and GLM-MoE, per-model launch scripts from Qwen2.5-7B up to GLM5-744B-A40B — and is not part of the contribution.

### 10. Evaluation

The three `eval_<bench>.sh` wrappers implement fixed-batch run-patch-rerun. The flag that matters for adapting this work is the generate/rerun split:

```bash
GENERATE_ONLY=1 bash scripts/eval_webshop.sh INPUT OUT_GENERATED
PATCH_SOURCE_ROOT=OUT_GENERATED bash scripts/eval_webshop.sh INPUT OUT_RERUN
```

`patch_source_root` reads `<root>/<batch_tag>/patch.json` — it moves a patch between machines for the cross-target protocol; it does **not** accumulate patches across batches. Nothing in the repo composes one patch onto another; every overlay is installed on the pristine base runtime.

Failure discipline is codified: `harness_r1_eval_max_attempts: 3` with `eval_retry_on_infra_failure: 2`, and the README's rule that "an environment failure is a missing evaluation and must be reported separately, never retried until it turns positive".

### The sandbox and the hook fire points (not in the ten, but the most instructive code here)

`code_runner.py` is 388 lines and is the reason a model can be allowed to write code into a live runtime. `compile_hook` (line 136) parses the source, requires exactly one top-level `def hook(ctx, nb)` plus at most 5 helpers, and rejects: any non-function top-level statement, any name or attribute starting with `_`, anything in `FORBIDDEN_NAMES` (`eval`, `exec`, `open`, `getattr`, `globals`, `__import__`, …), and any node in `FORBIDDEN_NODES` (`Import`, `ImportFrom`, `ClassDef`, `Lambda`, `While`, `With`, `Raise`, `Yield`, `Global`, `Delete`, async everything). `try` must catch `Exception` explicitly with no `else`/`finally`. Size caps: 8,000 source characters, 1,200 AST nodes, 700-character string literals, 900-character returns. The execution namespace holds only a 25-name safe-builtin subset plus `math`, `re`, and `SequenceMatcher`.

There is one benchmark-specific anti-leakage rule: `ALFWORLD_INSTANCE_ACTION_RE` rejects any string literal that looks like a numbered ALFWorld instance action (`go to cabinet 3`), and `_clean_action` additionally refuses a returned action matching that pattern unless it appears in the runtime's current admissible set — so a hook cannot smuggle a memorized solution past validation and cannot fabricate one at runtime.

`run_hook` (line 269) enforces a **0.05 s** wall clock via `signal.setitimer` and a **2,000 executed line** budget via `sys.settrace`, deep-copies `ctx` so a hook cannot mutate the runtime's view, and downgrades every exception and timeout to `None` — "a bad hook cannot crash an episode". `_normalize_hook_result` (325) then discards anything outside the declared return contract per hook name.

Hooks fire in the per-benchmark task drivers, not in the harness runtime classes. In `src/server/tasks/webshop/task.py`: `on_init` at line 449, `make_pre_hint` at 525, `on_before_action` at 661, `on_post_step` at 786, all routed through `_run_code_hooks` (333); ALFWorld and DBBench mirror this at `task.py:197/373` and `196/260`. Effects are applied by `_apply_before_action_effect`, which for `block_and_prompt` injects a user message plus a zero-reward history item and re-asks the frozen target, and for `force_action`/`rewrite_action` substitutes the action string.

Worth flagging: `src/server/harness/{webshop,alfworld,dbbench}.py` are 2,380, 1,580 and 2,736 lines of **hand-engineered base runtime** — BM25 skill retrieval, click-similarity thresholds, repeat-click blocking, search-loop and product-stall counters, budget checks, tuning constants named `h2_`…`h5_`. This is the harness the engineer's patches sit *on top of*, and it is human-written, benchmark-specific, and full of the tuning constants our own code standards treat as red flags. The paper's editable surface is the thin hook layer above it, not this.

## Relevance to this project

### Components we can directly reuse

- **The executable-hook action space and its return-value contract** — `src/server/harness/code_runner.py:30-34` (`HOOK_NAMES`, `ALLOWED_EFFECT_KINDS`) and `_normalize_hook_result` at line 325. The design principle worth taking whole: the hook returns a *declaration* and the host runtime performs the action, so the edit surface is auditable and the blast radius is bounded by the effect vocabulary rather than by what Python can do.
- **The AST sandbox** — `compile_hook` at line 136 and `run_hook` at 269, essentially as-is. This is the cheapest available answer to "how do we let a proposer write code into our harness" and it is 250 lines, not a container.
- **The same-batch rerun reward skeleton** — `reward_webshop_patch.py:924-953`. The mechanics of install-rerun-subtract are exactly what a horizon-discounted reward also needs; only the set of tasks rerun and the discounting change.
- **The baseline-identity protocol** — `validate_identity_metadata` and `webshop_batch_identity_v1`. Our score-before-update protocol has the same auditing problem in a harder form, and the pattern of freezing the before-half into immutable row metadata rather than recomputing it is directly transplantable.
- **`examples/heldout_generalization/`** — 27 patches from three editors on identical failure evidence, each with the realized held-out delta it earned and the evidence task ids. A small labelled set of (harness edit, realized outcome) pairs spanning an editor-quality gradient, usable immediately as a sanity dataset for a state-level critic even though it is far too small to train one.
- **The two released engineer checkpoints** — a ready-made greedy proposer for the zero-horizon arm of any comparison, removing the need to reimplement one.

### Components we need to modify

- **The reward.** Replace the same-batch delta with a return over later batches, or add a critic term. The dispatcher structure in `reward_mixed_codepatch.py` already isolates this to one function per environment.
- **Patch lifetime.** Overlays are per-batch and installed on the pristine runtime; `patch_source_root` moves a patch between machines, it does not compose patches. A stream setting needs a persistent harness state, which means composition rules, conflict handling between hooks, and a size budget — none of which exist here.
- **The zero-reward collapse.** Invalid, inert, and failed-evaluation patches all score 0.0, which equals the no-change reward and beats any regression. Under group normalization this rewards bailing out. Any objective of ours needs these three cases separated.
- **The action space.** Re-enabling `set_config`, `edit_tool_hint`, `add_or_edit_skill`, `add_guard_rule`, `add_recovery_rule` (they exist and validate; only `require_code_hook_only` blocks them) is the difference between a one-layer editor and one that can express moves and promotions across layers.

### Components that do not apply

- **The Relax/Megatron/SGLang stack** (`code/Relax/`, ~200 files) is a vendored snapshot of someone else's trainer, carried only so the launch script runs. Our RL stack choice is independent of anything this paper argues.
- **The hand-built per-benchmark base runtimes** (`src/server/harness/*.py`, ~6,700 lines of thresholds and counters) are WebShop/ALFWorld/DBBench plumbing. Read `webshop.py:121-150` once as an example of the tuning-constant density a hand-written harness accumulates, then leave it.
- **`dsl.py`'s typed condition evaluator** duplicates in a constrained expression language what the code hooks already do in Python. It is the abstraction the authors kept but stopped using, and it is not worth reviving as a language — only the *action types* it backs are worth reviving.

### Key code snippets worth studying

| Location | Why |
|---|---|
| `src/server/harness/code_runner.py:136-235` | `compile_hook` — the complete validation policy for model-written code, in one readable function |
| `src/server/harness/code_runner.py:269-324` | `run_hook` — time and line budgets, ctx deep-copy, exception downgrade |
| `src/server/harness/code_runner.py:325-378` | `_normalize_hook_result` — how a per-hook return contract is enforced rather than trusted |
| `code/Relax/examples/harness_r1/reward_webshop_patch.py:924-953` | the delta computation and every field the reward reports |
| `code/Relax/examples/harness_r1/reward_mixed_codepatch.py:66-101` | `_no_patch_reward` — the four zero-reward paths collapsing into one value |
| `src/server/tasks/webshop/task.py:333-400, 449, 525, 661, 786` | where the four hooks fire in a real episode and how `block_and_prompt` re-asks the frozen target |
| `scripts/harness_r1_patch.py:357-395` | `require_code_hook_only_patch` and `normalize_patch` — the 12-action cap and the switch that narrows six action types to one |
| `demo/run_demo.py` | 240 lines, runs offline, and is the fastest way to see compile → install → intervene end to end |
