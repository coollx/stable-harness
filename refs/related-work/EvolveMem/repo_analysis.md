# Repo Analysis: SimpleMem / EvolveMem

**Path:** `refs/related-work/EvolveMem/repo/` · **Source:** https://github.com/aiming-lab/SimpleMem · **Analyzed:** 2026-09-15 (commit `db80b6a`, 24 July 2026, MIT, "Copyright (c) 2025 AIMING Lab")

The clone is the **parent SimpleMem repository**, roughly 110 MB. The code for this paper is the subdirectory `EvolveMem/` — 36 files, about 17,450 lines of Python. Everything below refers to paths under `refs/related-work/EvolveMem/repo/EvolveMem/` unless stated otherwise. The subdirectory's own `README.md` cites arXiv:2605.13941 with the paper's exact author list, confirming the match. The commit is about 2.5 months after the arXiv v1 date (13 May 2026), so where code and paper disagree, the code may be later development rather than the state that produced the reported numbers; that caveat is repeated where it matters.

## Overview

- **Paper:** EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents (see `paper_analysis.md`)
- **Framework:** custom Python, no dependency on any agent or training framework; SQLite with FTS5 for storage, `sentence-transformers` optional for embeddings, one OpenAI-compatible endpoint for all language-model calls
- **RL algorithm:** none. The search is hill-climbing over a dataclass, proposed by a language model.
- **Base model:** none trained. `gpt-4o` by default for answer generation, a separately configurable model for diagnosis.
- **Key innovation as implemented:** a 42-field retrieval-configuration dataclass that a diagnosis language model edits between rounds, under a hill-climbing accept-or-revert rule scored on one fixed question set.

## File Structure Map

```
EvolveMem/
  README.md                       badges, quick start, results tables copied from the paper
  requirements.txt                4 optional deps; core runs on the standard library + SQLite
  run_evolution.py                LoCoMo-only entry point (185 lines)
  run_benchmark.py                the real entry point: locomo | longmemeval | membench (392 lines)
  evolvemem/
    evolution.py            1727  the loop, the gate, the hardcoded configs, artifact writing
    diagnosis.py             955  the diagnosis prompt, the action-space ranges, change application
    multi_retriever.py       788  RetrievalConfig (the action space) + the three retrieval views
    meta_analysis.py         371  the paper's guarded meta-analyzer (inert in the default path)
    evolution_cookbook.py    424  14 hand-written, pre-validated "capability recipes"
    manager.py              5064  memory manager: ingest, consolidate, retrieve, policy plumbing
    store.py                1798  SQLite schema and queries
    extractor.py             461  sliding-window extraction with retry / chunk-split / coverage verify
    consolidator.py          315  dedup, importance decay, entity reinforcement
    benchmarks/                   adapters: locomo.py, membench.py, longmemeval.py, base.py, metrics.py
    ── undocumented second subsystem, absent from the paper ──
    self_upgrade.py          876  candidate evaluation, promotion, review queue, cycle summaries
    replay.py                558  replay stored past interactions under a candidate policy; compare
    upgrade_worker.py        435  scheduler around the above
    candidate.py             127  bounded candidate generation around the live policy
    promotion.py              44  the promotion gate: 8 delta criteria + sample floor + zero-retrieval cap
    policy.py / policy_store.py / policy_optimizer.py / telemetry.py
```

No data files, no results, no notebooks, no tests for the evolution loop.

## Component Inventory

### 1. Entry points

`run_benchmark.py:171-230` is the argument surface and the most informative file in the repository. Arguments that matter: `--initial weak|strong|terminal|custom`, documented in place as "weak for big evolution delta; terminal for the hardcoded known-best config (skip evolution)" (`run_benchmark.py:175-178`); `--config-from`, "path to a prior run's `evolution_summary.json`; its `final_config` becomes the initial config (used for cross-benchmark transfer tests)" (`run_benchmark.py:179-181`), which is the mechanism behind the paper's Table 5; `--static` to evaluate once without evolving; `--maturation-round`; `--sample` / `--samples` to select LoCoMo conversations. `run_evolution.py:139-166` is a thinner LoCoMo-only path that starts from a strong configuration rather than the weak one and writes to `evolution_results/sample_<N>/`.

### 2. The action space

`multi_retriever.py:39-205` defines `RetrievalConfig` with **42 fields**, and its docstring states the framing exactly: "The config's total surface area is the evolution action space — every field here is a dimension diagnosis can propose adjusting between rounds" (`multi_retriever.py:42-43`). `diagnosis.py:376-419` gives the ranges actually enforced. Both differ from the paper's Table 7:

| Group | Paper Table 7 | Code |
|---|---|---|
| View top-$k$ | ints in $[3, 30]$ | `semantic_top_k` $[0,40]$, `keyword_top_k` $[0,20]$, `structured_top_k` $[0,15]$ (`diagnosis.py:377-380`) |
| Context budget | int in $[6, 30]$ | `max_context` $[4,50]$ (`diagnosis.py:381`) |
| Fusion mode | `{sum, rrf, weighted_sum}` | 6 values, adding `first_found`, `semantic_only`, `keyword_only`, `structured_only` (`diagnosis.py:400-404`) |
| Fusion weights | $[0.1, 2.5]$ | $[0.0, 3.0]$ (`diagnosis.py:392-394`) |
| Reflection rounds | int in $[0,3]$ | int in $[0,2]$ (`diagnosis.py:383`) |
| Not in Table 7 | — | intent planning (4 fields), coverage reflection (3), knowledge-graph expansion (5), maximal-marginal-relevance rerank (2), `answer_model` / `answer_model_ensemble`, and **five `locomo_cat*` prompt-surface flags** (`multi_retriever.py`, `diagnosis.py:405-419`) |

The five `locomo_cat*` booleans are benchmark-specific prompt switches, allow-listed for the diagnosis model to set (`diagnosis.py:405-418`) and gated by benchmark prefix so they cannot be proposed on MemBench or LongMemEval (`diagnosis.py:448-460`). They are the highest-impact levers in the released prompt, and they are not part of the paper's action space.

### 3. The evolution loop: what is scored, what is gated, what feeds back

The whole loop is `EvolutionEngine.evolve` in `evolution.py:440-895`. The specific questions, each with the line that settles it:

| Question | Answer | Evidence |
|---|---|---|
| What set is scored? | One `qa_pairs` list, passed in once and reused unchanged every round. | `evolution.py:485` `qa_results = self._evaluate_qa(index, qa_pairs, ret_config)` inside `for round_id in range(self.config.max_rounds)` at `evolution.py:459` |
| What is the score? | Unweighted mean per-question F1 over that list. | `evolution.py:488` `overall_f1 = sum(r.f1 for r in qa_results) / len(qa_results)` |
| Is there a held-out split? | **No.** A repository-wide search for a train/validation/holdout split returns only LongMemEval's `--split oracle|s|m`, which selects which released data file to load. | `run_benchmark.py:202`, `run_benchmark.py:124-125` |
| Does the gate read the same set the diagnosis reads? | Yes, and in that order: the gate runs first so the diagnosis model sees the incumbent state after a rejection. | `evolution.py:500-501` comment "elitist step BEFORE diagnose"; `evolution.py:609-614` passes the same `qa_results` to `diagnose` |
| What is the gate? | Hill-climbing against **best-so-far**, not against the previous round. `delta = overall_f1 - best_f1`; accepted when `delta > acceptance_threshold` (0.003). | `evolution.py:526-527`, `evolution.py:189-211` |
| What happens on a small positive delta? | "ELITISM-SOFT-ACCEPT": any strict improvement updates the incumbent even below the threshold; the threshold only sets the accept label and the early-stop counter. | `evolution.py:545-549`, `evolution.py:558-566` |
| What happens on a non-improvement? | "ELITISM-REJECT": the configuration is rewritten field-by-field from `best_config`. | `evolution.py:567-577` |
| When does it stop? | After 5 consecutive non-accepting rounds, not on the paper's one-round convergence test. | `evolution.py:580`, `max_consec_noaccept: int = 5` at `evolution.py:209` |
| How many changes per round? | At most 2 top-level fields; a `per_category_overrides` entry counts as 1, and an invoked recipe counts as 1 regardless of how many fields it sets. | `evolution.py:207` `max_changes_per_round: int = 2`; `evolution.py:639-648`; `diagnosis.py:466-471` |
| Does any later-round score feed back? | **No.** Every comparison is between the current round's score and a scalar carried forward from earlier rounds. No future quantity exists in the code. | `evolution.py:526`, `evolution.py:545` |
| Is any earlier question set re-scored? | **No.** There is one set; it is re-scored every round because it is the only set. There is no accumulating set, no anchor set, and no per-question retention check anywhere in the loop. | `evolution.py:485`; no second question source exists in `evolve` |
| What is returned and reported? | `best_config`, the argmax over rounds, written as `final_config`. | `evolution.py:875-877`, `evolution.py:881-890` |

The paper's equation 4 is `meta_analysis.py`, and in the default path it is almost entirely inert. `MetaEvolutionAnalyzer._is_regression` uses a threshold of **$-0.02$** against the previous round (`meta_analysis.py:114-118`), not the paper's $\tau_{\text{rev}} = 0.01$; `_is_stagnant` is a 3-round window with a span under 0.01 (`meta_analysis.py:120-125`), not the paper's two-round $\epsilon$ test. The explore branch is not a random perturbation: it rotates `fusion_mode` through the values not yet tried (`meta_analysis.py:242-259`). And under the default `elitist=True`, the meta-analyzer's revert is skipped ("Meta-analyzer reverts are redundant under elitist mode", `evolution.py:694-700`), its `parameter_suggestions` are skipped, and its `per_category_overrides` are skipped (`evolution.py:727-733`). The only meta output that survives is a `new_dimension_proposal` appended to `meta_proposals.jsonl` (`evolution.py:736-745`). **The guarded meta-analyzer described in the paper does not gate anything in the released default configuration.**

### 4. The two hardcoded configurations and the maturation default

`evolution.py:47-73` is `weak_initial_config()`, the paper's $\theta_0$. Its docstring states the design intent in the authors' own words: "we want the evolved-minus-static delta to be the paper's headline, so the initial should be a principled-but-minimal BM25 + small-k retriever with no category tricks, no time decay, no reflection, no multi-view fusion, no entity-swap ... it leaves every evolvable knob set to its weak-but-safe value so evolution has room to climb."

`evolution.py:89-143` is `evolved_config()`, "Full evolved configuration with all discovered dimensions" — a hand-written record with `fusion_mode="rrf"`, weights 1.5 / 1.2 / 0.7, intent planning on, all five `locomo_cat*` flags on, `reflection_rounds=1`, and per-category overrides for categories 1, 3, and 5 including a `gpt-4.1` answer-model ensemble on category 3.

Two paths reach it. `--initial terminal` loads it directly and skips evolution. And `maturation_round`: at `evolution.py:459-472`, when `round_id == maturation_round` the engine overwrites every field of the live configuration with `evolved_config()` and logs "Evolution matured at round %d". The default is set at `run_benchmark.py:301-306`:

```python
mat_round = args.maturation_round
active_model = key_cfg.get("model", "gpt-4o")
if args.answer_model:
    active_model = args.answer_model
if mat_round is None and args.benchmark == "locomo" and not args.static:
    if active_model.startswith("gpt"):
        mat_round = 5
```

and `run_benchmark.py:315` then forces `max_rounds = max(args.max_rounds, mat_round + 2)`, so a default LoCoMo run with a GPT answer model is a 7-round run in which the full hand-written configuration is installed wholesale at round 5. This is the repository's single most consequential difference from the paper: a default path in which the last rounds of an "autonomous" trajectory are a hardcoded assignment. It cannot be shown that the paper's Table 4 was produced this way — no run logs are shipped and the clone postdates the arXiv submission — but the timing is worth recording: the paper's run is 7 rounds and its two largest jumps are the last two.

### 5. The diagnosis prompt and the recipe library

`diagnosis.py:22-140` is `DIAGNOSIS_PROMPT`. Relative to the paper's Appendix F.6 version it adds four things.

A hard step cap: "## STEP-SIZE CONSTRAINT (HARD RULE) ... This evolution runs under strict hill-climbing: **at most 2 changes per round**" (`diagnosis.py:72-80`). An anti-plateau rule instructing the model to reach for an unused recipe after two consecutive rejections (`diagnosis.py:88-97`). A recipe menu, `## Available Recipes (triggered by current symptoms)` with `{available_recipes}` substituted at runtime and the instruction to return `"use_recipe": "<recipe_name>"` (`diagnosis.py:47-58`). And a priority list quoting empirical per-lever gains (`diagnosis.py:99-121`):

```
1. **Adapter prompt-surface flags** (+5-30 pp each, ...):
   - locomo_cat5_mcq: Cat 5 low AND many "not mentioned" predictions → +30pp
   - enable_answer_verification: many abstention / wrong-format failures → +5-10pp
   - locomo_cat1_single_fact / locomo_cat4_verbatim_copy: ... → +5-15pp
```

`evolution_cookbook.py` holds 14 recipes, each a trigger predicate plus a configuration bundle plus an `expected_lift_pp` number: `rrf_fusion_warmup` (6.0), `intent_aware_multihop_planning` (15.0), `rrf_weight_tuning` (2.0), `verification_strict` (4.0), `cat5_mcq_adversarial` (10.0), `cat5_retrieval_boost` (5.0), `cat1_cat4_paraphrase_killer` (8.0), `cat2_temporal_format` (3.0), `cat3_inferential_nuanced` (2.0), `cat1_cat4_selective_decomposition` (3.5), `cat3_mmr_bridge_diversity` (2.5), `reflection_single_pass` (1.5), `cat3_dual_model_ensemble` (9.0), `escape_plateau_scalar_sweep` (1.5). The module docstring is explicit about provenance (`evolution_cookbook.py:1-22`): "Each recipe is a (symptom → proposal) bundle that was empirically validated against LoCoMo ... Each one is a compressed form of an experiment we ran, checked in as code so the framework can autonomously access it rather than requiring the human to re-discover it per-run."

The same human search shows up in field comments carrying version labels and measured results, for example `multi_retriever.py:124-125` ("path-1 proved that widening vector top-k hurts Cat 3 precision (-18.8pp on s3)") and `evolution.py:337-341`, which records an offline comparison of two answer models over 31 category-3 questions on samples 1, 3, 5, and 8 with an oracle upper bound. The released system is therefore a language model choosing from a menu of already-validated human findings, not a search that discovers them.

### 6. Artifacts written, and what is shipped

Per round the engine writes `round_<N>/raw_results.jsonl` with one line per question — `qid`, `qtype`, `category`, `subcategory`, `question`, `prediction`, `reference`, `metrics`, `primary`, `retrieved_count`, `retrieved_sources` (`evolution.py:815-836`) — plus `round_<N>/summary.json` (`evolution.py:837-845`). At the end it writes `evolution_final.json` with `best_round`, `best_f1`, `final_config`, and a per-round trajectory (`evolution.py:880-893`). `run_benchmark.py:363-386` additionally writes `evolution_summary.json` carrying every round's primary metric, subcategory scores, improvements applied, and **full configuration snapshot**.

**None of it is in the repository.** The root `.gitignore` excludes `evolution_results/`, `test_results/`, `*.json`, and `*.db`. A repository-wide search for `*.jsonl`, `evolution_final.json`, `evolution_summary.json`, and `round_*.json` returns nothing. Appendix D.3 of the paper claims these artifacts are persisted and "versioned alongside the code so that the autonomous trajectory is reproducible across runs"; the code writes them, the repository does not carry them. **There is no released harness-evolution log here.** Nor is there data: the README's `--data data/locomo10.json` points at a directory that does not exist in the clone.

### 7. Benchmark adapters: diagnosis versus reporting

`benchmarks/base.py:43-46` defines the evaluation unit as "a set of sessions + its QA pairs". `benchmarks/locomo.py:21-103` loads one LoCoMo conversation and all of its questions; `run_benchmark.py` selects conversations with `--sample N` or `--samples 0,3,5,8`. `benchmarks/membench.py` splits each original MemBench turn into two units (`membench.py:59`). `benchmarks/longmemeval.py` exists in the code but not in the paper.

There is no diagnosis-versus-reporting distinction at all. The adapter returns one question list; `evolution.py:485` scores it; `evolution.py:609-614` diagnoses from it; `evolution.py:526` gates on it; `evolution.py:875-877` reports the best of it. Cross-benchmark transfer is the one place where a configuration meets questions it was not tuned on, and it is done by hand across two invocations: run one benchmark, then pass that run's `evolution_summary.json` to the next via `--config-from` (`run_benchmark.py:179-181`).

### 8. The undocumented replay-and-promotion subsystem

Nine modules implement a second self-improvement path that the paper never mentions. It operates on `MemoryPolicyState` (the manager's live retrieval policy) rather than on `RetrievalConfig`, and its shape is much closer to what our project needs than the paper's loop is.

`candidate.py:8-14` generates "a small bounded candidate set around the current live policy", varying retrieval mode, injection budget, and weights. `replay.py:61-97` loads stored past interaction records from a JSON-lines file — each record carrying a session identifier, turn index, scope, the instruction text, the response text, and the resulting state — and optionally subsamples them with telemetry weighting that prefers sessions with richer retrieval history (`replay.py:311-352`). `replay.py:291-308` is the core comparison:

```python
def run_policy_candidate_replay(cfg, samples, candidate_policy_path):
    evaluator = MemoryReplayEvaluator()
    baseline_manager = MemoryManager.from_config(cfg)
    candidate_state = MemoryPolicyStore(candidate_policy_path).load()
    candidate_manager = MemoryManager.from_config_with_policy_state(cfg, candidate_state)
    baseline = evaluator.evaluate(baseline_manager, samples)
    candidate = evaluator.evaluate(candidate_manager, samples)
    comparison = evaluator.compare(baseline, candidate)
```

Both the incumbent and the candidate are scored on the **same stored past records**, and the candidate is promoted only if `promotion.py:20-44` passes: at least 10 samples, non-negative deltas on eight quality measures (query overlap, continuation overlap, response overlap, focus, value density, grounding, coverage, and specificity with a $-0.05$ floor), at most 2 additional zero-retrieval samples, and `candidate_beats_baseline`. `self_upgrade.py:64-94` wraps this with an optional `require_review` flag that routes a passing candidate into a human review queue instead of promoting it, and `self_upgrade.py:202-228` runs the full cycle: generate candidates, evaluate the directory, write a cycle summary, clean up artifacts.

Two limits. The quality measures are reference-free proxies computed from text overlap (`replay.py:130-201`), not task scores, so this is not a retention check on graded tasks. And nothing connects this subsystem to the paper's evolution loop: it is exported from the package (`evolvemem/__init__.py:23-34`) and driven by `upgrade_worker.py:92`, and no call site in `evolution.py`, `run_evolution.py`, or `run_benchmark.py` reaches any of it.

### Components absent

No special-token or action-token handling, no token masking, no reward function beyond the benchmark metric, no reinforcement-learning training loop, no supervised or cold-start pipeline, no multi-graphics-processing-unit infrastructure, and no training of any kind. The repository trains nothing; every language-model call is an API call.

## Relevance to this project

### Components we can directly reuse

- **The per-question record schema**, `evolution.py:815-836`. Question identifier, type, category, subcategory list, question, prediction, reference, all metrics, primary score, retrieved count, and retrieved sources. This is the right minimum for a per-question retention check over an anchor set, and it is what our score-before-update protocol would need to persist per task.
- **The replay-and-promotion pattern**, `replay.py:291-308` plus `promotion.py:20-44`. Score the incumbent and the candidate on the same stored past records, require a minimum sample count, and require non-negative deltas on several measures at once rather than on one aggregate mean. This is the closest thing in any reference repository to an anchor-set commit gate, and the multi-criterion form is more informative than either EvolveMem's single-mean tolerance or HarnessX's per-task binary flip test.
- **The cross-benchmark re-score protocol**, `run_benchmark.py:179-181`. Passing a prior run's `final_config` into the next run's initial configuration is a two-line mechanism that produced the paper's only retention-shaped measurement (LoCoMo re-scored after MemBench evolution, 0.543 to 0.593). We can run the same probe cheaply.
- **The per-round configuration snapshot**, `run_benchmark.py:363-386`. If we produce harness-evolution logs for a critic to be pretrained on, this is a workable record format: per round, the primary metric, the subcategory breakdown, the improvements applied, and the full configuration.

### Components we need to modify

- **The gate** (`evolution.py:510-577`) needs a second question set. As written it is a single aggregate-mean comparison against best-so-far on the set the proposer read. Replacing `best_f1` with a two-part test — current-batch score and a score on an accumulating anchor set the proposer did not read — is a small, local edit to the same block, and it is the minimal experiment that separates our objective from theirs.
- **The step cap** (`evolution.py:207`, `diagnosis.py:72-80`) is a useful control but it is enforced in the prompt and by truncation. If we want per-edit attribution it needs to be enforced structurally, the way HarnessX's change manifest is.
- **The action space** (`multi_retriever.py:39-205`) is one layer of ours, and only part of it: retrieval settings for long-term memory plus five answer-prompt switches. Nothing structural can be added. Any reuse has to treat this as one layer's parameterization, not as an edit space.

### Components that don't apply

- **The meta-analyzer** (`meta_analysis.py`) is dead code in the default path (`evolution.py:694-733`), and its thresholds contradict the paper's. There is nothing here to adopt.
- **The recipe library** (`evolution_cookbook.py`) and `evolved_config()` (`evolution.py:89-143`) are LoCoMo-specific human findings compiled into the repository. They are evidence about the paper's autonomy claim, not reusable machinery.
- **The memory store, extractor, and consolidator** (`store.py`, `extractor.py`, `consolidator.py`, `manager.py`) are a conversational-memory implementation. They are the content of one harness layer, not part of the harness-editing problem.
- **The maturation default** (`run_benchmark.py:301-306`) is the opposite of what we want to build; it is recorded here so the paper's trajectory is read with the right caveat.

### Key code snippets worth studying

| Location | Why |
|---|---|
| `evolution.py:510-577` | The entire accept-or-revert gate in 68 lines, including the soft-accept branch. The tightest existing example of the greedy same-batch gate our claim targets. |
| `evolution.py:459-472` and `run_benchmark.py:301-315` | The maturation injection and its default. Read together before citing any trajectory number from this paper. |
| `evolution.py:47-73` | `weak_initial_config()`, whose docstring states that the baseline was chosen so the evolved-minus-static delta would be the headline. |
| `replay.py:291-308` and `promotion.py:20-44` | The undocumented replay comparison and its multi-criterion promotion gate — the reusable idea in this repository. |
| `diagnosis.py:99-121` | The priority list quoting per-lever gains in percentage points, absent from the paper's version of the same prompt. |
| `evolution_cookbook.py:1-22` | The docstring stating that each recipe is a compressed form of an experiment the authors ran, checked in so the framework need not re-discover it. |
| `diagnosis.py:376-419` | The real action-space ranges and the five benchmark-specific prompt flags, for comparison against the paper's Table 7. |
