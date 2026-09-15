# EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents

- **Original title**: EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents
- **Authors**: Jiaqi Liu, Xinyu Ye (corresponding), Peng Xia, Zeyu Zheng, Cihang Xie, Mingyu Ding, Huaxiu Yao (corresponding)
- **Affiliations**: University of North Carolina at Chapel Hill (Liu, Ye, Xia, Ding, Yao); University of California, Berkeley (Zheng); University of California, Santa Cruz (Xie)
- **Venue / year**: arXiv preprint, 2026 (arXiv:2605.13941v1 [cs.LG], submitted 13 May 2026; 22 pages including appendices A-F). The paper prints "Preprint."; the released README carries a "NeurIPS 2026" badge with no accompanying record.
- **Source URL**: https://arxiv.org/abs/2605.13941
- **Code**: https://github.com/aiming-lab/SimpleMem, subdirectory `EvolveMem/` (MIT, "Copyright (c) 2025 AIMING Lab") — see `repo_analysis.md`.
- **OpenReview**: none found (see section 7).
- **Classification**: **method paper**. The contribution is a built system — a typed memory store plus a four-step self-evolution loop over its retrieval configuration — that others could reproduce; the trajectory, transfer, and ablation tables exist to support the system claim rather than to establish a finding of their own.

Not to be confused with **MemEvolve** (arXiv:2512.18746, Zhang et al., "Meta-evolution of agent memory systems"), which this paper cites as reference [38] and lists as a separate row in its Table 1.

Short names used below: EvolveMem (the system), LoCoMo and MemBench (the two benchmarks), and "the configuration" for the retrieval configuration $\theta$ that the system edits.

---

## 1. Method Motivation

### a) Why was this method proposed?

The paper's complaint is one sentence in the abstract: "existing memory systems treat retrieval infrastructure as fixed: stored content evolves while scoring functions, fusion strategies, and answer-generation policies remain frozen at deployment." Section 1 develops this into a mismatch that grows over time — "as stored memories grow from dozens to hundreds of heterogeneous records, a retrieval policy calibrated for the small store becomes suboptimal, and different question categories require fundamentally different retrieval strategies." The proposed answer is co-evolution at two levels, the stored knowledge and the retrieval mechanism that queries it, driven by a language model that reads the system's own failure logs. The authors label this "AutoResearch": "the system autonomously conducts the observe-hypothesize-experiment-validate cycle that would otherwise require manual researcher effort."

### b) Pain points of existing methods

Section 2 and Table 1 sort prior memory systems into two families and fault both for the same omission. Organization-focused systems (MemGPT, Mem0, A-MEM, SimpleMem, SCM) and maintenance-focused systems (MemoryBank with forgetting curves, various consolidation mechanisms) "evolve stored content but keep the retrieval infrastructure frozen." Adaptive retrieval work (Self-RAG, CRAG, FLARE, Adaptive-RAG) "adapt retrieval triggers or post-retrieval filtering, but none adapts the retrieval parameters (scoring weights, fusion mode, context budgets) over a deployed system's lifetime." Self-improving agents (Voyager, ExpeL, EvolveR, SkillRL, MemRL, Memory-R1, Agentic Memory, MemEvolve) are credited with optimizing behavioral policies or stored content but not "the retrieval mechanism itself as the research subject." Table 1 gives EvolveMem the only row with all five checkmarks: content evolution, policy and parameter evolution, typed memory, consolidation, offline evaluation.

### c) Core hypothesis / intuition

If the whole retrieval configuration is exposed as a structured, clamped action space, then a language model reading per-question failure logs can search that space more effectively than grid or Bayesian search, and a small set of numeric guards (revert on regression, perturb on stagnation, stop on convergence) is enough to make the search safe.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

Three layers with one feedback loop around them (Sec. 3, Figure 2).

**Layer 1, the structured memory store (Sec. 3.1).** Input is a raw conversation $S = (u_1, \dots, u_T)$. A sliding window of length $W$ partitions it into overlapping segments; for each window the backbone language model emits a set of typed memory units, given the previous window as context to avoid duplication. Each unit is a tuple $m = (c, \mu, e, \eta)$ where $c$ is natural-language content, $e$ is a dense embedding, $\mu$ is one of six memory types (episodic, semantic, preference, project state, working summary, procedural), and $\eta$ holds importance, confidence, an entity-reinforcement score, extracted entities including persons and locations, topics, and a creation timestamp. Three extraction guards handle failures (eq. 6): retry with increasing wait on a failed call, preserving partial results; split a window that exceeds the context limit into 15-turn sub-windows and merge their outputs; and a coverage verifier that compares extracted memories against reference keywords from the source and re-extracts the missing subset. Three consolidation passes then maintain the store: deduplication merges any pair whose Jaccard similarity exceeds $\tau_J = 0.80$, keeping the higher-importance unit; importance decays linearly at $\alpha_d = 0.05$ per day with floor $\iota_{\min} = 0.15$; entity reinforcement adds $\delta_\rho = 0.05$ each time a memory's entities co-occur with a new query, capped at $\rho_{\max} = 0.30$.

**Layer 2, retrieval as an editable action space (Sec. 3.2).** A query goes to three independent views, each returning its own top-$k$: a lexical view scoring BM25 with $k_1 = 1.5$, $b = 0.75$ (eq. 7), a semantic view scoring embedding cosine similarity (eq. 8), and a structured view counting metadata-field overlap on persons, locations, and entities (eq. 9). The three candidate sets are combined under a fusion mode drawn from sum, weighted sum, and reciprocal rank fusion (RRF), where RRF sets the fused score to $\sum_v 1/(k + r_v(m_i))$ over per-view ranks. Two optional pre-retrieval mechanisms extend this: adversarial entity-swap strips detected person names and re-searches by topic, unioning the result with the original set (eq. 10); query decomposition splits a multi-hop question into at most $N_{\text{sub}}$ single-hop sub-queries and merges by RRF (eq. 11). An answer-generation model then produces a candidate answer under a configurable style, and an optional second pass reviews it when self-reported confidence falls below $\tau_{\text{ver}}$ or the answer falls in the "Unknown" or "not specified" class (eq. 13). Every one of these settings can be overridden per question category.

**Layer 3, the self-evolution engine (Sec. 3.3).** Each round $r$ runs the current configuration over the full evaluation question set, writes a per-question raw log, hands that log to a diagnosis language model, and applies the returned adjustment through a guard. The paper argues against standard hyperparameter search for this: "the space mixes continuous parameters (weights, budgets) with discrete choices (fusion mode, answer style, per-category overrides), and the objective requires a full evaluation pass per configuration."

### b) Architecture components

**The diagnosis module.** After round $r$ the evaluator writes $L_r = \{(q_j, \hat{y}_j, y_j^*, \text{score}_j, R(q_j; \theta_r))\}_{j=1}^{|Q|}$ to `raw_results.jsonl` (eq. 14) — every question, prediction, reference, score, and retrieved source. The diagnosis module invokes a language model with a structured rubric over common failure patterns and returns a structured proposal $\Delta\theta_r$ naming which parameters to change and to what. The paper's generality claim rests here: "The rubric is written in terms of failure patterns rather than specific benchmarks, so newly discovered configuration dimensions become immediately usable without rubric modification. This is how the evolution mechanism is self-expanding: the diagnosis LLM can propose entirely new parameters that were not in the original action space." The prompt is printed verbatim in Appendix F.6, and it does not match that description — see section 4d.

**The guarded meta-analyzer.** A three-branch update rule wraps the raw proposal (eq. 4). It reverts, explores, or applies; nothing else.

**Coverage-gap re-extraction.** If diagnosis returns a non-empty missing-keyword set, the engine augments the store with a targeted re-extraction pass before the next round (eq. 15), which is the only path by which evaluation feeds back into layer 1.

### c) Formulas and algorithms

**The action space (eq. 2)**: $\theta = (k_{\text{sem}}, k_{\text{kw}}, k_{\text{str}}, B_{\text{ctx}}, \text{mode}, \{w_v\}, \alpha, \{\theta_c\}_{c \in C}) \in \Theta$ — three per-view candidate counts, a context budget, the fusion mode, per-view fusion weights, an answer style, and a per-category sub-configuration that can override any global parameter. "Every dimension is clamped to a safe range before any proposed value is applied." Table 7 enumerates ten groups with their ranges: view top-$k$ integers in $[3, 30]$, `max_context` in $[6, 30]$, fusion weights in $[0.1, 2.5]$, `reflection_rounds` in $[0, 3]$, entity-swap and decomposition toggles with two integers each, a verification toggle with two styles, six answer-style branches, a time-decay half-life or null, and the per-category override map.

**The ranking score (eq. 1)**: $s(q, m_i; \theta) = s_{\text{fuse}}(q, m_i; \theta) + \lambda_\iota \iota_i + \lambda_r \operatorname{rec}(m_i) + \rho_i$, adding importance, a recency factor, and the entity-reinforcement score to the fused relevance.

**The objective (eq. 3)**: $\theta^* = \arg\max_{\theta \in \Theta} F(\theta; K, Q)$ with $F(\theta; K, Q) = \frac{1}{|Q|}\sum_{(q, y^*) \in Q} \operatorname{score}(\hat{y}(q; \theta, K), y^*)$, the mean token-level F1 over the evaluation question set.

**The update rule (eq. 4)**, the load-bearing part for us, quoted in full:

$$\theta_{r+1} = \begin{cases} \theta^\star_{r-1} & \text{if } f_{r-1} - f_r > \tau_{\text{rev}} \quad (\text{revert}) \\ \theta_r \oplus \eta_{\text{exp}} & \text{if } |f_r - f_{r-1}| < \epsilon \text{ for 2 consecutive rounds} \quad (\text{explore}) \\ \operatorname{clamp}_\Theta(\theta_r \oplus \Delta\theta_r) & \text{otherwise} \quad (\text{apply}) \end{cases}$$

The revert branch compares the current round against **the immediately preceding round**, not against the best so far, and rolls back to the best-so-far configuration when the one-round drop exceeds $\tau_{\text{rev}}$. The explore branch adds a random perturbation $\eta_{\text{exp}}$. The apply branch projects the proposed deltas onto each parameter's valid range. Appendix A gives $\epsilon = 0.005$ (0.5 percentage points); $\tau_{\text{rev}} = 0.01$ appears only in the Appendix C.1 case-study caption, not in the appendix that Section 3.3 points to for threshold values. No value for $\eta_{\text{exp}}$ is given anywhere.

**Termination and the reported number (eq. 16)**: the engine stops at $f_r - f_{r-1} < \epsilon$ or $r \geq R_{\max}$ and returns $\theta^\star = \arg\max_{0 \le r \le R} f_r$. Algorithm 1 line 17 makes this concrete: `if r > 0 and f_r − f_{r−1} < ε then break`. **Every reported number is therefore the best round, not the last round.** This stopping rule is also inconsistent with the paper's own reported run — see section 4d.

**Algorithm 2 (Appendix B)** places the loop inside a session stream: per session, ingest turn pairs, consolidate, retrieve and answer, record the per-question raw result, and "if $|K_{\text{active}}| \ge 5$ and new records since last round, execute Algorithm 1." Nothing in Algorithm 2 scores a session before the configuration that will be updated on it; the loop is offline over a fixed question set, and the session stream only supplies content.

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Against every memory system in Table 1, the difference claimed is that the retrieval mechanism is itself an optimization target rather than a set of deployment-time constants. Against automated database and index tuning (cited as $\lambda$-tune and reinforcement-learning index optimization), the difference is that the proposer is a language model reading per-question failure text rather than a workload-statistics optimizer. Against workflow and prompt optimizers, the difference is scope in the other direction: EvolveMem edits only a typed configuration record, never code.

### b) Key innovations and their significance

1. **Exposing a full retrieval configuration as one clamped, typed action space that a language model edits from failure logs — notable.** It is a clean, cheap formulation, and the released implementation is honest about where the surface lives: one dataclass. It is also much narrower than the paper's framing suggests, because nothing structural can be added; only fields already implemented can be switched on.
2. **The guarded update rule (eq. 4) — incremental.** Three branches with three constants. The revert branch is a one-round look-back with a fixed tolerance; the explore branch is undefined noise; the stop rule is a one-round improvement test. The released code replaces all three (see `repo_analysis.md`).
3. **The "self-expanding action space" claim — not supported by the paper's own text.** The abstract says the process discovers "entirely new configuration dimensions not present in the original action space", and Section 4.3 names them: adversarial entity-swap, query decomposition, answer verification. All three are printed as rows of Table 7, the paper's own enumeration of the action space ("Entity swap", "Decomposition", "Verification"), and all three are named as triggers in the Appendix F.6 diagnosis rubric (rules 4, 5, and 8). They were in the action space before the run started.
4. **Cross-benchmark transfer of an evolved configuration — notable as a measurement.** Table 5 is the only place in the paper where a configuration is scored on something other than the set it was tuned on, and it is the one result we can reuse directly.

### c) Applicability

The method works where the target distribution is a fixed, categorized question set with enough per-category signal for a language model to name the right lever, and where the useful levers are already implemented. It has no story at all for the regime our claim is about. There is no arriving task stream: Section 3.3 fixes one question set $Q$ and re-scores it every round. There is no score-before-update protocol: the score that drives the edit and the score that is reported come from the same pass over the same questions. There is no measurement of retention, because no earlier set is held out to retain. And there is no notion of the configuration's remaining capacity to be edited later — the only forward-looking object in the paper is a one-round revert tolerance.

### d) Comparison table

| Method | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| EvolveMem | Whole retrieval configuration is one clamped, serializable action space; diagnosis reads per-question text rather than a scalar, so proposals are targeted; per-category overrides let one round help one category without forcing a global choice; cheap (25-35 min and one diagnosis call per round on a single sample, no GPU); Table 5 is a genuine cross-benchmark re-score | Reward is mean F1 on the one fixed question set the diagnosis already read; the revert tolerance is a one-round look-back that provably lets sub-threshold declines through (R4 to R5, section 4d); the reported number is an argmax over rounds; the stated stopping rule contradicts the reported trajectory; the "new dimensions" are rows of the paper's own Table 7; nothing structural is editable, only pre-implemented fields | Score an edit on questions the diagnosis did not read; keep and re-score an accumulating earlier-question set rather than one static set; report the final round alongside the best; publish the per-round configuration history and raw logs that Appendix D.3 says are persisted |
| HarnessX / AEGIS (`refs/related-work/HarnessX/`) | Edit space reaches tool and middleware source; per-task binary no-regression gate over the full evaluation set; full trace observability with falsifiable per-edit predictions; variant isolation keeps the aggregate trajectory non-degrading | Reward is the current adaptation batch only; no held-out evaluation; per-edit gating misses sub-threshold accumulation by the authors' own analysis; needs a frontier closed-weight meta-agent at 100M-175M tokens per benchmark | Score an edit against later batches; grow the retention set from the stream |
| Memory systems with frozen retrieval (Mem0, A-MEM, MemGPT, SimpleMem, MemoryBank) | Mature, cheap, no evolution cost; retrieval behavior is stable and auditable | Retrieval policy calibrated once and never revisited as the store grows | Expose a small number of retrieval settings to an outer loop |
| Automated database and index tuning ($\lambda$-tune, reinforcement-learning index optimization) | Principled search with workload statistics; well-studied objective | No per-question failure semantics; cannot propose a mechanism, only tune existing knobs | Not applicable to our setting |

---

## 4. Experimental Validation

### a) Experimental design

Two benchmarks (Sec. 4.1). LoCoMo, the full LoCoMo-10 release: 10 conversations of 19-32 sessions and 369-689 turns, 1,986 question-answer pairs across five categories (single-hop, temporal, multi-hop or inferential, open-domain, adversarial name-swap), scored with token-level F1 and BLEU-1. MemBench, a memory-tool-use benchmark: 28 samples drawn as 7 low-level categories by 2 topics by 2 samples, scored with multiple-choice exact-match accuracy. Baselines are six memory systems on LoCoMo (MemVerse, Mem0, Claude-Mem, A-MEM, MemGPT, SimpleMem) and four on MemBench (RecentMemory, MemGPT, MemoryBank, SCMemory).

Implementation (Sec. 4.1, Appendix D-E): SQLite with FTS5 for storage, BAAI/bge-base-en-v1.5 at 768 dimensions for embeddings, an initial configuration of BM25-only fusion with the semantic and structured views disabled, $k_{\text{kw}} = 5$, $B_{\text{ctx}} = 8$, entity-swap and decomposition off, and $R_{\max} = 7$ rounds. All experiments ran on a single machine with an Apple M-series processor and no GPU. Appendix E is important to read against Table 2's "Backbone" column: "LLM calls for extraction and diagnosis use GPT-5.1 via Azure OpenAI API. Answer generation uses GPT-4o for all categories." The two backbone rows therefore differ in the answer-generation model only, and the diagnosis model is the same in both. Appendix E also states that default values "were chosen based on preliminary experiments on a held-out validation set (2 LoCoMo samples) and remained fixed for all reported results" — the only held-out data in the paper, used to set constants, not to score any edit.

### b) Key results

1. **LoCoMo headline (Table 2)**: overall F1 0.543 with GPT-4o answer generation against SimpleMem's 0.432, a 25.7% relative gain; and 0.543 against the 0.305 minimal starting point, a 78.0% relative gain (+23.8 percentage points). With GPT-5.1 answer generation, 0.572 against SimpleMem's 0.418, 36.8% relative. Per-category, the biggest gains are temporal (0.235 to 0.384, +63.4%) and single-hop (0.195 to 0.329, +68.7%).
2. **MemBench (Table 3)**: 67.9% overall with GPT-4o against a strongest baseline of 57.1%, 18.9% relative; 71.4% against 64.3% with GPT-5.1, 11.0% relative. Robustness is the weakest column (50.0% with GPT-4o, below MemoryBank's 75.0%), which the authors localize to missing memories rather than retrieval settings.
3. **The evolution trajectory (Table 4)**, single backbone, GPT-4o: R0 30.5 (BM25-only), R1 35.8 (intent planning plus RRF fusion), R2 34.8 (MMR diversity on category 3, reverted), R3 37.2 (entity-swap for category 5), R4 38.5 (per-category answer-style flags), R5 38.1 (query decomposition for categories 1 and 4), R6 45.4 (category 3 inferential subtypes plus entity-swap expansion), R7 54.3 (answer verification plus a hyperparameter sweep plus context-budget tuning). Two-thirds of the total 23.8-point rise arrives in the final two rounds.
4. **Cross-benchmark transfer (Table 5)**: the LoCoMo-evolved configuration scores 0.543 on LoCoMo and 54.3 on MemBench zero-shot; continuing evolution on MemBench yields 79.2 on MemBench and **raises LoCoMo from 0.543 to 0.593**, a Pareto improvement on both; evolving on MemBench from scratch reaches only 67.9. The authors read this as evidence that "the self-evolution process captures universal retrieval principles rather than benchmark-specific heuristics."
5. **Ablation (Table 6)**: removing the extraction guards is by far the most damaging (54.3 to 31.08, $-23.22$); then semantic search ($-10.32$), the diagnosis module replaced by random perturbation over the same action space ($-9.63$, landing at 44.67), BM25 ($-6.87$), entity-swap ($-3.10$), query decomposition ($-2.84$), structured metadata ($-2.33$), **self-evolution ($-2.03$, landing at 52.27)**, and answer verification ($-1.83$). The all-disabled baseline is 30.50, $-23.80$ from the full system.

### c) Where it shines

The categories that gain most are the ones with a single named mechanism attached: temporal (fixed by enabling the time-decay half-life) and adversarial name-swap (fixed by entity-swap). The Appendix C.1 case study makes the mechanism concrete at single-question resolution on one open-domain aggregation probe from conversation 26: the probe's F1 moves 0.00, 0.44, 1.00, 0.94, 1.00 across rounds, with the diagnosis model naming the exact confusion ("camping trip vs. Perseid meteor shower") from the round-0 log alone. Across the 70 category-4 probes in that sample, zero-F1 cases fall 26, 12, 11, 16, 9 and per-sample category-4 F1 rises from 0.350 at R0 to 0.520 at R4; at the population level category 4 rises from 41.0% at R0 to 49.6% at R7, +8.6 points.

### d) Limitations

**Acknowledged.** Only one: MemBench robustness is capped by store coverage, "indicating a coverage limitation that retrieval-level adjustments cannot resolve" (Sec. 4.2). The paper has no limitations section, no discussion of selection bias, no error bars, and no seed count anywhere.

**Unacknowledged, and material.** Five, in descending order of how much they matter to us.

1. **The reported score is an argmax over rounds on the set that produced it.** Equation 3 maximizes over $\Theta$, equation 16 returns $\arg\max_r f_r$, and equation 14 shows that the diagnosis module reads the per-question log of that same set. There is no held-out split in the evolution loop at any point; the two-sample held-out set in Appendix E only fixes constants. Every headline number is therefore a maximum over correlated round estimates on the questions the proposer was shown.

2. **The stated stopping rule contradicts the reported trajectory.** Algorithm 1 line 17 breaks when $f_r - f_{r-1} < \epsilon = 0.005$. In Table 4, $f_2 - f_1 = -0.010$, which satisfies the break condition, so the run should have terminated at R2 with F1 34.8. It also holds at R5 ($-0.004$). The reported eight-round trajectory ending at 54.3 could not have been produced by the algorithm as printed. Either the implementation differs from Algorithm 1 — which the released code confirms, using a different stopping rule entirely — or the trajectory was not produced by the printed loop.

3. **The revert guard misses exactly the decline our claim is about.** From R4 to R5, F1 falls 38.5 to 38.1, a drop of 0.004, which is below $\tau_{\text{rev}} = 0.01$, so the round is applied and the loop carries the degraded configuration forward. This is the same sub-threshold accumulation that HarnessX names as its own worst failure mode (`refs/related-work/HarnessX/paper_analysis.md`, Sec. 7.6: "five consecutive same-type edits accumulated sub-threshold coupling undetected by the seesaw constraint"). Separately, the one revert the paper does report is at best borderline at the printed precision: the R1-to-R2 drop is exactly 1.0 point, and the rule requires $f_{r-1} - f_r > 0.01$ strictly.

4. **The autonomy claim is undercut by the paper's own printed prompt.** Section 3.3 states the rubric "is written in terms of failure patterns rather than specific benchmarks", and Section 4.3 states each discovered dimension was found "by inspecting raw failure logs and proposing structural improvements, rather than by a benchmark-specific patch." Appendix F.6 prints the prompt. It contains a nine-item "Decision Rubric" mapping symptoms to named fields — "If temporal category weakness -> enable time_decay_half_life_days", "If adversarial category weakness -> enable_entity_swap=true", "If multi-hop weakness -> reflection_rounds >= 1", "If residual 'Unknown' or format-mismatch -> enable_answer_verification=true" — and a ninth rule that is benchmark-specific by name: "LoCoMo prompt-surface flags are highest-ROI when their symptom matches; propose them early." It also contains a "Tier-1 TODO checklist: levers that are still OFF in the incumbent ... Prefer picking ONE item per round until empty." The loop is walking down a hand-written checklist of already-implemented switches, and the paper's own appendix shows it.

5. **The ablation makes the loop look small.** Removing self-evolution costs $-2.03$ F1 while the headline delta over the all-disabled baseline is $+23.80$. Replacing the diagnosis model with random perturbation over the same action space still reaches 44.67, above every published baseline in Table 2. Read together, roughly 92% of the headline gain is attributable to the components being switched on at all, and a random search over the same space recovers most of what the language model finds. The paper does not state what "$-$ Self-evolution" holds fixed, so the exact reading is uncertain, but no reading makes the loop the dominant term.

Two smaller observations. Table 5's cross-benchmark row reports the LoCoMo-evolved configuration scoring 0.543 on LoCoMo (token F1) and 54.3 on MemBench (multiple-choice accuracy) — identical to three digits across two different metrics, which is worth verifying before citing. And that 54.3 zero-shot score is below MemBench's strongest baseline of 57.1, although the paper calls the transfer "effective" without comparing it to the baselines in its own Table 3.

---

## 5. Reproduction and Application

### a) Open source?

Yes, as a subdirectory of a larger repository: https://github.com/aiming-lab/SimpleMem, under `EvolveMem/`. MIT licensed. The clone in `repo/` is the parent SimpleMem repository at commit `db80b6a` (24 July 2026), about 110 MB, of which the EvolveMem subdirectory is 36 files and roughly 17,450 lines of Python. Full inventory in `repo_analysis.md`. Reproduction steps from the subdirectory README are three commands: install requirements, export an API key and model name, then `python run_evolution.py --data data/locomo10.json --max-rounds 7` or `python run_benchmark.py locomo --sample 0 --initial weak --max-rounds 3`. Neither the LoCoMo data nor any evolution output is in the repository.

Four gaps between paper and release matter for reproduction, all documented with line references in `repo_analysis.md`. The released gate is not equation 4 — it is hill-climbing against the best so far with an acceptance label at 0.003 and a five-round no-accept early stop, and the guarded meta-analyzer's revert and parameter branches are explicitly skipped in the default mode. The released action space is 42 fields, not Table 7's ten groups, with different ranges. The released diagnosis prompt adds a library of fourteen pre-validated recipes and a priority list quoting per-lever gains in percentage points. And the release contains a hand-written "full evolved configuration" function plus a default that injects it at round 5 of a 7-round LoCoMo run.

### b) Implementation details requiring attention

The cost model is the evaluation pass, not the proposer: Appendix D.3 gives 25-35 minutes for a full seven-round run on one LoCoMo sample of 200 questions and about 900 memories, of which 15-20 minutes per round is question answering and about 15 seconds is diagnosis. Index construction is about 5 seconds and per-query retrieval about 15 milliseconds; answer verification adds one language-model call and 2-3 seconds per question. Storage is under 5 MB per 1,000 memory units. The diagnosis model (GPT-5.1) is stronger than the answer-generation model (GPT-4o) in the headline configuration, which is worth holding fixed in any comparison. Three constants govern everything: $\epsilon = 0.005$, $\tau_{\text{rev}} = 0.01$, and $R_{\max} = 7$; the exploration perturbation $\eta_{\text{exp}}$ has no printed value.

Appendix D.3 claims that "for every run we persist: (i) per-round config snapshot, (ii) complete `raw_results.jsonl` containing every question, prediction, reference, every metric, and retrieved sources, (iii) per-round summary, (iv) the best-so-far configuration snapshot ... and (v) extracted memory cache." The code writes all five. **None of them is in the repository**; `evolution_results/` and `*.json` are both in the ignore list, and a repository-wide search finds no `.jsonl` file and no `evolution_final.json`.

### c) Transferability

The method transfers to any setting where an evaluation set is categorized and the useful mechanisms are pre-implemented; it does not transfer to settings where the mechanism has to be written. Below, what this means for us.

#### Transfer to our own work

**Where the paper sits relative to our claim.** EvolveMem is another instance of the horizon $\gamma = 0$ case, and a cleaner one than most because the objective is written down. Equation 3 maximizes mean F1 over one fixed question set $Q$; equation 14 shows the diagnosis module reads the per-question log of that same $Q$; equation 4's revert branch compares round $r$ to round $r-1$ on that same $Q$. Nothing after the current round enters any decision. There is no future term, no discount, and no second set. The system has a memory layer and nothing else — in our layer ordering it edits long-term-memory retrieval settings and the answer prompt surface, and nothing deeper.

**Exactly what the regression is measured on.** The same questions the diagnosis read, with no split. Equation 3 defines $F$ over $Q$; Algorithm 1 line 4 runs retrieval and answering over $Q$; line 5 scores it and writes the log; line 6 hands that log to diagnosis; line 7 compares the resulting round score to the previous round score. One set, one number, three uses. The only held-out data in the paper is the two LoCoMo samples in Appendix E used to fix constants before any run.

**The tolerance and thresholds.** Revert fires when the one-round drop exceeds $\tau_{\text{rev}} = 0.01$ (1 percentage point of mean F1). Explore fires when $|f_r - f_{r-1}| < \epsilon = 0.005$ for two consecutive rounds. Convergence fires when $f_r - f_{r-1} < \epsilon$ (eq. 16). All three are aggregate-mean tests; nothing is per-question and nothing is per-category, so a round that trades one category against another is invisible to the guard by construction. For comparison, HarnessX's gate is per-task binary (no previously passing task may flip) plus, in its release, an aggregate tolerance of 0.03 with a three-task-flip requirement; EvolveMem's is aggregate-only at 0.01 with a one-round look-back. EvolveMem's is therefore the weaker of the two same-round gates on both axes, and HarnessX's own Section 7.6 already shows the stronger version failing.

**Whether earlier rounds' questions are re-scored.** No. There is no accumulating set and no anchor set of any kind. The one retention-shaped measurement in the entire paper is Table 5: after evolving on LoCoMo and then continuing to evolve on MemBench, LoCoMo is re-scored and **rises** from 0.543 to 0.593 (+9.2% relative). This is a single measurement, in one direction, on two benchmarks, with no protocol behind it, but it is a real re-score of an earlier set after later editing and it is a data point against a naive catastrophic-forgetting expectation for configuration-level edits. It is worth citing exactly for what it is: one positive transfer observation at the retrieval-configuration layer, not evidence about retention under a stream.

**What the trajectory looks like, and whether the number is final or best.** The reported number is $\theta^\star = \arg\max_r f_r$ (eq. 16) — the best round, by definition. In Table 4 the best round happens to be the last (R7, 54.3), so the reported peak-to-final gap is zero, unlike HarnessX's $-24.3$ points on GAIA. But the trajectory is not monotone: R2 falls 1.0 point and is reverted, and R5 falls 0.4 points and is **not** reverted because the drop is below the tolerance. That is the mechanism we care about, visible in a seven-round run on a static set; it is HarnessX's sub-threshold accumulation at a smaller scale and with a weaker gate. The trajectory is also only seven rounds long on one benchmark, which is far too short for a peak-then-decline pattern to be either confirmed or excluded.

**Whether a usable harness-evolution log is released.** No. Appendix D.3 promises five persisted artifacts per run and the code writes all five, but the repository ships none of them, and its ignore rules exclude the directories they would land in. There is no per-round configuration history, no failure log, and no trajectory file we could use to pretrain a critic $\hat{\Phi}(h)$. The only round-level data in the public record is the eight rows of Table 4 and the four rows of Table 8, both transcribed by hand into the paper.

**What is reusable.** Three things. First, Table 5's protocol is directly reusable as a cheap retention probe: evolve on set A, evolve further on set B, re-score A. It costs one extra evaluation pass and it is the measurement the whole field is skipping. Second, the released `raw_results.jsonl` schema (eq. 14: question, prediction, reference, per-question score, retrieved sources) is the right minimum record for a per-question retention check, and it is what an anchor set would read. Third — and this is not in the paper at all — the release contains an undocumented second subsystem that replays candidate configurations against stored past interaction records and promotes only on a multi-criterion delta gate with a minimum sample count and a cap on newly empty retrievals. That is the closest thing in this repository to an anchor-set mechanism, and it is described with line references in `repo_analysis.md`.

**What is not reusable.** The guard as specified: a single aggregate mean with a one-round look-back and a 1-point tolerance is weaker than what we already have from HarnessX, and the paper's own R5 shows it failing. The "self-expanding action space" framing: the dimensions are enumerated in Table 7 before the run. And the headline delta: `weak_initial_config()` in the release documents in its own docstring that the starting point was chosen so that the evolved-minus-static difference would be the paper's headline.

---

## 6. Summary

### a) One-sentence core idea

Expose a memory system's whole retrieval configuration as a clamped action space and let a language model reading failure logs edit it each round.

### b) Quick-reference pipeline

1. Turn a long conversation into a table of small typed facts, each with an embedding, its entities, a timestamp, and an importance number; merge near-duplicates and re-extract anything a keyword check says is missing.
2. Answer a question by searching that table three ways at once — keyword match, embedding similarity, and metadata match — merging the three ranked lists, and passing the top few facts to a language model that writes the answer.
3. Score every question in a fixed list and write one line per question recording the question, the answer given, the correct answer, the score, and which facts were retrieved.
4. Give that file to a second language model, which reads the failures and returns a small set of settings to change — how many results each search returns, how to merge them, whether to strip names before searching, whether to split the question first, whether to double-check the answer, and whether to do any of this differently for one category of question.
5. Apply the change and score the same list again; if the score dropped by more than one point since last round, roll back to the best settings seen so far; if it barely moved twice in a row, jitter the settings; stop when a round no longer improves, and report the best round.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched "EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents", "EvolveMem", and arXiv:2605.13941 against OpenReview and the open web on 2026-09-15). The paper is marked "Preprint." on its title page. The subdirectory README carries a "NeurIPS 2026" badge, but no NeurIPS 2026 submission, decision, or review record is publicly visible, and the badge links only to the README's own citation section.

One near-miss is worth recording so a future reader does not mistake it for this paper: https://openreview.net/forum?id=dfPQrg1WA5 is "EVOLVE-MEM: A Self-Adaptive Hierarchical Memory Architecture for Next-Generation Agentic AI", a different paper with a different author list, which is not this work. A second distinct paper, MemEvolve (arXiv:2512.18746), is cited by this one as reference [38].
