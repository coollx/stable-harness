# MemEvolve: Meta-Evolution of Agent Memory Systems

- **Original title**: MemEvolve: Meta-Evolution of Agent Memory Systems
- **Authors**: core contributors Guibin Zhang, Haotian Ren; contributors Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu; corresponding authors Wangchunshu Zhou, Shuicheng Yan. (The OpenReview record lists seven authors and omits Zhenhong Zhou; the arXiv contributions page lists eight.)
- **Affiliations**: the byline reads "OPPO AI Agent Team, LV-NUS lab". No per-author affiliation is printed. Contact address given on the contributions page: guibinz@outlook.com.
- **Venue / year**: ICML 2026, accepted as a poster (Submission3358). Preprint: arXiv:2512.18746v1 [cs.CL], submitted 21 December 2025, paper dated 23 December 2025; 22 pages including appendices A-C.
- **Source URL**: https://arxiv.org/abs/2512.18746
- **Conference page**: https://icml.cc/virtual/2026/poster/61379
- **OpenReview**: https://openreview.net/forum?id=qpkG0eKx4v (public; three official reviews and two rebuttal threads — see section 7)
- **Code**: https://github.com/bingreeky/MemEvolve (Apache-2.0) — see `repo_analysis.md`.

**Classification: method paper.** The dominant contribution is two built artifacts others could reproduce — the MemEvolve dual-evolution loop and the EvolveLab codebase that re-implements twelve memory systems behind one interface — with the cross-system measurements of Table 3 as supporting evidence rather than as the finding itself.

Short names used below, all taken from the paper: MemEvolve (the meta-evolutionary framework), EvolveLab (the unified codebase), and the three discovered memory systems the paper names Lightweight, Riva, and Cerebra. This paper is unrelated to EvolveMem (arXiv:2605.13941) despite the similar name.

---

## 1. Method Motivation

### a) Why was this method proposed?

The paper's target is what it calls the "staticity of the memory system itself" (abstract). Self-improving memory lets an agent accumulate experience, but the pipeline that decides how experience is ingested, abstracted, and recalled is written once by a human and then frozen: "Researchers typically design a fixed memory pipeline (i.e., memory ingestion/abstraction/retrieval) and embed it within an agent, assuming it will sustain long-term evolution through mere exposure to new experiences" (Sec. 1). The authors' objection is that "distinct tasks are coupled with distinct memory affordances" — a skill library of reusable web-browsing functions is useless for mathematical reasoning, and a self-critique memory that helps reasoning degrades on tool use — so "a memory system that cannot adapt itself to the task at hand is fundamentally misaligned with the very premise of open-ended agent evolution" (Sec. 1).

The framing analogy, carried by Figure 2, is a ladder of human learners: a poor learner rote-memorizes errors, a skillful learner distills transferable insights, and an adaptive learner "dynamically alter[s] their learning strategies based on the subjects". Current memory systems are placed at the skillful rung; the paper's stated goal is the step to adaptive.

### b) Pain points of existing methods

Three concrete deficits are named. First, no existing self-improving memory generalizes: Sec. 5.3 reports that in the authors' own faithful re-implementations, DILU gains on xBench-DeepSearch and WebWalkerQA but "degrades GAIA by 2.42%", Dynamic Cheatsheet gains 1.76% on WebWalkerQA but "performs poorly on GAIA and xBench", and "ExpeL underperforms on all three benchmarks" because it was designed for ALFWorld-style and HotpotQA-style settings and "its prompts and mechanisms are ill-suited for long-horizon, long-context deep research". Table 3 quantifies this: against a no-memory Flash-Searcher baseline of 69.09 on GAIA, four of the seven human-designed memory systems score below it.

Second, the field has no shared implementation substrate, so cross-system comparison is confounded by framework differences. Third, the memory interface $\Omega$ "itself remains immutable" (Sec. 4.1), which structurally rules out architectural adaptation no matter how much experience accumulates.

### c) Core hypothesis / intuition

If any memory system can be written as four substitutable code slots — encode, store, retrieve, manage — then a language model reading execution traces can diagnose which slot is failing and rewrite it, so the memory architecture becomes a second, slower learning loop stacked on top of ordinary experience accumulation.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

The system has two parts: EvolveLab, the design space and codebase (Sec. 3), and MemEvolve, the dual-evolution loop that searches inside it (Sec. 4).

**The design space (Sec. 3.2).** Any memory system is decomposed as $\Omega = (\mathcal{E}, \mathcal{U}, \mathcal{R}, \mathcal{G})$: encode, store, retrieve, manage.

- Encode $\mathcal{E}$ turns a raw experience — a trajectory segment $\tau_t = (s_t, a_t, s_{t+1})$, a tool output, a self-critique — into a structured record $e_t = \mathcal{E}(\epsilon_t)$. "Encoding may be as simple as compressing raw traces or as sophisticated as extracting generalizable lessons."
- Store $\mathcal{U}$ commits the record to persistent memory, $M_{t+1} = \mathcal{U}(M_t, e_t)$; the paper names vector databases, knowledge graphs, and plain JSON as instances.
- Retrieve $\mathcal{R}$ returns task-relevant content, $c_t = \mathcal{R}(M_t, s_t, Q)$, which conditions the agent's next action.
- Manage $\mathcal{G}$ performs "offline and asynchronous operations such as consolidation, abstraction, or selective forgetting", $M_t' = \mathcal{G}(M_t)$.

The paper is explicit that this quadruple is the "genotype" on which the meta-evolution operates. Table 1 places twelve published systems in it — Voyager, ExpeL, Generative Agents, DILU, Agent Workflow Memory, Mobile-E, Dynamic Cheatsheet, SkillWeaver, G-Memory, Agent-KB, Memp, EvolveR — along four axes: single- versus multi-agent support, step-wise versus trajectory-wise granularity, on-the-fly versus offline update, and the four component implementations. The distribution is lopsided: eight of the twelve store to a vector database, four use plain semantic search for retrieval, and **seven of the twelve have no manage component at all** (Table 1, "Manage" column reads N/A for Voyager, ExpeL, Generative, DILU, Agent Workflow Memory, Mobile-E, Dynamic Cheatsheet). Only SkillWeaver (skill pruning), G-Memory (episodic consolidation), Agent-KB (deduplication), Memp (failure-driven adjustment) and EvolveR (update and pruning) implement any forgetting or consolidation.

**The agent-level formalism (Sec. 3.1).** An agentic system is $\mathcal{M} = \langle I, S, A, \Psi, \Omega \rangle$. At each step the active agent acts as $a_t = \pi_{\mu(t)}(s_t, H_t, Q, c_t)$ with $c_t \sim \Omega(M_t, s_t, H_t, Q)$; after the task, the recorded trajectory $\tau = (s_0, a_0, \dots, s_T)$ is scored by a terminal reward $R(\tau)$ and folded in as $M_{t+1} = \Omega(M_t, \epsilon)$. Two facts matter for us: the task is scored before the memory absorbs it, and the reward that the memory absorbs is the task's own reward — nothing later.

**Inner loop (Sec. 4.1, experience evolution).** At evolutionary iteration $k$, each candidate memory system $\Omega_j^{(k)}$ starts from an **empty** memory state and is run over a batch of trajectories $T_j^{(k)}$:

$$M_{t+1,j}^{(k)} = \Omega_j^{(k)}(M_{t,j}^{(k)}, \epsilon_\tau), \qquad \epsilon_\tau \in \mathcal{E}_j^{(k)}(\tau).$$

Each trajectory yields a three-dimensional feedback vector $f_j(\tau) \in \mathbb{R}^3$ measuring task success, token consumption, and latency; an aggregation operator $S$ summarizes them into $F_j^{(k)} = S(\{f_j(\tau)\}_{\tau \in T_j^{(k)}})$.

**Outer loop (Sec. 4.1, architectural evolution).** A meta-evolution operator $\mathcal{F}$ maps the current candidate set plus its summaries to the next candidate set:

$$\{\Omega_{j'}^{(k+1)}\}_{j' \in J^{(k+1)}} = \mathcal{F}\big(\{\Omega_j^{(k)}\}_{j \in J^{(k)}}, \{F_j^{(k)}\}_{j \in J^{(k)}}\big).$$

The iteration-0 candidate set is a singleton, $|J^{(0)}| = 1$, seeded with a hand-designed memory system. The paper's "unified view" equation makes the reset explicit: each iteration maps $(\{\varnothing\}, \{\Omega^{(k)}\})$ through the inner loop to $(\{M^{(k)}_{t+1}\}, \{\Omega^{(k)}\})$ and then through the outer loop to $(\{M^{(k)}_{t+1}\}, \{\Omega^{(k+1)}\})$ — the memory content is rebuilt from empty at every architectural iteration.

**Selection (Sec. 4.2).** Each candidate's summary is the vector

$$F_j^{(k)} \triangleq (\mathrm{Perf}_j^{(k)}, -\mathrm{Cost}_j^{(k)}, -\mathrm{Delay}_j^{(k)}),$$

higher preferred on all three. Candidates are ranked by non-dominated sorting to give a Pareto rank $\rho_j^{(k)}$, ties broken by raw performance, and the top $K$ become the parent set $P^{(k)} = \mathrm{Top}\text{-}K(\rho_j^{(k)}, \mathrm{Perf}_j^{(k)})$. This is the only gate in the method: a candidate is accepted because it ranked first on the batch, not because it cleared any absolute bar.

**Diagnose-and-Design (Sec. 4.2).** For each parent, $\mathcal{F}$ produces $S$ descendants in two phases.

1. *Diagnosis.* "Each parent architecture is examined using trajectory-level evidence from its own execution batch $T_p^{(k)}$." Per trajectory the system sees outcome statistics (success indicator, token cost) plus a structured description of the query, and "a replay interface grants access to the corresponding trajectories", enabling inspection of "retrieval failures, ineffective abstractions, or storage inefficiencies". The output is a structured defect profile $D(\Omega_p^{(k)})$ indexed by the four components.
2. *Design.* Conditioned on the defect profile, a language model writes $S$ variants, $\Omega_{p,s}^{(k+1)} = \mathrm{Design}(\Omega_p^{(k)}, D(\Omega_p^{(k)}), s)$ for $s \in \{1,\dots,S\}$, "modifying only the permissible implementation sites within the modular interface" so every variant still satisfies the interface and remains executable.

The next candidate set is the union of all descendants over all parents.

### b) Architecture components

EvolveLab's runtime contract (appendix A.1) is narrower than the four-slot design space. The abstract base class `BaseMemoryProvider` declares exactly three abstract methods: `provide_memory(request) -> MemoryResponse` (retrieve), `take_in_memory(trajectory_data) -> (bool, str)` (encode and store together), and `initialize() -> bool`. The paper states the collapse plainly: "While `take_in_memory` primarily integrates the Encode and Store stages, the Manage functionality that is responsible for offline consolidation or selective forgetting is typically implemented as auxiliary methods within the provider classes or invoked during specific lifecycle events." So manage has no interface slot; it exists only where an individual provider chooses to implement it.

Three standardized data carriers (appendix A.2) make providers interchangeable across agent frameworks: `MemoryItem` (text, insight, or executable code, with timestamps, confidence scores, source identifiers), `TrajectoryData` (query, full state-action trace, terminal reward), and `MemoryRequest`/`MemoryResponse`. Evaluation supports an online mode where memory updates as a stream of tasks is processed and an offline mode where memory is built from a static trajectory set and then frozen for testing on unseen tasks (Sec. 3.3).

### c) Formulas and algorithms

The three equations that carry the method are reproduced above: the inner-loop memory update, the outer-loop operator $\mathcal{F}$, and the Pareto summary vector. In plain language: run each candidate memory system on a batch of tasks from a blank memory; score it on success, tokens, and latency; keep the best by Pareto rank; show a language model the losing trajectories and let it rewrite the four code slots; repeat.

The one quantity with no equation is the meta-objective. There is no term for performance on any task outside $T_p^{(k)}$, no term for retention of earlier tasks, and no penalty on architectural size — all three of which turn out to matter (sections 4 and 5 below).

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Mainstream self-improving memory fixes $\Omega$ and evolves $M$. MemEvolve evolves both, and the object it edits in the outer loop is source code, not content: the descendants are new Python provider classes, generated by a language model and validated by execution. Relative to workflow or prompt optimizers, the edited layer is different — this is the long-term memory layer of the harness, and the edit rewrites the memory system's implementation rather than the agent's prompt or graph.

Relative to the neural-architecture-search tradition it structurally resembles, the search is far smaller and the fitness far noisier: nine candidate architectures total across the whole run, each scored on 60 tasks.

### b) Key innovations

| Innovation | What it is | Significance |
|---|---|---|
| Four-slot memory design space | Any memory system as $(\mathcal{E},\mathcal{U},\mathcal{R},\mathcal{G})$; twelve published systems mapped into it (Table 1) | **Notable** — the taxonomy is clean and the re-implementation is real, but three of the four slots collapse to two methods in the released interface |
| Diagnose-and-Design meta-operator | A language model reads failed trajectories, writes a defect profile per slot, then rewrites the slots | **Significant** — the rebuttal ablation shows it beats both random composition (+16.66 versus +3.33 on TaskCraft) and greedy per-module enumeration (+8.33) while using 2.7 times fewer trajectories |
| Three-objective Pareto selection | Rank on (performance, negative cost, negative delay) before choosing parents | **Incremental** — standard multi-objective selection; and the released code defaults it off |
| EvolveLab as shared substrate | One interface, twelve providers, four benchmark runners | **Notable** — the most durable contribution; reviewer OTYdSsSay5 says it "could become a useful benchmark substrate for future work" |

### c) Applicability

Where it works: a single task family with a stable tool set and an automatic judge, where 180 agent rollouts per round is affordable. The paper is candid that transfer has limits: "Memory systems evolved on TaskCraft are unlikely to transfer effectively to fundamentally different task families (e.g., embodied action), where environments, action space and tool sets differ substantially" (Sec. 5.3).

Where it struggles: any setting where the batch is small enough that a 60-task score is noise (reviewer IGfMnz2OHM's central objection), where three generations cannot express a meaningful architectural change, or where memory content needs to survive an architecture change — the method deliberately discards it.

### d) Comparison table

| Approach | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| Hand-designed memory (Voyager, ExpeL, Agent Workflow Memory, Dynamic Cheatsheet, ...) | Cheap, inspectable, no meta-loop cost | Do not transfer: Table 3 shows four of seven below the no-memory GAIA baseline; ExpeL below on all three | Publish the task family a design was tuned for |
| MemEvolve | Consistent 3.54-5.0% gains across three benchmarks from one evolved system; cost held at the no-memory level (GAIA $0.085 versus $0.086) | Architecture chosen on the batch that produced the diagnosis; memory content wiped each round; only three generations; no size penalty | Score candidates on a held-out batch; keep an anchor set of earlier tasks; add a size or cost term to the accept rule |
| Random architecture composition (rebuilt in rebuttal) | Same search budget, no language model needed | Peaks at round 2 then falls: TaskCraft 63.33 / 66.67 / 65.00, WebWalkerQA 64.51 / 67.84 / 65.88 | — (this is the paper's own evidence of peak-then-decline) |
| Greedy per-module substitution (rebuilt in rebuttal) | Systematic; covers each slot | 480 trajectories per round versus 180, and still only +8.33 versus +16.66 | Combine enumeration with trajectory diagnosis |

---

## 4. Experimental Validation

### a) Experimental design

Four benchmarks (Sec. 5.1, appendix B.1): GAIA (165 tasks: 53 Level 1, 86 Level 2, 26 Level 3), WebWalkerQA (a 170-query sample of 680), xBench-DeepSearch (100 tasks), and TaskCraft (a 300-query working subset, of which 120 are used for evolution). Two frameworks are evolved on — SmolAgent and Flash-Searcher — and two multi-agent frameworks are held out for transfer only: Cognitive Kernel-Pro and OWL.

Evolution configuration: $K_{\max} = 3$ iterations, survivor budget $K = 1$, descendants $S = 3$. "In the inner loop, each candidate architecture $\Omega_j^{(k)}$ is evaluated on a batch $T_j^{(k)}$ of 60 task trajectories, consisting of 40 newly sampled tasks and 20 tasks reused from the previous iteration to stabilize inter-iteration comparison" (Sec. 5.1). Appendix B.1 confirms 40 fresh queries per round on both GAIA (Level 1 plus 67 TaskCraft queries) and TaskCraft (120 queries across three rounds).

Backbone: GPT-5-mini both for the agent frameworks and for the meta-evolution operator. Cross-model transfer is tested with Kimi K2 and DeepSeek V3.2 (and, in rebuttal only, Claude-3.7-Sonnet).

The design point that matters most for us: **the memory systems evaluated on WebWalkerQA and xBench-DeepSearch were evolved on TaskCraft and then frozen** — "these evolved memory systems are then fixed and evaluated on WebWalkerQA and xBench-DS, i.e., without conducting dataset-specific meta-evolution" (Sec. 5.2). Those two columns are genuine held-out evaluation of an architecture. The TaskCraft column is not.

### b) Key results

1. **Main table (Table 2).** Flash-Searcher with GPT-5-mini goes 71.18 to 74.71 on WebWalkerQA, 69.0 to 74.0 on xBench-DeepSearch, 69.67 to 72.00 on TaskCraft, and 69.09 to 73.33 on GAIA average at pass@1. At pass@3 the GAIA average reaches 80.61, above OWL Workforce (60.61) and Cognitive Kernel-Pro (75.15). On SmolAgent the xBench-DeepSearch pass@1 moves 51.0 to 57.0 and pass@3 to 68.0.
2. **Cost is not the source of the gain (Table 3).** MemEvolve's per-task API cost is $0.085 on GAIA versus $0.086 for no memory, $0.136 versus $0.141 on xBench-DeepSearch, $0.040 versus $0.048 on WebWalkerQA. Latency does rise: 693.33 seconds on GAIA versus 505.46 with no memory, the largest delay in the table.
3. **Human-designed memory is inconsistent (Table 3).** On GAIA, Generative 66.67, DILU 66.67, ExpeL 66.06, Agent Workflow Memory 67.27, Dynamic Cheatsheet 68.48 all fall below the 69.09 no-memory baseline; only Voyager (69.70) and MemEvolve (73.33) exceed it.
4. **Cross-model transfer (Table 2).** The system evolved on TaskCraft with GPT-5-mini lifts Kimi K2 from 52.35 to 69.41 on WebWalkerQA — the headline "17.06%" — and from 58.00 to 68.00 on TaskCraft; DeepSeek V3.2 goes 69.41 to 72.35 and 68.0 to 70.0.
5. **The discovered architectures (Figure 6).** The trajectory from a frozen Agent-KB-style start runs through Adaptive Decision (nine skill granularities), Meta Mem (four storage levels plus a language-model guardrail at retrieval — the round-1 winner), Domain Mem, Riva (domain encoding, database storage, guide-probe-gating retrieval), Omni-Mem, and Cerebra (graph encoding, successful-trajectory-to-semantic store, text/tool bi-routing retrieval, node and edge pruning and consolidation). The direction is monotone: more agentic decisions inside the memory, more abstraction levels, more components. Cerebra is the first architecture in the chain with a real manage component.

### c) Where does it shine?

The most convincing evidence is the held-out transfer, because those numbers are produced by an architecture that was never scored on those benchmarks: "although the underlying memory system is evolved on TaskCraft, it consistently achieves improvements of 3.54% to 5.0% across all three evaluated benchmarks" (Sec. 5.3). Figure 4's cross-framework transfer to Cognitive Kernel-Pro and OWL is the same kind of evidence at the framework level. Figure 5 plots cumulative accuracy against question index and reports that MemEvolve "gradually stabilizes and converges to a consistently superior performance regime" after an early high-variance stretch.

### d) Limitations

Acknowledged: the transfer ceiling across task families (Sec. 5.3); dependence on the underlying framework's capability ("memory is not a panacea", Sec. 5.2).

Implicit, and important for us:

- **The architecture is scored on the batch that produced the diagnosis.** Sec. 4.2 says diagnosis uses "trajectory-level evidence from its own execution batch $T_p^{(k)}$", and Sec. 5.1 says each candidate is evaluated on a 60-task batch of which 20 are reused from the previous iteration. Two thirds of every selection batch is fresh, which is better than pure same-batch scoring, but the diagnosis and the score still share a task pool, and the released code is stricter still (see `repo_analysis.md`: the first selection stage scores candidates on exactly the same 20 tasks the diagnosis read).
- **No earlier task is ever re-checked.** The 20 reused tasks come from the immediately preceding iteration only; nothing from iteration $k-2$ returns. There is no retention measurement anywhere in the paper.
- **Memory content is discarded at every architectural iteration.** The inner loop is initialized "as an empty memory at the beginning of iteration $k$" (Sec. 4.1). This makes candidate comparison fair but means the system never measures what a long-lived memory does to a long-lived architecture.
- **Three generations is too short to show or refute decline.** $K_{\max} = 3$ with $K = 1$ and $S = 3$ gives at most nine architectures ever built. Reviewer IGfMnz2OHM raised exactly this.
- **No size or complexity penalty.** Cost and delay are in the selection vector, but nothing bounds the code size of the memory system, and the evolutionary direction in Figure 6 is uniformly toward more machinery.
- **Single-run numbers in the main tables.** Confidence intervals appear only in the rebuttal, for four systems on three benchmarks.

---

## 5. Reproduction and Application

### a) Open source

Yes: https://github.com/bingreeky/MemEvolve, Apache-2.0. The clone at `repo/` (commit `6035d56`, 2026-05-05) contains EvolveLab with thirteen provider implementations, the MemEvolve orchestrator, and three benchmark runners. Reproduction steps in outline: install the Flash-Searcher dependencies, place a benchmark JSONL under `data/`, set the model environment variables, then drive `MemEvolve/core/auto_evolver.py` through `evolve_cli.py` for the requested number of rounds. Two caveats found in the clone and detailed in `repo_analysis.md`: the TaskCraft runner that the main results depend on is not shipped, and no evolution logs, round summaries, or result files are released — only the two winning architectures, as source.

### b) Implementation details that require attention

The released defaults differ from the paper's reported configuration in ways that change the experiment: the batch is 20 tasks per provider rather than 40 new plus 20 reused; the reuse count is 5; Pareto selection is off by default in favour of a sort on accuracy then token count; and evaluation runs strictly sequentially (`--concurrency 1`). Candidate generation is constrained by a prompt rule that "Complexity must be >= template system (no simple fallbacks or trivial modifications)", so the search has no shrink path. Validation before evaluation is a compile-and-smoke gate over five tasks with up to three automatic repair attempts, not a performance gate. All of these are cited with file and line in `repo_analysis.md`.

### c) Transferability

The design space and the diagnose-and-design operator are not specific to memory. Any harness layer that can be expressed as a small set of substitutable code slots with a stable interface admits the same loop: run, collect traces, have a language model write a per-slot defect profile, generate variants, screen them, keep the best. The pieces that would have to change for a task stream are the scoring protocol (score candidates on tasks the diagnosis has not seen), the state handling (do not wipe accumulated content on every architectural edit), and the accept rule (add a retention check and a size bound).

---

## 6. Summary

### a) One-sentence core idea

Let a language model rewrite the agent's memory system code from diagnosed failures, keeping whichever rewrite wins on the current batch.

### b) Quick-reference pipeline

1. Express the memory system as four replaceable code pieces: how experience is turned into records, where records are kept, how records are recalled, and how records are cleaned up.
2. Run the agent on a batch of tasks with the current memory system, starting from an empty memory, and record success, token cost, and latency for every task.
3. Show a language model the failing trajectories; it writes a per-piece defect report and then three rewritten memory systems.
4. Run every rewrite plus the incumbent on a task batch, rank them on success first and resource use second, and keep the winner.
5. Repeat three times; the final winner is the memory system that ships, and it is then frozen and used on other benchmarks and other backbone models without further change.

---

## 7. Reviewer Reception (OpenReview)

### a) Link and outcome

https://openreview.net/forum?id=qpkG0eKx4v — ICML 2026, Submission3358, **accepted as a poster** (https://icml.cc/virtual/2026/poster/61379). Three official reviews, all public. Overall recommendations **4, 3, 4**; confidences 3, 2, 3. Per-axis: soundness 3/2/2, presentation 4/2/2, significance 3/2/3, originality 3/2/3. The reviewer who scored 4 (IGfMnz2OHM) states in the final justification that the score was revised downward from 5 to 4 after rebuttal, not upward.

### b) Main criticisms

1. **The evolutionary run is too small to support the claim** (IGfMnz2OHM). "The outer loop runs for only 3 iterations with 1 survivor and 3 descendants per iteration, yielding an evolutionary trajectory of at most ~9 candidate architectures total. Whether meaningful architectural adaptation can occur in three generations is unclear and insufficiently justified. Furthermore, each candidate is evaluated on only 60 trajectories (40 new + 20 reused), which introduces high variance." The same reviewer asked directly whether increasing the number of rounds "yield[s] further improvements, or does the process plateau or degrade".
2. **No component-level ablation, and no comparison against a simple search** (OTYdSsSay5 and IGfMnz2OHM). OTYdSsSay5: "the paper shows the full system and qualitative trajectories of evolved memories, but it is harder to tell which components of the proposed meta-evolution pipeline are essential." IGfMnz2OHM: "without such a baseline, it is unclear whether the LLM-driven diagnosis is necessary or whether a systematic architectural evolution is enough."
3. **The backbone assignment across Table 2 is inconsistent**, making comparisons unreliable (IGfMnz2OHM): "Claude is used for Cognitive Kernel-Pro and OAgents, but not for any other frameworks including MemEvolve", and "it is unclear what underlying model is used for Table 3."
4. **Search may be stuck in a local optimum** (OTYdSsSay5): "a very tight survivor budget ($K=1$) and expands to only three descendants ($S=3$) per iteration ... raises questions about whether the framework might get stuck in local optima". OTYdSsSay5 also flags debuggability as a limitation: because the system changes how memory is encoded, retrieved, and managed, "it may become difficult for developers to understand why an agent behaves a certain way."
5. **No statistical significance anywhere** (IGfMnz2OHM). Minor items: missing bold in tables, emoji used as framework names in prose, figure legend occlusion, and three typos in Figure 1.

### c) Author rebuttal and resolution

The authors added four experiments, all reported in the rebuttal threads rather than the camera-ready at the time of review.

- **Random and greedy search baselines, four rounds each.** MemEvolve on TaskCraft: 61.67, 70.00, 75.00, 78.33 (change +16.66). Random architecture composition: 61.67, 63.33, 66.67, **65.00** (+3.33). Greedy per-module substitution, enumerating two alternatives per slot for eight candidates and 480 trajectories per round: 61.67, 65.00, 68.33, 70.00 (+8.33). Held-out WebWalkerQA over the same rounds: MemEvolve 65.10, 67.06, 70.20, 76.27 (+11.17); random 65.10, 64.51, 67.84, **65.88** (+0.78); greedy 65.10, 65.88, 68.24, 70.59 (+5.49). The authors' reading is that "trajectory-grounded diagnosis, not brute-force enumeration, is the key". The number worth noting separately is that **both random-search columns peak at round 2 and fall at round 3**, on the evolution set and on the held-out set alike.
- **Component-level ablation on TaskCraft**, each variant evolving one slot and freezing the other three at the round-0 baseline. Change from round 0 to round 3: retrieve +10.00, encode +6.66, manage +5.00, store +3.33, full system +16.66. The authors conclude retrieve is the single most impactful module and that "no single module matches joint evolution".
- **Claude-3.7-Sonnet backbone**, with the TaskCraft-evolved memory transferred without re-evolution: MemEvolve with Flash-Searcher reaches 71.76 / 69.0 / 69.33 / 69.69 at pass@1 across WebWalkerQA, xBench-DeepSearch, TaskCraft, and GAIA average, above Cognitive Kernel-Pro at pass@1 on all four.
- **Three-run means and standard deviations** for four systems: MemEvolve 73.94 ± 0.86 on GAIA, 76.27 ± 1.47 on WebWalkerQA, 73.67 ± 0.47 on xBench-DeepSearch, against Voyager 69.9 ± 0.76 / 73.14 ± 1.0 / 68.0 ± 1.63, Agent Workflow Memory 67.68 ± 1.51 / 72.16 ± 0.28 / 70.33 ± 0.47, and Mobile-E 68.69 ± 1.03 / 71.76 ± 1.44 / 68.33 ± 1.25. The authors "commit to completing multi-run evaluation for the full Table 3 in the revised manuscript".

What stayed contested: the number of rounds. The authors never ran past three, so the reviewer's plateau-or-degrade question is unanswered. And the search-baseline objection was only partly resolved — IGfMnz2OHM's final justification says "I still have some concerns about the heuristic search comparisons, as random and greedy search are the most basic and naive implementations compared to more advanced algorithms, so I have updated my score to a weak accept (4) rather than an accept (5)."

### d) Net takeaway

The method works and the discovered architectures transfer, but the evidence base is thin by the authors' own account: a nine-architecture search, 60-task fitness batches, single-run main tables, and significance testing that exists only in a rebuttal thread. Read the OpenReview threads alongside the paper — the component ablation and the search baselines there are more informative about where the gains come from than anything in the camera-ready, and the random-search column is the only place in the whole record where a decline over rounds is visible.

---

## Transfer to our own work

**The horizon question, at both loops.** At the inner loop MemEvolve already satisfies score-before-update: Sec. 3.1 records the trajectory and scores it with $R(\tau)$, and only afterwards does the memory absorb it as $M_{t+1} = \Omega(M_t, \epsilon)$. The released runners make this exact (`repo_analysis.md`): the judge runs, then `take_in_memory` is called with the verdict attached. But the update itself is ungated — every task is absorbed regardless of outcome, and no score from any later task ever influences whether an earlier absorption was worth making. At the outer loop the paper is closer to the greedy case than to a held-out protocol. Diagnosis reads "trajectory-level evidence from its own execution batch $T_p^{(k)}$" (Sec. 4.2) and selection scores candidates on a 60-task batch of which 20 are carried over from the previous iteration (Sec. 5.1). The 40 fresh tasks make the outer loop better than same-batch-only, but this is still a one-step-ahead objective: no candidate is ever scored against anything beyond the batch immediately following its own construction, and the horizon is exactly one iteration.

**Retention is never measured.** The 20 carried-over tasks stabilize the comparison between consecutive iterations; they are not an anchor set. Nothing from two iterations back is re-run, and because the inner loop is "initialized as an empty memory at the beginning of iteration $k$" (Sec. 4.1), an architecture that would have degraded on earlier tasks cannot be caught — there are no earlier tasks in memory to degrade on. If we want the strongest version of the claim that no existing self-improving harness measures retention, MemEvolve is a clean instance at the memory layer, and unusually explicit about it: the reset is stated as an equation, not buried in code.

**Peak-then-decline evidence.** The paper's own MemEvolve curve is monotone across its three rounds, so it is not evidence for decline; $K_{\max} = 3$ is simply too short. The usable evidence is in the rebuttal's random-search column, which peaks at round 2 and falls at round 3 on both the evolution set (66.67 to 65.00 on TaskCraft) and the held-out set (67.84 to 65.88 on WebWalkerQA). That is a plasticity-blind search with the same accept rule declining once its own greedy selection has exhausted the easy gains. Reviewer IGfMnz2OHM asked the authors directly whether the process would "plateau or degrade" beyond three rounds and got no answer, which is a gap we can name precisely.

**A revert path exists but is not a gate.** The selection rule keeps the top-$K$ candidate by Pareto rank, and in the released implementation the incumbent is entered into the tournament alongside its descendants, so a round where every descendant loses leaves the incumbent in place. That is a comparative accept, not a gate: there is no threshold a candidate must clear, no check against previously solved tasks, and no bound on what the architecture may grow into. Compared with the per-edit gate in Harness Continual Learning this is weaker; compared with systems that accept unconditionally it is stronger.

**Nothing here can pretrain a critic.** The paper reports no evolution logs, and the repository ships none — no round summaries, no per-candidate scores, no trajectories, only the two winning provider files as source. So MemEvolve contributes architectures, not the realized returns a critic $\hat{\Phi}(h)$ would regress on. The one thing that could be harvested is the structural sequence in Figure 6: seven named architectures with their four-slot descriptions, from a frozen Agent-KB start to Cerebra, which at minimum gives labelled examples of what a language model considers an improvement at this layer.

**EvolveLab as a substrate for our experiments.** This is the most directly usable part. One abstract base class, thirteen provider implementations (twelve published systems plus a template; see Table 1 and the file inventory in `repo_analysis.md`), standardized carriers for trajectories and memory requests, and runners for GAIA, WebWalkerQA, and xBench-DeepSearch that already call judge-then-ingest in the right order. If we want a long-term memory layer with a dozen pre-built variants to edit and a scoring protocol that is already score-before-update, this is a shorter path than building one. Three changes would be needed: restore the manage operation to the interface rather than leaving it to individual providers, stop wiping storage between rounds so memory state can persist across an architectural edit, and replace the batch-relative accept with an anchor-set check plus a size bound. The missing TaskCraft runner and the absent evolution logs mean the paper's headline numbers cannot be reproduced from the release as it stands.

**What we should say about it in positioning.** MemEvolve evolves one harness layer — long-term memory — by rewriting its implementation, scores each architecture on the batch that follows its own construction, carries 20 tasks between consecutive iterations for comparison stability but keeps no anchor set, wipes accumulated memory content at every architectural edit, and stops at three generations. It is the closest existing work to a harness agent at the memory layer, and it is squarely inside the greedy regime our claim describes.
