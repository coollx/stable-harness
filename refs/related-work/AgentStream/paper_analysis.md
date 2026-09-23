# AgentStream: How Well Do Self-Evolving LLM Agents Perform Under Streaming Tasks?

- **Original title**: AgentStream: How Well Do Self-Evolving LLM Agents Perform Under Streaming Tasks?
- **Authors**: Dong Yan, Jian Liang (corresponding), Dapeng Hu (corresponding), Ran He, Nicholas Jing Yuan, Qi Zhang, Tieniu Tan
- **Affiliations**: School of Artificial Intelligence, University of Chinese Academy of Sciences (1); Microsoft (2); Institute of Automation, Chinese Academy of Sciences (3); Nanjing University (4). First author's work done during an internship at Microsoft.
- **Venue / year**: arXiv preprint, 2026 (arXiv:2608.00155v1 [cs.AI, cs.LG], submitted 31 Jul 2026, 43 pages including appendices A-E)
- **Source URL**: https://arxiv.org/abs/2608.00155
- **OpenReview**: no public page found (see section 7)
- **Code**: https://github.com/Jasper-Yan/AgentStream (Apache 2.0), mirrored at https://github.com/microsoft/Sico under `labs/AgentStream` — see `repo_analysis.md`

Short name used below: AgentStream, the paper's own name for its framework.

**Classification: empirical.** The contribution is what was measured — a 3 models × 5 methods × 3 stream shapes × 6 benchmarks grid and the three findings read off it — not an algorithm anyone would adopt. The evaluation framework is the instrument, and no new self-evolving method is proposed.

---

## 1. Research Motivation & Questions

### a) Why was this study conducted?

The gap is a mismatch between how self-evolving agents are sold and how they are measured. Methods that update a prompt, a memory, a skill library, or a whole harness are advertised as producing "increasingly capable agents over time" (Sec. 1), but the evidence for them comes from **independent evaluation**: each task is solved in isolation and the scores are averaged, with no state carried between tasks. Figure 1(a) draws this explicitly — every task starts from an empty state. So the very property being claimed, improvement over a sequence, is the one the standard protocol cannot observe. The authors' framing sentence: "it remains unclear whether the improvements reported for self-evolving agents persist once they are deployed in realistic streaming settings, where tasks may span diverse domains and arrive without clear task boundaries or supervision" (Sec. 1).

### b) What is missing in current understanding?

Three concrete blind spots, each named against a body of prior work:

1. **No cross-task state in the dominant protocol** (Sec. 1, Sec. 2.3). Agentic benchmarks — AppWorld, BFCL, SWE-bench, Humanity's Last Exam, Tau2 and the rest — score each task independently, so nothing about accumulation is testable on them as shipped.
2. **The few streaming studies stream one benchmark under one evolution component** (Sec. 1, Sec. 2.3, citing StreamBench and Evo-Memory). Within-domain streams cannot show whether experience from one domain helps or hurts another, and single-component studies cannot show whether the choice of what to evolve matters.
3. **No controlled comparison across model, method, and stream shape jointly** (Sec. 2.2). Existing self-evolving papers each report their own method on their own benchmark with their own base model, so the reported gains confound all three factors. The authors note that "recent analyses further reveal that self-evolution can degrade or fail to transfer across domains" (Sec. 2.2) without anyone having isolated when.

### c) Research questions

The paper poses no numbered hypotheses. The four section headings in Sec. 5 are its operative questions, and I restate them as the paper words them:

- Q1 (Sec. 5.1): "Does self-evolution help at all, and under which scenario?"
- Q2 (Sec. 5.2): "When does self-evolution help? The role of model capability."
- Q3 (Sec. 5.3): "How does model capability shape the choice of self-evolving method?"
- Q4 (Sec. 5.4): "How do self-evolving methods interact with streaming scenarios?"

The single guiding inquiry underneath them: measured on a stream rather than on isolated tasks, does self-evolution deliver a net gain, and what determines its sign and size?

### d) Type of study

Benchmarking plus ablation-driven characterization. AgentStream builds an evaluation instrument (six existing benchmarks re-organized into one configurable stream, three stream shapes, one scalar gain metric) and then runs a full factorial sweep over models × methods × stream shapes, reading findings off the marginals. It is not a probing or mechanistic study — the internal analyses in Appendix B are descriptive counts of what the evolution state looked like, not interventions.

---

## 2. Study Design

### a) Variables and conditions

**Independent variables (three factors of interest, crossed fully):**

| Factor | Levels | Count |
|---|---|---|
| Foundation model | GPT-5.4-medium, Gemini 3.1 Pro-medium, Claude Opus 4.7-high | 3 |
| Self-evolving method | ACE, A-Mem, ReasoningBank, AutoSkill, Harness | 5 (+ vanilla control) |
| Streaming scenario | Isolated, Sequential, Interleaved | 3 |

**Dependent variables:** per-task benchmark score (the native scoring pipeline of each benchmark, Sec. 4); the stream-level accuracy $\mathrm{Perf}(M, Q, S)$; and the derived **evolution gain** (eq. 1)

$$\Delta = \mathrm{Perf}(M, Q, S) - \mathrm{Perf}(M, Q, \emptyset)$$

where the control $\mathrm{Perf}(M, Q, \emptyset)$ is the same model solving every task with $S_t = \emptyset$ for all $t$. Appendix A adds per-task dollar cost, relative cost overhead, and average agent steps; Appendix B adds evolution-state size counts and edit counts; Appendix D adds cumulative accuracy against task index.

**Held fixed:** the task set (selection is seeded at 42 regardless of run seed, so all conditions see the same 300 tasks); the within-benchmark task order across the three scenarios ("Across all three streaming scenarios, the within-benchmark task exposure order is held constant to ensure comparability", Sec. 4); the embedding model (all-MiniLM-L6-v2); the agent runtime (Exgentic); the judge and user-simulator models (both unified to GPT-5.4); the Sequential benchmark order (AppWorld → BFCL → BrowseComp+ → HLE → SWE → Tau2).

**Nuisance factors that were not controlled:** reasoning-effort setting varies with the model (medium for GPT-5.4 and Gemini 3.1 Pro, high for Claude Opus 4.7, Sec. 4), and stream length varies with the scenario (Isolated runs six separate 50-task streams; Sequential and Interleaved each run one 300-task stream). Both are discussed in section 4c below.

### b) Subjects of study

Three frontier proprietary models; five self-evolving methods plus a no-state control; six agentic benchmarks at $N = 50$ tasks each, giving 300 tasks per stream; three random seeds that reshuffle task order while holding the task set fixed. The grid is 3 models × 5 methods × 3 scenarios = 45 evolving configurations, plus 3 vanilla controls, each at 3 seeds.

Benchmark splits, from Sec. 4: AppWorld `test_challenge`; BFCL `multi_turn_base`; Tau2 `telecom`; BrowseComp-Plus, Humanity's Last Exam, and SWE-bench Verified at their standard settings.

The five methods and what each one evolves (Sec. 4):

| Method | Evolution component | Coupling to execution |
|---|---|---|
| ACE | Agent context — a curated "playbook" of bullets, generated, reflected on, curated | Context-integrated (whole playbook enters the prompt) |
| A-Mem | Structured memory organized Zettelkasten-style with indexing and linking | Retrieval-based |
| ReasoningBank | Reasoning strategies distilled from successful **and** failed trajectories into memory items | Retrieval-based |
| AutoSkill | Reusable skills via extraction, structured representation, refinement, versioned maintenance | Retrieval-based |
| Harness | System prompt + long-term memory + skill library, jointly maintained | Context-integrated (prompt and memory) plus retrieval (top-3 skills) |

The Harness condition is the authors' own implementation, not a reimplementation of a released system: "inspired by [5, 11], we implement a harness evolution method that jointly maintains and updates system prompts, skills, and experience memory through reflection and revision" (Sec. 4). Its two inspirations are Harness Updating Is Not Harness Benefit (arXiv:2605.30621) and Agentic Harness Engineering (arXiv:2604.25850).

### c) Measurement and instruments

**Task score**: the native scoring pipeline of each benchmark — an oracle signal, used only for reporting.

**Evolution signal**: strictly non-oracle. "Notably, no ground-truth labels are accessible at test time, and the evolution relies entirely on feedback intrinsic to the interaction, such as execution outcomes and reflective self-evaluation" (Sec. 3.1). The update is $S_t = \mathrm{Evolve}(S_{t-1}, \tau_t)$ over the trajectory $\tau_t = (q_t, h_t, y_t, r_t)$ where $r_t$ is self-generated feedback. In the Harness implementation the update sees only the task text, the injected skill names, and the step-by-step trace — no score of any kind (verified in the code, see `repo_analysis.md`).

**Scoring protocol**: each task is solved with the state as it stood before that task arrived, and the state updates afterwards — $y_t = M(q_t, h_t \mid S_{t-1})$, then $S_t = \mathrm{Evolve}(S_{t-1}, \tau_t)$ (Sec. 3.1). This is the protocol of this project's framing, arrived at independently: each task is graded with the harness that existed when the task arrived, and the harness is edited only afterward.

**Aggregate metrics**: evolution gain $\Delta$ (eq. 1); positive rate (count of configurations with $\Delta > 0$); Top-1 rate (percentage of configurations where a scenario attains the highest accuracy); cumulative accuracy against task index (Appendix D).

**Metrics that do not exist in this paper**: there is no retention metric, no forgetting metric, no backward-transfer metric, and no re-testing of any earlier task. The words "forgetting", "retention", "plasticity", and "stability" appear only in the related-work paragraphs describing continual learning, never as quantities measured here. Each of the 300 tasks appears exactly once in a stream and is never revisited, so forgetting is unmeasurable by construction, not by oversight.

### d) Experimental conditions and protocol

Setup: pick 50 tasks per benchmark under selection seed 42; order them per the run seed; instantiate the scenario.

- **Isolated** (Sec. 3.2): each benchmark gets its own agent instance and its own evolution state $S_t^{(k)}$; six independent 50-task streams; nothing transfers across benchmarks. Measures within-domain accumulation.
- **Sequential**: one agent, one state, benchmarks in a fixed order, $Q = Q^{(1)} \oplus \cdots \oplus Q^{(K)}$; one 300-task stream with five ordered domain shifts. Measures forward transfer across a curriculum.
- **Interleaved**: one agent, one state, $Q = \mathrm{shuffle}(Q^{(1)} \cup \cdots \cup Q^{(K)})$; one 300-task stream with no domain boundaries. Measures retrieval of relevant experience under maximal diversity.

Manipulation → measurement, per task: solve with $S_{t-1}$ → score with the benchmark's native pipeline → evolve to $S_t$ from the trajectory alone. Repeat for 3 seeds.

The authors state plainly that the three scenarios are not a difficulty ladder: "Rather than being ordered by expected difficulty, the three streaming scenarios are designed to decouple distinct aspects of self-evolution" (Sec. 3.2).

### e) Statistical / analytical approach

No significance tests, no confidence intervals, no correction for multiple comparisons anywhere in the paper. The analysis is entirely descriptive: means over three seeds with a $\pm$ spread in Table 1, counts of configurations above and below baseline in Tables 2 and 3, and marginal averages in Tables 4-6. Sample size is 3 seeds per cell, 50 tasks per benchmark cell, 300 tasks per stream-level number.

Two denominators are in play and they are easy to conflate. Table 2 reports counts out of 45 per scenario — 15 model-method pairs at 3 seeds each, so per-seed granularity. Table 3 reports counts out of 5 methods per model per scenario, computed on seed-averaged numbers. The two are not directly comparable: Isolated is 34/45 in Table 2 (reported as 75.7%) but 12/15, i.e. 80%, when read off Table 3.

---

## 3. Findings & Observations (PRIMARY FOCUS)

### a) Headline findings

**Finding 1 — Self-evolution is not uniformly beneficial; its reliability depends on the stream shape, and Isolated is the most reliable.**

- *Evidence*: Table 2 — Isolated 34/45 positive, average gain $+1.37 \pm 0.80$, Top-1 rate 38%; Sequential 28/45 positive, $+0.75 \pm 0.48$, 29%; Interleaved 28/45 positive, $+0.90 \pm 0.34$, 33%. Across all 135 per-seed configurations, 90 beat vanilla and 44 fall below it (one Sequential cell ties exactly).
- *Conditions*: aggregated over all three models and all five methods. The failure share is large — roughly a third of configurations are worse than doing nothing.

**Finding 2 — The two cross-domain scenarios invert the expected difficulty ordering: Interleaved beats Sequential.**

- *Evidence*: Table 2 gains $+0.90$ (Interleaved) against $+0.75$ (Sequential), Top-1 rate 33% against 29%, and a smaller spread ($\pm 0.34$ against $\pm 0.48$). Pairwise on the 15 seed-averaged model-method configurations of Table 1: Interleaved higher in 10, Sequential in 5; Interleaved leads 4/5 on Gemini 3.1 Pro and 3/5 on Claude Opus 4.7, and still 3/5 on GPT-5.4 where both are net negative.
- *Conditions*: holds across all three models. The authors' reading: "the diversity of interleaved streams compensates for cross-domain noise more effectively than the ordered domain transitions in Sequential" (Sec. 5.1).

**Finding 3 — The benefit of self-evolution is gated by model capability: the weakest model loses under every scenario.**

- *Evidence*: Table 3 — GPT-5.4 gains $-0.35$ (Isolated), $-0.78$ (Sequential), $-0.62$ (Interleaved), with only 4 of 15 configurations above its own vanilla baseline. Gemini 3.1 Pro: $+2.71 / +1.98 / +2.41$, 14 of 15 positive. Claude Opus 4.7: $+1.75 / +1.05 / +0.90$, 13 of 15 positive.
- *Conditions*: consistent across all three seeds (Tables 11-13). Proposed mechanism, explicitly speculative: a bootstrap loop in which enough early successes are needed to seed the experience store with usable patterns (Sec. 5.2).

**Finding 4 — The gain is non-monotonic in model strength, and the non-monotonicity widens with stream complexity.**

- *Evidence*: Gemini 3.1 Pro averages $+2.37$ against Claude Opus 4.7's $+1.23$, despite Claude leading on the vanilla baseline (63.9% against 56.6%, Table 4 and Figure 3). The gap between them is 0.96 under Isolated, 0.93 under Sequential, and 1.51 under Interleaved (Table 3).
- *Conditions*: only two models sit above the capability gate, so the "non-monotonic" claim rests on a two-point comparison plus one below-gate point.

**Finding 5 — Method sensitivity shrinks as the model gets stronger, and the best method does not transfer between models.**

- *Evidence*: Table 4 — best-minus-worst method spread is 5.3 points on GPT-5.4 (48.8 for A-Mem against 43.5 for AutoSkill; only 1 of 5 methods beats vanilla), 2.0 on Gemini 3.1 Pro (all 5 above vanilla), 0.9 on Claude Opus 4.7 (all 5 above vanilla). Per-seed spreads corroborate: 7.0 / 5.6 / 7.7 for GPT-5.4 against 2.3 / 3.3 / 2.2 for Claude Opus 4.7. A-Mem is the best method on GPT-5.4 and the worst on Gemini 3.1 Pro; ACE leads on Gemini and ranks third on the other two; only ReasoningBank stays competitive on all three.
- *Consequence the authors draw*: averaged over methods, self-evolution **widens** the gap between the weakest and strongest model from 18.1 to 20.0 points; pairing each model with its own best method narrows it to 16.8 (Sec. 5.3).

**Finding 6 — Context-integrated methods peak under Isolated, retrieval-based methods peak under Interleaved.**

- *Evidence*: Table 5 — ACE $+2.28 / +2.26 / -1.26$ (Isolated / Sequential / Interleaved), Harness $+1.91 / -0.86 / +1.01$, against A-Mem $+1.93 / +1.71 / +2.22$, ReasoningBank $+0.78 / +0.46 / +1.79$, AutoSkill $-0.07 / +0.19 / +0.72$. The Isolated-higher / Interleaved-higher counts out of 9 split the families cleanly: ACE 7-2 and Harness 5-3 against ReasoningBank 2-7 and AutoSkill 2-7.
- *Mechanism offered*: tight coupling of stored experience to the prompt "converts experience into precise domain-specific strategies" within one domain but is "susceptible to cross-domain interference" across domains, while a retrieval gate "suppresses cross-domain interference ... by activating only task-relevant experience" (Sec. 5.4).

**Finding 7 (Appendix A) — Self-evolution does not necessarily cost more, but the cost-performance trade-off is set by the model.**

- *Evidence*: on Gemini 3.1 Pro four of five methods run **below** the vanilla per-task cost of $4.035 — ReasoningBank at 64%, Harness at 71%, AutoSkill at 82%, A-Mem at 84% — while also cutting agent steps. On GPT-5.4 (vanilla $0.297 per task) the pattern reverses: A-Mem reaches 577% of baseline cost, ACE 266%, and ReasoningBank alone reduces cost, to 92%. On GPT-5.4 only A-Mem is net positive (+3.2 accuracy at 5.77× cost); every other method is negative regardless of spend.
- *Caveat stated by the authors*: Figure 4 and Tables 7-8 are "based on a single evaluation", i.e. one seed. Claude Opus 4.7 cost is not reported at all.

**Finding 8 (Appendix B) — The three models accumulate very differently shaped states and edit them at very different rates.**

- *Evidence*: over the 300-task Interleaved stream (Table 9), Claude Opus 4.7 builds the largest states — 1251 ACE playbook bullets and 463 Harness skills — while Gemini 3.1 Pro builds the most compact (209 bullets, 117 AutoSkill skills) but concentrates 29,172 characters in the Harness system prompt and memory against Claude's 9,362 and GPT-5.4's 8,711. Table 10: revision rate 65% (GPT-5.4), 69% (Gemini), 96% (Claude); skills added 181 / 147 / 492; skills edited 5 / 66 / 460; memory-or-prompt edits 21 / 267 / 103.
- *Reading*: GPT-5.4 is "largely append-only, adding skills but rarely editing them ... and no modification to the system prompt"; Claude "performs the most intensive refinement, revising on 96% of tasks while concentrating almost entirely on the skill library" (Sec. B.2).

### b) Patterns and trends

**The effect sizes are small and the variances are not.** Every headline gain sits between $-0.8$ and $+2.7$ percentage points, while individual Table 1 cells carry standard deviations across three seeds of $\pm 36.2$ (A-Mem Isolated on Tau2 with GPT-5.4), $\pm 28.0$ (Harness Sequential on Tau2 with Claude Opus 4.7), $\pm 25.0$ (A-Mem Isolated on BFCL with Gemini), and $\pm 22.3$ (ACE Sequential on Tau2 with GPT-5.4). The aggregate claims survive only because they average 45 cells; no single cell is individually informative.

**Method spread contracts monotonically with model strength** (5.3 → 2.0 → 0.9 across GPT-5.4 → Gemini → Claude, Table 4), and the paper ties this directly to the capability gate: on a strong model every method works a little, so the choice stops mattering.

**The coupling axis is the only method property that predicts scenario preference.** The paper collapses five architecturally distinct methods onto one binary — is stored experience folded into the prompt, or retrieved per task — and that binary sorts Table 5 cleanly. ACE's Interleaved collapse is the sharpest instance: $+2.28$ under Isolated to $-1.26$ under Interleaved, a 3.5-point swing.

**Most falls in cumulative accuracy cannot be attributed to the evolving state; two panels can** (Appendix D, Figures 5-19). Four properties of the plots decide what they can show. Each curve is the mean of the three seeds: its endpoint matches the three-seed mean in Table 1 (checked for Figures 7 and 19). The horizontal axis, labeled "Task index in stream", restarts at 1 for each benchmark in all three scenarios, so a panel shows one benchmark's 50 tasks, not the 300-task stream. Within one seed, the order of a benchmark's tasks is identical across the three scenarios and across the five methods (`task_ordering.py`, see `repo_analysis.md`), so a shape that all three scenarios share can come from task order, or from the few tasks behind the early points, rather than from the evolving state. No vanilla curve is plotted. The paper reports the figures without comment: Appendix D is one sentence long.

Read with those properties, most falls are shared across scenarios and end near vanilla. In Figure 7 (ACE on Claude Opus 4.7), BFCL falls from 1.0 to between 0.81 and 0.85 under all three scenarios, against a vanilla score of 0.86. In Figure 5 (ACE on GPT-5.4), the Humanity's Last Exam panel peaks near 0.125 around task 13 and ends between 0.05 and 0.07; the peak rests on about five solved tasks across the three seeds. Three methods on Claude Opus 4.7 bend at the same Tau2 position, near task 23 (ACE and Harness under Isolated, A-Mem under Sequential), which points to a hard task at that position in the shared orders.

Two panels separate the scenarios on identical tasks, so there the difference comes from the state. In Figure 7, ACE on Tau2 under Interleaved scores about 0.89 over the first 9 tasks and about 0.68 over tasks 10-50, against about 1.00 and 0.92 for Isolated on the same tasks and 0.80 for vanilla: above vanilla early, below it later, and all three Interleaved seeds end below vanilla (78, 76, 62; Tables 11-13). In Figure 19, Harness on Tau2 under Sequential, whose state has absorbed 250 tasks from the other five benchmarks, falls to 0.74 while Isolated ends at 0.91; the per-seed scores are 94, 42 and 86, so one run collapsed and the average hides it. Such single-run collapses are rare and not specific to long streams: in Tables 11-13, 9 of the 270 model-method-scenario-benchmark cells have one seed at least 20 points below both others, three in each scenario, and 8 of the 9 are on BFCL or Tau2. All curve values in this pattern are read by eye from the figures; the window averages are derived from those readings.

### c) Surprises and counterintuitive results

1. **Interleaved beats Sequential.** The authors flag this as contradicting their own stated expectation: "A natural hypothesis is that Interleaved ... should pose the greatest challenge due to maximal cross-domain interference. However, the empirical results contradict this expectation" (Sec. 5.1).
2. **The mid-capability model gains more than the strongest one**, and the advantage grows with stream mixing (Finding 4).
3. **Self-evolution can be cheaper than not evolving.** Four of five methods on Gemini 3.1 Pro cut both cost and step count, which inverts the usual assumption that accumulated context is pure overhead.
4. **Self-evolution widens the inter-model gap** rather than levelling it, because it helps the strong and hurts the weak (Sec. 5.3).

### d) Negative and null results

- GPT-5.4 is net negative under all three scenarios; four of its five methods are below its own vanilla baseline (Table 4). This is the paper's largest single block of negative evidence.
- ACE's Interleaved gain is $-1.26$ and AutoSkill's Isolated gain is $-0.07$ (Table 5) — i.e. two of the five methods have a stream shape under which they are net harmful on average.
- **Harness is the only method with a negative Sequential gain ($-0.86$, Table 5).** The method that edits the most layers is the one that loses most under ordered domain shift.
- No method wins everywhere; the paper's third headline is a null result about method dominance.

### e) Robustness checks

Three seeds, reported per-seed in full (Tables 11-13, Appendix C), with the task set held constant so only arrival order varies. Within-benchmark order is held constant across the three scenarios so scenario comparisons are not confounded by ordering. That is the extent of it: one agent runtime, one embedding model, one judge model, one task count, no prompt-wording variations, no model-family replication within a capability tier, no re-running of the vanilla baseline across seeds.

---

## 4. Analysis & Interpretation

### a) Authors' explanations

**Demonstrated by the design**: the interaction between method family and stream shape (Finding 6) — the crossing is fully factorial, the same tasks in the same within-benchmark order appear in every scenario, so the scenario effect is a controlled contrast.

**Suggested, not demonstrated**: the bootstrap-loop account of capability gating (Sec. 5.2) — "when the base model's solve rate is low, the experience stream is dominated by failed or partially correct trajectories, from which the agent cannot reliably extract transferable knowledge". Nothing in the paper inspects the quality of stored entries as a function of the solve rate, so this is a plausible story fitted to a correlation between three points.

**Speculated**: the "virtuous cycle" language in Sec. 5.2, the claim that Interleaved diversity "drives the consolidation of transferable patterns" (Sec. 5.4), and the distributional-coherence explanation for Isolated's advantage (Sec. 5.1). None is measured.

### b) Evidence quality

The scenario contrast is clean, because scenario is manipulated with everything else fixed. The method contrast is clean within a model. The **capability** contrast is the weakest of the three and carries the paper's most quoted finding: model identity confounds family, training recipe, price, and — unflagged — reasoning-effort setting, since Claude Opus 4.7 was run at high effort while GPT-5.4 and Gemini 3.1 Pro were run at medium (Sec. 4). "Capability" is therefore a label attached to observed vanilla accuracy, which the authors themselves concede in Sec. 7.

Statistical support is thin in an absolute sense. With 300 tasks per stream, the binomial standard error on a stream-level accuracy near 50% is roughly 2.9 percentage points, which is larger than every headline gain the paper reports. The claims are defensible only as aggregates over 45 configurations, and the paper never says so.

### c) Confounds and alternative explanations

**Stream length is confounded with scenario.** Isolated runs six independent 50-task streams; Sequential and Interleaved each run one 300-task stream. So Isolated's "most reliable" status is equally consistent with a state-size explanation — a state built from 50 tasks is small, cheap to retrieve from, and hard to poison, while a state built from 300 tasks is large and interference-prone. The paper's own Table 9 shows exactly how large those 300-task states become (1251 ACE bullets, 463 Harness skills for Claude Opus 4.7), yet Sec. 5.1 attributes Isolated's advantage solely to "distributional coherence" and never tests the length account — for instance by truncating the Sequential or Interleaved stream to its first 50 tasks, which the recorded per-task metrics would make trivial.

**Judge/agent family overlap.** All judges and user simulators are GPT-5.4 (Sec. 4). This is uniform across conditions, so it does not bias scenario or method comparisons, but it does put the GPT-5.4 agent in the position of being graded and simulated by its own family, which is a live concern for the capability-gating claim specifically.

**Reasoning effort**, as above, tracks the model factor exactly.

**The vanilla baseline is a single run.** Table 1's vanilla rows carry no $\pm$, while every evolving row averages three seeds. Since evolution gain is defined against that single number, a baseline that happened to land high or low shifts every gain for that model-benchmark pair. On HLE with GPT-5.4 the vanilla score is 2.0% — one task out of 50 — so the reported HLE gains of $+3$ to $+12.7$ points correspond to two to seven additional tasks measured against a one-task reference.

### d) Generalizability

Credible: within frontier proprietary models of roughly this class, agentic benchmarks of roughly this kind, streams of a few hundred tasks, and self-evolving methods that write natural-language state read back at inference time. The scenario-by-method-family interaction is the most portable finding, because it follows from a structural property (coupling) rather than from any model's idiosyncrasy.

Risky: any extrapolation to open-weight or smaller models (untested — "capability" is defined by these three points); to longer streams (300 tasks is short for a continual-learning claim, and the cumulative-accuracy curves suggest the dynamics have not settled); to weight-updating methods (excluded by construction); and to any claim about which *specific* method to use, since the paper's own Finding 5 is that method ranking does not transfer across models.

---

## 5. Implications, Limitations & Transferability

### a) For practitioners

Four actionable rules, three of them the paper's own (Sec. 1, Sec. 6): apply self-evolution only above a capability threshold, since below it the machinery costs money and accuracy both; prefer context-integrated methods when the deployment stream stays within one domain and retrieval-based methods when it spans domains; select the method per model rather than adopting a leaderboard winner; and — added from Appendix A — measure cost per task, because the same method swings from 0.57× to 6.37× of baseline depending on the model.

### b) For researchers

The instrument is the contribution. Reorganizing existing benchmarks into a configurable stream with three named shapes, plus a one-line gain metric against a stateless control, is a protocol other work can adopt cheaply. The open questions it exposes: what governs the capability threshold; whether the coupling binary survives more methods; and — unasked by the paper — what happens past 300 tasks, given that at least one Appendix D panel shows the evolving state pushing accuracy below vanilla late in the stream (section 3b).

### c) Acknowledged limitations (Sec. 7)

Two, stated plainly. First, "the notion of model capability strength used throughout this work is grounded in the empirical observations under our specific experimental setup rather than a universally valid ranking of the models", since a model performs unevenly across benchmarks and everything runs inside Exgentic. Second, three scenarios and six benchmarks "cover representative but not exhaustive stream compositions and task domains"; broader coverage is left to future work.

### d) Unacknowledged limitations

1. **No retention measurement.** Each task is seen once and never revisited, so the paper cannot say whether a stream that improves on average is quietly losing earlier capability. This is the single largest gap between what "streaming evaluation" promises and what is delivered.
2. **The stream-length / scenario confound** (section 4c).
3. **Reasoning-effort mismatch across models** (section 4c).
4. **No significance testing**, against gains smaller than the sampling noise of a single configuration.
5. **Single-run vanilla baselines** anchoring every reported gain.
6. **Claude Opus 4.7 cost is never reported**, so the cost analysis covers two of three models and the cost-performance conclusion generalizes from them.
7. **Appendix D is presented without analysis.** Figures 5-19 are five pages of dynamics plots introduced by a single sentence, with no vanilla curve and no per-seed spread, so in most panels a reader cannot tell a state effect from task order; the two panels where scenarios diverge on identical tasks (section 3b) go unmentioned in the results sections.
8. **The Harness condition is the authors' own construction**, so its ranking is evidence about this implementation rather than about published harness-evolving systems.

### e) Reproducibility

Strong on code, absent on data. The repository is public under Apache 2.0 at both `Jasper-Yan/AgentStream` and `microsoft/Sico/labs/AgentStream`, bundles the Exgentic runtime and all five method implementations, and provides per-method runner scripts with the exact mode / seed / model / benchmark switches. Appendix E prints the full prompt set for all five methods. What is missing: no released run outputs, no per-task metric files, no evolved states — the repository ships zero result artifacts, so every number in the paper would have to be regenerated by re-running the grid against three proprietary APIs. At the reported per-task costs, one full seed of the Gemini grid alone is on the order of $4 × 300 × 15 configurations.

### f) Transfer to our own work

- The **evolution gain** $\Delta$ is, read literally, an undiscounted integral of stream performance, with each task graded by the harness that existed when it arrived, against a no-adaptation control. That is precisely the quantity this project's claim is about, already defined, already instrumented, already measured for five systems on three models.
- The **three stream shapes** are a ready-made knob for heterogeneity and non-stationarity, with a decisive empirical result attached: Sequential (ordered domain shift) is where deep-editing methods lose, so it is the discriminating regime for a plasticity claim, not Interleaved.
- **Cumulative accuracy against task index** becomes a decline readout only when compared with a reference on the same tasks, such as a no-state run or another scenario, because task order and small early samples shape a single curve; the code already writes the per-task scores needed (`cumulative_avg_score` in `online_metrics.jsonl`), alongside step count, cost, state character count, and skill count — a harness-size series that supports the equal-harness-size control our falsifier needs.
- **What must be added**: an anchor set and a retention metric. AgentStream supplies the stream, the protocol, the baselines, and the gain metric; it supplies nothing about what the harness has lost, and its own design forecloses measuring it.

---

## 6. Summary

### a) One-sentence headline finding

Across 45 configurations, self-evolution beats doing nothing by only 1-2 points on average and fails outright in about a third of cases, with the sign set by model capability and the best method set by stream shape.

### b) Quick-reference takeaways

- Self-evolution underperforms a stateless baseline in 44 of 135 per-seed configurations; average gains are $+1.37$ (Isolated), $+0.90$ (Interleaved), $+0.75$ (Sequential) percentage points.
- The weakest of three frontier models loses under every stream shape ($-0.35$ to $-0.78$), while the other two gain $+0.90$ to $+2.71$; the gain is not monotonic in model strength.
- A mixed, shuffled cross-domain stream is easier for self-evolving agents than an ordered curriculum with domain shifts — the reverse of the natural expectation.
- Methods that fold experience into the prompt peak on single-domain streams; methods that retrieve per task peak on mixed streams; no method wins across models, and the best method for one model is often among the worst for another.
- Over a 300-task stream the accumulated state grows without bound (up to 1251 playbook bullets or 463 skills) and is revised on up to 96% of tasks; nothing in the study checks whether earlier capability survives that churn.

### c) Bottom line for decision-making

Trust this for what it measures — the sign and rough size of stream-integrated gains across models, methods, and stream shapes — and not for claims about durability, because no task is ever re-tested and no retention quantity exists in the paper.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched OpenReview's note-search API for "AgentStream" and for the full title, plus a web search for the title with "openreview", on 2026-09-14). The paper is an arXiv preprint with no venue named on the abstract page or in the PDF, and no reviews are available. No review content is reported here.
