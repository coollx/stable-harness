# Adaptive Auto-Harness: Sustained Self-Improvement for Agentic System Deployment on Open-Ended Task Streams

- **Original title**: Adaptive Auto-Harness: Sustained Self-Improvement for Agentic System Deployment on Open-Ended Task Streams
- **Authors**: Zewen Liu, Zhan Shi, Yisi Sang, Bing He, Minhua Lin, Tianxin Wei, Dakuo Wang, Benoit Dumoulin, Wei Jin, Hanqing Lu
- **Affiliations**: Emory University (1); Amazon (2); The Pennsylvania State University (3); UIUC (4); Northeastern University (5)
- **Venue / year**: arXiv preprint, 2026 (arXiv:2606.01770v2 [cs.LG], v1 submitted 1 Jun 2026, v2 3 Jun 2026; 18 pages including appendices A–I)
- **Source URL**: https://arxiv.org/abs/2606.01770
- **OpenReview**: no public page found (see section 7)
- **Code**: https://github.com/A-EVO-Lab/a-evolve, branch `release/adaptive-auto-harness` — cloned and analyzed in `repo_analysis.md`

Short name used below: Adaptive Auto-Harness, the paper's own name for its system. Where this document says "harness", "layer", "task stream", or "score-before-update" it uses those terms as `docs/framing.md` defines them; the paper's own vocabulary (evolver, solver, evolution loss, adaptation loss, harness tree, agentic routing) is kept in italics on first use and attributed to the paper.

Classified as a **method paper**: the dominant contribution is a built system (a four-phase multi-agent *evolver*, a git-backed *harness tree*, a solve-time router, and human-steering hooks) that others could reproduce, with an analytical decomposition supplied to motivate its two components. The stream measurements in Figure 1 and Appendix C are diagnostic support for the system, not the contribution itself.

---

## 1. Method Motivation

### a) Why was this method proposed?

Existing systems that automatically rewrite a harness from execution feedback — the paper names A-Evolve, GEPA, Meta-Harness, Continual Harness, and SkillOS — are measured on fixed offline benchmarks such as SWE-bench, where there is a train/test cutoff, one task type, and a stationary distribution. The paper's driving observation is that real deployment is none of those things, and that when you run the same systems on a chronological stream their accuracy peaks early and then falls. Figure 1 is the paper's opening evidence: an unbounded A-Evolve run on prediction-market tasks grows its skill count from 12 to 34 and its system prompt from 2 KB to 68 KB over 51 evolution cycles, and the pass-rate lift over a no-evolution solver rises, peaks, and then decays for every stopping budget tried (3, 7, 15, 30, 51 cycles). The paper's summary sentence is that "all stopping budgets eventually peak and decline; later in the stream, shorter runs outperform longer ones" (Sec. 1).

The concrete failure mode is illustrated with a single artifact: an evolved skill file `news_from_future.md` that was right 138 times and wrong 16 times on sports markets, and that misfires on a politics market by reading a United States government shutdown as confirming evidence (Fig. 1 caption). One batch's useful patch is another batch's wrong prior.

### b) What are the pain points of existing methods?

The paper organizes them as three deployment dimensions that a static benchmark cannot expose (Sec. 1, Fig. 2):

1. **Unbounded streams.** There is no fixed endpoint; feedback, trajectories, and harness state accumulate forever, and a single-agent *evolver* must compress that growing history into one finite context window.
2. **Task heterogeneity.** A prediction-market platform mixes politics, sports, and finance in the same hour, each wanting different sources, tools, and prompting, but existing systems deploy one static dense harness across the whole stream with no per-task adaptation.
3. **Distributional non-stationarity.** Incoming tasks drift away from the experience the harness was last fitted on, so a harness optimized for recent cycles drifts out of fit even when its history is rich.

### c) Core hypothesis

The gap between a deployed self-editing harness and the best harness available at each moment splits into two independent, separately addressable deficits — what the *evolver* cannot build from history, and what a single committed harness cannot fit to the task in front of it — and attacking them with separate machinery (a stateful multi-agent builder, and a branch tree with solve-time routing) sustains stream performance where one densely-updated harness peaks and declines.

## 2. Method Design (primary focus)

### a) Pipeline overview

The system runs one loop per batch of tasks arriving in chronological order. Batch sizes are 100 (PolyBench), 20 (CTF-Dojo), and 20 (FutureX), producing 51, 14, and 26 evolution cycles respectively (Table 3).

**Step 1 — Route (solve time).** For each incoming task $x_t$ in the batch, a router agent reads the metadata of every branch of the *harness tree* (via `git show` against each branch's workspace) plus the task's own context, and returns a branch name. The router sees task context and branch metadata only: never labels, never branch outcomes (Appendix B). Tasks sharing a branch are grouped so the branch workspace is materialized once per group.

**Step 2 — Solve.** The chosen branch is checked out into the solver workspace, giving the solver that branch's system prompt, skills, tool registry, and memory. The solver (Claude Sonnet 4.6, temperature 0, 80-turn cap) produces the answer. The harness the task is scored with is the one that existed before the task arrived — score-before-update, implemented as a per-task release-time filter on what the solver may see (Appendix A).

**Step 3 — Reveal.** The *temporal-reveal gate* hands each task's evaluation signal to the *evolver* only after that task's real-world resolution date. A prediction market that resolves days after the trade is an unlabeled trajectory until then; CTF-Dojo tasks have no resolution date, so their labels stay hidden from the *evolver* entirely (config `temporal_reveal`, Appendix B).

**Step 4 — Evolve (four phases, one cycle per batch).** The *evolver* (Claude Opus 4.6, temperature 0, 128k token budget) runs four role-specialized agents, each with its own full context window and its own objective:

- **Analyze** — read the batch trajectories and the revealed outcomes, and update a persistent `task_board` with failure regimes and priorities.
- **Research** — dispatch parallel research agents (3 in all reported runs) on disjoint hypotheses drawn from the board; each writes a pass/fail verdict into a persistent research log.
- **Build** — construct the harness change (prompt edits, skill files, tool implementations, infrastructure pipelines) from verified research only.
- **Verify** — run tests against the built artifacts; on failure, roll back and retry, up to 3 times.

**Step 5 — Commit and branch.** The result is committed to the git-backed workspace. When failure evidence warrants specialization, the *evolver* creates a regime-specific branch (`branch/crypto-classical`, `branch/binary-reversing`, `branch/sports`, …) carrying its own prompt, skills, and tool registry, so specialization does not cross-contaminate the other regimes. Git supplies versioning, isolation, and lineage.

**Step 6 — Human hooks (optional, off in the main comparison).** Two structurally triggered hooks fire only when history cannot contain the needed signal: a research-phase hook when a research agent hits an authentication wall and needs a credential, and a task-board hook when the analyst's board has gaps a human can fill with source direction. In the reported human-in-the-loop run they fired twice for credentials at cycle 1 and once substantively for board steering at cycle 3, with the other four board prompts answered `skip` (Table 13).

### b) Architecture components

| Component | Function | Where it lives |
|---|---|---|
| Solver | Answers one task under the harness it is handed; never edits the harness | `agents/<benchmark>/solver.py` |
| Four-role *evolver* | Analyst → parallel Researchers → Builder → Verifier; writes harness edits | `algorithms/navigation/templates/structured_*.py` |
| Evolver workspace | Cross-cycle state: task board, research logs, architecture notes, verification tests; separate from the solver workspace | `_evolution_workspace.py` |
| Harness tree | Git branches, one per regime, each a complete harness; created by the *evolver*, never merged or pruned | `engine/versioning.py` |
| Router | Reads branch metadata at solve time and picks a branch for the incoming task | `algorithms/navigation/engine.py::navigate` |
| Temporal-reveal gate | Withholds each task's label from the *evolver* until its resolution date | `config.temporal_reveal`, `engine/observer.py` |
| Human interface | Terminal or Telegram channel exposed to the *evolver* as a tool | `engine/human_interface.py` |

### c) Formulas and algorithms

The analytical framework (Sec. 3.1–3.2) is a regret decomposition against an oracle harness. Tasks arrive as $x_1, x_2, \ldots, x_T$ with $x_t \sim P_t$; the agent holds history $\mathcal{H}_t = \{(x_i, r_i, \tau_i)\}_{i=1}^{t-1}$ of past tasks, optionally-realized rewards, and trajectories. An *evolver* $\varphi$ maps history to a harness $C = \varphi(\mathcal{H}_t)$ with capacity $|C| \le K$; a solver acts under $a_t \sim \pi(a \mid x_t, C_t)$.

Utility of a harness on a task (eq. 1):

$$V(C, x_t) = \mathbb{E}_{a_t \sim \pi(\cdot \mid x_t, C)}\big[r(a_t, y(x_t))\big].$$

The full-history ceiling is the best bounded harness constructible on $x_t$, $V(\mathcal{H}_t, x_t) := \sup_{C : |C| \le K} V(C, x_t)$, giving a non-negative regret (eq. 2):

$$\mathrm{Regret}(\varphi, x_t) = V(\mathcal{H}_t, x_t) - V(\varphi(\mathcal{H}_t), x_t) \ge 0.$$

The *solve-time optimal harness* is the best harness the *evolver* class $\Phi$ could produce **if it were allowed to see the incoming task** (eq. 3):

$$C^{*}_{\Phi}(x_t) = \arg\max_{C = \varphi(\mathcal{H}_t, x_t),\ \varphi \in \Phi,\ |C| \le K} V(C, x_t).$$

Pivoting the regret through that reference gives Proposition 1 (eqs. 4–6):

$$\mathbb{E}_{x_t}\big[\mathrm{Regret}(\varphi, x_t)\big] = L_{\mathrm{evo}}(\Phi) + L_{\mathrm{adapt}}(\varphi),$$
$$L_{\mathrm{evo}}(\Phi) = \mathbb{E}_{x_t}\big[V(\mathcal{H}_t, x_t) - V(C^{*}_{\Phi}(x_t), x_t)\big], \qquad L_{\mathrm{adapt}}(\varphi) = \mathbb{E}_{x_t}\big[V(C^{*}_{\Phi}(x_t), x_t) - V(\varphi(\mathcal{H}_t), x_t)\big].$$

In words: $L_{\mathrm{evo}}$ is the *evolution loss*, the structural ceiling of the *evolver* class — a single-agent prompt editor simply cannot produce multi-file infrastructure, and running it longer does not fix that. $L_{\mathrm{adapt}}$ is the *adaptation loss*, the cost of committing to one harness before seeing the task; it survives even a perfect *evolver* as long as tasks are heterogeneous. The multi-agent evolver targets the first, the tree plus router targets the second.

Two properties of this decomposition matter for reading the paper. First, both reference quantities are per-task: the oracle is defined at time $t$ against $x_t$ alone, so nothing in the objective represents what an edit does to later tasks — the decomposition has no future-stream term. Second, the paper is explicit in its Limitations (Sec. 6) that $L_{\mathrm{evo}}$ and $L_{\mathrm{adapt}}$ are "analytical quantities, not directly estimated oracle losses"; they are diagnosed through bottleneck analyses, ablations, and routing controls rather than by any estimator.

## 3. Comparison with Existing Methods

### a) Fundamental differences

Prior auto-harness systems keep one harness and rewrite it in place, cycle after cycle. Adaptive Auto-Harness keeps a *set* of harnesses in git branches and decides at solve time which one to run, and it replaces the single-context one-shot *evolver* with four role-specialized agents that share persistent cross-cycle state. Conceptually the shift is from "improve the harness" to "organize a harness space at build time and select from it cheaply per task."

Relative to `docs/framing.md`, the important structural fact is that the shift is architectural, not objective-level. Every edit is still proposed from the batch just solved and judged by whether the built artifact passes the Verifier's tests; there is no term for performance on tasks that have not arrived, no discount on future batches, and no re-check against earlier tasks. The system is a more capable builder and a smarter selector operating under the same current-batch objective — the greedy case in our framing — which is why its own runs still peak and decline (see 4d).

### b) Key innovations and significance

| Innovation | What it is | Significance |
|---|---|---|
| Evolution/adaptation regret split | Proposition 1 decomposes the gap to an oracle into an *evolver*-capability term and a commit-before-seeing-the-task term | Notable — it is a clean vocabulary for design axes, but it is analytical only and never estimated |
| Four-role stateful *evolver* | Analyze → Research → Build → Verify, each with its own context, plus a persistent task board, research log, and architecture notes | Significant — the largest single ablation effect (Fig. 6) |
| Harness tree + solve-time routing | Regime-specific git branches built at evolution time, selected per task by a router agent | Significant — this is the paper's distinctive mechanism, and the only one addressing heterogeneity |
| Temporal-reveal gate | Labels reach the *evolver* only after the task's real-world resolution date | Notable — a concrete, reusable implementation of score-before-update on a genuinely delayed-label stream |
| Structurally triggered human hooks | Credential and task-board hooks that fire only when history lacks the signal | Incremental — useful engineering, narrow effect (Fig. 9) |

### c) Applicability

The method excels where the stream has recurring, separable regimes with enough per-regime volume to justify a branch, and where the binding constraint is building infrastructure the *evolver* class could not previously produce (CTF-Dojo payload handling, PolyBench calibration). It struggles where the binding constraint is external access rather than branch choice: on FutureX the router underperforms the single `main` workspace (34.5 vs 39.7 Pass@1 on the replay subset, Table 11), because what the tasks need is source acquisition, not a specialized branch.

### d) Comparison table

| System | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| Adaptive Auto-Harness (this paper) | Leads all three streams; branch isolation prevents cross-regime contamination; delayed labels handled honestly | Still scores every edit on the batch that produced it; tree only grows (no merge, no prune in the released code); ~2.3× solver tokens and ~2.3× wall-clock of the no-evolution baseline on PolyBench (Table 5) | A term for future-stream performance; branch merge and retirement; an estimator for the losses it defines analytically |
| A-Evolve (single-agent linear chain) | Simple; strongest of the baselines on the two pass-rate streams (45.2 CTF-Dojo, 47.5 FutureX) | Peaks and declines; prompt grows 2 KB → 68 KB; **falls below the no-evolution baseline on PolyBench accuracy (18.4 vs 22.2)** | Bounded harness growth; per-regime isolation |
| GEPA (reflective Pareto prompt evolution) | Textual-feedback signal, no scalar reward needed | Weakest or near-weakest on two of three streams (13.4 PolyBench, 28.2 FutureX) | Beyond-prompt layers |
| Meta-Harness (filesystem archive + Claude Code proposer) | Strong on PolyBench (50.8 accuracy, +320 return) | **Below the no-evolution baseline on FutureX (29.4 vs 31.0)** | Cross-stream generality |
| Continual Harness (online adaptation in one run) | Continuous, no batch boundary | Weakest overall (8.5 PolyBench accuracy, 25.7 CTF-Dojo) | Structured verification before commit |
| SkillOS (learned skill curation) | Curation policy trained with reinforcement learning | Mid-pack everywhere (21.4 / 29.5 / 29.8) | Layers beyond skills |
| OctoTools (frozen human-designed) | No evolution cost; third on PolyBench return | Cannot adapt at all; last on FutureX (25.6) | — |

## 4. Experimental Validation

### a) Experimental design

Three chronological streams (Table 1): **PolyBench**, 5,075 Polymarket prediction-market tasks spanning Feb 6–22 2026 across politics, sports, finance, crypto, and entertainment; **CTF-Dojo**, 261 security challenges from `pwncollege/ctf-archive` ordered 2011–2024; **FutureX**, 503 event-forecasting questions over 82 days (Jan–Apr 2026) in English and Chinese. All three enforce strict temporal order.

Baselines are five no-evolution solvers (Sonnet 4.6, Haiku 4.5, DeepSeek-V3.2, GLM-4.7, Kimi-K2.5), five auto-harness systems (A-Evolve, GEPA, Meta-Harness, Continual Harness, SkillOS), and one frozen human-designed system (OctoTools). All algorithms share the same batch size, batch loop, and temporal-reveal gate; only the evolution algorithm varies. Solver is Sonnet 4.6, *evolver* is Opus 4.6, router is Sonnet 4.6, all at temperature 0, so the reported metrics are point estimates rather than averages over re-samples (Appendix B).

Crucially for reproducibility of the reported gains, the seed harness is nearly empty: 27, 24, and 114 prompt lines for PolyBench, CTF-Dojo, and FutureX, with zero seed skills, zero tools, and zero memory entries; only FutureX seeds an infrastructure directory (Table 4). Everything beyond the seed prompt is built by the *evolver*.

Metrics: PolyBench reports Accuracy (fraction of traded markets called correctly) and Return = Coverage × CWR, where CWR is a confidence-weighted profit-to-investment ratio and Coverage is the fraction of markets traded; CTF-Dojo and FutureX report the official Pass@1.

### b) Key results

1. **The three variants jointly lead every metric** (Table 2). Full System: 80.9% PolyBench Accuracy (vs 22.2% for the no-evolution Sonnet baseline and 18.4% for A-Evolve), 97.9% PolyBench Coverage, and 50.2% CTF-Dojo Pass@1 (vs 45.2% for the best baseline). Multi-agent variant leads FutureX at 49.5% (vs 47.5% for A-Evolve). Adaptive variant leads PolyBench Return at +352% (vs +320% for Meta-Harness).
2. **Greedy per-batch evolution can go net negative on a stream.** A-Evolve lands at 18.4% PolyBench Accuracy against 22.2% for the untouched solver; Meta-Harness lands at 29.4% FutureX Pass@1 against 31.0% for the untouched solver; Continual Harness is below the no-evolution baseline on all three (8.5 / 25.7 / 31.8). Evolving a harness on the batch in front of you is not reliably better than not evolving it at all.
3. **Multi-agent evolution with persistent state is the largest single component effect** (Fig. 6, on 100/60/80 tasks). Single-agent → full system: 20.3 → 44.3 CWR on PolyBench, 38% → 43% on CTF-Dojo, 38% → 44% on FutureX. Removing cross-cycle memory degrades most broadly (PolyBench collapses to 2.6 CWR); removing evaluation feedback mainly hurts PolyBench (7.5 CWR).
4. **Routing recovers only part of the adaptation gap** (Fig. 7, Table 11, replay subsets of 40/80/58 tasks). The Oracle−Naive gap — the paper's empirical estimate of $L_{\mathrm{adapt}}$ — is +37.5 pp on CTF-Dojo ($p_{\mathrm{adj}} = 4.8 \times 10^{-4}$) and +8.8 pp CWR on PolyBench ($p_{\mathrm{adj}} = 1.8 \times 10^{-5}$), with 95% bootstrap confidence intervals and Holm–Bonferroni-corrected paired Wilcoxon tests. The deployed router captures +17.5 of the 37.5 on CTF-Dojo and +2.7 of the 8.8 on PolyBench, and on FutureX it is 5.2 pp *worse* than never branching. Specialization creates headroom that the router only partly converts.
5. **Bottlenecks differ by stream** (Fig. 4). FutureX pass rate rises monotonically 34.0 → 47.6 → 57.1% across offline, date-filtered curated web, and unrestricted web retrieval, identifying source acquisition rather than reasoning as binding. CTF-Dojo pass rate falls with challenge payload size from 81.8% to 30.4% for the best single-agent variant and 90.9% to 39.1% for the multi-agent variant, a roughly 9-point margin sustained throughout — the multi-agent *evolver* mitigates the payload bottleneck without removing it.
6. **Cross-domain harness mixing is destructive** (Fig. 11). A PolyBench-evolved workspace reaches +60% CWR on PolyBench; the workspace formed by combining all three evolved workspaces reaches +3%, a loss of 57 points. Harness content learned on one regime actively dilutes another.

### c) Where it shines

Largest absolute gains appear where a whole missing capability had to be constructed: CTF-Dojo `web` (+27 pp over the Sonnet baseline) and `crypto` (+19 pp), Table 6; and PolyBench overall, where accuracy roughly triples over every baseline that is not Meta-Harness. English-language FutureX slices benefit most (finance 21.1 → 56.6, geopolitics 37.5 → 65.1, Table 7), while the single Chinese-finance slice shows the source-discovery failure neither evolution nor routing recovers.

Human steering is effective exactly where the paper predicts and nowhere else: FutureX slice lift is 0 on broad prediction-market questions, +5 on broad search-dependent questions, +20 on the finance-and-technology slice the steering directly targeted, and +15 on adjacent Western-specialty questions (Fig. 9). The takeaway the paper draws is that human input helps when it injects missing external signal, not when it offers generic advice.

### d) Limitations

**Acknowledged.** Only three streams, all of which the authors chose to exercise their three dimensions; and the two losses the framework is built on are never estimated, only diagnosed indirectly (Sec. 6).

**Implicit, and directly relevant to this project.** Table 9 reports the cycle index at which each run's cumulative mean was highest: PolyBench peaks at cycle 22 of 51, FutureX at cycle 10 of 26, CTF-Dojo at cycle 1 of 14. The paper's own system peaks roughly halfway through the stream and declines thereafter — on CTF-Dojo, immediately. The Figure 1 failure the paper opens with is pushed later, not removed. The paper acknowledges this only obliquely, noting in Appendix F that "a peak well before the final cycle is consistent with the overfitting trend in Figure 1."

Everything is single-seed by construction (temperature 0 throughout), so no variance estimate exists for the headline table; the confidence intervals and significance tests appear only in the routing analysis, which is a post-hoc replay rather than an end-to-end deployment. Cost is substantial: the Full System uses 233.2 M input and 12.2 M output solver tokens over 59.5 hours on PolyBench against 35.3 M / 5.0 M and 25.6 hours for the no-evolution baseline (Table 5), and *evolver*-side tokens were not persisted at all, so the true cost of evolution is unreported.

## 5. Reproduction and Application

### a) Open source

Yes: https://github.com/A-EVO-Lab/a-evolve, branch `release/adaptive-auto-harness` (MIT, cloned at commit `17bc9eb`; see `repo_analysis.md` for the component-level reading). The README reproduces the paper's runs as named cells — `bash scripts/poly_hypothesis.sh H0` (no evolution), `H1` (full single-agent evolution), `H4_multi` (four-phase multi-agent), `H4_multi_nav` (four-phase plus tree routing) — with sibling launchers for CTF-Dojo and FutureX. PolyBench needs only a SQLite snapshot of Polymarket markets fetched by `data/download_data.py`; CTF-Dojo and FutureX need Docker sandboxes. Everything runs against provider-hosted models, with no local GPU and no weight updates anywhere.

### b) Implementation details worth attention

Temperature 0 for solver, *evolver*, and router. Batch sizes 100/20/20 and 51/14/26 cycles. Router confidence threshold 0.7. Three parallel research agents, three build/verify retries. Solver turn cap 80; *evolver* token budget 128k. The system prompt is hard-truncated at 10,000 characters by a shared guardrail — this, plus branch isolation, is the entire defense against the harness bloat Figure 1 diagnoses. Every released config sets `holdout_ratio: 0.0`, which disables the inherited A-Evolve validation gate: in the paper's runs, no mutation is ever validated against held-out tasks before being committed.

### c) Transferability

The pieces that transfer to this project are the benchmark harnesses and the solve-time seam, not the objective. The three stream adapters (`benchmarks/{polybench,ctf_dojo,futurex}/`) give heterogeneous, non-stationary, temporally ordered streams with honest label delay already implemented. The `Adaptation` protocol (`select` → opaque key; `materialize` → realize the harness) is a small, clean interface with four implementations already in the repo, so a different selection policy drops in as a sibling class. The git-backed workspace supplies per-cycle tagged snapshots (`pre-evo-N`, `evo-N`), which is exactly the harness-evolution log a state-level estimator would train on.

## 6. Summary

### a) One-sentence core idea

Build many regime-specific harnesses with a stateful four-role evolver, keep them isolated in git branches, and route each incoming task to one at solve time.

### b) Quick-reference pipeline

1. Tasks arrive in real chronological order in batches; each task is answered with the harness that existed before it arrived.
2. A router agent reads the metadata of each stored harness variant and picks the one to run for this task; that variant is checked out and the solver answers under it.
3. Each task's true outcome is released to the harness builder only after the task's real-world resolution date.
4. Once per batch, four specialized agents — one reading failures, several researching fixes in parallel, one implementing, one testing — write the change into the store, creating a new isolated variant when the failures justify specializing.
5. Two hooks ask a human for exactly the things history cannot supply (credentials, source direction) and nothing else.

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched the exact title, the short name "Adaptive Auto-Harness", and the OpenReview note-search API on 2026-09-14). The paper is arXiv-only at v2 with no venue named in its comments field, and the only third-party commentary surfaced by search was an automated paper-aggregator summary, not peer review. Nothing is reported here about reviewer opinion because no reviews exist to report.

---

## Relevance to this project

**Confirms the premise of our claim, with the best external evidence available.** Figure 1 is a direct measurement of greedy patching lowering future stream performance: unbounded evolution grows the prompt from 2 KB to 68 KB and skills from 12 to 34, pass-rate lift peaks and decays for every stopping budget, and "later in the stream, shorter runs outperform longer ones." Table 2 supplies the stronger version — A-Evolve ends *below* the untouched solver on PolyBench accuracy (18.4 vs 22.2) and Meta-Harness ends below it on FutureX (29.4 vs 31.0). Greedy per-batch harness editing is not merely suboptimal on a stream; it can be worse than not editing.

**Confirms our positioning clause verbatim, and lets us sharpen it.** `docs/framing.md` says this paper "runs real task streams and fights bloat with prompt-level audits and branch isolation." That holds, and the code adds two facts worth carrying: the bloat defense is a single 10,000-character truncation constant on the system prompt, and the harness tree has no merge and no pruning — `promotion_threshold` and `staleness_window` exist in the config schema but are read by no code, so the store only grows. No edit is ever scored against anything but the batch that produced it, and no earlier task is ever re-checked, so the horizon is zero and there is no retention constraint of any kind — strictly weaker on retention than Harness Continual Learning, which at least gates each edit on bounded anchor damage.

**Their decomposition is missing exactly our term, and they say so.** Proposition 1 splits regret into an evolver-capability term and a commit-before-seeing-the-task term, both defined against a per-task oracle at time $t$. Nothing in it represents what an edit costs later tasks, which is why a system that optimizes both terms perfectly can still trace the Figure 1 curve — and does: Table 9 puts the multi-agent run's peak at cycle 22 of 51 on PolyBench, 10 of 26 on FutureX, and 1 of 14 on CTF-Dojo. Section 6 states plainly that both losses are "analytical quantities, not directly estimated oracle losses," diagnosed through ablations rather than by any estimator. A trained state-level estimate of future stream performance is the natural third term and the missing machinery, and both gaps are stated by the authors rather than inferred by us.

**Sharpens what "generalization to later tasks" costs.** Figure 11 measures harness-level negative transfer directly: a PolyBench-evolved workspace reaches +60% CWR, and combining all three evolved workspaces reaches +3% — 57 points lost to mixing. This is a number to cite when arguing that one dense harness cannot serve a heterogeneous stream, and also a caution for any design of ours that merges content across regimes (our `promotion` operation is precisely such a merge).

**Gives us infrastructure we do not have to build.** Three real streams with implemented delayed labels (5,075 / 261 / 503 tasks), a near-empty seed harness (27/24/114 prompt lines, zero skills, tools, or memory) so all harness content is agent-built, a clean two-method solve-time adaptation seam with four existing implementations, and per-cycle git tags that are a ready-made harness-evolution log.

**Reported-versus-released discrepancy to remember.** Table 3 describes EGL as an "Expected-Gain-from-Learning trigger that gates whether a cycle runs," with a threshold of 0.05 and a window of 3 on all benchmarks. In the released code, `egl.py` defines EGL as "Evolutionary Generality Loss" — new skills per thousand tasks solved — and both of its functions are dead: nothing calls them, `egl_threshold` is read by no code, and the loop actually stops on a plain score-plateau check using `egl_window` and a hardcoded epsilon of 0.01. The trigger described in the paper is not the trigger in the code.
