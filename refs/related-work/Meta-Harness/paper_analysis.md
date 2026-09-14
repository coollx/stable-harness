# Meta-Harness: End-to-End Optimization of Model Harnesses

- **Original title**: Meta-Harness: End-to-End Optimization of Model Harnesses
- **Authors**: Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn
- **Affiliations**: Stanford (Lee, Nair, Zhang, Finn — the IRIS lab, per the code release), KRAFTON (Kangwook Lee), MIT (Khattab)
- **Venue / year**: arXiv preprint, 2026 (arXiv:2603.28052v1 [cs.AI], submitted 30 Mar 2026, 26 pages including appendices A–E)
- **Source URL**: https://arxiv.org/abs/2603.28052
- **Project page**: https://yoonholee.com/meta-harness/
- **OpenReview**: no public page found (see section 7)
- **Code**: https://github.com/stanford-iris-lab/meta-harness (the framework and two reference experiments) plus https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact (the single discovered TerminalBench-2 harness). See `repo_analysis.md`.

Classification: **method paper**. The contribution is a built outer-loop system — a search procedure over harness code with a specified proposer, feedback channel, and evaluation loop — that others can reproduce and apply to a new domain; the three experiments exist to show the system works, not to characterize a phenomenon.

Short name used below: Meta-Harness, the paper's own name. Note the paper's own caution (Sec. 3): Meta-Harness *is itself* a harness, so "harness" unqualified below means the task-specific program being optimized, not the optimizer.

---

## 1. Method Motivation

### a) Why was this method proposed?

The driving observation is that the code around a fixed language model — what to store, what to retrieve, what to show at each step — moves measured performance as much as the weights do, and that this code is still written by hand. The paper's opening evidence is a reported 6x performance gap on the same benchmark from harness changes alone (Sec. 1, citing Tian et al. 2026). The authors ask whether the human loop of "inspect failures, adjust heuristics, iterate on a small number of designs" can be automated (Sec. 1).

The second, sharper motivation is a claim about *feedback bandwidth*. The natural tool for this job — existing text optimizers — compresses the history of prior attempts into scalar scores or short summaries. The authors argue this compression is fatal for harness engineering specifically, because "harnesses act over long horizons: a single choice about what to store, when to retrieve it, or how to present it can affect behavior many reasoning steps later. Compressed feedback often removes the information needed to trace downstream failures to earlier harness decisions" (Sec. 1).

### b) Pain points of existing methods

The paper quantifies the gap in Table 1, which is the cleanest statement of its thesis. Per optimization step, OPRO carries 0.002 MTok (a window of past (solution, score) pairs), TextGrad 0.015 MTok (textual feedback on the current artifact only), AlphaEvolve 0.022 MTok (program database plus scores), GEPA 0.008 MTok (summarized reflective feedback from rollout traces), Feedback Descent 0.012 MTok, TTT-Discover 0.026 MTok (a window holding the previous solution fragment). Meta-Harness carries 10.0 MTok and the history mode is "Full — all logs and scores". The authors state that in their most demanding setting a single evaluation can produce up to 10,000,000 tokens of diagnostic information, "roughly three orders of magnitude beyond the largest feedback budgets used in prior text optimization settings" (Sec. 1).

Three specific faults are named (Sec. 1, expanded in Appendix E):

1. **Memoryless or window-limited conditioning** — methods that see only the current candidate or a recent window (Self-Refine, OPRO, TextGrad) cannot compare a retrieval strategy that helps one problem class against one that hurts another.
2. **Scalar-only feedback** — AlphaEvolve and OpenEvolve give the proposer a program database with scores and apply fixed mutation strategies to tournament-selected parents; the authors argue this suits stateless functions with clean scalar objectives, not stateful programs that accumulate experience across many examples (Appendix E).
3. **Fixed critique formats** — GEPA provides rollout traces but one candidate at a time in a fixed critique format "that must anticipate what information is relevant" (Appendix E).

The authors are explicit that this is not a claim prior work was wrong: "This is a pragmatic scalability choice, not evidence that longer-range dependencies are uninformative" (Sec. 1).

### c) Core hypothesis / intuition

If a coding agent is given unrestricted filesystem access to the source code, scores, and raw execution traces of every harness tried so far, it can diagnose *why* a design failed rather than only *that* it failed, and the resulting search over harness code outperforms both hand-engineered harnesses and compressed-feedback text optimizers.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

**Objective (Sec. 3).** A harness $H$ is "a stateful program that wraps a language model and determines what the model sees at each step". With a fixed model $M$ and a task distribution $\mathcal{X}$, a rollout is $\tau \sim p_M(H, x)$ for $x \sim \mathcal{X}$, and a task-specific reward $r(\tau, x)$ scores the trajectory. The optimization target is

$$H^* = \arg\max_H \ \mathbb{E}_{x \sim \mathcal{X},\ \tau \sim p_M(H,x)}\ r(\tau, x).$$

Read this carefully, because it is what matters for our project: there is no time index on $H$, no sequence of edits, and no term for anything beyond the current expectation. The expectation is estimated by a fixed **search set** — a static subset of task instances reused at every iteration — so the quantity being maximized does not move as the search proceeds. When two objectives matter (accuracy and context cost), candidates are compared by Pareto dominance rather than by a scalarized combination (Sec. 3).

**The search loop (Algorithm 1, Fig. 2).** Input: tasks $\mathcal{X}$, a frozen model $M$, a proposer $P$, an iteration count $N$.

1. **Initialize** a population $\mathcal{H}$ of valid starting harnesses — in practice the hand-designed baselines for that domain (zero-shot, few-shot, ACE, MCE for classification; Terminus 2 and Terminus-KIRA for coding).
2. **Initialize** a filesystem $\mathcal{D}$ that will store code, scores, and traces.
3. **Seed**: evaluate every harness in $\mathcal{H}$ and write $(H, E_H)$ into $\mathcal{D}$.
4. **Per iteration** $t = 1 \ldots N$: the proposer $P$ *queries the filesystem* $\mathcal{D}$ — inspecting prior source code, scores, and execution traces with ordinary tools such as `grep` and `cat` — then proposes $k$ new harnesses $\{H_1, \ldots, H_k\}$.
5. Each proposed harness passes **interface validation** (an import-and-instantiate check); survivors are evaluated on the search set and the result is appended to $\mathcal{D}$.
6. **Return** the Pareto frontier of harnesses stored in $\mathcal{D}$.

After the loop ends, a **single final test-set evaluation** is run on the Pareto frontier. The paper is explicit: "The proposer never sees test-set results; its only feedback comes from the search set, the subset of task instances used to evaluate candidate harnesses during search and generate the feedback signal for improvement, and from execution traces logged during those search runs" (Sec. 3).

Three design choices are deliberately *absent* and the paper flags each: no parent-selection rule ("the proposer is free to inspect *any* prior harness and its execution trace when proposing new ones"), no fixed mutation operator set, no persistent memory mechanism outside the filesystem itself (Sec. 2, Sec. 3). The authors call this minimality deliberate: "by leaving diagnosis and edit decisions to the proposer rather than hard-coding search heuristics, Meta-Harness can improve automatically as coding agents become more capable" (Sec. 3).

### b) Architecture components

**The proposer.** A coding agent — Claude Code running Opus 4.6 in all experiments (Sec. 3, "Practical implementation"). The paper argues the choice of a coding agent rather than a raw model is load-bearing: "the amount of experience quickly exceeds context limits, so the proposer must decide *what* to inspect and validate edits through direct interaction with the codebase" (Sec. 1). It is guided by "a minimal domain-specific skill that describes where to write new harnesses, how to inspect previous harnesses and their execution traces, and what files it can and cannot modify."

**The filesystem as feedback channel.** Each evaluated harness contributes a directory holding its source code, scores, and execution traces (prompts, tool calls, model outputs, state updates). Appendix A.1 measures the access pattern on the TerminalBench-2 run: the proposer reads a **median of 82 files per iteration** (range 69–99), split **41% prior harness source code, 40% execution traces, 6% score/summary files, 13% other**, referencing over 20 prior candidates per step. The authors draw the right conclusion from this: "the proposer's access pattern is non-Markovian: it routinely inspects the majority of available history rather than conditioning only on the most recent parent" (Appendix A.1).

**The candidate harness.** A single-file Python program, 100–1000 lines (Appendix B). For text classification it implements a `MemorySystem` interface — `predict`, `learn_from_batch`, `get_state`, `set_state`. For coding it subclasses the benchmark's `Terminus2` agent class and may override any method: the LLM call, tool parsing, command execution, the main agent loop, the completion-confirmation step, the system prompt path, and context summarization. The search space is stated as "arbitrary Python code" subject only to the interface.

**Scale.** A typical run evaluates roughly 60 harnesses over 20 iterations (Sec. 3). Text classification: 20 iterations, 2 candidates each, 40 harnesses (Sec. 4.1). Math retrieval: 40 iterations, 109 candidate retrieval harnesses (Sec. 4.2). Coding: 10 iterations (Appendix A.1).

### c) Formulas / algorithms

There is exactly one equation, the objective above, and one algorithm, the loop above. This is unusual and worth naming: Meta-Harness has **no learned component and no training signal in the technical sense**. Nothing is fitted; nothing has parameters. The proposer is a prompted coding agent, the selection rule is Pareto dominance over search-set scores, and the "advantage" of a candidate is simply whether its search-set score exceeds the incumbent frontier. The intelligence is entirely in what the proposer reads, and the paper's claim is precisely that this is where it should be.

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Against **hand-engineered harnesses** (ACE, MCE, Terminus 2, Terminus-KIRA): the design loop is closed by a machine rather than by an engineer, and the artifact is discovered rather than specified.

Against **text optimizers** (OPRO, TextGrad, GEPA, AlphaEvolve/OpenEvolve, Feedback Descent, TTT-Discover): three differences stack. First, the optimization target is a complete executable procedure, not a prompt or a stateless function. Second, feedback is the full uncompressed history, accessed selectively, rather than a summary assembled in advance by the outer loop. Third, the search structure is minimal — no parent selection, no mutation operators — on the bet that a capable coding agent supplies better heuristics than hand-coded ones.

Against **agent-design search** (ADAS, AFlow, MemEvolve, and continual-memory search): the authors position those as searching over workflow graphs or memory designs within fixed scaffolds, while Meta-Harness "searches over domain-specific harnesses, including prompt construction, retrieval, and state update strategies that reset between tasks" (Sec. 2). The phrase "reset between tasks" is important and is discussed in section 5 below.

### b) Key innovations and significance

1. **Full-history filesystem access as the feedback channel** — *significant*. It is the paper's thesis, it is directly ablated (Table 3), and the ablation is decisive.
2. **Deliberate removal of search structure** — *notable*. No parent selection and no mutation operators is a real departure from evolutionary program search; the evidence that it helps is indirect (Meta-Harness beats OpenEvolve and TTT-Discover, but proposer strength is a confound).
3. **Treating the whole harness as the search space** — *notable*. Prior systems edit one layer; here the proposer may rewrite prompts, retrieval, memory, tool handling, and the control loop in one file.
4. **Pareto formulation over accuracy and context cost** — *incremental but useful*. It lets one search produce an operating curve rather than a point (Fig. 3, Table 9).

### c) Applicability

It excels where: the base model is fixed and the opportunity is in retrieval/memory/context/scaffolding; tasks repeat so a harness can be reused; a stable automatic metric exists; evaluation is cheap enough for ~50 full evaluations per run; and there are recurring error patterns visible in traces. The repo's `ONBOARDING.md` states these conditions explicitly and says it is "usually a poor fit when most of the gain would need to come from changing the base model or when solving a task that has no stable evaluation loop."

It struggles where: evaluation is noisy or subjective; the search set is too small to expose failure modes or so easy the baseline saturates; or the domain has no held-out set (the authors flag leakage as the first thing to worry about — "Be especially careful about evaluation leakage and hidden dependence on the final test set", `ONBOARDING.md`).

### d) Comparison table

| Method | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| Meta-Harness | Full uncompressed history, selectively accessed; searches whole executable harnesses; no hand-coded search heuristics; produces a Pareto curve; discovered artifacts are readable and transfer across models | Very expensive (~$500 and 4–6 h per coding iteration, per the repo README); quality tied to one strong proposer (only Claude Code tested); objective is a static search set with no notion of a later task; coding-domain results have no held-out split | Vary the proposer agent (the authors name this as future work); co-evolve harness and weights (also named); add a term that values future adaptability rather than only current-set score |
| Hand-designed harnesses (ACE, MCE, Terminus-KIRA) | Interpretable, cheap, embody real domain knowledge | Human-rate iteration over few designs; ACE and MCE pay 4x the context for lower accuracy (Table 2) | Use as search population seeds, which is exactly what Meta-Harness does |
| OpenEvolve / AlphaEvolve | Structured program search with an explicit database and mutation operators | Scalar-score feedback (4–22K tokens/step, Table 1); local mutations on tournament-selected parents; designed for stateless functions | Give the mutator trace access — that is the Meta-Harness delta |
| GEPA | Richest of the prior text optimizers; provides rollout traces | One candidate at a time, 2–8K tokens/step, fixed critique format chosen in advance | Remove the fixed format; let the agent choose what to read |
| TTT-Discover | Strong search-set performance among text optimizers | Window over the previous solution fragment only (0.026 MTok/step) | Replace the PUCT reuse rule with agentic selection |

---

## 4. Experimental Validation

### a) Experimental design

Three domains, each with a frozen base model and domain-appropriate baselines.

**Online text classification (Sec. 4.1).** Setup follows ACE (Zhang et al.) and MCE (Ye et al.): the model receives labeled examples one at a time, updates its memory, and is evaluated on a held-out test set. Base model GPT-OSS-120B. Three datasets chosen for difficulty and domain diversity: LawBench (criminal-charge prediction from case descriptions, 215 classes), Symptom2Disease (22 classes), USPTO-50k (precursor reactants from product molecules, 180 classes). Search population seeded from zero-shot, few-shot, ACE, and MCE. 20 iterations x 2 candidates = 40 harnesses. Comparison methods given the same evaluation budget: Best-of-N (independent seed samples, a compute-matched control for whether search structure matters at all), OpenEvolve, TTT-Discover, GEPA.

**Retrieval-augmented math reasoning (Sec. 4.2).** A retrieval corpus of 535,356 solved problems from eight open-source datasets (Table 10), deduplicated and decontaminated against every evaluation benchmark and the search set by exact prefix matching and fuzzy Jaccard similarity at threshold 0.8, with held-out problems manually inspected in top BM25 retrievals. Search: 40 iterations over a 250-problem OlympiadBench + Omni-MATH search set, producing 109 candidate retrieval harnesses, one selected by search-set score using GPT-OSS-20B. Evaluation: 200 previously unseen IMO-level problems (Table 11: 100 from IMO-AnswerBench, 60 IMO-ProofBench, 17 ArXivMath Dec 2025, 23 ArXivMath Jan 2026), pass@1 averaged over three samples, on **five held-out models** — GPT-OSS-20B plus GPT-5.4-nano, GPT-5.4-mini, Gemini-3.1-Flash-Lite, Gemini-3-Flash, none of which were used during search except the first.

**Agentic coding (Sec. 4.3).** TerminalBench-2, 89 long-horizon terminal tasks. Two base models: Claude Opus 4.6 and Claude Haiku 4.5. Search initialized from Terminus 2 and Terminus-KIRA. **Search and final evaluation are performed on the same 89-task benchmark**, framed as a "discovery problem": "the benchmark is small and expensive enough that introducing a separate split would materially weaken the search signal." The authors add manual inspection and regex-based audits for task-specific string leakage, and note the resulting harness is specialized to this regime.

### b) Key results

1. **Text classification, test set (Table 2)**: Meta-Harness 48.6 average accuracy with 11.4K additional context tokens, against ACE 40.9 / 50.8K and MCE 40.0 / 28.5K — **+7.7 points over ACE with 4x less context**, and +8.6 over MCE. Per dataset: USPTO 14.0, Symptom2Disease 86.8, LawBench 45.0 (versus ACE's 16.0 / 77.8 / 29.0).
2. **Feedback ablation (Table 3)** — the paper's most load-bearing result. Scores-only: 34.6 median, 41.3 best, 26 runs above zero-shot. Scores-plus-LLM-summary: 34.9 median, 38.7 best, 23 above zero-shot. Full interface with raw execution traces: **50.0 median, 56.7 best, 39 above zero-shot**. The median full-interface candidate beats the *best* candidate from either ablation. The authors' reading: "summaries do not recover the missing signal, and may even hurt by compressing away diagnostically useful details."
3. **Against text optimizers, search set (Table 4)**: Meta-Harness 50.0 median / 56.7 best, versus OpenEvolve 39.1 / 43.3, TTT-Discover 34.1 / 45.6, Best-of-N 34.0 / 44.2, GEPA 32.6 / 40.2. Figure 4 shows Meta-Harness reaching the *final* accuracy of OpenEvolve and TTT-Discover within its first four evaluations — a 10x evaluation-efficiency claim — and continuing to improve.
4. **Out-of-distribution classification (Table 5)**: on nine datasets never seen during search (SciCite, FiNER-139, Amazon Reviews, Financial PhraseBank, GoEmotions, Banking77, AG News, SciTail, TweetEval-Hate), the discovered harness reaches **73.1 average accuracy versus ACE 70.2**, is best on 6 of 9, and beats the next-best method by 2.9 points at 7.3K context. Naively adding 32 few-shot examples *hurts* on 7 of 9 tasks.
5. **Math retrieval across five held-out models (Table 6)**: the discovered harness averages **38.8 pass@1 versus 34.1 with no retriever (+4.7)**, beating BM25 retrieval (37.5) by 1.3 and dense retrieval at $k=5$ (38.1) by 0.7. Gains per model range +1.6 (GPT-5.4-mini) to +8.7 (GPT-5.4-nano). Dense retrieval and random few-shot each regress on some models; the discovered harness regresses on none.
6. **TerminalBench-2 (Table 7)**: on Opus 4.6, 76.4% pass rate, above hand-engineered Terminus-KIRA (74.7), Capy (75.3), and Claude Code (58.0), ranking #2 behind ForgeCode (81.8) — which the authors could not reproduce from its public code. On Haiku 4.5, **37.6% versus next-best Goose at 35.5 (+2.1)**, ranking #1.

### c) Where does it shine?

The weaker the base model and the more room for context engineering, the larger the gain: +2.1 points on Haiku 4.5 versus a #2 finish on Opus 4.6 (Table 7); +8.7 points on GPT-5.4-nano versus +1.6 on GPT-5.4-mini (Table 6). It also shines where the *shape* of the fix is structural rather than parametric — Appendix B shows discovered harnesses that are genuinely nontrivial programs: a two-call draft-then-verify classifier that re-retrieves confirming and contradicting examples conditioned on its own draft label (Fig. 5); a label-primed prompt built from a label primer, a per-class coverage block, and query-anchored contrastive pairs (Fig. 6); a four-route lexical router over BM25 retrieval policies with per-route deduplication and difficulty reranking, formed by *merging two successful search lineages* (Fig. 8).

The single most instructive result for our purposes is qualitative, in **Appendix A.2**, quoted verbatim from the proposer's own log on TerminalBench-2. Iterations 1 and 2 bundle structural bugfixes with prompt-template edits; both regress sharply from the 64.4% baseline. At iteration 3 the proposer diagnoses the confound in its own words: "Root cause of regressions: Prompt template changes (cleanup directives) caused the agent to delete necessary state before task completion. The structural bugfixes were confounded with harmful prompt changes." Iterations 4–6 keep regressing. At iteration 7 the proposer states the lesson: "All 6 prior iterations regressed from the 64.4% baseline because they modified the completion flow, prompt template, or observation processing. `evo_env_bootstrap` takes a different approach — **purely additive**. ... No other methods are changed." That candidate wins the run. Iteration 10 shows cross-run transfer: the proposer cites a result from a *separate earlier search run*.

The winning edit itself (Appendix B.3, and the artifact repo) is small: before the agent loop begins, run one compound shell command gathering working directory, an `/app` listing truncated to 20 entries, available languages and versions, package managers, and memory, and inject it as an `[Environment Snapshot]` block in the initial prompt, guarded by a 15-second timeout that fails silently. Roughly 80 lines on top of Terminus-KIRA. It gains on 7 of 89 tasks, concentrated in tasks needing domain-specific tooling whose availability cannot be assumed (bioinformatics, rendering, chess engines, CoreWars).

### d) Limitations

**Acknowledged.** Only one proposer agent tested — "our experiments demonstrate that harness search can work with one particularly strong coding-agent proposer (Claude Code); a broader study of how the effect varies across proposer agents remains for future work" (Sec. 5). The TerminalBench-2 harness is "specialized to the TerminalBench-2 regime" (Sec. 4.3). ForgeCode's higher score could not be reproduced (Sec. 4.3). Overfitting is present but the authors argue it is at least *inspectable*: "brittle if-chains or hard-coded class mappings are visible on inspection in a way that weight-space overfitting is not" (Sec. 5).

**Unacknowledged or under-flagged.**

- The **TerminalBench-2 experiment has no held-out split at all**, and the check against leakage is manual inspection plus regex audits — a procedure with no stated recall. Every headline coding number is a search-set number.
- The proposer skill files (shipped in the repo) contain hand-written, unmeasured anti-overfitting rules: "No dataset-specific hints", "Never mention dataset names in system code, prompts, or comments", "If in doubt, make it more general." These are prompt-level guards standing in for a quantity nobody measures. Their effect is never ablated.
- **The main text reports one selected harness per domain**, chosen by search-set score from the Pareto frontier. Table 9 shows the frontier's spread is wide (40.1 to 48.6 average accuracy across eight Pareto-optimal variants), so the headline number is a max over a population, compared against single hand-designed baselines.
- Cost is reported only in the repo, not the paper: the default 89x2 coding search is ~4–6 hours and **~$500 per iteration** (repo README). At 10 iterations that is roughly $5,000 for one search run.
- Text classification uses **one seed (42) and one base model**, with 30–50 validation examples per dataset (repo `config.yaml`), so "median across candidates" is carrying a lot of weight that a confidence interval would carry better.
- Figure 7 plots search-set against test accuracy per dataset and shows substantial scatter around the diagonal, but no correlation coefficient or overfitting magnitude is reported.

---

## 5. Reproduction and Application

### a) Open source

Yes, two repositories. `https://github.com/stanford-iris-lab/meta-harness` holds the framework, an `ONBOARDING.md` conversation script for adapting it to a new domain, and both reference experiments; `https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact` holds the single discovered coding harness as a standalone runnable agent. The framework README carries the caveat: "This is a cleaned up version of the code we used for the paper. It has not been tested beyond verifying that it runs." Full component map in `repo_analysis.md`.

Reproduction steps: `uv sync` in a reference example, then `uv run python meta_harness.py --iterations N`, then a separate `--test` invocation that freezes the run and evaluates the frontier once on the test split.

### b) Implementation details worth attention

- The proposer is invoked with `model="opus"`, `effort="max"`, and the tool set `[Read, Glob, Grep, Agent, Write, Edit, Bash]` — filesystem access is literally the allowlist.
- Test results live in a directory that simply **does not exist during evolution**; a run that has been finalized cannot be evolved further (the `finalized.json` state flips to `complete` and further evolution exits with an error). The separation is temporal, not an access-control mechanism.
- Candidate validity is one import check. In the coding domain there is an additional smoke test on a single task before a full evaluation.
- Appendix D's practical guidance is unusually candid and is the most transferable part of the paper: write a good proposer skill (it "should constrain outputs and safety-relevant behavior, not the proposer's diagnosis procedure"); build a search set the baseline finds hard, sized for roughly 50 full evaluations per run (50–100 classification examples, 88 math problems); log everything in a navigable machine-readable layout; validate cheaply before paying for a benchmark; run evaluation outside the proposer. The authors note that "after enough iterations, the accumulated traces often shape the proposer's behavior more than the skill itself", and that iterating on the skill mattered more to search quality than iteration count or population size.

### c) Transferability

High, and the repo is built for it: the `ONBOARDING.md` prompt walks a coding assistant through producing a `domain_spec.md` covering problem framing, harness interface, search-set and held-out evaluation, baselines, offline traces, online logging, and budget. Six independent community reimplementations and applications are already listed in the README (legal-agent benchmarks, a Claude Code skill port, long-video retrieval), which is itself evidence the interface is portable. For our project, the two things worth lifting are the filesystem-as-history design and the harness-evolution logs it produces; the objective is the thing to replace.

---

## 6. Summary

### a) One-sentence core idea

A coding agent with full filesystem access to every past harness's code, scores, and traces searches over harness programs better than compressed-feedback optimizers or human engineers.

### b) Quick-reference pipeline

1. Start from a population of existing hand-written harnesses for the task and score each one on a fixed search set.
2. Keep every candidate's source code, score, and raw execution trace in its own directory on a shared filesystem.
3. Each iteration, let a coding agent read whatever it wants from that filesystem, form a hypothesis about why past candidates failed, and write one to three new harness programs.
4. Check each new program imports and runs, score it on the same search set, write everything back to the filesystem, repeat for a fixed number of iterations.
5. Return the set of harnesses that are not dominated on accuracy and context cost, and evaluate them once on a held-out test set.

### c) Relation to our project

The optimizer's objective is expected reward on a fixed task distribution estimated by a static search set (Sec. 3, Algorithm 1 lines 5 and 12). No term rewards anything after the current evaluation; in our vocabulary the discount on future batch performance (horizon gamma) is zero, and more strictly than in the systems we usually call greedy — the score is not even a moving batch but one static set reused for every iteration. Harness plasticity, the future stream performance achievable from a harness, is neither measured nor constrained.

Held-out evaluation *does* exist and the survey's one-line characterization of Meta-Harness deserves this refinement. Three held-out measurements are reported: nine unseen classification datasets (Table 5, 73.1 versus ACE 70.2), 200 unseen IMO-level problems on five models not used in search (Table 6, +4.7), and a final test-set pass over the Pareto frontier that the proposer never sees (Sec. 3). But all three are *reports of generalization after the fact*, never optimization signal, and none of them is a later-task measurement: there is no stream, no arrival order, no anchor set, no re-check of previously passed tasks, and no scoring of a task with the harness as it stood before that task arrived (score-before-update). Each candidate harness is evaluated from a clean start on the same set rather than carried forward and patched, so the paper cannot exhibit — and does not look for — degradation across a sequence of edits. The TerminalBench-2 experiment has no held-out split whatsoever and is the cleanest instance of same-set optimization in the reference set.

The most valuable thing here for us is not the method but Appendix A.2: six consecutive regressions from prompt and control-flow edits, followed by the proposer articulating the rule "modifications to prompts and completion flow are high risk" and pivoting to purely additive changes. That is a working system discovering, at the level of a single search run, that parts of its harness have become expensive to modify — the phenomenon our claim says accumulates. It is an anecdote, not a measurement, and nothing in the paper converts it into one.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched the exact title "Meta-Harness: End-to-End Optimization of Model Harnesses" and the "Meta-Harness" keyword on OpenReview, 2026-09-14). The paper is an arXiv preprint (v1, 30 Mar 2026) with no venue listed in its comments field and no conference submission surfaced by search. Informal commentary exists — a HuggingFace papers page, a Pith Review entry, and a paper-notes issue — but none of these is peer review and none is summarized here as such.
