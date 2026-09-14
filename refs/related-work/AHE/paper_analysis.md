# Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses

- **Original title**: Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses
- **Authors**: Jiahang Lin, Shichun Liu (equal contribution), Chengjun Pan, Lizhi Lin, Shihan Dou, Zhiheng Xi, Xuanjing Huang, Hang Yan, Zhenhua Han (corresponding), Tao Gui (corresponding), Yu-Gang Jiang
- **Affiliations**: Fudan University (1), Peking University (2), Shanghai Qiji Zhifeng Co., Ltd (3); the first two authors did the work during an internship at Shanghai Qiji Zhifeng
- **Venue / year**: arXiv preprint, 2026 (arXiv:2604.25850v4 [cs.CL], submitted 28 Apr 2026, last revised 18 May 2026, 35 pages including appendices A-E). The front page is marked "Preprint"; no venue is stated.
- **Source URL**: https://arxiv.org/abs/2604.25850
- **OpenReview**: no public page found (see section 7)
- **Code**: https://github.com/china-qijizhifeng/agentic-harness-engineering — public, MIT, analyzed in `repo_analysis.md`

Short name used below: Agentic Harness Engineering (AHE), the paper's own abbreviation. Its harness framework is NexAU and its minimal seed configuration is written NexAU-0 (the paper writes it with a subscript zero).

**Classification: method.** The contribution is a built, released, reproducible loop plus a component substrate; the ten-iteration campaign, the transfer study, and the ablations exist to validate that artifact rather than to characterise a phenomenon. The self-attribution study of Sec. 4.4.2 is the one genuinely empirical part, and it is folded into section 4 below.

---

## 1. Method Motivation

### a) Why was this method proposed?

Harness design shifts task completion on long-horizon coding benchmarks even with the base model held fixed, which makes the harness "a first-class lever for improving coding agents" (Sec. 1). Two facts turn that lever into a problem. First, "the optimal harness is model-specific: a harness tuned for one base model often underperforms on another and must be re-adapted as the base model changes." Second, that re-adaptation is done by hand — "developers inspect trajectories, identify recurring failure patterns, and hand-craft edits across prompts, tools, middleware, and skills" — and as base models ship faster, "this manual loop struggles to keep pace, creating a widening gap between model capability and the harness needed to realize it" (Sec. 1). The paper's stated goal is to close that gap by handing the loop to another agent.

### b) Pain points of existing methods

The paper names one gap and three obstacles (Sec. 1, Sec. 2.2).

The gap: existing automatic optimisation edits **one** surface. "Few existing approaches jointly evolve the full set of editable components; most focus on a single component, typically the prompt, skills, or an in-context playbook" (Sec. 1). Sec. 2.2 sorts prior work by what the optimiser may observe and edit: episodic critique and reflection over the agent's own outputs; system prompts and instructions (structured playbooks such as ACE, semantic-advantage priors such as Training-Free GRPO, jointly optimised instruction-demonstration pipelines, reflective updates driven by Pareto-frontier traces such as GEPA); skill libraries; scored program and agent archives evolved through mutation; and graph-structured workflows. The complaint is about coverage, not quality: none of these reaches tools, middleware, or sub-agents, so cross-component trade-offs are invisible to them.

Three obstacles block joint evolution (abstract, Sec. 1):

1. **Heterogeneous action space.** Editable components differ in kind — prose, YAML, Python — so "edits beyond the prompt" are error-prone in "tightly coupled harness frameworks".
2. **Voluminous trajectories.** Rollout logs run past 10 million tokens per iteration; "long, unstructured trajectories yield little actionable signal".
3. **Attribution.** An edit's effect on a later score is hard to separate from noise and from other edits.

### c) Core hypothesis / intuition

"Our central insight is that this question is bottlenecked by *observability*, not by agent capability: once the evolution agent receives structured context over a clear action space, it can reliably converge on better harness designs" (Sec. 1). Each obstacle gets one matched observability pillar, and the three pillars compose into one unattended loop.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

**Input**: a seed harness $H_0$, a base model $M$, a benchmark $D$, a rollout count $k$ per task, and an iteration budget $N$. **Output**: $H_{\mathrm{best}}$, the harness from whichever iteration scored highest.

The seed is deliberately crippled. NexAU-0 "is deliberately minimal: a single shell-execution tool, no middleware, no skills, no sub-agents" and no long-term memory (Sec. 3.1, App. A). The stated reason is attribution hygiene, not modesty: "A seed already fitted to the target benchmark would contaminate every subsequent edit's attribution, since we could not tell whether a gain came from the loop or from the seed. The minimal seed forces every component AHE adds to earn its place against measured rollouts."

Algorithm 1 runs six phases per iteration $t$:

1. **Rollout**: $T_t \leftarrow \textsc{Rollout}(M, H_{t-1}, D, k)$ — run the *previous* iteration's harness over the whole benchmark, $k$ rollouts per task. Every rollout runs in a fresh remote sandbox so shell side effects cannot leak between tasks (App. A).
2. **Clean**: normalise traces into canonical form.
3. **Attribute, then roll back** (only when $t \ge 2$): $V_t \leftarrow \textsc{Attribute}(C_{t-1}, T_{t-1}, T_t)$, then $H_{t-1} \leftarrow \textsc{Rollback}(H_{t-1}, V_t)$. The previous round's change manifest $C_{t-1}$ is intersected with the observed task-level deltas to produce a per-edit verdict; edits whose predicted fixes did not materialise are candidates for reversion. Attribution runs *before* distillation "so its verdict lands inside the evidence corpus and binds each prior manifest entry as a contract rather than a rationale" (Sec. 3.3).
4. **Distil**: $R_t \leftarrow \textsc{AgentDebugger}(\tilde{T}_t)$ — layered evidence corpus.
5. **Evolve**: $(H_t, C_t) \leftarrow \textsc{Evolve}(H_{t-1}, R_t, V_t)$ — workspace edits plus a new manifest.
6. **Commit**: tag the iteration in the workspace git history; update $H_{\mathrm{best}}$ if $\mathrm{pass@1}(T_t) > \mathrm{pass@1}(H_{\mathrm{best}})$.

A one-shot explore agent runs in parallel with iteration 1 and seeds "a small number of reusable skills from the NexAU source and public coding-agent references. These skills receive no special protection: from iteration 2 onward the Evolve Agent may keep, refine, or remove them based on observed rollouts" (Sec. 3.3).

**The scoring protocol is score-before-update in the literal sense of `docs/framing.md`**: phase 1 of round $t$ measures $H_{t-1}$, the harness exactly as it stood before this round's edits. What differs from our setting is the task supply. $D$ is the *same* 89 Terminal-Bench 2 tasks every round; there is no stream and no unseen future batch. The "next round" that falsifies an edit is a re-run of the batch the edit was made on.

### b) Architecture components

**Pillar 1 — component observability (Sec. 3.1).** The harness $H$ is instantiated on the NexAU framework, "which exposes seven orthogonal component types as explicit files at fixed mount points in a single workspace". The component types are loosely coupled: "adding a middleware does not require editing the system prompt, and adding a skill does not require touching any tool." Each failure pattern maps to a single component class, which "gives the evolve agent a clean action space and localizes every pass-rate change to one file rather than scattering it across hundreds of lines of unstructured prompt prose. Each logical edit becomes one commit on the workspace's git history, which yields file-level diffs and rollback granularity for free."

The seven, with the files each occupies and what App. B.2's evolve prompt tells the agent each is for:

| Component | Files in the workspace | Character, per the evolve prompt |
|---|---|---|
| System prompt | `workspace/systemprompt.md` | Advisory, applies to all tasks: behavioural rules, workflow guidance |
| Tool description | `workspace/tool_descriptions/*.tool.yaml` | Co-located with the tool; what the model reads before calling it |
| Tool implementation | `workspace/tools/` | Controls tool behaviour directly: new capabilities, error handling, output formatting |
| Middleware | `workspace/middleware/` plus `code_agent.yaml` | Hooks into the agent loop pipeline; intercepts and transforms at execution level |
| Skill | `workspace/skills/` plus `code_agent.yaml` | On-demand, loaded when relevant; reusable workflow patterns |
| Sub-agent configuration | `workspace/sub_agents/{name}/` | Delegated execution in isolated context; offloads a specialised subtask |
| Long-term memory | `workspace/LongTermMEMORY.md` | Persistent cross-session knowledge: recurring pitfalls, proven strategies, environment quirks |

This list is exactly the layer decomposition of `docs/framing.md` minus model weights, which AHE holds fixed. Short-term memory (`ShortTermMEMORY.md`) exists but is session-scoped and explicitly off-limits to the Evolve Agent.

**Pillar 2 — experience observability (Sec. 3.2).** The Agent Debugger frames each rollout as "a navigable, file-based environment where each trajectory message lives in its own file and is reached through generic shell and scripting tools". Traces for the same query sit in one environment, and the debugger must "analyze the root cause of the failure or the success pattern", stored as a per-task analysis report that also carries the task's pass/fail status. A benchmark-level `overview.md` is aggregated from every report as the entry point for the iteration. Original traces are kept alongside the reports "in case the agents need to verify the claims in the reports", in raw and lightly cleaned form, as files supporting progressive disclosure. The figure-2 caption puts the compression at roughly 10 million raw trace tokens to a roughly 10 thousand token overview.

**Pillar 3 — decision observability (Sec. 3.3).** Two constraints govern the Evolve Agent.

*Controllability*: "the Evolve Agent writes only inside the harness workspace, while the runs directory, tracer, verifier, and LLM configuration are read-only, and the seed system prompt is marked non-deletable. These restrictions block the shortcuts an unconstrained self-modifier would take, such as disabling the verifier, swapping the model, or raising the reasoning budget, and keep every recorded gain attributable to harness edits." App. B.2's safety block adds: do not add task-specific logic or hardcoded solutions, do not reverse-engineer test cases from trajectories, and an "LLM Config Hands-Off Rule" because config changes "consistently cause broad, hard-to-diagnose regressions".

*Evidence-driven*: every change ships a manifest entry naming "the failure evidence, the inferred root cause, the targeted fix, and a predicted impact comprising both expected fixes and at-risk regressions". App. B.2 fixes the JSON shape:

```json
{
  "id": "chg-1",
  "type": "new|improvement|rollback",
  "description": "What was changed and why",
  "files": ["relative/to/workspace/file.py"],
  "failure_pattern": "The failure class this addresses",
  "predicted_fixes": ["task-name-a", "task-name-b"],
  "risk_tasks": ["task-name-c"],
  "constraint_level": "middleware|tool_impl|tool_desc|skill|prompt",
  "why_this_component": "Why this component level was chosen over alternatives"
}
```

"Each edit thereby becomes falsifiable by the next evaluation, which replaces rationale-driven self-justification with a measurable contract between rounds" (Sec. 3.3).

The evolve prompt also carries an explicit escalation rule — "**Anti-pattern:** If the same failure class persists across 2+ iterations despite fixes at one component level, that level may be the wrong choice. Rollback the ineffective change and re-approach from a different component level" — and, in the analysis-approach block, a four-way verdict vocabulary for prior changes: KEEP, IMPROVE, ROLLBACK + PIVOT.

### c) Formulas and algorithms

The paper carries almost no mathematics; its two numbered equations are both metric definitions, and the optimisation target is stated in prose.

**pass@1 (eq. 1, App. A)**: for a task set $D$ with $k$ rollouts per task and binary reward $r_{i,j} \in \{0,1\}$,

$$\mathrm{pass@1} = \frac{1}{k|D|}\sum_{i=1}^{|D|}\sum_{j=1}^{k} r_{i,j}.$$

Trials that terminate on an infrastructure exception "contribute $r = 0$ rather than being dropped, a strictly harsher convention than discarding failures that keeps our numbers comparable to the official terminal-bench leaderboard".

**Cost (eq. 2, App. A)**: $\mathrm{Succ/Mtok} = \mathrm{pass@1} \times 10^{6} / (\text{mean tokens per trial})$, expected successes per million tokens. It appears only in App. Table 5 and is never optimised.

**The objective**, from the evolve prompt (App. B.2), is one line of English: "**The sole optimization target is pass@1** — the probability that a single attempt succeeds. Every change you make should raise pass@1." There is no plasticity term, no retention term, no cost term, and no discount. In the vocabulary of `docs/framing.md` this is the horizon gamma set to zero: the harness edit is scored by the batch it was made on, re-measured one round later, and by nothing after.

**Attribution** is set intersection, not a formula (Sec. 3.3, and `evaluate_changes` in the released `evolve.py`): the round-$t$ flipped set (fail to pass) is intersected with round-$(t{-}1)$'s `predicted_fixes`, the regressed set (pass to fail) with its `risk_tasks`, and each change lands one of five verdicts — EFFECTIVE, PARTIALLY_EFFECTIVE, MIXED, INEFFECTIVE, HARMFUL. Regressions matching no change's predicted or risk list are collected as "unattributed regressions" and handed to the Evolve Agent with the instruction to diagnose them as possible interaction effects.

Two structural properties follow, and both matter for us. First, **the rollback of Algorithm 1 phase 3 is a judgement, not a gate.** The paper's prose ("ineffective ones are reverted at file granularity", abstract) reads as mechanical, but the released code computes the attribution report and hands it to the Evolve Agent with "You must use this report to decide whether to rollback previous changes. HARMFUL and INEFFECTIVE changes should be prioritized for rollback"; the source comment at the call site reads "report only, rollback decided by evolve agent". The deterministic `perform_auto_rollback` that does exist restores the whole workspace to the best-ever snapshot and is wired only into the resume path, not into the loop. So unlike Harness Continual Learning's hard admissibility gate, nothing in AHE deterministically refuses a harmful edit.

Second, **$H_{\mathrm{best}}$ is a reporting device, not a deployment rule.** Iteration $t+1$ evolves from $H_t$, the latest harness, whatever its score; the best-ever record is bookkeeping consulted at the end. Degradation therefore compounds inside the run, and only the final argmax is reported.

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Three, in the paper's own framing (Sec. 2.2, Sec. 4.2).

**Full-surface versus single-surface.** "AHE tunes the full harness as a combinatorial whole rather than a single editable surface, so cross-component trade-offs become legible to the optimizer." Its own results section turns this into the explanation for beating the two self-evolving baselines: "The gaps to ACE and TF-GRPO trace to a layer mismatch. ACE distills natural-language playbooks the agent reads in-context, and TF-GRPO is a trajectory-feedback variant of GRPO that reinforces successful tool sequences; neither method opens the surrounding scaffolding to edits. AHE instead jointly evolves system prompt, tools, middleware, and long-term memory, and Sec. 4.4.1 shows the gain concentrates in the latter three components, exactly the layers ACE and TF-GRPO leave untouched."

**Minimal human prior.** "It also keeps the human prior minimal, leaving methodology for the optimizer to discover from rollouts rather than fixing it by hand." The seed harness is nine lines of system prompt and one shell tool.

**Contracts instead of rationales.** Prior self-improving loops justify an edit in prose; AHE requires the edit to name in advance which tasks it will fix and which it may break, and the next round's ground truth checks that claim. This is the closest thing in the literature to scoring an edit on something it did not choose.

### b) Key innovations and their significance

1. **File-level decoupled component substrate across all seven editable layers** — *significant* for us specifically, because it is the first released harness where a proposer's action space spans prompt, tool description, tool implementation, middleware, skill, sub-agent, and memory with git-granular reverts. It is the action space our own harness agent needs, already implemented.
2. **Trace-to-evidence distillation as a navigable file environment** — *notable*. Roughly 10 million tokens to a roughly 10 thousand token overview with drill-down links. Engineering, but it is the engineering that makes any per-edit credit assignment affordable.
3. **The change manifest as a falsifiable contract, with precision and recall measured against it** — *significant*, and the paper's most transferable idea. It is the first instrumented measurement of a self-improving harness's foresight about its own damage, and its result (section 4b below) is the strongest external evidence we have for our claim.
4. **Attribution before distillation, so the verdict enters the evidence corpus** — *incremental* as a design choice, but it is what keeps the contract binding rather than decorative.
5. **Component-level ablation of an evolved harness** — *notable*. Swapping one evolved layer at a time into the seed is the cleanest published measurement of where harness value actually sits, and it produces the non-additivity result.

### c) Applicability

AHE needs three things: a benchmark with a deterministic verifier, a harness whose components are files, and enough compute to re-run the full benchmark every round. The reference run is 10 iterations over 89 tasks at $k = 2$ with a 1-hour per-task timeout, roughly 32 hours wall-clock (Sec. 4.2) on a 96-way concurrent sandbox fleet. It struggles where the benchmark is small (marginal regressions on SWE-bench-verified appear only on the three smallest repositories, Sec. 4.3), where the operating point is coupled to a timeout (Sec. 4.3, Limitations), and on the Hard tier, where component interference caps the gain and it trails the hand-written Codex harness (53.3% versus 56.7%, Table 1).

### d) Comparison table

| System | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| **AHE** | Edits all seven non-weight layers; file-level reverts; every edit carries a falsifiable prediction; frozen harness transfers across benchmarks and model families; released code | Optimises pass@1 on one fixed benchmark; no retention constraint and no deterministic gate; rollback is the proposer's own judgement; regression foresight near chance; component gains do not stack | A retention constraint over previously passed tasks; a learned proposer instead of a prompted one; evaluation on a task stream rather than a repeated batch |
| ACE (in-context playbook) | Cheap, model-agnostic, no framework coupling | Prompt layer only; on SWE-bench-verified it regresses below the seed (74.6% vs 75.2%) while spending 29% more tokens (Table 2) | Move the playbook's content down into tools or memory, which is what AHE's ablation says carries the gain |
| Training-Free GRPO | Reinforces successful tool sequences without weight updates | Prompt-resident; distilled on Terminal-Bench 2 traces, so on a different task surface "that text adds cost without reshaping the underlying policy"; 74.2% on SWE-bench-verified at 21% more tokens than AHE | Same as above |
| Codex-CLI (human-designed) | Best Hard-tier score on the panel (56.7%); no evolution cost | Static; must be re-engineered by hand per base model | This is the labour AHE automates |
| Harness Continual Learning (`refs/related-work/HCL/`) | Hard commit gate with an explicit retention term over an anchor set; the only prior system that refuses an edit for breaking old behaviour | Retention budget is a fixed swept constant, never an estimated quantity; four coarse components; no tools or weights; no public code | Its gate is the piece AHE lacks; AHE's substrate is the piece it lacks |
| Harness-R1 (`refs/related-work/Harness-R1/`) | Trains the proposer with reinforcement learning | One layer, same-batch reward | AHE's action space would give it something worth training over |

---

## 4. Experimental Validation

### a) Experimental design

Three research questions (Sec. 4): where AHE sits against existing harness design (RQ1), whether its product overfits its optimisation target (RQ2), and what inside the loop drives the gain and how reliable the loop's self-attribution is (RQ3).

**Evolution target**: the full 89 tasks of Terminal-Bench 2, officially split 4 easy / 55 medium / 30 hard, per-task timeout extended to 1 hour, $k = 2$ rollouts per task, 10 iterations, roughly 32 hours. **Transfer surfaces**: SWE-bench-verified (500 tasks, 7 repositories) and five alternate base models. **Models**: all four role agents — Code Agent, Agent Debugger, Evolve Agent, Explore Agent — share GPT-5.4; the Code Agent runs at high reasoning, the Evolve and Explore Agents at extra-high (App. Table 4). Sharing one base model across roles is what isolates the gain to harness edits rather than to a stronger analyser or editor (Fig. 1 caption). **Metrics**: pass@1 and mean tokens per trial. **Baselines**: three human-designed harnesses (OpenCode, Terminus-2, Codex) and two self-evolving loops (ACE, Training-Free GRPO) started from the same NexAU-0 seed.

### b) Key results

1. **Ten iterations lift Terminal-Bench 2 pass@1 from 69.7% to 77.0%** (Table 1), past Codex at 71.9%, Training-Free GRPO at 72.3%, and ACE at 68.9% — ACE ends *below* the seed. By difficulty: Easy 87.5% to 100.0%, Medium 78.2% to 88.2%, Hard 51.7% to 53.3%. Hard is the one tier where AHE trails Codex (53.3% vs 56.7%).

2. **The evolution curve is non-monotone and its end is not its peak.** Sec. 4.4.2 attributes "the non-monotone steps in the evolution curve of Sec. 4.2" directly to unforeseen regressions. Concrete text-sourced points: iteration 6 scored 75.8, iteration 7 dropped to 73.0 (App. C.1.4: "By iteration 7 the publish-state guard had been carried over for three rounds, the middleware for two, and the score had regressed from 75.8 to 73.0"), and iteration 8 reached 76.97, "the run's high-water mark". Figure 1 shows iterations 9 and 10 both sitting below iteration 8; the reported 77.0% is the best-so-far line, not the final harness. **This is a published peak-then-decline inside a greedy harness-optimisation loop, in the authors' own headline figure.**

3. **Regression blindness, measured.** The Evolve Agent predicts what it will fix far above chance and what it will break barely above chance (Fig. 4, Sec. 4.4.2): cross-iteration mean fix precision 33.7% against a 6.5% random-prediction baseline and fix recall 51.4% against 10.6% — roughly 5x — versus regression precision 11.8% against 5.6% and regression recall 11.1% against 5.4% — roughly 2x. App. D gives the cumulative form: "across the 9 rounds the agent issued 43 unique regression predictions and only 5 landed, giving cumulative precision 11.6%, while 40 regressions the agent did not foresee actually occurred, giving cumulative recall 11.1%." The authors' own conclusion: "The agent can justify why an edit should help, but it cannot reliably name the tasks the same edit is about to break ... Closing this gap is the clearest direction for future self-evolution loops."

4. **Components interact non-additively, capping the aggregate gain.** Single-layer swaps into the seed (Table 3): memory only 75.3%, tool only 73.0%, middleware only 71.9%, system prompt only 67.4% (below the 69.7% seed), full AHE 77.0%. "The three positive single-component gains sum to +11.1 pp against full AHE's +7.3 pp, and on Hard the memory-only variant exceeds full AHE: memory, middleware, and the system prompt all push toward the same closure-style verification, so stacking them spends turns on redundant re-checks within the long-horizon budget" (Sec. 4.4.1). Memory-only reaches 63.3% on Hard against full AHE's 53.3% — a 10 pp loss from accumulation. Memory-only also drops Easy from the seed's 87.5% to 50.0%, which the paper explains as lessons that "on Easy they reduce to superfluous re-verification".

5. **The frozen harness transfers.** On SWE-bench-verified with no re-evolution (Table 2), AHE takes the top aggregate success at 75.6% versus the seed's 75.2%, ACE's 74.6% and Training-Free GRPO's 74.2%, while cutting mean tokens per trial to 461k — 12% under the seed, 21% under Training-Free GRPO, 32% under ACE. Both prompt-resident baselines regress below the seed while spending 11% to 29% more tokens. Across five alternate bases (Fig. 3) every gain is positive: GPT-5.4 medium +2.3 pp, high +7.3 pp, extra-high +2.3 pp, gemini-3.1-flash-lite +5.1 pp (36.5 to 41.6), deepseek-v4-flash +10.1 pp (51.7 to 61.8), qwen-3.6-plus +6.3 pp (56.2 to 62.5). Cross-family gains exceed within-family ones, which the authors read as weaker bases leaning harder on coordination patterns a stronger base re-derives from its prompt.

6. **The gain lives below the prose layer.** The system prompt swapped in alone scores $-2.3$ pp against the seed; tools, middleware, and memory each carry the improvement on their own. "Factual harness structure transfers across tasks and models whereas prose-level strategy does not" (Sec. 1). Concretely, the evolved tool is "a 1364-line shell that auto-surfaces contract hints from files near each command", the memory is "12 boundary-case lessons", and the system prompt is "79 lines of universal discipline whose executability depends on the other three".

### c) Where it shines

The Medium tier, the largest at 55 tasks, carries the campaign: 78.2% to 88.2%, a +10 pp move, and the paper says the loop "converges to a Medium-heavy trade-off that returns part of the Hard memory effect" precisely because the aggregate is dominated by Medium. Cross-benchmark, the gain concentrates on django and sphinx-doc, "the two largest and most token-expensive repositories whose multi-step edit-and-verify loop matches the structure AHE's tools, middleware, and long-term memory compress from Terminal-Bench 2" (Sec. 4.3). Cross-model, the further a base sits from saturation the more it gains.

### d) Limitations

Acknowledged (Limitations section, Sec. 4.4.1):

- "This work studies a promising but high-variance setting."
- **Benchmark scope**: evolution is driven on one benchmark; "broader programming languages, repository-scale deployments, and human-in-the-loop workflows remain untested."
- **Evolution operating point**: the step budget and per-task timeout were fitted to GPT-5.4 high, so "cross-model transfer numbers conflate harness portability with operating-point coupling"; within one family the gain is non-monotone across reasoning tiers. Untangling this "will require re-running the loop under multiple operating points."
- **Self-modification governance**: "AHE bounds edits to a workspace, attributes every change in a versioned manifest, and rolls back ineffective edits at file granularity, but it does not provide a complete guardrail stack. Long-horizon harness cleanup and stronger misuse prevention remain incomplete, and AHE should be viewed as a controlled research prototype rather than a fully mature autonomous self-improvement system."
- **Interaction-aware evolution** is explicitly left to future work.

Unacknowledged, and load-bearing for us:

- **One campaign, one seed, no seed variance.** Every headline number comes from a single 10-iteration run; the paper reports no repeat run and no confidence interval on the 69.7 to 77.0 move, in a setting it itself calls high-variance and where $k = 2$.
- **The headline 77.0% is an argmax over ten iterations, not the loop's endpoint.** Selecting the best of ten noisy re-evaluations of the same 89 tasks inflates the number by an unreported amount, and the final harness is worse than the reported one.
- **No retention constraint anywhere.** The loop measures a retention rate (the released code computes and prints "previously passed and still passing" each round) and shows it to the Evolve Agent as information; nothing gates on it. A round that trades three Hard tasks for four Medium ones is a round the loop calls progress.
- **No held-out task set during evolution.** Every Terminal-Bench 2 task is visible to the Evolve Agent by name, with per-task failure analyses, for all ten rounds. The transfer study is the only defence against memorisation, and it is run after the fact.
- **The rollback described in prose is an LLM decision in code** (section 2c above). "Rolls back ineffective edits at file granularity" overstates what the released loop guarantees.

---

## 5. Reproduction and Application

### a) Open source?

**Yes**, at https://github.com/china-qijizhifeng/agentic-harness-engineering (MIT; the README's own clone command points at `Curry09/agentic-harness-engineering`, an author mirror). Cloned to `repo/` and analysed in `repo_analysis.md`. What ships: the full outer loop (`evolve.py`), the seed and evolve agent configurations, all five prompt blocks reproduced in App. B, the explore-agent prompts, and — most useful to us — `experiments/evolved_harness/`, the actual harness the ten-iteration campaign produced. What does not ship in full: "The current release ships a *partially* open-sourced Agent Debugger; due to company strategy, it cannot be fully open-sourced at this time" (README). The rollout traces, the per-iteration change manifests, and the change-evaluation files from the reference run are not released; App. C and the figures are the only window onto them.

Key steps to reproduce, in the repo's order: install with `uv sync` (Python 3.13); supply an LLM endpoint, a sandbox key, and a web-search key; build one sandbox template per benchmark task; launch `./scripts/evolve.sh configs/experiments/exp-simple-code-gpt54.yaml`. The loop is resumable from any iteration, restoring the workspace from that iteration's snapshot.

### b) Implementation details requiring attention

- **The Evolve Agent runs at a higher reasoning tier than the Code Agent** (extra-high vs high, App. Table 4) despite sharing GPT-5.4 — the proposer is given more thinking than the thing it edits.
- **Two rollouts per task is the minimum that produces a signal.** "We run $k \ge 2$ rollouts per task so each task carries a pass-rate signal, which stabilizes pass@1 and lets partial-pass tasks anchor comparative diagnosis" (Sec. 3.3). The evolve prompt calls partial-pass tasks "high-priority targets" and instructs the agent to read one passing and one failing rollout and find the divergence point.
- **Infrastructure exceptions count as failures** in pass@1 but are filtered out of the evidence the Evolve Agent sees, labelled "not agent issues, please ignore". Timeouts are *not* filtered — they are handed over as capability failures to analyse.
- **The seed system prompt is non-deletable**; later iterations may only append to it.
- **A best-of-N variant mode exists** (disabled by default): $N$ Evolve Agents run per iteration under complementary strategy constraints — one restricted to structural changes (middleware, tools, sub-agents), one to guidance changes (system prompt, skills, tool descriptions, memory) — all variants are evaluated, the highest pass-rate one is adopted, and the next round sees the cross-variant comparison. Selection is pure argmax on pass@1.
- **Registration is not automatic**: creating a component file is not enough; tools, middleware, skills, and sub-agents must each be registered in `code_agent.yaml`, and a validation script must pass after every round of edits.
- The reference run's per-iteration manifests, traces, and scores: [Information not available in provided text].

### c) Transferability

The substrate transfers to us almost directly. Our setting differs from AHE's in the supply of tasks and in the objective, not in the action space: we need a proposer that edits the same seven layers with the same file-level granularity, and NexAU plus this repo already provides it, with the workspace git history giving free per-edit reverts.

Three concrete adaptations follow. **First**, the loop is one config key away from a task stream: `harbor.dataset` names the benchmark, and replacing "re-run the same 89 tasks" with "run the next batch, then re-check an anchor set" changes what the round measures without touching the substrate. **Second**, the change manifest is the natural carrier for a plasticity signal — it already names at-risk tasks, and the measured 11% regression precision is the empirical statement that a prompted proposer cannot fill that field, which is the case for a trained critic reading the harness files rather than the proposer's introspection. **Third**, the released evolved harness plus the workspace git history is the shape of a harness-evolution log; the campaign that produced it is not released, but the repository shows what such a log looks like and what it would take to collect one.

---

## 6. Summary

### a) One-sentence core idea

Expose every editable harness layer as a file, distil rollout traces into readable evidence, and make each edit a prediction the next round verifies.

### b) Quick-reference pipeline

1. **Start from a deliberately bare harness** — one shell tool, a nine-line system prompt, nothing else — so every later addition has to earn its place.
2. **Run the whole benchmark with the harness as it currently stands**, two attempts per task, each in a fresh sandbox.
3. **Turn the traces into readable evidence**: one analysis report per task naming the root cause, plus one benchmark-wide overview, with the raw traces still reachable for checking.
4. **Check last round's promises.** Each previous edit said which tasks it would fix and which it might break; compare that against what actually flipped, label each edit effective, mixed, ineffective, or harmful, and let the editor decide what to undo.
5. **Edit the harness**, writing only inside the harness files, and record for every change: the failing evidence, the cause, the fix, the tasks it should fix, and the tasks it might break. Commit each change separately. Repeat from step 2.

The loop's only target is the success rate on the benchmark in front of it. It measures how many previously passing tasks still pass and shows that number to the editor, but nothing prevents an edit that breaks them; over ten rounds the score rises, dips, and peaks two rounds before the end.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched the exact title "Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses" and the short form, both generally and restricted to openreview.net, on 2026-09-14). The paper's front page is marked "Preprint" with no venue, and the OpenReview links that surface in searches are all entries in its own bibliography. No review record exists to report.
