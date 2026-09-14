# SIA: Self Improving AI with Harness & Weight Updates

- **Original title**: SIA: Self Improving AI with Harness & Weight Updates
- **Authors**: Prannay Hebbar (equal contribution), Yogendra Manawat (equal contribution), Samuel Verboomen, Alesia Ivanova, Selvam Palanimalai, Kunal Bhatia, Vignesh Baskaran
- **Affiliations**: Hexo Labs Palo Alto (1); Hexo Labs Brussels (2); Hexo Labs Toronto (3); University of Oxford (4)
- **Venue / year**: arXiv preprint, 2026 (arXiv:2605.27276v2 [cs.AI], submitted 26 May 2026, v2 28 May 2026, 15 pages including references)
- **Source URL**: https://arxiv.org/abs/2605.27276
- **OpenReview**: a page exists for a shorter workshop version titled *SIA-W: Self-Improving Agents with Test-Time Weight Updates* — https://openreview.net/forum?id=gZYsloW2bu (ICML 2026 Workshop AI4Research, accepted as poster). See section 7.
- **Code**: public, MIT licensed — https://github.com/hexo-ai/sia (analyzed separately in `repo_analysis.md`)

Short name used below: SIA, the paper's own abbreviation. The paper uses **scaffold** and **harness** interchangeably (Sec. 3.1) for what this project calls the harness minus the model weights.

**Classification: method paper.** The contribution is a built system — a three-agent improvement loop with two update levers — that others can reproduce and that ships as a pip-installable package; the three benchmark studies exist to validate it, not to characterize a phenomenon.

---

## 1. Method Motivation

### a) Why was this method proposed?

The stated driver is that humans are the rate limit on AI progress: "The models are designed and post-trained by researchers, and the agents built on top of them are scaffolded, prompted, debugged, and tuned by engineers" (Abstract). The paper wants a system that, given only a task specification and a verifier, improves both its scaffold and its model weights "without further human intervention" (Sec. 1.1). The narrower technical driver is that the two existing families of self-improvement each freeze exactly what the other changes.

### b) Pain points of existing methods

The paper splits prior work into two silos (Sec. 1.2, Sec. 4) and faults each for the same reason — a frozen half.

1. **Harness / scaffold self-improvement** (Sec. 1.2 Silo 1, Sec. 4.1). Darwin Gödel Machine, Meta-Harness, Hyperagents, automated design of agentic systems, AI Scientist, AutoResearcher. The complaint is empirical: "The recurring empirical observation in this silo is that scaffold edits concentrate on software-engineering hygiene parsing, retries, dispatch and rarely deliver domain-specific reasoning that the base model could not produce given any prompt." This is a ceiling claim about harness-only editing.
2. **Test-time post-training** (Sec. 1.2 Silo 2, Sec. 4.2). TTRL, the Discover line of test-time training, the surprising-effectiveness-of-test-time-training result. The complaint is that the pipeline delivering the gain "is engineered by humans and does not adapt to the task structure that a scaffolded agent would expose."

Table 1 (Sec. 4.4) formalizes the gap on two binary axes, "Edits harness?" and "Edits weights?", across thirteen prior systems. Every listed system is Yes/No or No/Yes; SIA is the only Yes/Yes row, and the paper claims it "is, to our knowledge, the only entry that updates both the scaffold and the weights in a single self-improving loop."

### c) Core hypothesis / intuition

The scaffold and the weights occupy distinct change spaces — the scaffold controls *how* the agent searches and acts, the weights control *what* the model knows — so neither saturates the gain available from the other, and a single loop that can pull either lever beats a loop restricted to one (Sec. 1.3, Sec. 7.1).

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

**Inputs.** A task specification $\mathcal{U}$ (benchmark description plus sample instances), an evaluation dataset $\mathcal{D}$, a deterministic verifier (grader) $V$, and optionally reference implementations $\mathcal{R}$ shipped with the benchmark (Sec. 3.2, Fig. 3).

**Initialization.** A Meta-Agent $\mathcal{M}$ — an LLM call whose output is itself an agent — writes the first scaffold: $A_1 = \mathcal{M}(\mathcal{U}, \mathcal{R})$ (Sec. 3.2). The scaffold is a Python program comprising a system prompt, tool-dispatch logic, answer-extraction code, and any supporting infrastructure; the paper defines it as "the union of the system prompt, tool-dispatch logic, answer extraction, and any supporting infrastructure, every part of the agent that is fixed code rather than model output" (Sec. 3.1).

**Per-generation loop.** Each generation $g$ runs a three-phase protocol (Sec. 3.3):

1. **Execution.** The current scaffold $A_g$ runs on $\mathcal{D}$ inside a sandbox with read-only access to the dataset directory and read/write access to a working directory. The full structured execution log — "every prompt, model response, tool call, tool result, and extracted answer for every task instance" — is captured as the trajectory $\tau_g$.
2. **Analysis.** A Feedback-Agent $\mathcal{F}$ receives $A_g$'s source code, $\tau_g$, the metrics $\mathcal{E}_g$, and optionally a set of sampled task descriptions "used to discourage single-instance overfitting."
3. **Improvement.** $\mathcal{F}$ emits an improvement report (prose analysis plus proposed changes) and the next-generation artifact.

The paper stresses that $\mathcal{F}$ reads the whole trajectory rather than aggregate metrics, so it "can diagnose specific failure modes rather than react to summary statistics" (Sec. 3.3).

**The two levers.** At each step the Feedback-Agent "dynamically selects... between two complementary actions" (Sec. 5.1):

- **Harness update** — scaffold evolution with the weights held fixed. One scaffold-evolution step, with the recurrence $A_{g+1} = \mathcal{F}(A_g, \tau_g(\pi_\theta), \mathcal{E}_g, \mathcal{U})$, where $\tau_g(\pi_\theta)$ denotes trajectories collected by executing $A_g$ with the current policy $\pi_\theta$ (Sec. 5.3). Only $A_g$ changes.
- **Training algorithm update** — a weight update by "an RL method of the Feedback-Agent's choosing, with the scaffold fixed" (Sec. 5.1). Only $\theta$ changes, via Low-Rank Adaptation.

"Harness Update Phase and Weight Update Phase are soft labels for these two action types, not rigid sequential stages" (Sec. 5.1); Fig. 2b illustrates a seven-step interleaved sequence. In practice, across all three reported tasks, "the Feedback-Agent begins with scaffold iteration and switches to weight updates once harness progress stalls" (Sec. 6.2) — so the observed behavior is two sequential phases, not free interleaving.

**Output.** An evolved scaffold plus an RL-adapted LoRA adapter over the base model.

**Termination.** The loop "repeats until the step budget is exhausted" (Fig. 3 caption). There is no acceptance test on an edit, no revert path, and no retention check: the reported operating points SIA-H and SIA-W+H are the best generation observed, not the loop's terminal state (Sec. 6.2; confirmed in the released code, see `repo_analysis.md`).

### b) Architecture components

| Component | Model | Role |
|---|---|---|
| Meta-Agent $\mathcal{M}$ | Claude Sonnet 4.6 (Sec. 5.2) | Writes the initial scaffold $A_1$ from $\mathcal{U}$ and $\mathcal{R}$ |
| Task-Specific Agent $A_g$ | gpt-oss-120b, or an RL-adapted checkpoint of it (Sec. 5.2) | Executes the task against $\mathcal{D}$, produces $\tau_g$ |
| Feedback-Agent $\mathcal{F}$ | Claude Sonnet 4.6 (Sec. 5.2) | Reads $(A_g, \tau_g, \mathcal{E}_g, \mathcal{U})$, picks the lever, writes $A_{g+1}$ or triggers training |
| Verifier $V$ | deterministic code (Sec. 3.1, Table 2) | Computes the per-instance reward; never edited |

The base model is `openai/gpt-oss-120b` throughout; weight updates adapt it via Low-Rank Adaptation (LoRA) at rank $r = 32$ with learning rate $4 \times 10^{-5}$ (Sec. 6.1). Weight updates execute on H100 GPUs through Modal, which "handles rollout generation, reward assignment, and gradient updates within a single managed pipeline" (Sec. 4.3).

**Sample-task regularisation** (Sec. 5.3) is the only generalization safeguard in the method: "The Meta-Agent is conditioned on a diverse set of task specifications during scaffold generation, which mitigates overfitting the initial scaffold to a single benchmark instance." It is a prompt-conditioning device with no measurement, no held-out check, and no ablation anywhere in the paper.

### c) Formulas and algorithms

The selector itself has no formula — it is an LLM call with a frozen prior (Sec. 9). The formulas in the paper are the six RL objectives the Feedback-Agent may write into a training step, each with a stated trigger condition (Sec. 7.3):

- **PPO with Generalized Advantage Estimation.** Observed when "step-level rewards are dense and training stability is the binding constraint." A learned value head $V_\phi$ produces per-token advantages $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$, and a clipped surrogate $\min(r_t \hat{A}_t, \mathrm{clip}(r_t, 1 \pm \varepsilon)\hat{A}_t)$ keeps the policy in a trust region. Used on LawBench.
- **Group Relative Policy Optimization.** Observed when "rollouts are cheap to sample and the verifier fires at episode end." Advantages are normalized inside a rollout group of size $G$: $\hat{A}_i = (r_i - \bar{r}) / \sigma_r$, eliminating the value network. Used on denoising.
- **Entropic advantage weighting.** Observed when "the reward histogram is heavily right-skewed." Rather than zeroing out below-average rollouts, gradient mass is redistributed by softmax with adaptive temperature: $w_i \propto \exp(r_i / \beta)$, with the temperature tuned online "so that the effective sample size stays above a floor threshold, preventing collapse onto a single trajectory." Used on TriMul.
- **REINFORCE with KL-to-base.** Monte Carlo returns $R_t = \sum_{t' \ge t} \gamma^{t'-t} r_{t'}$ serve as advantages directly, with a penalty $\alpha \, \mathrm{KL}(\pi_\theta \| \pi_{\theta_0})$ against the frozen reference policy.
- **Best-of-$N$ behavioural cloning.** A phase-zero cold start when $\mathbb{E}[r] \approx 0$ across all rollouts: the top-$k$ rollouts by verifier score are distilled into the model by cross-entropy.
- **Direct Preference Optimization.** For verifiers that rank but do not score absolutely: $-\log \sigma\!\left(\beta \log \frac{\pi_\theta(y^+)}{\pi_{\theta_0}(y^+)} - \beta \log \frac{\pi_\theta(y^-)}{\pi_{\theta_0}(y^-)}\right)$.

Note that the discount symbol appearing inside the PPO and REINFORCE objectives above is the *within-episode* token discount of standard RL. It is unrelated to any discount over generations of the improvement loop: the improvement loop itself has no discounted objective, no value over future generations, and no notion of a later task.

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Conceptually SIA is the union of two existing loops rather than a new optimization principle. Its one structural novelty is the *selector*: a component that decides, from a trajectory, which of two edit targets to modify. Every prior system in Table 1 has its edit target fixed by construction. Relative to Harness Continual Learning (HCL, arXiv:2608.19013, our nearest existing reference), SIA runs in the opposite direction on every axis except the harness itself: HCL freezes the weights and adds a retention gate over a task stream; SIA unfreezes the weights and drops every gate, running a single task to convergence.

### b) Key innovations and their significance

| Innovation | Significance | Basis |
|---|---|---|
| A single loop that edits scaffold and weights | **notable** — the Yes/Yes cell of Table 1 was genuinely empty, and the two-lever ablation (Table 3) is a clean, if narrow, demonstration that the levers do not substitute | Sec. 5.1, Table 3 |
| Trajectory-conditioned choice of RL algorithm | **incremental as evidence, notable as design** — six objectives with stated triggers, but only three are exercised, one per task, with no controlled comparison of the selection rule against a fixed choice | Sec. 7.3 |
| Evidence that harness-only improvement plateaus | **significant for this project** — three independent plateaus with the switch point explicitly triggered by stalling reward | Sec. 6.3.1–6.3.3 |
| The Feedback-Agent as an untrained selector | **stated as a limitation, not a contribution** — "currently selects between harness and weight updates using a frozen LLM prior" | Sec. 9 |

### c) Applicability

SIA as specified needs three things: a task specification in text, a *deterministic* verifier that scores a single instance, and enough compute to run RL on a 120B-parameter base model. It excels where a verifier is cheap and exact (classification accuracy, kernel runtime, reconstruction error) and where the base model plausibly holds latent domain knowledge that prompting cannot reach. It has nothing to say about tasks without a programmatic verifier, about task streams, or about settings where earlier capability must be preserved — none of which it evaluates. Out of scope by construction: self-editing of the verifier (the paper holds $V$ fixed and names the resulting Goodhart risk in Sec. 8).

### d) Comparison table

| Method | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| SIA | Both levers in one loop; large reported gains where harness-only stalls; a verifier-agnostic interface; open code | Untrained frozen-prior selector; no acceptance gate, no revert, no retention check; single fixed task per run; the loop optimizes directly on the reported test split; no repeated seeds or variance reported anywhere | Train the selector (the paper's own Sec. 9); add a gate on the edit; measure on tasks the loop did not optimize against |
| Harness-only self-improvement (Darwin Gödel Machine, Meta-Harness, Hyperagents) | Cheap, no GPU training, fully inspectable edits | Plateaus; Sec. 1.2 reports edits concentrate on "software-engineering hygiene" | Add the second lever (this is SIA's move) |
| Test-time training / test-time RL (TTRL, Discover) | Reaches knowledge no prompt encodes | Human-written pipeline, blind to the scaffold the agent runs inside | Let the scaffold that collects the rollouts also evolve |
| Harness Continual Learning (2608.19013) | Explicit retention gate over a heterogeneous task stream; measures forgetting | Weights frozen; retention budget is a swept constant, plasticity never estimated | Add weights as an editable layer; estimate rather than fix the budget |

---

## 4. Experimental Validation

### a) Experimental design

Three single-task benchmarks, one run each, "commonly used to evaluate other self-improving AI systems; we run on them specifically to enable direct comparison against prior work" (Sec. 6). Setup from Table 2 (Sec. 6.1):

| Task | Domain | Train / Test | Metric | Prior best | Verifier |
|---|---|---|---|---|---|
| LawBench, 191-class charge prediction | Chinese legal | 5,332 / 913 | top-1 accuracy | 0.450 | held-out test-split grader |
| AlphaEvolve TriMul | low-level GPU | n/a / fixed input shape | score $= 1500/\text{runtime}$, higher is faster | 1.292 | H100 timing |
| MAGIC scRNA-seq denoising | single-cell | n/a / pancreas scRNA-seq | `mse_norm` in $[0,1]$, higher is better | 0.24 | MAGIC reference against ground truth |

The baseline structure is stated explicitly (Sec. 6.2): the initial score is gpt-oss-120b run through the Meta-Agent's first scaffold $A_1$, "generated once from the benchmark specification before any Feedback-Agent iteration begins." Two operating points are then reported: **SIA-H** (best generation using harness updates only) and **SIA-W+H** (best generation with weight updates added on top of the harness-only best). Figures 4–6 add two external comparators, Codex 5.5 and Claude Code with Opus 4.7, run on the same tasks.

Everything the loop optimizes against is the same evaluation dataset every generation. On LawBench, "all evaluations are on the held-out test split" (Sec. 6.3.1) — meaning the split the improvement loop maximizes is the split the headline number reports. There is no separate development set, no repeated seed, and no variance estimate anywhere in the paper.

### b) Key results

**Table 3 (Sec. 7.1)** is the paper's core ablation:

| Task | Initial | Prior best | SIA-H (harness only) | SIA-W+H (harness + weights) |
|---|---|---|---|---|
| LawBench (top-1 accuracy) | 13.5% | 45.0% | 50.0% | **70.1%** |
| AlphaEvolve TriMul (score) | 0.105 | 1.292 | 0.120 | **1.475** |
| Denoising (`mse_norm`) | 0.048 | 0.240 | 0.241 | **0.289** |

1. **SIA-W+H beats SIA-H on all three tasks**: +20.1 percentage points on LawBench, a 91.9% runtime reduction on TriMul (12,483 to 1,017 microseconds), and +20% on denoising (Sec. 7.1). This is the paper's headline and it is the one claim its design actually licenses.
2. **Harness-only improvement plateaus on all three tasks, and the plateau triggers the switch.** LawBench: scaffold generations restructured the pipeline around TF-IDF plus a linear support-vector classifier, "iteratively tuning the character $n$-gram range and regulariser $C$, steadily improving accuracy until gains levelled off at 50.0%, a 36.5 percentage point gain over the initial run. At this point the Feedback-Agent detected stalling reward and switched to weight updates" (Sec. 6.3.1). TriMul: "Incremental scaffold changes (memory layout hints, compilation flags, retry logic) continued to yield smaller gains until the trajectory plateaued" at 1.14x (Sec. 6.3.2). Denoising: "Further scaffold iterations produced no meaningful improvement" past 0.241 (Sec. 6.3.3).
3. **Harness-only improvement failed to reach prior best on two of three tasks.** TriMul SIA-H scores 0.120 against a prior best of 1.292 — roughly a tenth. Denoising SIA-H scores 0.241 against 0.240 — a hair. Only LawBench's harness-only run (50.0% vs 45.0%) clears the prior bar. The abstract's three SOTA-beating numbers all require the weight lever.
4. **On TriMul, SIA-H is beaten by an off-the-shelf coding agent.** Figure 5: Claude Code with Opus 4.7 reaches 1.50x (9,481 microseconds) while SIA-H reaches 1.14x (12,483 microseconds); Codex 5.5 reaches 1.10x. The scaffold-evolution loop underperformed a single strong agent with no self-improvement at all on this task.
5. **Weight updates produced changes the scaffold loop never generated.** On denoising, "the first weight-update checkpoint introduced a structural transformation that the scaffold-only loop, across all harness iterations, never generated: a two-line post-processing step (`np.clip` + `np.rint`) that rounds imputed counts to non-negative integers, enforcing a biological invariant that is trivially correct yet absent from any prior scaffold version" (Sec. 6.3.3). On TriMul the weights internalized "H100-specific design patterns, shared-memory tiling, fp32 register accumulation, block-size selection, that no scaffold edit could encode" (Sec. 6.3.2).

### c) Where it shines

The largest margin is TriMul, where SIA-W+H is 12x the harness-only best. The paper's explanation (Sec. 7.4) is that kernel optimization needs hardware-specific priors that no textual instruction reaches. LawBench shows the cleanest lever separation: the harness lever built a classical machine-learning pipeline (a scaffold-shaped change), the weight lever sharpened 191-way charge disambiguation (a knowledge-shaped change), and neither substituted for the other.

### d) Limitations

**Acknowledged** (Sec. 8), one limitation only, named *coupled co-evolutionary Goodhart*: "Harness search and RL weight updates both optimise against the same fixed verifier $V$. Each pass shapes the distribution the other sees: the harness finds scaffolds that are easy for the current policy to exploit; the weights train on data collected through a scaffold that will subsequently change. The joint fixed point of this coupled system is a Nash equilibrium between two optimisers that are blind to each other's update history, not a point that maximises $V$ on out-of-distribution scaffolds or novel policies." The consequence they name: fixed points "can appear strong on the training verifier while being fragile under any perturbation to either component."

**Unacknowledged:**

- **One run per task, no seeds, no variance.** Every number in Table 3 is a single trajectory. With a best-of-generations reporting rule and no repeats, the margins carry no error bar.
- **The optimization target is the reported metric.** The loop reads test-split performance every generation and edits to raise it. Nothing in the protocol measures performance on anything the loop did not optimize against.
- **Baseline asymmetry.** The Meta-Agent and Feedback-Agent run Claude Sonnet 4.6 while the task-specific agent runs gpt-oss-120b (Sec. 5.2); the "initial" column is therefore a weak model behind a strong model's one-shot scaffold, and the comparators in Figures 4–6 are single agents with no iteration budget stated.
- **The selector is never ablated.** No experiment compares the Feedback-Agent's lever choice against a fixed schedule, even though the observed behavior in all three tasks reduces to the same fixed schedule (harness until stall, then weights).
- **Unverifiable scope claim.** "While this paper reports three tasks, we have been running SIA across a wider set of tasks not included here; the algorithm descriptions below reflect common patterns observed across that broader experimentation" (Sec. 7.3). Three of the six RL objectives are supported only by this unreported experience.
- **No cost accounting.** Neither GPU hours, token spend, nor number of generations per operating point is reported.

---

## 5. Reproduction and Application

### a) Open source?

Yes: https://github.com/hexo-ai/sia, MIT licensed, installable as `pip install 'sia-agent[claude]'`, with four bundled tasks (`gpqa`, `lawbench`, `longcot-chess`, `spaceship-titanic`). Full component-level analysis in `repo_analysis.md`. Two gaps between paper and code matter for reproduction: the lever choice is a launch-time flag (`--focus harness|weights`) fixed for the whole run rather than a per-step Feedback-Agent decision, and the weight-update path delegates to the `tinker-cookbook` library with the Meta-Agent generating a `train.py`, rather than implementing the six-objective menu of Sec. 7.3. Reproducing the LawBench numbers additionally requires a Tinker API key and Modal H100 access.

### b) Implementation details requiring attention

Base model `openai/gpt-oss-120b`; LoRA rank 32; learning rate $4 \times 10^{-5}$ (Sec. 6.1). Meta-Agent and Feedback-Agent both Claude Sonnet 4.6 (Sec. 5.2). Execution is sandboxed with read-only dataset access and read/write working directory (Sec. 3.3). The Feedback-Agent consumes the complete structured trajectory, not summary metrics — on a 913-instance evaluation this is a large context, and the released code truncates it (previews capped at 3,000 characters of agent code, 1,000 per trajectory, 500 per tool result), so what the Feedback-Agent actually diagnoses from is far smaller than the paper's description implies.

### c) Transferability

The interface — task specification plus deterministic verifier — is genuinely generic, and the released framework accepts an external task directory, so pointing the loop at a new task is cheap. What does not transfer is any claim about durability: nothing in the method survives a change of task, because nothing in it is ever measured after the task changes.

---

## 6. Summary

### a) One-sentence core idea

An LLM feedback agent reads an execution trajectory and chooses whether to rewrite the agent's scaffold or retrain its weights, repeating until the budget runs out.

### b) Quick-reference pipeline

1. A meta-agent reads a task description and writes the first version of a task-solving program.
2. That program runs on the evaluation set; every prompt, tool call, and answer is logged, and a deterministic grader scores it.
3. A feedback agent reads the whole log plus the score and picks one of two moves: rewrite the program, or run a reinforcement-learning pass that adjusts the model's weights while the program stays fixed.
4. Repeat for a fixed number of generations; report the best-scoring generation.
5. In every reported run the feedback agent rewrote the program until the score stopped rising, then switched to weight updates for the rest of the budget.

---

## 7. Reviewer Reception (OpenReview)

### a) Link and outcome

The arXiv paper itself has no OpenReview page. A shorter version by six of the same seven authors — *SIA-W: Self-Improving Agents with Test-Time Weight Updates* (Manawat, Hebbar, Verboomen, Palanimalai, Bhatia, Baskaran; Alesia Ivanova is not listed) — appears at https://openreview.net/forum?id=gZYsloW2bu, ICML 2026 Workshop on AI as a Tool for Mathematics, Computer Science, and Machine Learning (AI4Research), submission 26, accepted with venue field "ICML 2026 Workshop AI4Research Poster", preferred presentation poster, presentation type computer demo, posted 2026-06-01. Its `code_or_demo_url` field points to the same GitHub repository.

### b) Main criticisms

No reviews, ratings, or decision rationale are publicly retrievable for the workshop submission: querying the forum through the OpenReview API returns zero notes beyond the submission itself, which is the normal state for a workshop track without public review threads. Searched the OpenReview API by exact title, by paper title, and by author name on 2026-09-14; no reviewer text exists to summarize, and none is invented here.

One substantive public critique exists outside OpenReview. Lilian Weng's survey post on harness engineering (2026-07-04, https://lilianweng.github.io/posts/2026-07-04-harness/) writes: "There are a few confounding choices in SIA's experiments that make the results hard to interpret. For example, the task-specific agent is much weaker than the models used for the Meta-Agent and Feedback-Agent (`gpt-oss-120b` vs `Claude Sonnet 4.6`), and the baselines are too weak to cross-reference cleanly against related methods," concluding: "I would consider the direction interesting, but the evidence provisional."

### c) Author rebuttal and resolution

Not applicable — no public review thread.

One version-to-version discrepancy is worth recording. The workshop abstract reports the weights-over-harness gains as "16 percentage points on LawBench, 19% runtime reduction on GPU kernels, and 19% improvement on denoising," while arXiv v2 reports 20.1 percentage points, 91.9%, and 20% for the same three comparisons (Sec. 7.1). The LawBench and denoising numbers moved modestly; the TriMul figure moved from 19% to 91.9%, a different result rather than a restatement. The paper gives no changelog.

### d) Net takeaway

Peer review supplies no independent check on this paper: the workshop version was accepted to a non-archival poster track with no public reviews, and the arXiv version carrying all the headline numbers was never reviewed at all. The evidence should be weighted as a single-run engineering demonstration with an open-source artifact — the artifact is real and inspectable, the direction is well posed, and the quantitative claims rest on one trajectory per task scored on the split the loop optimized.

---

## 8. Relevance to this project

**Confirms our positioning of SIA, with one correction.** The framing file states that SIA "routes between two layers with an untrained selector and scores each step on its own batch." All three parts hold. *Two layers*: the action space is exactly {scaffold, weights} (Sec. 5.1, Fig. 2a), and the paper's "scaffold" collapses system prompt, tool dispatch, and answer extraction into one undifferentiated object (Sec. 3.1) — so SIA has no depth ordering inside the harness at all, and the two "layers" are better read as "the harness as a whole" versus "the weights." *Untrained selector*: confirmed verbatim in Sec. 9 — "The Feedback-Agent currently selects between harness and weight updates using a frozen LLM prior." *Scores each step on its own batch*: confirmed and stronger than stated — every generation is scored on the same fixed evaluation dataset, which on LawBench is the held-out test split that the headline number reports (Sec. 6.3.1), and no later task ever arrives.

**The correction worth making**: "routes between two layers" slightly overstates what the released system does. In the code the lever is a launch-time flag fixed for the whole run, so the selector the paper describes is not in the artifact (see `repo_analysis.md`).

**SIA is the purest instance of our horizon-zero case.** There is no task stream, no anchor set, no score-before-update protocol, and no gate on an edit. Every edit is scored on the batch it was made on, that batch never changes, and the reported result is the maximum over generations — so a harness trajectory that degrades is simply not reported.

**It supplies the strongest published evidence for a harness-only ceiling, and none for a decline.** Three independent plateaus, each explicitly triggering the lever switch (Sec. 6.3.1–6.3.3), with harness-only failing to reach prior best on two of three tasks (Table 3). But the plateau is where SIA stops looking: because no later task exists, the paper can show the ceiling and cannot show the decline our claim predicts. That is the gap our stream setting is built to measure.

**It validates model weights as the deepest layer.** Our setting lists model weights as the deepest editable layer; SIA is the existence proof that a self-improving loop can reach it, and Sec. 7.4 gives the mechanism split we can cite: harness edits are "externalised" (tools, parsers, retry policies — changing how the agent searches) while weight updates are "internalised" (domain patterns encoded in parameters — changing what the model knows). The denoising case (Sec. 6.3.3) is the sharpest illustration: a two-line invariant the scaffold loop never wrote, arriving through gradients.

**Its stated limitation extends our claim rather than contradicting it.** The coupled co-evolutionary Goodhart argument (Sec. 8) says two greedy optimizers over the same fixed verifier reach a Nash equilibrium that is "fragile under any perturbation to either component." Our framing asserts that greedy patching lowers the future stream performance achievable from a harness; SIA's argument is the within-task analogue, and adds a multi-lever mechanism our current formulation does not cover — a greedy edit at one layer can shrink the room available at another. Whether a plasticity estimate should be defined over the harness alone or over the harness-and-weights pair is a real open question this paper raises.

**Its future work is adjacent to ours and worth tracking.** Sec. 9 proposes to "run SIA across a distribution of tasks, treat each (trajectory, action, outcome) triple as a transition in an outer MDP, and train the selector via RL on that outer MDP." That is a trained proposer over a Markov decision process, which is our territory — but their state is the trajectory, their action space is two levers, their reward is outcome, and there is no discount over a stream and no estimate of what a harness can still become. The distinction we defend is the state-level estimate of future stream performance, not the mere fact of training the selector.
