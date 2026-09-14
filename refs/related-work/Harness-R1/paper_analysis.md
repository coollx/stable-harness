# Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories

- **Original title**: Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories
- **Authors**: Shuai Shao, Kangning Zhang (equal contribution), Qingyao Li, Shijian Wang, Hao Wang, Wenxiang Jiao (corresponding), Yuan Lu (corresponding), Yi Guo, Weiwen Liu (corresponding), Weinan Zhang (corresponding)
- **Affiliations**: Shanghai Jiao Tong University (1); Xiaohongshu Inc. (2); Southeast University (3). First three authors worked on this during an internship at Xiaohongshu Inc.
- **Venue / year**: arXiv preprint, 2026 (arXiv:2608.02276v1 [cs.AI], submitted 3 Aug 2026, 22 pages including appendices A-H)
- **Source URL**: https://arxiv.org/abs/2608.02276
- **OpenReview**: no public page found (see section 7)
- **Code**: https://github.com/DeepExperience/Harness-R1 (Apache-2.0; see `repo_analysis.md`)
- **Checkpoints**: https://huggingface.co/ShaoShuai0605/Harness-R1 (two Qwen3.5-9B harness engineers)

Short names used below: the trained editor model is the paper's **harness engineer**; the agent being edited is the **target**. The paper's own abbreviation for the method is Harness-R1.

---

## 1. Method Motivation

### a) Why was this method proposed?

A deployed agent continuously produces trajectories that expose systematic failures — tool misuse, lost state, protocol violations, repeated attempts, failed recovery — and the paper asks whether that experience can improve the agent continually after deployment (Sec. 1). It separates two places where improvement can land: model parameters, or the harness around a frozen model. It then makes the empirical observation that motivates everything else: direct harness modification by a fixed editor is not reliable. Figure 1 measures matched-baseline reward change across the three benchmarks and finds a hand-designed Self-Refine rule *lowers* reward on all three (−3.3 average) while six frontier models prompted as harness editors produce changes ranging from +0.1 (Kimi-K2.6) to +3.7 (GLM-5.2), with several reducing WebShop reward. The driving force is therefore not "can a harness be edited" but "can an editor be made trustworthy", and the paper's answer is to stop prompting an editor and start training one.

### b) Pain points of existing methods

Three distinct gaps, one per related-work subsection (Sec. 2):

1. **Against LLM-based harness evolution** (Meta-Harness, AutoHarness, Agentic Harness Engineering, Life-Harness, HarnessX). These are credited with searching or synthesizing whole harnesses and with turning recurring failures into scoped, regression-checked repairs, then faulted on one axis: "the harness proposer usually remains fixed: outcomes select or iteratively refine patches without directly updating proposer parameters". Where a system does train something from harness-editing outcomes, it trains the *task* model, not the editor — the paper names HarnessX's cross-harness GRPO as exactly this case.
2. **Against algorithmic optimization of harness components** (APE, OPRO, ProTeGi, TextGrad, EvoPrompt, Promptbreeder, GEPA, DSPy, MIPRO). These search over instruction candidates, propagate textual gradients, or compile pipelines, and "they generally do not post-train the proposer, reflector, or backward engine from editing outcomes; even when some fine-tune task modules, the product is a task-specific artifact or parameter rather than a dedicated harness-editor policy trained by editing outcomes".
3. **Against the few learned editors that exist**. Each "either restricts the edit space or couples editing with task solving" — training an editor over a single narrow target such as one context field or a revisable skill bank, generating workflows for a frozen executor, selecting among a few predefined structural actions under offline RL, or folding harness edits into one task actor's action space.

The paper also names two concrete obstacles that explain why nobody had done this: the editable runtime spans interdependent execution stages, so unrestricted code edits break interfaces or produce non-executable behavior; and text form and static rules cannot tell whether a patch is good, because "only the target agent's behavior after applying the patch can do so" (Sec. 1).

### c) Core hypothesis / intuition

If a dedicated editor is post-trained on the realized task outcome that each of its executable patches produces when the frozen target actually reruns the tasks, rather than on whether the patch looks reasonable, then failure-conditioned harness editing becomes a learned, transferable capability that a 9B editor can hold and a much larger prompted editor cannot.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

The loop has five stages and closes on itself once per update; Figure 2 is the diagram, Sec. 3.1 the setup, Algorithm 1 the training loop.

**Setting** (Sec. 3.1). $A$ is a frozen target agent — its model *plus* its surrounding base runtime. $B = \{x_i\}_{i=1}^{n}$ is a batch of $n$ tasks in environment $E$. Within an episode the target reads accumulated history plus the current observation and its frozen policy proposes an action in the environment's native interface (`search` and `click` in WebShop, navigation and object manipulation in ALFWorld, SQL in DBBench). The adapted component is explicitly *not* the weights: "The component we adapt is the base runtime: the code that assembles the context shown to the model, forwards each action to the environment, and relays the feedback back to the agent."

1. **Baseline rollout.** The unmodified target runs the batch, yielding trajectories $\tau_i^0$ and rewards $R_i^0$.
2. **Failure packet.** A deterministic extractor retains only failed episodes and compacts them into a failure packet $s_B$ holding task constraints, selected action-observation excerpts, outcomes, and necessary environment state. The engineer reads this packet **once**.
3. **Patch generation.** The engineer $H_\theta$ emits a `<think>` analysis followed by exactly one `<patch>` JSON object (Fig. A.3). The patch is a batch-conditioned executable overlay $P$. The engineer "neither answers the tasks nor participates in their rollouts" and never calls an environment tool (Appendix C).
4. **Validation and installation.** The patch is parsed, validated, and compiled; invalid patches have no effect. The overlay wraps the runtime as executable hooks at four lifecycle points, leaving weights untouched.
5. **Rerun and reward.** The frozen target reruns **every task in $B$, including tasks that originally succeeded**, producing $R_i^P$. The reward is the full-batch mean reward change; only $\theta$ is updated.

**The four lifecycle hooks** (Sec. 3.1, Table C.1). Each receives benchmark-specific runtime context and returns only an effect the host runtime interprets; the host, never the hook, executes anything.

| Hook | Lifecycle position | Invocation and permitted effect |
|---|---|---|
| `on_init` | Episode initialization | Before the first target decision; adds reusable task guidance or a tool hint (returns `skills` and/or `tool_hint`). |
| `make_pre_hint` | Pre-decision | Before a target decision; injects a state-conditioned `message` without executing an action. |
| `on_before_action` | Pre-action | After the target proposes an action but before environment execution; may `block_and_prompt`, or where supported `rewrite_action` or `force_action`. |
| `on_post_step` | Post-feedback | After environment feedback; may `inject_hint` or, where supported, `force_action` to schedule a next action. |

`block_and_prompt` suppresses the pending action and asks the frozen target to choose again; `rewrite_action` replaces it; `force_action` selects a concrete action for the current or next step. DBBench v1 accepts soft guidance and blocking but executes neither SQL rewrites nor forced commits (Appendix C) — a live asymmetry in the action space across environments.

### b) Architecture components

Three model roles and one non-model component.

- **Harness engineer $H_\theta$** — a separate Qwen3.5-9B, the only trainable object. Initialized by cold-start supervised fine-tuning, then trained by online GRPO.
- **Frozen target $A$** — Qwen3.5-9B in the main results, frozen within a stage; twenty further targets used only at test time (Table E.1).
- **Reference policy** — the frozen pre-update engineer used by GRPO's importance ratio.
- **The overlay $P$** — not a model: a JSON object holding up to twelve actions, each an `add_code_hook` carrying a Python `hook(ctx, nb)` body. `ctx` is per-call runtime context, `nb` a per-episode scratch dictionary the hooks use to carry state across steps. The host runtime honors only the hook's *return value*, never its side effects.

### c) Formulas and algorithms

**Engineer reward** (eq. 1, Sec. 3.1). The full-batch performance difference and the reward:

$$\Delta_B(P) = \frac{1}{n}\sum_{i=1}^{n}\left(R_i^P - R_i^0\right), \qquad r(B, P) = \begin{cases} \Delta_B(P), & \text{if valid and complete,} \\ 0, & \text{otherwise.} \end{cases}$$

In plain language: install the patch, rerun the *same* $n$ tasks with the *same* frozen target, and take the average change in task reward. The paper itself flags what this is: "Using the same tasks before and after controls task composition but defines a same-batch, transductive objective, with no iterative refinement within an instance or persistent patch memory across batches." A patch that rescues one failure and breaks one prior success scores exactly zero; a regression yields a negative reward, but an *invalid* patch yields zero, so invalidity is scored strictly better than damage.

**Cold-start supervised fine-tuning** (eq. 2, Sec. 3.2). A strong teacher (GPT-5.5) proposes serialized editing responses $y_j^T$ from failure packets $s_j$; candidates are validated on the frozen target and at most one executable, complete, non-regressive response is retained per packet. The engineer is then trained by teacher-forced next-token prediction over $\mathcal{D}_{\text{SFT}} = \{(s_j, y_j^T)\}_{j=1}^{M}$:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{\sum_{j=1}^{M}|y_j^T|}\sum_{j=1}^{M}\sum_{t=1}^{|y_j^T|}\log H_\theta\!\left(y_{j,t}^T \mid s_j, y_{j,<t}^T\right).$$

Teacher and RL instances use **disjoint task batches**.

**Group-relative advantage** (eq. 3, Sec. 3.2). For each failure packet, $K = 8$ candidate patches are sampled from the current policy, each independently installed and scored by rerunning the frozen target on the same full batch. With $\mu_B$ and $\sigma_B$ the empirical mean and standard deviation of $r_k = r(B, P_k)$ within the eight candidates:

$$\widehat{A}_k = \frac{r_k - \mu_B}{\sigma_B}.$$

**Clipped surrogate** (eq. 4, Sec. 3.2). With $\rho_{k,t}(\theta) = H_\theta(y_{k,t} \mid s_B, y_{k,<t}) / H_{\theta_{\text{old}}}(y_{k,t} \mid s_B, y_{k,<t})$ and the sequence-level advantage shared by all response tokens:

$$g_{k,t}(\theta) = \min\!\left\{\rho_{k,t}\widehat{A}_k,\ \text{clip}(\rho_{k,t}, 1-\epsilon_\ell, 1+\epsilon_h)\widehat{A}_k\right\}, \qquad \mathcal{J}(\theta) = \mathbb{E}\!\left[\frac{1}{K}\sum_{k=1}^{K}\frac{1}{T_k}\sum_{t=1}^{T_k} w_{k,t}\,g_{k,t}(\theta)\right].$$

$w_{k,t}$ is a truncated importance weight correcting the mismatch between the training engine's recomputed log probabilities and the rollout engine's recorded ones. The paper states explicitly: "no format-validity bonus or explicit KL loss is added. Only the engineer parameters $\theta$ are updated". Clipping bounds are $\epsilon_\ell = 0.20$, $\epsilon_h = 0.28$ (Appendix B.2).

**Algorithm 1** (Sec. 3.2). Base trajectories, rewards, and failure packets are cached before optimization; online evaluation reruns only the *patched* target for candidates sampled from the current engineer. Per update bundle: sample $K$ responses per packet, parse and validate each into $P_k$, independently install and rerun, set $r_k = 0$ for invalid/no-op/incomplete evaluations, normalize into $\widehat{A}_k$, update $\theta$. Iterate until the training budget is exhausted.

**What is deliberately absent from the objective.** There is no retention term, no anchor set, no held-out term, no edit-cost or inference-cost term, no size penalty in the reward, and no discount over future batches. The only bloat control is structural: at most 12 actions per patch, 8,000 characters and 1,200 AST nodes per hook, 5 helper functions, 700-character string literals, 900-character return text (repo, `code_runner.py`).

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

The learning target moves. Every neighbouring system treats the *harness* as the thing being optimized and the proposer as a fixed instrument — search, selection, iterative refinement, or a validity gate decides which candidate harness survives. Harness-R1 keeps the harness disposable and makes the *editing policy* the learned object: "Harness-R1 instead moves the learning target from the resulting harness to the editing policy" (Sec. 2). Everything else follows from that inversion — the reward must be realized rather than judged, the target must be frozen so the credit is unambiguous, and the payoff is a policy that transfers to targets it never trained on.

Against this project's setting, three differences matter and one of them cuts the other way from how it first reads.

The reward is computed on the same batch the failures were mined from, after the edit. In the vocabulary of `docs/framing.md` this is the horizon gamma set to zero — an edit is scored by the batch it was made on and nothing after — and it is additionally the *opposite* of score-before-update: the tasks used to score the harness are re-solved by the harness that was built to solve them. The paper is candid about this and names our objective as its own future work: "Our reward is also computed from same-batch task outcomes, which keeps training grounded but ties the signal to the tasks used to mine failures. Future work can enrich this reward with held-out performance, so that edits are explicitly optimized against regressions on unseen tasks" (Sec. 5).

There is no task stream and no harness state that persists. Each failure packet produces a fresh overlay installed on the *base* runtime; patches never compose, never accumulate, and never carry forward — the paper says so ("no persistent patch memory across batches", Sec. 3.1), and the repo confirms it, since the only cross-invocation patch plumbing is an evaluation convenience for generating on one machine and rerunning on another. There is therefore no harness that could lose plasticity, because nothing compounds. Harness-R1 is a one-shot patch generator measured twice, not a continual system.

The edit surface is one layer in our depth ordering, and the paper's own "lifecycle-wide" claim is on a different axis. The four hooks are all executable middleware wrapping a frozen policy; no prompt text, tool description, tool implementation, skill file, sub-agent configuration, memory store, or weight is edited as an artifact. What varies across the four positions is *when in the episode* the middleware runs — exactly the episode-time axis the Terms table in `docs/framing.md` distinguishes from layer depth. The one real nuance: `on_init` returns a `skills` list and a `tool_hint`, so middleware code synthesizes content that lands in other layers at runtime, and the released patch protocol defines five further action types (`set_config`, `edit_tool_hint`, `add_or_edit_skill`, `add_guard_rule`, `add_recovery_rule`) that were disabled for training and evaluation by `harness_r1_require_code_hook_only: true`. The reported system edits one layer; the released protocol could edit more.

### b) Key innovations and their significance

1. **Post-training an editor on realized rerun outcomes** (**significant**). This is the paper's actual claim to novelty and it holds up against the related work it surveys: no cited system updates proposer parameters from the measured effect of installing the proposal. The evidence that this matters is not the headline number but the comparison at fixed evidence — 9B trained beats 397B prompted by 4.8 points (53.6 vs 48.8, Table 1) and is the difference between +8.9 and −4.3 on held-out tasks (Table F.1).
2. **Non-differentiable outcome reward with no judge and no validity bonus** (**notable**). Every valid patch is compiled, installed, and scored by an actual rerun; `harness_r1_valid_bonus: 0.0` in the released config. This removes the usual failure mode where a format bonus teaches the editor to emit well-formed inert patches.
3. **A validated, sandboxed executable action space over four lifecycle points** (**notable**). The AST validator rejecting imports, I/O, dynamic evaluation, and hard-coded task answers is what makes "let a model write code into the runtime" trainable at all; the ALFWorld regex forbidding numbered instance actions is a direct anti-leakage measure.
4. **Evidence that editor and target co-evolve** (**notable, weakly supported**). One alternation, not a sequence: after target SFT lifts the actor 44.3 → 59.2, a retrained engineer adds a further 5.0 points. The paper correctly calls multi-round alternation future work.

### c) Applicability

It excels where failures are *recurring and observable in the runtime context* — a premature purchase with a required option unselected, an omitted transformation before placement, a schema error before a mutation. It is weakest wherever the failure is a wrong high-level choice made earlier: the paper's own WebShop case notes the guard "cannot repair an earlier choice of the wrong product" (Appendix H.5). It requires a computable per-task reward and an environment cheap enough to rerun a full batch eight times per update, which is the method's dominant cost. It has nothing to say about settings where the harness must persist and grow.

### d) Comparison table

| Approach | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| **Harness-R1** | Editor trained on realized effect; transfers to 20 unseen targets without retuning (+7.06 pp); positive on 1,270 held-out tasks (+8.9); executable multi-position edits; no judge | Same-batch transductive reward; no retention, cost, or size term in the objective; one adaptation round; patches never accumulate; requires a full-batch rerun per candidate (K=8) | Held-out or future-batch term in the reward; multi-round alternation with the target; an explicit regression constraint at commit |
| Prompted frontier editors (GLM-5.2, GPT-5.5, Qwen3.5-397B, DeepSeek-V4-Pro, Kimi-K2.6, Gemini-3.5-Flash) | No training cost; immediate; strong base models | Optimize for plausibility, never observe the rerun; gains unstable and sometimes negative (Fig. 1; Gemini-3.5-Flash −6.2 pp on ALFWorld, Appendix H.3); held-out deltas straddle zero across seeds | Give them the rerun signal — at which point they become an expensive Harness-R1 without the training |
| Fixed harness patterns (ReAct, Self-Refine, Reflection) | Zero infrastructure; well understood | One hand-crafted rule applied uniformly regardless of the target's failure modes; Self-Refine costs 2.5 points on average | Condition the rule on observed failures, which is the paper's thesis |
| Fixed-proposer harness evolution (Meta-Harness, AutoHarness, Life-Harness, HarnessX, Harness Continual Learning) | Can accumulate a persistent harness across a stream; some carry explicit retention gates | Proposer never improves from editing outcomes; per-edit gating cannot anticipate later tasks | Train the proposer (this paper) while keeping the persistent state and the gate (they do not) |
| Supervised-only engineer (this paper's own ablation) | Cheap; 877 examples; already beats most frontier editors on WebShop | 46.4 average, 7.1 points below the RL engineer; imitates a teacher's plausible edits, never measures them | This gap *is* the paper's core evidence |

---

## 4. Experimental Validation

### a) Experimental design

**Benchmarks** (Sec. 4.1). WebShop (grounded web navigation, 500 test tasks, fixed indices 0-499, goal seed 233, shaped continuous reward plus success), ALFWorld (text-based embodied household tasks, 500 test tasks spanning six families, binary success, additionally micro-averaged as "All"), DBBench from AgentBench (relational database tasks, 300 test tasks, binary success). Averages are the equal-weight mean of WebShop success, ALFWorld All, and DBBench success. Task splits are drawn before any trajectory collection: 17,843 training tasks, 299 validation, 1,300 test (Table D.1), with SFT and RL partitions disjoint.

**Comparisons.** The unmodified Qwen3.5-9B target against four groups: fixed prompt-based strategies (ReAct, Self-Refine, Reflection); six frontier models prompted as harness engineers; a supervised-only engineer; and Harness-R1. Reflection is reported under a separate two-episode $\text{success@2}$ protocol and is explicitly not ranked against single-episode rows. Beyond the primary target, twenty further target models spanning Llama, Gemma, and Qwen families at 1B-397B are used only at test time.

**Training** (Appendix B). Engineer: Qwen3.5-9B, full-parameter cold-start SFT on 877 teacher-filtered examples (381 WebShop, 248 ALFWorld, 248 DBBench), 2 epochs, LR $10^{-5}$, context 32,768, then online GRPO on roughly 1,500 failure packets with $K = 8$, rollout batch 4 prompts, global batch 32, LR $10^{-6}$ constant, temperature 0.7, top-$p$ 0.95, max prompt/response 28,672/12,288, truncated importance sampling on, entropy and KL coefficients both 0. Everything runs on one node of 8 NVIDIA H800 GPUs. Checkpoints are selected on aggregate development performance.

### b) Key results

1. **The outcome-trained engineer improves the frozen target on all three benchmarks** (Table 1). Average 44.3 → 53.6, +9.3 points. The largest single gain is ALFWorld, 40.6 → 53.2; WebShop success 31.2 → 42.2; DBBench 61.0 → 65.3.
2. **Training on realized outcomes is what does the work, not the editing format** (Table 1). The supervised-only engineer, trained on exactly the same patch format from a GPT-5.5 teacher, reaches 46.4 — the online RL stage adds **7.1 points** on top of it. The strongest frontier editor, GLM-5.2, reaches 48.8, so a 9B trained editor beats a far larger prompted one by 4.8 points.
3. **The gain survives improving the actor** (Table 1). Direct target SFT lifts the unmodified target to 59.2; a Harness-R1 engineer retrained against that stronger frozen target reaches 64.2, +5.0 points. ALFWorld All goes 71.2 → 84.0. The paper notes honestly that "a few individual metrics dip slightly" — WebShop shaped score falls 71.5 → 68.7 while WebShop success rises 42.6 → 43.0.
4. **The editing policy transfers to unseen targets without retuning** (Fig. 3, Table E.1). Across the twenty unseen targets the benchmark-averaged gain is **+7.06 points**; every target-level average is positive; of 63 target-benchmark combinations 56 improve, 4 are unchanged, and the 3 regressions are all $\le 2.0$ points (Llama-3.1-70B on WebShop −0.4, Qwen2.5-72B on ALFWorld −2.0, Gemma-4-31B-it on WebShop −0.2). Each target supplies its own failure traces and receives a newly generated patch, so this measures transfer of the policy, not replay of a fixed patch.
5. **Sparse evidence yields patches that help unseen tasks** (Fig. 4a, Table F.1). From the same ten sampled failures per benchmark, each editor writes one benchmark-level patch applied to all remaining tasks; pooled over 1,270 held-out tasks and three seeds, Harness-R1 gains **+8.9 ± 1.5 points** and is positive on every seed, with 9/9 patches valid. Qwen3.5-397B averages **−4.3 ± 2.5** (8/9 valid) and DeepSeek-V4-Pro **−0.4 ± 3.6** (6/9 valid). The frontier spreads straddle zero — the failure is not weak gains but inconsistency between marginal gains and sizable regressions.
6. **Action mediation and recovery carry the improvement, and which one is environment-dependent** (Fig. 4b, Table G.1). Holding the frozen target and generated patches fixed and disabling one lifecycle position at a time: full patch 53.1, no intervention 44.2. Removing pre-action costs 3.9 points (49.2), post-feedback 3.3 (49.8), episode initialization 0.9 (52.2), pre-decision 0.6 (52.5). Per benchmark, pre-action dominates WebShop (41.6 → 31.5) and post-feedback dominates ALFWorld (52.1 → 41.9). The paper warns these are conditional and "should not be summed into a universal importance ranking".

### c) Where does it shine?

Gains are largest on weak targets and on ALFWorld. In Table E.1 the biggest per-target averages are Qwen3-4B +12.3, Gemma-3-4B +12.0, Llama-3.1-8B +10.5, Gemma-3-12B +10.4, Qwen3-14B +10.0 — small models that make many mediatable execution mistakes. Gemma-3-1B gains only +1.2 and Llama-3.2-1B +2.2, so there is a floor below which there is no correct behavior to guard. Appendix H.2 shows the qualitative mechanism at its best: an ALFWorld patch coordinating all four positions — episode-init stage ordering, post-feedback state update, pre-decision subgoal hint, pre-action placement guard — records 56 stage hints or guard messages across a ten-task batch and lifts it 1/10 → 6/10.

### d) Limitations

**Acknowledged** (Sec. 5, Appendix C, Appendix H).

- The reward is same-batch, tying the training signal to the very tasks used to mine failures; the authors name held-out-enriched reward as future work.
- Only a single adaptation round from a vanilla target to a fine-tuned one is studied; whether multi-round alternation converges or compounds is open.
- No inference-efficiency or cost term, so nothing prevents useful patches from being expensive.
- Outcome-grounded post-training "increases useful executable edits without guaranteeing complete or regression-free rules" (Appendix H.5).
- The action space is unequal across environments — DBBench v1 executes neither SQL rewrites nor forced commits.

**Unacknowledged or implicit, and directly relevant to this project.**

- **Regressions are visible, unpenalized beyond the net, and traded one-for-one.** Appendix H.2: "the patch rescues six baseline failures but regresses one baseline success, producing a net change from 1/10 to 6/10". Eq. 1 prices that broken success at exactly the cost of a rescued failure. The reward is structurally incapable of preferring an edit that keeps both.
- **An invalid patch scores better than a harmful one.** Invalid, no-op, and incomplete evaluations all receive $r = 0$, which equals the no-change reward, while a regressive patch goes negative. Under group normalization this makes "emit garbage" a safe fallback whenever the alternative is risky.
- **Held-out transfer is measured once, from a single edit, on a non-compounding harness.** The +8.9 held-out number is one patch applied to one split. It is not evidence about a stream, and the paper never claims it is — but it is the closest thing in the literature to a counterweight to the claim that same-batch optimization damages later performance, and it should be read as such.
- **Every reported engineer is target-specific.** The README is explicit: "An engineer is only meaningful against the target it was trained for." The 20-target transfer result is generated patches from one trained policy, not one engineer serving heterogeneous targets in a single deployment.
- **Two seeds only where it matters most.** Main results report single-configuration numbers; only the held-out experiment carries three seeds and an error bar, and only the lifecycle ablation reruns three times per benchmark. The 9.3-point headline has no stated variance.
- **Benchmark-level, not stream-level, non-stationarity.** Rollout groups never mix benchmarks (README, "group records without mixing benchmarks inside a rollout group"), so the engineer never has to write one edit that serves a shifting task distribution.

---

## 5. Reproduction and Application

### a) Open source and key reproduction steps

Code is at https://github.com/DeepExperience/Harness-R1 under Apache-2.0, and both engineer checkpoints (`harness-r1` and `agent-sft-harness-r1`, Qwen3.5-9B) are released at https://huggingface.co/ShaoShuai0605/Harness-R1. What ships: the full patch protocol and AST sandbox, the three per-benchmark same-batch rerun rewards plus a dispatcher, the RL and SFT configs, dataset builders, three evaluation wrappers, an offline demo, protocol tests, a trimmed snapshot of the Relax RL framework, and the AgentBench-derived task runtimes. What does not ship: benchmark assets, target-agent weights, and training data. Detailed component-by-component notes are in `repo_analysis.md`.

Reproduction is a three-stage sequence: (1) cold-start SFT the engineer with `scripts/train_engineer_sft.sh`; (2) freeze a target, run no-harness trajectories on the training split, and build grouped failure-packet RL rows carrying immutable baseline metadata; (3) online GRPO with `scripts/train_engineer_rl.sh` pointing at `configs/rl/mixed_codepatch.yaml` and a live frozen-target endpoint. Evaluation is fixed-batch run-patch-rerun, optionally split into `GENERATE_ONLY=1` then `PATCH_SOURCE_ROOT=...` so patch generation and target rerun happen on different machines — the correct protocol for cross-target work.

### b) Implementation details requiring attention

The full hyperparameter tables are in Appendix B; the traps are in the README and are worth more than the tables.

- **Structured tool calls are load-bearing.** "The target server must return structured `message.tool_calls`. XML-looking tool text inside `message.content` is not equivalent and silently zeroes rewards." The repo ships `scripts/probe_openai_tool_calls.py` specifically to check this before a long run.
- **WebShop pairing is identity-checked, not index-checked.** Results are reportable only when `webshop_goal_seed=233` and the strict `webshop_batch_identity_v1` task manifests agree between baseline and patched rerun; "matching integer task indices are not proof of a paired comparison". The reward function raises rather than silently scoring if the configured seed disagrees with the baseline identity.
- **Static protocol drift destroys format validity.** SFT, RL, and evaluation must agree on the exact system prompt, static user prefix, schema, `schema_style`, and the `prefill_think_patch` response protocol. Baseline rewards from a different target model or serving protocol must never be reused.
- **Reward-metric and bonus choices.** The released config uses `harness_r1_reward_metric: delta_average_reward` (continuous) with `harness_r1_valid_bonus: 0.0`; `delta_pass_rate` is the discrete alternative. `harness_r1_reject_runtime_noop_patch: true` and `harness_r1_require_code_hook_only: true` are both set, the latter disabling the protocol's five non-code action types.
- **Failure and retry discipline.** An invalid patch is scored as no-patch with zero delta; an environment failure is a *missing* evaluation to be reported separately and "never retried until it turns positive".
- **Hook budgets.** 0.05 s wall-clock and 2,000 executed lines per hook call, enforced by `signal.setitimer` and a line-counting trace function; any hook exception or timeout degrades to no effect so a bad hook cannot crash an episode.

Not stated: total GPU-hours, the number of RL update bundles actually run, and any variance estimate for the main table. `NUM_ROLLOUT=174` appears in the README's example invocation but is not identified as the reference value. [Information not available in provided text.]

### c) Transferability

The recipe needs four things: a frozen target served behind an endpoint, a computable per-task reward, an environment whose full batch can be rerun eight times per update, and a host runtime with defined interception points. Any verifiable agentic environment satisfies this. The parts that transfer to this project most directly are the ones that are not the contribution: the executable-hook action space with its return-value contract, the AST sandbox, the same-batch rerun harness, and the identity protocol that makes a paired before-after comparison auditable.

Three adaptations follow from what the paper leaves open, and the first is the project's own claim. Replace $\Delta_B(P)$ with a horizon-discounted return over later batches, or add a term estimated by a state-level critic reading the harness files — the paper's limitations section asks for precisely the first half of this. Give the overlay persistence, so patches compose into a harness state that can accumulate and can therefore lose plasticity; as written there is nothing to retain because nothing survives the batch. And restore the five disabled action types, which would turn a one-layer editor into a cross-layer one and make moves and promotions expressible.

Two released artifacts are usable as-is. `examples/heldout_generalization/` holds 27 patches from three editors on identical failure evidence, each with the held-out delta it earned — a small labelled set of (harness edit, realized outcome) pairs across an editor-quality gradient, which is the shape of data a state-level critic would need, at a scale far below what training one requires. And the two released engineer checkpoints are a ready-made greedy proposer to run as the zero-horizon arm of a comparison rather than reimplementing one.

---

## 6. Summary

### a) One-sentence core idea

Post-train a dedicated 9B editor with online reinforcement learning so its executable runtime patches are optimized for the task success they actually produce when a frozen agent reruns the same batch.

### b) Quick-reference pipeline

1. **Run the agent on a batch of tasks and keep only the failures.** A deterministic extractor compacts them into one packet of constraints, action-observation excerpts, outcomes, and environment state.
2. **Ask the editor model for one patch.** The editor reads the packet once and writes short Python functions that hook into four points of the agent's execution: episode start, before each decision, before each action reaches the environment, and after each piece of environment feedback. The hooks may add guidance, block an action and ask again, rewrite or force an action, or track state across steps. They may not import anything, touch files, or hard-code answers.
3. **Install the patch and rerun the exact same tasks with the exact same frozen agent.** Reward = average task score after minus average task score before. Invalid, inert, or incomplete patches get zero.
4. **Sample eight patches per failure packet and compare them against each other.** Normalize the eight rewards into advantages and update only the editor's weights. The agent being edited never changes.
5. **Repeat until the training budget runs out.** The result is an editor policy, not a harness: point it at a different agent and it writes that agent a different patch.

The single load-bearing design choice is step 3 — that the score comes from a real rerun rather than a judge. The single unexamined one is that the rerun uses the same tasks the failures came from, and that the patch is discarded afterward.

### c) Bottom line for decision-making

Trust the central comparison — a 9B editor trained on realized outcomes beats far larger prompted editors, and the supervised-only ablation isolates why — and treat the transfer numbers as real but single-shot. Do not read anything in this paper as evidence about a task stream: nothing accumulates, nothing is retained, and the paper measures a fresh edit on a fresh runtime every time. For this project it is the strongest available instance of the zero-horizon case and the right greedy baseline to run, and its own limitations section asks for the objective we are claiming.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched the exact title "Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories", the short form "Harness-R1", and "Executable Runtime Harnesses" via web search and the OpenReview v1 and v2 search APIs on 2026-09-14; the v2 API returns zero notes for "Harness-R1"). The paper is an arXiv preprint with no venue stated on its front page and no comments field, so no review record exists to report. No reviews are summarized here because none are public.
