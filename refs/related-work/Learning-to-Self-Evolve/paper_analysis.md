# Learning to Self-Evolve

- **Original title**: Learning to Self-Evolve
- **Authors**: Xiaoyin Chen (work done during an internship at Snowflake), Canwen Xu (corresponding), Yite Wang, Boyi Liu, Zhewei Yao, Yuxiong He
- **Affiliations**: Mila – Quebec AI Institute (1); University of Montreal (2); Snowflake (3)
- **Venue / year**: ICLR 2026 Workshop on Lifelong Agents: Learning, Aligning, Evolving — accepted as poster; also arXiv:2603.18620v1 [cs.CL], submitted 19 Mar 2026, 17 pages including appendices A–B
- **Source URL**: https://arxiv.org/abs/2603.18620
- **OpenReview**: https://openreview.net/forum?id=zedEdPhmsA (venue field reads "LLA 2026 Poster"; see section 7 for what could and could not be retrieved)
- **Code**: https://github.com/chenyn66/learning-to-self-evolve — cloned to `repo/`, see `repo_analysis.md`

Short name used below: Learning to Self-Evolve (LSE), the paper's own abbreviation.

---

## 1. Method Motivation

### a) Why was this method proposed?

Post-training already uses reinforcement learning to improve a model on its own generated data, but that learning stops when training ends: "At deployment, an LLM applies the same policy regardless of how many problems it has solved in a domain, and discards all accumulated experience once the context resets" (Sec. 1). A body of work closes this gap at test time by having a frozen model rewrite its own context from feedback — automatic prompt optimization (GEPA, TextGrad, DSPy), self-referential agents (Gödel Agent, Darwin Gödel Machine, ADAS), skill and memory accumulation (Voyager, MemGen, Mem0). The paper's driving observation is that every one of these leans on an ability the model was never trained for: "These methods, however, rely entirely on the inherent ability of the LLM to analyze feedback and propose better context. The model is never explicitly trained for this self-improvement task" (Sec. 1). The proposal is to make proposing a good edit a trained skill rather than an emergent one.

### b) Pain points of existing methods

Three concrete complaints, in the paper's own order:

1. **The proposer is untrained.** Prompt optimizers and self-referential agents call a frozen model to write the next context. The paper argues the task is a reasoning problem with the structure of reinforcement learning — "An RL optimizer relies on dedicated algorithms to assign credit, estimate gradients, and balance exploration against exploitation. In self-evolution, the model must perform all three implicitly, through natural language reasoning alone" (Sec. 1) — so it should be optimized explicitly.
2. **Absolute post-edit score is the wrong training signal.** Rewarding the edit by the post-edit score "is biased toward contexts that are already effective" (Sec. 3.3): an edit that takes 80% down to 70% scores 0.7, while an edit that takes 30% up to 60% scores 0.6, so the ranking prefers the degradation. This is the paper's argument for an improvement-based reward.
3. **A linear evolution chain commits irreversibly.** "The linear evolution chain $c_0 \to c_1 \to \cdots$ greedily extends the most recent context, which risks committing irreversibly to a suboptimal evolution path" (Sec. 3.2).

### c) Core hypothesis / intuition

If the policy that rewrites the context is itself trained with reinforcement learning on the improvement each edit produces, a small model becomes a better self-evolver than a frontier model prompted to do the same job; the multi-step nature of evolution can then be left to a search loop at test time rather than to the training objective.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

LSE has two separable parts: a **test-time evolution loop** (Sec. 3.1–3.2, Algorithm 1) that works with any proposer, and a **training procedure** (Sec. 3.3) that produces a better proposer for that loop.

**Setting** (Sec. 3.1). A task is a triple $\mathcal{T} = (\mathcal{X}, \mathcal{Y}, R)$ — inputs, outputs, and a reward function. A policy $\pi$ maps inputs to outputs. A *self-evolving policy* is a function $f$ that updates the current policy from experience: over $T$ rounds, the current policy $\pi^{(t)}$ is applied to a batch of $k$ problems sampled from $\mathcal{X}$, producing experience tuples, and

$$\pi^{(t+1)} = f\big(\pi^{(t)}, \{(x_i, y_i, r_i)\}_{i=1}^k\big) \quad \text{(eq. 1)}$$

with the stated objective of maximizing the cumulative reward over the rounds,

$$\sum_{t=0}^{T} \mathbb{E}_{x \sim \mathcal{X}}\big[R(x, \pi^{(t)}(x))\big]. \quad \text{(eq. 2)}$$

A language-model policy $\pi_\theta$ is determined by parameters $\theta$ and context $c$, which admits two instantiations of $f$: gradient-based ($f$ modifies $\theta$) and prompt-based ($f$ modifies $c$ with $\theta$ frozen). The paper takes the prompt-based route explicitly to avoid forgetting: it "requires no gradient computation at test time, thereby avoiding the catastrophic forgetting problem with continual learning, and casts the evolution problem as a natural-language reasoning task that can itself be improved through training" (Sec. 3.1).

**What is edited** (Sec. 3.2). Inside the context, one field is mutable: "we designate a special *instruction field* within $c$ for the self-evolving policy to edit, leaving all other components (e.g., task description, format specification) fixed." Everything else in the harness — tools, memory, skills, sub-agent structure, weights — is untouched; the paper lists this as its third limitation (Sec. 5).

**The edit step** (eq. 3). The self-evolving policy maps the current context and a structured performance summary to a new context: $c_{t+1} = f_\psi(c_t, S_t)$, where $S_t = \{(x_i, y_i, y_i^*, r_i)\}_{i=1}^k$ holds the problems, the action model's outputs, the ground-truth answers, and per-problem correctness signals from round $t$. The repo also puts the action model's full chain of thought into $S_t$ (see `repo_analysis.md`).

**How an edit is scored** (eq. 4). The $k$ problems of a round are few and change each round, so in-batch accuracy is a noisy quality estimate. A *separate holdout set* $D \subset \mathcal{X}$ is fixed for the whole run and the score of a context is

$$\bar{R}(c) = \frac{1}{|D|} \sum_{x \in D} R(x, y), \quad y \sim \pi_\theta(\cdot \mid x, c). \quad \text{(eq. 4)}$$

Appendix A fixes $|D| = 50$ problems per domain, averaged over eight generations. This holdout is drawn from the **same domain** as the evolution batches — one BIRD database, or one MMLU-Redux subject — and is reused unchanged at every round of the run.

**Tree-guided evolution** (eq. 5, Algorithm 1). Rather than always extending the latest context, the system keeps an evolution tree $\mathcal{G}$ whose nodes store $(c_n, S_n, \bar{R}_n, v_n)$ — context, summary, mean holdout reward, visit count — and selects the node maximizing an Upper Confidence Bound score

$$n^* = \arg\max_{n \in \mathcal{G}} \ \bar{R}_n + C \sqrt{\frac{\ln N}{v_n}}, \quad \text{(eq. 5)}$$

with $N$ the number of completed rounds and $C > 0$ the exploration constant. The new context becomes a child of $n^*$. The run returns $\arg\max_{n \in \mathcal{G}} \bar{R}_n$ — the best node ever produced, not the last one.

**Training the proposer** (Sec. 3.3). The natural objective is the $T$-round sum $\max_{f_\psi} \sum_{t=0}^{T} \bar{R}(c_t)$ subject to $c_{t+1} = f_\psi(c_t, S_t)$ (eq. 6). The paper judges this too costly — "each rollout requires $T$ sequential rounds of evaluation and context generation, and the trajectory-level reward introduces a long-horizon credit-assignment problem" — and collapses it to the **single-step setting $T = 1$**: $f_\psi$ produces one update $c_1 = f_\psi(c_0, S_0)$ and is rewarded immediately. "This reduces the problem to a contextual bandit and avoids the long-horizon credit-assignment difficulty while still capturing the core challenge of learning to improve instructions from feedback" (Sec. 3.3).

**The reward** (eq. 7). Not the post-edit score but the improvement:

$$r_{\text{LSE}} = \bar{R}(c_1) - \bar{R}(c_0). \quad \text{(eq. 7)}$$

**Why the improvement reward is not vacuous** (eq. 8–10). The paper is honest that with a learned baseline the delta reward changes nothing: letting $V$ be the baseline under the post-edit reward and $V'$ under the delta reward, $V'(s) = V(s) - \bar{R}(c_0)$, so

$$A'(s, c_1) = \big(\bar{R}(c_1) - \bar{R}(c_0)\big) - \big(V(s) - \bar{R}(c_0)\big) = \bar{R}(c_1) - V(s), \quad \text{(eq. 8)}$$

"identical to the advantage under the post-edit reward alone." The gain comes from using the delta *instead of* baseline estimation: the pre-edit score $\bar{R}(c_0)$ is known before the policy acts and is exactly the value of the null edit, so it serves directly as the baseline, giving

$$A_{\text{LSE}} = \bar{R}(c_1) - \bar{R}(c_0), \qquad \nabla_\psi J = \mathbb{E}_{c_1 \sim f_\psi(\cdot \mid c_0, S_0)}\big[A_{\text{LSE}} \nabla_\psi \log f_\psi(c_1 \mid c_0, S_0)\big]. \quad \text{(eq. 9, 10)}$$

Because $\bar{R}(c_0)$ is action-independent it does not change the expected gradient; it is a control variate that "cancels prompt-specific offsets" from evaluation noise and between-prompt difficulty variation, and it removes the need for both multiple rollouts per prompt and a separate value network.

**Starting-state distribution** (Sec. 3.3, last paragraph). Training only from the seed context would mismatch deployment, where the policy must improve contexts it produced itself in earlier rounds. So the tree $\mathcal{G}$ is populated offline by multi-round evolution and RL steps start from nodes sampled out of it: "This exposes $f_\psi$ to a distribution of contexts similar to what it will see during multi-step evolution."

### b) Architecture components

| Component | Role |
|---|---|
| Action policy $\pi_\theta$ | Frozen model that solves the task under context $c$. Qwen3-4B-Instruct in the main tables; swapped for Arctic-Text2SQL-R1-7B in the transfer test. |
| Self-evolving policy $f_\psi$ | The trainable proposer. Reads the old prompt plus the performance summary, emits new instructions inside `<prompt>` tags. Also Qwen3-4B-Instruct. |
| Holdout evaluator | Runs $\pi_\theta$ under a candidate context on the fixed 50-problem holdout $D$ and returns $\bar{R}$. Used both as the tree's node value and as the RL reward source. |
| Evolution tree $\mathcal{G}$ | Stores every context produced, its summary, its holdout score and visit count; supplies UCB selection at test time and the starting-state distribution at training time. |

### c) Formulas and algorithms in plain language

Algorithm 1 (Prompt-Based Evolution with Tree Search), stated in words: start the tree with the seed context scored on the holdout; for each of $T$ rounds, pick the node with the best score-plus-exploration-bonus, sample $k$ fresh problems, run the action model under that node's context, score each answer, build a summary of problems-answers-truths-correctness, ask the proposer for a revised context, score the revision on the fixed holdout, attach it as a child and increment the parent's visit count; finally return the highest-scoring node in the tree.

The RL step in words: take a node out of a pre-built tree, hand its stored proposer prompt to the trainable model, sample several candidate rewrites, score each on the holdout, subtract the node's own pre-edit holdout score, and use that difference directly as the advantage for a clipped policy-gradient update.

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Every comparison point in the paper — GEPA, TextGrad, GPT-5 and Claude Sonnet 4.5 used as the proposer — leaves the proposer's weights untouched and differs only in the search or feedback machinery wrapped around it. LSE moves the contribution into the proposer's weights: the loop it uses at test time is deliberately plain (one edit per round, UCB over a tree), and the claim is that a trained 4B proposer inside a plain loop beats a frontier proposer inside a more elaborate one. The second conceptual difference is the reward: prompt optimizers select candidates by absolute validation score; LSE trains on the *change* in score, which makes the learning signal independent of how good the starting context happened to be.

### b) Key innovations and significance

1. **Improvement-based reward with the pre-edit score as an exact, free baseline** (eq. 7, 9) — *notable*. The derivation in eq. 8 is unusually candid: the delta is worth nothing against a learned baseline, and the real saving is bypassing value estimation and group sampling entirely. The ablation backs the choice empirically (4.3 points on BIRD, Sec. 4.3).
2. **Reduction of multi-step evolution to a single-step bandit, with the multi-step burden pushed to test-time search** — *significant as a design stance*, and the single most consequential choice for this project (see section 5c and the Notable list in the report).
3. **Training on a distribution of self-produced contexts sampled from pre-built evolution trees** — *notable*. It is the piece that makes a one-step objective usable in a many-step loop, and the mechanism transfers directly to any trained-proposer setup.
4. **UCB tree search over contexts** — *incremental* as a technique (GEPA already maintains a Pareto front to avoid greedy local optima), but its measurement here is valuable (Fig. 3).

### c) Applicability

It excels where problems within a domain share structure that an instruction can carry: the paper is explicit that the BIRD gains are larger than the MMLU-Redux gains because "in BIRD, all queries within a domain target the same database, so there is shared knowledge across problems... In MMLU-Redux, problems within the same subject are deliberately deduplicated and designed to cover broad topics" (Sec. 4.2). It struggles, by construction, where the useful edit is not an instruction — a new tool, a skill, a memory entry — and it says nothing about what happens when the domain changes under the evolved context.

### d) Comparison table

| | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| **LSE** | Trained proposer beats frontier proposers at 4B; reward is difficulty-normalized by construction; no value network, no group rollouts; the trained proposer transfers to a different action model without retraining (Table 3) | One policy per domain; only the instruction field is editable; the objective sees one edit ahead; no retention or degradation measure; headline metric is best-over-rounds | Multi-step objective (the paper's own first limitation); a single cross-domain policy (its second); editing tools, skill libraries and memory (its third) |
| **GEPA** (prompt optimizer) | Pareto front over per-instance performance guards against greedy local optima; no training needed | Proposer untrained; reflection quality bounded by the frozen model | — |
| **TextGrad** (prompt optimizer) | Two-call backward/forward decomposition gives interpretable textual "gradients" | Weakest of the compared optimizers on BIRD and the weakest overall on MMLU-Redux (69.1 average) | — |
| **Frontier model as proposer** (GPT-5, Claude Sonnet 4.5) | Strong zero-shot proposals with no infrastructure | Costly per round; still untrained for the task; loses to a trained 4B model on BIRD | — |
| **Linear evolution chain** | Simplest possible loop | Cascading failure: one bad edit poisons every descendant (Fig. 3) | Replace with tree search (what the paper does) |

---

## 4. Experimental Validation

### a) Experimental design

Two domains. **Text-to-SQL**: BIRD, where each database is a separate task domain; problems for both the evolution rounds and the holdout come from the same database. Training data comes from the BIRD training split; evaluation is on five randomly selected databases from the BIRD-SQL Mini-Dev split (Financial, Toxicology, Card Games, Formula 1, Codebase; 106–191 problems each, Table 4). **Question answering**: training on SuperGPQA converted to four-way multiple choice, evaluation on ten MMLU-Redux subjects (95–100 problems each, Table 4), each subject a separate task domain.

Both the action policy and the self-evolving policy are Qwen3-4B-Instruct unless stated. A separate self-evolving policy is trained per domain type. Protocol (Appendix A): holdout $|D| = 50$, scored as the average over eight generations; 25 evolution rounds at test time; each round samples a batch of 10 problems with replacement; the random seed is fixed so every method sees the same sequence of problem batches; **the reported number is the best holdout performance achieved over the rounds**. Training data: 200 data-generation runs of 20 evolution rounds each, roughly 4,000 tree nodes; nodes are sampled by a curriculum preferring "highest improvement potential, defined as the difference between a node's performance and the maximum performance in its own tree"; learning rate $1 \times 10^{-5}$, 32 nodes per batch, 4 rollouts per node, on-policy, no KL regularization, 4 epochs, best checkpoint chosen on a separate development set; implemented on veRL.

### b) Key results

1. **BIRD, execution accuracy, averaged over five databases (Table 1)**: seed prompt 57.2; untrained Qwen3-4B-Instruct as proposer 62.2; GEPA 62.8; TextGrad 63.1; Claude Sonnet 4.5 64.5; GPT-5 65.2; **LSE 67.3**. LSE is +2.1 over GPT-5, +2.8 over Claude Sonnet 4.5, +4.5 over GEPA, +4.2 over TextGrad, +10.1 over the seed prompt.
2. **MMLU-Redux, accuracy, averaged over ten subjects (Table 2)**: seed 67.6; untrained Qwen3-4B-Instruct 71.2; TextGrad 69.1; Claude Sonnet 4.5 72.0; GPT-5 72.5; GEPA 73.0; **LSE 73.3**. Here LSE essentially matches GEPA (+0.3) and GPT-5 (+0.8); the paper says so.
3. **Reward-design ablation (Sec. 4.3, Fig. 2a)**: training with the standard GRPO group-based advantage on the post-edit score reaches 63.0 on BIRD against 67.3 for the improvement-based advantage — a 4.3-point gap with all other settings identical.
4. **Search-strategy ablation (Sec. 4.3, Fig. 2b and Fig. 4)**, both with the *untrained* proposer: tree search 62.2 vs linear chain 59.8 on BIRD (+2.4), and 71.2 vs 69.0 on MMLU-Redux (+2.2).
5. **Transfer to a specialized action model (Table 3)**: the LSE-trained proposer, applied without further training to Arctic-Text2SQL-R1-7B (an RL-tuned text-to-SQL model) as the action policy, lifts BIRD execution accuracy from 57.7 to 64.4 (+6.7 average), with per-database gains from +4.7 (Formula 1) to +11.5 (Financial).

### c) Where it shines

The gap over untrained proposers is largest where instructions can carry reusable domain knowledge: BIRD (+5.1 over the untrained same-size proposer) versus MMLU-Redux (+2.1), with the paper's own explanation quoted in section 3c above. The transfer result is the strongest evidence that what was learned is a *policy for editing*, not a memorized prompt: the proposer was trained only against Qwen3-4B-Instruct and still improves a differently-trained 7B action model by 6.7 points.

### d) Limitations

Acknowledged (Sec. 5): the single-step reduction delegates exploration entirely to test-time search and a multi-step objective "could yield stronger policies"; one policy per task domain, with a single general policy requiring "large-scale training across many domains"; evolution is confined to the instruction field, leaving "tools, skill libraries, and external memory" unexplored; and "our training and evaluation environments are relatively small in scale," with environment curation named an open problem.

Unacknowledged or under-flagged:

- **The headline metric is the maximum over 25 rounds**, on a holdout of 50 problems reused at every round, with node values computed on that same holdout. Any decline over rounds is invisible to the reported number, and the search actively selects against it. The one place a within-run trajectory is shown is Fig. 3, and it shows a collapse.
- **The holdout is small and repeatedly optimized against.** 50 problems scored eight times per candidate, with UCB selecting on the same 50 problems for 25 rounds, is a validation set the loop has many opportunities to fit; the code contains a k-fold cross-validation path for round selection but the reported protocol does not use it.
- **No measurement of what an edit costs anywhere else.** Nothing is evaluated on problems from a different database or subject after an edit, so an instruction that helps Financial and hurts Toxicology is never observed — each run is confined to one domain.
- **Single seed.** "The random seed is fixed so that all methods observe the same sequence of problem batches" (Appendix A) controls the comparison but leaves no variance estimate; no error bars appear in Tables 1–3 or Figures 2–4.
- **A discrepancy between paper and code**: Appendix A states 32 nodes per batch and 4 rollouts per node, while the released training script defaults to `TRAIN_BATCH_SIZE=16` and `ROLLOUT_SAMPLES=8`.

---

## 5. Reproduction and Application

### a) Open source

Yes — https://github.com/chenyn66/learning-to-self-evolve, a complete release: the evolution loop, the tree, the BIRD and MMLU environments and agents, GEPA and TextGrad baselines, and a vendored veRL fork carrying the LSE reward function, dataset and advantage estimator. Full component map in `repo_analysis.md`. Reproduction path: install, download BIRD databases (the repo ships metadata only), run `scripts/run_bird.sh` with `n_round` and `tree.selection=ucb` to build evolution trees, then `scripts/rl/train_bird_ppo.sh` to train the proposer on those trees. Not shipped: the ~4,000-node trees themselves, any trained checkpoint, and a top-level README.

### b) Implementation details worth attention

The advantage estimator is registered as `delta` and does exactly eq. 9 — the scalar reward broadcast across response tokens, with no group mean subtracted and no critic (`critic.enable=false`, `use_kl_loss=false`, `use_kl_in_reward=false`). The reward function spins the action model up behind an OpenAI-compatible vLLM endpoint and scores candidate instructions by actually running the task; a malformed proposal (no `<prompt>` tag, or fewer than 10 characters) is scored at a fixed $-1.0$ before the delta is subtracted, which makes format failure far more costly than a merely bad edit. The proposer's prompt carries the action model's full chain of thought for every problem in the batch, which is why the training config allows a 65,536-token prompt.

### c) Transferability

Three pieces transfer cleanly to a different setting. The pre-edit score as a free baseline works for any edit-scoring setup where the state's value is measured before the action — it is not specific to prompts. The tree-as-starting-state-distribution trick is the general answer to "my proposer is trained on one step but deployed for many." And the tree serialization (`tree.json`: per node, the instructions, the summary, the holdout score, the parent link) is a ready-made record of harness evolution — exactly the shape a state-level model would need to be trained by regression on realized returns, except that the stored value is a one-round score rather than a discounted sum over later rounds.

---

## 6. Summary

### a) One-sentence core idea

Train the model that rewrites the prompt, rewarding each edit by how much the post-edit holdout score exceeds the pre-edit score, and let test-time tree search handle the multi-round part.

### b) Quick-reference pipeline

1. **Fix a scoring set and a seed prompt.** Set aside 50 problems from the domain; score the seed prompt on them by running the task model eight times and averaging.
2. **Run rounds.** Each round, pick a previously produced prompt by best-score-plus-exploration-bonus, run the task model under it on 10 freshly sampled problems, and collect what it answered, what was right, and its full reasoning.
3. **Ask the rewriter for a new prompt**, then score that new prompt on the same 50 held-out problems and hang it in the tree under the prompt it came from.
4. **Train the rewriter beforehand** on thousands of nodes harvested from earlier runs: for each node, sample several rewrites, score each on the holdout, subtract that node's own pre-edit score, and use the difference directly as the advantage — no value network, no group normalization.
5. **Report the best-scoring prompt ever produced** in the run.

The one field being rewritten is the instruction block inside the system prompt; nothing else about the system changes, and no problem outside the domain's own holdout is ever re-checked.

---

## 7. Reviewer Reception (OpenReview)

**a) Link and outcome.** A public OpenReview page exists: https://openreview.net/forum?id=zedEdPhmsA. Its venue field reads "LLA 2026 Poster" and its venue id is `ICLR.cc/2026/Workshop/LLA`, i.e. the paper was accepted as a poster at the ICLR 2026 Workshop on Lifelong Agents: Learning, Aligning, Evolving (confirmed independently by the ICLR 2026 virtual site entry, which lists it as a poster in that workshop). This is a workshop acceptance, not a main-conference one.

**b–d) Reviews.** No review text, scores, meta-review, or rebuttal could be retrieved. On 2026-09-14 the submission record itself was readable through the OpenReview search endpoint, but every request for the forum's replies (`notes?forum=zedEdPhmsA`, `notes?replyForum=zedEdPhmsA`) returned a challenge-verification error, and the rendered forum page returned a bot-check page rather than content. So the honest statement is: the acceptance and venue are confirmed, and nothing can be said about what reviewers wrote — it is unknown whether this workshop posts reviews publicly at all. Net takeaway: treat the paper as a workshop-reviewed preprint; the claims in it have no visible adversarial vetting on record, which matters most for the unflagged limitations in section 4d, since a best-over-rounds metric on a 50-problem reused holdout is exactly the kind of thing a reviewer would be expected to press on.
