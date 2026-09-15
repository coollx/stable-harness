# MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory

- **Original title**: MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory
- **Authors**: Shengtao Zhang, Jiaqian Wang (equal contribution), Ruiwen Zhou, Junwei Liao, Yuchen Feng, Zhuo Li, Yujie Zheng, Weinan Zhang, Ying Wen, Zhiyu Li, Feiyu Xiong, Yutao Qi, Bo Tang (corresponding, tangb@memtensor.cn), Muning Wen (corresponding, muningwen@sjtu.edu.cn).
- **Affiliations**: Shanghai Jiao Tong University; Xidian University; National University of Singapore; Shanghai Innovation Institute; MemTensor (Shanghai) Technology Co., Ltd.; University of Science and Technology of China.
- **Venue / year**: arXiv preprint, 2026 (arXiv:2601.03192v2 [cs.CL], v2 dated 12 Feb 2026, preprint line "February 13, 2026"; 41 pages including appendices A-I).
- **Source URL**: https://arxiv.org/abs/2601.03192
- **Code**: https://github.com/MemTensor/MemRL (MIT) — see `repo_analysis.md`.
- **OpenReview**: no public page found (see section 7).

Classified as a **method paper**: the dominant contribution is a built, reproducible system — an episodic memory store whose entries carry a learned scalar utility, a two-phase retriever that consumes it, and an update rule that maintains it — with the five-benchmark study serving to validate that system rather than to characterize a phenomenon.

Short names used below: MemRL (the paper's system), MemP (the prior procedural-memory system MemRL extends in code and its strongest baseline), Mem0 and Self-RAG (retrieval baselines), and the benchmarks BigCodeBench, Lifelong Agent Bench (its OS Interaction and DB Bench subsets), ALFWorld, and Humanity's Last Exam.

---

## 1. Method Motivation

### a) Why was this method proposed?

The driving force is that a deployed language-model agent is frozen: its weights do not move, so whatever it learns from running tasks has to live outside the parameters. The paper's premise is that episodic memory is the natural place for that learning to live, and that the field has built the storage half of the loop without the learning half. Agents accumulate trajectories, reflections, and procedural notes, but nothing in those systems ever tells the agent which of its stored experiences actually helped. Retrieval is decided by semantic similarity to the current task and nothing else, so an experience that led to a wrong answer is as retrievable as one that led to a right answer, provided it is worded similarly.

### b) Pain points of existing methods

Four concrete shortcomings are named. First, **value-blind retrieval**: similarity-based memory (retrieval-augmented generation, Mem0, MemP) ranks candidates by embedding distance, which measures relevance to the query and says nothing about outcome. Second, **no credit assignment**: experience is written once and never re-scored, so a memory that has been retrieved fifty times and preceded fifty failures carries the same standing as one that has never been used. Third, **single-task confinement**: reflection-style methods (the Reflexion line) generate a critique of the current attempt and consume it within the same task, so the lesson does not survive to a different task; the paper measures this directly as its own ablation (Table 3). Fourth, **the cost of the parametric alternative**: improving the agent by updating weights requires model access, gradient compute, and a training run per deployment, which is not available for the closed frontier models the paper uses as backbones.

### c) Core hypothesis / intuition

If every memory item carries a scalar utility learned from the outcomes of the tasks in which it was retrieved, and retrieval scores candidates by that utility as well as by similarity, then a frozen agent performs reinforcement learning at runtime with the memory store as its action space — no gradients, no weight access.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

MemRL casts memory-augmented generation as what the paper calls a Memory-Based Markov Decision Process: the state is the incoming task together with its context, the action space is the memory store, and the frozen language model is a fixed conditional policy that turns a (state, retrieved memory) pair into behavior. One pass over one task runs five stages.

**1. Intent extraction.** The incoming task is distilled into a compact intent string, which is embedded and used as the retrieval key. This separates what the task is asking from the full prompt text, so the key is stable across surface rewordings.

**2. Phase A retrieval, similarity recall.** The intent embedding is scored by cosine similarity against every stored intent embedding. Candidates below a similarity threshold $\delta$ are dropped outright; the surviving top $k_1$ form the candidate pool (eq. 6). This phase is a pure recall filter and never looks at utility.

**3. Phase B retrieval, value-aware selection.** Each candidate is scored by a convex blend of its normalized similarity and its learned utility $\hat{Q}$, controlled by a single weight $\lambda$ (eq. 7); the top $k_2$ are injected into the prompt. Because similarity and utility live on different scales, both are z-normalized before blending. This is the only place the learned quantity affects behavior.

**4. Execution and reward.** The frozen backbone runs the task with the retrieved experiences in context. An environment verifier returns a binary outcome — unit tests for BigCodeBench, environment success for Lifelong Agent Bench and ALFWorld, answer match for Humanity's Last Exam — which becomes the scalar reward $r$.

**5. Utility update and memory write.** Every memory item that was retrieved for this task receives the task's reward and has its utility moved toward it (eq. 4). Then the trajectory itself is written back as a new memory item: on success the procedural trace is stored; on failure the system generates a reflection over the failed trajectory and stores that instead, so a failed task still contributes a usable lesson rather than a poisoned example. Near-duplicate items above a novelty threshold are merged rather than appended, to bound store growth.

The critical structural fact is the ordering: the task is answered using the memory store exactly as it stood when the task arrived, and the store changes only afterward. MemRL therefore has a clean before-update scoring protocol at the level of the task, even though the implementation batches the writes (see `repo_analysis.md`).

### b) Architecture components

**The memory item (eq. 5).** Each entry is a triplet $m = (\text{intent}, \text{experience}, \text{utility})$. The intent field is the embedded retrieval key; the experience field is the procedural content injected into the prompt — a successful trajectory summary, or a reflection distilled from a failure; the utility field is the scalar $Q$ maintained by the update rule. Only the intent is indexed, and only the utility is learned; the experience text is immutable once written (the repo's adjustment strategy appends a note instead of rewriting, see `repo_analysis.md`).

**The memory-selection policy.** Retrieval is treated as a policy $\mu(m \mid s)$ over the store, composed with the frozen backbone policy $\pi_{\text{LLM}}(a \mid s, m)$ to give the agent's effective policy (eq. 1). Making this composition explicit is what lets the paper call utility maintenance reinforcement learning: $\mu$ is the only learnable object, and its greedy form is $\mu^*(s) = \arg\max_m Q(s, m)$ (eq. 2).

**The two-phase retriever.** Phase A is a hard recall gate (threshold plus top $k_1$); Phase B is a soft ranking gate (blend plus top $k_2$). The paper's justification for the split is that a single blended score over the whole store would let a high-utility but semantically unrelated memory be injected into an unrelated task; the threshold prevents that by construction.

**The curator.** A merge step folds a new item into an existing one when their similarity exceeds a novelty threshold, propagating a fraction of the new reward into the merged item's utility rather than creating a second near-identical entry.

### c) Formulas and algorithms

**Composed policy (eq. 1).** The agent's action distribution marginalizes over which memory is retrieved:

$$\pi(a \mid s) = \sum_{m \in \mathcal{M}} \mu(m \mid s)\, \pi_{\text{LLM}}(a \mid s, m)$$

The backbone term is frozen. All learning is in $\mu$.

**Greedy selection (eq. 2).** $\mu^*(s) = \arg\max_{m \in \mathcal{M}} Q(s, m)$ — the memory-selection policy is optimal when it retrieves the highest-utility admissible item.

**The general temporal-difference update (eq. 3).**

$$Q(s, m) \leftarrow Q(s, m) + \alpha\left[r + \gamma \max_{m'} Q(s', m') - Q(s, m)\right]$$

This is the full form, carrying a bootstrap term over the next state $s'$ and a discount $\gamma$.

**The update actually used (eq. 4).**

$$Q_{\text{new}} \leftarrow Q_{\text{old}} + \alpha\,(r - Q_{\text{old}})$$

The paper states plainly that eq. 4 "performs as a naturally simplified version of Equation 3 by setting $s'$ as a terminal state" (Sec. 4.3). Setting the next state terminal removes the bootstrap term, which is the same thing as setting the discount to zero. What remains is an exponential moving average of the rewards observed on the tasks in which this memory was retrieved, with learning rate $\alpha = 0.3$ in every reported experiment (Table 8). **No quantity from any later task enters this update.** The utility of a memory item is an estimate of the immediate reward of retrieving it for a similar intent, not a return over the stream that follows.

**Phase A (eq. 6).** $\mathcal{C} = \text{TopK}_{k_1}\{m : \cos(e_s, e_m) > \delta\}$ — recall by cosine similarity above threshold $\delta$.

**Phase B (eq. 7).** $\text{score}(m) = (1-\lambda)\,\tilde{s}(m) + \lambda\,\tilde{Q}(m)$, where $\tilde{s}$ and $\tilde{Q}$ are the z-normalized similarity and utility over the candidate pool; take the top $k_2$.

**Convergence argument (Appendix A).** Theorem A.1 shows the eq. 4 iteration converges to $\beta(s, m) = \mathbb{E}[r_t \mid s, m]$ under the usual stochastic-approximation conditions. This is worth reading carefully, because it states exactly what the learned quantity is: a **one-step expected reward**, not a value function over a horizon. The paper additionally frames the alternation between retrieval and utility update as generalized expectation-maximization — retrieval is the E-step, utility update the M-step — with a Kullback-Leibler semantic trust region arguing that the similarity threshold $\delta$ keeps successive selection policies close enough for monotone improvement (eqs. 8-9).

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Against similarity-only memory (retrieval-augmented generation, Mem0, MemP): those systems decide retrieval by relevance alone; MemRL adds an outcome-derived scalar to the ranking and maintains it online. Against reflection (the Reflexion line): those consume the critique inside the task that produced it; MemRL writes the reflection into a shared cross-task store and lets a later, different task retrieve it — Table 3 isolates exactly this change and attributes up to 9.0 points to it on OS Interaction. Against parametric reinforcement learning: MemRL performs no gradient step and touches no weight; the entire learned state is a float per memory item, which is why it runs on closed backbones (GPT-4o, GPT-5-mini, Gemini-3-pro) unchanged.

### b) Key innovations and their significance

1. **A scalar utility field on each memory item, maintained by an online update** — *notable*. The idea of scoring memories is not new, but maintaining the score by a reward-driven running average, per item, during deployment, and letting it enter retrieval is a concrete mechanism others can copy in a hundred lines.
2. **The two-phase retriever with a hard similarity gate before the value-aware blend** — *notable*. The ablation on the gate is the paper's own best evidence: removing normalization and the similarity gate raises the forgetting rate from 0.041 to 0.073 (Sec. 5.4.2).
3. **The Memory-Based Markov Decision Process formalization with a convergence theorem** — *incremental as reinforcement learning, notable as framing*. The theorem's content is standard stochastic approximation, and the process is collapsed to one step before use; its value is that it makes the design's horizon explicit rather than hidden.
4. **The utility calibration study (Figure 7a)** — *notable*. A Pearson correlation of 0.861 between a memory's learned utility bin and the downstream success rate of tasks that retrieved it, rising from 21.5% success in the lowest bin to 88.1% in the highest, is direct evidence that the scalar means something. This is the single most reusable measurement in the paper.

### c) Applicability

It shines where tasks in the stream resemble each other, because the mechanism can only fire when Phase A recalls something. The paper measures this itself: intra-dataset intent similarity is 0.518 on ALFWorld, 0.390 on OS Interaction, 0.308 on BigCodeBench, and 0.186 on Humanity's Last Exam (Appendix B.3, Figure 10), and the gains track that ordering for the multi-step benchmarks. It requires a verifier: the reward must be computable automatically per task, which excludes open-ended generation. It struggles where the backbone is already near ceiling (DB Bench, where MemRL and MemP tie at 0.960 last-epoch success) and where each task is genuinely novel.

### d) Comparison table

| Method | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| MemRL | Outcome signal enters retrieval; no weights touched; works on closed backbones; utility is calibrated against downstream success (Figure 7a) | Horizon is one step by construction (eq. 4); store grows without an effective bound (merging changes results by 0.004 or less, Table 6); needs a per-task verifier; five tuned constants per benchmark (Table 8) | Carry a real bootstrap term over later tasks; delete or demote low-utility items rather than only filtering them at retrieval; measure retention on a held-aside set rather than on the replayed training set |
| MemP (procedural memory) | Simple, strong, benchmark-agnostic; the reference this work extends | Value-blind retrieval; nothing is ever re-scored after writing | Add the utility field — which is precisely what MemRL does |
| Mem0 | Language-model-managed memory operations (add, update, delete) give explicit lifecycle control | Lifecycle decisions come from a prompted model, not from outcomes; weakest of the memory baselines here (0.712 average cumulative success) | Ground the delete decision in realized utility |
| Self-RAG / retrieval-augmented generation | No state to maintain; cheap; no verifier needed | Retrieval is relevance-only; can retrieve a failure as readily as a success | Any outcome signal at all |
| Reflexion-style single-task reflection | Immediate, targeted, no cross-task machinery | Lesson dies with the task; 0.761 versus 0.798 average here (Table 3) | Persist reflections into a shared store — again, what MemRL does |
| Parametric reinforcement learning | Updates the policy itself; unbounded capacity | Needs weight access and gradient compute; impossible on the closed backbones used here | Not comparable in this deployment setting |

---

## 4. Experimental Validation

### a) Experimental design

Five benchmarks spanning single-step and multi-step agency (Table 9): BigCodeBench-Instruct Full (1,140 runtime tasks, 1,140 transfer), Lifelong Agent Bench OS Interaction (500 / 500) and DB Bench (500 / 500) split 7:3 with seed 42, ALFWorld (3,553 runtime, 140 transfer instances drawn from valid_seen), and Humanity's Last Exam (2,500 tasks, runtime only). Backbones differ per benchmark (Table 7): GPT-4o for BigCodeBench, GPT-4o-mini for Lifelong Agent Bench, GPT-5-mini for ALFWorld, Gemini-3-pro for Humanity's Last Exam, with text-embedding-3-large throughout, temperature 0.0 (0.6 for Humanity's Last Exam) and top-p 1.0.

The protocol is the part that matters most for reading the numbers. The **runtime learning** setting runs the same task set for ten epochs, updating memory as it goes, and reports two quantities: last-epoch success rate, and a cumulative success rate defined as the fraction of tasks solved at least once across the ten epochs. The natural comparison for cumulative success is a Pass@10 control — the same backbone with no memory given ten independent attempts. The **transfer** setting freezes the memory store after runtime learning and evaluates it once on a disjoint task set.

Baselines: no memory, Pass@10, retrieval-augmented generation, Self-RAG, Mem0, MemP.

### b) Key results

1. **Runtime learning, five-benchmark average (Table 1)**: MemRL reaches 0.772 last-epoch success and 0.798 cumulative success, against MemP at 0.736 / 0.760, Mem0 at 0.681 / 0.712, Self-RAG at 0.673 / 0.726, retrieval-augmented generation at 0.679 / 0.699, and no memory at 0.631 with a Pass@10 control of 0.743. The headline claim is +3.8 points of average cumulative success over MemP.
2. **Per-benchmark cumulative success (Table 1)**: BigCodeBench 0.627 (MemP 0.602), OS Interaction 0.804 (0.742), DB Bench 0.972 (0.966), ALFWorld 0.981 (0.919), Humanity's Last Exam 0.606 (0.570). The two multi-step benchmarks carry the result: +6.2 points each on ALFWorld and OS Interaction (Table 4), against +3.6 on Humanity's Last Exam and +2.5 on BigCodeBench.
3. **Transfer (Table 2)**: with the store frozen, MemRL averages 0.794 against MemP's 0.766, so roughly three quarters of the runtime advantage survives being moved to unseen tasks — 0.979 versus 0.950 on ALFWorld, 0.746 versus 0.720 on OS Interaction.
4. **Cross-task retrieval is the load-bearing component (Table 3)**: replacing the shared store with single-task reflection drops the average from 0.798 to 0.761, with the damage concentrated exactly where the paper predicts — 0.804 to 0.714 on OS Interaction (9.0 points) and 0.981 to 0.930 on ALFWorld (5.1 points), while Humanity's Last Exam is unaffected and in fact slightly better without it (0.610 versus 0.606).
5. **The learned utility is calibrated (Figure 7a)**: Pearson correlation 0.861 between the utility bin of retrieved memories and the success rate of the tasks that retrieved them, spanning 21.5% in the 0.2-0.3 bin to 88.1% in the 0.9-1.0 bin. Figure 7b tempers this: roughly 12% of the items in the highest-utility bins are memories originally written from failures, meaning a distilled failure reflection can legitimately earn a high utility.
6. **Cross-model portability (Table 5)**: a memory store built on Humanity's Last Exam lifts Qwen3-235B from 0.150 to 0.531, Gemini-3-flash from 0.347 to 0.583, and GPT-5.2 (High) from 0.354 to 0.571 — the largest absolute gain going to the weakest backbone.
7. **Weak-backbone amplification (Tables 10-11)**: holding GPT-4o-mini fixed across all five benchmarks, MemRL averages 0.730 last-epoch and 0.819 cumulative against no memory's 0.604 and MemP's 0.665 / 0.707; on ALFWorld with this weaker model the gap is extreme (0.440 / 0.680 against MemP's 0.299 / 0.413 and no memory's 0.278). Frozen-store transfer under the same backbone gives 0.715 against 0.654 and 0.609.

### c) Where it shines

Multi-step, environment-grounded, internally homogeneous task sets with a weak-to-mid backbone. Every one of the paper's largest gaps sits in that cell: ALFWorld with GPT-4o-mini (+26.7 points of cumulative success over MemP), Humanity's Last Exam with Qwen3-235B (+0.381 absolute), OS Interaction with GPT-4o-mini. The paper's own explanation is the similarity analysis in Appendix B.3: the mechanism cannot fire unless Phase A recalls something, and recall probability tracks intra-dataset intent similarity, which is highest on ALFWorld (0.518) and lowest on Humanity's Last Exam (0.186).

### d) Limitations

Acknowledged: the method needs a verifiable reward; the store grows and the merging remedy is close to inert (OS Interaction 0.788 to 0.784, DB Bench unchanged at 0.960, Table 6); the cost per Humanity's Last Exam question is roughly 32K tokens, comparable to MemP (Appendix F).

Implicit, and more important for reading the headline numbers:

- **Cumulative success cannot decrease.** It is defined as "solved at least once so far", so it is monotone in the epoch index by construction. It measures coverage over ten attempts, not the state of the agent at the end. The quantity that *can* fall is last-epoch success rate, and the paper reports only the last epoch, never the epoch-by-epoch curve for all methods, so a peak-then-decline pattern would be invisible in Table 1.
- **The "stream" is the same task set replayed ten times in a fixed order.** There is no distribution shift within a run and no novelty after epoch 1; the transfer tables are the only measurement on unseen tasks, and they are a single frozen-store evaluation rather than a stream.
- **Retention is measured but never acted on.** Section 5.4.2 defines a forgetting rate as the number of tasks lost (solved in some earlier epoch, failed now) over the number of current failures, and reports 0.041 for MemRL against 0.051 for MemP and 0.073 for the ablated variant. That is a real retention measurement, and it is reported as a side effect of the similarity gate — no mechanism in the system consults it, and nothing is reverted when it rises.
- **Five constants are tuned per benchmark** (Table 8): the similarity threshold $\delta$ ranges from 0.25 on Humanity's Last Exam to 0.62 on ALFWorld, with $k_1 \in \{5, 10\}$ and $k_2 \in \{3, 5\}$; $\lambda = 0.5$ and $\alpha = 0.3$ are constant. The $\lambda$ sweep (Figure 5) over $\{0, 0.25, 0.5, 0.75, 1\}$ peaks in the middle, which is the paper's evidence that both terms matter, but it also means pure value-driven retrieval ($\lambda = 1$) is worse than the blend.
- **Backbones differ per benchmark**, so no single row of Table 1 is a controlled model comparison; Table 10 is the controlled version and is relegated to the appendix.

---

## 5. Reproduction and Application

### a) Open source?

Yes: https://github.com/MemTensor/MemRL, MIT-licensed, Python, built on the MemOS memory service with a Qdrant vector store and OpenAI-compatible model and embedding providers. Full component inventory in `repo_analysis.md`. The release is honest about its lineage — it describes itself as a layer plugged into the existing MemP service — and it ships runnable entry points for Lifelong Agent Bench, BigCodeBench, ALFWorld, and Humanity's Last Exam with the configuration files for each. Reproduction needs paid model access: nothing in the repository is a released result file, log, trajectory, or memory snapshot, so every number must be regenerated.

### b) Implementation details requiring attention

The discount is zero, and this is not a default left unexamined — it is written into the code default, into the configuration schema, and into all four shipped configuration files, with the comment "discount factor (0 for single-step credit assignment)" (see `repo_analysis.md`). Reward is $+1$ on success and $-1$ on failure, and **every memory retrieved during a trajectory receives the same trajectory-level reward**, with no per-step attribution. Utilities are initialized at 0.0 (Table 8). The Phase B blend normalizes similarity with a fixed mean and standard deviation but normalizes utility per call over the current candidate pool, so a candidate's effective utility score depends on which other candidates were recalled. A minimum-utility filter excludes low-utility items from retrieval entirely; it is the closest thing in the system to a gate, and it is a soft one — nothing is ever deleted or reverted. The updates are flushed in mini-batches (five tasks on Lifelong Agent Bench, twenty-five on BigCodeBench, thirty-two on Humanity's Last Exam), so the before-update scoring property holds at mini-batch granularity, not per task.

### c) Transferability

The method carries to any setting with a task stream, a verifier, and a retrievable store. The two pieces that transfer cleanly are the utility field itself (a float and a visit counter per item, updated by one line) and the calibration measurement of Figure 7a, which only needs retrieval logs plus task outcomes. The pieces that do not carry are the per-benchmark thresholds, which must be re-tuned because they depend on the embedding model and the intent distribution.

### d) Transfer to our own work

**Where MemRL sits against our claim.** It is a self-improving harness acting on the long-term-memory layer, and it is the clearest *self-aware* instance of the horizon $\gamma = 0$ case in this collection. Other systems in this index are greedy by omission — they simply never consider a future term. MemRL writes the general update with a discount and a bootstrap term (eq. 3), then removes both on purpose and says so: eq. 4 is eq. 3 "by setting $s'$ as a terminal state" (Sec. 4.3), the released configuration files all set the discount to zero, and the convergence theorem proves the learned quantity is the one-step conditional expectation $\mathbb{E}[r_t \mid s, m]$. When we write that existing systems score an edit only by the batch that produced it, MemRL is the citation where the authors derived that choice from the general case rather than never reaching it. It is also, in the released code, an attempted-and-abandoned exception: the runner calls a chained multi-step update method that does not exist anywhere in the repository (`repo_analysis.md`), so the one code path that could have propagated value across tasks is dead.

**What is reusable.**

1. *The utility field as a per-item plasticity proxy.* Our critic $\hat{\Phi}(h)$ reads the whole harness and returns one scalar. MemRL's utility is the same shape at a much smaller granularity — one scalar per memory item, learned by regression on realized outcomes — and Figure 7a is the validation study we would want for our own critic, computed only from retrieval logs and task outcomes with a Pearson correlation against downstream success. That measurement design transfers directly; the quantity it validates does not, because their target is one step ahead and ours is a discounted return.
2. *The forgetting rate as a retention statistic.* Section 5.4.2's ratio of tasks lost to current failures is computable over any replayed set and gives a per-epoch number (0.041 for MemRL, 0.051 for MemP, 0.073 ablated). It is not an anchor set — it is computed over the same replayed training tasks, not a set held for retention checking — but it is the closest published retention measurement on the memory layer, and our anchor set is exactly the instrument that would make it a commit-time signal rather than a post-hoc report. It is also, notably, not computed by any code they shipped.
3. *The two-phase retriever as a retention mechanism nobody labelled as one.* The paper's own ablation shows the hard similarity gate is what keeps the forgetting rate down: without it, 0.073. That is evidence that in memory-layer harnesses, retention damage arrives through *inappropriate retrieval* rather than through content loss — a high-utility memory injected into an unrelated task is the failure mode. That mechanism is specific to the memory layer and does not generalize to prompt or tool edits, which matters when we reason about which layer a retention gate should live at.
4. *The frozen-store transfer protocol.* Their transfer tables (2 and 11) hold the store fixed and evaluate once on a disjoint set. That is a generalization measurement, one of the three future quantities in our claim, and the one memory papers most often skip.

**What is not reusable.** There are no released logs, trajectories, or memory snapshots, so nothing here can pretrain a critic. There is no anchor set, no revert, and no gate on an edit — the minimum-utility filter suppresses retrieval of a bad memory but never removes it, so the store is append-only and the reported cost per task grows with it. And the ten-epoch replay of one fixed task set in one fixed order is not a task stream in our sense: it has no heterogeneity and no non-stationarity, so a loss-of-plasticity effect would have nowhere to show up. MemRL is evidence about what the memory layer can learn from one-step outcomes, not evidence about what happens to a harness over a stream.

---

## 6. Summary

### a) One-sentence core idea

Attach a reward-learned usefulness score to every stored experience and let retrieval rank by it, so a frozen agent improves without any weight update.

### b) Quick-reference pipeline

1. Boil the incoming task down to a short statement of what it is asking, and turn that into a vector.
2. Find stored experiences whose vectors are close enough to that one, keeping only the closest handful above a fixed closeness cutoff.
3. Re-rank that handful by an equal mix of closeness and a stored usefulness number, and put the best few into the prompt.
4. Run the task with the frozen model, and let the environment say pass or fail.
5. Nudge the usefulness number of every experience that was used toward the outcome just observed (up on pass, down on fail), then store this run as a new experience — the trajectory if it passed, a written-out lesson if it failed — merging it into a near-identical existing entry when one exists.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched "MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory" and a site-restricted search of openreview.net on 2026-09-15). The paper carries no venue line beyond "Preprint", and no submission, reviews, scores, or decision are publicly visible.
