# Gradient Episodic Memory for Continual Learning

- **Original title**: Gradient Episodic Memory for Continual Learning
- **Authors**: David Lopez-Paz, Marc'Aurelio Ranzato
- **Affiliation**: Facebook Artificial Intelligence Research
- **Venue / year**: 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach; paper ID 3235. Proceedings page: https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html
- **Source URL**: https://arxiv.org/abs/1706.08840 — local copy `paper.pdf` is arXiv v6 (13 Sep 2022; v1 26 Jun 2017, v5 4 Nov 2017 is the camera-ready; v6 changes are not described on the abstract page). 17 pages: 8 pages main text, references, Appendix A (hyper-parameter grids), Appendix B (the full 20×20 accuracy matrix for every model and dataset).
- **Code**: https://github.com/facebookresearch/GradientEpisodicMemory (CC BY-NC 4.0; stated in Section 1) — see `repo_analysis.md`.
- **OpenReview**: no public page (NIPS 2017 did not use OpenReview). The official referee reports are public on the proceedings site and are summarized in section 7.

Classified as a **method paper**: the dominant contribution is a built, reproducible training procedure — an episodic memory per task plus a gradient projection that forbids any update from increasing the loss on that memory — validated on three image streams. The paper also proposes an evaluation protocol (a matrix of per-task accuracies and three metrics read off it); that secondary contribution is treated in section 2a as part of the pipeline and in section 5d as the part of most direct use to us.

**Why this reference matters here**: it is the earliest weight-space instance of the object `docs/framing.md` calls a retention constraint at commit — "the loss on every past task's memory must not increase after this update" — and its evaluation matrix is a ready-made instrument for two of the three future costs in our claim (retention of earlier tasks, generalization to later tasks). Section 5d states exactly what carries over to a harness of files and what stays in weight space.

Short names used below: GEM (the paper's method), EWC (elastic weight consolidation, Kirkpatrick et al. 2017, a baseline), iCaRL (Rebuffi et al. 2017, a baseline), and the three datasets MNIST permutations, MNIST rotations, and incremental CIFAR-100.

---

## 1. Method Motivation

### a) Why was this method proposed?

Standard supervised learning assumes independent and identically distributed (iid) samples from one fixed distribution and takes many passes over them. The authors want the opposite regime, which they call "more human-like": the learner sees a continuum of examples $(x_1, t_1, y_1), \dots, (x_n, t_n, y_n)$ (eq. 1), one at a time, each exactly once, where $t_i$ is a task descriptor and whole runs of consecutive examples belong to one task before the next task starts. The goal is a single predictor $f : \mathcal{X} \times \mathcal{T} \to \mathcal{Y}$ that can be queried at any time on any task seen so far.

### b) Pain points of existing methods

Three, named in Section 1: (1) the data is not iid, so empirical risk minimization has no guarantee; (2) catastrophic forgetting — learning task $t$ hurts performance on earlier tasks; (3) transfer — related tasks should make later ones faster and earlier ones better, and existing methods do not exploit this. Against the two closest prior families the paper is specific. Synaptic-memory methods (EWC, synaptic intelligence) regularize parameters toward their old values; episodic-memory methods with distillation (iCaRL, learning without forgetting) keep old *predictions* fixed. Both treat the past as something to hold constant, which "would deem positive backward transfer impossible" (Section 3). The paper also faults the standard evaluation regime — few tasks, many examples per task, several passes, one average-accuracy number — for hiding both forgetting and transfer.

### c) Core hypothesis / intuition

If a small memory of each past task is kept, an update can be tested before it is applied: any step whose gradient points against a past task's gradient would raise that task's loss, so project it onto the cone of steps that do not. Framing the past as *inequality constraints* rather than as a fixed target leaves room for the past to improve.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

**Input.** A stream of triplets $(x, t, y)$ in task order; a total memory budget $M$ examples; a test set per task for evaluation; a scalar margin $\gamma \ge 0$ (the paper's symbol — this is *not* the horizon $\gamma$ of `docs/framing.md`; here it is a constant that biases the projection).

**Step 1 — store.** On observing $(x, t, y)$, add it to the episodic memory $\mathcal{M}_t$ of the current task. With $T$ tasks known in advance, each task gets $m = M/T$ slots; the memory holds "the last $m$ examples from each task" (Section 3). The paper notes that a coreset would be better and leaves it for future work.

**Step 2 — measure the past.** For every task $k < t$ seen so far, compute the loss on its memory (eq. 5),
$$\ell(f_\theta, \mathcal{M}_k) = \frac{1}{|\mathcal{M}_k|} \sum_{(x_i, k, y_i) \in \mathcal{M}_k} \ell(f_\theta(x_i, k), y_i),$$
and its gradient $g_k = \nabla_\theta\, \ell(f_\theta, \mathcal{M}_k)$. This is one forward-backward pass per past task per update — the paper names it as the method's bottleneck.

**Step 3 — propose.** Compute the ordinary gradient on the current example, $g = \nabla_\theta\, \ell(f_\theta(x, t), y)$.

**Step 4 — check.** Test the angle between $g$ and each $g_k$ (eq. 7). If $\langle g, g_k \rangle \ge 0$ for all $k < t$, the step is accepted unchanged: to first order it does not raise any past loss.

**Step 5 — project on violation.** Otherwise replace $g$ by the closest $\tilde g$ (in squared Euclidean distance) that satisfies every constraint (eq. 8). This is a quadratic program; the paper solves its dual in $t-1$ variables (eq. 11), recovers $\tilde g = G^\top v^\star + g$, and adds the margin $\gamma$ to $v^\star$ so the projected step leans toward *decreasing* past losses.

**Step 6 — update.** $\theta \leftarrow \theta - \alpha \tilde g$ (Algorithm 1). Plain stochastic gradient descent, mini-batch 10, one pass.

**Step 7 — evaluate.** After the last example of each task $i$, evaluate on the test set of every task $j$ and write row $i$ of the matrix $R \in \mathbb{R}^{T \times T}$, $R_{i,j}$ = test accuracy on task $j$ after finishing task $i$. Also record $\bar b$, the accuracy vector at random initialization. Three metrics are read off $R$ (eqs. 2–4):
$$\mathrm{ACC} = \frac{1}{T} \sum_{i=1}^{T} R_{T,i}, \qquad \mathrm{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} \left(R_{T,i} - R_{i,i}\right), \qquad \mathrm{FWT} = \frac{1}{T-1} \sum_{i=2}^{T} \left(R_{i-1,i} - \bar b_i\right).$$
ACC is final average accuracy. BWT (backward transfer) is how much each task's accuracy moved between the moment it was learned and the end of the stream — negative BWT is forgetting, positive BWT is later tasks improving earlier ones. FWT (forward transfer) is accuracy on a task *before any of its data has been seen*, relative to an untrained network — the zero-shot benefit of what came before. Larger is better for all three; "if two models have similar ACC, the most preferable one is the one with larger BWT and FWT". The paper notes $R$ can be made finer-grained by evaluating more often than once per task, which the code does (every 100 mini-batches).

**Output.** The trained predictor $f_\theta$ and the matrix $R$ (Algorithm 1 returns both).

### b) Architecture components

There is no new architecture. GEM wraps an ordinary network: a fully-connected net with two hidden layers of 100 rectified linear units on the MNIST streams; a reduced ResNet-18 with three times fewer feature maps on CIFAR-100, with "a final linear classifier per task" — implemented as masking the output to the current task's five classes, so the task descriptor selects the head (Section 4.2). The only method-specific state is the memory tensor $\mathcal{M}_1, \dots, \mathcal{M}_T$ and, transiently, the $p \times (t-1)$ matrix $G$ of past-task gradients.

### c) Formulas and algorithms

**The constrained problem (eq. 6)**, solved at every step:
$$\min_\theta\ \ell(f_\theta(x,t), y) \quad \text{subject to} \quad \ell(f_\theta, \mathcal{M}_k) \le \ell(f_\theta^{t-1}, \mathcal{M}_k)\ \ \text{for all } k < t,$$
where $f_\theta^{t-1}$ is the predictor at the end of task $t-1$. In words: lower the current loss without letting any past task's memory loss rise above where it stood when that task was last trained. Two observations make it tractable. First, the old predictors need not be stored: it suffices that no single update raises a past loss, so the reference point becomes the current parameters. Second, assuming the loss is locally linear over one small step and the memory is representative of its task, "loss on $\mathcal{M}_k$ does not increase" becomes the first-order condition

**(eq. 7)** $\qquad \langle g, g_k \rangle := \left\langle \dfrac{\partial \ell(f_\theta(x,t),y)}{\partial \theta},\ \dfrac{\partial \ell(f_\theta, \mathcal{M}_k)}{\partial \theta} \right\rangle \ge 0 \quad \text{for all } k < t.$

**The projection (eq. 8)**: $\min_{\tilde g} \tfrac12 \lVert g - \tilde g \rVert_2^2$ subject to $\langle \tilde g, g_k \rangle \ge 0$ for all $k < t$. Writing $G = (g_1, \dots, g_{t-1})$, this is the primal quadratic program $\min_z \tfrac12 z^\top z - g^\top z + \tfrac12 g^\top g$ subject to $Gz \ge 0$, in $p$ variables (the parameter count, 1,109,240 for the CIFAR-100 network). By quadratic-programming duality (eqs. 9–10, Dorn 1960) the same solution comes from

**(eq. 11)** $\qquad \min_v\ \tfrac12 v^\top G G^\top v + g^\top G^\top v \quad \text{subject to } v \ge 0,$

a program in $t - 1 \ll p$ variables, after which $\tilde g = G^\top v^\star + g$. Geometrically $\tilde g$ is $g$ plus a non-negative combination of past-task gradients — the projection can only *add* past-task direction, never remove it. "In practice, we found that adding a small constant $\gamma \ge 0$ to $v^\star$ biased the gradient projection to updates that favoured beneficial backwards transfer." The grid in Appendix A sweeps $\gamma \in \{0.0, 0.1, \dots, 1.0\}$ and selects $0.5$ on all three datasets; the code realizes it as the lower bound $v \ge 0.5$ inside the program rather than as a post-hoc addition (see `repo_analysis.md`).

**A causal reading (Section 3, "A causal compression view")**: a predictor that never increases any past task's loss is learning "the subset of correlations common to a set of distributions", which the authors connect to invariant prediction in causal inference. This is offered as interpretation, not tested.

**Algorithm 1** is Steps 1–7 above with the evaluation loop: for each task $t$, for each $(x, y)$ in its training continuum, update memory, compute $g$ and all $g_k$, project, step; after the task, write row $t$ of $R$.

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Every prior method the paper positions against treats the past as a *target to hold*: EWC and synaptic intelligence pull parameters back toward their old values weighted by importance; iCaRL and learning-without-forgetting pull old predictions back toward their old values by distillation. GEM treats the past as a *constraint direction*: the update is free to move anywhere in parameter space as long as no stored past loss rises to first order. Because the constraint is an inequality, past losses may fall — the paper's headline distinction is that GEM "allows positive backward transfer" where the others cannot by construction. Structurally, GEM is also the only one of these whose extra work is per-update and scales with the number of tasks rather than with the number of parameters.

### b) Key innovations and their significance

1. **The gradient-angle constraint and its dual solution** — *significant*. Turning "do not forget" into a per-step feasibility test with a cheap violation check (dot products) and an expensive fix only on violation (a small quadratic program) is the reusable idea; the dual trick is what makes it affordable at $p \approx 10^6$.
2. **Inequality rather than equality treatment of the past** — *notable*. Small change in formulation, but it is what makes positive backward transfer observable, and the paper shows it (BWT $> 0$ on two of three streams).
3. **The $R$ matrix and the ACC / BWT / FWT triple** — *notable*. Not mathematically deep, but it is the first place these three are separated and reported together in this literature, and it fixes the protocol (single pass, evaluate all tasks at each boundary, record the untrained baseline) that later work adopted. Reviewer 2 (section 7) disputed the novelty of the surrounding framing but not of the metrics.
4. **Task descriptors as a first-class input** — *incremental*; the paper itself uses only integer descriptors and leaves structured ones for future work.

### c) Applicability

GEM excels when tasks are *related* enough that a shared network is right and *distinct* enough that a plain shared network forgets: MNIST rotations is the ideal case (positive backward transfer, forward transfer of $0.66$). It struggles when the memory budget is tiny relative to the task's variability (CIFAR-100 at 200 stored examples: ACC $0.487$, Table 2) and when the number of tasks is large, since every update pays one backward pass per past task. It requires a task descriptor at training *and* test time in the CIFAR-100 experiment (the head is selected by $t$), which is the task-incremental rather than class-incremental setting.

### d) Comparison table

| Method | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| **GEM** | Past can improve (inequality constraint); ACC at or near the iid oracle on MNIST rotations (Table 3); ACC rises monotonically with memory, no tuning cliff (Table 2); cheaper than EWC on permutations (77 s vs 179 s, Table 1) | One backward pass per past task per step; stores raw examples; first-order check can miss curvature; margin $\gamma$ is an unreported, tuned constant; no forward term at all | Coreset memory; amortize past gradients across steps; structured descriptors for forward transfer |
| single (one shared net) | Fastest (11 s); strong current-task learning | Forgets heavily (BWT $-0.20$ on permutations) | — |
| independent (one net per task, $T\times$ smaller) | Zero forgetting by construction | Zero transfer; each net too small (ACC $0.45$ on permutations) | Warm-start from previous net (the paper's `finetune`) |
| multimodal (per-task input layer) | Near-zero forgetting on MNIST | Not applicable when inputs share a representation (skipped on CIFAR-100); forgets under repeated passes ($-0.28$ BWT at 5 epochs) | — |
| EWC | No stored examples in principle | Forgets nearly as much as single here (BWT $-0.17$ / $-0.20$); slowest on MNIST (179 s); regularization strength swept over four orders of magnitude | Better Fisher estimate than one memory batch |
| iCaRL | Designed for class-incremental; competitive at small memory | Pins old predictions, so BWT $\le 0$; ACC saturates with memory (0.508 at 5,120) | — |

---

## 4. Experimental Validation

### a) Experimental design

Three streams of $T = 20$ tasks each, every example seen once, mini-batch 10, plain stochastic gradient descent, hyper-parameters grid-searched per method and dataset (Appendix A) with the best reported:

| Stream | Task construction | Per task | Network |
|---|---|---|---|
| MNIST permutations | fixed random pixel permutation per task; input distributions unrelated | 1,000 training examples, 10 classes | 2 hidden layers × 100 units |
| MNIST rotations | digits rotated by an angle fixed per task, angles spanning 0°–180° across the 20 tasks | 1,000 training examples, 10 classes | same |
| incremental CIFAR-100 | each task is a disjoint block of 5 of the 100 classes | 2,500 training examples, 5 classes | ResNet-18 with $3\times$ fewer feature maps, output masked to the task's classes |

Evaluation on the full test partition of each task after each task boundary (and every 100 mini-batches in the code). Baselines: single, independent, multimodal (MNIST only), EWC, iCaRL (CIFAR-100 only); GEM, EWC and iCaRL share single's architecture plus a memory. GEM's memory: 5,120 examples total, 256 per task — that is 25.6% of the MNIST training data per task and 10.2% of the CIFAR-100 data per task.

### b) Key results

All final numbers below are from Appendix B's matrices (the paper's Figure 1 plots them without printing values):

| Stream | Method | ACC | BWT | FWT |
|---|---|---|---|---|
| MNIST permutations | single | 0.602 | −0.198 | 0.009 |
| | EWC | 0.619 | −0.165 | 0.005 |
| | multimodal | 0.756 | −0.028 | 0.006 |
| | **GEM** | **0.826** | **+0.025** | 0.009 |
| MNIST rotations | single | 0.531 | −0.090 | 0.425 |
| | EWC | 0.546 | −0.205 | 0.552 |
| | multimodal | 0.759 | −0.024 | 0.018 |
| | **GEM** | **0.861** | **+0.005** | **0.665** |
| CIFAR-100 | single | 0.463 | −0.123 | −0.001 |
| | EWC | 0.498 | −0.080 | −0.008 |
| | iCaRL | 0.546 | −0.083 | 0.000 |
| | **GEM** | **0.678** | **+0.004** | −0.008 |

(Appendix B's BWT and FWT are averaged over $T$ rather than $T-1$; see `repo_analysis.md`. The ordering is unaffected.)

1. **GEM has the best ACC on all three streams and the only positive BWT anywhere in the study.** On CIFAR-100 the gap to the next best (iCaRL) is 13 points.
2. **GEM reaches the iid oracle** (Table 3, MNIST rotations): single trained on shuffled data from all tasks gives ACC 0.83 / 0.87 / 0.89 at 1 / 2 / 5 epochs; GEM gives 0.86 / 0.88 / 0.89 with BWT +0.05 / +0.02 / −0.02. Every other method *loses* accuracy as epochs increase (single 0.53 → 0.43, multimodal 0.76 → 0.59) because repeated passes deepen forgetting.
3. **ACC is monotone in memory size and GEM beats iCaRL at every size** (Table 2, CIFAR-100): GEM 0.487 / 0.579 / 0.633 / 0.654 versus iCaRL 0.436 / 0.494 / 0.500 / 0.508 at 200 / 1,280 / 2,560 / 5,120 stored examples.
4. **Cost sits between the trivial baselines and EWC** (Table 1, CPU seconds): permutations 77 s (GEM) vs 179 s (EWC) vs 11 s (single); rotations 135 s vs 169 s vs 11 s.
5. **First-task accuracy over the stream** (Figure 1 right): GEM's curve is flat or rising on all three streams while single and EWC decay; on CIFAR-100 GEM's first-task accuracy ends *above* where it started.

### c) Where it shines

MNIST rotations, where the tasks are related by a smooth transformation: the constraint keeps earlier angles' accuracy while the shared representation improves for all, giving both the best ACC and a forward transfer of 0.66 — a network that has seen 19 rotations is already at 66 points above chance on the twentieth before seeing it. The paper explains the CIFAR-100 gains by memory: Table 2 shows the gap to iCaRL widening with memory size, so GEM converts stored examples into accuracy where distillation saturates.

### d) Limitations

*Acknowledged (Section 6).* Integer descriptors only, so no zero-shot forward transfer is attempted; memory is the last $m$ examples, no coreset; one backward pass per past task per iteration.

*Implicit.*
- **Twenty tasks, one order, one seed.** No error bars anywhere; `run_experiments.sh` fixes `--seed 0`. Reviewer 1 asked whether the multimodal-versus-GEM BWT difference on rotations is significant; the paper does not answer.
- **The margin $\gamma$ is never ablated.** The grid runs 0.0 to 1.0 and 0.5 wins everywhere, but no table shows what $\gamma = 0$ (the pure projection the derivation describes) achieves. With $\gamma = 0.5$, every projection step adds at least half of every past-task gradient, which is a replay component on top of a projection; how much of the positive BWT is projection and how much is this replay is not separable from the reported results.
- **Task-incremental, not class-incremental, on CIFAR-100.** The task descriptor selects the head at test time for every method including iCaRL (the code masks all logits outside the task's five classes). That is the easier protocol and not the one iCaRL was designed for — Reviewer 1's first question, unanswered in the text.
- **Memory is a quarter of the data.** 256 of 1,000 examples per MNIST task are retained. The paper's Table 2 shows ACC falling to 0.487 at 200 total (10 per task) on CIFAR-100, so the headline numbers depend on a generous budget.
- **The first-order check is only as good as the linearization and the memory.** The paper assumes both; neither is tested. Nothing measures how often the projection fires or how far $\tilde g$ is from $g$.
- **The current-task learning cost is not discussed.** The feasible cone is the intersection of $t-1$ half-spaces and cannot grow as tasks accumulate; whether that impairs learning of *new* tasks is never asked. It can be read from the diagonals of Appendix B: mean $R_{i,i}$ is 0.801 for GEM vs 0.800 for single on permutations, 0.856 vs 0.620 on rotations, 0.674 vs 0.586 on CIFAR-100, and GEM's last five diagonals exceed its first five on every stream — so within 20 tasks there is no visible cost, though single uses a different learning rate on each stream, so this is not a clean comparison.
- **The stream is non-stationary but homogeneous** — every task is ten-way or five-way image classification with the same loss. Nothing recurs, nothing shifts in kind.

---

## 5. Reproduction and Application

### a) Open source?

Yes: https://github.com/facebookresearch/GradientEpisodicMemory, PyTorch, 1,490 lines, one script reproduces everything (`run_experiments.sh`: build the three datasets, run all model/dataset pairs, plot). Reproduction steps: download raw MNIST and CIFAR-100 (`data/raw/raw.py`), generate the three task files, run `main.py --model gem --n_memories 256 --memory_strength 0.5 --lr 0.1` per dataset. Details in `repo_analysis.md`.

### b) Implementation details requiring attention

- The quadratic program is solved with the `quadprog` package on the $ (t-1) \times (t-1)$ Gram matrix $G G^\top$, symmetrized and regularized with $10^{-3} I$ for positive-definiteness (the repo's last commit is titled "Fix PSD bug for quadprog").
- The projection runs only when at least one dot product is negative; otherwise the raw gradient is used. Selected values: learning rate 0.1, 256 memories per task, margin 0.5, on all three streams.
- Memory is a ring buffer per task; because each task has more examples than slots, the buffer ends up holding the last 256 examples, matching the paper.
- The CIFAR-100 head is a single 100-way output with logits outside the current task's five classes set to $-10^{11}$ before the loss — the "linear classifier per task" is a mask, not separate layers.
- EWC's Fisher is estimated from one batch of stored examples at the task boundary (squared gradients), not from the full task.

### c) Transferability

The method needs gradients, so it transfers to any differentiable learner trained on a task sequence: language-model fine-tuning, reinforcement-learning policies, multi-task regression. Its cost is linear in the number of tasks per update, so beyond a few dozen tasks one would sample a subset of past tasks per step or batch the memories. The evaluation protocol (Step 7) needs no gradients and transfers to anything scored per task.

### d) Transfer to our own work

**Where GEM sits against our claim.** `docs/framing.md` writes the per-edit objective as $\max_{a_t} r_t(h_t, a_t)$ subject to $\Phi(h_{t+1}) \ge \Phi_{\min}$ and leaves open whether the retention constraint at commit is hard or soft. GEM is that constrained form with the future term deleted and the retention term made *hard and per-step*: eq. 6 maximizes current-example performance subject to "no past task's memory loss rises". There is no forward term of any kind — forward transfer is measured (FWT) but never enters the objective; with integer descriptors the authors say one "cannot expect significant positive forward transfer" and stop there. In our vocabulary GEM is the horizon $\gamma = 0$ case *with* a retention constraint, which is a different point from the self-improving harnesses in this index, which are $\gamma = 0$ *without* one. Its evidence is that the constraint alone recovers the iid oracle on a related-task stream (Table 3) — with a memory that holds a quarter of the data.

**The three things that transfer.**

1. *The evaluation matrix is our instrument for two of the three future costs.* $R_{i,j}$ under GEM's protocol is exactly score-before-update: row $i$ is written after task $i$ finishes and before task $i+1$'s first example, and $R_{i-1,i}$ is accuracy on task $i$ with nothing of task $i$ seen. BWT is retention of earlier tasks; FWT is generalization to later tasks. Two adaptations are needed. The baseline $\bar b$ (an untrained network) has no harness meaning; the right reference is the base, unedited harness scored on task $i$, so harness FWT becomes $R_{i-1,i} - R_{0,i}$. And the third cost in our claim, the ability to keep patching, is the *diagonal* $R_{i,i}$ plotted against $i$ — GEM never reports it, but the matrix contains it, and the harness version should be a first-class curve. Figure 1 right (first-task accuracy over the whole stream) is the retention plot to copy.

2. *The two-stage gate: cheap check always, expensive fix only on violation.* GEM tests $t-1$ dot products every step and solves the quadratic program only when one is negative. For a harness the expensive operation is re-running the anchor set; the cheap check is whatever surrogate we trust — the critic $\hat\Phi(h)$, or a retrieval-overlap test between the edit and the anchors. GEM is precedent that a first-order surrogate gating an exact correction is enough to hold retention at zero loss over 20 tasks, *and* it is a warning: the surrogate is never validated in the paper, and a harness edit has no linearization, so the surrogate for us must be learned or measured rather than derived.

3. *Inequality, not equality.* GEM's decisive design choice is that the past may improve. That is the same fault line as the framing amendment recorded for Harness Continual Learning (restoration edits inadmissible under its commit gate): a gate that only forbids regressions and never demands invariance is what makes positive BWT observable at all. GEM's +0.025 and +0.004 BWT are small but they are the only positive values in the study, and they are the existence proof that a retention constraint and improvement of old tasks are compatible.

**What does not transfer.** The constraint's *mechanism* is entirely weight-space: gradient angles, a cone in $\mathbb{R}^p$, a dual program in $t-1$ variables. A harness edit is a text diff with no gradient, so eq. 7 has no analogue and the anchor set must be re-scored, not differentiated. The memory of raw $(x, y)$ pairs is the *shape* of an anchor set — previously seen tasks retained for re-checking — but the paper's selection rule (the last $m$ examples) is what the authors themselves flag as the weak point; the anchor set's rule "previously *passed* tasks, grown from the stream" is a different and more defensible rule. The margin $\gamma = 0.5$ is a tuned constant with no ablation; nothing in it should be imported. And GEM's stream — 20 homogeneous classification tasks, one order, one seed, nothing recurring — is not a task stream in our sense, so its results say nothing about heterogeneity or about whether a constraint like this one degrades the ability to learn new tasks over hundreds of edits. The diagonal computation in section 4d suggests it does not over 20 tasks; that is the extent of the evidence.

---

## 6. Summary

### a) One-sentence core idea

Keep a few examples of each past task and project every update so no past task's loss increases, letting the past improve but never regress.

### b) Quick-reference pipeline

1. As tasks arrive one after another, save a fixed number of the most recent examples from each task.
2. For every new update, compute the usual gradient on the current example and also the gradient of the loss on each saved past task's examples.
3. If the new gradient points against any past task's gradient (negative dot product), the update would make that task worse; replace it with the nearest gradient that points against none of them, solved as a small optimization with one variable per past task.
4. Apply the (possibly corrected) gradient step.
5. After each task, test on every task seen so far and record the full table of accuracies; report the final average, how much old tasks changed since they were learned, and how well new tasks were handled before any training on them.

---

## 7. Reviewer Reception

### a) Link and outcome

No OpenReview page exists — NIPS 2017 ran its reviews on a closed system, and an OpenReview title search on 2026-09-16 returned only a DBLP mirror record and an unrelated 2020 ICLR submission titled "Revisiting Gradient Episodic Memory for Continual Learning". The official venue published the three referee reports at https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Reviews.html. Outcome: accepted. No numeric scores, no author rebuttal, and no meta-review are shown on the page.

### b) Main criticisms

1. **The framing overclaims novelty** (Reviewer 2, the most critical report). "The way the paper presents itself is misleading to the readers about where the core innovation of the paper is." The reviewer objected to the coinage "continuum learning" as "a very straightforward instantiation of lifelong learning" and rejected the claim that task descriptors are novel: even Caruana's multitask learning "must send a task identifier with the training data". The reviewer nonetheless called the approach itself "interesting" with "reasonable baselines".
2. **The iCaRL comparison may not be like-for-like** (Reviewer 1). iCaRL "would be able to solve a full 100-classes problem, while GEM would solve only any of 20 5-classes problems" — was iCaRL "ever given a hint which of 5 classes to look at?" — and the two were run at different memory sizes. The reviewer also asked whether the multimodal-versus-GEM difference on rotations is significant, whether baselines shared GEM's architecture, and for a batch-trained upper bound.
3. **Missing cost and scale accounting** (Reviewer 3). "How long do the various methods take?" given the inner optimization, and "exactly how big your CIFAR100 network is" since the argument rests on $t \ll p$. Reviewer 3 also wanted a no-single-pass benchmark accuracy to locate the method, while finding the idea "quite interesting and worthy of inclusion".
4. **A probable typo in eq. 6** (Reviewer 1): "in its current form I don't see how it captures this idea."

### c) Author rebuttal and resolution

No rebuttal is published. Comparing the referee requests with the camera-ready shows what was addressed: Table 1 (CPU seconds per method) answers Reviewer 3's timing question; the sentence giving $p = 1{,}109{,}240$ for the CIFAR-100 network answers the size question; Table 3's "single, shuffled data" row is the batch-trained oracle both Reviewers 1 and 3 asked for; Table 2 runs GEM and iCaRL at matched memory sizes, answering Reviewer 1's second point; and the "same architecture as single, plus episodic memory" sentence in Section 4.3 answers the fifth. The class-incremental versus task-incremental objection to the iCaRL comparison (Reviewer 1's first point) is not addressed in the text; the code shows the task descriptor selects the head for every method. Whether the "continuum learning" terminology was softened cannot be determined from the final text, which still uses "continuum of data" throughout; the term does not appear as the name of a setting.

### d) Net takeaway

The reviews contest the paper's *framing* — the vocabulary and the claim of novelty for descriptors and setting — and not its method or results; two of three referees explicitly called the idea interesting. The camera-ready visibly absorbed the concrete requests (timing, oracle row, matched memory sizes, network size). Trust the mechanism and the metric protocol; read the CIFAR-100 comparison to iCaRL as task-incremental and therefore favourable to GEM; treat the framing sentences about a new setting as the part the referees pushed back on.
