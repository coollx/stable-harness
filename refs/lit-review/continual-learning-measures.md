# Measures of continual learning across the references

Written 2026-09-19. Scope: a map of what the seven continual-learning references in this folder measure, how each measure is defined, and which pairs are the same quantity under different names. No recommendation is made here about which measures this project adopts; that belongs to the task that builds the evaluation.

## Question

When a learner takes tasks one after another, the literature reports many numbers with overlapping names: forgetting, backward transfer, forward transfer, intransigence, plasticity loss, average accuracy, evolution gain, dead units, effective rank. Which of these are distinct quantities, which are the same quantity under two names, and how does each relate to the two abilities every method trades against each other: stability, the ability to keep what was learned, and plasticity, the ability to learn something new?

## Papers covered

- [GEM](../related-work/GEM/paper_analysis.md): Lopez-Paz and Ranzato, *Gradient Episodic Memory for Continual Learning*, NeurIPS 2017 ([arXiv:1706.08840](https://arxiv.org/abs/1706.08840)). Source of the accuracy matrix protocol and of average accuracy, backward transfer, and forward transfer.
- [Riemannian-Walk](../related-work/Riemannian-Walk/paper_analysis.md): Chaudhry, Dokania, Ajanthan and Torr, *Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence*, ECCV 2018 ([arXiv:1801.10112](https://arxiv.org/abs/1801.10112)). Source of the forgetting measure with a best-ever reference and of intransigence.
- [HCL](../related-work/HCL/paper_analysis.md): *Harness Continual Learning: Continual Adaptation Beyond Model Parameters*, 2026 ([arXiv:2608.19013](https://arxiv.org/abs/2608.19013)). Reports average accuracy and forgetting for a harness rather than a network.
- [Loss-of-Plasticity](../related-work/Loss-of-Plasticity/paper_analysis.md): Dohare et al., *Loss of plasticity in deep continual learning*, Nature 2024. Origin of the term this project borrows; measures plasticity as a curve against stream position and introduces three internal-state statistics.
- [Understanding-Plasticity](../related-work/Understanding-Plasticity/paper_analysis.md): Lyle et al., *Understanding Plasticity in Neural Networks*, ICML 2023 ([arXiv:2303.01486](https://arxiv.org/abs/2303.01486)). The only formula for plasticity in the folder, plus a falsification test of the internal-state statistics.
- [Dormant-Neurons](../related-work/Dormant-Neurons/paper_analysis.md): Sokar et al., *The Dormant Neuron Phenomenon in Deep Reinforcement Learning*, ICML 2023 ([arXiv:2302.12902](https://arxiv.org/abs/2302.12902)). Defines a per-unit dormancy score, the most developed of the internal-state statistics.
- [AgentStream](../related-work/AgentStream/paper_analysis.md): *AgentStream: How Well Do Self-Evolving LLM Agents Perform Under Streaming Tasks?*, 2026 ([arXiv:2608.00155](https://arxiv.org/abs/2608.00155)). Source of evolution gain, the only whole-stream measure against a no-adaptation control.

## Synthesis

### The shared object: a table of scores

Every outcome measure below is read off one table. After the learner finishes task $i$, test it on task $j$ and record the score $R_{i,j}$. GEM writes this as a matrix $R \in \mathbb{R}^{T \times T}$ over $T$ tasks and also records $\bar b$, the score vector of an untrained network. Riemannian Walk writes the same entries as $a_{k,j}$ and only fills $j \le k$. HCL writes $R_{s,j} = \mathrm{Eval}(H^{(s)}, D_j^{\mathrm{test}})$ for $j \le s$ with $H^{(s)}$ the harness after task $s$. Three protocols, one table. The measures differ in which cells they read and what they compare them to.

### Two families

The references split into two families that answer different questions. Outcome measures read test scores and say how well the system performs. Internal-state statistics read the learner's internals and are proposed as causes or early warnings of the outcomes. Only the plasticity papers use the second family; the two harness papers report outcome measures only.

### Outcome measures, consolidated to five

**1. Forgetting: how much of an earlier task's accuracy the learner has lost by the end.** A stability measure. Two conventions differ only in the starting point. GEM's backward transfer (eq. 3) starts from the score right after the task was learned:

$$\mathrm{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1}\left(R_{T,i} - R_{i,i}\right),$$

with higher better, negative values meaning forgetting and positive values meaning later tasks improved earlier ones. Riemannian Walk's forgetting (eq. 3) starts from the best score the task ever reached:

$$F_k = \frac{1}{k-1}\sum_{j=1}^{k-1}\left(\max_{l \in \{1,\dots,k-1\}} a_{l,j} - a_{k,j}\right),$$

with lower better. HCL's $\mathrm{Fgt}_T$ (eq. 15) is the same formula as Riemannian Walk's, written with $\max_{r \in \{j,\dots,T\}} R_{r,j}$. The two conventions coincide exactly when every task peaked at the moment it was learned; otherwise the best-ever version is larger, because it also counts gains that a later task produced and a still-later task took away. Riemannian Walk's worked example: task scores of 0.7, 0.8, 0.6, 0.5 across four stages give backward transfer of $-0.2$ and forgetting of $0.3$. Same quantity, second convention stricter, and the two are not interchangeable across papers.

**2. Forward transfer: how much having learned earlier tasks helps on a new task before the learner has trained on it.** A plasticity measure read one cell above the diagonal. GEM (eq. 4):

$$\mathrm{FWT} = \frac{1}{T-1}\sum_{i=2}^{T}\left(R_{i-1,i} - \bar b_i\right),$$

the score on task $i$ after finishing task $i-1$, minus what a randomly initialized network scores on task $i$. The comparison point is a floor, not a ceiling. Only GEM reports it; Riemannian Walk borrows the words "positive forward transfer" and "negative forward transfer" as sign labels for its intransigence, which is a different measurement (see below), and says GEM's forward transfer is "complementary to our approach". AgentStream's sequential stream shape is described as measuring forward transfer across a curriculum but it reports no cell-level version of it.

**3. Intransigence: how much having learned earlier tasks hurts the learner's ability to learn a new task once it trains on it.** A plasticity measure read on the diagonal and compared against a learner that never went through the stream. Three papers use this shape with three different comparison learners, and only one of them uses the name.

- Riemannian Walk (eq. 4): $I_k = a_k^* - a_{k,k}$, where $a_k^*$ is the score on task $k$ of a model trained in one ordinary run on the union of all data from tasks 1 through $k$, and $a_{k,k}$ is the incremental learner's score on task $k$ right after training on it. Lower is better. The comparison model is rebuilt for each $k$ and has seen exactly the same examples, so the only difference is order and constraint. Reported at the final task only, never as a curve. The paper allows a cheaper stand-in for $a_k^*$ when the joint model is too expensive, provided it is named.
- Loss of Plasticity gives no formula. In its class-incremental CIFAR-100 experiment it reports the streamed network's accuracy on the current class set minus that of a network retrained from scratch on exactly those classes, and plots it against the number of classes seen. In its other experiments the comparison is a linear network or a fully reinitialized one. The curve, not the number at one point, is what the paper calls loss of plasticity.
- Understanding Plasticity (eqs. 4 and 5): fix an optimizer $\mathcal{O}$ with a fixed step budget and a distribution $\mathcal{L}$ of synthetic regression targets. Plasticity of parameters $\theta_t$ is $\mathcal{P}(\theta_t) = b - \mathbb{E}_{\ell \sim \mathcal{L}}[\ell(\mathcal{O}(\theta_t, \ell))]$, with $b$ the loss of a constant predictor, and plasticity loss along a trajectory is $\mathcal{P}(\theta_t) - \mathcal{P}(\theta_0)$, the current checkpoint against a freshly initialized network of the same architecture. This is the only closed-form definition of plasticity in the folder. The probe task is deliberately synthetic so the number is not about any one future task.

All three are the same subtraction: streamed learner minus stream-free learner, on the task in front of it. They differ in the comparison learner (jointly trained, retrained from scratch, or freshly initialized on a probe) and in whether the trend against stream position is plotted (the two plasticity papers plot it; Riemannian Walk does not). Riemannian Walk adds one caution that carries to the others: a learner that has abandoned all old tasks can score negative intransigence on the newest one, so the number is misleading unless forgetting is shown next to it.

**4. Average accuracy: how well the learner does on every task seen so far, averaged at one point in the stream.** GEM's $\mathrm{ACC} = \frac{1}{T}\sum_{i=1}^{T} R_{T,i}$ (eq. 2), Riemannian Walk's $A_k = \frac{1}{k}\sum_{j=1}^{k} a_{k,j}$, HCL's $\mathrm{Avg}_T$ (eq. 15) are identical. Every paper that reports it also says it is insufficient alone: GEM prefers the model with larger backward and forward transfer when accuracies tie, and Riemannian Walk states it "does not provide any information about forgetting or intransigence profile". It mixes the two failures and cannot separate them.

**5. Evolution gain: how much better the adapting system scores over the whole stream than the same system with adaptation switched off.** AgentStream (eq. 1): $\Delta = \mathrm{Perf}(M, Q, S) - \mathrm{Perf}(M, Q, \emptyset)$, where $\mathrm{Perf}$ is accuracy over a stream $Q$ with model $M$ and evolving state $S$, each task scored before the update it triggers, and $\emptyset$ is the same model with no state. This is the only measure in the folder that scores the whole stream against a control that never adapts, and the only one that never re-tests an old task: each task appears once, so forgetting is invisible to it by construction. AgentStream also reports cumulative accuracy against task index, which is the running form of the same sum.

### Internal-state statistics, four

These are functions of the learner's state, taken before a task rather than from test scores. All come from the plasticity papers; neither harness paper has an analogue.

1. **Silent units.** Loss of Plasticity counts units whose activation is zero on every example of a fixed sample of 2,000 taken before each task, softened in its class-incremental code to units active on under 1% of samples. Dormant Neurons (Definition 3.1) generalizes this to a layer-normalized score, $s_i^\ell = \mathbb{E}_x|h_i^\ell(x)| \,/\, \frac{1}{H^\ell}\sum_k \mathbb{E}_x|h_k^\ell(x)|$, a unit's mean absolute activation divided by the layer's mean, and calls a unit dormant when the score is at or below a threshold $\tau$, used at 0, 0.025 and 0.1 in different sections.
2. **Effective rank.** Loss of Plasticity (eq. 2): for a representation matrix $\Phi$ with singular values $\sigma_k$ and $p_k = \sigma_k / \|\sigma\|_1$, $\mathrm{erank}(\Phi) = \exp\{-\sum_k p_k \log p_k\}$, a continuous count between one and the rank of how many independent directions the hidden representation spans. Understanding Plasticity measures rank of the weight matrices and rank of the learned features separately.
3. **Weight norm.** Loss of Plasticity: sum of absolute weights divided by their count. Understanding Plasticity: parameter norm.
4. **Loss curvature.** Understanding Plasticity reads the eigenvalue spectrum of the Hessian and the gradient covariance matrix, and proposes curvature as the explanation that survives its own test.

Understanding Plasticity's falsification design (Figure 3, 128 deep Q-network agents) is the one controlled test of these statistics in the folder. For each of parameter norm, weight-matrix rank, dead-unit count and feature rank, there is one experimental setting where the statistic correlates positively with plasticity loss and another where it correlates negatively; the sign flips with observation space, architecture, or reward structure. The paper's conclusion is that none of the four is a causal explanation or a reliable stand-in for the outcome measures. Loss of Plasticity, one year later in publication, reports the first three as consistent correlates in its own settings without such a test; the two papers therefore conflict on whether these statistics mean anything outside the setting they were measured in. Dormant Neurons is the one paper that validates its statistic by intervention: pruning the units it flags leaves performance unchanged, and recycling them restores the ability to fit a new target, on two Atari games.

### Where the field agrees and where it conflicts

Agreed by every paper that measures both: forgetting and the diagonal-side measures move in opposite directions when a retention mechanism is turned up. Riemannian Walk's Table 3 shows it in weight space over a five-order-of-magnitude sweep of regularization strength; HCL's tolerance sweep shows forgetting rising monotonically while final average accuracy peaks at an intermediate setting. Agreed by GEM, Riemannian Walk and HCL: average accuracy alone cannot tell the two failures apart. Agreed by the three plasticity papers: plasticity is measured against a learner that did not go through the stream, never in absolute terms.

Conflicts: the two forgetting conventions coexist without being distinguished; the words "forward transfer" name two different measurements in GEM and Riemannian Walk; "plasticity" has a formula in one paper, a protocol in two, and no definition at all in HCL, which uses the word throughout and names two configurations after it; and the internal-state statistics are endorsed by one paper and disqualified by another.

### How the measures relate to this project's quantities

Stated as a mapping only. `docs/framing.md` defines harness plasticity $\Phi(h)$ as the future stream performance achievable from harness $h$, discounted by the horizon $\gamma$, and names three future quantities that current-batch scoring cannot see: retention of earlier tasks, generalization to later tasks, and the ability to keep patching. Forgetting is the loss of the first. Forward transfer is the second, read one task ahead against a floor. Intransigence is the third, read one task ahead against a ceiling. Evolution gain is the undiscounted sum of stream performance under score-before-update against a no-adaptation control, which is the shape of the project's claim with $\gamma = 1$ and no plasticity-preserving comparison. Every plasticity measure above looks exactly one task ahead with no discounting, so each is the one-step case of $\Phi(h)$. None of the seven papers measures anything with a horizon longer than one task. The Terms table in `docs/framing.md` currently lists none of forgetting, forward transfer, or intransigence; the analyses under `related-work/` use the words with attribution to their papers.
