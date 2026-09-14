# Loss of plasticity in deep continual learning

- **Original title**: Loss of plasticity in deep continual learning
- **Authors**: Shibhansh Dohare (corresponding), J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton
- **Affiliations**: Department of Computing Science, University of Alberta, Edmonton, Canada; Canada CIFAR AI Chair, Alberta Machine Intelligence Institute (Amii), Edmonton, Canada
- **Venue / year**: Nature **632**, 768–774 (2024), issue 8026. Received 11 August 2023, accepted 12 June 2024, published online 21 August 2024. Open access (CC BY 4.0).
- **DOI**: [10.1038/s41586-024-07711-7](https://doi.org/10.1038/s41586-024-07711-7)
- **Source URL**: https://www.nature.com/articles/s41586-024-07711-7 · local copy `paper.pdf` (26 pages: 7 pages main text, Methods, Extended Data Figs. 1–5, Extended Data Tables 1–5)
- **Earlier preprint**: [arXiv:2306.13812](https://arxiv.org/abs/2306.13812), *Maintaining Plasticity in Deep Continual Learning* (2023). The Nature version adds the class-incremental CIFAR-100 experiment and all the reinforcement-learning experiments, and simplifies the algorithm; see section 7.
- **OpenReview**: no public page found (searched 2026-09-14; see section 7). Nature published the complete referee reports and author rebuttals as a [Peer Review File](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07711-7/MediaObjects/41586_2024_7711_MOESM1_ESM.pdf), which is summarized in section 7.
- **Code**: https://github.com/shibhansh/loss-of-plasticity (stated in the paper's Code availability section; analyzed in `repo_analysis.md`)

**Classification: empirical.** The paper's dominant contribution is a phenomenon — that standard deep-learning systems progressively lose the ability to learn new tasks — established across five protocols, many hyperparameter settings and 30–100 runs each. A mitigation (continual backpropagation) is proposed and evaluated, but the authors themselves state in the rebuttal that "the main purpose of our paper is to demonstrate loss of plasticity and not to benchmark all the possible solutions", and all three referees judged the demonstration strong and the proposed algorithm the weaker half. The method lens is folded into sections 2d, 3a (finding 5) and 5a rather than given its own framework.

**Why this reference matters here**: this is the origin of the term "loss of plasticity" that `docs/framing.md` borrows for harnesses. Section 5f states precisely which parts of the paper's definition and measurement design transfer to a harness of files and which do not.

---

## 1. Research Motivation and Questions

### a) Why was this study conducted?

Deep learning is used in two phases that almost never interleave: a training phase in which weights change, and a deployment phase in which they are frozen. The authors observe that this split is not a preference but a workaround — "with current methods, it is usually not effective to simply continue training a network when new data become available", and so "the most common strategy for incorporating substantial new data has been simply to discard the old network and train a new one from scratch on the old and new data together" (Introduction). When the network is a large language model and the new data is a substantial portion of the internet, "each retraining may cost millions of dollars in computation". The study asks whether the workaround is necessary: does a deep network that keeps training on a stream of new tasks keep learning as well as it did at the start?

### b) What is missing in current understanding?

Three concrete gaps. First, loss of plasticity had been "visible in some recent works" and "most recently ... begun to be explored explicitly", but never demonstrated systematically — prior demonstrations were incidental, at small scale, or confined to one setting. Second, the phenomenon was conflated with catastrophic forgetting; the authors insist the two are distinct — "loss of plasticity is different from catastrophic forgetting, which concerns poor performance on old examples even if they are not presented again". Third, no study had run long enough to see the effect: the authors argue that exposing it "requires ... three or four orders of magnitude more computation ... compared with what would be needed to train a single network", which is why standard practice, by stopping training early, "simply hides the issue" (rebuttal to the editor).

### c) Research questions

The paper does not number its hypotheses. Stated as they function in the design:

1. Do standard deep-learning systems lose the ability to learn new tasks when training continues over a long stream? (The central claim.)
2. Is this robust across architectures, optimizers, activation functions, step sizes, network widths, rates of distribution change, and both supervised and reinforcement learning? (The systematicity claim — what makes it a Nature paper rather than an observation.)
3. What accompanies loss of plasticity in the network's internal state? (The three correlates: dead units, growing weight magnitude, falling effective rank.)
4. Do commonly used deep-learning techniques — L2 regularization, dropout, normalization, Adam, shrink-and-perturb — mitigate it?
5. Does continually reinitializing a small fraction of the least-used units maintain plasticity indefinitely?

### d) Type of study

Failure-mode characterization with a controlled comparison across learning algorithms, plus a proposed mitigation benchmarked inside the same protocols. It is a measurement study: the object measured is a learning system's ability to learn, not a model's output quality.

---

## 2. Study Design

### a) Variables and conditions

**Independent variable of interest**: position in the task stream — the index of the task the learner is currently on. Everything the paper calls loss of plasticity is a decline of a performance measure against this axis.

**Second independent variable**: the learning algorithm — backpropagation (the base system), L2 regularization, shrink-and-perturb, dropout, online normalization, Adam, continual backpropagation, ReDo, full reinitialization, head resetting.

**Dependent variables**: (i) per-task performance on the task the learner is currently being trained on; (ii) three state statistics — fraction of dead or dormant units, average weight magnitude, and effective or stable rank of the representation.

**Held fixed / controls**: task difficulty is deliberately held constant or explicitly factored out. In Continual ImageNet "the difficulty of tasks remains the same over time", so a drop in performance "would mean the network is losing its learning ability, a direct demonstration of loss of plasticity". In class-incremental CIFAR-100 the task genuinely gets harder, so the measure is accuracy *relative to a network retrained from scratch on the same subset of classes* — "if the incrementally trained network performs better than retraining, then there is a benefit owing to training on previous classes, and if it performs worse, then there is genuine loss of plasticity". The same sequences of class pairs and the same hyperparameters are used across algorithms.

**Nuisance factors addressed by sweeping rather than fixing**: step size, network width, activation function, rate of distribution change, and each mitigation's own hyperparameters are all varied, so a finding cannot be blamed on one unlucky setting.

### b) Subjects of study

Five protocols, none of them a language model:

| Protocol | Learner | Stream | Scale |
|---|---|---|---|
| Continual ImageNet | 3 convolution-plus-max-pooling layers + 3 fully connected layers (Extended Data Table 3), SGD with momentum 0.9 | Binary classification of class pairs drawn from 1,000 downsampled 32×32 ImageNet classes; 1,200 training and 200 test images per task; 250 epochs, mini-batch 100 per task | up to 5,000 tasks; 30 runs per setting |
| Class-incremental CIFAR-100 | 18-layer residual network with heads added as classes arrive (Extended Data Table 1); batch normalization, data augmentation, L2, learning-rate schedule reset per increment | Start with 5 classes, add 5 at a time to 100; 450 train / 50 validation / 100 test images per class; 200 epochs per increment, 4,000 epochs total | 20 increments; 30 runs |
| Online Permuted MNIST | 3 hidden layers, 2,000 units per layer by default (also 100, 1,000, 10,000); single pass, no mini-batches | A new random pixel permutation per task, all 60,000 images shown once each, no signal at task switch | 800 tasks (150 for the width sweep); 30 runs |
| Slowly-Changing Regression | 1 hidden layer, 5 units; six activation functions | Input of 21 bits plus bias, 15 of them flipping; one flipping bit changes every 10,000 steps; target is a fixed random network of 100 linear threshold units | 3 million examples; 100 runs |
| Ant locomotion (reinforcement learning) | Proximal policy optimization; policy and value networks each 2 hidden layers of 256 units | Ant-v3 with ground friction resampled log-uniformly from [0.02, 2.00] every 2 million steps (non-stationary), plus stationary Ant-v3, Hopper-v3, Walker-v3 | 20M–100M steps; 30 runs (100 runs for Fig. 3c) |

### c) Measurement and instruments

**Plasticity is never given a formula.** It is operationalized throughout as *performance attainable on the task currently being learned, benchmarked against a learner that did not go through the stream*. Three concrete reference points do this work:

- **A linear network** (Continual ImageNet, Slowly-Changing Regression). Its accuracy is flat across tasks by construction, and the paper averages it over thousands of tasks to get a low-variance estimate. Falling to or below this line is "direct evidence that these methods do not work well in continual-learning problems".
- **A network retrained from scratch** on exactly the classes available at that point (class-incremental CIFAR-100). This is the cleanest reference because it cancels the rising intrinsic difficulty of the task.
- **Full reinitialization** (Online Permuted MNIST), whose performance "stays at the level of the first point".

**The three state measures**, all defined explicitly:

- *Dead units*: for ReLU networks, the number of units whose activation is zero for every example in a random sample of 2,000 taken before each task. In the class-incremental code the threshold is softened to units active on under 1% of samples ("dormant units", Fig. 2c).
- *Average weight magnitude*: the sum of absolute weights divided by the number of weights.
- *Effective rank* (eq. 2): for a matrix $\Phi \in \mathbb{R}^{n \times m}$ with singular values $\sigma_k$, $k = 1,\dots,q$, $q = \max(n,m)$, and $p_k = \sigma_k / \lVert\sigma\rVert_1$, $$\mathrm{erank}(\Phi) \doteq \exp\{H(p_1,\dots,p_q)\}, \qquad H(p_1,\dots,p_q) = -\sum_{k=1}^{q} p_k \log(p_k).$$ It is a continuous measure between one and the rank. A low effective rank means "many of the units in the hidden layer are not providing any useful information".
- *Stable rank* (used in Figs. 2d and 4b) is a cheaper variant: $\min\{k : \sum_{i\le k}\sigma_i / \sum_j \sigma_j > 0.99\}$.

All state measures are taken *before* each task, on the network as it stands, not on the training trace — they are functions of the learner's state.

### d) Experimental conditions and the proposed algorithm

Continual backpropagation is ordinary backpropagation plus one extra step per update: reinitialize a small fraction of the lowest-utility mature units in every layer.

**Contribution utility** (eq. 1) for the $i$th hidden unit in layer $l$ at time $t$, with decay rate $\eta$ fixed at 0.99 in all experiments: $$u_l[i] = \eta \times u_l[i] + (1-\eta) \times \lvert h_{l,i,t}\rvert \times \sum_{k=1}^{n_{l+1}} \lvert w_{l,i,k,t}\rvert.$$ In words: a unit is useful in proportion to how large its activation is and how heavily its consumers weight it; a unit with a small product "can be overwhelmed by contributions from other hidden units", so it is not useful to its consumer.

**Two hyperparameters carry the algorithm.** The *replacement rate* $\rho$ is the fraction of mature units reinitialized per step in every layer; the *maturity threshold* $m$ is the number of updates for which a newly created unit is protected from replacement — needed because a new unit's outgoing weights are set to zero (so it does not perturb the learned function) and therefore its utility starts at zero. Typical values are tiny: at $\rho = 10^{-5}$ in a 512-unit layer, "roughly $512\times10^{-5} = 0.00512$ units are replaced. This corresponds to roughly one replacement after every $1/0.00512 \approx 200$ updates or one replacement after every eight epochs on the first five classes". Values used: $\rho = 10^{-5}$, $m = 1{,}000$ for class-incremental CIFAR-100; a grid over $\rho$ and $m \in \{1{,}000, 10{,}000\}$ for the other problems.

The shipped algorithm reinitializes input weights from the original initialization distribution, sets outgoing weights to zero, and zeroes the unit's utility and age. For Adam it also resets that unit's optimizer moment estimates.

### e) Statistical and analytical approach

Means over independent runs with shaded bands: standard error of the mean at ±1 for the supervised experiments (30 runs, 100 for Slowly-Changing Regression), 95% bootstrapped confidence intervals for the reinforcement-learning experiments. No hypothesis tests, no multiple-comparison correction — the effects are large and the run counts high enough that the bands separate visibly. Hyperparameters for each algorithm were chosen by grid search on average accuracy over the full stream (Extended Data Table 2), with ten runs per grid point and twenty more added for the winner to reach thirty.

---

## 3. Findings and Observations (primary focus)

### a) Headline findings

**Finding 1 — Standard deep learning on Continual ImageNet degrades to the level of a linear network.**
*Evidence*: Fig. 1b. The convolutional network reaches up to 88% test accuracy on early tasks; by task 2,000 it has "lost substantial plasticity for all values of the step-size parameter", and for the larger step sizes it falls below the flat linear-network baseline drawn at roughly 77%. The paper reads this as a pattern rather than an artifact: "for a well-tuned network, performance first improves and then falls substantially, ending near or below the linear baseline. We have observed this pattern for many network architectures, parameter choices and optimizers."
*Conditions*: step sizes 0.01, 0.001, 0.0001; 30 runs each; task difficulty constant by construction.

**Finding 2 — A residual network in a realistic class-incremental setting loses plasticity too, and the size of the loss is equivalent to removing batch normalization.**
*Evidence*: Fig. 2b. Accuracy relative to a network retrained from scratch starts at about +2% (the benefit of prior training), crosses zero near 40 classes, and reaches about −5% at 100 classes. The paper's own calibration: "the accuracy of our incrementally trained base system was 5% lower than the retrained network (a performance drop equivalent to that of removing a notable algorithmic advance, such as batch normalization)". Continual backpropagation's final accuracy on all 100 classes was 76.13% (Methods).
*Conditions*: 18-layer residual network with batch normalization, data augmentation, L2 and a learning-rate schedule already in place — that is, the loss survives the full modern recipe.

**Finding 3 — Plasticity loss also appears in reinforcement learning, including in a stationary environment, and is catastrophic there.**
*Evidence*: Fig. 3c, slippery ant — standard proximal policy optimization climbs above 2,000 reward per episode and then decays toward zero over 20 million steps, while tuned, L2-regularized and continual-backpropagation variants hold a sawtooth around 4,000–5,000, recovering after each friction change (100 runs, 95% bootstrapped confidence interval). Fig. 4a, *stationary* Ant-v3 — "the average performance of PPO increased for about 3 million steps but then collapsed. After 20 million steps, the ant is failing every episode and is unable to move forwards." Tuned proximal policy optimization (Adam with $\beta_1 = \beta_2 = 0.99$) delays but does not prevent the collapse; L2 holds about 4,000; continual backpropagation plus L2 keeps improving past 5,000 through 50 million steps. Hopper-v3 and Walker-v3 mirror this (Extended Data Fig. 5a).
*Conditions*: standard algorithm, standard hyperparameters, no replay buffer. The authors' point is that "the current practice simply hides the issue by prematurely stopping training".

**Finding 4 — Three state correlates move together with the loss, and continual backpropagation is the only tested method that fixes all three.**
*Evidence*: (i) *dormant units* — in class-incremental CIFAR-100 the base system's dormant fraction rises past 50% by 100 classes, shrink-and-perturb reaches about 8%, continual backpropagation stays near 1% (Fig. 2c); in reinforcement learning standard and tuned proximal policy optimization both approach 50–60% while L2 and continual backpropagation stay under 5% (Fig. 4c); in Online Permuted MNIST "for the step size of 0.01, up to 25% of units die after 800 tasks" (Extended Data Fig. 3c). (ii) *weight magnitude* — rises from about 0.05 to 0.11 for standard and tuned proximal policy optimization over 50 million steps, stays flat near 0.03 for continual backpropagation and falls to about 0.005 under L2 (Fig. 4d). (iii) *rank* — stable rank falls from about 95 to 73 for the base system on CIFAR-100 versus about 93 for continual backpropagation (Fig. 2d), and from about 95 to 25 for proximal policy optimization versus about 82 for continual backpropagation (Fig. 4b).
*Conditions*: measured before each task on a fixed random sample of 2,000 examples (1,000 for the residual network).

**Finding 5 — Only methods that continually inject variability maintain plasticity; the standard toolkit does not, and some of it makes things worse.**
*Evidence*: Extended Data Fig. 4a, Online Permuted MNIST over 800 tasks, online classification accuracy. Continual backpropagation and shrink-and-perturb stay flat near 95%; L2 declines mildly to about 94%; backpropagation falls to about 93.5%; online normalization starts best and then drops below backpropagation; dropout falls to about 92.5% with the decline steepening as the dropout probability rises; Adam is the worst at about 91.3%, its "loss of plasticity ... categorized as catastrophic", with dead units that "plateau at around 60%" and a large early drop in effective rank. The abstract's conclusion: "Plasticity is maintained indefinitely only by algorithms that continually inject diversity into the network."
*Conditions*: all algorithms at their best grid-searched hyperparameters, 10–30 runs.

**Finding 6 — Selective reinitialization beats indiscriminate reinitialization, and the gain is in speed of relearning rather than in ceiling.**
*Evidence*: Methods. "Continual backpropagation achieves the best accuracy ten times faster than a fully reinitialized network on the 5,000th task of Continual ImageNet, ten epochs versus about 125 epochs." The paper is candid that the final accuracy of continual backpropagation "is only slightly better than a fully reinitialized network" on Continual ImageNet (Fig. 1b) — the advantage is that it "takes advantage of what it has previously learned to speed up learning on new data". In reinforcement learning with a small or no replay buffer, full reinitialization "will lead the agent to forget everything that it has learned, and its performance will be down to the starting point".

**Finding 7 — Continual backpropagation beats the closest published alternative, but that comparison is explicitly preliminary.**
*Evidence*: Extended Data Fig. 5b. On Ant-v3, proximal policy optimization with ReDo plus L2 "performs much better than standard PPO. However, it still suffers from performance degradation and its performance is worse than PPO with L2 regularization." The authors label this "a preliminary comparison" and "leave a full comparison and analysis for future work".

### b) Patterns and trends

The characteristic shape is *rise then fall*: performance improves over the first tens of tasks and then declines monotonically — the improvement is real transfer, and the decline overtakes it. In Continual ImageNet the crossover is within the first few tasks at large step sizes; in class-incremental CIFAR-100 it is at about 40 classes; in stationary Ant-v3 it is at about 3 million steps. The decline is *not* a plateau, which matters for how a monitoring metric would have to be shaped: a system watching only the current task sees a healthy number for a long time.

Severity scales with the step size (larger step size, faster and deeper loss), inversely with network width (100 units degrade most, 10,000 least, but "even the largest networks show some loss of plasticity"), and is insensitive to the rate of change of the data stream — permuting every 10,000, 100,000 or 1 million examples over 48 million examples all show it (Extended Data Fig. 3b). It holds for all six activation functions tested on Slowly-Changing Regression, severely for ReLU and tanh and less so for the exponential linear unit (Extended Data Fig. 2b).

### c) Surprises and counterintuitive results

The clean surprise is that *the standard toolkit is not neutral but often harmful*: "Surprisingly, popular methods such as Adam, Dropout and normalization actually increased loss of plasticity." Adam — the default optimizer for both deep reinforcement learning and large language models — is the worst method tested. The paper traces this to a mismatch between the two moment decay rates: "a sudden large gradient can cause a very large update, as a large value of $\beta_2$ means that the running estimate for the square of the gradient, which is used in the denominator, is updated much more slowly than the running estimate for the gradient, which is the numerator", and shows that setting $\beta_1 = \beta_2 = 0.99$ substantially reduces both the largest weight updates and the plasticity loss (Extended Data Fig. 5c).

A second surprise is that plasticity loss appears in a *stationary* reinforcement-learning environment. Nothing about Ant-v3 changes; the non-stationarity is generated by the agent's own improving policy, which changes the data it sees. This is why the paper argues reinforcement learning is "inherently continual".

A third, weaker surprise is the L2 trade-off: L2 fixes weight growth but overshoots, producing "very small weights, which prevented the agent from committing to good behaviour" — mitigating one correlate at the cost of the ability to specialize.

### d) Negative and null results

- Dropout is worse than plain backpropagation, and the paper cannot explain it with its own framework: "the poor performance of Dropout is not explained by our three correlates of loss of plasticity, which means that there are other possible causes of loss of plasticity."
- Online normalization (the online analogue of batch normalization) is better early and worse late.
- L2 regularization and shrink-and-perturb both reduce but do not eliminate the loss, and both are "slightly sensitive to hyperparameter values" — they help only for a narrow range and make things worse outside it. Shrink-and-perturb "does not fully resolve the three correlates ... it has a lower effective rank than backpropagation and it still has a high fraction of dead units".
- Shrink-and-perturb "did not provide a marked performance improvement over L2 regularization" on the ant problem, so it is absent from the reinforcement-learning figures.
- In class-incremental CIFAR-100 "there was no correlation between when a class was presented and the accuracy of that class, implying that the temporal order of classes did not affect performance" — a null result that rules out a simple recency story.
- Continual backpropagation's own utility choices are near-ties: running-average utility and instantaneous utility perform the same (Extended Data Fig. 5d), and the earlier preprint's more elaborate utility measure was dropped in revision because "for activations like ReLU, the contribution utility works almost as well".

### e) Robustness checks

Extensive on the phenomenon, thin on the mitigation. The phenomenon is replicated across five problems, four architectures (fully connected, convolutional, residual, actor-critic), two learning paradigms, six activation functions, four network widths, three rates of distribution shift, three to four step sizes each, and 30–100 seeds per cell. The mitigation is evaluated at its grid-searched best in each problem, with a replacement-rate sensitivity sweep (Extended Data Fig. 1b) showing $\rho = 10^{-5}$ beating $10^{-4}$ and $10^{-6}$ on CIFAR-100 by under one percentage point, and the authors note that in reinforcement learning "performance was highly sensitive to parameter settings", which is why continual backpropagation is only ever reported there in combination with L2.

---

## 4. Analysis and Interpretation

### a) Authors' explanations

The proposed mechanism is loss of the properties of the random initial weight distribution. "The only difference in the learner over time is the network weights ... the starting weights for the next task are qualitatively different from those for the first task. As this difference in the weights is the only difference in the learning algorithm over time, the initial weight distribution must have some unique properties that make backpropagation plastic in the beginning. The initial random distribution might have many properties that enable plasticity, such as the diversity of units, non-saturated units, small weight magnitude etc." Each correlate is then given its own argument: dead units contribute zero gradient so "once a unit in the first layer dies, it stays dead forever" under non-negative inputs, directly reducing capacity; large weights are "often associated with slower learning" through the condition number of the Hessian; and a low-rank representation "might be a bad starting point for learning from new observations because most of the hidden units provide little to no information", with the loop closed by "after each task, the learning algorithm finds a low-rank solution for the current task, which then serves as the initialization for the next task".

The remedy follows from the diagnosis: if the useful thing about initialization is randomness, keep injecting it. "Continual backpropagation involves a form of variation and selection in the space of neuron-like units, combined with continuing gradient descent", placed in a lineage running back to Selfridge's Pandemonium (1959) and to generate-and-test methods. The abstract's general conclusion: "methods based on gradient descent are not enough — that sustained deep learning requires a random, non-gradient component to maintain variability and plasticity."

### b) Evidence quality

The phenomenon itself is established about as well as an empirical claim in this literature can be: the comparisons are controlled (same streams, same seeds, same difficulty or an explicit from-scratch control), the reference baselines are principled, the sweeps are broad, and the run counts are high. The *existence* claim is demonstrated.

The *mechanism* claim is weaker and the paper's own language marks it. The three quantities are introduced as "correlates" and the paper says loss of plasticity "is accompanied by" and "is correlated with" them, and that their loss "partially explains" the degradation. The supporting argument is indirect: methods that fix the correlates also maintain plasticity. That is consistent with causation but does not isolate it — dropout breaks the pattern (Section 3d), which the paper acknowledges. No intervention isolates one correlate while holding the other two fixed.

The *superiority* claim for continual backpropagation is the weakest of the three. Against full reinitialization it wins on relearning speed, not on ceiling. Against ReDo the comparison is self-described as preliminary. Against shrink-and-perturb the margins in Fig. 1c and Extended Data Fig. 4a are under one percentage point.

### c) Confounds and alternative explanations

- **Head resetting** at every task change in Continual ImageNet gives the learner "privileged information on the timing of task changes"; the authors flag this and note they do not use it in other experiments. It could flatter every algorithm equally, so it is unlikely to explain the cross-algorithm differences, but it does inflate the absolute per-task numbers.
- **No learning-rate schedule in Continual ImageNet.** Referee 1 pressed exactly here: the rise-then-fall shape is what schedules exist to handle, and the residual-network experiment does use a schedule (reset per increment) while the ImageNet one does not. The authors did not run the schedule ablation on ImageNet.
- **Task structure.** Permuted MNIST tasks share no structure by construction, so nothing there can distinguish loss of plasticity from loss of transfer; Referee 3 made this point and the authors moved permuted MNIST to Methods in response, promoting the ant problem to the main text.
- **Single pass, never repeated.** Because tasks never recur, the online performance measure is, in Referee 2's words, "heavily, if not fully, biased towards plasticity", which is a confound for calling the setting "continual learning" but not for the plasticity claim itself.
- **Hyperparameter choice for Adam.** Referee 1 argued the catastrophic Adam result "might be more due to poor hparam choice"; the revision answers this by adding tuned Adam ($\beta_1 = \beta_2 = 0.99$) as a separate condition throughout, which does improve matters substantially while still losing plasticity.

### d) Generalizability

Credibly extends to: gradient-trained networks of any of the tested families on any long non-stationary or self-induced-non-stationary data stream, with or without the standard regularization toolkit, in both supervised and reinforcement settings. The reinforcement-learning result extends the claim to *stationary* environments whenever the agent's own policy is the source of distribution change.

Risky to extrapolate to: transformer-scale models on natural streams. The authors explicitly say "a systematic study with large language models would not be possible today because just a single training run with one of these networks would require computation costing millions of dollars". Also risky: settings with large replay buffers, which the authors note "can hide the effect of new data". And risky for any system whose adapting state is not weights, which is the case for us — see 5f.

---

## 5. Implications, Limitations and Transferability

### a) For practitioners

If a system trains continuously, expect ability-to-learn to decay and measure it against a from-scratch reference rather than against its own past. Do not assume the standard toolkit protects you — Adam, dropout and normalization all made things worse here. If you need a fix, the cheapest one with the strongest evidence is continual backpropagation, which is a few lines on top of an existing optimizer, costs one unit replacement every few hundred steps, and has two hyperparameters. Keep weights from growing (L2 helps) but do not drive them to zero (the agent then cannot commit). Finally: if your training run ends before the collapse would have started, you have not established that your method is stable, only that you stopped early.

### b) For researchers

The paper opens three named directions. First, a principled utility measure — "our current utility measure is not a global measure of utility as it does not consider how a given unit affects the overall representation function. One possibility is to develop utility measures in which utility is propagated backwards from the loss function." Second, joint stability and plasticity: continual backpropagation "does not tackle the forgetting problem. Its current utility measure only considers the importance of units for current data. One idea to tackle forgetting is to use a long-term measure of utility that remembers which units were useful in the past." Third, whether the optimization sense of plasticity and the generalization sense (Ash and Adams's warm-starting) are one phenomenon or two — an open question the paper takes a side on without settling.

### c) Acknowledged limitations

The evaluation measures plasticity and not forgetting, and the authors state this explicitly in the extended discussion after being pressed in review. The utility measure "is based on heuristics". Continual backpropagation is sensitive to hyperparameters in reinforcement learning and is reported there only with L2. The ReDo comparison is preliminary. Benchmark choice determines which property is visible: "in Continual ImageNet, previous tasks are rarely repeated, which makes it effective for studying plasticity but not forgetting."

### d) Unacknowledged limitations

- No wall-clock or compute accounting for continual backpropagation anywhere in the paper, despite Referee 1 asking; the rebuttal gives CIFAR-100 training times but they did not reach the camera-ready.
- Task boundaries are abrupt and known to the experimenter in every protocol; no protocol has gradual or unannounced drift of the kind a deployed system would face.
- Every stream is synthetic. The three referees converged on this independently and it survives into the published version: Continual ImageNet, permuted MNIST, class-incremental CIFAR-100 and friction-randomized Ant are all constructed non-stationarities.
- The ceiling is unexplored: the paper shows continual backpropagation does not *decline*, not that it reaches what a specialist would.

### e) Reproducibility

Strong. Code is public and, in Referee 1's words after inspection, "cleanly written and including the implementations used in the submission". All datasets and environments are public; the downsampled ImageNet subset is distributed by the authors. Every architecture and hyperparameter appears in Extended Data Tables 1–5, and the repository ships one configuration file per published condition. The cost is the barrier: about 12 GPU-hours per Continual ImageNet run and 30 runs per condition, about 6 GPU-hours per CIFAR-100 run, about 24 CPU-hours per 50-million-step reinforcement-learning run.

### f) Transfer to this project

`docs/framing.md` defines harness plasticity $\Phi(h)$ as the future stream performance achievable from harness $h$, and names its decrease under greedy patching loss of plasticity. This paper is where that term comes from. The transfer is partial and the boundary is worth stating exactly.

**What transfers.**

*The measurement design is the most valuable import.* The paper turns an unobservable capacity into an observable delta by always benchmarking against a learner that did not go through the stream — a linear network, or a network retrained from scratch on exactly the data available now. The harness analogue is direct: score a task with the current harness and with a reference harness (the unedited base, or a harness rebuilt from scratch for that task under the same budget), and read plasticity as the gap. Under the project's score-before-update protocol, this is a second scoring pass per task, not a new pipeline.

*The rise-then-fall shape justifies the project's central claim about what the current batch cannot see.* In every protocol here, per-task performance improves before it declines, and the decline is only visible against the task axis. A harness agent scoring each edit on the batch it was made on is looking at exactly the part of the curve that stays healthy longest.

*The state-function framing carries over.* The paper measures its three correlates on the network as it stands before each task, not on the training trace. That is the same shape as a critic $\hat\Phi(h)$ reading the harness files: cheap state-level statistics evaluated between tasks. The paper is the precedent for the idea, though it never validates the correlates as *predictors* — it only shows they move together with the loss, and dropout breaks even that.

*Selective reinitialization with a utility measure and a maturity threshold is a mechanism that is not intrinsically about weights.* It needs only a set of units, a per-unit usefulness score, a protection period for new units, and a rate. A harness of discrete entries — skills, memory entries, tool descriptions, prompt clauses — satisfies all four. The paper's finding that *selective* beats *indiscriminate* replacement mainly on relearning speed rather than on ceiling (Section 3a, Finding 6) is the argument for why a harness should prune rather than reset.

*The negative results are directly usable as predictions about harness-level analogues.* A size penalty on the harness is the analogue of L2: by this evidence it would reduce but not eliminate the loss, and it has a named failure mode — weights driven too small "prevented the agent from committing to good behaviour". That is a concrete warning for bloat-fighting harness approaches that audit for length, and a reason the project's objective is not a size constraint.

**What does not transfer.**

*All three correlates are properties of continuous weights and activations and have no harness counterpart.* There are no singular values of a prompt, no magnitude of a skill, and "dead" means something different for a text entry (never retrieved, never matched) than for a ReLU unit (provably zero gradient). An unused-entry fraction is a plausible analogue; a diversity or rank analogue would require embedding the entries, which introduces a measurement artifact the weight version does not have. The correlates must be re-derived, not ported.

*The diagnosed mechanism is specific to gradient descent and does not apply.* The paper's thesis is that gradient descent loses the variability of the random initialization because nothing refreshes it, and that the fix is "a random, non-gradient component". A harness is not updated by gradient descent; it is updated by a language model proposing text edits, which is already a non-gradient, stochastic, variability-injecting process. Whatever degrades a greedily patched harness, it is not loss of randomness — the more likely candidate is accumulation and interference among committed entries. Borrowing the paper's name should not import its mechanism.

*Reinitialization is lossy in a way harness editing is not.* A reinitialized unit is gone, which is why continual backpropagation's generate step must sample from the initial distribution. Harness content is text under version control, so the harness action set is strictly richer: revert, move content between layers, merge several narrow entries into one general entry. The paper's generate-and-test has no analogue of a move or a promotion.

*The paper covers only one of the project's three future costs.* The claim in `docs/framing.md` names generalization to later tasks, retention of earlier tasks, and the ability to keep patching. This paper studies the third and deliberately excludes the second — tasks never recur, the metric is online, and the authors conceded in review that their evaluation "is focused on plasticity, not forgetting". Its measurement design cannot be borrowed for the retention term or for the anchor set.

*The word is ambiguous in the source literature, and the project's usage sits on the other side of the ambiguity.* Referee 3 argued at length that two distinct phenomena share the name: the inability to *optimize* a new objective (this paper: dead units, no gradient flow) and the inability to *generalize* after warm-starting (Ash and Adams). The authors kept the position that they may be related; the referee accepted this as a difference of opinion the field would settle later, and the published extended discussion says outright that "it is unclear if the two types of plasticity loss are fundamentally different or if the same mechanism can explain both phenomena". The project's $\Phi(h)$ is future *stream performance* — the generalization sense, not the optimization sense. A one-line note at first use in project writing would be cheap insurance.

*The paper's operationalization is one-step; $\Phi(h)$ is a discounted sum.* Every measurement here is performance on the next task. That is the one-step lookahead special case of $\Phi(h)$, not the full quantity. Adopting the paper's measurement without saying so would silently replace a discounted future with a single step.

---

## 6. Summary

### a) One-sentence headline finding

Standard deep-learning systems trained continuously on a stream of tasks progressively lose the ability to learn new ones, eventually learning no better than a linear network.

### b) Quick-reference takeaways

- The phenomenon is real, large, and general: it appears in convolutional, residual and actor-critic networks, in supervised and reinforcement learning, across step sizes, widths, activation functions and rates of change, and even in a stationary reinforcement-learning environment where the only non-stationarity is the agent's own improving policy.
- The size of the loss is practically meaningful: about 5 percentage points below a from-scratch retrained network after 100 classes of CIFAR-100 — the authors calibrate this as equivalent to removing batch normalization — and total collapse to zero reward in Ant-v3 after 20 million steps.
- The standard toolkit does not protect against it and sometimes causes it: Adam is the worst method tested, dropout and normalization are worse than plain backpropagation, and L2 and shrink-and-perturb help only within a narrow hyperparameter range.
- Three state measures track the loss — the fraction of dead or dormant units (rising past 50%), average weight magnitude (roughly doubling), and effective or stable rank of the representation (falling by a quarter to three quarters) — but they are correlates, and dropout's failure is not explained by any of them.
- Continual backpropagation — reinitializing a tiny fraction of the lowest-utility mature units every step, on the order of one unit every few hundred updates — holds all three measures flat and maintains performance over 5,000 ImageNet tasks and 50 million reinforcement-learning steps; it beats full reinitialization on relearning speed (ten epochs versus about 125 on the 5,000th task) rather than on final accuracy.

### c) Bottom line for decision-making

Trust the phenomenon and the measurement design without reservation; treat the three correlates as diagnostics rather than as causes; treat continual backpropagation as the best-evidenced of several roughly comparable mitigations rather than as a settled answer.

---

## 7. Reviewer Reception

### a) Link and outcome

**No public OpenReview page exists.** Searched the OpenReview API for both the Nature title and the earlier preprint title *Maintaining Plasticity in Deep Continual Learning* on 2026-09-14; the only hits were reviews of unrelated submissions that mention the phrase. The work went from arXiv preprint (June 2023) directly to Nature and was never submitted to a conference with public reviews.

**Nature published the full review record instead**, as a [Peer Review File](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07711-7/MediaObjects/41586_2024_7711_MOESM1_ESM.pdf) — three referee reports on the initial version, the author rebuttals, three reports on the first revision, and the second rebuttal. Nature names the referees in the published article: Pablo Castro, Razvan Pascanu and Gido van de Ven. Outcome: accepted after two rounds; received 11 August 2023, accepted 12 June 2024 — ten months. No numeric scores; Nature does not use them.

### b) Main criticisms

1. **The demonstration is the contribution; the proposed algorithm is the weak half.** Referee 2: "I was rather disappointed by the solution — continual backpropagation — that is proposed in the manuscript, as well as by the way this solution is evaluated. It seems obvious that periodically reinitialising part of the network should help mitigate loss of plasticity", and suggested "an option could also be to leave out the part about the proposed solution". The editor agreed: "the thorough exposition of the problem is a clear strength of this paper, and ... the latter 'solution' aspect is somewhat weaker".
2. **Missing comparison to ReDo / the dormant neuron phenomenon.** Referee 1 called the references "one of the weakest points in this paper" and noted that the ReDo paper already compared against continual backpropagation using the arXiv version and found "no real gain from the scoring method used here", concluding that "relegating it to the end in such a shallow manner is certainly not enough for publication".
3. **Forgetting is ignored, and the online metric is biased toward plasticity.** Referee 2, across both rounds: "this manuscript simply ignores forgetting ... This is like studying the plasticity/stability dilemma by only considering plasticity"; and on the revision, "this paper uses artificial data streams in which old tasks are never repeated. Because of this, the online performance metric is heavily, if not fully, biased towards plasticity." The referee also objected that on ImageNet and CIFAR-100 the evaluation is not actually online, since networks train for 200–250 epochs before measurement.
4. **All the settings are synthetic, and the strong framing claims are not licensed by them.** All three referees raised this. Referee 3: "All datasets provided are synthetic. In particular permuted MNIST is generally seen as being too toyish." Referee 1, on the revision, attacked the framing directly: the claim that continued training is not possible in practice is "simply not true, as supervised fine-tuning (SFT) is doing exactly this ... If continued training were truly not possible, none of these methods would work", and concluded "I don't feel the paper is sufficiently convincing for the claims being made", recommending the authors "soften the claims made".
5. **The mechanism is asserted rather than derived.** Referee 3: "Right now there is very little precise information of why these 3 metrics matter and no discussion whether there are other properties that matter", and separately asked for a technical justification of the utility measure comparable to the Taylor-expansion arguments used in the pruning literature.
6. **Two different phenomena share the name.** Referee 3 argued that Ash and Adams's warm-starting failure is an inability to *generalize* while this paper's is an inability to *optimize*, and that "I strongly disagree that it is the same phenomenon."

### c) Author rebuttal and resolution

The first rebuttal is where most of the published paper's content came from. The authors added the class-incremental CIFAR-100 residual-network experiment (answering the "unrealistic architecture" objection), added all the stationary and non-stationary reinforcement-learning experiments and promoted the ant problem into the main text (answering "not relevant to practitioners"), *simplified* the algorithm by dropping the preprint's more elaborate utility measure in favour of contribution utility alone, added the preliminary ReDo comparison, narrowed the friction range in the ant problem from an "unrealistic" wide range to [0.02, 2.00], and cut the main text to about 4,000 words. They explicitly stated in the paper that continual backpropagation does not address forgetting, and added the acknowledgement that their metrics measure plasticity and not forgetting. On the ReDo priority question they noted their own work had been public longer.

Resolution by referee: **Referee 3 moved to accept** — "taking into account the rebuttal and changes done with the paper ... I'm happy with the overall paper and I think it is worth accepting" — while recording the optimization-versus-generalization disagreement as "a disagreement of opinion" that "does not affect the correctness of their claims", a position the authors accepted. **Referee 2 was "mostly satisfied"** and narrowed to asking for clearer statements about what the evaluation does not measure, which the authors made. **Referee 1 remained unconvinced** through the second round, maintaining that the claims are too strong for synthetic settings, but did verify the code: "I did a quick examination of the code and it seems to be cleanly written and including the implementations used in the submission."

### d) Net takeaway

The review record confirms the classification: two of three referees explicitly separated a strong phenomenon from a weaker proposed remedy, and the published paper is materially stronger than the preprint because of it — the residual-network and reinforcement-learning results exist only because reviewers demanded realism. Read the demonstration as well-vetted. Read the mechanism, the superiority of continual backpropagation over ReDo and shrink-and-perturb, and the framing about deep learning being incapable of continued training as contested and, on the last point, contested by a referee who never withdrew the objection.
