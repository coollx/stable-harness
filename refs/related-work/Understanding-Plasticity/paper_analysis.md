# Understanding Plasticity in Neural Networks

- **Original title**: Understanding Plasticity in Neural Networks (the arXiv listing lowercases it as *Understanding plasticity in neural networks*)
- **Authors**: Clare Lyle (corresponding, clarelyle@deepmind.com), Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, Will Dabney
- **Affiliations**: Google DeepMind (Lyle, Zheng, Avila Pires, Pascanu, Dabney); Montreal Institute for Learning Algorithms (Nikishin — work done during an internship at DeepMind)
- **Venue / year**: International Conference on Machine Learning 2023 (the 40th, Honolulu), PMLR 202:23190–23211. Presented as an **oral** ([ICML virtual page](https://icml.cc/virtual/2023/oral/25517)).
- **PMLR**: https://proceedings.mlr.press/v202/lyle23b.html
- **arXiv**: [arXiv:2303.01486](https://arxiv.org/abs/2303.01486) (v4, 27 November 2023 — this is the version in `paper.pdf`)
- **OpenReview**: [openreview.net/forum?id=odqQB2OXsG](https://openreview.net/forum?id=odqQB2OXsG) — metadata only; ICML 2023 did not release reviews (see section 7)
- **Code**: **none released.** No repository is named anywhere in the paper, and no official or author-affiliated implementation could be found (see section 5e). There is therefore no `repo_analysis.md` in this folder.
- **Local copy**: `paper.pdf`, 22 pages — 9 pages of main text, 3 pages of references, Appendix A (experiment details), Appendix B (additional analysis), Figures 1–22.

**Classification: empirical.** The contribution is what was learned, not what was built: the paper runs a falsification study over five previously conjectured causes of plasticity loss, rules all of them out as causal, and benchmarks seven existing interventions it did not invent. Nothing new is proposed as an artifact — the one thing carried to a large benchmark, layer normalization, is an off-the-shelf layer from 2016. The method lens would have nothing to describe.

**Why this reference matters here**: `docs/framing.md` borrows "plasticity" and "loss of plasticity" from continual learning. `refs/related-work/Loss-of-Plasticity/` holds the paper that made the phenomenon famous; this one is the paper that asks what *causes* it, and reaches the opposite conclusion about the two correlates that the Nature paper leans on hardest — dead units and weight growth. Section 5f states exactly which of its definitions, measurements and causes have an analogue in a harness made of text files, and which do not.

---

## 1. Research Motivation and Questions

### a) Why was this study conducted?

A cluster of published interventions all claim to restore a neural network's ability to keep learning: resetting whole layers (Nikishin et al., 2022), reinitializing individual activation units (Dohare et al., 2021 — the preprint that became the Nature paper in `refs/related-work/Loss-of-Plasticity/`), and regularizing the learned features (Kumar et al., 2020; Lyle et al., 2021). The authors' opening observation is that these fixes act on completely different objects, so "it is unlikely that they are all obtaining these improvements by the same mechanism. As a result, it is difficult to know how to improve on these interventions to further preserve plasticity" (Introduction). The field had a growing list of remedies and no diagnosis.

### b) What is missing in current understanding?

Prior work had conjectured, "implicitly or explicitly, that a variety of network properties might cause plasticity loss" — saturated units, large parameter norms, low-rank features, low-rank weight matrices — but, in the authors' words, "the explanatory power of these hypotheses has not been rigorously tested. While observing a correlation between a particular variable and plasticity loss can be useful for diagnosis, only a causal relationship indicates that intervening on that variable will necessarily increase plasticity" (Section 5.2). The concrete gap is that the whole literature runs on correlations measured inside a single experimental setting.

### c) Research questions

The paper states three, verbatim, in Section 3.3:

1. **"What happens when neural networks lose plasticity?"** — answered in Sections 4 and 5 by two interpretable case studies plus a qualitative analysis of what the probe-task learning curves look like.
2. **"What properties cause plasticity loss?"** — answered in Section 5 by a falsification design over four candidate statistics.
3. **"How can we mitigate plasticity loss?"** — answered in Section 6 by a sweep over scaling and seven interventions, then one intervention carried to the Arcade Learning Environment.

### d) Type of study

Mechanism-oriented failure-mode characterization with an explicit **falsification design**. The authors borrow the design from the study of causally robust predictors of generalization (Dziugaite et al., 2020): a statistic that genuinely explains an outcome must correlate with it *with the same sign* across experimental conditions, so exhibiting a sign reversal is enough to disqualify it. This is the paper's methodological core and the part most worth importing.

---

## 2. Study Design

### a) Variables and conditions

**Independent variables that are manipulated.** Observation space (CIFAR-10 images versus MNIST images); reward structure of the environment (three variants, described below); network architecture (multi-layer perceptron, convolutional network, 18-layer residual network, small vision transformer); network width (base width 16, multiplied by 1, 2, 4, 8, 12, 16); how often the bootstrap target is refreshed (target network update period of 1, 100 or 1,000 steps); optimizer and its hyperparameters; the output parameterization (plain regression versus a categorical "two-hot" encoding); the intervention applied (seven, listed in 2d); and random seed.

**The dependent variable of interest is a single scalar, plasticity loss**, defined in 2c. The paper also logs candidate explanatory statistics — parameter norm, rank of the weight matrices, number of dead units, rank of the learned features — and two loss-landscape objects: the eigenvalue spectrum of the Hessian (the matrix of second derivatives of the loss with respect to the parameters, whose largest eigenvalue measures how sharply the loss curves) and the gradient covariance matrix (how aligned the per-example gradients are with each other).

**Held fixed.** The probe protocol itself — the same target distribution, the same optimizer, the same 2,000-step budget — is identical across every condition, which is what makes plasticity loss comparable across architectures and datasets. The paper also deliberately designs two of its three environments so that the transition dynamics do **not** depend on the agent's action, which isolates the non-stationarity coming from the moving bootstrap target from the non-stationarity coming from a changing state distribution.

### b) Subjects of study

| Setting | Learner | Non-stationarity | Scale |
|---|---|---|---|
| Memorization case study (Section 4.1, Figure 1) | Two-hidden-layer fully connected network, width 1,024; Adam, learning rate 0.001 | Random labels on 5,000 MNIST images, re-randomized after each fixed training budget | About 90,000 steps across several task boundaries |
| Random-walk comparison (Section 4.2, Figure 2) | Same architecture family; plain stochastic gradient descent, learning rate 0.001, batch size 512 | Q-learning on the easy classification environment with the target network refreshed every 5,000 steps | 20 iterations compared against a matched random walk |
| Falsification sweep (Section 5, Figure 3) | Deep Q-network agents, multi-layer perceptron and convolutional variants | Bootstrap target refreshed every 1,000 steps | **128 agents** over tasks, observation spaces, optimizers and seeds |
| Width sweep (Section 6.1, Figure 5) | Multi-layer perceptron and convolutional network on MNIST and CIFAR-10 | Target update periods 1, 100, 1,000 | Widths up to the memory limit of a single graphics card |
| Intervention sweep (Section 6.2, Figures 6, 8–11) | Four architectures, both output parameterizations | Same environments | 100 iterations of 1,000 steps per run; means ± standard deviation reported |
| Arcade Learning Environment (Section 6.3, Figure 7) | Double deep Q-network, the standard implementation from DQN Zoo | The ordinary reinforcement-learning training loop | **57 games, 3 seeds each**, 200 million frames per run |

The environments are "a simple analogue of image classification" cast as a decision problem with ten states and ten actions, where the observation for a state is an image of that class. Three variants: **true-label** (the image's real class indexes the state; reward is 1 when the action matches the state), **random-label** (each image is given a permanently random label, so the task fights the network's inductive bias), and **sparse-reward** (reward only in one state, and the agent's own actions now influence which states it visits).

### c) Measurement and instruments — the plasticity protocol

This is the part of the paper most worth reading closely, because it is the only operational definition of plasticity on offer anywhere in this reference folder.

**The definition (Section 2.2, equations 4 and 5).** Fix an optimization algorithm $\mathcal{O} : (\theta, \ell) \mapsto \theta^*$ that takes parameters and an objective and returns new parameters after a *fixed budget* — "the parameters $\theta^*$ need not be an optimum: $\mathcal{O}$ could, for example, run gradient descent for five steps." Fix a distribution $\mathcal{L}$ over loss functions, each one a regression problem against a randomly drawn target function:

$$\ell_{f,\mathbf{X}}(\theta) = \mathbb{E}_{x \sim \mathbf{X}}\left[(f(\theta, x) - g_\omega(x))^2\right]$$

Fix a baseline value $b$ — for a regression loss, the variance of the targets, i.e. what a constant predictor achieves. Then plasticity at parameters $\theta_t$ is how far below that baseline the network can get within the budget:

$$\mathcal{P}(\theta_t) = b - \mathbb{E}_{\ell \sim \mathcal{L}}\left[\ell(\theta_t^*)\right] \quad \text{where} \quad \theta_t^* = \mathcal{O}(\theta_t, \ell)$$

and **loss of plasticity along a training trajectory is the difference $\mathcal{P}(\theta_t) - \mathcal{P}(\theta_0)$** — the current checkpoint's plasticity minus the fresh network's. The authors highlight exactly why the difference form matters: "this definition of plasticity loss is independent of the value of the baseline $b$, i.e. the difficulty of the probe task for the network, allowing us to measure the relative change in performance of checkpoints taken from a training trajectory."

**The concrete protocol (Sections 3.1, 5.1 and Appendix A.2), step by step.**

1. The input set $\mathbf{X}$ is the transitions already sitting in the agent's replay buffer — so the probe tests the network on data it has actually seen, not on new data.
2. The probe target is $g(x) = a + \sin(10^5 f(x; \omega_0))$, where $\omega_0$ is a *fresh random initialization* of the same architecture and $a$ is the network's own current mean prediction. The extreme frequency multiplier $10^5$ is deliberate: it makes the target effectively a random, high-frequency function of the input, so fitting it requires perturbing predictions "in random directions sampled effectively uniformly over the input space". Centering on the network's mean prediction stops the probe from unfairly favouring freshly initialized networks, whose outputs sit near zero.
3. Training on the real task **pauses every 5,000 optimizer steps**. From a *copy* of the current parameters $\theta_t$, the network is trained on **10 independently sampled probe targets**, each for **2,000 optimizer steps**, with the **same optimizer it uses on the real task**. The loss at the end of those 2,000 steps is logged, the copies are discarded, and real training resumes from the saved $\theta_t$ — the probe never contaminates the trajectory.
4. The 2,000-step budget was chosen because it "minimized iteration time while also providing enough opportunity for networks starting from random initializations to solve the task" — i.e. calibrated so a fresh network succeeds.

The authors are candid that the target distribution is a modelling choice: "A different distribution over future target functions might give different numerical results; however, we believe that a uniform distribution captures a more universal notion of plasticity."

### d) Experimental conditions — the interventions tested

Seven, all pre-existing: **resetting the last layer** at every target network update (a simplified Nikishin et al. 2022); **layer normalization** after each convolutional and fully connected layer (Ba et al., 2016); **shrink and perturb** — multiply the weights by a small scalar and add the weights of a fresh random network, at every target update (Ash and Adams, 2020); a **two-hot categorical encoding**, where the network outputs a probability distribution over fixed bins and minimizes a cross-entropy loss instead of predicting a number directly; **spectral normalization** of the first linear layer (Gogianu et al., 2021); **weight decay** with penalty coefficient $10^{-5}$; and, mentioned only in Appendix B.1, **resetting only the optimizer state**.

### e) Statistical and analytical approach

No hypothesis tests anywhere. The falsification study is read off scatterplots — one point per training run, a fitted line per condition, and the argument is about the *sign* of the slope, not its significance. The intervention sweep reports mean ± standard deviation across seeds in heatmap cells. The Arcade results are three seeds per game with a human-normalized improvement score per game. The paper never states how many seeds back each intervention cell; the standard deviations are large enough (frequently as large as the mean) that this matters — see 4c.

---

## 3. Findings and Observations (primary focus)

### a) Headline findings

**Finding 1 — An abrupt change in the learning target can make an adaptive optimizer destroy the network, and this is a hyperparameter artefact, not a law.**
*Evidence*: Figure 1. A network that has perfectly memorized random MNIST labels is given a fresh random labelling. Default Adam "quickly leads ... to diverge, saturating most of its ReLU units and resulting in trivial performance on the task that a freshly initialized network could solve perfectly": the fraction of dead units in the run jumps from near zero to roughly 0.6–0.8 and stays there, and accuracy collapses to near chance. The mechanism is spelled out from Adam's update rule $u_t = \alpha \frac{\hat m_t}{\sqrt{\hat v_t + \bar\epsilon} + \epsilon}$ (eq. 6): gradient norms are roughly proportional to the loss, so when the loss jumps at a task boundary, the fast-moving numerator $\hat m_t$ updates long before the slow-moving denominator $\hat v_t$ catches up, and "the updates immediately after a task change will scale as a large number divided by a much smaller number". Changing two hyperparameters — second-moment decay from 0.999 to 0.9 and $\epsilon$ from $10^{-9}$ to $10^{-3}$ (Appendix A.1) — removes the failure entirely.
*Conditions*: a deliberately extreme setting (perfectly memorized labels, then full re-randomization). The authors close with a telling aside: the large $\epsilon$ used by deep Q-networks "suggests that the community has implicitly converged towards optimizer hyperparameters which promote stability under nonstationarity."

**Finding 2 — Gradient descent sharpens the loss landscape for *new* tasks in a way that a random walk of the same step size does not.**
*Evidence*: Figure 2, a controlled comparison. Two trajectories start from identical random parameters and take steps of identical norm; one follows gradient-based optimization on the non-stationary objective, the other adds Gaussian noise in a random direction. After five target updates, the Hessian eigenvalue density of the gradient-descent network has developed a large outlier peak around eigenvalues of 20–30 while the random walk stays concentrated near the origin, and the gradient covariance heatmaps show the gradient-descent network developing strong negative off-diagonal entries (gradients that point against each other, so reducing the loss on one group of inputs raises it on another) that the random walk does not. The authors' summary: "the inductive bias induced by gradient descent can push the parameters towards regions of the parameter space where the local loss landscape is less friendly to optimization towards arbitrary new objectives than what would be obtained by blindly perturbing randomly initialized parameters."
*Conditions*: one architecture, one environment. This is the paper's cleanest causal isolation — same starting point, same step size, only the direction differs — and it is run at the smallest scale of any experiment here.

**Finding 3 — None of the four conjectured causes survives falsification (the central negative result).**
*Evidence*: Figure 3, 128 deep Q-network agents, four scatterplots. "For each of four quantities, there exists a learning problem where the quantity positively correlates with plasticity, and one in which it exhibits a negative correlation. In many learning problems the correlation between plasticity loss and the quantity of interest is nonexistent." Specifically: **parameter norm** correlates positively with plasticity loss under CIFAR-10 observations and slightly negatively under MNIST observations; **rank of the weight matrices** flips the same way with observation space; **number of dead units** flips with architecture (convolutional versus fully connected); **rank of the learned features** flips with the reward structure (easy, hard, sparse). The authors add a second, independent complaint: "In all cases we note that the correlation with plasticity is already quite weak; even so, the ability to reverse the sign of this correlation is a further mark against the utility of these simple statistics as causal explanations of plasticity."
*Conditions*: the toy environments, deep Q-network agents, target refresh every 1,000 steps. **Note what is not in this list: gradient norm is never tested as a candidate cause.** It appears only inside the Adam mechanism of Finding 1.

**Finding 4 — Plasticity loss looks like slower optimization, not a lower ceiling.**
*Evidence*: Figure 4, probe-task learning curves taken from checkpoints at iterations 0, 10, 20, 50 and 100 of training on the real task. The iteration-0 checkpoint drives the probe loss from about 0.5 to near zero within roughly the first 250 of its 2,000 steps; by iteration 100 the curve is still above 0.1 when the budget runs out. "We see that parameters from early training checkpoints quickly attain low losses, but that the slopes of these learning curves become more shallow as training progresses on the main task." The distinction the authors draw is diagnostic: hitting an early plateau at a high loss would mean the optimizer is stuck in a bad minimum, whereas a uniformly flatter slope means it is having trouble making progress at all. They also note the curves become visibly noisier, which in full-batch optimization is a known signature of a sharper loss landscape (Cohen et al., 2021). "In short, we see that it is increasing difficulty of navigating the loss landscape that drives plasticity loss in this problem."

**Finding 5 — Scale reduces plasticity loss consistently but never removes it.**
*Evidence*: Figure 5, width sweep across three target update frequencies and four architecture-dataset combinations, plasticity loss on a logarithmic axis. The curves decline monotonically with width in every panel; the caption states the limit plainly: "even when scaling the architecture to the point where it no longer fits on a single GPU, we are still unable to completely eliminate plasticity loss on these simple classification-inspired problems." The authors' reading is conditional rather than triumphant: plasticity loss "is unlikely to be the limiting factor for sufficiently large networks on sufficiently simple tasks", but persists "for tasks which do not align with the inductive bias of the network ... or for which the network is not sufficiently expressive", and "we typically cannot guarantee a priori that a learning problem will fall in the first category".

**Finding 6 — In the intervention sweep, changing how the prediction target is represented beats every method that perturbs or regularizes the parameters; and two popular interventions are actively harmful.**
*Evidence*: Figure 6 (CIFAR-10 observations; the entry is the change in probe loss between the first and last epoch of training, so **lower is better** and the no-intervention row is the reference).

| Intervention | ResNet | Transformer | CNN | MLP |
|---|---|---|---|---|
| None (reference) | 0.20 ± 0.21 | 0.13 ± 0.03 | 0.31 ± 0.06 | 0.27 ± 0.21 |
| Two-hot categorical output | **0.12 ± 0.15** | **0.05 ± 0.04** | **0.09 ± 0.17** | **0.02 ± 0.02** |
| Reset last layer | **0.00 ± 0.00** | **0.04 ± 0.07** | 0.36 ± 0.32 | **0.10 ± 0.10** |
| Layer normalization | — | — | 0.24 ± 0.16 | 0.25 ± 0.21 |
| Weight decay | 0.26 ± 0.18 | 0.17 ± 0.10 | 0.34 ± 0.08 | 0.28 ± 0.23 |
| Spectral normalization | — | — | 0.41 ± 0.21 | 0.43 ± 0.23 |
| Shrink and perturb | 0.79 ± 0.21 | 0.23 ± 0.03 | 0.56 ± 0.20 | 0.61 ± 0.07 |

Read down the columns: the two-hot categorical output is the only intervention that improves on doing nothing in all four architectures, cutting plasticity loss by roughly 40% (ResNet), 62% (transformer), 71% (CNN) and 93% (MLP). Layer normalization gives a modest 23% (CNN) and 7% (MLP) improvement. Weight decay and spectral normalization are neutral-to-worse everywhere. **Shrink and perturb — the intervention whose entire principle is injecting fresh randomness — is the worst method tested, roughly quadrupling plasticity loss on the residual network (0.79 versus 0.20) and roughly doubling it on the convolutional and fully connected networks.** Resetting the last layer is excellent on three architectures and worse than nothing on the convolutional one. The authors' verdict: "selecting a network parameterization which smooths out the loss landscape is the most effective means of preserving plasticity of all approaches we have considered in this setting, and even has a greater effect on plasticity than resetting the final layer of the network in some instances."

**Finding 7 — The output parameterization dominates every other choice, to the point where it reverses the ranking of the other interventions (appendix only).**
*Evidence*: Figures 8 (CIFAR-10) and 9 (MNIST) split the same runs by output parameterization. Under the categorical encoding, layer normalization takes CNN plasticity loss to $-0.00 \pm 0.00$ and MLP to $0.05 \pm 0.16$, against $0.24 \pm 0.16$ and $0.25 \pm 0.21$ under plain regression. Shrink and perturb, disastrous under regression ($0.79 \pm 0.21$ on the residual network), becomes harmless under the categorical encoding ($0.08 \pm 0.06$). The same pattern holds on MNIST (Figure 9): layer normalization under regression is $0.32 \pm 0.13$ (CNN) and $0.43 \pm 0.28$ (MLP), both **worse** than the no-intervention reference of $0.27 \pm 0.08$ and $0.34 \pm 0.24$, while under the categorical encoding it is $0.03 \pm 0.05$ and $0.07 \pm 0.07$. Appendix B.1's own summary is narrower than the main text's: "resetting the last layer, incorporating a two-hot output representation, and performing layer normalization have beneficial effects on plasticity, whereas shrink and perturb, weight decay, and resetting only the optimizer state do not improve plasticity."

**Finding 8 — Layer normalization added to a standard double deep Q-network improves 45 of 57 Atari games.**
*Evidence*: Figure 7, human-normalized improvement score per game, 3 seeds, 200 million frames, "the only difference between the baseline implementation and our modification is the incorporation of layer normalization after each hidden layer". 45 games improve, 2 are exactly flat (Montezuma's Revenge, Gravitar — both games where the baseline scores nothing), 10 regress. The top of the distribution: Enduro +144.2%, Space Invaders +140.9%, Wizard of Wor +132.6%, Tennis +124.4%, Video Pinball +119.9%, Asterix +105.4%. The bottom: Double Dunk −52.4%, Bowling −8.7%, Centipede −7.3%, Venture −4.6%, Atlantis −3.3%. The authors do not overclaim the attribution: "While this improvement cannot be definitively attributed to a reduction in plasticity loss from the evidence provided, it points towards the regularization of the optimization landscape as a fruitful direction."

**Finding 9 — At Atari scale, gradient degeneracy tracks learning progress one-for-one, while Hessian sharpness tracks nothing (appendix).**
*Evidence*: Appendix B.3, Figures 19–22. "In the game Freeway, which is known to produce extremely high variance outcomes wherein agents either maximize the game score or fail to learn at all, we saw a one-to-one mapping between gradient degeneracy and learning progress. All random seeds where the agent made learning progress exhibited heavier weight on as opposed to off the diagonal, whereas the random seeds which did not ever improve preserved their initial degenerate gradient structure." Against that: "We did not observe any obvious correlations between the spectrum of the Hessian and the agent's performance." The authors explicitly decline to say which way the causation runs: "it is not clear whether gradient degeneracy is a symptom or a cause of performance plateaus."

### b) Patterns and trends

Two cross-cutting patterns. First, **every candidate statistic the paper tests is condition-dependent while the phenomenon is not** — plasticity loss shows up in all four architectures, both observation spaces, all three reward structures and all three target-refresh frequencies, but no single statistic points the same way across those conditions. Second, **the interventions sort cleanly by what they act on**: the ones that change how the objective is represented (categorical output, normalization layers) help, the ones that perturb or shrink parameters (shrink and perturb, weight decay, spectral normalization) do not, and the one that discards parameters (resetting the last layer) is the most architecture-sensitive of all. That sorting is the paper's thesis in one line.

### c) Surprises and counterintuitive results

The genuine surprise is Finding 6's bottom row. Shrink and perturb is the closest thing in this sweep to "continually inject diversity into the network", which is the remedy class the Nature paper in `refs/related-work/Loss-of-Plasticity/` advocates as the only one that works. Here it is the worst of seven, by a wide margin, on three of four architectures. The second surprise is Finding 7: the ranking of interventions is not a property of the interventions, it is a property of the interventions *crossed with the output parameterization*, and that cross-term is larger than most of the main effects — a result confined to the appendix.

A third, quieter surprise is that plasticity loss appears as a flatter slope rather than an earlier plateau (Finding 4). The intuitive story — the network gets stuck in a bad local optimum and cannot escape — is not what happens.

### d) Negative and null results

- The four conjectured causes are all rejected (Finding 3). This is the paper's headline and it is a negative result.
- Scaling does not eliminate plasticity loss (Finding 5).
- Weight decay and spectral normalization do not improve plasticity in any of the eight architecture-dataset cells where they were run; several cells are worse than no intervention (Figure 6, Figure 9).
- Layer normalization under the plain regression parameterization is worse than no intervention on MNIST for both the convolutional and fully connected networks (Figure 9), which sits awkwardly with the main text calling it "the best-performing intervention".
- No relationship was found between the Hessian spectrum and agent performance at Atari scale (Finding 9) — which partially undercuts the paper's own curvature thesis at the scale that matters most.
- The two-hot categorical encoding, the strongest intervention by raw number, is explicitly *not* recommended as a drop-in: "it does so at the cost of stability of the learned policy in several instances we considered. Additionally, this intervention required significantly different optimizer hyperparameters from the regression parameterization, suggesting that while it can be a powerful tool to stabilize optimization, it might not be suitable as a plug-in solution."

### e) Robustness checks

The falsification design *is* the robustness check, and it is the strongest part of the paper: 128 agents crossed over tasks, observation spaces, optimizers and seeds, with the explicit goal of finding conditions that reverse a correlation. The intervention sweep is repeated on two datasets (Figures 8 and 9), split by output parameterization, and decomposed into initial loss and final loss separately (Figures 10 and 11), and it is checked for interference with the primary reinforcement-learning task (Figures 12 and 13) — where the authors report honestly that "the methods which perturb the network weights often interfere with learning, particularly in the more challenging 'hard' reward structure". The Atari transfer is 57 games, which is the standard, but only 3 seeds, which is thin for a benchmark whose per-game variance is notorious.

---

## 4. Analysis and Interpretation

### a) Authors' explanations

The proposed mechanism is **curvature of the loss landscape induced by new tasks on trained parameters**. It is built from three pieces: the controlled random-walk comparison showing gradient descent specifically (not parameter movement in general) grows Hessian outliers and negative gradient interference (Finding 2); the learning-curve shape showing plasticity loss as slowed progress, which is what a sharper landscape predicts (Finding 4); and the intervention ranking, in which the methods that help are exactly the ones conjectured to smooth the landscape (Finding 6). Figure 14 supplies a direct confirmation at small scale: adding layer normalization holds the largest Hessian eigenvalue of a convolutional network roughly flat while the default architecture's climbs steeply over 30 training iterations.

The authors are careful about the status of this claim. The abstract says loss of plasticity "is deeply connected to changes in the curvature of the loss landscape" — connected, not caused by — and the paper concedes it is "difficult to characterize explicitly". The conclusion frames curvature control as a direction rather than an answer: "The findings of this paper point towards stabilizing the loss landscape as a crucial step towards promoting plasticity."

The demonstrated / suggested / speculated split:
- **Demonstrated**: the four conjectured statistics do not have sign-stable correlations with plasticity loss (Figure 3); adaptive optimizer divergence at task boundaries is real and hyperparameter-fixable (Figure 1); gradient descent grows curvature faster than a matched random walk (Figure 2); the intervention numbers in Figures 6, 8 and 9; layer normalization improves 45 of 57 Atari games (Figure 7).
- **Suggested**: that curvature is the actual driver — supported by convergent but circumstantial evidence, and undercut by the absence of any Hessian-performance relationship at Atari scale.
- **Speculated**: that the Atari improvement is *because of* plasticity — the authors say outright it "cannot be definitively attributed"; and that gradient degeneracy causes rather than accompanies performance plateaus.

### b) Evidence quality

Three tiers. The **falsification result is the best-supported claim in the paper and is close to airtight for what it actually says** — a sign reversal is a valid disqualifier, the sweep is wide, and the conclusion is stated at exactly the right strength: these statistics are not *uniquely* attributable causes. It does not show they are irrelevant, only that none of them alone explains the phenomenon.

The **curvature claim is one controlled experiment plus a pile of consistent circumstantial evidence.** The Brownian-motion comparison is genuinely controlled (matched starting point, matched step norm, single manipulated variable) but runs at one architecture, one environment, 20 iterations. Everything else supporting curvature is correlational, and by the paper's own falsification standard, a correlation observed in one family of settings is not enough — the authors do not subject curvature to the same sign-reversal test they use to reject the other four statistics. Appendix B.3 then reports no Hessian-performance relationship on Atari.

The **Atari result is a solid benchmark improvement with an unestablished mechanism**, and the paper says so.

### c) Confounds and alternative explanations

- **No per-intervention hyperparameter tuning**, acknowledged: "we did not perform extensive hyperparameter tuning for each intervention, so it is possible that a more carefully tuned optimizer would produce better performance." Given that Finding 1 shows plasticity outcomes swinging on two Adam constants, this is a large caveat on the entire ranking in Figure 6 — an intervention may be losing because of its optimizer, not its mechanism.
- **The error bars swallow most of the differences.** In Figure 6 the reference cell for the residual network is $0.20 \pm 0.21$ and for the fully connected network $0.27 \pm 0.21$. Only the extremes (two-hot, shrink and perturb) clearly separate from those bands. The paper reports no seed counts for these cells and runs no test.
- **The probe target is one specific choice** — a high-frequency sine of a random network — and the paper acknowledges that a different distribution "might give different numerical results". Nothing establishes that this target's difficulty profile matches the tasks a practitioner cares about, and a $10^5$ frequency multiplier is an adversarial choice in a way the paper does not quantify.
- **The probe measures training loss on data already in the replay buffer.** By construction it cannot distinguish "can no longer fit new targets" from "can no longer fit new targets *on this fixed data*", and it says nothing about generalization.
- **Three seeds on Atari.** Double Dunk at −52.4% could be one seed.
- **Figure 6's baseline is "None" in a sweep with no shared control across datasets**, and the main text calls layer normalization "the best-performing intervention" on evidence that, taken across both datasets and both parameterizations (Figures 8 and 9), it is not — the two-hot encoding is, and the authors know it and decline to recommend it for stability reasons they do not quantify.

### d) Generalizability

Credibly extends to value-based reinforcement learning with bootstrapped targets, across the four architecture families tested, for the specific claim that plasticity loss occurs and that the four named statistics do not explain it. The Atari result extends layer normalization's benefit to a standard, full-scale agent.

Risky to extrapolate: the curvature mechanism to large models (the paper's own Atari-scale Hessian analysis finds nothing); anything at all to policy-gradient methods, to supervised learning with a stationary objective, or to networks whose adaptation is not by gradient descent; and the intervention ranking to any setting where the optimizer has been tuned per intervention.

### e) Relation to forgetting, backward transfer, forward transfer, and intransigence

The paper draws one boundary itself and leaves the rest implicit. Section 2 opens by naming and excluding **catastrophic forgetting** — "training a network first on one task and then a second will result in reduced performance on the first task" — and stating the contrast: "This paper concerns itself with a different phenomenon: in certain situations, training a neural network on a series of distinct tasks can result in worse performance on later tasks than what would be obtained by training a randomly initialized network of the same architecture." Every measurement in the paper is forward-facing; nothing is ever re-measured on an earlier task.

Mapping that onto the metric vocabulary used by the other references in this folder (the mapping below is mine, not the paper's — none of these three terms appears in it):

- **Intransigence** (Chaudhry et al., ECCV 2018, *Riemannian Walk*, in `refs/related-work/Riemannian-Walk/`) is defined there as $I_k = a_k^* - a_{k,k}$: the accuracy on task $k$ of a reference model trained jointly on all data up to $k$, minus the accuracy on task $k$ of the model that got there incrementally. This is the closest existing term to what Lyle et al. measure, and the correspondence is structural — both are "how much worse is the streamed learner than a reference learner on the task in front of it". Two differences matter. The reference model differs: Chaudhry's is a joint-training oracle, Lyle's is the *same architecture freshly initialized* and given the same fixed budget. And the task differs: Chaudhry measures on the real next task, Lyle measures on a synthetic probe drawn from a deliberately uninformative distribution, precisely so that the number is not about any one future task.
- **Backward transfer** (Lopez-Paz and Ranzato, NIPS 2017, in `refs/related-work/GEM/`) — how much learning a later task moved an earlier task's accuracy — **has no counterpart here at all.** Probe targets are drawn fresh every time and never revisited; the reinforcement-learning tasks never recur. The paper cannot see retention, and does not claim to.
- **Forward transfer** in the same source is accuracy on a task *before* seeing any of its data, relative to an untrained network. Lyle's plasticity shares the sign convention and the "relative to an untrained network" reference point, but it is not zero-shot: the network gets 2,000 optimizer steps on the probe. The quantity therefore sits between forward transfer (zero steps of adaptation) and intransigence (full training) — it is **trainability of an arbitrary next task under a fixed budget**.
- The paper also flags, in Section 2.2, that the word "plasticity" covers two different failures in the literature: reduced *generalization* after warm-starting (Ash and Adams, 2020; Berariu et al., 2021) and an impaired ability "to even reduce the learning objective on the training distribution" (Dohare et al., 2021; Lyle et al., 2021; Nikishin et al., 2022). It adopts the second — the optimization sense — explicitly, noting that the formulation it inherits from Lyle et al. (2021) "is applicable to learning problems which do not admit a straightforward train-test split, as is the case in many deep RL environments." Every number in this paper is a training loss.

---

## 5. Implications, Limitations and Transferability

### a) For practitioners

Four concrete actions. Check your adaptive optimizer's constants before blaming your architecture: a second-moment decay of 0.999 with a tiny $\epsilon$ is a live failure mode at every abrupt target change, and moving to 0.9 and $10^{-3}$ removed it here. Do not use a diagnostic statistic — dead-unit count, weight norm, feature rank — as a health metric without first establishing that it points the same way in your setting as in the setting you borrowed it from; this paper shows all four reverse sign across ordinary experimental choices. Prefer interventions that change how the objective is represented (normalization layers, categorical output heads) over ones that perturb or shrink parameters; shrink and perturb made things substantially worse in three of four architectures here. And if you want a single change with the broadest evidence behind it, add layer normalization after each hidden layer: on 57 Atari games with a standard double deep Q-network and no other tuning, 45 improved.

### b) For researchers

The falsification design is the transferable methodological tool, and it generalizes far beyond plasticity: for any statistic proposed as the cause of any training pathology, construct a family of learning problems and check whether the correlation's *sign* survives. The paper also leaves three explicit openings: an explicit characterization of the curvature property that actually matters (it says only that it is "difficult to characterize explicitly"); whether gradient degeneracy is symptom or cause (Appendix B.3, named as "an exciting avenue for future work"); and the tension its conclusion names between memorization and generalization — "an exciting direction for future work to better disentangle the complementary roles of memorization and generalization in plasticity."

### c) Acknowledged limitations

The probe target distribution is one choice among many and may change the numbers. No extensive hyperparameter tuning per intervention. The Atari improvement "cannot be definitively attributed to a reduction in plasticity loss". The curvature property is not explicitly characterized. The two-hot encoding costs policy stability and needs its own optimizer settings. The conclusion volunteers a scope limit that is rare and worth quoting: "it is possible that in many settings, plasticity loss is not a limiting factor in network performance and so need not be a concern for many of the relatively small environments used to benchmark algorithms today, we conjecture that as the complexity of the tasks to which we apply RL grows, so will the importance of preserving plasticity."

### d) Unacknowledged limitations

- **No seed counts and no statistical tests anywhere**, while the standard deviations in the headline intervention figure are frequently as large as the means.
- **The main text's "best-performing intervention, layer normalization" is not what the appendix shows.** By raw numbers the two-hot categorical encoding wins in all four architectures on CIFAR-10 and three of four on MNIST, and layer normalization under plain regression is worse than doing nothing on MNIST. The choice to carry layer normalization to Atari is defensible (it is a drop-in; the two-hot encoding is not) but the label is not supported.
- **Curvature is never subjected to the paper's own falsification test.** The statistics it rejects were rejected for having condition-dependent correlations; the statistic it adopts is supported by correlations from a narrower set of conditions plus one small controlled experiment.
- **Every environment is synthetic** apart from Atari, and the synthetic ones are ten-state decision problems built on image classification.
- **The probe never tests generalization**, so "plasticity" here cannot distinguish a network that can still fit anything from one that can still learn anything useful.

### e) Reproducibility

**Weak — no code was released.** The paper names no repository; the arXiv and PMLR listings carry no code link; Papers with Code lists none; the community survey repository `Probabilistic-and-Interactive-ML/awesome-plasticity-loss` indexes this paper with a paper link and no code link while tagging neighbouring papers with both (searched 2026-09-17). The only implementation the paper points at is the external baseline it builds on, DQN Zoo (`github.com/deepmind/dqn_zoo`). What *is* fully specified in Appendix A: all four architectures with exact widths, kernel sizes and channel counts; both Adam configurations for the optimizer case study; the probe protocol's every constant (10 targets, 2,000 steps, probe every 5,000 steps, target update period, $\epsilon$-greedy with $\epsilon = 0.1$, replay buffer of 100,000, 200 million frames on Atari); and the weight decay coefficient of $10^{-5}$. The probe protocol is reimplementable from the paper alone in a few dozen lines on top of any existing agent. The environments are not released, but they are simple enough to rebuild from the three-paragraph description in Section 3.2.

### f) Transfer to our own work

`docs/framing.md` defines harness plasticity $\Phi(h)$ as the future stream performance achievable from harness $h$ and calls its decrease under greedy patching loss of plasticity. This paper contributes three things to that: a measurement protocol with a property we should copy, a methodological standard we should adopt before naming any harness statistic a cause, and a set of causal claims that mostly do not survive the crossing into text.

**What has a direct analogue.**

*The measurement protocol, and specifically its four design choices.* The paper turns an unobservable capacity into a number by fixing a budget, drawing a probe from a distribution deliberately unrelated to the current task, running the real update machinery on a discarded copy, and reporting a difference against the same measurement taken at the start. All four choices port. The harness analogue: at a checkpoint in the stream, take a copy of the harness, draw a small set of probe tasks from a distribution that is not the current batch, let the harness agent make a **fixed budget of edits** (the analogue of 2,000 optimizer steps — a count of edits or of tokens, named as a volume argument), score the result, discard the copy, and report the score minus the same quantity measured from the base harness. The difference form is the valuable part: because the probe's intrinsic difficulty enters both terms, the number is comparable across checkpoints even though we can never calibrate "how hard is this probe for a harness" in absolute terms. This is a second scoring pass at checkpoints, not a new pipeline, and it fits inside score-before-update.

*The falsification standard, which is the single most useful import.* The temptation in a text harness is to reach for an obvious health statistic — total characters, entry count, fraction of memory entries never retrieved, redundancy between entries, depth of the layer that content sits at — and treat a correlation with declining stream performance as a cause. This paper is a worked demonstration that exactly this reasoning failed for four statistics that the field had accepted for years, and that the failure was only visible once someone varied the observation space and the architecture. The standard to adopt before we call any harness statistic a cause of plasticity loss: it must correlate with the same sign across at least two materially different task streams and two materially different base models. Harness size is the statistic most at risk here, and `docs/framing.md` already keeps size out of the objective.

*The shape of the degradation tells us what to measure.* Finding 4 says plasticity loss showed up as a flatter learning curve, not an earlier plateau — the network was not stuck, it was slow. The harness analogue is a measurable prediction: a harness that has lost plasticity should need **more edits, or more attempts, to reach a given score on a new task**, rather than topping out below it. That argues for logging edits-to-fix and attempts-to-pass alongside final score, since a metric that only reads the ceiling would miss the phenomenon this paper actually found.

*The intervention ranking is a usable prior, stated as a prediction rather than a result.* The methods that worked changed how the target was represented; the methods that perturbed or shrank parameters did not. If that sorting has any harness analogue, it is that **how the harness represents what it has learned** — one general entry versus five narrow ones, which layer a piece of content sits at — should matter more than pruning rate or a size penalty. That is precisely what `docs/framing.md` already names as *move* and *promotion*. This is the paper's suggestion, not its demonstration, and it should be labelled as a hypothesis if it enters our work.

*Shrink and perturb's failure is a concrete warning.* Randomly perturbing and partially resetting the parameters was the worst of seven interventions (0.79 ± 0.21 against a 0.20 ± 0.21 reference on the residual network). The harness analogue of that operation is periodically rewriting or randomly dropping committed content. This is evidence — in a different substrate, so weak evidence, but evidence — that adding noise to a degraded state is not automatically restorative.

**What has no analogue.**

*Every quantity in the paper's mechanism is a property of continuous parameters, and none of them survives the crossing.* Hessian eigenvalues, gradient covariance, the largest curvature eigenvalue, Adam's moment estimates, weight norm, weight-matrix rank, feature rank, dead units, network width: a harness of text files has no loss surface, no gradients, no units and no width. There is nothing to be gained by looking for the "curvature of the harness" — the mechanism does not port even as a metaphor, because curvature here is defined by second derivatives that do not exist for us.

*The dead-unit analogue in particular is a trap.* An entry that is never retrieved looks like a dead unit, and the resemblance is superficial: a dead rectified-linear unit is provably stuck because its gradient is exactly zero, whereas a never-retrieved skill file is one prompt away from being retrieved. The mechanism that makes dead units matter has no counterpart. This paper's Figure 3 gives an independent reason to be sceptical of the metric even in its home domain.

*Gradient norm is not among the falsified statistics* — the paper tests parameter norm, weight-matrix rank, dead units and feature rank. Anyone citing this paper as having ruled out gradient-based diagnostics would be overreading it.

*The quantity is one-step and optimization-sense; $\Phi(h)$ is a discounted sum and generalization-sense.* Each probe measures what the network can fit *next*, with no discounting and no horizon, and it measures a training loss on data already in the buffer, not held-out performance. $\Phi(h)$ is expected discounted future stream performance on unseen tasks. Adopting this protocol wholesale would silently substitute a one-step training-loss probe for a discounted future. The honest import is: this protocol measures the one-step, in-distribution component of $\Phi(h)$, and the discount and the generalization component have to come from somewhere else.

*The paper measures none of our retention term.* Probe targets are never revisited and tasks never recur, so nothing here informs the anchor set or the retention constraint. Same limitation as the Nature paper, for the same reason.

**Where this paper disagrees with `refs/related-work/Loss-of-Plasticity/`, which matters because we cite both.**

They agree on the phenomenon: plasticity loss is real, it arises under non-stationarity including the self-induced non-stationarity of a moving bootstrap target, it is not cured by scale, and the standard optimizer toolkit is implicated rather than protective — Dohare et al. find Adam the worst method tested, Lyle et al. trace a specific Adam failure at task boundaries and fix it with two constants.

They disagree, sharply, on cause and remedy. **Dohare et al. build their entire diagnosis on three correlates — dead or dormant units, growing weight magnitude, falling rank — and their remedy (continual backpropagation) on restoring unit diversity, concluding that "sustained deep learning requires a random, non-gradient component to maintain variability and plasticity". Lyle et al. run a falsification design over three of those same statistics and reject all of them as causes: the sign of each correlation reverses across ordinary experimental choices (Figure 3), and the abstract states that plasticity loss "often occurs in the absence of saturated units."** The disagreement extends to the remedy: shrink and perturb, the nearest thing in Lyle's sweep to injecting fresh randomness, is its worst intervention, whereas Dohare et al. report shrink and perturb as one of only two methods that hold plasticity flat on permuted MNIST. Neither paper measures what the other's thesis rests on — Dohare et al. never look at curvature; Lyle et al. never run a 5,000-task stream.

Timing and personnel make this a real disagreement rather than two groups talking past each other. This paper appeared as a preprint in March 2023 and at ICML in July 2023; the Nature paper was submitted in August 2023 and published in August 2024, and Razvan Pascanu — a co-author here — was one of the three referees Nature names on it. The referee record summarized in `refs/related-work/Loss-of-Plasticity/paper_analysis.md` section 7 contains exactly this objection, pressed hard: "Right now there is very little precise information of why these 3 metrics matter and no discussion whether there are other properties that matter." The published Nature text keeps the word "correlates" and states that the correlates only "partially" explain the loss. The published record does not say which referee wrote which report, so the attribution stops there.

**What this means for us.** When project writing says "loss of plasticity", it is borrowing a name whose *mechanism* is contested in its home field and whose *correlates* have been falsified by one of the two papers we cite. The defensible position is the one both papers support: the phenomenon is real and is invisible to a metric that only reads current-task performance. The undefensible one is importing dead-unit or size-growth analogues as causes. What we should copy is the measurement discipline — fixed budget, uninformative probe distribution, difference against a reference that did not traverse the stream, and a sign-stability check before any statistic is promoted to a cause.

---

## 6. Summary

### a) One-sentence headline finding

Loss of plasticity cannot be attributed to dead units, weight norm, or the rank of features or weights — every one of those correlations reverses sign across ordinary experimental choices — and it tracks the curvature of the loss landscape instead.

### b) Quick-reference takeaways

- Plasticity is measured, not asserted: pause training, copy the parameters, fit 10 randomly drawn high-frequency target functions for exactly 2,000 optimizer steps each with the same optimizer, and report the final loss relative to what the same measurement gave at initialization. The difference form cancels the probe's intrinsic difficulty, which is what makes checkpoints comparable.
- Four statistics that the field had treated as causes — parameter norm, weight-matrix rank, number of dead units, feature rank — each correlate with plasticity loss positively in one setting and negatively in another (128 agents, Figure 3), so none of them is a cause. Correlations are weak even where they do not reverse.
- An abrupt change in the prediction target can make Adam diverge and kill 60–80% of the network's units; changing the second-moment decay from 0.999 to 0.9 and $\epsilon$ from $10^{-9}$ to $10^{-3}$ removes the failure entirely.
- Plasticity loss shows up as slower optimization, not a lower ceiling: probe learning curves get flatter as training proceeds, they do not hit an early plateau.
- Interventions that change how the target is represented beat interventions that perturb parameters. Against a no-intervention reference of 0.20 to 0.31 (change in probe loss, lower is better), a two-hot categorical output gives 0.02 to 0.12 across four architectures while shrink and perturb gives 0.23 to 0.79 — worse than doing nothing. Adding layer normalization after each hidden layer of a standard double deep Q-network improved 45 of 57 Atari games, up to +144.2% on Enduro, with 10 regressions of which one is −52.4%.
- Scaling the network reduces plasticity loss consistently but never eliminates it, even past the memory of a single graphics card.

### c) Bottom line for decision-making

Trust the falsification result without reservation and stop using the four rejected statistics as causal diagnostics; treat the curvature mechanism as the best current hypothesis rather than a finding, since the paper never subjects it to its own sign-stability test and finds no Hessian-performance relationship at Atari scale; treat the intervention ranking as a prior weakened by the absence of per-intervention tuning, seed counts and tests.

---

## 7. Reviewer Reception

**No public reviews exist.** The paper has an OpenReview page at [openreview.net/forum?id=odqQB2OXsG](https://openreview.net/forum?id=odqQB2OXsG), but it carries bibliographic metadata only: ICML 2023 was the first year the conference used OpenReview and it ran the venue in a closed configuration, publishing titles, authors, abstracts and PDFs of accepted papers while keeping review threads, scores, rebuttals and meta-reviews private to authors, reviewers and area chairs. The public review policy changed for ICML 2025, not 2023. The OpenReview interface and both API endpoints returned a challenge-verification wall rather than notes when queried directly (searched 2026-09-17).

The one piece of public signal from the process: the paper was accepted as an **oral** presentation, listed at [icml.cc/virtual/2023/oral/25517](https://icml.cc/virtual/2023/oral/25517), which at ICML 2023 was a small fraction of accepted papers.

Net takeaway: there is no external check on this paper beyond the camera-ready text and its oral designation, so the limitations in 5d stand unanswered — in particular the missing seed counts, the absent statistical tests, and the gap between the main text's "best-performing intervention" and what Figures 8 and 9 show.
