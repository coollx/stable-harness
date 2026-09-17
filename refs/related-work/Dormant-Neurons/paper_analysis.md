# The Dormant Neuron Phenomenon in Deep Reinforcement Learning

- **Original title**: The Dormant Neuron Phenomenon in Deep Reinforcement Learning
- **Authors**: Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci (Castro and Evci are marked as equal advising)
- **Affiliations**: Eindhoven University of Technology, The Netherlands (Sokar; the paper states the work was done while she was an intern at Google DeepMind); Google DeepMind (Agarwal, Castro, Evci); Mila (Agarwal)
- **Venue / year**: International Conference on Machine Learning 2023, Honolulu, Hawaii. Proceedings of Machine Learning Research volume 202, pages 32145–32168. Presented as an oral.
- **PMLR URL**: https://proceedings.mlr.press/v202/sokar23a.html
- **arXiv**: [arXiv:2302.12902](https://arxiv.org/abs/2302.12902) (v1 24 February 2023, v2 13 June 2023; the local `paper.pdf` is v2, 24 pages: 8 pages main text, references, Appendices A–D)
- **OpenReview**: no public review record; see section 7.
- **Code**: https://github.com/google/dopamine/tree/master/dopamine/labs/redo (stated in a first-page footnote of section 5; it is a subdirectory of Google's Dopamine reinforcement-learning framework, analyzed in `repo_analysis.md`)

**Classification: empirical.** The title, the abstract's opening sentence, and the two formal definitions (3.1 and 3.2) all name a phenomenon rather than an artifact, and the recycling method the paper proposes is five lines of pseudocode with no new machinery whose evaluation exists mainly to show that the phenomenon is causal rather than incidental. The method lens is folded into sections 2d (the algorithm as an experimental condition), 3a (findings 6 to 9) and 4e (the point-by-point comparison with the closest published reset rule) rather than given its own framework. This also keeps the classification aligned with the sibling reference `Loss-of-Plasticity`, whose structure is the same: a phenomenon plus a reset-based remedy.

**Why this reference matters here**: our project suspects that a greedily edited harness accumulates pieces nothing uses any more, and that periodically recycling those pieces keeps the harness able to improve. This paper is the closest published match: it defines a per-unit dormancy score, shows dormancy rising during training and damaging the ability to fit a new target, proposes a recycling rule, and — uniquely among the reset papers — ablates the *selection rule* against random selection and against inverted selection. Section 5f states exactly which parts transfer to a harness of text pieces measured from run logs, and which do not.

---

## 1. Research Motivation and Questions

### a) Why was this study conducted?

Two facts sat uncomfortably next to each other. In supervised learning, more parameters reliably buy more performance ("scaling laws"). In reinforcement learning, networks "lose their expressivity and ability to fit new targets over time, despite being over-parameterized" (Introduction, citing Kumar et al. 2021a and Lyle et al. 2021). The existing remedies were blunt: Igl et al. (2020) and Nikishin et al. (2022) periodically reinitialize some or all layers of the agent's network. The authors call these "somewhat drastic: reinitializing the weights can cause the network to 'forget' previously learned knowledge and require many gradient updates to recover." The study asks what is actually going wrong inside the network, so that the remedy can be targeted instead of drastic. The framing question is stated verbatim as: "Do RL agents use neural network parameters to their full potential?"

### b) What is missing in current understanding?

Prior work established *that* expressivity falls (measured by effective rank of the representation, or by the ability to fit a new target) but not *which part of the network* is responsible. Nothing in the literature gave a per-unit, layer-comparable measure of "this unit has stopped contributing", so nothing could distinguish a network that is uniformly less expressive from a network in which a growing subset of units has switched off. Without that, the only available intervention was resetting whole layers, which discards the working units along with the broken ones.

### c) Research questions

The paper does not number its hypotheses. As they function in the design:

1. Does the fraction of practically inactive units in a deep reinforcement-learning network grow steadily during training? (The existence claim; formalized as Definitions 3.1 and 3.2.)
2. Is it specific to reinforcement learning, or does supervised learning show it too? (The contrast claim.)
3. Which of the two non-stationarities of reinforcement learning — the data the agent collects, or the learning target it chases — produces it?
4. Do these units ever come back, and do they cost anything while they are down?
5. Does dormancy *cause* reduced ability to learn something new, or merely accompany it?
6. Does recycling exactly those units, and no others, repair the damage?

### d) Type of study

Failure-mode characterization with controlled ablations, followed by a proposed remedy benchmarked inside the same protocols. The object measured is a learning system's capacity utilization and its ability to fit a new target, not the quality of any single output.

---

## 2. Study Design

### a) Variables and conditions

**Independent variable of interest**: training progress, measured in gradient steps. Everything the paper calls the dormant neuron phenomenon is a rise of the dormant fraction against this axis.

**Second independent variable, and the paper's main lever**: the replay ratio, defined as the number of gradient updates performed per environment step. It is varied over $\{0.25, 0.5, 1, 2\}$ for DQN and $\{1, 2, 4, 8\}$ for DrQ. This variable matters because it separates "more training" from "more experience": raising it means more gradient updates on the same amount of collected data.

**Third independent variable**: the training algorithm under test — the base agent, the base agent with recycling, the base agent with periodic resets of the penultimate layer (Nikishin et al. 2022), and the base agent with weight decay.

**Dependent variables**: (i) the percentage of dormant units in the network, pooled across layers; (ii) agent performance (average return per game, or interquartile mean of human-normalized score aggregated across games); (iii) in one appendix table, effective rank of the penultimate-layer representation.

**Controls**: the three sharpest comparisons in the paper are each a control rather than a measurement. Supervised learning on CIFAR-10 with *fixed* labels controls for "deep networks just do this" (Figure 3). Offline reinforcement learning on a fixed dataset controls for the agent's own data collection (Figure 4). A network trained with *fixed random* targets controls for target movement (Figure 4). A freshly initialized network trained on the identical regression objective controls for the pretrained network in the fit-a-new-target probe (Figure 8).

### b) Subjects of study

| Setting | Learner | Environment / data | Scale |
|---|---|---|---|
| Discrete control, main | DQN with the convolutional network of Mnih et al. (2015); also the residual network of the IMPALA agent | 17 Arcade Learning Environment games (Asterix, Demon Attack, Seaquest, Wizard of Wor, Bream Rider, Road Runner, James Bond, Qbert, Breakout, Enduro, Space Invaders, Pong, Zaxxon, Yars' Revenge, Ms. Pacman, Double Dunk, Ice Hockey) | 40 iterations of 250K environment steps by default; 10M frames for the replay-ratio sweep; 5 seeds per game |
| Discrete control, sample-efficient | DrQ with an epsilon-greedy policy | 26 games of the Atari 100K benchmark | 400K steps; 10 seeds |
| Continuous control | Soft actor-critic, 2 hidden layers of 256 units in both actor and critic | HalfCheetah-v2, Hopper-v2, Walker2d-v2, Ant-v2 (MuJoCo) | 1M environment steps; 10 seeds |
| Supervised control | 3-convolution-plus-dense network, stochastic gradient descent, cross-entropy | CIFAR-10, 10,000 samples, with fixed labels and with labels reshuffled every 20 epochs | 100 epochs; 3 seeds |
| Offline reinforcement learning | DQN on the logged dataset of Agarwal et al. (2020) | DemonAttack, Asterix | 6M gradient steps; 5 seeds |

Two representative games (DemonAttack and Asterix) carry the analysis figures; the appendix repeats the measurements on the rest.

### c) Measurement and instruments

**The dormancy score (Definition 3.1) is the paper's central instrument.** For a neuron $i$ in layer $\ell$, with $h_i^\ell(x)$ its activation on input $x$ drawn from a distribution $D$ and $H^\ell$ the number of neurons in the layer:

$$s_i^\ell = \frac{\mathbb{E}_{x\in D}\lvert h_i^\ell(x)\rvert}{\frac{1}{H^\ell}\sum_{k\in h}\mathbb{E}_{x\in D}\lvert h_k^\ell(x)\rvert} \qquad (1)$$

In plain language: take a batch of inputs, average the absolute activation of this neuron over the batch, and divide by the average of that same quantity over all neurons in the same layer. A score of 1 means the neuron is exactly as active as the typical neuron in its layer; a score of 0.1 means it is one tenth as active. A neuron is **$\tau$-dormant** if $s_i^\ell \le \tau$.

The denominator is the whole point. The paper: "We normalize the scores such that they sum to 1 within a layer. This makes the comparison of neurons in different layers possible." (The arithmetic in that sentence is off — under equation 1 the scores in a layer sum to $H^\ell$, not to 1 — but the intent, and what the code does, is division by the layer mean, which makes one threshold usable in every layer regardless of width or activation scale. The mismatch is a known point of confusion; see `repo_analysis.md`.) In a convolutional layer a whole feature map counts as one neuron, so the average also runs over the spatial dimensions.

**Definition 3.2** promotes the score into the phenomenon: an algorithm exhibits the dormant neuron phenomenon "if the number of $\tau$-dormant neurons in its neural network increases steadily throughout training". The paper reads this as under-utilization: such an algorithm "is not using its network's capacity to its full potential, and this under-utilization worsens over time".

**Thresholds actually used.** The analysis sections of section 3 use $\tau = 0$, which is the strictest possible reading — a neuron counts as dormant only if its average absolute activation is exactly zero. The benchmarking sections loosen it: "we use a threshold of $\tau = 0.1$, unless otherwise noted, as we found this gave a better performance than using a threshold of 0 or 0.025" (section 5, Implementation details). Table 1 records $\tau = 0.025$ for the default DQN setting and 0.1 otherwise; Figure 9 is explicitly at $\tau = 0.025$; Table 4 records $\tau = 0$ for soft actor-critic. The grid searched was $\tau \in \{0, 0.01, 0.1\}$ crossed with a recycling period in $\{1000, 10000, 100000\}$, tuned once on DQN at replay ratio 1 with the convolutional network and then applied unchanged everywhere else (Appendix A).

**Probe batch.** The score is computed on a minibatch sampled from the replay buffer: 64 transitions for DQN and DrQ (Table 1), 256 for soft actor-critic (Table 4). Appendix C.3 (Figure 24) sweeps the probe batch over $\{32, 64, 256, 1024\}$ on five games and finds the identified dormant percentage "approximately the same" for all of them, which is the paper's evidence that the measurement is not batch-size sensitive.

**Set-persistence instrument.** To ask whether dormant neurons recover, the paper tracks the overlap coefficient between the currently dormant set and the historical dormant set, $\mathrm{overlap}(X,Y) = \frac{\lvert X\cap Y\rvert}{\min(\lvert X\rvert,\lvert Y\rvert)}$ (footnote 1).

### d) The proposed algorithm as an experimental condition

Recycling dormant neurons (the paper writes it *ReDo*) is Algorithm 1, complete:

> **Input**: network parameters, threshold $\tau$, training steps $T$, frequency $F$.
> For $t = 1$ to $T$: update the parameters with the regular reinforcement-learning loss; **if** $t \bmod F = 0$ **then** for each neuron $i$: **if** $s_i^\ell \le \tau$ **then** reinitialize the incoming weights of neuron $i$, and set the outgoing weights of neuron $i$ to 0.

Six properties of this rule matter and are easy to lose in a summary.

1. **The count is data-dependent, not budgeted.** Every neuron below the threshold is recycled at that step. There is no fixed fraction and no ranking; a step may recycle none or many.
2. **The incoming weights are resampled from the network's original initialization distribution** — the same distribution used to create the network, not a perturbation of the current weights.
3. **The outgoing weights are set to zero, and this is the load-bearing choice.** The paper's justification: "Note that if $\tau$ is 0, we are effectively leaving the network's output unchanged; if $\tau$ is small, the output of the network is only slightly changed." A recycled neuron therefore starts invisible to the rest of the network and has to earn its way back in through gradient descent. This is what separates recycling from the layer resets it competes with, which change the function discontinuously and then need many updates to recover.
4. **The schedule is periodic, on gradient steps**: $F = 1000$ for DQN and DrQ (Table 1), $F = 200{,}000$ for soft actor-critic (Table 4).
5. **The output layer is never recycled** (this is in the code, not the pseudocode; see `repo_analysis.md`).
6. **No protection period for recycled neurons.** A neuron recycled at step $t$ can be recycled again at $t + F$ if it is still below threshold. (The contrast with continual backpropagation, which needs such a period, is in section 4e.)

Two design alternatives were tried and reported (section 4, detailed in Appendix C.2). Initializing the outgoing weights randomly instead of to zero gives "similar or worse returns" — Figure 22 shows zero initialization ahead on Asterix, Breakout and DemonAttack across five games, and the paper's explanation is that "the newly added random weights change the output of the network". Scaling the incoming weights by the average norm of the non-dormant neurons in the same layer, instead of sampling from the initialization distribution, "performed similarly" (Figure 23).

### e) Statistical and analytical approach

Means over independent seeds with 95% confidence intervals; for aggregations across games the paper uses the interquartile mean recommended by Agarwal et al. (2021) — the mean after discarding the bottom and top 25% of normalized scores — with stratified bootstrap confidence intervals. Seeds: 5 per DQN experiment, 10 for DrQ and soft actor-critic, 3 for the CIFAR-10 control. No hypothesis tests and no multiple-comparison correction; the reported effects are large relative to the intervals.

---

## 3. Findings and Observations (primary focus)

### a) Headline findings

**Finding 1 — The dormant fraction rises steadily through training in deep reinforcement learning, and the effect is not small.**
*Evidence*: Figure 2, DQN at $\tau = 0$. Read from the figure, the dormant percentage climbs from near zero at the start to roughly 27% on DemonAttack and roughly 30% on Asterix after 6 million gradient steps, with no sign of levelling off. Appendix B repeats this for DrQ on five Atari 100K games (Figure 17, rising to between 8% and 16%) and for soft actor-critic on MuJoCo (Figure 18, where actor and critic networks on Ant-v2 rise to roughly 10% and 17% respectively over 1 billion gradient steps).
*Conditions*: 5 seeds, $\tau = 0$ — that is, these are neurons whose average absolute activation over the probe batch is exactly zero, the most conservative count available.

**Finding 2 — Supervised learning with fixed labels does the opposite; it is the moving target that produces dormancy.**
*Evidence*: Figure 3, CIFAR-10, 3 seeds. With fixed targets the dormant percentage *falls* over 100 epochs (read from the figure, from about 10% to near 2%). With labels reshuffled every 20 epochs it rises, and "the sharp increases in the figure correspond to the points in training when the labels are shuffled", settling in the 30–40% band. This is the cleanest control in the paper: identical architecture, identical optimizer, identical data, one variable changed.
*Conditions*: 3-convolution network, stochastic gradient descent, 10,000 CIFAR-10 samples (Table 5).

**Finding 3 — Of the two non-stationarities in reinforcement learning, the target is the culprit and the data collection is not.**
*Evidence*: Figure 4. In *offline* reinforcement learning, where the dataset is fixed and the agent collects nothing, the rise is still there (read from the figure, DemonAttack from about 20% to about 33%, Asterix from about 10% to about 50% over 6 million gradient steps). Replacing the moving bootstrap target with fixed random targets reduces the rise substantially (DemonAttack to about 22%, Asterix to about 20%). The paper's conclusion: "target non-stationarity in RL training is the primary source of the dormant neuron phenomenon."
*Conditions*: the offline dataset of Agarwal et al. (2020); 5 seeds.

**Finding 4 — Dormant neurons do not come back, and they are inert while they are down.**
*Evidence*: two independent tests. (i) Figure 5: the overlap coefficient between the currently dormant set in the penultimate layer and the historical dormant set rises through training to roughly 65% on DemonAttack and roughly 80% on Asterix, so "once a neuron becomes dormant, it remains that way for the rest of training". (ii) Figure 6: explicitly *pruning* every neuron found dormant during training leaves the learning curves on DemonAttack and Asterix indistinguishable from standard training. Figure 19 repeats the pruning test for soft actor-critic on Ant-v2 and HalfCheetah-v2 with the same result.
*Why this is the most quotable finding for us*: the pruning test is the direct demonstration that the score identifies genuinely unused capacity rather than quiet-but-useful capacity. It is the cheap test that any dormancy measure must pass before recycling on it is justified.

**Finding 5 — More gradient updates per unit of experience means more dormancy, and the dormancy tracks the performance collapse.**
*Evidence*: Figure 7 (and Figure 16 for five games individually). Read from the figure, the dormant percentage after 40 million frames rises monotonically with the replay ratio: roughly 15% at 0.25, 30% at 0.5, 35% at 1, and above 45% at 2. Final return moves the opposite way over the same four settings: roughly 4,300, 2,000, 1,200 and 700 on DemonAttack. The paper is careful here — "although difficult to assert conclusively, this finding could account for the difficulty in training RL agents with higher replay ratios."

**Finding 6 — A network carrying many dormant neurons is measurably worse at fitting a new target than a fresh network. This is the causal claim.**
*Evidence*: Figure 8, the paper's key probe. Take a DQN agent trained at replay ratio 1 (high dormancy) and fine-tune it by distillation toward a well-performing DQN network using an ordinary regression loss; compare against a randomly initialized network trained on the same loss. The pretrained network's loss *degrades* over 50,000 gradient steps while the randomly initialized one improves continuously. Meanwhile the pretrained network's dormant fraction keeps climbing toward 100% while the random baseline stays flat at roughly 45–50%.
*Why it matters*: this converts a correlation over the training axis into a controlled comparison against a learner that did not traverse the stream, on a target both learners can in principle fit. It is the same measurement design that the `Loss-of-Plasticity` reference uses throughout, applied once, at small scale.

**Finding 7 — Recycling dormant neurons removes most of the dormancy and turns the replay-ratio curve around.**
*Evidence*: Figure 9 ($\tau = 0.025$, replay ratio 0.25) shows the dormant percentage on DemonAttack dropping from roughly 25–30% to under 5%, with the final return at 40 million frames rising from roughly 4,000 to roughly 7,000. Figure 10, four panels, is the headline result: without recycling, the interquartile mean falls as the replay ratio rises (read from the leftmost panel: about 0.45, 0.42, 0.30, 0.12 at replay ratios 0.25, 0.5, 1, 2); with recycling it rises and then holds (about 0.47, 0.55, 0.60, 0.55). The same reversal holds with 3-step returns, with the residual architecture, and for DrQ on Atari 100K at replay ratios 1, 2, 4, 8.
*Conditions*: 17 games, 5 seeds (10 for DrQ), 10 million frames rather than the usual 200 million because the replay-ratio sweep is expensive.

**Finding 8 — The gain is not something a bigger network, a smaller learning rate, or a different activation function would have bought.**
*Evidence*: three separate controls. (i) *Width* (Figure 12): doubling and quadrupling the width of every convolutional and dense layer gives "at most a mild positive effect", and — the surprising part — the dormant percentage is "similar across the varying widths", so the extra units go dormant too. Recycling improves performance at every width and improves *more* as width grows. (ii) *Learning rate* (Figure 11): dividing the learning rate by four at replay ratio 1 reduces dormancy and improves performance but leaves the dormant percentage high (read from the figure, roughly 40% versus roughly 47% for the baseline and roughly 10% with recycling), and the recycled agent at the default learning rate still wins. (iii) *Activation function* (Appendix C.1, Figure 21): leaky rectified linear units with the default negative slope of 0.01 give "a mild decrease in the number of dormant neurons, but the phenomenon is still present". So it is not purely an artifact of rectified units saturating at zero.

**Finding 9 — Selecting the *least* used units is what makes recycling work. Recycling at the same rate with any other selection rule destroys the agent.**
*Evidence*: Figure 15, section 5.5, the ablation that matters most for us. Three arms share an identical budget: every 1000 steps, a fraction of the neurons is recycled, the fraction following a cosine schedule from 0.1 down to 0. The arms differ only in which neurons are chosen. Recycling the lowest-scoring neurons reaches roughly 3,500 average return on DemonAttack and roughly 2,200 on Asterix at 9 million frames. Recycling *random* neurons stays near 500 on both. Recycling the *highest*-scoring neurons (the paper calls this the inverse rule) collapses to near zero on DemonAttack and near 500 on Asterix. The paper's summary: "recycling active or random neurons hinders learning and causes performance collapse."

**Finding 10 — Against the two established alternatives, recycling wins on Atari and is the only one that does not visibly hurt on MuJoCo.**
*Evidence*: Figure 13 (17 Atari games, DQN at replay ratio 1): read from the bar chart, interquartile mean of roughly 0.30 for the base agent, 0.32 with periodic resets of the penultimate layer, 0.42 with weight decay, and 0.60 with recycling; "weight decay is comparable to periodic resets, but ReDo is superior to both". Figure 14 and Figure 20 (soft actor-critic on Ant-v2, HalfCheetah-v2, Hopper-v2, Walker2d-v2): the reset arm shows a sawtooth — a sharp drop after each reset followed by recovery — and recycling "is the only method that does not suffer a performance degradation".
*Conditions*: weight decay tuned over $\{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}\}$, best $10^{-5}$; reset period tuned over $\{5\times10^{4}, 10^{5}, 2.5\times10^{5}, 5\times10^{5}\}$ gradient steps, best $10^{5}$; soft actor-critic resets every $2\times10^{5}$ environment steps, following the original paper (Appendix A).

### b) Patterns and trends

The shape is monotone, not a plateau: the dormant fraction rises for as long as training continues, in every setting where it rises at all. Severity scales with the replay ratio (Finding 5) and is roughly invariant to network width (Finding 8) and to the probe batch size (Figure 24). The recycled agent's curves show the mirror pattern: dormancy drops within the first few hundred thousand gradient steps and then holds at a low level for the rest of training, rather than being pushed down repeatedly.

One trend runs the other way and the paper flags it as a puzzle: dormancy is roughly constant across widths, which "is somewhat at odds with" Sankararaman et al. (2020), who showed in supervised learning that wider networks have less gradient confusion and train faster. If that carried over to reinforcement learning, wider networks should have fewer dormant neurons. They do not.

### c) Surprises and counterintuitive results

The genuine surprise is Finding 2 in reverse: dormancy *decreases* in ordinary supervised training. The phenomenon is not a general property of gradient descent on deep networks; it is produced by chasing a target that keeps moving. That makes the paper a statement about a training regime rather than about an architecture.

The second surprise is the pruning result (Finding 4). One might expect a low-activation unit to still carry some signal, especially with the loose threshold. It carries none that the agent's return can detect.

The third is that over-parameterization does not help. The extra units acquired by quadrupling the width go dormant at the same rate, so the capacity argument for scaling reinforcement-learning networks does not survive contact with the phenomenon — which is exactly why the paper closes by suggesting recycling "can be an important component in being able to successfully scale RL networks in a sample-efficient manner".

### d) Negative and null results

- Random selection and inverted selection of recycling targets both collapse performance (Finding 9). These are reported as negative results for the alternatives and therefore as positive evidence for the score.
- Random initialization of the outgoing weights is worse than zeroing them (Appendix C.2, Figure 22).
- Scaling the incoming weights by the norm of the surviving neurons is no better than sampling from the initialization distribution (Figure 23).
- Leaky rectified linear units do not fix the phenomenon (Appendix C.1).
- Larger networks do not fix it (Figure 12).
- Lower learning rates do not fix it (Figure 11).
- On MuJoCo, recycling does not *improve* soft actor-critic; it merely avoids the degradation that resets and weight decay cause. The paper's own explanation: "we hypothesize that ReDo does not provide gains here as the state space is considerably low and the typically used network is sufficiently over-parameterized." Table 6 supports this — shrinking the actor and critic networks to a quarter and a half of their width on Ant-v2 makes recycling help (2016.18 ± 102 becomes 2114.52 ± 212 at quarter width; 3964.04 ± 953 becomes 4471.61 ± 648 at half width), which is a small effect with overlapping intervals at half width.
- The continual backpropagation utility measure, substituted into the same fixed schedule, gives "similar or worse performance" (Appendix C.4, Figure 25). Read from the figure, the two curves are nearly identical on Asterix and the activation-only score is ahead on DemonAttack. This is a null result on the *score*, and it is important: see section 4e.

### e) Robustness checks

Broad on the phenomenon, adequate on the method. The phenomenon is shown for three algorithm families (value-based, sample-efficient value-based, actor-critic), two domains (Atari, MuJoCo), two architectures (convolutional and residual), online and offline training, four replay ratios, three network widths, two activation functions, and four probe batch sizes. Hyperparameters for the method were searched once, on one agent at one replay ratio on one architecture, and transferred unchanged to every other setting — which is a real robustness claim in itself, though it means no setting other than DQN at replay ratio 1 has its own tuned threshold. Seeds are 5 or 10 per cell, lower than the 30–100 of the `Loss-of-Plasticity` reference.

---

## 4. Analysis and Interpretation

### a) Authors' explanations

The proposed mechanism is a one-way door created by the combination of rectified activations and moving targets. A rectified linear unit "consists of a linear part (positive domain) with unit gradients and a constant zero part (negative domain) with zero gradients. Once the distribution of pre-activations falls completely into the negative domain, it would stay there since the weights of the neuron would get zero gradients" (Appendix C.1). A moving target repeatedly pushes units into that domain; once there, no gradient arrives to pull them out, and the overlap and pruning results confirm they never return. This is *demonstrated* for the persistence half (Figures 5, 6) and *suggested* for the mechanism half: the leaky-activation experiment shows the story is incomplete, since giving the negative domain a non-zero gradient reduces dormancy only mildly.

The paper is careful about what it claims causally. Its strongest statement is that the phenomenon "does result in reduced expressivity and inability to adapt to new tasks", resting on Figure 8 (the fit-a-new-target probe) and on Table 7 (effective rank of the penultimate representation rises from 449.2 ± 5.77 to 470.8 ± 1.16 with recycling, on DemonAttack at 10 million frames over 5 seeds) — a 5% change in a 512-unit layer, which the paper itself calls a "preliminary" measurement.

### b) Evidence quality

Three tiers, and they should not be collapsed.

*Demonstrated*: the phenomenon exists and grows (many settings, consistent). Dormant neurons stay dormant and cost nothing while down (overlap plus pruning, two independent tests, replicated on MuJoCo). Target non-stationarity produces it and input non-stationarity does not (the CIFAR-10 control and the offline control are both single-variable manipulations). Recycling reverses the replay-ratio collapse (17 games, 4 architectures/agents, 5–10 seeds). Selection by low score is necessary (Figure 15, matched budget).

*Suggested*: that dormancy causes the loss of learning ability rather than sharing a common cause with it. Figure 8 is one probe on one game family with one distillation target, and both arms differ in far more than their dormant fraction — the pretrained network also has larger weights, a different rank, and a representation shaped by a different objective. Nothing isolates dormancy while holding those fixed. The honest reading is that Figure 8 rules out the *benign* interpretation (a network that has merely specialized) but does not isolate the mechanism.

*Speculated*: the rectified-unit saturation story, which the paper's own leaky-activation experiment partially contradicts; and the claim that dormancy explains the difficulty of high replay ratios, which the paper itself hedges as "difficult to assert conclusively".

### c) Confounds and alternative explanations

- **Recycling injects randomness, and randomness alone could be the active ingredient.** Figure 15 is the paper's answer and it is a good one: random recycling at the same rate collapses performance, so the benefit is not churn per se. But Figure 15 uses the scheduled fixed-fraction variant, not the threshold rule that the rest of the paper benchmarks, so the ablation is run on a sibling of the shipped algorithm rather than the shipped algorithm itself.
- **Threshold and schedule were tuned for the recycled arm only.** The baseline has no equivalent knob, so some of the margin in Figure 10 is tuning. The learning-rate control (Figure 11) partly addresses this by giving the baseline one extra degree of freedom.
- **Recycling and resets are compared at the resets paper's own best period**, but the reset baseline resets only the penultimate layer on Atari while recycling touches every layer. The comparison is between two published configurations, not between two mechanisms at matched scope.
- **Ten million frames instead of two hundred million.** The replay-ratio sweep — the paper's headline — is run at one twentieth of the standard budget for cost reasons. Whether the baseline recovers later is untested.
- **Dormancy is measured on a replay-buffer batch**, whose distribution itself shifts as the policy changes. A neuron could be dormant on recent data and active on the data it was trained for. The paper does not separate "this unit is dead" from "this unit is off-distribution".

### d) Generalizability

Credibly extends to: value-based and actor-critic agents with rectified activations trained on a bootstrapped target, at any replay ratio, in discrete or continuous control. The CIFAR-10 control extends the *cause* (moving targets) beyond reinforcement learning, which means any training loop whose target is regenerated from the model's own current state is a candidate.

Risky to extrapolate to: settings with no fixed unit budget; models whose adapting state is not weights; and networks whose activations are not rectified (the paper tested exactly one alternative). Also risky: reading the MuJoCo null result as evidence that recycling is harmless — it means recycling neither helped nor hurt where the network was already over-sized for the problem.

### e) Relation to continual backpropagation (the `Loss-of-Plasticity` reference)

Both papers reset low-usefulness units in place, both keep the rest of training untouched, and each ran a preliminary comparison against the other in which its own method won. The differences are precise and worth holding.

| Aspect | Recycling dormant neurons (this paper) | Continual backpropagation (Dohare et al., Nature 2024) |
|---|---|---|
| Selection criterion | Absolute threshold on the score: every neuron with $s_i^\ell \le \tau$ is recycled. The number recycled is whatever the data says. | Rank-and-rate: the fraction $\rho$ of eligible units with the lowest contribution utility is replaced. The number replaced is fixed by $\rho$. |
| What the score uses | Activation magnitude only, normalized by the layer mean, measured on one fresh probe batch at the recycling step. | Activation magnitude times the sum of outgoing weight magnitudes, maintained as a decayed running average (decay 0.99) updated every step. |
| Cost of the score | One forward pass every $F$ steps on a 64-sample batch. | Storage and an update for every unit at every step. This paper names the difference explicitly in Appendix C.4. |
| Protection for new units | None. | A maturity threshold of 1,000 to 10,000 updates, required because a new unit's outgoing weights start at zero and so its utility starts at zero. |
| Frequency | Every $F$ steps: 1,000 for DQN and DrQ, 200,000 for soft actor-critic. | Every update, at rate $\rho = 10^{-5}$ to $3\times10^{-4}$ of eligible units per layer — roughly one replacement every few hundred updates. |
| Incoming weights | Resampled from the original initialization distribution. | Resampled from the original initialization distribution. |
| Outgoing weights | Set to zero. | Set to zero, and the next layer's bias is compensated by the mean activation the removed unit contributed. |
| Optimizer state | Adam first and second moment estimates zeroed for the affected weights. | Adam moments and step counts zeroed for the replaced unit. |
| Evidence base | Reinforcement learning only; 17 to 26 games plus 4 MuJoCo tasks; 5–10 seeds. | Supervised and reinforcement streams up to 5,000 tasks; 30–100 seeds. |
| Head-to-head | Appendix C.4, Figure 25: the continual backpropagation utility substituted into this paper's fixed schedule gives "similar or worse performance". | Extended Data Figure 5b: proximal policy optimization with recycling plus weight decay beats the plain agent but loses to weight decay alone; labelled "a preliminary comparison". |

Three readings follow, and the third is the one that changes what we would build.

First, **neither head-to-head is neutral.** Each is run inside the author's own codebase, with the other method's distinctive schedule replaced by the local one — this paper strips the running average and the maturity threshold from continual backpropagation, and the Nature paper runs recycling inside its own proximal-policy-optimization stack. A referee of the Nature paper made exactly this point about the priority question. Treat both comparisons as weak.

Second, **the two selection scores are close to interchangeable.** This paper's own Appendix C.4 says so ("similar results"), and the Nature paper independently reports that its elaborate preprint utility was dropped in revision because "for activations like ReLU, the contribution utility works almost as well". Two independent groups found that making the usefulness score more sophisticated buys nothing. That is a strong, cheap result to inherit: start with the simplest usage measure.

Third, **what is not interchangeable is whether selection is by low usefulness at all.** Figure 15 here is the only experiment in either paper that holds the recycling budget fixed and varies only the selection rule, and it shows a collapse for both alternatives. So the design effort belongs in *not recycling the wrong pieces*, not in refining the score.

---

## 5. Implications, Limitations and Transferability

### a) For practitioners

If a training loop chases a target derived from the model's own current state, measure the fraction of units whose average absolute activation is far below their layer's average, and plot it against gradient steps rather than against wall-clock or reward. If it rises, extra width will not save you and a smaller learning rate will only slow it. Recycle by threshold: reinitialize the incoming weights of sub-threshold units from the initialization distribution, zero their outgoing weights so the network's function does not jump, and clear the optimizer state for both. Never recycle on a schedule without a usefulness criterion. Before recycling, run the pruning test: delete the units your criterion flags and confirm performance is unchanged; if it is not, the criterion is mislabeling useful units.

### b) For researchers

The paper names three openings. The dormancy threshold "is a hyperparameter that requires tuning; having an adaptive threshold over the course of training could improve performance even further". Recycling "reduces dormant neurons significantly but it doesn't completely eliminate them", so the initialization and optimization of recycled capacity is unfinished. And the relationship between "the task's complexity, network capacity, and the dormant neuron phenomenon" is unmapped — Table 6 is the one datapoint, showing recycling helps most when the network is small for the task.

### c) Acknowledged limitations

Better recycling approaches likely exist; dormancy is reduced, not eliminated; the threshold needs tuning; recycling gives no gain on MuJoCo where networks are over-sized for a low-dimensional state space; the expressivity measurement (Table 7) is explicitly "preliminary".

### d) Unacknowledged limitations

- The replay-ratio headline runs at 10 million frames instead of the standard 200 million, so the comparison is about early training.
- Seed counts (5 for DQN) are low for Atari, where per-game variance is large; the interquartile mean mitigates but does not remove this.
- The paper's own text about normalization ("scores sum to 1 within a layer") does not match equation 1 or the released code, which divide by the layer mean. Harmless to the results, confusing to anyone reimplementing.
- No compute accounting for recycling anywhere, despite the paper arguing in Appendix C.4 that the competing method costs more.
- No forgetting measurement. The claimed advantage over layer resets is that recycling does not discard learned knowledge, and the mechanism (zeroing outgoing weights) makes that plausible, but no experiment measures retention of anything.
- Dormancy is only ever *tested* in the direction "dormant units are useless" (pruning). It is never tested in the direction "the units the threshold spares are useful", which is the direction that matters when the threshold is loosened to 0.1.

### e) Reproducibility

Good. The official implementation is public, inside Google's Dopamine framework, under the Apache 2.0 licence, and it contains not only the shipped rule but the ablation variants (random selection, inverted selection, the continual backpropagation utility) as switchable options. Every hyperparameter is in Tables 1–5. The barrier is compute: 17 Atari games times 4 replay ratios times 5 seeds at 10 million frames each, plus the 26-game Atari 100K sweep and the MuJoCo runs.

### f) Transfer to this project

Our setting is a harness around a frozen language model — prompts, tool descriptions, skills, memory files — edited over a task stream. The suspicion this reference was fetched to test is that greedy editing leaves pieces nobody reads any more, and that recycling them keeps the harness improvable. Here is what actually carries.

**The score becomes a usage rate normalized within a layer.** Equation 1 has three properties worth preserving and only one of them is about neurons. It is measured on a probe sample of the *current* input distribution, not accumulated over all history. It is divided by the layer mean, so one threshold works in every layer. And it measures contribution to what the layer emits, not mere existence. The harness reading from run logs: for each piece $p$ in layer $\ell$, let $u_p$ be the fraction of tasks in a trailing window of the stream in which that piece was both placed in the model's context and acted on — a skill invoked, a tool called, a memory entry retrieved and quoted — and set the score to $u_p$ divided by the mean of $u$ over all pieces in the same layer. The layer normalization is not optional: a memory store with 400 entries and a tool list with 12 entries have base rates that differ by more than an order of magnitude, and a single absolute count threshold would empty the memory store while sparing every tool. The window length is the one parameter with no counterpart in the paper; it plays the role of the recycling period $F$, which the authors searched over $\{1000, 10000, 100000\}$ steps, so the corresponding sweep for us is over window lengths, not over thresholds.

**A better score is unlikely to be worth building, and the paper says so twice.** The obvious refinement is to weight usage by downstream reliance — did the retrieved entry actually change the answer — which is precisely the continual backpropagation utility (activation times outgoing weight mass). Appendix C.4 here and the revision history of the Nature paper both report that this refinement buys nothing. Build the plain usage count first, and spend the saved effort on the threshold and on the pruning test.

**Recycling a text piece: the rule has two halves and only one of them transfers cleanly.** Zeroing the outgoing weights means "make this piece invisible to the rest of the system at the instant of the edit, so the edit cannot damage the current task". In a harness that is deletion, or unlinking from the retrieval index, and Finding 4's pruning result is the argument that it is free when the piece is genuinely dormant. Reinitializing the incoming weights from the initialization distribution means "hand the freed slot a fresh draw from the distribution that made the network plastic in the first place". Of the three candidate text actions, *rewriting the piece from recent failures* is the true analogue, because it draws new content from the current data distribution the way resampling draws from the initial one; *merging a piece into a neighbour* is not a recycling action at all but what `docs/framing.md` calls a promotion, and its nearest relative in this paper's related-work section is the merge-similar-features line (Zhou et al. 2012) rather than the reset line; *deletion alone* is the first half without the second.

**Whether the second half is even needed is the sharpest disanalogy, and it should be stated before any experiment is designed.** A network has a fixed number of units, so a dormant unit is stranded capacity that must be recycled *in place* — that is the entire reason the paper reinitializes rather than prunes, and Figure 6 shows that pruning alone costs nothing and buys nothing. A harness has no fixed unit count. A dormant harness piece costs context length and retrieval noise, not a slot. So the argument for recycling in a harness cannot be the capacity argument; it has to be an interference argument, and it has to be made separately. The one place the capacity argument does survive intact is the always-resident layers — the system prompt and the tool descriptions — where context length is a hard budget and a dormant clause genuinely occupies a slot another clause could use. For memory files behind retrieval, it does not survive, and deletion there needs a different justification.

**The ablations we would owe are Figure 15's three arms, and they are cheap.** Fix a recycling budget — some number of pieces per window on a fixed schedule — and run three arms that differ only in which pieces are selected: the least used, a random selection, and the most used. The paper's result is a collapse in both of the latter two, which is what licenses the claim that the selection rule rather than the churn does the work. If our version shows random selection matching least-used selection, the dormancy story is wrong for harnesses and the benefit, if any, is from churn. Two further controls come free with the paper. The pruning control (Figure 6): delete everything the measure flags and check stream performance is unchanged, which validates the measure before anything is built on it. And the fit-a-new-target probe (Figure 8): take a heavily edited harness and a fresh one and measure how fast each can be brought to a fixed, known-good behaviour on a held-aside task — this is the closest thing in either this paper or the `Loss-of-Plasticity` reference to a direct measurement of harness plasticity $\Phi(h)$, and it is one extra arm rather than a new pipeline.

**The danger of recycling useful pieces is the paper's most transferable warning, and our granularity makes it worse.** Figure 15's inverted arm recycles at most 10% of a layer, decaying to zero over training, and it takes DemonAttack from roughly 3,500 average return to near zero. Random selection at the same budget is barely better. So the error is asymmetric: removing an inert piece is free, removing a load-bearing one is catastrophic, and the threshold is the only thing separating the two cases. The authors concede the threshold "is a hyperparameter that requires tuning", and the value they ship for benchmarking (0.1) is the loosest of the three they searched — meaning the published configuration deliberately recycles units that are not silent, accepting the risk because one unit out of 512 is a small bet. Our units are whole skills and whole memory files; one wrong deletion is a far larger fraction of the system, so the same threshold looseness is not available to us. Three guards follow. Set the threshold from the score distribution rather than as an absolute number, which the layer normalization already half does. Run the pruning control first and treat any performance drop as a failure of the measure, not of the idea. And exploit the one advantage the text setting has over the weight setting: harness content is under version control, so a recycling action is reversible and a wrong deletion is recoverable, which is not true of a reinitialized unit.

**The cause identified here has a direct harness reading, and it is testable.** The phenomenon is produced by chasing a target derived from the system's own current state (Finding 2, Finding 3). Under our score-before-update protocol the harness agent is in exactly that position: each edit is evaluated against a harness the previous edit produced, so the thing being optimized against moves because of the optimizer. The prediction that follows is sharp and cheap to test — a harness edited against a *fixed* frozen benchmark should not accumulate dormant pieces, while a harness edited against a stream it itself shapes should. That is the harness version of Figure 3, and it would be the first piece of direct evidence that the analogy holds at all.

**What does not transfer.** The mechanism, entirely: no gradients, no rectified units, no one-way door, so the reason our pieces would go dormant has to be found rather than assumed. The effective-rank measurement (Table 7), for the same reason it does not transfer from the `Loss-of-Plasticity` reference: computing a rank over embeddings of harness text measures the embedding model. The optimizer-state reset, which has no counterpart. And the absence of a protection period — this paper can afford to have none because a neuron recycled twice in a row costs nothing, whereas rewriting the same harness piece every window would be visible churn, so the maturity threshold from continual backpropagation is the detail to borrow from the *other* paper if we build this.

---

## 6. Summary

### a) One-sentence headline finding

Deep reinforcement-learning agents accumulate neurons that stop activating, the cause is the moving learning target, and recycling exactly those neurons reverses the performance collapse that more gradient updates otherwise produce.

### b) Quick-reference takeaways

- A neuron's dormancy score is its average absolute activation over a probe batch divided by the layer's average of the same quantity; a neuron is dormant when that ratio falls at or below a threshold, set to 0 for the analyses and to 0.1 (or 0.025) when benchmarking the fix. Dividing by the layer average is what makes one threshold usable in every layer.
- The dormant fraction climbs steadily with training — to roughly 27–30% on two Atari games after 6 million gradient steps at the strictest threshold, and above 45% at high replay ratios — and it climbs in value-based agents, sample-efficient agents, and actor-critic agents, online and offline.
- The cause is the moving target, not the changing data: an identical supervised network on CIFAR-10 shows dormancy *falling* with fixed labels and rising with labels reshuffled every 20 epochs, and offline reinforcement learning on a frozen dataset still shows the rise.
- Dormant neurons never recover (the overlap of the dormant set with its own history rises to 65–80%) and cost nothing while down (pruning them changes returns not at all) — but a network carrying them is measurably worse than a fresh network at fitting a new fixed target, which is the paper's causal evidence.
- The fix is five lines: every 1,000 gradient steps, for every neuron below threshold, resample the incoming weights from the initialization distribution and set the outgoing weights to zero, so the network's function is unchanged at the moment of the edit. It turns the replay-ratio curve from falling to rising (interquartile mean across 17 games roughly 0.30 to 0.60 at replay ratio 1) and beats both periodic layer resets and weight decay.
- The single most important ablation: at an identical recycling budget, selecting random neurons or the *most* active neurons collapses performance to near zero. The selection rule, not the injected randomness, is what works. Meanwhile two different selection *scores* (activation only, versus activation times outgoing weight mass) perform the same, here and in the continual backpropagation paper independently.

### c) Bottom line for decision-making

Trust the phenomenon, the moving-target cause, and the pruning and selection ablations. Treat the causal chain from dormancy to lost learning ability as supported by one probe rather than established. Treat the margin over periodic resets and weight decay as real but measured at one twentieth of the standard training budget with a threshold tuned only for the recycled arm.

---

## 7. Reviewer Reception

**No public review record exists.** ICML 2023 hosted accepted papers on OpenReview as metadata only and did not release reviews, scores, or rebuttals for any submission; the reported forum identifier for this paper is `skb34O7hFp`. Searched on 2026-09-17: a direct fetch of the forum page returns an automated-access challenge rather than content, and the OpenReview search interface returns only a bibliographic mirror record for this exact title, with no submission note carrying reviews. Nothing about the review process can be reported. Do not fabricate.

Two pieces of indirect reception are on the record and worth noting because they bear on how much of the method claim to believe. The paper was accepted as an **oral** at ICML 2023, the venue's top tier. And the referees of the `Loss-of-Plasticity` reference discussed this paper at length in Nature's published peer-review file: referee 1 called the Nature paper's citation handling "one of the weakest points" and noted that this paper had already compared against continual backpropagation and found "no real gain from the scoring method used here". The two papers' mutual comparisons are described in section 4e; each favours its own method, and neither is neutral.
