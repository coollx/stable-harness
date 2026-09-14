# Repo Analysis: loss-of-plasticity

**Path:** `refs/related-work/Loss-of-Plasticity/repo/` · **Source:** https://github.com/shibhansh/loss-of-plasticity · **Analyzed:** 2026-09-14 (commit `a6b7958`, 2026-07-14, "Minor fix RL README")

## Overview

- **Paper:** *Loss of plasticity in deep continual learning*, Nature 632, 768–774 (2024) — see `paper_analysis.md`
- **Framework:** custom PyTorch 2.1, no training framework. 56 Python files, about 6,200 lines total. Experiments are plain scripts reading JSON or YAML configs; no Hydra, no Lightning, no Hugging Face.
- **RL Algorithm:** hand-written proximal policy optimization in `lop/algos/rl/` (`ppo.py`, `agent.py`, `buffer.py`, `learner.py`); MuJoCo via `free-mujoco-py` and `gym` 0.23.1
- **Base Model:** none — no language model anywhere. Networks are small: 3-convolution + 3-fully-connected (ImageNet), an 18-layer residual network (CIFAR-100), 3 fully connected layers of 2,000 units (permuted MNIST), 1 hidden layer of 5 units (slowly-changing regression), 2 hidden layers of 256 units (policy and value networks).
- **Key Innovation:** continual backpropagation — a generate-and-test loop bolted onto an ordinary optimizer that reinitializes the lowest-utility mature units at a tiny per-step rate, plus the measurement code for the three plasticity correlates (dormant units, average weight magnitude, effective and stable rank).

**Why this repo was analyzed:** we may adapt the measurement design, so this analysis is weighted toward *how plasticity is measured in code* — where each metric is computed, what is logged versus derived, and how the reference learner that defines "plasticity" is implemented. The generate-and-test algorithm gets second billing; the standard ten-component inventory is largely inapplicable and is summarized in one table rather than forced.

## File Structure Map

```
lop/
├── algos/                          # all learning algorithms
│   ├── bp.py                       # plain backpropagation wrapper (the base system)
│   ├── cbp.py                      # ContinualBackprop: optimizer step then gnt.gen_and_test
│   ├── gnt.py                      # GnT — the generate-and-test core for fully connected nets
│   ├── res_gnt.py                  # ResGnT — same for residual nets, operates on conv out_channels
│   ├── convGnT.py, convCBP.py      # same for the ImageNet convolutional net
│   ├── cbp_linear.py, cbp_conv.py  # newer layer-style API: drop a CBPLinear between two layers
│   ├── gntRedo.py                  # GnTredo — the ReDo baseline (periodic, threshold-based)
│   └── rl/                         # custom PPO: ppo.py, agent.py, buffer.py, learner.py
├── nets/                           # architectures: conv_net, deep_ffnn, ffnn, linear,
│                                   #   policies, valuefs, torchvision_modified_resnet
├── utils/
│   ├── miscellaneous.py            # >>> rank metrics live here (effective/approximate/abs rank)
│   ├── AdamGnT.py                  # Adam variant exposing per-parameter moment reset
│   └── ltu.py                      # linear threshold units (the regression target network)
├── imagenet/                       # Continual ImageNet: multi_param_expr.py + single_expr.py,
│                                   #   cfg/{bp,cbp,cbp2,l2,snp}.json, class_order
├── incremental_cifar/
│   ├── incremental_cifar_experiment.py   # the class-incremental training loop
│   ├── post_run_analysis.py        # >>> the plasticity-measurement script (all four metrics)
│   └── cfg/{base_deep_learning_system, continual_backpropagation, head_resetting,
│            shrink_and_perturb, retrained_network}.json
├── permuted_mnist/                 # online_expr.py logs rank + dead units inline during training
├── slowly_changing_regression/     # expr.py, ~70 configs sweeping activation x optimizer x method
├── rl/                             # run_ppo.py, run.sh, cfg/{ant,sant,hopper,walker}/*.yml,
│                                   #   plots/{fig3,fig4a,fig4b,rlapp-b,rlapp-c,rlapp-d}.py
└── envs/slippery_ant.py            # Ant-v3 with resettable ground friction
```

Everything an experiment produces is a pickle of tensors; there is no results database and no experiment tracker. Plots are read back from those pickles by per-figure scripts.

## How plasticity is measured in code

This is the part worth copying. The paper's three correlates are computed in three different places with three different conventions, and the *definition of plasticity itself* is not a metric at all but a config flag on a reference run.

### The reference learner is a config flag, not separate code

The paper's operational definition — ability to learn the next task, benchmarked against a learner that did not traverse the stream — is implemented by running the *same* experiment script with one boolean flipped. `lop/incremental_cifar/cfg/retrained_network.json` is byte-for-byte `base_deep_learning_system.json` except for `"reset_network": true`, and `incremental_cifar_experiment.py:468` acts on it at every task boundary:

```python
if self.reset_network:
    self.net.apply(kaiming_init_resnet_module)
```

Plasticity loss is then the difference between two curves from two runs of one script. This is the cheapest possible implementation of the design and it is the one thing here that ports directly to a harness: an arm of the same stream run with the harness reset at each task boundary, differing by one flag, yields the reference curve.

Two conventions in the same file shape what "performance" means. `class_increase_frequency = 200` (line 130) fixes one task at 200 epochs; `early_stopping: true` makes `extend_classes` restore the best-validation parameters before measuring and before moving on (line 451-452), so the reported per-task number is best-on-validation *within* the task, not final. The learning-rate schedule is re-applied per task, hard-coded at epochs 0/60/120/160 of each increment (`set_lr`, line 418).

### Metric 1 — dormant units: three different definitions

| Where | Definition | Sample |
|---|---|---|
| `incremental_cifar/post_run_analysis.py:106` | fraction of units active on fewer than 1% of samples | one batch, `num_samples = 1000` |
| `permuted_mnist/online_expr.py:149` | units whose summed absolute activation is exactly zero | matrix of activations, every 60,000 examples |
| `rl/plots/fig4b.py:52` | activity rate logged during training, thresholded at 0.01 **in the plotting script** | trailing 1,000 environment steps |

The residual-network version is the general one:

```python
dead_neurons[layer_idx] = ((features_per_layer[layer_idx] != 0).float().mean(dim=(0, 2, 3)) < dormant_unit_threshold).sum()
dead_neurons[-1] = ((features_per_layer[-1] != 0).float().mean(dim=0) < dormant_unit_threshold).sum()
...
return dead_neurons.sum().item() / number_of_features, last_layer_activations.numpy()
```

Note what it actually is: a unit is dormant if the *rate at which it fires over a fixed probe set* falls below a threshold, and the reported number is a single scalar pooled over all layers, denominated by the total unit count. Nothing about gradients is involved despite the paper's gradient-based justification. The reinforcement-learning variant makes the pattern even clearer — `run_ppo.py:244` logs only the raw rate, `pol_features_activity[step//1000] = (short_term_feature_activity>0).float().mean(dim=0)`, and the threshold that turns a rate into a "dead unit" is applied post hoc at plot time. The threshold is a presentation choice, not an experimental one.

### Metric 2 — average weight magnitude: one line, whole network

```python
for p in net.parameters():
    num_weights += p.numel()
    sum_weight_magnitude += torch.sum(torch.abs(p))
return sum_weight_magnitude.cpu().item() / num_weights
```

`post_run_analysis.py:92`. It is the mean absolute value over *every* parameter including biases and batch-norm scales, not per-layer. The permuted MNIST runner instead logs per-layer sums every iteration (`online_expr.py:162`).

### Metric 3 — rank: four variants, two implementations

`lop/utils/miscellaneous.py:143` is the shared entry point, returning four numbers at once:

```python
sv = torch.tensor(svd(np_m, compute_uv=False, lapack_driver="gesvd"), device=m.device)  # use_scipy
rank = torch.count_nonzero(sv).to(torch.int32)
effective_rank = compute_effective_rank(sv)
approximate_rank = compute_approximate_rank(sv, prop=prop)
approximate_rank_abs = compute_abs_approximate_rank(sv, prop=prop)
```

*Effective rank* (`miscellaneous.py:165`, the paper's eq. 2) normalizes singular values into a distribution and exponentiates its entropy — $\mathrm{erank} = \exp\{-\sum_k p_k \log p_k\}$ with $p_k = \sigma_k / \lVert\sigma\rVert_1$:

```python
norm_sv = sv / torch.sum(torch.abs(sv))
entropy = torch.tensor(0.0, ...)
for p in norm_sv:
    if p > 0.0:
        entropy -= p * torch.log(p)
effective_rank = torch.tensor(np.e) ** entropy
```

*Approximate rank* squares the singular values first and counts how many reach 99% of the total; *abs approximate rank* is the same without squaring. *Stable rank* (`post_run_analysis.py:146`) is a third count-based variant and is what Figs. 2d and 4b actually plot:

```python
sorted_singular_values = np.flip(np.sort(singular_values))
cumsum_sorted_singular_values = np.cumsum(sorted_singular_values) / np.sum(singular_values)
return np.sum(cumsum_sorted_singular_values < 0.99) + 1
```

So four rank notions coexist: a hard count of nonzero singular values, a continuous entropy-based one, and two cumulative-mass thresholds. The paper reports effective rank in some figures and stable rank in others; the code confirms they are different functions, not synonyms. The `lapack_driver="gesvd"` and the CPU fallback are there because `torch.linalg.svdvals` fails to converge on GPU for large matrices — a comment in the source says so.

### When measurement happens

Three different cadences, all *between* gradient updates and all on the network as it stands:

- **Class-incremental CIFAR-100**: offline. `incremental_cifar_experiment.py` checkpoints parameters every 200 epochs (`checkpoint_save_frequency = class_increase_frequency`); `post_run_analysis.py` is a separate program run afterward over the saved checkpoints (`analyze_results`, line 225). Measurement is fully decoupled from training and costs nothing at train time.
- **Permuted MNIST**: inline, every `rank_measure_period = 60000` examples — that is, once per task — with the results accumulated into preallocated tensors (`online_expr.py:118-125`).
- **Reinforcement learning**: inline at fixed step counts, rank every 10,000 steps over the trailing 1,000-step activation buffer, activity rates every 1,000 steps (`run_ppo.py:241-245`), selected by a `to_log` list in the YAML config (`to_log: ['pol_features_activity', 'pol_weights', 'val_weights', 'stable_rank']`).

The offline pattern is the strongest of the three: checkpoint the state, measure later with a separate program that can be changed without rerunning the experiment.

## The algorithm: generate-and-test

`GnT` (`lop/algos/gnt.py:8`) is the core; `ContinualBackprop` (`cbp.py:7`) is a thin wrapper whose `learn` is an ordinary step followed by one extra call:

```python
def learn(self, x, target):
    ...  # forward, loss, opt.step()
    self.gnt.gen_and_test(features=self.previous_features)
```

Four steps per call.

**1. `update_utility` (`gnt.py:79`)** maintains the decayed running average of eq. 1. Utility type is a string switch with seven options; `contribution` is the published one, and the ImageNet config ships `adaptable_contribution` instead:

```python
output_wight_mag = next_layer.weight.data.abs().mean(dim=0)
input_wight_mag  = current_layer.weight.data.abs().mean(dim=1)
...  # 'contribution': output_wight_mag * features.abs().mean(dim=0)
bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]
```

The Adam-style bias correction matters: without it a newly created unit's utility, which starts at zero, would look low for the first $1/(1-\eta)$ steps for the wrong reason.

**2. `test_features` (`gnt.py:122`)** picks the victims. Three details are load-bearing. Eligibility is by age, not by score — `eligible_feature_indices = torch.where(self.ages[i] > self.maturity_threshold)[0]` — so a unit is protected for its first $m$ updates. The count is a fraction of the *eligible* population, not of the layer. And because that fraction is far below one unit per step at published settings, the code has an explicit sub-unit branch: either accumulate the fractional debt across steps (`accumulate=True`) or flip a biased coin (`if torch.rand(1) <= num_new_features_to_replace`). Selection is then `torch.topk(-self.bias_corrected_util[i][eligible_feature_indices], num_new_features_to_replace)`.

**3. `gen_new_features` (`gnt.py:181`)** resamples input weights uniformly inside Kaiming-style bounds, **zeroes the outgoing weights**, and compensates the next layer's bias by the mean activation the old unit contributed. Zeroing the output is what makes replacement function-preserving at the instant it happens — the new unit starts invisible to the rest of the network and earns its way back in. Ages and utilities reset to zero.

**4. `update_optim_params` (`gnt.py:207`)** zeroes the replaced unit's Adam moment estimates and step counts, via `lop/utils/AdamGnT.py`. Reinitializing a weight without clearing its optimizer state would have the old momentum immediately undo the reset.

Variants: `ResGnT` (`res_gnt.py:31`) operates on convolution `out_channels` and defaults to `util_type='weight'`, `maturity_threshold=1000`; `CBPLinear` (`cbp_linear.py:36`) is a newer nn.Module you insert between two layers and drive with forward hooks (`call_reinit`, `log_features`), which is the form to copy if adapting the idea; `GnTredo` (`gntRedo.py:5`) is the ReDo baseline and differs structurally — it fires on a period (`reset_period=1000`) and replaces every unit below an absolute threshold on normalized activity (`feature_utility = features[i] / features[i].mean()`; `feature_utility <= self.threshold`), so its replacement count is data-dependent rather than rate-controlled.

Published hyperparameters, read from the shipped configs:

| Experiment | Config | Replacement rate | Maturity threshold | Utility | Notes |
|---|---|---|---|---|---|
| Class-incremental CIFAR-100 | `incremental_cifar/cfg/continual_backpropagation.json` | `1e-5` | 1000 | `contribution` | step size 0.1, weight decay 5e-4, momentum 0.9, early stopping on |
| Continual ImageNet | `imagenet/cfg/cbp.json` | `3e-4` | 100 | `adaptable_contribution` | 30 runs, 5000 tasks, 250 epochs per task, decay rate 0.99 |
| Ant-v3 (stationary) | `rl/cfg/ant/cbp.yml` | `1e-4` | 10000 | `contribution` | `wd: 1e-4` — CBP is shipped *with* L2, matching the paper's statement that it is only reported in combination |
| Slippery Ant | `rl/cfg/sant/cbp.yml` | `1e-4` | 10000 | `contribution` | `wd: 1e-4`, friction changes every 2M steps from `cfg/frictions` |

`decay_rate` is 0.99 everywhere it appears, matching the paper.

## Component Inventory (standard ten)

Most of the standard template does not apply — this is a supervised-and-control research codebase with no language model, no tools, and no text.

| Component | Present | Where / why absent |
|---|---|---|
| 1. Entry points | yes | `imagenet/multi_param_expr.py` and `single_expr.py`; `incremental_cifar/incremental_cifar_experiment.py --config --experiment-index --verbose`; `incremental_cifar/post_run_analysis.py`; `permuted_mnist/online_expr.py`; `slowly_changing_regression/expr.py`; `rl/run_ppo.py -c cfg/ant/std.yml -s 0` and `rl/run.sh`. Configs are JSON for supervised, YAML for reinforcement learning. |
| 2. Special token / action handling | **absent** | no language model, no tokens |
| 3. Environment / tool interaction | partial | MuJoCo gym environments only (`lop/envs/slippery_ant.py` subclasses Ant-v3 to make ground friction settable); no HTTP, no subprocess, no tool calls |
| 4. Token masking | **absent** | not applicable |
| 5. Reward function | environment-supplied | standard MuJoCo returns; no custom reward shaping anywhere. Advantage is generalized advantage estimation (`g: 0.99`, `lm: 0.95` in the YAML) |
| 6. RL training loop | yes | `lop/algos/rl/ppo.py` — clipped objective, `clip_eps: 0.2`, batch 2048, 10 inner iterations over 16 slices, learning rate 1e-4, Adam `beta_1 = beta_2 = 0.99` for the tuned condition |
| 7. SFT / cold start | **absent** | no pretraining phase |
| 8. Data pipeline | yes | downsampled 32x32 ImageNet as per-class binary files with a fixed `imagenet/class_order`; CIFAR-100 through a custom `CifarDataSet` with a 450/50/100 train/validation/test split per class; MNIST loaded and permuted in memory. Results are pickles of tensors. |
| 9. Multi-GPU infrastructure | **absent** | single-GPU or CPU throughout; parallelism is across seeds by launching independent processes. No FSDP, DeepSpeed, or DDP |
| 10. Evaluation | yes | per-task accuracy during the run; `post_run_analysis.py` for the state metrics; per-figure plotting scripts (`rl/plots/fig3.py`, `fig4a.py`, `fig4b.py`, `rlapp-*.py`) that also carry the bootstrap confidence intervals (`scipy.stats.bootstrap`, `confidence_level=0.95`) |

Cost, from the subdirectory READMEs: about 12 GPU-hours per Continual ImageNet run and about 6 per CIFAR-100 run, times 30 seeds; about 24 CPU-hours per 50-million-step reinforcement-learning run; about 15 CPU-minutes per slowly-changing-regression run, which is the only experiment here anyone can rerun casually.

## Relevance to this project

### Components we can directly reuse

- **The reference-arm-as-config-flag pattern** (`incremental_cifar/cfg/retrained_network.json` versus `base_deep_learning_system.json`, acted on at `incremental_cifar_experiment.py:468`). One boolean separates the measured arm from the reference arm in a single script. Directly applicable to measuring harness plasticity as a two-arm difference over one stream.
- **The offline measurement split** (`incremental_cifar_experiment.py` checkpoints; `post_run_analysis.py` measures). Metrics computed by a separate program over saved state can be redefined without rerunning the stream — worth copying given that any harness-level plasticity statistic will be revised several times.
- **Logging the raw quantity and thresholding at analysis time** (`run_ppo.py:244` logs activity rates; `rl/plots/fig4b.py:52` applies the 0.01 cutoff). Keeps a threshold from being baked into an expensive run.
- **`scipy.stats.bootstrap` over seeds** (`rl/plots/rlapp-b.py:26`) as the uncertainty convention for stream curves, rather than parametric error bars.

### Components we need to modify

- **The utility measure.** The shape — a decayed running average of usefulness, per item, with a maturity threshold before an item can be removed — carries over to harness entries. The contents do not: `features.abs().mean(dim=0)` and `next_layer.weight.data.abs().mean(dim=0)` have no text analogue. A harness version would need a retrieval or invocation frequency in place of activation magnitude and some measure of downstream reliance in place of outgoing weight mass, and both would be counts rather than continuous quantities.
- **`test_features`'s fractional-replacement logic** (`gnt.py:145-158`) is genuinely useful and genuinely awkward. At the published rates, fewer than one unit per step is replaced, so the code either accumulates a fractional debt or flips a coin. A harness prunes far fewer, far larger items, so the same arithmetic applies with even more extreme fractions — but the accumulate branch is the one to take, since a coin flip over rare expensive edits adds variance for nothing.
- **The dormant-unit metric** would have to be re-derived, not ported. The code's definition is "fires on under 1% of a probe set", which maps onto "entry retrieved or matched on under 1% of recent tasks" — but the three implementations in this repo disagree with each other about the threshold and the probe, which is a warning that the choice is arbitrary and needs to be fixed once and documented.

### Components that don't apply

- **All rank code** (`miscellaneous.py:143-214`, `post_run_analysis.py:134-151`). Effective rank, stable rank, and both approximate ranks are functions of the singular values of an activation matrix. A harness has no activation matrix. Computing a rank over embeddings of harness entries would measure the embedding model, not the harness.
- **`compute_average_weight_magnitude`** — no parameters to sum.
- **`gen_new_features` and `update_optim_params`** (`gnt.py:181-226`) — resampling from an initialization distribution and clearing Adam moments are both weight-space operations with no counterpart in a text diff, which is reversible and inspectable in a way a reinitialized unit is not.
- **The entire PPO stack** (`lop/algos/rl/`) — a from-scratch continuous-control implementation on gym 0.23.1 and `free-mujoco-py`, unrelated to anything we would train.
- **`GnTredo`** as a baseline — it is a baseline for *this* paper's claim, not for ours.

### Key code snippets worth studying

| File:lines | What it shows |
|---|---|
| `lop/incremental_cifar/post_run_analysis.py:91-151` | all four plasticity metrics in sixty lines; read this first |
| `lop/incremental_cifar/incremental_cifar_experiment.py:445-470` | the task boundary: early-stopping restore, class extension, head reset, full network reset — where the reference arm diverges |
| `lop/utils/miscellaneous.py:143-181` | the rank family and why there are four of them |
| `lop/algos/gnt.py:122-180` | `test_features` — maturity gating, fractional replacement, top-k selection |
| `lop/algos/gnt.py:181-206` | `gen_new_features` — zeroing outgoing weights and compensating the downstream bias, the trick that makes replacement function-preserving |
| `lop/algos/cbp_linear.py:36-146` | the hook-based layer API; the form to imitate if the mechanism is adapted |
| `lop/rl/run_ppo.py:190-250` | inline metric logging at fixed step counts driven by a `to_log` list |
| `lop/algos/gntRedo.py:45-64` | the contrast: threshold-and-period replacement instead of rate-controlled, and why the replacement count becomes data-dependent |
