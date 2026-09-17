# Repo Analysis: dopamine (labs/redo)

**Path:** `refs/related-work/Dormant-Neurons/repo/` · **Source:** https://github.com/google/dopamine (the ReDo code lives at `dopamine/labs/redo/`) · **Analyzed:** 2026-09-17 (commit `5873f54`, 2026-03-24, "Replace `gsutil` commands with `gcloud storage` in Dopamine Colab notebooks")

## Overview

- **Paper:** *The Dormant Neuron Phenomenon in Deep Reinforcement Learning*, ICML 2023 — see `paper_analysis.md`
- **Framework:** Google's Dopamine reinforcement-learning framework. The agents are JAX plus Flax with optax optimizers; all configuration is gin (text files of `Class.attribute = value` bindings). The one exception is the continuous-control path, which is TensorFlow plus TF-Agents. Apache 2.0 licence. The clone is 39M; the ReDo subdirectory is 12 files and 3,186 lines.
- **RL Algorithm:** the recycler is agent-agnostic and is wired into four agents — DQN, Rainbow, the Atari 100K Rainbow variant (the paper's DrQ setting), and soft actor-critic.
- **Base Model:** none. Networks are the 3-convolution-plus-dense network of Mnih et al. (2015), the residual network of the IMPALA agent, and 2-hidden-layer fully connected actor and critic networks of 256 units.
- **Key Innovation:** `weight_recyclers.py` — one file holding the dormancy score, the threshold rule, the incoming and outgoing weight reset, the optimizer-state reset, and — importantly for us — the three ablation arms of Figure 15 as switchable options on one class.

**Why this repo was analyzed:** the task was to pin down five mechanisms exactly — how the dormancy score is computed, what threshold is applied, how incoming and outgoing weights are reset, whether optimizer state is cleared, and on what schedule. All five live in `dopamine/labs/redo/weight_recyclers.py`, so this analysis is weighted heavily toward that file and toward the gap between what the paper reports and what the shipped configs actually set. The standard ten-component inventory is largely inapplicable (no language model, no tools, no text) and is compressed into one table.

## File Structure Map

```
dopamine/labs/redo/
├── weight_recyclers.py                  # 746 lines — ALL the mechanism; read this first
│                                        #   BaseRecycler      : score, logging, schedule predicates
│                                        #   LayerReset        : the Nikishin et al. reset baseline
│                                        #   NeuronRecycler    : the shipped ReDo rule (threshold-based)
│                                        #   NeuronRecyclerScheduled : the Fig. 15 ablation harness
├── recycled_dqn_agents.py               # 262 — JaxDQN subclass; replay-ratio loop; where recycling is called
├── recycled_rainbow_agent.py            # 210 — same wiring for Rainbow
├── recycled_atari100k_rainbow_agent.py  # 232 — same wiring for the DrQ / Atari 100K setting
├── networks.py                          # 341 — 4 networks, each tagging its post-ReLU activations
├── train.py                             # 158 — entry point: --base_dir --gin_files --gin_bindings
├── configs/
│   ├── dqn_dense.gin                    # DQN, nature network, replay ratio 1
│   ├── DrQ_eps_dense.gin                # Atari 100K
│   └── rainbow_dense.gin
└── tfagents/
    ├── sac_train_eval.py                # 1223 — the whole MuJoCo path, TensorFlow, self-contained
    └── configs/sac_mujoco_dense_config.gin
```

There is no README in `labs/redo/`, no requirements file of its own, and no test. The subdirectory inherits Dopamine's top-level `setup.py`. Nothing here writes results to disk in a structured form; metrics go through Dopamine's `collector_dispatcher` to the console or TensorBoard.

## Component Inventory

### The dormancy score (`weight_recyclers.py:303-327`)

Twelve lines, and every claim in the paper about equation 1 rests on them:

```python
reduce_axes = list(range(activation.ndim - 1))
if self.sub_mean_score or is_cbp:
  activation = activation - jnp.mean(activation, axis=reduce_axes)

score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
if not is_cbp:
  # Normalize so that all scores sum to one.
  score /= jnp.mean(score) + 1e-9

return score
```

Four things to read off it. The reduction runs over every axis except the last, so for a convolutional activation of shape (batch, height, width, channels) the whole feature map averages down to one number per channel — a channel is one neuron, as the paper says. The normalizer is `jnp.mean(score)`, the **layer mean**, so a score of 1 means "as active as the typical unit in this layer" and the scores in a layer sum to the layer width. The code comment says "sum to one" and the paper's section 3 says the same; both are wrong about the arithmetic and right about the intent, which is to make one threshold usable in layers of different widths and activation scales. Mean subtraction is off by default (`sub_mean_score=False`) and is forced on only for the continual-backpropagation score variant. The `1e-9` guards a layer that has gone entirely silent.

The activations reaching this function are **post-ReLU** outputs, tagged in the network definition. `networks.py:40-44` wraps each layer's activation in a named identity module:

```python
def _record_activations(self, x, layer):
  if self.is_initializing():
    name = '/'.join(layer.scope.path)
    self.layer_names.append(name)
  return IdentityLayer(name=f'{layer.name}_act')(x)
```

and `recycled_dqn_agents.py:249-262` harvests them with Flax's `capture_intermediates`, filtering on `'act' in l.name`. The probe batch is drawn fresh from the replay buffer at `recycled_dqn_agents.py:238-247` with `batch_size_statistics` transitions (default 256 in code; the paper's Table 1 says 64 for DQN and DrQ).

### The threshold (`weight_recyclers.py:567-570`)

```python
def _score2mask(self, activation, param, next_param, key):
  del key, param, next_param
  score = self.estimate_neuron_score(activation)
  return score <= self.dead_neurons_threshold
```

That is the entire selection rule of the shipped algorithm: an absolute comparison, no ranking, no budget. The number recycled at a step is whatever the data produces. The default is `dead_neurons_threshold=0.0` (`:147`), which recycles only exactly-silent units; the paper benchmarks at 0.025 and 0.1, which must be supplied as a gin binding.

The same threshold is reused for *reporting* at `:261-302` (`log_dead_neurons_count`), which counts `jnp.count_nonzero(m <= self.dead_neurons_threshold)` per layer, **skips any key containing `'final_layer'`**, and returns one pooled percentage over all remaining layers plus per-layer breakdowns. So the dormant percentages plotted in the paper exclude the output layer and are denominated by total unit count across layers — the same pooling convention as the `Loss-of-Plasticity` repo.

### The reset of incoming and outgoing weights (`weight_recyclers.py:473-565`)

`recycle_dead_neurons` runs four stages. First `create_masks` (`:572-650`) turns the per-neuron boolean into weight-shaped masks and, along the way, **zeroes the recycled neuron's bias** (`:637-642`) — an operation the paper's Algorithm 1 does not mention:

```python
bias_key = 'params/' + k + '/bias'
new_bias = jnp.zeros_like(param_dict[bias_key])
param_dict[bias_key] = jnp.where(neuron_mask, new_bias, param_dict[bias_key])
```

Then the incoming weights are resampled by `weight_reinit_random` (`:65-111`), which draws from `nn.initializers.xavier_uniform()` — the same initializer the networks are built with (`networks.py:55`), so "the original weight distribution" of the paper is literal. The optional `weight_scaling` branch is the alternative reported in Appendix C.2 and Figure 23: it normalizes the fresh weights and rescales them to the mean norm of the *non*-recycled neurons in the layer. It is off by default.

Then the outgoing weights are zeroed (`:547-551`):

```python
elif self.init_method_outgoing == 'zero':
  weight_zero_reset_fn = jax.jit(functools.partial(jax.tree.map, weight_reinit_zero))
  params = weight_zero_reset_fn(params, outgoing_mask)
```

`init_method_outgoing='zero'` is the default (`:409`); setting it to `'random'` reproduces the worse-performing variant of Figure 22. Zeroing is what makes the edit function-preserving at the instant it happens, which is the same trick continual backpropagation uses (`Loss-of-Plasticity` `repo_analysis.md`, `gnt.py:181`), except that continual backpropagation additionally compensates the downstream bias for the removed unit's mean contribution and this code does not.

The mask expansion is the fiddly part and is worth reading if the pattern is ever reimplemented: `create_mask_helper` (`:652-697`) has to broadcast a 1-D neuron mask onto a 4-D convolution kernel in one direction and a 2-D dense kernel in the other, and it handles the convolution-to-dense flatten by repeating each feature-map entry across the spatial positions that the flatten produced.

### The optimizer-state reset (`weight_recyclers.py:554-564`)

```python
reset_momentum_fn = jax.jit(functools.partial(jax.tree.map, reset_momentum))
new_mu = reset_momentum_fn(opt_state[0][1], incoming_mask)
new_mu = reset_momentum_fn(new_mu, outgoing_mask)
new_nu = reset_momentum_fn(opt_state[0][2], incoming_mask)
new_nu = reset_momentum_fn(new_nu, outgoing_mask)
opt_state_list = list(opt_state)
opt_state_list[0] = optax.ScaleByAdamState(opt_state[0].count, mu=new_mu, nu=new_nu)
```

with `reset_momentum(momentum, mask) = momentum * (1.0 - mask)` at `:51-53`. Both Adam moment estimates are zeroed, for both the incoming and the outgoing weights, and **the step count is carried over unchanged** — so Adam's bias correction is not restarted for the recycled weights. That differs from continual backpropagation, which resets the per-unit step count as well (`lop/utils/AdamGnT.py`). The paper does not mention the optimizer reset at all; it exists only here.

### The schedule (`weight_recyclers.py:144-163`, `:168-169`, `:451-456`)

| Knob | Code default | Paper value (Table 1 / Table 4) | Set by shipped gin config? |
|---|---|---|---|
| `reset_period` | `200_000` | 1,000 gradient steps for DQN and DrQ; 200,000 for soft actor-critic | **no** |
| `dead_neurons_threshold` | `0.0` | 0.025 default DQN setting, 0.1 otherwise; 0 for soft actor-critic | **no** |
| `reset_start_step` | `0` | not reported | no |
| `reset_end_step` | `100_000_000` | not reported | yes: 2,500,000 (DQN), 400,000 (Atari 100K) |
| `logging_period` | `20_000` | not reported | no |
| `batch_size_statistics` | `256` | 64 (DQN, DrQ), 256 (soft actor-critic) | no |

The firing predicate is two lines:

```python
def is_update_iter(self, step):
  return step > 0 and (step % self.reset_period == 0)
```

gated by `is_reset` (`:451-456`) to the window `[reset_start_step, reset_end_step)`. The step counter is the **gradient** step, not the environment step: `recycled_dqn_agents.py:155-166` runs `_num_updates_per_train_step` inner updates per environment step and increments `self.gradient_step` inside that loop. This matters for reading the paper — at replay ratio 2, a recycling period of 1,000 gradient steps fires twice as often per unit of experience.

**The shipped configs do not reproduce the paper.** `configs/dqn_dense.gin` sets neither `reset_period` nor `dead_neurons_threshold`, so a plain `train.py --gin_files=...dqn_dense.gin` run recycles every 200,000 gradient steps at a threshold of exactly zero, not every 1,000 at 0.025. It also selects `reset_mode='neurons_scheduled'` territory by binding `NeuronRecyclerScheduled.score_type` and `recycle_rate`, meaning the shipped config is aimed at the **ablation** variant rather than at the headline algorithm. Reproducing Figure 10 requires supplying `--gin_bindings` for the period, the threshold, and `RecycledDQNAgent.reset_mode='neurons'`.

### The ablation harness (`weight_recyclers.py:700-746`)

This is the most useful sixty lines in the repo for our purposes, because it is Figure 15 in executable form. `NeuronRecyclerScheduled` subclasses the recycler and replaces only `_score2mask`, so selection changes while everything else — period, reset mechanics, optimizer handling — stays fixed:

```python
score = self.estimate_neuron_score(activation, is_cbp=is_cbp)
if self.score_type == 'redo':
  pass
elif self.score_type == 'random':
  new_key = random.fold_in(key, self._last_update_step)
  score = random.permutation(new_key, score, independent=True)
elif self.score_type == 'redo_inverted':
  score = -score
# Metric used in Continual Backprop pape.
elif self.score_type == 'cbp':
  ...
  score *= jnp.sum(jnp.abs(next_param), axis=next_axes) / jnp.sum(jnp.abs(param), axis=current_axes)
multiplier = max(0, self._last_update_step / self.reset_end_step)
ones_fraction = float(jnp.cos(jnp.pi * 0.5 * multiplier))
ones_fraction *= self.recycle_rate
return leastk_mask(score, ones_fraction)
```

Four selection rules on one switch: the activation score, a random permutation of that score (so the *count* is preserved and only the *identity* of the victims is randomized), the negated score (recycle the most active), and the continual-backpropagation contribution utility. The last one is a genuinely faithful port of the competing metric's formula — activation magnitude times outgoing weight mass, here additionally divided by incoming weight mass — but it is computed fresh at each recycling step rather than as the decayed running average the Nature paper maintains, and there is no maturity threshold protecting new units. That is the shape of the comparison behind Appendix C.4 and Figure 25, and it is why the comparison should be read as weak in both directions.

The budget is the other half. `leastk_mask` (`:27-48`) negates the scores, sorts, and keeps the `k` lowest where `k = max(1, round(n * ones_fraction))` — so unlike the shipped rule this one recycles a **fixed fraction**, which is exactly what makes the three arms comparable. `ones_fraction` follows a quarter-cosine from `recycle_rate` down to zero across `reset_end_step`, giving the paper's "cosine schedule starting at 0.1 ending at 0". The shipped `dqn_dense.gin` sets `recycle_rate = 0.3`, not 0.1.

### How recycling is wired into the training loop (`recycled_dqn_agents.py:177-236`)

The order within one gradient step is: sample a batch, compute gradients, apply the optax update, **then** decide whether to measure and recycle. Measurement is skipped entirely unless needed (`is_intermediated_required` at `:458-461` returns true only on a logging step or a recycling step), so the extra forward pass costs nothing on ordinary steps. Selection of the recycler is a four-way switch at `:128-140`: `'neurons'` gives the shipped rule, `'neurons_scheduled'` the ablation harness, `'weights'` the layer-reset baseline, and `None` a `BaseRecycler` that only logs dormancy — which is how the paper's phenomenon figures are produced for unmodified agents. The weight-decay baseline is three lines at `:117-123`, `optax.chain(optax.add_decayed_weights(weight_decay), self.optimizer)`.

### The baselines

`LayerReset` (`:329-389`) is the Nikishin et al. periodic-reset baseline: it reinitializes whole layers from `reset_start_layer_idx` onward, zeroes their biases, and clears the same Adam moments. `NeuronRecycler.__init__` (`:392-441`) restricts scope in two ways worth knowing — `self.reset_layers = self.reset_layers[:-1]` with the comment "we don't recycle the neurons in the output layer", and for the residual architecture only layers named `Conv_1`, `Conv_3` or `Dense` are eligible.

### The continuous-control path (`tfagents/sac_train_eval.py`)

A separate 1,223-line TensorFlow implementation that duplicates the mechanism rather than sharing it. Same score at `:518-529` (`neurons_score / (tf.reduce_mean(neurons_score) + 1e-9)` — the layer-mean normalization again), same threshold test at `:743-750`, same incoming-random / outgoing-zero / bias-zero / momentum-clear sequence at `:781-891`, plus an option absent from the JAX path: `reset_target_models`, which applies the same reset to the target network. It also carries two extra score variants at `:530-558` — `activation_grad` (activation times its gradient) and `connected_weights` (incoming weight mass times outgoing weight mass, with no activation term at all). The agent defaults are `reset_algo='low_score'`, `neuron_score_algo='activation'`, `dead_neuron_threshold=0.0`, `recycled_incoming_scaler=1`, `recycled_outgoing_scaler=0` (`:292-297`). Note the asymmetry: the neuron path only implements `low_score` (`:768-771` raises on anything else, with a TODO saying random and high-score are unimplemented here), so the Figure 15 ablation exists **only** in the JAX code. The shipped `sac_mujoco_dense_config.gin` has `reset_mode = None` and `reset_frac = 0.0`, so out of the box it runs the unmodified agent.

### Standard ten-component inventory

| Component | Present | Where / why absent |
|---|---|---|
| 1. Entry points | yes | `train.py --base_dir --gin_files --gin_bindings` for all Atari agents; `tfagents/sac_train_eval.py` for MuJoCo. Configuration is entirely gin. |
| 2. Special token / action handling | **absent** | no language model |
| 3. Environment / tool interaction | partial | Arcade Learning Environment and MuJoCo through Dopamine's standard wrappers; no HTTP, no subprocess, no tools |
| 4. Token masking | **absent** | not applicable. The word "mask" throughout this repo means a weight-shaped boolean selecting which parameters to reinitialize. |
| 5. Reward function | environment-supplied | clipped Atari rewards and standard MuJoCo returns; no shaping |
| 6. RL training loop | yes | `recycled_dqn_agents.py:155-175` — the replay-ratio loop, with a `target_update_strategy` switch choosing whether the target network syncs on environment steps or gradient steps (`:164-173`), a detail that matters at high replay ratios and is flagged by a comment in the source |
| 7. SFT / cold start | **absent** | `min_replay_history` warm-up only |
| 8. Data pipeline | yes | Dopamine's circular replay buffer; the offline experiments read the logged dataset of Agarwal et al. through `--replay_dir` |
| 9. Multi-GPU infrastructure | **absent** | single-device JAX; parallelism is across seeds and games by launching independent processes |
| 10. Evaluation | partial | metrics flow to `collector_dispatcher` (console or TensorBoard); the human-normalized aggregation, interquartile means and bootstrap intervals of the paper's figures are **not** in this repo |

## Relevance to this project

### Components we can directly reuse

- **The layer-mean normalization of the usage score** (`weight_recyclers.py:322-324`). One line, and it is what lets a single threshold apply to a 32-channel convolution and a 512-unit dense layer at once. The harness analogue — dividing a piece's usage rate by the mean usage rate of pieces in the same layer — is the only part of equation 1 that transfers unchanged, and it is the part that makes a threshold portable across a 12-entry tool list and a 400-entry memory store.
- **The selection-only ablation pattern** (`NeuronRecyclerScheduled`, `:700-746`). One subclass overriding one method, so the budget, the schedule and the edit mechanics are provably identical across arms and only the choice of victim varies. This is the cleanest experimental design in either this repo or the `Loss-of-Plasticity` repo, and it is the shape our own random-versus-least-used-versus-most-used comparison should take: one code path, one switch.
- **Measuring only when about to act** (`:458-461` plus `recycled_dqn_agents.py:210-216`). The probe is computed on logging steps and recycling steps and skipped otherwise. For us the equivalent is deriving usage counts from run logs on a window boundary rather than instrumenting every task.
- **Threshold reuse between reporting and acting** (`:261-302` and `:567-570` use the same `dead_neurons_threshold`). Keeps the number in the plot and the number driving the edit from drifting apart.

### Components we need to modify

- **The probe distribution.** The batch comes from the replay buffer (`recycled_dqn_agents.py:238-247`), so "dormant" means "silent on recent experience", not "silent in general". Our window-over-run-logs measure inherits exactly this property and exactly this ambiguity: a harness piece unused in the last 50 tasks may be a piece whose task type has not recently appeared. The repo offers no separation of the two cases and neither would a naive port.
- **The count-versus-budget choice.** The shipped rule (`:567-570`) recycles everything below threshold with no cap; the ablation harness (`leastk_mask`, `:27-48`) recycles a fixed fraction. For a harness the uncapped version is the wrong default — a threshold that misfires deletes an unbounded number of skills in one edit — so the fixed-fraction path is the one to adapt, with the fraction small and the cosine decay replaced by something we can justify.
- **The per-layer scope restriction** (`:436-441`, keeping only `Conv_1`, `Conv_3` and `Dense` for the residual network, and `:432` dropping the output layer). The idea that some layers are simply not eligible for recycling ports directly and should: an always-resident system prompt and a retrieval-backed memory store have different risk profiles, and the code's precedent is that eligibility is declared up front rather than inferred.

### Components that don't apply

- **All weight-space mechanics**: `weight_reinit_random` (`:65-111`), `weight_reinit_zero` (`:56-62`), `create_mask_helper` (`:652-697`), and the Adam moment clearing (`:554-564`). Resampling from Xavier-uniform, broadcasting masks onto convolution kernels, and zeroing second-moment estimates have no counterpart in a text edit. The *reason* for zeroing the outgoing weights — make the edit invisible to the current task so it cannot cause damage at the moment it lands — does transfer, but it arrives as a design principle, not as code.
- **`LayerReset`** (`:329-389`) — it is a baseline for this paper's claim, not for ours.
- **The whole TF-Agents path** (`tfagents/sac_train_eval.py`) — a duplicate implementation for continuous control, on a framework we do not use.
- **The `activation_grad` score** (`:530-545` in the SAC path) — requires gradients of activations, which do not exist for us.

### Key code snippets worth studying

| File:lines | What it shows |
|---|---|
| `dopamine/labs/redo/weight_recyclers.py:303-327` | the dormancy score in twelve lines, including the layer-mean normalization and the comment that misstates it |
| `dopamine/labs/redo/weight_recyclers.py:567-570` | the entire selection rule of the published algorithm: one absolute comparison, no ranking, no budget |
| `dopamine/labs/redo/weight_recyclers.py:473-565` | the full recycle: masks, incoming resample, outgoing zero, Adam moments cleared — the four operations the paper describes as two |
| `dopamine/labs/redo/weight_recyclers.py:637-642` | the bias zeroing that Algorithm 1 omits |
| `dopamine/labs/redo/weight_recyclers.py:700-746` | the Figure 15 ablation harness — four selection rules behind one switch, budget held fixed; the design to copy |
| `dopamine/labs/redo/weight_recyclers.py:27-48` | `leastk_mask` — fixed-fraction selection, the alternative to thresholding |
| `dopamine/labs/redo/recycled_dqn_agents.py:155-236` | where recycling sits in the training step, and why the schedule counts gradient steps rather than environment steps |
| `dopamine/labs/redo/networks.py:40-44` | tagging post-activation outputs with named identity layers so they can be harvested later without changing the forward pass |
| `dopamine/labs/redo/configs/dqn_dense.gin` | the shipped configuration, and the gap between it and the paper's Table 1 |
