# Repo Analysis: facebookresearch/agem

**Path:** `refs/related-work/Riemannian-Walk/repo/` · **Source:** https://github.com/facebookresearch/agem · **Analyzed:** 2026-09-17 (commit `45421499`, authored 2019-02-28) · **Licence:** MIT (Copyright 2017-present, Facebook, Inc.) · **Size:** 1.9 MB working tree (3.0 MB with git history), of which 1.1 MB is the CUB and AWA image-list text files; 7,863 lines of Python across 14 files.

## Overview

- **Paper:** Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence (see `paper_analysis.md`).
- **Provenance caveat, read this first.** This is *not* a release for the RWalk paper. There is no dedicated RWalk repository: the project page hosts paper, supplementary, poster and slides only. This repository is the first author's own later code, described by its README as "the official implementation of the Averaged Gradient Episodic Memory (A-GEM) and Experience Replay with Tiny Memories in Tensorflow"; it lists `chaudhry2018riemannian` (this paper) among the four papers in its citation block, ships `RWALK` as a selectable method implemented from this paper's equations, and gives the first author's Oxford address (`arslan.chaudhry@eng.ox.ac.uk`) as the contact. It is therefore authorial code for the method but not for the paper: it reproduces the ICLR 2019 A-GEM tables (`replicate_results_iclr19.sh`), not this paper's ECCV tables, and its CIFAR split is 20 tasks of 5 classes rather than this paper's 10 tasks of 10.
- **Framework:** custom TensorFlow 1.x (graph mode, explicit `tf.assign` op lists, one `Model` class). No configuration files; every setting is an `argparse` flag with a module-level constant as default.
- **Learning algorithm:** supervised incremental classification. No reinforcement learning, no language model, no tool use. Method is selected by the string `--imp-method` from `{VAN, PI, EWC, MAS, RWALK, A-GEM, S-GEM, ER, PNN, FTR_EXT, GEM}`.
- **Base models:** a small fully-connected network (MNIST), a 4-convolution network (CIFAR-100), ResNet-18 (CUB, AWA), VGG-16 (`utils/vgg_utils.py`).
- **Key relevance here:** it contains a direct, 10-line implementation of the paper's forgetting measure, and it contains no implementation of the intransigence measure. That asymmetry is the main finding of this analysis.

## File Structure Map

```
repo/
├── README.md                      installation, dataset download, example runs, citation block
├── LICENSE                        MIT
├── model/model.py                 1,600+ lines: the single Model class; every method's ops live here
├── utils/utils.py                 metrics (compute_fgt), sample selection, episodic memory, reservoir sampling
├── utils/data_utils.py            dataset loading and task splitting for MNIST, CIFAR, CUB, AWA
├── utils/resnet_utils.py          ResNet-18 blocks
├── utils/vgg_utils.py             VGG-16 blocks
├── utils/vis_utils.py             plotting helpers
├── fc_permute_mnist.py            entry point: permuted MNIST stream
├── conv_split_cifar.py            entry point: split CIFAR-100 stream
├── conv_split_cub.py              entry point: split CUB
├── conv_split_awa.py              entry point: split AWA
├── conv_split_{cub,awa}_hybrid.py entry points: joint-embedding variants
├── replicate_results_iclr19.sh    the paper-reproduction driver for A-GEM (ICLR 2019), includes RWALK runs
├── replicate_results_er.sh        driver for the Experience Replay paper
├── dataset_lists/                 CUB and AWA train/val/test image lists and attribute pickles
└── plotting_code/agem_plots.ipynb figure generation
```

## Component Inventory

The template's ten components assume a reinforcement-learning language-model repository. This is a supervised vision repository, so six of them are absent: special token handling, environment and tool interaction, reward function, reinforcement-learning loop, supervised cold-start pipeline, and multi-GPU infrastructure (training is single-GPU; `os.environ["CUDA_VISIBLE_DEVICES"]` is the only device control). The four that apply, plus the two metric implementations that the caller asked about, are below.

### 1. Where the forgetting measure is computed — `utils/utils.py:381-392`

This is the single most important location in the repository for our purposes. It is eq. 3 of the paper, averaged, with no deviation:

```python
def compute_fgt(data):
    """
    Given a TxT data matrix, compute average forgetting at T-th task
    """
    num_tasks = data.shape[0]
    T = num_tasks - 1
    fgt = 0.0
    for i in range(T):
        fgt += np.max(data[:T,i]) - data[T, i]

    avg_fgt = fgt/ float(num_tasks - 1)
    return avg_fgt
```

Reading it against the paper: `data` is the accuracy matrix with `data[k][j]` equal to $a_{k,j}$; `np.max(data[:T,i])` is $\max_{l \in \{1,\dots,k-1\}} a_{l,j}$, the best score task `i` ever reached *before* the final row (the slice `[:T]` excludes row `T`, exactly as eq. 3's index set excludes $l = k$); `data[T, i]` is $a_{k,j}$; the division is by $k-1$. The maximum-over-the-process reference that distinguishes this measure from GEM's backward transfer is the one call to `np.max`. Note that the loop reads the *columns* of the matrix, so the convention here is `data[after_task][evaluated_task]`.

Wrappers and call sites:

| Location | What it does |
|---|---|
| `utils/utils.py:369-379` | `average_fgt_stats_across_runs(data, key)` — runs `compute_fgt` per random seed, returns mean and standard deviation across seeds |
| `utils/utils.py:358-367` | `average_acc_stats_across_runs(data, key)` — the average-accuracy measure $A_k$, computed as `np.mean(data[i][-1])`, the mean of the final row, times 100 |
| `conv_split_cifar.py:738-739` | both wrappers called at the end of the run, printed with mean and standard deviation |
| `fc_permute_mnist.py:573` | `compute_fgt(acc_mean)` printed inline in the summary line, alongside `acc_mean[-1, :].mean()` |
| `conv_split_awa.py:25`, `conv_split_cub.py` | imported, same usage |

### 2. Where the intransigence measure is computed — nowhere

Searched the whole tree for `intransigence`, `intransigent`, `a_star`, `reference_model`, `target_model`, `joint_acc`, and for any second accuracy matrix. The only hit anywhere is the word in the paper's own title inside the README citation block (`README.md:53`). The measure the paper introduces alongside forgetting is not implemented, and neither is the reference model $a_k^*$ it requires. Three structural facts in the code explain and reinforce this.

- **The accuracy matrix is deliberately lower-triangular.** In `conv_split_cifar.py:593-595`, inside `test_task_sequence`:

  ```python
  if not MULTI_TASK:
      if tt > task:
          return final_acc
  ```

  Evaluation stops as soon as it reaches a task that has not yet been trained on. Every entry $a_{k,j}$ with $j > k$ stays at its initialized zero. So this matrix supports the forgetting measure and average accuracy, and supports neither GEM's forward transfer (which needs the cell one *above* the diagonal, $R_{i-1,i}$) nor intransigence (which needs an externally trained model that is not a cell of the matrix at all).
- **A joint-training mode exists, but only for the union of all tasks.** `MULTI_TASK` is a hardcoded module constant, `False` (`conv_split_cifar.py:74`). When set to `True`, `conv_split_cifar.py:248-257` concatenates every task's training data before task 0 and unmasks all logits, producing exactly the kind of jointly trained reference the intransigence formula calls for — but for $k = T$ only, once, not per task. Turning it on gives the offline upper bound reported as a separate baseline row; it does not give the per-task sequence $a_1^*, \dots, a_T^*$.
- **Per-task accuracy is never compared to anything outside the stream.** The only quantities persisted are the accuracy matrix and timing.

The practical consequence for us: a reimplementation of intransigence starts from scratch. The forgetting half can be copied; the intransigence half is ten lines of arithmetic once someone decides what the reference model is and pays for training it, which is the choice discussed in `paper_analysis.md` section 5d.

### 3. Entry points and how a run is configured

`if __name__ == "__main__"` in each of the five per-dataset scripts; `replicate_results_iclr19.sh` is the driver. Each script defines its constants at module level and its flags in `get_arguments()`. The RWalk-relevant ones, with the values as shipped:

| Constant | `conv_split_cifar.py` | `fc_permute_mnist.py` | Paper's value |
|---|---|---|---|
| `IMP_METHOD` | `'EWC'` (line 49) | `'EWC'` | selected per run |
| `SYNAP_STGTH` (the regularization strength $\lambda$) | `75000` (line 50) | `10` | swept in Table 3 |
| `FISHER_EMA_DECAY` ($\alpha$ in eq. 6) | `0.9` (line 51) | `0.9` (line 50) | 0.9 (supplementary B.3) |
| `FISHER_UPDATE_AFTER` ($\Delta t$ in eq. 7) | `50` (line 52) | `10` (line 51) | 50 for CIFAR, 10 for MNIST (B.3) |
| `SAMPLES_PER_CLASS` | `13` (line 53) | — | 25 for CIFAR, 10 for MNIST (Table 1) |
| `TOTAL_CLASSES` / `NUM_TASKS` | `100` / `20` (line 73) | — | 100 / 10 in the ECCV paper |
| `MULTI_TASK` | `False` (line 74) | `False` | both head settings reported |

The two Fisher constants match the paper exactly; the task split and the sample budget do not, because this script targets the A-GEM setup. `replicate_results_iclr19.sh:20-22, 47-49` set RWALK's learning rate to 0.1 on MNIST and 0.03 on CIFAR with `--synap-stgth 1`.

### 4. The RWalk method implementation

Ops are built in `Model.create_fisher_ops` and `Model.create_pathint_ops`, wired together for `RWALK` at `model/model.py:264-266` and `388-390`, and executed per iteration from the training loop of each entry script.

**The regularizer, eq. 8** — `model/model.py:859-862`:

```python
elif imp_method == 'RWALK':
    reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * (f + scr)) for w, w_star,
        f, scr in zip(self.trainable_vars, self.star_vars, self.normalized_fisher_at_minima_vars,
            self.normalized_score_vars)])
```

Both weight terms enter already normalized to $[0,1]$, matching the paper's requirement that the Fisher and path terms be commensurate.

**The online Fisher, eq. 6** — `model/model.py:1086-1097`: `set_tmp_fisher` accumulates `tf.square(d)` of the gradients of `unweighted_entropy` (so this is the empirical Fisher, as the paper's Appendix A specifies), and the moving average is

```python
self.set_running_fisher = [tf.assign(f, (1 - self.fisher_ema_decay) * f + (1.0/ self.fisher_update_after) *
                            self.fisher_ema_decay * tmp) for f, tmp in zip(self.running_fisher_vars, self.tmp_fisher_vars)]
```

Note the `1.0 / fisher_update_after` factor: the average is applied once every $\Delta t$ steps over the squared gradients accumulated since the last application, which is a mean over that window rather than the per-step update eq. 6 literally describes.

**The path importance score, eq. 7** — `model/model.py:1022-1037`. The numerator, the loss decrease attributable to each parameter, is accumulated every step under a control dependency on the training op so the parameter delta is measured after the update:

```python
with tf.control_dependencies([self.train]):
    update_small_omega_ops.append(tf.assign_add(self.small_omega_vars[v],
        -(self.vanilla_gradients_vars[v][0] * (self.trainable_vars[v] - self.weights_old[v]))))
```

and the division by the Fisher-weighted distance, every $\Delta t$ steps:

```python
elif self.imp_method == 'RWALK':
    # Update the big omegas after small intervals using distance in riemannian manifold (KL-divergence)
    update_big_omega_riemann_ops.append(tf.assign_add(self.big_omega_riemann_vars[v],
        tf.nn.relu(tf.div(self.small_omega_vars[v],
            (PARAM_XI_STEP + self.running_fisher_vars[v] * tf.square(self.trainable_vars[v] - self.weights_delta_old_vars[v]))))))
```

`tf.nn.relu` is the paper's "negative scores are set to zero". `PARAM_XI_STEP = 1e-3` (`model/model.py:18`) is $\epsilon$. The PI branch immediately above (`model/model.py:1029-1032`) is the same expression with the Fisher factor removed, which is the paper's own statement that Euclidean distance recovers PI, visible as a one-term diff.

Two deviations from the paper to flag for anyone reimplementing:

1. **The factor of one half is dropped.** The paper's denominator is $\frac{1}{2}F^t_{\theta_i}\Delta\theta_i(t)^2 + \epsilon$; the code uses $\epsilon + F^t_{\theta_i}\Delta\theta_i(t)^2$. This rescales every score by 2 before normalization, so it is absorbed by the min-max normalization and is harmless here, but it means the printed score values are not the paper's.
2. **The score-averaging step is an accumulation, not an assignment** — `model/model.py:1050-1053`:

   ```python
   # For the first task, scale the scores so that division does not have an effect
   self.scale_score = [tf.assign(s, s*2.0) for s in self.big_omega_riemann_vars]
   # To reduce the rigidity after each task the importance scores are averaged
   self.update_score = [tf.assign_add(scr, tf.div(tf.add(scr, riemm_omega), 2.0))
           for scr, riemm_omega in zip(self.score_vars, self.big_omega_riemann_vars)]
   ```

   `tf.assign_add` makes this `score := score + (score + omega)/2`, that is `score := 1.5*score + 0.5*omega`. The paper's rule (the line following eq. 8) is `score := (score + omega)/2`, which would be `tf.assign`. Under the released code the accumulated score grows across tasks instead of staying bounded, which is the opposite of the stated purpose of the step — the paper's justification for averaging is precisely "to reduce the rigidity after each task", the wording the comment above the line repeats. The min-max normalization at `model/model.py:1063-1064` partly hides the effect by rescaling to $[0,1]$ after every task, but the *relative* weighting between old and new scores still drifts toward the old ones. This is unverified against the authors' intent; it is reported as a discrepancy between the released code and the published formula, not as a bug that changes any published number.

**End-of-task consolidation** — `model/model.py:1279-1305`, inside `task_updates`: at task 0 run `scale_score` (the ×2 that makes the first averaging a no-op), then `update_score`, then the min and max reductions and `normalize_scores`, then `get_fisher_at_minima` and `normalize_fisher_at_minima`, then store the weights and reset the accumulators. A `sparsify_scores` dropout path exists and is commented out at lines 1289-1295 and 1302, with a `# TODO: Tmp remove this?` beside it.

**Per-iteration driver** — `conv_split_cifar.py:442-461`: at task 0 iteration 0, seed the running Fisher (`set_initial_running_fisher`) and store the weights; every `fisher_update_after` iterations run `update_big_omega_riemann`, `set_running_fisher`, `weights_delta_old_grouped`, `reset_small_omega`; every step run `set_tmp_fisher`, `weights_old_ops_grouped`, `train`, `update_small_omega`, `reg_loss` in one `sess.run`.

### 5. Data pipeline, and the second thing that is missing: the single-head setting

`utils/data_utils.py` loads each dataset and splits it into tasks with disjoint label sets. The head setting is implemented as a logit mask fed into the graph rather than as separate output heads: `model.output_mask` is a 0/1 vector over `TOTAL_CLASSES`, and only the entries it switches on contribute to the prediction. The mechanism is clean and would support either setting.

It is only ever used for the multi-head setting. Tracing every assignment to `logit_mask` in `conv_split_cifar.py` (lines 270, 276, 280, 283, 316-317, 618-620): at training time the mask is the current task's labels, at evaluation time it is the evaluated task's own labels (`conv_split_cifar.py:618-620`), and the one place it is opened to everything, `logit_mask[:] = 1.0` at line 283, is the joint-training branch guarded by `MULTI_TASK`. Nowhere is the mask set to the union of labels seen so far, which is what the paper's single-head setting requires (Section 2.1, $\mathcal{Y}^k = \cup_{j=1}^{k}\mathbf{y}^j$).

So the released code implements neither of the two things the ECCV paper argued hardest for: not the intransigence measure, and not the single-head evaluation setting whose absence the paper says "gives an impression that IL problem is solved". Both omissions have the same cause, that this repository reproduces the A-GEM tables, which are multi-head throughout. Anyone using this code to study the ECCV paper's claims would, without noticing, be running exactly the evaluation the paper wrote its Section 6.1 against.

### 6. Sample selection and episodic memory

`utils/utils.py` holds `samples_for_each_class`, `sample_from_dataset`, `sample_from_dataset_icarl` (the mean-of-features rule), `get_sample_weights`, `update_episodic_memory`, and `update_reservior` (reservoir sampling, from the Experience Replay paper, not this one). The paper's plane-distance and entropy rules are not separately implemented under those names.

## Relevance to this project

### Components we can directly reuse

- **`utils/utils.py:381-392`, `compute_fgt`** — ten lines of NumPy over an accuracy matrix, with no dependency on TensorFlow, on images, or on anything else in the repository. It is a correct implementation of the paper's forgetting measure and can be transcribed as-is once our anchor-set re-checks produce the lower triangle of $a_{k,j}$. The only adaptation is that our scores are task scores on a harness, not classification accuracies, which the formula does not care about.
- **`utils/utils.py:358-367`, `average_acc_stats_across_runs`** — the average-accuracy measure and the per-seed mean and standard deviation wrapper, the same shape our stream-performance reporting needs.
- **The logit-mask pattern for the head setting** (`conv_split_cifar.py:270-283`, `618-620`) — as a design idea rather than code: a single switch that decides whether the system is told which task it faces, set independently at training and at evaluation time. The analogue for us is whether the harness is given a task-type label, and this repository shows the cheapest way to make that switch an experimental variable rather than a fixed assumption.

### Components we need to modify

- **Intransigence has to be written, not adapted** — there is nothing to modify. What the repository does supply is the joint-training path (`conv_split_cifar.py:248-257` under `MULTI_TASK`) that shows the shape of the reference model: concatenate all data seen so far, unmask all outputs, train normally. Our version needs it per task rather than once, or needs the cheaper specialist reference argued for in `paper_analysis.md` section 5d.
- **The accuracy matrix must be filled above the diagonal** — `conv_split_cifar.py:593-595` returns early on unseen tasks. Under score-before-update we get the cell $a_{k-1,k}$ for free from the stream itself, so our table is naturally one cell wider than this one, and the early return is exactly the line that would have to go.

### Components that don't apply

- **The entire RWalk mechanism** — Fisher matrices, path integrals over an optimization trajectory, and a quadratic penalty on parameter displacement all need a differentiable parameter vector. A harness is text; there is no Fisher of a prompt. `model/model.py:859-1120` is of interest for understanding the paper, not for reuse.
- **TensorFlow 1.x graph construction, the `Model` class, `resnet_utils`, `vgg_utils`, `data_utils`** — vision-specific and framework-obsolete.
- **`update_reservior` and the episodic-memory machinery** — these belong to A-GEM and Experience Replay, not to this paper; our memory layer is a harness component with different semantics.
- **`replicate_results_iclr19.sh`** — reproduces the A-GEM tables, not this paper's.

### Key code snippets worth studying

| Location | Why |
|---|---|
| `utils/utils.py:381-392` | the forgetting measure, complete, transcribable |
| `conv_split_cifar.py:593-595` | the three lines that make the accuracy matrix lower-triangular, and therefore the three lines that explain why forward transfer and intransigence are absent |
| `conv_split_cifar.py:248-257` with `MULTI_TASK = True` | the joint-training reference model, the thing intransigence needs, present but used only once at the end |
| `model/model.py:1029-1037` | the PI and RWalk score updates side by side: the paper's Euclidean-versus-Fisher claim as a one-term code diff |
| `model/model.py:1050-1053` | the score-averaging discrepancy against the paper's formula |
| `conv_split_cifar.py:270-283` and `618-620` | every logit-mask assignment: the mechanism supports both head settings, the code only ever uses multi-head |
