# Repo Analysis: GradientEpisodicMemory

**Path:** `refs/related-work/GEM/repo/` · **Source:** https://github.com/facebookresearch/GradientEpisodicMemory · **Analyzed:** 2026-09-16 (commit `34c6b8e`, 2018-10-22, "Fix PSD bug for quadprog, and adapt code to PyTorch 0.4."; 11 commits total, first 2017-11-09)

## Overview

- **Paper:** *Gradient Episodic Memory for Continual Learning*, NIPS 2017 — see `paper_analysis.md`
- **Framework:** plain PyTorch (written for 0.4; `torch.load`/`torch.save` of tensor tuples, no Lightning, no Hydra, no experiment tracker). 16 Python files, 1,490 lines. `requirements.txt` lists standard-library modules alongside real dependencies; the real ones are `torch`, `torchvision`, `numpy`, `quadprog`, `matplotlib`, `PIL`.
- **RL Algorithm:** none. Supervised classification only.
- **Base Model:** none — no language model. A 2-hidden-layer, 100-unit fully-connected net for MNIST (`model/common.py:22-36`) and a ResNet-18 with `nf=20` base feature maps for CIFAR-100 (`model/common.py:70-105`).
- **Key Innovation:** `model/gem.py` — one `observe(x, t, y)` call per mini-batch that (i) writes the batch into a per-task ring-buffer memory, (ii) takes one backward pass per past task to fill a gradient matrix, (iii) checks dot products against the current gradient, and (iv) on any negative one solves a dual quadratic program with `quadprog` and overwrites the gradient before `opt.step()`. Plus `metrics/metrics.py`, which turns the logged accuracy trace into the paper's ACC / BWT / FWT.

**Why this repo was analyzed:** two things are of use to us — the *evaluation harness* (how the accuracy matrix $R$ is logged during the stream and reduced to three numbers afterward) and the *gate shape* (cheap check every step, expensive correction only on violation). The weight-space projection itself does not transfer; it is documented for completeness. The standard ten-component inventory is mostly inapplicable and is given briefly at the end.

## File Structure Map

```
repo/
├── run_experiments.sh          # the whole paper: build 3 datasets, 15 model×dataset runs, plot
├── main.py                     # Continuum iterator, eval loop (life_experience), CLI, result saving
├── metrics/metrics.py          # confusion_matrix(): R-matrix rows at task boundaries → ACC/BWT/FWT
├── model/
│   ├── common.py               # MLP (Xavier-uniform init) and the nf=20 ResNet-18
│   ├── single.py               # one shared net, plain SGD — the forgetting baseline
│   ├── independent.py          # one net per task, T× narrower; optional clone from previous task
│   ├── multimodal.py           # per-task input layer, shared hidden + output (MNIST only)
│   ├── ewc.py                  # elastic weight consolidation, Fisher from one memory batch
│   ├── icarl.py                # iCaRL re-implementation: herding exemplars, nearest-mean, distillation
│   └── gem.py                  # >>> GEM: ring-buffer memory, store_grad/overwrite_grad, project2cone2
├── data/
│   ├── raw/raw.py              # wget CIFAR-100 and mnist.npz → cifar100.pt, mnist_{train,test}.pt
│   ├── mnist_permutations.py   # one random pixel permutation per task
│   ├── mnist_rotations.py      # one angle per task, drawn inside the t-th of n_tasks equal bins of [min,max]
│   └── cifar100.py             # 100/n_tasks consecutive classes per task
└── results/plot_results.py     # Figure 1: bar plot of (ACC,BWT,FWT) and first-task-accuracy curves
```

Every model exposes the same three-method interface — `Net(n_inputs, n_outputs, n_tasks, args)`, `forward(x, t)`, `observe(x, t, y)` — and `main.py:227` swaps them with `importlib.import_module('model.' + args.model)`. A method is one file; nothing else changes.

## The data stream and the evaluation harness

### The stream (`main.py:33-88`)

`Continuum` materializes the whole stream up front as a list of `[task, example_index]` pairs: for each task in order (optionally `--shuffle_tasks yes`), `--samples_per_task` random examples (1,000 for MNIST, 2,500 for CIFAR-100, out of 60,000 and 2,500 available), repeated `--n_epochs` times (1 in the paper; 1/2/5 for Table 3), shuffled within the task. `__next__` returns a mini-batch that never crosses a task boundary (`main.py:81-83` stop when the task id changes). Each yielded tuple is `(x, t, y)` with `t` an integer — the task descriptor is a Python int throughout.

### The evaluation loop (`main.py:122-151`)

```python
for (i, (x, t, y)) in enumerate(continuum):
    if(((i % args.log_every) == 0) or (t != current_task)):
        result_a.append(eval_tasks(model, x_te, args))   # accuracy on EVERY task's test set
        result_t.append(current_task)
        current_task = t
    ...
    model.observe(v_x, t, v_y)
result_a.append(eval_tasks(model, x_te, args))           # one final row
```

Two properties matter for us. First, evaluation at a task switch happens *before* `observe` sees the new task's first batch, so the row logged at the switch is "after the last example of task $i$, before any of task $i+1$" — the paper's $R_{i,\cdot}$, and the same discipline as our score-before-update. Second, evaluation is on *all* tasks every time, including tasks not yet seen, which is what makes FWT computable: the accuracy on task $i+1$ in that row is zero-shot. The full trace is finer than $T$ rows (every 100 mini-batches plus every boundary) and is saved whole; the reduction to the paper's $T \times T$ matrix happens afterward.

`eval_tasks` (`main.py:93-119`) runs the whole test set as one batch and has an off-by-one (`b_to = min(b_from + eval_bs, x.size(0) - 1)`) that drops the last test example of every task. Harmless at 10,000 (MNIST) or 500 (CIFAR-100) examples.

### From trace to metrics (`metrics/metrics.py:24-62`)

```python
baseline = result_a[0]                                   # accuracy at random init = b̄
changes = torch.LongTensor(changes + [result_a.size(0)]) - 1
result = result_a[changes]                               # the row just before each task switch → R (T×T)
acc = result.diag()                                      # R[t,t]
fin = result[nt - 1]                                     # R[T,:]
bwt = result[nt - 1] - acc                               # R[T,t] − R[t,t]
fwt = torch.zeros(nt)
for t in range(1, nt):
    fwt[t] = result[t - 1, t] - baseline[t]              # R[t−1,t] − b̄[t]
stats = [fin.mean(), bwt.mean(), fwt.mean()]
```

**Paper-versus-code discrepancy, verified.** Eqs. 3 and 4 divide by $T-1$; the code takes `.mean()` over all $T$ entries, where `bwt[T-1]` and `fwt[0]` are identically zero. Every BWT and FWT in Appendix B is therefore $(T-1)/T = 0.95$ of the paper's definition — recomputing from the Appendix B matrix for single on MNIST permutations gives $-0.2085$ by eq. 3 versus the printed $-0.1980$, and $-0.2085 \times 19/20 = -0.1981$. Ordering between methods is unaffected; absolute BWT values in the paper are 5% too small in magnitude. The `.txt` output written at `metrics.py:42-54` is byte-for-byte the format of Appendix B (baseline row, `|`, the matrix, then `Final Accuracy / Backward / Forward`).

Each run also saves a `.pt` tuple `(result_t, result_a, model.state_dict(), stats, one_liner, args)` (`main.py:252`); `plot_results.py:52` reads `stats` for the bar chart and `plot_results.py:97` reads `data[1][:, 0]` — column 0 of the full trace, first-task accuracy at every log point — for the evolution curves of Figure 1 right.

## The algorithm: `model/gem.py`

### Memory (`gem.py:117-122`, `gem.py:159-172`)

```python
self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)
self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
...
bsz = y.data.size(0)
endcnt = min(self.mem_cnt + bsz, self.n_memories)
effbsz = endcnt - self.mem_cnt
self.memory_data[t, self.mem_cnt: endcnt].copy_(x.data[: effbsz])
...
self.mem_cnt += effbsz
if self.mem_cnt == self.n_memories:
    self.mem_cnt = 0
```

A pre-allocated `[n_tasks, n_memories, n_inputs]` tensor of raw inputs (`--n_memories 256` per task in the paper's runs), written as a ring buffer with a single write cursor `mem_cnt` shared across tasks. Because the cursor is not reset at a task switch and every task has more examples than slots (1,000 or 2,500 versus 256), each task's 256 slots are all overwritten during its last 26 mini-batches and the memory ends up holding the last 256 examples — the paper's "last $m$ examples", by accident of the counts rather than by design. With `samples_per_task < n_memories` the buffer would be partially stale from the previous task. Total budget is `n_tasks × n_memories = 5,120`, the paper's $M$.

### Past-task gradients (`gem.py:174-190`, `gem.py:32-48`)

```python
if len(self.observed_tasks) > 1:
    for tt in range(len(self.observed_tasks) - 1):
        self.zero_grad()
        past_task = self.observed_tasks[tt]
        offset1, offset2 = compute_offsets(past_task, self.nc_per_task, self.is_cifar)
        ptloss = self.ce(self.forward(self.memory_data[past_task], past_task)[:, offset1: offset2],
                         self.memory_labs[past_task] - offset1)
        ptloss.backward()
        store_grad(self.parameters, self.grads, self.grad_dims, past_task)
```

One full forward-backward over all 256 stored examples of each past task, every step, flattened into column `past_task` of `self.grads` (shape `[p, n_tasks]`, `gem.py:128`). This is the $O(T)$-per-step cost the paper names as its bottleneck; nothing is cached across steps.

### The gate and the projection (`gem.py:199-213`, `gem.py:70-90`)

```python
store_grad(self.parameters, self.grads, self.grad_dims, t)
indx = torch.LongTensor(self.observed_tasks[:-1])
dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
if (dotp < 0).sum() != 0:
    project2cone2(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
    overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)
self.opt.step()
```

```python
def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()          # G : (t-1) × p
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())            # G Gᵀ
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps             # symmetrize + 1e-3 I  (the "PSD fix")
    q = np.dot(memories_np, gradient_np) * -1                   # −G g
    G = np.eye(t)
    h = np.zeros(t) + margin                                    # v ≥ margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np                    # g̃ = Gᵀ v + g
    gradient.copy_(torch.Tensor(x).view(-1, 1))
```

`quadprog.solve_qp(P, q, C, h)` minimizes $\tfrac12 v^\top P v - q^\top v$ subject to $C^\top v \ge h$, so with $q = -Gg$ the objective is $\tfrac12 v^\top G G^\top v + g^\top G^\top v$ — eq. 11 exactly — and the constraint is $v \ge \text{margin}$.

**Second paper-versus-code discrepancy.** The paper says the constant $\gamma$ is *added to $v^\star$ after* solving with $v \ge 0$; the code instead solves with the *lower bound* $v \ge \gamma$ inside the program (`h = margin`, `--memory_strength 0.5`). These are not the same solution in general — the code's version re-optimizes given the floor — though both add at least $\gamma \sum_k g_k$ of past-task direction. What the code does in words: whenever *any* past task would be hurt, the update becomes the current gradient plus at least half of every past task's memory gradient, plus whatever more the program needs to make all dot products non-negative. With margin 0.5 this is projection plus a forced replay step; with margin 0 it would be the pure projection of eq. 8. The paper reports no run at margin 0.

The check itself is $t-1$ dot products of length-$p$ vectors; the fix is a $(t-1)$-variable program on the CPU in float64 with a fresh Gram matrix each time. Nothing records how often the branch fires.

### Multi-head masking (`gem.py:141-151`)

For CIFAR-100 the single 100-way output has every logit outside `[t·5, (t+1)·5)` filled with $-10^{11}$ before the loss and before `argmax` at evaluation. The task descriptor therefore selects the head at test time. The same mask is in `single.py`, `ewc.py`, `independent.py` and `icarl.py:forward` (which restricts nearest-mean classification to the task's five class means), so every CIFAR-100 number in the paper is task-incremental.

## The baselines, briefly

- **`single.py`** (69 lines): shared net, `observe` = one SGD step. The forgetting reference.
- **`independent.py`**: `n_tasks` nets each with `n_hiddens / n_tasks` units (5 for MNIST) or ResNet-18 with `20 / n_tasks = 1` feature map; `--finetune yes` copies the previous task's weights at each switch (`independent.py:64-69`).
- **`multimodal.py`**: one input layer per task, shared hidden and output layers, log-softmax + negative log-likelihood.
- **`ewc.py`**: at each task switch, one backward pass over the stored `n_memories` examples of the finished task; squared gradients become that task's diagonal Fisher (`ewc.py:73-93`); penalty `reg · Σ_tt Σ_i F_tt[i] (θ − θ*_tt)²` summed over all past tasks (`ewc.py:113-117`). `--memory_strength` is the regularization weight here (1,000 on rotations, 3 on permutations, 1 on CIFAR-100 — the same flag means the margin for GEM, per `main.py:168`'s "meaning depends on memory").
- **`icarl.py`** (212 lines): CIFAR-100 only, asserts one epoch; accumulates the whole task, and at the last mini-batch (`examples_seen == samples_per_task`) selects herding exemplars per class so the total stays at `n_memories`, then stores the network's outputs on them as distillation targets; classification is nearest exemplar-mean within the task's classes.

## Component Inventory (standard ten)

| # | Component | Status |
|---|---|---|
| 1 | Entry points | `main.py` (CLI, 16 flags at `main.py:155-200`); `run_experiments.sh` (all 15 runs, seed 0, CPU for MNIST, GPU for CIFAR-100); three data scripts |
| 2 | Special tokens / actions | absent — no language model |
| 3 | Environment / tools | absent; the only external call is `wget` for the raw datasets (`data/raw/raw.py:18,24`) |
| 4 | Token masking | absent; the only mask is the per-task logit mask above |
| 5 | Reward | absent; loss is cross-entropy (`gem.py:108`) |
| 6 | RL loop | absent |
| 7 | SFT / cold start | absent |
| 8 | Data pipeline | `torch.save([tasks_tr, tasks_te], ...)` lists of `[descriptor, X, y]` per task; `load_datasets` (`main.py:23-30`) infers input width and class count from them |
| 9 | Multi-GPU | absent; `--cuda yes/no` only, `cudnn` disabled for determinism (`main.py:213`) |
| 10 | Evaluation | `eval_tasks` + `confusion_matrix`, described above; benchmarks are the three generated streams |

## Relevance to this project

### Components we can directly reuse

- **The evaluation-loop discipline** (`main.py:129-133`): score every task, including unseen ones, at every boundary *before* the new task's first update, keep the full trace, reduce afterward. This is score-before-update with FWT for free. Our harness runner should log the equivalent of `result_a` (per-task score vector at every commit boundary) and derive ACC / BWT / FWT in analysis code, never in the runner.
- **`confusion_matrix`'s reduction** (`metrics/metrics.py:24-62`) as the reference implementation of BWT and FWT from a trace — with the $T-1$ denominator corrected, and with the untrained-network baseline row replaced by the *base harness* scored on every task.
- **The evolution plot** (`results/plot_results.py:89-114`): first-task accuracy at every log point across the stream. The direct visual for retention; a second panel of the diagonal $R_{i,i}$ over $i$ would give the plot the paper never made — current-task learning over the stream, our third future cost.
- **The one-file-per-method, one-interface pattern** (`Net / forward / observe`, swapped by `importlib`): a method is a file, the stream and the evaluation never change. The right shape for comparing edit policies over one task stream.

### Components we need to modify

- **The gate shape** (`gem.py:205-212`): cheap check every step, expensive correction only when it fails. For us the cheap check is the critic $\hat\Phi(h)$ or a retrieval-overlap test between the proposed edit and the anchor set; the expensive correction is re-running the anchors. The *structure* carries over; the dot product does not.
- **The memory as an anchor set** (`gem.py:117-122`): per-task retained examples of fixed size, filled with the most recent items. Ours holds previously *passed* tasks and grows from the stream; the fixed-per-task budget and the "most recent" rule are the two things to change, the paper's own conclusion having flagged the second.
- **`store_grad`'s per-task column layout** (`gem.py:32-48`) is the natural shape for a per-anchor score vector at commit time, replacing gradient columns with pass/fail or score columns.

### Components that don't apply

- **`project2cone2`** and everything about eq. 7–11: gradient angles and a dual program over $\mathbb{R}^p$. A harness edit has no gradient; the constraint must be evaluated by re-scoring, not linearized.
- **The margin `memory_strength=0.5`**: an unablated tuned constant whose code semantics differ from the paper's; nothing to import.
- **`ewc.py`, `icarl.py`, `independent.py`, `multimodal.py`**: baselines for *this* paper's claim in weight space.
- **`common.py`**: two small image networks.

### Key code snippets worth studying

| File:lines | What it shows |
|---|---|
| `main.py:122-151` | the evaluation loop: when rows of $R$ are written relative to updates — read this first |
| `metrics/metrics.py:24-62` | trace → $R$ → ACC / BWT / FWT in 40 lines, and the $T$ vs $T-1$ discrepancy |
| `model/gem.py:153-213` | the whole method in one `observe`: memory write, past gradients, check, project, step |
| `model/gem.py:70-90` | the dual quadratic program and how the margin enters as a lower bound |
| `model/gem.py:141-151` | the logit mask that makes CIFAR-100 task-incremental for every method |
| `main.py:33-88` | `Continuum`: a stream materialized up front, mini-batches never crossing a task boundary |
| `results/plot_results.py:89-114` | first-task-accuracy-over-stream curves from column 0 of the saved trace |
