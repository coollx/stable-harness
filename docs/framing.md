# Research framing

The research identity: what this project claims, in what setting, and the words it uses. Edited rarely — additive amendments batch after they survive; retractions are edited in immediately. Items marked (open) are hypotheses or undecided details to be revisited.

## Claim

Optimizing a harness for current-batch task success trades away three kinds of future performance: generalization to later tasks, retention of earlier tasks, and the ability to keep patching. All three are performance on the task stream that the current batch cannot measure.

Harness plasticity $\Phi(h)$ is the future stream performance achievable from harness $h$. Greedy per-batch patching lowers it (loss of plasticity). Every existing self-improving harness is the $\gamma = 0$ case: it scores an edit by the batch it was made on and nothing after.

Claim: a harness agent that balances single-turn task performance against harness plasticity sustains stream performance where greedy systems peak and then decline.

Falsifier: a plasticity-blind, per-edit-gated system (Harness Continual Learning with a swept retention budget) matches the integral of stream performance at equal harness size.

## Setting

- Object: a harness around a language model, decomposed into layers ordered by depth — system prompt, tool description, tool implementation, middleware, skill, sub-agent configuration, long-term memory, model weights. An edit may add, remove, rewrite, merge, or move content across layers. (open: exact proposer action space.)
- Regime: a heterogeneous, non-stationary task stream. Each task is scored with the harness as it stood before the task arrived (score-before-update); labels may arrive late.
- Objective, per edit: $\max_{a_t}\ r_t(h_t, a_t)\ \ \text{s.t.}\ \ \Phi(h_{t+1}) \ge \Phi_{\min}$, where $r_t$ is task performance on the current batch and $h_{t+1}$ is the harness after the edit. Its relaxation $r_t + \gamma\,\Phi(h_{t+1})$ is the per-step reward of a Markov decision process over harness states, whose stream objective is $\mathbb{E}\big[\sum_t \gamma^t r_t\big]$. The horizon is set by $\gamma$; nothing slides. (open: $\gamma$ and the retention constraint at commit, hard versus soft.)
- Plasticity: $\Phi(h)$ is not observed at edit time; it is estimated by a state-level critic $\hat{\Phi}(h)$ that reads the harness files and outputs a scalar, trained by regression on realized returns $G_t = \sum_k \gamma^k r_{t+k}$. The critic can be pretrained on released harness-evolution logs. (open: critic architecture and input representation.)
- Training recipe (open): a group of candidate edits sampled from $h_t$, each applied and scored on the next batch, reward $r_t + \gamma\,\hat{\Phi}(h_{t+1})$, group-normalized advantage with a clipped policy objective, one candidate committed through a deterministic retention gate. Alternative training methods are under consideration and this recipe will be revisited.
- Out of scope: token or dollar cost as an objective term; self-editing of evaluators.

Positioning, one clause each: Harness Continual Learning (2608.19013) uses plasticity and stability as labels for a fixed retention budget, never as an estimated quantity, and freezes weights; SIA (2605.27276) routes between two layers with an untrained selector and scores each step on its own batch; Adaptive Auto-Harness (2606.01770) runs real task streams and fights bloat with prompt-level audits and branch isolation; Harness-R1 (2608.02276) trains the proposer with reinforcement learning at one layer on same-batch reward ($\gamma = 0$); AgentStream (2608.00155) evaluates existing methods on streams and proposes nothing.

## Terms

The vocabulary allowlist. A term or abbreviation may be used in project documents only if listed here; only the researcher approves additions; agents never coin terms.

| Term | Meaning |
|---|---|
| harness | Everything around the model that shapes its behavior: prompts, tool descriptions and implementations, middleware, skills, sub-agent configuration, long-term memory; plus the model weights when they are editable. |
| layer | One of the harness components above, ordered by edit depth (cost to make, persistence, cost to undo). Depth axis only; not the episode-time axis of Life-Harness. |
| task stream | Tasks arriving one after another, mixed in type, with a distribution that shifts over time. |
| score-before-update | Scoring protocol: each task is scored with the harness as it stood before that task arrived; the harness updates afterward. |
| harness plasticity $\Phi(h)$ | The future stream performance achievable from harness $h$; its decrease under greedy patching is loss of plasticity. |
| critic $\hat{\Phi}(h)$ | A model that estimates $\Phi(h)$ from the harness files, trained on realized returns. |
| horizon $\gamma$ | Discount on future batch performance; $\gamma = 0$ is the greedy, current-batch-only objective of existing systems. |
| harness agent | The trained model that proposes harness edits and balances $r_t$ against $\Phi(h_{t+1})$. |
| anchor set | Previously passed tasks, grown from the stream, re-checked at commit time to measure retention. |
| move | An edit that relocates content from one layer to another. |
| promotion | A move that replaces several narrow entries with one general entry at a higher layer. |
