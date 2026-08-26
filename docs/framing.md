# Research framing

The research identity: what this project claims, in what setting, and the words it uses. Edited rarely — additive amendments batch after they survive; retractions are edited in immediately. Items marked (open) are hypotheses or undecided details to be revisited.

## Claim

Optimizing a harness for current-batch task success trades away three kinds of future performance: generalization to later tasks, retention of earlier tasks, and the ability to keep patching. All three are performance on the task stream that the current batch cannot measure.

Harness entropy is a label-free property of the harness, computable at the time of an edit, that predicts these losses. (open: the bound form `perf_future ≥ perf_batch − λ·H` and the mapping of entropy components to loss channels are hypotheses, to be tested by log mining and by a λ ablation.)

Claim: a proposer trained on task success minus harness entropy sustains stream performance where entropy-blind, per-edit-gated systems peak and then decline.

Falsifier: an entropy-blind gated system (Harness Continual Learning with a swept retention budget) matches the integral of stream performance at equal harness size.

## Setting

- Object: a harness around a language model, decomposed into layers ordered by depth — system prompt, tool description, tool implementation, middleware, skill, sub-agent configuration, long-term memory, model weights. An edit may add, remove, rewrite, merge, or move content across layers. (open: exact proposer action space.)
- Regime: a heterogeneous, non-stationary task stream. Each task is scored with the harness as it stood before the task arrived (score-before-update); labels may arrive late.
- Objective: constrained — maximize stream task success subject to a harness-entropy budget; Lagrangian form `U = perf − λ·H`. Retention on the anchor set is the commit-time constraint. (open: hard constraint versus soft term for retention.)
- Harness entropy: thermodynamic sense (disorder that grows unless work is spent), not Shannon entropy. Operational components: narrowness of rules, redundancy across entries and layers, contradiction between entries. (open: measurement.)
- Proposer: a trained model, cold-started by supervised fine-tuning on teacher edits, then trained by group relative policy optimization (GRPO) against U. (open: recipe details.)
- Out of scope: token or dollar cost as an objective term; self-editing of evaluators; failures of transmission between agents (the object of the "Entropy Principle" paper, arXiv 2606.08162, whose entropy is behavior decay of a running multi-agent system, not a property of the harness).

Positioning, one clause each: Harness Continual Learning (2608.19013) handles stability as a retention constraint with frozen weights and no entropy measure; SIA (2605.27276) routes between two layers with an untrained selector and no stability term; Adaptive Auto-Harness (2606.01770) runs real task streams and fights bloat with prompt-level audits and branch isolation, never an objective term; Harness-R1 (2608.02276) trains the proposer with GRPO at one layer on same-batch reward; AgentStream (2608.00155) evaluates existing methods on streams and proposes nothing.

## Terms

The vocabulary allowlist. A term or abbreviation may be used in project documents only if listed here; only the researcher approves additions; agents never coin terms.

| Term | Meaning |
|---|---|
| harness | Everything around the model that shapes its behavior: prompts, tool descriptions and implementations, middleware, skills, sub-agent configuration, long-term memory; plus the model weights when they are editable. |
| layer | One of the harness components above, ordered by edit depth (cost to make, persistence, cost to undo). Depth axis only; not the episode-time axis of Life-Harness. |
| task stream | Tasks arriving one after another, mixed in type, with a distribution that shifts over time. |
| score-before-update | Scoring protocol: each task is scored with the harness as it stood before that task arrived; the harness updates afterward. |
| harness entropy (H) | Disorder of a harness that grows as patches accumulate unless consolidation work is spent; components: narrowness, redundancy, contradiction. Not Shannon entropy. |
| anchor set | Previously passed tasks, grown from the stream, re-checked at commit time to measure retention. |
| proposer | The model that proposes harness edits. |
| move | An edit that relocates content from one layer to another. |
| promotion | A move that replaces several narrow entries with one general entry at a higher layer. |
| GRPO | Group relative policy optimization: reinforcement learning that scores each sample relative to the mean of a group of samples for the same input. |
