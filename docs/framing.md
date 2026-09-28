# Research framing

The research identity: what this project claims, in what setting, and the words it uses. Edited rarely; every amendment is edited in immediately. Items marked (open) are hypotheses or undecided details to be revisited.

## Problem

A harness $h$ around a frozen language model is decomposed into layers ordered by depth: system prompt, tool description, tool implementation, middleware, skill, sub-agent configuration, long-term memory, model weights. It is edited after each chunk of a heterogeneous, non-stationary task stream: the model solves a chunk with $h_t$, a harness agent reads $h_t$ with the chunk's trajectories and pass rate, and outputs one edit (or none) that yields $h_{t+1}$. The stream has no end. The objective is the mean task pass rate over the stream, each task graded with the harness that existed when it arrived; labels may arrive late, as on a prediction market where the answer is known only when the event happens. (open: stream design; chunk size and whether chunks are fixed.)

Unregulated patching accumulates redundant and inconsistent content in $h$. The accumulated harness then learns a new task type more slowly than a harness built from scratch; this deficit is intransigence, $I$. Existing self-improving harnesses score each edit on the batch it was made from, which cannot detect the deficit; their stream pass rate peaks and then declines (AHE at round 8 of 10, Adaptive Auto-Harness at cycle 22 of 51, HarnessX from 73.8% to 49.5% on GAIA), and none measures why. Retention and generalization are the two other ways an edit lowers later pass rate; both are measured by prior work.

## Hypothesis

A harness agent whose training reward includes an intransigence term sustains stream pass rate where an agent trained on pass rate alone peaks and declines. Harness plasticity $\Phi(h)$, the future stream performance achievable from $h$, is the quantity the term protects; $I$ is its measured deficit.

## Measurement

Intransigence at stream position $k$: let $D_k$ be the next chunk, drawn from a task type new to the stream, and $Q_k$ a held-out set of the same type that never enters the stream. The current harness $h_k$ and a fresh copy of $h_0$ each process $D_k$ under the same editor and are then scored on $Q_k$; each score is corrected by the harness's score on $Q_k$ before $D_k$. $I_k$ is the corrected score of the fresh copy minus that of $h_k$. Reported as a curve over $k$, alongside forgetting and forward transfer following GEM (1706.08840). One measurement costs $|D_k| + 3|Q_k|$ task runs. (open: $|Q_k|$, seeds, and measurement frequency.)

## Estimator

One measurement of $I_k$ is too expensive to score candidate edits during training. An intransigence estimator $\hat{I}(h)$ takes the harness files as input and outputs a predicted intransigence; it is trained on protocol measurements from the baseline streams and used in the training reward. (open: a learned regressor or a formula over harness statistics such as entry count, contradictions and never-retrieved entries; its input representation.)

## Method

The harness agent is a language model that receives $h_t$, the chunk's trajectories and pass rate, and outputs one structured edit (add, remove, rewrite, merge, or move content across layers) or no edit. It is trained by policy gradient with reward $p(h_{t+1}) - \lambda\,\hat{I}(h_{t+1})$, where $p$ is the pass rate after the edit on the chunk's own tasks. $\lambda = 0$ is Harness-R1's objective with persistent patches. The estimator, any critic and the benchmark are training-time only; the deployed agent receives none of them. (open: identical or held-out same-type tasks for $p$; the advantage baseline, a learned critic, a group mean, or the pre-edit score; the exact edit format; a retention gate at commit, hard versus soft.)

## Claim

The agent trained with $\lambda > 0$ sustains stream pass rate where the $\lambda = 0$ agent and untrained self-improving harnesses peak and decline, and its $I_k$ curve stays flat where theirs rise.

Falsifier: $I_k$ does not rise along the stream under unregulated patching; or the $\lambda = 0$ agent, or a per-edit-gated system (Harness Continual Learning with a swept retention budget), matches the $\lambda > 0$ agent in mean stream pass rate at equal harness size.

Out of scope: token or dollar cost as an objective term; self-editing of evaluators.

Positioning, one clause each: Harness Continual Learning (2608.19013) uses plasticity and stability as labels for a fixed retention budget, never as a measured or estimated quantity, and freezes weights; SIA (2605.27276) routes between two layers with an untrained selector and scores each step on its own batch; Adaptive Auto-Harness (2606.01770) runs real task streams and fights bloat with prompt-level audits and branch isolation; Harness-R1 (2608.02276) trains the proposer with reinforcement learning at one layer on same-batch reward, discarding each patch after its batch; AgentStream (2608.00155) evaluates existing methods on streams and proposes nothing.

## Terms

The vocabulary allowlist. A term or abbreviation may be used in project documents only if listed here; only the researcher approves additions; agents never coin terms.

| Term | Meaning |
|---|---|
| harness | Everything around the model that shapes its behavior: prompts, tool descriptions and implementations, middleware, skills, sub-agent configuration, long-term memory; plus the model weights when they are editable. |
| layer | One of the harness components above, ordered by edit depth (cost to make, persistence, cost to undo). Depth axis only; not the episode-time axis of Life-Harness. |
| task stream | Tasks arriving one after another, mixed in type, with a distribution that shifts over time. |
| harness plasticity $\Phi(h)$ | The future stream performance achievable from harness $h$; intransigence is its measured deficit. |
| intransigence estimator $\hat{I}(h)$ | A function of the harness files that predicts intransigence, trained on protocol measurements; used only in the training reward. |
| chunk | A fixed number of consecutive stream tasks solved with one harness; the harness agent acts once per chunk. |
| harness agent | The trained language model that reads the harness and the chunk's outcomes and outputs one edit; its reward is pass rate minus $\lambda$ times estimated intransigence. |
| anchor set | Previously passed tasks, grown from the stream, re-checked at commit time to measure retention. |
| move | An edit that relocates content from one layer to another. |
| promotion | A move that replaces several narrow entries with one general entry at a higher layer. |
| intransigence | How much the edits accumulated so far hurt a harness's ability to learn a new task, compared with a harness that starts from scratch; measured by the protocol under Measurement as $I_k$. Term from Riemannian Walk (1801.10112). |
