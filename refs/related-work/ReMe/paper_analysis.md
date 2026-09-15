# Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution

- **Original title**: Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution
- **Authors**: Zouying Cao (1,3, work done during an internship at Tongyi Lab), Jiaji Deng (2), Li Yu (2), Weikang Zhou (2), Zhaoyang Liu (2, corresponding), Bolin Ding (2), Hai Zhao (1,3, corresponding)
- **Affiliations**: 1 — AGI Institute, School of Computer Science, Shanghai Jiao Tong University; 2 — Tongyi Lab, Alibaba Group; 3 — Key Laboratory of Shanghai Education Commission for Intelligent Interaction and Cognitive Engineering, Shanghai Jiao Tong University.
- **Venue / year**: Findings of the Association for Computational Linguistics: ACL 2026 (acceptance announced July 2026); preprint arXiv:2512.10696, v1 December 2025, v2 dated 15 April 2026 (the version in `paper.pdf`, 20 pages including appendices A–E).
- **arXiv URL**: https://arxiv.org/abs/2512.10696
- **ACL Anthology URL**: https://aclanthology.org/2026.findings-acl.829/
- **OpenReview**: none. ACL 2026 ran through ACL Rolling Review, whose reviews are not public for this paper; see section 7.
- **Code**: https://github.com/agentscope-ai/ReMe (Apache 2.0). The paper's own experiments live at tag `v0.2.0.6`; `main` has since been rewritten into a different product. See `repo_analysis.md`.

**Classification: method paper.** The contribution is a built, released, three-phase memory pipeline (acquisition, reuse, refinement) that others can run; the benchmark results exist to validate it rather than to characterize a phenomenon.

The paper's own short name for its system is ReMe (from *Remember Me, Refine Me*), used throughout below. Two other short names appear because the paper uses them: BFCL-V3 (Berkeley Function Calling Leaderboard version 3) and AppWorld (a simulated world of nine applications behind 457 application programming interfaces).

---

## 1. Method Motivation

### a) Why was this method proposed?

The paper's premise is that "how-to" knowledge extracted from an agent's own past runs — which it calls procedural memory — is the substrate for improving an agent without retraining it, and that current systems handle that substrate badly. The opening complaint is a single phrase: existing frameworks "predominantly suffer from a 'passive accumulation' paradigm, treating memory as a static append-only archive" (Abstract). Section 1 sets out three criteria an ideal procedural memory system must meet, and the method is built one mechanism per criterion: high-quality extraction (distill generalized, reusable knowledge rather than raw observations), task-grounded utilization (adapt what is retrieved to the current task), and progressive optimization (reinforce effective entries and remove outdated ones "to prevent degradation over time").

### b) Pain points of existing methods

Three concrete failures are named (Sec. 1, Sec. 2). First, storing whole trajectories as experiences means "coarse-grained trajectory-level experiences may introduce irrelevant information that can prevent the agent from grasping the core logic" — the cited examples are Synapse and HiAgent, which store complete trajectories for retrieval. Second, "fetched experiences are applied without adaptation, leading to failures in slightly shifted scenarios." Third, and the one the paper treats as decisive: "lack of timely update strategies causes the experience pool to degrade into a mixture of valid insights and toxic noise," and the related-work section sharpens this into a gap statement — prior methods "neglect strategic experience removal mechanism, since harmful experiences inevitably exist even with human validation and initial helpful ones can also degrade over time." The degradation claim is attributed to Xiong et al., 2025, not demonstrated here.

### c) Core hypothesis / intuition

If experiences are distilled at the level of individual key steps rather than whole trajectories, indexed by a model-written description of *when to use them* rather than by the original task text, and continuously pruned by how often they have actually helped, then a small model plus this memory can match a larger model without memory.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

ReMe is three phases running in a cycle (Fig. 2): experience acquisition builds a pool from past runs, experience reuse injects entries into new tasks, experience refinement adds and removes entries as tasks complete.

**Unit of storage.** Section 3.2 defines the experience set $\mathcal{E}$, with each individual experience written

$$E = \langle \omega, e, \kappa, c, \tau \rangle$$

where $\omega$ states the scenario in which to use the experience, $e$ is the core experience content, $\kappa = \{\kappa_1, \ldots, \kappa_m\}$ is a keyword set for categorization, $c \in [0,1]$ is a confidence score, and $\tau$ enumerates the tools used. The extraction prompts (Tables 10–12) additionally emit the originating task query and a generalized rewrite of it, which exist so that section 4.3 can ablate four candidate retrieval keys.

**Phase 1 — Experience acquisition (Sec. 3.2, App. B.4.1).** An execution model, $\text{LLM}_{execute}$, runs the training tasks. For each task query $q$ it samples $N$ trajectories ($N = 8$, temperature 0.9) "to capture diverse execution paths and thereby increase the likelihood of obtaining valuable success/failure pairs for comparisons." Within each per-task group, trajectories are sorted by reward and only the lowest-scoring and highest-scoring are kept. A summarizer model, $\text{LLM}_{summ}$, then runs three complementary analyses over them:

1. **Success pattern recognition** — successful trajectories are those exceeding a score threshold (empirically 1.0, i.e. full success); the summarizer names the key step that produced the success.
2. **Failure analysis** — for failed trajectories, the summarizer determines "the earliest key step that leads to suboptimal outcomes" and writes a preventive lesson.
3. **Comparative insight generation** — when a reward gap exists between the selected pair, the summarizer articulates "which specific decision or action distinguishes higher-scoring from lower-scoring attempts."

Two content filters follow. A language-model judge (prompt in Table 13) scores each candidate for actionability, accuracy, relevance, clarity, and uniqueness, and the prompt instructs: "Mark as invalid if score is below 0.3." Surviving entries are deduplicated: each is embedded and compared by cosine similarity against existing entries, "and any candidate exceeding a predefined similarity threshold is discarded as redundant." Retained entries are indexed by the embedding of the usage-scenario field $\omega$ and written to a vector database.

**Phase 2 — Experience reuse (Sec. 3.3, App. B.4.2).** On a new task, the top-$K$ entries are retrieved by cosine similarity between the new query and the stored usage-scenario embeddings:

$$\mathcal{E}_r = \arg\text{top}_k\left[\text{sim}_{cos}(E_i, q_{new})\right] \quad \text{(eq. 2)}$$

$$\text{sim}_{cos}(E, q_{new}) = \frac{\phi(\omega) \cdot \phi(q_{new})}{\lVert\phi(\omega)\rVert \, \lVert\phi(q_{new})\rVert} \quad \text{(eq. 3)}$$

with $\phi(\cdot)$ the embedding model. Two optional language-model stages follow: a reranker (prompt in Table 14) re-orders the candidates against the current task's constraints and objectives, and a rewriter (prompt in Table 15) merges the retrieved entries into "a cohesive, task-specific guidance" instead of a list. Retrieval happens once, at the start of each task — the paper states this as a limitation.

**Phase 3 — Experience refinement (Sec. 3.4).** Two mechanisms, one adding and one removing.

*Selective addition.* The paper compares full addition (summarize every new trajectory regardless of outcome) against selective addition (distill only trajectories that led to success) and adopts the latter, with a stated reason: during initial pool construction many failed trajectories can be analyzed collectively, but "in real-time task execution, a single failed trajectory often provides insufficient context for accurate failure analysis, potentially leading to misguided experiences."

*Failure-aware reflection.* When a new task fails, $\text{LLM}_{summ}$ analyzes the attempt, extracts improvement points, and $\text{LLM}_{execute}$ "starts a new trial based on these lessons. When such trial succeeds, the corresponding lessons are incorporated into memory; otherwise, they are discarded without cluttering the experience pool." The number of self-reflections is capped at 3.

*Utility-based deletion.* Each entry carries two counters: $f(E)$, the number of times it has been retrieved, and $u(E)$, a historical utility that "increments by 1 each time its recall contributes to a successful task completion." An entry is removed when it is frequently retrieved yet fails to improve performance:

$$\phi_{remove}(E) = \begin{cases} \mathbb{1}\left[\frac{u(E)}{f(E)} \le \beta\right], & \text{if } f(E) \ge \alpha \\ 0, & \text{otherwise} \end{cases} \quad \text{(eq. 1)}$$

with $\alpha = 5$ (retrieval threshold) and $\beta = 0.5$ (utility threshold) in all experiments; the paper notes "we only consider an experience for removal after it has been retrieved at least $\alpha$ times" and attributes the threshold choice to prior work (Xiong et al., 2025).

**Two configurations.** ReMe (fixed) freezes the pool after acquisition; ReMe (dynamic) runs refinement during evaluation. Section 4.1: "The configuration difference between ReMe (fixed) and ReMe (dynamic) lies in whether the experience pool is dynamically updated during agent execution."

### b) Architecture components

There is no trained model and no new architecture. The moving parts are three roles for an off-the-shelf language model — $\text{LLM}_{execute}$ (runs the task), $\text{LLM}_{summ}$ (distills and reflects), $\text{LLM}_{rerank}$ (optional re-ordering, with the rewriter as a fourth call) — plus an embedding model and a vector database. In the main experiments $\text{LLM}_{summ} = \text{LLM}_{execute}$, so the system improves itself with no stronger teacher, and section 4.4 ablates what happens when the summarizer is upgraded.

### c) Formulas and algorithms

Only three equations exist and all are above: the deletion rule (eq. 1), the top-$K$ retrieval (eq. 2), and cosine similarity over usage-scenario embeddings (eq. 3). Everything else is prompt text; the six prompts are printed verbatim in Tables 10–15, which makes the method unusually legible.

The deletion rule is the piece worth dwelling on. It is the only quantity in the paper computed from tasks *after* the one that created the entry: $f$ and $u$ accumulate across every subsequent retrieval, and the ratio $u/f$ is an empirical estimate of the entry's hit rate on the tasks it has actually been used on. It fires only downward — it can delete, never rank or accept — and only after $\alpha = 5$ retrievals.

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Against trajectory-storing systems (Synapse, HiAgent), ReMe stores key-step insights instead of trajectories and shows the granularity difference is worth more than the memory itself (Table 2, below). Against agentic memory systems (A-Mem, which builds a memory-centric knowledge graph, and LangMem, LangChain's long-term memory module), the difference the paper emphasizes is removal: A-Mem and LangMem append and reorganize, ReMe additionally prunes on realized usage. Against reflection systems (Reflexion), ReMe's reflection is a persistence filter rather than an in-context scratchpad — the lesson from a failed attempt is kept only if the retried attempt succeeds.

### b) Key innovations and their significance

| Innovation | Significance | Why |
|---|---|---|
| Key-step (rather than trajectory) extraction, with success, failure, and comparative variants | Notable | The largest single ablation effect in the paper (Table 2: +4.17 Avg@4 against +2.67 for trajectory-level on Qwen3-8B; +4.23 against +1.00 on Qwen3-14B). |
| Indexing by a model-written "when to use" field | Incremental | Best of four keys on five of six model-benchmark cells (Table 9), but the margin over raw task query is 0.5 to 1.8 points. |
| Utility-based deletion keyed on later-task success | Notable | The only mechanism here whose signal comes from tasks after the entry was written; contributes +3.34 Pass@4 in Table 3, more than reflection and selective addition combined on that metric. |
| Failure-aware reflection with persist-only-on-retry-success | Incremental | +0.67 Avg@4, 0.00 Pass@4 in Table 3. |
| Reranking and rewriting of retrieved entries | Incremental | +1.83 Avg@4 / +6.01 Pass@4 in Table 4, but measured only in a reduced setting (thinking mode disabled). |

### c) Applicability

It excels where tasks repeat structure across a stream — tool-calling suites where the same application programming interfaces recur — because an entry must be retrieved five times before deletion can even be considered, so the utility signal needs recurrence to exist at all. It should struggle where tasks are one-shot or the pool is small: on 96 to 218 entries over 150 to 168 evaluation tasks, most entries never reach $\alpha = 5$ retrievals, and the paper never reports how many entries were actually deleted.

### d) Comparison table

| | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| ReMe | Only system here that removes entries on realized later-task usage; extraction granularity is the dominant gain and is cheap; all six prompts published; a released pool per model and benchmark | One retrieval per task, fixed at the start; deletion needs 5 retrievals before acting; credit assignment is co-occurrence, not counterfactual; two tuned constants ($\alpha$, $\beta$) imported from prior work | Counterfactual utility (score with and without the entry); retrieval refreshed mid-task; report deletion counts |
| A-Mem | Structured knowledge graph, autonomous organization | No removal; loses to No Memory on AppWorld for all three models (Table 1) | — |
| LangMem | Production library, prompt refinement built in | No removal; loses to No Memory on AppWorld for all three models, worst on Qwen3-8B (26.79 against 32.85 Pass@4) | — |
| No Memory | No cost, no failure mode | No transfer between tasks | — |

---

## 4. Experimental Validation

### a) Experimental design

Two tool-use benchmarks. **BFCL-V3**: 50 tasks from the base multi-turn category build the initial pool ("the default dataset does not provide training split"), the remaining 150 are the evaluation set; appendix D.2 adds a 750-task run over the complete multi-turn category. **AppWorld**: 90 training tasks for acquisition, 168 test-normal tasks for evaluation. Metrics are Avg@4 (average success across four independent trials) and Pass@4 (at least one of four succeeds), each averaged over three independent runs with standard deviations reported. Backbones are Qwen3-8B, 14B, and 32B instruct models; appendix D.1 adds six more (GPT-4.1, o4-mini, Qwen3-Max, Kimi-K2-Thinking, DeepSeek-V3.2, GLM-4.7). Baselines are No Memory, A-Mem, and LangMem, with the protocol equalized: "all methods perform experience retrieval only once at the beginning of each task. Additionally, the memory addition operation for these systems is triggered only upon the collection of successful trajectories." Maximum 30 iterations per task. Embedding model text-embedding-v4 at dimension 1024; $K = 5$.

### b) Key results

1. **Main table (Table 1).** ReMe (dynamic) is best on every model and both benchmarks. Averaged over the two benchmarks: Qwen3-8B 27.65 → 34.94 Avg@4 and 46.20 → 55.03 Pass@4; Qwen3-14B 35.62 → 44.66 and 54.65 → 63.71; Qwen3-32B 40.89 → 49.10 and 61.52 → 69.97. Both memory baselines lose to No Memory on AppWorld at every model size — A-Mem 14.97 → 12.95 Avg@4 on Qwen3-8B, LangMem 14.97 → 11.46 — so on that benchmark the comparison ReMe wins is against methods that actively hurt.

   A bookkeeping note: the abstract and section 4.2 both report "8.83% in Avg@4 and 7.29% in Pass@4" for Qwen3-8B, but Table 1's own average columns give +7.29 Avg@4 and +8.83 Pass@4. The two labels are swapped in the prose; the magnitudes are right.

2. **Memory substitutes for scale (Table 1).** Qwen3-8B with ReMe (dynamic) reaches 55.03 average Pass@4 against vanilla Qwen3-14B's 54.65; Qwen3-14B with ReMe reaches 44.66 Avg@4 and 63.71 Pass@4 against vanilla Qwen3-32B's 40.89 and 61.52. This is the paper's headline and it is a comparison of a memory-equipped smaller model against a memoryless larger one, not against the larger model with the same memory.

3. **Extraction granularity dominates (Table 2).** On BFCL-V3 under ReMe (fixed), trajectory-level extraction gains +2.67 Avg@4 / +0.45 Pass@4 on Qwen3-8B and +1.00 / +1.11 on Qwen3-14B, while key-step extraction gains +4.17 / +6.22 and +4.23 / +4.22. The single design choice of what to store is worth more than the rest of the pipeline on the Pass@4 metric.

4. **Deletion carries the refinement gain (Table 3, Qwen3-8B, BFCL-V3).** Full addition 40.83 / 62.00; selective addition 44.33 / 64.66; plus reflection 45.00 / 64.66; plus deletion 45.17 / 68.00. Reading the increments: switching to selective addition is +3.50 / +2.66, reflection adds +0.67 / +0.00, and deletion adds +0.17 / +3.34. Note that full addition (40.83 Avg@4) is *below* the No Memory baseline of 40.33 by only half a point — indiscriminate accumulation of experiences nearly erases the benefit of having a memory at all.

5. **Generality across backbones (Table 7, BFCL-V3 Avg@4).** ReMe (dynamic) gains on all six additional models: GPT-4.1 48.25 → 54.67, o4-mini 54.67 → 60.00, Qwen3-Max 59.00 → 64.00, Kimi-K2-Thinking 57.17 → 66.00, DeepSeek-V3.2 53.06 → 57.33, GLM-4.7 68.00 → 73.83. The gain does not shrink with backbone strength in any obvious way — the largest, +8.83, belongs to Kimi-K2-Thinking.

### c) Where it shines

On the harder BFCL-V3 sub-categories (Table 8, Qwen3-8B, Pass@4): Missing Parameters 31.00 → 38.50 (+7.50) and Base 59.55 → 65.77 (+6.22) beat Long Context 47.00 → 52.00 (+5.00) and Missing Functions 19.00 → 21.50 (+2.50). The error analysis is the sharpest evidence (Fig. 6): total failures drop 62 → 47, ReMe fixes 17 baseline failures and introduces 2 new ones, and the reduction is concentrated in reasoning errors (22 → 14) and action omissions (19 → 16), with function-calling errors unchanged at 13. So the memory changes what the agent decides to do, not whether it can format a call.

A second place it shines is stronger summarization: with $\text{LLM}_{execute}$ fixed at Qwen3-8B, raising $\text{LLM}_{summ}$ to 14B gives +1.83 Avg@4 and to 32B gives +3.33 (Table 5). The pool quality, not the executor, is the bottleneck.

### d) Limitations

The paper acknowledges three (Limitations section): retrieval is fixed at one call per task start, validation rests entirely on a language-model judge that "may overlook nuanced aspects of experience quality," and a larger summarizer would help. Unacknowledged ones that matter here:

- **No stream-time curve anywhere.** Every number is an aggregate over an evaluation set. Nothing plots performance against position in the task stream, so the paper's own motivating claim — that pools degrade over time — is never measured, only inherited from a citation.
- **No count of what deletion actually removed.** Given $\alpha = 5$ and pools of 96 to 218 entries over 150 to 168 tasks, the number of entries eligible for removal is plausibly small, and the +3.34 Pass@4 attributed to deletion (Table 3) is reported without saying how many entries were deleted.
- **The utility signal is co-occurrence, not attribution.** Every one of the five retrieved entries gets $u \mathrel{+}= 1$ when the task succeeds, whether or not the agent used it. With $K = 5$ this is a noisy credit assignment that a single strong entry can carry.
- **Nothing is ever re-tested.** No earlier task is re-run after the pool changes, so no number in the paper measures whether a memory addition broke something that previously worked.
- **The full-addition row is a weak floor.** Table 3's "full addition" arm is the paper's stand-in for passive accumulation, and it is one configuration on one model on one benchmark.

---

## 5. Reproduction and Application

### a) Open source?

Yes: https://github.com/agentscope-ai/ReMe, Apache 2.0. The paper's code is the `v0.2.0.6` tag (7 January 2026), which carries the three-phase pipeline as composable operations, the BFCL-V3 and AppWorld drivers, all six prompts as yaml files, and six released experience pools under `docs/library/paper_data/task/`. The `main` branch has since been rewritten into a general local-first memory product and no longer contains the benchmark drivers. Full component inventory and file-level evidence in `repo_analysis.md`.

Reproduction is four steps (documented in `docs/cookbook/bfcl/quickstart.md`): split the benchmark into train and evaluation halves, collect trajectories with memory off, start the ReMe service and build the pool from those trajectories, then re-run the evaluation half with memory on. Two external installations are required (the Gorilla BFCL evaluator, AppWorld's simulated environment) and no result files are shipped, so every number must be regenerated against paid model interfaces.

### b) Implementation details requiring attention

The published constants are $N = 8$ trajectories per training task at temperature 0.9, success threshold 1.0, $K = 5$, $\alpha = 5$, $\beta = 0.5$, self-reflections capped at 3, 30 agent iterations maximum, embedding text-embedding-v4 at dimension 1024. Three paper-to-code gaps are worth knowing before reproducing: the validation prompt says to reject below 0.3 while the shipped default rejects below 0.5; the paper names Elasticsearch as the vector database while the quickstart runs a local backend; and the shipped defaults disable both the reranker and the rewriter, which are exactly the two modules Table 4 credits with +6.01 Pass@4. Latency cost is modest — 21.42 to 23.96 seconds per AppWorld task, +2.54 seconds (Table 6) — but the acquisition phase costs $8 \times$ the training tasks in rollouts plus three summarizer calls and one judge call per candidate.

### c) Transferability

The extraction-prompt set and the two-counter bookkeeping transfer directly to any memory layer. The deletion rule transfers as a template but needs its credit assignment fixed: as written it rewards every co-retrieved entry equally.

### d) Transfer to our own work

**Where ReMe sits in our setting.** Memory is one of our harness layers, so ReMe is a self-improving harness that edits exactly one layer — long-term memory — and nothing else. Its edits are adds and deletes of pool entries; no prompt, tool description, tool implementation, middleware, skill, or sub-agent configuration is ever touched, and weights are frozen. It is the narrowest action space of any system in this index, and the narrowness is what lets it run a per-entry statistic that wider systems cannot.

**On score-before-update.** ReMe satisfies our protocol at task granularity without naming it: retrieval happens once at the start of a task and every pool change happens after that task is scored, so in the dynamic configuration each task is scored with the pool as it stood when the task arrived. This is the second instance in this index of the protocol (after AgentStream) and the first where it falls out of the system's own design rather than from an evaluation harness built to impose it. The exception is deliberate and is the failure-aware reflection loop: when a task fails, the lesson written from that failure is injected into a retry of *the same task*, and the lesson is kept only if that retry succeeds (Sec. 3.4). That inner loop is the inverse of score-before-update and the purest horizon $\gamma = 0$ objective — the edit is scored on precisely the task that produced it — matching Harness-R1's same-batch rerun reward, at a different layer.

**On the horizon.** ReMe is the first partial counterexample to our claim that existing systems score an edit "by the batch it was made on and nothing after." Its deletion rule does not: $u(E)/f(E)$ is accumulated over tasks strictly later than the one that wrote $E$, and an entry cannot even be considered for removal until it has been retrieved $\alpha = 5$ times on later tasks (eq. 1). That is a realized, undiscounted return estimate — the same object our critic estimates, computed empirically instead of predicted, at per-entry rather than harness-state granularity. Four qualifications keep it short of our objective, and they are the precise statement of the gap we are filling:

1. It is **removal-only**. No later-task evidence ever affects whether an entry is accepted; acceptance is decided entirely by the producing trajectory's own success plus a content-quality judge and a duplicate check. The forward-looking term appears only as a garbage collector.
2. It is **per-item, not per-state**. There is no quantity attached to the pool as a whole, so nothing can express that a pool of individually-useful entries has become collectively unusable — which is the crowding effect the paper's own Figure 5 exhibits as $K$ grows.
3. It is **retrospective and unpredicted**. Nothing estimates $u/f$ before the entry has been used; the entry sits in the pool doing damage for at least 5 retrievals first.
4. Its credit assignment is **co-occurrence**. All $K = 5$ retrieved entries are credited on success.

So the framing sentence survives for acceptance decisions and needs a qualifier for removal decisions: existing systems score an *edit at commit time* on the batch that produced it; ReMe additionally scores an entry's *continued presence* on later batches. Worth raising with the researcher as a candidate framing amendment rather than assumed.

**On retention.** There is no retention measurement and no mechanism that could produce one. No earlier task is ever re-run; there is no anchor set, no revert, no regression check; the only downward pressure on the pool is the statistical deletion rule. ReMe therefore cannot tell a pool entry that is merely unused from one that actively broke a task it used to pass — both look like low $u/f$ only if the entry happens to be retrieved on the tasks it breaks.

**On peak-then-decline.** The paper asserts the phenomenon in its motivation ("the experience pool [degrades] into a mixture of valid insights and toxic noise," Sec. 1) and cites it rather than measuring it. Two pieces of its own evidence bear on it, neither along the stream-time axis we care about. Figure 5 varies the number of retrieved entries $K$ from 0 to 10 on Qwen3-8B / BFCL-V3 and reports the peak at $K = 5$ (40.33 / 59.55 at $K = 0$, rising to 44.50 / 65.77 at $K = 5$, with no further gain out to $K = 10$), with the text concluding "beyond the saturation point, retrieving more may degrade performance, primarily due to the higher chance of incorporating noisy experiences." That is crowding within one context window, not accumulation over a stream, though it is the same failure mode viewed through a different variable. Table 3 is the stronger hint: full addition scores 40.83 Avg@4 against a No Memory baseline of 40.33, so accumulating every trajectory's distillate leaves essentially nothing of the +4.84 that selective addition plus refinement achieves. Against our claim, the paper's run lengths (150 and 168 evaluation tasks, pools of 96 to 218 entries) are short enough that decline may simply not have had room to appear — and dynamic beats fixed everywhere in Table 1, i.e. over these horizons, updating during the stream is still strictly better than freezing.

**What is reusable.** Three things, one of them not what the abstract advertises.

The **released pools** (`docs/library/paper_data/task/`, six files, 96 to 218 entries each, split into success, failure, and comparative categories) are a released memory-layer harness state, not a harness-evolution log: every entry carries `freq: 0` and `utility: 0`, so they are acquisition-phase snapshots taken before any reuse counters accumulated, with no per-task outcomes attached. They cannot pretrain a critic on realized returns. They are usable as seed memory content and as a corpus for studying what a distilled experience looks like — which is what the paper claims for `reme.library`.

The **two-counter schema plus its two service endpoints** — record retrieval and success on every retrieved entry, sweep periodically for low hit rate — is about sixty lines of code total and is the cheapest existing implementation of a forward-looking signal on harness content. We would change three things: credit counterfactually rather than by co-occurrence, let the signal inform acceptance and not only deletion, and aggregate to state level.

The **six prompts** (Tables 10–15, shipped as yaml) are a complete proposer prompt set for the memory layer, with stable wording across the comparison, and are a reasonable starting point for our own memory-layer proposer rather than writing one from scratch.

**What is missing for us.** No stream construction of any kind: the evaluation order is whatever order the benchmark file happens to be in, sharded round-robin across four concurrent workers that all write into one shared pool, and the train/evaluation split is a random shuffle. There is no non-stationarity, no domain shift, and no ordering control — so none of this paper's protocol can be borrowed as a task-stream design.

---

## 6. Summary

### a) One-sentence core idea

Distill per-step lessons from past runs, index them by when to use them, and delete the ones that keep getting retrieved without helping.

### b) Quick-reference pipeline

1. Run the agent eight times on each training task, keep the best and worst run of each, and have a model write short lessons from them — what worked, what went wrong, and what separated the good run from the bad one.
2. Score each lesson with a model judge, throw away the low scores and anything too similar to a lesson already stored, and file the rest by a one-line description of the situation in which it applies.
3. On a new task, look up the five lessons whose situation description is closest to the new request, optionally re-order them and merge them into one piece of guidance, and paste that in front of the task.
4. After the task, add lessons from it only if it succeeded; if it failed, retry once with the failure lesson in context and keep that lesson only if the retry then succeeded.
5. Count for every lesson how often it was retrieved and how often the task it was retrieved for succeeded; once a lesson has been retrieved five times, delete it if it succeeded less than half the time.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched the exact title, "ReMe", and the ACL Rolling Review venue on OpenReview and by web search on 2026-09-15; also queried the OpenReview note-search interface directly, which returns no matching record). The paper went to Findings of ACL 2026 through ACL Rolling Review, whose review threads are hosted on OpenReview but are not public for this submission, and the ACL Anthology entry carries no reviews. Nothing about the review process is therefore available, and none is invented here. The only external signal is venue: Findings rather than the main conference, which at ACL indicates scores above the acceptance bar but below the main-track cut, and which is consistent with the paper's own shape — a well-executed engineering combination on two benchmarks rather than a new measurement or a new claim.
