# ALMA: Learning to Continually Learn via Meta-learning Agentic Memory Designs

- **Original title**: Learning to Continually Learn via Meta-learning Agentic Memory Designs
- **Authors**: Yiming Xiong, Shengran Hu, Jeff Clune
- **Affiliations**: University of British Columbia (all three authors); Vector Institute (Hu, Clune); Canada CIFAR AI Chair (Clune)
- **Venue / year**: arXiv preprint, 2026 (arXiv:2602.07755v1 [cs.AI], submitted 8 Feb 2026, dated "Preprint. February 10, 2026", 71 pages including appendices A–C). The authors' project page and the code README report two ICLR 2026 workshop orals with paper awards; that claim is not peer-review evidence we could read, see section 7.
- **Source URL**: https://arxiv.org/abs/2602.07755
- **Project page**: https://yimingxiong.me/alma
- **OpenReview**: https://openreview.net/forum?id=sOq52KnJmR (the page could not be read; see section 7 for exactly what it shows)
- **Code**: https://github.com/zksha/alma (Apache 2.0). Discovered designs and evaluation logs are distributed separately through a Google Drive folder linked from the README. See `repo_analysis.md`.

Classification: **method paper**. The contribution is a built outer loop — a proposer, an archive with a stated sampling rule, a sandboxed evaluation protocol, and a code-level interface for the thing being searched over — that another group could run on a new domain; the four benchmarks exist to show the system works, not to characterize a phenomenon.

Names used below, all the paper's own. ALMA (Automated meta-Learning of Memory designs for Agentic systems) is the outer loop. The **Meta Agent** is the proposer that writes memory designs. A **memory design** is a Python program implementing the memory interface. The **agentic system** is the inner agent being helped by that memory; it is fixed throughout and is never edited. In our vocabulary the memory design is one layer of a harness, and ALMA is a search over that one layer.

---

## 1. Method Motivation

### a) Why was this method proposed?

Foundation models are stateless at inference, so an agent built on one re-solves every task from scratch (Sec. 1). The field's standard fix is a memory module that stores and re-reads past experience, and the paper's observation is that every such module in use is hand-written: "memory design, the architectural specification that determines how memories are represented, stored, retrieved, and updated, is still predominantly handcrafted by humans" (Sec. 1). Because different domains reward different things to remember — user facts for conversational agents, abstract strategies for strategic games (Sec. 1) — a human has to re-tailor the design per domain, which the authors call "difficult and labor-intensive".

The framing is explicitly the AI-generating-algorithms tradition of Clune (2020): hand-crafted components get replaced by learned ones, as hand-designed vision features were replaced by learned representations (Sec. 1, Sec. 2). ALMA is placed in the second pillar of that program, meta-learning the learning algorithm, where the learning algorithm here *is* the memory design.

### b) Pain points of existing methods

Three concrete shortcomings are named.

1. **Hand-crafted and fixed memory designs** (ReasoningBank, Dynamic Cheatsheet, G-Memory, trajectory retrieval): all of what to store, how to update, and how to retrieve is manually specified, so a design cannot adapt to a new domain's structure (Sec. 4.3). The paper's own Table 1 supplies the evidence that this hurts: on TextWorld with GPT-5-nano, three of the four hand-designed baselines score *below* the no-memory baseline (Trajectory Retrieval 2.7 versus 5.4; Dynamic Cheatsheet 4.3; G-Memory 2.1), that is, a memory design tuned elsewhere actively damages performance here.
2. **Agentic-system search that evaluates one-shot performance only.** The paper's positioning against ADAS and AgentSquare is sharp: "while existing works evaluate only the one-shot performance of agentic systems, ALMA explicitly optimizes for a memory design's capability to facilitate continual learning from past experience" (Sec. 2). The fitness signal is meant to reward accumulating and reusing experience, not solving one task well.
3. **Greedy, initialization-heavy search over memory designs.** A concurrent work is criticized for relying on "initialization with many existing handcrafted memory designs and greedy selection of top-performing designs, which limits open-ended exploration" (Sec. 2). ALMA starts from a single design written from scratch and keeps a non-zero sampling probability on every archive member.

### c) Core hypothesis / intuition

If memory designs are written as executable code and searched open-endedly — sampling stepping stones from an archive by a score that rewards high performance and penalizes repeated visits, rather than always expanding the current best — a proposer model can discover per-domain memory designs that let a fixed, weak inner agent accumulate and reuse experience better than any hand-written design.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

**What is being searched over (Sec. 3.1, Appendix A.1).** The search space is arbitrary Python subject to one interface. A memory design exposes two methods to the inner agent: `general_update()`, called after interactions to extract experience into memory, and `general_retrieve()`, called when a new task arrives to fetch relevant experience into the prompt. Internally a design may compose any number of sub-modules, each with its own `update()`, `retrieve()`, and optional database, and the output of one sub-module may feed the next, "enabling hierarchical and modular memory designs" (Sec. 3.1). The abstraction is stated to be reverse-engineered from existing hand-written designs: ReasoningBank is a vector-database sub-module plus an experience-extraction sub-module, G-Memory is a graph sub-module plus a vector sub-module (Sec. 3.1).

**How a candidate is scored (Sec. 3.2, Appendix A.2, Algorithm 1).** Formally a memory design is a triplet $\mathcal{M} = (U, D, R)$ — update process, storage, retrieval process. A dataset $G$ is split evenly into $G_{\text{collection}}$ and $G_{\text{deployment}}$.

1. **Memory Collection Phase.** Starting from $D_0 = \emptyset$, the inner agent runs every task in $G_{\text{collection}}$ **with no memory access at all**, producing trajectories $\tau^{(j)}_{\text{collection}}$ and task-level scores $f^{(j)}_{\text{collection}} = F(\tau^{(j)}_{\text{collection}})$. Memory is then updated one trajectory at a time, $D_j = U\big(\tau^{(j)}_{\text{collection}}, f^{(j)}_{\text{collection}}, D_{j-1}\big)$, ending at $D_N$. Because collection is memory-free, the trajectories are identical in distribution across all candidate designs, which the authors use to argue the cost metric is unbiased (Appendix A.3).
2. **Deployment Phase.** For each task $g^{(i)}_{\text{deployment}}$, retrieve $e^{(i)} = R(D_N, s^{(i)}_1)$ from the *frozen* memory, then act with the retrieved context in the prompt, $a_t \sim \pi(\cdot \mid s_{\le t}, a_{<t}, e^{(i)})$, and score $f^{(i)}_{\text{deployment}} = F(\tau^{(i)}_{\text{deployment}})$.
3. **Fitness.** $f_{\mathcal{M}} = \frac{1}{K}\sum_{i=1}^{K} f^{(i)}_{\text{deployment}}$, the plain mean over deployment tasks (Algorithm 1 line 30). The whole Deployment Phase is repeated three times and the reported number is the mean of the three pass averages, with the standard error taken across those three passes (Sec. 4.1).

The Deployment Phase has two modes. **Static**: memory stays frozen, every deployment task sees the same $D_N$. **Dynamic**: after each deployment task the memory is updated with that task's trajectory, $D_{N+i}$, and the updated memory is used for task $i+1$ (Appendix A.2). The paper is explicit that **learning uses static mode only**, "to reduce variance during evaluations" (Sec. 3.2); dynamic mode appears once, in a single test on ALFWorld (Sec. 4.4, Fig. 5).

**The outer loop (Sec. 3.3, Algorithm 1).** Input: an empty archive, a step count $L$, and the two task halves.

1. Initialize the archive with one memory design written from scratch by the Meta Agent from the abstract classes (Appendix B.3); evaluate it.
2. Each step, sample up to five designs from the archive without replacement by the score below.
3. In parallel for each sampled design: a **Planner** reads the design's source code, its stratified-sampled interaction logs, and its success rate, and proposes a change; an **Implementor** writes the new code; a **Trial Run** executes it; on error a **Debugger** revises, up to three attempts.
4. Evaluate the new design by the collection-then-deployment protocol above; insert $(\tilde{\mathcal{M}}, f_{\tilde{\mathcal{M}}}, S_{\tilde{\mathcal{M}}}, t_{\tilde{\mathcal{M}}})$ into the archive.
5. After $L$ steps, return $\mathcal{M}^* = \arg\max_{\mathcal{M} \in A_L} f_{\mathcal{M}}$ (Algorithm 1 line 35).

Scale: 11 learning steps, up to five designs per step, 43 memory designs discovered in total (Sec. 4.1, Appendix B.3).

### b) Architecture components

**The Meta Agent.** GPT-5 for the search (GPT-5 at medium effort for the initial from-scratch design, Appendix B.3). It is given "Provided Tools" with minimal usage examples for invoking reasoning or conversational models, embedding functions, and database operations, and it may call GPT-4o-mini, GPT-4.1, and text-embedding-3-small *from inside* the memory designs it writes (Sec. 4.1). The system prompt for planning is fixed across all benchmarks and across the whole run; only a per-benchmark description varies (Appendix B.3, which prints the four benchmark descriptions verbatim).

**What the proposer sees.** Per sampled design: (1) its source code, (2) stratified-sampled interaction logs from its evaluation, (3) its overall success rate. Plus (4) a one-shot example consisting of a previous design's code, the suggestion that was applied to it, and the resulting performance delta, and (5) the benchmark description (Appendix B.3). This is a far narrower channel than Meta-Harness's full filesystem: the archive stores a fixed-size proxy $S_{\mathcal{M}}$ of size $K_f$ drawn by stratified sampling over reward bins, "instead of storing the entire interaction logs" (Appendix A.4).

**The inner agent.** Fixed for the whole search: GPT-5-nano at low effort (Sec. 4.1). Memory-internal models are pinned to GPT-4o-mini and text-embedding-3-small at test time "to ensure a fair comparison" (Sec. 4.1).

**Sandboxing.** Every model-written design is validated and executed inside an isolated container, "with access confined to the sandbox to prevent any interference with the external system", and the authors additionally report human inspection of learned designs for things like prompt injection (Sec. 5).

### c) Formulas / algorithms

**Archive sampling (Appendix A.4)** is the only real equation set and the part most worth reusing. For each design, normalize performance against the no-memory baseline $f_0$ measured on the same deployment set,

$$\hat f_{\mathcal{M}_i} = \sigma\big(f_{\mathcal{M}_i} - f_0\big) = \frac{1}{1 + \exp\!\big(-\lambda (f_{\mathcal{M}_i} - f_0)\big)},$$

subtract a visit-count penalty to get a sampling score,

$$J_i = \hat f_{\mathcal{M}_i} - \alpha \log\big(1 + t_{\mathcal{M}_i}\big),$$

and turn scores into probabilities by a temperature softmax,

$$p_i = \frac{\exp\big(J_i/T - \max_j J_j/T\big)}{\sum_{j=1}^{|A_l|} \exp\big(J_j/T - \max_j J_j/T\big)} .$$

Settings: $\alpha = 0.5$, $T = 0.5$, five designs per step, sampled without replacement (Appendix B.3). The design intent is stated plainly: prioritize high-performing but under-sampled designs, and keep every design reachable — "designs with moderate performances retain a non-zero sampling probability to maintain diversity" (Appendix A.4).

Two things are *absent* from the formulas and both matter for us. There is **no acceptance test**: a newly evaluated design enters the archive regardless of whether it beat its parent, and nothing is ever reverted or removed. And there is **no term that references anything beyond the current evaluation** — $f_{\mathcal{M}}$ is a mean over one fixed deployment set, computed from an empty memory every time.

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Against **hand-designed memory systems** (Trajectory Retrieval, ReasoningBank, Dynamic Cheatsheet, G-Memory): the design is discovered rather than specified, and it is specialized per domain. Figure 3 is the evidence for specialization: the ALFWorld design stores an affordance graph, spatial experience, trajectory skeletons and failure patterns, while the MiniHack design stores trajectory schemas, spatial analysis, risk-and-interaction experience, a strategy library, and reflex rules. The paper's reading is that object-manipulation domains get fine-grained factual memory and reasoning-heavy domains get abstract strategy memory (Sec. 4.4).

Against **agentic-system search** (ADAS, AgentSquare, DGM): the search space is narrowed to the memory layer alone, and the fitness is defined so that a design is rewarded for turning collected experience into later task success rather than for one-shot quality (Sec. 2). The archive machinery — sampling proportional to performance and inversely to visit count — is taken from DGM and Go-Explore (Sec. 3.3, Appendix A.4).

Against **Meta-Harness** (`refs/related-work/Meta-Harness/paper_analysis.md`), the closest existing entry in our set and the fairest comparison, since both search over code with an archive-conditioned proposer: three differences and one shared blind spot.
- *Search space*: ALMA searches one layer (memory) behind a two-method interface; Meta-Harness searches the whole harness program including prompts, control flow, and tool handling.
- *Feedback channel*: ALMA hands the proposer a stratified sample of logs plus a scalar, roughly the scale Meta-Harness explicitly argues is too small (its Table 1 puts prior optimizers at 0.002–0.026 MTok per step against its own 10.0 MTok); Meta-Harness hands the proposer a filesystem.
- *Search structure*: ALMA has an explicit parent-selection rule with an exploration bonus; Meta-Harness deliberately has none.
- *Shared blind spot*: both score a candidate on one fixed set that never moves, from a clean start, and neither ever re-tests a task it has already passed.

### b) Key innovations and significance

1. **Fitness defined as collect-then-deploy rather than one-shot accuracy** — *notable*. It is the paper's genuine conceptual move: the score of a design is the inner agent's performance on tasks it has not seen, using memory built from tasks it has. That makes the objective about experience reuse rather than about single-task quality. It stops short of being a stream objective because collection and deployment are two disjoint blocks, not an interleaved sequence.
2. **Archive sampling with a visit-count penalty over memory designs** — *notable*, though transplanted from DGM and Go-Explore. The ablation against greedy search (Sec. 4.4, Table 2, Appendix C.2) is the only direct evidence it helps, and the evidence is mixed (see 4d).
3. **A minimal code interface for the memory layer** — *incremental but genuinely useful*. Two abstract methods plus an optional per-sub-module database is a small, copyable contract.
4. **Per-domain specialization as a demonstrated outcome** — *notable*. Figure 3 plus Table 1 together make the case that no single hand-written design wins everywhere and that search finds different structures per domain.

### c) Applicability

It works where the inner agent is weak enough that memory moves the number, where tasks in a domain share reusable structure, where an automatic scalar score exists per episode, and where one can afford roughly $|G_{\text{collection}}| + 3|G_{\text{deployment}}|$ episodes per candidate times 43 candidates per domain.

It struggles where the score is saturated or near-floor — the GPT-5-nano numbers in Table 1 top run from 2.1 to 19.0 percent success, so most designs are being ranked on a handful of solved tasks, and TextWorld fitness values during search sit between 0.01 and 0.05 (Fig. 11) — and where task distribution shifts mid-run, since the learning set is fixed in advance. The authors name the latter themselves: "the memory designs are learned using a pre-defined learning set, instead of dynamically learning memory designs when facing new tasks" (Sec. 5).

### d) Comparison table

| Method | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| ALMA | Fitness rewards experience reuse, not one-shot accuracy; per-domain specialization is demonstrated across four domains; explicit, simple, reusable archive-sampling rule; cheapest end-to-end memory cost of all compared designs ($0.09, 1,319 retrieved tokens, Fig. 6); sandboxed execution of model-written code | Fitness is one fixed set evaluated from an empty memory, so nothing after the current evaluation is scored; no acceptance gate and no revert; no earlier task is ever re-tested; the proposer sees only a stratified log sample plus a scalar; absolute success rates are very low with the search-time inner agent (overall 12.3 percent) | Interleave collection and deployment into one stream and score each task before its update; keep an anchor set of previously passed tasks; add a multi-objective term, which the authors suggest for cost (Sec. 4.4) |
| Hand-designed memory (ReasoningBank, Dynamic Cheatsheet, G-Memory, Trajectory Retrieval) | Interpretable, no search cost, portable across domains without re-running anything | Fixed structure; can be worse than no memory off-domain (TextWorld, Table 1: 2.1–4.3 against 5.4 no-memory) | Use as archive seeds, which ALMA deliberately does not do |
| Meta-Harness | Whole-harness search space; uncompressed history as feedback channel | Same fixed-set objective; no held-out split in its coding domain | Narrow the space to one layer when evaluation is expensive, as ALMA does |
| DGM-style greedy expansion (the ablation) | Simpler; concentrates budget on the current best | Ends 10 points lower when transferred to the stronger inner agent (77.1 against 87.1, Table 2) | Keep the visit penalty; it is one line of code |

---

## 4. Experimental Validation

### a) Experimental design

**Domains, all text-interfaced sequential decision making** (Sec. 4.2, Appendix B.1): ALFWorld (3,553 train tasks, 140 valid-seen, 134 valid-unseen; binary reward; step limit 100); TextWorld Treasure Hunter and Cooking through BALROG (52 tasks total; score in $[0,1]$ measuring partial completion; step limit 80); Baba Is AI through BALROG (52 tasks; binary; step limit 20); MiniHack navigation games through BALROG (114 tasks; binary; step limit 100).

**Splits** (Sec. 4.1, Appendix B.4): each dataset is divided into a learning set and a testing set, and *each of those* is split evenly into a collection half and a deployment half. TextWorld and Baba Is AI use the second halves of their datasets for testing; MiniHack uses the remaining 70 percent; ALFWorld uses valid-seen for in-distribution testing and valid-unseen for the distribution-shift test.

**Models**: Meta Agent GPT-5; inner agent GPT-5-nano at low effort during search and during the first test setting, GPT-5-mini at medium effort for the transfer setting; memory-internal calls pinned to GPT-4o-mini and text-embedding-3-small (Sec. 4.1).

**Baselines** (Sec. 4.3): no memory, Trajectory Retrieval, ReasoningBank, Dynamic Cheatsheet (cumulative), G-Memory.

**Protocol**: the design with the highest success rate observed *during learning* is taken as the learned design and evaluated on the testing set in static mode; every number is a mean over three Deployment Phase repeats with the standard error over those three (Sec. 4.1, Sec. 4.4).

### b) Key results

1. **Table 1 top (GPT-5-nano, same model as search).** Overall averages: no memory 6.1, Dynamic Cheatsheet 7.2, ReasoningBank 7.5, G-Memory 7.7, Trajectory Retrieval 8.6, ALMA 12.3 — a gain of 6.2 points over no memory and 3.7 over the best hand-designed baseline. Per domain: ALFWorld 12.4 ± 0.5, TextWorld 6.2 ± 1.7, Baba Is AI 19.0 ± 2.4, MiniHack 11.7 ± 2.2.
2. **Table 1 bottom (GPT-5-mini, transfer to a stronger inner agent).** Overall: no memory 41.1, best hand-designed 48.6 (Trajectory Retrieval), ALMA 53.9 — a gain of 12.8 over no memory. ALFWorld 87.1 ± 1.4 against 67.6 no memory. The authors note the gain grows with model strength (12.8 against 6.2, a delta of 6.6 that exceeds every baseline's delta) and read this as the designs supporting stronger agents better (Sec. 4.4).
3. **Scaling with collected experience (Fig. 4).** On ALFWorld valid-seen with GPT-5-mini, sweeping the number of collection tasks from 0 to 70, the learned design "achieves higher performance faster with limited data and scales better" than the baselines. The authors also note honestly that more trajectories is not monotonically better, "since trajectories may vary in quality" (Sec. 4.4).
4. **Distribution shift, dynamic mode (Fig. 5).** Memory collected from 70 valid-seen tasks, then deployed on valid-unseen with updates after each task: the learned design reaches 84.1 percent, above all hand-designed baselines. This is the only experiment in the paper where memory changes while it is being scored.
5. **Cost (Fig. 6).** Averaged across benchmarks, the learned designs sit at 53.9 percent success for $0.09 end-to-end memory cost and 1,319 retrieved tokens per task, against Trajectory Retrieval at 9,149 tokens, G-Memory at 6,095, Dynamic Cheatsheet at 2,258, ReasoningBank at 240.

### c) Where does it shine?

The biggest margins are on ALFWorld: +9.5 over no memory at nano (12.4 against 2.9) and +19.5 at mini (87.1 against 67.6), which is also the domain with the largest learning set and the only one with a genuine unseen-object split. The transfer result is the most convincing single finding, because the memory design was searched entirely against GPT-5-nano and the 87.1 number comes from a model that was never in the loop.

The weakest cell is Baba Is AI at nano, where the paper's blanket claim that learned designs "outperform all state-of-the-art human-designed memory baselines" (Sec. 4.4) does not hold cell-wise: ALMA and Trajectory Retrieval are both at 19.0, differing only in standard error (± 2.4 against ± 6.3). The claim holds on the overall average, not on every benchmark.

### d) Limitations

Acknowledged by the authors (Sec. 5): the learning set is fixed in advance rather than arriving online, and they say why — "we do not pursue this approach due to computational budget constraints, as evaluating each explored memory design would require a large number of rollouts"; capability is bounded by the underlying models; memory and the agentic system are not learned jointly; learned components raise safety concerns, handled here by sandboxing plus human inspection.

Implicit, from our reading:
- **The greedy ablation is thinner than it is presented.** Table 2 gives greedy 11.9 ± 0.5 against ALMA 12.4 ± 0.5 at nano — a 0.5-point gap with standard errors of the same size, which is not a separation — and 77.1 ± 0.8 against 87.1 ± 1.4 at mini, which is. The abstract-level claim that open-ended exploration beats greedy rests on the transfer setting.
- **Single-seed search.** One search run per domain, 43 designs; no repeated searches, so the variance of the *search* (as opposed to the variance of one design's evaluation) is unmeasured.
- **Selection on the learning set.** The reported design is the argmax over 43 noisy fitness values on the learning deployment half, each a mean of three passes at success rates near 0.05–0.25. With a selection over 43 candidates at that noise level, some of the learning-set advantage is selection bias; the held-out testing set is what protects the headline numbers, and it is used correctly.
- **Low absolute performance at search time.** The inner agent used for the entire search solves between 2.9 and 9.5 percent of tasks without memory (Table 1 top), so the fitness signal that drives 43 design decisions is built on very few successes.

---

## 5. Reproduction and Application

### a) Open source

Yes: https://github.com/zksha/alma, Apache 2.0, analyzed in `repo_analysis.md`. The repository contains the outer loop, the archive, the four environment adapters, the Docker sandbox, the five baseline memory designs, and the exact training and testing commands. The discovered memory designs and the evaluation logs are **not** in the repository; the README points to a Google Drive folder for "the learned memory designs and logs required to give the results presented in our work", and the four design identifiers referenced in `testing.sh` are absent from the checked-in archive.

Reproduction steps, from the repo: build the two container images (one for ALFWorld, one for BALROG-hosted environments), set API keys, run `training.sh` (one command per domain, batched rollouts, GPT-5 proposer, GPT-5-nano inner agent), then `testing.sh` with the resulting design identifier.

### b) Implementation details worth attention

- Search uses **static mode only**; the dynamic path exists but is used once, in the ALFWorld distribution-shift test (Sec. 3.2, Sec. 4.4).
- Sampling constants: $\alpha = 0.5$, $T = 0.5$, five designs per step, without replacement, 11 steps (Appendix B.3).
- Debugging budget: three retries on a failed trial run, after which the candidate is dropped (Sec. 3.3, Algorithm 1 lines 10–18).
- The normalization baseline $f_0$ is measured once, by running the no-memory design on the same deployment half before the search starts (Appendix A.4).
- Evaluation noise control is three full Deployment Phase repeats; the standard error is across those three repeats, not across tasks (Sec. 4.1).
- Collection is always memory-free, which is what makes the cost metric comparable across designs (Appendix A.3) and also what makes the collection half useless as a learning-curve measurement.

### c) Transferability

The interface is domain-agnostic: any environment that can produce a recorder object with an initial state, a step list, and a scalar reward can be dropped in, and the four shipped adapters differ mainly in prompt construction and task listing. The learned designs themselves transfer across inner models within a domain (Table 1 bottom) but are not claimed to transfer across domains, and Figure 3 argues they should not be expected to.

---

## 6. Summary

### a) One-sentence core idea

A proposer model searches an archive of Python memory designs, scoring each by how well a fixed weak agent does on held-out tasks using memory built from other tasks.

### b) Quick-reference pipeline

1. Write one memory design from scratch as Python code behind two required methods, one that stores experience and one that fetches it.
2. Score a design: run the agent on the first half of the tasks with no memory at all, feed those trajectories through the design's update method one at a time, freeze the memory, then run the agent on the second half of the tasks with retrieval turned on and average the scores over three repeats.
3. Keep every scored design in an archive together with its score, its parent, and a small sample of its logs.
4. Each step, pick up to five designs by a score that rises with performance and falls with how often the design has been picked before, and have the proposer read each one's code, logs and score, then write a modified version; retry up to three times if it crashes.
5. After eleven steps and about forty-three designs, take the highest-scoring design and evaluate it once on a testing set that the search never touched.

### c) Relation to our project

**The outer-loop objective is the greedy case, on a set that never moves.** Fitness is $f_{\mathcal{M}} = \frac{1}{K}\sum_i f^{(i)}_{\text{deployment}}$ over the deployment half of one benchmark's learning set (Appendix A.2, Algorithm 1 line 30), averaged over three replicate passes. It carries no time index, no term for anything after this evaluation, and no dependence on which designs came before. In our vocabulary the discount on future batch performance (horizon gamma) is zero. Moreover, as with Meta-Harness, this is stricter than a moving batch: every candidate is evaluated from $D_0 = \emptyset$ on the *same* deployment half at every one of the eleven steps, so the harness never accumulates history and the objective never moves. Harness plasticity, the future stream performance achievable from a harness, is neither measured nor constrained.

**Meta-test sequences do differ from meta-train ones, and this is done properly.** The learning set and testing set are disjoint (Sec. 4.1), ALFWorld's testing split even uses unseen objects and rules (Appendix B.1), and the transfer test swaps the inner model entirely (Table 1 bottom). This is a real held-out protocol and it is better than Meta-Harness's coding domain, which has none. But it is a one-shot report after the search ends, never optimization signal, and it is a different *set*, not a later position in a stream: there is no arrival order, no anchor set, and no notion of a task being re-encountered.

**Score-before-update exists in exactly one experiment and is never plotted as a curve.** The default static mode freezes memory before scoring, so every deployment task sees identical memory and there is no inner learning curve to speak of — the protocol deliberately removes within-deployment learning "to reduce variance" (Sec. 3.2). The dynamic mode used for the ALFWorld distribution-shift test (Appendix A.2, Appendix B.4, Fig. 5) *is* score-before-update in our sense: retrieve from $D_{N+i-1}$, act, score, then update to $D_{N+i}$, task by task in order. But only the final average over the sequence is reported (84.1 percent); the paper never plots success against task position, so a rise-then-decline over the sequence would be invisible in the reported statistic.

**No earlier task is ever re-tested, so retention is not measured anywhere.** Collection tasks and deployment tasks are disjoint halves; a collection task, which is the only thing the memory has actually seen, is never scored again after it is collected. Nothing re-checks a previously passed task at any point in either loop, so there is no retention measurement and nothing that could reveal forgetting.

**No gate and no revert at the outer loop.** Every evaluated design is inserted into the archive with its score, including designs that are worse than their parents; the greedy-ablation archive tree (Fig. 14) visibly contains many nodes at 0.00. The archive keeps bad designs deliberately, as diversity. This is the opposite of a retention gate: it is a mechanism for *not* committing, since the current harness is never a single object that gets patched in place.

**The closest thing to plasticity evidence is the greedy ablation, and it is suggestive rather than decisive.** A greedy chain that always expands the current best explored the same 43-design budget and ended at 11.9 (nano) and 77.1 (mini) against 12.4 and 87.1 for archive sampling (Table 2). The nano gap is inside the standard errors; the mini gap of 10 points is not. Read our way: a chain of edits each selected for being the best-so-far ends up worse than a diversified search at equal budget — an edit sequence optimized greedily degrades what can be reached later. The unit degrading here is the search trajectory, not a harness carried across a task stream, so it is adjacent evidence, not our phenomenon.

### d) Transfer to our own work

- **A score-before-update stream protocol**: partially present, not usable as published. The dynamic path is exactly our scoring order, but it exists only on the ALFWorld out-of-distribution test, is reported as one aggregate, and the tasks arriving in it are never re-tested. The repository does implement it, which is the useful part (see `repo_analysis.md`).
- **Harness-evolution logs usable to pretrain a critic**: the best offer in this paper, with a caveat. The archive record is $(\mathcal{M}, f_{\mathcal{M}}, S_{\mathcal{M}}, t_{\mathcal{M}})$ plus, in the code, the parent identifier, the proposer's suggestion, and the realized one-step delta against the parent. That is exactly the shape of a critic-training tuple — a harness state as code, an edit described in words, and a realized return — for four domains and 43 designs each. The caveat is that the return is one step on a fixed set, so it trains an estimate of immediate edit value, not of future stream performance, and the logs ship through Google Drive rather than the repository.
- **A retention or anchor-set mechanism**: absent. Nothing here re-checks past tasks.
- **A gate or revert mechanism**: absent by design; the archive accepts everything.
- **Evidence of loss of plasticity**: only the greedy-versus-open-ended ablation described above, at the level of the search tree.
- **Direct reuse for us**: the sampling score $J_i = \sigma(f_i - f_0) - \alpha \log(1 + t_i)$ is a clean, one-line exploration rule for any archive of harness states we keep, and the normalization against a measured no-memory baseline is a good habit — it makes scores comparable across domains with wildly different difficulty. The two-method memory interface is a plausible contract for our own long-term-memory layer, and the sandboxed execution path is a working example of running model-written harness code safely.
- **Positioning sentence, if ALMA earns one in framing**: ALMA meta-learns the memory layer as code, scoring each candidate by a fixed collect-then-deploy split with the memory frozen during scoring (horizon gamma zero), and never re-tests a collected task.

---

## 7. Reviewer Reception (OpenReview)

An OpenReview forum exists at https://openreview.net/forum?id=sOq52KnJmR, but **no review content could be read, and none is reported here** (checked 2026-09-15). What the page shows: navigating to the forum redirects to `https://openreview.net/challenge?redirect=%2Fforum%3Fid%3DsOq52KnJmR`, a page titled "Verifying your browser | OpenReview" whose visible text is "Complete the check below to continue to OpenReview"; a scripted browser reached the same challenge page with no completable widget exposed. The REST endpoints `https://api2.openreview.net/notes?forum=sOq52KnJmR` and the version-1 equivalent both return `{"name":"ChallengeRequiredError", ..., "status":403}`. A search-engine cache of the forum page surfaced only abstract-level text with no ratings, no decision, and no review threads. Therefore: no scores, no criticisms, no rebuttal, and no decision are known from OpenReview.

Venue information from the authors themselves, recorded as author claims rather than as verified peer review: the project page banner reads "Best Paper @ MemAgents Workshop · Outstanding Paper @ RSI Workshop, ICLR 2026", and the repository README carries two badges, "ICLR 2026 Workshop: AI with Recursive Self-Improvement (Oral)" and "ICLR 2026 Workshop: MemAgents (Oral)". Workshop papers at ICLR generally receive lighter review than main-track papers, and in this case we could not read any of it.

Net takeaway: treat the paper on the strength of its own artifact. The code is public and the protocol is specified precisely enough to check, which is what our reading above relies on; the two workshop orals are an author-reported signal of positive reception with no accessible review record behind them.
