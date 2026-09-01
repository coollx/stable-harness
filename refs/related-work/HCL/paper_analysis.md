# Harness Continual Learning: Continual Adaptation Beyond Model Parameters

- **Original title**: Harness Continual Learning: Continual Adaptation Beyond Model Parameters
- **Authors**: Borui Kang, Jinrui Gu, Junhan Lv, Wenbin Li (corresponding), Lei Wang, Yang Gao
- **Affiliations**: State Key Laboratory for Novel Software Technology, Nanjing University, China (1); University of Wollongong, Australia (2)
- **Venue / year**: arXiv preprint, 2026 (arXiv:2608.19013v1 [cs.LG], submitted 19 Aug 2026, 21 pages including appendices A-C)
- **Source URL**: https://arxiv.org/abs/2608.19013
- **OpenReview**: no public page found (see section 7)
- **Code**: no public release (see section 5a)

Short name used below: Harness Continual Learning (HCL), the paper's own abbreviation.

---

## 1. Method Motivation

### a) Why was this method proposed?

Continual learning has been studied almost exclusively as a property of model parameters: a stream of tasks arrives, the parameters change, and the research question is how to acquire new behavior without destroying old behavior. The paper's driving observation is that an agent built around a frozen foundation model has a second, entirely separate mutable state — the harness of prompts, memories, tools, skills, and routing rules — and that this state also accumulates experience and also reshapes future behavior. The paper's framing sentence: "Prompts, memories, tool and skill specifications, and routing policies can persist and evolve across interactions even when the foundation model remains frozen. Agent adaptation is therefore no longer confined to model state" (Sec. 1). The consequence they want to make into a research object is that a harness edit made to fix a recent failure can silently break a task that used to succeed, with the model untouched. They name this **harness-level forgetting** and position it as "the classical stability-plasticity problem" relocated "from model state to harness state" (Sec. 1).

### b) Pain points of existing methods

Two distinct gaps, one per related-work subsection:

1. **Against harness engineering / agent self-optimization** (Sec. 2.1). Work that revises prompts, declarative programs, memories, tool-use policies, skills, and workflows from execution feedback (ReAct, Toolformer, MRKL, HuggingGPT, MemGPT, Reflexion, Voyager, DSPy, AFlow, and recent configuration-search and failure-diagnosis systems) is credited, then faulted: "Their main objective, however, is usually the quality of a component or the next configuration on a current task or target distribution. Repeated improvement alone does not provide a general retention criterion for the full harness state." Two sub-complaints are packed in here: the objective is the current batch (not a sequence), and the object is one component (not the whole coupled state).
2. **Against model-centric continual learning** (Sec. 2.2). After surveying the five standard families (representation-, architecture-, optimization-, regularization-, replay-based): "the state being learned remains model knowledge, representations, architectures, or parameters. Our work moves the continual learning object outside the model."

### c) Core hypothesis / intuition

If the whole mutable harness is treated as one jointly versioned continual-learning state, and if every proposed update must pass an explicit retention check on previously solved cases before it is deployed, then an agent around a frozen model can accumulate capability across a heterogeneous task stream while its loss of earlier behavior stays measurable and adjustable.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

The system has two interleaved loops: an **execution** loop that runs each interaction through the currently deployed harness, and a **guarded evolution** loop that turns post-execution feedback into a possible replacement for that harness.

**Setting and notation** (Sec. 3.1). A foundation model $F_\theta$ is fixed for the entire stream; $\theta$ never changes. $H_n$ is the harness deployed at interaction step $n$. Harness Continual Learning is defined as "the problem of sequentially updating the deployed harness to acquire new behavior while retaining behavior that was reliable before the update", where previously reliable behavior may be "a correct response, a valid tool call, or an action trajectory that satisfies an environment goal", re-checked "under the same input and execution conditions".

**Execution path, step by step.**

1. **Input**: a raw interaction $u_n$ — an instruction, an observation, or a multimodal input.
2. **Parse (Task Interface)**: $i_n = I_n(u_n) = (x_n, g_n, k_n)$ (eq. 5), where $x_n$ is the available input, $g_n$ the task objective, and $k_n$ the constraints ("output format, legal tool use, and environment restrictions"). $I_n$ is implemented as an LLM-based parser driven by prompts, task templates, and parsing/normalization rules. Its purpose is to map heterogeneous task data (text QA, images, environment observations) into one representation so a single stream can mix modalities.
3. **Assemble context (Adaptive Router)**: $z_n = R_n(i_n, M_n, C_n)$ (eq. 8). The Router retrieves relevant experience from Experience Memory, selects capabilities from the Capability Map, and orders them into an execution context. $z_n$ holds the structured task representation, the selected experience and capabilities, and the workflow to be used. The Router is itself an LLM call parameterized by routing prompts, selection criteria, and workflow templates.
4. **Execute**: the frozen model plus the external runtime (tools, environment) consume $z_n$ and emit the outcome $y_n$ — a response, an action, or a trajectory.
5. **Feedback**: post-execution feedback $f_n$ arrives from the environment or a verifier. The interaction record is collected as $e_n = (u_n, i_n, z_n, y_n, f_n)$ (eq. 1).

**Guarded evolution path, step by step.**

6. **Propose (Continual Optimizer)**: $\tilde{H}_{n+1} = O_{F_\theta}(H_n, e_n)$ (eq. 2). The Optimizer is a **prompt template** over the same frozen model $F_\theta$: it is handed the deployed harness, the interaction evidence, and an update rule, and asked to return a candidate harness. Nothing here is trained; the operator $O$ has no parameters.
7. **Gate (Continual Evaluator)**: three checks — current improvement, historical retention, validity — combined into a binary decision $G_n \in \{0,1\}$.
8. **Commit or discard**: $H_{n+1} = \tilde{H}_{n+1}$ if $G_n = 1$, else $H_{n+1} = H_n$ (eq. 3). "Therefore, a candidate affects later interactions only when it is committed."

The candidate is all-or-nothing at commit time: "HCL treats all proposed changes as one complete candidate harness. The candidate replaces $H_n$ only after it satisfies current improvement, historical retention, and validity requirements. Otherwise, none of its changes enters the deployed harness" (Sec. 3.2).

### b) Architecture components

The mutable harness state is a closed four-tuple (eq. 4):

$$H_n = (I_n, M_n, C_n, R_n)$$

Table 1 of the paper enumerates exactly which contents inside each component are editable. This enumeration is the method's real action space.

| Component | Function during execution | Contents updated by HCL |
|---|---|---|
| Task Interface $I_n$ | Transforms raw interactions into structured representations | Prompts, task templates, and parsing and normalization rules |
| Experience Memory $M_n$ | Provides concrete interactions and abstract guidance for reuse | Raw interaction records and LLM-generated Abstract Memory entries |
| Capability Map $C_n$ | Provides external operations and reusable inner skills | Inner skills extracted from Abstract Memory |
| Adaptive Router $R_n$ | Selects and organizes memory and capabilities | Routing prompts, selection criteria, and workflow templates |

**Task Interface $I_n$** (Sec. 3.2.1). Input-processing layer; produces $(x_n, g_n, k_n)$. Versioned with the harness "since interface updates may change how tasks are interpreted".

**Experience Memory $M_n = (M_n^{\mathrm{raw}}, M_n^{\mathrm{abs}})$** (eq. 6, Sec. 3.2.2). Raw Memory stores $u_n$, the resulting response or trajectory $y_n$, and the feedback $f_n$; it "retains a fixed number of interactions from each task in arrival order" to bound storage. Abstract Memory is produced by prompting an LLM to summarize Raw Memory into "scoped guidance, such as output conventions, reliable reasoning patterns, and common errors to avoid". The paper maps these to replay (raw) and representation transfer (abstract).

**Capability Map $C_n = (C_n^{\mathrm{outer}}, C_n^{\mathrm{inner}})$** (eq. 7, Sec. 3.2.3). Outer capabilities are supplied by the runtime — "APIs, retrieval services, perception models, calculators, and environment actions", each entry specifying "its function, expected inputs and outputs, invocation protocol, availability conditions, and known limitations". Inner capabilities are reusable skills distilled by an LLM from Abstract Memory, "with explicit inputs, outputs, execution steps, and applicable scopes". Only the inner half is updated: Table 1 lists the Capability Map's editable contents as exactly "Inner skills extracted from Abstract Memory", and Table 7 says "the Optimizer may add or revise internal skills". The outer entries — the tools themselves and their descriptions — are never revised.

**Adaptive Router $R_n$** (Sec. 3.2.4). Connects the other three to execution via eq. 8, using an LLM plus routing prompts, selection criteria, and workflow templates, all of which "can also be revised across interactions".

**Anchor set $A_n$** (Sec. 3.3.2, Table 7). Not part of $H_n$. It is Evaluator-only state: "Used only by the Evaluator; unavailable to execution and candidate generation. Updated at the end of each task and then fixed during candidate generation and evaluation for the next task." Each anchor is a raw input plus a success criterion, so it can be rerun under both the deployed and the candidate harness. Anchors are drawn "using a predefined ratio of previously successful and failed cases"; if one group is short, "the remaining slots are filled from the other group".

### c) Formulas and algorithms

**Interaction record (eq. 1)**: $e_n = (u_n, i_n, z_n, y_n, f_n)$ — everything the Optimizer is allowed to look at when diagnosing what went wrong. Note $i_n$, $z_n$, $y_n$ are transient; only $H_n$ persists (Appendix A.1).

**Candidate generation (eq. 2)**: $\tilde{H}_{n+1} = O_{F_\theta}(H_n, e_n)$. Plain language: ask the frozen model, in one templated prompt, to rewrite its own surrounding configuration in light of what just happened. The concrete edit space, quoted from Sec. 3.3.1: "It may modify prompts or parsing rules in the Task Interface, record or summarize experience in Memory, add or revise skills in the Capability Map, or adjust selection and workflow rules in the Adaptive Router."

**Candidate search strategy** (Sec. 3.3.1, no equation number). When several components need revision, they are visited "in a predefined order"; for each component the Optimizer generates "up to $K$ alternatives one at a time", each evaluated "by replacing only the selected component in the current candidate harness while keeping all other components fixed". "The highest-scoring admissible alternative is retained as the basis for revising the next component. If no alternative passes the gate, that component remains unchanged." The stated reason for this coordinate-wise sweep rather than joint search is cost: "To provide alternative update directions while limiting repeated LLM calls".

**Deployment rule (eq. 3)**: $H_{n+1} = \tilde{H}_{n+1}$ when $G_n = 1$, $H_{n+1} = H_n$ otherwise. Rejection is a no-op, not a rollback: the deployed harness simply never changed.

**Current improvement (eq. 9)**: with $V_n$ the validation cases for the current task and $P(H, V_n)$ the performance of harness $H$ on them,

$$\Delta_n = P\big(\tilde{H}_{n+1}, V_n\big) - P\big(H_n, V_n\big)$$

and the criterion is $\Delta_n \ge \delta_n$ for a predefined minimum improvement $\delta_n$. $P$ "may measure answer accuracy, tool-use success, or environment completion". Both harnesses are run "under the same model, decoding, tool, environment, and seed conditions".

**Historical retention (eqs. 10-11)**: for each anchor $a \in A_n$, a binary success indicator $q(H, a) \in \{0,1\}$ (eq. 10), equal to 1 when harness $H$ meets that anchor's success criterion. The historical loss is

$$D_n = \sum_{a \in A_n} \mathbb{1}\Big[q(H_n, a) = 1 \ \wedge\ q\big(\tilde{H}_{n+1}, a\big) = 0\Big]$$

(eq. 11) — a count of anchors solved by the deployed harness that fail under the candidate. The criterion is $D_n \le B_n$, with $B_n$ "the predefined tolerance for historical loss"; $B_n = 0$ "requires the candidate to preserve every anchor currently solved by $H_n$". Two properties worth noting: the count is one-sided (Appendix C: "improvement on another anchor does not offset it"), and it is thresholded per anchor, so a graded regression counts as a loss only if it crosses the criterion (their example: a RefCOCO intersection-over-union drop from 0.68 to 0.41 crosses the 0.5 threshold and contributes one loss).

**Validity (eq. 12)**: for each check $\ell$ in a set $L_n$, $v_{n,\ell}(\tilde{H}_{n+1}) \in \{0,1\}$. Checks "may cover artifact syntax, output-schema compliance, legal tool use, task constraints, and environment consistency".

**The commit gate (eq. 13)**, the paper's central object:

$$G^{(k)}_n = \mathbb{1}\Big[\big(\Delta^{(k)}_n \ge \delta_n\big) \ \wedge\ \big(D^{(k)}_n \le B_n\big) \ \wedge\ \big(\forall \ell,\ v_{n,\ell}(\tilde{H}^{(k)}_{n+1}) = 1\big)\Big]$$

where $k$ indexes candidates. This is "a hard admissibility gate". When several candidates pass, "the Continual Evaluator ranks them using a composite score that aggregates their current-performance, validity, and historical-retention scores"; the top one is committed, ties broken randomly; if none passes, $H_n$ stays. The composite score's weighting is not specified. [Information not available in provided text]

Two structural consequences of eq. 13 are worth stating plainly, because they are properties of the conjunction rather than of any single term. First, **current-task improvement is a necessary condition for every commit**: an edit that only consolidates, deduplicates, generalizes, or repairs — with $\Delta_n < \delta_n$ on the current validation set — cannot be committed, no matter how much it helps later. Second, **$B_n$ constrains only the anchors in $A_n$**, so retention is enforced on a sample, not on the population of earlier behavior; the paper itself draws this consequence (Sec. 4.4).

**Evaluation protocol (eq. 14)**: with $H^{(s)}$ the harness after learning task $D_s$, $R_{s,j} = \mathrm{Eval}(H^{(s)}, D_j^{\mathrm{test}})$ for $j \le s$ — evaluate the current harness on the current task and every earlier task. Validation cases and anchors belong to the Evaluator; final test sets are disjoint from both and used only for reporting.

**Reported metrics (eq. 15)**:

$$\mathrm{Avg}_T = \frac{1}{T}\sum_{j=1}^{T} R_{T,j}, \qquad \mathrm{Fgt}_T = \frac{1}{T-1}\sum_{j=1}^{T-1}\Big(\max_{r \in \{j,\dots,T\}} R_{r,j} - R_{T,j}\Big)$$

Final average performance across the stream, and average decline of earlier tasks from their best observed value. Forgetting is reported as "-" for Zero-shot and Static Harness because they make no sequential updates.

**Mapping to model-centric families (Sec. 3.4)**, stated by the authors as "conceptual rather than one-to-one": Raw Memory plays the role of replay; the Capability Map plays the role of representation learning (reusable abstractions); the Router plays the role of architecture-based module selection; and the Optimizer-Evaluator pair plays the role of optimization/regularization constraints on the update.

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

**Versus model-centric continual learning**: the learning object moves out of the model. There is no gradient anywhere in the method; the "update" is a text edit to persistent configuration, and the "regularizer" is a rerun of stored test cases. The frozen model is categorical, not incidental — asserted in the title, the abstract, the formalism ("The model parameters $\theta$ remain unchanged", Sec. 3.1), and every experiment ("the same foundation model is used for all comparisons and remains frozen throughout the continual-learning stream. Any adaptation therefore comes from harness updates rather than model training", Sec. 4). No fine-tuning, LoRA, distillation, or continual pre-training is proposed, run, or listed as future work, and the system never generates data intended for a later model update; what accumulates is memory consumed at execution time.

**Versus harness engineering / self-improving agent systems**: two moves. (i) The unit of update is the *whole* harness, versioned jointly and committed atomically, rather than one component optimized in isolation. (ii) Commitment is gated on rerunning earlier cases, so retention is a deployment condition rather than an after-the-fact evaluation.

**What is not different, and matters for reading the paper**: nothing in the system is learned. The Continual Optimizer "implements the update operator $O$ in Eq. (2) using a prompt template for the foundation model $F_\theta$" (Sec. 3.3.1) — an untrained prompt, not a trained proposer. The Evaluator's thresholds ($\delta_n$, $B_n$, the validity set $L_n$, the alternative count $K$, the anchor-set size and composition ratio) are hand-set constants, "chosen heuristically to balance current-task improvement with candidate reliability" and held "identical across all settings" (Sec. 4.4). The gate itself is a deterministic conjunction. So HCL is a search-and-filter procedure with fixed hyperparameters, not a training method.

**Two scope facts that the abstract's wording can obscure.** The abstract advertises a harness "of prompts, memories, tools, skills, and routing rules", and Sec. 1 says "tool and skill specifications ... can persist and evolve". The implemented method evolves *skills* ($C_n^{\mathrm{inner}}$), not *tools*: outer capabilities — APIs, retrieval services, perception models, calculators, environment actions, and their descriptions and invocation protocols — appear in the state but are never listed as updatable in Table 1, Table 7, or Table 9, and Sec. 4.3.1 states the closed set flatly: "HCL updates only the Task Interface, Experience Memory, Capability Map, and Adaptive Router." Likewise, the machinery around the harness is outside the mutable state: the Optimizer's own prompt template, the Evaluator, the anchor-selection procedure, the environment adapters and task validators mentioned in Sec. 2.1, and the runtime are never targets of an update. There is no self-modification of the optimizing or evaluating apparatus.

**On "stability" and "plasticity"**: both words are used throughout — in the abstract, twice in the introduction, in Sec. 3.3.2, in Sec. 4.1/4.2.1/4.3, in the conclusion, and as the names of the two headline configurations Stability-HCL and Plasticity-HCL — but neither is ever defined. There is no sentence of the form "we define plasticity as ..."; the concept is imported by reference: "It extends the classical stability-plasticity problem from model state to harness state" (Sec. 1). Nor is either quantity measured: there is no plasticity metric, score, or state variable anywhere in the paper. What is measured is the outcome on each side — $\mathrm{Avg}_T$ and $\mathrm{Fgt}_T$ (eq. 15) — plus two per-decision quantities inside the gate, $\Delta_n$ (current improvement) and $D_n$ (a count of damage already done to anchors). Nothing reads out how much future adaptability the harness currently retains. The operational reading (plasticity = adaptation to the current task, stability = retention of earlier behavior) is inferable from usage, for example "Plasticity-HCL therefore favors capability acquisition, whereas Stability-HCL provides a better balance between adaptation and retention" (Sec. 4.2.1), but is never stated. The paper cites no loss-of-plasticity literature; the word "budget" appears three times and always means the per-decision tolerance $B_n$, never an allowance that is drawn down — "we fix $B_n \equiv b$ for all candidate decisions", so a candidate that costs three anchors under $b = 3$ does not reduce what a later candidate may cost.

### b) Key innovations and their significance

1. **Naming and formalizing harness-level forgetting, and demonstrating it is measurable** (significant). The stage-wise protocol of eq. 14 applied to a harness rather than a model is the paper's most transferable contribution: it makes "the harness got worse at what it used to do" a number instead of an anecdote.
2. **The commit gate of eq. 13 as a deployment condition** (notable). Separating proposal from commitment, and requiring a rerun of stored earlier cases before deployment, is a clean and reusable mechanism. Its novelty is in the retention conjunct; validation-gated acceptance alone is standard in prompt- and workflow-optimization work.
3. **Treating four harness components as one jointly versioned state with atomic commitment** (notable). This is the paper's answer to cross-component coupling, and the ablation is built to defend it.
4. **A single scalar operating-point control $B_n$ with a sweep** (incremental as a mechanism, notable as an empirical result). The knob is a hand-set threshold on a counter. What makes it worth reading is the non-monotone finding it produces (section 4b below).
5. **The conceptual mapping of the five model-centric families onto harness mechanisms** (incremental; the authors themselves hedge it as "conceptual rather than one-to-one implementations").

### c) Applicability

**Where it excels, per the paper's own evidence**: task streams where the frozen model is capable but badly configured — output formats, parsing conventions, workflow order. The largest gains are precisely there: multimodal detection 4.27 to 65.34 and grounding 43.00 to 91.60 (Table 4), where "the harness must organize spatial information into task-specific outputs", and GSM8K 49.40 to 92.00 (Table 3). It also excels in long-horizon environments where reusable procedures compound: Minecraft, where a static harness plateaus at 15 of 50 tasks and HCL completes all 50 (Fig. 3).

**Where it struggles**: tasks the frozen model already does well by default, where harness structure can only get in the way — VQAv2 is the one multimodal task where Zero-shot (84.87) beats both HCL profiles (79.33 / 79.80), and MuSiQue *regresses* under both textual profiles (35.00 zero-shot to 27.60 / 29.00). Both are cases where the harness accumulated for other tasks in the stream is imposed on a task that did not need it.

**Structural limits on applicability**: the gate needs current-task validation cases with a computable score and rerunnable anchors with binary success criteria, so it requires a verifiable task family and the ability to replay earlier inputs under identical conditions. The evaluation cost scales with the anchor set (80 anchors per earlier task in the sweep) times the number of candidates, and no cost accounting is reported.

### d) Comparison table

| Method | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| **HCL (this paper)** | Whole-harness state updated jointly; retention enforced before deployment, not diagnosed after; one scalar sets the operating point; works on a frozen model of any family (four different backbones used); best final average in all four settings | Every commit requires current-task improvement, so consolidation-only edits are inadmissible by construction; retention enforced on a finite anchor sample, so $b=0$ still forgets; thresholds hand-set and unprincipled; nothing in the system is learned; rerun cost of anchors unreported; tools/outer capabilities never revised despite abstract wording | Learn or adapt the proposer instead of a fixed prompt template; make the retention constraint an estimated quantity rather than a count over a sampled anchor set; admit maintenance edits with a non-improvement objective; schedule $B_n$ over the stream rather than fixing it per run; report evaluation cost |
| **Static Harness** (no evolution) | Zero forgetting by construction; zero adaptation cost | Plateaus: 47.12 on ALFWorld, 15/50 Minecraft tasks | — (it is a control) |
| **RAG baseline** (ALFWorld) | Cheapest adaptive baseline; lowest forgetting of the adaptive baselines (1.74); 47.12 to 55.56 | "retrieval alone cannot revise reusable procedures or routing rules" | Combine retrieval with procedure/routing revision, which is what HCL does |
| **MemP** (Fang et al. 2026), **MemRL** (Zhang et al. 2026c) | Procedural / episodic memory adaptation improves individual categories | Performance "varies considerably across the stream"; highest forgetting among the adaptive baselines (5.18, 5.64); no retention criterion | Add a retention gate over memory edits |
| **DGG** (Li et al. 2026b, multimodal) | A trained sequential multi-task method with a matching setting; low forgetting (0.26) | Much lower final average (42.73 vs 68.92) and much worse VQAv2 (62.60 vs 79.33), i.e. it damages a task the frozen model already did well | — |
| **Zero-shot** | Strongest on tasks the frozen model already handles (VQAv2 84.87; MuSiQue 35.00) | No accumulation; 45.50 textual, 39.40 multimodal, 34.84 in the ablation stream | — (it is a control) |

Caveat on the comparisons: MemP and MemRL are "reimplemented within our framework with unified data processing and action selection" and, in Minecraft, "reproduced within our harness as memory-management baselines, rather than run from their official repositories".

---

## 4. Experimental Validation

### a) Experimental design

Four settings, four different frozen backbones, chosen "to examine whether HCL generalizes across model families and scales rather than depending on a particular model". Within each setting the backbone is identical across all compared methods and frozen throughout.

| Setting | Frozen model | Stream | Data per task | Reporting |
|---|---|---|---|---|
| ALFWorld (Shridhar et al. 2021) | Qwen3.5-9B | 6 categories in order: Pick-and-Place, Look-in-Light, Clean, Heat, Cool, Two-object | 10 training episodes per category, max 50 interaction steps per episode | 134 official evaluation episodes; category macro-average; forgetting over the first five categories |
| Minecraft (Voyager-style, Wang et al. 2024a) | Qwen3.6-27B | 50-task curriculum: collection, crafting, mining, tool use, placement, smelting, multi-step dependencies | Sequential environment feedback; retained skill tests as anchors | Cumulative task completion, recovery events, validated skill changes; "Completed tasks are not systematically replayed after every update" |
| Textual reasoning | DeepSeek-V4-Flash | MuSiQue, ProofWriter, GSM8K, HotpotQA | 250 adaptation / 50 validation | 500 test per task |
| Multimodal perception | Qwen3.6-27B | COCO detection, COCO captioning, RefCOCO grounding, VQAv2 | 250 adaptation / 50 validation | 500 test per task |
| Retention sweep (textual) | DeepSeek-V4-Flash | Same textual order | 300 adaptation / 80 validation, 80 anchors per earlier task, 40 proposal opportunities per profile (ten per stage) | 600 test per task |
| Component ablation (multimodal) | Qwen3.5-4B | Same multimodal order | 250 / 50 | 500 test per task |

The evaluation protocol is eq. 14 applied after every stage. The three data pools are kept disjoint by role: adaptation examples drive the Optimizer, validation cases and anchors belong to the Evaluator only, and the final test sets are used only for reporting. Anchors are additionally hidden from candidate generation.

Anchor success criteria are pinned per task in Appendix C, which is what makes $q(H,a)$ concrete: exact normalized answer match (MuSiQue, HotpotQA); label match plus valid schema (ProofWriter); normalized numeric equality (GSM8K); correct category with intersection-over-union at least 0.5 and valid box schema (COCO detection); sentence-level CIDEr at least 0.5 on a normalized scale (captioning); intersection-over-union at least 0.5 (RefCOCO); VQA consensus score exactly 1.0 (VQAv2); the environment goal predicate true within 50 steps (ALFWorld); the retained skill test reaching its inventory or world-state predicate (Minecraft).

Gate settings. Main profiles: a candidate "must improve by at least one validation case for discrete metrics or strictly improve the designated continuous score, without introducing an invalid outcome"; Stability-HCL uses $B_n = 0$ and Plasticity-HCL $B_n = \infty$. Sweep: $\delta_n$ requires two additional correct predictions among 80 validation cases, validity requires at least 90.00% output-format compliance plus no syntax, tool-use, or environment violations, and only $B_n \equiv b$ varies over $\{0, 1, 3, \infty\}$. Minecraft applies $B_n = 0$ to retained skill tests only, "and therefore evaluates skill-level rather than full task-level retention".

### b) Key results

1. **ALFWorld (Table 2)**: Static Harness 47.12 final average; RAG 55.56 (forgetting 1.74); MemP 53.15 (5.18); MemRL 51.51 (5.64); Stability-HCL 61.74 (2.64); Plasticity-HCL 62.98 (10.94). Plasticity-HCL solves 100.00 on the final Two-object category but forgets roughly four times as much as Stability-HCL; Stability-HCL is best in four of six categories.
2. **Textual reasoning (Table 3)**: Zero-shot 45.50; Stability-HCL 52.20 with forgetting exactly 0.00; Plasticity-HCL 64.70 with forgetting 0.07, driven mainly by GSM8K 49.40 to 92.00 and ProofWriter 42.80 to 77.00. Both profiles lose on MuSiQue (35.00 to 27.60 / 29.00).
3. **Multimodal perception (Table 4)**: Zero-shot 39.40; DGG 42.73 (0.26); Plasticity-HCL 67.96 (0.81); Stability-HCL 68.92 (0.22) — here the retention-strict profile wins on both axes. Detection 4.27 to 65.34, grounding 43.00 to 91.60; VQAv2 is the exception (84.87 zero-shot vs 79.33).
4. **Retention sweep (Table 5), the paper's most interesting result**: forgetting is monotone in $b$ — 0.39, 1.22, 2.00, 3.45 for $b = 0, 1, 3, \infty$ — but final average is not: 61.25, **63.46**, 62.04, 60.13. The best final harness is at $b = 1$, and the *unconstrained* setting is the *worst* of the four. The authors' hedged explanation: "One possible explanation is that each committed update changes the subsequent evolution trajectory: without historical constraints, locally beneficial updates may overwrite reusable harness contents, weakening both retention and the experience or capabilities available for later tasks." Note that $b$ is a fixed per-run constant, so this is a comparison of four whole trajectories, not of a schedule.
5. **Minecraft (Fig. 3, no table)**: HCL completes all 50 curriculum tasks while the Static Harness "follows HCL for 15 tasks and then plateaus"; cumulative environment actions 83 (HCL) vs 88 (MemRL) vs 91 (MemP), read as less redundant execution.
6. **Component ablation (Tables 6 and 10, Qwen3.5-4B)**: Zero-shot 34.84; w/o Interface update 62.37 (forgetting 0.11, 24 commits); w/o Memory update 62.28 (0.83, 46); w/o Capability update 63.12 (0.06, 16); w/o Router update 62.77 (0.14, 4); Full HCL 63.41 (0.45, 18). Memory and Interface matter most for final performance; disabling Memory is the only variant that *increases* forgetting, which the paper reads as evidence that evolving memory supports both acquisition and retention. Capability updates matter least here, attributed to the domain: this stream "relies less on reusable executable procedures than the Minecraft curriculum". The commit counts are explicitly not comparable acceptance rates, "because variants do not necessarily share a proposal sequence".

The abstract's headline, "relative gains exceeding 10% over corresponding baselines in multiple settings", is consistent with the tables: Stability-HCL over RAG on ALFWorld is 61.74 vs 55.56 (about 11% relative, computed from the table), Plasticity-HCL over Zero-shot on textual is 64.70 vs 45.50 (about 42%), Stability-HCL over DGG on multimodal is 68.92 vs 42.73 (about 61%).

### c) Where it shines

The clearest evidence is where the frozen model has latent capability blocked by configuration: multimodal detection (a 15x improvement over zero-shot, 4.27 to 65.34) and grounding (43.00 to 91.60) — the paper attributes these to the harness learning to "organize spatial information into task-specific outputs" — and GSM8K under the permissive profile (49.40 to 92.00). The second clear case is long-horizon skill accumulation: Minecraft, where progression is the metric and the static control stops at 15/50.

The most decision-relevant result for anyone designing a retention mechanism is the sweep: unrestricted commitment is not merely riskier, it produced the *worst* final harness of the four settings tested.

### d) Limitations

Acknowledged in the paper:

- **A zero tolerance does not buy zero forgetting.** "The remaining forgetting at $b = 0$ occurs because the constraint covers a finite anchor set, whereas forgetting is evaluated on separate historical test cases. Preserving all anchors currently solved by $H_n$ cannot guarantee unchanged behavior on historical cases not represented by $A_n$" (Sec. 4.4). The forgetting numbers for $B_n = 0$ are 0.39 in the sweep and 0.22 in the multimodal setting; textual Stability-HCL reached 0.00, which the anchor argument says is not guaranteed.
- **Thresholds are unprincipled**: "chosen heuristically ... and remain identical across all settings" (Sec. 4.4).
- **Candidate generation is deliberately cheap**: the coordinate-wise sweep with up to $K$ alternatives exists "while limiting repeated LLM calls" (Sec. 3.3.1), not because it searches well.
- **Minecraft retention is weaker than elsewhere**: skill-level rather than task-level, and "Completed tasks are not systematically replayed after every update".
- **One headline result is explained only speculatively** ("One possible explanation ...").
- **Open problems named in the conclusion**: "efficient retention evaluation, harness-content consolidation, and evaluation over longer interaction streams".

Implicit or structural, not discussed by the paper:

- **Consolidation is inadmissible by construction.** Because eq. 13 conjoins $\Delta_n \ge \delta_n$ with the retention and validity checks, every committed edit must improve the current task. A pure maintenance edit — merging redundant memory entries, generalizing three narrow skills into one, deleting stale guidance — has $\Delta_n \approx 0$ and cannot be committed. This sits directly against the conclusion's own open problem of "harness-content consolidation": the gate as specified forbids the operation the conclusion asks for.
- **$B_n$ is a fixed per-run knob, not a state.** The subscript $n$ admits a schedule in principle, but "we fix $B_n \equiv b$ for all candidate decisions"; nothing adapts it, and nothing in the system estimates how much retention headroom remains.
- **No cost accounting.** The Evaluator reruns up to 80 anchors per earlier task for every candidate of every component; no wall-clock, token, or dollar figures appear anywhere.
- **No variance.** No seeds, error bars, or repeated runs are reported; only that the deployed and candidate harnesses are compared under matched seed conditions. Several headline gaps are around 1 percentage point (the entire ablation spread is about 1.1pp).
- **No harness artifacts shown.** No prompts, templates, skill definitions, or tool schemas are reproduced, so the reader cannot see what an edit actually looks like.
- **The Minecraft comparison is three scalars in a figure**, with no table and no per-task breakdown.
- **No limitations section**, and no safety or adversarial discussion of a self-editing harness.

---

## 5. Reproduction and Application

### a) Open source?

**No public code release.** The paper contains no repository link, no project page, no reproducibility statement, and no supplementary code; a grep of the full text finds "GitHub" only inside a bibliography entry (the SWE-bench title). The arXiv abstract page carries no code link and no comments field. Searches for the exact title plus "github", for the arXiv identifier plus "code", and a GitHub repository search for the paper's distinctive component names returned nothing by these authors; the obvious GitHub handles for the first author (kangborui, borui-kang, BoruiKang) do not exist. Two same-named repositories are unrelated work: `sethkarten/continual-harness` is Karten et al., "Continual Harness: Online Adaptation for Self-Improving Foundation Agents" (arXiv 2605.09998), and `NVlabs/HCL` is Heterogeneous Continual Learning (CVPR 2023). Verified 2026-08-31.

**Key steps to reimplement**, in the order the paper specifies them:

1. Build the four-component harness as separately addressable text artifacts: interface prompts and parsing rules; a raw memory store with a fixed per-task capacity in arrival order; an abstract memory store written by an LLM summarizer; a capability registry split into fixed outer entries and editable inner skills; routing prompts, selection criteria, and workflow templates. Version them jointly so a candidate is one object.
2. Implement the execution path: eq. 5 (parse), eq. 8 (route and assemble), execute, collect $e_n$ per eq. 1.
3. Implement the Optimizer as a single prompt template taking $(H_n, e_n)$ and returning a candidate; add the coordinate-wise loop over components in a fixed order, up to $K$ alternatives each, other components held fixed, best admissible alternative carried forward.
4. Implement the anchor set as evaluation-only state: raw input plus binary success criterion per anchor, refreshed at the end of each task using the predefined success/failure ratio, then frozen for the next task's decisions.
5. Implement the gate of eq. 13 plus the composite ranking score over passing candidates, and make rejection a no-op.
6. Implement the stage-wise protocol of eq. 14 and the two metrics of eq. 15, keeping adaptation, validation/anchor, and test pools disjoint.

### b) Implementation details requiring attention

- **Matched conditions**: deployed and candidate harnesses must be compared "under the same model, decoding, tool, environment, and seed conditions", otherwise $\Delta_n$ and $D_n$ are noise.
- **Thresholds actually used**: main profiles require at least one additional correct validation case (discrete) or a strict improvement (continuous) and no invalid outcome; the sweep requires two additional correct predictions among 80 validation cases plus at least 90.00% output-format compliance. $B_n \in \{0, 1, 3, \infty\}$, fixed within a run.
- **Anchor budget**: 80 anchors per earlier task in the sweep; composition set by a predefined success/failure ratio with shortfalls filled from the other group. Anchors must be invisible to the Optimizer.
- **Memory bound**: Raw Memory keeps a fixed number of interactions per task in arrival order.
- **Skill dependency**: inner skills are distilled from Abstract Memory, so disabling memory updates also removes the source of new skills — a dependency the ablation explicitly flags.
- **One-sided, thresholded historical loss**: gains on other anchors do not offset a loss, and a graded regression counts only when it crosses the anchor's criterion.
- **Value of $K$**, the composite score's weights, the anchor success/failure ratio, and the fixed component visiting order: [Information not available in provided text].
- **Prompt texts for the Optimizer, Interface, and Router**: [Information not available in provided text].
- **Compute, latency, and token cost**: [Information not available in provided text].

### c) Transferability

The mechanism is domain-agnostic in the way that matters: it needs a task with a computable current-task score, earlier cases that can be rerun deterministically enough for a binary success check, and harness artifacts that an LLM can rewrite as text. Any verifiable-task agent stream satisfies this — code repair, tool-use agents, retrieval pipelines, structured extraction. The paper itself demonstrates the span with four different frozen backbones across text, vision-language, and two interactive environments.

Three adaptations follow naturally from what the paper leaves open. The retention conjunct can be transplanted alone into any existing self-improving harness, since it is independent of how candidates are proposed. The anchor set can be replaced by any cheaper retention estimate, which is the paper's own first open problem ("efficient retention evaluation") and the main cost driver. And the improvement conjunct can be relaxed for edits whose purpose is consolidation rather than current-task gain — the paper's second open problem, which its own gate as written forbids.

---

## 6. Summary

### a) One-sentence core idea

Treat the agent's whole harness as continual-learning state, and commit each LLM-proposed edit only if it improves the current task, breaks at most $B_n$ previously solved cases, and stays valid.

### b) Quick-reference pipeline

1. **Run the task with the current setup.** The agent parses the input into (input, goal, constraints), pulls relevant past records and skills, assembles a context, acts, and records what happened together with the feedback.
2. **Ask the model to rewrite its own setup.** The same frozen model is shown the current setup and the record of that interaction, and returns a proposed new setup — revised prompts and parsing rules, new or edited memory notes, new or edited skills, new routing and workflow rules. Components are revised one at a time, a few alternatives each, others held fixed.
3. **Test the proposal twice.** Re-score it on the current task's held-out validation cases, and re-run a stored set of earlier cases the current setup already solves.
4. **Accept only if all three conditions hold**: it improves the current task by at least the set margin, it breaks no more than the allowed number of earlier cases, and it passes format and legality checks. Otherwise keep the old setup unchanged.
5. **After each task, re-score everything seen so far**, and report final average performance and how far earlier tasks fell from their best.

The single dial is step 4's allowed number of broken earlier cases. Set to zero it forgets least; set to unlimited it forgets most and, in the paper's own sweep, ends up with the worst final harness of the four values tested.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched the exact title "Harness Continual Learning: Continual Adaptation Beyond Model Parameters" and the short form on openreview.net and via the OpenReview API on 2026-08-31). The paper is an arXiv preprint with no venue stated on its front page, so no review record exists to report.
