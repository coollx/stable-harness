# MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents

- **Original title**: MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents
- **Authors**: Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, Wenya Wang (correspondence: Haozhen Zhang, Wenya Wang)
- **Affiliations**: Nanyang Technological University (Zhang, Long, Bao, Wang); University of Illinois Urbana-Champaign (Feng); University of Illinois Chicago (Weizhi Zhang); Tsinghua University (Yue). Funded by an NTU Start-Up Grant and a Singapore MOE AcRF Tier 1 Seed Grant.
- **Venue / year**: arXiv preprint, 2026 (arXiv:2602.02474v2 [cs.CL], v2 dated 24 May 2026; the title-page footer reads "Preprint. May 26, 2026"). 33 pages including appendices A-D. The paper claims no venue and carries no camera-ready marks.
- **Source URL**: https://arxiv.org/abs/2602.02474
- **Code**: https://github.com/ViktorAxelsen/MemSkill (Apache 2.0) — cloned to `repo/`, see `repo_analysis.md`.
- **OpenReview**: no public page found (see section 7).

Classification: **method paper** — the contribution is a built artifact (a three-component architecture, a reinforcement-learning objective over an evolving action space, and a released implementation plus the final evolved skill files); the experiments exist to validate the build rather than to measure a phenomenon.

Short names used below: MemSkill (the system); controller, executor, designer (its three components); skill bank (the shared store of memory skills, which evolves during training) and memory bank (the per-trace store of extracted memories, which does not survive the trace).

---

## 1. Method Motivation

### a) Why was this method proposed?

The driving complaint is that agent memory systems are built from a small, fixed vocabulary of hand-written operations. The abstract states it directly: "Most Large Language Model (LLM) agent memory systems rely on a small set of static, hand-designed operations for extracting memory. These fixed procedures hard-code human priors about what to store and how to revise memory, making them rigid under diverse interaction patterns and inefficient on long histories." The introduction names the vocabulary — "fixed operation primitives (e.g., add/update/delete/skip)" plus "heuristic modules that govern what to store, how to revise it, and when to prune it" — and the consequence: such designs "bake in strong human assumptions and often suffer under diverse interaction patterns, scaling poorly as histories grow" (Sec. 1).

The paper's move is to raise memory extraction itself to a learnable object: "Rather than treating memory as the output of fixed operations or hand-designed modules, we propose to elevate memory extraction itself into a learnable abstraction" (Sec. 1). A memory skill is "a structured behavior that specifies when and how interaction traces should be transformed into memory and revised over time" — in other words, a written routine that an executing language model reads, not code.

### b) Pain points of existing methods

Three properties are set out as what an ideal system should satisfy, each stated against a named failing (Sec. 1). First, **minimal reliance on human priors**: "Instead of manually encoding what is worth remembering for a domain, memory behaviors should be shaped by interaction data and updated as task demands evolve." Second, **support for larger extraction granularity**: "Many approaches are tuned to a fixed unit, such as per-turn processing, and can weaken when applied to longer spans." Figure 1 contrasts the two regimes — prior systems "interleave handcrafted operations with LLM calls to incrementally extract and revise memory turn by turn", whereas MemSkill processes a span in one pass. Third, **skill-conditioned, compositional memory construction**: "Existing systems often decompose memory construction into specialized modules. In contrast, we prefer to select and compose a small set of relevant skills for the current context and apply them in one generation step."

Learned memory managers are acknowledged and then faulted for scope, not for method: Memory-R1 and Mem-alpha "optimize memory management with reinforcement learning using downstream task signals", but "memory management is still largely governed by static, hand-crafted routines for extraction, consolidation, and pruning" (Sec. 2.1) — that is, they learn to choose among fixed operations rather than to change the operations. Concurrent self-evolving memory work is separated by target (Sec. 3, related-work paragraph): "EvoMemory evaluates streaming memory evolution, MemEvolve optimizes predefined memory architectures, MemGen targets latent memory for reasoning, and ReasoningBank distills reasoning strategies from experience. By contrast, we target the evolution of memory skills themselves."

### c) Core hypothesis / intuition

If the operations that build memory are written as reusable natural-language routines rather than hard-coded procedures, then a small trainable policy can learn which routines to apply to the current context, and a language model can periodically rewrite and extend the routine set from observed failures — improving both how memory is used and what memory behaviors exist.

---

## 2. Method Design (PRIMARY FOCUS)

### a) Pipeline overview: Input to Processing to Output

MemSkill keeps **two stores with different lifetimes**, and the distinction carries most of the paper's design weight (Sec. 3.1): "The memory bank is trace-specific and stores memories for each training trace (e.g., a long dialogue). In contrast, the skill bank is shared across traces and contains reusable memory skills. During training, the controller and executor build each trace's memory bank, while the designer updates the shared skill bank between phases." Figure 2 labels the memory bank "Reset Every Trace" and the skill bank "Refines and Evolves Over Training".

**Per-trace inner loop (controller plus executor).** Input is one interaction trace — a long dialogue, an expert trajectory, or a document set. A segmenter splits it into fixed-length contiguous spans by token count (Sec. 3.3.1: "we split each interaction trace into fixed-length contiguous spans by token count and process them sequentially"); on conversational benchmarks a dialogue session is the processing unit, and at evaluation time the default span size is 512 tokens (Sec. 4.1, App. B.1). Spans are processed in order. For span $t$: (1) the retriever pulls up to $R$ memories $M_t$ from the current trace's memory bank, which is empty for the first span; (2) a fixed embedding model encodes the span text $x_t$ and each retrieved memory, and the controller maps them to a state vector $h_t$ (eq. 1); (3) the controller scores every skill currently in the bank against $h_t$ and samples an ordered Top-K subset $A_t$ without replacement (eq. 2 and eq. 3); (4) the executor — a fixed language model, never fine-tuned — receives the span, the retrieved memories, and the selected skills' full content, and emits structured memory updates "in one pass" (Sec. 3.3.2), which are parsed and applied to this trace's memory bank. Output of the inner loop is a completed memory bank for that trace.

**Reward and controller update.** After the whole trace is processed, "we evaluate the resulting memory bank on the trace's memory-dependent training queries and use the resulting task performance as the reward (e.g., F1 or success rate)" (Sec. 3.3.3). The scalar becomes the episode reward $R$ (eq. 12), assigned to the terminal step; the controller is trained by proximal policy optimization over the span sequence (eq. 14 to eq. 18).

**Periodic skill evolution (designer).** While the controller trains, every query that is answered incorrectly is written into a sliding hard-case buffer, "recording the query along with its ground-truth and metadata (e.g., retrieved memories and model prediction), as well as summary statistics such as task performance and the number of failures observed so far" (Sec. 3.4). The buffer expires cases by age and by capacity. Every fixed number of controller-training steps (100 in the main experiments, Sec. 4.1), the designer: scores each case by difficulty (eq. 4); clusters cases "by semantic similarity of their queries" with KMeans and mines representative cases per cluster, so "the designer feedback is not dominated by a single frequent error mode" (App. B.2); then runs **two-stage evolution** (Sec. 3.4) — stage one, "it employs an LLM to analyze the selected hard cases and identify what memory behaviors are missing or mis-specified"; stage two, "it uses the resulting analysis to propose concrete edits to existing skills and to introduce new skills", capped at 3 edits per round (Sec. 4.1).

**Closed loop with a gate.** Sec. 3.5: "Each cycle begins with controller training on the current skill bank, where the executor constructs memories and accumulates challenging cases. The designer then updates the skill bank using representative hard cases, optionally rolling back to a prior snapshot if performance regresses. The next cycle resumes controller training on the updated skill bank, with additional exploration to encourage early use of new skills." The rollback criterion is eq. 8 and the exploration incentive is eq. 5 to eq. 7, both detailed below.

### b) Architecture components

**Skill bank (Sec. 3.2).** Each skill $s \in S$ carries "(i) a short description for skill representation and selection, and (ii) a detailed content specification that instructs the executor on memory extraction or revision." The content is a four-field template rendered in Figures 1 and 4: Purpose / When to use / How to apply / Constraints. The bank is initialized with exactly four primitives — INSERT, UPDATE, DELETE, SKIP — "to ensure a stable and functional initialization", and "from this minimal set, the designer progressively refines existing skills and expands the bank by proposing new skills that address uncovered failure modes."

**Controller.** Three independent lightweight multilayer perceptrons (Sec. 4.1), $f_{\theta}^{ctx}$, $f_{\theta}^{skill}$, and $f_{\theta}^{score}$, wrapped around a frozen embedding model $e(\cdot)$ instantiated as Qwen3-Embedding-0.6B, which also serves as the memory retriever. The design constraint that shapes it is that the action space grows: "Instead of producing a fixed-dimensional action head tied to a fixed number of skills, the controller concatenates the state representation with each candidate skill representation and applies a shared scorer to all such state-skill pairs in parallel" (Sec. 3.3.1), so the logit vector $z_t \in \mathbb{R}^{|S_t|}$ "adapts as the skill bank evolves." Note that skills are represented by their *description* only, "as a compact and stable semantic signal rather than the full skill content".

**Executor.** A fixed language model (LLaMA-3.3-70B-Instruct for training, Qwen3-Next-80B-A3B-Instruct for transfer), prompted with span, retrieved memories, and selected skills. Its output is parsed into memory operations. The efficiency argument lives here: "By composing several skills for the same text span and extracting memory in one LLM call, MemSkill reduces repeated per-turn processing and scales better to long interaction histories" (Sec. 3.3.2).

**Designer.** A fixed language model that reads mined hard cases and writes skill edits. It never touches the controller's weights and never touches any memory bank; its only output is a changed skill bank.

### c) Formulas and algorithms

**State and skill representations (eq. 1).** $h_t = f_{\theta}^{ctx}([e(x_t); \bar{m}_t])$ with $\bar{m}_t = \frac{1}{R}\sum_{r=1}^{R} e(m_{t,r})$, and $u_i = f_{\theta}^{skill}(e(\mathrm{desc}(s_i)))$. The retrieved-memory term is "omitted when $M_t$ is empty", that is, on the first span of every trace.

**Skill scoring (eq. 2).** $z_{t,i} = f_{\theta}^{score}([h_t; u_i])$, $p_{\theta}(i \mid h_t) = \mathrm{softmax}(z_t)_i$. One shared scorer applied to every state-skill pair, which is what makes a variable-size action space possible.

**Top-K joint probability (eq. 3, restated as eq. 10 and eq. 11).** $\pi_{\theta}(A_t \mid h_t) = \prod_{j=1}^{K} \frac{p_{\theta}(a_{t,j} \mid h_t)}{1 - \sum_{\ell<j} p_{\theta}(a_{t,\ell} \mid h_t)}$ — the probability of drawing an ordered set without replacement, which "reduces to the single-action case when $K=1$". Sampling is done by the Gumbel-Top-K trick: add independent Gumbel noise to the logits and take the $K$ largest.

**Difficulty score for hard cases (eq. 4).** $d(q) = \big(1 - r(q)\big) \cdot c(q)$, where $r(q) \in [0,1]$ is the task reward for query $q$ and $c(q)$ is "its cumulative failure count within the buffer window". So a case is prioritized both for being badly answered and for being repeatedly badly answered. (The released code computes $\mathrm{severity} \cdot \log(1+c)$ instead of $\mathrm{severity} \cdot c$; see `repo_analysis.md`.)

**Exploration incentive for new skills (eq. 5 to eq. 7).** After an evolution round, newly added skills $S_{new}$ have never been selected, so the controller would ignore them. The fix acts on logits: require $\sum_{i \in S_{new}} p_{\theta}(i \mid h_t) \ge \tau_q$ (eq. 5); when violated, add a uniform gain $\delta_q$ to every new skill's logit, "chosen as the minimal value that makes" the constraint hold (eq. 6); and decay the target linearly over the window, $\tau_q = \tau_0 (1 - q/T_{explore})$ (eq. 7), with $T_{explore} = 50$ steps and $\tau_0 = 0.3$. "By operating on logits, this mechanism preserves the controller architecture"; the decay yields "a smooth transition back to the controller's learned selection behavior."

**Cycle score, rollback, early stopping (eq. 8).** "Because the reward signal can be volatile immediately after a skill-bank update, we assess whether a cycle improves performance using a stabilized reward estimate: we compute the average task reward over the last quarter of training steps within the cycle" — $\bar{r}_{tail} = \frac{1}{L/4}\sum_{t=3L/4+1}^{L} r_t$ for a cycle of $L$ controller-training steps. "We compare $\bar{r}_{tail}$ against the best score observed so far. If the current cycle does not improve this criterion, then before performing the next skill evolution step, we roll back the skill bank to the previously best-performing snapshot and restart evolution from that snapshot." And: "if the stabilized reward fails to improve for several consecutive evolution cycles (we use a fixed patience), we early stop training and return the best skill bank snapshot encountered during training."

**Return and policy objective (eq. 12 to eq. 18).** $R \triangleq \mathrm{Eval}(\text{memory bank}; \text{training queries})$ (eq. 12). Returns use a discount over spans, $G_t = \sum_{\tau=t}^{T} \gamma^{\tau-t} r_{\tau}$ (eq. 13), and the paper is explicit that the reward is terminal: "In our default setting, reward is provided only after memory construction completes, i.e., $r_T = R$ and $r_{\tau} = 0$ for $\tau < T$, so $G_t = \gamma^{T-t} R$." A value function $V_{\phi}(h_t)$ is learned and advantages come from generalized advantage estimation. The importance ratio (eq. 14) uses the Top-K joint log-probability, the surrogate is the standard clipped objective (eq. 15), and the total objective is $\max_{\theta,\phi} L_{policy}(\theta) - c_v L_{value}(\phi) + c_H H(\theta)$ (eq. 18). Entropy is computed on the full categorical distribution, not the executed subset, "which encourages exploration of the evolving skill bank even though the executed action is a Top-K set."

---

## 3. Comparison with Existing Methods

### a) Fundamental differences

Three separations, in decreasing order of how much they matter.

**The operation set is data, not code.** Prior memory systems fix add/update/delete/skip and vary only the surrounding heuristics; MemSkill starts from those same four and lets a language model rewrite and extend them during training. Figure 4 shows what comes out: on LoCoMo, skills named "Capture Temporal Context" and "Capture Activity Details"; on ALFWorld, "Capture Action Constraints" and "Track Object Location". The paper's reading is that "the evolved skill bank reflects recurring information needs from the data, rather than a fixed notion of what to remember" (Sec. 4.5).

**Selection is trained, not heuristic.** Memory-R1 and Mem-alpha train with downstream task signal but over a fixed operation set; MemSkill trains a selector whose action space changes size mid-training, which is what the shared-scorer design (eq. 2) and the new-skill logit gain (eq. 5 to eq. 7) exist to support.

**Granularity is a span, not a turn.** One executor call per 512-token span replaces one or more calls per conversational turn. This is the source of the efficiency result in Table 3 and is orthogonal to the skill idea — it would help any extraction pipeline.

### b) Key innovations and their significance

1. **Skill bank as an evolving, natural-language action space with a trained selector over it** — *significant*. The combination of a variable-size action space, a joint Top-K log-probability for policy-gradient training over it (eq. 10 and eq. 11), and an exploration incentive for freshly added actions (eq. 5 to eq. 7) is a complete and non-obvious recipe; the ablation in Table 2 shows both halves are load-bearing.
2. **Designer driven by clustered, difficulty-ranked hard cases** — *notable*. Difficulty ranking (eq. 4) plus KMeans coverage is a concrete answer to "which failures should drive the next edit", a question most self-improving systems answer with "the most recent batch".
3. **Snapshot, rollback, and early stopping on a stabilized reward** (eq. 8) — *notable*, and the piece most relevant to us. It is a working gate-and-revert mechanism on a harness layer, with an explicit answer to reward volatility right after an edit.
4. **Span-level one-pass extraction** — *incremental but high-value*: Table 3 shows it is where most of the cost win comes from.

### c) Applicability

It excels where a corpus of interaction traces has recurring structure that a written routine can name — dialogue histories with temporal and activity structure, ALFWorld task templates that share goal structure, long document sets. It also excels where inference cost matters: Table 3 shows quality held while calls drop roughly six-fold against A-MEM.

It should struggle where failures are not describable as a missing extraction behavior (the designer can only write a new routine, never change the retriever, the segmenter, or the executor), where traces are short (the span-level win vanishes), and where the training signal is sparse — the reward is a single scalar per trace, spread by discount across all spans, so credit assignment to an individual span's skill choice is weak by construction (eq. 13).

### d) Comparison table

| Approach | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| Hand-designed memory pipelines (MemoryBank, Mem0, LangMem, MemoryOS, A-MEM) | No training; predictable; easy to audit | Fixed operation set and fixed granularity; "brittle under distribution shift" (Sec. 1); expensive — 685 to 1548 LLM calls on LoCoMo (Table 3) | Expose the operation set as editable text, which is exactly what MemSkill does |
| Learned memory managers over fixed operations (Memory-R1, Mem-alpha) | Downstream signal drives management; end-to-end | The operation vocabulary is still hand-written; cannot invent a behavior it was not given | Let the action space grow; MemSkill's shared scorer (eq. 2) shows one way |
| Experience distillation (ExpeL, ReasoningBank, AWM) | Cheap; reuses trajectories; no policy training | Distilled insights are retrieved, not selected by a trained policy; no gate on whether an added insight helps | Add a selection policy and a rollback criterion |
| MemSkill | Evolving natural-language action space; trained selector robust to action-set growth; explicit rollback and early stop; best quality-cost point in Table 3 | Trains only on 6 LoCoMo conversations; reward is one scalar per trace; evolution capped at a handful of rounds; no measurement of what earlier behavior is lost | Score candidate skill edits on held-out earlier traces rather than on the current cycle's own training reward; publish per-round curves |

---

## 4. Experimental Validation

### a) Experimental design

Four benchmarks (Sec. 4.1): **LoCoMo** (10 long conversations, roughly 200 training queries each, split 6/2/2 by sample into train/validation/test, adversarial queries removed), **LongMemEval-S** (roughly 100K tokens per example, abstention questions removed, evaluated on a stratified test sample of about 100 examples), **HotpotQA** at 50/100/200 concatenated documents (Sec. 4.3, transfer only), and **ALFWorld** ALF-Seen and ALF-Unseen with expert trajectories from the training split as the memory corpus. Appendix A adds **AppWorld** (Test-Normal and Test-Challenge) and a smaller-backbone study with Llama-3.1-8B-Instruct.

The critical scope fact: **training happens on LoCoMo only, with LLaMA-3.3-70B-Instruct only** (plus a separate ALFWorld training run). "LongMemEval is also evaluated in a transfer setting, where we directly apply the skills learned on LoCoMo without further training" (Sec. 4.1), and every Qwen number is marked with the transfer triangle. Metrics are F1 and an LLM-judge score (judge: openai/gpt-oss-120b, App. B.1) for conversational tasks, success rate and environment steps for ALFWorld, pass rate and steps for AppWorld. Training used A6000 GPUs; base models were called through hosted APIs. Key settings: $K=3$ skills per unit during training, $K=7$ at evaluation for LoCoMo and LongMemEval, $K=5$ for ALFWorld; evolution every 100 training steps with at most 3 edits per round; up to 20 memory items retrieved, for MemSkill and for every baseline, "for consistency".

The ALFWorld training protocol deserves separate note (App. B.3), because it is the only place the reward comes from data the memory was not built on: "For each task type, we randomly sample a subset of trajectories as the experience corpus used for memory construction, and sample another non-overlapping subset of trajectories from the same type as evaluation cases... while still ensuring that evaluation traces are held out from the traces used to build memory."

### b) Key results

1. **Best LLM-judge score on both conversational benchmarks, under both base models (Table 1).** With LLaMA: LoCoMo 44.21 F1 / 53.82 judge and LongMemEval 31.12 / 60.89, average judge 57.36. The strongest baseline by average judge score is CoN at 49.08 (41.72 on LoCoMo, 56.44 on LongMemEval); the strongest dedicated memory system, MemoryOS, reaches only 44.24 (48.64 and 39.83). With Qwen, purely transferred: 42.08 / 54.14 and 25.29 / 60.40, average 57.27, against CoN at 47.51.
2. **Best ALFWorld success rates (Table 1, Table 4).** LLaMA average success 80.36 against the best Table 1 baseline Mem0 at 77.82; Appendix Table 4 adds three experience-reuse baselines, where 80.36 leads LightMem's 74.83 by 5.53 points, the biggest gain on the unseen split — "MemSkill improves the success rate from 75.37 to 83.58 while reducing the average number of steps from 20.69 to 16.63" (App. A.1). Transferred to Qwen, average 81.29 against Expel at 73.68.
3. **Transfer under distribution shift holds and grows with difficulty (Figure 3).** LoCoMo-trained skills applied to HotpotQA give 67.57 judge at 50 documents (MemoryOS 66.02, A-MEM 65.63), 71.48 at 100 documents, and 67.97 at 200 documents (MemoryOS 60.55, A-MEM 61.72) — "with gains becoming more pronounced in the challenging long-context setting" (Sec. 4.3). Sensitivity to $K$ is described as mild, with $K=7$ best in all three settings.
4. **Both components are load-bearing, and the designer matters more (Table 2).** LoCoMo judge score under LLaMA / Qwen: full system 53.82 / 54.14; random skill selection instead of the trained controller 48.43 / 42.84; designer disabled and the bank frozen at the four primitives 46.50 / 36.15; refinement allowed but no new skills 47.45 / 48.88. The Qwen column is the sharper one — freezing the bank costs 17.99 points there, and the refine-only variant recovers most but not all of it.
5. **Better quality at a fraction of the cost (Table 3).** At span size 512 on LoCoMo: 53.82 judge with 249K input tokens, 18K output tokens, and 215 LLM calls, against MemoryOS at 48.64 with 1013K/165K/1288 and A-MEM at 49.71 with 2850K/362K/1548. Quality falls at span size 1024 (48.11) — "larger span sizes reduce cost, but may hurt performance because each memory update covers a longer span."

### c) Where it shines

The clearest margins appear exactly where a fixed operation set should hurt most. On the no-demonstration ALFWorld setting (Table 5), which the authors introduce because "in-context demonstrations can also act as an external form of memory" and may confound the comparison, MemSkill averages 57.71 success against LightMem's 45.99 — an 11.72-point gap, twice the 5.53-point gap in the demonstration setting — and on ALF-Unseen it lifts success from 46.27 to 59.70. On Qwen without demonstrations it reaches 63.83 against the strongest baseline at 55.80. The second standout is the Qwen ablation column already quoted: when the base model changes, a frozen four-primitive bank collapses to 36.15 while the evolved bank holds at 54.14, which is the strongest single piece of evidence that what the designer wrote is a genuine, portable behavior specification rather than a prompt tuned to one model.

### d) Limitations

The paper has no limitations section; the following are partly acknowledged in passing and partly implicit.

**Training data is very small and one-dimensional.** Six LoCoMo conversations constitute the entire conversational training set (App. B.1, 6/2/2 split of 10 samples). Every conversational claim, including all transfer claims, rests on a skill bank evolved from six traces.

**No per-round evidence for the central claim about evolution.** Appendix A.4 asserts "we observe that the learned skill bank improves progressively over evolution rounds", but no figure or table reports reward, score, or skill count by round. The ablations compare endpoints (full versus frozen versus refine-only), never trajectories. Given that the paper's own stability machinery — rollback plus early stopping on patience — exists precisely because rounds can make things worse, the absence of a per-round curve is the paper's largest evidentiary gap.

**The gate is measured on the same distribution that produced the edit.** Eq. 8 compares the mean reward over the last quarter of the current cycle's controller-training steps against the best such value seen so far. Those steps are fresh episodes sampled from the same six-conversation training pool. A validation split is constructed (App. B.1) and is used only "to tune dataset-specific configurations" for LongMemEval; it gates nothing, and in the released code it is never read at all (`repo_analysis.md`).

**No earlier trace is ever re-scored, so nothing measures what an edit costs.** The memory bank is destroyed at the end of every trace, and a skill edit is judged only by rewards collected after it. A skill rewritten in round 4 is never tested against the queries that round 1 already answered correctly.

**Credit assignment is weak by construction.** One scalar per trace (eq. 12), zero intermediate reward, distributed over every span decision by discounting alone (eq. 13). With $K=3$ selections per span over a bank of a dozen skills, the number of distinct actions vastly exceeds the reward resolution.

**Cost accounting excludes training.** Table 3 covers inference only; the paper argues "since MemSkill learns and evolves the skill bank before deployment, this preparation cost is amortized over repeated use" (Sec. 4.5). The executor and designer are 70-to-80-billion-parameter API models called once per span across hundreds of controller-training steps, so the preparation cost is large and unreported.

**The designer's reach is narrow.** It writes skills. It cannot change the retriever, the segmenter, the span size, the number of retrieved items, or the executor prompt scaffold — all of which are hand-tuned constants (Sec. 4.1) and several of which Table 3 shows to matter more than any single skill.

---

## 5. Reproduction and Application

### a) Open source?

Yes: https://github.com/ViktorAxelsen/MemSkill, Apache 2.0, linked from the abstract's last line. The clone contains the full training loop, the controller, the designer, the ALFWorld pair-training variant, launch scripts for LoCoMo and ALFWorld, and — usefully — the final evolved skill banks as markdown files (9 conversational skills, 7 embodied task skills), which match the case studies in Figure 4 and Appendix C. It does not contain training logs, result files, or trajectories. See `repo_analysis.md` for the file-level reading.

Reproduction path: obtain LoCoMo and LongMemEval-S, stand up an inference endpoint for a 70-billion-parameter instruct model, download Qwen3-Embedding-0.6B, then run the provided LoCoMo script (100 inner epochs, 10 outer epochs, batch size 4, $K=3$ during training, evolution every outer epoch, at most 3 edits per round, F1 reward) and evaluate with $K=7$, span size 512, top-20 retrieval.

### b) Implementation details requiring attention

- **The reward metric is the training signal, and it is F1 for LoCoMo but an LLM judge for ALFWorld.** Difficulty scoring (eq. 4) reads the same metric, so the judge choice propagates into which cases the designer sees.
- **$K$ differs between training (3) and evaluation (7).** The controller is trained to compose three skills and deployed composing seven; Figure 3 shows the gap is worth several points, but nothing trains for it.
- **Skills are selected from descriptions, not content** (eq. 1). A designer edit that improves a skill's instructions without touching its description is invisible to the controller's selection.
- **The new-skill logit gain has a hard 50-step window and a 0.3 initial mass** (eq. 7). Too short and a new skill is never adopted; too long and it crowds out learned behavior. This is the one knob that makes an evolving action space trainable at all.
- **Rollback restores the skill bank only.** The controller weights trained under the rejected bank are kept (confirmed in code, `repo_analysis.md`), so a revert is partial by design.
- **Span size is the main cost-quality knob** (Table 3), and it is not monotone: 512 beats both 256 and 1024 on quality.

### c) Transferability — and transfer to our own work

**Which harness layer does each component edit?** This is the question that decides how MemSkill should be cited, and the answer is not the one the title suggests.

- The **executor** writes the **long-term memory layer** — but only a per-trace scratch copy of it. The memory bank is initialized empty at the start of every trace (Figure 2, "Reset Every Trace"; Sec. 3.1) and never survives to the next one. During training, nothing MemSkill puts in memory persists across tasks. What the title calls self-evolving memory is, at the level of durable state, self-evolving skills.
- The **designer** writes the **skill layer**. Its edits are the only durable harness change the system makes: insert a new skill or rewrite an existing one, at most 3 per round, into a shared bank seeded with 4 primitives. The edit format — Purpose / When to use / How to apply / Constraints (Figure 1) — is a concrete, reusable specification for what a skill-layer edit may contain.
- The **controller** edits nothing. It is a trained selection policy over the skill layer, and its weights are persistent state that is not one of our layers. The right reading is that MemSkill splits the work we assign to one harness agent into two: a fast trained policy that decides which existing skill applies now, and a slow language model that decides what skills should exist at all.

So MemSkill edits the skill layer, and that layer governs how the memory layer gets written. That two-level arrangement — a durable layer whose content is a policy for writing another layer — is worth noting against our framing, which orders layers by edit depth but does not name the case where one layer's content controls another's.

**How is the controller trained, and over what horizon?** The reward is eq. 12: the score of the *same trace's own* memory-dependent training queries, computed immediately after that trace's memory was built. The discount $\gamma = 0.99$ in eq. 13 operates over spans *inside one trace* and nothing else; episodes are independent, and the memory bank resets between them. In our vocabulary, the controller is trained at horizon zero at the task level: an edit — here, a selection — is scored by the batch that produced it and by nothing after. The only thing $\gamma$ buys is credit assignment from a terminal reward back to earlier spans of the same document.

**What signal does the designer use, and when is it scored?** Failures from the sliding hard-case buffer: queries answered incorrectly during recent controller training on the training traces, each recorded with its reward at the moment it was answered, under whatever skill bank was in force then. Cases expire by age and capacity (Sec. 3.4), are ranked by eq. 4, and are clustered so that several error types are represented. Two properties matter for us. First, the signal is *recent-window*, so a failure mode that the bank fixed three rounds ago drops out of view and cannot be regressed against. Second, the cases are scored *before* the edit that responds to them and are never re-scored after it; the buffer measures what is currently failing, never what an edit broke.

**Where MemSkill sits against our claim.** It is a horizon-zero system at the task level, with the most developed gate-and-revert mechanism in our index so far. Eq. 8 is a genuine commit gate: an evolution cycle that does not beat the best stabilized reward is not merely ignored, the bank is rolled back to the best snapshot before the next edit is attempted, and after a fixed patience of non-improving cycles training stops and the best snapshot is returned. But the criterion is a mean over the last quarter of the *current cycle's own freshly sampled training episodes*, drawn from the same six-conversation pool. It is same-distribution, forward-looking-by-a-few-steps, and it re-tests no earlier task. There is no anchor set, no held-out earlier-task check, and no retention number anywhere in the paper; a grep of the released code for retention, anchor, holdout, forgetting, and regression vocabulary returns nothing (`repo_analysis.md`). MemSkill therefore confirms rather than complicates our claim about retention measurement, while raising the bar on the gate: when we say existing systems score an edit only on the batch that produced it, MemSkill is the strongest counter-example to the weakest version of that statement and no counter-example at all to the actual one.

**One partial exception worth citing precisely.** The ALFWorld protocol (App. B.3) builds memory from one subset of expert trajectories and scores it on a *non-overlapping* subset of the same task type, with the stated motivation of "a controlled generalization signal ... while still ensuring that evaluation traces are held out from the traces used to build memory." This is the nearest thing in the paper to scoring an edit on data it did not see. It is still same-distribution and same-cycle — both subsets are sampled fresh each episode from the training pool — and it gates nothing about earlier tasks, but it is the design we would extend, not replace, if we wanted a within-task-family generalization term.

**Evidence bearing on loss of plasticity.** Indirect and unquantified. The machinery in Appendix A.4 and eq. 8 — snapshot, rollback, early stopping on patience, a cap of at most 3 edits per round — is built on the premise that continued evolution stops paying and can actively hurt; the released configuration caps the whole LoCoMo run at a small number of evolution cycles. Against that, the paper asserts without evidence that "the learned skill bank improves progressively over evolution rounds". So MemSkill neither shows nor refutes peak-then-decline on a task stream; it shows a team that built defenses against it and then published no curve. That absence is itself citable: a system whose stability section describes three separate mechanisms for surviving bad edits, and which reports no per-round performance, is precisely the gap our stream measurement fills.

**What is directly reusable.** The exploration incentive of eq. 5 to eq. 7 solves a problem we will hit the moment our harness agent adds an entry to a layer that a selector must then choose: a newly written entry starts with no history and is never picked. A minimal-gain-to-target-mass rule with linear decay over a fixed window is a clean answer that does not touch the policy architecture. The snapshot-and-rollback implementation is a working gate-and-revert on a text layer that we can keep structurally and re-key to a retention criterion. The hard-case buffer with per-query failure counts is a ready mechanism for choosing which past failures drive the next edit. The four-field skill template is a good default edit format. What is *not* reusable: nothing in the release lets us pretrain a critic — there are no evolution logs, no per-round scores, no trajectories, and no result files, only the final skill bank.

**One honest tension.** MemSkill's transfer results — LoCoMo-trained skills reaching the best LongMemEval scores untouched (Table 1), holding up across three HotpotQA context lengths (Figure 3), and surviving a base-model swap (the Qwen blocks throughout) — are evidence that a harness edit selected greedily on one batch can still generalize widely. We should cite this rather than skirt it. The reconciliation is that they measure the *final* skill bank's transfer at one point in time, on task families whose structure the skills happen to fit, never the stream behavior of the edit sequence, and never retention of anything earlier. Generalization of the endpoint and plasticity along the path are different quantities, and MemSkill measures only the first.

---

## 6. Summary

### a) One-sentence core idea

Make memory operations editable text, train a small policy to pick which ones apply, and let a language model rewrite them from failures.

### b) Quick-reference pipeline

1. Keep a shared list of written memory routines, starting from four basics (insert, update, delete, skip); each routine states its purpose, when it applies, how to apply it, and what it must not do.
2. For each long interaction history, cut it into fixed-size chunks and walk through them in order; for each chunk, a small trained network reads the chunk plus what is already remembered and picks a handful of routines, and a language model applies all of them in one call to write memory entries.
3. When the history is finished, answer that history's own questions from the memory just built; the score becomes the training signal for the picker, and every wrong answer goes into a rolling failure list.
4. Every hundred training steps, group the recent failures by topic, take the worst and most repeated from each group, and have a language model diagnose what memory behavior was missing and then rewrite up to three routines or add new ones. Nudge the picker to try the new routines for the next fifty steps.
5. If a round's average score does not beat the best so far, restore the previous best routine list; if several rounds in a row fail to beat it, stop and keep the best.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched OpenReview for the exact title "MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents" and for the paper under ICML, NeurIPS, and COLM 2026 on 2026-09-15). The paper carries no venue footer — its title page reads "Preprint. May 26, 2026" — so no peer-review record exists to summarize. A third-party paper-aggregator page lists a venue string for this arXiv identifier, but it links no reviews and conflicts with the paper's own footer, so it is not recorded here.
