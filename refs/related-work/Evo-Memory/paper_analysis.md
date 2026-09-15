# Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory

- **Original title**: Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory
- **Authors**: Tianxin Wei (corresponding, twei10@illinois.edu), Noveen Sachdeva, Benjamin Coleman, Zhankui He, Yuanchen Bei, Xuying Ning, Mengting Ai, Yunzhe Li, Jingrui He, Ed H. Chi, Chi Wang, Shuo Chen, Fernando Pereira, Wang-Cheng Kang, Derek Zhiyuan Cheng
- **Affiliations**: University of Illinois Urbana-Champaign (1); Google DeepMind (2). The dagger note "Work done while at Google DeepMind" is attached to Tianxin Wei and Yunzhe Li. The title page carries the Google DeepMind report banner, the date line 2026-5-19, and "© 2026 Google DeepMind".
- **Venue / year**: arXiv preprint (arXiv:2511.20857v2 [cs.CL, cs.AI], v1 submitted 25 November 2025, v2 18 May 2026; 27 pages including appendices A-I). No venue is named anywhere in the paper; Appendix E.2 says the code "will be released under the permissive open-source license ... upon acceptance", so the work is under submission somewhere with an artifact-documentation checklist (Appendix E) of the kind used by the Association for Computational Linguistics venues.
- **Source URL**: https://arxiv.org/abs/2511.20857
- **OpenReview**: no public review page found (see section 7). The identifier `iaAX2TFzRP` that a web search surfaces is an automatically imported bibliographic record (venue field "CoRR 2025", invitation `DBLP.org/-/Record`), not a submission with reviews.
- **Code**: no release from the authors (see section 5e). A third-party reimplementation exists at https://github.com/zhaosnw/evo_mem (MIT, created 2025-12-16, 21 stars, unaffiliated account, contains the paper as a committed PDF and an agent-written branch); it is not the authors' code and is not cloned into this folder.

Names used below are the paper's own: Evo-Memory (the benchmark and framework), ReMem (the proposed agent), ExpRAG and ExpRecent (the proposed simple baselines), and the method names of the systems it re-implements (ReAct, Amem, SelfRAG, MemOS, Mem0, LangMem, Dynamic Cheatsheet in its curated and retrieval variants DC-Cu and DC-RS, Agent Workflow Memory abbreviated AWM by the paper). "History" is the paper's label for putting the full prior interaction history in context; "Baseline" is its label for the same model with no memory at all.

**Classification: empirical.** The dominant contribution is an instrument and the measurements taken with it — over ten memory systems re-implemented behind one interface, run as task streams over eleven dataset columns and five backbones, with the findings read off the resulting grid. ReMem is a method contribution and is folded into section 2 below (study design, since it is one of the conditions) rather than given its own framework, because the paper's own title, its three contribution bullets, and its four research questions are all organized around the benchmark, and because ReMem is described in one page with no training, no learned component, and no ablation of its own parts.

---

## 1. Research Motivation & Questions

### a) Why was this study conducted?

The gap is between what memory systems are sold as and what memory benchmarks measure. The paper's distinction, drawn in Figure 1(a) and stated in Sec. 1, is between *conversational recall* — retrieving a past fact, such as the solution to one quadratic equation — and *experience reuse* — retrieving a strategy, such as the quadratic formula, and applying it to a new problem. Existing memory evaluations measure the first: "agents remember what was said but not what was learned" (Sec. 1). Nothing in the standard protocol tests whether an agent that has solved fifty tasks solves the fifty-first better, because the standard protocol gives each task a fresh memory.

The authors' word for the missing capability is *test-time evolution*: "where LLMs retrieve, integrate, and update memory continuously during deployment" (abstract). Their claim is that no unified framework exists for evaluating it, so every memory system is reported on its own benchmark under its own protocol, and the field cannot say which design choice — retrieval policy, update policy, representation — is doing the work.

### b) What is missing in current understanding?

Three named blind spots (Sec. 1, Sec. 2.2):

1. **Prior streaming benchmarks measure the wrong thing or too little of it.** StreamBench "evaluates sequential learning but mainly measures factual retention without reasoning or trajectory reuse"; LifelongBench "focuses on retention without modeling memory structure or updates"; other long-term-memory studies "assess long-term conversational consistency but do not test how agents evolve their memory during deployment" (Sec. 1).
2. **No unified interface across memory designs.** Memory systems span passive buffers, read-write controllers, policy-driven stores, and structured representations (Sec. 2.2), and each ships with its own harness, so published comparisons confound the memory design with everything around it.
3. **Experience reuse is untested as a distinct capability.** The paper asserts that without it "models repeatedly solve similar problems, as long-term assistants often recall context yet fail to adapt across sessions" (Sec. 1), and that this is precisely what current evaluations cannot see.

### c) Research questions

Four, stated verbatim in Sec. 4:

- RQ1: "How do LLM agents perform on Evo-Memory across domains and task types, and does ReMem enhance their test-time learning ability?"
- RQ2: "What factors influence the effectiveness of memory in different tasks, and how does experience reuse improve task efficiency?"
- RQ3: "How does task sequence difficulty (e.g., easy vs. hard trajectories) affect memory adaptation and generalization?"
- RQ4: "How do varying feedback types impact learning dynamics and memory refinement?"

### d) Type of study

Benchmarking, with a proposed method evaluated inside it. The instrument is a re-organization of existing datasets into ordered task streams plus a single loop contract that every memory system is re-implemented against; the findings are the marginals of a methods × backbones × datasets grid, plus two designed manipulations (task-order difficulty, Sec. 4.2.3; whether failed experiences are stored, Sec. 4.2.4).

---

## 2. Study Design

### a) The protocol, which is the contribution

Evo-Memory formalizes a memory-augmented agent as a tuple $(F, U, R, C)$: base model $F$, memory update pipeline $U$, retrieval module $R$, context construction $C$ (Sec. 3.1). A dataset is converted into a stream $\tau = \{(x_1, y_1), \ldots, (x_T, y_T)\}$. At step $t$ the agent holds memory $M_t$, retrieves $R_t = R(M_t, x_t)$, builds a prompt $C_t = C(x_t, R(M_t, x_t))$, answers $\hat{y}_t = F(C_t)$, forms an experience entry $m_t = h(x_t, \hat{y}_t, f_t)$ from the answer and the feedback $f_t$, and updates

$$M_{t+1} = U(M_t, m_t),$$

producing the trajectory $(x_1, \hat{y}_1, M_1) \to (x_2, \hat{y}_2, M_2) \to \cdots \to (x_T, \hat{y}_T, M_T)$. Sec. 4.1.1 writes the same loop compactly as $(x_t, M_t) \xrightarrow{\text{search}} R_t \xrightarrow{\text{synthesis}} \hat{y}_t \xrightarrow{\text{evolve}} M_{t+1}$ and fixes the signal: "Feedback $f_t$ is considered the correctness signal."

Two properties of this protocol matter for anyone reusing it. First, **the task is answered with the memory as it stood before the task arrived** — $\hat{y}_t$ depends on $M_t$, and $M_{t+1}$ is formed afterwards. This is score-before-update, implemented over nine datasets. Second, **the only quantity that enters the update is the current task's own correctness**: $f_t$ is the current task's feedback, $m_t$ is built from the current task's input, output, and feedback, and no later task's outcome is ever routed back to an earlier update.

### b) Variables and conditions

| Factor | Levels |
|---|---|
| Memory method | Baseline (no memory), History, ReAct, Amem, SelfRAG, MemOS, Mem0, LangMem, DC-Cu, DC-RS, AWM, ExpRecent, ExpRAG, ReMem — 14 in single-turn; MemOS and LangMem are dropped in multi-turn as "incompatible with embodied environments" (Sec. 4.1.2), leaving 12 |
| Backbone | Gemini 2.5 Flash, Gemini 2.5 Flash-Lite, Gemini 2.5 Pro, Claude 3.5 Haiku, Claude 3.7 Sonnet |
| Dataset | Single-turn: AIME-24, AIME-25, GPQA-Diamond, MMLU-Pro (Economics / Engineering / Philosophy), ToolBench. Multi-turn: AlfWorld, BabyAI, PDDL, ScienceWorld |
| Task order (Sec. 4.2.3) | Easy→Hard, Hard→Easy |
| What is stored (Sec. 4.2.4) | Successful experiences only (the default of Table 1) versus successful and failed |

**Held fixed** (Appendix A.2): one retriever for all methods (BAAI/bge-base-en-v1.5), the same retrieval budget (top-$k$, default $k = 4$), the same prompt-length truncation, and "a unified task sequence ordering within each dataset, ensuring consistent memory evolution dynamics for all models". Methods that decide *whether* and *what* to retrieve (SelfRAG, ReMem) do so on top of the same retrieval pool.

**Dependent variables** (Appendix A.3): answer accuracy (exact match for single-turn), success rate and progress rate (multi-turn, reported as the paired columns S and P), step efficiency (Figure 4), and "sequence robustness ... whether the LLM maintains consistent knowledge and performance across varying task orders" — operationalized as Table 2's two orderings, not as re-testing any earlier task.

### c) Subjects of study

Dataset sizes (Appendix F): MMLU-Pro Engineering 969, Economics 844, Philosophy 499; GPQA-Diamond 198; AIME-24 and AIME-25 30 each; ToolBench 750; AlfWorld 134; BabyAI 112; ScienceWorld 90; PDDL 60. Streams are therefore short by the standards of a deployment claim — the longest multi-turn stream is 134 tasks — and the MMLU-Pro and ToolBench streams (499 to 969 tasks) are the only ones long enough to show slow drift.

### d) The two proposed methods, as conditions

**ExpRAG** (Sec. 3.2) is the paper's deliberately simple reference point: each finished task becomes one structured experience text $m_i = S(x_i, \hat{y}_i, f_i)$; retrieval is $R_t = \text{Top-}k_{m_i \in M_t}\ \phi(x_t, m_i)$ for a similarity score $\phi$; the model answers in context; the update is a plain append, $M_{t+1} = M_t \cup \{(x_t, \hat{y}_t, f_t)\}$. Nothing is merged, rewritten, or removed. **ExpRecent** is the same with recency instead of similarity as the retrieval rule.

**ReMem** (Sec. 3.3) adds one action. At each step the agent picks $a_t^n \in \{\text{Think}, \text{Act}, \text{Refine}\}$ and transitions by $o_t^n = \text{Agent}(x_t, M_t, a_t^n)$, with state $s_t^n = (x_t, M_t, o_t^{1:n-1})$; Think emits reasoning, Refine performs "meta-reasoning over memory ... exploiting useful experiences, pruning noise, and reorganizing $M_t$", and the step ends when Act is chosen. The multi-turn prompt in Appendix D makes Refine concrete: the model may answer `Think-Prune: <IDs>` to "remove unhelpful experiences from 'RELEVANT EXPERIENCE' section", where each retrieved experience is printed as goal, trajectory, and a `Correctness: [success/failure]` line. The paper describes the loop as inducing a Markov decision process (Sec. 3.3), but this is a description of the within-task decision loop only: there is no reward function, no discount, no policy learning, and no training of any kind anywhere in the paper.

### e) Statistical approach

None. Appendix G.2 states only that "reported results are averaged across multiple task instances and trajectories". No seed count, no variance, no confidence interval, and no significance test appears in the paper. Total API spend is given as "on the order of tens of thousands of US dollars" (Appendix G.1); the utilities are PyTorch 2.7.1 and Weights & Biases (Appendix G.3).

---

## 3. Findings & Observations

### a) Headline findings

**1. Memory helps far more in multi-turn embodied streams than in single-turn reasoning streams.**
Multi-turn (Table 1b, Table 4), averaged success / progress: on Claude 3.7 Sonnet, Baseline 0.24/0.52 → ReMem 0.78/0.91; on Claude 3.5 Haiku, 0.18/0.39 → 0.51/0.69; on Gemini 2.5 Flash, 0.27/0.46 → 0.50/0.64; on Gemini 2.5 Pro, 0.21/0.45 → 0.50/0.65. Single-turn (Table 1a, Table 5), average score: Gemini 2.5 Flash 0.59 → 0.65, Gemini 2.5 Flash-Lite 0.58 → 0.61, Claude 3.7 Sonnet 0.54 → 0.58, Claude 3.5 0.38 → 0.41. The paper's own reading: "Performance gains are notably larger in multi-turn settings, underscoring that continual adaptation becomes increasingly valuable as task horizons lengthen" (Sec. 4.2.1).

**2. A plain retrieve-and-append baseline is competitive with every published memory architecture, and beats ReMem in single-turn.** On Claude 3.7 Sonnet single-turn, ExpRAG averages 0.59 against ReMem 0.58, with every re-implemented system between 0.48 (AWM) and 0.56 (Amem) (Table 1a). On Gemini 2.5 Flash-Lite, ExpRAG and ReMem tie at 0.61 (Table 5). The paper states it plainly: "ExpRAG serves as a simple yet highly effective baseline, outperforming several more complex designs" (Sec. 4.2.1).

**3. Accumulating raw interaction history is actively harmful, sometimes catastrophically.** The History condition on Gemini 2.5 Flash scores 0.31/0.26 on ToolBench against the no-memory Baseline's 0.71/0.61 — a collapse of 40 and 35 points from doing nothing but keeping the transcript (Table 1a, repeated in Table 5). On Gemini 2.5 Flash-Lite, History averages 0.49 against Baseline 0.58 across all seven single-turn columns. ReAct on Gemini 2.5 Flash scores 0.05 on GPQA against a 0.48 baseline and averages 0.37 against 0.59 (Table 1a).

**4. The size of the gain tracks how similar the stream's tasks are to each other.** Figure 3 regresses ReMem's improvement over History against within-dataset task similarity (mean cosine distance from each task embedding to its dataset centroid, computed with the retriever encoder): Pearson $r = 0.717$ on Gemini 2.5 Flash and $r = 0.563$ on Claude 3.7 Sonnet. PDDL and AlfWorld sit at the high-similarity, high-gain end; AIME-25 and GPQA at the low end, with AIME-25 on Claude 3.7 Sonnet showing a memory improvement of roughly −40% (Figure 3, left panel).

**5. Memory buys steps as well as accuracy.** Average steps to complete a task (Figure 4): AlfWorld 22.6 (History) → 17.5 (ExpRecent) → 14.3 (ExpRAG) → 11.5 (ReMem); BabyAI 18.8 → 14.0; PDDL 20.5 → 17.5; ScienceWorld 24.9 → 17.8.

**6. Storing failures alongside successes degrades the simple methods sharply and ReMem mildly.** Comparing Table 3 (successes and failures stored) against Table 1b (the default): on Gemini 2.5 Flash AlfWorld, ExpRAG falls 0.59/0.79 → 0.25/0.60, ExpRecent 0.37/0.65 → 0.22/0.57, and ReMem 0.66/0.81 → 0.57/0.76. The paper's reading: "naive memory accumulation introduces noise and hinders retrieval", while ReMem "remain[s] robust by actively refining stored experiences" (Sec. 4.2.4).

**7. Task order matters less for the evolving methods than for the baseline.** Table 2 (the Base row reproduces the Claude 3.7 Sonnet History row of Table 1b exactly, though the caption names no backbone): Base averages 0.41/0.74; ReMem reaches 0.77/0.92 on Easy→Hard and 0.81/0.94 on Hard→Easy, ExpRAG 0.57/0.79 and 0.69/0.87, ExpRecent 0.57/0.83 and 0.60/0.83. Every method does better on Hard→Easy; the paper reads this as "successful experiences from harder tasks are more transferable" (Sec. 4.2.3).

### b) Patterns and trends

The ordering of method families is stable across backbones in multi-turn and unstable in single-turn. In Tables 1b and 4, ReMem has the best average success rate on all four backbones and ExpRAG or ExpRecent is second; the published architectures (SelfRAG, Mem0, DC-Cu, DC-RS, AWM) cluster within a few points of each other and of the History baseline on every backbone. In Tables 1a and 5 the ordering is a scramble: LangMem wins on Claude 3.5 (0.43), ExpRAG on Claude 3.7 Sonnet (0.59), ReMem on Gemini 2.5 Flash (0.65), a tie on Flash-Lite.

Memory pruning is selective in a way that tracks dataset breadth (Figure 5): fraction of memory entries pruned is 36.8% on GPQA, 32.2% on MMLU-Pro Philosophy, 29.2% on MMLU-Pro Engineering, 23.8% on ToolBench, 20.0% on MMLU-Pro Economics, 17.5% on AIME-24, 10.8% on AIME-25. The paper's interpretation is that heterogeneous streams produce more entries that later look irrelevant (Appendix B.2).

### c) Surprises

The strongest backbone is not the strongest without memory. Gemini 2.5 Pro scores 0.04/0.39 on AlfWorld with no memory, below Gemini 2.5 Flash (0.12/0.34 success) and far below Claude 3.7 Sonnet (0.18/0.49), and its overall no-memory multi-turn average (0.21/0.45) is the second-worst of the four backbones (Table 4). Memory closes most of that gap (0.50/0.65), which means a large part of what the memory condition supplies is format and procedure rather than transferred experience.

Memory is not free even where it is supposed to help: on Gemini 2.5 Flash's BabyAI column the no-memory Baseline (0.61/0.71) is the best entry in the table and beats ReMem (0.53/0.61) and every other method (Table 1b) — one of four multi-turn environments where the entire apparatus loses to doing nothing.

### d) Negative and null results

On AIME-24 and AIME-25 with Claude 3.7 Sonnet, ReMem scores 0.13/0.13 against a no-memory baseline of 0.17/0.13 and against Amem's 0.27/0.17 (Table 1a) — the proposed method is below the no-memory baseline on the two hardest single-turn sets. Figure 7 shows GPQA and MMLU-Pro Engineering curves for ReMem and History converging to the same value by the end of the stream on Gemini 2.5 Flash-Lite, i.e. no benefit at all after several hundred tasks. AWM collapses on AIME (0.03/0.03 on both Claude 3.7 Sonnet and Gemini 2.5 Flash-Lite, Tables 1a and 5). The paper reports these numbers in its tables but does not discuss them.

### e) Robustness checks

Two designed manipulations (task-order direction, Sec. 4.2.3; stored-failure condition, Sec. 4.2.4) and one breadth check (two extra backbones in Appendix B.1, Tables 4 and 5). There are no seed repetitions, no prompt-variation checks, and no ablation of ReMem's own components — Think, Act, and Refine are never separated, so the paper cannot attribute ReMem's multi-turn gain over ExpRAG to the Refine action specifically.

---

## 4. Analysis & Interpretation

### a) Authors' explanations

*Demonstrated*: that gains correlate with within-dataset task similarity (Figure 3, with the correlation coefficient computed), and that unfiltered failure storage hurts retrieval-only methods more than it hurts ReMem (Table 3 against Table 1b). *Asserted with supporting numbers but not isolated*: that the benefit comes from "continual reflection and refinement ... [improving] procedural knowledge accumulation" (Sec. 4.2.1) — no ablation separates reflection from the extra tokens, the extra steps, or the format scaffolding that the memory prompt supplies. *Speculated*: that "successful experiences from harder tasks are more transferable" as the explanation for the Hard→Easy advantage (Sec. 4.2.3); the design cannot distinguish this from the simpler account that the easy tail of a stream is easier.

### b) Evidence quality

The controlled comparison is genuinely tight for its scope: one retriever, one retrieval budget, one fixed task order per dataset, one loop contract, methods re-implemented behind it rather than run in their own harnesses. Within that, the claim "memory helps more in multi-turn than single-turn" is licensed by the design. The claim "ReMem is the best memory design" is weaker than it looks: single-run numbers with no variance estimate, differences of 1-3 points in single-turn, and a method that is simultaneously the only condition allowed to emit an extra action type.

### c) Confounds and alternative explanations

1. **Token budget and step budget are not controlled.** ReMem may Think and Refine repeatedly before Act; the baselines answer once. Nothing in Appendix A.2 equalizes calls or tokens across conditions, so part of every ReMem gain is an inference-compute gain.
2. **The no-memory baseline's prompt is not the memory prompt with an empty memory.** The Appendix D templates for memory conditions carry environment instructions, static few-shot demonstrations, a help line, and an output-format contract. How much of the multi-turn jump (0.24 → 0.78 on Claude 3.7 Sonnet) is scaffolding rather than accumulated experience is not separable from the reported tables.
3. **Re-implemented baselines are the authors' versions.** MemOS, Mem0, LangMem, Dynamic Cheatsheet, and AWM are all re-implemented behind the unified interface, which is what makes the comparison clean and also means their scores are evidence about this re-implementation.
4. **One fixed order per dataset.** Sec. A.2 fixes the order for comparability; the price is that every per-dataset number is one draw from the space of orderings, and Sec. 4.2.3 itself shows the ordering is worth tens of points.

### d) Generalizability

Credible for API-served frontier models on text-based single-turn reasoning and text-based embodied simulators, at stream lengths of 60 to about 1,000 tasks. Extrapolation is risky in two directions the paper does not flag: to streams that mix domains (every stream here is within one dataset, so the entire study measures near-transfer and Figure 3 is the evidence that near-transfer is where the gains live), and to streams long enough for the store to saturate (nothing reports memory size, retrieval latency, or entry count over time).

---

## 5. Implications, Limitations and Transferability

### a) For practitioners

Four rules the tables support: put memory where the tasks repeat structure (PDDL, AlfWorld) and not where they do not (AIME, GPQA); do not put raw interaction history in the context, which is worse than no memory on two of the four single-turn backbones and catastrophic on ToolBench; filter what is written by whether the task succeeded, because unfiltered failure storage costs ExpRAG 34 points of AlfWorld success rate; and try retrieve-and-append before adopting any published memory architecture, since none of the five re-implemented ones beats it on average.

### b) For researchers

The protocol — one loop contract, one retriever, one budget, many memory systems behind the same interface — is the reusable piece, and it is the first to score memory systems on an ordered stream with the memory as it stood before each task. The open questions it exposes and does not ask: what happens when the stream spans domains, what happens when the store saturates, and whether anything that was learned early is still available late.

### c) Acknowledged limitations (Appendix I)

Three, stated briefly: budget and API limits restrict the study to a selected set of strong proprietary models rather than a broad sweep including open-weight or multilingual models; the benchmark is textual and goal-oriented, with multimodal and real-world environments left to future work; and Appendix C adds that memory management depends on the reliability of the model's own judgements and that memory stores are vulnerable to poisoning by adversarial or misleading interactions.

### d) Unacknowledged limitations

1. **No retention measurement.** Every task in every stream is seen exactly once and never re-tested. Nothing in the paper can say whether the memory that raises the average has cost the agent a capability it had at task 10. "Sequence robustness" (Appendix A.3) sounds like the missing quantity and is not: it is measured by re-running a *different* order, not by re-checking an earlier task.
2. **No variance, no seeds, no significance.** Differences of 1-3 points in the single-turn tables carry the claim that ReMem is the best single-turn method, with nothing to compare them against.
3. **No compute or token accounting per condition**, despite ReMem being the condition with the extra action and Figure 4 showing it uses the fewest environment steps — the step saving is reported, the token cost is not.
4. **No ablation of the proposed method.** Think, Act, and Refine are never separated, so the paper's central mechanism claim rests on comparing ReMem to ExpRAG, which differs in several ways at once.
5. **Memory size is never reported.** Pruning rates appear (Figure 5) but not the store's growth, so the reader cannot tell whether pruning holds the store flat or merely slows it.
6. **A text-to-table mismatch in the headline sentence.** Sec. 4.2.1 says the methods reach "0.92/0.96 on BabyAI and 0.95/0.62 on ScienceWorld"; in Table 1b, 0.92/0.96 is the Claude 3.7 Sonnet ReMem AlfWorld cell, BabyAI is 0.73/0.83, 0.95 is the PDDL progress rate, and 0.62 is the ScienceWorld success rate. The tables are internally consistent (Table 1b matches Table 4); the prose is not.
7. **Figure 6 and Figure 7 are cumulative running averages.** The caption of Figure 6 says so explicitly — "shown as rolling averages over fixed task sequences (not learning curves)" — which means a falling curve conflates a harder tail of the stream with an agent getting worse, and the paper draws no conclusion from the shapes beyond the two short paragraphs of Appendix B.3.

### e) Reproducibility

Poor at present. No code, no configuration files, no run outputs, no evolved memory stores, and no per-task metric files are released; Appendix E.2 promises release "upon acceptance", and neither the arXiv page, the abstract, nor the body carries a repository link. The first author's public GitHub account (`weitianxin`, UIUC, matching the paper) holds no Evo-Memory repository, and no repository exists under the Google DeepMind or Google Research organizations. The unaffiliated `zhaosnw/evo_mem` repository that search engines surface for the title was created three weeks after the v1 preprint, carries the paper as a committed PDF, has a branch named for an agent coding session, and names no authors: it is a third-party reimplementation from the PDF, and any claim about it would be a claim about that reimplementation and not about this paper. Reproducing the study means rebuilding the loop contract, re-implementing ten memory systems against it, and spending the reported order of tens of thousands of dollars of API budget.

### f) Transfer to our own work

**Score-before-update stream protocol: present, clean, and the best-specified one we have for the long-term-memory layer.** Sec. 3.1 defines the loop so that $\hat{y}_t$ is produced from $M_t$ and $M_{t+1}$ is formed only afterwards, and Appendix A.2 fixes one task order per dataset so that every method walks the same stream. This is our score-before-update protocol applied to one harness layer, instantiated over eleven dataset columns and five backbones. What we would have to add is heterogeneity: every Evo-Memory stream stays inside one dataset, and Figure 3 is direct evidence that the method's whole advantage is a function of how self-similar the stream is — the correlation of 0.717 between task similarity and gain is, read against our setting, a prediction that these methods degrade as the stream becomes the heterogeneous, non-stationary one our claim concerns.

**The horizon $\gamma = 0$ case, on the memory layer, stated in the paper's own notation.** The update is $M_{t+1} = U(M_t, m_t)$ with $m_t = h(x_t, \hat{y}_t, f_t)$ and "feedback $f_t$ is considered the correctness signal" (Sec. 4.1.1). The only score that ever touches an update is the score of the task that produced it. No later task's outcome revises an earlier entry, no return is accumulated, and the discount on future batch performance — the horizon $\gamma$ of `docs/framing.md` — is zero by construction, not by omission. This is the cleanest single equation in our index for the claim that existing self-improving harnesses score an edit only on the batch that produced it.

**Retention or anchor set: absent, and the absence is structural.** No task in any stream is re-evaluated. The nearest thing, "sequence robustness", is order variation across separate runs. So Evo-Memory reproduces the exact hole we found in AgentStream: a benchmark built to measure improvement over a stream that cannot measure what the improvement cost.

**Gate and revert: two partial mechanisms, neither a gate in our sense.** First, a write filter — the default configuration stores successful experiences only, and Sec. 4.2.4 exists precisely to measure what happens when failures are admitted too (ExpRAG loses 34 points of AlfWorld success rate on Gemini 2.5 Flash). That filter is a one-bit gate keyed on the current task's own outcome, the strictest possible $\gamma = 0$ gate. Second, a removal path — ReMem's Refine action prunes entries (`Think-Prune: <IDs>` in the Appendix D template), pruning 10.8% to 36.8% of entries depending on dataset (Figure 5). Pruning is driven by the model's judgement of relevance at retrieval time, never by a measured score, and there is no rollback: no earlier memory state is ever restored and no edit is ever undone after being shown to hurt.

**Loss of plasticity: suggestive evidence, not a measurement.** Three pieces. The History condition, which is pure undiscounted accumulation, is worse than no memory at all on two of the four single-turn backbones and loses 40 points on ToolBench with Gemini 2.5 Flash (Table 1a) — accumulation without editing destroys capability. The stored-failure experiment shows the store's contents degrading the agent that built them (Table 3). And Figure 6's History curves fall over their streams while the ReMem curves on the same fixed orders do not, which is the peak-then-decline shape our claim predicts — with the caveat the authors themselves attach, that these are rolling averages over a fixed order and not learning curves, so the shape alone does not establish degradation. Evo-Memory gives us the strongest circumstantial evidence yet that the memory layer is where greedy accumulation breaks first, and no instrument that would settle it.

**Released artifacts for pretraining a critic: nothing.** No trajectories, no evolved memory stores, no per-task logs, no code. The 10-dataset stream construction would have to be rebuilt from the paper.

---

## 6. Summary

### a) One-sentence headline finding

Scored on ordered task streams with the memory as it stood before each task, simple retrieve-and-append beats every published memory architecture, and adding one memory-pruning action on top of it lifts multi-turn success rates by a factor of two to three.

### b) Quick-reference takeaways

- Multi-turn embodied streams are where memory pays: averaged success rate rises from 0.24 to 0.78 on Claude 3.7 Sonnet and 0.27 to 0.50 on Gemini 2.5 Flash; single-turn reasoning streams gain 3-6 points at most and lose on the hardest sets.
- A retrieve-top-4-and-append baseline (ExpRAG) matches or beats five published memory systems on every backbone, and beats the paper's own proposed method in single-turn on Claude 3.7 Sonnet (0.59 against 0.58).
- Keeping raw interaction history is worse than having no memory at all on two of the four single-turn backbones, including a 40-point collapse on ToolBench with Gemini 2.5 Flash.
- The gain from memory is a function of how similar the stream's tasks are to each other: Pearson correlation 0.717 on Gemini 2.5 Flash, 0.563 on Claude 3.7 Sonnet, with negative gains on the least self-similar dataset.
- Writing failures into the store as well as successes costs the retrieval baseline 34 points of AlfWorld success rate, which makes the default success-only write filter a load-bearing and under-discussed design choice.
- No task in any stream is ever re-tested, so the study measures the gain from accumulated memory and nothing about what the accumulation cost.

### c) Bottom line for decision-making

Trust the protocol and the ranking of memory designs within one domain; treat every ReMem-versus-baseline gap under 5 points as unresolved, since there are no seeds, no variance, and no token-budget control; and take nothing from it about durability, because no earlier task is ever revisited.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched the OpenReview note-search APIs, both versions, for the full title and for "Evo-Memory", and web-searched the title with "openreview", on 2026-09-15). The one identifier that matches — `iaAX2TFzRP` — is an automatically imported bibliographic record carrying the venue string "CoRR 2025" under the `DBLP.org/-/Record` invitation, with no reviews, no ratings, and no decision attached; the "Revision History" page a search engine surfaces for it is the history of that bibliographic record. The paper names no venue, and Appendix E.2's "upon acceptance" wording indicates it was under submission at the time of the v2 revision. No review content is reported here.
