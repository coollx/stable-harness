# Agent Harness Engineering: A Survey

- **Original title**: Agent Harness Engineering: A Survey
- **Authors**: Junjie Li, Xi Xiao, Yunbei Zhang, Chen Liu (co-first authors), Lin Zhao, Xiaoying Liao, Yingrui Ji, Janet Wang, Yingqiang Ge, Weijie Xu, Xi Fang, Xiang Xu, Tianchen Zhao, Youngeun Kim, Jihun Hamm, Tianyang Wang (corresponding), Chandan K. Reddy (corresponding)
- **Affiliations**: Carnegie Mellon University, University of Alabama at Birmingham, Tulane University, Yale University, Northeastern University, Stanford University, Amazon, University of Chicago, Virginia Tech, Rutgers University, independent researchers
- **Venue / year**: OpenReview submission `eONq7FdiHa`, 2026. The OpenReview copy is indexed under the header "Under review as submission to TMLR" (Transactions on Machine Learning Research). Project page lists the online date as 2026-05-16; the PDF analysed here was compiled 2026-07-20 and runs 71 pages (51 pages of body, references, appendices A and B).
- **Source URL**: https://picrew.github.io/LLM-Harness/ (PDF: `main.pdf` on that page)
- **OpenReview**: https://openreview.net/forum?id=eONq7FdiHa (page is behind a challenge check; see section 7)
- **Code / data**: living catalog https://github.com/Picrew/awesome-agent-harness (cloned to `repo/`, commit `91c5381`, 2026-08-31; see `repo_analysis.md`); Hugging Face dataset https://huggingface.co/datasets/ChenLiu1996/Agent-Harness-Engineering (a PDF mirror, no tabular data)

Classification: empirical (survey with a systematic mapping study). The contribution is a taxonomy plus an observational characterisation of an ecosystem of open-source projects, not an artifact to reproduce. The paper's own acronym for its taxonomy, ETCLOVG (Execution, Tooling, Context, Lifecycle, Observability, Verification, Governance), is used below as a quoted label.

---

## 1. Research Motivation and Questions

### a) Why was this study conducted?

The paper starts from a body of evidence that the infrastructure around a language model moves measured agent performance while the weights stay fixed. It cites three items (Sec. 1): Bölük (2026) changed only the edit-tool format and tool harness across 15 models and reports gains on coding benchmarks "including an improvement of up to 10x for one model"; Trivedy (2026) raised a fixed GPT-5.2-Codex agent from 52.8% to 66.5% on Terminal-Bench 2.0 by restructuring the system prompt, injecting context through middleware, and adding self-verification hooks; Meta-Harness (Lee et al., 2026) shows parts of the harness can be optimised automatically. The authors immediately bound the claim: "The strongest controlled evidence currently comes from coding-agent benchmarks, and these results do not establish that the harness matters more than the model in every setting."

Against that evidence the research vocabulary is fragmented. Academic work studies planning, memory, retrieval, tool use, verification, coordination, and safety as separate capabilities; production write-ups describe context pipelines, tool contracts, execution policies, traces, retries, handoffs, permission boundaries, and runtime safeguards. Three questions are hard to answer across these bodies of work (Sec. 1): where the harness begins and ends, how its interacting mechanisms should be organised, and which parts of the design space existing systems cover.

### b) What is missing in current understanding?

Two specific gaps. First, no shared systems-level account exists, so "closely related engineering problems appear under different names, while choices that interact during execution are analyzed in isolation" (Sec. 1). Second, the nearest prior framework, Meng et al. (2026), describes an individual harness through six components (execution loop, tool registry, context manager, state store, lifecycle hooks, evaluation interface). The authors position that as a component-level view of one harness and offer instead a layer-level view of the ecosystem, which separates observability and governance as distinct engineering surfaces (Sec. 1, Sec. 2.3).

### c) Research questions / hypotheses

The paper is organised around three claims rather than numbered hypotheses (Sec. 1, restated at each chapter opening as "Claim 1/2/3"):

- Claim 1 (the "binding-constraint thesis"): the harness is an independent system layer whose design jointly determines agent reliability with the model.
- Claim 2: the seven-layer taxonomy separates production concerns along seven engineering surfaces.
- Claim 3: mapping open-source projects onto that taxonomy reveals adoption patterns, coverage gaps, and design principles.

### d) Type of study

A survey combined with a systematic mapping study: candidate artifacts are collected through documented source streams, filtered by inclusion criteria, and coded onto a taxonomy (Sec. 2.5 to 2.9, Appendix A). The reporting discipline is explicitly borrowed from systematic reviews (PRISMA 2020, Page et al., 2021). There is no controlled experiment and no new measurement of agent performance; every quantitative performance claim in the paper is a citation.

## 2. Study Design

### a) Variables and conditions

Nothing is manipulated. The unit of analysis is a publicly documented harness artifact. Recorded attributes per artifact (Sec. 2.5): project name, URL, artifact type, source type, availability status, release year when identifiable, GitHub metadata when available, and the public evidence used for coding. The coded outcome is the set of ETCLOVG layers an artifact exhibits, split into one primary layer (the mechanism most central to the artifact) and optional secondary layers (assigned "only when the documentation exposes an independent capability rather than an incidental dependency", Sec. 2.7). Aggregate coverage per layer is the reported dependent quantity.

### b) Subjects of study

The corpus size is stated several ways in the same document:

- "more than 170 open-source projects" (abstract, Sec. 1, Sec. 2.9, Sec. 13);
- "the corpus of over 100 projects" (Sec. 6 opening);
- Appendix A: the catalog snapshot was verified on 2026-05-08 with 171 public entries, 146 GitHub entries, 142 GitHub entries in project categories, and 0 broken URLs;
- Table S1 (Appendix A), which keeps only technical artifacts and drops readings, ecosystem maps, list repositories, and handbooks, lists 20 + 12 + 9 + 47 + 15 + 21 + 14 = 138 rows across the seven layers.

Smaller focused samples appear in the body: Table 2 classifies 18 orchestration systems by pattern and execution model (stars as of 2026-05-12); Table 4 scores 10 governance systems on six feature columns; Appendix B profiles 6 reference harness implementations (Claude Code, OpenCode, Codex CLI, OpenHands, SWE-agent, Symphony). The living catalog in the companion repository had grown to 367 entries in 9 categories by 2026-08-31 (see `repo_analysis.md`).

### c) Measurement and instruments

Layer coding uses the public artifact itself as evidence: README files, documentation, papers, examples, release notes, repository structure (Sec. 2.7). The rule for ambiguous cases is conservative: "if the public evidence did not clearly show an agent-facing mechanism, the layer assignment was withheld." The authors state plainly that the snapshot "uses a single-primary-coder protocol with author audit rather than a formal multi-coder agreement study, so we do not report Cohen's kappa or a comparable inter-annotator statistic." All metrics are paper-specific (layer membership, counts); none are standard.

### d) Protocol

Figure 5 and Sec. 2.5 to 2.7 give the pipeline. Four source streams: prior surveys and benchmark papers; reproducible GitHub searches over names, descriptions, README text, topics, stars, recency, and archival status (queries combined terms such as agent harness, coding agent, LLM agent sandbox, MCP server, agent observability, agent memory, agent evaluation, agent governance); curated project lists and package registries; company engineering blogs and release notes. Candidates are deduplicated, then included when three conditions hold (Sec. 2.6): publicly documented, implementing or specifying a concrete harness-level mechanism, and evidenced enough to assign at least one layer. Excluded: chatbot demos, prompt packs, thin model-client wrappers, static datasets or leaderboards without an agent runtime, generic infrastructure not adapted to agents, and product pages whose behaviour cannot be inspected. Borderline cases are resolved "by mechanism rather than label". Metadata was frozen on 2026-05-08.

### e) Statistical / analytical approach

Counts and qualitative synthesis only. No hypothesis tests, no confidence intervals, no agreement statistics. The two inferential-sounding numbers in the text (a 6 percentage point shift with $p < 0.01$ in Sec. 7.3, and $\kappa = 0.88$ in Sec. 7.4) are quoted from Anthropic (2026d) and Cemri et al. (2026) respectively, not computed here.

## 3. Findings and Observations

### a) Headline findings

**F1. Harness changes alone move benchmark scores by large margins, on coding tasks.**
Claim: agent performance "depends jointly on the model and its execution system" (Sec. 1). Evidence (all cited, none measured here): up to 10x for one of 15 models from edit-tool and harness changes (Bölük, 2026); 52.8% to 66.5% on Terminal-Bench 2.0 for a fixed GPT-5.2-Codex (Trivedy, 2026); LangChain's DeepAgents making the same jump "through changes at the harness" (Sec. 2.1). Conditions: coding and terminal agents; the authors decline to generalise beyond that.

**F2. Coverage across the seven layers is uneven; the structural core is dense and the control plane is thin.**
Claim (Sec. 2.9): "Execution, tool interfaces, lifecycle orchestration, and verification have the densest visible coverage"; "Context and memory appear across many projects but are often embedded inside larger frameworks rather than released as standalone harness components"; "Observability and governance are thinner in open-source coverage and more often appear through commercial platforms, SDK features, or engineering writeups." Evidence: Table S1 primary-layer counts, L 47, V 21, E 20, O 15, G 14, T 12, C 9. Note that the table ranks Tooling (12) below Observability (15), so the prose ranking of the "densest" four layers is not fully borne out by the appendix's own primary-layer counts; the prose statement is about visible coverage across primary and secondary assignments, which the appendix does not break out.

**F3. Complete systems are cross-layer.**
Claim (Sec. 2.9): "the most complete systems combine sandboxing, tool protocols, orchestration, tracing, evaluation, and permission controls, which supports the central claim that harness engineering is an integrated systems problem rather than a collection of isolated add-ons." Evidence: Appendix B, Table S2, where the six reference implementations each cover three to five layers (Claude Code: E, T, C, L, G; OpenHands: E, T, C, L, V; Symphony: C, L, V, G).

**F4. Governance is the sparsest layer in deployed agents.**
Claim (Sec. 9.6, 9.7): "no agent fully implements all defense categories"; "Information flow control, identity management, and formal verification are absent from every surveyed system"; monitoring is partial in all six case-study agents (Codex, Gemini CLI, OpenHands, Browser Use, Nanobrowser, Skyvern). Evidence: Table 4 (10 systems by permission, hooks, hardening, constitution, audit, multi-agent support; most cells partial or absent) and the case studies of Kim et al. (2026), which the paper relays.

**F5. Context drift is a distinct, unsolved failure mode; every current technique slows it and none prevents it.**
Claim (Sec. 5.7): context rot is a single-step property (more tokens degrade retrieval and reasoning); context drift is a trajectory property, where after 100 or more turns agents "repeat work already done, contradict earlier decisions without awareness, and lose track of the motivating goal", and "nothing in the current architecture detects this divergence." Evidence cited: accuracy drops by more than 30% when the relevant document sits mid-context across 20 documents (Liu et al., 2024a); all 18 frontier models degrade with input length, with loss visible at 50K tokens for models rated for 200K (Hong et al., 2025); QSAF (Atta et al., 2025) finds hallucinated outputs stored in persistent memory and reused in later sessions, "creating self-reinforcing degradation loops that cross session boundaries" (Sec. 7.4).

**F6. A benchmark score is a property of the model and harness pair, and the evaluation pipeline is itself noisy.**
Claim (Sec. 8.1): "A harness-aware evaluation should treat the reported score as a property of the model–harness pair, not of the model alone." Evidence relayed: infrastructure configuration alone shifts agentic coding scores by 6 percentage points, $p < 0.01$ (Anthropic, 2026d, Sec. 7.3); single-run pass rates hide substantial variance (Bjarnason et al., 2026, Sec. 12.3); 89% of surveyed teams use observability while 52.4% run offline evaluations (LangChain, 2026c, Sec. 12.3); 38% of task failures in one production system trace to parsing errors that deterministic checks would catch (AgentFixer, Mulian et al., 2026, Sec. 7.4).

**F7. Every harness component encodes an assumption about what the model cannot do, and those assumptions go stale.**
Claim (Sec. 7.4, 7.5, 12.5): "harness complexity should track model capability"; "As model capabilities change, harness interventions should be re-estimated rather than assumed to remain beneficial." Evidence: Anthropic (2026c) removed the sprint construct and context resets when moving from Opus 4.5 to Opus 4.6, "reducing cost from $200 to $125 while maintaining output quality." The paper names the resulting design goal "adaptive simplification" and proposes a "meta-monitoring layer that tracks which interventions (context resets, evaluator feedback loops, tool restrictions) are still load-bearing" (Sec. 7.5).

**F8. Sandboxes are both a cage and a licence, and current ones are partially escapable.**
Claim (Sec. 3.1.2, 3.3): sandboxing serves security, reproducibility, and liveness; the liveness role is new to agents because it converts per-action permission prompts into a session configuration. Evidence: sandboxing reduced Claude Code permission prompts by 84% (Anthropic, 2025c); SandboxEscapeBench reports 15% to 35% escape success against Docker containers depending on configuration (Marchand et al., 2026); IsolateGPT reports under 30% overhead for three quarters of queries; transactional sandboxing reports about 14.5% overhead.

### b) Patterns and trends

- Control-plane layers (O, V, G) mature later than the structural core (E, T, C, L); the authors attribute this to coding, web, terminal, and computer-use agents needing runnable environments, tool contracts, control loops, and repeatable evaluation before anything else (Sec. 2.9).
- Isolation strategy is bifurcating: managed sandboxes move from shared-kernel containers toward microVMs, while OS-level permission sandboxes narrow the host view without a separate environment; "Plain Docker containers occupy a diminishing middle ground" (Sec. 3.2 synthesis).
- A recurring bundle-versus-compose split: framework-integrated runtimes bundle capability at the cost of coupling; sandbox abstraction layers (SWE-ReX, smolagents executors, Kubernetes agent-sandbox) make the substrate replaceable (Sec. 3.2.4, 3.2.7).
- "Fewer but better tools" outperforms brute-force tool exposure because it shrinks prompt footprint and planner branching (Sec. 4.2).
- Three-tier memory (active window, session state, cross-session store) has become the dominant organising frame, mapped onto the operating-system memory hierarchy (Sec. 5.2).
- Ecosystem movement from agent frameworks toward agent platforms that add durable workspaces, identity, billing, observability, evaluation, governance, and human handoff (Sec. 11.4).

### c) Surprises and counterintuitive results

- Removing scaffolding improved cost without hurting quality when the model got stronger (F7). The default assumption that harness design "move[s] monotonically toward more scaffolding" is rejected explicitly (Sec. 12.5).
- Token elasticity: setting a chain-of-thought token budget too low can increase token consumption because the model overflows and generates more than a moderately budgeted prompt would (TALE, Han et al., 2025, Sec. 7.3).
- MCP servers can attack clients before any tool is invoked, through poisoned tool descriptions read at registration time (Trail of Bits, Sec. 9.3).
- Open-source models hallucinate nonexistent package names at 21.7% across 576,000 samples, opening a supply-chain attack the authors call slopsquatting (Spracklen et al., 2025, Sec. 9.3).

### d) Negative and null results

- No current technique solves context drift (Sec. 5.7).
- No widely adopted standard exists for hook interfaces (Sec. 9.2), permission specification languages (Sec. 9.1), constitution schemas (Sec. 9.4), or audit schemas (Sec. 9.5).
- No systematic empirical comparison exists across self-hosted, cloud, and hybrid sandbox deployment modes (Sec. 3.4).
- Standard benchmarks complete within a few dozen steps and do not capture failures that appear at 100 or more turns (Sec. 5.7).
- Interaction effects between stacked governance hooks are "largely unstudied"; an upstream sanitiser may weaken a downstream detector (Sec. 9.2).

### e) Robustness checks

None. One coder, one snapshot date, no sensitivity analysis over inclusion criteria, no agreement study. The corpus counts vary across the abstract, Sec. 6, and Appendix A (Sec. 2b above), and the paper does not reconcile them.

## 4. Analysis and Interpretation

### a) Authors' explanations

Demonstrated (given the coding): the per-layer counts and the cross-layer nature of complete systems. Speculated: the causal story for why observability and governance lag (later commercial maturation), the "harness coupling problem" as the reason agent scores cannot be attributed to the model (Sec. 11.3, which imports the closed-loop framing of Zhang et al., 2026c, a paper by six of this survey's authors), and the "harness-as-assumption principle" (Sec. 7.5), which generalises one vendor anecdote into a design law.

### b) Evidence quality

The mapping is observational and documentation-based; the authors say absence from a layer means "not publicly evidenced" rather than "not implemented" (Sec. 2.8). Every performance number in the paper is borrowed from blog posts or other papers, and several of the most-quoted ones (10x, 52.8% to 66.5%, 84% fewer prompts, $200 to $125) come from vendor engineering posts that were not peer reviewed. Claim 1 therefore rests on secondary evidence; Claims 2 and 3 are the paper's own work and are supported to the extent a single-coder classification can be.

### c) Confounds and alternative explanations

- Visibility bias: English-language, GitHub-visible, open-source, and coding-agent artifacts dominate (acknowledged, Sec. 2.8). Thin observability and governance coverage may reflect that these functions ship inside commercial platforms, exactly as the authors suggest; the data cannot separate "rare" from "closed".
- Coder and taxonomy come from the same team, so the taxonomy can look well-fitted to the corpus by construction; no external coder tested it.
- The companion catalog uses nine practitioner categories, not the seven layers; Appendix A folds categories into layers and the fold is not machine-readable (see `repo_analysis.md`). The repository's verification report for the 2026-05-08 snapshot lists 142 entries outside the readings category, all of them GitHub projects, while Table S1 has 138 rows, so four technical entries were dropped in the fold without explanation. The layer counts therefore depend on an undocumented mapping step.
- GitHub star counts are used as the ordering within tables (Table 2, Table S1) and as a proxy for adoption in the prose (Mem0, everything-claude-code); stars measure attention, not use.

### d) Generalisability

Credible for the open-source coding-agent ecosystem as of May 2026. Extrapolation to closed production systems and non-coding agents is flagged as risky by the authors themselves (Sec. 13). The taxonomy is descriptive; the authors say turning it into a normative design guide "is the natural next step" and not something this paper does.

## 5. Implications, Limitations, and Transferability

### a) For practitioners

- Test harness changes as system changes: "A prompt, tool, memory, sandbox, verifier, or monitor may look beneficial in isolation while degrading the whole rollout when combined with the rest of the control loop" (Sec. 11.3); "local improvements can create global regressions" (Sec. 8.6.1).
- Record the harness configuration (tool registry, context policy, permission policy, budget, timeout, model configuration, runtime interface) as evaluation metadata (Sec. 8.3.2), and repeat rollouts to expose variance rather than hide it (Sec. 8.4.1).
- Treat traces as primary evaluation data and the evaluator as a component under test; layer deterministic checks, judge models, and human audit (Sec. 8.4.2, 8.5.3).
- Re-estimate every intervention when the model changes; remove controls that are no longer load-bearing (Sec. 12.5).
- Keep the prompt prefix stable and context append-only for cache reuse (Sec. 5.3).

### b) For researchers

The open agenda (Sec. 11.5, 12): benchmarks that vary harness interventions rather than only model weights; factorial model-by-harness evaluation; trace-native failure attribution across layers; context management recast as state estimation with bounds on divergence between the agent's internal state and the task state (Sec. 12.2); cross-layer handoff contracts carrying intent, constraints, permissions, provenance, budget, and risk (Sec. 12.4); and methods that let harnesses optimise and simplify their own controls (Sec. 12.5), with the stated risk that "a harness that optimizes itself only against a narrow suite may become brittle."

### c) Acknowledged limitations

Sec. 2.8 and Sec. 13: corpus biased toward English, GitHub, open source, and coding agents; commercial systems under-represented; absence means not evidenced; single primary coder with author audit and no agreement statistic; taxonomy descriptive rather than normative.

### d) Unacknowledged limitations

- Inconsistent corpus counts (170+, over 100, 171, 146, 142, 138) across sections.
- The appendix's primary-layer counts do not match the prose ranking of dense layers (F2).
- Heavy reliance on Anthropic, OpenAI, LangChain, and Manus engineering posts as evidence; the same vendors' products populate the tables.
- Six of the authors co-wrote Zhang et al. (2026c), which supplies the "binding-constraint" and "closed-loop controller" framing that anchors Sec. 5.7, 8.1, 11.3, and 12; this is disclosed only through the citation.
- The affiliation list includes a University of Chicago superscript that no author on the PDF's title page carries, one sign that the author block differs between versions.

### e) Reproducibility

The catalog is reproducible from the companion repository: `data/projects.yaml` is the single source of truth; three scripts sync GitHub metadata, render the bilingual READMEs, and verify schema, link health, and mirror consistency; dated verification reports exist for every sync, including one for the paper's 2026-05-08 snapshot. What is not reproducible from the repository is the ETCLOVG coding itself: entries carry a catalog category and free-form tags but no layer field, so the primary/secondary layer assignments live only in Table S1.

### f) Transfer to this project

The survey is a positioning source and a vocabulary map, not a baseline.

- **Where our work sits in their agenda.** Sec. 12.5 ("Keeping Harnesses Useful as Models Improve") and Sec. 8.6.3 name harness self-optimisation as an open direction and list its current instances: Meta-Harness (search over harness structure), Natural-Language Agent Harnesses (ablatable modules), Lin et al. (2026) (observability-driven evolution of coding-agent harnesses), plus catalog entries such as auto-harness ("benchmark-gated optimization loop that mines failures, edits agent code, and guards against regressions"), SkillOpt, EverOS, and Hermes Agent. Every one is scored on the suite or failures in front of it. In our vocabulary all of them are the $\gamma = 0$ case. The survey's own warning about brittleness from narrow-suite self-optimisation is the closest it comes to our claim, and it frames the danger as benchmark overfitting rather than as lost future adaptability.
- **Two distinct staleness mechanisms.** Their "harness-as-assumption principle" says a component stops paying for itself because the model improved. Our harness plasticity claim says greedy patching forecloses future edits regardless of the model. Both call for re-estimating an intervention's value over time; ours estimates it with a critic before commit, theirs proposes monitoring after the fact. The paper's related-work paragraph on this survey should state that contrast in one sentence.
- **Coupling motivates system-level scoring.** Sec. 11.3 and 8.6.1 argue that a harness edit must be evaluated on the whole rollout because layers interact. That is the justification for scoring candidate edits on the next batch as a system and for the anchor set, and it is the field's stated reason for per-edit regression gates, which is what our falsifier baseline implements.
- **Layer vocabulary.** ETCLOVG is an engineering-surface axis; our layer list is an edit-depth axis. The edits our harness agent proposes land in their C (system prompt, skills, memory), T (tool description and implementation), and L or G (middleware, sub-agent configuration, hooks). Their V and G layers are exactly what we hold fixed (evaluators are out of scope for self-editing). Their scope statement excludes model training, so weights, the deepest layer on our axis, are outside their taxonomy.
- **A mechanism for plasticity loss at the memory layer.** QSAF's observation that hallucinated outputs stored in persistent memory get reused across sessions is a concrete, cited example of an edit to one layer degrading the future stream. It is worth citing when we motivate why the memory layer needs a plasticity estimate.
- **Evaluation hygiene for our stream experiments.** The 6 percentage point infrastructure-noise figure and the randomness result set a floor on the effect size a single-run stream comparison can claim; repeated rollouts and recorded harness configuration per run follow directly from Sec. 8.3 and 8.4.
- **Catalog as a source list.** The repository's 367 entries are a scan list for inspectable open harnesses to run our task streams on (for example mini-swe-agent, OpenHands SDK, smolagents) and for further references to add: Meta-Harness (arXiv:2603.28052), Natural-Language Agent Harnesses (arXiv:2603.25723), Lin et al., Agentic Harness Engineering (arXiv:2604.25850), Zhang et al., Stop Comparing LLM Agents Without Disclosing the Harness (arXiv:2605.23950), and Meng et al. (2026), the six-component harness survey.

## 6. Summary

### a) One-sentence headline finding

The harness is a separable engineering layer; over 171 catalogued artifacts, execution, orchestration, and evaluation are dense while observability and governance are thin, and harness self-optimisation is named an open problem.

### b) Quick-reference takeaway list

- A seven-layer taxonomy (execution, tooling, context, lifecycle, observability, verification, governance) organises the harness ecosystem; the first four are the structural core, the last three the control plane.
- Primary-layer counts in the appendix: lifecycle 47, verification 21, execution 20, observability 15, governance 14, tooling 12, context 9; context tooling is mostly embedded in larger frameworks, governance is the sparsest in deployed agents.
- Cited evidence that harness changes alone move coding benchmarks by double-digit points or more, with fixed weights; the authors restrict the claim to coding agents.
- Harness layers are coupled, so edits must be evaluated on the whole rollout; per-edit regression testing is the recommended practice.
- Harness components encode assumptions about model deficits that go stale; interventions should be re-estimated and removed when no longer load-bearing.
- Context drift over long trajectories is unsolved by compaction, retrieval, or sub-agent isolation.

### c) Bottom line for decision-making

Trust the taxonomy and the coverage picture as a map of the open-source coding-agent ecosystem in May 2026. Do not cite the performance numbers as this paper's findings; cite their primary sources. Use it to position self-improving harness work as the $\gamma = 0$ instances of an agenda the survey itself calls open.

## 7. Reviewer Reception (OpenReview)

An OpenReview record exists (forum `eONq7FdiHa`; the indexed PDF header reads "Under review as submission to TMLR"). On 2026-09-14 the forum page and both API endpoints (`api.openreview.net`, `api2.openreview.net`) returned a challenge-verification page (HTTP 403, `ChallengeRequiredError`) instead of content, so no decision, scores, or review text could be retrieved. The companion catalog's own link checker records the same 403 for the OpenReview PDF. No reviews are reported here; nothing has been inferred from third-party summaries.
