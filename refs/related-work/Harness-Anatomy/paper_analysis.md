# Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents — A Source-Code Study of Eleven Systems

- **Original title**: Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents — A Source-Code Study of Eleven Systems
- **Authors**: Paul Barbaste (lead and corresponding; Inclusive Brains and Wavestone AI Lab), Tristan Darrigol, Germain Vu, Tom Wiltberger (Wavestone AI Lab)
- **Venue / year**: arXiv preprint, 2026 (arXiv:2609.00006v1 [cs.SE, cs.MA], submitted 15 July 2026; 83 pages, 7 figures, 18 tables). Second, expanded edition of an April 2026 study: corpus grown from 8 to 11 systems plus one meta-harness contrast point, with the April snapshots retained for a longitudinal comparison.
- **Source URL**: https://arxiv.org/abs/2609.00006
- **OpenReview**: no public page found (see section 7)
- **Code**: no public artifact (see section 5e). The 90-line scaffold in Listing 3 is printed in the paper and labelled "illustrative scaffold, not production code".
- **AI-assistance disclosure** (paper's Acknowledgments): written "with substantial assistance from Anthropic's Claude (via the Claude Code CLI), which was used both for the source-code analysis and for drafting the manuscript"; findings "verified against the referenced codebases".

Short form used below: "the study". The paper gives itself no abbreviation. Abbreviations used in this document, expanded at first use here: Model Context Protocol (MCP), Agent Client Protocol (ACP), Agent-to-Agent protocol (A2A), retrieval-augmented generation (RAG), lines of code (LoC), just-in-time (JIT), Agentic Harness Engineering (AHE, the system of Lin et al., arXiv:2604.25850), Harness Continual Learning (HCL, arXiv:2608.19013, see `refs/related-work/HCL/`).

---

## 1. Research Motivation & Questions

### a) Why was this study conducted?

"Harness engineering" was named as a discipline in early 2026 (the term entered circulation in February 2026 via LangChain's Deep Agents work, Sec. 2.1) and acquired practitioner guides, formal definitions, and automated-evolution systems within months, but, the authors argue, no reference account "grounded in production source code, of what a harness actually is, what it is made of, and how the field's leading implementations differ" (Sec. 1). The study sets out to be that reference and arrives with a thesis: between 2025 and mid-2026 "the harness has stopped being a tool and become a platform".

### b) What is missing in current understanding?

Three named gaps (Sec. 1, Sec. 3): benchmark surveys report SWE-bench scores without touching architecture; conceptual taxonomies describe patterns without implementations; the one concurrent source-level taxonomy (Rombaut, "Inside the Scaffold", arXiv:2604.03515, 13 open-source scaffolds, 12 dimensions) excludes the provider-native systems (Claude Code, Codex, Gemini CLI, Mistral Vibe) "that define the field's frontier". None reaches "the engineering decisions that determine whether a harness is reliable, extensible, or deployable: how the loop is structured, how tools are defined and sandboxed, how safety is enforced, how context is rationed", nor asks where the category is heading.

### c) Research questions / hypotheses

The study is exploratory and states no numbered hypotheses. Its guiding inquiry: what is a coding-agent harness made of, how do eleven production implementations differ along that anatomy, how did eight of them change over one quarter, and does the accumulated evidence support the claim that the harness has become a platform? The one explicitly labelled hypothesis is the "scaffold–capability frontier" of Sec. 15.5, offered as "a guiding intuition rather than a formal hypothesis" and not tested.

### d) Type of study

Observational, qualitative source-code audit (a comparative multiple-case study of software architecture) with a longitudinal sub-study (same eight systems source-diffed across April to July 2026) and a descriptive landscape sweep. Explicitly not a benchmark: "It does not benchmark or rank; it describes and compares how the systems are built" (abstract). No runtime measurements were made (Sec. 15.6).

---

## 2. Study Design

### a) Variables & conditions

Nothing is manipulated. The factors of interest are (i) the system (eleven harnesses plus one meta-harness contrast point) and (ii) the snapshot date (April 2026 versus July 2026, for the eight systems carried over from the first edition). The "dependent variables" are qualitative implementation positions along seven analysis dimensions D1 to D7 (agent loop; LLM integration and model–agent co-design; tool and action systems; memory and context management; safety and permission models; multi-agent orchestration; extensibility mechanisms), plus counts (built-in tools, adoption of standards such as MCP, skills, hooks, ACP), plus order-of-magnitude code size. Held fixed: each system is read at one pinned release (Table 3); the same seven dimensions are applied to every system. Nuisance factors the authors name: counting conventions for LoC and tools differ across languages and systems, so figures "position systems by order of magnitude" only (Sec. 4.2, Table 4 footnote).

### b) Subjects of study

Eleven harnesses and one contrast layer, pinned to July 2026 releases (Table 3): OpenHands (Python, V1 SDK v1.34.0), Aider (Python, v0.86.3.dev, maintenance mode), Claude Code (TypeScript, a "circulated source snapshot" of March 2026; shipping binary 2.1.206), Codex (Rust, rust-v0.144.1), Gemini CLI (TypeScript, v0.50.0), Mistral Vibe (Python, v2.19.1), Mini-SWE-Agent (Python, v2.4.5), Hermes (Python, 0.18.2), Pi (TypeScript, v0.80.6), OpenCode (TypeScript, v1.17.18), OpenClaw (TypeScript, v2026.6.11, a multi-channel assistant gateway used as a non-coding contrast point), and Omnigent (Python, v0.4.0, Databricks' meta-harness, scored on none of the seven dimensions and analysed separately in Sec. 14.4). Code size spans three orders of magnitude: Mini-SWE-Agent about 5K LoC, Codex about 1.1M lines of Rust across 126 crates; the corpus totals "roughly four million lines" (abstract). Three systems (Hermes, Pi, OpenCode) are new in this edition; eight are carried over with their April snapshots retained.

### c) Measurement & instruments

Source reading anchored in named modules, types, and flags; dependency-manifest inspection; import grep across three languages for framework and vector-store dependencies (Sec. 13.2); reading of each system's canonical system prompt for rhetorical content (Sec. 7.3, Table 16). All instruments are paper-specific and qualitative. Deliberate design choices: no line-number citations ("finer precision would be obsolete by the time the paper is read", Sec. 15.6); code-size figures dated to the July pins; benchmark numbers removed from the main comparison table because they are "self-reported, obtained on different model generations and configurations" (footnote 4 to Table 4; the spring-2026 self-reported SWE-bench Verified figures are kept in the footnote: OpenHands 77.6%, Mini-SWE-Agent 74%+, Claude Code 72.7%, Codex 69.1%).

### d) Experimental conditions / protocol

Setup: select systems for spread along design philosophy, maturity, and market position, one per major commercial provider plus seven open-source niches (Sec. 4.1); pin each to a release. Reading: dissect every system along D1 to D7 (Sections 6 to 12), recording minimal and maximal observed implementation per subsystem (Table 1). Longitudinal: for the eight carried-over systems, diff the April and July snapshots (Sec. 14.5). Synthesis: derive 13 numbered cross-cutting observations, a catalog of 29 recurring patterns (17 from April with membership updated, 12 new; Tables 11 and 12), a five-axis trade-off framework (Sec. 13.4), 18 design recommendations each citing its supporting observations and implementing systems (Sec. 16), and a 90-line Python scaffold (Listing 3). A market sweep of systems outside the corpus (Table 2) is drawn from announcements and repository metadata and is flagged as "not source-verified" (footnote 3).

### e) Statistical / analytical approach

None. All quantitative claims are counts out of 11 systems (with a parallel count out of 10 wherever OpenClaw's inclusion matters, Sec. 4.1), out of 9 multi-agent systems, or out of 9 skills adopters. No inter-rater reliability, no sampling, no significance testing. The longitudinal sample is 8 systems over one interval. The authors state the qualitative scoring "involves judgment calls" and invite disagreement (Sec. 15.6).

---

## 3. Findings & Observations (PRIMARY FOCUS)

### a) Headline findings

**F1. The twin absences: no agentic framework, no RAG over code (Observation 9, Sec. 13.2, Table 13).**
- Claim: across all eleven agent runtimes, no production code path imports a general-purpose agentic framework, and no system retrieves code with vector embeddings.
- Evidence: dependency manifests and import grep over "roughly 4 M lines of Python, TypeScript, and Rust" for LangChain, LangGraph, LlamaIndex, AutoGen, CrewAI, Pydantic AI, Genkit, Haystack agents, Semantic Kernel, Google ADK, Smolagents, Swarm, Agno: zero hits in agent code. Gemini CLI "uses neither of Google's own frameworks (Genkit, ADK)". Vector stores (Chroma, Pinecone, Weaviate, Qdrant, Milvus, FAISS, LanceDB, sqlite-vec, Elasticsearch in vector mode): zero for code retrieval. Every loop is hand-rolled (asyncio, blocking Python, Promise/async-iterator, Tokio); every tool registry is custom (Pydantic, Zod, TypeBox, Effect Schema, Rust enums). Retrieval runs on ripgrep, glob, tree-sitter, and auto-discovered Markdown context files (Table 13).
- Conditions and boundary cases: the finding "survived a threefold corpus expansion and a three-month re-audit". Three boundary cases are named: OpenCode delegates inner LLM plumbing to Vercel's AI SDK (a provider-abstraction layer, not orchestration; an in-house replacement client sits behind a flag); Aider's `/help` can install llama-index for documentation lookup over Aider's own docs; OpenClaw's default memory plugin runs hybrid sqlite-vec plus FTS5 search with embeddings on by default, "for chat recall, never for reading the source tree". Hermes's core conversation search is deliberately lexical (SQLite FTS5, BM25 plus trigram, "no LLM calls anywhere").

**F2. Loop sophistication does not predict benchmark performance; production mass goes elsewhere (Observation 1, Sec. 5).**
- Claim: systems spanning three orders of magnitude in code size target similar tasks, and the minimal linear loop reports results in the same range as the event-sourced engines.
- Evidence: the self-reported figures in footnote 4 (Mini-SWE-Agent 74%+, OpenHands 77.6%, Claude Code 72.7%, Codex 69.1% on SWE-bench Verified). "Roughly three-fifths of OpenCode's non-test source is its TUI, web, desktop, and SDK clients rather than the harness proper"; Codex "devotes six-figure line counts to app-server transports, plugins, and a realtime voice layer that no benchmark will ever measure". Mini-SWE-Agent implements all seven subsystems in about 100 lines (Listing 1; Sec. 2.4).
- Conditions: the authors stress the figures are not comparable (different models, dates, sandboxing) and "do not draw a head-to-head conclusion from them" (Sec. 13.4, Axis 1).

**F3. Extensibility standards resolved in favour of skills; hooks became the extension substrate (Observation 8, Sec. 12.5, Table 10; Sec. 14.1 Signal 2).**
- Claim: SKILL.md skill bundles are the most-adopted extensibility standard; deferred loading is near-universal among adopters; a supply chain and non-human authors have appeared.
- Evidence: skills 9/11 (only Aider and Mini-SWE-Agent abstain) versus MCP 8/11 (7/10 coding-first); the April tie (6/8 each) was broken by Pi, which implements the agentskills.io specification while rejecting MCP outright ("build CLI tools with READMEs"). Deferred loading of skill bodies in 8 of 9 adopters. The `.agents/skills/` path is accepted by six systems; OpenCode additionally searches `~/.claude/skills`, so skills installed for Claude Code work in OpenCode unmodified. Remote skill registries in four systems (Mistral Vibe, OpenCode, Hermes, OpenHands); trust tiers, pre-install scanning, and quarantine in Hermes; provenance verification in OpenClaw. Agent-authored skills in Hermes (a background review agent creates and patches skills from completed tasks, with a curator maintaining the collection) and Gemini CLI (an extraction sub-agent mines sessions into skill patch files for a human-reviewed inbox). User-facing lifecycle hooks: 9/11 systems (Codex's event names are "verbatim Claude Code's"; Gemini CLI exposes eleven loop-lifecycle events; Pi's extension API has about 33 typed events; OpenCode's plugins return about 20 typed hooks).
- Conditions: counts are at the July 2026 pins; Claude Code cells reflect the March 2026 snapshot.

**F4. Convergence became imitation; patterns diffuse in weeks; trees move at platform speed (Observation 12, Sec. 14.5).**
- Claim: April's convergences were "mostly independent rediscovery"; July's are traceable to copying, and the half-life of a competitive distinctive "is currently measurable in weeks".
- Evidence: Codex adopted Claude Code's hook event vocabulary verbatim (PreToolUse, PermissionRequest, PostToolUse, PreCompact, SessionStart, UserPromptSubmit, SubagentStart, SubagentStop, Stop, plus a PostCompact event with no Claude Code counterpart) and ships an importer that converts `~/.claude/projects` session JSONL into Codex rollout items and offers to translate `~/.claude/settings.json` to `~/.codex/config.toml`; OpenHands's manifest loader accepts `.claude-plugin` directories and its TaskTool signature is "conspicuously Claude Code-like"; Hermes's source comments credit OpenCode (edit matcher), Codex (smart approvals), OpenClaw (orchestrator prompt), and Goose (context hints). Diffusion counts over the quarter: deferred tool loading 1 to 3 systems (plus seven skills variants); read-only plan modes 2 to all 4 provider-native systems; LLM approval classifiers 1 to 2 (Codex's Guardian); turn-level checkpointing 1 to 3; safety-aware scheduler partitioning 1 to 2. Codex's workspace grew from 621K to about 1.12M lines of Rust and 89 to 126 crates; Mistral Vibe grew 77% (35.6K to 63K lines); Aider had 18 commits in the window.
- Conditions: eight systems, one quarter (April to July 2026).

**F5. Behavioural policy is migrating from prompt prose to configuration, and prompt rhetoric thins as trust calibrates (Observation 3, Sec. 7.3, Table 16).**
- Claim: prompt directives converge where engineering experience converges, then thin or move into feature flags and precedence contracts as models internalise norms.
- Evidence: anti-gold-plating language "in nearly isomorphic phrasings across six independently developed scaffolds"; the no-autonomous-commit rule was universal in April 2026 and split three ways by July (kept in Claude Code, OpenCode, Hermes, Codex 5.2 prompt; reversed in Mistral Vibe, whose changelog reads "Loosened the no-git-commit constraint" and which now mandates a co-author trailer; dropped entirely by Codex's GPT-5.6 prompts, which "contain no occurrence of 'commit'"; a `codex_git_commit` flag briefly governed the behaviour and is retired). Mistral Vibe replaced its "CRITICAL:" tags and Hard Rules section with a seven-level instruction-precedence contract. Codex's prompts are now server-delivered model-catalog data (`models.json` `base_instructions`, refreshed from a remote endpoint) rather than compiled-in templates, so scaffold behaviour is re-tuned per model release without a client update. 9 of 11 prompts contain verbosity directives (Claude Code's quantified limits of 25 and 100 words are gated to internal A/B builds; OpenCode demands fewer than 4 lines; Mistral Vibe under 150 words). 11 of 11 prompts contain no policy-level refusal language: "alignment work is never duplicated in the harness prompt". OpenCode's edit-tool description claims the tool "will error if you attempt an edit without reading the file" but "no runtime read-tracking exists in the tool's source" ("prompt text is a behavioral wish, not a mechanism").

**F6. Compaction converged; persistent memory is the new frontier, differentiated by who writes and who reviews (Observation 5, Sec. 9, Figure 4).**
- Claim: seven of eleven systems use threshold-triggered LLM compaction; what now differentiates systems is the memory write path.
- Evidence (compaction): Claude Code fires when the token estimate enters a 13,000-token buffer below the context window and post-restores up to 5 files under a 50,000-token budget; Gemini CLI compresses at 50% of the limit and preserves the most recent 30% verbatim; Hermes triggers at 50% of context minus max-tokens with a 64K floor and rotates to a child session instead of rewriting the transcript ("lineage compaction"); Pi triggers at context window minus 16,384 tokens, keeps 20K recent, and merges summaries iteratively over an append-only JSONL session tree; OpenCode uses a chars/4 estimate ("no tokenizer exists anywhere in the tree") and merges into mandated Markdown sections; Aider's recursive halving defaults to 1,024 tokens with up to 3 levels; OpenHands consolidated ten condenser classes to three in its V1 SDK.
- Evidence (memory governance, four models): agent-maintained (Codex: a background two-phase pipeline extracts memories per rollout into SQLite and a sandboxed internal sub-agent consolidates them over a git-baselined `~/.codex/memories/` root, injected with citation tracking and usage-based ranking; "the first system in the corpus whose long-term memory is maintained by an agent rather than by code"); human-gated (Gemini CLI removed its `save_memory` tool; an extraction sub-agent writes patch files to a per-project `.inbox/` that the user reviews and applies via `/memory`, "the only such design in the corpus"); model-direct-but-bounded (Hermes: MEMORY.md capped at 2,200 characters and USER.md at 1,375, injected as a frozen snapshot for cache hygiene; OpenHands and Claude Code via context-file conventions); pre-turn agentic recall (OpenClaw's Active Memory sub-agent runs before each reply). "None of the eleven uses embedding-based retrieval as its primary memory substrate."

**F7. Safety is code-expensive and a choice, not a consequence of scale; both flagships converged on a three-stage permission shape (Observation 6, Sec. 10).**
- Claim: the April correlation "largest systems are the sandboxers" broke; where a native sandbox is built it is a multi-thousand-line investment; Claude Code and Codex now share the static-rules to LLM-classifier to backstop pattern.
- Evidence: Hermes (about 642K lines) ships zero OS-level isolation primitives and spends on content-borne threats (a 3,200-line approval module, twelve hardline patterns that survive `--yolo` with the bypass flag frozen at module import "so a prompt-injected skill cannot flip it at runtime", 47 dangerous-command patterns matched on deobfuscated variants); OpenCode (about 578K lines) is policy-only with tree-sitter-parsed commands and a roughly 130-entry LLM-generated command-arity dictionary scoping "always allow" grants. Codex has four layers (Starlark execution policy with parse-time-validated inline examples; lifecycle hooks; a Guardian LLM approval reviewer that fails closed; vendored Bubblewrap on Linux, Seatbelt on macOS, restricted tokens on Windows). Claude Code has three (PreToolUse hook rules; LLM permission classifier; interactive dialog). Pi documents the absence of a sandbox as a security argument ("would be easy to misunderstand as a security boundary").

Further observations, briefly: Observation 2, provider-native optimisations (cache boundaries, thinking budgets, reasoning effort) are "gated on who pays the per-provider conditional-code cost", not on tight coupling; Hermes, Pi, and OpenCode reach the full menu from multi-provider substrates, and what only vendors retain is "server-side co-evolution". Observation 4, editing contracts split into an exact camp (Claude Code; Mistral Vibe deleted its fuzzy SEARCH/REPLACE tool and converged on exact unique-substring matching within the quarter) and a drift-tolerant camp (OpenCode's nine-stage cascade at Levenshtein 0.65, Hermes's nine-strategy chain "inspired by OpenCode", Aider's RelativeIndenter), with Gemini CLI's LLM edit-fixer subcall as a third option. Observation 7, coordinator-worker shapes appear in seven systems (eight with OpenClaw) while sub-agent coordination stays in-process in 8 of 9 multi-agent systems. Observation 11, ACP ships in 6 of 11 systems and acquired a third role, harness hosting (OpenHands runs Claude Code, Codex, or Gemini CLI as interchangeable step backends). Observation 13, the 90-line scaffold "implements 10 of the 18 recommendations directly".

### b) Patterns & trends

- Deferred loading generalised from tools (Claude Code) to skills (8 of 9 adopters) and to MCP catalogs (Codex's BM25 `tool_search`; Hermes collapses MCP and plugin schemas into three bridge tools once they would exceed 10% of the context window).
- Threshold compaction is the de facto standard (7 of 11), with newcomers hybridising it: iterative summary merging (Pi, OpenCode), pluggable engines (Hermes's ContextEngine, Pi's veto hook), session-tree substrates (Pi, Hermes).
- Cross-harness lineage is now "written in the code itself" (crediting comments, verbatim vocabularies, format readers, importers).
- The authors distinguish "inventory claims (tool counts, feature cells, version pins)" that "decay in weeks" from "structural claims (loop taxonomy, subsystem anatomy, the absences)" that "have so far proven durable" (Sec. 14.5).
- Every system reads hierarchical Markdown context files (AGENTS.md, CLAUDE.md, GEMINI.md, `.cursorrules`), and the newest read several ecosystems' filenames (Hermes, OpenCode, OpenHands).

### c) Surprises & counterintuitive results

- The framework absence held even for Google's own Gemini CLI; the authors "expected to find at least some use of LangChain" and "the search for counterexamples ... ran for several weeks before we accepted the result" (footnote 8).
- Skills overtook MCP, and a production harness (Pi) rejects MCP as a position.
- The size-implies-sandbox correlation of April was falsified by Hermes and OpenCode.
- A provider-native flagship (Codex) adopted a rival's extension vocabulary verbatim and shipped an importer for the rival's on-disk state.
- The corpus's tightest April convergence (no autonomous commits) dissolved in one quarter.
- Prompt claims can overstate mechanisms (OpenCode's edit-tool description); Claude Code's numeric verbosity limits and its anti-false-success section ship to internal builds first.
- Codex's `ultra` reasoning tier "changes orchestration behavior" (maximum reasoning plus automatic sub-agent delegation), an effort level that alters harness topology.

### d) Negative / null results

- No relationship found between loop sophistication and self-reported benchmark performance (Observation 1), with the caveat that the numbers are not comparable.
- Three April observations required "substantive revision in place": the coupling dichotomy (Observation 2), size-implies-sandbox (Observation 6), and protocol placement (Observation 11).
- April errors corrected in July: Codex's policy language was Starlark, not TOML (footnote 7); Aider resolves 13 edit formats, not 14 (footnote 5); Mistral Vibe's earlier fuzzy matcher "only generated 'closest match' diagnostics for error messages; it never applied fuzzy edits" (footnote 6).
- The study produced no cross-system benchmark, no operational definition of the scaffold–capability frontier, and no test of the 90-line scaffold (Observation 13 is "a conjecture, without proof").

### e) Robustness checks

The corpus was expanded threefold in coverage terms (8 to 11 systems plus the meta-harness) and re-audited after three months; the framework and vector-store sweep was repeated across all twelve trees. Adoption counts are reported both with and without OpenClaw. Both April and July snapshots are retained. No inter-rater checks, no runtime replication, and no alternative scoring rubrics were applied.

---

## 4. Analysis & Interpretation

### a) Authors' explanations

Demonstrated at the level of the pinned source: the absences (grep and manifest evidence), the lineage of imitation (crediting comments, verbatim event names, format readers, importers), and the diffusion counts. Proposed mechanisms, offered as interpretation: frameworks are absent because "hand-rolled debuggable code wins out over reusable abstractions once the agent is mutating real source code" (silent prompt corruption, opaque caching, version-incompatible tool schemas become too expensive; Sec. 13.2), corroborated by Anthropic's December 2024 guidance; RAG is absent because code "carries dense deterministic structural metadata", "changes minute to minute, so pre-indexed embeddings are stale almost by construction", every environment ships ripgrep and glob, and CodeRAG-Bench found retrieval gains "highly variable across tasks" (Sec. 13.2); policy thins because "models internalize norms and harnesses grow governance surfaces" (Sec. 14.5); the platform turn is argued from four signals (skills as declarative programs, hooks as extension substrate, disappearing tool/workflow boundary, harness as service surface; Sec. 14.1) and the harness–framework merger (Table 15). Speculated and labelled as such: whether the alignment with Anthropic's guidance "reflects shared empirical reality, public-guidance influence, or both is an open question" (Observation 10); the concave scaffold–capability frontier (Sec. 15.5); the conjecture that the 90-line scaffold would match Mini-SWE-Agent's numbers (Observation 13).

### b) Evidence quality

Strong for existence and absence claims at the pinned snapshots: they are verifiable by anyone with the ten public trees (the Claude Code snapshot excepted). Moderate for the longitudinal claims: n = 8 systems, one interval, qualitative reading of diffs. Weak for anything about performance: the only outcome numbers are self-reported SWE-bench figures under different models, and the authors themselves quarantine them to a footnote. The 18 recommendations are prescriptions inferred from what popular systems do, with no outcome variable connecting a design choice to a result; the study is candid that "whether the production scaffolding produces measurable improvements on the same benchmark under matched conditions is an open question that this study does not answer" (Sec. 13.4).

### c) Confounds & alternative explanations

- Selection: systems were chosen for spread and market position, not sampled; popularity is not efficacy, so convergence may reflect fashion or copying rather than fitness. The authors partially address this by relabelling July's convergence as imitation.
- Shared upstream: Anthropic's four engineering articles and the March 2026 Claude Code source exposure (which also spawned Claw Code, about 100k stars "within days") mean the "independently developed" systems are not independent draws; Observation 10 flags this and Sec. 15.6 proposes interview work.
- AI-assisted reading: the analysis and drafting used Claude Code; verification is asserted, not documented.
- Staleness asymmetry: all Claude Code cells are from March 2026 while the other ten are July 2026 (Table 4 caption, Sec. 15.6).
- Counting heterogeneity: tool counts mix built-ins, default surfaces, edit formats, and registered-but-hidden tools (Figure 3 caption).

### d) Generalizability

The findings describe coding-agent harnesses in mid-2026, mostly terminal-first. OpenClaw serves as a single non-coding contrast, and parallel counts out of 10 let a reader restrict claims to coding agents. The authors' own framing limits inventory claims to weeks; the structural claims (seven subsystems, the absences, the governance taxonomy of memory writes) are the parts they expect to endure. Extrapolation to non-coding agent harnesses, to harnesses on open-weight models (Hermes's model-gated prompt blocks suggest behaviour differs by model family), or to future model generations (Mistral Vibe's fuzzy-to-exact migration is read as "stronger models shift the optimum") is risky by the paper's own evidence.

---

## 5. Implications, Limitations & Transferability

### a) For practitioners

The 18 recommendations (Sec. 16), condensed: start with a linear while loop and graduate to a middleware pipeline once three or more independent turn policies exist (R1); couple tightly to a home provider with a generic fallback, or budget for per-model metadata if multi-provider (R2); begin with a bash tool alone and add tools only against observed failures (R3); adopt deferred tool loading past about 15 tools (R4); match the edit contract to the model tier, exact for frontier models and fuzzy for weaker ones, never line numbers (R5); auto-discover hierarchical Markdown context files and read neighbours' filenames (R6); threshold compaction with a verbatim tail, incremental summary merging, and reactive firing on overflow (R7); no RAG over code (R8); a three-mode approval system for developer tools (R9) and OS-level sandboxing with policy-as-code for enterprise or automated contexts (R10); safety rules as data with a floor beneath any bypass mode (R11); stay single-agent until a breadth-first exploration phase justifies parallel isolation (R12); ship an ACP server and keep sub-agents in-process (R13); skills for capability templates, MCP for external integrations, treat third-party skills as packages (R14); no general-purpose agentic frameworks (R15); no vector-embedding code retrieval (R16); no one-to-one SaaS API wrapping (R17); ship cheap stuck-detection caps, nothing elaborate (R18). Listing 3 is a 90-line Python scaffold implementing ten of these.

### b) For researchers

Future work named in Sec. 17.2: unified evaluation of safety, cost, extensibility alongside correctness; a reference architecture specification for the catalogued patterns; formal verification of policy-as-code; empirical study of model–agent co-evolution across model generations, including how automatically evolved scaffolds (AHE) compare to hand-crafted ones; a cross-system benchmark under controlled conditions; protocol-adoption tracking; quarterly re-pinning to turn the study into a time series measuring "pattern-diffusion half-lives, prompt thinning, and the platformization trajectory"; meta-harness economics; causal analysis of the Anthropic–industry alignment. The scaffold–capability frontier is given an operational sketch: for a fixed task distribution $T$, fixed model $M$, a scaffold-complexity metric, and a target success rate $p$, the "minimum viable scaffold for $(T, M, p)$" is the lower envelope of complexity values whose empirical success on $T$ with $M$ exceeds $p$; the study does not compute it.

### c) Acknowledged limitations

Sec. 15.6: source reading, not runtime measurement ("we do not claim to know how fast any of these systems are"); qualitative scoring with judgment calls; no line-number citations and approximate LoC; the Claude Code analysis rests on a March 2026 circulated snapshot and is "the weakest link on reproducibility"; the framework-absence sweep did not trace internal forks, dynamic imports, or transpiled distributions ("We would be surprised, but not shocked"); the Anthropic-alignment mapping does not establish causation; no shared-task head-to-head run. Landscape claims in Sec. 3.7 are announcement-sourced, not source-verified.

### d) Unacknowledged limitations

- No outcome variable: the study has no measurement of task success, cost, or reliability of its own, so every recommendation rests on the prevalence of a choice among selected popular systems.
- Single analyst team with no inter-rater reliability for the qualitative scores; the April-to-July corrections show the reading method's error rate is non-trivial (three revised observations, three factual corrections) without an estimate of how many remain.
- The longitudinal "half-life measurable in weeks" is an impression from eight systems over one interval, not a measured rate.
- The imitation claims infer direction from code comments and vocabulary identity; simultaneous adoption from a common public source (documentation, blog posts) is not excluded.
- The "platform turn" thesis mixes source evidence with market events (acquisitions, tier withdrawals) whose interpretation is not falsifiable within the study.
- The 90-line scaffold is untested; its compaction routine, for instance, drops the middle of the transcript including tool results the summary may not cover.

### e) Reproducibility

No code, data tables, scoring sheets, or snapshot archives are released. Reproduction relies on the version pins in Table 3 (exact tags or commits for ten public systems) and on re-reading the sources; the Claude Code analysis cannot be reproduced from public artifacts. The April snapshots are said to be retained by the authors but are not published. Replicating the audit would mean re-cloning the ten trees at the pinned releases and re-running the manifest and import sweep; replicating the qualitative scoring would require a rubric the paper does not include.

### f) Transfer to our own work

Read against `docs/framing.md`:

1. **The layer list is grounded in production.** Our harness layers (system prompt, tool description, tool implementation, middleware, skill, sub-agent configuration, long-term memory, model weights) each appear as a documented subsystem or pattern in the corpus: middleware as Mistral Vibe's composable turn policies and Recommendation 1's upgrade path; skills in 9 of 11 systems; sub-agent configuration as Markdown-plus-frontmatter agent definitions in Claude Code, OpenHands, Gemini CLI, Pi, and OpenCode; long-term memory as the four governance models of Observation 5. No system in the corpus edits weights, which matches our "plus the model weights when they are editable" clause. Table 1 and Sections 6 to 12 are citable support for the setting's decomposition.

2. **Every in-the-wild self-improving harness is a horizon-zero system.** Three production write paths edit the harness from experience: Codex's agent-maintained memory (sandboxed sub-agent consolidates over a git-baselined store), Hermes's self-improving skill loop (a background agent authors and patches skills, a curator prunes), and Gemini CLI's extraction inbox (human-gated). None scores an edit on later tasks; none measures retention; the only after-the-fact signal is Codex's usage-based ranking of injected memories, a popularity proxy rather than a return. These are concrete, named instances for the claim that existing systems are the $\gamma = 0$ case, and they are production systems rather than research prototypes, which strengthens the motivation beyond HCL, SIA, Adaptive Auto-Harness, and Harness-R1.

3. **Human-authored moves across layers are observed longitudinally.** Observation 3 and Sec. 14.5 document rules migrating from prompt prose to feature flags and configuration (Codex's commit rule to `codex_git_commit` and then out entirely; Mistral Vibe's hard rules to a precedence contract), and Recommendation 1 describes the middleware promotion path. This is our `move` and `promotion` vocabulary enacted by human maintainers, and it gives the proposer action space an empirical anchor: the directions humans actually take are prose to configuration, per-run flags to structural mode (plan mode became "structural rather than prompt-only" in Mistral Vibe), and many narrow entries to one general mechanism (OpenHands's ten condensers to three).

4. **Human harness-evolution logs exist and are pinned.** The framing says the critic "can be pretrained on released harness-evolution logs". Table 3's version pins, the retained April snapshots, and the public git histories of ten harnesses are exactly such logs, at a scale of millions of lines and hundreds of releases. The caveat: they carry edit trajectories without task-stream returns, so they can supply the input distribution for a critic reading harness files but not the regression targets; commit messages and changelogs are at best weak labels.

5. **The static frontier complements our temporal claim.** The study's scaffold–capability frontier is a curve over scaffold complexity at fixed task distribution and fixed model: beyond a low floor, scaffold work buys safety and operations, not completion rate. Our plasticity claim is about the time axis at fixed complexity: a greedy edit that helps the current batch lowers the future stream performance reachable from the resulting harness. The two are orthogonal and the paper's Axis 5 (scaffold complexity versus model capability) can be cited as the static baseline our stream setting extends. Note also Mistral Vibe's fuzzy-to-exact editing migration: an edit optimal for one model generation is not optimal for the next, so model updates are one source of non-stationarity a task stream could include.

6. **What a critic should read.** The authors' distinction between inventory claims that decay in weeks and structural claims that endure is a hint for the critic's input representation: tool counts and specific rule text are volatile, while loop topology, layer placement of policy, and governance of the write path are stable. A critic that scores harness files should weight structure over inventory.

7. **Guarding the update machinery from the edited content.** Hermes freezes its bypass flag at module import so "a prompt-injected skill cannot flip it at runtime"; Pi trust-gates repository-controlled configuration while loading context files regardless and documents that acceptance. Our framing rules out self-editing of evaluators; these are production precedents for the boundary between editable harness content and the machinery that scores or commits it.

8. **Literature pointers surfaced by this paper and its search.** Lin et al., Agentic Harness Engineering (arXiv:2604.25850): a bash-only seed evolved with observability-driven feedback reaches 71.9% on SWE-bench Verified, transfers across four model families, and its ablation attributes the gain to tools (+3.3 pp), middleware (+2.2 pp), and long-term memory (+5.6 pp) while the system prompt alone regresses (−2.3 pp), which is layer-level evidence for where edits pay off and is not yet in our positioning list. Also cited by the paper: Rombaut's source-code taxonomy (arXiv:2604.03515), HARBOR (arXiv:2604.20938, harness optimisation against benchmarks), Code as Agent Harness (arXiv:2605.18747), and Macedo's definitional paper (arXiv:2606.10106). Surfaced by the web search for this reference, title only and unread: "Don't Blame the Large Language Model: How Scaffolding Evolution Shapes Coding Agent Quality" (arXiv:2607.03691) and "What Do Evolutionary Coding Agents Evolve?" (arXiv:2605.20086).

---

## 6. Summary

### a) One-sentence headline finding

Eleven production coding harnesses, read at source, share hand-rolled loops, deterministic retrieval, skills-over-MCP, and imitation-driven convergence; none uses frameworks or code RAG.

### b) Quick-reference takeaway list

- Across about 4 million lines in 11 harnesses, 0 agent runtimes import an agentic framework and 0 retrieve code with embeddings; retrieval is ripgrep, glob, tree-sitter, and Markdown context files.
- Skills (9/11) lead MCP (8/11) as the extensibility standard; hooks are in 9/11; deferred loading is near-universal; skill registries, trust tiers, and agent-authored skills appeared within one quarter.
- Compaction converged on threshold-triggered LLM summarisation (7/11); the differentiator is now who writes persistent memory: an agent (Codex), a human-reviewed inbox (Gemini CLI), bounded model-direct files (Hermes), or a pre-turn sub-agent (OpenClaw).
- Over April to July 2026, convergence became imitation (Codex copied Claude Code's hook vocabulary verbatim and ships an importer for its sessions), patterns diffused in weeks, and behavioural policy moved from prompt prose into configuration.
- Loop sophistication does not track self-reported benchmark scores; production mass goes to safety, clients, transport, and extensibility, and OS sandboxing is a choice (Hermes and OpenCode, among the largest, have none).

### c) Bottom line for decision-making

Trust the structural and absence claims as of July 2026 (they are verifiable against pinned public trees); treat every count and every recommendation as a dated inventory with no outcome evidence behind it; use the paper as the best available map of what production harnesses contain and how humans evolve them, not as evidence of what works.

---

## 7. Reviewer Reception (OpenReview)

No public OpenReview page found (searched the exact title and the short form "Anatomy, Architecture, and Evolution of Coding Agents" together with "openreview" on 2026-09-14). The paper is an arXiv preprint (cs.SE, cs.MA) with no venue stated on its front page or arXiv record, so no review record exists to report. The only OpenReview hit for adjacent terms was a different paper ("Agent Systems with Harness Engineering", Tang et al.), not this one.
