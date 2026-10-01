# Harness components across the references: what is named, what is edited, how an edit is written and checked

Written 2026-10-01. Scope: for every reference in this folder that has an agent harness, the parts of the harness the paper names, the parts its method changes in the reported experiments, the form an edit takes, and the check an edit must pass. Each entry was extracted from the reference's analyses and then re-checked against its PDF and, where a clone exists, its code. The last section records the layer list the researcher chose on 2026-10-01 after reading this survey.

## Question

Papers on self-improving harnesses divide the harness into parts, and the divisions differ: seven component types in AHE, nine dimensions in HarnessX, four components in HCL, four positions in Harness-R1. Which parts do the methods change in their experiments, as opposed to parts they only name? In what form is a change written, and what must it pass before it is installed? Which parts are changed by many papers and which by none?

## Papers covered

- [AHE](../related-work/AHE/paper_analysis.md): *Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses* ([arXiv:2604.25850](https://arxiv.org/abs/2604.25850)).
- [Adaptive-Auto-Harness](../related-work/Adaptive-Auto-Harness/paper_analysis.md): *Adaptive Auto-Harness: Sustained Self-Improvement for Agentic System Deployment on Open-Ended Task Streams* ([arXiv:2606.01770](https://arxiv.org/abs/2606.01770)).
- [HarnessX](../related-work/HarnessX/paper_analysis.md): *HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry* ([arXiv:2606.14249](https://arxiv.org/abs/2606.14249)).
- [HCL](../related-work/HCL/paper_analysis.md): *Harness Continual Learning: Continual Adaptation Beyond Model Parameters* ([arXiv:2608.19013](https://arxiv.org/abs/2608.19013)).
- [AgentStream](../related-work/AgentStream/paper_analysis.md): *AgentStream: How Well Do Self-Evolving LLM Agents Perform Under Streaming Tasks?* ([arXiv:2608.00155](https://arxiv.org/abs/2608.00155)).
- [SIA](../related-work/SIA/paper_analysis.md): *SIA: Self Improving AI with Harness & Weight Updates* ([arXiv:2605.27276](https://arxiv.org/abs/2605.27276)).
- [Meta-Harness](../related-work/Meta-Harness/paper_analysis.md): *Meta-Harness: End-to-End Optimization of Model Harnesses* ([arXiv:2603.28052](https://arxiv.org/abs/2603.28052)).
- [Harness-R1](../related-work/Harness-R1/paper_analysis.md): *Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories* ([arXiv:2608.02276](https://arxiv.org/abs/2608.02276)).
- [Learning-to-Self-Evolve](../related-work/Learning-to-Self-Evolve/paper_analysis.md): *Learning to Self-Evolve* ([arXiv:2603.18620](https://arxiv.org/abs/2603.18620)).
- [MemSkill](../related-work/MemSkill/paper_analysis.md): *MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents* ([arXiv:2602.02474](https://arxiv.org/abs/2602.02474)).
- [MemEvolve](../related-work/MemEvolve/paper_analysis.md): *MemEvolve: Meta-Evolution of Agent Memory Systems* ([arXiv:2512.18746](https://arxiv.org/abs/2512.18746)).
- [EvolveMem](../related-work/EvolveMem/paper_analysis.md): *EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents* ([arXiv:2605.13941](https://arxiv.org/abs/2605.13941)).
- [ALMA](../related-work/ALMA/paper_analysis.md): *Learning to Continually Learn via Meta-learning Agentic Memory Designs* ([arXiv:2602.07755](https://arxiv.org/abs/2602.07755)).
- [ReMe](../related-work/ReMe/paper_analysis.md): *Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution* ([arXiv:2512.10696](https://arxiv.org/abs/2512.10696)).
- [Evo-Memory](../related-work/Evo-Memory/paper_analysis.md): *Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory* ([arXiv:2511.20857](https://arxiv.org/abs/2511.20857)).
- [MemRL](../related-work/MemRL/paper_analysis.md): *MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory* ([arXiv:2601.03192](https://arxiv.org/abs/2601.03192)).
- [Harness-Anatomy](../related-work/Harness-Anatomy/paper_analysis.md): *Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents — A Source-Code Study of Eleven Systems* ([arXiv:2609.00006](https://arxiv.org/abs/2609.00006)).
- [Agent-Harness-Survey](../related-work/Agent-Harness-Survey/paper_analysis.md): *Agent Harness Engineering: A Survey* (OpenReview eONq7FdiHa).

Six references in the folder have no agent harness and are not covered: GEM, Riemannian-Walk, Loss-of-Plasticity, Understanding-Plasticity, Dormant-Neurons, Lehman-Laws.

## Synthesis

### Two ways to divide a harness

The references divide a harness along two different lines. The first asks what kind of thing is changed: the system prompt text, a tool's description, a tool's code, a skill file, stored memory, and so on. AHE, Adaptive-Auto-Harness, HarnessX, HCL, AgentStream and both surveys divide this way. The second asks at what moment during one attempt at a task the changed code runs. Harness-R1 divides this way: its four positions are the start of the attempt, before the model decides, before the model's action is executed, and after the result returns. All four positions hold the same kind of thing, a Python function that the runtime calls without the model asking for it. `docs/framing.md` uses the first division and calls its parts layers.

### Middleware

Between the model's step and the environment there is code the model never calls. It runs on every turn, reads the message list or a tool result, and can add text, change the proposed action, or block it. AHE and the two surveys call this middleware; HarnessX calls one unit a processor; Harness-R1 calls one unit a code hook. Three examples from the cloned code:

| Source | File | When it runs | What it changes |
|---|---|---|---|
| AHE | `refs/related-work/AHE/repo/experiments/evolved_harness/middleware/execution_risk_hints.py` (525 lines) | before each model call and after each shell call | appends risk notes to the tool output and hint messages to the prompt |
| Harness-R1 | `refs/related-work/Harness-R1/repo/examples/webshop_patch.json`, hook `make_pre_hint` | before each decision of the task model | returns a message that the runtime shows to the model |
| HarnessX | `refs/related-work/HarnessX/repo/harnessx/processors/control/loop_detection.py` | before and after each tool call | appends a warning to the tool result, or stops the run when the same call repeats |

Middleware differs from a tool in who starts it: a tool runs when the model asks for it, middleware runs whether or not the model asks. It differs from the system prompt in that it can act on a condition: a prompt rule is shown at every step, the loop detector adds its warning only after a call has repeated. Middleware content stays from round to round in AHE and HarnessX and is discarded after each batch in Harness-R1.

### Per reference: parts named, parts changed, edit form, check

| Reference | Components claimed | Edited in experiments (paper's own names) | How an edit is expressed | What validates it |
|---|---|---|---|---|
| Adaptive-Auto-Harness | 4 in prose, 5 in code (`evolve_*` flags) | prompts, skills, tools, tool registry, memory (switched off in the PolyBench runs), infra — 6 names, 5 gated layers | Free-form whole-file writes through one bash tool in a Docker sandbox. A typed write API (`write_prompt`, `write_skill`, `write_tool`, `add_memory`) ships in the repo and the evolver never calls it. | A separate Verifier agent writes and runs its own tests; accepted on the literal string `VERDICT: PASS`. Git rollback to a pre-build tag, 3 retries. Post-cycle sweeps: regex-strip search caps, truncate the prompt at 10,000 characters, smoke-run one pipeline. The inherited held-out gate is dead code. |
| AHE | 7 | system prompt, tool description, tool implementation, middleware, long-term memory (5; plus `code_agent.yaml` registration) | Whole-file `write_file` and exact-string `replace` on a git workspace, plus a separate JSON manifest per edit. No patch envelope is wired; an `apply_patch` tool exists in the repo and is unused. | A static validator (YAML schema, middleware import paths, `ast.parse`) that the prompt tells the agent to run and the loop never calls. After the edit is live: next-round verdicts (HARMFUL…EFFECTIVE) handed back to the same agent, which decides rollback. |
| HarnessX | 9 dimensions; 4 logged edit types | prompt, tools, processor, config (plus task-model weights in the co-evolution setting) | Whole new files written into a scratch directory — a complete replacement `config.yaml`, optional new Python tool and processor modules, optional prompt template — plus a manifest with one-line prose change summaries. No diff text anywhere. | Canonicalize, processor and tool dry-fire, contract check (ships in `warn` mode, covers 2 of 8 hooks), novelty/evidence policy. Acceptance is an aggregate score against the historical best with a tolerance. The paper's per-task regression constraint has no code. |
| HCL | 4 | Task Interface, Experience Memory, Capability Map (inner skills), Adaptive Router | Whole-component replacement proposed by one prompt over the frozen model. No format, schema or operation list is given, and no artifact is shown in 21 pages. | The strongest gate in the set, run before deployment: improvement of at least δ on the current task's validation cases, zero regressions on an anchor set of previously solved cases, and validity checks (≥90% format compliance, legal tool use). Failing is a no-op. |
| MemEvolve | 4 (all inside one layer) | Encode, Store, Retrieve, Manage — 1 layer | A complete new Python provider class, scraped out of a Markdown reply by four regexes, written as a new file plus enum/mapping/config insertions. Repairs go through a shell agent told to prefer `sed`. | `ast.parse`, four literal substring checks, then a five-call in-process smoke test on hard-coded strings (no benchmark task is run; the real-task branch prints "Not implemented"). Then a tournament against the incumbent on 20–25 tasks. |
| SIA | 4 named parts plus weights | system prompt, tool-dispatch logic, answer extraction, supporting infrastructure (all one Python file); weights by a run-level flag | Free-text whole-file rewrite: the editor writes a new `target_agent.py` (or `train.py`) per generation. | None. No syntax or compile check, no accept/reject, no revert; generation N+1 replaces N whichever way the score moved. |
| Harness-R1 | 4 lifecycle positions; 6 declared action types | `add_code_hook` at `on_init`, `make_pre_hint`, `on_before_action`, `on_post_step` — 1 layer; the other 5 action types are switched off | One JSON object with a list of typed actions; each names a hook slot and carries Python source for `def hook(ctx, nb)` returning an effect from a closed vocabulary. | The strictest in the set, and the only one that runs before acceptance: schema plus a 12-action cap; AST allowlist (exactly one top-level hook, ≤5 helpers, forbidden nodes and names, size caps); then runtime wall-clock and line budgets with every exception downgraded to no effect. No regression check of any kind. |
| MemSkill | 3 components, 2 stores | skill bank — 1 layer (the memory bank is per-trace output; the controller is trained) | JSON with a `changes` list of at most 3 items, each `add_new` / `refine_existing` / `no_change`; payload is free text in a fixed five-section shape. | Field presence, `update_type` enum, placeholder ban, name existence, 30-entry cap with lowest-reward eviction. Then a cycle-level performance gate that restores the best snapshot of the bank. |
| EvolveMem | none stated (3 evolvable retrieval parts) | RetrievalConfig fields, per-category overrides, answer-style selection — 1 layer | A JSON object of field name → new value, assigned onto a typed dataclass; plus named recipe bundles and a hand-written whole-config overwrite at a fixed round. | Field allowlist with per-field ranges and enum sets, 2 changes per round; hill-climbing accept against best-so-far — on the same task set the proposer read. Recipes and the fixed-round overwrite bypass the allowlist. |
| ALMA | none stated (memory design = 3 parts) | memory design (update, storage, retrieval) — 1 layer | Free-form whole-file Python rewrite: the proposer is handed the current file and returns a complete new one, saved under a new identifier. | A 5-task trial run in a container; on an exception, debug and retry, up to 3 times. No score gate — designs that score worse than their parent stay in the archive and stay selectable. |
| ReMe | none stated | experience pool — 1 layer | Structured operations on a vector store: insert rows, delete by id, sweep-delete by rule, increment two counters. Every field change is a whole-record overwrite. | Success threshold on the producing trajectory, an LLM judge with a score threshold, embedding near-duplicate rejection, and a retrieval-count/usefulness deletion rule. Nothing re-runs a finished task. |
| Evo-Memory | 4 (base model, update, retrieval, context construction) | memory state — 1 layer | Append one free-text entry; delete entries by id named in a text line (`Think-Prune: <IDs>`). | None. No parse step is described, no score gate, no rollback. |
| MemRL | 3 (of the method, not of a harness) | memory bank rows plus one utility number per row — 1 layer | Three operations (`add`, `update`, `noop`) over store rows; payload is written text plus one float. | Nothing rejects a write: the pass/fail signal picks which template is written, and both branches write. No deletion, revert or rollback path. |
| LSE | none stated (3 named context parts) | the instruction field of the system prompt — 1 layer | Free-text whole-block rewrite inside one `<prompt>` tag. | Effectively nothing: the retry check and one reward branch are both unreachable, so the only live check is "at least 10 characters". No accept/reject; a harmful edit stays in the tree. |
| AgentStream (its Harness arm) | 4 component families; 3 layers in its own arm | system prompt, skills, experience memory | Structured write operations issued as tool calls (`edit_prompt`, `edit_memory`, `add_skill`, `edit_skill`, `delete_skill`); prompt and memory are whole-document replacements. | Argument presence, name existence, duplicate refusal, one prompt edit and one memory edit per task. Rollback only on a crash. No score gate anywhere — `edit_memory` lacks the required-field check its siblings have, so a missing `body` wipes the document. |
| Meta-Harness | none stated (the paper lists prompting, retrieval, memory, orchestration logic) | prompt template, memory, retrieval, orchestration logic, observation processing, command execution (all inside one Python file) | A complete new single Python file per candidate, made by copying an earlier candidate and changing it. | The paper describes a test that imports the file, builds the class and calls its methods on a few examples. The released classification code only imports the file, with a 30-second limit; a candidate that crashes later is scored 0. |
| Agent-Harness-Survey | 7 layers | none (survey) | n/a | Recommends benchmark regression evaluation triggered by each harness change; names consistency checking of policy files as a gap nobody has filled. |
| Harness-Anatomy | 7 subsystems plus 2 cross-cutting surfaces | none (read-only source audit of 11 systems) | n/a | Documents others' checks: a human review inbox, agent review of a diff under isolation, character caps on model-written memory, parse-time test cases inside policy files. |

### Per layer: how many references change it

| Layer | References whose method changes it | Named but not changed |
|---|---|---|
| long-term memory | 12: AHE, Adaptive-Auto-Harness, AgentStream, HCL, Meta-Harness, ALMA, Evo-Memory, EvolveMem, MemEvolve, MemRL, MemSkill, ReMe | HarnessX |
| system prompt | 8: AHE, Adaptive-Auto-Harness, HarnessX, HCL, AgentStream, SIA, Meta-Harness, Learning-to-Self-Evolve | Harness-R1, MemEvolve, ReMe, Evo-Memory, both surveys |
| tool implementation | 5: AHE, Adaptive-Auto-Harness, HarnessX, SIA, Meta-Harness | HCL |
| skill | 4: Adaptive-Auto-Harness, AgentStream, MemSkill, HCL | AHE, Harness-R1, HarnessX |
| middleware | 3: AHE, HarnessX, Harness-R1 | Adaptive-Auto-Harness, both surveys |
| tool description | 2: AHE, Adaptive-Auto-Harness | Harness-R1, HCL, HarnessX, Meta-Harness, both surveys |
| weights of the task model | 2: HarnessX, SIA | stated as frozen in the others |
| sub-agent configuration | 0 | AHE, Adaptive-Auto-Harness, HarnessX, Harness-Anatomy |

SIA and Meta-Harness each change one Python file that holds several of these parts together; they are counted under each part the file contains.

### What the references agree on

- Memory and the system prompt are the parts changed most often. Seven of the references (ALMA, Evo-Memory, EvolveMem, MemEvolve, MemRL, MemSkill, ReMe) change one memory or skill store and nothing else.
- No reference changes sub-agent configuration in its experiments, although four name it.
- The number of parts a paper names is larger than the number it changes. AHE names seven and its released evolved harness contains five: skills and sub-agents were left as in the seed. HarnessX names nine dimensions and logs four kinds of edit. Harness-R1's patch format declares six action types and the training configuration allows one.

### Where they differ

- Edit form. Most methods write whole files or whole text blocks (AHE, Adaptive-Auto-Harness, HarnessX, SIA, Meta-Harness, ALMA, Learning-to-Self-Evolve). A smaller group uses a fixed list of operations with typed arguments (AgentStream, MemSkill, EvolveMem, ReMe, MemRL). Harness-R1 alone writes a JSON object that carries Python source for a named function.
- Check before installation. Harness-R1 has the strictest check on form: a schema, a list of allowed and forbidden Python constructs, size limits, and time and line limits when the function runs. It has no check on whether earlier tasks still pass. HCL has the strictest check on effect: an edit must improve the current task and must not fail any previously solved case. SIA, Evo-Memory and MemRL install every edit with no check.
- Checks described and checks run. AHE ships a validator its loop never calls. Adaptive-Auto-Harness ships a typed write interface its editor never uses. HarnessX's contract check runs in warning mode and covers two of eight hook points. Meta-Harness's released check is weaker than the one its appendix describes.

### What it implies for this project

The researcher chose, on 2026-10-01, to start with five layers: system prompt, tool, middleware, skill, long-term memory. Tool covers a tool's description and its implementation together, changed by one update-tool operation. Sub-agent configuration is left out because no reference changes it. Weights of the task model are left out. Edits are to be written as a patch in JSON against a harness built to accept such patches, so that each edit can be parsed and checked automatically, following the Harness-R1 form and extending it from one layer to five. `docs/framing.md` is the authority for the layer list; this file is the evidence behind it.

Three facts from the survey bear on that design. First, Harness-R1's checker forbids file access and imports, so a tool whose code reads or writes outside the process cannot pass it unchanged. Second, AHE, Adaptive-Auto-Harness and HarnessX each require a registry file to be updated when a component is added; if the add and remove operations perform the registration, the registry does not become a sixth thing to edit. Third, every check found in the references tests form, size, duplication or a replayed call; none detects a new entry that contradicts an existing one.
