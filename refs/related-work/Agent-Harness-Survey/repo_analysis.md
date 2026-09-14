# Repo Analysis: awesome-agent-harness

**Path:** `refs/related-work/Agent-Harness-Survey/repo/` · **Source:** https://github.com/Picrew/awesome-agent-harness · **Analyzed:** 2026-09-14 (commit `91c5381`, pushed 2026-08-31)

## Overview

- **Paper:** Agent Harness Engineering: A Survey (see `paper_analysis.md`)
- **Framework:** none. This is the paper's living catalog, not an implementation: one YAML data file, three Python maintenance scripts, generated bilingual READMEs, and dated verification reports.
- **RL Algorithm:** none
- **Base Model:** none
- **Key artifact:** a curated, data-driven list of 367 agent-harness resources (333 GitHub projects, 90.7%) in 9 practitioner categories, verified for link health on 2026-08-31, with 1,759 GitHub stars at clone time. The paper's Appendix A is a frozen 2026-05-08 snapshot of this catalog, refolded from 9 categories into the 7 ETCLOVG layers.

## File Structure Map

```
README.md                      generated English catalog (132 KB): featured blogs, category table, per-category project tables
README_zh.md                   generated Chinese mirror, same order and entries
CONTRIBUTING.md                edit data/projects.yaml only; run sync, render, verify; commit generated files together
data/projects.yaml             single source of truth: catalog metadata, 9 categories, 367 entries (5,210 lines)
scripts/sync_github_metadata.py  refreshes stars_snapshot / updated_at / license from the GitHub API (token from GITHUB_TOKEN or `gh auth token`)
scripts/render_readme.py         renders both READMEs from the YAML; exports DATA_FILE, REQUIRED_ENTRY_FIELDS, github_ratio, render_readmes
scripts/verify_catalog.py        schema check, GitHub-ratio check, URL health (HEAD then GET, Cloudflare challenge counted as reachable), README mirror check; writes reports/verification/<date>.md
docs/curation_policy.md          scope, inclusion criteria, fixed 9-category taxonomy, data contract
docs/taxonomy_iterations.md      five iterations from generic tool buckets to the harness-first 9-category scheme
docs/sources_and_verification.md, docs/github_query_templates.md   source streams and GitHub search templates
reports/verification/*.md        one report per sync, 2026-03-30 to 2026-08-31 (about 60 files), including 2026-05-08 (the paper's snapshot)
```

## Component Inventory

The ten-component template targets ML training repositories; most components are absent here and are listed as such.

| Component | Status | Location and details |
|---|---|---|
| Entry points | present | Three scripts, run in order: `sync_github_metadata.py` (parallel GitHub API fetches, falls back to `gh api`), `render_readme.py` (YAML to two Markdown tables, entries sorted by stars descending within category, star badges rendered from `stars_snapshot`), `verify_catalog.py` (schema, ratio, links, mirror consistency, report). No CLI arguments beyond the defaults matter for reuse. |
| Data pipeline | present | `data/projects.yaml`. Top-level keys: `catalog` (titles, `last_verified`, descriptions), `categories` (9 entries with `id`, `anchor`, `name_en`, `name_zh`), `entries`. Required entry fields per `CONTRIBUTING.md`: `name`, `repo_url`, `category`, `summary_en`, `summary_zh`, `tags`, `stars_snapshot`, `updated_at`, `why_included`; `license` is also present. There is no ETCLOVG layer field; the paper's per-artifact layer coding exists only in its Table S1. |
| Evaluation | present, non-ML | `verify_catalog.py:44` (`check_url`) treats HTTP 403 with a `cf-mitigated: challenge` header as reachable, which is why vendor blog pages behind bot checks pass. The 2026-08-31 report lists exactly one broken URL: the paper's own OpenReview PDF (HTTP 403). |
| Special token / action handling | absent | not applicable |
| Environment / tool interaction | absent | only the GitHub REST API for metadata |
| Token masking | absent | not applicable |
| Reward function | absent | not applicable |
| RL training loop | absent | not applicable |
| SFT / cold-start | absent | not applicable |
| Multi-GPU / infrastructure | absent | pure Python, `pyyaml` only |

## Catalog snapshot versus the paper's Appendix A

The 2026-05-08 verification report (the paper's frozen snapshot) gives 171 entries, 146 GitHub (85.4%), 142 of 142 GitHub in project categories, 9 categories, 0 broken URLs, matching the paper's Appendix A text. Category counts at that snapshot: Harness Architecture & Orchestration 21, Context & Working-State Engineering 9, Execution Substrates & Sandboxing 18, Protocols, Tool Interfaces & Agent Contracts 11, Evaluation Harnesses & Benchmarks 21, Observability & Reliability Operations 14, Guardrails, Security & Governance 12, Reference Harness Implementations 36, Essential Readings & Ecosystem Maps 29.

The paper's Table S1 reports primary-layer counts E 20, T 12, C 9, L 47, O 15, V 21, G 14 (138 rows). The fold from 9 categories to 7 layers is done outside the repository: the 36 reference implementations are distributed into their dominant layer (most land in L, which rises from 21 to 47), readings are dropped, and 4 of the 142 technical entries do not appear in the table. Category counts on 2026-08-31: 64, 29, 27, 42, 29, 21, 27, 89, 39 in the same order, so the catalog has more than doubled since the paper's snapshot, with the largest growth in reference implementations and protocols.

## Relevance to this project

### Components we can directly reuse

- **The entry list as a related-work scan.** Entries whose summaries or tags describe self-improving, self-evolving, or benchmark-gated harness optimisation, all scored on the current suite or current failures (the $\gamma = 0$ case in our framing):
  - Meta-Harness, https://github.com/stanford-iris-lab/meta-harness, 1.5k stars: automated search over task-specific model harnesses, with memory-system and terminal-agent scaffold experiments.
  - auto-harness, https://github.com/neosigmaai/auto-harness, 534 stars: benchmark-gated optimisation loop that mines failures, edits agent code, and guards against regressions overnight.
  - Agentic Harness Engineering, https://github.com/china-qijizhifeng/agentic-harness-engineering, 859 stars: observability-driven evolution of coding-agent harness components through evaluate, analyse, improve loops (companion to Lin et al., 2026).
  - SkillOpt, https://github.com/microsoft/SkillOpt, 16.5k stars: trains reusable natural-language agent skills through trajectory edits and validation gates.
  - EverOS, https://github.com/EverMind-AI/EverOS, 12.6k stars: self-evolving memory runtime persisting Markdown with local indexes and reusable cases and skills.
  - Hermes Agent, https://github.com/NousResearch/hermes-agent, 238k stars: self-improving agent runtime with memory, skill creation, subagents, scheduled automations.
  - OpenClaw.NET, https://github.com/clawdotnet/openclaw.net, 487 stars: runtime with a governance ledger and harness regression tests.
- **Candidate open harnesses for our task streams.** Small, inspectable loops listed under Harness Architecture & Orchestration and Reference Harness Implementations: mini-swe-agent, SWE-agent, OpenHands SDK, smolagents, Codex CLI, OpenCode. Their layer files (prompts, tool descriptions, skills, memory) are the edit surface a harness agent would touch.
- **The Essential Readings category** (39 entries) is the practitioner corpus the paper leans on: Anthropic, OpenAI, LangChain, Google, Cognition engineering posts. Useful when the paper needs a citation for a production practice.

### Components we need to modify

None. We do not maintain a catalog.

### Components that don't apply

The three maintenance scripts and the bilingual rendering. The verification-report pattern (dated Markdown report per data change) is already covered by our spine.

### Key code snippets worth studying

- `scripts/verify_catalog.py:44` (`check_url`): HEAD-then-GET link check with an explicit allowance for Cloudflare challenge pages; the reason the catalog's link-health numbers stay at zero broken while vendor pages sit behind bot checks.
- `data/projects.yaml` entry for any project, for the `why_included` field: a one-line evidence statement per entry, which is the closest the repository comes to the paper's coding protocol.
