---
name: ref
description: Use when adding a reference paper to the repo, or analyzing/comparing reference papers and their code, or synthesizing a cross-paper literature review. Given a paper (PDF path, arXiv ID/URL, any URL, or just a title to search), "/ref add" creates refs/related-work/<Name>/ holding paper.pdf, a structured paper_analysis.md, the cloned code in repo/, and a repo_analysis.md, then updates refs/index.md. "/ref lit-review <topic>" synthesizes across papers into refs/lit-review/. ALSO use this skill — not ad-hoc grepping — whenever the researcher says "add ref", "add a reference/paper", "analyze this paper", "analyze repo", "how does <repo> implement <X>", "compare repos", or "find the <component> in <repo>". This is the single entry point for everything under refs/.
---

# /ref

This skill manages everything under `refs/`. Anything that writes there is a [reference] operation regardless of how it was invoked. Modes: **add** (primary), **lit-review**, **compare**, **quick-find**.

## Project context

For the project's goal and what to prioritize when analyzing papers/repos, see `docs/framing.md` (the research identity). That is the single source — do not restate it here.

## Reference folder layout

```
refs/
  index.md                     <- single cross-reference index (one row per ref)
  related-work/                <- per-paper refs (one self-contained folder each)
    <Name>/
      paper.pdf                <- the paper (renamed to paper.pdf; original title kept in the analysis header + index)
      paper_analysis.md        <- structured analysis (method OR empirical prompt)
      repo_analysis.md         <- repo-analysis-protocol output (omit entirely if no code exists)
      repo/                    <- the cloned code repository (gitignored)
  lit-review/                  <- cross-paper reviews/syntheses (docs that span MANY refs, not one)
```

`<Name>` is a short, clean, human-readable folder name (e.g. `Adaptive-Planner`); every reference is one self-contained folder under `refs/related-work/`. There is no global accumulator file — `refs/index.md` is the place to scan across refs. Every `refs/related-work/*` glob below (e.g. compare-mode's `refs/related-work/*/repo/`) ranges over ref folders only; lit-reviews are never `<Name>/` folders and never table rows.

# Primary mode: add a reference

Input may be a local PDF path, an arXiv ID or URL, any web URL, or just a paper title. Run these steps in order.

## Step 1 — Resolve the input to a PDF + title

| Input form | Action |
|---|---|
| Local `.pdf` path | Use it directly. |
| arXiv ID or `arxiv.org` URL | Download the PDF (`https://arxiv.org/pdf/<id>`). |
| Other URL (landing page or PDF) | Fetch it; if a landing page, resolve to the PDF link and download. |
| Just a title / description | Search (deepxiv CLI or web) to find the paper, then download. |

**On ambiguity, confirm with the researcher before proceeding** — multiple plausible search hits, an unclear ID, or you're unsure which paper they mean. Don't guess silently. Record the source (URL/arXiv ID) — it goes in the analysis header and index.

## Step 2 — Name the ref and create the folder

Derive a short folder name `<Name>` from the paper's short title or method name (PascalCase or kebab, matching existing entries). If the name is unclear or could collide with an existing folder, confirm with the researcher. Then create `refs/related-work/<Name>/` and save the PDF as `paper.pdf`.

## Step 3 — Read the paper and classify its type

Read the PDF with the built-in Read tool — it renders each page as text and image, preserving tables, equations, and figures. Only fall back to a pdf-processing skill if Read returns garbled/empty text (a scanned PDF needing OCR). Then decide whether it is a **method/system paper** or an **empirical-study/findings paper**:

- **Method/artifact/pipeline/system** → the contribution is something built (a model, pipeline, algorithm, system) others could reproduce.
- **Empirical study / findings** → the contribution is what was learned (a probing/behavioral/scaling/failure-mode study, benchmark, measurement, ablation characterization).

Auto-classify, announce your choice in one line, and proceed unless the researcher corrects you. Many papers are hybrids — pick the prompt matching the dominant contribution, follow it as the backbone, and fold the secondary lens in where it fits. State which you chose and why in one line.

## Step 4 — Write `paper_analysis.md`

Read the matching prompt file and follow it exactly to produce `refs/related-work/<Name>/paper_analysis.md`:

- Method paper → `references/method-paper-prompt.md`
- Empirical paper → `references/empirical-paper-prompt.md`

Start the file with a header recording original title, authors, venue/year, and source URL. Both prompts end with a **Reviewer Reception** section: search OpenReview for the paper (title + venue); if a public page exists, embed its link in the header and summarize the reviews/decision/rebuttal there; if not, note that you searched and found none. Never fabricate reviews.

## Step 5 — Find the code repository

1. Scan the paper text (already extracted in Step 3) for a GitHub/GitLab URL — abstract, "Code available at" lines, first-page footnotes, links sections.
2. If none found, web-search `"<paper title>" github code`.
3. If still nothing, treat the paper as code-unavailable — skip Step 6, note "no public code" in the analysis and the index, and finish at Step 7.

Before cloning, briefly confirm the repo URL looks right (matches the paper/authors). Then `git clone <repo-url> refs/related-work/<Name>/repo`. If the researcher already has the code locally, accept that path and move it there instead.

## Step 6 — Analyze the repo

Follow `references/repo-analysis-protocol.md` to produce `refs/related-work/<Name>/repo_analysis.md`.

## Step 7 — Update `refs/index.md`

Upsert a row for this ref. Column order: Ref, Paper, Type, Code, Summary, Analyses. Column guidance: **Ref** — folder name. **Paper** — original title (italicized), linking the local `related-work/<Name>/paper.pdf`; the upstream URL goes in the Summary or Code column. **Type** — `method` or `empirical`. **Code** — ✅ linking the local clone `[repo/](related-work/<Name>/repo)` plus the upstream `[src](<url>)`, or ❌ if unavailable. **Summary** — 2–4 sentences covering both the paper's contribution AND what the code provides — this is the human-scannable heart of the index, make it genuinely informative. **Analyses** — links to both analysis files (omit the repo link if no code). Rows ordered by date added, newest at the bottom.

## Step 8 — Report

Tell the researcher what was created (folder path, classification, whether code was found) and surface anything notable from the analyses relevant to this project. Spine line + commit via the owning task.

# Mode: lit-review

`/ref lit-review <topic>`: cross-paper synthesis into `refs/lit-review/<slug>.md`. If scope is unclear, interview first (batches of 3 questions, each with a recommended answer). Structure: question · papers covered (linking their `related-work/` rows; add missing ones via `add` first) · synthesis — what the field agrees on, where it conflicts, what it implies for this project. List it in the index's lit-review section, never as a table row.

# Mode: compare repos

Trigger: "compare repos", "compare how A and B do X". Produce a comparison table across the cloned repos in `refs/related-work/*/repo/`: rows = aspect / location (file:line) / method / key difference, one column per ref, ending with a recommendation given this project. Lean on the existing `repo_analysis.md` files first; only re-grep the code for details they don't cover. Offer to save the comparison into the relevant `repo_analysis.md` files or `refs/lit-review/`.

# Mode: quick-find

Trigger: "where is X in repo Y", "how does Y implement Z". Skip the full workflow: check `refs/related-work/<Y>/repo_analysis.md` first — it may already pinpoint the component; otherwise grep within `refs/related-work/<Y>/repo/`. Explain what it does with file:line references and how it could be adapted for this project.
