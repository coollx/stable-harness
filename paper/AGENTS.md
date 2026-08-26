# paper/AGENTS.md — the paper rulebook

This folder holds everything for the project's paper. Read this file before responding to any paper-related request; where it conflicts with the root AGENTS.md, this file wins for paper work.
The LaTeX source lives in `overleaf/`, an independent git clone whose remote is shared with co-authors; everything else in this folder is tooling and never syncs to them.
Fill every `<...>` slot when the paper phase starts; delete slots that do not apply.

## The paper

Title: **"<TITLE>"**. Venue: <VENUE> <YEAR>, <SUBMISSION-KIND: anonymous / single-blind / camera-ready>; deadlines: <DEADLINES>.
If the paper names an artifact (method, system, benchmark): **<NAME>**, written `\name{}` in all .tex files; never type the letters directly. Delete this line if the paper names nothing.
Story: <ONE-PARAGRAPH: setting, core idea, how it is established (training / study / proof / measurement), domain, headline evidence>.
Evidence source: <path — code repo, study data, proof file, dataset>. All paper claims, numbers, methods, and protocol details are grounded in that source's actual artifacts; read the primary artifact before describing any protocol; never invent or approximate. When numbers are banked in the paper and declared frozen, the paper is the source of record and the evidence source is consulted only for genuinely new material.

## Linking Overleaf (once, at writing kickoff)

1. `git clone <overleaf-git-url> paper/overleaf` (the outer repo already ignores `paper/overleaf/` and `paper/local_workspace/`).
2. Write `paper/overleaf/.git/info/exclude` with LaTeX build junk and OS junk only (`*.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk *.synctex.gz .DS_Store`); all tooling lives outside the clone, so nothing else can leak.
3. Resolve venue packaging now and record it in the style guide bindings: length limit and what counts against it; whether extra material is an appendix in the PDF, a separate upload, or forbidden; required checklists or statements; anonymity rules; submission file formats.
4. Restructure the main .tex into thin shell + `sections/` (one file per chapter, `\input` lines in the shell); for a project with existing co-author content, coordinate first or keep the monolith with the same conventions.
5. Add to the shell preamble: `\newcommand{\name}{<NAME>}` and `\newcommand{\note}[1]{\textcolor{red}{#1}}` (draft slots; all `\note{}` removed before submission).

## Sync (`/paper:push`)

- `git -C paper/overleaf pull --rebase <REMOTE> <BRANCH>` first every session; web-editor edits land continuously.
- Push only through `/paper:push`: pull, stage, show the staged set, confirm with the researcher, then commit and push. Never add `Co-Authored-By` or any AI-attribution trailer; the commit log is shared with co-authors.
- Compile-check (`/paper:compile` or `bash paper/compile.sh`) before every push.

## Working contract

discuss → plan → approve → draft in chat → approve → file write.
- A question is never edit authorization; discussion turns get chat answers only. Edit `overleaf/` sources, captions included, only on an explicit write instruction.
- Declare deviations from an approved plan; an edit request authorizes that change only, and deleting an existing clause needs separate approval, especially the unique carrier of a fact.
- Researcher edits are authoritative: diff file state before revising anything drafted earlier; never reintroduce deleted wording.
- The contract is artifact-agnostic: slides, figures, and appendix prose follow it exactly as main-text prose.

## Style

`paper/style/style-guide.md` governs all prose. §1 Core is fixed; every later section is appended only by an approved `/paper:style-synthesis` proposal. Any style-guide edit — core or appended — is proposed in chat and written only on explicit approval, never opportunistically mid-draft.
At writing kickoff, run `/paper:style-synthesis` on the paper's reference works to build the guide's paper-specific sections; re-run it whenever new reference works arrive.
The press-conference-principle skill governs every paper-facing selection decision — what to include, emphasize, or omit, in prose, figures, experiment selection, and review responses; invoke it whenever shaping paper content.
Where this paper's style guide and a writing skill disagree, the style guide wins.

## Grounding and verification

- Never attach a citation lineage to a mechanism without evidence it is that work's contribution; never describe a protocol from what would be methodologically normal.
- Recompute every derived quantity (deltas, averages, any sentence summarizing a table) from its source cells before writing it; numbers pasted from other sessions are unverified inputs.
- **Researcher-supplied numbers are authoritative.** When the researcher provides or edits a result — their own, or a collaborator's from outside the evidence source — enter it exactly as given; never refuse, demand local verification, or treat absence from the evidence source as fabrication. Acknowledge the number as researcher-supplied, and report (without blocking) any inconsistency it creates with numbers already in the paper.
- Any new number sharing an entity with a frozen table is reconciled against the published value, and the reconciliation reported unprompted.
- A correction generalizes: fix every sibling item in the same pass, grep the whole tree for every phrasing of a corrected fact, and propose the style-guide binding in the same turn.
- Verification means the artifact the researcher opens: after any float, figure, or layout edit, compile, refresh the PDF, and visually read the rendered result before reporting done.
- Before showing any draft: numbers recomputed in-session; reconciled against frozen tables; press-conference pass (no defensive sentences, no generality claims from null results, no bookkeeping quantities in prose); register diffed against the named template section; no forward references.
- Every bib entry is verified against its primary source (title, full authors, published venue and year, identifier) before submission.
- Talks and other derived artifacts are a subset of the paper: one idea per slide, every claim traceable to a paper sentence, no new explanations; reuse tables and figures by cropping the compiled PDF.

## Figures

- One folder per figure under `local_workspace/figures/<name>/`: one CSV, `derive.py`, `plot.py`, `caption.tex`, a minimal README, `out/`.
- `derive.py` is the verification gate: it reads the figure's source of truth in place (run artifacts, study data, published table), recomputes the numbers, and asserts them against an independent statement of them before writing the CSV; on failure nothing is written. `plot.py` reads only the CSV.
- "Replot" runs `plot.py` only; the CSV is researcher-controlled — report derive/CSV mismatches, never overwrite.
- Before any new figure or substantive revision, ask alignment questions in batches of about four, at least one batch; axis labels + ranges and data provenance always require explicit approval.
- Colors come from `paper/style/palette.md` only; render at the venue's column size in the venue's body font family (record both in the style-guide bindings); every render passes the figure checker and a visual read before being called done.
- Caption edits are paper text: propose wording in chat, never edit unasked.
- Schematic and pipeline diagrams use the drawio skill; once the researcher hand-edits a `.drawio`, that file is canonical and is never regenerated.

## Compile

`bash paper/compile.sh` builds the shell from `overleaf/` and reports pages, undefined citations, and bibliography warnings; set MAIN and ENGINE at its top. It runs with zero model inference via `! bash paper/compile.sh` or any terminal; `/paper:compile` is the wrapper for when the result should be interpreted.
