---
name: style-synthesis
description: Synthesize paper-specific style-guide sections from reference works — study their narration, structure, claims, wording, floats, and venue mechanics, then propose evidence-backed rules for the researcher's approval. Invoke at writing kickoff with the initial reference corpus, and again whenever new reference works arrive.
---

# Style Synthesis

Turn a corpus of reference works into appended sections of `paper/style/style-guide.md`.
Incremental by design: the first run builds the paper-specific sections; later runs study new works and propose additions or revisions to existing rules.
Nothing is written to the guide without the researcher's explicit approval of the proposed text.

## Inputs

From the researcher: the reference works (files or identifiers; or approval of a candidate list you gather via the deepxiv skill), the venue and its author kit path, and any tier assignments.
Tier the corpus before reading: **closest-topic** works supply logic, structure, and argument; a **house-voice** tier (the group's or advisor's own papers, when given) supplies tone and register only — never mine it for structure.

## Step 0 — concept ontology and narrative spine

Classify the paper against the two field priors, then verify the prior against the actual corpus; the corpus overrules the prior.

- **AI/ML prior.** Organizing concepts: prior-work gap, method/mechanism, training or inference recipe, benchmarks and metrics, baselines, ablations, comparison to the state of the art. Chain: gap → approach → headline results → ablation support. Typically absent: participants, design requirements, qualitative findings.
- **HCI/HAI prior.** Organizing concepts: motivating challenges (from literature or a formative study), design goals/requirements, the interaction concept, system features mapped to requirements, study design (participants, tasks, measures), quantitative and qualitative findings, discussion with implications for design. Chain: challenge → requirement → feature → study evidence → implication, traceable end to end. Typically absent: state-of-the-art deltas, ablation grids, benchmark tables.
- **Hybrid** (common in human-AI work): assign an ontology per section — ML neighbors for technical sections, HCI neighbors for study sections — and tier the corpus accordingly.

Output of Step 0, and the spine of everything after it: the field's concept vocabulary, the canonical chain linking the concepts, which concepts the corpus weights heavily, and which it conspicuously never uses.

## Extraction axes

Read the corpus section by section, **verbatim, never from summaries**, and extract rules along these axes, instantiated over the Step-0 ontology:

1. **Narration flow** — the paper-spanning story arc: the order the organizing concepts are introduced, how each section discharges its link in the chain, how the chain stays traceable end to end.
2. **Claim-evidence** — where claims anchor, how headline claims are discharged, quantifier calibration, precision from abstract to results.
3. **Contributions** — format, verb choice, whether claims pair with payoffs or named alternatives.
4. **Wording** — voice, tense, hedging asymmetry, sentence length and rhythm, discourse markers, emphasis discipline.
5. **Section anatomy** — the slot template of each section: what its paragraphs are, in what order, with what openers.
6. **Terminology and naming** — how artifacts are named and expanded, coined constructs, acronym hygiene.
7. **Numbers and statistics** — prose-vs-table division, the uncertainty convention (seeds, intervals, protocol disclosure), units and formats. This axis varies sharply by neighborhood and venue; never inherit it.
8. **Floats** — table style and marking grammar, figure conventions, caption obligations, how float pointers ride claim sentences.
9. **Comparison and weakness rhetoric** — how alternatives are named, credited, and differentiated; the shape of concessions; how negative results are phrased.
10. **Length calibration** — measured sentence and paragraph budgets per section, so "too long" is a number.
11. **Negative space** — what the corpus never does; absence rules carry the same evidence discipline as presence rules.
12. **Notation and math placement** — where formalism lives, glossing conventions, notation minimalism.

## Evidence discipline

A general rule needs at least 3 corpus works; a named variant needs at least 2; every rule carries one verbatim quote locating its evidence, and quotes are spot-checked against the sources before the proposal goes to the researcher.
State each rule as an instruction, not an observation; note the works it came from by name.

## Venue mechanics

Extracted from the venue's actual author kit and call for papers, never from the corpus: length limit and what counts against it, packaging (appendix in PDF / separate upload / forbidden), anonymity rules, forbidden packages and commands, citation style, required checklists and statements, figure and file format rules.

## Output contract

Propose the new or revised guide sections in chat as ready-to-append markdown: the paper-specific sections after §1 Core, ending with a **Bindings** section (fixed terminology, venue mechanics, per-paper style decisions).
The researcher approves, adjusts, or rejects per section; only approved text is appended to `paper/style/style-guide.md`.
On incremental runs, diff new findings against the existing guide: propose additions, and flag any existing rule the new corpus contradicts rather than silently rewriting it.
