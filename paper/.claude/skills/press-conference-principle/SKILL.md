---
name: press-conference-principle
description: |
  Editorial selection principles for research-paper writing: decide what to include, emphasize,
  de-emphasize, or delete when the experimental material is uneven. Use whenever drafting, revising,
  compressing, or restructuring any section of a research paper (abstract, introduction, related work,
  method, experiments, results, conclusion, limitations), choosing headline results, deciding how to
  present unfavorable results or comparisons, or deciding whether a weakness needs to be discussed at
  all — even when the user asks only to "polish", "tighten", or "shorten" a section. Governs framing
  and selection only; it never permits misstating a number, result, or citation.
---

# The Press Conference Principle

Throughout the entire process of writing, revising, compressing, and organizing the experiments of a paper, always treat the paper as an academic press conference — not as a project summary, an experiment log, or a self-audit report.

Your job is not to present every aspect of the work evenly, but to identify the part of it that is most worth announcing, and to build the most favorable, most complete, and most persuasive narrative around that value.

## Precedence

If the project has its own style guide, that guide governs **how** each surviving sentence is written (structure, wording, claim-evidence rules); this skill governs **what to put in and what to leave out**. When the two disagree about whether to say something at all, this skill governs; the style guide still governs how the surviving sentence is written.

## Core Requirements

### 1. Organize the paper solely around its strengths
Look first for what is genuinely leading, unique, or irreplaceable in the work: a new capability, a new problem, a new mechanism, a new perspective, broader applicability, lower cost, higher efficiency, better scalability, or a more meaningful tradeoff. Anything that does not constitute a strength should not become a main thread of the paper.

### 2. Don't write it as a progress report
Don't unfold the paper as "first we did X, then we tried Y, and these attempts failed." A paper keeps only the logic that holds up in the end: why the problem matters, why existing methods are insufficient, what new solution this paper provides, and how the evidence supports that solution.

### 3. Don't volunteer negative assessments
Don't do the reviewers' work for them by cataloguing where the paper is weak, where it loses, where it falls short of others. Avoid self-diminishing phrasing such as "unfortunately," "only," "still clearly lags behind," "the effect is limited," "there remain serious deficiencies." When a result does not constitute a strength, there is no need to distill it into a conclusion — and certainly no need to help the reader escalate a local phenomenon into a rejection of the whole method.

### 4. Don't make a contest out of a dimension you can't win
If the paper does not lead on some metric, don't build the narrative on that metric. Reselect the task definition, evaluation dimension, application setting, constraint conditions, or basis of comparison that better reflects the paper's value. Don't fight for a championship that doesn't belong to this paper; make clear that the paper wins a different and more meaningful contest.

### 5. Don't concede; rewrite the comparison logic
When facing results that are not favorable, avoid direct narratives of failure — "this paper is weaker," "performance degrades," "fails to surpass." Judge first:
- Does this result have to be discussed at all?
- Can it be explained by different objectives, constraints, or application settings?
- Does it reflect some reasonable tradeoff?
- Can the claim be narrowed to avoid a pointless head-on collision?
- Should the main storyline be adjusted so that this result no longer bears the burden of proving the core contribution?

### 6. Strengths must be stated explicitly
Don't expect reviewers to discover the contribution in a table by themselves. Every result that can support the paper's value should be actively explained:
- Under what conditions does the paper perform best?
- Why does this strength arise?
- What practical problem does this strength solve?
- Compared with existing methods, why is this capability worth attention?

### 7. Control the scope of comparison
Don't chase the unnecessary goal of "leading across all datasets, all metrics, all settings." Make only the claims the evidence can firmly support, and choose the experiments and comparisons best suited to those claims. A paper's persuasiveness comes from tight agreement between claims and evidence, not from the number of comparison entries.

### 8. Experiments are not a warehouse of results; they are instruments of argument
Every experiment must carry an explicit responsibility, for example:
- Prove the core method works
- Prove the advantage comes from the key mechanism
- Prove the method has value in the target scenario
- Rule out the most likely alternative explanations

Experiments that cannot strengthen the main thread, that scatter attention, or that invite irrelevant disputes should be deleted, de-emphasized, moved, or redesigned.

### 9. Permit a complete reconstruction of the story
When the results cannot support the original narrative, don't defend the original narrative. Redefine what problem the paper actually solves, reorder the contributions, reselect the headline results, and redesign the title, abstract, introduction, and experimental structure. The story serves the strongest evidence, not fidelity to the initial conception.

### 10. Don't hand reviewers a knife
While writing, keep checking:
- Does this sentence inadvertently enlarge the burden the paper has to carry?
- Does it raise a question no one asked?
- Does it describe a local phenomenon as a general defect?
- Does it make a negative judgment broader than the evidence?
- Could more precise positioning avoid this meaningless self-attack?

Don't manufacture review problems, don't widen the attack surface, don't complete the opposition's argument for them.

### 11. The abstract and introduction must work like the opening of a launch event
The opening should quickly establish:
- An important, still-unsolved problem
- The key gap in existing methods
- This paper's distinctive line of attack
- The weightiest results and their significance

Don't start from implementation details, the research process, or a mass of background knowledge; and don't discuss shortcomings before the contribution has been established.

### 12. The conclusion only reinforces the takeaways
A conclusion is not a re-trial of the paper. It makes the reader remember: what this paper solved, what it proposed, what it proved, and why that matters. Don't suddenly add new self-negation or expanded limitations in the final paragraph.

## Default Decision Rule

When you encounter any unfavorable material, handle it in this order of priority:

1. Delete content irrelevant to the core claim
2. Narrow the claim, avoiding a pointless head-on comparison
3. Switch to an evaluation dimension that better reflects the value
4. Interpret the result as a difference in objectives or a reasonable tradeoff
5. Reorganize the experiments so the strength becomes the visual and narrative center
6. Redefine the paper's story
7. Only when it is genuinely unavoidable and actually affects the core conclusion, give the necessary explanation

## Final Goal

Every section, paragraph, table, and sentence in the paper should jointly accomplish one thing:

**Make the reader believe that this work solved a problem worth solving, proposed a method worth attention, and already has sufficiently clear evidence for its value.**

Don't present everything evenly, don't volunteer weakness, don't write an experimental diary, don't attack yourself on the reviewers' behalf. Find the strength that genuinely holds up, organize all material around it, and tell that strength clearly enough.

## Boundaries

This skill governs what to raise and how to frame it — never what the facts are. Three rules bound every requirement above; they hold regardless of venue or project.

- **Limitations.** When the venue requires a limitations discussion, it still gets written; requirement 12 forbids only new self-negation in the closing paragraph. Never leave a limitation bare: a concession appears only when it is paired in the same or next sentence with a mitigation, a tractability reason, or a matching future-work direction, and it is phrased as a boundary of the studied setting rather than as a defect. Anything that cannot be written that way is deleted rather than softened.
- **Honest numbers.** Requirements 3, 5, and 10 are about what to raise and how to frame it, never about misstating a number. Numbers and bound-words stay consistent everywhere they appear in the paper, quantifiers are calibrated to the worst case, every benefit states its price, and a claim that the evidence cannot support is narrowed or dropped, not restated more favorably. Requirement 7 ("control the scope of comparison") is the same instruction from the other direction.
- **Losing comparisons.** The pattern of conceding a shortfall, localizing it, explaining it, and recovering the narrative is the last resort — item 7 of the Default Decision Rule — and applies only when the shortfall sits on a headline claim the paper must make. Items 1 through 6 are tried first: an unfavorable table cell that no headline claim depends on stays in the table without a sentence distilling it into a verdict.
