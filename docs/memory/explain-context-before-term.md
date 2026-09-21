---
name: explain-context-before-term
description: "Introduce the concrete situation before naming a term (say what a prediction market resolves before saying 'resolution date'); never lean on an unlisted term such as 'score-before-update'"
metadata:
  type: feedback
---

When explaining anything to the researcher, describe the concrete situation first, then attach the name. Example: say "the market asks whether team A wins on Feb 10; the true answer is known only that day" before using "resolution date". Do not use a compact term as if it were already shared vocabulary; if the term is not in the framing.md Terms table, spell out the sentence it stands for.

**Why:** On 2026-09-21 the researcher hit "score-before-update" and "resolution date" cold and could not tell what either meant; the first term had been in framing.md since the initial commit without their explicit approval and was removed the same day.

**How to apply:** In every explanation, context sentence first, term second. Prefer the spelled-out sentence over a coined label.
