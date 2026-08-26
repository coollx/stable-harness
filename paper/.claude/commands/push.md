---
description: Pull, stage the paper source, show the staged set, confirm with me, then commit + push to the shared paper remote
argument-hint: "commit message"
allowed-tools: Bash
---

Sync the paper source in `paper/overleaf/` to its shared remote. All commands run with `git -C paper/overleaf`. Stop and report on any error.

1. `git -C paper/overleaf pull --rebase <REMOTE> <BRANCH>` — grab co-author edits first. On CONFLICT, STOP immediately, show the conflicting files, and stage/push nothing.
2. `git -C paper/overleaf add -A` — safe because all tooling lives outside the clone; only paper content and anything not covered by `.git/info/exclude` can be staged.
3. Show the staged set: `git -C paper/overleaf status --short` and `git -C paper/overleaf diff --cached --stat`. Prominently call out NEWLY tracked files and staged DELETIONS — the only surprises `-A` can introduce.
4. STOP and ask me to confirm. Commit/push nothing until I say yes; drop files with `git -C paper/overleaf restore --staged <file>` and re-show.
5. On confirmation: commit message = `$ARGUMENTS` (if empty, propose one and ask). NO `Co-Authored-By` or any AI-attribution trailer — the commit log is shared with co-authors. Then `git -C paper/overleaf push <REMOTE> <BRANCH>`.
6. Report the pushed commit hash and the remote it landed on.
