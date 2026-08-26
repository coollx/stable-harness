#!/bin/bash
# Session-start context injection: project status for a fresh agent session,
# plus silent self-repair of the git hook path and the harness memory symlink.
set +e
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$ROOT" || exit 0

# self-heal: the commit gate lives in-repo
git config core.hooksPath .claude/hooks

# self-heal: harness memory path -> docs/memory (slug = cwd with / _ . as -)
mkdir -p "$ROOT/docs/memory"
SLUG=$(pwd | sed 's|[/_.]|-|g')
MEM="$HOME/.claude/projects/$SLUG/memory"
if [ -L "$MEM" ]; then
  ln -sfn "$ROOT/docs/memory" "$MEM"
elif [ ! -e "$MEM" ]; then
  mkdir -p "$(dirname "$MEM")" && ln -s "$ROOT/docs/memory" "$MEM"
elif [ -z "$(ls -A "$MEM" 2>/dev/null)" ]; then
  rmdir "$MEM" && ln -s "$ROOT/docs/memory" "$MEM"
else
  echo "note: $MEM holds real files; move them into docs/memory/ and relink."
fi

# regenerate the memory index from file descriptions (agents never hand-edit it)
if ls docs/memory/*.md >/dev/null 2>&1; then
  { echo "# Memory index (generated from each file's description: line)"
    for f in docs/memory/*.md; do
      b=$(basename "$f"); [ "$b" = "MEMORY.md" ] && continue
      d=$(grep -m1 '^description:' "$f" | sed 's/^description: *//')
      echo "- $b — ${d:-no description}"
    done
  } > docs/memory/MEMORY.md
fi

echo "== project status ($(date +%F)) =="
for f in docs/tasks/T-*.md; do
  [ -e "$f" ] || continue
  st=$(grep -m1 '^status:' "$f" | awk '{print $2}')
  case "$st" in done|killed) continue ;; esac
  id=$(basename "$f" | grep -oE '^T-[0-9]+')
  last=$(grep -h "· $id ·" docs/log/*.md 2>/dev/null | tail -1 | awk -F' · ' '{sub(/^- /,"",$1); print $1}')
  echo "  ${st:-?}  $(basename "$f" .md)  last spine line: ${last:-none yet}"
done
SPINE=$(ls docs/log/2*.md 2>/dev/null | tail -1)
if [ -n "$SPINE" ]; then
  echo "-- last spine lines ($SPINE):"
  tail -5 "$SPINE" | sed 's/^/  /'
fi
echo "-- uncommitted: $(git status --porcelain | wc -l | xargs) file(s)"
git status --porcelain | cut -c4- | cut -d/ -f1 | sort | uniq -c | sort -rn | sed 's/^/  /'

# stray-file baseline: prune resolved entries, never absorb unresolved ones;
# reset the once-per-session shown list
SNAP=.git/harness-strays
touch "$SNAP"
git status --porcelain | grep '^??' | cut -c4- | sort | comm -12 "$SNAP" - > "$SNAP.tmp" && mv "$SNAP.tmp" "$SNAP"
: > .git/harness-strays-shown
exit 0
