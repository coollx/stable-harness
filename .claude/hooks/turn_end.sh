#!/bin/bash
# Close-out check: at each agent turn end, surface untracked files not yet triaged.
# Each needs one verdict: track / ignore / relocate. An unresolved file re-surfaces
# once per session (never silently absorbed into the baseline).
set +e
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$ROOT" || exit 0
SNAP=.git/harness-strays
SHOWN=.git/harness-strays-shown
[ -f "$SNAP" ] || exit 0
touch "$SHOWN"
NEW=$(git status --porcelain | grep '^??' | cut -c4- | sort | comm -13 <(sort -u "$SNAP" "$SHOWN") -)
[ -z "$NEW" ] && exit 0
echo "untracked files needing a verdict (track / ignore / relocate):"
echo "$NEW" | sed 's/^/  /'
echo "$NEW" >> "$SHOWN"
exit 0
