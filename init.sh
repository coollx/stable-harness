#!/bin/bash
# One-time project setup. Usage: ./init.sh <package-name> [project-name]
set -e
PKG=${1:?usage: ./init.sh <package-name> [project-name]}
PROJECT=${2:-$(basename "$(pwd)")}

mkdir -p "$PKG" docs/log docs/tasks docs/decisions docs/memory \
         refs/related-work refs/lit-review experiments analyses scratch
touch "$PKG/__init__.py" scratch/scratch_pad.md

sed -i "s/{{PROJECT}}/$PROJECT/g; s|{{PACKAGE}}|$PKG|g" AGENTS.md .claude/CLAUDE.md .claude/skills/*/SKILL.md

# the template's README describes the harness; the project gets its own stub
printf '# %s\n\nRun under the research-harness workflow — AGENTS.md is how we work; docs/framing.md is what we work on.\n' "$PROJECT" > README.md

[ -d .git ] || git init
git config core.hooksPath .claude/hooks
chmod +x .claude/hooks/* init.sh

# harness memory path for this machine -> docs/memory
SLUG=$(pwd | sed 's|[/_.]|-|g')
MEM="$HOME/.claude/projects/$SLUG/memory"
if [ ! -e "$MEM" ]; then
  mkdir -p "$(dirname "$MEM")" && ln -s "$(pwd)/docs/memory" "$MEM"
fi

command -v deepxiv >/dev/null 2>&1 || echo "note: deepxiv CLI not found — /ref uses it for paper search; install it when convenient"

MONTH=$(date +%Y-%m)
touch "docs/log/$MONTH.md"

cat > docs/tasks/T-001-init.md <<EOF
# T-001 initialize research harness
status: done
context: new project created from the research-harness template.
goal: working harness — rulebook, hooks, skeletons, first commit through the gate.
plan:
  1. [meta] run init.sh — AGENTS.md, hooks, skeleton stores
EOF

echo "- $(date +%F) · T-001 · [meta] initialized from research-harness template · AGENTS.md" >> "docs/log/$MONTH.md"

git add AGENTS.md README.md .gitignore .claude \
        "docs/log/$MONTH.md" docs/tasks docs/framing.md refs/index.md "$PKG" paper
git rm -q --cached --ignore-unmatch -- "$0"
git commit -m "T-001: initialize research harness"

echo ""
echo "done. next steps:"
echo "  1. fill docs/framing.md (the research identity) — the first real session's job"
echo "  2. start working: open Claude Code and say what you want"

# one-time script: never part of the project it creates
rm -- "$0"
