#!/bin/bash
# Zero-inference compile-and-report for the paper. Run with `! bash paper/compile.sh`
# in Claude Code (no model call), or from any terminal. /paper:compile wraps this.
set -e
MAIN="<MAIN>"        # shell .tex basename in overleaf/, without extension
ENGINE="-pdf"        # -pdf (pdflatex) | -xelatex | -lualatex — match the venue kit
PAPER=$(cd "$(dirname "$0")" && pwd)
BUILD="/tmp/paperbuild-$(basename "$(dirname "$PAPER")")"

cd "$PAPER/overleaf"
mkdir -p "$PAPER/local_workspace"
rm -f "$PAPER/local_workspace/$MAIN.pdf"
if ! latexmk "$ENGINE" -interaction=nonstopmode -outdir="$BUILD" "$MAIN.tex" > /dev/null 2>&1; then
  echo "COMPILE FAILED — last 30 log lines:" >&2
  tail -30 "$BUILD/$MAIN.log" >&2
  exit 1
fi
cp "$BUILD/$MAIN.pdf" "$PAPER/local_workspace/"

if command -v pdfinfo > /dev/null; then
  PAGES=$(pdfinfo "$PAPER/local_workspace/$MAIN.pdf" | awk '/^Pages/{print $2}')
else
  PAGES="? (pdfinfo not installed)"
fi
UNDEF=$(grep -c 'Citation.*undefined' "$BUILD/$MAIN.log" || true)
BIBWARN=$(cat "$BUILD"/*.blg 2>/dev/null | grep -cE '^(Warning--|WARN - )' || true)
echo "paper/local_workspace/$MAIN.pdf: $PAGES pages | undefined citations: $UNDEF | bibliography warnings: $BIBWARN"
