---
name: framing-deck-pipeline
description: "How the scratch/framing-presentation0826 deck is built — ppt-master quick-generate commands, SVG conventions, and a local PNG preview trick"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 9b1c3350-a98e-47a9-b23e-79cab10c533b
  modified: 2026-08-26T13:21:28.723Z
---

The framing deck lives at `scratch/framing-presentation0826/framing-deck_ppt169_20260826/` (svg_output/NN_slug.svg, notes/total.md, icons/tabler-outline/, exports/). Built with ppt-master at `/home/xiang_li/ppt-master/skills/ppt-master/scripts/` (`S` below), quick profile.

**How to apply:** add a slide = write `svg_output/NN_slug.svg` (1280×720, Arial, palette purple #745EA5/#D8D0F0, grey #A8ACB8/#848998, yellow #F8E8A4/#A28A05, green #98B898/#5B995D, red #C8A0A0/#9F4C51, text #58584E; header label 16px at y=74, title 36px bold at y=124, page number at (1208,688)), append `# Slide NN` to `notes/total.md`, then: `python3 $S/icon_sync.py <project> tabler-outline/<name>…` for new icons; `python3 $S/preset_shape_svg.py render chevron|rightArrow|leftRightArrow --id … --frame x y w h --fill … --stroke …` for arrows; `python3 $S/total_md_split.py <project>`; `python3 $S/svg_quality_checker.py <project> --quick-generate --stage final --json` (must be error-free — checker estimates text width, ~9px/char at 18px; ~20px/char at 36px bold); `python3 $S/svg_to_pptx.py <project> --quick-generate --with-notes -a auto`. Formula blocks: `data-pptx-replace-with="formula"` with JSON metadata — LaTeX backslashes must be doubled in the JSON (`\\;` not `\;`) or the checker rejects it.

No LibreOffice on this machine. Preview a slide by inlining `<use data-icon>` icons as stroked `<g transform="translate scale(w/24)">` groups and rendering with pymupdf (`fitz.open(stream=svg, filetype='svg')`), then Read the PNG.
