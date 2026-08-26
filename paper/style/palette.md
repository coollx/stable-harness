# House Palette (fixed across all papers)

This six-color pastel palette, with its pinned darker same-hue stroke companions, is the fixed palette for ALL papers, not a per-paper choice.
Assign colors to series in the order below and never cycle or generate extra hues; a method/series keeps its color pair across every figure in a paper (color follows the entity, not its rank within one plot).
Preview sheet: `palette-preview.png` in this folder.

| # | name | fill (pastel) | stroke (companion) |
|---|------|---------------|--------------------|
| 1 | pale yellow | `#F8E8A4` | `#A28A05` |
| 2 | pale lavender | `#D8D0F0` | `#745EA5` |
| 3 | sage green | `#98B898` | `#5B995D` |
| 4 | dusty rose | `#C8A0A0` | `#9F4C51` |
| 5 | gray-blue | `#A8ACB8` | `#848998` |
| 6 | warm gray | `#D0D0C7` | `#58584E` |

Slot 6 (warm gray) is preferred as the neutral: `#D0D0C7` for gridlines and background/"other" shading, `#58584E` for reference lines and annotation ink.

```python
FILL = ["#F8E8A4", "#D8D0F0", "#98B898", "#C8A0A0", "#A8ACB8", "#D0D0C7"]
STROKE = ["#A28A05", "#745EA5", "#5B995D", "#9F4C51", "#848998", "#58584E"]
```

## Usage rules

- Line plots draw the line in the stroke color, with the pastel reserved for confidence bands and area fills.
- Bars and filled markers use the pastel as face color with the stroke color as edge (`linewidth ~ 1.4`).
- Grid and axes stay recessive (light gray, thin); never a dual y-axis.
- Which color pair a single-series figure uses is the researcher's per-figure choice.
- More than six series: never extend the palette. Split into small multiples, or gray out all but the two or three series the sentence argues about (`#D0D0C7`) and label them directly, or move the comparison to a table.

## Accessibility contract

The companions were derived in OKLCH: same hue, lightness lowered in alternating tiers so adjacent series also differ in lightness, chroma capped to keep the muted feel.
The stroke set passes the dataviz-skill palette checker (re-run it after any palette change; lightness band, >=3:1 contrast on white, worst adjacent colorblind delta-E 9.9, worst normal-vision delta-E 16.2; the two gray companions read gray by design).
The pastel fills alone are intentionally low-chroma and light and FAIL the same checker (worst adjacent pair `#98B898` vs `#C8A0A0` has deutan delta-E 1.2; all six are below 3:1 contrast on white), so color is never the only encoding: every multi-series plot must also differ per series in marker shape AND line style, and carry a legend or direct labels.
This redundancy rule is what satisfies the scipilot figure skill's colorblind-safety requirement; ignore that skill's default colorblind palette.
