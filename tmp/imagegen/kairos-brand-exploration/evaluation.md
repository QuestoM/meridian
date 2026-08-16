# Kairos generated-mark exploration

Mode: built-in `image_gen` (two calls; no CLI fallback). Both returned 1254×1254 RGBA PNGs with zero-alpha corners, so no chroma-key removal was necessary. Production code was not changed.

## Evaluation layout

The light and dark comparison sheets use the same ordering:

- Rows, top to bottom: current code-native `KairosMark`, Candidate A, Candidate B.
- Columns, left to right: 16 px, 34 px, 44 px, 48 px.
- Each icon was first rendered on a 64×64 canvas at the exact target size with Lanczos reduction, then enlarged 8× with nearest-neighbor solely to expose pixel behavior.
- Comparison color was applied from each asset's alpha mask to test monochrome reversal consistently; this does not repair or vectorize either generated source.

## Measured opaque bounds

| Mark | Source canvas | Alpha bounds | Fraction of square box | Approx. visible bounds at 16 px |
| --- | ---: | ---: | ---: | ---: |
| Current | 512×512 exact SVG render | 384×416 | 0.750×0.813 | 12×13 px |
| Candidate A | 1254×1254 RGBA | 432×960 | 0.344×0.766 | 6×12 px |
| Candidate B | 1254×1254 RGBA | 400×880 | 0.319×0.702 | 5×11 px |

## Verdict

REJECT both generated candidates. Candidate A becomes two thin slashes at 16 px, loses the splice, and reads as a lightning/H-like construction at larger sizes. Candidate B's negative tracks collapse at 16 px and resolve as a digital H/4-like letterform at larger sizes. Both are raster assets with generated edge noise and require alpha-mask tinting to reverse color; the current two-path inline SVG remains clearer, broader, deterministic, `currentColor`-adaptable, and substantially cheaper to ship.

## Final prompts

### Candidate A

```text
Use case: logo-brand
Asset type: compact brand mark for a CNN-grade broadcast operations dashboard, displayed at 16, 34, 44, and 48 pixels
Primary request: create exactly ONE original abstract symbol that encodes two parallel schedule rails, one decisive editorial splice, and uninterrupted continuity through the splice. The concept should feel like a precise timeline handoff: two tall narrow rails offset at the center and locked together by one angular cut, with an asymmetrical silhouette that is unmistakable even when tiny.
Scene/backdrop: perfectly flat solid #00ff00 chroma-key background for local background removal
Subject: one centered standalone geometric mark only
Style/medium: strict vector-logo geometry rendered as a flat solid near-black silhouette; minimal; hard-edged; broadcast-control seriousness; no line-art detail thinner than one eighth of the mark width
Composition/framing: square canvas, mark occupies about 64% of canvas height, generous even padding, upright orientation
Color palette: subject is a single uniform near-black #11110f only; background #00ff00 only
Constraints: exactly one mark, strong distinctive silhouette, balanced negative space, no internal texture, no shadow, no mockup, no wordmark, no extra symbols; retain meaning and clean edges after reduction to 16px; background must be one uniform #00ff00 with no shadows, gradients, texture, reflections, floor plane, lighting variation, or antialias haze; do not use #00ff00 in the subject
Avoid: letters, monograms, typography, play triangle, broadcast waves, circles, rings, rounded blobs, gradients, glow, mascots, generic AI motifs, infinity symbols, chain links, film strips, clocks, arrows, chevrons, brackets, multiple concepts, presentation boards, watermark, 3D, bevels
```

### Candidate B

```text
Use case: logo-brand
Asset type: compact brand mark for a CNN-grade broadcast operations dashboard, displayed at 16, 34, 44, and 48 pixels
Primary request: create exactly ONE original abstract symbol for schedule control: a single compact upright slab whose negative space contains TWO parallel vertical schedule tracks; at one centered editorial cut, the tracks exchange sides through a single square stepped splice, then continue cleanly. The result should communicate ordered blocks before and after one edit while remaining one coherent continuity mark.
Scene/backdrop: perfectly flat solid #ff00ff chroma-key background for local background removal
Subject: one centered standalone geometric mark only
Style/medium: strict flat vector-logo geometry; solid uniform cream #f2eee3; hard square corners; high-reliability broadcast-control character; extremely reduced, with no detail thinner than one sixth of the mark width
Composition/framing: square canvas, upright mark occupying about 62% of canvas height, generous even padding
Color palette: subject uses exactly one uniform cream color #f2eee3; background uses exactly one uniform #ff00ff
Constraints: exactly one mark; one compact silhouette with two broad negative-space tracks and one stepped edit seam; optically balanced but slightly asymmetrical; clean enough to redraw as a handful of rectangles; legible at 16px; background perfectly uniform with no shadow, gradient, texture, floor, reflection, or lighting variation; do not use #ff00ff in the subject
Avoid: any recognizable letter or monogram (especially H, K, N, S, or Z), lightning bolt, arrow, chevron, brackets, play triangle, broadcast waves, circles, rings, rounded shapes, gradients, glow, mascots, generic AI motifs, infinity symbols, links, film strips, clocks, multiple concepts, presentation board, wordmark, typography, watermark, 3D, bevel, texture
```
