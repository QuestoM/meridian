# UX overhaul asset inventory

Captured from the implementation and production build, then synchronized to the frozen production direction on 16 August 2026. This inventory separates runtime assets from art-direction artifacts; generated concepts and mark explorations are evidence of art direction and do not ship in the application.

## Runtime type

| Asset | Package / version | Loaded subsets and weights | Runtime treatment | Licence reported by package |
| --- | --- | --- | --- | --- |
| Noto Sans Hebrew Variable | `@fontsource-variable/noto-sans-hebrew@5.3.0`, locally copied Hebrew subset | Hebrew variable 100–900 | Local WOFF2 and local `@font-face`; critical Hebrew face preloaded; `font-display: swap` | OFL-1.1, copied beside the asset |
| IBM Plex Sans Hebrew | `@fontsource/ibm-plex-sans-hebrew@5.3.0` | Hebrew 400, 500, 600 | Local compatibility fallback; `font-display: swap` | OFL-1.1 |
| IBM Plex Sans | `@fontsource/ibm-plex-sans@5.3.0` | Latin 400, 500, 600 | Local package import; regular and semibold WOFF2 preloaded; `font-display: swap` | OFL-1.1 |
| IBM Plex Mono | `@fontsource/ibm-plex-mono@5.3.0` | Latin 400, 500 | Local package import; not preloaded; `font-display: swap` | OFL-1.1 |

The imports and three preload links are implemented in [`src/index.jsx`](../../tv-break-dashboard/src/index.jsx). The local Noto source, hash, and licence are documented in [`src/assets/fonts/README.md`](../../tv-break-dashboard/src/assets/fonts/README.md). Font stacks and role assignment are in [`tokens.css`](../../tv-break-dashboard/src/tokens.css) and [`shell/theme.js`](../../tv-break-dashboard/src/shell/theme.js).

The production build emits 17 local font files: the 12.26KB Noto Hebrew variable subset, IBM Hebrew fallback at three weights in WOFF and WOFF2, Latin Sans at three weights in WOFF and WOFF2, and Latin Mono at two weights in WOFF and WOFF2. No remote font request is part of the implementation. Noto Hebrew plus regular and semibold Latin are preloaded; the remaining faces load on demand.

Fallback stacks are:

- Hebrew: `Noto Sans Hebrew Variable`, `IBM Plex Sans Hebrew`, `IBM Plex Sans`, `Arial Hebrew`, `Arial`, `sans-serif`
- Latin: `IBM Plex Sans`, `Noto Sans Hebrew Variable`, `IBM Plex Sans Hebrew`, `Arial`, `sans-serif`
- Identifiers: `IBM Plex Mono`, `Courier New`, `monospace`

Noto was selected for the Hebrew glyph rhythm and variable-weight fit observed in the user's local reference; the subset contains no Latin glyphs. IBM Plex therefore remains the deliberate voice for Latin, figures, dates, and identifiers, avoiding an accidental mixed-script handoff inside numbers or channel names.

## Runtime icons and marks

| Asset family | Source | Treatment | Shipping status |
| --- | --- | --- | --- |
| Interface icons | `lucide-react@1.22.0` | One live icon source; canonical Studio buttons render at a 20px optical box and 1.75px stroke; domain navigation also specifies 1.75px | Ships |
| Kairos master mark | Original local `KairosMark` SVG in `shell/kairos-icons.jsx` | Transparent 32px-grid SVG with exactly two filled `currentColor` paths: opposed frames interrupted by an off-axis negative-space splice; instances at 34px rail, 40px auth check, 44px login, and 48px gate | Ships |
| Desktop-gate mark | The same canonical transparent `KairosMark` | 48px; inherits `currentColor`, with no image request, background plate, or alternate gate logo | Ships |
| Status, loading, focus, timelines, bars | React/CSS/SVG paths already in components | Role-token colour and geometry; no raster dependency | Ships |

Repository inspection found 121 direct `lucide-react` import modules in `src` and no import from Font Awesome, React Icons, Heroicons, or MUI Icons. That proves one general interface-icon source, not a completed wrapper migration. The seven domain glyphs and Kairos mark are original local SVGs; the domain glyphs use the interface stroke grammar, while the two-path master mark deliberately uses solid frame geometry so it remains identifiable at small sizes. Feature modules still import Lucide directly, while canonical actions normalize size and stroke. New features must use the canonical action components and must not introduce a second icon library.

No runtime `<img>`, `<picture>`, or CSS `url(...)` image usage was found in the application source. The selected direction does not require product photography or decorative raster imagery.

## Generated art-direction artifacts

All three concept renders use the same normalized Kairos Today brief and were generated to compare complete visual/interaction systems. They materially influenced the selected palette, density, rail, type hierarchy, and list/detail composition. They are unretouched evaluation artifacts: none is imported by application code, none is presented as a product screenshot, and the generated logo visible in the concepts was not promoted to a runtime asset.

Later image-generation passes explored bow-tie/gradient, equalizer-like, slab-letter, diagonal-rail, and nested-track directions. Every generated candidate was rejected as generic, letter-like, or too fragile after inspection at 16, 34, 44, and 48px. None was copied into runtime source. Their useful contribution was negative art direction: the final mark had to remain institutional, transparent, code-native, and legible without becoming a literal `K`, media-play glyph, or decorative broadcast trope. The shipping two-path frame-splice SVG was drawn and tuned directly in code from that conclusion.

| Artifact | Dimensions | Bytes | SHA-256 | Disposition |
| --- | ---: | ---: | --- | --- |
| [Direction A — Studio Ledger](./concepts/direction-a-studio-ledger.png) | 1586×992 | 1,310,564 | `7e66e46c801e5a6a3369eff0485915885f95ff19f872794b2b03c89600026690` | Selected direction; composition and material language translated into code, not used as a bitmap |
| [Direction B — Signal Room](./concepts/direction-b-signal-room.png) | 1586×992 | 1,386,828 | `aec597d4c4da178a858c102d581c0b839a8ee0aa7eca4f320c82f8fb44d1b5d8` | Rejected: dark control-room treatment overstates real-time monitoring |
| [Direction C — Signal Paper](./concepts/direction-c-signal-paper.png) | 1586×992 | 1,300,170 | `b3c172d4a9068ea4ee52551ef0e01e6dd510b18e603874597cf9ff2297d0a7f3` | Rejected: clear but insufficiently ownable and too border-led |

The full comparison and selection rationale is in [`phase-2-direction.md`](./phase-2-direction.md).

## Asset rules for future work

- Structural UI remains code/SVG first. A raster asset needs a capability that CSS or SVG cannot provide.
- Runtime imagery needs explicit dimensions, meaningful alternative text when informative, modern format, and lazy loading below the first viewport.
- Generated imagery is raw material, never a drop-in. It must be retouched, cropped, graded to the Studio palette, and reviewed for stock/AI artifacts before it can enter the runtime inventory.
- Icons remain Lucide until an explicitly versioned Kairos icon set replaces it as a whole. Do not mix sources or use emoji as interface icons.
- The local transparent `KairosMark` is the single shipping product mark in the rail, auth loading state, login, and desktop gate. Its cue is the off-axis void splicing two opposed frames, never a filled play symbol, letterform, equalizer, or background badge; generated mark explorations and concept-render logos remain unapproved documentation artifacts.
