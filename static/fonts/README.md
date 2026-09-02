# Webfonts

Bundled, not fetched from a CDN: the app runs locally and must not depend on
the network for its typeface. 120 KB total, cached by the browser after the
first load. Registered in `.streamlit/config.toml` under `[[theme.fontFaces]]`
and served from `./static` (`server.enableStaticServing = true`).

| Family | Weights | Role |
|---|---|---|
| **Archivo** | 400, 500, 600, 700 | Everything that is words — body copy, headings, prose. A neutral grotesque in the Swiss/International line, which is the family Bloomberg's own identity sits in: tight apertures, low stroke contrast, no personality of its own. That last part is the point; a UI face should get out of the way of the numbers. |
| **IBM Plex Mono** | 400, 500, 600 | Every figure in the app, plus uppercase micro-labels. Digits are all 600 units wide, so decimal points line up down a ranking column. |

Source: [Fontsource](https://fontsource.org) (`@fontsource/archivo`,
`@fontsource/ibm-plex-mono`), latin subset, WOFF2.

## What these replaced, and why

Latin Modern — the OpenType Computer Modern that LaTeX sets by default. It was
handsome and wrong for a dark screen. Measured on a rendered page: with caption
colour set to `rgb(211,219,230)`, the brightest pixel actually drawn was
`rgb(131,137,146)` — about **63% of the ink asked for**. Below roughly 15px its
stems are thinner than a pixel, so anti-aliasing eats most of them, and no
colour value fixes that because effective brightness is colour x coverage.
Switching to the demibold optical cut helped but did not close the gap.

A grotesque has thick, even stems and simply does not have the problem, which
is why the text tokens in `utils/theme.py` could go back to a real luminance
hierarchy instead of every level sitting at near-white to compensate.

## Glyph coverage

Neither family ships a Greek subset, so the beta and sigma used in labels
("Market beta", volatility sigma) fall back — as do `₹`, `✓`, `▲`, `▼`. The CSS
stacks therefore name Segoe UI and DejaVu Sans after the primary family, both
of which carry them.

## Licence

Both are SIL Open Font License 1.1 — free to use, modify and redistribute,
including bundled in an application.
