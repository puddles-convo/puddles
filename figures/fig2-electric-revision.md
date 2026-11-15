# Figure 2: published electricity correction

## Proven cause

The GitHub README at `0355d0c` loaded `figures/fig2-inference.svg` directly (not a Camo image in the observed authenticated browser). Fetching that exact image returned bytes identical to the local export. Actual README widths were approximately 838px at a 1440px viewport and 423px at an 800px viewport.

The previous effect was five 28-unit endpoint filaments on a 2840-unit canvas: about 8.3px long at desktop README width and 4.2px on the narrow view. Its bright filament was only 3.8 units wide. It had no traveling neon current. Prior QA enlarged crops and checked geometry changes, not perceptual visibility at README size. That was inadequate verification.

The original remote SVG's filaments still changed at 25 and 49 seconds. A finite animation stopping, wrong asset, and SVG stripping were not the cause observed here. The original export fails `qa_fig2_readme.py`, retained as `fig2-published-evidence/before-visibility.json`.

## Correction

Only the existing `audio_in`, `audio_out`, and `local_play` routes use regular edges now, enabling the engine's native neon **pulse primitive** (not the circuit-tree macro). The original source/destination ports are unchanged. Each has two staggered green-glow / bright-core / white-centre pulses traveling along its full path, indefinitely repeating every 1.4 seconds. Ordinary particles on those three routes were removed, not recolored. All other data motion is unchanged.

The base paths remain dashed using supported constant dash-offset tracks, preserving planned-integration meaning. The green effect illustrates audio-signal activity, not physical power/network wiring or a working audio integration. Labels paint above the current. Every block, other route, memory-bank category, icon and architecture relationship remains unchanged; no new wires were added.

## Verification

- Real published README and original remote raw SVG inspected in an authenticated Chromium browser; exact-byte identity established before correction.
- New visual regression test first failed on the published original, then passed on the corrected export at 423px and 838px widths, light and dark backgrounds, seven phases including 25.15s and 49.45s.
- Visibility gate: each audio corridor must contain at least 15 green pixels spanning at least 10px, including a sample after 24 seconds. No weakened threshold between red and green.
- All 18 native pulse layers change dash position in early, 25-second and 49-second sample pairs. All repeat indefinitely. Normal script-disabled SVG playback also changes their positions.
- Full `render_fig2.sh` passed: schema, static HTML/SVG audits, architecture/route/label checks, fixed memory positions, icons, SVG player byte identity, Replay phase reset, and both electrical tests.
- No wire crossings, block intrusions or text-box overlaps. Other figures unchanged.
- Engine animation/full human-baseline profiles and slide compatibility are not claimed. The existing 24-second seeded data-particle schedule may have a loop-boundary discontinuity; native electrical pulses are independent indefinitely repeating tracks.

## Evidence

- `fig2-published-evidence/github-before-dark.png`: original real README.
- `fig2-published-evidence/before.svg`: original published bytes, retained locally.
- `fig2-published-evidence/before-remote-motion.json`: original remote motion beyond 24 seconds.
- `fig2-published-evidence/{before,after}-visibility.json`: red/green visual regression evidence.
- `fig2-published-evidence/after-{dark,light}-{423,838}-*.png`: final exported SVG at actual README widths.
- `fig2-electric-qa.json`, `fig2-qa.json`, `fig2-replay-qa.json`: current automated checks.
- `fig2-player.html`: unchanged exported SVG embedded in a Replay-capable image player.

Post-push remote rendering evidence is retained locally alongside these files and reported separately after publication; generation and byte equality alone are not described as verified GitHub playback.
