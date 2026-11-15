# Figure 2 · verification

## Current result

- **Schema and static profiles passed** through the local glass-diagrams engine. `fig2-static.json` points to retained engine evidence. The static profile explicitly skips the engine animation and human-baseline visual profiles; these are not claimed as passes.
- **Custom browser QA passed** (`fig2-qa.json`). Tested the actual exported SVG with page JavaScript disabled, not merely the engine HTML.
- Export inspected at **0.5, 4 and 16 seconds**, all nested SVG clocks paused and set consistently. Normal script-disabled wall-clock playback also changed geometry.
- **14 semantic icons** changed actual animated point geometry over the phase sweep 0.1, 0.8, 1.9 and 2.4 seconds. Every icon is left of its title and vertically aligned. Agent titles and all six note-bank category labels use **34-unit typography**, approximately **18.96 CSS px at the tested 1584-px-wide SVG**. Secondary text is 25 units. No title truncation or text-box overlap was found.
- All **16 expected single-strand paths** were found and sampled at 301 points each. **Zero pairwise wire crossings, shared segments or block intrusions**. All integration paths retain their dashed strokes. Shared endpoints are intentional: independent text fan-out and note-bank input/output.
- The six private-bank category positions were identical in every inspected phase. Only eligibility opacity changes. There is no cluster cloud and no animation of stored originals.
- Initial player data-URI bytes equal the exported SVG exactly. Replay also yields byte-identical SVG content from a new Blob URL. Initial and post-Replay image playback both animate.
- **Replay resets phase**, independently checked on the keyword scanner crop: early-vs-reset mean pixel difference **0.859**, versus early-vs-late **11.213**. See `fig2-replay-qa.json` and retained crop PNGs.
- SHA-256 of the exported SVG before the electrical-port revision: `69439392819675c68d9865328f510eaf9cda40cd6273b12aab108c1a46f0484a`. Current SVG, after that revision (see `fig2-electric-revision.md`): `35a44c3e6b01abc25189f4f0dbe3d8e43d13dfe798d39ecc54302c2a2bc6447d`.
- **150 pre-existing figure files verified byte-for-byte unchanged**, including figure 1 and its archives. Tracked git status is unchanged. No commit, push, source-code or engine modification.

## Issues found and resolved

1. Initial wide-bank icon placement used a small-card offset. The generator now derives the native left header slot from card dimensions while retaining fixed icon size.
2. An initial apply-to-bank curve crossed the bank. Separate lower fan-in/fan-out curves now have no block intrusions or overlap.
3. An added per-turn caption collided with the audio link label. It was moved above Microphone; the final text-box overlap check is empty.
4. Two initial icon sample times landed on repeated sinusoidal states / check-mark holds. The failed diagnostic is preserved as `fig2-qa-initial-phase-alias.json`; a multi-phase sweep now checks every icon's geometry.
5. The inherited clear-and-restore-same-data-URI Replay pattern did **not** reliably reset Chromium's cached image animation clock. The failed phase-reset report is preserved as `fig2-replay-before-fix.json`. Figure 2 now creates a fresh Blob URL from the identical SVG bytes for each Replay. Figure 1 was deliberately not modified.

## Deliverables

- `fig2-inference.svg`: transparent, script-free animated SVG.
- `fig2-player.html`: self-contained inline base64-image player, dark viewing surface, Replay and expandable scope notes.
- `fig2-inference.json`, `gen_fig2.py`, `fig2_icons.py`: editable recipe and generator; shared `glass_icons.py` is imported unchanged.
- `fig2-inference.html`: engine-built preview, not the inline player.
- `package_fig2.py`, `render_fig2.sh`, `qa_fig2.py`, `fig2_replay_probe.py`: reproducible packaging / verification.
- `fig2-preview-0.5.png`, `fig2-preview-4.png`, `fig2-preview-16.png`, `fig2-player-preview.png`, `fig2-icons-sheet.png`: real browser captures/crops.
- `fig2-architecture.md`, `fig2-source-evidence.json`, validation reports and QA JSON: provenance and evidence.

## Limits

- This verifies the figure, not live robot integration or branch functionality. Branch implementation badges reflect inspected source, not newly executed integration tests.
- No human-approved visual baseline, slide-app playback, mobile performance or seamless loop boundary is claimed. The finite seeded 24-second particle schedule may jump at replay.
- Best viewed at full width. Typography scales down with the image in narrow chat panes; open the player or zoom for the complete architecture.
- Speaker deliberately remains audio output / TTS-or-stored-response because source docs do not establish one implemented crisis-audio path.
