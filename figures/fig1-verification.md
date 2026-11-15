# Figure 1 verification

## Published bundle

- `fig1-training.svg`: transparent, engine-generated animated SVG.
- `fig1-training.json` and `fig1-training.html`: recipe and engine HTML export.
- `fig1-player.html`: self-contained replayable player; download and open locally.
- `gen_fig1.py`, `glass_icons.py`, `package_fig1.py`, `qa_fig1.py`, `render_fig1.sh`: generation, packaging and browser checks.
- `fig1-qa.json`: actual exported-SVG and player check results. Preview paths identify local generated evidence; screenshots and historical archives are intentionally not committed.

## Verified before publication

`qa_fig1.py` passed in Chromium against the final SVG and repackaged player:

- SVG XML parses; all nested SVG clocks seek correctly at five phases.
- Script-disabled SVG playback moves normally.
- All 360 cloud stars preserve the historical normalized geometry and SMIL keyframes, checked against a fingerprint embedded in the test rather than an archived SVG dependency.
- All seven semantic icons change their actual `animatedPoints`.
- All 10 regular paths were sampled at 251 positions: no pairwise crossings, shared segments or block-interior intrusions (allowing native 3-unit docking).
- Planned Judge paths have stationary dashes and no particle schedules; no gate-to-simulated-user connection exists.
- Four memory category titles and sampled agent titles are 34 SVG units.
- The player's embedded data URI matches the SVG bytes exactly. Pixels change during initial playback and after Replay.
- The portable render shell script passes `sh -n`.

SVG SHA-256: `8469805699d6b652bf794f4d8ac4eaa139e551e884f98c6885f23db514618115`.

## Prior validation and limitations

The preparation run passed engine schema and static HTML/SVG audits. The engine's animation profile did not pass: it reported `animation_23_0` (`ic_sim_0_move`, a points track) as stationary. That audit reads the `points` attribute through a computed CSS property; the dedicated `animatedPoints` and real playback checks above pass. This is not an all-engine-checks-passed claim. Historical diagnostic reports and bulky preview artifacts remain local.

GitHub animation playback, slide-app playback and a human-approved visual baseline were not tested. The finite seeded particle export may jump at the loop boundary. Motion is illustrative, not measured embeddings, throughput or execution order. Early no-pattern/no-evidence exits are explained in player notes rather than drawn as extra wires. State inputs are named in strips rather than long intersecting return wires.

## Reproduction

See `fig1-architecture.md` for the source snapshot and engine prerequisites. Run:

```sh
GLASS_DIAGRAMS_ROOT=/path/to/glass-diagrams sh figures/render_fig1.sh
```

The engine checkout must support `cluster_cloud` with `legend: corners`; engine source changes are outside this repository and are not bundled. Set `GLASS_DIAGRAMS_BIN` for a non-default engine environment. Playwright and Chromium are required for browser QA. The script regenerates the recipe, exports, player, local QA previews and schema/static validation evidence. To check existing exports without re-rendering, run `python figures/qa_fig1.py` in that environment.
