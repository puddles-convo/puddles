# Figure 2: green port-local electrical activity

## Scope

- `audio_in`: `mic.bottom` contact and `listener.top` audio-input boundary.
- `audio_out`: `playback.left` contact only. Cloud Speaker and the intervening semantic audio handoff are not depicted as electrical wiring.
- `local_play`: `offline.top` and `playback.bottom` device-only audio contacts.

Each contact uses a 28-unit electric filament on the existing route, three native polyline layers (green sleeve, bright green current, white-hot core), animated geometry and intensity on a 1.6-second cycle. Five local contacts; no new connections and no new/recolored particle stream. Existing particles, dashed paths, semantic NLP/memory flow, positions, labels and dimensions are unchanged.

## Engine constraint and chosen fallback

Read source engine `docs/animations.md` and `docs/PRIMITIVES.md` (pulse definition at lines 595–608; overlay anchors at 610–626). The native `neon` macro targets circuit edges; its `pulse` primitive requires a regular connection, not the approved dashed spaghetti connections. Therefore use the requested supported port-local electric-pulse fallback, authored as native polyline overlays plus `points` and `opacity` tracks. This is not the `neon` macro and is not ordinary green particles. No engine modification or exported-SVG editing.

## Verification

- Full `render_fig2.sh` completed; schema and static HTML/SVG validation passed. Engine animation/full-baseline profiles were not requested or run.
- Existing Figure 2 QA passed: no wire crossings, shared segments, block intrusions, text overlaps or truncated titles; icon motion and fixed note-bank positions preserved.
- Dedicated electrical QA passed at 0.08, 0.2, 0.46 and 0.9 seconds: all 15 layers change geometry/intensity; ten green layers and five white-hot cores present.
- All electric layers also verified moving during normal standalone SVG playback with page scripts disabled.
- Actual screenshot inspection: complete figure at 0.46 seconds; audio-input closeups at 0.2 and 0.9 seconds; robot input at 0.46 seconds; local playback at 0.2 seconds. Green filaments and contact highlights visible, labels clear, original layout retained.
- Player embeds unchanged SVG bytes. Fresh-Blob Replay logic retained; equality, continued motion and independent phase reset passed.
- Recipe comparison proves every pre-existing animation, block and connection unchanged. 150 unrelated figure files hash-verified unchanged, including Figure 1.
- Native SVG finite seeded particle schedule retains the existing possible loop-boundary discontinuity. No slide-app playback or human baseline approval claimed.

## Artifacts

- `fig2-inference.svg`: transparent, standalone animated SVG.
- `fig2-player.html`: inline-ready player.
- `fig2-electric-phase-0.46.png`: complete preview.
- `fig2-electric-{input,output,local}-{0.08,0.2,0.46,0.9}.png`: phase closeups.
- `fig2-electric-qa.json`, `fig2-qa.json`, `fig2-replay-qa.json`, `fig2-static.json`: evidence.
- Prior version archived at `fig2-archive-before-electric-20261114-202053/`; original validation history retained in place.

Only inference Figure 2 sources/artifacts changed. No commit or push.
