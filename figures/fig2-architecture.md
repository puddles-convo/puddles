# Figure 2 · inference architecture audit

## Snapshot and status language

Inspected local refs without checking out or modifying branches:
- main `e5ea2a8`: reusable Helper function exists; no `puddles/inference` pipeline on this ref.
- crisis-path `78377e409a5cb30be461fdc6405bcf24b9b394d9`: text-turn keyword, crisis checker, concurrent Helper and Release code.
- notes `23cdbb853557d1159e191a2e6f8641727f3f56c8`: private note storage and note-taker/action code, not session integration.

MAIN is a component/function, not an end-to-end deployment claim. BRANCH marks inspected implementation on the named local branch, not main. PLANNED means intended audio/device component or lifecycle integration. Every wire is dashed because the complete depicted system is not integrated on main. User-approved motion is illustrative target flow, not observed execution or throughput. No source code outside figures was modified.

## Edge trace

| Figure link | Payload / actual consumer | Evidence | Status on main |
|---|---|---|---|
| Microphone → Listener | Audio → speech transcription | docs/agents/i1-listener.md; docs/plan/stage-4-inference.md:13 | Planned audio-stack boundary |
| Listener → Keyword | Transcribed text → deterministic keyword match | docs/plan/stage-4-inference.md:19–25; crisis-path:puddles/inference/turn.py:128–132 | Planned audio integration; text consumer on branch |
| Listener → Checker | Text; recent conversation supplied separately | same plan; crisis-path:turn.py:134–146 | Planned audio integration; concurrent model call on branch |
| Listener → Helper | Text + session history; policy snapshot and user_notes | main:loop/helper.py:26–37; crisis-path:turn.py:134–146 | Helper function exists, wiring on branch/planned |
| Keyword → Release | Match, not clearance | crisis-path:turn.py:179–182; release.py:32–33 | Implemented on crisis-path only |
| Checker → Release | Valid risk/category, or failed/late/unreadable status | crisis-path:turn.py:148–182; release.py:23–43 | Implemented on crisis-path only |
| Helper → Release | Draft or unavailable/failure | crisis-path:turn.py:167–182 | Implemented on crisis-path only |
| Release → Speaker | Selected reply text | docs/agents/i4-speaker.md:10,25–35 | Planned audio integration |
| Speaker → Robot | Audio output | docs/agents/i4-speaker.md:5–6,35 | Planned audio integration |
| No cloud → Offline audio → Robot | Device-only unavailable-cloud trigger → local stored recording | docs/agents/i4-speaker.md:78–81; intended device fallback | Planned; bypasses all cloud stages |
| Private memory → Helper | Eligible notes loaded once at session start | docs/memory.md:82–95,127–146,173; notes:note_store.py:219–236 | Planned session wiring; note reader on notes branch |
| Session end → Note-taker | Finished text turns, date, remember consent | docs/memory.md:93–108; notes:notes.py:47–51,110–120 | Planned lifecycle wiring |
| Private memory → Note-taker | Existing notes with IDs, kinds, dates | notes:notes.py:47–51; note_store.py:219–233 | Reader implemented on notes branch |
| Note-taker → Apply actions | add/replace/confirm/delete; forget-all flag | notes:notes.py:110–180 | Implemented within notes branch function |
| Apply actions → Private memory | Indexed writes/removals; delete retries | notes:notes.py:128–180; note_store.py:269 onward | Implemented on notes branch |

These are separate temporal panels: per-turn processing, device-only failure fallback, session-end notes. The figure does not falsely route audio into Note-taker or imply Speaker calls Note-taker.

## Guardrails

- Keyword detection is synchronous code before the two model workers start on crisis-path. The three branches are independent inputs, not a claim of three parallel model jobs. Checker and Helper are concurrent. No Checker → Helper edge exists.
- Release first honors keyword/crisis fixed category text, discarding Helper output. Keyword with no checker category uses `unsure`. Categories are `self`, `danger`, `unsure`.
- Draft requires no keyword plus a valid `none`/`concern` check for this session and turn. Failure/timeout/unreadable check or unavailable draft gives holding text. Two consecutive failed checks give offline text, subject to keyword/crisis priority.
- Device offline audio is a separate bypass. Speaker is labelled audio output / TTS or stored response because the docs mention device recordings; the figure does not assert all crisis responses are synthesized.
- Memory's six fixed categories match main:schemas.py:262–271: situation, people, what_helps, goals, follow_up, preferences. No training-style cloud, clustering, moving originals, or measured embeddings.
- Remember off allows deletions only; forget-all blocks add, replace and confirm. Apply removes old IDs from the live index first. The bank's note records and index are a single logical private store, not two invented databases.
- Only Helper and Note-taker read private notes. There is no private-memory arrow to Checker, Listener, Release, Speaker or training. Training-transcript consent is separate and no fake training pipeline was added.
- Policy snapshot and history are explicit Helper inputs. Helper does not fetch memory itself. This figure omits model names rather than turning planned model selections into implementation claims.

## Motion schedule

Role icons loop independently at 1.8–4 seconds and explain the local action. Dashed target handoffs carry seeded slow particles and faster shooters; fast/slow use different travel speeds. The 24-second baked interval is illustrative and can jump at replay. Fixed category chips change opacity to indicate eligibility; their coordinates never change. Text remains fixed. Memory-read particle copies move toward Helper and Note-taker; writes run only on the session-end panel. This is not a chronological simulation of session start, turn and end.

## Artifacts and scope

`gen_fig2.py` + `fig2_icons.py` generate `fig2-inference.json`; all geometry exports through the existing glass-diagrams engine. `render_fig2.sh` produces SVG, engine HTML, unchanged-byte player and QA. `package_fig2.py` adds only a dark player surface and Replay. No generated SVG edits, engine patches, training figure changes, commit or push.
