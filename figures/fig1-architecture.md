# Figure 1 v6: implementation-grounded architecture

This replaces the misleading v5 training-loop overview. There is **no gate-to-simulated-user wire and no automatic next-batch loop**. The exported SVG is engine-generated, never post-edited. Long-distance state dependencies are named in input strips, rather than routed as overlapping wires.

Source callsites below refer to the implementation snapshot at commit `58f9391`; line numbers may shift in later commits. The figure documents the script-driven path, not every newer orchestration entry point.

## Displayed handoffs and source evidence

| Producer → consumer | Payload | Status / callsite |
|---|---|---|
| Simulated user ↔ Helper | Alternating dialogue turns | Implemented: `loop/runner.py:43–58`; scenario drives the user, policy drives Helper |
| Conversation runner → Dialogues | Completed conversation records | Implemented: `loop/runner.py:61–70`, `99–114`; displayed after Helper as the runner's output, not an extra AI |
| Dialogues → Judge → Scored dialogues | Scoring integration | **Planned**: stationary dashed paths, no particles. Judge schemas do not establish an implemented caller |
| Pre-scored JSON → Critic | ScoredConversation objects | Actual current entry: `scripts/run_round.py:24–38`; fixture default `fixtures/bad_conversations.json` |
| Critic → Researcher | FailurePattern | Implemented: `puddles/round.py:66–79` |
| Researcher ↔ Exa | Search queries / returned sources | Implemented: `scripts/run_round.py:36–37`, `puddles/agents/researcher.py:57–75` |
| Critic → Proposer | Direct failure pattern | Implemented: `puddles/round.py:87`; distinct lower route, not only an indirect Researcher handoff |
| Researcher → Proposer | Research sources | Implemented: `puddles/round.py:87`; `puddles/agents/proposer.py:38–42` |
| Proposer → Gatekeeper | Candidate policy + edit context | Implemented development path: `puddles/round.py:90–106` |
| Manual keep/revert → Gatekeeper | Developer-selected outcome | `scripts/run_round.py:26,38`; **development stand-in**, no scored evaluation |
| Gatekeeper → Finish round | GateOutcome | `scripts/run_round.py:38–39`; `puddles/round.py:154–186` |
| Finish round → Shared searchable memory | Copies of available records after logging | `puddles/round.py:181–186`; no direct immediate agent-to-memory writes shown |
| Finish round → Policy in force | Promotion, **kept only** | `puddles/round.py:167–173`; reverted keeps the prior policy |

The policy-improvement row reads right to left; the generation row reads left to right. Finishing is orchestration code, not an AI agent. The view shows the candidate-producing path. Early exits for no pattern or no evidence still call finish_round with no gate (`scripts/run_round.py:38–39`, `puddles/round.py:69–71,83–85`). These branches are described in the player's notes, not drawn as extra wires.

## State dependencies shown as named input strips

- **Policy in force → Helper, Critic, Proposer**: `loop/runner.py:53`; `puddles/round.py:63,66,87`. A new kept artifact does not automatically launch a new batch.
- **Decision log → Critic prior pattern status**: `puddles/agents/critic.py:20–24`.
- **Decision log → Researcher previous queries**: `puddles/agents/researcher.py:22–30`.
- **Decision log → Proposer earlier edits / pattern status**: `puddles/agents/proposer.py:28–42`.
- **Memory → Critic supported records**: recall is not restricted to a single record type (`critic.py:22–24`).
- **Memory → Researcher prior queries**: only ResearchNote results contribute query strings (`researcher.py:25–27`).
- **Memory → Proposer gate reasons**: only GateOutcome reasons enrich earlier edits (`proposer.py:32–37`).
- **No memory read by Gatekeeper** is shown or implied.

## One shared searchable memory, four real types

| Display label | Type | Author |
|---|---|---|
| Failure patterns | FailurePattern | Critic (`critic.py:68–76`) |
| Research notes | ResearchNote, containing sources | Researcher (`researcher.py:78–81`) |
| Policy edits | PolicyEdit | Proposer (`proposer.py:91–106`) |
| Gate outcomes | GateOutcome | Development gate (`puddles/round.py:96–106`) |

The decision log and local files are authoritative. Agent records are saved locally during execution (`puddles/round.py:75,82,92`), and the gate outcome and round event are persisted during finishing (`164,181`). Memory holds searchable copies; recalled IDs resolve through local records and content hashes (`puddles/memory.py:186–205`). These are not independent competing stores. The figure's “Written by finish_round” refers to the round-log write, not exclusive authorship of every local file.

Model labels follow `README.md:163–164` and `docs/agents/t1-simulated-user.md:12`, `t2-i3-helper.md:12`. Improvement agents use the configured OpenAI model, not a fabricated pinned model name.

## Motion and visual conventions

- Solid implemented paths carry slow particles and faster shooters. Bidirectional conversations and search have reverse movers on the same wire.
- Planned scoring paths are native one-strand paths with constant `stroke_dashoffset` tracks: stationary dashes, no particle emitters.
- Animation indicates relationships, not measured throughput or literal execution timing. Concurrent animated streams do not assert concurrent agent calls.
- The original 360 cloud stars preserve v3's 24-second gather/rotate/split geometry and all motion keyframes, after normalizing only world-position translation and generated block IDs. Cloud positions are not measured embeddings.
- Large native `legend: corners` labels preserve the animated cloud. No engine files were changed in this revision.
- All seven agent icons retain role-specific left-hand geometry and SMIL animation.

Reproduce with `GLASS_DIAGRAMS_ROOT=/path/to/glass-diagrams sh figures/render_fig1.sh`. Set `GLASS_DIAGRAMS_BIN` if the engine environment is not under `~/.local/share/glass-diagrams/uv-venv/bin`. The engine must support `cluster_cloud` with `legend: corners`; this bundle does not vendor or publish engine changes. Browser QA requires Playwright and Chromium. Historical revisions and transient previews are excluded; the normalized historical cloud fingerprint is embedded in `qa_fig1.py`. Detailed checks and limitations are in `fig1-verification.md`.
