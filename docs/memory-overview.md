# Memory overview

How training memory is stored today, the format of each part, and what to improve next. The full design is in [memory.md](memory.md); every format is defined in [`schemas.py`](../schemas.py).

There are two memories:

- **Training memory** is built.
- **User memory** (notes about a person) has its schemas, but no store yet.

Training memory has three layers:

- **The official copy:** files in the run folder (the decision log, record files and policy files).
- **A searchable copy:** in past.dev.
- **A read log:** records who used what.

| Layer | Where | Role |
|---|---|---|
| Record files | `records/round-NN/<id>.json` | The full content of each agent output |
| Decision log | `log/decisions.jsonl` | The official history; code works everything out from it |
| Policy files | `policies/policy-vN.md` + `.meta.json` | The rules in force and where each one came from |
| past.dev | audience `training-agents` | Search only: finds earlier records relevant to this round |
| Read log | `log/memory_reads.jsonl` | Which agent read which record |

## 1. Record files

One JSON file per agent output. Every record has `id`, `schema_version` (3), `written_by`, `created_at` and `parent_ids` (the IDs it was made from).

| Record | ID | Written by | Main fields |
|---|---|---|---|
| `FailurePattern` | `pattern-rNN` | critic | `name`, `description`, `seen_in` / `out_of`, `example_conversation_ids`, `same_as_pattern_id` |
| `ResearchNote` | `research-rNN` | researcher | `pattern_id`, `queries`, `sources[{url, title, technique}]` |
| `PolicyEdit` | `edit-rNN` | proposer | `pattern_id`, `research_id`, `action` (add / reword / remove), `rule_id`, `rule_text`, base and new policy version |
| `GateOutcome` | `gate-rNN` | gatekeeper (stand-in for now) | `edit_id`, `decision` (kept / reverted), scores `before` / `after`, `safety_passed`, `policy_in_force`, `reason` |

Code assigns the IDs, never the model. A file counts only if its SHA-256 matches the hash in the log.

## 2. Decision log

Append-only, one JSON event per line. There are two event types.

**`round`**: one per round. A second one for the same round is refused.

```json
{"event": "round", "round": 7, "outcome": "kept",
 "records": {"pattern": {"id": "pattern-r07", "path": "records/round-07/pattern-r07.json", "sha256": "…"},
             "research": {"…": "…"}, "edit": {"…": "…"}, "outcome": {"…": "…"}},
 "pattern": {"id": "pattern-r07", "same_as_pattern_id": "pattern-r03", "name": "advice-before-understanding",
             "seen_in": 3, "out_of": 8, "status_before": "fixed"},
 "edit": {"id": "edit-r07", "action": "reword", "rule_id": "C9", "rule_text": "…"},
 "policy": {"base": "policy-v4", "base_sha256": "…", "new": "policy-v5", "new_sha256": "…", "in_force": "policy-v5"},
 "scores": null, "safety": null, "reason": "…"}
```

`outcome` is one of `kept`, `reverted`, `no_pattern` or `no_evidence`.

**`memory_copy`**: one or more per record.

```json
{"event": "memory_copy", "round": 7, "record_id": "pattern-r07", "status": "confirmed", "attempt": 1}
```

From this log, code works out:

- each pattern's status (open, fixed or recurring)
- earlier edits and queries
- the policy in force
- which copies are pending or missing

## 3. Policy files

- **`policy-vN.md`:** a `## Safety` section of S rules, which are locked, and a `## Conversation` section of C rules, which can be edited.
- **`policy-vN.meta.json`:** where each rule came from.

```json
{"policy_version": "policy-v5", "parent": "policy-v4", "policy_sha256": "…", "safety_sha256": "…",
 "base_rules": ["C1", "C3"],
 "learned_rules": {"C9": {"edit": "edit-r07", "action": "reword", "pattern": "pattern-r03", "round": 7, "text_sha256": "…"}}}
```

**Replay check** (`puddles/replay.py`): runs at the start and end of every round. If any of these fails, the loop stops.

- Replaying the log's kept edits from the starting policy must reproduce the policy file in force exactly: the same rule IDs and the same text.
- The Safety section's hash must match the starting policy's.
- The `.meta.json` must match what the log says.

## 4. past.dev

One entry per record:

```json
{"id": "pattern-r07", "label": "critic: failure pattern, round 7", "timestamp": "<created_at>",
 "identity": "critic", "audience": "training-agents",
 "content": "Failure pattern pattern-r07, advice-before-understanding, round 7: … Seen in 3 of the 8 lowest-scoring conversations reviewed. Same pattern as pattern-r03.",
 "metadata": {"type": "failure_pattern", "round": 7, "record_id": "pattern-r07", "record_path": "…", "sha256": "…"}}
```

- **Written** only after the `round` event is logged. Each write is checked until past.dev reports `completed`.
- **Failed writes** are logged `pending` and retried at the start of the next round. Three failures stop the loop.
- **Read** with a plain-language question, asked as one identity: `critic`, `researcher`, `proposer` or `gatekeeper`.
- **Hidden from** the simulated user, the helper, the judge and the gatekeeper, which never read it.

## 5. Read log

```json
{"reader": "proposer", "query": "…", "at": "…", "record_ids": ["gate-r03", "edit-r03"]}
```

This log produces the "written by X, later read by Y" report.

## 6. How an agent's input is built

1. **From the agent before it:** records from earlier in this round are passed directly, from Critic to Researcher to Proposer.
2. **From the log:** a history list. It shows the items about this pattern and the 10 newest others (`HISTORY_RECENT`), then one line counting the rest. Duplicate checks still use the whole log.
3. **From past.dev:** a search returns record IDs. Each must be in the log, and its file hash must match. At most 5 records (`FROM_MEMORY_LIMIT`) go into the prompt.

**Pattern identity:** the first sighting's ID is the pattern's identity. A later sighting gets its own ID and points back with `same_as_pattern_id`. If the critic leaves that field empty, code matches the name against the log.

## 7. Restart safety

`prepare_round` runs before each round:

- **Policy in force:** taken from the last `round` event, with its hash checked.
- **Unlogged promotion:** a promoted policy file that no round event names, and that is newer than the policy in force, is removed. This undoes a crash between writing the file and logging the round.
- **Missing copies:** a record with no `memory_copy` event is copied.
- **Pending copies:** they are resent, and the three-attempt limit is checked again afterwards.

The stage-0 graph runner (`scripts/run_graph_round.py`) does not run these steps yet; see improvement 1.

## 8. User memory (designed, not built)

- **`UserNote`:** one fact about one person, with `user_id`, `session_id`, `kind`, `fact`, `last_confirmed` and an optional `due_date`.
- **`NoteAction`:** the note-taker's change to a note: `add`, `replace`, `delete` or `confirm`.
- **`LiveNoteIndex`:** lists which notes are live and enforces a limit for each kind. Only the notes it lists reach the helper.

Still to build: the note-taker, the store, expiry and deletion. See [memory.md](memory.md) section 10.

## 9. How to improve it

| # | Improvement | Why | Effort |
|---|---|---|---|
| 1 | Run the replay check and log-based policy choice in the graph runner too | Only `scripts/run_round.py` has them; the graph runner still picks the newest policy file | Small, but its acceptance tests must change |
| 2 | A fixed list of pattern names, built from ESConv | Repeat sightings rely on the critic's judgement or an exact name match, so a renamed pattern counts as new | About an hour |
| 3 | Helpful and harmful counts for each rule | At the 800-token cap, nothing says which rule to drop | Medium |
| 4 | Gate scores for each problem type | A rule that helps one type and hurts another looks neutral overall | Medium |
| 5 | Reject an edit nearly identical to a reverted one; set a pattern aside after 3 reverts | Stops the loop retrying the same failed idea | Small |
| 6 | A fifth outcome for a round with a pattern but no edit | Today it is logged as `no_pattern` (issue #1) | Small |
| 7 | A `TYPE:` tag in each entry's text, or one audience per record type | past.dev does not filter by metadata, so a search can return the wrong kind of record | Small |
| 8 | Measure how useful recall is, using the read log | It is unknown whether recalled records change what agents do | Medium |
| 9 | Commit or archive `runs/` logs and records | The official copy lives in a folder git ignores | Small |
| 10 | Build user memory: store, delete call, one audience per person, expiry | Needed before real users; how fast past.dev stops returning a deleted entry is untested | Large |
| 11 | An index over the decision log | The log is read in full for every lookup, which is fine now but slow on long runs | Small |
