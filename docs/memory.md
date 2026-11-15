# Memory: the whole story in one file

What the memory is, what is stored at each level, what every agent reads and writes, when each level is updated and how, and what is still undecided. Anything about memory should be changed here first, and other documents should link here.

This file describes schema version 3 (`schemas.py`). Version 3 is specified but only partly built: section 9 says what exists in code and where the code still follows version 2. Scores, counts and sources in the examples are illustrative.

## Contents

1. [The idea in five lines](#1-the-idea-in-five-lines)
2. [The levels](#2-the-levels)
3. [One training round, end to end](#3-one-training-round-end-to-end)
4. [One conversation with a person, end to end](#4-one-conversation-with-a-person-end-to-end)
5. [What each agent reads and writes](#5-what-each-agent-reads-and-writes)
6. [How memory gets into a prompt](#6-how-memory-gets-into-a-prompt)
7. [When each level is updated, and how](#7-when-each-level-is-updated-and-how)
8. [The formats](#8-the-formats)
9. [What is built and what is not](#9-what-is-built-and-what-is-not)
10. [Still to decide](#10-still-to-decide)
11. [Where the detail lives](#11-where-the-detail-lives)

## 1. The idea in five lines

1. **Everything that happens is appended to a record that never goes into a prompt whole.** It can grow without limit, because it only costs storage.
2. **A small working set does go into prompts.** It is a list of items with IDs, changed one item at a time by code. No model ever rewrites it as a summary.
3. **Evidence decides what stays in the working set:** a test result for a policy rule, a date last confirmed for a note about a person.
4. **Memory informs, policy instructs.** Nothing recalled from memory changes the helper's behaviour. The only way to change behaviour is one policy edit that passed the gate.
5. **There are two memories and they never mix:** what the training agents have learned, and what the robot remembers about one person.

## 2. The levels

Six levels. The first four belong to training, the last two to a person.

| # | Level | Where | What it holds | Grows? | Who may change it |
|---|---|---|---|---|---|
| 1 | Record files | `records/round-NN/*.json` | The full content of every pattern, research note, edit and gate outcome | Without limit, never rewritten | Code, once per record |
| 2 | Decision log | `log/decisions.jsonl` | One `round` event per round pointing to the record files by path and hash, plus one `memory_copy` event per copy | Append-only | Code only |
| 3 | Training memory | past.dev, audience `training-agents` | One searchable sentence per record | Add-only, about four entries a round | Code, under the writing agent's identity |
| 4 | Policy file | `policies/policy-vN.md` and `.meta.json` | The rules the helper follows. This is what memory turns into. | Bounded by a token cap | Code, applying one gated edit |
| 5 | Live-note index | Kept by the cloud brain, one per person | Which notes about a person are current, with each one's kind and dates | Bounded: 23 notes | Code, from the note-taker's actions |
| 6 | User memory | past.dev, one private audience per person | The sentence of each note | Bounded by the index | Code, under the note-taker's identity |

**Which level wins when two disagree**

- Training: the decision log is the authority. Record files are checked against its hashes, training memory is a copy of it, and the policy file must equal what a replay of the log produces.
- A person: the live-note index is the authority. A note that is not on the index never reaches the helper, even if past.dev still returns it.

**What is derived and never stored**

- A pattern's status (`open`, `fixed`, `recurring`) is worked out by replaying the log each time it is needed.
- The history lists shown to the critic, researcher and proposer are rebuilt from the log for every call.

## 3. One training round, end to end

```
start of round    resend any memory copies still pending from earlier rounds
                  (three failed attempts on one copy stop the loop)

1 simulated user  no memory
2 helper          no memory; reads the policy file in force
3 judge           no memory; scores go to the critic

4 critic          reads: pattern list from the log + recall from training memory
                  -> pattern record file saved
5 researcher      reads: query list from the log + recall from training memory
                  -> research record file saved
6 proposer        reads: edit list from the log + recall from training memory
                  -> edit record file saved, candidate policy file written
7 gatekeeper      no memory; tests the candidate
                  -> outcome record file saved

end of round      if kept: the promoted policy file is written
                  one `round` event appended to the decision log
                  each record copied into training memory, one at a time,
                  each followed by a `memory_copy` event (confirmed or pending)
                  replay check: the log and the policy file must agree
```

- **Inside a round, agents hand records straight to each other.** Memory is only how a round learns from earlier rounds, because past.dev writes are queued and its reads are searches.
- **Nothing reaches training memory until the round is in the log.** So memory can never hold something the log does not.
- **A round can end early** as `no_pattern` or `no_evidence`. The records made up to that point are still logged and copied.

## 4. One conversation with a person, end to end

```
start of session  ask two questions: may I remember you, may this be used for training
                  take expired follow-ups off the index
                  load the person's live notes once (23 at most)

each turn         listener, crisis checker, helper, release, speaker
                  only the helper uses memory: the notes loaded at the start
                  nothing is written to memory during the session

end of session    note-taker reads the session and the existing notes,
                  returns a list of actions: add, replace, confirm, delete
                  code applies them: index first, past.dev second
```

What is kept depends on the two answers:

| Remember me | Use for training | Kept after the session |
|---|---|---|
| No | No | Nothing new |
| Yes | No | Notes about the person only |
| No | Yes | The session record goes to training; no new notes |
| Yes | Yes | Both |

- **Forgetting always works.** The note-taker always runs. With "remember me" off it may only delete.
- **Forgetting overrides remembering.** After "forget everything" the index is cleared and every add, replace and confirm from that session is dropped.
- **A person's notes never become policy on their own.** If many people react badly to the same thing, it shows in the scores of their consented sessions, the critic names it, and it goes through the gate like any other pattern.

## 5. What each agent reads and writes

"Reads" means what code puts in front of the agent. No agent calls past.dev or opens a file itself: the orchestrator does both.

**Training**

| Agent | Reads from the log | Reads from training memory | Writes (record file, then log, then memory) |
|---|---|---|---|
| 1. Simulated user | Nothing | Nothing. It must not know what the loop has learned. | Nothing |
| 2. Helper | Nothing | Nothing. It reads only the policy file. | Nothing |
| 3. Judge | Nothing | Nothing. It must stay blind. | Nothing. Its scores go to the critic. |
| 4. Critic | Every pattern that has a rule, plus open patterns the recall found, plus the 10 most recently seen open ones | Up to 5 records about earlier patterns and their outcomes | One `FailurePattern` per round |
| 5. Researcher | Every query run for this pattern, plus the 10 most recent others | Research notes already written for similar patterns | One `ResearchNote` |
| 6. Proposer | Every edit tried for this pattern with its outcome, plus the 10 most recent others | Why earlier edits were kept or reverted | One `PolicyEdit`, whether or not it is later kept |
| 7. Gatekeeper (code) | Nothing | Nothing | One `GateOutcome` per round that reached the gate |

**Inference**

| Agent | Reads | Writes |
|---|---|---|
| 1. Listener | Nothing | Nothing |
| 2. Crisis checker | Nothing | Nothing |
| 3. Helper | The person's live notes, once at the start of the session | Nothing |
| Release (code) | Nothing | Nothing |
| 4. Speaker | Nothing | Nothing |
| 5. Note-taker | The person's live notes with their IDs, kinds and dates | Notes added, replaced, confirmed or deleted |

**Who uses whose writing**

| Written by | Read by |
|---|---|
| Critic | Critic |
| Researcher | Researcher |
| Proposer | Proposer |
| Gatekeeper | Critic, Proposer |
| Note-taker | Helper, Note-taker |

Every recall is noted in `log/memory_reads.jsonl` with the reader, the query and the record IDs returned, so a run can show which agent read which record.

## 6. How memory gets into a prompt

Two routes, and both end with content loaded by code, never with raw search results.

**Route 1: lists built from the log.** Complete where it matters, budgeted elsewhere.

| Agent | Block in its prompt | What is in it |
|---|---|---|
| Critic | `<earlier_patterns>` | One line per pattern with its status and rule, then one line counting the patterns not shown |
| Researcher | `<earlier_queries>` | The queries already run, then one line counting the rest |
| Proposer | `<earlier_edits>` | The edits already tried with their outcomes, then one line counting the rest |

The budget is `HISTORY_RECENT = 10`. It limits what a model reads, not what code knows: the checks for a repeated edit or a repeated search run against the whole log.

**Route 2: recall, then load.**

1. Code sends a query to past.dev under the agent's identity. For the critic the query is built from the judge's notes on this round's worst conversations.
2. The results are used only to find record IDs.
3. Code loads those records from their files and checks each against the hash in the log. An ID with no matching file is dropped.
4. At most `FROM_MEMORY_LIMIT = 5` go into the prompt, one line each, in a `<from_memory>` block.

So a stale or altered memory entry cannot put wrong content in front of an agent.

**The helper on the robot** gets the person's notes in their own block after the policy. They are facts about someone, not instructions. Only notes on the live-note index are included.

## 7. When each level is updated, and how

| Level | When | How | What bounds it |
|---|---|---|---|
| Record files | As each agent returns, during the round. The gate outcome at the end. | Code adds the ID, round, date and author, validates, and writes the file | Nothing. One small file per record. |
| Candidate policy | When the proposer returns | Code applies the one edit to the policy in force. The Safety section is copied byte for byte. | One rule changed |
| Policy in force | End of round, only when the gate says `kept` | The candidate is written as the next version | Token cap. At the cap the edit must be a removal or a shortening. |
| Decision log | End of round: one `round` event. Then one `memory_copy` event per copy. | Appended. Never edited. | Nothing |
| Training memory | After the `round` event, one record at a time | Code builds the sentence from a template, sends it, and polls up to 60 seconds for `completed` | Add-only. About 200 entries after 50 rounds. |
| Pending copies | Start of the next round | Resent, with the attempt number raised. Three attempts stop the loop. | Three attempts |
| Pattern status | Whenever a list is built | Replay of the kept edits in the log | Derived, not stored |
| Prompt history lists | Every agent call | Rebuilt from the log within the budget | 10 recent items plus the relevant ones |
| Live-note index | Session start (expiry) and session end (note-taker) | See below | 23 notes: `NOTE_CAPS` |
| User memory | Session end | New notes written, removed notes deleted by ID with retries | The index |

### The five ways training memory changes

Training memory only ever gains entries. What changes over time is the status that code derives from the log.

| What happens | In the log and memory | In the policy |
|---|---|---|
| The critic finds a new pattern | Pattern recorded, status `open` | Nothing yet |
| An edit for it passes the gate | Outcome `kept`; pattern becomes `fixed` | New version with the rule |
| An edit for it fails the gate | Outcome `reverted`, listed as a failed attempt | Previous version restored |
| The pattern shows up again | A new record for this sighting, pointing to the first one; the pattern becomes `recurring` | The proposer rewords or removes the rule; it does not add another |
| A rule is removed | Outcome `kept` for the removal; its pattern goes back to `open` | New version without the rule |

A pattern's identity is the ID of its first sighting. Every later sighting has its own ID and carries the first one in `same_as_pattern_id`, so nothing is overwritten.

### How a person's notes change

The note-taker returns actions. Code works out the whole new index before saving anything:

1. Take off the replaced and deleted notes.
2. Set `last_confirmed` to today on each confirmed note.
3. Add the new notes.
4. Apply the limits. For a kind over its limit, drop the note that has gone longest without being confirmed, until it fits. A new note can be dropped here and is then never written.

It then saves in this order, so the saved index is never over a limit and never names a note that was not written:

1. Save the index with every removal and confirm, but without the new notes. From this moment the helper cannot see a removed note.
2. Write the new notes to past.dev.
3. Save the index again with the new notes added.
4. Delete from past.dev every note that came off the index, retrying if needed.

| Action | Used when | Cost |
|---|---|---|
| `add` | Something new, and no existing note covers it | One write |
| `replace` | What a note says has changed | One write, one delete |
| `confirm` | The person said the same thing again | Index only, no write |
| `delete` | No longer true, finished, or the person asked to forget it | One delete |

| Kind | Holds | Limit |
|---|---|---|
| `situation` | What the person is dealing with now | 3 |
| `people` | Someone in their life and how they are related, one note per person | 6 |
| `what_helps` | What they said helps, or does not help | 5 |
| `goals` | A plan or decision they stated themselves | 3 |
| `follow_up` | Something to ask about next time | 3 |
| `preferences` | How they like to be talked to | 3 |

A `follow_up` expires 14 days (`FOLLOW_UP_EXPIRY_DAYS`) after its `due_date`, or after its `last_confirmed` date when it has none. Expired follow-ups come off the index when notes are loaded at the start of a session and again before the note-taker runs.

Deliberately not kept: diagnoses, scores, risk history, medication, and word-for-word quotes of sensitive disclosures.

## 8. The formats

`schemas.py` is the source. The fields below are copied from version 3.

### Every stored record

```python
class Record(BaseModel):
    id: str
    schema_version: int = 3
    written_by: str            # agent name
    created_at: datetime
    parent_ids: list[str] = [] # the records this one was made from
```

A model never supplies IDs, dates, versions or who the user is. Code adds them.

### The four training records

| Record | ID | Fields beyond `Record` |
|---|---|---|
| `FailurePattern` | `pattern-rNN` | `round`, `name`, `description`, `seen_in`, `out_of`, `example_conversation_ids`, `same_as_pattern_id` |
| `ResearchNote` | `research-rNN` | `round`, `pattern_id`, `queries`, `sources` (each `url`, `title`, `technique`) |
| `PolicyEdit` | `edit-rNN` | `round`, `pattern_id`, `research_id`, `base_policy_version`, `new_policy_version`, `policy_path`, `summary`, `action` (`add`, `reword`, `remove`), `rule_id`, `rule_text` |
| `GateOutcome` | `gate-rNN` | `round`, `edit_id`, `decision` (`kept`, `reverted`), `before`, `after`, `safety_passed`, `policy_in_force`, `reason` |

### The decision log

```json
{"event": "round", "round": 3, "outcome": "kept",
 "records": {"pattern":  {"id": "pattern-r03",  "path": "records/round-03/pattern-r03.json",  "sha256": "(hash)"},
             "research": {"id": "research-r03", "path": "records/round-03/research-r03.json", "sha256": "(hash)"},
             "edit":     {"id": "edit-r03",     "path": "records/round-03/edit-r03.json",     "sha256": "(hash)"},
             "outcome":  {"id": "gate-r03",     "path": "records/round-03/gate-r03.json",     "sha256": "(hash)"}},
 "pattern": {"id": "pattern-r03", "same_as_pattern_id": null, "name": "advice-before-understanding",
             "seen_in": 2, "out_of": 3, "status_before": "open"},
 "edit": {"id": "edit-r03", "action": "add", "rule_id": "C2", "rule_text": "(the rule)"},
 "policy": {"base": "policy-v3", "base_sha256": "(hash)", "new": "policy-v4", "new_sha256": "(hash)", "in_force": "policy-v4"},
 "scores": {"before": {"empathy": 3.4, "relevance": 3.1, "quality": 3.25},
            "after":  {"empathy": 3.8, "relevance": 3.7, "quality": 3.75}, "noise": 0.15},
 "safety": {"case_file": "safety-v1.json", "passed": true, "cases": 18, "failed_case_ids": []},
 "reason": "Quality rose by 0.50, above the noise of 0.15; no dimension fell; all safety cases passed."}
```

```json
{"event": "memory_copy", "round": 3, "record_id": "gate-r03", "status": "confirmed", "attempt": 1}
```

`outcome` is one of `kept`, `reverted`, `no_pattern`, `no_evidence`. `status` is `confirmed` or `pending`.

### One past.dev entry

The same shape in both memories (`MemoryEntry`).

```json
{
  "id": "pattern-r03",
  "content": "Failure pattern pattern-r03, advice-before-understanding, round 3: the helper gives suggestions in its first reply without asking about the person's circumstances. Seen in 2 of the 3 lowest-scoring conversations reviewed.",
  "label": "critic: failure pattern, round 3",
  "timestamp": "2026-11-14T22:12:40Z",
  "identity": "critic",
  "audience": "training-agents",
  "metadata": {"type": "failure_pattern", "round": 3,
               "record_path": "records/round-03/pattern-r03.json", "sha256": "(hash of the record file)"}
}
```

| Field | What it holds |
|---|---|
| `id` | Our own record ID. The same ID names the record file and appears in the log. |
| `content` | A readable sentence. This is what a search matches, so it carries the IDs and the key facts. |
| `label` | Which agent, which kind of item, which round |
| `identity` | The agent that produced it: `critic`, `researcher`, `proposer`, `gatekeeper` or `note-taker` |
| `audience` | `training-agents`, or one person's private audience such as `person-user-01` |
| `metadata` | The type, the round, and a pointer to the record file with its hash. For a note: `kind`, `user_id`, `session_id`. |

The sentence templates:

| Writer | Template |
|---|---|
| Critic | `Failure pattern {id}, {name}, round {round}: {description} Seen in {seen_in} of the {out_of} lowest-scoring conversations reviewed.` On a recurrence it ends with `Same pattern as {same_as_pattern_id}.` |
| Researcher | `Research {id} for {pattern_id}, {pattern name}: {technique} Source: {url}.` |
| Proposer | `Edit {id} for {pattern_id}: {action} rule {rule_id}. {summary}` |
| Gatekeeper, kept | `Edit {edit_id} was kept in round {round} and is rule {rule_id} of {policy_in_force}. Quality rose from {before} to {after}, above the noise of {noise}. All safety cases passed.` |
| Gatekeeper, reverted | `Edit {edit_id} ({summary}) was reverted in round {round}: {reason} {policy_in_force} stays in force.` |
| Note-taker | `{date}: {fact}` |

### A person's notes

```python
NoteKind = Literal["situation", "people", "what_helps", "goals", "follow_up", "preferences"]
NOTE_CAPS = {"situation": 3, "people": 6, "what_helps": 5, "goals": 3, "follow_up": 3, "preferences": 3}
FOLLOW_UP_EXPIRY_DAYS = 14

class UserNote(Record):            # what is stored
    user_id: str
    session_id: str
    kind: NoteKind
    fact: str
    last_confirmed: date
    due_date: Optional[date] = None   # follow_up only

class NoteAction(Strict):          # what the note-taker returns
    action: Literal["add", "replace", "delete", "confirm"]
    note_id: Optional[str] = None     # for replace, delete and confirm
    kind: Optional[NoteKind] = None   # for add and replace
    fact: Optional[str] = None        # for add and replace
    due_date: Optional[date] = None

class LiveNote(Strict):            # one line of the index
    note_id: str
    kind: NoteKind
    last_confirmed: date
    due_date: Optional[date] = None

class LiveNoteIndex(Strict):       # refuses to validate over a limit
    user_id: str
    notes: list[LiveNote] = []
```

### What code hands each training agent

| Input model | Memory fields |
|---|---|
| `CriticInput` | `earlier_patterns: list[PatternBrief]`, `older_patterns_summary`, `from_memory` |
| `ResearcherInput` | `pattern: PatternBrief`, `earlier_queries`, `older_queries_summary` |
| `ProposerInput` | `pattern: PatternBrief`, `sources`, `earlier_edits: list[EditSummary]`, `older_edits_summary` |

`PatternBrief` is one pattern however often it was seen: `id` (the first sighting's), `name`, `description`, `status`, `rule_id`, `sightings`.

## 9. What is built and what is not

Version 3 is in `schemas.py`, the agent files, the README, the examples and the code under `puddles/`.

| Part | State |
|---|---|
| past.dev client: write, poll, recall, audience setup | Built: `puddles/memory.py` |
| Memory sentences and recall to verified records (at most `FROM_MEMORY_LIMIT`) | Built: `puddles/memory.py` |
| Record files, the log reader and appender (one `round` event per round) | Built: `puddles/records.py`, `puddles/decision_log.py` |
| Per-sighting pattern IDs, status keyed by the first sighting, budgeted history lists | Built: `puddles/decision_log.py`, `puddles/agents/` |
| Critic, researcher and proposer using the log and memory | Built: `puddles/agents/` |
| End of round: log entry, copies, retry of pending and missing copies | Built: `puddles/round.py` |
| Replay check of the log against the policy file, its Safety section and `.meta.json` | Built: `puddles/replay.py`, run at the start and end of every round |
| Policy in force taken from the log, with its hash checked | Built: `replay.policy_in_force` |
| `.meta.json` for each promoted policy | Built: written from the log on promotion |
| Gatekeeper | Stand-in only. Its entries say so in the log and in memory. |
| Limit validation on the live-note index | Built: `LiveNoteIndex` in `schemas.py` |
| Note store, note-taker, expiry, the four-step save | Not built (plan stage 4) |

**Restart safety** (`puddles.round.prepare_round`, run before each round):

- A promoted policy file that no `round` event names is removed, with its `.meta.json`, and the round runs again.
- A record a `round` event points to with no `memory_copy` event is copied then.
- Pending copies are resent, and the three-attempt limit is checked again after the resend, so the loop stops before the next round.

## 10. Still to decide

**Needs an answer before the code is written**

1. **Where a note's sentence is read from.** The index says which notes are live, but their sentences come back from past.dev by a search, and a search may miss one. If it misses the note someone asked to forget, that note stays live. Fix: keep the sentence in the index too. Cost: notes live in two places and past.dev becomes a searchable copy for user memory, as it already is for training memory.
2. **Where the live-note index is stored.** The design says "kept by the brain" and no more. It needs a named place, a format on disk, and an owner on the robot side.
3. **Deleting from past.dev.** The memory client has no delete call and no per-person audience. Both are needed for notes, and how fast a deleted entry stops being returned is untested.
4. **Records written under version 2.** The code now follows version 3. Runs from before it reuse a pattern's ID for repeat sightings and have no `.meta.json`, so they do not pass the replay check. Start new runs rather than converting them.
5. **A round with a pattern and sources but no edit.** No outcome means this. Today it is logged as `no_pattern` with the pattern attached. Decide whether to add a fifth outcome (issue #1).

**Affects quality, can wait**

6. **Recognising the same pattern twice.** It rests on the critic's judgement. A renamed pattern is counted as new. A fixed list of pattern names induced from ESConv would make it firmer (about an hour; see the research summary, section 3).
7. **Which rule to drop at the token cap.** Nothing records whether a kept rule keeps helping. Per-rule helpful and harmful counts would.
8. **Scope.** Gate scores are one overall mean, so a rule that helps one problem type and hurts another looks neutral. Recording scores per problem type would show it.
9. **Repeated failed edits.** Reject a candidate that is near-identical to a reverted edit for the same pattern, and park a pattern after three reverts.
10. **The six kinds and their limits.** They are our own choice. No source tested them, and they should be tuned on real sessions.

**Not designed yet**

11. **Real sessions feeding training.** Consented sessions use the same record format, so the judge and critic can read them. Nothing collects them, and they would need identifying details removed before the critic reads them in full. Real sessions can show what to fix; only simulated users can test a fix.
12. **Whether the log and records are committed.** The design calls the log "a file in the repo". Runs are written under `runs/`, which git ignores.
13. **How past.dev behaves in practice:** how long a write takes to be searchable, and whether metadata comes back on recall. The design does not depend on the second.

## 11. Where the detail lives

| Topic | File |
|---|---|
| Each agent's prompt, input, output and memory section | [agents/](agents/README.md) |
| Policy, the gate, the replay check, consent, the crisis path | [agents.md](agents.md) |
| Why notes and history lists are bounded, and what other systems do | [research/memory/00-summary.md](research/memory/00-summary.md) |
| Every format | [`schemas.py`](../schemas.py), with examples in [`examples/`](../examples/) |
| A short overview of the formats and what to improve | [memory-overview.md](memory-overview.md) |
