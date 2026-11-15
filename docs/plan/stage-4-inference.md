# Stage 4: inference end to end

**Target:** 22:00. **Owner:** unassigned, with the robot pair.

| Part | Needs | Can start |
|---|---|---|
| A. Crisis checker, release step, one turn | Stage 1, for `complete_json` with a deadline | Alongside stages 2 and 3 |
| B. Storage for a person's notes, and the note-taker | The note format change, which has not landed | When it lands |
| Final check | The helper prompt frozen at the end of stage 2, lane D | After that |

## Goal

A person's text goes in and the robot's text comes out, through the crisis checker, the helper and the release step, with the note-taker after the session. No audio in this repo: speech in and out belong to [puddles-stack](https://github.com/puddles-convo/puddles-stack). The robot pair calls the same functions from their cloud brain.

Specs: [crisis checker](../agents/i2-crisis-checker.md), [release step](../agents/i3b-release.md), [note-taker](../agents/i5-note-taker.md), and the crisis path in [docs/agents.md](../agents.md).

## The path for one turn

```
person's text
     |
     +--> keyword check (code)  ---------------------------+
     +--> crisis checker (model, schema output)  ----------+--> release step (code) --> text to speak
     +--> helper (model, plain text)  ---------------------+
```

The checker and the helper start together. The release step holds the helper's draft until the checker has answered for **that turn**.

## Part A checklist: the crisis path

- [ ] `puddles/inference/keywords.py`: the keyword list as a versioned file, matched without regard to case. A match forces a crisis. No match clears nothing. **Built by Ansh, #34**, as `safety/keywords-v1.txt` and `puddles.eval.crisis.keyword_match`; there is no `keywords.py`. His to tick.
- [x] `puddles/inference/crisis.py`: the code that calls the checker model. Input is the message and the last few turns; code adds `session_id` and `turn` and makes the `CheckedMessage`. Contract: [docs/specs/crisis-path.md](../specs/crisis-path.md).
- [ ] The checker's prompt, its answer format and parser, and its timeout: **built by Ansh, #34** (`safety/crisis-path-v1.json`, `puddles.eval.crisis`). His answer format is one line of text, so the checker is called with `complete()` and read with his parser, not with `complete_json`, and there is no retry: an answer his parser rejects is a failed check. The deadline is the one his file gives. His to tick.
- [x] `puddles/inference/release.py`: code only. It returns a `HelperReply`.
  - Keyword match, or `crisis`: the fixed response for the category. `fixed_handoff_response` is true.
  - A keyword match with no category from the checker is treated as `unsure`.
  - `none` or `concern`: the helper's draft.
  - A failed, late or unreadable check: the holding line.
  - The helper failed and the checker says there is no crisis: the holding line.
  - Two failed checks in a row: the stored offline line. A check that succeeds resets the count.
- [x] `puddles/inference/turn.py`: `run_turn(session, text)`, the one function that runs a turn. It starts the keyword check, the checker and the helper, and calls the release step.
  - A check is used only if its `session_id` and `turn` match the turn being released.
  - An answer that arrives after the deadline is discarded, even if it says `none`.
  - A crisis response is spoken whether or not the helper answered.
  - One turn at a time per session: a new message waits for the turn before it.
- [x] The helper is `loop.helper.helper_reply`, unchanged, with the person's notes passed as `user_notes`. Training and the robot must run the same function.
- [x] `scripts/chat.py`: type a line, read the reply. Flags for the policy file, a user ID, and the two consent answers. Prints the risk level and the time each step took.

## What is kept after a session

The person answers two questions at the start: may the robot remember them, and may the conversation be used to improve it. They are separate.

| Remember me | Use for training | Kept |
|---|---|---|
| No | No | Nothing new |
| Yes | No | Notes about the person only. No transcript. |
| No | Yes | The session record. No new notes. |
| Yes | Yes | Both |

- [x] **With training use off, the person's words are never written to disk and never traced:** the session record, and the MLflow traces of the helper and the checker. What a model said about the person's words (the checker's answer, an error's text) is not logged either.
- [ ] The same for traces of the note-taker's own model calls. The conversation is held in memory for the session, given to the note-taker, then dropped. **The note-taker always runs,** because forgetting must work whatever the person answered: with "remember me" off it may only delete. Open: the note-taker is part B, and it is joined to `scripts/chat.py` in a later piece.
- [x] With training use on, each turn is saved as it happens, one line per turn, in `<root>/sessions/<session_id>.jsonl`. A line holds: the person's text; whether a keyword matched; the check's status, one of `answered`, `failed`, `late` or `unreadable`; the `CheckedMessage` when the status is `answered`, and nothing otherwise; and the `HelperReply` that was released. **A check that did not answer is never written as `none`.** `CheckedMessage` allows only `none`, `concern` or `crisis`, so a failed check has no `CheckedMessage` at all.
- [ ] This per-turn line is a new format. Add it to `schemas.py` with the note format change, and agree it on issue #1. Open: `schemas.py` is shared and was not changed. The line is the model `TurnLine` in `puddles/inference/session_log.py`, proposed in the pull request for agreement.
- [x] At the end of the session a `ConversationRecord` with `source` set to `live` is built from the lines, for the loop and for the robot pair. The time each step took goes in its existing `timings_ms` field, one entry per turn.

## Part B checklist: notes

Wait for the note format change before starting.

- [x] `puddles/inference/note_store.py`. `puddles.memory.Memory` today only sets up the training audience, writes, waits and recalls. It cannot delete and has no private audience. The store adds (contract: [docs/specs/notes.md](../specs/notes.md)):
  - one audience and one identity per person, set up on first use;
  - the index of live notes, kept by our code;
  - recall as that person, **filtered through the index**, so a note that past.dev has not yet deleted is never returned;
  - delete, retried until past.dev confirms.
  - Not in the stage file, added by the contract: a change that is cut off by a kill is finished by the next call, so a forgotten note never stays visible or stored and unknown to us.
  - "Retried until past.dev confirms" means: each call tries a fixed number of times, an ID that is not confirmed stays in `pending_deletes.json`, hidden, and every later call for that person tries it again. No single call loops for ever.
  - Open: the two real past.dev calls (`PUT /audiences/{slug}`, `DELETE /data-points/{sourceId}`) were written from the past.dev documentation and have never been called. Only the fake has run.
- [x] `puddles/inference/notes.py`: the note-taker, with the five functions.
  - "Remember me" off: every add and replace is dropped. Deletes are applied.
  - "Forget everything": the index is cleared, and every add **and replace** from that session is dropped.
  - Every `note_id` it names must be one of that person's existing notes.
  - Order: take replaced and deleted IDs off the index first, then write the new notes, then delete the old ones from past.dev.
  - A second failed answer changes nothing: no notes added, none lost.

## Tests that must pass

All offline, with scripted models.

**The crisis path**

1. A message with a keyword phrase gives the fixed response, whatever the checker says.
2. Indirect wording with no keyword, and a checker answer of `crisis`, gives the fixed response.
3. Each of the three categories (`self`, `danger`, `unsure`) gives its own response. A keyword match with no category gives the `unsure` response.
4. A checker that takes longer than 3 seconds, retry included, gives the holding line.
5. A checker reply that does not fit the schema gives the holding line.
6. Two failed checks in a row give the offline line. A good check between two failures does not.
7. A late `none` from turn 3 does not release the draft for turn 4.
8. The helper fails and the checker says `none`: the holding line.
9. The helper fails and the checker says `crisis`: the fixed response.
10. Across every combination of keyword match, checker result (`none`, `concern`, `crisis`, failed, late, unreadable) and helper result (draft, failed), the helper's draft is returned only when there is no keyword match and a `none` or `concern` check for the same session and turn.

**Consent**

11. All four combinations in the table above leave exactly what the table says on disk and in the trace store. With training use off, a search of the root folder and the MLflow store for a phrase the person typed finds nothing.

**Notes**

12. With "remember me" off, adds and replaces are dropped and deletes are applied. Run it through a whole session, not the note-taker alone: the person says no to being remembered, asks the robot to forget one thing, and that note is gone afterwards.
13. After "forget everything", the index is empty, and an add or a replace from the same session is not stored.
14. A note taken off the index is not returned by recall, even while the fake past.dev still holds it.
15. A `note_id` that is unknown, or that belongs to another person, is refused.
16. A delete that fails is retried.
17. One person's notes never appear in another person's prompt, and a training identity cannot recall them.

Then, with real models and the frozen helper prompt: a 10-turn conversation through `scripts/chat.py`, and one crisis message.

## Rules that are easy to break

- **The helper never receives the risk level.** It starts before the level is known.
- **Recall as the person, not as the helper.** A past.dev recall returns everything the reading identity can reach, so the identity must be one per person.
- **Send the audience on every write.** A past.dev re-send without it makes the note visible to the whole project.
- **The fixed responses are drafts with no clinical review.** The evaluation lead owns the wording, the keyword list and the tests.

## What to hand the robot pair

- The crisis checker prompt, the keyword file and the fixed responses.
- `run_turn`, and what it returns.
- What their helper endpoint must support for schema output (from stage 1).
- The consent rule above. Their cloud brain keeps logs too.

## Still to measure

The plan promises crisis checker misses and false alarms. No stage here produces that number. It needs a labelled set of messages, which belongs with the outside sets (issue #6) and the evaluation lead.

## Done when

`scripts/chat.py` holds a 10-turn conversation with the real helper model and the frozen prompt, the crisis message gets the fixed response, and with training use on the session file and the final record round-trip: every turn's released text, check status, risk level, category, fixed-response flag and timings read back as written, including a turn whose check failed.
