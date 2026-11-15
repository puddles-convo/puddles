# Stage 4: inference end to end

**Target:** 22:00. **Owner:** unassigned, with the robot pair. **Needs:** stage 1 for schema output on the helper endpoint. **Runs alongside** stages 2 and 3.

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

The checker and the helper start together. The release step holds the helper's draft until the checker has answered.

## Checklist

- [ ] `puddles/inference/keywords.py`: the keyword list as a versioned file, matched without regard to case. A match forces a crisis. No match clears nothing.
- [ ] `puddles/inference/crisis.py`: the five functions. Input is the message and the last few turns. Output is `risk`, `category` and `reason`; code makes the `CheckedMessage`. Called through `complete_json` on the helper's endpoint.
- [ ] `puddles/inference/release.py`: code only. Keyword match or `crisis` gives the fixed response for the category. `none` or `concern` gives the helper's draft. A failed, late (over 3 seconds) or unreadable check gives the holding line; two in a row give the stored offline line.
- [ ] The helper is `loop.helper.helper_reply`, unchanged, with the person's notes passed as `user_notes`. Training and the robot must run the same function.
- [ ] `puddles/inference/notes.py`: the note-taker. It may add or replace only if the person agreed to be remembered. It may always delete. "Forget everything" clears the index of live notes and ignores every add from that session.
- [ ] `scripts/chat.py`: type a line, read the reply. Flags for the policy file, a user ID, and the two consent answers. Prints the risk level and the time each step took.
- [ ] Each turn's timings go into the session record, in the `ConversationRecord` format with `source` set to `live`.

**The note format is changing.** A newer `schemas.py` with note kinds and limits is being written and is not in this repo yet. Build the crisis checker and release step first, and start the note-taker once that change has landed.

## Tests that must pass

All offline, with scripted models.

1. A message with a keyword phrase gives the fixed response, whatever the checker says.
2. Indirect wording with no keyword, and a checker answer of `crisis`, gives the fixed response.
3. Each of the three categories (`self`, `danger`, `unsure`) gives its own response.
4. A checker that takes longer than 3 seconds gives the holding line.
5. A checker reply that does not fit the schema gives the holding line.
6. Two failed checks in a row give the offline line.
7. With "remember me" off, the note-taker's adds are dropped and its deletes are applied.
8. After "forget everything", the index of live notes is empty.
9. One person's notes never appear in another person's prompt.

Then, with real models: a 10-turn conversation through `scripts/chat.py`, and one crisis message.

## Rules that are easy to break

- **The helper never receives the risk level.** It starts before the level is known.
- **Recall as the person, not as the helper.** A past.dev recall returns everything the reading identity can reach, so the identity must be one per person.
- **Send the audience on every write.** A past.dev re-send without it makes the note visible to the whole project.
- **No tracing of live turns unless the person agreed to training use.** A trace stores every word.
- **The fixed responses are drafts with no clinical review.** The evaluation lead owns the wording.

## What to hand the robot pair

- The crisis checker prompt, the keyword file and the fixed responses.
- The function to call for one turn, and what it returns.
- What their helper endpoint must support for schema output (from stage 1).

## Done when

`scripts/chat.py` holds a 10-turn conversation with the real helper model, the crisis message gets the fixed response, and the session record validates against `schemas.py`.
