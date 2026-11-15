# Note-taker

**Inference, step 5, after the session ends**

- Decides what the robot should remember about this person for next time.
- It may add or replace notes only if the person agreed to be remembered. It may always delete.
- Every note has a kind, and each kind has a limit: 23 notes for one person at most. Code enforces the limits, so the notes stay the same size however many sessions there are.

|  |  |
|---|---|
| Receives from | The finished session |
| Hands to | User memory: that person's private notes |
| Model | A small OpenAI model. It runs after the session, so speed does not matter. |
| Reads from memory | This person's existing notes |
| Writes to memory | Notes added, replaced, confirmed or deleted |
| Task | [puddles-stack #7](https://github.com/puddles-convo/puddles-stack/issues/7), [#10](https://github.com/puddles-convo/puddles-stack/issues/10) |

## 1. The prompt

### System message

The fixed instructions. They are the same on every call.

```
You keep notes for a support robot so that it can remember a person between conversations. Read the finished conversation and the notes that already exist, and decide what should change.

You are told whether you may add notes. If you may not, return only delete actions.

Every note has one kind:
- situation: what the person is dealing with now.
- people: someone in their life and how they are related. One note per person.
- what_helps: something they said helps, or does not help.
- goals: a plan or decision they stated themselves.
- follow_up: something to ask about next time, or that they asked the robot to bring up. Give its date if they said one.
- preferences: how they like to be talked to.

Something they asked the robot to remember goes under the kind that fits it. If something fits none of the kinds, do not keep it.

Leave out: your own interpretations or diagnoses, and word-for-word quotes of sensitive disclosures. Write "said they have been feeling very low" rather than quoting it.

For each change, give one action:
- add: a new note, as one plain sentence that still makes sense read alone months later, with its kind.
- replace: a new sentence that takes the place of an existing note, with that note's ID and its kind. Use it when what the note says has changed. A follow_up keeps its date unless you give a new one.
- confirm: an existing note's ID, when the person said the same thing again and the note still holds as written.
- delete: an existing note's ID, when it is no longer true, is finished, or the person asked for it to be forgotten.

The notes are a short picture of the person as they are now, not a history. Before adding a note, check whether an existing one already covers it, and replace or confirm that one instead. Each kind has a limit, shown with the existing notes. When a kind is full, replace or delete a note of that kind to make room. If you do not, the note of that kind that has gone longest without being confirmed is dropped.

If the person asked the robot to forget something, do not write a new note about it. If they asked it to forget everything, delete every note and add nothing.

Return a JSON list of actions, each with: action, note_id (for replace, confirm and delete), kind and fact (for add and replace), due_date (for a follow_up with a date, as YYYY-MM-DD). Return an empty list if nothing should change.
```

### User message

```
<today>{today}</today>
<may_add_notes>{may_add_notes}</may_add_notes>

<existing_notes>
<{kind} count="{count}" limit="{limit}">
- {note_id}, last confirmed {last_confirmed}{, due {due_date}}: {fact}
...
</{kind}>
...
</existing_notes>

<conversation>
user: {turn text}
helper: {turn text}
...
</conversation>
```

## 2. Where the input goes in the prompt

| Place in the prompt | Filled with | Comes from |
|---|---|---|
| `<today>` | `today` | The clock, supplied by code. The model is never trusted for dates. |
| `<may_add_notes>` | `may_add_notes` | The person's answer to "may I remember this for next time?" |
| `<existing_notes>` | `existing_notes`, grouped by kind, one line each with its ID, the date it was last confirmed and, for a follow-up that has one, its due date. Every kind is shown with its count and its limit (`NOTE_CAPS`), including the empty ones. | The sentence comes from user memory. The kind and the dates come from the brain's index of live notes. Expired follow-ups are taken off first. |
| `<conversation>` | `turns` | The finished session |

## 3. Input

Format: `NoteTakerInput` in [`schemas.py`](../../schemas.py). Example: [`examples/note_taker_input.json`](../../examples/note_taker_input.json).

```json
{
  "today": "2026-11-15",
  "may_add_notes": true,
  "existing_notes": [
    {
      "note_id": "note-user01-002",
      "kind": "goals",
      "fact": "Planning to go back to work next week.",
      "last_confirmed": "2026-11-08"
    },
    {
      "note_id": "note-user01-003",
      "kind": "situation",
      "fact": "Works in a warehouse.",
      "last_confirmed": "2026-11-08"
    }
  ],
  "turns": [
    {
      "speaker": "user",
      "text": "I'm on short term disability and I keep worrying I'll lose my job."
    },
    {
      "speaker": "helper",
      "text": "That's a lot to carry. What's making the job feel at risk?"
    },
    {
      "speaker": "user",
      "text": "I've decided I'm not going back yet. I'm not ready."
    },
    {
      "speaker": "helper",
      "text": "So you need more time, and you're afraid of what that costs you."
    },
    {
      "speaker": "user",
      "text": "Yeah. When it gets bad I do that slow breathing, in for four, out for six. It helps a bit."
    }
  ]
}
```

## 4. Output

JSON, requested as a structured output: a list of changes, which may be empty.

Format: `NoteAction (a list)` in [`schemas.py`](../../schemas.py).

| Field | Type | Meaning |
|---|---|---|
| `action` | `add`, `replace`, `confirm` or `delete` | The change |
| `note_id` | ID or null | The existing note, for a replace, a confirm or a delete |
| `kind` | one of the six kinds, or null | For an add or a replace |
| `fact` | text or null | One plain sentence, for an add or a replace |
| `due_date` | date or null | For a `follow_up` that has a date |

Example:

```json
[
  {
    "action": "replace",
    "note_id": "note-user01-002",
    "kind": "goals",
    "fact": "Has decided not to go back to work yet; does not feel ready."
  },
  {
    "action": "add",
    "note_id": null,
    "kind": "situation",
    "fact": "On short-term disability and worried about losing their job."
  },
  {
    "action": "add",
    "note_id": null,
    "kind": "what_helps",
    "fact": "Slow breathing (in for four, out for six) helps a bit when the worry gets bad."
  }
]
```

A note that came up again and still holds is confirmed, not rewritten: `{"action": "confirm", "note_id": "note-user01-003"}`.

### What code does with it

Code adds the date, `user_id`, `session_id` and a new note ID to each added or replacing note, and sets its `last_confirmed` to today, making a `UserNote`. A replacing `follow_up` that gives no `due_date` keeps the old note's date.

Code first works out the new index in full, without saving anything:

1. Take off the replaced and deleted notes.
2. Set `last_confirmed` to today on each confirmed note.
3. Add the new notes.
4. Apply the limits. For any kind that is over its limit, drop notes of that kind, the one that has gone longest without being confirmed first, until it fits. On a tie the older note goes first. A new note can be dropped here too, and is then never written.

It then saves the result in this order, so the saved index is never over a limit and never names a note that was not written:

1. Save the index with every removal and every confirm, but without the new notes. From this moment the helper cannot see a replaced, deleted or dropped note. A confirm changes only the index, so it costs no write to past.dev.
2. Write the new notes to past.dev.
3. Save the index again with the new notes added.
4. Delete from past.dev every data point that came off the index, retrying if needed. The IDs dropped by a limit are logged.

`LiveNoteIndex` refuses to validate with more notes of a kind than its limit, so an over-limit index cannot be saved or loaded.

### The limits

| Kind | Holds | Limit |
|---|---|---|
| `situation` | What the person is dealing with now | 3 |
| `people` | Someone in their life and how they are related, one note per person | 6 |
| `what_helps` | What they said helps, or does not help | 5 |
| `goals` | A plan or decision they stated themselves | 3 |
| `follow_up` | Something to ask about next time | 3 |
| `preferences` | How they like to be talked to | 3 |

- The limits are `NOTE_CAPS` in `schemas.py`, 23 notes in all. They are our choice, to be tuned on real sessions.
- A `follow_up` expires 14 days (`FOLLOW_UP_EXPIRY_DAYS`) after its `due_date`, or after its `last_confirmed` date when it has no due date. Code takes expired follow-ups off the index when the helper's notes are loaded at the start of a session, and again before the note-taker runs.
- The index holds each live note's kind and dates (`LiveNote`), so the limits and the expiry are worked out by code, without a search.

### Checks before it is accepted

- When `may_add_notes` is false, every `add`, `replace` and `confirm` is dropped.
- If the person asked the robot to forget everything, the index is cleared and every `add`, `replace` and `confirm` from this session is dropped.
- Every `note_id` is one of the existing notes.
- Every `add` and `replace` has a kind and a fact. A `due_date` on any kind but `follow_up` is ignored.

## 5. Memory

### Reads

The existing notes about this person, so it can replace or delete them by ID:

```json
{
  "query": "What should the robot remember about this person?",
  "identity": "user-01",
  "level": "low"
}
```

Only notes whose IDs are on the brain's index of live notes are passed on.

### Writes

One entry per note, in that person's private audience.

Template for `content`:

```
{date}: {fact}
```

```json
{
  "id": "note-user01-005",
  "content": "15 Nov 2026: slow breathing (in for four, out for six) helps a bit when the worry gets bad.",
  "label": "note-taker: user note",
  "timestamp": "2026-11-14T22:12:40Z",
  "identity": "note-taker",
  "audience": "person-user-01",
  "metadata": {
    "type": "user_note",
    "kind": "what_helps",
    "user_id": "user-01",
    "session_id": "sess-0007"
  }
}
```

A replaced or deleted note is removed from past.dev by its ID. Nothing here ever reaches the training memory.

## 6. When it goes wrong

- Output that fails validation is retried once. A second failure changes nothing: no notes are added, and none are lost.
- A delete that past.dev does not confirm is retried. The note is already off the index, so the helper cannot see it in the meantime.

---

[All agents](README.md) · [Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
