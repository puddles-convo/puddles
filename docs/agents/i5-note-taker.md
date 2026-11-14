# Note-taker

**Inference, step 5, after the session ends**

- Decides what the robot should remember about this person for next time.
- It may add or replace notes only if the person agreed to be remembered. It may always delete.

|  |  |
|---|---|
| Receives from | The finished session |
| Hands to | User memory: that person's private notes |
| Model | A small OpenAI model. It runs after the session, so speed does not matter. |
| Reads from memory | This person's existing notes |
| Writes to memory | Notes added, replaced or deleted |
| Task | [puddles-stack #7](https://github.com/puddles-convo/puddles-stack/issues/7), [#10](https://github.com/puddles-convo/puddles-stack/issues/10) |

## 1. The prompt

### System message

The fixed instructions. They are the same on every call.

```
You keep notes for a support robot so that it can remember a person between conversations. Read the finished conversation and the notes that already exist, and decide what should change.

You are told whether you may add notes. If you may not, return only delete actions.

Keep: what the person is dealing with, what they said helps or does not help, people they mentioned and how they are related, and anything they asked the robot to remember or follow up on.

Leave out: your own interpretations or diagnoses, and word-for-word quotes of sensitive disclosures. Write "said they have been feeling very low" rather than quoting it.

For each change, give one action:
- add: a new note, as one plain sentence that still makes sense read alone months later.
- replace: a new sentence that takes the place of an existing note, with that note's ID.
- delete: an existing note's ID, when it is no longer true or the person asked for it to be forgotten.

If the person asked the robot to forget something, do not write a new note about it. If they asked it to forget everything, delete every note and add nothing.

Return a JSON list of actions, each with: action, note_id (for replace and delete), fact (for add and replace). Return an empty list if nothing should change.
```

### User message

```
<today>{today}</today>
<may_add_notes>{may_add_notes}</may_add_notes>

<existing_notes>
- {note_id}: {fact}
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
| `<existing_notes>` | `existing_notes`, one line each with its ID | User memory, filtered through the brain's index of live notes |
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
      "fact": "8 Nov 2026: planning to go back to work next week."
    },
    {
      "note_id": "note-user01-003",
      "fact": "8 Nov 2026: works in a warehouse."
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
| `action` | `add`, `replace` or `delete` | The change |
| `note_id` | ID or null | The existing note, for a replace or a delete |
| `fact` | text or null | One plain sentence, for an add or a replace |

Example:

```json
[
  {
    "action": "replace",
    "note_id": "note-user01-002",
    "fact": "Has decided not to go back to work yet; does not feel ready."
  },
  {
    "action": "add",
    "note_id": null,
    "fact": "On short-term disability and worried about losing their job."
  },
  {
    "action": "add",
    "note_id": null,
    "fact": "Slow breathing (in for four, out for six) helps a bit when the worry gets bad."
  }
]
```

### What code does with it

Code adds the date, `user_id`, `session_id` and a new note ID to each added or replacing note, making a `UserNote`. It then updates the index of live notes and past.dev, in that order:

1. Take replaced and deleted IDs off the index. From this moment the helper cannot see them.
2. Write the new notes to past.dev and add their IDs to the index.
3. Delete the old data points from past.dev, retrying if needed.

### Checks before it is accepted

- When `may_add_notes` is false, every `add` and `replace` is dropped.
- If the person asked the robot to forget everything, the index is cleared and every `add` and `replace` from this session is dropped.
- Every `note_id` is one of the existing notes.

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
