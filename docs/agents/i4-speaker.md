# Speaker

**Inference, step 4**

- Turns the reply into speech, a sentence at a time.
- The robot plays each piece as it arrives.

|  |  |
|---|---|
| Receives from | [Release step](i3b-release.md): the text to say |
| Hands to | The robot's speaker. When the session ends, the [note-taker](i5-note-taker.md) |
| Model | A text-to-speech model: Kokoro, on the same GPU as the helper. |
| Reads from memory | Nothing |
| Writes to memory | Nothing |
| Task | puddles-stack #1, #2 |

## 1. The prompt

A speech model, so it has no prompt.

## 2. Where the input goes in the prompt

Not applicable: there is no prompt.

## 3. Input

The `text` of a `HelperReply`:

```
That's a heavy thing to be carrying. What's making you feel the job is at risk right now?
```

## 4. Output

Audio: 16 kHz, mono, 16-bit, the format the robot already plays. One piece per sentence, sent as soon as each is ready. Kokoro produces 24 kHz, so the audio is resampled in the cloud.

### What code does with it

When the session ends, code saves the whole session as a `ConversationRecord` with `source: "live"`:

```json
{
  "id": "sess-0007",
  "written_by": "speaker",
  "created_at": "2026-11-15T10:02:11Z",
  "parent_ids": [],
  "source": "live",
  "scenario_id": null,
  "split": null,
  "policy_version": "policy-v4",
  "turns": [
    {
      "speaker": "user",
      "text": "I'm feeling anxious that I'm going to lose my job."
    },
    {
      "speaker": "helper",
      "text": "That's a heavy thing to be carrying. What's making you feel the job is at risk right now?"
    }
  ]
}
```

### Checks before it is accepted

- The robot's microphone is off while it is speaking, so it does not answer itself.

## 5. Memory

### Reads

None. It only speaks.

### Writes

None. The session record is a file. It goes to training only if the person agreed.

## 6. When it goes wrong

- If speech fails, the robot shows the reply on its screen and plays the holding line when it can.
- The crisis and offline lines are stored on the device as audio, so they do not depend on this step.

---

[All agents](README.md) · [Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
