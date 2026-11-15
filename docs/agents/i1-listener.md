# Listener

**Inference, step 1**

- Turns the person's speech into text.
- Sends that text to the crisis checker and the helper at the same time.

|  |  |
|---|---|
| Receives from | The robot: audio streamed over one open WebSocket |
| Hands to | [Crisis checker](i2-crisis-checker.md) and [Helper](t2-i3-helper.md), together |
| Model | A speech-to-text model: faster-whisper with large-v3-turbo, on the same GPU as the helper. |
| Reads from memory | Nothing |
| Writes to memory | Nothing |
| Task | puddles-stack #1, #2 |

## 1. The prompt

A speech model, so it has no prompt.

## 2. Where the input goes in the prompt

Not applicable: there is no prompt.

## 3. Input

Audio: 16 kHz, mono, 16-bit, sent in 250 ms chunks, then an end-of-speech marker. Example: `utterance-sess-0007-01.wav`, 3.2 seconds.

The robot decides the person has finished after about one second of silence.

## 4. Output

A `UserMessage`. Code adds `session_id` and `turn`.

Format: `UserMessage` in [`schemas.py`](../../schemas.py).

| Field | Type | Meaning |
|---|---|---|
| `session_id` | text | Added by code |
| `turn` | number | Added by code |
| `text` | text | What the person said |

Example:

```json
{
  "session_id": "sess-0007",
  "turn": 1,
  "text": "I'm feeling anxious that I'm going to lose my job."
}
```

### Checks before it is accepted

- An empty transcript does not start a turn.

## 5. Memory

### Reads

None. It only transcribes.

### Writes

None. The audio is not stored anywhere. The text becomes part of the session record.

## 6. When it goes wrong

- With no connection the robot cannot understand speech at all. It plays the offline line stored on the device, which gives the 911 and 988 numbers.
- Mishearing is a real risk in a noisy room. Session records keep the transcript, so it can be checked afterwards.

---

[All agents](README.md) · [Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
