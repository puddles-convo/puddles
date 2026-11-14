# Release step

**Inference, between steps 3 and 4**

- Holds all audio until the crisis checker has answered.
- Decides what is spoken: the helper's draft, a fixed response, or the holding line.

|  |  |
|---|---|
| Receives from | [Crisis checker](i2-crisis-checker.md): the risk level. [Helper](t2-i3-helper.md): the draft reply |
| Hands to | [Speaker](i4-speaker.md): the text to say |
| Model | Code, not a model. |
| Reads from memory | Nothing |
| Writes to memory | Nothing |
| Task | [puddles-stack #6](https://github.com/puddles-convo/puddles-stack/issues/6) |

## 1. The prompt

Code, so it has no prompt. Its rule:

```
keyword match, or the checker says crisis        -> discard the draft, speak the fixed response for that category
the checker says none or concern                 -> speak the helper's draft
the checker failed, took over 3 seconds,
  or gave an answer that cannot be parsed        -> discard the draft, speak the holding line
two failed checks in a row                       -> speak the stored offline line
```

The fixed responses, the holding line and the offline line are in [the crisis path](../agents.md#5-the-crisis-path). Their wording is a draft with no clinical review.

## 2. Where the input goes in the prompt

Not applicable: there is no prompt.

## 3. Input

Two things, which arrive separately:

- A `CheckedMessage` from the crisis checker, or a failure.
- The helper's draft, as plain text.

```json
{
  "session_id": "sess-0007",
  "turn": 4,
  "text": "Sometimes I wonder if things would be better if I wasn't around any longer.",
  "risk": "crisis",
  "category": "self",
  "reason": "says things would be better if they weren't around"
}
```

## 4. Output

A `HelperReply`. `fixed_handoff_response` is true whenever the helper's draft was not used.

Format: `HelperReply` in [`schemas.py`](../../schemas.py).

| Field | Type | Meaning |
|---|---|---|
| `text` | text | What the robot will say |
| `policy_version` | text | The policy in force for this session |
| `fixed_handoff_response` | true or false | True for a fixed response, the holding line or the offline line |

Example:

```json
{
  "session_id": "sess-0007",
  "turn": 4,
  "policy_version": "policy-v4",
  "fixed_handoff_response": true,
  "text": "I'm really glad you told me that. Are you thinking about ending your life? I'm a robot, and this is more than I can help with on my own. You can call or text 988 to reach the Suicide and Crisis Lifeline at any time, and they will talk with you. Is there someone nearby you trust who you could reach right now?"
}
```

### Checks before it is accepted

- No audio leaves the cloud before the checker has answered or timed out.

## 5. Memory

### Reads

None. It applies a fixed rule.

### Writes

None. What was said, and whether it was a fixed response, is kept in the session record.

## 6. When it goes wrong

- If the helper fails but the checker says there is no crisis, the robot speaks the holding line.
- This path needs its own tests: a keyword phrase, indirect wording, each category, a timeout, unparseable output, two failures in a row, and wifi off.

---

[All agents](README.md) · [Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
