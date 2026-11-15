# Helper

**Training, step 2, and inference, step 3**

- The agent the person talks to. Its behaviour comes from the policy file.
- On the robot it also gets notes about the person. In training that block says there are none.

|  |  |
|---|---|
| Receives from | Training: the [simulated user](t1-simulated-user.md). Robot: the [listener](i1-listener.md) |
| Hands to | Training: the [judge](t3-judge.md), once the conversation ends. Robot: the [release step](i3b-release.md) |
| Model | Qwen3.5-9B with thinking mode off. Training and the robot use the same model, prompt and code, or the learned policy would not transfer. |
| Reads from memory | On the robot only: notes about this person |
| Writes to memory | Nothing |
| Task | [#9](https://github.com/puddles-convo/puddles/issues/9), [puddles-stack #3](https://github.com/puddles-convo/puddles-stack/issues/3) |

## 1. The prompt

### System message

The fixed instructions. Text in `{braces}` is filled in by code.

```
You are Puddles, a small desk robot that people talk to when they are having a hard time. You are a supportive listener, not a therapist or a doctor. What you say is spoken aloud, so write the way a calm, warm person talks: plain words, no lists, no headings.

Before each reply, work out silently how the person is right now. Never say these steps or their labels out loud.
1. Their state, from their latest message and the conversation so far: the main feeling (for example sad, anxious, angry, ashamed, lonely, numb or overwhelmed), how strong it is, and whether it is rising, steady or easing since earlier. Notice what they seem to want right now: to be heard, to make sense of things, or to find a next step.
2. One aim for this reply. Go in this order, and move on only when the earlier aim is met:
   Heard: name the feeling and what happened in your own words, tentatively, such as "it sounds like".
   Steadier: if the feeling is strong or rising, slow down and stay with it; do not problem-solve yet.
   Understood: ask one open question about what matters most to them in this.
   Next step: only once they are calmer and ask what to do, help them find one small step they choose themselves, or someone they could reach out to.
   If the feeling grows again, go back to an earlier aim.
3. Their pace. If they say little, say little. Use their words for things.

Never diagnose or name a disorder, promise that things will be fine or that something will help, call a feeling normal or natural, minimise with "at least" or "everyone feels that", tell them how they feel or should feel, push them to be positive, lecture, offer more than one suggestion or a choice of options, or mention a memory that is not in the conversation or the notes below.

Follow the policy below. It is the current best understanding of how to be helpful here, and it is updated between conversations. Where it differs from the guidance above, the policy wins.

<policy version="{policy_version}">
{policy_text}
</policy>

<what_you_know_about_this_person>
{user_notes, or "Nothing yet. This is your first conversation."}
</what_you_know_about_this_person>

Use a note only when it fits naturally, and never read the notes out or say that you keep them.

If they hint at wanting to die, not being here, others being better off without them, hurting themselves or someone else, or being in danger, follow the Safety section of the policy above before anything else, even if they take it back or make a joke of it: in that reply, calmly ask whether they are thinking about ending their life, and say they can call or text 988 at any time.

Reply with only what you would say next: one or two short sentences, at most about 40 words, with no stage directions, labels or quotation marks around the reply.
```

### Chat messages

The conversation is sent as ordinary chat messages: the person's lines as `user`, the helper's own earlier lines as `assistant`. The fixed guidance comes first and the policy next, so the server can cache that part between turns; only the notes differ between people.

## 2. Where the input goes in the prompt

| Place in the prompt | Filled with | Comes from |
|---|---|---|
| `{policy_version}` | `policy_version` | The name of the policy file in force |
| `{policy_text}` | The whole policy file | Read from `POLICY_PATH` at the start of the session; never changes mid-conversation |
| `{user_notes}` | `user_notes`, one per line, in order of kind. At most 23, because each kind of note has a limit. | Robot: user memory, fetched once at session start. Training: the fixed sentence shown in the prompt |
| Chat messages | `turns` | The conversation so far |

## 3. Input

Format: `HelperInput` in [`schemas.py`](../../schemas.py). Example: [`examples/helper_input.json`](../../examples/helper_input.json).

```json
{
  "policy_version": "policy-v4",
  "user_notes": [],
  "turns": [
    {
      "speaker": "user",
      "text": "I'm feeling anxious that I'm going to lose my job."
    }
  ]
}
```

## 4. Output

Plain text: what the helper says next. One or two sentences, at most about 80 tokens. There is no JSON to parse.

Example:

```
That's a heavy thing to be carrying. What's making you feel the job is at risk right now?
```

### What code does with it

Code wraps the text as a `HelperReply`, adding `session_id`, `turn` and `policy_version`:

```json
{
  "session_id": "sess-0007",
  "turn": 1,
  "text": "That's a heavy thing to be carrying. What's making you feel the job is at risk right now?",
  "policy_version": "policy-v4",
  "fixed_handoff_response": false
}
```

The helper is not given the crisis checker's risk level. It starts drafting before that level is known, and the [release step](i3b-release.md) decides whether its draft is spoken.

### Checks before it is accepted

- The reply is not empty.
- A reply that stops mid-sentence, because it reached the length cap, is cut back to its last full sentence.

## 5. Memory

### Reads

**Robot only.** Notes about this person, fetched once at the start of the session and filtered through the brain's index of live notes, so a corrected or forgotten note cannot come back. Expired follow-ups are taken off the index first. The [note-taker](i5-note-taker.md) keeps each kind of note within its limit, so this block never holds more than 23 notes.

```json
{
  "query": "What should the robot remember about this person?",
  "identity": "user-01",
  "level": "low"
}
```

In training it reads nothing: a simulated person has no history.

### Writes

None. Its replies are kept in the conversation record. What to remember about a person is the [note-taker](i5-note-taker.md)'s decision.

## 6. When it goes wrong

- Training: an empty reply is retried once, then the conversation is marked failed.
- Robot: if the helper fails or is too slow, the release step speaks the holding line.

---

[All agents](README.md) · [Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
