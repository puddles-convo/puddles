# Simulated user

**Training, step 1**

- Plays a person from a scenario, so the helper has someone to talk to.
- Sees only its scenario and the conversation. It has no access to memory or to the policy.

|  |  |
|---|---|
| Receives from | The orchestrator: one scenario, and the conversation so far |
| Hands to | [Helper](t2-i3-helper.md): the person's next line |
| Model | Qwen3.6-35B-A3B on Tinker. A different model from the helper, so the helper is not talking to itself. |
| Reads from memory | Nothing |
| Writes to memory | Nothing |
| Task | [#10](https://github.com/puddles-convo/puddles/issues/10) |

## 1. The prompt

### System message

The fixed instructions. Text in `{braces}` is filled in by code.

```
You are playing a person who has come to a support helper to talk. You are not an assistant.

Who you are:
- Situation: {situation}
- Main feeling: {emotion_type}, intensity {initial_intensity} out of 5
- Kind of problem: {problem_type}
- Openness: {openness}   (open, guarded or resistant)

How to play this person:
- Write the way people talk when they are upset: short, sometimes unfinished, one thought at a time.
- Give details only when asked or when it comes naturally. Do not explain your whole situation in your first message.
- React honestly to what the helper says. Being listened to makes you open up a little. Generic advice, being rushed or being told what to do makes you pull back or push back.
- Your feelings change slowly. One kind sentence does not fix anything.
- Never say you are simulated, never praise the helper's advice unless it truly fits, and never coach the helper.
- When the conversation has reached a natural end, or you would realistically give up, reply with exactly: [END]

Reply with only what the person says next.
```

### Chat messages

The conversation is sent as chat messages with the roles swapped, because this model is playing the person: the helper's lines arrive as `user` messages, and this agent's own earlier lines are `assistant` messages. The first message is always a `user` message, `(The helper is listening. Begin.)`, so the chat never opens with an `assistant` message, which many model servers reject.

## 2. Where the input goes in the prompt

| Place in the prompt | Filled with | Comes from |
|---|---|---|
| `{situation}` | `scenario.situation` | The scenario, loaded from ESConv |
| `{emotion_type}` | `scenario.emotion_type` | The scenario |
| `{initial_intensity}` | `scenario.initial_intensity` | The scenario; the user's own rating in ESConv |
| `{problem_type}` | `scenario.problem_type` | The scenario |
| `{openness}` | `scenario.openness` | Assigned by code when the scenario is loaded, so a share of scenarios are guarded or resistant |
| Chat messages | The opening message, then `turns` | The conversation so far, roles swapped |

## 3. Input

Format: `SimulatedUserInput` in [`schemas.py`](../../schemas.py). Example: [`examples/simulated_user_input.json`](../../examples/simulated_user_input.json).

```json
{
  "scenario": {
    "scenario_id": "esconv-validation-0042",
    "split": "validation",
    "emotion_type": "anxiety",
    "problem_type": "job crisis",
    "situation": "I am on short term disability and I am afraid I will lose my job if I don't go back soon.",
    "initial_intensity": 4,
    "openness": "guarded"
  },
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

## 4. Output

Plain text: the person's next line, or exactly `[END]`. There is no JSON to parse.

Example:

```
I'm on short term disability and I'm not ready to go back. I haven't been there long enough to have any protection.
```

### What code does with it

Code wraps the line as a `Turn` with `speaker: "user"` and appends it to the conversation. `[END]` closes the conversation.

### Checks before it is accepted

- The line is not empty.
- It is not longer than 80 words.
- It does not say it is an AI or a simulation.

## 5. Memory

### Reads

None. If it could read memory it would know what the loop has learned, and the test would stop being fair.

### Writes

None. Its lines are kept in the conversation record, which is a file, not a memory entry.

## 6. When it goes wrong

- An empty or out-of-character line is retried once.
- A second failure marks the conversation as failed. It is counted and logged, never silently dropped, because dropped conversations bias the scores.

---

[All agents](README.md) · [Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
