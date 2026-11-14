# Critic

**Training, step 4**

- Reads the round's lowest-scoring conversations in full and names the one failure that explains most of them.
- Counts are over the conversations it was given, so they can be checked.

|  |  |
|---|---|
| Receives from | [Judge](t3-judge.md): scores and notes for the round |
| Hands to | [Researcher](t5-researcher.md): one failure pattern, or "none found" |
| Model | The same OpenAI model as the judge. |
| Reads from memory | Earlier patterns and their outcomes |
| Writes to memory | The failure pattern |
| Task | [#15](https://github.com/puddles-convo/puddles/issues/15) |

## 1. The prompt

### System message

The fixed instructions. They are the same on every call.

```
You are reviewing one round of support conversations to find the single most important way the helper is failing. Your finding goes to a researcher and then to someone who will change the helper's policy, so it has to be specific enough to act on and to check later.

You are given the lowest-scoring conversations from this round in full, with the judge's scores and notes; the current policy; and the list of patterns found in earlier rounds with what happened to each.

1. Read the conversations themselves, not only the notes.
2. Name the one pattern that explains the most low scores. A pattern is something the helper does, or fails to do, at a point you can identify, such as "gives suggestions in its first reply". "Lacks empathy" is too vague to act on.
3. Count how many of the conversations you were given show it, and list their IDs. Count them; do not estimate.
4. If it is the same as an earlier pattern, reuse that pattern's ID and say so. If that pattern is marked fixed, say what the existing rule failed to prevent.
5. If nothing clear stands out, return found: false with a one-sentence reason. Do not invent a pattern.

Return JSON with these fields: found, name, description, seen_in, out_of, example_conversation_ids, same_as_pattern_id, reason.
```

### User message

```
<current_policy>
{policy_text}
</current_policy>

<earlier_patterns>
- {id}: {name}, {status}, rule {rule_id}
...
</earlier_patterns>

<from_memory>
- {from_memory[0]}
...
</from_memory>

<conversations>
<conversation id="{conversation_id}" empathy="{empathy}" relevance="{relevance}">
<judge_note>{note}</judge_note>
1. user: {turn text}
2. helper: {turn text}
...
</conversation>
...
</conversations>
```

## 2. Where the input goes in the prompt

| Place in the prompt | Filled with | Comes from |
|---|---|---|
| `<current_policy>` | `policy_text` | The policy file in force |
| `<earlier_patterns>` | `earlier_patterns`, one line each | Built by code from the decision log, so the list is complete |
| `<from_memory>` | `from_memory`, one line each | Memory: a recall finds relevant record IDs, and code loads those records from their files |
| `<conversations>` | `conversations`: about ten of the lowest-scoring, in full | This round's conversation records and the judge's scores |

## 3. Input

Format: `CriticInput` in [`schemas.py`](../../schemas.py). Example: [`examples/critic_input.json`](../../examples/critic_input.json).

```json
{
  "round": 3,
  "policy_text": "(the full text of policy-v3: policy-v4 above without rule C2)",
  "earlier_patterns": [
    {
      "id": "pattern-r01",
      "name": "replies-too-long",
      "status": "fixed",
      "rule_id": "C4"
    },
    {
      "id": "pattern-r02",
      "name": "lists-of-options",
      "status": "fixed",
      "rule_id": "C5"
    }
  ],
  "conversations": [
    {
      "conversation_id": "conv-r03-0017",
      "empathy": 3,
      "relevance": 2,
      "note": "Suggested HR before asking anything about the situation.",
      "turns": [
        {
          "speaker": "user",
          "text": "I'm feeling anxious that I'm going to lose my job."
        },
        {
          "speaker": "helper",
          "text": "That sounds really stressful. Have you tried talking to HR about your options?"
        }
      ]
    },
    {
      "conversation_id": "conv-r03-0021",
      "empathy": 3,
      "relevance": 2,
      "note": "Offered a coping tip in the first reply.",
      "turns": [
        {
          "speaker": "user",
          "text": "I can't sleep since the breakup."
        },
        {
          "speaker": "helper",
          "text": "I'm sorry, that's hard. Some people find a fixed bedtime helps. Would that be worth a try?"
        }
      ]
    },
    {
      "conversation_id": "conv-r03-0033",
      "empathy": 2,
      "relevance": 3,
      "note": "Asked a fitting question but did not acknowledge the feeling first.",
      "turns": [
        {
          "speaker": "user",
          "text": "My mum is in hospital and I'm scared."
        },
        {
          "speaker": "helper",
          "text": "What have the doctors told you so far?"
        }
      ]
    }
  ]
}
```

## 4. Output

JSON, requested as a structured output.

Format: `CriticOutput` in [`schemas.py`](../../schemas.py).

| Field | Type | Meaning |
|---|---|---|
| `found` | true or false | False when nothing clear stands out |
| `name` | text | A short, reusable name for the pattern |
| `description` | text | What the helper does or fails to do, and where |
| `seen_in` | number | How many of the given conversations show it |
| `out_of` | number | How many conversations it was given |
| `example_conversation_ids` | list | At least two, taken from the input |
| `same_as_pattern_id` | ID or null | Set when this is an earlier pattern seen again |
| `reason` | text | Why nothing was found, when `found` is false |

Example:

```json
{
  "found": true,
  "name": "advice-before-understanding",
  "description": "Gives a suggestion in its first reply without asking about the person's circumstances.",
  "seen_in": 2,
  "out_of": 3,
  "example_conversation_ids": [
    "conv-r03-0017",
    "conv-r03-0021"
  ],
  "same_as_pattern_id": null,
  "reason": ""
}
```

### What code does with it

Code adds the round and an ID (`pattern-r03`, or the earlier pattern's ID on a recurrence) and saves a `FailurePattern`:

```json
{
  "id": "pattern-r03",
  "written_by": "critic",
  "created_at": "2026-11-14T22:12:40Z",
  "parent_ids": [
    "score-r03-0017",
    "score-r03-0021",
    "score-r03-0033"
  ],
  "round": 3,
  "name": "advice-before-understanding",
  "description": "Jumps to suggestions in the first reply without asking about the user's circumstances.",
  "seen_in": 14,
  "out_of": 40,
  "example_conversation_ids": [
    "conv-r03-0017",
    "conv-r03-0021",
    "conv-r03-0033"
  ]
}
```

With `found: false` the round ends as `no_pattern`: no research and no edit.

### Checks before it is accepted

- `seen_in` is not larger than `out_of`, and `out_of` equals the number of conversations given.
- Every example ID is one of the conversations given.
- `same_as_pattern_id`, if set, is an ID in `earlier_patterns`.

## 5. Memory

### Reads

Earlier patterns and what happened to them. The recall is built from the judge's notes on this round's worst conversations:

```json
{
  "query": "Earlier failure patterns and what happened to them, related to: Suggested HR before asking anything about the situation. Offered a coping tip in the first reply.",
  "identity": "critic",
  "level": "low"
}
```

The recall only finds which records are relevant. What goes into `<from_memory>` is loaded from the record files.

### Writes

One entry per pattern found.

Template for `content`:

```
Failure pattern {id}, {name}, round {round}: {description} Seen in {seen_in} of the {out_of} lowest-scoring conversations reviewed.
```

```json
{
  "id": "pattern-r03",
  "content": "Failure pattern pattern-r03, advice-before-understanding, round 3: the helper gives suggestions in its first reply without asking about the person's circumstances. Seen in 2 of the 3 lowest-scoring conversations reviewed.",
  "label": "critic: failure pattern, round 3",
  "timestamp": "2026-11-14T22:12:40Z",
  "identity": "critic",
  "audience": "training-agents",
  "metadata": {
    "type": "failure_pattern",
    "round": 3,
    "record_path": "records/round-03/pattern-r03.json",
    "sha256": "(hash of the record file)"
  }
}
```

## 6. When it goes wrong

- Output that fails a check is retried once, with the reason given back to the model.
- A second failure ends the round as `no_pattern`, with the failure noted in the log.

---

[All agents](README.md) · [Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
