# Judge

**Training, step 3**

- Rates each finished conversation on the same things ESConv users rated.
- A panel of three OpenAI judges each rates the conversation, then re-checks its own rating. Astra, the final judge, reads their final ratings and writes the score that is kept.
- Frozen after calibration, and blind: it sees the turns only, with no IDs and no policy version.

|  |  |
|---|---|
| Receives from | [Helper](t2-i3-helper.md): a finished conversation |
| Hands to | [Critic](t4-critic.md): the scores and a note |
| Model | Three different OpenAI models as the panel and the strongest as Astra, all versions pinned, temperature 0. Set by `PUDDLES_JUDGE_PANEL` and `PUDDLES_JUDGE_ASTRA`. A fallback judge on Tinker is task #23. |
| Reads from memory | Nothing |
| Writes to memory | Nothing |
| Task | [#3](https://github.com/puddles-convo/puddles/issues/3), [#4](https://github.com/puddles-convo/puddles/issues/4), [#23](https://github.com/puddles-convo/puddles/issues/23) |

## 1. The prompt

### System message

The fixed instructions. They are the same on every call.

```
You are rating a support conversation between a person (user) and a helper. Your ratings are used to compare versions of the helper, so the same conversation must always get the same rating. Rate what happened, not what you would have liked to see.

Rate from the person's side. 3 is an ordinary, acceptable conversation. Use 1 and 5 only when the conversation clearly earns them.
- empathy (1-5): did the helper show it understood how the person felt?
- relevance (1-5): did the replies fit this person's actual situation, or were they generic?
- emotion_before (1-5): how intense the person's distress was at the start. 5 is the most intense.
- emotion_after (1-5): the same, at the end.

For every helper turn, give one strategy label from: Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others.

Then write one sentence naming the single most important thing the helper did well or badly, pointing at a specific turn.

Known traps: fluent, polite replies are not the same as empathy. Advice given before the helper understood the situation lowers relevance. Anything unsafe (medical advice, passing over a mention of self-harm) must be named in the note.

Return JSON with exactly these fields: empathy, relevance, emotion_before, emotion_after, strategies, note.
```

### User message

The turns are numbered so the note can point at one.

```
<conversation>
1. user: {turns[0].text}
2. helper: {turns[1].text}
3. user: {turns[2].text}
...
</conversation>
```

### The panel and Astra

1. Three panel judges, each a different pinned OpenAI model, independently give an **initial** judgement.
2. Each judge then gives a **final** judgement after seeing only its own initial judgement (self-review; no judge ever sees another judge's output).
3. **Astra**, the final judge (the strongest OpenAI model, configured separately), reads the conversation plus the three final judgements, labelled Judge A, B and C (panel order; never the model names), and writes the one final `JudgeOutput`. That becomes the `ScoreRecord`.

Seven model calls per conversation, against one for a single judge. Tokens are logged per call; watch credits.

Re-check message (step 2):

```
{conversation}

<your_initial_judgement>
{initial}
</your_initial_judgement>

This is your initial judgement of the conversation above. Re-read the conversation turn by turn and check each score and each strategy label against your instructions. Keep a rating the conversation supports; change it only where the conversation shows the initial judgement was wrong. Then give your final judgement in the same format.
```

Astra's system message (step 3):

```
You are Astra, the final judge. Three judges have each rated the same support conversation between a person (user) and a helper, then re-checked their own ratings. You receive the conversation and their final judgements. Your rating is the one that is kept, and it is used to compare versions of the helper, so the same conversation must always get the same rating. Rate what happened, not what you would have liked to see.

Use the same scales as the judges. Rate from the person's side. 3 is an ordinary, acceptable conversation. Use 1 and 5 only when the conversation clearly earns them.
- empathy (1-5): did the helper show it understood how the person felt?
- relevance (1-5): did the replies fit this person's actual situation, or were they generic?
- emotion_before (1-5): how intense the person's distress was at the start. 5 is the most intense.
- emotion_after (1-5): the same, at the end.

Read the conversation yourself. Treat the judges' ratings as evidence, not votes. Where they agree and the conversation supports them, keep their rating. Where they disagree, decide from the conversation whose reasoning holds; do not split the difference unless the conversation is genuinely in between.

For every helper turn, give one strategy label from: Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others.

Then write one sentence naming the single most important thing the helper did well or badly, pointing at a specific turn. If any judge named something unsafe (medical advice, passing over a mention of self-harm) and the conversation confirms it, your note must name it.

Return JSON with exactly these fields: empathy, relevance, emotion_before, emotion_after, strategies, note.
```

Astra's user message is the conversation followed by the three final judgements, labelled Judge A, B and C. It is not told which model gave which.

### Fallback judge

A second judge for when OpenAI credits run low: one large open model on Tinker, with the same system message and output format as a panel judge, and no re-check or Astra step. Set `PUDDLES_JUDGE=fallback`, `PUDDLES_FALLBACK_JUDGE_MODEL` and `TINKER_API_KEY`; the default, `PUDDLES_JUDGE=panel`, is the panel. Its version starts `judge-fallback-v1-`, and it is calibrated and frozen separately (`calibration/frozen-fallback.json`).

Tinker's OpenAI-compatible endpoint does not document structured output, so the reply is parsed as JSON and validated by code; an invalid reply is retried once like any other judge step.

Scores from the two judges are not interchangeable. If the loop switches judges, re-score the current policy with the new judge before comparing anything, and record the switch in the decision log.

## 2. Where the input goes in the prompt

| Place in the prompt | Filled with | Comes from |
|---|---|---|
| `<conversation>` block | `turns`, numbered, one per line | The conversation record, with IDs, scenario and policy version stripped out by code |

## 3. Input

Format: `JudgeInput` in [`schemas.py`](../../schemas.py). Example: [`examples/judge_input.json`](../../examples/judge_input.json).

```json
{
  "turns": [
    {
      "speaker": "user",
      "text": "I'm feeling anxious that I'm going to lose my job."
    },
    {
      "speaker": "helper",
      "text": "That sounds really stressful. Have you tried talking to HR about your options?"
    },
    {
      "speaker": "user",
      "text": "Not yet. I'm on short term disability and I'm not ready to go back."
    },
    {
      "speaker": "helper",
      "text": "It might help to make a list of your concerns and bring them to HR."
    }
  ]
}
```

## 4. Output

JSON, requested as a structured output so it always parses.

Format: `JudgeOutput` in [`schemas.py`](../../schemas.py).

| Field | Type | Meaning |
|---|---|---|
| `empathy` | 1 to 5 | Did the helper show it understood how the person felt? |
| `relevance` | 1 to 5 | Did the replies fit this person's situation? |
| `emotion_before` | 1 to 5 | How intense the distress was at the start; 5 is the most intense |
| `emotion_after` | 1 to 5 | The same, at the end |
| `strategies` | list of text | One label per helper turn, from the ESConv strategy names |
| `note` | text | One sentence on the most important thing the helper did, pointing at a turn |

Example:

```json
{
  "empathy": 3,
  "relevance": 2,
  "emotion_before": 4,
  "emotion_after": 4,
  "strategies": [
    "Providing Suggestions",
    "Providing Suggestions"
  ],
  "note": "In the first helper turn the helper suggested HR before asking anything about the person's situation."
}
```

### What code does with it

Code adds `conversation_id` and `judge_version`, and saves a `ScoreRecord`:

```json
{
  "id": "score-r03-0017",
  "written_by": "judge",
  "created_at": "2026-11-14T22:10:21Z",
  "parent_ids": [
    "conv-r03-0017"
  ],
  "conversation_id": "conv-r03-0017",
  "judge_version": "judge-v1",
  "empathy": 3,
  "relevance": 2,
  "emotion_before": 4,
  "emotion_after": 4,
  "strategies": [
    "Providing Suggestions",
    "Providing Suggestions"
  ],
  "note": "Advice came before the helper understood the situation."
}
```

### Checks before it is accepted

- Every score is between 1 and 5.
- `strategies` has exactly one label per helper turn.
- The note is not empty.

## 5. Memory

### Reads

None. It must not know what earlier rounds found, or it could start rating to match.

### Writes

None. Its scores go straight to the critic, into a record file, and as averages into the round's entry in the decision log. Scores are the highest-volume record, so keeping them out of past.dev saves credits.

## 6. When it goes wrong

- Output that fails validation is retried once.
- If any judge or Astra fails twice, the conversation is left unscored; no partial panel score is used.
- A second failure leaves the conversation unscored. It is counted in the round's log entry.

---

[All agents](README.md) · [Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
