# Researcher

**Training, step 5**

- Searches the web through Exa for how trained counsellors handle the pattern.
- Returns the sources it kept, each with a one-line technique.

|  |  |
|---|---|
| Receives from | [Critic](t4-critic.md): one failure pattern |
| Hands to | [Proposer](t6-proposer.md): sources and techniques, or "none found" |
| Model | The same OpenAI model, with one tool: an Exa search. This is the Exa call in the main loop. |
| Reads from memory | Techniques found in earlier rounds |
| Writes to memory | The research note |
| Task | [#16](https://github.com/puddles-convo/puddles/issues/16) |

## 1. The prompt

### System message

The fixed instructions. They are the same on every call.

```
You help improve a support helper by finding how trained counsellors handle a specific problem. You are given one failure pattern and the searches already run in earlier rounds. Find practical guidance on what a skilled helper does instead.

- Write one to three search queries in terms a counselling trainer would use, such as "reflective listening" or "exploring before advising". These work better than describing the fault. Do not repeat an earlier query.
- Run them with the search tool. You will get back titles, URLs and text.
- Keep sources that give concrete technique: counselling skills guides and training material from universities, health services or professional bodies. Skip marketing pages and forums.
- For each source you keep, write one sentence stating the technique in a form that could become a rule for the helper.
- Use only pages the search returned, and copy their URLs exactly. If nothing useful came back, return an empty list of sources. Do not fill the gap from your own knowledge.

Return JSON with these fields: queries, sources (each with url, title, technique).
```

### User message

```
<failure_pattern id="{pattern.id}">
{pattern.name}: {pattern.description}
</failure_pattern>

<earlier_queries>
- {earlier_queries[0]}
...
{older_queries_summary}
</earlier_queries>
```

## 2. Where the input goes in the prompt

| Place in the prompt | Filled with | Comes from |
|---|---|---|
| `<failure_pattern>` | `pattern` | The critic's output for this round |
| `<earlier_queries>` | `earlier_queries`, one line each, then `older_queries_summary` | Built by code from the decision log: every query run for this pattern, then the 10 most recent for other patterns (`HISTORY_RECENT`). Any others are counted in one last line, such as `31 older queries are not shown`. |
| Tool results | Titles, URLs and page text | Exa, returned to the model as tool messages after each search it runs |

## 3. Input

Format: `ResearcherInput` in [`schemas.py`](../../schemas.py). Example: [`examples/researcher_input.json`](../../examples/researcher_input.json).

```json
{
  "pattern": {
    "id": "pattern-r03",
    "name": "advice-before-understanding",
    "description": "Gives a suggestion in its first reply without asking about the person's circumstances."
  },
  "earlier_queries": [
    "how long should a counsellor's responses be",
    "offering choices in counselling"
  ]
}
```

What one search returns to the model (placeholders; the real values come from Exa):

```json
{
  "query": "counselling skills exploring before giving advice",
  "results": [
    {
      "url": "(url 1 from Exa)",
      "title": "(title 1)",
      "text": "(page text)"
    },
    {
      "url": "(url 2 from Exa)",
      "title": "(title 2)",
      "text": "(page text)"
    }
  ]
}
```

## 4. Output

JSON, requested as a structured output.

Format: `ResearcherOutput` in [`schemas.py`](../../schemas.py).

| Field | Type | Meaning |
|---|---|---|
| `queries` | list of text | The searches it ran, one to three |
| `sources` | list | The pages it kept |
| `sources[].url` | text | Copied exactly from a search result |
| `sources[].title` | text | The page title |
| `sources[].technique` | text | One sentence that could become a rule for the helper |

Example:

```json
{
  "queries": [
    "counselling skills exploring before giving advice"
  ],
  "sources": [
    {
      "url": "(url 1 from Exa)",
      "title": "(title 1)",
      "technique": "Ask one open question and reflect the answer before any suggestion."
    }
  ]
}
```

### What code does with it

Code adds the round and `pattern_id`, and saves a `ResearchNote`:

```json
{
  "round": 3,
  "pattern_id": "pattern-r03",
  "queries": [
    "counselling skills exploring before giving advice"
  ],
  "sources": [
    {
      "url": "(url 1 from Exa)",
      "title": "(title 1)",
      "technique": "Ask one open question and reflect the answer before any suggestion."
    }
  ]
}
```

With no sources left, the round ends as `no_evidence`: no edit is made, and the pattern keeps its status.

### Checks before it is accepted

- Every URL is one that Exa returned in this round. Any other is dropped.
- No query repeats one already run. The search tool checks each query before it goes to Exa, against every query in the decision log, not only the ones shown, ignoring case and spacing. A repeat is not run: the model is told so and can write another.
- At most three searches.

## 5. Memory

### Reads

Techniques found in earlier rounds, so it does not rediscover them:

```json
{
  "query": "Research already done for: advice-before-understanding. Gives a suggestion in its first reply without asking about the person's circumstances.",
  "identity": "researcher",
  "level": "low"
}
```

The earlier queries shown come from the decision log, not from this recall.

### Writes

One entry per research note.

Template for `content`:

```
Research {id} for {pattern_id}, {pattern name}: {technique} Source: {url}.
```

```json
{
  "id": "research-r03",
  "content": "Research research-r03 for pattern-r03, advice-before-understanding: ask one open question and reflect the answer before any suggestion. Source: (the page Exa returned).",
  "label": "researcher: research note, round 3",
  "timestamp": "2026-11-14T22:12:40Z",
  "identity": "researcher",
  "audience": "training-agents",
  "metadata": {
    "type": "research_note",
    "round": 3,
    "record_path": "records/round-03/research-r03.json",
    "sha256": "(hash of the record file)"
  }
}
```

## 6. When it goes wrong

- If Exa fails, the search is retried once.
- If no usable source remains, the round ends as `no_evidence`. The researcher must not fill the gap from its own knowledge.

---

[All agents](README.md) · [Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
