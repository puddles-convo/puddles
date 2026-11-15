# Build spec: case files for the critic, researcher and proposer

Covers lanes C and D of stage 2 ([docs/plan/stage-2-real-agents.md](../plan/stage-2-real-agents.md)), as far as code can go. Tuning a prompt and declaring a case checked are a person's work; this piece prepares them.

Not the judge: it is Ansh's.

## Ground rules

- Nothing under `puddles/eval/` or `safety/` is created or edited, and no `tests/test_eval_*.py`.
- `schemas.py`, the agents' modules under `puddles/agents/` and the prompts in `docs/agents/` are not changed.
- **The cases are written by hand for this file.** None is taken from ESConv, CounselBench, VERA-MH, `safety/` or any held-out set. They are drafts: every case carries `"reviewed_by": null` until a person has read it.
- No real model call, no Exa call and no past.dev call is made while building or testing this piece.
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 /Users/moamen/Documents/puddles-runall-venv/bin/python -m pytest -q
```

## Layout

```
fixtures/cases/critic.json        new: 6 cases
fixtures/cases/researcher.json    new: 5 cases
fixtures/cases/proposer.json      new: 5 cases
fixtures/cases/README.md          new: what a case is, how to check one, and the lane D reading list
scripts/try_agent.py              new
tests/test_try_agent.py           new
```

## 1. A case file

A JSON list. Each case:

```json
{
  "case_id": "critic-01",
  "what_it_tests": "Advice in the first reply, before any question.",
  "input": { },
  "expect": {
    "must_mention": ["advice", "question"],
    "must_not_mention": ["diagnos"],
    "fields": {"found": true}
  },
  "reviewed_by": null
}
```

- `input` validates as the agent's input model: `CriticInput`, `ResearcherInput` or `ProposerInput`.
- `expect.must_mention`: each word or phrase must appear, ignoring case, somewhere in the answer's JSON text. `expect.must_not_mention`: none may appear. `expect.fields`: each named top-level field of the answer must equal the value given. All three are optional; a case has at least one of them.
- The critic's cases start from the four conversations in `fixtures/bad_conversations.json` and add: one where nothing is clearly wrong (`found` false), and one that repeats an earlier pattern listed in `earlier_patterns`.
- The researcher's cases each give one pattern, some with earlier queries that must not be repeated.
- The proposer's cases include one at the length limit (`at_length_limit` true: the edit must be a removal or a rewording), and one where an earlier edit for the same pattern was reverted.

## 2. `scripts/try_agent.py`

```
python -m scripts.try_agent critic fixtures/cases/critic.json --dry-run
```

`main(argv: Sequence[str] | None = None, llm: LLM | None = None, search: Callable[[str], list[dict]] | None = None) -> int`.

| Argument | Meaning |
|---|---|
| `agent` | `critic`, `researcher` or `proposer` |
| `case_file` | A case file |
| `--case` | Run only the case with this ID |
| `--dry-run` | Validate every case and print the system and user messages each would send. No model is built and nothing is called. |

- For each case it renders the user message with the agent's own `render_user` (the proposer's takes the base policy version, read from the case's `policy_text`), sends it with the agent's own system prompt, and prints: the case ID, what it tests, the answer as JSON, then one line per expectation, `ok` or `MISSING`/`PRESENT`/`DIFFERENT`, and a last line `N of M cases met their expectations`.
- It does not write to the decision log, to record files or to memory, and it does not run the agent's own retry: it shows the first answer as it came.
- The model is the `llm` argument when given; otherwise `puddles.llm.OpenAILLM()`. The researcher's answer is asked for with `structured_with_search`, using the `search` argument when given and otherwise `puddles.exa.search`.
- Without `--dry-run` and without an `llm` argument it needs `OPENAI_API_KEY`, and for the researcher `EXA_API_KEY`. A missing key is one line on standard error and exit code 2, before any call.
- Exit code 0 when every case met its expectations or with `--dry-run`; 1 when a case did not; 2 for a case file that cannot be read or does not validate, an unknown agent, or missing configuration.

## 3. Tests

`tests/test_try_agent.py`, all offline:

1. Every case in the three files validates as its input model, has a unique ID and at least one expectation, and has `reviewed_by` null.
2. `--dry-run` on each file exits 0 and prints every case ID.
3. With a scripted `llm` whose answers meet the expectations, the exit code is 0; with one answer that misses a `must_mention`, it is 1 and the line says `MISSING`.
4. A case file that does not validate gives exit code 2.
5. Without `--dry-run`, without an `llm` argument and without `OPENAI_API_KEY`, the exit code is 2 and no client is built.

## 4. What stays open for a person

`fixtures/cases/README.md` lists it: reading and signing each case (`reviewed_by`), the tuning runs against a real model, the one live Exa search, and for lane D the five real conversations to read and the four things to check in them, with the exact command from the stage file.
