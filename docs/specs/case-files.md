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
- `expect.must_mention`: each word or phrase must appear, ignoring case, somewhere in the answer's JSON text. `expect.must_not_mention`: none may appear. `expect.fields`: each named top-level field of the answer must equal the value given. 
- Added after the first check, because the three above could not tell an empty or a lengthened answer from a good one:
  - `expect.non_empty`: a list of top-level fields; each must be there and not be null, an empty string, a string of spaces or an empty list.
  - `expect.one_of`: `{field: [values]}`; the field's value must be one of them.
  - `expect.max_chars`: `{field: n}`; the field must be text of at most `n` characters. A field that is null or missing does not meet it, unless the case's `one_of` allows a `remove` and the answer's `action` is `remove`.
- All six are optional; a case has at least one of them. A line is printed for each, `ok` or `EMPTY`, `NOT ONE OF`, `TOO LONG`.
- **Every case must be able to fail:** an empty answer, an answer about something else, and for the proposer an edit of the wrong kind, must each miss at least one expectation. Every researcher case requires `queries` and `sources` not to be empty. Every critic case with `found` true requires a non-empty `name` and `description`; with `found` false, a non-empty `reason`. The length-limit case allows only `remove` or `reword`, and a rewording must be shorter than the rule it rewords (`max_chars`). An expectation must not be so tight that a good answer misses it: a `must_not_mention` names a harmful thing to say, not a phrase a good answer might quote.
- **A case whose expectations cannot fail does not validate** (exit code 2): `expect` must hold at least one check that can reject an answer. An empty list or object under any key, and an empty or all-space string in `must_mention`, `must_not_mention` or `non_empty`, are refused by the loader.
- **`non_empty` goes all the way down.** A field is empty when it is null, an empty or all-space string, an empty list or object, or a list or object in which any text value at any depth is empty or all spaces. So a researcher source with a blank `url` or `technique` makes `sources` empty.
- **A field the answer did not give is not a blank.** The blank test is made on the answer as the model gave it. A field that the agent's output model makes optional, that the answer leaves out, and that the model fills with its default (version 4 of the record formats added `Source.quote`, default `""`) is not there to be empty and does not make its parent empty. A field the answer does give, as null, empty or all spaces, is empty as before, optional or not; and a field named directly in `non_empty` must still be given. So the five good researcher answers pass again, and blanking any field an answer holds still fails.
- **A proposer answer that rewords or removes a rule must name it.** Every proposer case lists `rule_id` under `non_empty`; the script does not apply it to an answer whose `action` is `add` when the proposer's own output model allows an addition without a rule ID (read `puddles/agents/proposer.py`; if an addition must carry one too, it is applied to every action).
- A proposer case that allows `reword` or `add` requires the replacement text: for those actions `rule_text` must not be null, empty or spaces. Where the case also allows `remove`, the script applies this to an answer whose `action` is not `remove`, the same way `max_chars` does. Write it as `expect.non_empty` on `rule_text`; for a case that allows `remove`, `non_empty` on `rule_text` is not applied to an answer whose `action` is `remove`.
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
- Exit code 0 when every case met its expectations or with `--dry-run`; 1 when a case did not; 2 for a case file that cannot be read (a missing file, or bytes that are not UTF-8 text) or does not validate, an unknown agent, or missing configuration. No traceback reaches the person for any of these.

## 3. Tests

`tests/test_try_agent.py`, all offline:

1. Every case in the three files validates as its input model, has a unique ID and at least one expectation, and has `reviewed_by` null.
2. `--dry-run` on each file exits 0 and prints every case ID.
3. With a scripted `llm` whose answers meet the expectations, the exit code is 0; with one answer that misses a `must_mention`, it is 1 and the line says `MISSING`.
4. A case file that does not validate gives exit code 2.
5. Without `--dry-run`, without an `llm` argument and without `OPENAI_API_KEY`, the exit code is 2 and no client is built.

## 4. What stays open for a person

`fixtures/cases/README.md` lists it: reading and signing each case (`reviewed_by`), the tuning runs against a real model, the one live Exa search, and for lane D the five real conversations to read and the four things to check in them, with the exact command from the stage file.
