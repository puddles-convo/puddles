# Case files for the critic, researcher and proposer

Hand-written cases for tuning the three training agents' prompts (stage 2, lane C). The format and the script are specified in [docs/specs/case-files.md](../../docs/specs/case-files.md).

**These are drafts.** Every case was written by hand for this folder; none is taken from ESConv, CounselBench, VERA-MH, `safety/` or any held-out set. Each one carries `"reviewed_by": null` until a person has read it and signed it. No case is about suicide or self-harm: that is the safety cases' job, not these.

| File | Cases | What they cover |
|---|---|---|
| `critic.json` | 6 | The four conversations in `fixtures/bad_conversations.json` (advice first, a long list of options, "you should"), a different pattern with no history, a counting case, one where nothing is clearly wrong (`found` false), and one that repeats an earlier pattern listed in `earlier_patterns` while its rule is marked fixed. |
| `researcher.json` | 5 | One pattern each; four of them give earlier queries that must not be run again. |
| `proposer.json` | 5 | An open pattern; one at the length limit (`at_length_limit` true, with a policy long enough that `puddles.policy.at_length_limit` agrees); one where an earlier edit for the same pattern was reverted; a recurring pattern with an active rule; and one next to safety rule S2. |

## What a case is

```json
{
  "case_id": "critic-01",
  "what_it_tests": "One sentence.",
  "input": {"...": "a CriticInput, ResearcherInput or ProposerInput"},
  "expect": {
    "must_mention": ["each must appear, ignoring case, in the answer's JSON"],
    "must_not_mention": ["none may appear"],
    "fields": {"found": true}
  },
  "reviewed_by": null
}
```

- The answer's JSON is the compact form (`model_dump_json()`), so a phrase such as `"action":"add"` has no spaces.
- `fields` compares a top-level field of the answer exactly; `null` matches a field that is empty.
- The source URLs in the proposer's cases are `example.org` placeholders, not real pages. The proposer only reads them; nothing fetches them.
- The proposer's base policy version is read from the first line of the case's `policy_text` (`# policy-v3`).

## Running them

From the repo root, with `PYTHONPATH` unset:

```
python -m scripts.try_agent critic fixtures/cases/critic.json --dry-run
python -m scripts.try_agent proposer fixtures/cases/proposer.json --case proposer-02
```

`--dry-run` validates every case and prints the system and user messages each would send; it builds no model and calls nothing. Without it the script needs `OPENAI_API_KEY` and `PUDDLES_AGENT_MODEL`, and for the researcher `EXA_API_KEY`. It sends each case once, with no retry, prints the answer and one line per expectation (`ok`, `MISSING`, `PRESENT` or `DIFFERENT`), and ends with `N of M cases met their expectations`. It writes nothing: no decision log, no record file, no memory. Exit codes: 0 when every case met its expectations (or with `--dry-run`), 1 when one did not, 2 for a bad case file, an unknown agent or missing configuration.

## How to check a case

1. Read the input as the agent would: run with `--dry-run --case <id>` and read the user message.
2. Ask whether a good answer must contain what `expect` says, and whether a good answer could miss it anyway. An expectation that a good answer can fail is a bad expectation: change or drop it.
3. Check the conversations are plain, respectful and believable, and stay out of crisis territory.
4. Put your name in `reviewed_by`. Only signed cases count as hand-checked for the stage file.

## What stays open for a person

- **Reading and signing each case** (`reviewed_by`). All 16 are unsigned.
- **The tuning runs against a real model.** Change the prompt in `docs/agents/t4-critic.md`, `t5-researcher.md` or `t6-proposer.md` (the code reads the system message from there), re-run the case file, keep what helps.
- **The one live Exa search** through the researcher, with the returned URLs checked against what Exa sent. `try_agent` prints the answer but not the raw search results; compare them by hand, or run the researcher through `scripts/run_round.py`, which saves the searches beside the note.
- **Lane D: the helper and the simulated user.** Run the five fixture scenarios with the real models:

  ```
  python -m loop.run_batch --policy policies/policy-v1.md --scenarios fixtures/scenarios.jsonl --round 0 --out runs/dev/conversations.jsonl
  ```

  Read all five conversations and check four things:
  1. the resistant scenario stays resistant;
  2. nobody feels better after one reply;
  3. helper replies are one or two spoken sentences;
  4. the simulated user ends with `[END]` at a natural point.

  Change `loop/prompts/*.txt` and the matching file in `docs/agents/` together, and freeze the helper prompt at the end of the lane.
