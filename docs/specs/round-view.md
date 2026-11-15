# Build spec: the round view

Covers item 7 of [docs/plan/prompts/stages-2-3-4.md](../plan/prompts/stages-2-3-4.md): what we look at after each round to see what every part did and what went where. No web app. It is three things: a summary on each node's span in MLflow, a report per round, and numbers on the round's MLflow run.

Training rounds only. Nothing here reads or reports a live session.

This file is the contract.

## Ground rules

- Nothing under `puddles/eval/` or `safety/` is created or edited, and no `tests/test_eval_*.py`.
- `schemas.py` is not changed. No existing test file is changed.
- The view only reads what a round left behind. It never changes the decision log, a record, a policy file, a conversations file or a scores file, and it never calls a model, Exa or past.dev.
- No network and no API keys in any test.
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 /Users/moamen/Documents/puddles-runall-venv/bin/python -m pytest -q
```

## Layout

```
puddles/round_view.py                    new
puddles/graph.py                         changed: each node wrapped once; four more metrics
scripts/show_round.py                    new
tests/test_round_view.py                 new: the implementer's tests
tests/acceptance/test_acc_round_view.py  new: written from this file alone, by someone else
```

## 1. A summary on each node's span

```python
SUMMARY_KEY = "puddles.summary"
def node_summary(node: str, state: Mapping[str, Any], update: Mapping[str, Any], root: Path) -> dict
def with_summary(node: str, fn: Callable[[RoundState], dict], root: Path) -> Callable[[RoundState], dict]
```

- `build_round_graph` registers every node as `with_summary(name, fn, root)`. The node names and what each node does are unchanged.
- `with_summary` calls `fn`, and on its way out sets the attribute `puddles.summary` on the MLflow span named after the node to `node_summary(...)`, where `state` is what the node received and `update` what it returned. It returns `fn`'s result unchanged. If there is no active span, or MLflow raises, the node's result is returned all the same: the view never breaks a round. An exception from `fn` passes through.
- If the span that LangGraph tracing opens for a node cannot be reached from inside the node, put the summary on a child span named `summary:<node>` instead, with the same attribute. (The builder says in its own report to the manager whether that happened; the round's report does not mention it.) Do not change how tracing is set up.

`node_summary` returns a small dict that can be written as JSON. It holds no conversation text.

**"No conversation text" means no turn of any conversation is copied**, in a summary or in the report: nothing a helper or a simulated user said. A round's reason, a pattern's name, a rule's text and a source's URL are shown as the decision log and the records hold them; they are the loop's own records of a training round, and the view does not rewrite them.

| Node | Keys |
|---|---|
| `converse` | `finished`, `failed` (counts), `file` (the train conversations file, relative to the root) |
| `judge` | `scored`, `unscored` (counts), `mean_empathy`, `mean_relevance` (over every conversation that was scored this round, not only the ones passed on to the critic; `null` when nothing was), `judge_version` (from the scores file's first line; `null` when there is no scores file) |
| `critic` | `pattern` (its name) and `seen_in`, `out_of`; or `pattern: null` and `reason` |
| `researcher` | `queries` (count), `sources` (their URLs) |
| `proposer` | `action`, `rule_id`, `rule_text`; or `edit: null` and `reason` |
| `gate` | `decision`, `quality_before`, `quality_after`, `noise`, `safety_passed`, `failed_checks` (names) |
| `finish` | `outcome`, `policy_in_force`, `copies` (a list of `{record_id, status}` from the memory-copy events of this round) |

- A value that was not measured is `null`, never 0 and never `true`. For the gate, `quality_before`, `quality_after` and `noise` come from `result.scores` and are `null` when it is `None`; `failed_checks` comes from `<root>/scores/round-NN-gate.json` and is `null` when that file does not exist (a stand-in gate). **The noise is a measurement only when it was measured.** `noise` is `null` in the summary, is left out of the run's metrics, and is shown in the report as `not measured`, with the figure that was passed in beside it in words (`not measured; the figure given was 0.00`), whenever `<root>/scores/round-NN-gate.json` does not mark `noise_not_measured` as `passed`. `result.scores.noise` holds whatever figure the gate was given, measured or not, so it is not enough on its own. `safety_passed` is `result.safety.passed` and is `null` when `result.safety` is `None`: the gate outcome's own `safety_passed` is a plain true or false even for a stand-in, so it is never used for the view.

## 2. Numbers on the round's MLflow run

`run_round` sets the tag `root` on the round's MLflow run to the root folder's resolved absolute path. It also logs, when a gate outcome exists and `result.scores` is set: `quality_before`, `quality_after` and, when it was measured (above), `noise`; and whenever a gate outcome exists: `kept` (1 or 0). They are left out, not logged as 0, when they were not measured. `conversations_failed`, `conversations_unscored`, tokens and cost are already logged.

## 3. `scripts/show_round.py`

```
python -m scripts.show_round --root runs/dev [--round N] [--mlflow <tracking address>]
```

`main(argv: Sequence[str] | None = None) -> int`. `--round` defaults to the last round in the log.

```python
def build_report(root: Path, round_: int) -> str         # in puddles/round_view.py
def report_path(root: Path, round_: int) -> Path          # <root>/reports/round-NN.md
```

`build_report` returns the text and writes nothing. The script writes it to `report_path(...)`, prints it, and attaches the file to the MLflow run named `round-NN` in the experiment `puddles` as an artifact under `reports/`. The tracking address is `--mlflow`, else `MLFLOW_TRACKING_URI`, else the default in `puddles/tracing.py`. If there is no such run, or attaching fails for any reason, it prints one line saying the report was not attached, and still exits 0. Several roots can share one tracking store, and each has a `round-01`. The round's run is the most recent run named `round-NN` whose tag `root` is this root's resolved absolute path; a run with another root's tag, or with no such tag, is never used, and with none that matches the report is not attached. Exit code 2, with one line on standard error, when the root has no log, the log cannot be read, or the round is not in it.

The report is markdown with these sections, in this order:

1. `# Round N: <outcome>`, then the reason.
2. `## Nodes`: one table row per node that ran: what it received, what it produced, where that went (a file path relative to the root, the decision log, memory). A node that did not run has a row that says so. Which nodes ran is worked out from the files: `converse`, `judge` and `finish` always; `critic` unless the event's reason starts with one of the two sentences graph.md gives for a round with nothing scored (judging.md may append a sentence about unscored conversations after it); `researcher` when the event has a pattern; `proposer` when the round has a research record and its outcome is not `no_evidence`; `gate` when the round has a gate outcome record.
3. `## Conversations`: a table of ID, empathy, relevance, and whether the critic was given it. A line of the scores file whose conversation is not in the round's conversations file (left by an attempt at the same round number that was killed) is not joined to anything and never stops the report: it is counted in one line under the table, `N score(s) in the scores file belong to no conversation of this round`, and left out of the table and of the judge's means. Unscored conversations are listed with empty scores and the word `unscored`. Failed conversations are listed from the failures file with their reason. No conversation text.
4. `## Memory`: each record of the round, who wrote it and who read it later, from `puddles.round.memory_trace`; and each memory copy event of the round with its status, one line per event.
5. `## Policy change`: the edit as a diff of rules (`+ C9. ...`, `- C4. ...`, or both lines for a rewording), and the version of the policy in force after the round. `No edit.` when there was none.
6. `## Gate`: every check in `scores/round-NN-gate.json` with `passed`, `failed` or `not run`; quality before and after and the noise; the safety cases that failed. For a stand-in gate, one line that says it was a stand-in and nothing was measured. `The gate did not run.` when it did not.
7. `## Cost`: per agent, calls, tokens in and out, and dollars to four decimal places, from the finished calls in `<root>/log/usage.jsonl` for this round.

Which conversations the critic was given is `puddles.judging.select_for_critic` applied to the round's scores file (`<root>/scores/round-NN-train.jsonl`) with the default number; with no scores file (the stub judge) the column is empty. A round run with another `critic_conversations` is not told apart; that is a stated limit.

`build_report` works from the files alone, so it can be run on a root long after the round, and on a round that ended at any of its exits.

## 4. Tests

`tests/test_round_view.py`, offline, on rounds run with the fakes:

1. After a round that reached the gate, every node's span (or its `summary:` child) has `puddles.summary` with the keys above, and none holds a line any client spoke.
2. A round that ends as `no_pattern` and one that ends as `no_evidence` both give a report, with the later nodes marked as not run.
3. The report of a kept round shows the rule diff; of a reverted round with the real gate, the failed checks by name.
4. With a stand-in gate, the gate's numbers are absent from the MLflow run and `null` in the summary, never 0.
5. `show_round` attaches the report to the round's run; with no such run it still writes and prints it and exits 0.
6. `build_report` leaves every file under the root byte-for-byte as it was and writes none; `show_round` adds only the report file.

The acceptance tests are written from this file alone, by someone who has not seen the code.
