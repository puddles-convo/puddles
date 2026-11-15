# Build spec: the gate joined to the real judge and the real safety check

The last small job of stage 2. [gate.md](gate.md) built the gatekeeper against a judge and a safety check passed in as arguments. This joins it to Ansh's: the judge from `puddles.eval.judge.judge_from_env()` (#33, #38) and `puddles.eval.safety.run_safety` with his cases (#34).

This file is the contract.

## Ground rules

- Nothing under `puddles/eval/` or `safety/` is created or edited, and no `tests/test_eval_*.py`. His code is called through its public functions and is not copied.
- `puddles/gate.py` is not changed: it still imports nothing from `puddles.eval`. The join lives in a module of its own.
- `schemas.py` is not changed. No existing test file is changed, with two exceptions that the removal of the old refusal makes necessary, each corrected by its own author and only as far as that: `test_the_real_gate_without_fake_stops_before_anything_is_written` in `tests/test_gate.py` (it expected the words "later piece"; it now expects exit 2, one line, nothing written, for the configuration that is missing) and `test_without_fake_constructs_documented_dependencies` in `tests/acceptance/test_acc_graph_script.py` (without `--fake` the gate is now `real`, so the test supplies the safety check's configuration or stands in for its factory).
- No network and no API keys in any test.
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 /Users/moamen/Documents/puddles-runall-venv/bin/python -m pytest -q
```

## Layout

```
puddles/gate_join.py                   new
scripts/run_graph_round.py             changed: the real gate without --fake
tests/test_gate_join.py                new: the implementer's tests
```

## 1. `puddles/gate_join.py`

```python
def safety_check(asker: puddles.eval.safety.YesNoLLM, cases: Optional[list[SafetyCase]] = None,
                 case_file: str = "safety-v1.json") -> puddles.gate.SafetyCheck
def real_gate_deps(*, helper_client: ChatClient, user_client: ChatClient, judge: Any, root: Path,
                   scenarios: list[Scenario], noise_measured: bool = False, max_turns: int = 10,
                   seed: Optional[int] = None, asker: Optional[YesNoLLM] = None) -> puddles.gate.GateDeps
def safety_models(asker: Any) -> list[str]
```

- `safety_check(asker, cases, case_file)` returns a callable `(policy_version, reply) -> SafetyResult` that is one call to `puddles.eval.safety.run_safety(policy_version, reply, asker, cases=cases, case_file=case_file)`. With `cases` `None` his loader reads his file. Nothing about the cases, the question prompt or the yes-or-no rule is restated here.
- `real_gate_deps` builds the `GateDeps`: `judge` is wrapped in `puddles.judging.MeteredJudge` unless it already is one, so the gate's scoring is one call per conversation to the same judge the round uses and its tokens are counted; `safety` is `safety_check(asker or puddles.eval.safety.OpenAIYesNo.from_env())`. A `ValueError` from `from_env` passes through: it is missing configuration.
- `safety_models(asker)` is `[asker.model]`, for the price check.
- **An empty case list is not a pass.** The join does not check for it: the gate does ([gate.md](gate.md), "A safety check that ran no case did not run"). With his loader returning no cases, `run_safety` returns `passed` with no cases, and the gate fails `safety_cases` with `no safety case was run` and keeps nothing.
- **A safety check that raises** is not caught, retried or turned into a result by the join. The gate fails `safety_cases` for it (gate.md) and the candidate is reverted.
- His yes-or-no client logs its token counts and does not return them, so the tokens of the safety questions are not in the usage log. This module does not change his client to get them. ("The report" here and below means the builder's last message to the manager, not a file and not program output; no test is asked of it.)

## 2. `scripts/run_graph_round.py`

- `--gate` has no fixed default any more: with `--fake` it is `reverted`, as before; without `--fake` it is `real`. So `python -m scripts.run_graph_round --root runs/small --scenarios fixtures/scenarios.jsonl --max-turns 6` runs the real gatekeeper.
- Without `--fake` and with `--gate real`, the gate is `puddles.gate.make_gate(real_gate_deps(...), validation_scenario_ids=<every ID in the --validation file>, noise=args.noise, token_cap=args.token_cap)`, with the same judge object the round's `judge_panel` uses, the same helper and simulated user clients, and `noise_measured=args.noise_measured`. The earlier refusal ("joined in a later piece") is removed.
- Missing configuration for the safety model (`PUDDLES_SAFETY_MODEL` or `PUDDLES_JUDGE_ASTRA`) is one line on standard error and exit code 2, before anything is written. The safety model's name is added to the models whose prices are checked before the round starts, and a model with no price is one line on standard error and exit code 2, before memory is touched, before anything is written under the root and before any model is called; if the graph has no way to take extra model names for that check, call `deps.usage`'s price check from the script before `run_round`, and report the choice.
- `--fake` behaves as before for every `--gate` value.

## 3. Tests

`tests/test_gate_join.py`, offline:

1. The gate with his real `run_safety`, his real cases and a scripted asker that always gives the passing answer for each question: the safety check passes and the round event's `safety` names his case file and the number of his cases.
2. **A yes-or-no answer that is neither word, twice, fails that assertion:** with a scripted asker that answers `maybe` twice for one question, the candidate is reverted with `safety_cases` in the failed checks, and the case is in `failed_case_ids`. This is the last line of test 3 in the stage file.
3. The gate with his real `JudgePanel`, built from three fake `JudgeLLM`s and a fake Astra: a clean candidate with `noise_measured` on is `kept`, and with it off is `reverted` with a reason that says the checks would have kept it.
4. A judge whose Astra fails twice on one validation conversation (an answer that fails his own checks, at both tries, so that his `score` raises): `reverted`, `complete_evidence`.
5. The tokens his judge reports for the gate's conversations are in `<root>/log/usage.jsonl` under the agent `gatekeeper`.
6. The script without `--fake` and with the judge variables unset exits 2 with one line and writes nothing under the root.

The passing answers in test 1 are worked out from his case file at run time (a `must` question gets yes, a `must_not` question gets no); the helper's replies are scripted so that his `contains` assertions hold, by reading the values from his file. No case text is typed into the test.
