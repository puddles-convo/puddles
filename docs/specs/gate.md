# Build spec: the gatekeeper

Covers lane B of stage 2 ([docs/plan/stage-2-real-agents.md](../plan/stage-2-real-agents.md)), except the safety case file, which is Ansh's (`safety/safety-v1.json`, #34). The decision rule is in [docs/agents/t7-gatekeeper.md](../agents/t7-gatekeeper.md).

This file is the contract: it fixes the names, signatures and file formats. Where it and the stage file disagree, this file is right.

The gate is built against a judge and a safety check that are **passed in as arguments**. This piece never imports `puddles.eval`. A later piece joins the gate to the real judge and the real safety check.

## Ground rules

- Nothing under `puddles/eval/` or `safety/` is created or edited, and no `tests/test_eval_*.py`.
- `schemas.py` is not changed. No existing test file is changed.
- No network and no API keys in any test.
- Much of this exists on branch `stage-2-gate` (`git show origin/stage-2-gate:puddles/gate.py`, `tests/test_gate.py`, `loop/runner.py`). Carry it over, minus its own judge (`puddles/agents/judge.py`), loader (`puddles/esconv.py`), calibration script, safety file (`fixtures/safety-v1.json`), its yes-or-no asking code, and `puddles/meta.py` (main now has `puddles/replay.py`, which writes `.meta.json` for promoted policies).
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 /Users/moamen/Documents/puddles-runall-venv/bin/python -m pytest -q
```

## Layout

```
puddles/gate.py                          new
loop/runner.py                           changed: a tag in conversation IDs
puddles/round.py                         changed: RoundResult.scores and .safety; finish_round writes them
puddles/graph.py                         changed: RoundDeps.start_policy
scripts/run_graph_round.py               changed: --gate real and its arguments
puddles/replay.py                        changed: how it reads a policy file, nothing else (section 8, rule 11)
tests/test_gate.py                       new: the implementer's tests
tests/test_gate_boundaries.py            new: the implementer's tests for section 8, rules 1 to 6
tests/test_finish_guards.py              new: the implementer's tests for section 8, rules 7 to 11
tests/unit/test_runner.py                changed: tests for the tag, added at the end
tests/acceptance/test_acc_gate.py        new: written from this file alone, by someone else
```

## 1. `loop/runner.py`: a tag in conversation IDs

```python
def conversation_id(round: int, scenario_id: str, tag: str | None = None) -> str
def run_conversation(..., tag: str | None = None) -> ConversationRecord      # new last keyword
def run_batch(..., tag: str | None = None) -> BatchResult                    # new last keyword
```

- `conversation_id(3, "s1")` is `conv-r03-s1`. `conversation_id(3, "s1", "val")` is `conv-r03-val-s1`.
- A tag must match `[a-z][a-z0-9]*`. Anything else raises `ValueError`, from `conversation_id` and from `run_batch` before any conversation starts.
- With `tag=None` both functions behave exactly as before.

## 2. `puddles/gate.py`

```python
VALIDATION_TAG = "val"        # the candidate's validation conversations: conv-rNN-val-<scenario>
CURRENT_TAG = "current"       # the policy in force: conv-r00-current-<scenario>
BASELINE_ROUND = 0
NOT_MEASURED = Scores(empathy=0, relevance=0)     # scores are 1 to 5, so 0 can only mean "not measured"

CHECKS: dict[str, str]        # check name -> words for the reason. Keys, in deciding order:
                              # candidate_file, safety_section, one_rule, token_cap,
                              # safety_cases, complete_evidence, quality_gain, dimension_drop,
                              # noise_not_measured

class GateError(RuntimeError)

class Judge(Protocol):
    version: str
    def score(self, record: ConversationRecord) -> Any: ...   # the returned object has `.score`, a ScoreRecord

Reply = Callable[[list[Turn]], str]
SafetyCheck = Callable[[str, Reply], SafetyResult]             # (policy_version, reply) -> SafetyResult

@dataclass
class GateDeps:
    helper_client: ChatClient
    user_client: ChatClient
    judge: Judge
    safety: SafetyCheck
    root: Path
    scenarios: list[Scenario]            # where validation_scenario_ids are looked up
    noise_measured: bool = False         # stage 5 turns it on; until then nothing is promoted
    max_turns: int = 10
    seed: Optional[int] = None

@dataclass
class Evidence:
    expected: int
    scores: Optional[Scores] = None      # set only when every scenario came back with a score
    failed: list[str]                    # scenario IDs whose conversation failed (default empty)
    unscored: list[str]                  # conversation IDs the judge did not score (default empty)
    @property
    def complete(self) -> bool           # scores is set, and failed and unscored are empty

@dataclass
class GateReport:
    outcome: GateOutcome
    failed: list[str]                    # names from CHECKS, in CHECKS order; empty only when kept
    would_keep: bool                     # true when none of the eight checks before noise_not_measured failed
    checks: dict[str, str]               # every key of CHECKS -> "passed", "failed" or "not_run"
    log_scores: Optional[LogScores] = None    # None when validation gave no complete scores
    log_safety: Optional[LogSafety] = None    # None when the safety check did not run
    safety: Optional[SafetyResult] = None

def average(scores: list[ScoreRecord]) -> Scores
def score_policy(deps: GateDeps, policy_path: Path, scenario_ids: list[str], round_: int, tag: str) -> Evidence
def current_scores(deps: GateDeps, policy_path: Path, scenario_ids: list[str]) -> Scores
def run_gate(inp: GateInput, deps: GateDeps) -> GateReport
def make_gate(deps: GateDeps, *, validation_scenario_ids: list[str], noise: float,
              token_cap: int = puddles.policy.TOKEN_CAP,
              safety_case_set: str = "safety-v1") -> Callable[[RoundResult], GateOutcome]
def ensure_starting_meta(root: Path, start_policy: Path) -> Path
```

### Scores

- `average`: `empathy` and `relevance` are the means over the records; `quality` is the mean of `(empathy + relevance) / 2` per record; `emotion_change` is the mean of `emotion_before - emotion_after`. An empty list raises `GateError`.
- Emotion change is reported and never gated.

### `score_policy`

1. Loads the policy with `puddles.graph.check_policy_file(policy_path)`.
2. Picks the scenarios named in `scenario_ids` from `deps.scenarios`, in the order of `scenario_ids`. An ID that is not there, a repeated ID or an empty list raises `GateError`.
3. `loop.runner.run_batch(deps.helper_client, deps.user_client, policy, scenarios, round=round_, max_turns=deps.max_turns, seed=deps.seed, tag=tag, out_path=<root>/conversations/round-NN-<name>.jsonl)`, where `<name>` is `validation` for the tag `val` and the tag itself otherwise (`round-00-current.jsonl`).
4. For each record, one call `deps.judge.score(record)`. Any exception from it leaves that conversation unscored; the rest are still scored. A result whose `score.conversation_id` is not the record's ID is unscored too.
5. Saves the `ScoreRecord`s it got to `<root>/scores/round-NN-<name>.jsonl`, one per line, even when some are missing. When nothing was scored the file is written empty.
6. Returns `Evidence`. `scores` is set, to `average(...)`, only if no conversation failed and none is unscored. **The conversations that survived are never averaged.**

### `current_scores`

The scores of the policy in force on the validation scenarios. Kept in `<root>/scores/current.json`:

```json
{"policy_version": "policy-v1", "policy_sha256": "(hash of the policy file's text)",
 "scenario_ids": ["a", "b"], "judge_version": "judge-v1-ab12cd34",
 "scenarios_sha256": "(see below)", "max_turns": 10, "seed": null,
 "scores": {"empathy": 3.4, "relevance": 3.1, "quality": 3.25, "emotion_change": 0.6}}
```

- If that file exists and its `policy_version`, `policy_sha256`, `scenario_ids`, `judge_version`, `scenarios_sha256`, `max_turns` and `seed` equal the policy file's version, the hash of its text (`puddles.records.sha256_text`), the given IDs in the same order, `deps.judge.version`, the hash of the scenarios, `deps.max_turns` and `deps.seed`, its scores are returned and no model is called. A file that cannot be read, does not have every one of these fields, or whose `scores` is missing or does not validate as `Scores`, counts as absent.
- `scenarios_sha256` is `puddles.records.sha256_text` of the picked scenarios, in the order of `scenario_ids`, each as `model_dump_json()`, joined with `"\n"`. A scenario whose text changed under the same ID is therefore scored again.
- Otherwise it calls `score_policy(deps, policy_path, scenario_ids, BASELINE_ROUND, CURRENT_TAG)`. Complete evidence is written to `current.json` and returned. Incomplete evidence raises `GateError` and writes no `current.json`; a `current.json` that was already there and did not match is left as it is, since it can never match.
- **A limit, stated:** the file does not record which helper or simulated-user model produced the conversations. A root is for one pair of models. Change either and start a new root, or delete `current.json`.

### `run_gate`

`inp.current_policy` and `inp.candidate_policy` are paths to policy files. `inp.current_scores` is trusted as given.

**It never raises.** Its whole evaluation sits inside one `except Exception`; anything that goes wrong reverts the candidate with `complete_evidence` in `failed` and the exception's type and text in the reason. That includes a file the gate could not save: a decision whose evidence could not be written down is a revert, and the gate still returns.

In this order:

1. **Static checks, no model call.**
   - `candidate_file`: the candidate does not load with `check_policy_file`, or the current policy does not.
   - `safety_section` (check 1): `loop.policy.Policy.safety_sha256` of the candidate differs from the current policy's.
   - `one_rule` (check 2): the number of Conversation rules added, removed or reworded between the two files (by rule ID, text compared exactly) is not exactly 1.
   - `token_cap` (check 3): `puddles.policy.estimate_tokens(candidate text) > inp.token_cap`.

   If `candidate_file` fails, the other three are `not_run`. Otherwise all three are evaluated, and each is `passed` or `failed`. If any of the four fails, nothing else runs: no conversation, no judge call, no safety check, and the five checks after them are `not_run`, `noise_not_measured` included.
2. **The safety check** (check 4): one call `deps.safety(candidate.version, reply)`, where `reply(turns)` returns `loop.helper.helper_reply(deps.helper_client, candidate, turns, session_id=f"safety-r{round:02d}", turn=1).text` under the candidate policy, with no crisis checker in front of it. `safety_cases` fails when the result's `passed` is false, and when `deps.safety` raises; a safety check that raised leaves no `SafetyResult`, and validation still runs. **A safety check that ran no case did not run.** When the result's `cases` list is empty, `safety_cases` is `failed` with the reason `no safety case was run`, whatever the result's `passed` says; the candidate cannot be kept; the result is still saved as it came back; and the gate's own statements about it are false, not true: `GateOutcome.safety_passed` is `False` and `log_safety.passed` is `False`, with `cases` 0. (Found when the real safety check was joined: an empty case file made `run_safety` return `passed` with no cases. It holds for every safety check the gate is given, the script's `--fake` stand-in included, so a `--fake --gate real` round now always names `safety_cases` among its failed checks.) **A result that contradicts itself is not a pass either:** when any case in the result has `passed` false, `safety_cases` is `failed` and names those cases, whatever the result's own `passed` says, and `GateOutcome.safety_passed` and `log_safety.passed` are `False`. So `safety_cases` is `passed` only when the result says passed, at least one case ran, and no case in it failed. **Scores are 1 to 5, so a baseline outside that range was not measured.** When `inp.current_scores` has `empathy` or `relevance` below 1 or above 5 (`NOT_MEASURED` is the plain example), `complete_evidence` is `failed` with the reason `the policy in force has no measured scores`, `quality_gain` and `dimension_drop` are `not_run`, and nothing is kept; and a `scores/current.json` whose scores are outside that range counts as absent, like one that does not validate, so the policy in force is scored again. The `SafetyResult` is saved as JSON to `<root>/scores/round-NN-safety.json`. The gate does not load cases and asks no yes-or-no question itself; that is inside the safety check it is given.
3. **Validation**: `score_policy(deps, candidate_path, inp.validation_scenario_ids, inp.round, VALIDATION_TAG)`. Steps 2 and 3 both run even if one fails. `complete_evidence` fails when the evidence is not complete; then `quality_gain` and `dimension_drop` are `not_run`. **It also fails if any validation conversation ID is found in `<root>/conversations/round-NN-train.jsonl`** (a training scenario named `val-a` and a validation scenario named `a` give the same ID): evidence that cannot be told apart from the round's own is not evidence. The validation IDs are `loop.runner.conversation_id(inp.round, scenario_id, VALIDATION_TAG)` for each ID in `inp.validation_scenario_ids`; the training IDs are the `id` of each line of that file. With no such file there is nothing to clash with. On a clash the gate treats the validation as not measured: `quality_gain` and `dimension_drop` are `not_run`, `after` is `NOT_MEASURED` and `log_scores` is `None`, whatever the judge said. The validation files already written stay where they are; the gate deletes nothing.
4. With complete evidence, where `before = inp.current_scores` (its `quality` filled in as `(empathy + relevance) / 2` when it is `None`) and `after` is the candidate's scores. Every difference is rounded to 9 decimal places before it is compared:
   - `quality_gain` (check 5) fails unless `after.quality - before.quality > inp.noise`. A gain exactly equal to the noise fails.
   - `dimension_drop` (check 6) fails if `before.empathy - after.empathy > inp.noise`, or the same for relevance. A fall exactly equal to the noise passes.
5. `would_keep` is true when none of the eight checks above failed (the four static ones, `safety_cases`, `complete_evidence`, `quality_gain`, `dimension_drop`).
6. `noise_not_measured`: `failed` when `deps.noise_measured` is false, `passed` otherwise. It is the last name in `failed`.
7. The decision is `kept` only if `would_keep` and `deps.noise_measured`. Otherwise `reverted`.

The `GateOutcome`:

- `id` is `gate-rNN`, `written_by` is `gatekeeper`, `parent_ids` is `[inp.edit_id]`.
- `before` is `inp.current_scores` with quality filled in. `after` is the candidate's scores, or `NOT_MEASURED` when validation was not complete or did not run.
- `safety_passed` is the safety result's `passed`, and false when the safety check did not run or raised.
- `policy_in_force` is the candidate's version without `-candidate` when kept, and the current policy's version when reverted. If the current policy file does not load, it is the version `puddles.policy.parse_policy` reads from it, and the file's stem if that fails too.
- `reason` is sentences a person can read. It names every failed check with the words in `CHECKS`, the deciding one first. With `noise_measured` false it says the noise has not been measured and says what the checks found: that the candidate would have been kept, or which checks failed. A kept candidate's reason gives the quality gain and the noise.

`log_scores` is `LogScores(before, after, noise=inp.noise)` when the evidence was complete, else `None`. `log_safety` is built from the `SafetyResult`: `case_file`, `passed`, the number of cases and the IDs of the failed ones; `None` when the safety check did not run or raised.

The report is also saved to `<root>/scores/round-NN-gate.json`: `{"round", "decision", "would_keep", "failed", "checks", "reason"}`.

**The gate writes no policy file and changes none.** The candidate file the proposer wrote stays where it is. A reverted candidate leaves the policy in force byte-for-byte as it was.

**Every check and every measurement is about the same bytes.** The gate takes the hash of the candidate file's text (`puddles.records.sha256_text`) when it first reads it for the static checks, and of the current policy's likewise. Every later time the gate loads either file (for the safety replies, for validation, for anything else) it compares the hash of what it read with the first one, and it compares once more after the last judge call, before it decides. Where an existing loader reads the file itself, the gate hashes the file before and after that load and requires both to equal the first hash. A difference at any point is a revert: `complete_evidence` fails, `quality_gain` and `dimension_drop` are `not_run`, `after` is `NOT_MEASURED`, `log_scores` is `None`, and the reason says the candidate (or the current policy) changed while it was being tested. Validation is the steps of `score_policy` carried out on that one tied reading of the candidate; it need not call the public `score_policy`, which loads the file itself. A file that was replaced and put back while the conversations ran on the tied reading changes nothing: every check and measurement was still about the first bytes, and they are the bytes on disk at the end. The baseline is tied the same way: `make_gate` carries the hash of the policy text that `current_scores` scored (or matched in `current.json`) into the gate, and a policy in force that no longer holds that text is a revert before the candidate is tested. `GateReport` has one more field, `candidate_sha256: Optional[str] = None`: the hash the checks were made on, `None` when the candidate could not be read.

### `make_gate`

Returns a callable for `RoundDeps.gate`. For a `RoundResult` with an edit it:

1. gets the current scores with `current_scores(deps, result.base_policy_path, validation_scenario_ids)`. If that raises, the candidate is reverted untested: `failed` is `["complete_evidence"]` plus `noise_not_measured` when the flag is off, `before` and `after` are `NOT_MEASURED`, and the reason says the policy in force could not be scored;
2. builds the `GateInput` (`current_policy` and `candidate_policy` are the two paths as text, `safety_case_set` as given) and calls `run_gate`;
3. sets `result.scores = report.log_scores`, `result.safety = report.log_safety` and `result.candidate_sha256 = report.candidate_sha256`;
4. on `kept`, rewrites `current.json` with the candidate's scores under the promoted version. The candidate is read once more for this, and its hash must equal `report.candidate_sha256`; if it does not, the outcome returned is a revert as described under "the same bytes", and `current.json` is not rewritten. Its `policy_sha256` is the hash of the text `finish_round` will write for the promoted policy: `puddles.policy.render_policy` of the parsed candidate with its version set to the promoted version. If this write fails the outcome stands and nothing is raised: the file is then stale or missing, it cannot match the promoted policy, and the next round scores the policy in force again;
5. returns `report.outcome`.

`safety_case_set` is only recorded on the `GateInput`. The safety check that was passed in decides which cases it runs; the gate cannot choose them through the callback. The piece that joins the real safety check makes the two agree.

The callable never raises. It keeps the last `GateReport` on its attribute `last_report`.

### `ensure_starting_meta`

Writes `<root>/policies/<version>.meta.json` for the starting policy if it is not there, and returns its path. The content validates as `PolicyMeta`: `parent` is `None`, `policy_sha256` is the hash of the file's text, `safety_sha256` is what `puddles.replay.expected_meta` gives for the starting policy, every Conversation rule ID is in `base_rules`, and `learned_rules` is empty. An existing file is left as it is. `puddles/replay.py` is edited only as section 8, rule 11 says.

## 3. `puddles/round.py`

- `RoundResult` gets four fields, all `None` by default: `scores: Optional[LogScores]`, `safety: Optional[LogSafety]`, `candidate_sha256: Optional[str]` and `base_sha256: Optional[str]`. `None` means not measured, never that it passed. `make_gate` sets `base_sha256` to the hash of the policy text the before-scores were measured on (the text `current_scores` scored, or matched in `current.json`), on every call, kept or reverted, and to `None` when the policy in force could not be scored.
- **`finish_round` files measurements only under the bytes they were measured on.** When a gate outcome is given and `result.base_sha256` is set, whatever the decision, `finish_round` reads the base policy file once, before it writes anything, and if the hash of that text is not `result.base_sha256` it raises `puddles.round.PolicyChanged` (a `RuntimeError`): no policy file, no `.meta.json`, no record, no round event, no memory copy. The event's `base_sha256`, and everything else `finish_round` needs from the base policy, comes from that one reading. `CandidateChanged` is a subclass of `PolicyChanged`.
- **`finish_round` files nothing under bytes the gate did not test.** When a gate outcome is given and `result.candidate_sha256` is set, **whatever the decision** (kept, or reverted for any reason, a static rejection included), `finish_round` reads the candidate file once, and if the hash of that text is not `result.candidate_sha256` it raises `puddles.round.CandidateChanged` (a `RuntimeError`) before it writes anything: no policy file, no `.meta.json`, no record, no round event, no memory copy. A reverted round's event carries the candidate's hash beside its scores and its safety result, so the rule is the same as for a keep. The promoted policy, and every hash of the candidate that is written, come from that same reading. With `candidate_sha256` `None` (the stand-in gate, or a candidate the gate could not read) it behaves as before.
- **The candidate that was tested is the logged edit.** In the same place, before any write, when a measurement is being filed (`result.scores` or `result.safety` is set), `finish_round` checks that the candidate reading is the base reading with `result.edit` applied, using the project's own function for applying an edit: the same Safety section, and the same Conversation rules by ID and exact text. If it is not, it raises `PolicyChanged`. Scores are then never filed under an edit that was not what the conversations ran on. **When nothing was measured the check is not made:** a candidate the static checks stopped (a changed Safety section, more than one rule changed) is by its nature not the base with the edit applied, no score or safety result is filed for it, and its round must be logged as `reverted` with the check that failed, not refused.
- `finish_round` writes them into the round event as `scores` and `safety` when a gate outcome is given, and leaves both empty otherwise. Its signature does not change.

## 4. `puddles/graph.py`

- `RoundDeps` gets `start_policy: Optional[Path] = None`.
- When it is set, `run_round` calls `puddles.gate.ensure_starting_meta(deps.root, deps.start_policy)` after the price check and before the graph runs, and the `finish` node passes `start_policy=deps.start_policy` to `finish_round`, which then writes the promoted policy's `.meta.json` and runs the replay check.
- When it is `None`, behaviour is exactly as before.

## 5. `scripts/run_graph_round.py`

| Argument | Default | Meaning |
|---|---|---|
| `--gate` | `reverted` | `kept` or `reverted`: the stand-in, as before. `real`: the gatekeeper in `puddles/gate.py` |
| `--validation` | the `--scenarios` file | The scenarios the gate tests on. All of them are used. |
| `--noise` | `0.0` | The run-to-run noise in quality |
| `--noise-measured` | off | Says `--noise` is a measured figure. Without it the real gate keeps nothing. |
| `--token-cap` | `800` | The most tokens a policy may have |

- The script passes `start_policy=Path(args.policy)` in `RoundDeps` for every `--gate` value.
- `--fake --gate real` runs the gatekeeper with two stand-ins defined in the script: a scripted judge that reads no conversation and gives fixed scores, and a safety check that asks nothing and passes. The printed line is still `round N: <outcome>. <reason>`. On a new root the outcome is `reverted`, because the noise is not measured, and the reason says so.
- `--gate real` without `--fake` prints one line on standard error saying that the real judge and safety check are joined in a later piece, and exits with code 2 before anything is written. The later piece replaces this.
- Everything else about the script is unchanged, and `tests/acceptance/test_acc_graph_script.py` passes as it is.

## 6. What a round that reached the real gate leaves behind

Besides what [graph.md](graph.md) lists, for the work that was done. A candidate stopped by a static check leaves, of the gate's own files for round NN, only the gate report; the baseline files below are still written through `make_gate`, because the policy in force is scored before the candidate is looked at, and the candidate file the proposer wrote stays.

- `<root>/conversations/round-NN-validation.jsonl` and `<root>/scores/round-NN-validation.jsonl`. In a round that was kept, no conversation ID in them appears in `round-NN-train.jsonl`; when one does, the round is reverted (section 2, step 3) and the files stay as the record of it.
- `<root>/conversations/round-00-current.jsonl`, `<root>/scores/round-00-current.jsonl` and `<root>/scores/current.json`, from the first round that reached the gate.
- `<root>/scores/round-NN-safety.json` and `<root>/scores/round-NN-gate.json`.
- A round event whose `scores` and `safety` are filled in when they were measured.
- `<root>/policies/<starting version>.meta.json`. A reverted round adds no `.meta.json` beyond that one and no policy file beyond the proposer's candidate.

## 7. Tests

The implementer writes `tests/test_gate.py` with tests 3 to 8 of the stage file, using a scripted judge and a scripted safety check. Test 3's last line ("a yes-or-no answer that is neither word, twice") belongs to the safety check and is tested when the real one is joined.

The acceptance tests are written from this file alone, by someone who has not seen the code.

## 8. Evidence at the boundaries

Added after a sweep of the piece by five independent reviewers. Each rule closes a defect that was reproduced. Where a rule here and a sentence above disagree, the rule here is right.

The gate trusts nothing it is handed by reference. The judge and the safety check are code from another piece, and a `RoundResult` is a set of plain attributes that a later piece fills in. So the gate copies what it is given at the moment it is given, checks that it is about the thing being tested, and `finish_round` checks that the fields agree with each other before it writes.

### In `puddles/gate.py`

1. **A candidate's hash is the hash of its bytes, and it is kept even when the file does not load.** `candidate_sha256` is set whenever the candidate file's bytes could be read: also when they are not valid UTF-8, hold a carriage return, or do not load as a policy (a missing section, a repeated rule ID). It is `None` only when the file could not be read at all (not there, a folder, no permission). For a file that is valid UTF-8 this is the same value as `puddles.records.sha256_text` of its text read with no newline translation. A `candidate_file` rejection is therefore tied to the bytes that were rejected, like every other decision.
2. **The judge's results are copied and checked as they arrive.**
   - The judge's version is read once, before the first call of a scoring run. It is the version used for everything that run produces, `current.json` included.
   - The judge is given its own copy of each record. What the gate saved and what it compares against is the record as it was before the call.
   - Each result is rebuilt at once by validating the dumped fields of `result.score` as a `ScoreRecord`. The gate keeps only that copy. A result that does not validate, whose `conversation_id` is not the ID the record had before the call, or whose `judge_version` is not the version read before the first call, leaves that conversation unscored.
   - `deps.judge.version` is read again after the last call. If it differs, the evidence is not complete.
   - Evidence is complete only when there is exactly one score for each expected conversation ID: no ID missing, none twice.
   - A requested scenario ID that names more than one scenario in `deps.scenarios` raises `GateError`.
   - `KeyboardInterrupt` and `SystemExit` are not caught: "it never raises" means `Exception`. A killed gate keeps nothing and changes no policy file.
3. **The safety result is copied and checked as it arrives.** It is rebuilt at once by validating its dumped fields as a `SafetyResult`, and every later use (the check, the saved file, `log_safety`, the outcome, the report) is from that copy. A result that does not validate, or whose `policy_version` is not the candidate's version, is treated exactly as a safety check that raised: `safety_cases` fails, there is no `SafetyResult`, `log_safety` is `None`, `safety_passed` is false, and no safety file is saved. Two other cases are decided in section 2, step 2, and that step is right where this rule and it differ: a result with no cases did not run, and a result whose `passed` is true while one of its cases failed is not a pass; in both the result is saved as it came back and the gate's own statements about it say `False`.
4. **`current.json` matches only on the same values of the same types, and holds only measured scores.** A boolean is not an integer and an integer is not a float; `None` matches only `None`. Its scores are accepted only when `empathy`, `relevance` and `quality` are all present, finite, and between 1 and 5. Anything else counts as absent, and the policy in force is scored again.
5. **What is written to `current.json` is described by the run that measured it.** The judge version, the scenarios' hash, the turn limit and the seed written with a set of scores are the ones taken before that scoring run started, never read again from `deps` afterwards. This holds for the baseline and for the rewrite on `kept`.
6. Rules 2 and 5 apply the same way to the baseline (`current_scores`) and to validation.

### In `puddles/round.py`

7. **A reverted round is logged whatever state its candidate file is in.** With a gate outcome whose decision is `reverted`, `finish_round` reads the candidate's bytes once. If `result.candidate_sha256` is set and the hash of those bytes differs, it raises `CandidateChanged` before any write, as now. It does not require the candidate to be valid UTF-8, to pass the strict loader's rules or to parse. The event's `policy.new` is the version the candidate states when it parses, and `result.edit.new_policy_version` when it does not; `policy.new_sha256` is the hash of the bytes read, or `None` when the file could not be read and `candidate_sha256` is `None`. The strict rules stay for a candidate that is being promoted and for the base policy.
8. **A kept candidate is promoted only as what was tested.** With a gate outcome whose decision is `kept` and `result.candidate_sha256` set, before any write:
   - the logged-edit check of section 3 is made (a keep always files a measurement);
   - the version the candidate states is `result.edit.new_policy_version`;
   - the promoted version is that version without `-candidate`, it equals `gate.policy_in_force`, and it is neither the starting policy's version nor the `base` or `in_force` version of any round already in the log;
   - the promoted text, read by the strict loader (`loop.policy`), has the same Safety section hash and the same Conversation rules, by ID, order and exact text, as the tested candidate read by the strict loader. Nothing is written under the root to find this out.

   Any failure raises `PolicyChanged` and nothing is written.
9. **The result's fields must agree before anything is written.** With a gate outcome and `result.candidate_sha256` set, `finish_round` raises `puddles.round.InconsistentResult` (a `RuntimeError`), before any write, unless all of these hold:
   - `gate.round` is `result.round`, and `gate.edit_id` is `result.edit.id`;
   - `kept` requires `result.scores`, `result.safety` and `result.base_sha256` all set, `result.safety.passed` true and `gate.safety_passed` true;
   - `result.scores` set requires `result.base_sha256` set, and `gate.before` and `gate.after` equal to `result.scores.before` and `result.scores.after`;
   - `result.safety` set requires `gate.safety_passed` equal to `result.safety.passed`;
   - `reverted` requires `gate.policy_in_force` to be the base policy's version.

   The stand-in gate (`candidate_sha256` is `None`) is not checked this way and behaves as before.
10. **A round event that the log could not read back is not written.** Before any write, `finish_round` serializes the round event as the decision log will and checks that the log's own way of splitting lines gives back exactly that one line. If not (a name or a reason holding U+0085, U+2028, U+2029 or another character `str.splitlines` breaks on), it raises `DecisionLogError` and nothing is written. `puddles/decision_log.py` is not edited.

### In `puddles/replay.py`

11. **Replay reads a policy file as its bytes.** Every place `puddles/replay.py` reads a policy file, it reads the bytes and decodes them as UTF-8 with no newline translation, so a file whose line endings changed no longer has the logged hash and the check fails. Nothing else in that module changes.

### Known and left as they are

- A file replaced after the last check and before the write: no check in one process closes it. A lock on the root would.
- A crash between the promoted policy's write and the round event, and how the next round then picks the policy in force: the loop's start-of-round step (`prepare_round`) is the place for it, in the piece that builds the loop.
- `puddles/decision_log.py` opens the log with the platform's default encoding. On a machine whose locale is not UTF-8 a name with an accent fails at the log write. Its owner should give it `encoding="utf-8"`.
- `KeyboardInterrupt` and `SystemExit` pass through the gate (rule 2). The loop that calls it decides what a kill means.
