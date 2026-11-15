# Build spec: the crisis path, one turn, and what a session leaves behind

Covers part A of stage 4 ([docs/plan/stage-4-inference.md](../plan/stage-4-inference.md)). The rule the release step applies is in [docs/agents/i3b-release.md](../agents/i3b-release.md).

This file is the contract: it fixes the names, signatures and file formats. Where it and the stage file disagree, this file is right.

**This is the path between a person in distress and what the robot says to them. Where this file says a draft is held, discarded or not stored, that is a safety rule.**

## Ground rules

- Ansh owns the keywords and how they are matched, the checker's prompt, its answer format and parser, its timeout, and every fixed line. They are in `puddles/eval/crisis.py`, `safety/keywords-v1.txt` and `safety/crisis-path-v1.json`. **Load them; never restate a keyword, a prompt sentence or a fixed line in code or in a test.** Tests read the expected text from his files through his functions.
- Nothing under `puddles/eval/` or `safety/` is created or edited, and no `tests/test_eval_*.py`. There is no `puddles/inference/keywords.py`: the stage file asks for one, and his `keyword_match` replaces it.
- `schemas.py` is not changed. The per-turn line is a new format; it lives in `puddles/inference/session_log.py` until it is agreed on issue #1.
- The checker's answer stays one line of text in his format. It is asked for with `complete()` and parsed with his `parse_checker_line`. It is not changed to JSON.
- No network and no API keys in any test. No test sleeps for more than 0.5 seconds in total: tests pass a short deadline instead of waiting 3 seconds.
- `puddles/inference/__init__.py` exists and is empty. Do not edit it. Another piece is being built in the same package at the same time (`note_store.py`, `notes.py`); do not create or edit those.
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 /Users/moamen/Documents/puddles-runall-venv/bin/python -m pytest -q
```

## Layout

```
puddles/inference/crisis.py         new: calls the checker model
puddles/inference/release.py        new: the release step, code only
puddles/inference/session_log.py    new: the per-turn line and the session file
puddles/inference/turn.py           new: Session, run_turn, end_session
scripts/chat.py                     new
tests/test_inference_crisis.py      new: the implementer's tests
tests/acceptance/test_acc_crisis.py new: written from this file alone, by someone else
```

## 1. `puddles/inference/crisis.py`

```python
CheckStatus = Literal["answered", "failed", "late", "unreadable"]
RECENT_TURNS = 6
CHECKER_MAX_TOKENS = 40

@dataclass(frozen=True)
class CrisisPath:
    version: str
    system: str                      # the checker's system prompt
    user_template: str               # with {recent_turns} and {text}
    timeout_s: float
    fixed_responses: dict[str, str]  # keys: self, danger, unsure
    holding_line: str
    offline_line: str
    keywords: list[str]

class CheckFailed(Exception):
    status: Literal["failed", "unreadable"]

def load_path() -> CrisisPath
def build_input(session_id: str, turn: int, turns: Sequence[Turn], text: str) -> CrisisCheckerInput
def render_user(inp: CrisisCheckerInput, path: CrisisPath) -> str
def check(line: str) -> tuple[str, Optional[str], str]
def run(client: ChatClient, inp: CrisisCheckerInput, path: CrisisPath) -> CheckedMessage
def to_record(inp: CrisisCheckerInput, risk: str, category: Optional[str], reason: str) -> CheckedMessage
```

- `load_path` builds the `CrisisPath` from `puddles.eval.crisis.load_crisis_path()` and `load_keywords()`. Nothing in it is typed out in this repo a second time.
- `build_input`: `recent_turns` is the last `RECENT_TURNS` of `turns`, the turns before this message. `text` is the message.
- `render_user` fills `path.user_template`. `{recent_turns}` is one line per turn, `<speaker>: <text>`; with no turns it is empty.
- `check` is `puddles.eval.crisis.parse_checker_line(line.strip())`. It raises `ValueError` on a line his parser rejects.
- `run` sends one request: `client.complete(path.system, [{"role": "user", "content": render_user(inp, path)}], max_tokens=CHECKER_MAX_TOKENS, temperature=0.0)`. There is no retry. Any exception from the client raises `CheckFailed` with status `failed`. A line `check` rejects raises `CheckFailed` with status `unreadable`. Otherwise it returns `to_record(...)`.
- `to_record` copies `session_id`, `turn` and `text` from the input. The model never supplies them.
- `run` knows nothing about time. The deadline is kept by `run_turn`, on the clock.

## 2. `puddles/inference/release.py`

```python
def release(*, session_id: str, turn: int, policy_version: str, path: CrisisPath,
            keyword: Optional[str], check_status: CheckStatus, checked: Optional[CheckedMessage],
            draft: Optional[HelperReply], failed_before: int) -> tuple[HelperReply, int]
```

Code only: no model, no clock, no file. It returns the reply to speak and the new count of failed checks in a row.

A check **counts** only if `check_status` is `answered`, `checked` is not `None`, and `checked.session_id` and `checked.turn` equal `session_id` and `turn`. Anything else is a check that did not answer, whatever `checked` holds.

The new count is 0 when the check counts, and `failed_before + 1` when it does not.

The text, first rule that applies:

| # | When | Text | `fixed_handoff_response` |
|---|---|---|---|
| 1 | `keyword` is not `None` | The fixed response for the category of a counting check that says `crisis`; `unsure` in every other case, including a counting crisis check with no category | true |
| 2 | The check counts and says `crisis` | The fixed response for its category; `unsure` if it has none | true |
| 3 | The check counts, says `none` or `concern`, and `draft` is not `None` | `draft.text` | false |
| 4 | The check counts, says `none` or `concern`, and `draft` is `None` | The holding line | true |
| 5 | The check does not count and the new count is 2 or more | The offline line | true |
| 6 | The check does not count and the new count is 1 | The holding line | true |

- **The helper's draft is returned only by rule 3**: no keyword match, and a `none` or `concern` check for the same session and turn.
- A keyword match gives a fixed response whatever the checker did, including when it failed twice in a row.
- The reply's `session_id`, `turn` and `policy_version` are the arguments, never the draft's or the check's.
- The texts come from `path`.

## 3. `puddles/inference/session_log.py`

```python
class TurnLine(Strict):
    session_id: str
    turn: int
    user_text: str
    keyword_matched: bool
    check_status: CheckStatus
    checked: Optional[CheckedMessage] = None
    reply: HelperReply
    timings_ms: dict[str, int] = {}

def session_path(root: Path, session_id: str) -> Path          # <root>/sessions/<session_id>.jsonl
def record_path(root: Path, session_id: str) -> Path           # <root>/sessions/<session_id>.record.json
def append_line(root: Path, line: TurnLine) -> None
def read_lines(root: Path, session_id: str) -> list[TurnLine]
def to_conversation_record(lines: list[TurnLine], *, policy_version: str, remember_me: bool,
                           consent_for_training: bool, now: datetime | None = None) -> ConversationRecord
```

- `TurnLine` does not validate unless `checked` is set exactly when `check_status` is `answered`. **A check that did not answer has no `CheckedMessage`; it is never written as `none`.**
- A session ID must match `[A-Za-z0-9][A-Za-z0-9_-]*`; anything else raises `ValueError` from `session_path` and `record_path`.
- `append_line` adds one JSON line and creates the folder. `read_lines` returns them in order; a session with no file gives an empty list.
- `to_conversation_record`: `id` is the session ID, `source` is `live`, `written_by` is `helper`, `scenario_id` is `None`. `turns` are, per line, the person's text then the reply's text, which is what was spoken. `timings_ms` has one entry per line. An empty list raises `ValueError`.

## 4. `puddles/inference/turn.py`

```python
@dataclass
class Session:
    session_id: str
    policy: loop.policy.Policy
    helper_client: ChatClient
    checker_client: ChatClient
    path: CrisisPath
    user_id: Optional[str] = None
    remember_me: bool = False
    consent_for_training: bool = False
    user_notes: Sequence[str] = ()
    root: Optional[Path] = None           # where sessions/ goes
    deadline_s: Optional[float] = None    # None: path.timeout_s
    helper_wait_s: float = 30.0
    turns: list[Turn]                     # what was said so far; starts empty
    lines: list[TurnLine]                 # one per turn, in memory; starts empty
    failed_checks: int = 0                # failed checks in a row

@dataclass
class TurnResult:
    reply: HelperReply
    keyword: Optional[str]
    check_status: CheckStatus
    checked: Optional[CheckedMessage]
    helper_failed: bool
    timings_ms: dict[str, int]            # keys: keyword, checker, helper, total

def run_turn(session: Session, text: str) -> TurnResult
def end_session(session: Session, now: datetime | None = None) -> Optional[ConversationRecord]
```

### `run_turn`

1. One turn at a time per session: a second call on the same session waits until the first has returned.
2. The turn number is `len(session.lines) + 1`.
3. The keyword check: `puddles.eval.crisis.keyword_match(text, session.path.keywords)`.
4. The checker and the helper **start together**, each on its own thread. The checker is `crisis.run`. The helper is `loop.helper.helper_reply(session.helper_client, session.policy, session.turns + [the person's turn], session_id=..., turn=..., user_notes=session.user_notes)`, unchanged. **The helper is never given the risk level, the keyword result or the checker's answer.**
5. The checker has `session.deadline_s` seconds on the clock (`path.timeout_s` when it is `None`), counted from the moment it was started. If it has not answered by then its status is `late`, and **whatever it returns afterwards is discarded, even `none`**. `CheckFailed` gives its status. Any other exception is `failed`. If the `CheckedMessage` that comes back does not carry this session's ID and this turn's number, the status is `failed` and no `CheckedMessage` is kept: a line marked `answered` always holds the check for that same turn.
6. The helper's draft is waited for only when it can be used: no keyword match and a check for this session and turn that answered `none` or `concern`. Then it is waited for up to `session.helper_wait_s` seconds from the start of the turn. A helper that raised, or has not answered by then, is a failed helper: `helper_failed` is true and the draft is `None`. In every other case the turn does not wait for the helper, and `helper_failed` says only whether the helper had already raised when the reply was released: a crisis response, the holding line and the offline line are spoken whether or not the helper answered.
7. `release(...)` decides the reply, with `failed_before=session.failed_checks`. `session.failed_checks` becomes the count it returns.
8. A `TurnLine` is made and appended to `session.lines`. `session.turns` gets the person's turn and then the released text as the helper's turn.
9. **If and only if `session.consent_for_training` is true**, the line is appended to the session file with `append_line(session.root, line)`. With consent on and `root` `None`, `run_turn` raises `ValueError` before any model is called.
10. Returns the `TurnResult`.

`run_turn` raises nothing for a failed model. Whatever the checker and the helper do, it returns a reply. The checker and the helper run on daemon threads; nothing either returns after the turn stopped waiting for it is used, in that turn or a later one, in that session or another.

In `timings_ms`, `helper` is `-1` when the helper had not answered by the time the reply was released: not measured, which is different from 0.

### Consent

| Remember me | Use for training | After the session, on disk and in the trace store |
|---|---|---|
| No | No | Nothing |
| Yes | No | Nothing from this piece. Notes are another piece. |
| No | Yes | The session file and the session record |
| Yes | Yes | The session file and the session record |

- **With `consent_for_training` false, nothing the person typed is written to disk or traced**: no session file, no record file, no MLflow span, trace or run, no log line that holds their words. The same holds for what the models returned and for the text of a model's error, because either can repeat the person's words: with consent off, neither is logged, traced or written. The clients are used as they were given, without `TracedClient`.
- With `consent_for_training` true, each turn runs inside one MLflow span named `turn`, and the two clients are wrapped in `puddles.tracing.TracedClient`, named `helper` and `crisis_checker`. `run_turn` does not call `tracing.setup`; the caller does.
- `end_session`: with consent, builds the record with `to_conversation_record` from the lines read back from the session file, writes it to `record_path(...)` as JSON, and returns it. Without consent it writes nothing and returns `None`. A session with no turns returns `None` and writes nothing. In both cases it then clears `session.turns` and `session.lines`.

## 5. `scripts/chat.py`

```
python -m scripts.chat --policy policies/policy-v1.md --user-id user-01 --remember-me no --training no --root runs/chat
```

`main(argv: Sequence[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int`.

| Argument | Default | Meaning |
|---|---|---|
| `--policy` | `policies/policy-v1.md` | The policy file, loaded with `puddles.graph.check_policy_file` |
| `--user-id` | `user-01` | Who is talking |
| `--remember-me` | `no` | `yes` or `no` |
| `--training` | `no` | `yes` or `no`: may this session be used for training |
| `--root` | `runs/chat` | Where `sessions/` goes |
| `--session-id` | made from the clock | |
| `--fake` | off | Scripted helper and checker. No keys, no network. |
| `--mlflow` | none | Tracking address, passed to `tracing.setup`, which is called only when `--training yes` |

- It reads one message per line from `stdin` until the end of input or a line that is `/quit`. Blank lines are skipped.
- For each message it prints the reply on one line starting `robot: `, then one line starting `  [` that holds the risk level (`none`, `concern` or `crisis`; or the check's status when it did not answer), whether a keyword matched, and the milliseconds for the checker, the helper and the whole turn.
- At the end it calls `end_session` and prints one line saying what was kept: the session file's path, or that nothing was kept.
- With `--fake` the helper speaks a fixed line and the checker answers from a short rule in the script, so a session can be scripted from a file. The fake checker answers in his format and its answers go through his parser.
- Without `--fake` both clients are `loop.llm.client_from_env("HELPER")`: the checker runs on the helper's endpoint.
- Exit code 0 when the session ended. Exit code 2, with one line on standard error, for a policy file that cannot be read or is malformed and for missing configuration.
- The note-taker is not connected here. `--remember-me` is recorded in the session record and nothing else; another piece joins the notes.

## 6. Tests

The implementer writes `tests/test_inference_crisis.py` with tests 1 to 11 of the stage file. In test 11 the notes are out of scope, so the two "remember me" rows leave the same files as their "no" rows.

Test 10 is a table over every combination of keyword match (yes, no), check (`none`, `concern`, `crisis`, failed, late, unreadable, answered for another turn, answered for another session) and helper (draft, failed): the draft's text is returned only by rule 3.

The acceptance tests are written from this file alone, by someone who has not seen the code.
