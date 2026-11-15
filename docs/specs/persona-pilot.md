# Build spec: the persona pilot

A demo of the robot's memory of a person changing over time. Three synthetic personas in `fixtures/personas.json` each talk to the live session code three times. After each session the note-taker updates that person's notes, and the next session's helper is given them. One persona asks the robot in session 2 to forget something from session 1. The existing judge scores each session. Every persona runs in two arms: `notes_on`, and `notes_off`, where the helper is never given notes. This lets session 3 be compared with memory and without it.

This file is the contract.

## Ground rules

- No existing file is created over, edited or deleted. In particular `schemas.py`, everything under `puddles/eval/`, `safety/` and `loop/`, `puddles/inference/*.py`, `puddles/judging.py`, `scripts/chat.py` and every existing test stay as they are. The pilot only imports them.
- No network and no API keys in any test. Each test that runs a session sets `MLFLOW_TRACKING_URI` to a `sqlite:///` file under `tmp_path`, or calls `puddles.tracing.setup` with one, so that no trace is written into the repo.
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 /Users/moamen/Documents/puddles-runall-venv/bin/python -m pytest -q
```

## Layout

```
fixtures/personas.json                      given (written with this contract)
puddles/inference/pilot.py                  new: the logic
scripts/persona_pilot.py                    new: the command
tests/test_persona_pilot.py                 new: the implementer's tests
tests/acceptance/test_acc_persona_pilot.py  new: acceptance tests, written separately from this contract
```

## 1. The persona file

`fixtures/personas.json` is one JSON object, `{"personas": [Persona, ...]}`. Each persona is:

```
{"persona_id": str, "description": str, "sessions": [PersonaSession, PersonaSession, PersonaSession]}
PersonaSession: {"session": int, "opener": str, "scenario": Scenario, "fake_markers": [str, ...]}
```

- `persona_id` matches `[a-z0-9][a-z0-9-]*` and is at most 12 characters, so `<persona_id>-off` passes `puddles.inference.note_store.check_user_id`.
- `scenario` is a `schemas.Scenario`. The simulated user's prompt (`loop.prompts.render_simulated_user_system`) uses only `situation`, `emotion_type`, `initial_intensity`, `problem_type` and `openness`. So each later session's `situation` restates what the person told the helper before, and `Scenario.opener` is left unset. The session's own `opener` field is what is said.
- `opener` is the person's exact first message of that session, sent unchanged in both arms and in both modes.
- `fake_markers` are used only by `--fake` in the `notes_on` arm (section 5). Every `{user_id}` in a marker is replaced by the arm's user ID. Real mode ignores them.
- Personas, in file order: `laidoff` (laid off, wants to be heard, dislikes advice), `examprep` (exam anxiety, wants concrete steps), `grief` (grieving their father; session 2's opener asks the robot to forget the argument with their brother from session 1).

## 2. `puddles/inference/pilot.py`

```python
ARMS = ("notes_on", "notes_off")
SESSIONS_PER_PERSONA = 3
DEFAULT_TURNS = 6
SESSION_GAP_DAYS = 7
POLICY_SENTENCE = ("The policy is the same for every person: there are no per-person policy rules. "
                   "The personal layer is the notes (situation, people, what_helps, goals, follow_up, "
                   "preferences), which the note-taker keeps and the helper is given at the start of a session.")
LIMITS_SENTENCE = ("Limits: three personas, one run each, simulated users and a model judge; "
                   "differences this small are not evidence of an effect.")
NOT_MEASURED = "not measured"
FAILED = "failed"

class PersonaSession(Strict):   # fields as in section 1
    session: int
    opener: str
    scenario: Scenario
    fake_markers: list[str] = []

class Persona(Strict):
    persona_id: str             # validated as in section 1
    description: str
    sessions: list[PersonaSession]

@dataclass
class Clients:
    helper: ChatClient          # the helper's client
    checker: ChatClient         # the crisis checker's client; a different object from `helper`
    user: ChatClient            # the simulated user's client
    path: CrisisPath            # puddles.inference.crisis.load_path() in both modes

@dataclass
class NoteChanges:
    added: list[dict]           # [{"note_id", "fact"}], from NotesResult.added
    replaced: list[dict]        # from NotesResult.replaced
    confirmed: list[dict]       # from NotesResult.confirmed
    deleted: list[dict]         # from NotesResult.deleted
    failed: bool                # NotesResult.failed
    reason: str                 # NotesResult.reason

@dataclass
class SessionResult:
    persona_id: str
    session: int
    arm: str
    user_id: str
    session_id: str
    notes_given: list[str]                # what the helper was given as user_notes
    record: Optional[ConversationRecord]  # None when the session failed
    notes_before: list[dict]              # snapshot_notes before the session
    notes_after: list[dict]               # snapshot_notes after finish_session
    changes: NoteChanges
    failed: bool
    reason: str                           # "" when not failed

@dataclass
class SessionScore:
    status: Literal["scored", "unscored", "failed"]
    score: Optional[ScoreRecord]          # set only when status is "scored"
    reason: str
```

Functions:

```python
def load_personas(path: Path) -> list[Persona]
def arm_user_id(persona_id: str, arm: str) -> str
def session_day(start: date, session: int) -> date
def snapshot_notes(store: NoteStore, user_id: str) -> list[dict]
def note_changes(before: list[dict], after: list[dict], result: NotesResult) -> NoteChanges
def run_session(persona: Persona, session: int, arm: str, *, clients: Clients, policy: Policy,
                store: NoteStore, note_llm: LLM, root: Path, turns: int = DEFAULT_TURNS,
                start: date, fake: bool = False) -> SessionResult
def score_session(judge: Judge, result: SessionResult) -> SessionScore
def run_pilot(personas: list[Persona], *, make_clients: Callable[[Persona, int, str], Clients],
              policy: Policy, store: NoteStore, note_llm: LLM, judge: Judge, root: Path,
              turns: int = DEFAULT_TURNS, start: date, fake: bool = False) -> dict
def mean_or_not_measured(values: list[float]) -> float | str
def render_report(pilot: dict) -> str
```

Here `Judge` is `puddles.judging.Judge`, `Policy` is `loop.policy.Policy`, `LLM` is `puddles.llm.LLM`, `NoteStore` and `NotesResult` are the existing ones.

### `load_personas`

Reads the file as UTF-8 JSON and returns the personas in file order. Raises `ValueError` naming the persona, or the session, when any of these fails. The top level is an object with the key `personas`, and that key holds a list of at least one persona. Each `persona_id` matches section 1 and appears only once. Each persona has exactly `SESSIONS_PER_PERSONA` sessions numbered 1, 2, 3 in order. Every `scenario` validates as `Scenario`, and every `opener` has at least one character other than whitespace. Each `scenario_id` appears only once in the file. An empty `personas` list raises `ValueError`.

### `arm_user_id`, `session_day`

- `arm_user_id(p, "notes_on")` is `f"{p}-on"`, and `arm_user_id(p, "notes_off")` is `f"{p}-off"`. Any other arm raises `ValueError`.
- `session_day(start, n)` is `start + timedelta(days=SESSION_GAP_DAYS * (n - 1))`. It is the `today` passed to `start_notes` and `finish_session`. The `now` passed to `finish_session` is that day at 12:00 UTC. Session IDs are `f"{user_id}-s{n}"`, for example `grief-on-s2`.

### `snapshot_notes`

Returns one dict per note in `store.index(user_id).notes`, in index order, for notes whose ID is also in `store.stored(user_id)`:

```
{"note_id": str, "kind": str, "fact": str, "last_confirmed": "YYYY-MM-DD", "due_date": "YYYY-MM-DD" | null}
```

`fact` comes from `store.stored(user_id)[note_id].fact`, and the other fields come from the index. Reading changes nothing. It never calls past.dev, because `index` and `stored` read the files only. A person with no folder gives `[]`.

### `note_changes`

Builds `NoteChanges` from `result`. Each ID in `added` takes its fact from `after`. Each ID in `replaced`, `deleted` or `confirmed` takes its fact from `before`, or from `after` if `before` does not have it. An ID found in neither has fact `""`. Lists keep the order of `result`. An empty `NotesResult` gives four empty lists.

### `run_session`

Runs one session through the existing live code, in this order:

1. `user_id = arm_user_id(persona.persona_id, arm)`, `today = session_day(start, session)`, `notes_before = snapshot_notes(store, user_id)`.
2. `notes_given = start_notes(store, user_id, today=today)` when `arm == "notes_on"`, and `[]` when `arm == "notes_off"`. **This is the only difference in what the helper sees.** The rest of the arm differences are the user ID and `remember_me`.
3. `Session(session_id=f"{user_id}-s{session}", policy=policy, helper_client=clients.helper, checker_client=clients.checker, path=clients.path, user_id=user_id, remember_me=(arm == "notes_on"), consent_for_training=True, user_notes=notes_given, root=root)`. Both arms have `consent_for_training` true because the text is synthetic, so `end_session` writes the record.
4. `puddles.inference.turn.run_turn(session, opener)` with the session's `opener` unchanged.
5. Then, while fewer than `turns` turns have run: `line = loop.simulated_user.next_user_line(clients.user, scenario, session.turns)`. `None` ends the session early, and that is not a failure. Otherwise `run_turn(session, line)`. `turns` counts the person's messages, the opener included. `turns` below 1 raises `ValueError` before anything is done.
6. `record, result = puddles.inference.live.finish_session(session, store, note_llm, today=today, now=<12:00 UTC on today>)`.
7. `notes_after = snapshot_notes(store, user_id)`, `changes = note_changes(notes_before, notes_after, result)`.

Failure rules:

- `run_turn` itself never raises for a model failure: a failed helper gives a line from the crisis path. Such a turn counts as a turn.
- An exception from `next_user_line` (`loop.errors.SimulatedUserError`, `loop.errors.LLMError` or anything else that is an `Exception`) stops the turns. The session still goes through step 6 so its notes are taken. The result then has `failed=True`, `record=None`, and `reason` set to `f"{type(exc).__name__}: {exc}"`.
- An exception out of `finish_session` gives `failed=True`, `record=None`, the `reason` as above, and `changes` from an empty `NotesResult(failed=True)`.
- A session that ends with `record is None` for any other reason (no turns ran) has `failed=True` and `reason="no record"`.
- `KeyboardInterrupt` is not caught.

### `score_session`

- A failed `result` returns `SessionScore("failed", None, result.reason)` and never calls the judge.
- Otherwise it calls `puddles.judging.score_conversations(judge, [result.record])`. One score in `run.scores` gives `"scored"`. A record left in `run.unscored` gives `"unscored"` with `run.reasons[record.id]`. `score_conversations` already catches the judge's exceptions and checks that the score belongs to the record, so `score_session` raises nothing for a judge failure.
- The judge is injected: any object with `version` and `score(record)` that returns an object with `.score` (a `ScoreRecord`) and `.usage` (a list). Tests pass a scripted one. The command passes `puddles.eval.judge.judge_from_env()`, unwrapped. It does not wrap it in `MeteredJudge`, because `score_conversations` already logs usage.

### `run_pilot`

For each persona in order, for each arm in `ARMS` order, for sessions 1, 2, 3: `make_clients(persona, session, arm)`, then `run_session`, then `score_session`. A failure in one session is recorded and the pilot goes on to the next session of the same arm. Returns the `pilot.json` object (section 4) and writes nothing. The command writes the files.

### `mean_or_not_measured`, and what counts

- `mean_or_not_measured([])` is `NOT_MEASURED`, never `0`. Otherwise it is the arithmetic mean, rounded to 2 decimals.
- Only `"scored"` sessions go into a mean. A `"failed"` or `"unscored"` session is shown as `failed` or `unscored`, is never counted as zero, and is left out of every mean and count of scored sessions.
- A difference (`notes_on` minus `notes_off`) is a number only when both sides are numbers. Otherwise it is `NOT_MEASURED`.
- Every number in `report.md` is computed from the `pilot` dict passed in. `render_report` never reads a file, and the command never reads a derived number back from a file it wrote.

## 3. The command: `scripts/persona_pilot.py`

```
python -m scripts.persona_pilot --personas fixtures/personas.json --policy policies/policy-v2.md \
    --out results/persona-pilot [--turns 6] [--fake] [--only PERSONA_ID] [--start YYYY-MM-DD]
```

```python
FAKE_REMEMBER = "I remember what you told me last time."
def main(argv: Sequence[str] | None = None) -> int
def fake_clients(persona: Persona, session: int, arm: str, path: CrisisPath) -> Clients
class FakeJudge   # version = "fake-judge"; score(record) -> object with .score and .usage = []
```

| Argument | Default | Meaning |
|---|---|---|
| `--personas` | required | The persona file |
| `--policy` | required | Read with `puddles.graph.check_policy_file`. The same policy for every persona and both arms |
| `--out` | required | The run folder |
| `--turns` | 6 | The person's messages per session, opener included. At least 1 |
| `--fake` | off | Offline: section 5 |
| `--only` | all | Run only this persona. An ID not in the file is an error |
| `--start` | today | The day of session 1 |

Real mode, built from the environment as `scripts/chat.py` and `scripts/run_graph_round.py` build it:

- Helper and checker: `loop.llm.client_from_env("HELPER")`, called twice so they are two clients. This needs `HELPER_URL` and `HELPER_MODEL`, and optionally `HELPER_API_KEY`, `HELPER_EXTRA_BODY` and `HELPER_MAX_TOKENS_PARAM`.
- Simulated user: `client_from_env("SIMUSER")`. This needs `SIMUSER_URL` and `SIMUSER_MODEL`, and the optional ones with the same suffixes.
- Notes: `NoteStore(PastUserMemory(), <out>/store)`, which needs `PAST_API_KEY`. The note-taker is `puddles.llm.OpenAILLM()`, never wrapped, as `scripts/chat.py` uses it with `--training yes`. It needs `OPENAI_API_KEY` and `PUDDLES_AGENT_MODEL`, and optionally `OPENAI_BASE_URL` and `PUDDLES_AGENT_TEMPERATURE`.
- Judge: `puddles.eval.judge.judge_from_env()`. `PUDDLES_JUDGE` is `panel` by default, which needs `PUDDLES_JUDGE_PANEL` and `PUDDLES_JUDGE_ASTRA`. With `fallback` it needs `PUDDLES_FALLBACK_JUDGE_MODEL`, and the Tinker key `TINKER_API_KEY` that model's client reads. `PUDDLES_JUDGE_TEMPERATURE` is optional.
- Tracing: `puddles.tracing.setup(tracking_uri=f"sqlite:///{<out>/mlflow.db}")` before the first session, because every session has consent for training on and `run_turn` opens MLflow spans.

Every missing variable is checked before the first session, by building every client. A `ConfigError` or `ValueError` from that is one line on standard error, `error: <type>: <text>`, and exit code 2, before anything is written.

Exit codes:

- `0`: the pilot ran and the three files of section 4 were written, even if some sessions failed.
- `2`: bad arguments, an unreadable or invalid persona or policy file, an unknown `--only`, missing configuration, or an `--out` that already holds a run. Nothing is written.

**It refuses to write into a run.** It writes only when `--out` does not exist or is an empty directory. Any other `--out` is refused with exit code 2, whether it holds `pilot.json` or anything else, because the note store under it would carry notes from the earlier run into session 1. The store lives at `<out>/store`, so notes always start empty.

Standard output gets one line per session as it finishes: `<persona_id> <arm> s<n>: <scored|unscored|failed>, notes +A ~R -D`, and at the end `report: <out>/report.md`. The line holds no conversation text and no note text.

## 4. Output files under `--out`

```
<out>/store/sessions/...                         written by the existing session code (consent is on)
<out>/store/notes/<user_id>/...                  written by NoteStore
<out>/sessions/<persona_id>/<arm>/s<n>.json      one per session
<out>/pilot.json
<out>/report.md
<out>/mlflow.db                                  real mode only
```

`s<n>.json`, written with `json.dumps(..., indent=2)`:

```
{
  "persona_id": str, "session": int, "arm": "notes_on" | "notes_off", "user_id": str, "session_id": str,
  "failed": bool, "reason": str,
  "notes_given": [str, ...],
  "record": <ConversationRecord.model_dump(mode="json")> | null,
  "notes_before": [Snapshot, ...],
  "notes_after": [Snapshot, ...],
  "changes": {"added": [{"note_id": str, "fact": str}, ...], "replaced": [...], "confirmed": [...],
              "deleted": [...], "failed": bool, "reason": str},
  "score": {"status": "scored" | "unscored" | "failed", "reason": str,
            "empathy": int | null, "relevance": int | null,
            "emotion_before": int | null, "emotion_after": int | null,
            "judge_version": str | null, "note": str | null}
}
```

`Snapshot` is the dict from `snapshot_notes`. The score's number fields are `null` unless `status` is `"scored"`.

`pilot.json` is what `run_pilot` returns:

```
{
  "policy_version": str, "policy_path": str, "turns": int, "fake": bool, "start": "YYYY-MM-DD",
  "judge_version": str,
  "policy_sentence": POLICY_SENTENCE, "limits_sentence": LIMITS_SENTENCE,
  "personas": [
    {"persona_id": str, "description": str,
     "sessions": {"notes_on": [S1, S2, S3], "notes_off": [S1, S2, S3]}}
  ]
}
```

Each `S` has exactly the shape of `s<n>.json`. `report.md` is `render_report(pilot)`.

## 5. `--fake`

Fully offline: no key, no network, no past.dev.

- Checker: `scripts.chat.fake_checker(path)`, where `path` is `crisis.load_path()`. Helper: the fake helper below, not `scripts.chat.fake_helper()`, which always says the same line.
- Note-taker: `scripts.chat.FakeNoteTaker()`, unwrapped. It acts only on `[note:<kind>:<fact>]` and `[forget:<note_id>]` markers in the conversation.
- Memory: `puddles.inference.note_store.FakeUserMemory()`, as `scripts.chat` uses it, one for the whole pilot.
- Judge: `FakeJudge`. Its `score(record)` returns a `ScoreRecord` with `id=f"score-{record.id}"`, `conversation_id=record.id`, `judge_version="fake-judge"`, `written_by="judge"`, `created_at` = `record.created_at`, `strategies` = one `"Question"` per helper turn, and `note="scripted"`. Its scores are empathy 4 and relevance 4 when the record's first helper turn starts with `FAKE_REMEMBER`, and empathy 3 and relevance 3 otherwise. `emotion_before` is 4 and `emotion_after` is 3.
- Simulated user: `fake_clients` builds `user = loop.llm.FakeClient(<callable>)`, and `run_session` still calls the real `next_user_line` with it. Its first answer in the `notes_on` arm is `"Yes, that is part of it. " + " ".join(markers)`, where `markers` are the session's `fake_markers` with `{user_id}` replaced by `arm_user_id(persona_id, "notes_on")`. In the `notes_off` arm, and for a session with no markers, it is `"Yes, that is part of it."`. Every later answer is `"I suppose so."`. It never returns `[END]`. Markers are sent only in `notes_on` because a `[forget:<id>]` for a note the person does not have is refused by `puddles.inference.notes.check`, and in `notes_off` adds are dropped anyway.
- Note IDs are `note-<user_id>-NNN` from `NoteStore.next_note_id`, counted from 001 in a fresh store. That is why `grief`'s session 2 marker `[forget:note-{user_id}-002]` deletes the session 1 note about the argument with their brother. A run with `--fake` therefore shows notes added in every persona and one note deleted in `grief` session 2.
- Helper: `fake_clients` gives `helper = loop.llm.FakeClient(<callable>)`. The callable answers `scripts.chat.FAKE_HELPER_LINE`, except for the first reply of a session (the messages hold one user message) whose system message does not contain `loop.prompts.NO_NOTES`, that is, notes were given. That reply is `FAKE_REMEMBER + " " + FAKE_HELPER_LINE`, with `FAKE_REMEMBER = "I remember what you told me last time."` defined in the script. The side-by-side table then shows a difference, and the fake helper learns about notes only from the system message `loop.prompts.render_helper_system` built.
- `--fake` calls `puddles.tracing.setup` with `<out>/mlflow.db` as real mode does, so the session code runs the same way.

## 6. `report.md`, exactly

```
# Persona pilot

<POLICY_SENTENCE>

Policy: <policy_version> (<policy_path>). Turns per session: <turns>. Mode: <real|fake>. Judge: <judge_version>.

## <persona_id>: <description>

### Notes after each session (notes_on)

| Session | Kind | Fact | Change |
|---|---|---|---|
```

- One row per note in `notes_after` of each `notes_on` session, in index order. `Change` is `added` if the note ID is in that session's `changes.added`, otherwise empty. A row is also written for each note in `changes.deleted`, with its fact and `Change` `deleted`, and for each ID in `changes.replaced`, with `Change` `replaced`. The forgotten note in `grief` session 2 appears as a `deleted` row. A session with no notes after it and no changes has one row: `| <n> | - | (no notes) | |`. A failed session has one row: `| <n> | - | (session failed) | |`. The `notes_off` arm keeps no notes and has no notes table.
- `|` and newlines inside a fact are written as `\|` and a space.

```
### Judge scores

| Session | notes_on empathy | notes_off empathy | Difference | notes_on relevance | notes_off relevance | Difference |
|---|---|---|---|---|---|---|
| 1 | 4 | 3 | +1 | ... |
```

- A cell is the integer score, `failed` or `unscored`. The difference is signed (`+1`, `0`, `-2`) when both cells are integers, and `not measured` otherwise.

```
### Session 3, first two helper replies

| Reply | notes_on | notes_off |
|---|---|---|
| 1 | <text> | <text> |
| 2 | <text> | <text> |
```

- Replies are the first two `helper` turns of each arm's session 3 record. A missing one (a failed session or a shorter conversation) is `-`.

Then, after all personas:

```
## Overall

| Measure | notes_on | notes_off | Difference |
|---|---|---|---|
| Mean empathy, session 3 | ... | ... | ... |
| Mean relevance, session 3 | ... | ... | ... |
| Mean empathy, all sessions | ... | ... | ... |
| Mean relevance, all sessions | ... | ... | ... |
| Sessions scored | <k> of <n> | <k> of <n> | |
| Sessions failed | <f> | <f> | |
| Notes kept after session 3 | <count> | 0 | |
| Notes deleted on request | <count> | 0 | |

<LIMITS_SENTENCE>
```

- Means come from `mean_or_not_measured` over the `"scored"` sessions only, so they read `not measured` when there are none. Differences are `notes_on` minus `notes_off`, with 2 decimals and a sign, or `not measured`. `<n>` counts the sessions that ran in that arm. `Notes kept after session 3` sums the length of each persona's `notes_on` session 3 `notes_after`. A failed session 3 adds nothing to the sum and is counted in `Sessions failed`. `Notes deleted on request` sums the length of `changes.deleted` over all `notes_on` sessions.
- `POLICY_SENTENCE` and `LIMITS_SENTENCE` appear exactly once each, word for word.

## 7. Tests (`tests/test_persona_pilot.py`)

All offline, with the fakes of section 5 or scripted clients, in `tmp_path`, with tracking pointed at `tmp_path`.

1. `load_personas("fixtures/personas.json")` gives 3 personas, `laidoff`, `examprep` and `grief`, each with sessions 1, 2, 3, and every scenario a `Scenario`. Each of these raises `ValueError`: a duplicate `persona_id`, a 13-character ID, an ID with a capital letter, two sessions, sessions out of order, an empty opener, an invalid scenario field, and an empty `personas` list.
2. `arm_user_id` gives `x-on` and `x-off` and refuses any other arm. `session_day` puts session 3 fourteen days after session 1.
3. `snapshot_notes` on a person with no folder gives `[]`. After a session that adds two notes, it gives both, with kind, fact and dates, in index order.
4. `note_changes` with an empty `NotesResult` gives four empty lists, and a deleted ID takes its fact from `before`.
5. `run_session` in `notes_off`: the helper's system message in every turn contains the "no notes" text from `loop.prompts`, even though the `notes_on` store already holds notes. The session's `remember_me` is false, its user ID ends in `-off`, and no note is stored for that user ID.
6. `run_session` in `notes_on`, session 2: the helper's system message holds session 1's facts, and the record is written under `<root>/sessions`.
7. Both arms call `run_turn` with the same opener text, and the same number of turns with the same scripted user. Spy on `puddles.inference.pilot.run_turn` and `finish_session`.
8. `--turns 1` runs only the opener. A scripted user that returns `[END]` at once ends the session after the opener, and that session is not failed.
9. A simulated user that raises `SimulatedUserError` in session 2 gives a failed session 2 with a reason and no record. Session 3 still runs, `score_session` does not call the judge for session 2, and the report shows `failed` for it, never `0`.
10. `score_session` with a judge that raises gives `"unscored"`. A judge that returns a score for another conversation also gives `"unscored"`.
11. `mean_or_not_measured([])` is `"not measured"`. A pilot where every session in one arm failed shows `not measured` for that arm's means and differences, and `0 of 3` for scored sessions.
12. `main([... "--fake"])` with `--out` under `tmp_path` exits 0 and writes `pilot.json`, `report.md` and 18 `s<n>.json` files. The report has a notes table per persona with at least one `added` row, `grief` session 2 has a `deleted` row for the brother note, and that note is gone from `notes_after` of sessions 2 and 3. `POLICY_SENTENCE` and `LIMITS_SENTENCE` each appear exactly once.
13. `main` with `--out` that holds any file exits 2 and changes nothing in it. `--only nobody` exits 2. `--only grief` writes 6 session files.
14. Real mode with `HELPER_URL` unset exits 2 with one `error:` line, before `--out` is created.
15. The numbers in `report.md` come from the dict: `render_report` on a hand-built `pilot` dict, with no files on disk, gives the expected means and differences.

## Left outside

- More than one run per persona, seeds, confidence intervals or any significance test.
- Per-person policy rules. The policy file is shared, and that is what the pilot shows.
- The crisis path's behaviour. The personas hold no crisis content, and a session that a checker sends down the fixed path is recorded as it happened.
- Changing `FakeNoteTaker` to replace or confirm notes. `--fake` shows only adds and deletes.
- Tracing without consent, and the `--training no` path of `scripts/chat.py`.
- Cost accounting for the pilot (no usage meter is set). Judge usage is logged only if a meter is current.
- A README section. It is written from a real run's `report.md` afterwards.

## Amendments (15 November, after the test author's defect list)

These override the text above where they differ.

1. **`run_pilot` has two more keyword arguments**, both with defaults: `policy_path: str = ""`, which is what `pilot.json` carries as `policy_path`, and `on_session: Callable[[dict], None] | None = None`, called with each session's `s<n>.json` dict as the session finishes. The command passes both.
2. **"Writes nothing" means the pilot's own files.** `run_pilot` writes no `s<n>.json`, `pilot.json` or `report.md`; the existing session code still writes the session file and the record under `root`, and the note store its files.
3. **The two arms differ in what the helper sees only by the notes.** With `--fake` the scripted user's first line also differs, because the markers that make the fake note-taker act are sent only in `notes_on`. That is a property of the fakes, not of the pilot, and the report of a `--fake` run proves only that the report renders.
4. **`<out>/mlflow.db` is written in both modes.**
5. **A `deleted` or `replaced` row takes its kind from `notes_before`**, and `-` when the note is not there.
6. **Rounding:** a mean is rounded to 2 decimals for display only. A difference of means is computed from the unrounded means and then rounded to 2 decimals. A per-session difference of zero is written `0`; a difference of means of zero is written `0.00`, without a sign.
7. **`--fake` shows notes added in every persona only when `--turns` is at least 2**, since the markers are in the simulated user's first answer.
8. **`--workers N`** (default 1, at least 1): the six chains (persona, arm) run on up to N threads. The three sessions of one chain always run in order on one thread. Results, files, `pilot.json` and the report are in the same order whatever finishes first, and the lines on standard output may come in any order. `run_pilot` takes `workers: int = 1`. Each chain has its own user ID, so no two threads touch the same person's notes. A failure in one chain does not stop the others.
9. **Real mode builds the note store with the store's defaults**, so the wait for past.dev to confirm a note and the timeout of the recall at session start follow `PUDDLES_NOTE_WRITE_WAIT_S` and `PUDDLES_NOTE_RECALL_TIMEOUT_S` once those exist in `NoteStore`. The pilot passes no timeout of its own.
10. **Differences in the Overall table are paired.** A mean of one arm is still taken over that arm's scored sessions. But the `Difference` cell of each "Mean ..." row is computed only over the pairs (persona, session) that are scored in both arms: the mean, over those pairs, of `notes_on` minus `notes_off`, from unrounded values, shown with a sign and 2 decimals, and `not measured` when there is no such pair. A new row follows the four mean rows: `| Pairs compared, session 3 | <k> of <n> | | |` and `| Pairs compared, all sessions | <k> of <n> | | |`, where `<n>` is the number of (persona, session) pairs that ran. So a session that failed in one arm never makes the difference compare different people.
11. **Proposed notes that were not stored are shown.** `NoteChanges` has two more fields, `not_written: int = 0` and `not_written_reasons: dict[str, int]` (empty by default), which `note_changes` copies from the `NotesResult` (see `NotesResult.not_written_reasons` in `docs/specs/notes.md`). Both are written under `changes` in each `s<n>.json` and in each session of `pilot.json`. In `render_report`, the notes table of a `notes_on` session whose `not_written` is above zero gets one extra row after that session's rows: `| <n> | - | (<k> proposed notes not stored: <reason> x<count>, ...) | not stored |`, the reasons sorted by name. The Overall table gets one row after "Notes deleted on request": `| Notes proposed but not stored | <sum over notes_on sessions> | <sum over notes_off sessions> | |`. In `notes_off` every proposed note is not stored by design (reason `consent`), which is why both columns are shown. With `--fake` the `notes_off` column is 0, because the markers that make the fake note-taker propose notes are sent only in `notes_on` (amendment 3). A `pilot.json` written before this amendment, with neither key, renders as zero, without error.
