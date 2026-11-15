# Build spec: the turn service

The checked text path as an HTTP service, so the robot's cloud brain can use it. The brain (`puddles-stack`, FastAPI on Modal) sends the person's text and speaks the text that comes back. Today its voice goes straight to a speech model, past the policy, the crisis checker and the release step. After this piece it can call `run_turn` over HTTP, or mount the same app inside its own.

Text in, text out. No audio in this repo.

This file is the contract. It builds on [crisis-path.md](crisis-path.md) and [chat-notes.md](chat-notes.md) and changes neither.

## Ground rules

- Nothing under `puddles/eval/` or `safety/` is created or edited, and no `tests/test_eval_*.py`.
- `schemas.py`, every existing file under `puddles/`, `loop/` and `scripts/`, and every existing test file are not changed. The service imports what it needs from them, including the fakes in `scripts/chat.py`.
- No network, no listening socket and no API keys in any test. Tests reach the app in process with `fastapi.testclient.TestClient`.
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 /Users/moamen/Documents/puddles-runall-venv/bin/python -m pytest -q
```

## Layout

```
puddles/inference/service.py                 new
scripts/serve.py                             new
tests/test_inference_service.py              new: the implementer's tests
tests/acceptance/test_acc_turn_service.py    new: the acceptance tests
pyproject.toml, requirements.txt             changed: fastapi and uvicorn added
```

`fastapi>=0.110` and `uvicorn>=0.30` are added to the dependencies in both files. Both are already installed in the test interpreter (MLflow needs them). FastAPI was chosen over the standard library's HTTP server because the robot's brain is a FastAPI app and can mount this one, and because it brings request validation and an in-process test client.

## 1. `puddles/inference/service.py`

```python
@dataclass
class ServiceConfig:
    token: str                                   # the bearer token every request must carry
    policy_path: Path                            # read again at the start of every session
    root: Path                                   # where sessions/ goes
    store: NoteStore
    client_factory: Callable[[], ChatClient]     # called twice per session: helper, then checker
    note_llm_for: Callable[[bool], LLM]          # consent_for_training -> the note-taker's model
    path: Optional[CrisisPath] = None            # None: crisis.load_path(), once, in create_app
    mlflow_uri: Optional[str] = None
    idle_timeout_s: float = 900.0
    max_sessions: int = 64
    deadline_s: Optional[float] = None           # passed to Session
    helper_wait_s: float = 30.0                  # passed to Session
    clock: Callable[[], float] = time.monotonic

class TurnService:
    def live(self) -> list[str]                  # IDs of the sessions held now, sorted
    def sweep(self) -> list[str]                 # ends every idle session; returns their IDs, sorted

def create_app(config: ServiceConfig) -> fastapi.FastAPI
```

- `create_app` raises `ValueError` for a token that is not a non-empty string, for `max_sessions` below 1, and for an `idle_timeout_s` that is not a finite number above zero. A crisis path that cannot be loaded raises what `crisis.load_path` raises. Nothing is served by an app that could not be built.
- The app's `TurnService` is at `app.state.service`.
- Every handler that can wait on a model is a plain `def`, so FastAPI runs it on a worker thread. A slow turn in one session does not delay a turn in another session, and the service holds no lock of its own across a model call that would make it do so.

### Who may call

Every route except `GET /health` needs the header `Authorization: Bearer <token>`. A request is let through only when the header is present, has that exact form and the token equals `config.token`, compared with `hmac.compare_digest`. Everything else gets `401` and `{"error": "unauthorized"}` before the body is looked at, before any session is looked up and before anything else happens.

`GET /health` returns `200` and `{"status": "ok"}`, and nothing else.

### Start a session: `POST /sessions`

Request body:

```json
{"user_id": "user-01", "remember_me": false, "consent_for_training": false, "session_id": "sess-0007"}
```

- `user_id`, `remember_me` and `consent_for_training` are required. The two consent answers must be JSON booleans: a string, a number or a missing answer is refused, never read as yes or as no. `session_id` is optional. Any other field is refused.
- `user_id` must pass `note_store.check_user_id`. `session_id`, when given, must pass the check `session_log.session_path` makes, and be at most 64 characters. When it is not given the service makes one: `sess-`, the UTC time as `YYYYMMDDHHMMSS`, `-`, and 8 random hex characters.

In this order, after the body is valid:

1. A live session with this `session_id`: `409`, `{"error": "session_exists"}`.
2. With `consent_for_training` true: a session file or a record file for this `session_id` already under `root`: `409`, `{"error": "session_exists"}`. An earlier process's session is never appended to. A file counts as absent only when the look says it is not there (`FileNotFoundError`); any other error looking is a `500`.
3. A live session for this `user_id`: `409`, `{"error": "user_busy"}`. One person has one session at a time, because two would change the same notes.
4. `max_sessions` sessions already live: `429`, `{"error": "too_many_sessions"}`.
5. The policy is loaded with `puddles.graph.check_policy_file(config.policy_path)`. A policy that cannot be read or is refused (`PolicyError`, `OSError`, `UnicodeDecodeError`): `503`, `{"error": "policy_unavailable"}`.
6. `config.client_factory()` twice, and `config.note_llm_for(consent_for_training)` once. Any exception from them: `503`, `{"error": "model_unavailable"}`.
7. With `consent_for_training` true, `puddles.tracing.setup(tracking_uri=config.mlflow_uri)`. With it false the service makes no MLflow call of its own.
8. `live.start_notes(store, user_id)` gives the session's `user_notes`. If it raises, the session starts with no notes and `notes_failed` is true.

A refusal at any step leaves no session behind and nothing on disk.

Response, `201`:

```json
{"session_id": "sess-0007", "policy_version": "policy-v1", "notes_loaded": 2, "notes_failed": false}
```

The policy is read once per session, at its start. A policy file that changes afterwards reaches only sessions started later.

### One turn: `POST /sessions/{session_id}/turns`

Request body: `{"text": "..."}`. `text` is required and a string; any other field is refused. With leading and trailing whitespace removed it must be 1 to 2000 characters (`MAX_TEXT_CHARS = 2000`). The stripped text is what `run_turn` is given.

The service calls `puddles.inference.turn.run_turn(session, text)` and nothing else decides what is said. Response, `200`:

```json
{"session_id": "sess-0007", "turn": 1, "text": "That sounds heavy. What is worrying you most?",
 "fixed_handoff_response": false, "policy_version": "policy-v1",
 "check_status": "answered", "risk": "none", "category": null,
 "keyword_matched": false, "helper_failed": false,
 "timings_ms": {"keyword": 0, "checker": 412, "helper": 655, "total": 655}}
```

- `text`, `turn`, `policy_version` and `fixed_handoff_response` are the released `HelperReply`'s.
- `risk` and `category` are the `CheckedMessage`'s when `check_status` is `answered`, and `null` otherwise. A check that did not answer is never reported as `none`.
- `keyword_matched` says whether a keyword matched; the keyword itself is not returned. The checker's `reason` is not returned.
- A model that fails does not fail the request: `run_turn` gives a line from the crisis path and the response is still `200`.
- A session that is not live, because it never existed, was ended or was swept: `404`, `{"error": "session_not_found"}`. No model is called.
- If `run_turn` itself raises: `500`, `{"error": "turn_failed", "say": "<the crisis path's offline line>"}`, so the brain has a safe line to speak. The session stays live.

### End a session: `POST /sessions/{session_id}/end`

No body is needed, and one that is sent is ignored.

The service does what `scripts/chat.py` does at the end: `live.finish_session(session, store, note_llm)`, and if that raises, `turn.end_session(session)` with the notes reported as failed. The note-taker's model is the one `note_llm_for` gave at the start. The session is no longer live afterwards, whatever happened. Response, `200`:

```json
{"session_id": "sess-0007", "kept": false, "record": null,
 "notes": {"added": 0, "replaced": 0, "confirmed": 0, "deleted": 0, "undeleted": [], "failed": false}}
```

- `record` is the `ConversationRecord` as JSON when one was written, which needs `consent_for_training` and at least one turn, and `null` otherwise. `kept` is true exactly when `record` is not `null`.
- `notes` holds counts, the note IDs not yet confirmed deleted, and whether the notes step failed. Never a note's text and never an error's text.
- A session that is not live: `404`, `{"error": "session_not_found"}`. Ending twice gives `404` the second time.
- **A turn and the end of the same session never overlap.** The end waits for a turn that is running. A turn that arrives once the end has begun gets `404` and reaches no model, also when it arrived while the end was still running.

### Idle sessions

A session holds the person's words in memory, and a brain that crashes never ends it. `TurnService.sweep()` ends, exactly as the end route would, every session whose last activity is `idle_timeout_s` or more ago by `config.clock`. Activity is the session's start, and the start and the finish of each turn. A session with a turn running is not idle. A session that fails to end cleanly is still removed. `sweep` raises nothing.

Nothing in a request calls `sweep`: a person's turn never waits for another person's note-taker. `scripts/serve.py` calls it on a timer.

### Requests that are refused

- A body that is not valid JSON, or does not fit the shapes above, gets `422` and `{"error": "invalid_request", "fields": ["text"]}`: the names of the top-level fields at fault, sorted, and `[]` when the body as a whole is at fault. **The response never repeats a value that was sent.** FastAPI's own validation response does, so it is replaced.
- An error inside a handler that this contract does not name gets `500` and `{"error": "internal"}`.
- No error response, at any status, holds the person's text, a note's text or an exception's message.

### What is kept and what is logged

The table in [chat-notes.md](chat-notes.md), section 3, holds for a session of the service, row by row. The service adds no file of its own.

The service logs through `logging.getLogger("puddles.inference.service")`. A log line may hold a session ID, a route, a status, a count and an exception's type name. It never holds the person's text, a reply's text, a note's text, the token or an exception's message. Nothing is printed.

## 2. `scripts/serve.py`

```
python -m scripts.serve --policy policies/policy-v1.md --root runs/serve [--notes-root DIR]
                        [--host 127.0.0.1] [--port 8765] [--mlflow URI] [--sweep-every 30] [--fake]
```

```python
def build_config(args: argparse.Namespace, env: Mapping[str, str]) -> ServiceConfig
def main(argv: Sequence[str] | None = None, *, run: Callable[..., None] | None = None) -> int
```

- The token comes from the environment variable `PUDDLES_SERVICE_TOKEN`, with `--fake` too. It is never an argument, and never printed or logged.
- `--fake`: `client_factory` returns `scripts.chat.fake_helper()` on its first call for a session and `scripts.chat.fake_checker(path)` on its second; the store's memory is `FakeUserMemory`; the note-taker is `scripts.chat.FakeNoteTaker`, wrapped in `tracing.TracedLLM` for a session with consent for training. No keys, no network.
- Without `--fake`: `client_factory` is `loop.llm.client_from_env("HELPER")`; the memory is `PastUserMemory()`; `note_llm_for(True)` is `puddles.llm.OpenAILLM()` and `note_llm_for(False)` is `live.PlainLLM.from_env()`. The helper's variables, `PAST_API_KEY`, `OPENAI_API_KEY` and `PUDDLES_AGENT_MODEL` are checked before anything listens: one of each client is built once at startup to find a missing variable.
- A missing token, a missing variable, a policy file that `check_policy_file` refuses at startup, or a crisis path that does not load: one line on standard error starting `error: `, and exit code 2. Nothing listens.
- `main` builds the app, starts a daemon thread that calls `app.state.service.sweep()` every `--sweep-every` seconds, and calls `run(app, host=..., port=...)`, where `run` is `uvicorn.run` unless one is passed. It returns 0 when `run` returns.
- The default host is `127.0.0.1`. Reaching it from outside the machine is the deployer's decision.

## 3. Tests

`tests/test_inference_service.py`, offline, with `TestClient`, scripted clients and `FakeUserMemory`:

1. A session of three turns through the routes gives the replies `run_turn` gives for the same scripted clients, with turns numbered 1 to 3.
2. A keyword message, and a scripted `crisis` check, each give the fixed response with `fixed_handoff_response` true; a failed check gives the holding line with `risk` `null`.
3. The four rows of the consent table: what is under `root` and the notes root after the end is what the row says, and with training use off a phrase the person typed is in no file under either root, in no log record and in nothing printed.
4. Every route but `/health` without the token, with a wrong token and with a malformed header: `401`, and no session is made or changed.
5. A consent answer that is a string or is missing: `422`, no session, and the response does not repeat what was sent.
6. A turn text that is empty, only whitespace, or 2001 characters: `422`, the session's turn count is unchanged, and the response does not repeat the text.
7. The same `session_id` twice, a `session_id` whose file is already on disk, and a second session for the same person: `409` each.
8. `max_sessions` reached: `429`. After one session ends, a new one starts.
9. A policy file changed between two sessions: the second session reports the new version, and the first keeps the old one to its end.
10. A missing policy file: `503`, no session.
11. A turn after the end, and a second end: `404`.
12. A turn that is running when the end is asked for finishes first; a turn sent while the end is running gets `404` and the scripted helper is not called for it.
13. With an injected clock, `sweep` ends a session idle for `idle_timeout_s` and leaves a fresher one; the swept person's forget request was applied by the note-taker.
14. A slow helper in one session does not delay a turn in another.
15. `scripts.serve.main` with `--fake` and a stand-in `run`: the app it passes answers a turn. Without the token: exit code 2, one `error: ` line, and `run` is not called.

## Left outside

- Audio, speech to text and text to speech: `puddles-stack`.
- Calling the service from the robot's brain, and deploying it: `puddles-stack`. The hand-over is this file.
- Sessions alive when the process stops are dropped without the note-taker. A forget request made in such a session is lost with it.
- A session whose turn failed with `turn_failed` under training consent may have a session file one line short of what was said.
- Two sessions with different consent answers running at the same moment: a session without consent turns MLflow autologging off when it ends, which can lose a trace of the other session's note-taker. No person's words are exposed by it.
- More than one service process sharing a `root`: the `session_exists` and `user_busy` checks are per process.
- Rate limits, and tokens per device.
