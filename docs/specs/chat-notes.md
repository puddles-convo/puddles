# Build spec: the notes joined to a live session

A small job that joins two finished pieces: the crisis path ([crisis-path.md](crisis-path.md)) and the notes ([notes.md](notes.md)). After it, `scripts/chat.py` loads a person's notes at the start of a session and runs the note-taker at the end.

This file is the contract.

## Ground rules

- Nothing under `puddles/eval/` or `safety/` is created or edited, and no `tests/test_eval_*.py`.
- `schemas.py`, `puddles/inference/crisis.py`, `release.py`, `turn.py`, `session_log.py`, `note_store.py` and `notes.py` are not changed. No existing test file is changed.
- No network and no API keys in any test.
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 /Users/moamen/Documents/puddles-runall-venv/bin/python -m pytest -q
```

## Layout

```
puddles/inference/live.py           new
scripts/chat.py                     changed
tests/test_inference_live.py        new: the implementer's tests
```

## 1. `puddles/inference/live.py`

```python
def start_notes(store: NoteStore, user_id: str, *, today: date | None = None) -> list[str]
def finish_session(session: Session, store: NoteStore, llm: LLM, *, today: date | None = None,
                   now: datetime | None = None) -> tuple[Optional[ConversationRecord], NotesResult]
```

- `start_notes` returns `store.helper_notes(user_id, today)`: the facts the helper is given as `user_notes`. Expired follow-ups come off here. It is called for every session, whatever the person answered about being remembered: notes they agreed to earlier are still theirs. A past.dev failure gives the notes the store has on file; nothing is raised.
- `finish_session` does, in this order:
  1. copies `session.turns`, because `end_session` clears them;
  2. `puddles.inference.turn.end_session(session, now)`: the session record, kept only with consent for training;
  3. `puddles.inference.notes.take_notes(store, llm, user_id=session.user_id, session_id=session.session_id, turns=<the copy>, remember_me=session.remember_me, today=today, now=now)`. **It always runs, because forgetting must work whatever the person answered.** A session with no turns skips it and returns a `NotesResult` with nothing in it;
  4. drops the copy and returns both results.
- `session.user_id` `None` raises `ValueError` before anything is done.
- **Tracing follows consent, whatever the process did before.** A process can hold a session with training use on and then one with it off (the robot's cloud brain does), and `tracing.setup` turns on OpenAI autologging for the whole process. So with `consent_for_training` false, `finish_session` turns tracing off before the note-taker runs: if `mlflow` is already imported in this process it disables OpenAI and LangChain autologging (`mlflow.openai.autolog(disable=True)`, `mlflow.langchain.autolog(disable=True)`) and makes sure no span is open around the call; it does not import `mlflow` to do so. It stays off afterwards; a later session with consent calls `tracing.setup` again. `scripts/chat.py` with `--training no` does the same before the first message is read. `finish_session` passes `llm` to `take_notes` as it was given and opens no span. It also checks what it can: if `consent_for_training` is false and `llm` is a `puddles.tracing.TracedLLM`, it raises `ValueError` before the note-taker runs.

## 2. `scripts/chat.py`

New arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--notes-root` | the `--root` value | Where `notes/<user_id>/` goes |

- At the start it builds a `NoteStore(memory, notes_root)` and sets `session.user_notes = start_notes(store, user_id)`. It prints one line, `notes: N loaded`, **on standard error**.
- At the end it calls `finish_session` in place of `end_session`, and prints **on standard error** one line with the counts of notes added, replaced, confirmed and deleted, and the IDs past.dev has not yet confirmed deleted. Both notes lines are diagnostics and hold no fact about the person: no note text, only counts and note IDs. Standard output stays exactly as [crisis-path.md](crisis-path.md) describes it, with the line that says what was kept of the session as its last line.
- With `--fake`: the store's memory is `FakeUserMemory`, and the note-taker's model is a scripted one defined in the script that adds nothing and deletes nothing unless a message of the session holds a marker: `[note:<kind>:<fact>]` adds a note, `[forget:<note_id>]` deletes one, `[forget-everything]` sets `forget_everything`. The fake past.dev lives only as long as the process; the index and `notes.json` are on disk, so a second `--fake` session of the same person sees the first one's notes.
- Without `--fake`: the memory is `PastUserMemory()` and the model is `puddles.llm.OpenAILLM()`, never wrapped: it reports its own requests, and MLflow traces them only when `tracing.setup` has been called. With `--training yes` the script calls `tracing.setup` as it does today, so the note-taker's calls are traced with the rest of the session. With `--training no` MLflow is not set up and `mlflow.openai.autolog` is not turned on, so the note-taker's calls leave no trace. Missing `PAST_API_KEY` or `OPENAI_API_KEY` is one line on standard error and exit code 2, before the first message is read.
- A failure of the notes at the start or at the end (past.dev down, a model error) never loses the session and never raises out of `main`: the session ends as it would have, the notes line says the notes step failed, and the exit code is still 0.
- Everything else about the script is unchanged, and `tests/test_inference_crisis.py` and `tests/acceptance/test_acc_crisis.py` pass as they are, with one exception. **Real mode now needs the notes' configuration as well as the helper's**: a session whose note-taker cannot run would leave a request to forget silently undone, so it does not start. `test_chat_real_mode_uses_helper_endpoint_and_no_setup_without_consent` in `tests/acceptance/test_acc_crisis.py` ran real mode with only the helper's variables set; its author corrects it to set the two keys and stand in for `PastUserMemory` and `OpenAILLM`. Nothing else in that file changes.

## 3. What a session leaves behind

| Remember me | Use for training | Under `--root` and `--notes-root`, and in the trace store |
|---|---|---|
| No | No | Nothing new. Notes the person asked to forget are gone. |
| Yes | No | Notes only: `notes/<user_id>/`. No session file, no record, no trace. |
| No | Yes | The session file, the record and the traces. No new notes. |
| Yes | Yes | Both |

A note is a sentence the note-taker wrote about the person, so with "remember me" on it is on disk by their consent. With training use off, nothing the person typed is anywhere else.

## 4. Tests

`tests/test_inference_live.py`, offline, through `scripts.chat.main` with `--fake` and scripted input:

1. All four rows of the table: what is on disk and in a temporary MLflow store after the session is exactly what the row says. With training use off, a search of the session root and the MLflow store for a phrase the person typed finds nothing; with "remember me" on, the only files are under `notes/`. With training use off and "remember me" on, the search is also made of everything the script printed on both streams.
2. **Stage 4's test 12, through a whole session:** session one, the person agrees to be remembered and two notes are kept. Session two, the same person says no to being remembered and asks the robot to forget one of them: afterwards that note is gone from the index and from `notes.json`, the other is still there, and nothing new was added.
3. The notes loaded at the start reach the helper: the helper's system message in session two holds the first session's fact, and a different person's session does not.
4. `[forget-everything]` empties the index even with "remember me" on, and a note added in the same session is not stored.
5. `finish_session` refuses a traced model when training use is off.
6. **A session with training use on, then one with it off, in the same process**, with the real `puddles.llm.OpenAILLM` pointed at a local stand-in (an `httpx.MockTransport` or a loopback server; no network): the second session's note-taker request, once answering and once failing with an error whose text repeats the person's words, leaves no trace, no run and no marker in the MLflow store, in any file, or on either stream.
7. Row by row of the table, the MLflow store is checked for runs as well as traces, and with `--notes-root` set apart from `--root` the search for the person's phrase covers the notes root outside `notes/` too.
