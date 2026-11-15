# Build spec: a person's notes

Covers part B of stage 4 ([docs/plan/stage-4-inference.md](../plan/stage-4-inference.md)): the note store and the note-taker. The agent is described in [docs/agents/i5-note-taker.md](../agents/i5-note-taker.md) and the memory design in [docs/memory.md](../memory.md).

This file is the contract: it fixes the names, signatures and file formats. Where it and the stage file disagree, this file is right.

**Three rules here are promises to a person: nothing is remembered without their consent, forgetting always works, and one person's notes never reach another person or a training agent.**

## Ground rules

- Nothing under `puddles/eval/` or `safety/` is created or edited, and no `tests/test_eval_*.py`.
- `schemas.py` is not changed. `puddles/memory.py` is not changed.
- No network and no API keys in any test.
- `puddles/inference/__init__.py` exists and is empty. Do not edit it. Another piece is being built in the same package at the same time (`crisis.py`, `release.py`, `session_log.py`, `turn.py`, and `scripts/chat.py`); do not create or edit those.
- This piece never calls MLflow and never wraps a model in a traced client. Whether the note-taker's model calls are traced is the caller's decision, and it depends on the person's consent.
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 /Users/moamen/Documents/puddles-runall-venv/bin/python -m pytest -q
```

## Layout

```
puddles/inference/note_store.py     new
puddles/inference/notes.py          new
docs/agents/i5-note-taker.md        changed: the last paragraph of the system message, and the opening of section 4
tests/test_inference_notes.py       new: the implementer's tests
tests/acceptance/test_acc_notes.py  new: written from this file alone, by someone else
```

## Decisions this spec takes

[docs/memory.md](../memory.md), section 10, leaves three things open. This spec settles them for now, in the way that is simplest to undo:

1. **Where a note's sentence is read from.** The store keeps each live note's full `UserNote` in `notes.json`, beside the index. A recall from past.dev is still made as the person and filtered through the index; a live note the recall did not return is filled in from the file. past.dev is a searchable copy.
2. **Where the index is stored.** Under the store's root: `<root>/notes/<user_id>/index.json`.
3. **Deleting from past.dev.** The store talks to past.dev through the `UserMemory` protocol below. The real client's `delete` and `ensure_audience` follow the documented past.dev API (https://past.dev/docs/memory-api). **If the documentation cannot be read, or has no such call, the method raises `NotImplementedError` with a message that says so. Do not guess an endpoint.**

## 1. `puddles/inference/note_store.py`

```python
NOTE_TAKER = "note-taker"
RECALL_QUERY = "What should the robot remember about this person?"

def audience_for(user_id: str) -> str          # "person-<user_id>"
def check_user_id(user_id: str) -> str

class UserMemory(Protocol):
    def ensure_audience(self, audience: str, identities: list[str]) -> None: ...
    def ingest(self, entry: MemoryEntry) -> str: ...
    def wait(self, ingestion_id: str, timeout_s: float = 60) -> bool: ...
    def recall(self, query: str, identity: str, level: str = "low", limit: int = 20) -> list[dict]: ...
    def delete(self, entry_id: str) -> bool: ...          # True when past.dev confirmed it

class PastUserMemory(puddles.memory.PastClient)           # adds ensure_audience and delete
class FakeUserMemory(puddles.memory.FakeMemory)           # adds ensure_audience and delete, for tests and offline runs
    fail_deletes: int = 0                                 # the next N delete calls return False and delete nothing

class NoteStore:
    def __init__(self, memory: UserMemory, root: Path, *, delete_attempts: int = 3) -> None
    def ensure_person(self, user_id: str) -> None
    def index(self, user_id: str) -> LiveNoteIndex
    def save_index(self, index: LiveNoteIndex) -> None
    def stored(self, user_id: str) -> dict[str, UserNote]
    def next_note_id(self, user_id: str) -> str
    def expire(self, user_id: str, today: date) -> list[str]
    def live_notes(self, user_id: str, today: date) -> list[ExistingNote]
    def helper_notes(self, user_id: str, today: date) -> list[str]
    def write(self, note: UserNote) -> None
    def take_off(self, user_id: str, note_ids: list[str]) -> None
    def delete(self, user_id: str, note_ids: list[str]) -> list[str]
    def pending_deletes(self, user_id: str) -> list[str]
```

### Who is who

- `check_user_id` returns the ID if it matches `[a-z0-9][a-z0-9-]*`, is at most 33 characters (past.dev allows an audience name 40, and the audience is `person-<user_id>`), and is not one of `puddles.memory.TRAINING_IDENTITIES`, `note-taker` or `project`. Anything else raises `ValueError`. Every method that takes a `user_id` checks it first, so a user ID can never name a path outside the root or a training identity.
- A person's audience is `person-<user_id>`. Its identities are exactly `[user_id, "note-taker"]`. The person's own identity for a recall is `user_id`.
- `ensure_person` calls `memory.ensure_audience(audience_for(user_id), [user_id, NOTE_TAKER])`. Every method that reads or writes past.dev for a person calls it first.

### Files, all under `<root>/notes/<user_id>/`

| File | Holds |
|---|---|
| `index.json` | The `LiveNoteIndex`. Only notes listed here reach the helper. |
| `notes.json` | `{note_id: UserNote}` for the notes on the index. Between a `write` and the `save_index` that follows it, it also holds the new note that is not yet on the index. |
| `counter.json` | `{"next": N}`: the next note number. A number is never used twice, even after a delete. While a write is on its way it also holds `"writing": [{"note_id", "owner"}]`: the notes that have been or are about to be sent to past.dev and are not yet on the index. With no write on its way the file is exactly `{"next": N}`. |
| `pending_deletes.json` | A list of note IDs that are off the index and that past.dev has not yet confirmed deleted |

Every file is written to a temporary file in the same folder and then renamed over the old one.

**A change that is cut off is finished by the next call.** A change that touches more than one of the four files and past.dev is written down before its first step: `pending_deletes.json` is the intent "these notes are coming off", and `writing` is the intent "this note is being sent". Every method that changes anything (`ensure_person` and `next_note_id` included), and `live_notes` and `helper_notes`, first finishes what an interrupted call left: every ID in `pending_deletes.json` is taken off the index and out of `notes.json` before anything else happens and before any remote delete is asked for, and a `writing` entry left by a store that is gone, for a note that never reached the index, is queued for deletion. A note's ID is in `writing` before the note is sent, so "forget everything" after a kill at any point still finds it. The rule errs towards forgetting: a note in flight may be lost; a forgotten note never comes back, and `save_index` refuses an index that names a note waiting to be deleted. `index`, `stored` and `pending_deletes` only read. Two sessions of one person at the same time are outside this contract. The `owner` of a `writing` entry names the `NoteStore` instance that made it: a new instance on the same root is a restart, and treats an entry it did not make as left by a store that is gone.

**A note ID belongs to one person.** An ID belongs to `user_id` when it is `note-<user_id>-` followed by digits only. No method of the store writes, indexes, takes off or asks past.dev to delete an ID that does not belong to the person it was called for: `write`, `save_index`, `take_off` and `delete` either raise `ValueError` for such an ID or leave it out, and either way the other person's files and their past.dev entries are exactly as they were.

- `index` returns an empty index for a person with no file.
- `save_index` writes `index.json`, and removes from `notes.json` every note that is not on the index it was given. **Every note it removes this way is added to `pending_deletes.json`,** so a note that was written but never reached the index (a crash between the two) is still deleted from past.dev. A `LiveNoteIndex` over a limit cannot be built, so it cannot be saved.
- `next_note_id` returns `note-<user_id>-<NNN>`, with the user ID exactly as it is, three digits or more, starting at 001, and moves the counter on. The user ID is not shortened or changed in any way: `user-01` and `user01` are two people and must never share a note ID, because past.dev stores and deletes by that ID. (The example IDs in the agent's file, `note-user01-004`, are older than this rule.)

### Reading

- `expire` takes off the index every `follow_up` whose expiry day has passed: `today` is later than `due_date + FOLLOW_UP_EXPIRY_DAYS` days, or than `last_confirmed + FOLLOW_UP_EXPIRY_DAYS` days when it has no due date. It saves the index, deletes those notes with `delete`, and returns their IDs.
- `live_notes` calls `expire` first. Then it recalls as the person (`memory.recall(RECALL_QUERY, identity=user_id)`), and keeps only results whose source ID is on the index. **A note that is off the index is never returned, even while past.dev still holds it.** A note on the index that the recall did not return is taken from `notes.json`. A recall that raises, or an audience that cannot be set up, is ignored here: the facts are in `notes.json`. Each `ExistingNote` has the fact from `notes.json`, and the kind and dates from the index. The order is the index's.
- `helper_notes` is the facts of `live_notes`, in the same order. It is what goes into `loop.helper.helper_reply(..., user_notes=...)`.

### Writing

- `write(note)` checks that `note.id` is not already in `notes.json`, sends one `MemoryEntry` and waits for it. The entry: `id` is the note's ID; `content` is `{date}: {fact}` with the date as in `15 Nov 2026`, taken from `last_confirmed`; `label` is `note-taker: user note`; `identity` is `note-taker`; **`audience` is the person's audience, on every write**; `metadata` is `{"type": "user_note", "kind", "user_id", "session_id"}`. If past.dev does not confirm the write within the wait, or raises, `write` raises `puddles.memory.PastError` and the note is in neither file. **Once the entry has been handed to `memory.ingest`, an unconfirmed write is always queued:** the note's ID is added to `pending_deletes.json` whether or not an ingestion ID came back, because a response can be lost after past.dev stored the entry. Only a failure before `memory.ingest` is called leaves nothing queued. When it is confirmed the note is added to `notes.json`. It does not touch the index.
- `take_off` removes the IDs from the index and from `notes.json`, adds them to `pending_deletes.json`, and saves. From this moment the helper cannot see them. An ID that is not on the index is ignored.
- `delete` first takes the IDs off the index (`take_off`), so nothing can be deleted while it is still visible. It then calls `memory.delete(note_id)` for each ID, up to `delete_attempts` times each, until it returns true. An exception from `memory.delete` counts as a failed attempt. A confirmed ID is taken out of `pending_deletes.json`. It returns the IDs that are still not confirmed; they stay in `pending_deletes.json`. It never raises for a failed delete. Every call to `delete` also retries the IDs already pending for that person.

**Two time limits.** `NoteStore(..., write_wait_s=300.0, recall_timeout_s=5.0)`. `write` passes `write_wait_s` to `memory.wait`; the old 60 s default lost every note, because past.dev took 54 s, 81 s and over 63 s to complete a note's ingestion (first real calls, 15 Nov 2026). `recall_timeout_s` is given to the recall in `live_notes` as `timeout_s` (an optional keyword on `UserMemory.recall`; a memory whose `recall` has no such keyword is called as before); a timeout counts as a failed recall, and the facts still come from `notes.json`. The first recall on a new audience took 7.7 s once and hit the client's 30 s read timeout twice, which would hold up `helper_notes` at the start of a session for nothing it uses. When not passed, each is read from `PUDDLES_NOTE_WRITE_WAIT_S` or `PUDDLES_NOTE_RECALL_TIMEOUT_S` if set, else the default. A value that is not a finite number of seconds above zero raises `ValueError` naming the setting or the variable.

## 2. `puddles/inference/notes.py`

```python
SYSTEM = puddles.agents.system_prompt("i5-note-taker.md")

class NoteTakerOutput(Strict):
    forget_everything: bool = False
    actions: list[NoteAction] = []

@dataclass
class NotesResult:
    added: list[str]              # new note IDs written, replacements included
    replaced: list[str]           # old note IDs that were replaced
    confirmed: list[str]
    deleted: list[str]            # note IDs taken off the index by a delete or by "forget everything"
    dropped_by_limit: list[str]   # existing note IDs dropped by a limit
    not_written: int              # adds and replaces that were dropped: by consent, by "forget everything" or by a limit
    not_written_reasons: dict[str, int] = field(default_factory=dict)  # why each of those was not stored
    undeleted: list[str]          # IDs off the index that past.dev has not confirmed deleted
    forgot_everything: bool
    failed: bool                  # the model's answer failed twice; nothing was changed
    reason: str = ""

def build_input(store: NoteStore, user_id: str, turns: list[Turn], *, remember_me: bool,
                today: date) -> NoteTakerInput
def render_user(inp: NoteTakerInput) -> str
def check(out: NoteTakerOutput, inp: NoteTakerInput) -> list[str]
def run(inp: NoteTakerInput, llm: LLM) -> NoteTakerOutput
def to_record(action: NoteAction, *, user_id: str, session_id: str, note_id: str, today: date,
              now: datetime, old: Optional[ExistingNote] = None) -> UserNote
def take_notes(store: NoteStore, llm: LLM, *, user_id: str, session_id: str, turns: list[Turn],
               remember_me: bool, today: date | None = None, now: datetime | None = None) -> NotesResult
```

### The model's answer

- A list cannot be asked for through `LLM.structured`, so the answer is a `NoteTakerOutput`. It is a new output format; it lives in this module until it is agreed on issue #1. The only change to the system message in `docs/agents/i5-note-taker.md` is its last paragraph, which now asks for a JSON object with `forget_everything` (true only if the person asked the robot to forget everything about them) and `actions`. Section 4 of that file ("Output") is brought into line: its first sentences say the answer is an object with those two fields.
- `build_input`: `existing_notes` is `store.live_notes(user_id, today)`, `may_add_notes` is `remember_me`, `today` is the ISO date.
- `render_user` follows the user message in the agent's file: every kind is shown with its count and its limit, the empty ones too.
- `check` returns the problems, an empty list when there are none:
  - a `replace`, `confirm` or `delete` with no `note_id`, or with a `note_id` that is not one of `inp.existing_notes`. **A note ID that is unknown, or that belongs to another person, is refused**;
  - an `add` or `replace` with no `kind` or no `fact`, or with a fact that is empty or only spaces;
  - an `add` that has a `note_id`;
  - the same `note_id` named by two actions.
- `run` is `puddles.llm.call_checked` around `llm.structured(SYSTEM, render_user(inp) + feedback, NoteTakerOutput)`: a failed check is retried once with the reasons, and a second failure raises `puddles.llm.AgentFailed`.

### `take_notes`: the whole step after a session

**It always runs, whatever the person answered, because forgetting must always work.** `today` and `now` default to the clock.

1. `build_input`, then `run`. If `run` raises (`AgentFailed`, or any other exception from the model), it returns a result with `failed=True` and the reason. **Nothing is changed: no note added, none lost.** Follow-ups that had expired come off in `build_input`, before the model is asked; that is expiry, not a change by the note-taker, and it stands when the answer fails. On this path `undeleted` still lists every ID that is off the index and not yet confirmed deleted, and the other lists are empty.
2. Which actions are kept:
   - with `remember_me` false, every `add`, `replace` and `confirm` is dropped. Deletes are kept;
   - with `forget_everything` true, every existing note is deleted, and every `add`, `replace` and `confirm` from this session is dropped;
   - a `due_date` on any kind but `follow_up` is ignored. A replacing `follow_up` that gives no `due_date` keeps the old note's.
3. The new index is worked out in full before anything is saved: take off the replaced and deleted notes; set `last_confirmed` to `today` on each confirmed note; add the new notes, each with a new ID from `store.next_note_id` and `last_confirmed` set to `today`; then, for any kind over its limit in `NOTE_CAPS`, drop notes of that kind, the one with the oldest `last_confirmed` first and on a tie the one with the lower note number, until it fits. A new note can be dropped here and is then never written.
4. It is saved in this order:
   1. `store.take_off(...)` for every note that came off, and the index saved with the confirms, **without the new notes**. From this moment the helper cannot see a replaced, deleted or dropped note.
   2. `store.write(...)` for each new note. A note whose write raises `PastError` is left out and counted in `not_written`; the others go on.
   3. The index saved again with the new notes that were written.
   4. `store.delete(...)` for every note that came off. The IDs it returns go into `undeleted`.
5. Returns the `NotesResult`.

`not_written_reasons` says why each proposed `add` or `replace` counted in `not_written` was not stored. Its keys come from a fixed set: `"consent"` (dropped because "remember me" is off), `"forget_everything"` (dropped because the same answer asked to forget everything; this wins over `"consent"` when both hold), `"limit"` (a new note dropped by a per-kind limit in step 3), and `"write_failed:" + type(exc).__name__` for a note whose `store.write` raised in step 4.2. Only the exception's type name is kept, never its message, which can repeat the person's words or a note's text. `NoteStore.write` wraps every failure, an ingest that raised and a wait that ran out alike, in `PastError`, so today that key is always `"write_failed:PastError"`. The values sum to `not_written`. An ordinary session, and a session whose model answer failed (step 1), give an empty dict. Nothing else about `take_notes` changes.

`replaced` lists the old notes that came off because of a replace, whether or not the new note's write then succeeded. The order of the IDs in each list of the result is not fixed.

`take_notes` raises nothing for a model or a past.dev failure, including an audience that cannot be set up and a recall that raises. After it returns, the index never names a note that was not written and is never over a limit.

## 3. Tests

The implementer writes `tests/test_inference_notes.py` with tests 12 to 17 of the stage file, using a scripted model and `FakeUserMemory`:

12. With "remember me" off, adds and replaces are dropped and deletes are applied. Run through `take_notes` on a whole session: the person said no to being remembered, asks the robot to forget one thing, and that note is gone afterwards from the index, from `notes.json` and from the fake past.dev.
13. After "forget everything" the index is empty, and an add or a replace from the same session is not stored.
14. A note taken off the index is not returned by `live_notes`, even while the fake past.dev still holds it (`fail_deletes`).
15. A `note_id` that is unknown, or that belongs to another person, is refused.
16. A delete that fails is retried, in the same call and again on the next one.
17. One person's notes never appear in another person's `helper_notes` or `NoteTakerInput`, and a recall under each of `puddles.memory.TRAINING_IDENTITIES` returns none of them.

Also: a second failed answer changes nothing; the limits; the order of the four saves; a follow-up's expiry; a user ID that is refused.

The acceptance tests are written from this file alone, by someone who has not seen the code.
