# Note store against past.dev: first real calls

Date: 2026-11-15. Branch `inference-smoke`. Store: `NoteStore(PastUserMemory(), Path("runs/smoke/notes"))`, driven through `take_notes` with a scripted model (`untraced = True`) that returns a fixed `NoteTakerOutput`. All note text is synthetic. HTTP calls were logged by wrapping `PastUserMemory._http.request` in the test snippet (method, path, status and time only; for DELETE and error responses, the first 300 characters of the body after checking the key is not in it). Raw logs: `runs/smoke/notes/run1.log`, `run2.log`, `run2.json`, `status1.log`.

Two runs:

- **Run 1** (person `smoke-b-749c`): the store exactly as shipped.
- **Run 2** (person `smoke-b-3b18`, other person `smoke-b-d761`): the same, except that the snippet swapped `mem.wait` for `PastClient.wait(..., timeout_s=300, poll_s=5)`. Without that change no note could be written (see defect 1), and steps b to e could not run. No repo file was changed.

## Results

| Step | Result | Evidence |
|---|---|---|
| a. Add two notes (`situation`, `what_helps`) | **FAIL as shipped**; PASS with the 300 s wait | Run 1: both `POST /ingest` returned 202, but `GET /ingest/{id}` still said `processing` when the 60 s wait ran out (30 polls each). `take_notes` returned `added=[]`, `not_written=2`. Both IDs were queued and then deleted (`DELETE` 202, `ids` named each one). Index, notes.json and pending all empty. A later `GET /ingest/{id}` showed `status=completed`, `sources.deleted=1`. Run 2: `added=['note-smoke-b-3b18-001','note-smoke-b-3b18-002']`, both on the index and in notes.json, pending empty. Ingestion took **81.0 s** and **54.2 s**. |
| b. Read back via `helper_notes` | PASS | `helper_notes(A)` returned both facts in index order straight away; they come from notes.json. A direct `recall` as the person, made 0.4 s after `take_notes` returned, already listed both source IDs, because the write wait had covered ingestion. Results carry `sourceId`, `label`, `metadata` (`type`, `kind`, `user_id`, `session_id`) and `excerpts`, so metadata does come back on recall. |
| c. Another person sees nothing | PASS | `helper_notes('smoke-b-d761')` = `[]`. `recall` as `smoke-b-d761` returned no source IDs. `recall` as `critic` returned only training records (`pattern-r*`, `research-r*`, `edit-r*`) and no note. |
| d. Delete one note | PASS | `take_notes` with a `delete` of `note-smoke-b-3b18-001` → `deleted=[…-001]`, `undeleted=[]`, finished in 0.9 s. The index was `[…-002]` straight away. `DELETE /data-points/note-smoke-b-3b18-001` → **202** `{"operationId":…,"ids":["note-smoke-b-3b18-001"]}`, confirmed on the **1st attempt**. `pending_deletes` = `[]`. The first recall poll (0.4 s later) no longer returned `-001`, only `-002`. `helper_notes` = the `what_helps` fact only. |
| e. Forget everything | PASS | `take_notes(forget_everything=True)` → `deleted=[…-002]`, `undeleted=[]`, `forgot_everything=True`, 0.6 s. Index `[]`, notes.json `{}`, pending `[]`. `DELETE /data-points/note-smoke-b-3b18-002` → 202, `ids` named it, 1st attempt. Recall as the person was empty on the first poll (0.2 s). `GET /audiences/person-smoke-b-3b18` reports `memoryCount: 0`. |

The two calls never run before both work as the code expects:

- `PUT /audiences/{slug}` answers **201** the first time and **200** after that, echoing `kind: fixed` and `identities: ["note-taker", "<user_id>"]`.
- `DELETE /data-points/{sourceId}` answers **202** with `{"operationId", "ids": [<sourceId>]}`, which is the shape `PastUserMemory.delete` checks for (`note_store.py:67-73`). All 4 deletes across both runs were confirmed on the first attempt.

## Timings

| What | Time |
|---|---|
| Ingestion, `POST /ingest` to `status=completed` (run 2) | 81.0 s and 54.2 s |
| Ingestion in run 1 | over 60 s for both notes (the wait gave up after about 63 s each) |
| `take_notes` with two adds | 133 s in run 1 (both failed), 167 s in run 2 (300 s wait) |
| `take_notes` with a delete / forget everything | 0.9 s / 0.6 s |
| Note visible to recall after the write was confirmed | already visible on the first poll (0.4 s) |
| Deleted note gone from recall | gone on the first poll (0.4 s; 0.2 s for forget-everything) |
| `PUT /audiences` | 87-460 ms |
| `DELETE /data-points` | 110-200 ms |
| `POST /recall`, warm | 230-450 ms |
| `POST /recall`, first one for a newly created audience | 7.7 s (run 1), **30 s ReadTimeout** (run 2, inside `live_notes`), plus another 30 s ReadTimeout during run 1's polling |

## HTTP call log, run 2

S = made by the store (`take_notes`, `helper_notes`); T = made directly by the test snippet (recall polls and checks).

| # | By | Method | Path | Status |
|---|---|---|---|---|
| 1 | S | PUT | /audiences/person-smoke-b-3b18 | 201 |
| 2 | S | POST | /recall | ReadTimeout (30 s), swallowed by `live_notes` |
| 3 | S | PUT | /audiences/person-smoke-b-3b18 | 200 |
| 4 | S | POST | /ingest | 202 |
| 5-20 | S | GET | /ingest/b5b157f4-b031-4b2c-86bc-e632e0613908 | 200 ×16 (15 × `processing`, then `completed`) |
| 21 | S | PUT | /audiences/person-smoke-b-3b18 | 200 |
| 22 | S | POST | /ingest | 202 |
| 23-33 | S | GET | /ingest/9d875994-8471-424c-8f0b-dc6318bd5559 | 200 ×11 (10 × `processing`, then `completed`) |
| 34 | S | PUT | /audiences/person-smoke-b-3b18 | 200 |
| 35 | S | PUT | /audiences/person-smoke-b-3b18 | 200 (`helper_notes(A)`) |
| 36 | S | POST | /recall | 200 |
| 37 | T | POST | /recall (as A) | 200 |
| 38 | S | PUT | /audiences/person-smoke-b-d761 | 201 (`helper_notes(B)` creates B's audience) |
| 39 | S | POST | /recall | 200 |
| 40 | T | POST | /recall (as B) | 200 |
| 41 | T | POST | /recall (as critic) | 200 |
| 42 | S | PUT | /audiences/person-smoke-b-3b18 | 200 |
| 43 | S | POST | /recall | 200 |
| 44 | S | PUT | /audiences/person-smoke-b-3b18 | 200 |
| 45 | S | DELETE | /data-points/note-smoke-b-3b18-001 | 202 |
| 46 | T | POST | /recall (as A) | 200 |
| 47 | S | PUT | /audiences/person-smoke-b-3b18 | 200 |
| 48 | S | POST | /recall | 200 |
| 49 | S | PUT | /audiences/person-smoke-b-3b18 | 200 |
| 50 | S | POST | /recall | 200 |
| 51 | S | PUT | /audiences/person-smoke-b-3b18 | 200 |
| 52 | S | DELETE | /data-points/note-smoke-b-3b18-002 | 202 |
| 53 | T | POST | /recall (as A) | 200 |

## HTTP call log, run 1 (store as shipped)

| # | Method | Path | Status |
|---|---|---|---|
| 1 | PUT | /audiences/person-smoke-b-749c | 201 |
| 2 | POST | /recall | 200 (7.7 s) |
| 3 | PUT | /audiences/person-smoke-b-749c | 200 |
| 4 | POST | /ingest | 202 |
| 5-34 | GET | /ingest/8182241c-c07b-4d3d-9076-4f41e4e51105 | 200 ×30, still not completed → wait returns False |
| 35 | PUT | /audiences/person-smoke-b-749c | 200 |
| 36 | POST | /ingest | 202 |
| 37-66 | GET | /ingest/86e12ad1-81c0-4ab5-961c-1d243fb73750 | 200 ×30, still not completed → wait returns False |
| 67 | PUT | /audiences/person-smoke-b-749c | 200 |
| 68 | DELETE | /data-points/note-smoke-b-749c-001 | 202, `ids` names it |
| 69 | DELETE | /data-points/note-smoke-b-749c-002 | 202, `ids` names it |
| 70 | PUT | /audiences/person-smoke-b-749c | 200 (`helper_notes`) |
| 71 | POST | /recall | 200 |
| 72-79 | POST | /recall (test polls) | 200 ×7, then ReadTimeout (30 s); the snippet stopped there |

## Defects found

1. **Every note write fails because the write wait is shorter than past.dev's ingestion (high: "remember me" never stores anything).**
   - Call: `GET /ingest/{ingestionId}`, polled by `PastClient.wait`.
   - Status: 200 every time, body `status: processing`, for longer than 60 s.
   - Expected: `status: completed` within the 60 s default. `NoteStore.write` calls `self.memory.wait(ingestion_id)` with no timeout (`puddles/inference/note_store.py:328`), so it uses `timeout_s=60` from `puddles/memory.py:82`.
   - Returned: `completed` after 54 s and 81 s (run 2), and after more than 63 s for both notes in run 1.
   - Effect: `write` raises `PastError` ("did not confirm … within the wait"), `take_notes` counts each note in `not_written`, and the store then deletes the note from past.dev. In run 1 both notes were lost. The fail-safe works as designed (nothing is left orphaned), but the feature does not.
   - Fix: in `write`, pass a longer timeout matched to measured ingestion (for example 180-300 s), or make the timeout a `NoteStore` setting. The ingestion status already reports `sources.current: 1` while still `processing`. Treating that as "stored" would be faster, but it is not the confirmation the spec asks for, so that is a design decision. A 3-minute wait inside the after-session step is also worth a look.

2. **The first recall on a new audience is slow or times out, and `live_notes` makes it on every read even though it ignores the result (medium: latency).**
   - Call: `POST /recall` as the person.
   - Result: 7.7 s in run 1 and a 30 s `httpx.ReadTimeout` in run 2, both the first recall after the audience was created, plus another 30 s ReadTimeout during run 1's polling. Later recalls took 0.2-0.45 s.
   - At fault: `note_store.py:290` calls `self.memory.recall(...)` and drops the result (the facts come from notes.json), and `puddles/memory.py:63` sets a 30 s client timeout. The exception is swallowed, so the output is correct, but `helper_notes` (and `build_input` in `take_notes`) can take 30 s or more.
   - Fix: make the recall in `live_notes` best-effort with a short timeout (a few seconds), or drop it, since nothing it returns is used.

3. **Reading a person's notes creates a past.dev audience, and forgetting never removes it (low: a leftover, no data).**
   - `helper_notes` for a person with no notes still calls `ensure_person`, which runs `PUT /audiences/person-<id>` and gets 201 (run 2, call 38). "Forget everything" deletes every data point but leaves the audience in place (`memoryCount: 0`).
   - This follows the spec as written (`note_store.py:287`, `:162`). It does mean every user ID ever looked up gets an audience on past.dev, and a forgotten person's audience, with their user ID in its name and identities, stays behind.
   - Left from this test: empty audiences `person-smoke-b-749c`, `person-smoke-b-3b18` and `person-smoke-b-d761`. They hold no data points, and the store has no call to remove them.

No defect in the two calls under test: `PUT /audiences/{slug}` and `DELETE /data-points/{sourceId}` behave as `PastUserMemory` expects. Deletes were confirmed on the first attempt and gone from recall within half a second.
