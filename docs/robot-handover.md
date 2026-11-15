# Hand-over: the robot's brain on the checked text path

For the `puddles-stack` team. This page is everything needed to move the robot's brain off talking straight to a speech model and onto our checked text path: what changes, how to run the service, the three calls, and what to do with every answer. The contract behind it is [specs/turn-service.md](specs/turn-service.md); where this page and the contract differ, the contract is right and this page needs fixing.

Every response shown in sections 4 and 9 was produced by the service in fake mode on 15 November 2026, unless it is marked as the contract's example.

**Real mode works on the `integration` branch as of 15 November 2026**, with pull request #74 merged: the crisis checker's deadline is 5 s, the note store waits up to 300 s for past.dev to confirm a note, and the note-taker runs with and without training consent. A real run over HTTP that morning followed this page; its report is [results/http-replay/README.md](../results/http-replay/README.md).

## 1. The swap in five lines

1. Speech-to-text turns what the person said into text.
2. The brain POSTs that text to `/sessions/{session_id}/turns`.
3. The service runs the keyword check, the crisis checker, the helper and the release step, and returns one `text`.
4. The brain speaks exactly that `text` with text-to-speech.
5. The speech model no longer writes the reply. Nothing else decides what the robot says.

## 2. What you must have on your side

**Requirement:** speech-to-text and text-to-speech as two separate steps, with our service in between.

**Risk:** a speech-to-speech session that generates its own reply bypasses the policy, the crisis checker and the release step. A person who says they want to die would get whatever the speech model improvises, not the fixed response with 988. That is the thing this swap exists to stop.

If your Realtime session can run with transcription on and its own response generation off, and can speak a text you give it, that is the shape needed. We cannot see your API, so we do not claim how it does that; check it, and if it cannot, use a separate speech-to-text and text-to-speech step instead.

## 3. Start the service

From the root of this repo (the working directory matters; see `--mlflow` below), with the project's Python environment: the one with this repo's dependencies installed by `pip install -r requirements.txt`, which now includes `fastapi`, `uvicorn` and MLflow. Every `python` on this page means that interpreter. With a Python that lacks one of them, `python -m scripts.serve` dies at import with a traceback ending in `ModuleNotFoundError` (for example `No module named 'mlflow'`), not with the `error:` line described below.

**Fake mode** (scripted helper, checker and note-taker; no keys, no network):

```
PUDDLES_SERVICE_TOKEN=dev-token python -m scripts.serve --policy policies/policy-v1.md --root runs/serve --fake
```

**Real mode:**

Set the token first, without it landing in your shell history (paste it at the silent prompt), then start:

```
read -rs PUDDLES_SERVICE_TOKEN && export PUDDLES_SERVICE_TOKEN
python -m scripts.serve --policy policies/policy-v2.md --root runs/serve
```

The brain needs the same token, from its own secret store.

Options: `--notes-root DIR` (default: the `--root` value), `--host` (default `127.0.0.1`), `--port` (default `8765`), `--mlflow URI` (used only for sessions with training consent; see below), `--sweep-every SECONDS` (default 30). It binds to `127.0.0.1` unless you pass `--host`; opening it beyond the machine is your decision.

**Where training-consent traces go.** For sessions with training consent the service records MLflow traces, and those hold the person's words. Without `--mlflow` (and without `MLFLOW_TRACKING_URI`), MLflow writes `runs/mlflow.db` and `mlruns/` in the service's working directory, outside `--root`. Pass `--mlflow sqlite:///<root>/mlflow.db` (for example `--mlflow sqlite:///runs/serve/mlflow.db`) so the store lands under the root. That moves only the tracking metadata: the trace artifacts still land in `./mlruns` of the working directory (a known open defect). Protect both places, and when a person's data must go, delete their records and artifacts from both; do not delete the working directory itself.

Real mode reads these environment variables. Names and non-secret values are the ones the team runs with; the keys are yours to fill in.

**Required** (the service will not start without them):

| Variable | Value | For |
|---|---|---|
| `PUDDLES_SERVICE_TOKEN` | a long random string | the bearer token every request must carry; never an argument, never printed |
| `HELPER_URL` | `https://api.openai.com/v1` | the helper and the crisis checker |
| `HELPER_MODEL` | `gpt-6-luna` | the helper and the crisis checker |
| `OPENAI_API_KEY` | your key | the note-taker |
| `PUDDLES_AGENT_MODEL` | `gpt-6-luna` | the note-taker |
| `PAST_API_KEY` | your key | past.dev, where "remember me" notes live |

**Provider settings** (optional to the code, but needed for OpenAI and `gpt-6-luna`):

| Variable | Value | For |
|---|---|---|
| `HELPER_API_KEY` | your key | the helper and the crisis checker |
| `HELPER_MAX_TOKENS_PARAM` | `max_completion_tokens` | the helper and the crisis checker |
| `HELPER_EXTRA_BODY` | `'{"temperature": 1, "reasoning_effort": "none"}'` | the helper and the crisis checker |

**Optional:**

| Variable | Default | For |
|---|---|---|
| `OPENAI_BASE_URL` | OpenAI | the note-taker, with and without training consent |
| `PAST_BASE_URL` | `https://api.past.dev/api/v1` | past.dev |
| `PUDDLES_AGENT_MAX_TOKENS_PARAM` | `max_completion_tokens` | the output-limit field of the note-taker for sessions without training consent only; set `max_tokens` for servers that want the older name; empty is the same as unset; the note-taker for sessions with training consent always uses `max_completion_tokens` |

`PUDDLES_AGENT_TOOL_REASONING_EFFORT`, which the training scripts set, applies only to requests with tools and does not affect the note-taker.

At startup one of each client is built, so a missing variable stops it before anything listens: one line on standard error starting `error: ` and exit code 2. Without the token, for example:

```
error: PUDDLES_SERVICE_TOKEN is not set
```

### Two ways to use it

**As its own process:** run the command above and call `http://127.0.0.1:8765`.

**Mounted inside your FastAPI app:** with this repo installed or on the path, `PUDDLES_SERVICE_TOKEN` and the variables above in the environment. The snippet is real mode; change `fake=False` to `fake=True` for fake mode:

```python
import argparse, os
from puddles.inference.service import create_app
from scripts.serve import build_config

args = argparse.Namespace(policy="policies/policy-v2.md", root="runs/serve", notes_root=None, mlflow=None, fake=False)
turns = create_app(build_config(args, os.environ))
app.mount("/puddles", turns)
```

The routes are then under `/puddles`, for example `/puddles/sessions`, and the token is still required. Mounted, nothing sweeps idle sessions for you: call `turns.state.service.sweep()` every 30 seconds or so from a background thread. `scripts.serve` does that on its own. The trace locations above apply mounted too: pass `mlflow="sqlite:///runs/serve/mlflow.db"` in the `Namespace` (or set `MLFLOW_TRACKING_URI`), and treat the host app's working directory, where `mlruns/` lands, as holding personal data.

## 4. The three calls

Every call but `GET /health` needs `Authorization: Bearer <token>`.

```
T=dev-token
B=http://127.0.0.1:8765
```

### Start a session

`user_id`, `remember_me` and `consent_for_training` are required, and both consent answers must be JSON booleans (`true` or `false`). Ask the person both questions at the start of every session and send what they said. Never assume an answer, and never fill in a default. `session_id` is optional; leave it out or send `null` and the service makes one. If you send it, it must be a string of letters, digits, `_` and `-`, starting with a letter or digit, at most 64 characters. Send only these four fields; any other field is refused with `422`.

```
curl -s -X POST $B/sessions -H "Authorization: Bearer $T" -H "Content-Type: application/json" \
  -d '{"user_id": "user-01", "remember_me": false, "consent_for_training": false, "session_id": "sess-0007"}'
```

`201`:

```json
{"session_id": "sess-0007", "policy_version": "policy-v1", "notes_loaded": 0, "notes_failed": false}
```

### One turn

`text` is required and must be a JSON string: 1 to 2000 characters once leading and trailing whitespace is removed. Send no other field; any other field is refused with `422`.

```
curl -s -X POST $B/sessions/sess-0007/turns -H "Authorization: Bearer $T" -H "Content-Type: application/json" \
  -d '{"text": "Work has been really stressful lately."}'
```

`200`, the contract's example (real mode):

```json
{"session_id": "sess-0007", "turn": 1, "text": "That sounds heavy. What is worrying you most?",
 "fixed_handoff_response": false, "policy_version": "policy-v1",
 "check_status": "answered", "risk": "none", "category": null,
 "keyword_matched": false, "helper_failed": false,
 "timings_ms": {"keyword": 0, "checker": 412, "helper": 655, "total": 655}}
```

### End the session

No body.

```
curl -s -X POST $B/sessions/sess-0007/end -H "Authorization: Bearer $T"
```

`200`:

```json
{"session_id": "sess-0007", "kept": false, "record": null,
 "notes": {"added": 0, "replaced": 0, "confirmed": 0, "deleted": 0, "undeleted": [], "failed": false}}
```

`record` is the session record when training consent was given and there was at least one turn, otherwise `null`; `kept` is true exactly when `record` is not `null`. `notes` holds counts and note IDs only, never text.

### The same in Python

Every call handles a lost connection, a timeout, an unexpected status and an answer that is not JSON; a turn with no usable answer speaks the stored offline line. The end call runs on its own thread, a normal (not daemon) thread kept in a table with one entry per person: the next session for that person waits for it, and `finish_all()` joins every one before the process exits or the service shuts down, so no end request is cut off. `speak` is yours: it sends the text to text-to-speech and must not print, log or trace it.

```python
import logging
import threading

import httpx

log = logging.getLogger("brain.puddles")  # IDs, statuses and flags only; never the person's text or the reply
OFFLINE = ("I can't connect right now, so I can't listen properly. If you are in danger, please call 911. "
           "If you are thinking about hurting yourself, call or text 988.")  # stored on the robot
brain = httpx.Client(base_url="http://127.0.0.1:8765", headers={"Authorization": "Bearer dev-token"})
ending: dict[str, threading.Thread] = {}  # user_id -> its end call, still running (one per person)


def speak(text: str) -> None:
    """Yours: send text to text-to-speech. Do not print, log or trace it."""
    raise NotImplementedError


def call(path: str, body: dict | None = None, timeout: float | None = 35.0) -> tuple[int | None, dict]:
    """(status, JSON object), or (None, {}) when there is no usable answer: no connection, timeout, not JSON."""
    try:
        r = brain.post(path, json=body, timeout=timeout)
        data = r.json()
    except (httpx.HTTPError, ValueError):
        return None, {}
    return (r.status_code, data) if isinstance(data, dict) else (None, {})


def start(user_id: str, remember_me: bool, consent_for_training: bool) -> str | None:
    previous = ending.get(user_id)
    if previous is not None:
        previous.join()  # this person's last session is still ending: wait for it (409 user_busy otherwise)
    status, data = call("/sessions", {"user_id": user_id, "remember_me": remember_me,
                                      "consent_for_training": consent_for_training})
    log.info("start: %s %s", status, data.get("error"))
    if status == 201 and isinstance(data.get("session_id"), str):
        return data["session_id"]
    speak(OFFLINE)
    return None


def turn(session_id: str, heard: str) -> dict:
    status, data = call(f"/sessions/{session_id}/turns", {"text": heard})
    log.info("turn %s: %s %s fixed=%s", session_id, status, data.get("error"), data.get("fixed_handoff_response"))
    if status == 200 and isinstance(data.get("text"), str):
        speak(data["text"])  # always, exactly as returned
    elif status == 500 and isinstance(data.get("say"), str):
        speak(data["say"])  # turn_failed: the offline line; the session is still live
    else:
        speak(OFFLINE)  # see the error table for what to do next
    return data


def end_in_background(user_id: str, session_id: str) -> None:
    def run() -> None:
        status, data = call(f"/sessions/{session_id}/end", timeout=None)  # can take minutes
        notes = data.get("notes") or {}
        log.info("end %s: %s kept=%s notes_failed=%s undeleted=%d", session_id, status, data.get("kept"),
                 notes.get("failed"), len(notes.get("undeleted") or []))
        ending.pop(user_id, None)

    thread = threading.Thread(target=run, name=f"end-{session_id}")  # not a daemon: the end must complete
    ending[user_id] = thread
    thread.start()


def finish_all() -> None:
    """Call before the process exits or the service shuts down, so every end request is sent and completed."""
    for thread in list(ending.values()):
        thread.join()


sid = start("user-01", remember_me=False, consent_for_training=False)  # the person's two answers, asked
if sid:
    for heard in ["Work has been really stressful lately.", "Some days I just want to die."]:
        turn(sid, heard)
    end_in_background("user-01", sid)
finish_all()  # in your brain: on shutdown, not after every session
```

What it did against the fake-mode app, with `speak` recording which line it was given, and more calls added after the page's own lines to try each failure. The log lines are the client's own and hold no text; session IDs differ from run to run:

```
== the page's own lines: start, two turns, end in the background, finish_all
log: start: 201 None
log: turn sess-20261115141857-9d44a4a5: 200 None fixed=False
speak: the helper's reply
log: turn sess-20261115141857-9d44a4a5: 200 None fixed=True
speak: the unsure crisis response
log: end sess-20261115141857-9d44a4a5: 200 kept=False notes_failed=False undeleted=0
ends still running after finish_all: [] | live sessions: []
== a session ended in the background, then the same person again at once: start waits for that end
log: start: 201 None
log: turn sess-20261115141857-e3452ba1: 200 None fixed=False
speak: the helper's reply
log: end sess-20261115141857-e3452ba1: 200 kept=False notes_failed=False undeleted=0
log: start: 201 None
second session: True | end finished first: True
== a turn after the end of the first session (404)
log: turn sess-20261115141857-9d44a4a5: 404 session_not_found fixed=None
speak: the offline line
== turn_failed (run_turn made to raise)
log: turn sess-20261115141857-9fb600cc: 500 turn_failed fixed=None
speak: the offline line
log: end sess-20261115141857-9fb600cc: 200 kept=False notes_failed=False undeleted=0
== no connection (nothing listening)
log: start: None None
speak: the offline line
start returned: None
== an answer that is not JSON (a proxy's error page, status 502)
log: turn sess-0007: None None fixed=None
speak: the offline line
== wrong token (401)
log: start: 401 unauthorized
speak: the offline line
start returned: None
```

The client speaks the offline line for every turn it cannot use; what to do next (start a new session, alert, retry) is in the error table in section 6.

## 5. What to do with each turn's response

- **Always speak the returned `text`, exactly as given.** Do not shorten, rephrase, translate or add to it. The service has already chosen the safe line; the brain never chooses between lines itself.
- **A keyword match forces the crisis response even when the check failed.** The keyword check runs first and wins over everything the checker did or did not say.
- **`risk` and `category` are `null` when the check did not answer** (`check_status` is `late`, `failed` or anything but `"answered"`). Treat that as unknown, never as calm. On a crisis response forced by a keyword, `risk` can read `"none"` while `category` is `null` (section 9 shows one); `category` is never `"none"`.
- **So the only reliable sign of a fixed line is `fixed_handoff_response` true.** A fixed line is one of the three crisis responses, the holding line or the offline line, never the helper's words. Speak all of it, do not let the person's next words cut it off, and append nothing (no filler, no follow-up from another model).
- **The only reliable sign of a crisis response is `fixed_handoff_response` true together with `keyword_matched` true or `risk` equal to `"crisis"`.** Use that, not `risk` alone, for anything that must react to a crisis.
- **When the check did not answer and no keyword matched**, the text is the holding line ("Sorry, I lost my thread for a moment. Could you say that again?") on the first failed check in a row, and the offline line on the second and later ones.
- **`risk` and `category`** may drive the face or the lights. They never change the words.
- **`helper_failed` true** means the helper failed or had no usable draft in time; the text is still safe to speak. In `timings_ms`, a `helper` of `-1` means no timely helper result was recorded when the turn closed; it does not say whether the turn waited.

## 6. Every error and what the robot should do

Errors are JSON with an `error` code; none repeats what was sent.

| Status | `error` | Means | What the robot does |
|---|---|---|---|
| 401 | `unauthorized` | token missing, wrong or malformed | Do not retry. Speak the offline line, and alert: the brain's configuration is wrong. |
| 404 | `not_found` | not one of the routes | A bug in the brain's URL. Speak the offline line, log it. |
| 404 | `session_not_found` | the session never existed, was ended, or was ended by the idle sweep | Start a new session (ask the two consent questions again), then send the text again. |
| 405 | `method_not_allowed` | the HTTP method is wrong for that route | A bug in the brain's request; fix it. Speak the offline line if something must be said. |
| 409 | `session_exists` | that `session_id` is live, or (with training consent) already on disk | Start again without `session_id`, so the service makes one. |
| 409 | `user_busy` | this person already has a live session, or the last one is still ending | Wait for that session's end call to return, then start. If you lost track of it, wait and retry; an idle session is ended by a later sweep, and one that is already ending holds the person until its end finishes, which has no fixed limit. |
| 422 | `invalid_request` | the body is wrong; `fields` names the fields at fault | A brain bug, except for empty text (for example speech-to-text heard nothing): say nothing and keep listening. Never fix a consent field by guessing. |
| 429 | `too_many_sessions` | 64 sessions are live | Speak the offline line; retry the start shortly. |
| 503 | `policy_unavailable` | the policy file cannot be read | Speak the offline line; alert. No session was made. |
| 503 | `model_unavailable` | the models could not be set up | Speak the offline line; retry the start shortly. |
| 500 | `turn_failed` | the turn itself crashed; the body has `say` | Speak `say` (it is the offline line). The session is still live; carry on. |
| 500 | `internal` | anything else | Speak the offline line; log it. |

**No connection at all** (connection refused, DNS failure, a turn with no answer within the timeout, or any status above that leaves you without a line to speak): play the offline line stored on the robot. It is in `safety/crisis-path-v1.json` as `offline_line`:

> I can't connect right now, so I can't listen properly. If you are in danger, please call 911. If you are thinking about hurting yourself, call or text 988.

Store it on the device, because with no connection nothing can be fetched.

## 7. Timing

- **Turn timeout: at least 35 s.** The service waits up to 30 s for the helper, counted from the start of the turn. The crisis checker's deadline is 5 s. Normal turns take 1 to 2 s on `gpt-6-luna` (section 8a has today's measurements).
- **Do not block on the end call.** Ending runs the note-taker, and the note store waits up to 300 s for past.dev to confirm each note. In this morning's real runs the end calls took 50.3 s for a session that wrote four notes, 3.6 s and 2.3 s for two others, 1.4 to 3.3 s for the pilot sessions, and 64 ms for a session with no turns; allow longer. Run the end call in the background with no short timeout, as the client in section 4 does.
- **Next session for the same person only after the end returns.** Until then a new start for that `user_id` gets `409 user_busy`. A different person can start at once.
- **Idle sessions** become eligible to be ended after 900 s (15 minutes) without activity, and are ended, notes and all, when the next sweep reaches them (`scripts.serve` sweeps every 30 s; endings are run one after another, so a sweep can take a while). A session with a turn running is not idle. A turn after the end gets `404 session_not_found`. A session that is already ending holds its person until the end finishes; there is no guaranteed release at 15 minutes.

## 8. Consent and logs

| Remember me | Use for training | What is kept after the session |
|---|---|---|
| No | No | Nothing new. A request to forget is acted on, but the end response cannot confirm a specific note went (below). |
| Yes | No | Notes only: `notes/<user_id>/`. No session file, no record, no trace. |
| No | Yes | The session file, the record and the traces. No new notes. |
| Yes | Yes | Both |

**Forgetting is attempted whatever the person answered, but it can fail.** The note-taker runs at the end of every session with at least one turn, so a request to forget is acted on even with "remember me" off. But the notes step can fail before it changes anything (`notes.failed` true in the end response), and past.dev can be slow to confirm a deletion. A deletion past.dev has not confirmed stays pending, with its note ID in `notes.undeleted`, and is retried by later note-store operations for that person; there is no separate background retry. Even `failed` false and an empty `undeleted` do not prove that the particular thing the person asked to forget was deleted, because the response gives counts, not which notes: `deleted` above zero means some note was removed, not which one. So the robot must not confirm that a specific fact was forgotten; the most it can say is that it has passed the request on. Forgetting notes does not remove a session file, record or trace kept with training consent.

**Your brain keeps logs too.** With training consent off, nothing the person said may be kept anywhere, so the brain must not log the person's text, the transcript, the reply text or the audio either: not in Modal logs, not in error reports, not in traces. Log session IDs, statuses, counts and timings only. The service's own log lines hold no text, with one exception: with training consent on, MLflow autologging can print a warning to the service's standard error that contains a fragment of a note. So treat the service's standard error as holding personal data for consented sessions, and keep and delete it accordingly.

## 8a. Things the robot will do that may surprise you

Each of these was seen in saved real runs on 15 November 2026 or follows from the current code, and is how today's build behaves. The denial example was seen in the persona pilot that morning (`results/persona-pilot/` on the `persona-pilot` branch; the laid-off persona with notes on, session 2, turn 6): "No. I'm not thinking about ending my life." got the fixed response because the keyword `ending my life` matched while the checker said `none`. None comes with a promise of a fix.

- **A crisis disclosure can follow the person into the next session.** With "remember me" on, after a session in which the person disclosed a crisis, the note-taker may keep a note of it ("They disclosed suicidal thoughts and a plan for this weekend." in the real run). The helper may raise it at the start of the next session, and the checker may then send the person's following turns down the crisis path too, so they get the fixed crisis response even for ordinary messages, including a request to forget something.
- **"No, I'm not thinking about ending my life" gets the fixed crisis response.** The keyword check matches the phrase, and a keyword match forces the crisis response whatever the checker says.
- **Replies can end abruptly.** The helper's draft asks for at most 80 tokens (the default in `loop/helper.py`) and is trimmed to whole sentences; if it holds no complete sentence, the incomplete reply is kept as it is. Fixed responses go through neither step.
- **Latency per turn, measured over HTTP:** client wall time had a median of 1203.5 ms and a maximum of 1748 ms. Wall time minus the service's own `total` had a median of 2 ms, but that difference does not isolate the HTTP overhead.

## 9. A two-minute check before the robot is plugged in

Terminal 1, from this repo's root:

```
PUDDLES_SERVICE_TOKEN=dev-token python -m scripts.serve --policy policies/policy-v1.md --root /tmp/puddles-check --fake
```

Terminal 2, once `curl -s $B/health` answers (startup takes about 1 s):

```
T=dev-token
B=http://127.0.0.1:8765
curl -s -w '\n' $B/health
curl -s -w '\n' -X POST $B/sessions -H "Content-Type: application/json" -d '{}'
curl -s -w '\n' -X POST $B/sessions -H "Authorization: Bearer $T" -H "Content-Type: application/json" \
  -d '{"user_id": "user-01", "remember_me": false, "consent_for_training": false, "session_id": "sess-0007"}'
curl -s -w '\n' -X POST $B/sessions/sess-0007/turns -H "Authorization: Bearer $T" -H "Content-Type: application/json" \
  -d '{"text": "Work has been really stressful lately."}'
curl -s -w '\n' -X POST $B/sessions/sess-0007/turns -H "Authorization: Bearer $T" -H "Content-Type: application/json" \
  -d '{"text": "Some days I just want to die."}'
curl -s -w '\n' -X POST $B/sessions/sess-0007/end -H "Authorization: Bearer $T"
curl -s -w '\n' -X POST $B/sessions/sess-0007/turns -H "Authorization: Bearer $T" -H "Content-Type: application/json" \
  -d '{"text": "hello"}'
```

What you should see, one line per call (the timings can differ by a few milliseconds):

```
{"status":"ok"}
{"error":"unauthorized"}
{"session_id":"sess-0007","policy_version":"policy-v1","notes_loaded":0,"notes_failed":false}
{"session_id":"sess-0007","turn":1,"text":"That sounds like a lot to carry. What has been on your mind most?","fixed_handoff_response":false,"policy_version":"policy-v1","check_status":"answered","risk":"none","category":null,"keyword_matched":false,"helper_failed":false,"timings_ms":{"keyword":0,"checker":0,"helper":0,"total":1}}
{"session_id":"sess-0007","turn":2,"text":"I want to make sure you're safe. I'm a robot, so I can't send help myself. If anyone is in danger right now, please call 911. If you're thinking about hurting yourself, you can call or text 988 at any time. Can you tell me a bit more about what's happening?","fixed_handoff_response":true,"policy_version":"policy-v1","check_status":"answered","risk":"none","category":null,"keyword_matched":true,"helper_failed":false,"timings_ms":{"keyword":0,"checker":0,"helper":-1,"total":0}}
{"session_id":"sess-0007","kept":false,"record":null,"notes":{"added":0,"replaced":0,"confirmed":0,"deleted":0,"undeleted":[],"failed":false}}
{"error":"session_not_found"}
```

The second turn holds `want to die`, a phrase from `safety/keywords-v1.txt`: it gets a fixed response with 988, `fixed_handoff_response` true and `keyword_matched` true. For these two messages, which carry no checker marker, the fake checker answers `none`, which is why `risk` says `none` there; the keyword alone forced the crisis response. (The fake checker gives other outcomes when a message holds a marker such as `[checker:concern]`, `[checker:crisis-self]` or `[checker:fail]`.) `helper` `-1` means no timely helper result was recorded; a crisis response does not wait for the helper. Timings are near zero because fake mode calls no model.

Then run the Python client from section 4 against it with your `speak`: it should speak the helper's reply, then the crisis response, and log `end ...: 200`. Stop the service with Ctrl-C.

## 10. Known limits today

- **One process only.** Sessions live in that process's memory, and the `user_busy` and `session_exists` checks are per process. Run exactly one copy; do not let it scale to several containers or workers.
- **Sessions alive when the process stops are dropped without notes.** The note-taker does not run for them, and a forget request made in such a session is lost. End sessions before a restart or deploy.
- **The person ID is whatever you send.** The service does not know who the person is; it trusts `user_id`. It must be lowercase letters, digits and hyphens, starting with a letter or digit, at most 33 characters (a few reserved names such as `project` are refused), and the same person must get the same ID every time, or their notes will not follow them.
- Rate limits and one token per device are not built; there is one shared token.
- The fixed responses are drafts with no clinical review.
