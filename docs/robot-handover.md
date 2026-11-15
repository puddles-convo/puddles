# Hand-over: the robot's brain on the checked text path

For the `puddles-stack` team. This page is everything needed to move the robot's brain off talking straight to a speech model and onto our checked text path: what changes, how to run the service, the three calls, and what to do with every answer. The contract behind it is [specs/turn-service.md](specs/turn-service.md); where this page and the contract differ, the contract is right and this page needs fixing.

Every response shown in sections 4 and 9 was produced by the service in fake mode on 15 November 2026, unless it is marked as the contract's example.

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

From the root of this repo, with its dependencies installed (`fastapi` and `uvicorn` are in `requirements.txt`).

**Fake mode** (scripted helper, checker and note-taker; no keys, no network):

```
PUDDLES_SERVICE_TOKEN=dev-token python -m scripts.serve --policy policies/policy-v1.md --root runs/serve --fake
```

**Real mode:**

```
PUDDLES_SERVICE_TOKEN=<a long random string> python -m scripts.serve --policy policies/policy-v2.md --root runs/serve
```

Options: `--notes-root DIR` (default: the `--root` value), `--host` (default `127.0.0.1`), `--port` (default `8765`), `--mlflow URI` (used only for sessions with training consent), `--sweep-every SECONDS` (default 30). It binds to `127.0.0.1` unless you pass `--host`; opening it beyond the machine is your decision.

Real mode needs these environment variables. Names and non-secret values are the ones the team runs with; the keys are yours to fill in.

| Variable | Value | For |
|---|---|---|
| `PUDDLES_SERVICE_TOKEN` | a long random string | the bearer token every request must carry; never an argument, never printed |
| `HELPER_URL` | `https://api.openai.com/v1` | the helper and the crisis checker |
| `HELPER_MODEL` | `gpt-6-luna` | the helper and the crisis checker |
| `HELPER_API_KEY` | your key | the helper and the crisis checker |
| `HELPER_MAX_TOKENS_PARAM` | `max_completion_tokens` | the helper and the crisis checker |
| `HELPER_EXTRA_BODY` | `'{"temperature": 1, "reasoning_effort": "none"}'` | the helper and the crisis checker |
| `OPENAI_API_KEY` | your key | the note-taker |
| `PUDDLES_AGENT_MODEL` | `gpt-6-luna` | the note-taker |
| `PUDDLES_AGENT_TOOL_REASONING_EFFORT` | `none` | the note-taker (optional) |
| `OPENAI_BASE_URL` | optional | the note-taker without training consent; default OpenAI |
| `PAST_API_KEY` | your key | past.dev, where "remember me" notes live |
| `PAST_BASE_URL` | optional, default `https://api.past.dev/api/v1` | past.dev |

At startup one of each client is built, so a missing variable stops it before anything listens: one line on standard error starting `error: ` and exit code 2. Without the token, for example:

```
error: PUDDLES_SERVICE_TOKEN is not set
```

### Two ways to use it

**As its own process:** run the command above and call `http://127.0.0.1:8765`.

**Mounted inside your FastAPI app:** with this repo installed or on the path, `PUDDLES_SERVICE_TOKEN` and the variables above in the environment (drop `fake=True` for real mode):

```python
import argparse, os
from puddles.inference.service import create_app
from scripts.serve import build_config

args = argparse.Namespace(policy="policies/policy-v2.md", root="runs/serve", notes_root=None, mlflow=None, fake=False)
turns = create_app(build_config(args, os.environ))
app.mount("/puddles", turns)
```

The routes are then under `/puddles`, for example `/puddles/sessions`, and the token is still required. Mounted, nothing sweeps idle sessions for you: call `turns.state.service.sweep()` every 30 seconds or so from a background thread. `scripts.serve` does that on its own.

## 4. The three calls

Every call but `GET /health` needs `Authorization: Bearer <token>`.

```
T=dev-token
B=http://127.0.0.1:8765
```

### Start a session

`user_id`, `remember_me` and `consent_for_training` are required, and both consent answers must be JSON booleans (`true` or `false`). Ask the person both questions at the start of every session and send what they said. Never assume an answer, and never fill in a default. `session_id` is optional; leave it out and the service makes one.

```
curl -s -X POST $B/sessions -H "Authorization: Bearer $T" -H "Content-Type: application/json" \
  -d '{"user_id": "user-01", "remember_me": false, "consent_for_training": false, "session_id": "sess-0007"}'
```

`201`:

```json
{"session_id": "sess-0007", "policy_version": "policy-v1", "notes_loaded": 0, "notes_failed": false}
```

### One turn

`text` is required: 1 to 2000 characters once leading and trailing whitespace is removed.

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

```python
import httpx

OFFLINE = ("I can't connect right now, so I can't listen properly. If you are in danger, please call 911. "
           "If you are thinking about hurting yourself, call or text 988.")

brain = httpx.Client(base_url="http://127.0.0.1:8765", headers={"Authorization": "Bearer dev-token"},
                     timeout=httpx.Timeout(35.0))

def say(text: str) -> None:
    print("robot:", text)  # your text-to-speech goes here

r = brain.post("/sessions", json={"user_id": "user-01", "remember_me": False, "consent_for_training": False})
sid = r.json()["session_id"]
for heard in ["Work has been really stressful lately.", "Some days I just want to die."]:
    try:
        r = brain.post(f"/sessions/{sid}/turns", json={"text": heard})
    except httpx.TransportError:
        say(OFFLINE); continue
    body = r.json()
    say(body["text"] if r.status_code == 200 else body.get("say", OFFLINE))
print(brain.post(f"/sessions/{sid}/end", timeout=None).json())  # in your brain: in the background
```

## 5. What to do with each turn's response

- **Speak `text`, exactly as given.** Do not shorten, rephrase, translate or add to it.
- **`fixed_handoff_response` true:** `text` is a fixed line from the crisis path, not the helper's words: one of the three crisis responses, the holding line or the offline line. Speak all of it, do not let the person's next words cut it off, and append nothing (no filler, no follow-up from another model).
- **A crisis response** is `fixed_handoff_response` true with `keyword_matched` true, or with `risk` `"crisis"`. Do not rely on `risk` alone: a keyword match forces the crisis response even when the checker said `none`, and the response then still says `"risk": "none"` (see section 9).
- **`risk` and `category`** may drive the face or the lights. They never change the words.
- **`check_status` not `"answered"`** (`late`, `failed` and so on): the checker gave no usable answer, so the service gave the holding line ("Sorry, I lost my thread for a moment. Could you say that again?"), or the offline line on a second failed check in a row. `risk` and `category` are then `null`; treat that as unknown, never as calm.
- **`helper_failed` true:** the helper did not produce a draft; the text is still safe to speak.

## 6. Every error and what the robot should do

Errors are JSON with an `error` code; none repeats what was sent.

| Status | `error` | Means | What the robot does |
|---|---|---|---|
| 401 | `unauthorized` | token missing, wrong or malformed | Do not retry. Speak the offline line, and alert: the brain's configuration is wrong. |
| 404 | `not_found` | not one of the routes | A bug in the brain's URL. Speak the offline line, log it. |
| 404 | `session_not_found` | the session never existed, was ended, or was swept after 15 idle minutes | Start a new session (ask the two consent questions again), then send the text again. |
| 409 | `session_exists` | that `session_id` is live, or (with training consent) already on disk | Start again without `session_id`, so the service makes one. |
| 409 | `user_busy` | this person already has a live session, or the last one is still ending | Wait for that session's end call to return, then start. If you lost track of it, wait and retry; the sweeper ends it after 15 idle minutes. |
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

- **Turn timeout: at least 35 s.** The service waits up to 30 s for the helper, counted from the start of the turn, and the crisis checker has 5 s. Normal turns take 1 to 2 s: the median measured on `gpt-6-luna` is about 1.4 s per turn. (This checkout's `safety/crisis-path-v1.json` still says `"timeout_seconds": 3`; the change to 5 s is in pull request #74. At 3 s one ordinary turn in eleven lost its reply to the holding line.)
- **Do not block on the end call.** Ending runs the note-taker, and past.dev took about a minute (54 s and 81 s measured) to confirm each note, so an end can take minutes. Call it in the background with no short timeout, and let the robot carry on.
- **Next session for the same person only after the end returns.** Until then a new start for that `user_id` gets `409 user_busy`. A different person can start at once.
- **Idle sessions are ended after 15 minutes** without activity, notes and all. A turn after that gets `404 session_not_found`.

## 8. Consent and logs

| Remember me | Use for training | What is kept after the session |
|---|---|---|
| No | No | Nothing new. Notes the person asked to forget are gone. |
| Yes | No | Notes only: `notes/<user_id>/`. No session file, no record, no trace. |
| No | Yes | The session file, the record and the traces. No new notes. |
| Yes | Yes | Both |

Forgetting always works: if the person asks the robot to forget something, it is deleted whatever they answered.

**Your brain keeps logs too.** With training consent off, nothing the person said may be kept anywhere, so the brain must not log the person's text, the transcript, the reply text or the audio either: not in Modal logs, not in error reports, not in traces. Log session IDs, statuses, counts and timings only. The service itself never logs text, so the gap would only be on your side.

## 9. A two-minute check before the robot is plugged in

Terminal 1, from this repo's root:

```
PUDDLES_SERVICE_TOKEN=dev-token python -m scripts.serve --policy policies/policy-v1.md --root /tmp/puddles-check --fake
```

Terminal 2, once `curl -s $B/health` answers (startup takes about 10 s):

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

The second turn holds `want to die`, a phrase from `safety/keywords-v1.txt`: it gets a fixed response with 988, `fixed_handoff_response` true and `keyword_matched` true. The fake checker always answers `none`, which is why `risk` says `none` there; the keyword alone forced the crisis response. `helper` `-1` means the helper's draft was not waited for. Timings are near zero because fake mode calls no model.

Then point your brain at it and run the Python client from section 4: it should print the same two replies and the end result. Stop the service with Ctrl-C.

## 10. Known limits today

- **One process only.** Sessions live in that process's memory, and the `user_busy` and `session_exists` checks are per process. Run exactly one copy; do not let it scale to several containers or workers.
- **Sessions alive when the process stops are dropped without notes.** The note-taker does not run for them, and a forget request made in such a session is lost. End sessions before a restart or deploy.
- **The person ID is whatever you send.** The service does not know who the person is; it trusts `user_id`. It must be lowercase letters, digits and hyphens, starting with a letter or digit, at most 33 characters (a few reserved names such as `project` are refused), and the same person must get the same ID every time, or their notes will not follow them.
- Rate limits and one token per device are not built; there is one shared token.
- The fixed responses are drafts with no clinical review.
