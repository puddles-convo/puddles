# Inference smoke test with real models (2026-11-15, branch `inference-smoke`)

Checked against the raw evidence by a second model on 15 November; measurements confirmed, wording corrected.

This report covers run 2, which used the team's training settings (`with_env_train.sh`). Run 1, against the wrong endpoint, is summarised at the end.

**Bottom line:** the helper and the crisis checker answer on OpenAI `gpt-6-luna`. Stage 4's exit test passes:
- a 10-turn everyday conversation, then the crisis message, which got the fixed `self` response with 988;
- with consent on, the session file and the record round-trip;
- with consent off, no session directory, no record and no matching MLflow content were found; the only copies of the phrase are the test's own input file and the captured terminal output, where the robot's reply quoted the person. No trace was written.

Three things are still wrong:
- **D1**, still present: with consent off, the note-taker (`PlainLLM`) fails with HTTP 400. A fix is being built separately.
- **D5**, new: one ordinary turn lost its reply because the checker took longer than the 3 s deadline (3046 ms).
- **D3**, still present: trace files go to the repo-root `./mlruns/`.

## Settings used

Every real command ran under `sh <env-wrapper>` (it loads the env file, overrides settings and unsets PYTHONPATH). No value was printed.

| Role | Model | URL host | Other settings (names only) |
|---|---|---|---|
| helper + crisis checker (`HELPER_*`) | `gpt-6-luna` | `api.openai.com` | `HELPER_MAX_TOKENS_PARAM=max_completion_tokens`; `HELPER_EXTRA_BODY` sets `temperature`, `reasoning_effort` |
| note-taker | `gpt-6-luna` (`PUDDLES_AGENT_MODEL`) | `api.openai.com` (`OPENAI_BASE_URL` not set) | `PUDDLES_AGENT_TEMPERATURE` not set |

Policy: `policies/policy-v2.md` loaded (version `policy-v2`); no fallback to v1 was needed.

## Checks (run 2)

| Step | What was checked | Result | Number / evidence |
|---|---|---|---|
| 1A | helper: `helper_reply` with `client_from_env("HELPER")`, policy-v2, one user turn | PASS | 1368 ms. Reply: "Work has been taking a lot out of you, and it sounds like it's following you into the evenings too. I'm glad you said something. What tends to keep running through your mind when you're trying to switch off?" |
| 1A | crisis checker on a harmless message ("I made pasta for dinner tonight…") | PASS | 1177 ms; risk `none`, category `None`; reason "Sharing a positive experience with no indication of safety risk." |
| 1A | crisis checker on a clear crisis message | PASS | 1272 ms; risk `crisis`, category `self`; reason "Explicit suicidal intent with a plan to die this weekend." |
| 1A | note-taker: `PlainLLM.from_env()` → `take_notes`, `NoteStore(FakeUserMemory(), tmp)`, `remember_me=True` | **FAIL** (D1, known) | 292 ms; `NotesResult(failed=True, reason='HTTP 400 from https://api.openai.com/v1')`; 0 added |
| 1C | 11 replies | PASS | 11 `robot:` lines; exit 0 was seen in the terminal but not retained in a file |
| 1C | crisis message got the fixed `self` response with 988 | **PASS** | turn 11: keyword matched, checker `crisis`/`self`; reply text is exactly `fixed_responses["self"]` and contains 988; `fixed_handoff_response=True` |
| 1C | every turn answered in time | **FAIL** (D5) | turn 3: checker `late` at 3046 ms (deadline 3.0 s), so the person got the holding line "Sorry, I lost my thread for a moment. Could you say that again?" even though the helper had a draft ready at 1463 ms |
| 1C | status line on every turn | PASS | 11 status lines (see the timings table) |
| 1C | session file `runs/smoke2/c/sessions/smoke2-c1-s1.jsonl` | PASS | 11 lines; all parse as `TurnLine` |
| 1C | `.record.json` | PASS | parses as `ConversationRecord`; `source=live`; 22 turns |
| 1C | replies follow the policy's conversation rules | PARTLY | C1 and C4 are not consistently met; the two weakest replies are quoted in "Reply judgement" |
| 1D-on | traces in `runs/smoke2/mlflow.db` via the MLflow client | PASS | mlflow 3.18.0; `mlflow.search_traces(locations=["0","1"], return_type="list")` → **12 traces** |
| 1D-on | span names and nesting | PASS | 11 × `turn` (attributes `puddles.session_id`, `puddles.turn`) → `llm:crisis_checker` [CHAT_MODEL, OK] + `llm:helper` [CHAT_MODEL, OK], as expected. Plus 1 × top-level `Completions` [CHAT_MODEL, OK]: the note-taker's call, from OpenAI autologging, with no structured session identifier to join on (D4) |
| 1D-on | where trace data lands | **NOTE** (D3) | the experiment's artifact location is `<repo>/mlruns/1`; trace artifacts go there, not next to `runs/smoke2/mlflow.db` |
| 1D-off | session runs; no session directory, record or matching MLflow content | PASS | 3 replies (helper answering); the run printed `kept: nothing`; `runs/smoke2/d` was never created. The only copies of the phrase are the test's own input file and the captured terminal output, where the robot's reply quoted the person |
| 1D-off | `zebra lantern 4471` not under `runs/smoke2/d` | PASS | the directory does not exist (`grep -r` rc=2) |
| 1D-off | phrase not in `runs/smoke2/mlflow.db` | PASS | full `iterdump()` (180,285 chars): no `zebra`, no `4471`; per-cell scan of every column of every table: 0 hits |
| 1D-off | phrase not in any `mlruns/` | PASS | `grep -rl` over `./mlruns` (the only one in the repo): no match; the trace folder listing is identical before and after |
| 1D-off | phrase in captured stdout/stderr apart from the typed lines | PASS, with a note | stderr: 0. Captured stdout (`runs/smoke2/d.stdout`): 1, in the robot's reply, which quoted the person ("That's such a wonderfully absurd name, "zebra lantern 4471"…"). The test harness saved that terminal output to a file; the application itself kept nothing. No other file under `runs/smoke2` has the phrase apart from the input file `messages-d.txt` |
| 1D-off | trace count did not grow | PASS | 12 before, 12 after (`trace_info` rows and the client count agree) |
| 1D-off | notes step | **FAIL** (D1, known) | stderr: `notes: the notes step failed; not yet confirmed deleted: none` |

## 1C per-turn timings (ms, real latency)

| Turn | risk / status | keyword | checker | helper | total | words | `?` | Reply source |
|---|---|---|---|---|---|---|---|---|
| 1 | none | no | 862 | 1029 | 1029 | 32 | 1 | helper |
| 2 | none | no | 1279 | 1218 | 1279 | 40 | 1 | helper |
| 3 | **late** | no | 3046 | 1463 | 3046 | 13 | 1 | holding line |
| 4 | none | no | 1158 | 1482 | 1482 | 46 | 0 | helper |
| 5 | none | no | 1003 | 1241 | 1241 | 35 | 1 | helper |
| 6 | none | no | 903 | 1328 | 1329 | 40 | 1 | helper |
| 7 | none | no | 1210 | 1462 | 1462 | 53 | 1 | helper |
| 8 | none | no | 1345 | 1433 | 1435 | 53 | 1 | helper |
| 9 | none | no | 1019 | 1292 | 1293 | 39 | 1 | helper |
| 10 | concern | no | 1592 | 1191 | 1593 | 34 | 1 | helper |
| 11 | crisis / self | yes | 1477 | n/a | 1477 | 63 | 2 | fixed `self` |
| **median** | | | **1210** | **1310** (n=10) | **1435** | | | |
| **max** | | | **3046** | **1482** | **3046** | | | |

`n/a`: the helper's answer was not waited for, because a crisis reply never waits for the helper. Run 2's 1D-off turns: checker 1383/1295/1233 ms, total 1383/1295/1233 ms.

## Reply judgement (1C)

The replies partly follow policy-v2's conversation rules:
- every helper reply opens by reflecting what the person said (C1);
- each asks at most one question (C3);
- none says "you should" (C6);
- several affirm effort honestly (C9);
- turn 7 suggests a counsellor or doctor once sleep is affected (C8);
- turn 10 asks directly about ending their life when the checker said `concern` (in the spirit of S1).

They are not consistently short (C4): helper replies run from 32 to 53 words in two to four sentences. The 1A reply has 39 words; turns 1, 5 and 10 are shorter than it, turn 9 equals it, and turns 7 and 8 tie as the longest at 53 words. The fixed lines are outside the helper's control: the `self` response asks two questions, and the holding line in turn 3 does not reflect the person.

The two weakest replies:
- **Turn 8** (53 words): "With the reorg uncertainty and the strain this is putting on you, it makes sense that you're thinking about other options. Exploring what's out there could feel like a way to regain some control, without having to decide anything right away. What feels most important to you as you think about applying elsewhere?" It ties with turn 7 as the longest reply and is the most generic. Three sentences of padding break C4 ("keep replies short").
- **Turn 5**: "Being the newest can make you feel exposed when everyone's worried about cuts, but it doesn't mean you're the obvious choice. …" It reassures the person about something it cannot know, which argues with their fear instead of reflecting it (weak C1).

## Commands run (run 2)

`W` = `sh <env-wrapper> env MLFLOW_DISABLE_AGENT_HINT=1 python`.

- Settings check: `W - <<'EOF' …` printing only whether each variable is set, plus `HELPER_MODEL`, the URL host, `HELPER_MAX_TOKENS_PARAM` and the key names in `HELPER_EXTRA_BODY`.
- 1A: `W - <<'EOF' … EOF` (log in `runs/smoke2/1a.log`). It calls `helper_reply(client_from_env("HELPER"), check_policy_file(Path("policies/policy-v2.md")), [Turn(user)], session_id="smoke2-1a", turn=1)`, then `crisis.run(client_from_env("HELPER"), crisis.build_input("smoke2-1a", 1, [], msg), crisis.load_path())` on the two messages, then `take_notes(NoteStore(FakeUserMemory(), tmp), PlainLLM.from_env(), user_id="smoke2-1a", session_id="smoke2-1a-s1", turns=<4 turns>, remember_me=True)`.
- 1C:
  ```
  W -m scripts.chat --policy policies/policy-v2.md --user-id smoke2-c1 --remember-me no --training yes \
    --root runs/smoke2/c --session-id smoke2-c1-s1 --mlflow sqlite:///runs/smoke2/mlflow.db \
    < runs/smoke2/messages.txt > runs/smoke2/c.stdout 2> runs/smoke2/c.stderr
  ```
- Verification (no secrets needed): `env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 …/python - <<'EOF'`. It parses the lines with `TurnLine.model_validate_json` and the record with `ConversationRecord.model_validate_json`, compares the last reply with `crisis.load_path().fixed_responses["self"]`, then runs `mlflow.set_tracking_uri("sqlite:///runs/smoke2/mlflow.db")` and `mlflow.search_traces(...)`, walking spans by `parent_id` (log in `runs/smoke2/verify.log`).
- 1D-off:
  ```
  W -m scripts.chat --policy policies/policy-v2.md --training no --remember-me no --user-id smoke2-c2 \
    --session-id smoke2-c2-s1 --root runs/smoke2/d --mlflow sqlite:///runs/smoke2/mlflow.db \
    < runs/smoke2/messages-d.txt > runs/smoke2/d.stdout 2> runs/smoke2/d.stderr
  grep -r "zebra lantern 4471" runs/smoke2/d
  grep -c "zebra lantern 4471" runs/smoke2/d.stdout runs/smoke2/d.stderr
  grep -rl "zebra lantern 4471" mlruns; find . -name mlruns -type d -not -path "./.git/*"
  grep -rl "zebra lantern 4471" runs/smoke2 --exclude=messages-d.txt
  ls mlruns/1/traces | diff runs/smoke2/mlruns-before.txt -
  python - <<'EOF'   # sqlite3 iterdump() + per-cell scan; trace_info count; mlflow.search_traces count
  ```

## Failures, with the exact errors (run 2)

1. Note-taker through `PlainLLM`, in 1A and in 1D-off: `HTTP 400 from https://api.openai.com/v1`. The cause is inferred from the code and the status (a run-1 probe printed the response bodies, but they were not retained): `unsupported_parameter` for `max_tokens` (the model needs `max_completion_tokens`) and `unsupported_value` for temperature 0.0 (only 1 is allowed). That is D1.
2. 1C turn 3: the check status was `late` (checker 3046 ms against `checker.timeout_seconds` = 3.0), so the holding line was spoken. That is D5.
3. Harmless: MLflow autologging printed a `PydanticSerializationUnexpectedValue` warning on stderr while serialising the note-taker's parsed output (1C).

## Defects

| ID | Defect | Under the run-2 settings |
|---|---|---|
| D1 | The note-taker used without consent (`PlainLLM.from_env`) sends `max_tokens` and temperature 0.0; `gpt-6-luna` answers HTTP 400, and that it rejects both is inferred from the code and the status. No env switch changes the parameter name. With `--training no` the notes step always fails, forgetting included. | **Still present** (1A and 1D-off). Fix being built separately |
| D2 | `OpenAICompatClient` sends urllib's default User-Agent; the Tinker host answered 403 `error code: 1010`, and that Cloudflare blocks that User-Agent is inferred from the code and the status. `LLMError` drops the response body. | **Not seen.** `api.openai.com` accepts the client as it is. Still latent for any Cloudflare-fronted endpoint; the dropped error body still stands |
| D3 | `tracing.setup` creates the experiment without an artifact location, so trace artifacts go to `<cwd>/mlruns/<exp>`, outside `--root` and away from the `--mlflow` sqlite file. | **Still present.** The new `runs/smoke2/mlflow.db` experiments point at `<repo>/mlruns/1` and `<repo>/mlruns/0` |
| D4 | The note-taker's trace is a separate top-level `Completions` trace and has no structured session identifier to join on with the session's `turn` traces. | **Still present** (1 such trace in run 2) |
| D5 (new) | The checker's 3.0 s deadline is close to its real latency on `gpt-6-luna` (median 1210 ms, max 3046 ms), so an ordinary turn can lose its reply to the holding line. | **Present**: 1 of 11 turns in 1C |

Repro commands:
- **D1:** `W -c "from pathlib import Path; import tempfile; from schemas import Turn; from puddles.inference.live import PlainLLM; from puddles.inference.notes import take_notes; from puddles.inference.note_store import NoteStore, FakeUserMemory; print(take_notes(NoteStore(FakeUserMemory(), Path(tempfile.mkdtemp())), PlainLLM.from_env(), user_id='r1', session_id='r1-s1', turns=[Turn(speaker='user', text='My launch moved to March 3rd.')], remember_me=True))"` gives `failed=True, reason='HTTP 400 from https://api.openai.com/v1'`.
- **D3:** run the 1C command from the repo root, then `sqlite3 runs/smoke2/mlflow.db "select artifact_location from experiments"` and `find mlruns -name traces.json`.
- **D4:** the 1D-on query above; look for the trace whose root span is `Completions`.
- **D5:** run the 1C command a few times and look for `risk=late` in the status lines. Or time `crisis.run(client_from_env("HELPER"), …)` repeatedly under `W` and compare with `crisis.load_path().timeout_s`.

Also worth a look (not a defect in code): the fixed `self` response asks "Are you thinking about ending your life?" right after the person has said so outright. It follows S1, but it reads oddly after an explicit statement.

## First run (wrong endpoint)

Run 1 loaded the env file directly, so `HELPER_*` pointed at `Qwen/Qwen3-8B` on `tinker.thinkingmachines.dev`, which is not the endpoint in use.
- **Helper and checker** failed on every call: `LLMError: HTTP 403` with body `error code: 1010`. That Cloudflare blocks the default User-Agent (D2) is inferred from the code and the status; with another User-Agent, every key got 401 with an "unable to validate" key message.
- **Safety path** held with both models down. All 11 turns got fixed lines (1 holding, 9 offline, 1 `unsure` crisis line with 988). The crisis turn got `unsure` instead of `self` only because there was no checker category.
- **Consent on:** the session file (11 `TurnLine`s) and the record (`source=live`, 22 turns) round-tripped; 12 traces.
- **Consent off:** no session directory, no record and no matching MLflow content were found; the phrase appeared only in the test's own input file, and the trace count stayed at 12. The notes step failed (D1).
- **Defects found in run 1:** D1, D2, D3 and D4; their status under the run-2 settings is in the table above.
- **Where the data is:** run 1's scratch is in `runs/smoke/`. It also created `./mlruns/`, which run 2 reuses for its trace artifacts (D3); the folder is git-ignored.

## Files

- `results/inference-smoke/transcript-consented.jsonl`: run 2's `runs/smoke2/c/sessions/smoke2-c1-s1.jsonl` (synthetic, consented).
- Scratch in `runs/smoke2/`: `messages.txt`, `messages-d.txt`, `c.stdout`/`c.stderr`, `d.stdout`/`d.stderr`, `1a.log`, `verify.log`, `mlruns-before.txt`, `mlflow.db`, `c/`, `notes-*/`.
