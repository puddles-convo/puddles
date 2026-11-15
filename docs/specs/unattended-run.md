# The unattended run: unexpected output maps to something safe

Tonight's loop runs for hours on real services with nobody watching and no workflow engine around it.
This contract is the set of rules that make that safe. Each rule comes from a reproduced finding of a
five-reviewer robustness review (2026-11-15) or from the first real rounds.

**Safe** means all of:

- a round ends in one of the four outcomes and is in the decision log, or the loop stops with a reason
  written down;
- the policy in force changes only through a `kept` from the real gatekeeper;
- nothing unmeasured is recorded as measured;
- the loop does not spend money going round in circles;
- memory is not left wrong for the rounds that follow.

Out of scope here, as everywhere tonight: a file changed by another process, an object built to
change while it is read, a caller that forges values.

## 1. The overnight driver (`scripts/run_loop.py`, `puddles/loop.py`)

**1.1 The real gatekeeper.** `--gate real` builds the gatekeeper the way `scripts/run_graph_round.py`
does: `puddles.gate_join.real_gate_deps` and `puddles.gate.make_gate` without `--fake`, the fake safety
check with it. Its validation scenarios are the ones named by `loop.validation_ids`; its noise and
`noise_measured` come from the `[loop]` table; `max_turns` and `seed` come from the same table. The
safety model's price is checked before the first call, as that script does.

**1.2 The judge goes through the round's own adapter.** In real mode the judge from `judge_from_env()`
is given to the round as `RoundDeps.judge_panel`, as `scripts/run_graph_round.py` does. A provider error
in one judgment then leaves that conversation unscored and the round goes on.

**1.3 A stand-in never promotes with real models.** Without `--fake`, `--gate kept` is refused before
any call (exit 2). `--gate reverted` without `--fake` is a pilot: the driver says so in its first
printed line, and nothing can be kept.

**1.4 No improvement run on unmeasured noise.** `--gate real` without `--fake` is refused before any
call (exit 2) unless `loop.noise_measured` is true and `loop.noise` is a finite number greater than 0.
(The gatekeeper would revert every candidate; the loop would pay for rounds it can never keep.)

**1.5 Which judge must be frozen.** A panel judge (`kind == "panel"`) must be frozen, as now. The
single-model judge (`kind == "fallback"`) needs no frozen file: its calibration is parked by the
owner's decision. In both cases, when `loop.judge_version` is set and is not the judge's `version`,
the run is refused before any call: the baseline and the noise were measured with another judge.
`scripts/score_baselines.py` follows the same rule for which judge must be frozen.

**1.6 An unexpected error costs one round, not the night.** Any `Exception` raised by one round (not
`KeyboardInterrupt`, not `SystemExit`) is caught by the loop. It appends one line to `log/loop.jsonl`:
`{"event": "round_error", "round": n, "error": "<type>: <message cut to 300 characters>", "at": ...}`
and goes on to the next turn of the loop, which starts as a restart does (`prepare_round`). Three
round errors in a row stop the loop. A round that ends with an outcome resets the count.

**1.7 Three rounds with nothing scored stop the loop.** When three rounds in a row end with no scored
training conversation at all (the helper's or the judge's server is down), the loop stops.

**1.8 Why it stopped is written down.** Every way the loop ends appends one line to `log/loop.jsonl`:
`{"event": "stopped", "reason": ..., "rounds": n, "spend_usd": ..., "policy_in_force": ..., "at": ...}`
and prints the same. The exit code is 0 for a cap (rounds, spend) and 3 for a stop caused by failures
(1.6, 1.7, the existing same-agent streak).

## 2. Memory is a copy of the log and never stops the run (`puddles/memory.py`, `puddles/round.py`, `puddles/graph.py`)

**2.1 A failed read is "memory unavailable".** When a recall fails (`PastError`, an `httpx` error, or a
response that is not the expected shape), `recall_records` returns no records and writes a line to
`log/memory_reads.jsonl` with `"unavailable": true` and the error. The agent runs without memory.
An unavailable read is told apart from an empty one in that log.

**2.2 The audience call is not fatal.** When `ensure_training_audience` fails at a round's start, the
failure is written to `log/memory_reads.jsonl` (`"reader": "round"`, `"unavailable": true`) and the
round goes on.

**2.3 A pending copy stays pending.** A record whose copy to memory failed is retried once at each
round start, and no more than once per round start. A pending copy never raises, never stops the loop
and never blocks a start, whatever its attempt count. (This replaces the stop after three attempts.)

**2.4 A malformed answer from the memory service is a `PastError`.** A 200 with a body that is not
JSON, a missing field, `sources` that is not a list, metadata that is not an object: `PastClient`
raises `PastError`, so 2.1 and 2.3 apply.

**2.5 Only this run's records are recalled.** A recalled result is used only when its source metadata
names a record ID that this root's decision log holds and the metadata's `sha256` equals the log's
hash for that record. Record IDs are not taken from a result's free text. (Entries left by an earlier
run in the same past.dev project carry the same IDs, `pattern-r01` and so on, and were being served
as this run's records.)

**2.6 A recall query fits the service's limit.** A recall query is cut to the memory service's limit
of 1,000 characters before it is sent. (Found in the first real training round: the critic's query,
built from a judge's note, was longer; the service answered 400 and the critic ran without memory.)

## 3. The judge and the agents (`puddles/eval/judge.py` single-model judge, `puddles/agents/`, `puddles/policy.py`, `puddles/llm.py`)

**3.1 Strategy labels are normalised before they are checked.** In the single-model judge, a label is
stripped of surrounding white space and quotes and of a leading turn marker (`Turn 2: `, `2. `, `2: `,
`Helper turn 2 - `), and matched to the canonical list without regard to case. A label that still
matches nothing fails the check as before. (A real judge wrote `Turn 2: Reflection of feelings`
twice and the conversation went unscored; one unscored validation conversation reverts a round.)

**3.2 The single-model judge keeps the safety flags.** When the judge's answer carries safety flags
and evidence, the score record carries them. (The gatekeeper does not yet act on them; see the end.)

**3.3 Rule C8 is protected.** The proposer cannot remove or reword C8 (the rule that recommends a
counsellor or doctor when a problem goes beyond listening), in the same way it cannot touch the
Safety rules. The protected set is written in one place.

**3.4 A malformed search-tool call is an error result, not a crash.** When the model's tool call has
arguments that are not JSON, no `query`, a blank or non-string `query`, or names another tool, the
model gets a tool result saying so, it counts as one of its searches, and no exception escapes. A tool
turn cut off at the output limit fails the researcher (`AgentFailed`).

**3.5 A failed search is counted.** When the researcher's search fails (section 5 of
`provider-settings.md`), one `failed` line is written for the researcher, so the loop's same-agent
streak sees an Exa outage.

(A rule 3.6, "the critic's `seen_in` equals the number of distinct example IDs", was written and then
withdrawn before the check: the example IDs are a sample of the sightings, as the critic's own doc
shows with 14 sightings and 3 examples, so the rule would have failed correct answers.)

## 4. A kept policy is measured again (`puddles/gate.py`, `puddles/gate_join.py`, `scripts/run_loop.py`)

Added at 05:00 on 2026-11-15, from the run itself. A candidate is kept because one measurement of it
came out high; the gate then saved that same measurement as the score every later candidate must
beat. A measurement picked for being high overstates the policy. Seen: `policy-v3` was kept at
quality 4.15; two fresh measurements of the same file on the same scenarios gave 3.90 and 3.90. The
next candidate then needed 4.425 to be kept.

- **4.1 The setting.** `GateDeps` gets `remeasure_kept: bool = False`. `real_gate_deps` takes
  `remeasure_kept: bool = False` and passes it on. `scripts/run_loop.py` reads
  `PUDDLES_GATE_REMEASURE_KEPT`: `1` is on, unset or `0` is off, anything else is a refusal before any
  model call (exit 2, like the other refusals), naming the variable.
- **4.2 Off is today's behaviour, byte for byte.** With the setting off nothing changes: the same
  calls, the same files, the same `current.json` text. No existing test changes.
- **4.3 On: a keep does not save the winning scores.** On `kept`, the gate does not rewrite
  `current.json` (step 4 of `make_gate` in `docs/specs/gate.md` is skipped). Everything else about the
  round is unchanged: the decision, the checks, the report, and the scores the round logs, which are
  what that round measured.
- **4.4 On: only a fresh measurement is believed.** `current.json` gets one more field,
  `"measured": "fresh"`, written when `_current_scores` scores the policy in force and saves the
  result. With the setting on, a file matches only if it has that field with that value as well as
  every field of the key. A file without it, which is what any keep wrote before this rule, counts as
  absent: the policy in force is scored again as round 0 with the tag `current`, and that result is
  saved with the field. So the first gated round after a keep, and the first after a restart that
  turns the setting on, measures the policy in force afresh, once; later rounds reuse it.
- **4.5 A limit, stated.** The files `round-00-current.jsonl` are replaced by each fresh measurement,
  as they already were whenever the policy in force was scored again. The numbers of earlier
  measurements stay in the decision log.

## 5. What was tried and rejected is not tried again (`puddles/agents/critic.py`, `puddles/agents/proposer.py`, `puddles/decision_log.py`, `scripts/run_loop.py`)

Added at 05:20 on 2026-11-15, from the run itself. The loop proposed the same idea it had already seen
rejected: rounds 5 and 7 both made C9 stricter, and both candidates scored about 0.65 below the policy
in force; rounds 2, 3 and 6 all reworded C1. The cause is in the code, not the models: when the critic
names an earlier pattern that already has a rule, the proposer's check forbids adding a rule, so it
must reword that same rule again, and nothing counts how often that has failed.

- **5.1 The setting.** `PUDDLES_PARK_REVERTED`: `1` is on, unset or `0` is off, anything else is a
  refusal before any model call in `scripts/run_loop.py` (exit 2, like 4.1). Off is today's behaviour:
  the same prompts, the same checks. No existing test changes.
- **5.2 What is parked.** Worked out by code from the decision log's round events only, for the policy
  version in force, P. A *failed try* is a round whose outcome is `reverted`, whose `policy.base` is
  P, and that has an edit. It is a *negative result* when both logged qualities are present and
  `scores.after.quality` is lower than `scores.before.quality`.
  - A **rule** is parked on P when one failed try that reworded or removed it was a negative result,
    or when two failed tries reworded or removed it.
  - A **pattern** (by its identity, following `same_as_pattern_id` the way the log already does) is
    parked on P when one failed try for it was a negative result, or when two failed tries were for it.
  - A keep changes the policy in force, so nothing is parked on the new version until it has failed
    tries of its own.
- **5.3 The critic does not name a parked pattern.** With the setting on, the critic's input marks each
  parked pattern as parked, with the rounds that tried it and what they measured. The check rejects
  an answer whose `same_as_pattern_id` is a parked pattern, with a reason that says it is parked on
  this policy and asks for a different pattern or none. The usual single retry applies; a second
  failure takes the existing path (the round ends as `no_pattern` with the failure as its reason).
- **5.4 The proposer does not edit a parked rule.** With the setting on, the proposer's input lists the
  parked rules, each with the edits that tried it and the measured result. The check rejects a
  `reword` or `remove` of a parked rule, with a reason naming the earlier edit and its result and
  asking for a different rule or a new one. The existing check "this pattern already has an active
  rule; reword or remove it instead of adding" does not apply when that rule is parked, so the
  proposer always has a legal move.
- **5.5 Limits, stated.** The critic can describe an old weakness under a new name; then 5.3 does not
  fire, and 5.4 still keeps the parked rule from being edited. (A new rule that restates a rejected
  rewording was a limit here until 5.7.)
- **5.6 A clearly worse result is remembered for the whole run.** Added at 05:45, after round 8's keep
  emptied the list while C9's stricter wording had been rejected twice at about 0.65 below the policy
  in force. A failed try is *clearly worse* when both qualities are logged, `scores.noise` is logged,
  and `scores.after.quality` is lower than `scores.before.quality` by more than `scores.noise`. The
  rule and the pattern of a clearly worse failed try on version B are parked on B and on every
  version the run reaches from B through keeps, whatever is kept in between. (In a run every new
  version comes from a keep, so this is every later version.) Everything else in 5.2 stays per version.
- **5.7 The same idea in other words is not tried again.** With the setting on, and only when there
  is at least one *rejected edit to compare with*, the proposer's answer gets one more check after
  its coded checks pass. The rejected edits to compare with are: every failed try on the policy in
  force, and every clearly worse failed try of the run (5.6). The check is one structured model call
  on the proposer's own model: it is given the proposed edit (action, rule, text) and each rejected
  edit (ID, action, rule, text, measured result), and answers whether the proposed edit would change
  the helper's behaviour in the same way as one of them, and which. The answer's schema has
  `same_as_edit_id` (an ID from the list, or null) and `reason`.
  - A non-null `same_as_edit_id` that is in the list fails the proposer's check with a reason naming
    that edit and its result and asking for a different idea. The usual single retry applies.
  - Anything else lets the edit through: null, an ID not in the list, an invalid answer, or a failed
    call (`AgentFailed` or a provider error). The gatekeeper remains the guard; this check only saves
    a round.
  - The call is made inside the proposer's step, so its tokens are counted with the proposer's.
  - The system prompt for this call is a new file beside the other agents' prompts.
- **5.8 How many failed tries park is a setting.** Added at 06:40, after round 11. With 5.7 on, a rule
  that has one failed try on the policy in force is not parked, so the critic names its pattern
  again; the proposer is then required to reword that same rule (the pattern has an active rule),
  and any rewording is the same idea as the one just rejected. Round 11 ended that way with no
  candidate. `PUDDLES_PARK_AFTER` is the number of failed tries on the policy in force that parks a
  rule or a pattern: a whole number from 1 up, default 2 (the behaviour of 5.2). With `1`, any
  rejected edit sets its rule and its pattern aside until a keep. A negative result still parks at
  once, and 5.6 is unchanged. `scripts/run_loop.py` refuses any other value before any model call
  (exit 2, like 5.1); unset means 2.

## Knowingly left out tonight

Each is a reproduced finding and becomes an issue.

- The gatekeeper does not veto on the judge's safety flags. (A keep confirmed against a fresh
  measurement was on this list until section 4.)
- A reply cut off at the output limit still counts as evidence.
- No deadline on one call or one round beyond each client's own timeout.
- Safety-check calls and failed judgments are missing from the spend total; the loop's spend limit is
  set below the real budget to cover them.
- Nothing parks a pattern after repeated reverts; a renamed pattern passes the duplicate-rule guard;
  a source is accepted if its URL came from the search, whatever its text says; repeated and
  unrecorded search queries.
- A locked MLflow store, two loops on one root.
