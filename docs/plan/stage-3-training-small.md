# Stage 3: training end to end, small

**Target:** 22:00. **Owner:** unassigned. **Needs:** stage 2. **Blocks:** stages 5 and 6.

## Goal

Real rounds on the 5 fixture scenarios with real models, Exa and past.dev. This is where the pieces first meet. The point is to find what breaks and to measure what a round costs, while a round is still cheap.

**Nothing is promoted in this stage.** The noise has not been measured, so the gate runs with `noise_measured` off and reverts every candidate (stage 2, lane B). The reason in the log says what the checks found.

## Setup

- [ ] A separate past.dev project for development, with its own `PAST_API_KEY`, so test records do not sit beside the real run's.
- [ ] Every variable in the [plan README](README.md#environment) set in the shell, none in a file that git tracks.
- [ ] The prices table from stage 1 filled in.
- [ ] `runs/small/` as the root. It is ignored by git. **The overnight run uses a different root and a different past.dev project,** so nothing from this stage becomes its starting state.
- [ ] Validation for the gate, for this stage only: the same 5 fixture scenarios. The policy in force is scored on them once before round 1, to give the gate its `current_scores`.

## Checklist

- [ ] `python -m scripts.run_graph_round --root runs/small --scenarios fixtures/scenarios.jsonl --max-turns 6`: round 1.
- [ ] Read the five conversations, the scores, the pattern, the sources and the edit. Does each follow from the one before it?
- [ ] Run it again: round 2. Agents that run should recall round 1's records. `log/memory_reads.jsonl` shows who read what.
- [ ] Force a stop in the middle of round 3 (Ctrl-C during the conversations), then run again. The round number must be 3 again, with nothing half-written in the log.
- [ ] Fill in the table below from the MLflow run and `log/usage.jsonl`.

## What to measure

| Per round | Value |
|---|---|
| Seconds, total and per node | |
| Tokens in and out, per agent, gate included | |
| Exa searches | |
| Cost in dollars, from the prices table | |
| Conversations that failed, out of 5 | |
| Conversations left unscored | |
| Agents whose first answer failed its checks | |
| Seconds for past.dev to confirm a copy | |

From these, work out and write down:

- [ ] How many conversations per round the overnight run can afford.
- [ ] How many rounds fit between 23:30 and 08:00.
- [ ] The spend cap for stage 6, and the most one round can cost.

## Tests that must pass

A real round may end at any of its exits, so these depend on the path it took.

1. Each finished round has one `round` event in `runs/small/log/decisions.jsonl`, pointing at record files whose hashes match.
2. One MLflow trace per round with a span for every node that ran.
3. Every memory copy is `confirmed`, or is `pending` and was resent at the start of the next round.
4. No event has the outcome `kept`, and `runs/small/policies/` holds no promoted policy.
5. **If** round 1 produced an edit **and** round 2 reached the proposer, round 2's proposer input lists round 1's edit and its outcome.
6. After the forced stop, the log still parses line by line.

Recall of earlier edits is proved offline, not here: a scripted two-round test in `tests/test_graph.py` in which round 1 always produces an edit and round 2 always reaches the proposer.

## What usually goes wrong here

- **A model answers outside its schema twice.** What happens next depends on the agent; the table is in the [plan README](README.md#when-an-agents-answer-fails-twice). It is always counted and logged.
- **The simulated user ends at once or never ends.** Fix in the prompt, then re-run stage 2 lane D.
- **Exa returns nothing useful.** The round ends as `no_evidence`. That is a valid outcome, not a failure.
- **past.dev is slow to confirm.** Copies wait up to 60 seconds each. Note the real figure; it sets how long a round's tail is.

## Not in this stage

- No ESConv scenarios beyond the fixtures.
- No promotion.

## Done when

The table is filled in, the three numbers under it are written down, and a teammate who was not in the room can read `runs/small` and the trace and follow why round 1 ended as it did.
