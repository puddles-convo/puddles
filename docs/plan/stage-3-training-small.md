# Stage 3: training end to end, small

**Target:** 22:00. **Owner:** unassigned. **Needs:** stage 2. **Blocks:** stages 5 and 6.

## Goal

One real round, then a second, on the 5 fixture scenarios with real models, Exa and past.dev. This is where the pieces first meet. The point is to find what breaks and to measure what a round costs, while a round is still cheap.

## Setup

- [ ] A past.dev project or audience used only for development, so test records do not sit beside the real run's.
- [ ] Every variable in the [plan README](README.md#environment) set in the shell, none in a file that git tracks.
- [ ] `runs/small/` as the root. It is ignored by git.

## Checklist

- [ ] `python -m scripts.run_graph_round --root runs/small --scenarios fixtures/scenarios.jsonl --max-turns 6`: round 1.
- [ ] Read the five conversations, the scores, the pattern, the sources and the edit. Does each follow from the one before it?
- [ ] Run it again: round 2. The critic, researcher and proposer should recall round 1's records. `log/memory_reads.jsonl` shows who read what.
- [ ] Force a stop in the middle of round 3 (Ctrl-C during the conversations), then run again. The round number must be 3 again, with nothing half-written in the log.
- [ ] Fill in the table below from the MLflow run.

## What to measure

| Per round | Value |
|---|---|
| Seconds, total and per node | |
| Tokens in and out, per agent | |
| Cost in dollars | |
| Conversations that failed, out of 5 | |
| Agents whose first answer failed its checks | |
| Seconds for past.dev to confirm a copy | |

From these, work out and write down:

- [ ] How many conversations per round the overnight run can afford.
- [ ] How many rounds fit between 23:30 and 08:00.
- [ ] The spend cap for stage 6.

## Tests that must pass

1. Two `round` events in `runs/small/log/decisions.jsonl`, each pointing at record files whose hashes match.
2. One MLflow trace per round with a span for every node that ran.
3. Every memory copy is `confirmed`, or is `pending` and was resent at the start of the next round.
4. Round 2's proposer input lists round 1's edit and its outcome.
5. After the forced stop, the log still parses line by line.

## What usually goes wrong here

- **A model answers outside its schema.** The retry carries the reason; if it fails twice the round should stop with the agent's name, not carry on.
- **The simulated user ends at once or never ends.** Fix in the prompt, then re-run stage 2 lane D.
- **Exa returns nothing useful.** The round ends as `no_evidence`. That is a valid outcome, not a failure.
- **past.dev is slow to confirm.** Copies wait up to 60 seconds each. Note the real figure; it sets how long a round's tail is.

## Not in this stage

- No ESConv scenarios beyond the fixtures.
- No promotion that counts: the gate still runs on provisional noise.

## Done when

The table is filled in, the three numbers under it are written down, and a teammate who was not in the room can read `runs/small` and the trace and follow why round 1 ended as it did.
