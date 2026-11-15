# Stage 6: the overnight loop and the plots

**Target:** start by 23:30. **Owner:** unassigned. **Needs:** stage 5.

Issues: [#21](https://github.com/puddles-convo/puddles/issues/21), [#12](https://github.com/puddles-convo/puddles/issues/12), [#23](https://github.com/puddles-convo/puddles/issues/23).

## Goal

The loop runs round after round with nobody watching, stops itself at its caps, survives a crash, and leaves a log and a set of plots that explain what it did. The final policy goes to the robot pair by Sunday 09:00.

## Checklist

**The loop**

- [ ] `scripts/run_loop.py`: reads a config file, then runs the stage 0 graph once per round until a cap is reached.
- [ ] Config: `max_rounds`, `conversations_per_round`, `conversations_for_critic`, `max_turns`, `spend_limit_usd`, the prices table, the training scenario ID file, the validation ID file, the seed, the judge version, the noise. Values come from stages 3 and 5.
- [ ] A fresh root folder and the real past.dev project. Not `runs/small`.
- [ ] One line printed per round: number, outcome, quality before and after, cost so far.

**Which scenarios a round uses**

- [ ] Training scenarios come from the ESConv training split. Each round draws `conversations_per_round` of them with a sampler seeded by the seed and the round number, so a re-run of a round draws the same ones.
- [ ] The validation IDs and the training IDs share nothing. The loop checks this at start and refuses to run otherwise.
- [ ] The critic gets the `conversations_for_critic` lowest-scoring conversations (`select_for_critic`, stage 2).

**Starting again after a stop**

The code as it stands does not restart safely. Three things need building.

- [ ] **The policy in force comes from the log.** `puddles.round.policy_in_force` picks the newest policy file by its modified time. Replace that: take `policy.in_force` from the last round event, and check the file's hash against the event. When the log has no round event yet, as in a fresh root, the policy in force is the starting policy named in the config, the same one stage 5 scored.
- [ ] **An interrupted promotion is undone.** `finish_round` writes the promoted policy file before it appends the round event. A crash between the two leaves a policy file the log knows nothing about. At start, a promoted policy file with no round event is removed, and the round is run again.
- [ ] **A round is written once.** `DecisionLog.append` does not refuse a second `RoundEvent` for the same round number. Add the guard for round events only. Memory-copy events carry the round number too, and there are several per round; they must still append.
- [ ] `DecisionLog.next_round()` gives the next number; that part works today.

**Copies into memory**

- [ ] `puddles.round.retry_pending` checks the three-attempt limit before it retries. A copy that fails its third attempt lets the round go ahead, and the loop stops one round later. Check the limit again after the retries.
- [ ] A crash after the round event but before a copy leaves no `pending` entry, so nothing resends it. At start, compare the records each round event points to with the copy events, and send any that are missing.

**Spend**

- [ ] Spend so far is the sum of `<root>/log/usage.jsonl` (stage 1), so it survives a restart. It includes failed attempts, the gate's conversations, Exa searches, and the calls of a round that was interrupted: finished calls at their real cost, and calls that were sent but never answered at their estimate.
- [ ] Before each round: stop if spend so far plus the most a round has cost so far, times 1.5, is over the limit. Before round 1, use the cost per round measured in stage 3, scaled to the overnight settings.
- [ ] **This is a stop rule, not a guarantee.** It is based on what rounds have cost so far, and some token counts are estimates, so one round can overshoot. Put a hard spending limit on each provider account as well, where the provider offers one.
- [ ] past.dev credits are not counted. Check the dashboard before leaving.

**Stopping on repeated failure**

- [ ] Three rounds in a row that fail on the same agent stop the loop with that agent's name. A round has failed on an agent if `usage.jsonl` has a `failed` line for that round and agent. Group the lines by round first, then count rounds in a row; a round has many lines.

**The end-of-round check**

Run after every round. Any failure stops the loop.

- [ ] Replay the kept edits from the log and compare with the policy in force: same rule IDs, same text, nothing extra.
- [ ] The Safety section's hash equals the starting policy's.
- [ ] The `.meta.json` of the policy in force (written in stage 2, for the starting policy and for each promoted one) agrees: every rule is a base rule or links to a kept edit in the log.

**If a round is too slow**

- [ ] Run the conversations in parallel on Modal (#12). Workers return records. Only the loop process writes files and writes to past.dev.

**If OpenAI credits run low**

A different judge gives different numbers, so its scores cannot be compared with the first judge's.

- [ ] Default: the loop stops, and the policy in force is the result.
- [ ] To carry on with the fallback judge (#23), start a new stretch of rounds: calibrate the fallback, measure the baselines and the noise again with it (stage 5), score the policy in force with it, and record the switch in the log. Plots mark the two stretches separately and never join their lines.

**The plots**

- [ ] `scripts/plots.py`, into `docs/results/`:

  | Plot | Drawn from |
  |---|---|
  | Quality by round, with the noise band and a mark for each kept and reverted edit | `scores` in the round events; the noise from stage 5 |
  | A table of every round: pattern, source, edit, outcome, reason | The round events and record files |
  | Judge score against user score | `runs/calibration/pairs.jsonl` (stage 2, lane A) |
  | Final policy against both baselines, three bars with the spread | The final evaluation results (issue #7). The format is agreed with the evaluation lead before Sunday. |

- [ ] Small made-up input files for all four plots under `tests/`, so the script can be tested before the real data exists.
- [ ] Every number in a plot comes from a file in the repo or in `runs/`.

## Tests that must pass

1. A fake loop of five rounds stops at `max_rounds`.
2. A fake loop with a low spend limit stops before the round that would pass it. Restarted, it still knows what was spent, including the calls of a round that was interrupted part-way through its conversations.
14. Three rounds in a row with a `failed` line for the same agent stop the loop. Two such rounds, a clean round, then a third do not. Several `failed` lines in one round count as one round.
3. Kill the process during round 3's conversations and start it again: the log has rounds 1, 2, 3 once each.
4. Kill it after the promoted policy is written and before the log entry: on restart the stray policy file is gone and the round runs again.
5. A log whose last line is cut off half-way is reported, not read as a round.
6. A second round event for an existing round number is refused. A memory-copy event for that round still appends.
7. A fresh root with an empty log starts from the starting policy in the config.
8. A memory copy that fails three times stops the loop before the next round starts.
9. A round event with a record that has no copy event gets its copy sent at the next start.
10. A policy file edited by hand between rounds, a changed Safety section, and a missing `.meta.json` for the policy in force each stop the loop at the next check.
11. A validation ID that also appears in the training file stops the loop at start.
12. The same seed and round number draw the same training scenarios.
13. `scripts/plots.py` draws all four plots from the made-up files.

## Before leaving it

- [ ] One full round watched from start to finish with the real settings.
- [ ] The spend cap set below what the team can afford to lose.
- [ ] The machine will not sleep, and the run is in a session that survives a closed laptop.
- [ ] Someone knows where the log is and how to read the last line.

## Sunday morning

- [ ] 08:00: read the log. Count kept, reverted, `no_pattern` and `no_evidence` rounds.
- [ ] 09:00: hand the policy in force to the robot pair, and to the evaluation lead for the held-out test and the outside sets (#6, #7).
- [ ] Run the plots. Pick one kept round and one reverted round to show.
- [ ] Freeze at 12:30. Submissions close at 14:00.

## Not in this stage

- No change to the judge, the validation set, the noise or the gate rule within a stretch of rounds. A change makes earlier rounds impossible to compare with later ones.

## Done when

The loop has been running for an hour without anyone touching it, and the first plot draws from its log.
