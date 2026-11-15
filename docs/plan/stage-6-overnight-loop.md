# Stage 6: the overnight loop and the plots

**Target:** start by 23:30. **Owner:** unassigned. **Needs:** stage 5.

Issues: [#21](https://github.com/puddles-convo/puddles/issues/21), [#12](https://github.com/puddles-convo/puddles/issues/12), [#23](https://github.com/puddles-convo/puddles/issues/23).

## Goal

The loop runs round after round with nobody watching, stops itself at its caps, survives a crash, and leaves a log and a set of plots that explain what it did. The final policy goes to the robot pair by Sunday 09:00.

## Checklist

**The loop**

- [ ] `scripts/run_loop.py`: reads a config file, then runs the stage 0 graph once per round until a cap is reached.
- [ ] Config: `max_rounds`, `conversations_per_round`, `max_turns`, `spend_limit_usd`, the validation ID file, the seed, the judge and its fallback. Values come from stages 3 and 5.
- [ ] Each round starts from `puddles.round.policy_in_force` and `DecisionLog.next_round()`, so a restart continues where the log ends.
- [ ] `puddles.round.retry_pending` runs at the start of every round. It already stops the loop when a memory copy is still pending after three attempts.
- [ ] A running total of spend from the token counts. The loop stops before a round it cannot afford.
- [ ] Three rounds in a row that fail on the same agent stop the loop with that agent's name.
- [ ] After every round, run the check that the policy file and the log agree: replay the kept edits from the log and compare with the policy in force, rule by rule. A mismatch stops the loop.
- [ ] One line printed per round: number, outcome, quality before and after, cost so far.

**If a round is too slow**

- [ ] Run the conversations in parallel on Modal (#12). Workers return records. Only the loop process writes files and writes to past.dev.

**If OpenAI credits run low**

- [ ] Switch to the fallback judge (#23), re-score the policy in force with it before comparing anything, and record the switch in the log.

**The plots**

- [ ] `scripts/plots.py`, from the decision log and the baseline runs, into `docs/results/`:
  - quality by round, with the noise band and a mark for each kept and reverted edit;
  - final policy against both baselines, as three bars with the spread;
  - judge score against user score, from the calibration run;
  - a table of every round: pattern, source, edit, outcome, reason.
- [ ] Every number in a plot comes from a file in the repo or in `runs/`.

## Tests that must pass

1. A fake loop of five rounds stops at `max_rounds`.
2. A fake loop with a low spend limit stops before the round that would pass it.
3. Kill the process during round 3 and start it again: the log has rounds 1, 2, 3 once each.
4. A policy file edited by hand between rounds stops the loop at the next check.
5. `scripts/plots.py` runs on the log from stage 3 without real data from the overnight run.

## Before leaving it

- [ ] One full round watched from start to finish with the real settings.
- [ ] The spend cap set below what the team can afford to lose.
- [ ] The machine will not sleep, and the run is in a session that survives a closed laptop.
- [ ] Someone knows where the log is and how to read the last line.

## Sunday morning

- [ ] 08:00: read the log. Count kept, reverted, `no_pattern` and `no_evidence` rounds.
- [ ] 09:00: hand the policy in force to the robot pair, and to the evaluation lead for the held-out test (#7).
- [ ] Run the plots. Pick one kept round and one reverted round to show.
- [ ] Freeze at 12:30. Submissions close at 14:00.

## Not in this stage

- No change to the judge, the validation set or the gate rule once the loop has started. A change makes earlier rounds impossible to compare with later ones.

## Done when

The loop has been running for an hour without anyone touching it, and the first plot draws from its log.
