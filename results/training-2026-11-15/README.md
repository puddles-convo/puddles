# Training run of 2026-11-15

The records of one unattended training run and of every measurement behind its results. The run started at 03:02 and stopped itself at 06:40 after 13 rounds. Everything here was copied from the machine it ran on; nothing was edited.

## Result

**The final policy is `policies/training-2026-11-15/policy-v4.md`.** It differs from the starting policy in two rules: C1 was reworded (rounds 1 and 8) and C9 was added (round 4). Its safety rules are the starting policy's, unchanged, and it passed all 19 safety cases.

Quality is the average of empathy and relevance, each scored 1 to 5. Each policy below was measured on two fresh sets of the same 20 validation scenarios (40 conversations per policy), and each set was scored by both judges. The loop judge is `gpt-6.1-sol`; the final judge, `gpt-6-astra`, took no part in the loop. Both use the strict grading guide in `calibration/judge-skill-strict/SKILL.md`.

| Policy | Loop judge | Final judge |
|---|---|---|
| `policy-v1`, the run's starting policy | 3.29 | 3.19 |
| Control: the same safety rules, no conversation rules | 3.60 | 3.55 |
| `policy-v4`, the final policy | 3.98 | 3.80 |

| Comparison, scenario by scenario (95% interval) | Loop judge | Final judge |
|---|---|---|
| Final policy over the starting policy | +0.69 (+0.51 to +0.86) | +0.61 (+0.45 to +0.76) |
| Final policy over the control | +0.38 (+0.17 to +0.57) | +0.25 (+0.07 to +0.44) |
| Control over the starting policy | +0.31 (+0.14 to +0.50) | +0.36 (+0.19 to +0.55) |

Held-out check: 20 scenarios from the same split that the gatekeeper never used, one set per policy, both judges.

| Policy | Loop judge | Final judge |
|---|---|---|
| `policy-v1` | 3.38 | 3.20 |
| Control | 3.55 | 3.45 |
| `policy-v4` | 3.90 | 3.73 |

| Comparison, held-out (95% interval) | Loop judge | Final judge |
|---|---|---|
| Final policy over the starting policy | +0.53 (+0.20 to +0.88) | +0.53 (+0.15 to +0.90) |
| Final policy over the control | +0.35 (+0.05 to +0.65) | +0.28 (-0.05 to +0.57) |
| Control over the starting policy | +0.18 (-0.17 to +0.55) | +0.25 (-0.07 to +0.57) |

All 40 scenarios together (the gatekeeper's 20 with two sets per policy, the held-out 20 with one):

| Comparison, 40 scenarios (95% interval) | Loop judge | Final judge |
|---|---|---|
| Final policy over the starting policy | +0.61 (+0.42 to +0.79) | +0.57 (+0.36 to +0.78) |
| Final policy over the control | +0.36 (+0.19 to +0.54) | +0.26 (+0.09 to +0.44) |
| Control over the starting policy | +0.24 (+0.04 to +0.44) | +0.31 (+0.12 to +0.49) |

Reproduce every table above from the saved scores, with no model call:

```
python results/training-2026-11-15/measurements/tools/paired_stats.py
```

## What the numbers do and do not show

- The loop improved its starting policy by about 0.6 of a point under a judge it was not trained against, and by about 0.5 on scenarios it never saw, under both judges.
- The final policy is ahead of a helper with no conversation rules by 0.25 to 0.38 in every comparison. The interval is above zero over all 40 scenarios and on the gatekeeper's 20, and just reaches zero on the held-out 20 under the final judge.
- The hand-written conversation rules scored below having none. The difference is distinguishable from zero on the gatekeeper's scenarios and over all 40, and not on the held-out 20 alone.
- The people in every conversation are a language model playing an ESConv scenario. Nobody real took part.
- The grading guide was made stricter during the night and is not calibrated against human ratings.
- The safety wording of the starting policy was rewritten by hand before the run so that it passes the 19 safety cases. It has not been reviewed by the team.
- Two sets of 20 conversations from one policy differed by as much as 0.45, which is wider than the 0.275 the gatekeeper used as its bar. Single-set numbers, including each round's own decision, carry that uncertainty.

## The run

| | |
|---|---|
| Rounds | 13: 3 kept (rounds 1, 4, 8), 7 reverted, 3 ended with no candidate |
| Why it stopped | The proposer failed its checks in three rounds in a row (rounds 11 to 13), and the loop's own stop rule ended the run |
| Models | Helper, simulated person, critic, researcher, proposer: `gpt-6-luna`. Judge and safety questions: `gpt-6.1-sol` |
| Per round | 8 training conversations from a pool of 910; 20 fixed validation conversations; 19 safety cases; 6 helper replies per conversation |
| Calls and cost | 4,832 model calls, none failed at the provider; $6.67 by the run's usage log |
| Code | Branch `train-run`. The loop was restarted four times on newer commits, at round boundaries: after round 7 (`bf840e2`, a kept policy is measured again), after round 8 (`8143577`, rejected rules are set aside), after round 9 (`01ac931`, a clearly worse result is remembered; same-idea check) and after round 13 (`c1febb7`, which stopped at once on the three failures already in the log) |

Rounds 5 to 7 were decided against a bar that was too high: `policy-v3` had been kept on a reading of 4.15 and measured 3.75 to 3.90 afterwards. From round 8 the gatekeeper measures a kept policy again before using its score.

## What is in this folder

| Path | What it is |
|---|---|
| `run/log/decisions.jsonl` | The decision log: one line per round with the pattern, the edit, the scores before and after, the safety result and the reason. The authority for everything else |
| `run/log/usage.jsonl` | Every model call: agent, model, tokens, cost, time |
| `run/log/loop.jsonl` | The driver's own events, including why it stopped |
| `run/log/memory_reads.jsonl` | Every read of the shared memory, and whether the service answered |
| `run/policies/` | Each kept policy with its record of origin (`policy-vN.meta.json`), and every candidate (`policy-vN-candidate-rNN.md`) |
| `run/records/round-NN/` | The pattern, the research with its sources, the edit and the gatekeeper's outcome for each round |
| `run/conversations/`, `run/scores/` | Every conversation of the run and its scores. `round-00-current.*` is the last fresh measurement of the policy in force (`policy-v4`) |
| `measurements/astra/` | The final judge's scores. `sets/<label>.conversations.jsonl` are the conversations it scored, `sets/<label>.sol.scores.jsonl` the loop judge's scores of the same conversations, `<label>.scores.jsonl` the final judge's |
| `measurements/validation-rechecks/` | Fresh sets of the 20 validation scenarios for `policy-v1`, `policy-v3`, `policy-v4`, the plain baseline and the control |
| `measurements/heldout/` | The held-out check and its 20 scenario IDs |
| `measurements/stage5-baseline/` | The three baseline runs the noise of 0.275 was measured from, and the comparison of the two grading guides |
| `measurements/controls/` | The control policy: the run's safety rules with no conversation rules |
| `measurements/tools/` | The scripts and settings used, as run. Some hold paths of the machine they ran on |

Labels in `measurements/astra/`: `v1`, `v1b` are the two sets of the starting policy; `v2`, `v3` one set each; `v4`, `v4b` the final policy; `plain` the plain baseline with the original safety wording; `plain-s2`, `plain-s2b` the control; `heldout-v1`, `heldout-control`, `heldout-v4` the three held-out sets.
