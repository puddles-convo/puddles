# Reproduction of the training run of 2026-11-15

Does last night's result ([`results/training-2026-11-15/`](../training-2026-11-15/README.md)) hold when it is measured again? Three checks, all run on 15 November with last night's models (helper `gpt-6-luna`; judge `gpt-6.1-sol` with the strict guide), on the 20 held-out scenarios no gatekeeper used.

## Result

**Yes.** Over 10 fresh sets per policy, the final policy (`policy-v4`) beats the starting policy (`policy-v1`) by **+0.63** (95% interval +0.53 to +0.73), and was higher in **10 of 10** sets. Last night's single set gave +0.53.

| Policy | Mean of 10 sets | Spread (sd) | Range | Last night (one set) |
|---|---|---|---|---|
| `policy-v1`, the starting policy | **3.26** | 0.09 | 3.10 to 3.43 | 3.38 |
| Control: the same safety rules, no conversation rules | **3.63** | 0.12 | 3.48 to 3.93 | 3.55 |
| `policy-v4`, the loop's final policy | **3.89** | 0.10 | 3.73 to 4.03 | 3.90 |

| Comparison (per scenario, averaged over the 10 sets; 95% interval over the 20 scenarios) | Difference | Set by set |
|---|---|---|
| Final over starting policy | **+0.625 (+0.53 to +0.73)** | final higher in 10 of 10 |
| Final over control | **+0.252 (+0.15 to +0.37)** | final higher in 10 of 10 |
| Control over starting policy | +0.372 (+0.26 to +0.49) | control higher in 10 of 10 |

All 600 conversations completed and were scored. Cost: $9.94.

## The three checks

| Check | What it shows | Result |
|---|---|---|
| 1. Last night's tables from its saved scores (`paired_stats.py`) | The published numbers are computed correctly | Identical |
| 2. One fresh set per policy (`fresh-heldout/`) | Whether the gain shows on new conversations | v1 3.50, control 3.53, v4 3.78: final over start +0.275 (-0.07 to +0.68). Same direction, but one set is not enough: v1 happened to score high |
| 3. Ten fresh sets per policy (`repeat-measure/`) | The gain once the sampling noise is averaged out | the table above |

## What this does and does not show

- The loop's final policy is better than its starting policy, and better than having no conversation rules, under the loop's own judge, on scenarios it never saw. The result is not an artefact of one lucky set.
- One set of 20 moves by up to about 0.45 (control: 3.48 to 3.93). A single-set comparison, such as each round's own gate decision, carries that uncertainty.
- The hand-written conversation rules of `policy-v1` score below having none, in all 10 sets.
- Only the loop judge scored these sets. Last night's final judge (`gpt-6-astra`) did not take part.
- The people are a language model playing ESConv scenarios. The grading guide is the strict one written last night, not calibrated against human ratings.
- The helper code includes integration's reply-trimmer change (#77), which last night's run did not have.

## Reproduce

```
python scripts/repeat_stats.py results/reproduction-2026-11-15/repeat-measure    # no model call
N=10 P=6 sh scripts/repeat_measure.sh                                            # the sets again (~$10)
```

`repeat-measure/<policy>/<version>-run-<n>/` and `fresh-heldout/` hold each set's `scores.jsonl` and `result.json`. The conversations themselves are in `runs/` on the machine that ran them (git-ignored).

The whole loop is also being run again from `policy-v1` (`sh scripts/reproduce_run.sh loop`, root `runs/repro-2026-11-15`); its rounds are in [`results/toolbox-2026-11-15/RUNLOG.md`](../toolbox-2026-11-15/RUNLOG.md).
