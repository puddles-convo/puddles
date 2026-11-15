# Stage 5: baselines and noise

**Target:** 23:00. **Owner:** unassigned. **Needs:** stage 3, and the judge frozen in stage 2. **Blocks:** stage 6.

Issue: [#19](https://github.com/puddles-convo/puddles/issues/19).

## Goal

Two numbers the rest of the project stands on: how good the starting policies are, and how much the score moves when nothing has changed. The gate keeps an edit only if the gain is larger than that movement, so until it is measured, nothing can be promoted.

## Checklist

- [ ] Pick the validation set: a fixed list of ESConv validation scenario IDs, saved to a file under `fixtures/` (IDs only, no ESConv text). The gate uses this same list every round. The size comes from the cost measured in stage 3.
- [ ] Fix the seed and the turn limit. Write both down here.
- [ ] Score `policies/baseline-plain.md` once on the validation set.
- [ ] Score `policies/policy-v1.md` three times on the validation set, with the same settings each time.
- [ ] Noise = the highest quality of the three minus the lowest. Quality is the mean over conversations of (empathy + relevance) / 2.
- [ ] Write the results into the table below and into `runs/baseline/`.
- [ ] Give the gate its threshold: the noise figure goes into the loop's config and `noise_measured` is turned on. From here the gate can keep an edit.
- [ ] Give the gate its starting `current_scores`: the mean of the three `policy-v1` runs. This choice is not in the gatekeeper's spec; write it into the spec if the team agrees.
- [ ] Pick the training scenario IDs for the overnight run from the ESConv training split, saved the same way. No ID appears in both files.
- [ ] Log each scoring run to MLflow as its own run, tagged `baseline`.

## Results

| Policy | Run | Conversations | Empathy | Relevance | Quality |
|---|---|---|---|---|---|
| baseline-plain | 1 | | | | |
| policy-v1 | 1 | | | | |
| policy-v1 | 2 | | | | |
| policy-v1 | 3 | | | | |

Noise: ____. Validation scenarios: ____. Seed: ____. Turn limit: ____. Judge version: ____.

## Tests that must pass

1. All four runs used the same scenario IDs, seed, turn limit and judge version. The script refuses to compare runs that differ.
2. The noise is not negative, and it is stored where the gate reads it.
3. No scenario from the test split was loaded.

## How to read the result

- **Noise close to the gap between the two baselines** means the judge or the simulated user is too variable for this many conversations. Use more conversations per run before starting the loop, not a smaller threshold.
- **Three runs is a small sample.** The figure is a floor for the threshold, not a precise estimate. Say so when reporting it.
- **The same seed does not make runs identical.** The helper and simulated user are sampled. That variation is what the noise measures.

## Not in this stage

- No test split. It is used once, on Sunday, for the final comparison (issue #7).
- No CounselBench-Adv or VERA-MH. They are the outside check and the loop must never see them.

## Done when

The table is filled in, the noise is in the gate's configuration, and one command reproduces a baseline run.
