# ESConv dataset run (stand-in for a failed live training run)

Made with `python -m scripts.dataset_rounds --root runs/esconv` (seed 0, starting from `policies/policy-v1.md`),
then copied here. The files have the same formats as a live run and pass the replay check.

Every model step was replaced by a step that reads ESConv (revision `32c54c1`), so nothing here was tested by the
live helper, judge or gatekeeper:

- **Critic:** the 10 lowest-rated of 60 seeded ESConv training conversations; names the catalogue pattern seen
  most often among them that also lowers seeker ratings across the training split.
- **Researcher:** the pattern's counselling sources plus the training-split figures.
- **Proposer:** the one rule edit written for that pattern.
- **Gate:** seeker ratings on the validation split, for conversations that follow the new rule against those that
  break it. Noise is the range of mean quality over 3 seeded thirds of the validation split (0.20). No
  conversations or safety cases were run; only the Safety section was checked unchanged. The test split is unused.

| Round | Outcome | Pattern | Validation quality: break rule → follow rule |
|---|---|---|---|
| 1 | reverted | no-reflection-of-feelings | 4.58 (60) → 4.51 (114) |
| 2 | no_pattern | | |
| 3 | kept → `policy-v2` | no-reassurance | 4.35 (17) → 4.55 (157) |
| 4 | no_pattern | | |
| 5 | reverted | suggestion-heavy | 4.68 (11) → 4.53 (163); relevance fell 0.44 |
| 6–8 | no_pattern | | |

Result: `policies/policy-v2.md` adds one rule, C9 (affirm the person's effort or strengths at least once).
Its gain (0.20) is only just above the noise (0.199), and ESConv ratings are near the top of the scale
(mean 4.5 of 5), so treat it as weak evidence. The comparison is between existing human sessions, not a test of
the helper following the rule.

Files:
- `policies/policy-v2.md` and `policy-v2.meta.json`: the policy in force and where each rule came from
- `log/decisions.jsonl`: the decision log
- `records/round-NN/`: the pattern, research, edit and gate records for each round with a pattern
