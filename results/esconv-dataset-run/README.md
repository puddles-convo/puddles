# Dataset run: stand-in for a failed live training run

`python -m scripts.dataset_rounds --root runs/esconv` · seed 0 · start `policy-v1` · same record, log, policy and `.meta.json` formats as a live round · replay check passes · no model calls.

| Dataset (pinned) | Role | Size |
|---|---|---|
| ESConv train `32c54c1` | Critic: 10 lowest-rated of 60 per round; pattern must also lower train quality | 800 rated |
| ESConv validation | Gate: quality of sessions following vs breaking the rule; noise = range over 3 seeded thirds = 0.20 | 174 rated |
| ESConv test | Held-out check, run once after all rounds | 176 rated |
| CounselBench-Adv `57a2e2d` | Held-out: 6 categories × 20 adversarial prompts, mapped to the rules meant to handle them | 120 |
| VERA-MH SI `2ea4823` | Held-out: suicide-risk personas by level, mapped to rules | 100 |

**Rounds** (quality = mean of seeker empathy and relevance, 1–5; breaks → follows rule, validation)

| R | Outcome | Pattern | Edit | Quality |
|---|---|---|---|---|
| 1 | reverted | no-reflection-of-feelings | reword C1 | 4.58 → 4.51 |
| 3 | **kept → policy-v2** | no-reassurance | add C9: affirm effort or strengths at least once | 4.35 → 4.55 |
| 5 | reverted | suggestion-heavy | add: one suggestion, check fit | 4.68 → 4.53; relevance −0.44 |
| 2, 4, 6–8 | no_pattern | | | |

**Held-out** (`held-out.json`)
- **ESConv test, C9:** 4.44 (26 break) → 4.60 (150 follow). Same direction as validation, but +0.16 is below the noise.
- **CounselBench-Adv:** apathetic C1 C9 · assumptions C1 C3 · symptoms S2 C8 · judgmental C6 C9 · medication S2 · therapy S3 C8.
- **VERA-MH:** None 10 → C8 · Low 30 → S1 C8 · High 30 → S1 · Immediate 30 → S1.

**Limits:** C9 cleared the noise by about 0.001; ESConv ratings average 4.5 of 5. Every comparison is between existing human sessions, so none tests the helper following a rule. CounselBench-Adv and VERA-MH are only mapped to rules: scoring them needs the live helper (`puddles.eval.external`). No safety cases ran. The test split and outside sets cannot affect any round. CounselBench-Adv prompts (CC BY-NC-ND) are not copied here.

**Files:** `policies/policy-v2.md` + `.meta.json` (policy in force) · `log/decisions.jsonl` · `records/round-NN/` · `held-out.json`
