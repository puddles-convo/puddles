# Memory: what we have, what others do, and what to change

14 November 2026. This is the manager's synthesis of the four reports in this folder. The synthesis itself has not been independently reviewed; the reports it rests on have been (see "How this was checked").

| File | What it is | Written by | Checked by |
|---|---|---|---|
| [01-agent-memory-and-cogsci.md](01-agent-memory-and-cogsci.md) | Agent memory systems and cognitive science | Claude Opus 5.5 | Codex (04) |
| [02-therapist-memory-and-schema-learning.md](02-therapist-memory-and-schema-learning.md) | How helpers keep track; learning the structure from data | Claude Opus 5.5 | Codex (04) |
| [03-current-design-audit.md](03-current-design-audit.md) | Audit of our own design, with `path:line` references | Codex Sol | Manager spot-check, 6 claims |
| [04-verification.md](04-verification.md) | Fact-check of 01 and 02 against the live sources | Codex Sol | Not reviewed |

## 1. Updating memory without it exploding

The systems that stay bounded share one shape:

1. **An append-only record that never goes into a prompt whole.** It can grow without limit because it only costs storage.
2. **A small working set that does go into prompts.** It is a list of items with IDs, changed by small per-item operations (add, update, delete, confirm) that code applies. It is never rewritten as one summary.
3. **Evidence decides what stays in the working set:** a test result, a counter, or a date last confirmed.

ACE is the clearest source for item 2. It reports a case where letting the model rewrite its whole context shrank it from 18,282 tokens to 122 and dropped accuracy from 66.7 to 57.1, below the 63.7 baseline. Its fix is bullets with IDs and helpful/harmful counters, with deltas merged by code. ExpeL does the same with vote counts and removes an insight at zero. Mem0 does it for user facts with ADD / UPDATE / DELETE / NOOP against the ten most similar memories.

Clinical record-keeping has a similar split: session notes (SOAP, DAP, BIRP) are appended, while the formulation, goals and safety plan are revised as things change. That is an analogy, not evidence.

**Where we stand**

| Part of our design | Matches this shape? |
|---|---|
| Decision log → policy file (one tested edit per round, applied by code, token cap) | Yes. This is the ACE design with a stricter gate. |
| Training memory (four add-only entries per round) | Yes for the store. About 200 short entries after 50 rounds. |
| History lists put into the critic, researcher and proposer prompts ("complete list, built from the log") | No. They grow with every round. Small now (about 840 tokens at 10 rounds, 16,800 at 200, by the audit's assumptions). |
| User notes | No. One flat list, no cap, all of it in the helper prompt. At three net notes a session: 150 notes and about 4,500 tokens after 50 sessions, 45,000 after 500. |

So the training side will not explode during the hackathon. The user side is the real gap.

## 2. Are the concepts enough?

**Training side: the four record types are the right ones.** Failure pattern, research note, policy edit and gate outcome are the same "scored experiment store" that ACE, GEPA, AlphaEvolve and ExpeL keep. What is missing is attributes, plus one optional type:

| Missing | Why it matters | Source |
|---|---|---|
| A canonical pattern identity, separate from each round's occurrence | Today a recurrence reuses the earlier record's ID. That collides with add-only storage, and `copy_status` would treat the new copy as already confirmed. A renamed duplicate also never becomes `recurring`. | Audit, section 5 |
| Scope: which problem type, emotion or openness a pattern or rule applies to | Scenario labels exist but gate scores are one overall mean, so a rule that helps "job crisis" and hurts "ongoing depression" looks neutral. | Audit, section 4 |
| Evidence over time per rule (does a kept rule keep helping) | Nothing tells the proposer which rule to drop at the token cap. | ACE counters, ExpeL votes |
| What worked (positive patterns), optional | Only failures are recorded. ExpeL compares successes with failures. | ExpeL |

**User side: one untyped "note" is too coarse.** It cannot be capped or expired sensibly. Proposed kinds, taken from what a non-clinical helper may properly keep:

| Kind | Holds | Updated by | Cap |
|---|---|---|---|
| `situation` | What they are dealing with now | Replace when it changes | 3 |
| `people` | One note per person and their relation | Replace that person's note | 6 |
| `what_helps` | What helped or did not | Replace on new evidence | 5 |
| `goals` | Plans the person stated | Replace, delete when done | 3 |
| `follow_up` | A dated thing to ask about | Expires after its date | 3 |
| `preferences` | How they like to be talked to | Replace | 3 |

Each note also gets `last_confirmed`, set by a new `confirm` action. Deliberately left out: diagnoses, scores, beliefs in the CBT sense, risk history, medication, verbatim disclosures. That exclusion list is a product and safety decision, not a literature finding; a clinician should confirm the risk-history line.

The kinds, caps and expiry are our choices. No source tested them, and the verification found no evidence that categorised notes improve answer quality. What they give is a hard upper bound (23 notes, roughly 500 to 700 tokens) and a principled choice of what to drop.

## 3. Learning the structure from the dataset

| Structure | Can ESConv teach it? | How |
|---|---|---|
| Failure-pattern taxonomy | Yes | See below |
| User-note kinds | Partly: it can check coverage, not discover update rules | ESConv is single-session, so it shows what people disclose, not what matters next time or goes stale |
| Update and decay rules for notes | No | Needs multi-session data. EvoEmo (the ES-MemEval dataset: 18 virtual users, up to 33 sessions, seeded from ESConv) tests exactly this: knowledge updates and conflict detection. Licence not checked. |

What ESConv has for the failure taxonomy:

- A strategy label on every supporter turn. Checked directly in the Hugging Face release: 1,429 labelled supporter turns in the first 100 dialogues.
- Seeker empathy and relevance ratings per conversation.
- Seeker feedback scores after every two supporter turns, in the raw GitHub `ESConv.json` only. The Hugging Face preprocessing drops them.
- `FailedESConv.json`: 196 rejected conversations.

Procedure, about an hour:

1. Take low-rated segments from the raw JSON, plus FailedESConv, plus our own low-scoring simulated conversations.
2. Have a model write one sentence per segment: what the helper did that made it worse, and which strategy was involved.
3. Induce a taxonomy from those sentences three times on different shuffles (the TnT-LLM method), each code with a definition and inclusion criteria (the LLooM idea).
4. Keep a code only if it appears in at least two of three runs and is clearly more common in low-rated than high-rated conversations.
5. Give the critic the resulting list as its allowed pattern names, plus "new: ...". That replaces free-text names, which is what makes recurrence matching fragile today.

A low rating marks a segment, not one guilty utterance.

## 4. Can we do better? Ranked

| # | Change | Fixes | Hours |
|---|---|---|---|
| 1 | Give user notes a `kind`, `last_confirmed`, per-kind caps enforced by code, a `confirm` action and follow-up expiry | Unbounded helper prompt | 1 to 2 |
| 2 | Budget the history lists in prompts: everything for this pattern, the last 10 overall, one line of counts for the rest. Code checks exact duplicates against the full log. | Prompt growth per round | 0.5 to 1 |
| 3 | Separate canonical pattern ID from occurrence ID; critic chooses from the induced taxonomy or "new" | ID collision, missed recurrences | 1 to 2 |
| 4 | Run the taxonomy induction in section 3 | Hand-picked pattern names | 1 |
| 5 | Record gate scores per problem type as well as overall | Hidden conditional harm | 1 |
| 6 | Reject a candidate rule that is near-identical to a reverted edit for the same pattern; park a pattern after three reverts | Repeated failed edits | 1 to 2 |
| 7 | Per-rule helpful/harmful counters to choose removals at the cap | Policy bloat | 3 to 4, after the hackathon |
| 8 | Merging old notes into summaries | Very long-term users | Defer. Gist merging can create false facts. If built, merge two notes at a time and cite both source IDs. |

Two things not to do: do not let any model rewrite the policy or the notes wholesale, and do not keep "invalidated" user notes for history (forgetting must really delete).

## 5. Limits of this review

- Memory structure is unlikely to raise our scores at this scale. Mem0's own paper has full context beating its memory on LOCOMO (72.90 against 66.88) while cutting tokens by over 90%. The honest claim for the past.dev challenge is bounded cost and traceability, not accuracy.
- Most of the memory design is not in code yet. Under `loop/` there is policy loading, note rendering, the batch runner and the log. There is no memory client, note lifecycle or log replay.
- Verification corrected two overreaches in the reports. ACE argues against wholesale rewriting, not against all merging. And nothing shows that comparing a new fact only with notes of the same kind is enough; the note-taker should keep seeing all notes, which is cheap under the caps.
- Not covered in depth: Voyager, A-MEM, Agent Workflow Memory, Reflective Memory Management, MemoryBank's full mechanism. Clinical sources for note formats are vendor guides. Reconsolidation and schema theory had no primary source opened.

## How this was checked

- All 27 arXiv IDs cited in 01 and 02 resolve to the stated papers, and the numbers quoted above were confirmed against the papers by Codex (04).
- Six claims from the audit (03) were checked against the files by the manager and held.
- The ESConv strategy-label count was checked directly against the Hugging Face rows API.
