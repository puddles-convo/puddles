# Stage 2: real agents in place of the dummies

**Target:** 21:00. **Owner:** one per lane, unassigned. **Needs:** stages 0 and 1. **Blocks:** stage 3.

## Goal

The judge and the gatekeeper exist, and every agent's prompt has been tuned against cases a person has checked.

Lanes A, C and D do not depend on each other. **Lane B needs lane A's judge and scenario loader to run for real.** Build the gate against a scripted judge and the fixture scenarios first, and join it to the real judge once lane A is frozen.

## Lane A: the judge (issues #2, #3, #4)

Spec: [docs/agents/t3-judge.md](../agents/t3-judge.md).

- [ ] `puddles/agents/judge.py` with the five functions. Input is `JudgeInput` (turns only). Output is `JudgeOutput`. `to_record` makes a `ScoreRecord`; code adds `conversation_id` and `judge_version`.
- [ ] A second failed answer leaves that conversation unscored. The round counts it; it does not stop.
- [ ] The training conversations' scores are saved to `<root>/scores/round-NN-train.jsonl`, one `ScoreRecord` per line.
- [ ] A helper that turns a `ConversationRecord` and its `ScoreRecord` into the `ScoredConversation` the critic takes.
- [ ] `select_for_critic(scored, n)`: the `n` lowest by (empathy + relevance) / 2, ties broken by conversation ID. The critic's spec says about ten. Today `critic.build_input` passes on whatever it is given, so without this the critic reads every conversation.
- [ ] The judge model comes from its own variable, with the version pinned and temperature 0.
- [ ] ESConv loader (#2): scenarios by split, and real conversations with the users' own ratings. Loaded at run time, never committed. The test split needs an explicit argument.
- [ ] `scripts/calibrate_judge.py`: runs the judge on about 100 real ESConv conversations from the validation split and prints rank correlation and mean absolute error against the users' ratings, for empathy and for relevance. It saves the pairs of scores (conversation ID, judge score, user score; no text) to `runs/calibration/pairs.jsonl`, which the scatter plot in stage 6 reads.
- [ ] Tune the prompt, re-run, and stop when a change no longer helps. Then freeze: write the `judge_version` and the two agreement numbers into this file.

**Order of fields.** `JudgeOutput` has the scores first and `note` last. Rule 3 in the [plan README](README.md) says explanation first. Moving `note` to the top changes a shared format, so settle it on issue #1 before calibrating, not after.

## Lane B: the gatekeeper (issues #5, #20)

Spec: [docs/agents/t7-gatekeeper.md](../agents/t7-gatekeeper.md). It is code. It has no prompt apart from one yes-or-no question per `judged` safety assertion.

- [ ] `fixtures/safety-v1.json`: safety cases as `SafetyCase`, covering rules S1, S2 and S3, **written by hand**. Do not take them from CounselBench-Adv or VERA-MH. Those are the outside check, and the gate's decisions shape the policy, so a case taken from them would let the loop see them.
- [ ] `puddles/gate.py`: takes a `GateInput`, runs the validation scenarios under the candidate policy (converse, then judge), runs the safety cases against the helper alone, and applies the six checks in the spec. Returns a `GateOutcome` with real `Scores` and a `reason` that names the check that decided it.
- [ ] **The gate's conversations never overwrite the round's training evidence.** `run_batch` writes its file with `write_text`, and a conversation ID is `conv-rNN-<scenario>`, so a second batch in the same round on the same scenario would replace both the file and the ID. Give `run_batch` and `run_conversation` a tag that goes into the ID (`conv-rNN-val-<scenario>`), and save the gate's batches as `conversations/round-NN-validation.jsonl` and `scores/round-NN-validation.jsonl`. The scoring of the current policy before round 1 uses round 0 and the tag `current`.
- [ ] **Complete evidence or revert.** Every validation scenario must come back with a score. A failed conversation, an unscored conversation or a crashed run reverts the candidate, with the reason "incomplete evidence". Averaging the conversations that survived can look like a gain.
- [ ] A yes-or-no answer that is neither word is asked again once, then the assertion counts as failed.
- [ ] `GateInput` needs `validation_scenario_ids` and `current_scores`. The loop keeps the current policy's scores: measured once before round 1, then replaced by the candidate's scores each time an edit is kept.
- [ ] **No promotion on provisional noise.** The gate takes the noise and a `noise_measured` flag. While the flag is false it runs every check, writes what it would have decided in the reason, and returns `reverted`. The spec says nothing is promoted until the noise has been measured. Stage 5 turns the flag on.
- [ ] `finish_round` takes the scores, the noise and the safety results, and writes them into the round event as `scores` (`LogScores`) and `safety` (`LogSafety`). Today it leaves both empty, and the plots need them.
- [ ] `policies/<version>.meta.json`, as `PolicyMeta`, exists for every policy that is ever in force:
  - for the starting policy, written when a run's root folder is first set up: no parent, every rule a base rule, no learned rules;
  - for each promoted policy, written by `finish_round` when an edit is kept: the parent version, the policy hash, the same Safety hash, the base rules, and for each learned rule its edit, pattern, round and text hash.
  A reverted round writes none; the policy in force keeps the one it has.
- [ ] Replaces `stand_in_gate` in the graph. Keep the stand-in for tests.

## Lane C: the critic, researcher and proposer

They are built. This lane is tuning.

- [x] For each agent, 5 to 10 cases in `fixtures/cases/`: an input and what a good answer must contain (6 for the critic, 5 each for the researcher and the proposer). `fixtures/bad_conversations.json` (four conversations) is the start for the critic. Contract: [docs/specs/case-files.md](../specs/case-files.md).
- [ ] The cases are hand-checked. Open: they were written by a model as drafts, every one has `reviewed_by` null, and a person has to read them. `fixtures/cases/README.md` lists the one-word checks to look at first.
- [x] `scripts/try_agent.py <agent> <case file>`: runs the real model on each case and prints the answer beside the expectation. Only `--dry-run` and scripted models have been run; no real model call has been made. "Met its expectations" covers the expectations written in the case, not the agent's own checks; the README says which those are.
- [ ] Change the prompt in `docs/agents/<agent>.md` (the code reads it from there), re-run, keep what helps.
- [ ] One live Exa search through the researcher, with the returned URLs checked against what Exa sent.

## Lane D: the helper and the simulated user

They are built. This lane is tuning the persona.

- [ ] Run the 5 fixture scenarios with the real helper and simulated user models: `python -m loop.run_batch --policy policies/policy-v1.md --scenarios fixtures/scenarios.jsonl --round 0 --out runs/dev/conversations.jsonl`.
- [ ] Read all five. Check: the resistant scenario stays resistant, nobody feels better after one reply, helper replies are one or two spoken sentences, and the simulated user ends with `[END]` at a natural point.
- [ ] Change `loop/prompts/*.txt` and the matching file in `docs/agents/` together.
- [ ] **Freeze the helper prompt at the end of this lane** and tell whoever has stage 4. The robot runs the same prompt, and stage 4's final check must be run against the frozen one.

## Tests that must pass

1. The judge's offline tests: a scripted answer becomes a valid `ScoreRecord`; an out-of-range score is retried once; a second failure leaves the conversation unscored and counted; the rendered input contains no IDs.
2. `select_for_critic` returns the lowest `n` in a fixed order, and all of them when there are fewer than `n`.
3. The gate's offline tests, each giving `reverted` with the right reason:
   - a changed Safety section; two rules changed; over the token cap; a failed safety case;
   - a gain smaller than the noise, and a gain exactly equal to the noise (the rule is "greater than");
   - one dimension falling by more than the noise;
   - one validation conversation failed; one left unscored; all failed;
   - a yes-or-no answer that is neither word, twice.
4. The gate's offline tests giving `kept`: a clean case; and one dimension falling by less than the noise, or by exactly the noise, while quality rises by more than the noise.
5. With `noise_measured` false, the clean case gives `reverted`, the reason says what the checks found, and no policy file is written.
6. A reverted candidate leaves the policy file in force byte-for-byte unchanged.
7. A kept round's event has `scores` and `safety` filled in, and the `.meta.json` validates as `PolicyMeta` with the Safety hash of the starting policy. A new root has the starting policy's `.meta.json` before round 1. A reverted round adds no `.meta.json`.
8. After a round that reached the gate, the training conversations file and the validation file both exist, and no conversation ID appears in both.
9. Every agent has made one real call that passed its checks.

## Not in this stage

- No full round with real models. That is stage 3.
- No overnight settings.

## Done when

The graph from stage 0 runs offline with the real judge and gate code in place of the dummies, and each lane's list above is ticked.
