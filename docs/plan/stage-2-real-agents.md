# Stage 2: real agents in place of the dummies

**Target:** 21:00. **Owner:** one per lane, unassigned. **Needs:** stages 0 and 1. **Blocks:** stage 3.

## Goal

The judge and the gatekeeper exist, and every agent's prompt has been tuned against cases a person has checked. The lanes below do not depend on each other, because stage 0 fixed what each node receives and returns.

## Lane A: the judge (issues #2, #3, #4)

Spec: [docs/agents/t3-judge.md](../agents/t3-judge.md).

- [ ] `puddles/agents/judge.py` with the five functions. Input is `JudgeInput` (turns only). Output is `JudgeOutput`. `to_record` makes a `ScoreRecord`; code adds `conversation_id` and `judge_version`.
- [ ] A helper that turns a `ConversationRecord` and its `ScoreRecord` into the `ScoredConversation` the critic takes.
- [ ] The judge model comes from its own variable, with the version pinned and temperature 0.
- [ ] ESConv loader (#2): scenarios by split, and real conversations with the users' own ratings. Loaded at run time, never committed. The test split needs an explicit argument.
- [ ] `scripts/calibrate_judge.py`: runs the judge on about 100 real ESConv conversations from the validation split and prints rank correlation and mean absolute error against the users' ratings, for empathy and for relevance.
- [ ] Tune the prompt, re-run, and stop when a change no longer helps. Then freeze: write the `judge_version` and the two agreement numbers into this file.

**Order of fields.** `JudgeOutput` has the scores first and `note` last. Rule 3 in the [plan README](README.md) says explanation first. Moving `note` to the top changes a shared format, so settle it on issue #1 before calibrating, not after.

## Lane B: the gatekeeper (issues #5, #20)

Spec: [docs/agents/t7-gatekeeper.md](../agents/t7-gatekeeper.md). It is code. It has no prompt apart from one yes-or-no question per `judged` safety assertion.

- [ ] `fixtures/safety-v1.json`: safety cases as `SafetyCase`, covering rules S1, S2 and S3. Written by hand or taken from CounselBench-Adv openers at run time; no scraped posts.
- [ ] `puddles/gate.py`: takes a `GateInput`, runs the validation scenarios under the candidate policy (converse, then judge), runs the safety cases against the helper alone, and applies the six checks in the spec. Returns a `GateOutcome` with real `Scores` and a `reason` that names the check that decided it.
- [ ] Replaces `stand_in_gate` in the graph. Keep the stand-in for tests.
- [ ] Until stage 5 has measured the noise, the gate takes the noise as an argument and writes "provisional noise" in its reason.

## Lane C: the critic, researcher and proposer

They are built. This lane is tuning.

- [ ] For each agent, 5 to 10 hand-checked cases in `fixtures/`: an input and what a good answer must contain. `fixtures/bad_conversations.json` is the start for the critic.
- [ ] `scripts/try_agent.py <agent> <case file>`: runs the real model on each case and prints the answer beside the expectation.
- [ ] Change the prompt in `docs/agents/<agent>.md` (the code reads it from there), re-run, keep what helps.
- [ ] One live Exa search through the researcher, with the returned URLs checked against what Exa sent.

## Lane D: the helper and the simulated user

They are built. This lane is tuning the persona.

- [ ] Run the 5 fixture scenarios with the real helper and simulated user models: `python -m loop.run_batch --policy policies/policy-v1.md --scenarios fixtures/scenarios.jsonl --round 0 --out runs/dev/conversations.jsonl`.
- [ ] Read all five. Check: the resistant scenario stays resistant, nobody feels better after one reply, helper replies are one or two spoken sentences, and the simulated user ends with `[END]` at a natural point.
- [ ] Change `loop/prompts/*.txt` and the matching file in `docs/agents/` together.

## Tests that must pass

1. The judge's offline tests: a scripted answer becomes a valid `ScoreRecord`; an out-of-range score is retried once; the rendered input contains no IDs.
2. The gate's offline tests, one per check: a changed Safety section, two rules changed, over the token cap, a failed safety case, a gain smaller than the noise, one dimension falling. Each must give `reverted` with the right reason. One clean case must give `kept`.
3. A reverted candidate leaves the policy file in force byte-for-byte unchanged.
4. Every agent has made one real call that passed its checks.

## Not in this stage

- No full round with real models. That is stage 3.
- No overnight settings.

## Done when

The graph from stage 0 runs offline with the real judge and gate code in place of the dummies, and each lane's list above is ticked.
