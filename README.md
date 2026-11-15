# puddles

A support-conversation helper that improves its own behaviour, built at the SF AI Research Hackathon (November 14-15, 2026), Track 5 (Open).

The helper listens, reflects, asks questions, suggests coping techniques and points people to professional help. It does not diagnose or treat.

**Status:** this README is the agreed design. No pipeline code is in this repo yet. Scores, counts and sources in the examples are illustrative.

- One file per agent, with its prompt, where the input goes in it, its input and output formats, and its memory reads and writes: [docs/agents/](docs/agents/README.md)
- The design behind memory, policy and the crisis path: [docs/agents.md](docs/agents.md)
- Record formats: `schemas.py` (version 2), with an example of each in `examples/`. They are a proposal for the team to settle under [issue #1](https://github.com/puddles-convo/puddles/issues/1).
- The device and its cloud brain: [puddles-stack](https://github.com/puddles-convo/puddles-stack)

## Contents

1. [Training steps](#1-training-steps)
2. [Training agents](#2-training-agents)
3. [Inference steps](#3-inference-steps)
4. [Inference agents](#4-inference-agents)
5. [The memory](#5-the-memory)
6. [From memory to policy](#6-from-memory-to-policy)
7. [The proposer's toolbox](#7-the-proposers-toolbox)
8. [Handoffs](#8-handoffs)
9. [How training and inference connect](#9-how-training-and-inference-connect)
10. [How we measure the result](#10-how-we-measure-the-result)
11. [Data](#11-data)
12. [Robot side](#12-robot-side)
13. [Related work](#13-related-work)
14. [Improvements from the literature](#14-improvements-from-the-literature)

## 1. Training steps

Training is the offline loop that improves the policy. It runs in the cloud in rounds, with no human in it.

```
TRAINING  (one round; down the left is what each agent hands the next)
+------------------------------+                                        +----------------------+
| 1. SIMULATED USER            |     (no access)                        |                      |
| plays a person in distress   |                                        |                      |
+------------------------------+                                        |                      |
              |  user message                                           |                      |
              v                                                         |                      |
+------------------------------+                                        |                      |
| 2. HELPER                    |     (no access during training)        |                      |
| replies, following policy    |                                        |                      |
+------------------------------+                                        |                      |
              |  conversation, turns only                               |                      |
              v                                                         |                      |
+------------------------------+                                        |                      |
| 3. JUDGE                     |     (no access)                        |                      |
| rates the conversation       |                                        |                      |
+------------------------------+                                        |                      |
              |  scores and notes                                       |                      |
              v                                                         |                      |
+------------------------------+                                        |   TRAINING MEMORY    |
| 4. CRITIC                    | <- reads: past patterns, outcomes ---- |      (past.dev)      |
| names the main failure       | -- writes: failure pattern ----------> |                      |
+------------------------------+                                        |    shared by the     |
              |  failure pattern, or none found                         |   training agents    |
              v                                                         |                      |
+------------------------------+                                        |                      |
| 5. RESEARCHER                | <- reads: past searches -------------- |                      |
| finds how counsellors fix it | -- writes: sources and technique ----> |                      |
+------------------------------+                                        |                      |
              |  sources, or none found                                 |                      |
              v                                                         |                      |
+------------------------------+                                        |                      |
| 6. PROPOSER                  | <- reads: past edits and outcomes ---- |                      |
| writes one policy edit       | -- writes: the proposed edit --------> |                      |
+------------------------------+                                        |                      |
              |  candidate policy                                       |                      |
              v                                                         |                      |
+------------------------------+                                        |                      |
| 7. GATEKEEPER (code)         | -- writes: outcome, from the log ----> |                      |
| tests it; keeps or reverts   |                                        |                      |
+------------------------------+                                        |                      |
              :  policy in force; the next round starts again at 1      |                      |
              :                                                         +----------------------+
```

- **Down the left** is the order of the steps and what each agent hands straight to the next.
- **Across** is what each agent reads from and writes to the shared memory.
- **A round can also end early:** with no pattern found, or no usable source found. Then no edit is made.

## 2. Training agents

The agent names link to a file for each one, with its prompt and formats.

| Agent | What it does | Pulls from memory | Writes to memory | Example input | Example output |
|---|---|---|---|---|---|
| [**1. Simulated user**](docs/agents/t1-simulated-user.md) | • Plays a person from an ESConv scenario<br>• Reacts honestly and opens up slowly | Nothing (no access) | Nothing | Scenario: anxiety, job crisis, "I am on short term disability and I am afraid I will lose my job"<br>plus the helper's last line | `Not yet. I'm on short term disability and I'm not ready to go back.` |
| [**2. Helper**](docs/agents/t2-i3-helper.md) | • Replies to the person, following the policy file<br>• Same model, prompt and code as on the robot | Nothing during training | Nothing | `policy-v4` and the turns so far:<br>user: "I'm feeling anxious that I'm going to lose my job." | `That's a heavy thing to be carrying. What's making you feel the job is at risk right now?` |
| [**3. Judge**](docs/agents/t3-judge.md) | • Rates each finished conversation<br>• Blind to which policy produced it, and frozen after calibration | Nothing (no access) | Nothing. Its scores go to the critic and the decision log. | The turns of one conversation, with no IDs and no policy version | `{"empathy": 3, "relevance": 2, "emotion_before": 4, "emotion_after": 4, "note": "Suggested HR before asking anything."}` |
| [**4. Critic**](docs/agents/t4-critic.md) | • Reads the lowest-scoring conversations in full<br>• Names the one failure that explains most of them | Earlier patterns, and what happened to each | The failure pattern | About ten of the lowest-scoring conversations with the judge's notes, and the current policy | `{"found": true, "name": "advice-before-understanding", "seen_in": 2, "out_of": 3}` |
| [**5. Researcher**](docs/agents/t5-researcher.md) | • Searches Exa for how counsellors handle that pattern<br>• Keeps sources, each with a one-line technique | Searches already run | The sources and techniques | The failure pattern | `{"sources": [{"url": "(from Exa)", "technique": "Ask one open question and reflect the answer before any suggestion."}]}` |
| [**6. Proposer**](docs/agents/t6-proposer.md) | • Picks exactly one action from its [toolbox](#7-the-proposers-toolbox): a harness change or, on escalation, a model update<br>• States a hypothesis and a prediction the gate can check<br>• Does not repeat an action that was reverted | Earlier actions and their outcomes, and the lever scoreboard | The proposed action | The current harness version, the pattern, the sources, the scoreboard and the remaining budget | `{"action": "edit_rule", "args": {"op": "add", "rule_text": "In your first two replies, ask one open question about their situation and do not give suggestions yet, because advice that comes before understanding rarely fits."}, "hypothesis": "...", "prediction": "relevance +0.2"}` |
| [**7. Gatekeeper**](docs/agents/t7-gatekeeper.md) (code, no model) | • Tests the candidate on validation conversations and safety cases<br>• Keeps it, or restores the previous policy | Nothing | The outcome, copied from the decision log | The candidate policy, the current scores and the noise level | `{"decision": "kept", "policy_in_force": "policy-v4", "safety_passed": true}` |

## 3. Inference steps

Inference is the live conversation with a person. The policy is fixed for the whole conversation.

```
INFERENCE  (2 and 3 start together; RELEASE decides what is spoken)
+------------------------------+                                        +----------------------+
| 1. LISTENER                  |     (no access)                        |                      |
| turns speech into text       |                                        |                      |
+------------------------------+                                        |                      |
              |  user message, to 2 and 3 at the same time              |                      |
              v                                                         |                      |
+------------------------------+                                        |                      |
| 2. CRISIS CHECKER            |     (no access)                        |                      |
| flags risk to life or safety |                                        |                      |
+------------------------------+                                        |                      |
              :  runs side by side with                                 |                      |
              :                                                         |                      |
+------------------------------+                                        |                      |
| 3. HELPER                    | <- reads: notes about this person ---- |                      |
| drafts the reply             |                                        |     USER MEMORY      |
+------------------------------+                                        |      (past.dev)      |
              |  risk level from 2, draft reply from 3                  |                      |
              v                                                         |  one private space   |
+------------------------------+                                        |      per person      |
|    RELEASE (code)            |     (no access)                        |                      |
| holds audio until 2 answers  |                                        |                      |
+------------------------------+                                        |                      |
              |  the draft, or a fixed response                         |                      |
              v                                                         |                      |
+------------------------------+                                        |                      |
| 4. SPEAKER                   |     (no access)                        |                      |
| turns the reply into speech  |                                        |                      |
+------------------------------+                                        |                      |
              |  the finished session                                   |                      |
              v                                                         |                      |
+------------------------------+                                        |                      |
| 5. NOTE-TAKER                | <- reads: existing notes ------------- |                      |
| keeps notes on this person   | -- writes: note changes -------------> |                      |
+------------------------------+                                        +----------------------+
```

- **The crisis checker and the helper start together**, so the check adds no delay.
- **Nothing is spoken until the crisis checker has answered.** On a crisis the helper's draft is discarded.
- **With no connection** the robot cannot understand speech, so it plays a line stored on the device that gives the 911 and 988 numbers.

## 4. Inference agents

| Agent | What it does | Pulls from memory | Writes to memory | Example input | Example output |
|---|---|---|---|---|---|
| [**1. Listener**](docs/agents/i1-listener.md) (speech model) | • Turns the person's speech into text | Nothing | Nothing | About 3 seconds of audio | `I'm feeling anxious that I'm going to lose my job.` |
| [**2. Crisis checker**](docs/agents/i2-crisis-checker.md) | • Checks every message for risk to life or safety<br>• Runs at the same time as the helper | Nothing | Nothing | The latest message and the last few turns:<br>"Sometimes I wonder if things would be better if I wasn't around any longer." | Level `crisis`, category `self`, reason "says things would be better if they weren't around" |
| [**3. Helper**](docs/agents/t2-i3-helper.md) | • Drafts the reply, following the policy file<br>• Same model, prompt and code as in training | Notes about this person, once at the start of the session | Nothing | `policy-v4`, the notes ("on short-term disability and worried about losing their job") and the turns so far | `That's a heavy thing to be carrying. What's making you feel the job is at risk right now?` |
| [**Release**](docs/agents/i3b-release.md) (code, no model) | • Holds all audio until the crisis checker answers<br>• Crisis: fixed response. Checker failed: holding line. Otherwise: the draft. | Nothing | Nothing | The risk level and the helper's draft | The draft, or a fixed response such as "I'm really glad you told me that. Are you thinking about ending your life? ... You can call or text 988 ..." |
| [**4. Speaker**](docs/agents/i4-speaker.md) (speech model) | • Turns the reply into speech, a sentence at a time | Nothing | Nothing | The reply text | Audio |
| [**5. Note-taker**](docs/agents/i5-note-taker.md) | • After the session, decides what to remember about this person<br>• Adds notes only if the person agreed; can always delete | The existing notes about this person | Notes added, replaced or deleted | The finished session and the existing notes | `[{"action": "add", "fact": "Slow breathing helps a bit when the worry gets bad."}]` |

## 5. The memory

### Three stores

| Store | Where | What it is |
|---|---|---|
| Decision log | A file in this repo, append-only | The trusted record of every round. Code reads it. |
| Training memory | past.dev, audience `training-agents` | A searchable copy of what the log holds, which the training agents ask questions of. |
| User memory | past.dev, one audience per person | What the robot remembers about that person. |

### What is in the memory

| Item | Store | Written by | Read by | What it keeps track of |
|---|---|---|---|---|
| Failure pattern | Training | Critic | Critic | What the helper keeps getting wrong, and how often |
| Research note | Training | Researcher | Researcher | What has been searched, and the techniques found |
| Policy edit | Training | Proposer | Proposer | Every change that was proposed |
| Gate outcome | Training | Gatekeeper | Critic, Proposer | Which edits were kept or reverted, with the scores and the reason |
| User note | User | Note-taker | Helper, Note-taker | One fact about one person, to carry into the next conversation |

Examples of each:

| Item | Example entry |
|---|---|
| Failure pattern | Failure pattern pattern-r03, advice-before-understanding, round 3: the helper gives suggestions in its first reply without asking about the person's circumstances. Seen in 2 of the 3 lowest-scoring conversations reviewed. |
| Research note | Research research-r03 for pattern-r03: counselling skills guidance says to ask one open question and reflect the answer before any suggestion. Source: (the page Exa returned). |
| Policy edit | Edit edit-r03 for pattern-r03: add rule C2, ask an open question and hold suggestions for the first two replies. |
| Gate outcome | Edit edit-r03 was kept in round 3 and is rule C2 of policy-v4. Quality rose from 3.25 to 3.75, above the noise of 0.15. All safety cases passed. |
| Gate outcome | Edit edit-r05 (add a rule offering three coping options in each reply) was reverted in round 5: quality fell from 3.75 to 3.60. policy-v4 stays in force. |
| User note | 15 Nov 2026: slow breathing (in for four, out for six) helps a bit when the worry gets bad. |

### The shape of one memory entry

Every entry, in either past.dev memory, has the same shape:

```json
{
  "id": "pattern-r03",
  "content": "Failure pattern pattern-r03, advice-before-understanding, round 3: the helper gives suggestions in its first reply without asking about the person's circumstances. Seen in 2 of the 3 lowest-scoring conversations reviewed.",
  "label": "critic: failure pattern, round 3",
  "timestamp": "2026-11-14T22:12:40Z",
  "identity": "critic",
  "audience": "training-agents",
  "metadata": {"type": "failure_pattern", "round": 3,
               "record_path": "records/round-03/pattern-r03.json", "sha256": "(hash of the record file)"}
}
```

| Field | What it holds |
|---|---|
| `id` | Our own record ID. The same ID names the record file and appears in the decision log. |
| `content` | A readable sentence. This is what a search matches, so it carries the IDs and the key facts. |
| `label` | A short title: which agent, which kind of item, which round. |
| `timestamp` | When it happened. |
| `identity` | The agent that produced it: `critic`, `researcher`, `proposer`, `gatekeeper` or `note-taker`. |
| `audience` | Who may read it: `training-agents`, or one person's private audience. |
| `metadata` | The item type, the round, and a pointer to the full record file with its hash. |

### Who is in control

- **No agent is in charge of the memory.** The orchestrator is: the plain code that runs a round. An agent returns its result, and the orchestrator saves it as a record file, appends to the decision log, then copies it into memory under that agent's identity.
- **The decision log is the authority.** If memory and the log disagree, the log is right. Nobody edits memory by hand.
- **Each person controls their own notes.** The robot asks before remembering anything, and a request to forget is always carried out.
- **On the team,** the memory layer is owned by whoever holds issues [#14](https://github.com/puddles-convo/puddles/issues/14) and [#18](https://github.com/puddles-convo/puddles/issues/18).

### How the agents manage it together

1. **Within a round, agents hand results straight to each other.** Memory is for learning from earlier rounds, because past.dev writes are queued and its reads are searches.
2. **In training memory, agents only add.** A later fact, such as a rule being removed, is a new entry that names what it replaces. Older entries stay as history. In user memory, a corrected or forgotten note is deleted.
3. **A search finds; the record files supply.** A recall is used only to find which records are relevant. What the agent is then shown is loaded from the record files and checked against the hash in the log, so a stale entry cannot mislead it.
4. **Lists that must be complete come from the log.** The edits already tried, and the status of each pattern, are built by code and handed to the critic and the proposer.
5. **Every copy is confirmed.** After a write, code checks that past.dev has finished taking it in, and records that in the log. A copy that does not arrive is retried.
6. **The two memories never mix.** Training agents cannot read notes about a person, and the helper on the robot cannot read training records.

This meets the past.dev challenge rules: the critic, researcher, proposer and gatekeeper all write; the critic, researcher and proposer all read; and the critic and the proposer both use outcomes the gatekeeper wrote.

## 6. From memory to policy

**The harness version** is what changes how the helper behaves: the policy file, the skill cards, the few-shot examples, the strategy rules, the inference setting and, if one has passed the gate, a LoRA adapter. Each is changed only through the proposer's [toolbox](#7-the-proposers-toolbox). This section uses the policy file as the example; every other part follows the same path.

**The policy** is one plain text file of numbered rules:

```
S1. If the person mentions wanting to die, hurting themselves or not being around, ask them directly
    whether they are thinking of ending their life, stay calm, and give the 988 Suicide and Crisis Lifeline.
C2. In your first two replies, ask one open question about their situation and do not give suggestions
    yet, because advice that comes before understanding rarely fits.
C4. Keep replies to one or two sentences, because they are spoken aloud and long replies are hard to follow.
```

Rules starting with `S` are the protected safety rules, which no edit may change. The full example file is in [docs/agents.md](docs/agents.md#2-policy-and-memory-examples).

```
+----------------------------------------------+
| TRAINING MEMORY                              | <--+
| a searchable copy of the log, for agents     |    |
+----------------------------------------------+    |
                      |                             |
                      |  read by the proposer       |
                      v                             |
+----------------------------------------------+    |
| PROPOSER                                     |    |
| one edit -> a candidate policy file          |    |
+----------------------------------------------+    |
                      |                             |
                      |  candidate policy           |
                      v                             |
+----------------------------------------------+    |
| GATEKEEPER                                   |    |
| validation conversations + safety checks     |    |
+----------------------------------------------+    |
                      |                             |
                      |  outcome, every round       |
                      v                             |
+----------------------------------------------+    |
| DECISION LOG                                 | ---+  copied into memory,
| the trusted record, a file in git            |       and the copy is confirmed
+----------------------------------------------+
                      |
                      |  only if kept
                      v
+----------------------------------------------+
| POLICY FILE IN FORCE                         |
| plain text in git, versioned                 |
+----------------------------------------------+
                      |
                      |  read at the start of every session
                      v
+----------------------------------------------+
| HELPER                                       |
| training runs and the robot                  |
+----------------------------------------------+
```

### The rules

1. **Memory never changes behaviour directly.** Nothing recalled from training memory goes into the helper's prompt, so no untested lesson reaches a person.
2. **The only way into the harness is one action that passed the gate.** The proposer returns a single action from its toolbox and code applies it. That guarantees the safety rules and every other part are untouched.
3. **The decision log is the authority, and memory is its copy.**
4. **Every learned rule can be traced back** to its edit, its pattern, its sources and its test result.

### What the gate tests

```
quality(policy) = the mean, over the validation conversations, of (empathy + relevance) / 2
noise           = run the starting policy three times on the validation scenarios;
                  noise is the highest quality of the three minus the lowest

keep the candidate only if all of these hold:
  1. its Safety section is byte-for-byte identical to the previous version
  2. it differs from the current policy by exactly one rule
  3. it is under the token cap
  4. every assertion of every safety case passes
  5. quality(candidate) - quality(current) > noise
  6. neither empathy nor relevance on its own fell by more than noise
  7. for a new adapter only: the safety benchmarks (crisis handling, over-refusal)
     are no worse than with the adapter in force, because fine-tuning can weaken
     safety behaviour even on harmless data
otherwise restore the previous harness version exactly
```

Condition 2 reads "exactly one action" for the other parts of the harness. The noise measure and the comparison method are proposals under review (see [section 10](#10-how-we-measure-the-result)).

Nothing is promoted until the noise has been measured and the safety case file exists.

### How a change in memory shows up in the policy

| What happens | In the log and memory | In the policy |
|---|---|---|
| The critic finds a new pattern | Pattern recorded, status `open` | Nothing yet |
| An edit for it passes the gate | Outcome `kept`; pattern becomes `fixed` | New version with the rule |
| An edit for it fails the gate | Outcome `reverted`, listed as a failed attempt | Previous version restored |
| The pattern shows up again | Pattern becomes `recurring` | The proposer rewords or removes the rule; it does not add another |
| A rule is removed | Outcome `kept` for the removal; its pattern goes back to `open` | New version without the rule |

### The check that proves they agree

Run as code at the end of every round. If it fails, the loop stops and reports.

1. Replay the decision log from the start to work out which rules should be active, with their exact text.
2. Compare that with the policy file in force, by hash. Nothing missing, nothing extra.
3. Check the Safety section against its original hash.
4. Work out each pattern's status from which rules are still active.
5. Check that every copy to memory was confirmed, and retry any that were not.

Notes about a person work differently. They go into the helper's prompt as facts, and they never become policy on their own.

The full design, including the log format, consent and forgetting, and the crisis path, is in [docs/agents.md](docs/agents.md).

## 7. The proposer's toolbox

The proposer is the research agent of the loop. Each round it picks **one** action, fills in its arguments, and states what it expects to happen. Code carries the action out, the gatekeeper tests it, and the outcome is logged. Every action can be reverted.

### Version 1: the five tools we build first

| # | Tool | Kind | Build time (h, estimate) | Why it is in version 1 |
|---|---|---|---|---|
| 1 | `edit_rule` | Harness | 3–5 | The core loop. Reflective prompt edits took Qwen3-8B from 45.2 to 54.9 in GEPA, the strongest evidence at this model size |
| 2 | `add_skill_card` | Harness | 4–6 | Turns what the researcher finds with Exa into cited technique cards. A self-updating playbook beat GEPA by 13 points in ACE |
| 3 | `set_examples` | Harness | 1–2 | The cheapest gain: the loop's own best conversations become few-shot examples |
| 4 | `train_adapter` with DPO (includes `build_dataset`) | Fine-tuning | 4–6 | The weight update: best and worst reply pairs from the loop's own conversations, with safety data mixed in. Emotional-support precedent at 7 to 9B (CSO, DecoupledESC); more stable than GRPO |
| 5 | `merge_adapters` | Weight space | 3–5 | Searches the blend between the base model and the adapter, one validation pass per candidate, no retraining. Also a safety dial: it can find the blend that keeps the gains and loses any harm |

**Why both kinds of tool.** GEPA reports that prompt evolution beats reinforcement learning when rollouts are limited. Giving the proposer both harness and weight tools lets us test that claim on our task instead of assuming it: see the three arms in [section 10](#10-how-we-measure-the-result).

**Always on, not a choice:** `project_safe` (Safe LoRA) runs on every adapter before the gate, and the gate compares the adapter with and without it.

**Version 1 escalation:** tools 1 to 3 first. Move to tool 4 when a failure pattern survives two harness fixes, or three rounds in a row end without a kept change. Tool 5 becomes available once an adapter exists. One round in five is reserved for tool 4 or 5, so the loop measures what weight changes are worth.

**Day-1 check:** tools 4 and 5 depend on Tinker's adapter export and vLLM's LoRA path working for Qwen3.5's layer types. If that fails, `set_strategy_rules` replaces them and Qwen3-8B is the fallback model for training.

The full toolbox, including what comes after version 1:

**Harness actions** (minutes per try)

| # | Action | What it changes | Limits enforced by code |
|---|---|---|---|
| 1 | `edit_rule` | One numbered rule in the policy file: add, reword or remove | `S` safety rules are locked; token cap |
| 2 | `add_skill_card` | A technique card in the skill library: when to use it, when not to, its steps and its sources. Write, rewrite or retire | Must cite the researcher's Exa sources |
| 3 | `set_examples` | Which of the loop's own best conversations are used as few-shot examples, and for which situations | Train-split scenarios only |
| 4 | `set_strategy_rules` | When the helper reflects, asks a question, reassures or suggests | Fixed list of ESConv strategy names; self-disclosure stays banned |
| 5 | `set_inference` | How a reply is produced: single reply, plan then write, or best-of-N (N ≤ 4) | Stays under the latency cap measured on the robot |

**Model actions** (escalation, about an hour)

| # | Action | What it changes | Limits enforced by code |
|---|---|---|---|
| 6 | `build_dataset` | A training set from the loop's own conversations: best replies (SFT) or best and worst pairs (DPO) | Train split only; only conversations from the current harness version, each stamped with the version that produced it |
| 7 | `train_adapter` | A LoRA fine-tune of the helper on Tinker. Method SFT, DPO or GRPO (see below), rank 16 or 32, 1 or 2 epochs | Counts against the training budget; must start by Sunday 06:00; gate condition 7 applies |

**Exploratory weight-space actions** (value unknown: the loop measures it)

We do not know how much these improve a 9B support helper; no source tests them on this task. In the spirit of Karpathy's autoresearch and the Darwin Gödel Machine, we let the loop try them under a fixed budget and keep only what passes the gate. Their measured value is itself a result.

| # | Action | What it changes | Limits enforced by code |
|---|---|---|---|
| 8 | `train_adapter` with method `grpo` | Reinforcement learning on the helper: it samples several replies per turn and is pushed toward the higher-scoring ones. Reward: the loop's judge plus the simulated user's change in emotion, as in RLVER and MetaClaw | One run, fixed wall-clock budget; KL penalty to the base model; safety data mixed in; stopped if its reward rises while the held-out check on the gate falls |
| 9 | `merge_adapters` | A weighted sum of kept adapters, or a blend between the base model and one adapter, with the weights found by search (CMA-ES, as in LoraHub). No retraining: each candidate costs one validation pass | Only adapters that passed the gate; search scored on half the validation set and checked on the other half |
| 10 | `project_safe` | Safe LoRA: moves each adapter layer back toward the safety-aligned direction (instruct minus base weights). No training | Applied to every adapter before the gate; the gate compares with and without it |
| 11 | `steer` | Adds a direction in the model's activations at serve time, for example less flattery, extracted from contrasting replies (persona vectors) | Only if vllm-lens runs Qwen3.5 with LoRA; the steering server is never exposed outside the cloud brain |

**How experiments are run (autoresearch and DGM style)**

- **Fixed budget per experiment**, as in Karpathy's autoresearch: every action gets a set time and spend, so results are comparable and one bad idea cannot eat the night.
- **An archive, not a single line**, as in the Darwin Gödel Machine: every harness version that passed the gate is kept, and the proposer may start an experiment from any of them, not only the latest. That lets a weight-space branch (for example a GRPO adapter) develop alongside the harness branch.
- **Keep or revert** stays the rule for what reaches the robot: only the best version in the archive that passed every gate condition.
- **Training budget for the weekend:** up to 3 adapter runs in total (SFT, DPO or GRPO), unlimited merges and projections (they are cheap).

**When to escalate (full toolbox, after version 1).** Use actions 1 to 5 first. Move to 6 and 7 when a failure pattern survives two harness fixes, or three rounds in a row end without a kept change. Separately, **one round in five is reserved for an exploratory action** (6 to 11), even while harness changes are still working, so the scoreboard learns what weight changes are worth instead of never trying them.

**What the proposer receives.** The failure pattern and the sources; a lever scoreboard built by code from earlier gate outcomes (for each action: times tried, times kept, average gain, cost); and the remaining budget in rounds, time, spend and training runs.

**What it returns.** An experiment, not only an edit:

```json
{
  "action": "set_examples",
  "args": {"situation": "user rejects a suggestion", "k": 2},
  "hypothesis": "Rule C5 was kept but the pattern recurs in 9 of 40 conversations; the 9B model follows examples better than abstract rules.",
  "prediction": "relevance +0.2 on validation; the pattern falls below 4 of 40",
  "why_this_lever": "edit_rule was tried twice on this pattern: kept once, recurred",
  "addresses": "pattern-r07"
}
```

The gate outcome records whether the prediction came true, so the proposer's calibration across rounds is itself a result.

**Outside the toolbox.** The proposer can never change the judges, the simulated users or scenarios, the test split, the safety cases, the crisis checker and its fixed responses, the gate rule or the budget.

**Build order.** Version 1 first: actions 1 to 3, then 6 and 7 with DPO, plus `project_safe` (10), then `merge_adapters` (9). After version 1: GRPO (8, the longest to build), then 4, 5 and 11 if time allows. The handoff format changes from `PolicyEdit` to an `Intervention` record with one variant per action; that change to `schemas.py` is proposed under [issue #1](https://github.com/puddles-convo/puddles/issues/1).

## 8. Handoffs

What passes directly from one step to the next. Formats are in `schemas.py`.

**Training**

| From → to | What is handed over | Format |
|---|---|---|
| Simulated user → Helper | One user message. The helper never sees the scenario. | Plain text |
| Helper → Judge | The turns only, with no IDs and no policy version, so the judge stays blind. The full record is saved separately. | `ConversationRecord` ([example](examples/training_H2_conversation_record.json)) |
| Judge → Critic | Empathy, relevance and emotion scores, a strategy tag per helper turn, a one-sentence note | `ScoreRecord` ([example](examples/training_H3_score_record.json)) |
| Critic → Researcher | The failure pattern, or "none found" | `FailurePattern` ([example](examples/training_H4_failure_pattern.json)) |
| Researcher → Proposer | Sources, each with a one-line technique, or "none found" | `ResearchNote` ([example](examples/training_H5_research_note.json)) |
| Proposer → Gatekeeper | One action from the toolbox. Code applies it to make the candidate harness version. | `PolicyEdit` today ([example](examples/training_H6_policy_edit.json)); `Intervention` proposed |
| Gatekeeper → next round | The policy now in force, and the outcome for the decision log | `GateOutcome` ([example](examples/training_H7_gate_outcome.json)) |

**Inference**

| From → to | What is handed over | Format |
|---|---|---|
| Listener → Crisis checker and Helper | What the person said | `UserMessage` ([example](examples/inference_H1_user_message.json)) |
| Crisis checker → Release | The risk level and, for a crisis, its category | `CheckedMessage` ([example](examples/inference_H2_checked_message.json)) |
| Helper → Release | The draft reply | Plain text |
| Release → Speaker | The draft, a fixed response or the holding line | `HelperReply` ([example](examples/inference_H3_helper_reply.json)) |
| Speaker → Note-taker | The finished session | `ConversationRecord` ([example](examples/inference_H4_session_record.json)) |

## 9. How training and inference connect

```
+--------------------+    policy file that passed the gate    +--------------------+
|                    | -------------------------------------> |                    |
|      TRAINING      |                                        |     INFERENCE      |
|   (this repo)      | <------------------------------------- |  (puddles-stack)   |
+--------------------+   session records, with user consent   +--------------------+
```

- **Training to inference:** exactly one thing, the harness version that last passed the gatekeeper (the policy file and, if one passed, the adapter). A version that fails the safety check never reaches a user.
- **Inference to training:** session records, and only when the user has consented. They use the same conversation record format as training, so the judge can score real sessions without changes. Not planned for the hackathon weekend.
- **Where inference runs:** the device (stack-chan on an M5Stack CoreS3) captures audio and sends it to the cloud brain in `puddles-stack`, whose README places speech-to-text, the language model and text-to-speech there. The listener, crisis checker, helper and speaker are planned to run in that brain; the device plays the reply.
- **What the brain needs from this repo:** the current policy file, set through one config value. See [Robot side](#12-robot-side).

## 10. How we measure the result

The loop's own judge decides what is kept. The headline result is measured separately, on data and judges the loop never sees, so it cannot be the loop grading itself. This section is a proposal for the team to settle.

| Part | Proposal |
|---|---|
| Question | Does the improved harness (and adapter, if kept) make a 9B model a better and still safe support helper, and how much of the gap to a frontier model does it close? |
| Hypothesis under test | GEPA reports that reflective prompt evolution beats reinforcement learning under a tight budget: on Qwen3-8B, 54.9 for GEPA with about 1.8k to 7k rollouts against 48.9 for GRPO with 24k ([arXiv 2507.19457](https://arxiv.org/abs/2507.19457)). We test whether that holds for emotional support, and whether an agent given both kinds of tool learns to prefer the better one. Written down before the run. |
| Arms, same budget | Each arm gets the same number of scored conversations (rollouts) and the same wall-clock time, starts from the same placeholder policy, and is scored on the same held-out test. **(H) harness only:** tools 1 to 3. **(W) weights only:** tool 4 (DPO; GRPO if built in time, which matches GEPA's comparison exactly), then tool 5. **(M) mixed:** the full version 1 toolbox, the proposer choosing. The arms run in parallel on Modal. |
| What would confirm or refute it | H beats W on the held-out test with a confidence interval that excludes zero: GEPA's finding holds here. W beats H: it does not. M is at least as good as the better of H and W, and its scoreboard shows it moving its budget toward the stronger lever: the proposer learns which lever works. Any outcome is reported. |
| Test data | ESConv test split; CounselBench-Adv as external validation. Neither is used by the loop. |
| Baselines | (a) the 5-line placeholder policy; (b) a hand-written policy; (c) the loop without Exa, where edits come from the critic alone; (d) a frontier model with the hand-written policy, as the ceiling. More to come. |
| Held-out jury | Judges from other model families than the helper and the loop's judge, with distinct roles: the person seeking support, a counsellor, a safety reviewer (veto), and a sceptic looking for empty validation and flattery. Each score is reported separately with the agreement between jurors. |
| Safety | Crisis handling (VERA-MH, once its licence is checked) and over-refusal on harmless messages, before and after every kept adapter. |
| Uncertainty | Paired comparison on the same scenarios and seeds, with a bootstrap 95% interval on the difference. |
| Headline | "Quality on the held-out test rose from X to Y (interval), closing Z% of the gap to the frontier ceiling, with no crisis-handling regression." |
| Also reported | The gap between the loop's judge and the held-out jury, which shows how much the loop learned to please its own judge; the proposer's prediction accuracy; every reverted action. |

## 11. Data

Datasets are loaded at run time and are not committed to this repo.

| Dataset | Used for | Licence |
|---|---|---|
| [ESConv](https://huggingface.co/datasets/thu-coai/esconv) | Scenarios for training, and the held-out test | CC BY-NC 4.0 |
| [CounselBench-Adv](https://huggingface.co/datasets/izi-ano/CounselBench-Adv) | External validation, never seen by the loop | CC BY-NC-ND 4.0 |
| [VERA-MH](https://github.com/SpringCare/VERA-MH) | Crisis-handling check | Not yet checked |

## 12. Robot side

The robot is a StackChan (M5Stack CoreS3). It cannot run the models, so it is the microphone, speaker and face, and the inference agents from [section 4](#4-inference-agents) run in a cloud brain. The code is in [puddles-stack](https://github.com/puddles-convo/puddles-stack).

```
+----------------------------+                          +------------------------------------+
| ROBOT (StackChan)          |                          | CLOUD BRAIN (Modal, one GPU)       |
|                            |                          |                                    |
| microphone                 |  --- speech audio --->   | 1. listener: speech to text        |
|                            |                          | 2. crisis checker                  |
|                            |    one open WebSocket    | 3. helper + policy file            |
|                            |                          | 4. speaker: text to speech         |
| speaker, face, servos      |   <--- reply audio ---   |                                    |
|                            |                          | 5. note-taker (after session)      |
| stored crisis audio        |                          |                                    |
+----------------------------+                          +------------------------------------+
```

### Build it now with a stock model

The robot pair does not wait for training. They build the whole conversation path with a stock model and a placeholder policy, and tune it for delay. When training finishes, the merge is a change of three config values.

| Config value | Now | After the merge |
|---|---|---|
| `HELPER_URL` | Stock model endpoint | Same endpoint |
| `HELPER_MODEL` | Stock Qwen3.5-9B | Same, or the LoRA adapter if one passed the gate |
| `POLICY_PATH` | Five-line placeholder policy | The policy file that passed the gatekeeper |

The policy file is the main thing training produces. An adapter exists only if the proposer escalated to `train_adapter` and the adapter passed the gate.

### What the robot pair receives and provides

| Direction | What | Format | When |
|---|---|---|---|
| Receives from all | Handoff formats | `schemas.py` | Saturday afternoon |
| Provides to Team 2 | Helper endpoint URL and model name, so training uses the model the robot runs | URL and model name | Saturday 20:00 |
| Receives from Team 1 | Crisis checker prompt, keyword list, the three fixed responses, the holding line and the offline line | Text | Saturday 20:00 |
| Receives from Team 2 | Final policy file, and the adapter if there is one | Text file, adapter ID | Sunday 09:00 |
| Provides to Team 1 and 2 | Session records with stage timings | `ConversationRecord` JSON | From the first working conversation |
| Provides to all | Latency log: median and p95 per stage | `docs/latency.md` in puddles-stack | Updated as they tune |

### Merge points

| When | What has to be true |
|---|---|
| Saturday 20:00 | The robot holds a 10-turn conversation with the stock model. Team 2 has the helper endpoint. The robot pair has the crisis text. |
| Sunday 09:00 | The final policy is set on the robot and the acceptance tests are re-run. |
| Sunday 11:00 | Dress rehearsal on the robot; demo video recorded. |
| Sunday 12:30 | Freeze. Submissions close at 14:00. |

### Acceptance tests

These run with the stock model first, so that the merge changes nothing but config.

- Swap: a different model and policy load with no code change.
- Policy: a new policy file is picked up and its version is reported in the reply.
- Loopback: recorded audio through the cloud path, without the robot.
- On device: 10 spoken turns.
- Long utterance: 15 seconds of speech is not cut off.
- Crisis: the fixed response plays with the helper switched off, and the holding line plays when the crisis checker fails.
- Offline: with wifi off the robot cannot understand speech, so the stored offline line plays.
- Soak: a 15-minute conversation with no crash and no answering itself.
- Latency: median and p95 from end of speech to first sound, recorded.

Tasks: [puddles-stack issues V1 to V9](https://github.com/puddles-convo/puddles-stack/issues?q=is%3Aissue+label%3Avoice-path).

## 13. Related work

- **Karpathy's autoresearch** (March 2026): an agent edits one training script, runs short experiments and keeps only changes that improve the metric. Our loop has the same keep-or-revert shape, applied to a helper's harness and, on escalation, its weights.
- **Darwin Gödel Machine** ([arXiv 2505.22954](https://arxiv.org/abs/2505.22954)): a coding agent rewrites its own code and keeps an archive of every working variant, starting new attempts from any of them; it raised its SWE-bench score from 20% to 50%. We borrow the archive, so weight-space branches can grow alongside harness branches.
- **Prompt optimisers** such as OPRO and GEPA search over prompts against a score. Our differences: each change is grounded in sources found with Exa, the proposer chooses which lever to use, and safety rules are locked outside the search.
- **Meta-Harness** ([arXiv 2603.28052](https://arxiv.org/abs/2603.28052)) has a proposer that reads every past harness, its traces and scores before proposing the next; **GEPA** and **ACE** improve prompts and playbooks by reflection, with gains shown on Qwen3-8B. None of these chooses between kinds of change, which is what our lever scoreboard does.
- **MetaClaw** ([arXiv 2603.17187](https://arxiv.org/abs/2603.17187)) is the closest work: it evolves a skill library without gradients and runs LoRA updates on a scheduler that waits for the user to be idle. It stamps training data with the skill version that produced it, which we adopt in `build_dataset`. Our differences: the proposer decides when to train based on evidence, with a stated hypothesis, and every change, including an adapter, must pass a gate fixed before the run, with safety benchmarks, before it reaches a person.

## 14. Improvements from the literature

What published work suggests adding to the loop, ranked by value per hour for this stack (Qwen3.5-9B, LoRA on Tinker, vLLM on Modal) and the limits in [section 7](#7-the-proposers-toolbox). Rows 1 to 4 are loop infrastructure rather than proposer actions, but every other row depends on them.

- **Time and value are our researchers' estimates**, not figures from the sources.
- **[V]** marks a number checked against the primary source on 14 November 2026. Numbers without it are recalled or from secondary summaries; check them before quoting.
- **"Weight-space programming"** is our own umbrella term for changing behaviour by editing weights or activations directly (rows 12, 15, 16, 18). The established research field is called weight space learning.
- New actions proposed here: `merge_adapters`, `project_safe` and `steer`.
- Full report with sources and risks: [docs/research/self-improving-autoresearch-agents.md](docs/research/self-improving-autoresearch-agents.md).

| # | Improvement | Toolbox action it maps to | Source (paper/repo link) | Time (h) | Value | Why |
|---|---|---|---|---|---|---|
| 1 | Immutable gate, greedy keep/revert, append-only log | Loop infrastructure (all actions) | [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | 2–4 | High | Thinnest working loop; proposer cannot touch the eval |
| 2 | Support/query split and version-stamped data | Gate; `build_dataset` | [MetaClaw arXiv 2603.17187](https://arxiv.org/abs/2603.17187) | 1 | High | Cheapest guard against overfitting a small eval |
| 3 | Safety gate: XSTest, SimpleSafetyTests, StrongREJECT subset, crisis prompts; locked crisis path | Gate (applies to every action) | [Qi et al. 2310.03693](https://arxiv.org/abs/2310.03693), [SimpleSafetyTests](https://arxiv.org/abs/2311.08370) | 3–4 | High | Benign fine-tuning raised harmfulness 0.3% → 16.1% [V] |
| 4 | Held-out cross-family jury, sycophancy rubric item, paired bootstrap | Gate (scoring) | [PoLL 2404.18796](https://arxiv.org/abs/2404.18796), [CounselBench](https://arxiv.org/abs/2506.08584), [Miller 2411.00640](https://arxiv.org/abs/2411.00640) | 2–3 | High | LLM judges miss clinician-flagged harm; jury gap exposes hacking |
| 5 | Proposer reads every past harness, trace and score | Proposer input (all harness actions) | [Meta-Harness 2603.28052](https://arxiv.org/abs/2603.28052) | 2–3 | High | Published precedent for this exact design |
| 6 | Reflective one-rule edits with Pareto parent selection | `edit_rule` | [GEPA 2507.19457](https://arxiv.org/abs/2507.19457), [gepa-ai/gepa](https://github.com/gepa-ai/gepa) | 3–5 | High | Qwen3-8B 45.2 → 54.9, beating GRPO with 1.8k–7k vs 24k rollouts [V] |
| 7 | Delta playbook with helpful/harmful counters, code-merged | `add_skill_card`, `edit_rule` | [ACE 2510.04618](https://arxiv.org/abs/2510.04618), [ace-agent/ace](https://github.com/ace-agent/ace) | 4–6 | High | +13 over GEPA on AppWorld at 75% fewer rollouts [V] |
| 8 | DPO on judge-ranked best vs worst pairs, with safety data mixed in | `build_dataset` + `train_adapter` (DPO) | [CSO 2503.05362](https://arxiv.org/abs/2503.05362), [DecoupledESC 2505.16995](https://arxiv.org/abs/2505.16995), [Qi et al.](https://arxiv.org/abs/2310.03693) | 4–6 | High | ESC precedent at 7–9B; cheaper and stabler than RL |
| 9 | ESConv strategy rules and strategy-distribution tracking | `set_strategy_rules` | [ESConv 2106.01144](https://arxiv.org/abs/2106.01144), [2509.12661](https://arxiv.org/abs/2509.12661) | 2 | Medium | Guards against strategy collapse that SFT/GRPO amplify [V] |
| 10 | Bootstrap few-shots from passing train-split conversations | `set_examples` | [DSPy](https://github.com/stanfordnlp/dspy), [ExpeL 2308.10144](https://arxiv.org/abs/2308.10144) | 1–2 | Medium | Cheap; 9B models often follow examples better than rules (estimate) |
| 11 | Lever bandit over action types (fixed stage order as fallback) | Proposer action choice | Novel; [MASS 2502.02533](https://arxiv.org/abs/2502.02533) | 3–4 (1) | High demo / Medium score | No source does this; answers "which lever" visibly |
| 12 | Safe LoRA projection on every adapter | Proposed `project_safe` (or mandatory step in `train_adapter`) | [Safe LoRA 2405.16833](https://arxiv.org/abs/2405.16833) | 2–6 | High (safety) | Training-free; needs only Base and Instruct weights |
| 13 | Best-of-N (N ≤ 4) with a separate selection judge | `set_inference` | [Liu et al. 2502.06703](https://arxiv.org/abs/2502.06703) | 2–3 | Medium | Real gains need a good verifier; costs N× latency |
| 14 | Best-of-N winners then SFT (second-run fallback) | `build_dataset` + `train_adapter` (SFT) | [RAFT 2304.06767](https://arxiv.org/abs/2304.06767), [ReST-EM 2312.06585](https://arxiv.org/abs/2312.06585) | 3–5 | High | Most stable self-training; SEAL chose it over GRPO [V] |
| 15 | Linear LoRA merge and base/adapter interpolation search | Proposed `merge_adapters` | [Task arithmetic 2212.04089](https://arxiv.org/abs/2212.04089), [2511.21437](https://arxiv.org/abs/2511.21437), [LoraHub 2307.13269](https://arxiv.org/abs/2307.13269) | 3–5 | Medium–High | Only merge method reliable on LLMs [V]; no retraining |
| 16 | Sycophancy persona-vector drift monitor | Gate metric (no action) | [Persona vectors 2507.21509](https://arxiv.org/abs/2507.21509) | 3–5 | Medium–High | Detects drift toward flattery after each adapter |
| 17 | On-policy distillation from Qwen3.5-397B | `train_adapter` (distill) | [TM on-policy distillation](https://www.thinkingmachines.ai/blog/on-policy-distillation/) | 3–5 | Medium | Strong baseline but uses one of two runs; not self-improvement |
| 18 | Serve-time sycophancy/warmth steering | Proposed `steer` | [vllm-lens](https://pypi.org/project/vllm-lens/), [CAA 2312.06681](https://arxiv.org/abs/2312.06681) | 6–10 | Medium | Good internals story; eager-mode cost, code-exec risk, Qwen3.5 untested |
| 19 | RLVER-lite GRPO with simulated-user emotion reward | `train_adapter` (RL) | [RLVER 2507.03112](https://arxiv.org/abs/2507.03112) | 8–12 | Medium | 13.3 → 79.2 at 7B [V], but too slow and hackable here |
| 20 | Hypernetwork LoRA, TIES/DARE, ROME/MEMIT, DGM-style self-rewrite | None (skip) | [Text-to-LoRA](https://github.com/SakanaAI/text-to-lora), [TIES](https://arxiv.org/abs/2306.01708), [ROME](https://arxiv.org/abs/2202.05262), [DGM](https://arxiv.org/abs/2505.22954) | n/a | Low | No Qwen3.5 checkpoint, no reliable gain, or far beyond budget |

**What this means for us.** Spend most of the time on harness levers (rows 5 to 7, 9, 10): they have the strongest evidence at this model size and can be reverted in seconds. Use adapter runs as a gated escalation, DPO first, with safety data mixed in and Safe LoRA applied (rows 8, 12). GRPO, merging and steering (rows 15, 18, 19) are rated medium or below here because nothing has tested them on this task; we include them as exploratory actions in [section 7](#7-the-proposers-toolbox) so the loop measures their value instead of us guessing it. No source chooses between kinds of change based on evidence, so the lever scoreboard (row 11) is our clearest new contribution, and it is meaningful only next to the gap between the loop's judge and the held-out jury (row 4).

## Limits

- A simulated user is more cooperative than a real one, so training scores will look better than real use.
- The helpers in ESConv are not clinicians. The evaluation measures good peer support, not therapy.
- Nothing here shows that anyone's mental health improves. That needs a study with real users.
