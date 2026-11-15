# puddles

A support-conversation helper that improves its own behaviour, built at the SF AI Research Hackathon (November 14-15, 2026), Track 5 (Open).

The helper listens, reflects, asks questions, suggests coping techniques and points people to professional help. It does not diagnose or treat.

**Status:** this README is the agreed design. The memory layer and the critic, researcher and proposer (steps 4 to 6) are in [`puddles/`](puddles/); see [Running the training agents](#running-the-training-agents). The rest of the pipeline is not built yet. Scores, counts and sources in the examples are illustrative.

- One file per agent, with its prompt, where the input goes in it, its input and output formats, and its memory reads and writes: [docs/agents/](docs/agents/README.md)
- The memory, end to end, in one file: [docs/memory.md](docs/memory.md). Go there for anything about memory.
- The design behind policy and the crisis path: [docs/agents.md](docs/agents.md)
- Record formats: `schemas.py` (version 3), with an example of each in `examples/`. They are a proposal for the team to settle under [issue #1](https://github.com/puddles-convo/puddles/issues/1).
- The device and its cloud brain: [puddles-stack](https://github.com/puddles-convo/puddles-stack)

## Contents

1. [Training steps](#1-training-steps)
2. [Training agents](#2-training-agents)
3. [Inference steps](#3-inference-steps)
4. [Inference agents](#4-inference-agents)
   - [Models](#models): which model plays each agent
5. [The memory](#5-the-memory)
6. [From memory to policy](#6-from-memory-to-policy)
7. [Handoffs](#7-handoffs)
8. [How training and inference connect](#8-how-training-and-inference-connect)
9. [Data](#9-data)
10. [Robot side](#10-robot-side)
- [Running the training agents](#running-the-training-agents)

## 1. Training steps

The agreed design is an offline policy-improvement loop. The figure below distinguishes implemented handoffs from planned integration; the current development gate is selected manually.

![Puddles training implementation snapshot](figures/fig1-training.svg)

**Implementation snapshot:** solid paths show implemented handoffs; dashed paths show planned Judge integration. The round script consumes pre-scored JSON, and the gate is a developer-selected keep/revert stand-in, not measured evaluation. There is **no automatic next-batch loop**.

- **State inputs** name policy, decision-log and memory dependencies; the log and hash-checked local records are authoritative.
- **Finish round** logs the outcome, copies available records to shared memory and promotes the policy only when kept. No-pattern/no-evidence exits finish without an edit.
- **Motion is illustrative**, not execution timing or measured embeddings. The broader tables below remain the agreed design, not a claim that all planned components are built.

[Replayable player](figures/fig1-player.html) (download and open locally) · [Architecture and reproduction notes](figures/fig1-architecture.md)

## 2. Training agents

The agent names link to a file for each one, with its prompt and formats.

| Agent | What it does | Pulls from memory | Writes to memory | Example input | Example output |
|---|---|---|---|---|---|
| [**1. Simulated user**](docs/agents/t1-simulated-user.md) | • Plays a person from an ESConv scenario<br>• Reacts honestly and opens up slowly | Nothing (no access) | Nothing | Scenario: anxiety, job crisis, "I am on short term disability and I am afraid I will lose my job"<br>plus the helper's last line | `Not yet. I'm on short term disability and I'm not ready to go back.` |
| [**2. Helper**](docs/agents/t2-i3-helper.md) | • Replies to the person, following the policy file<br>• Same model, prompt and code as on the robot | Nothing during training | Nothing | `policy-v4` and the turns so far:<br>user: "I'm feeling anxious that I'm going to lose my job." | `That's a heavy thing to be carrying. What's making you feel the job is at risk right now?` |
| [**3. Judge**](docs/agents/t3-judge.md) | • Rates each finished conversation<br>• Blind to which policy produced it, and frozen after calibration | Nothing (no access) | Nothing. Its scores go to the critic and the decision log. | The turns of one conversation, with no IDs and no policy version | `{"empathy": 3, "relevance": 2, "emotion_before": 4, "emotion_after": 4, "note": "Suggested HR before asking anything."}` |
| [**4. Critic**](docs/agents/t4-critic.md) | • Reads the lowest-scoring conversations in full<br>• Names the one failure that explains most of them | Earlier patterns, and what happened to each | The failure pattern | About ten of the lowest-scoring conversations with the judge's notes, and the current policy | `{"found": true, "name": "advice-before-understanding", "seen_in": 2, "out_of": 3}` |
| [**5. Researcher**](docs/agents/t5-researcher.md) | • Searches Exa for how counsellors handle that pattern<br>• Keeps sources, each with a one-line technique | Searches already run | The sources and techniques | The failure pattern | `{"sources": [{"url": "(from Exa)", "technique": "Ask one open question and reflect the answer before any suggestion."}]}` |
| [**6. Proposer**](docs/agents/t6-proposer.md) | • Writes exactly one change to the policy<br>• Does not repeat an edit that was reverted | Earlier edits and their outcomes | The proposed edit | The current policy, the pattern and the sources | `{"action": "add", "rule_text": "In your first two replies, ask one open question about their situation and do not give suggestions yet, because advice that comes before understanding rarely fits."}` |
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
| [**5. Note-taker**](docs/agents/i5-note-taker.md) | • After the session, decides what to remember about this person<br>• Adds notes only if the person agreed; can always delete | The existing notes about this person | Notes added, replaced, confirmed or deleted | The finished session and the existing notes | `[{"action": "add", "kind": "what_helps", "fact": "Slow breathing helps a bit when the worry gets bad."}]` |

## Models

Which model plays each agent. These are starting choices and none has been benchmarked yet. No model name is hard-coded: each one is set in configuration, so changing a model does not change the code.

### Training

| Agent | Model | Runs on |
|---|---|---|
| [1. Simulated user](docs/agents/t1-simulated-user.md) | Qwen3.6-35B-A3B. A different model from the helper, so the helper is not talking to itself | Tinker |
| [2. Helper](docs/agents/t2-i3-helper.md) | Qwen3.5-9B, thinking mode off | Modal |
| [3. Judge](docs/agents/t3-judge.md) | Three pinned OpenAI judges, each rating and then re-checking its own rating, and Astra (the strongest pinned OpenAI model), who writes the final score. Temperature 0 | OpenAI API |
| [4. Critic](docs/agents/t4-critic.md) | The same OpenAI model as the judge | OpenAI API |
| [5. Researcher](docs/agents/t5-researcher.md) | The same OpenAI model, with Exa search as its one tool | OpenAI API and Exa |
| [6. Proposer](docs/agents/t6-proposer.md) | The same OpenAI model as the judge | OpenAI API |
| [7. Gatekeeper](docs/agents/t7-gatekeeper.md) | Code, not a model. It asks the judge model one yes-or-no question per judged safety check | |

### Inference

| Agent | Model | Runs on |
|---|---|---|
| [1. Listener](docs/agents/i1-listener.md) | faster-whisper, large-v3-turbo (speech to text) | Modal, the same GPU as the helper |
| [2. Crisis checker](docs/agents/i2-crisis-checker.md) | A keyword list in code first, then the helper model with its own prompt | Modal |
| [3. Helper](docs/agents/t2-i3-helper.md) | Qwen3.5-9B, the same model and prompt as in training | Modal |
| [Release](docs/agents/i3b-release.md) | Code, not a model | |
| [4. Speaker](docs/agents/i4-speaker.md) | Kokoro (text to speech) | Modal |
| [5. Note-taker](docs/agents/i5-note-taker.md) | A small OpenAI model, run after the session ends | OpenAI API |

### Still to decide

- **The exact OpenAI model** for the judge, critic, researcher and proposer. It must be pinned before the judge is calibrated ([#3](https://github.com/puddles-convo/puddles/issues/3)), because changing it afterwards invalidates the scores. A fallback judge on Tinker is [#23](https://github.com/puddles-convo/puddles/issues/23).
- **The crisis checker's model.** The helper model adds no delay. A stronger model would miss less, if it can answer inside the 3-second limit.
- **The helper's size.** 4B, 9B or larger is settled by the comparison in [#24](https://github.com/puddles-convo/puddles/issues/24).

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
| User note | User | Note-taker | Helper, Note-taker | One fact about one person, to carry into the next conversation. Each note has one of six kinds, and one person has 23 notes at most. |

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

### What past.dev holds and what the repo holds

| Kept in the repo (the authority) | Copied into past.dev, audience `training-agents` |
|---|---|
| `log/decisions.jsonl`: one `round` event per round, plus a `memory_copy` event per copy | Nothing from the log itself, only the records it points to |
| `records/round-NN/pattern-rNN.json`, written by the critic | One sentence per failure pattern, identity `critic` |
| `records/round-NN/research-rNN.json`, written by the researcher, plus the raw Exa results beside it | One sentence per research note: technique and source URL, identity `researcher` |
| `records/round-NN/edit-rNN.json` and `policies/policy-vN-candidate.md`, written by the proposer | One sentence per edit: action, rule ID and summary, identity `proposer` |
| `records/round-NN/gate-rNN.json` and the promoted `policies/policy-vN.md` | One sentence per outcome: kept or reverted and why, identity `gatekeeper` |
| `log/memory_reads.jsonl`: which agent recalled which record IDs | Nothing |
| Conversations, scores, the simulated user's scenarios and the policy text | Nothing |

Each copy goes in only after the round is in the log, and is polled until past.dev reports it `completed`; a copy still processing after 60 s is logged `pending` and resent at the start of the next round. A recall only tells an agent which record IDs are relevant: the agent loads those records from their files and drops any whose hash does not match the log. The simulated user, the helper and the judge never get a past.dev client.

Who reads what, in [`puddles/agents/`](puddles/agents/):

| Agent | Reads from memory | Written by |
|---|---|---|
| Critic | Earlier patterns and their outcomes | Critic, gatekeeper |
| Researcher | Earlier research notes and their queries | Researcher |
| Proposer | Why earlier edits were kept or reverted | Gatekeeper |

### Who is in control

- **No agent is in charge of the memory.** The orchestrator is: the plain code that runs a round. An agent returns its result, and the orchestrator saves it as a record file, appends to the decision log, then copies it into memory under that agent's identity.
- **The decision log is the authority.** If memory and the log disagree, the log is right. Nobody edits memory by hand.
- **Each person controls their own notes.** The robot asks before remembering anything, and a request to forget is always carried out.
- **On the team,** the memory layer is owned by whoever holds issues [#14](https://github.com/puddles-convo/puddles/issues/14) and [#18](https://github.com/puddles-convo/puddles/issues/18).

### How the agents manage it together

1. **Within a round, agents hand results straight to each other.** Memory is for learning from earlier rounds, because past.dev writes are queued and its reads are searches.
2. **In training memory, agents only add.** A later fact, such as a rule being removed, is a new entry that names what it replaces. Older entries stay as history. In user memory, a corrected or forgotten note is deleted, and each kind of note has a limit, so one person's notes stay the same size however many sessions there are.
3. **A search finds; the record files supply.** A recall is used only to find which records are relevant. What the agent is then shown is loaded from the record files and checked against the hash in the log, so a stale entry cannot mislead it.
4. **History lists come from the log, within a budget.** The edits already tried, the searches already run and the status of each pattern are built by code. The researcher and the proposer are shown everything about the pattern in hand and the 10 most recent others, with a count of the rest. The critic is shown every pattern that has a rule, and the unsolved ones most like this round's failures. So a prompt does not grow with every round. The checks for a repeated edit or search run against the whole log.
5. **Every copy is confirmed.** After a write, code checks that past.dev has finished taking it in, and records that in the log. A copy that does not arrive is retried.
6. **The two memories never mix.** Training agents cannot read notes about a person, and the helper on the robot cannot read training records.

The reasons for the limits on notes and history lists, and what other systems do, are in [docs/research/memory/00-summary.md](docs/research/memory/00-summary.md).

This meets the past.dev challenge rules: the critic, researcher, proposer and gatekeeper all write; the critic, researcher and proposer all read; and the critic and the proposer both use outcomes the gatekeeper wrote.

## 6. From memory to policy

**The policy** is one plain text file of numbered rules, and it is the only thing that changes how the helper behaves:

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
2. **The only way into the policy is one edit that passed the gate.** The proposer returns a single rule change (add, reword or remove) and code applies it. That guarantees the safety rules and every other rule are untouched.
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
otherwise restore the previous policy file exactly
```

Nothing is promoted until the noise has been measured and the safety case file exists.

### How a change in memory shows up in the policy

| What happens | In the log and memory | In the policy |
|---|---|---|
| The critic finds a new pattern | Pattern recorded, status `open` | Nothing yet |
| An edit for it passes the gate | Outcome `kept`; pattern becomes `fixed` | New version with the rule |
| An edit for it fails the gate | Outcome `reverted`, listed as a failed attempt | Previous version restored |
| The pattern shows up again | A new record for this sighting, pointing to the first one; the pattern becomes `recurring` | The proposer rewords or removes the rule; it does not add another |
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

## 7. Handoffs

What passes directly from one step to the next. Formats are in `schemas.py`.

**Training**

| From → to | What is handed over | Format |
|---|---|---|
| Simulated user → Helper | One user message. The helper never sees the scenario. | Plain text |
| Helper → Judge | The turns only, with no IDs and no policy version, so the judge stays blind. The full record is saved separately. | `ConversationRecord` ([example](examples/training_H2_conversation_record.json)) |
| Judge → Critic | Empathy, relevance and emotion scores, a strategy tag per helper turn, a one-sentence note | `ScoreRecord` ([example](examples/training_H3_score_record.json)) |
| Critic → Researcher | The failure pattern, or "none found" | `FailurePattern` ([example](examples/training_H4_failure_pattern.json)) |
| Researcher → Proposer | Sources, each with a one-line technique, or "none found" | `ResearchNote` ([example](examples/training_H5_research_note.json)) |
| Proposer → Gatekeeper | One edit. Code applies it to make the candidate policy file. | `PolicyEdit` ([example](examples/training_H6_policy_edit.json)) |
| Gatekeeper → next round | The policy now in force, and the outcome for the decision log | `GateOutcome` ([example](examples/training_H7_gate_outcome.json)) |

**Inference**

| From → to | What is handed over | Format |
|---|---|---|
| Listener → Crisis checker and Helper | What the person said | `UserMessage` ([example](examples/inference_H1_user_message.json)) |
| Crisis checker → Release | The risk level and, for a crisis, its category | `CheckedMessage` ([example](examples/inference_H2_checked_message.json)) |
| Helper → Release | The draft reply | Plain text |
| Release → Speaker | The draft, a fixed response or the holding line | `HelperReply` ([example](examples/inference_H3_helper_reply.json)) |
| Speaker → Note-taker | The finished session | `ConversationRecord` ([example](examples/inference_H4_session_record.json)) |

## 8. How training and inference connect

```
+--------------------+    policy file that passed the gate    +--------------------+
|                    | -------------------------------------> |                    |
|      TRAINING      |                                        |     INFERENCE      |
|   (this repo)      | <------------------------------------- |  (puddles-stack)   |
+--------------------+   session records, with user consent   +--------------------+
```

- **Training to inference:** exactly one thing, the policy file that last passed the gatekeeper. A policy that fails the safety check never reaches a user.
- **Inference to training:** session records, and only when the user has consented. They use the same conversation record format as training, so the judge can score real sessions without changes.
- **Where inference runs:** the device (stack-chan on an M5Stack CoreS3) captures audio and sends it to the cloud brain in `puddles-stack`, whose README places speech-to-text, the language model and text-to-speech there. The listener, crisis checker, helper and speaker are planned to run in that brain; the device plays the reply.
- **What the brain needs from this repo:** the current policy file, set through one config value. See [Robot side](#10-robot-side).

## 9. Data

Datasets are loaded at run time and are not committed to this repo.

| Dataset | Used for | Licence |
|---|---|---|
| [ESConv](https://huggingface.co/datasets/thu-coai/esconv) | Scenarios for training, and the held-out test | CC BY-NC 4.0 |
| [CounselBench-Adv](https://huggingface.co/datasets/izi-ano/CounselBench-Adv) | External validation, never seen by the loop | CC BY-NC-ND 4.0 |
| [VERA-MH](https://github.com/SpringCare/VERA-MH) | Crisis-handling check | Not yet checked |

## 10. Robot side

The robot is a StackChan (M5Stack CoreS3). It cannot run the models, so it is the microphone, speaker and face, and the inference agents from section 4 run in a cloud brain. The code is in [puddles-stack](https://github.com/puddles-convo/puddles-stack).

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
| `HELPER_MODEL` | Stock Qwen3.5-9B | Same, or the fine-tuned adapter if one exists |
| `POLICY_PATH` | Five-line placeholder policy | The policy file that passed the gatekeeper |

The policy file is the main thing training produces. A fine-tuned model is optional.

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

## Running the training agents

Python 3.10 or later. Keys come from the environment and are never committed.

```bash
pip install -r requirements.txt
python -m pytest                                   # offline: scripted model, Exa and memory
PAST_API_KEY=past_sk_... python -m scripts.pastdev_smoke   # task B1: write one record and read it back
```

A live round needs `PAST_API_KEY`, `OPENAI_API_KEY`, `PUDDLES_AGENT_MODEL` and `EXA_API_KEY`. Until the gatekeeper (J7) exists, `--gate` picks a stand-in outcome, which the log and memory label as a stand-in. Use a separate past.dev project for these runs so they stay out of the team's memory.

```bash
python -m scripts.run_round --root runs/demo --gate reverted   # round 1, on fixtures/bad_conversations.json
python -m scripts.run_round --root runs/demo --gate kept       # round 2 recalls round 1
```

Each run ends by printing, for every record, the agent that wrote it and the agents that later read it from memory. [`tests/test_memory_wiring.py`](tests/test_memory_wiring.py) checks the same thing over two rounds offline.

| Module | What it does |
|---|---|
| `puddles/memory.py` | past.dev client (ingest, poll, recall, audience setup), memory sentences, recall to verified records |
| `puddles/agents/critic.py`, `researcher.py`, `proposer.py` | Build each input from the log and memory, call the model, check the output, retry once, make the record |
| `puddles/policy.py` | Parse the policy and apply one edit in code; the Safety section is never touched |
| `puddles/decision_log.py` | Reader and appender for the log; builds the budgeted history lists and pattern status |
| `puddles/replay.py` | The replay check: the log, the policy in force, its Safety section and `.meta.json` must agree |
| `puddles/round.py` | Steps 4 to 6 with direct handoffs, then the log entry and the copies into memory |

## Limits

- A simulated user is more cooperative than a real one, so training scores will look better than real use.
- The helpers in ESConv are not clinicians. The evaluation measures good peer support, not therapy.
- Nothing here shows that anyone's mental health improves. That needs a study with real users.
