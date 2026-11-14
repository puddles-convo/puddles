# puddles

A support-conversation helper that improves its own behaviour, built at the SF AI Research Hackathon (November 14-15, 2026), Track 5 (Open).

The helper listens, reflects, asks questions, suggests coping techniques and points people to professional help. It does not diagnose or treat.

**Status:** this README is the agreed design. No pipeline code is in this repo yet. The handoff formats are in `schemas.py`, with one example per handoff in `examples/`; the field names are a proposal for the team to fix before building.

This repo holds the training loop and the evaluation. The device and its cloud brain live in [puddles-stack](https://github.com/puddles-convo/puddles-stack).

## Contents

1. [Training](#1-training)
2. [Training handoffs](#2-training-handoffs)
3. [Inference](#3-inference)
4. [Inference handoffs](#4-inference-handoffs)
5. [How training and inference connect](#5-how-training-and-inference-connect)
6. [Data](#6-data)
7. [Robot side](#7-robot-side)

## 1. Training

Training is the offline improvement loop. It runs in batches in the cloud, never on the device, and repeats with no human in it. Each box is one agent. Each arrow is a handoff (H1 to H7), described in the next section.

```
TRAINING (offline, repeats each round)
+------------------------------------------------+
| 1. SIMULATED USER                              | <--+
| plays a person from an ESConv scenario         |    |
+------------------------------------------------+    |
                        |                             |
                        |  H1: user message           |
                        v                             |
+------------------------------------------------+    |
| 2. HELPER                                      |    |
| replies using the current policy file          |    |
| (1 and 2 alternate until the turn limit)       |    |
+------------------------------------------------+    |
                        |                             |
                        |  H2: conversation record    |
                        v                             |
+------------------------------------------------+    |
| 3. JUDGE                                       |    |
| rates the conversation                         |    |
+------------------------------------------------+    |
                        |                             |
                        |  H3: score record           |
                        v                             |
+------------------------------------------------+    |
| 4. CRITIC                                      |    |
| reads the worst ones, names one failure        |    |
+------------------------------------------------+    |
                        |                             |
                        |  H4: failure pattern        |
                        v                             |
+------------------------------------------------+    |
| 5. RESEARCHER                                  |    |
| searches Exa for how to handle it              |    |
+------------------------------------------------+    |
                        |                             |
                        |  H5: sources + technique    |
                        v                             |
+------------------------------------------------+    |
| 6. PROPOSER                                    |    |
| writes one edit to the policy                  |    |
+------------------------------------------------+    |
                        |                             |
                        |  H6: candidate policy       |
                        v                             |
+------------------------------------------------+    |
| 7. GATEKEEPER                                  | ---+ H7: policy + outcome
| re-tests, runs safety check, keeps or reverts  |
+------------------------------------------------+
```

A round ends when the gatekeeper keeps or reverts the proposed edit. The next round starts from whichever policy file is then in force.

## 2. Training handoffs

| Handoff | From → to | What is handed over | Format |
|---|---|---|---|
| H1 | Simulated user → Helper | One user message. The helper never sees the scenario itself. | Plain text |
| H2 | Helper → Judge | Scenario ID and split, all turns with speaker, policy version | JSON, one record per conversation |
| H3 | Judge → Critic | Empathy, relevance and emotion-change scores, a strategy tag per helper turn, a one-sentence note | JSON, written to memory |
| H4 | Critic → Researcher | Name of the failure pattern, how often it occurred, two or three example conversation IDs | JSON, written to memory |
| H5 | Researcher → Proposer | Source links and a one-line technique from each | JSON, written to memory |
| H6 | Proposer → Gatekeeper | The full new policy file, and which pattern and source it answers | Text file plus a JSON note |
| H7 | Gatekeeper → next round | The policy now in force (new or restored); kept or reverted, scores before and after, safety result | Text file, plus one line in the decision log |

Shared memory: the judge, critic, researcher, proposer and gatekeeper all write their handoff records to a shared memory layer (past.dev), and each reads what the earlier agents wrote. The simulated user has no access to it. The policy file and the decision log live in this repo as plain text.

## 3. Inference

Inference is the live conversation with a person. It makes one pass per user turn, using the policy file that training last approved. Nothing is learned or changed during a conversation.

```
INFERENCE (live, one pass per user turn)
+------------------------------------------------+
| 1. LISTENER                                    | <--+
| turns the person's speech into text            |    |
+------------------------------------------------+    |
                        |                             |
                        |  H1: user message           |
                        v                             |
+------------------------------------------------+    |
| 2. CRISIS CHECKER                              |    |
| flags suicide risk or emergency                |    |
+------------------------------------------------+    |
                        |                             |
                        |  H2: message + risk flag    |
                        v                             |
+------------------------------------------------+    |
| 3. HELPER                                      |    |
| replies using the frozen policy file           |    |
| and its notes about this user                  |    |
+------------------------------------------------+    |
                        |                             |
                        |  H3: reply text             |
                        v                             |
+------------------------------------------------+    |
| 4. SPEAKER                                     | ---+ next turn
| turns the reply into speech                    |
+------------------------------------------------+
                        |
                        |  H4: turn added to the session record
                        v
+------------------------------------------------+
| 5. NOTE-TAKER                                  |
| after the session, saves facts about the user  |
+------------------------------------------------+
```

The listener and the speaker are speech components and make no decisions. They are drawn as boxes so the order is complete.

## 4. Inference handoffs

| Handoff | From → to | What is handed over | Format |
|---|---|---|---|
| H1 | Listener → Crisis checker | What the person said | Plain text |
| H2 | Crisis checker → Helper | The same message and a risk flag: none, concern or crisis. On crisis the helper uses a fixed hand-off response that points to a crisis line. | Text plus a flag |
| H3 | Helper → Speaker | The reply | Plain text |
| H4 | Speaker → Note-taker | The full session, once it ends | JSON, same record as training H2 |
| | Note-taker → memory | Facts about this user for next time | JSON, written to memory |

## 5. How training and inference connect

```
+--------------------+    policy file that passed the gate    +--------------------+
|                    | -------------------------------------> |                    |
|      TRAINING      |                                        |     INFERENCE      |
|   (this repo)      | <------------------------------------- |  (puddles-stack)   |
+--------------------+   session records, with user consent   +--------------------+
```

- **Training to inference:** exactly one thing, the policy file that last passed the gatekeeper. A policy that fails the safety check never reaches a user.
- **Inference to training:** session records, and only when the user has consented. They use the same conversation record format as training (H2), so the judge can score real sessions without changes.
- **Where inference runs:** the device (stack-chan on an M5Stack CoreS3) captures audio and sends it to the cloud brain in `puddles-stack`, whose README places speech-to-text, the language model and text-to-speech there. The listener, crisis checker, helper and speaker are planned to run in that brain; the device plays the reply.
- **What the brain needs from this repo:** the current policy file, set through one config value. See [Robot side](#7-robot-side).

## 6. Data

Datasets are loaded at run time and are not committed to this repo.

| Dataset | Used for | Licence |
|---|---|---|
| [ESConv](https://huggingface.co/datasets/thu-coai/esconv) | Scenarios for training, and the held-out test | CC BY-NC 4.0 |
| [CounselBench-Adv](https://huggingface.co/datasets/izi-ano/CounselBench-Adv) | External validation, never seen by the loop | CC BY-NC-ND 4.0 |
| [VERA-MH](https://github.com/SpringCare/VERA-MH) | Crisis-handling check | Not yet checked |

## 7. Robot side

The robot is a StackChan (M5Stack CoreS3). It cannot run the models, so it is the microphone, speaker and face, and the five inference blocks from section 3 run in a cloud brain. The code is in [puddles-stack](https://github.com/puddles-convo/puddles-stack).

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
| Receives from Team 1 | Crisis checker prompt and fixed hand-off response | Text | Saturday 20:00 |
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
- Crisis: the fixed response plays with the helper off, and with wifi off.
- Soak: a 15-minute conversation with no crash and no answering itself.
- Latency: median and p95 from end of speech to first sound, recorded.

Tasks: [puddles-stack issues V1 to V9](https://github.com/puddles-convo/puddles-stack/issues?q=is%3Aissue+label%3Avoice-path).

## Limits

- A simulated user is more cooperative than a real one, so training scores will look better than real use.
- The helpers in ESConv are not clinicians. The evaluation measures good peer support, not therapy.
- Nothing here shows that anyone's mental health improves. That needs a study with real users.
