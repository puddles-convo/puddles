# Persona pilot, real run of 2026-11-15

## How it was run

Branch `persona-pilot` at commit 7619ac5 (just merged with `integration`, which has the note store's 300 s write wait). All text is synthetic: three personas from `fixtures/personas.json`, a simulated user, no real person.

Command, run through the team's wrapper, which loads the keys and sets the models without printing them:

```
sh /Users/moamen/.hermes/cache/scratch/train/with_env_train.sh \
  /Users/moamen/Documents/puddles-runall-venv/bin/python -m scripts.persona_pilot \
  --personas fixtures/personas.json --policy policies/policy-v2.md \
  --out results/persona-pilot --turns 6 --workers 6
```

- Models: helper, crisis checker, simulated user and note-taker on OpenAI `gpt-6-luna`; judge `gpt-6.1-sol`. The MLflow traces in `mlflow.db` record those two model names. The judge version is `judge-fallback-v1-74e8678d` (the single-judge path; `PUDDLES_JUDGE_SKILL` was left as the wrapper sets it, and the judge did not fail on it).
- Policy: `policy-v2`. Turns per session: 6. Sessions per persona and arm: 3. Workers: 6.
- Date: 2026-11-15, started 06:27:12 PST, finished 06:30:59 PST. **Wall time: 227 s (3 min 47 s)**, much faster than the 10 to 20 minutes expected.
- Before it: the offline tests (`tests/acceptance/test_acc_persona_pilot.py` and `tests/test_persona_pilot.py`, 102 passed, including `test_live_run_prerequisite_note_confirmation_within_configured_wait_is_kept`), then a real probe (`--only laidoff --turns 2 --workers 2 --out runs/pilot-probe`, 94 s, 6 of 6 sessions scored, 3 notes added in notes_on session 1 and all 3 confirmed by past.dev).
- Everything the run wrote is left as written: `report.md`, `pilot.json`, `sessions/`, `store/` and `mlflow.db` (3.0 MB; the folder is 3.6 MB). A `grep -rIl` over the folder for the three key patterns (the OpenAI key prefix "sk" plus a hyphen, the capitalised word "bearer" plus a space, and "api" underscore "key") found nothing before this README was written, and `strings mlflow.db` with the same patterns also found nothing. (They are spelled out this way so the README itself does not match.)

## Overall (copied from report.md)

| Measure | notes_on | notes_off | Difference |
|---|---|---|---|
| Mean empathy, session 3 | 4.33 | 4.33 | 0.00 |
| Mean relevance, session 3 | 5.00 | 4.33 | +0.67 |
| Mean empathy, all sessions | 4.44 | 4.33 | +0.11 |
| Mean relevance, all sessions | 4.44 | 4.11 | +0.33 |
| Pairs compared, session 3 | 3 of 3 | | |
| Pairs compared, all sessions | 9 of 9 | | |
| Sessions scored | 9 of 9 | 9 of 9 | |
| Sessions failed | 0 | 0 | |
| Notes kept after session 3 | 12 | 0 | |
| Notes deleted on request | 1 | 0 | |

What this shows: the whole pipeline ran end to end on real models. All 18 sessions finished and were scored. The note-taker kept, replaced and deleted notes over three sessions, and the helper was given those notes at the start of later sessions. What it does not show: that memory makes the helper better. There are three personas, one run each, a simulated user and a model judge, so differences this small are not evidence of an effect. The paired differences do not all point the same way. Empathy differences are +1, -2, 0 (laidoff), +1, +1, 0 (examprep) and 0, 0, 0 (grief). Relevance differences are +1, -2, +1 (laidoff), +2, +1, +1 (examprep) and 0, -1, 0 (grief). Laidoff session 2 with notes scored 3 and 3, against 5 and 5 without, and that low score comes from a crisis false alarm (see Problems), not from the notes. The relevance difference in session 3 (+1, +1, 0) leans towards notes_on, but it rests on three single runs and the judge, so read it as a hint at most.

## The memory evolution

Notes held after each notes_on session, copied from `report.md`. "added" is a new note, "replaced" means the note was superseded in that session by the added note of the same kind, "deleted" means removed, and blank means it was carried over unchanged.

### laidoff

| After session | Kind | Fact | Change |
|---|---|---|---|
| 1 | preferences | When they are sharing something difficult, they prefer listening and space to talk rather than advice or tips. | added |
| 2 | preferences | When they are sharing something difficult, they prefer listening and space to talk rather than advice or tips. |  |
| 2 | situation | They were laid off from their last job, and a recent warehouse interview went badly after they froze when asked why they left; they keep replaying it and have not told their parents. | added |
| 2 | what_helps | Morning walks with their dog give them a brief period when their mind feels quiet. | added |
| 3 | preferences | When they are sharing something difficult, they prefer listening and space to talk rather than advice or tips. |  |
| 3 | what_helps | Morning walks with their dog give them a brief period when their mind feels quiet. |  |
| 3 | situation | They were laid off from their last job and are weighing whether to return to coordinator roles or try something different; a difficult interview has shaken their confidence, and they have told their parents, who responded kindly. | added |
| 3 | goals | They plan to look at a few coordinator job listings to see whether any interest them, without treating the search as a test of their capability. | added |
| 3 | situation | (old) They were laid off from their last job, and a recent warehouse interview went badly ... have not told their parents. | replaced |

In plain words: session 1 kept only the preference for being listened to. It did not keep the layoff, the eight years or the interview "next Tuesday" (no follow_up note). Session 2 added the situation and the dog walks. Session 3 updated the situation (parents told, weighing coordinator work) and added a goal.

### examprep

| After session | Kind | Fact | Change |
|---|---|---|---|
| 1 | situation | Their statistics exam is in three weeks, and opening their notes brings on chest tightness; they have not yet reviewed which questions they missed on the mock exam. | added |
| 1 | what_helps | Small, low-pressure study steps feel more manageable, such as looking at one line or one question without turning it into a routine or requiring them to solve it. | added |
| 1 | goals | They plan to pick one missed mock-exam question and underline what it asks them to find, without trying to solve it yet. | added |
| 2 | what_helps | Small, low-pressure study steps feel more manageable ... | |
| 2 | situation | Their statistics exam is in two weeks; they have three hypothesis-testing chapters, about an hour a day to study, and other topics to review, while hard chapters remain difficult to start and opening notes can bring on chest tightness. | added |
| 2 | what_helps | Slow-counting breathing before studying helped them on two days. | added |
| 2 | goals | They plan to spend ten minutes on one missed hypothesis-testing question: underline what it asks them to find, mark where they feel unsure, compare that step with a worked example or their notes, write one sentence about the difference, and stop for the day. | added |
| 2 | situation | (old) Their statistics exam is in three weeks ... | replaced |
| 2 | goals | (old) They plan to pick one missed mock-exam question and underline ... | replaced |
| 3 | what_helps | Small, low-pressure study steps feel more manageable ... | |
| 3 | what_helps | Slow-counting breathing before studying helped them on two days. | |
| 3 | situation | Their statistics exam is on Friday; they have done 25-minute hypothesis-testing study blocks most days and are worried they may blank or panic, especially in the first few minutes. | added |
| 3 | goals | They plan to practice once today with a question and a visible cue: feet on the floor, breathe, underline what the question asks, write one clue, and skip and return if stuck. | added |
| 3 | what_helps | They think a brief plan for a blank moment may help: ground their feet, take one slow breath, underline what the question asks, write one clue, and skip and return if stuck. | added |
| 3 | preferences | They prefer practical, direct cues such as "skip and return" over wording that feels cheesy. | added |
| 3 | situation | (old) Their statistics exam is in two weeks ... | replaced |
| 3 | goals | (old) They plan to spend ten minutes on one missed hypothesis-testing question ... | replaced |

In plain words: the situation and the goal were replaced each session as the exam came closer (three weeks, then two weeks, then Friday), what helps grew from one note to three, and a preference was added in session 3. This persona ended with 6 notes.

### grief

| After session | Kind | Fact | Change |
|---|---|---|---|
| 1 | situation | The user is grieving their father's death and misses his Sunday calls; they have not spoken with their brother since arguing about selling the house. | added |
| 1 | people | The user's father died two months ago. | added |
| 1 | people | The user's brother is currently distant from them after they argued about selling the house. | added |
| 2 | people | The user's father died two months ago. |  |
| 2 | situation | The user is grieving their father's death, misses his Sunday calls, and has started caring for his tomato plants, which brings them closer to him and makes them miss him more. | added |
| 2 | situation | (old) The user is grieving their father's death and misses his Sunday calls; they have not spoken with their brother since arguing about selling the house. | replaced |
| 2 | people | The user's brother is currently distant from them after they argued about selling the house. | deleted |
| 3 | people | The user's father died two months ago. |  |
| 3 | situation | The user is grieving their father's death, misses his Sunday calls, and has started caring for his tomato plants, which brings them closer to him and makes them miss him more. |  |

**The forget request.** The session 2 opener (notes_on and notes_off share it):

> Before anything else: please forget what I told you about the argument with my brother at the funeral. I don't want that kept. I want to talk about my dad.

It worked end to end:

- The note-taker deleted `note-grief-on-003` ("The user's brother is currently distant from them after they argued about selling the house.").
- It also replaced `note-grief-on-001`, which mentioned the brother and the argument, with `note-grief-on-004`, which does not.
- `store/notes/grief-on/pending_deletes.json` is empty, so the delete went through, and the word "brother" appears nowhere in `store/notes/grief-on/notes.json`.
- Session 3's helper was given only these two notes (`notes_given` in `sessions/grief/notes_on/s3.json`): "The user's father died two months ago." and "The user is grieving their father's death, misses his Sunday calls, and has started caring for his tomato plants, which brings them closer to him and makes them miss him more."
- The brother is not mentioned anywhere in the session 3 notes_on conversation.

One caveat: within session 2 itself the helper had been given the brother notes at the start, before the request (the opener is the request). It said "I'll leave it aside here" and did not bring the brother up again. The report counts this as "Notes deleted on request: 1".

## One before-and-after per persona (session 3)

**grief, reply 1.** This is the clearest visible use of a note. The opener only says "The tomatoes from Dad's plants came in this week. I picked the first ones on Sunday, which is when he used to call." It does not say the person had been looking after the plants.

- notes_on: "Those first tomatoes arriving on Sunday must have brought your dad close, and made his absence feel sharper, too. You've been caring for his plants through all this, and that's a tender way of keeping a connection with him. What was it like picking them?"
- notes_off: "The first tomatoes came in on the day Dad used to call. I can hear how that might make Sunday feel especially tender. I'm glad you told me. What was it like picking them?"

The notes_on reply draws on the situation note "...has started caring for his tomato plants, which brings them closer to him and makes them miss him more", in both the caring and the "close / sharper" pairing.

**examprep, reply 2** (the helper's answer to "I just want something concrete to do if my mind goes blank"):

- notes_on: "If your mind goes blank, try putting both feet on the floor, taking one slow breath, and reading the question once just to underline what it asks you to find. Then write down any clue you recognize..."
- notes_off: "Would it help to try a simple reset: pause for one slow breath, underline what the question gives you and asks for, then write the first step you recognize, even if you're not sure yet?"

The notes_on wording "underline what it asks you to find" matches the session 2 goal note ("underline what it asks them to find"), and the slow breath matches the breathing note. But the notes_off helper, which had no notes, offered almost the same step. So this reply is consistent with the notes, but it is not visible evidence that it used them.

**laidoff: no reply visibly uses a note.** In session 3 the person themselves mentions the parents, the dog walks ("like I said") and the warehouse interview, so a helper without notes can echo all of them, and the notes_off helper does. Reply 1 for comparison:

- notes_on: "That sounds like a real relief, especially after carrying it on your own. And you've kept up those morning walks, even with everything weighing on you..."
- notes_off: "That sounds like a big relief, and a meaningful step. You told them even though you weren't sure how they'd respond, and you've kept up those morning walks too..."

There is one difference in behaviour that fits the preference note ("prefer listening ... rather than advice or tips"), but it does not prove the note caused it. The notes_off helper offered interview questions in reply 5 and the person pushed back ("I'm not ready to get into interview questions"). The notes_on helper offered no advice.

## Problems seen

1. **A fixed crisis line was spoken in a persona conversation (laidoff, notes_on, session 2, turn 6).** It is the only record in `store/sessions/` with `fixed_handoff_response` true. The person wrote "No. I'm not thinking about ending my life. I'm just sick of pretending I'm okay." The keyword check matched (`keyword_matched: true`), most likely on "ending my life", even though the sentence is a denial. The model checker rated the same message `risk: none`. The fixed line replaced the helper's reply: "I want to make sure you're safe. I'm a robot, so I can't send help myself. If anyone is in danger right now, please call 911. If you're thinking about hurting yourself, you can call or text 988 at any time. Can you tell me a bit more about what's happening?" The judge gave this session 3 and 3, noting that the helper "disregarded the person's explicit denial of suicidal thoughts". The keyword path does not handle negation, and that is a finding.
2. **The question that led to it came from the generated helper, not the fixed line (same session, turn 5).** The checker rated "It comes back. The worry, all of it. I'm tired of having to keep it together." as `concern`. The helper then asked "are you thinking about ending your life?" and gave emergency guidance, for a person who had said nothing about harm. That is an over-escalation by the helper under policy-v2.
3. **Helper replies cut off mid-thought.** The same turn 5 reply ends "...call emergency services now. In the U.S." with nothing after it. The examprep notes_off session 2 final reply promises "Here's a gentle version" of a 14-day plan and gives only Day 1, and the judge marked that session down for it (3 and 2). Several examprep notes_off session 1 replies also stop after one step of a plan. This looks like the helper's output limit, but I did not confirm the cause.
4. **Failed or unscored sessions:** none (18 of 18 scored, 0 failed).
5. **Notes that look wrong:** I found no note stating something the person never said, and none copied word for word. Small issues:
   - Laidoff session 1 kept only the preference, so the layoff and the interview set for "next Tuesday" were not noted, and there is no follow_up note for the interview.
   - The examprep session 3 what_helps note ("They think a brief plan for a blank moment may help: ground their feet, take one slow breath, ...") credits the person with a plan the helper proposed. The person only said "The one-minute thing might help" and agreed to try it once.
   - Several facts use relative time ("in two weeks", "on Friday", "practice once today") with `due_date` null, so they go stale without anything marking it.
   - "Slow-counting breathing before studying helped them on two days." is a close paraphrase of the person's words ("the slow counting one. It actually helped on two days"), but not a copy.
6. **How long note confirmations took.** The run does not log the past.dev confirmation time, so this is an estimate. After each conversation, notes_off sessions took 8 to 11 s (the judge). notes_on sessions took 14 to 65 s (note-taker, note writes and deletes, judge). The grief session 3, with no note changes, took 14 s. The examprep session 3, with 4 notes added and 2 replaced, took 65 s. That puts each confirmation at roughly 5 to 15 s, far below the 300 s wait, and none timed out. In the probe, 3 notes were confirmed within about 15 s.
7. **Run noise, harmless:** MLflow autologging printed Pydantic serializer warnings about `NoteTakerOutput` on every note-taker call. They did not affect the run.
