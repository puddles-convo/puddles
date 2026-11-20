# Vision

<p class="draft"><b>Draft.</b> Written from the repository's README and design documents for the team to edit. It states intentions, not results; what was measured is under <a href="site:results.html">Results</a>.</p>

## The idea

Many people have no one to talk to at the moment they need it. Puddles is a small desk robot meant to be that first listener: it listens, reflects, asks questions, suggests coping techniques and points people to professional help. It does not diagnose or treat, and it says plainly that it is a robot, not a person and not a therapist.

The harder question is how such a helper gets better without getting less safe. A model that is tuned by hand improves only as fast as someone reads its transcripts, and a model that tunes itself can drift. Vibe Therapist is an attempt at the middle: a helper that improves its own behaviour, one small written change at a time, with every change tested before it is kept and every decision kept on record.

## What we are building

**A helper whose behaviour is a readable file.** What Puddles does is set by a policy: a short list of numbered rules that anyone can read, question and diff. Improvement means a change to that file, not an opaque change to a model. [See the policy the run ended with](policies/training-2026-11-15/policy-v4.md).

**A loop that proposes, and a gate that decides.** Each round the loop reads its weakest conversations, researches how counsellors handle the failure, and proposes one change. A gate written in code, fixed before the run, keeps the change only if quality rises by more than the noise and every safety case still passes. Most proposals are reverted, and that is the design working. [How it learns](site:how-it-works.html).

**Safety that the loop cannot touch.** The safety rules are locked outside the search. In a live conversation nothing is spoken until a crisis checker has answered, and on a crisis the helper's draft is discarded for a fixed response that gives the 988 number. [The live conversation](site:how-it-works.html#the-live-conversation).

**Memory that belongs to the person.** The policy is the same for everyone. What is personal is a short set of notes, kept only with consent, bounded in number, and removed from active memory when the person asks. [The memory](docs/memory.md).

**A body.** The helper is meant to live in a StackChan robot on an M5Stack CoreS3, speaking and listening. The text path behind it runs today; speech is in a separate repository and is not connected through it yet. [The hand-over page for the robot](docs/robot-handover.md).

## What it is not

- Not therapy, not a medical device and not a crisis service.
- Not a replacement for a counsellor, a doctor or the people in someone's life. A good outcome is often that Puddles points a person towards one of them.
- Not validated. Everything measured so far used scripted or simulated people and model judges. See [Limits](site:limits.html).

## Where it stands

A hackathon research prototype, built on 14 and 15 November 2026. The training loop ran unattended for 13 rounds and produced a policy that scored higher than the one it started from, and the result held when measured again. The live text path ran end to end on real models. The honest list of what broke and what is still wrong is under [Results](site:results.html).

## What stands between this and real use

1. **Evidence with people.** Clinician review of the fixed crisis responses, the crisis labels and a sample of replies; a judge calibrated against human ratings; and, eventually, a study with real users. None of this has happened.
2. **The known crisis-path problems.** For example, a keyword match currently overrides the checker even on an explicit denial, and the crisis test set has no hard negatives. These are listed, not fixed.
3. **The voice.** Routing the robot's speech through the checked text path, and meeting the latency a spoken conversation needs.
4. **Operations and privacy.** A deployable service with per-client access, retention rules, and a forget request that reaches every stored copy, not only active memory.
5. **A model of our own.** The loop already runs with a Qwen helper served on Modal, and weight training between rounds exists but has only been tested offline.

Until the first of these is done, Puddles stays a research prototype.
