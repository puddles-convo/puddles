# W2. Memory for a support helper, and learning its structure from data

**Scope and limits.** Every source cited below was opened during this session unless it is marked `[not verified]`. Several pages could not be read: NCBI and PMC returned a captcha, the Stanley-Brown form PDF had no extractable text, and one MI textbook page returned 403. Claims that rest only on search-result snippets are marked `[not verified]`. The commercial systems (Woebot, Wysa, Replika) and Conversation Chronicles were not checked, so they are left out. Labels: **Source:** means what the page says. **Inference:** means my own reasoning.

**Bottom line.** Keep the flat, one-sentence notes. Add four things: a small fixed set of categories, a per-category cap, a `last_confirmed` date, and an update rule that differs by category. Together these stop the notes growing without bound, and they cost about two hours of work. Learning the schema from ESConv is feasible in about two hours for the failure taxonomy. For the user-note schema it can only validate categories, not discover them, because ESConv is single-session.

---

## Part A. How human helpers keep track

| Practice | Categories it tracks | How it is updated | Source |
|---|---|---|---|
| SOAP note | **S**: what the client reports (concerns, mood, events, self-report scores). **O**: what the clinician observes. **A**: the clinician's interpretation, including progress toward treatment goals. **P**: interventions, homework, next-session focus, referrals. | A new note each session, appended. Progress is judged against goals that live elsewhere, in the treatment plan. | [Supanote, SOAP vs DAP vs BIRP](https://www.supanote.ai/blog/soap-notes-the-gold-standard) (vendor blog) |
| DAP note | **D**: the client's report and the clinician's observations, combined. **A**: interpretation and progress toward goals. **P**: next steps. | Appended per session | Same |
| BIRP note | **B**: how the client presents. **I**: the technique used. **R**: how the client responded to it. **P**: next steps. | Appended per session | Same |
| Case formulation, 5 Ps | Presenting problem; predisposing factors (long-standing vulnerabilities); precipitating factors (recent triggers); perpetuating factors (what keeps the problem going); protective factors (strengths, supports). | **Revised in place.** The source calls it "hypothesis, not certainty", to be updated when new information arrives or after crises or changes in support. | [Supanote, 5 Ps guide](https://www.supanote.ai/blog/5-ps-case-formulation-framework-a-clinical-guide) |
| CBT cognitive conceptualisation diagram (CCD) | Relevant history, core beliefs, intermediate beliefs, coping strategies, and for each situation: automatic thoughts, emotions, behaviours. | One diagram per client, refined over time `[not verified]`. PATIENT-Ψ uses 106 CCDs, each tied to one situation. | [PATIENT-Ψ, arXiv 2405.19660](https://arxiv.org/html/2405.19660v2) |
| Helping-skills model (Hill) and ESConv | Hill: exploration, insight, action. ESConv replaces insight with *comforting*, because insight "usually requires re-interpreting users' behaviors and feelings", which is risky for untrained supporters. It has 8 strategies: Question, Restatement or Paraphrasing, Reflection of Feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others. | The stages run within one session. Nothing is carried across sessions. | [ESConv paper, ar5iv 2106.01144](https://ar5iv.labs.arxiv.org/html/2106.01144) |
| Motivational interviewing | Change talk is "any client statement that favors change". Preparatory change talk is DARN (Desire, Ability, Reasons, Need); mobilising change talk is CAT (Commitment, Activation, Taking steps). Also tracked: sustain talk and values. | Tracked within and across sessions; mobilising talk signals readiness `[not verified as a recording practice]`. | [Agents of Change, change talk](https://agentsofchangeprep.com/blog/what-is-change-talk-in-motivational-interviewing/) |
| Stanley-Brown Safety Plan | A form titled "Stanley-Brown Safety Plan", author Gregory K. Brown, University of Pennsylvania (from the PDF metadata). Its six steps are `[not verified]`; the text was unreadable and the steps come from search snippets: warning signs, internal coping strategies, social contacts and settings that distract, people to ask for help, professionals and crisis lines, and making the environment safer. | A single document written with a clinician and revised `[not verified]`. | [Form PDF](https://dsamh-training.utah.gov/_documents/evidencebasedprogramming/SAFETY_PLAN_form_8.21.12.ada.pdf) (metadata only) |
| Problem list and treatment plan with goals | Problems, goals, interventions. SOAP and DAP assessments refer back to the goals. | The plan is revised; session notes are appended. | Supanote (above) supports only the link to goals. The rest is `[not verified]`. |
| Measurement-based care | Short scales (for example PHQ-9) given repeatedly, with the scores fed back into care. | A time series, one score per session `[not verified]`. | The PMC and NCBI pages were blocked by captcha. |
| Therapeutic alliance | Agreement on goals and tasks, and the bond `[not verified]`. | Rated periodically `[not verified]`. | Not opened. |

**One pattern runs through all of this (inference).** Clinicians keep two kinds of record. One is an **append-only session log**: SOAP, DAP and BIRP notes. The other is a **small, revised-in-place summary**: the formulation, the problem list and goals, and the safety plan. Puddles has the same split in its training memory (the decision log versus the policy). Its user memory has only one kind, a flat list of notes. The way the clinical world keeps records bounded is that the summary is revised in place, never appended to.

### What a non-clinical peer-support robot may keep

| Keep (paraphrased, from the person's own words) | Do not keep |
|---|---|
| The current situation and stressors, which overlap with ESConv's `problem_type` and `situation`. | Diagnoses, symptom severity, or scale scores (measurement-based care is clinical). |
| People in the person's life and how they are related. This is the protective-supports row of the 5 Ps, recorded as facts only. | Core beliefs, intermediate beliefs and automatic thoughts. These are the clinician's interpretation, and the same reason ESConv drops insight applies. |
| What helped or did not. This is the R of BIRP and the coping row of a safety plan, in lay terms. | The predisposing and perpetuating rows of the 5 Ps. These are causal hypotheses. |
| Goals and plans the person stated, close to MI's mobilising talk ("I'm going to call my sister"). | A safety plan or risk history. Safety planning is a clinician-led intervention. A stored note saying someone is "at risk" or "not at risk" could bias the helper or falsely reassure it (inference). |
| Things to follow up on (an upcoming event the person asked to have remembered). | Medication details, and verbatim quotes of disclosures. |
| How the person likes to be talked to. This is a lay version of the "tasks" part of the alliance. | Alliance scores or readiness ratings about the person. |

---

## Part B. Published agents with long-term memory

| System or study | What it stores | How it updates | Evidence |
|---|---|---|---|
| MemoryBank / SiliconFriend | The abstract says only that the model can "summon relevant memories" and adapt to the user's personality. It does not give a storage format. | Inspired by the Ebbinghaus forgetting curve: memories weaken or strengthen with elapsed time and significance. SiliconFriend is additionally tuned on psychological dialogues. | [arXiv 2305.10250](https://arxiv.org/abs/2305.10250) |
| PATIENT-Ψ | An explicit client model: the 8-part CCD above, used to simulate patients. | Static per simulated patient | User study with 13 trainees and 20 experts. Experts judged it closer to real patients than GPT-4. [arXiv 2405.19660](https://arxiv.org/abs/2405.19660) |
| Cactus | Client personas plus CBT techniques used by the counsellor. The abstract does not list the persona fields. | Static, one session per persona | The model trained on it (Camel) is reported to beat others on counselling skills. EMNLP 2024 Findings. [arXiv 2407.03103](https://arxiv.org/abs/2407.03103) |
| Multi-Session Chat | Human-human chats over several sessions, in which partners refer back to what they learned earlier. | Retrieval-augmented and summarise-and-recall models beat plain encoder-decoders. Whether there are per-session summary annotations is `[not verified]`. | [arXiv 2107.07567](https://arxiv.org/abs/2107.07567) |
| LoCoMo | LLM-generated dialogues grounded in personas and temporal event graphs, edited by humans. On average 300 turns and 9K tokens per conversation, up to 35 sessions. | Benchmark only: QA, event summarisation, multimodal generation | [arXiv 2402.17753](https://arxiv.org/abs/2402.17753) |
| LongMemEval | 500 questions testing five abilities: information extraction, multi-session reasoning, temporal reasoning, **knowledge updates**, **abstention**. | Proposes indexing, retrieval and reading stages, with session decomposition, fact-augmented keys and time-aware query expansion. | Commercial assistants and long-context LLMs show a **30% accuracy drop** across sustained interactions. ICLR 2025. [arXiv 2410.10813](https://arxiv.org/abs/2410.10813) |
| ES-MemEval / EvoEmo | Multi-session emotional-support dialogues with fragmented, implicit disclosures and evolving user states. Abilities tested: extraction, temporal reasoning, **conflict detection**, abstention, **user modelling**. | Benchmark | Explicit long-term memory reduces hallucination. RAG improves factual consistency but struggles with temporal dynamics and evolving user states. WWW 2026. [arXiv 2602.01885](https://arxiv.org/abs/2602.01885). Search snippets say it has 18 users, up to 33 sessions, and is seeded from ESConv `[not verified on abs page]`. |
| CareCall (CHI 2024) | An LLM voice chatbot for public-health check-ins, with long-term memory | Not detailed in the abstract | From 1,252 call logs and 9 interviews: memory increased health disclosure and familiarity. It was harder to get disclosure about chronic conditions, and users raised **privacy concerns**. The authors recommend choosing deliberately which topics to remember. [arXiv 2402.11353](https://arxiv.org/abs/2402.11353) |

**Harms, from search snippets only (`[not verified]`; full papers not opened).** Users of ChatGPT's memory were surprised by what it remembered and asked for visibility and control ([UGA summary](https://grady.uga.edu/research/ai-never-forgets-exploring-users-perceptions-of-ai-memory-and-chatgpts-memory-feature/)). Users who customised emotional-support bots wanted to delete negative content so it would not resurface (arXiv 2504.12943). Deletion controls led to more honest disclosure (arXiv 2607.06371). I found no opened source measuring stale-memory harms, sycophancy from memory, or dependence in support bots specifically.

**Inference.** Two benchmarks were built specifically to test the cases a flat, add-only store handles worst: **knowledge updates** and **conflicts and evolving state** (LongMemEval, ES-MemEval). Puddles' note-taker already replaces and deletes, which is the right behaviour. What it lacks is a way to see which note a new fact conflicts with, other than reading all of them. Grouping notes by category narrows that search.

---

## Part C. Learning the structure from data

### What ESConv contains

- **Size.** The paper reports 1,053 dialogues and 31,410 utterances (14,855 from supporters, 16,555 from seekers), about 29.8 utterances per dialogue ([ar5iv](https://ar5iv.labs.arxiv.org/html/2106.01144)). The repo now says 1,300 conversations ([GitHub](https://github.com/thu-coai/Emotional-Support-Conversation)). The Hugging Face card shows a train/validation/test split of 910/195/195 ([HF card](https://huggingface.co/datasets/thu-coai/esconv)).
- **Fields** (HF card): `experience_type`, `emotion_type`, `problem_type`, `situation`, `survey_score` (the seeker part has `initial_emotion_intensity`, `empathy`, `relevance`, `final_emotion_intensity`), and `dialog`. The paper also describes per-turn strategy labels, and **seeker helpfulness feedback after every second supporter utterance** on a 1-5 scale. Whether that feedback is present in the released JSON is `[not verified]`; check before relying on it.
- **Extra data.** `FailedESConv.json` holds 196 negative samples dropped during collection (GitHub).
- **Licence.** The HF card says CC BY-NC 4.0. The repo says "for academic research use only". Fine for a hackathon, not for a product.
- **Single-session.** ESConv tells us what people *disclose* in one conversation. It cannot tell us what *turns out to matter next time*, what goes stale, or how facts change. So from ESConv we can induce categories of disclosure, but not update or decay rules. EvoEmo/ES-MemEval (multi-session, seeded from ESConv per snippets) and LoCoMo or Multi-Session Chat (multi-session, general chit-chat) fill this gap. Their licences were not checked.

### Methods for inducing a schema

| Method | How it works | Reported validation | Source |
|---|---|---|---|
| **TnT-LLM** | Stage 1: an LLM generates a label taxonomy zero-shot in several steps and refines it iteratively over batches. Stage 2: the LLM labels data, and the labels train a lightweight classifier. Applied to Bing Copilot conversations. | Human and automatic evaluation. Taxonomies judged more accurate and relevant than baselines. The abstract gives no numbers. | [arXiv 2403.12173](https://arxiv.org/abs/2403.12173) |
| **LLooM** (concept induction) | An LLM repeatedly synthesises sampled text into concepts of increasing generality, each with **explicit inclusion criteria**, then scores documents against the criteria. | Better quality and data coverage than topic models; expert case studies. | [arXiv 2404.12259](https://arxiv.org/abs/2404.12259) |
| **TopicGPT** | The LLM proposes topics with natural-language labels and descriptions, refines them, then assigns documents to them. | Harmonic mean purity against human Wikipedia categories: **0.74** versus 0.64 for the best baseline. | [arXiv 2311.01449](https://arxiv.org/abs/2311.01449) |
| **Clio** | Extracts facets per conversation, clusters semantically, describes each cluster while omitting private details, then builds a hierarchy. Privacy layers include a minimum cluster size and a final check that summaries hold no identifying information. | r = 0.71 correlation with Trust and Safety classifier concern ratings (blog). Accuracy and privacy evaluations are in the paper. | [Anthropic blog](https://www.anthropic.com/research/clio), [arXiv 2412.13678](https://arxiv.org/abs/2412.13678) |

Dialogue slot induction and persona-extraction literature: not opened in this session.

**Inference.** For our purposes, LLooM's *explicit inclusion criteria* and TnT-LLM's *iterative refinement over batches* matter most. The criteria are what the note-taker and critic prompts need. The batch refinement is what makes the process cheap.

### A two-hour procedure on ESConv

Use one strong model for induction and a cheaper one for assignment. Fix the seeds. Every run writes JSON.

**(a) User-note schema (about 60 minutes)**

1. **Sample.** Take 150 train conversations, stratified by `problem_type`. Hold out 50 validation conversations.
2. **Open coding.** For each conversation, prompt: "List facts about the seeker worth remembering for a conversation next week. One plain sentence each. Use the current note-taker's keep and leave-out rules. For each fact, mark it `stable` (true for months) or `episodic` (an event with a date)." This gives roughly 600 to 1,000 facts.
3. **Induce categories, three times.** Use three independent runs on different shuffles of the facts, in the TnT-LLM style. Batch 1 proposes at most 8 categories, each with a name, definition, inclusion criteria, exclusion criteria and 2 examples. Each later batch of 100 facts may add, merge or rename categories, and must keep an `other` category.
4. **Check stability.** An LLM maps each category in run A to its best match in runs B and C, or to none. Report the matched fraction. Then assign 200 facts with two different runs or models and compute Cohen's kappa per category. Keep categories that match in at least 2 of 3 runs and have kappa ≥ 0.6. That threshold is a convention, not something taken from a source.
5. **Check coverage.** Code the 50 held-out conversations and assign their facts. Report the share that falls in `other` (target under 10%) and the share of conversations that touch each category.
6. **Crosswalk.** Map each surviving category to the Part A "keep" list, and reject any that fall in the "do not keep" list. This is where hand-picked safety constraints override what the data suggests.

**(b) Failure-pattern taxonomy (about 60 minutes)**

1. **Find failure segments.** If per-turn feedback is present, take supporter utterances directly before a seeker rating of 1 or 2. Otherwise take the 100 conversations with the lowest `empathy` plus `relevance`, the 196 FailedESConv samples, and the round's own low-scoring simulated conversations.
2. **Open coding.** For each segment, prompt: "In one sentence, what did the helper do that made this worse? Name the ESConv strategy involved." Pass the strategy labels in.
3. **Induce and stabilise** as in (a), steps 3 and 4.
4. **Discrimination check.** Tag 100 low-rated and 100 high-rated conversations with the taxonomy. Keep a code only if it appears clearly more often in the low-rated set, for example at a ratio of 1.5 or more (my threshold). A failure code that is just as common in good conversations is noise.
5. **Output.** `failure-taxonomy-v1.json`. Each entry has `code`, `definition`, `include`, `exclude`, `examples[2]`, `esconv_strategy`, `freq_low`, `freq_high`, and `stability`.

**What to do with it (inference).** Give the critic this list as its allowed `name` values, plus "new: …" for anything outside it. This addresses the fragility noted in `docs/agents.md` §3 "Still open", where a recurrence is matched to an earlier pattern only if the critic happens to reuse the same free-text name.

---

## Part D. Recommendation: user-memory schema

| Category | What goes in it (from Part A "keep" list) | Updates by | Cap |
|---|---|---|---|
| `situation` | What they are dealing with now | **Replace** when the situation changes; delete when they say it is resolved. | 3 |
| `people` | One note per person: name or role, relation, one fact | Replace that person's note | 6 |
| `what_helps` | A coping step or approach, and whether it helped or did not | Replace when new evidence contradicts it | 5 |
| `goals` | Plans or commitments the person stated (MI mobilising talk) | Replace when progress is made; delete when done or dropped. | 3 |
| `follow_up` | A dated event, or something they asked to have remembered | **Expires**: code removes it 14 days after `due_date` unless it is confirmed again. | 3 |
| `preferences` | How they like to be talked to | Replace | 3 |

**Fields on each note.** Add these to `UserNote` and the note-taker's `NoteAction`:
- `category`: one of the six.
- `fact`
- `first_seen`
- `last_confirmed`: code sets it when the note-taker marks the note `confirm`, a new no-op action meaning "still true, mentioned again".
- `source_session`
- `due_date`: only for `follow_up`.

**How this stops the store "exploding" (question 1).**
- The caps add up to at most 23 notes, roughly 500 to 700 tokens in the prompt (my estimate).
- Code rejects an `add` to a full category unless the same output also deletes or replaces a note in that category.
- The note-taker sees notes **grouped by category**, so it compares a new fact only with its neighbours. This is a lightweight answer to the conflict-detection weakness found by ES-MemEval.
- `last_confirmed` gives the note-taker a principled choice of what to drop: the least recently confirmed note.
- This applies the clinical pattern from Part A: a small summary revised in place, with history living elsewhere. MemoryBank-style decay by time is not needed, because explicit caps are simpler and auditable.

**Deliberately excluded.** Diagnoses, scores, beliefs and thoughts in the CBT sense, risk or safety-plan content, medication, verbatim disclosures, and sensitive details about third parties beyond their relation. Consent gating and "forget always works" stay unchanged.

**How it differs from the current notes.** The current notes are one flat list with no categories, no cap and no dates other than the date in the text. The proposal keeps the same one-sentence notes and the same add/replace/delete actions. It adds one enum, two dates, a `confirm` action, caps enforced by code, and an expiry for `follow_up`.

**Is it worth building before tomorrow? Yes, the minimal version.** That is the `category` enum, the caps, `last_confirmed`, and a grouped prompt block. It changes `schemas.py`, the note-taker prompt and its checks, which is about 1 to 2 hours. It turns "no cap" into a guarantee you can demo. **Skip** the data-driven schema induction for the user notes until after the hackathon: ESConv cannot tell you about cross-session value, and the six categories above already map onto the existing keep rules. **Do** run part (b) if anyone has an hour, because it directly firms up the critic's pattern names. There is no evidence in the sources I opened that categorised notes improve support quality. The case for them is bounded size, easier review, and simpler conflict handling.

---

## Open issues and weak evidence

- The clinical sources for SOAP, DAP, BIRP and the 5 Ps are vendor blogs. The details of measurement-based care, the therapeutic alliance, the treatment plan and the Stanley-Brown steps are `[not verified]`; the primary pages were blocked or unreadable. Before quoting them, check against Stanley & Brown (2012, *Cognitive and Behavioral Practice*) and a measurement-based-care review.
- Whether ESConv's released JSON contains per-turn strategy labels and seeker feedback was not confirmed from the HF preview. Part C(b) step 1 depends on it.
- ESConv size: the paper says 1,053, the repo says 1,300. The HF splits total 1,300.
- ES-MemEval's dataset size and its ESConv seeding come from search snippets only.
- The harm and user-perception studies other than CareCall were not opened.
- Woebot, Wysa, Replika and Conversation Chronicles were not checked.
- The caps (23 notes), the kappa ≥ 0.6 bar, the 10% `other` target, the 1.5× discrimination ratio and the 14-day expiry are my choices, not values from any source.
- Excluding risk history from notes is a safety judgement. A clinician on the team should confirm it, since some might argue that remembering a prior crisis disclosure aids continuity.
