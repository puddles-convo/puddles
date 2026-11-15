# W1. Agent memory systems and cognitive science, mapped onto puddles

Scope: how LLM-agent systems and human-memory theory structure and update memory, and what that means for our two stores: training memory plus policy file, and user memory. Every cited claim is from a page I opened in this session. Claims I could not open a source for are marked `[not verified]`. My own reasoning is marked **Inference**.

Blocker note: PubMed, PMC and Nature pages refused automated fetches, so several cognitive-science primaries (reconsolidation, interference, schema theory, Soar) are thin or unverified. See the last section.

## Short answers to the four questions

1. **Bounding growth.** The systems that work keep memory as **itemised entries with IDs, change them with small local edits, and prune by a score or by deduplication**. They do not keep rewriting one summary. ACE shows that whole-context rewriting can collapse. Our training side already works this way. Our user side has no bound on the prompt, and it will need one.
2. **Concept types.** The four training record types are enough. What is missing is **per-rule evidence counters** and a **canonical pattern identity**. On the user side, untyped notes work while there are only a few. A small `kind` field plus a `last_confirmed` date is the cheapest structure that makes capping and expiry possible.
3. **How others do it.** See Part A (table) and Part B.
4. **Learning the structure from data.** This is mostly out of scope for this file. A short pointer is in C.3.

---

## Part A. LLM-agent memory systems

### A.1 Summary table

| System | What it stores (schema) | Update operations, who decides, when | What bounds growth | Reported evidence | Known failure modes |
|---|---|---|---|---|---|
| **Generative Agents** (Park et al.) [1] | A memory stream of natural-language observations, plus *reflections* stored back in the stream with pointers to the memories they cite | Append observations. An LLM writes reflections when the summed importance of recent events exceeds 150, about 2 to 3 times a day | Retrieval only: score = recency (exponential decay, factor 0.995 per game hour since last retrieval) + importance (LLM rating 1 to 10, made at write time) + relevance (embedding cosine). Each is min-max normalised, all weights are 1, and the top memories that fit the context go in. The store itself is not bounded | Ablations: observation, planning and reflection each contribute to believability [1] | Store grows without limit, and only the prompt is bounded |
| **MemGPT / Letta** [2][3] | Memory tiers: in-context memory blocks and out-of-context archival storage | The agent moves data between tiers with function calls. In Letta's *sleep-time agents*, a separate agent rewrites the primary agent's memory blocks during idle time, and the primary agent cannot edit its own core memory | Fixed-size in-context blocks, with paging to external storage | The sleep-time compute paper (arXiv 2504.13171) reports a "Pareto improvement" on AIME and GSM. The blog gives no figures [3] | Continuous rewriting is exposed to the summary drift that ACE documents (below). **Inference** |
| **Mem0** [4] | Short natural-language facts in a vector store. Mem0g adds a graph of entities and relations | For each candidate fact, retrieve the top s = 10 similar memories, then the LLM picks **ADD / UPDATE / DELETE / NOOP** by tool call. UPDATE replaces a memory only if the new fact carries more information. Extraction sees a running summary, the last m = 10 messages and the new message pair | Deduplication against similar memories at write time, plus retrieval top-k | LOCOMO LLM-as-judge: Mem0 66.88, Mem0g 68.44, **full context 72.90**. The paper reports 91% lower p95 latency and over 90% token savings compared with full context [4] | **Full context still scores higher.** Memory buys cost and latency, not accuracy. Base Mem0 deletes a contradicted memory. Mem0g instead marks a relation invalid, to keep history for temporal questions [4] |
| **Zep / Graphiti** [5] | A temporally aware knowledge graph that keeps historical relationships | Not opened in detail | Not opened in detail | DMR 94.8% vs MemGPT 93.4%. LongMemEval: up to 18.5% higher accuracy and 90% lower latency than baselines [5] | `[not verified]` beyond the abstract |
| **Reflexion** [6] | Verbal self-reflections in an episodic buffer | The agent reflects after task feedback, and the reflection is used in the next trial | Not stated on the abstract page | HumanEval pass@1 91% vs 80% for GPT-4 [6] | Bound not verified |
| **ExpeL** [7] | A list of natural-language *insights*, each with an importance count | The LLM applies **ADD, EDIT, UPVOTE, DOWNVOTE** while comparing failed and successful trajectories for the same task, and while reviewing sets of successes. Each new insight starts at count 2. UPVOTE and EDIT raise the count, DOWNVOTE lowers it, and the insight is **removed at 0** [7] | Removal by vote count | The abstract says performance rises as experience accumulates. I did not open the numbers | Rests on LLM judgements of agreement |
| **Dynamic Cheatsheet** [8] | A self-curated "cheatsheet" of short strategies, code and insights | The LM rewrites the cheatsheet at test time, with no labels | Concision of the curated sheet | Game of 24 (GPT-4o): 10% → 99%. AIME accuracy more than doubled for Claude 3.5 Sonnet. GPQA-Diamond +9%, MMLU-Pro +8% [8] | **Context collapse** (from ACE [9], next row) |
| **ACE (Agentic Context Engineering)** [9] | A "playbook" of **bullets**. Each bullet has an ID and **helpful / harmful counters**, and holds one strategy, concept or failure mode | Generator → Reflector → Curator. The Curator emits small *delta* entries, and **non-LLM code merges them**. The full context is never regenerated | **Grow-and-refine**: append new bullets, update existing ones in place, then de-duplicate by embedding similarity. Refinement runs either after every delta or lazily, only when the context window is exceeded | +10.6% on agent benchmarks, +8.6% on finance. Matches the top production agent on the AppWorld leaderboard average [9] | Named failure modes: **brevity bias** and **context collapse**. In one case a Dynamic Cheatsheet context on AppWorld went from 18,282 tokens (accuracy 66.7) to **122 tokens (57.1)**, below the no-adaptation baseline of 63.7. Without reliable feedback, the context "can be polluted by spurious signals" [9] |
| **GEPA** [10] | A pool of prompt candidates, from whose **Pareto frontier** it combines lessons | It reflects in natural language on sampled trajectories, proposes prompt edits, and tests them | The frontier keeps only non-dominated candidates (selection detail not opened) | About 6% better than GRPO on average and up to 20% better, with up to 35x fewer rollouts. Over 10% better than MIPROv2, e.g. +12% on AIME-2025 [10] | |
| **AlphaEvolve** [11] | A **program database** of scored programs | LLMs propose programs, automated evaluators score them, and an evolutionary algorithm picks which stored programs seed future prompts | Evolutionary selection over scored entries | E.g. 0.7% of Google's fleet compute recovered. Matched the state of the art on about 75% of more than 50 open problems and improved on about 20% [11] | Needs an objective, automatic evaluator |
| **CoALA** [12] | A framework, not a system. Memory types: **working, episodic, semantic, procedural**. Internal actions: retrieval, reasoning, learning (writing to long-term memory) | | | | The paper states that "modifying and deleting (a case of 'unlearning') are understudied" [12] |
| **LongMemEval** [13] | A benchmark of 500 questions over five abilities: information extraction, multi-session reasoning, **temporal reasoning, knowledge updates, abstention** | | Proposed design: indexing → retrieval → reading. Session decomposition, fact-augmented keys, time-aware query expansion | Commercial assistants and long-context LLMs show "a 30% accuracy drop" across sustained interactions [13] | Updates and temporal questions are a named hard category |

### A.2 The two cases closest to ours

**Self-improving loops with an experiment record** (our training memory plus policy). The systems that cope with growth share three habits:
- **A scored store, not a narrative.** AlphaEvolve keeps scored programs, GEPA keeps a Pareto frontier of tested candidates, and ExpeL and ACE keep items with counters [7][9][10][11].
- **An external evaluator decides what survives.** Programs are kept on evaluator scores, and GEPA tests every proposed edit [10][11]. ACE's own limitations section says that without reliable feedback the context degrades [9].
- **Small, attributable edits applied by code.** ACE's deltas are merged by non-LLM logic so that the context is never rewritten whole [9].

Our design already has all three: the gatekeeper measures against noise, the proposer makes one edit that code applies, and the decision log is the record. None of the pages I opened gives a mechanism for "do not repeat a failed attempt" beyond keeping scored history in view. That is my reading of these pages, not a claim any of them makes. Our complete `earlier_edits` list built from the log is the same idea.

**Shared memory across agents.** The clearest example I opened is Letta's sleep-time split: one agent owns writes to memory and the other only reads [3]. ACE separates the roles too: the Reflector proposes and the Curator plus code commits [9]. Our "orchestrator writes, agents return results" rule is the same single-writer pattern. Per-agent identity and audiences add provenance on top.

---

## Part B. Cognitive science

| Mechanism | Claim | Source | Computational form we could implement |
|---|---|---|---|
| Memory taxonomy | Working, episodic (past experience), semantic (knowledge about the world and self), procedural (skills in weights or code) | CoALA's adaptation of the classic split [12]. The original psychology sources were not opened `[not verified]` | Our mapping: decision-log records = episodic. The policy file = procedural, because it changes behaviour. User notes = semantic facts about a person. Failure patterns = semantic generalisations drawn from episodes |
| Complementary learning systems | The hippocampus learns fast, the neocortex slowly. New memories are replayed *interleaved* with old ones so they integrate without catastrophic interference | McClelland, McNaughton & O'Reilly 1995, *Psychological Review* 102(3):419-457. Seen via a search-result listing of the author's PDF. The PDF itself failed on a certificate error, so treat this as `[partly verified]` | Fast store = raw records and session notes. Slow store = the policy, changed rarely and only after testing against *all* validation conversations together. The gate's validation set works like interleaved replay: a new rule must not break old behaviour. **Inference** |
| Forgetting curve (Ebbinghaus) | Replicated. A summed-exponential (Memory Chain) fit is slightly better than a power function (AIC difference about 1.5). There is probably a rise from the 24-hour point, possibly a sleep effect | Murre & Dros 2015, PLOS ONE [14] | Retention R(t) as a decreasing function of time since the last rehearsal. Use it as a ranking signal, not a deletion rule |
| ACT-R base-level activation | B_i = ln( Σ_{j=1..n} t_j^(−d) ), where n is the number of prior uses of chunk i, t_j is the time since the j-th use and d is the decay. Full activation is A_i = B_i + Σ_j W_j·S_{j,i}, where the second term is context association. The equation follows the power law of forgetting | Equations as written in arXiv 2505.05083 [15], which cites Anderson & Schooler 1991 and Anderson et al. 2004. The ACT-R site says retrieval depends on "subsymbolic retrieval equations" over context and usage history [16]. d = 0.5 is the conventional default `[not verified from a primary source]` | Per note, store the list of session indices where it was used or confirmed. Rank notes by B_i (frequency and recency combined) plus a relevance term |
| Gist vs verbatim (fuzzy-trace theory) | Verbatim and gist traces are encoded, stored and retrieved independently. Verbatim fades with delay and gist persists. Gist also drives false memories, including vivid "phantom recollection" | Reyna, Corbin, Weldon & Brainerd 2016, *J. Applied Research in Memory and Cognition* [17] | Store gist sentences (which we already do: "said they have been feeling very low"). Also a warning: an LLM consolidating several notes into one gist can *create* false facts, so merges must cite their source note IDs |
| Reconsolidation | A retrieved memory becomes labile and can be updated, not just strengthened | Nader et al. 2000 and later reviews. Pages blocked, `[not verified]` | Update on retrieval: when a note is used in a session and the person confirms or corrects it, the note-taker replaces it. Our note-taker already sees all notes after every session, which is this pattern. **Inference** |
| Schema theory, interference, spacing, Soar | Not opened | `[not verified]` | Not used in the recommendations |

---

## Part C. Recommendations

### C.1 Where we already match good practice

- **The policy is an itemised playbook with permanent IDs, changed by one small delta that code applies, and merging is banned.** This is the ACE design, and it is the stated fix for context collapse [9]. Keep the merge ban: ACE's 18,282 → 122-token collapse is what a "merge and shorten" step can do.
- **A gate with measured noise is the external evaluator** that ACE, GEPA and AlphaEvolve rely on [9][10][11]. ACE warns that contexts degrade without reliable feedback, and our gate is that feedback. **Inference:** this matters more for us than for ACE, because LLM-judge scores are noisy.
- **Single writer, the log as authority, memory as a searchable copy.** This matches the sleep-time and Curator role splits [3][9].
- **The note-taker's add / replace / delete over the existing notes is essentially Mem0's ADD / UPDATE / DELETE / NOOP** [4]. An empty list is our NOOP.
- **The training store will not explode in a hackathon.** It gets four add-only entries per round, so 50 rounds is about 200 short sentences. Recall is top-k, and full content is loaded from files. No change needed.

### C.2 Where the literature says it will break, and what to change (ranked)

| # | Store | Problem | Mechanism and source | Concrete change | Hours | Risk |
|---|---|---|---|---|---|---|
| 1 | User | **All live notes go into the prompt, with no cap.** The prompt grows linearly with sessions. The Mem0 numbers cut both ways: full context beats selective memory on accuracy (72.90 vs 66.88 [4]), so all-in is right while notes are few and wrong only once they cost too much | Retrieval top-k under a budget [1][4] | Keep "all notes" up to a **budget of about 600 tokens or 30 notes**. Above it, include pinned notes first (rec. 2), then rank the rest by recency of confirmation. Notes that are left out are *not deleted* | 1-2 | Low. Risk that a relevant note falls outside the budget. Log which notes were included |
| 2 | User | Untyped notes cannot be capped or expired sensibly. A done follow-up stays forever | CoALA types [12]. Our note-taker guidance already lists the categories | Add `kind` ∈ {`situation`, `helps`, `does_not_help`, `person`, `follow_up`, `asked_to_remember`} and `last_confirmed` (date) to `UserNote`. The note-taker returns `kind`. Code pins `asked_to_remember` and open `follow_up`. A `follow_up` the helper has raised is deleted or turned into a `situation` note by the next note-taker run | 2 | Low. The LLM may mislabel kinds, which only affects ordering |
| 3 | User | No use history, so recency and frequency cannot drive the ranking | ACT-R base-level activation [15]. Generative Agents recency decay [1] | Store `used_in_sessions: [int]`. Rank by B = ln Σ (s_now − s_j + 1)^(−0.5), with time measured in sessions. Fall back to `last_confirmed` if that is too much for tomorrow | 1 | Low. d = 0.5 is a convention, not a source-verified value |
| 4 | Training | **The critic reusing a pattern ID is fragile** (the repo's own open issue). Near-duplicate patterns split the counts, so recurrences look new and `recurring` never fires | Mem0 top-s similar plus an LLM decision [4]. ACE embedding dedup [9] | Before saving a pattern, code fetches the **top 5 existing patterns by embedding similarity** (or past.dev recall) and the critic must return `existing_id` or `new`. If cosine > 0.85 to an existing pattern and the critic says `new`, log a warning | 2 | Medium. The threshold is untuned, and a wrong merge hides a distinct failure. Log every merge decision |
| 5 | Training | **Repeated failed edits.** The proposer is told not to repeat a reverted edit, but a reworded version passes the check | Scored candidate store [10][11]. Turning that into a code-level near-duplicate check is my inference | Code compares the candidate rule text with every reverted edit **for the same pattern** by embedding similarity. Above a threshold, reject unless the summary states what differs (the existing escape hatch). Also cap attempts per pattern (e.g. 3 reverts → mark `parked`, skip in the to-do list for N rounds) | 2 | Low |
| 6 | Training / policy | **Policy bloat and choosing what to remove at the cap.** The proposer has no evidence for which rule to drop | ACE helpful/harmful counters [9]. ExpeL importance counts with removal at 0 [7] | In `policy-vN.meta.json`, per learned rule keep `helpful`, `harmful` (did the judge or critic tie it to a good or bad conversation) and `last_linked_round`. When `at_length_limit`, give the proposer the 3 rules with the lowest (helpful − harmful) as removal candidates. **Removal still goes through the gate**, unlike ExpeL's automatic deletion | 3-4 | Medium. Attributing a conversation to a rule needs the judge or critic to name rule IDs, and the counts will be sparse in a short run |
| 7 | Training | The `earlier_edits` prompt block grows linearly with rounds | Retrieval top-k [1][4] | Pass every edit *for this pattern*, the last 10 edits overall, and one line of counts for the rest (e.g. "23 older edits: 9 kept, 14 reverted") | 0.5 | Low |
| 8 | User | Consolidation, once a person has many notes | Sleep-time rewriting [3]. Generative Agents reflections with pointers [1]. Fuzzy-trace false-gist warning [17]. ACE collapse [9] | **Defer.** If added: only when notes exceed the budget, merge at most 2 notes per run, the output must cite both source IDs, and the person's own `asked_to_remember` notes are never merged | 3 | **High.** Gist merging can create false facts [17]. Repeated rewriting erodes detail [9] |
| 9 | User | Temporal questions ("last time you said...") once notes are replaced | Mem0g marks relations invalid instead of deleting them [4]. LongMemEval names knowledge updates and temporal reasoning as hard [13] | **Do not adopt invalidation for user notes.** Forgetting must really delete. Keep the date prefix, which we already have, and set `last_confirmed` on replace | 0 | Keeping history would conflict with consent and the right to be forgotten |

### C.3 On learning the structure from data (question 4)

Not researched in depth here. **Inference only:** ESConv's `problem_type` and `emotion_type` labels could seed the `situation` kind, and its per-turn strategy labels could seed the critic's pattern vocabulary. That would mean clustering critic outputs against strategy labels rather than inventing names. Sibling reports should be preferred on this question.

### C.4 Build order for tomorrow

7 → 1 → 2 → 5 → 4 → 3 → 6. That is roughly 9-12 hours. Items 1, 2, 5 and 7 together take about 6 hours and close the main explosion risks. Item 8 should wait.

---

## Sources opened in this session

1. Park et al., Generative Agents. https://arxiv.org/abs/2304.03442 and https://arxiv.org/html/2304.03442
2. Packer et al., MemGPT. https://arxiv.org/abs/2310.08560
3. Letta, sleep-time compute blog. https://www.letta.com/blog/sleep-time-compute
4. Chhikara et al., Mem0. https://arxiv.org/abs/2504.19413 and https://arxiv.org/html/2504.19413
5. Rasmussen et al., Zep. https://arxiv.org/abs/2501.13956
6. Shinn et al., Reflexion. https://arxiv.org/abs/2303.11366
7. Zhao et al., ExpeL. https://arxiv.org/abs/2308.10144 and https://arxiv.org/html/2308.10144v3
8. Suzgun et al., Dynamic Cheatsheet. https://arxiv.org/abs/2504.07952
9. Zhang et al., ACE. https://arxiv.org/abs/2510.04618 and https://arxiv.org/html/2510.04618
10. Agrawal et al., GEPA. https://arxiv.org/abs/2507.19457
11. Google DeepMind, AlphaEvolve blog. https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
12. Sumers et al., CoALA. https://arxiv.org/abs/2309.02427 and https://arxiv.org/html/2309.02427
13. Wu et al., LongMemEval. https://arxiv.org/abs/2410.10813
14. Murre & Dros 2015, Replication and Analysis of Ebbinghaus' Forgetting Curve. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0120644
15. ACT-R equations as stated in arXiv 2505.05083. https://arxiv.org/html/2505.05083
16. ACT-R site, about page. https://act-r.psy.cmu.edu/about/
17. Reyna, Corbin, Weldon & Brainerd 2016, fuzzy-trace theory. https://pmc.ncbi.nlm.nih.gov/articles/PMC4815269

## Open issues and weak evidence

- **Not opened, so not covered:** A-MEM, MemoryBank (reportedly Ebbinghaus-based decay `[not verified]`), Voyager, Agent Workflow Memory, HippoRAG, LoCoMo's own paper, The AI Scientist, Agent Laboratory, FunSearch, OPRO, TextGrad, and 2024-2026 memory surveys. Several could change the ranking, Voyager's skill library and MemoryBank's decay in particular.
- **Abstract-only evidence:** For Zep, Reflexion, GEPA, Dynamic Cheatsheet and MemGPT I saw only abstracts. Mechanism details (Zep's bi-temporal edges, Reflexion's buffer bound, GEPA's frontier selection) are therefore not reported.
- **ACT-R:** The equation comes from a secondary paper [15], not the ACT-R reference manual. The d = 0.5 default is unverified.
- **CLS:** I saw the Psychological Review citation and abstract points only through a search-result page. The author's PDF failed on a certificate error.
- **Reconsolidation, schema theory, interference, spacing, Soar:** No primary source opened, because PubMed, PMC and Nature blocked fetches. The reconsolidation mapping is inference.
- **The Mem0 comparison** is on LOCOMO with long synthetic dialogues. That full context wins at our scale (a few notes) is inference, though the direction is consistent.
- **The thresholds in C.2** (600 tokens, 30 notes, cosine 0.85, 3 reverts) are my guesses, not from any source. Tune them on the first real runs.
- **Rule-level helpful/harmful attribution (rec. 6)** assumes the judge or critic can name which rule a conversation shows. Our judge is deliberately blind to the policy, so the critic would have to do it. Check whether that leaks into judge blindness before building it.
