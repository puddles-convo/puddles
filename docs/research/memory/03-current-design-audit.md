**W3 — Read-only audit of the current memory design**

The policy has a designed size limit; the surrounding memory does not. Several guarantees—canonical record lookup, policy replay, note replacement, forgetting, and copy confirmation—remain specifications rather than executable behavior. The implemented batch runner also permits evidence to be overwritten. (`README.md:298–305`; `docs/agents.md:309–349`; `loop/runner.py:112–123`)

This is a static audit of the requested files and tests. Tests were inspected, not executed; no files were modified.

**1. Inventory**

“Writer” below means the logical producer. The design assigns persistence to orchestrator code, not direct agent writes. Stored `Record` subclasses share `id`, `schema_version`, `written_by`, `created_at`, and `parent_ids`. (`README.md:212`; `schemas.py:27–34`)

| Stored thing | Producer, timing, fields, readers, lifecycle |
|---|---|
| Decision-log `round` | Orchestrator/gatekeeper at round completion, including early exits. Holds round/outcome, record references, pattern/count/status, edit, policy versions/hashes, aggregate scores/noise, safety summary, reason. Code reads it; append-only. (`schemas.py:284–340`; `docs/agents.md:264–277`) |
| Decision-log `memory_copy` | Copy-confirmation code after ingestion waits. Fields: round, record ID, confirmed/pending, attempt. Read by retry/status code; append-only. (`schemas.py:343–350`; `docs/agents.md:331`) |
| Canonical training files | Critic: round, name, description, counts, example conversations, recurrence reference. Researcher: round, pattern, queries, sources with URL/title/technique. Proposer: pattern/research references, versions/path, summary, action/rule/text. Gatekeeper: edit, decision, before/after scores, safety, policy, reason. Written when each step succeeds; downstream agents and recall hydration read them. Historical files have no correction/deletion protocol. (`schemas.py:85–148`; `docs/agents.md:275–277,329`) |
| Training-memory entries | Copies of those four records, under agent identities. Fields: ID, content sentence, label, timestamp, identity, audience, metadata containing type/round/path/hash. Critic reads patterns/outcomes; researcher reads research; proposer reads edits/outcomes. Add-only, including rejected edits. (`README.md:164–208,219–224`; `docs/agents/t6-proposer.md:173–177`) |
| Policy rules and versions | Code applies one proposed add/reword/remove; gate retains or restores the policy. Helper reads the current version; critic/proposer inspect it. Permanent rule IDs, protected safety rules; historical versions remain in git. (`README.md:270–287`; `docs/agents.md:109,258–262`) |
| Policy metadata | Adjacent file: version, parent, policy/safety hashes, base-rule IDs, learned-rule map containing edit/action/pattern/round/text hash. Integrity code reads it; excluded from prompts. No retention policy. (`schemas.py:355–371`; `docs/agents.md:130,316`) |
| Scenarios | Dataset-loading code supplies ID, split, emotion/problem/situation, intensity, openness, optional originating session. Simulator reads them; conversation records reference them. Current loader only reads JSONL; no update lifecycle. (`schemas.py:44–54`; `loop/runner.py:28–31`) |
| Conversations and live sessions | Training runner or live speaker/orchestrator records source, scenario/split, policy, turns; live fields include both consents, helped, timings. Judge/critic and note-taker consume these; consented sessions enter training. Current batches overwrite their output path. (`schemas.py:57–69`; `docs/agents/i4-speaker.md:39–76`; `loop/runner.py:112–123`) |
| Scores and safety results | Judge produces conversation/judge IDs, empathy/relevance, emotion before/after, strategies, note. Gate stores aggregate comparisons and case pass/reason results. Scores feed critic/gate/log, not past.dev; no revision protocol. Safety cases are versioned inputs. (`schemas.py:72–82,315–325,374–398`; `docs/agents/t3-judge.md:157`) |
| User notes and live index | Post-session note-taker proposes add/replace/delete. `UserNote` adds user/session IDs and fact; index stores user ID/current note IDs. Helper and note-taker read live notes. Replacement creates a new ID, removes old visibility, then deletes old remote data with retries. (`schemas.py:230–239,431–451`; `docs/agents/i5-note-taker.md:148–158,202–207`) |
| Failure diagnostics | Runner stores scenario ID/reason separately and returns batch counts. Files are replaced or removed on rerun; partial failed conversations are discarded. (`loop/runner.py:73–123`; `tests/acceptance/test_acc_runner.py:228–272`) |

Audio is explicitly unstored. Crisis classifications and fixed-response flags are promised in session records, but their persistence is structurally missing, discussed below. (`docs/agents/i1-listener.md:65`; `docs/agents/i2-crisis-checker.md:136`; `docs/agents/i3b-release.md:89`)

**2. Growth and prompt costs**

Let \(T\) be transcript tokens, \(P\) policy tokens, \(N(S)\) live-note tokens, \(q\leq3\) queries per research round, and \(D\) recalled detail. Orders assume bounded individual records; most string/list fields lack enforced limits. (`schemas.py:85–110,194–239`; `docs/agents/t5-researcher.md:150–152`)

| Store | Growth and present bound |
|---|---|
| Log | \(R\) round events plus up to \(12R\) copy events under the designed three-attempt limit and four copies/round. Code imposes no attempt ceiling. (`docs/agents.md:331`; `schemas.py:350`) |
| Canonical files / training memory | Up to \(4R\) records/copies each; early exits reduce this. No compaction, expiry, or deletion. (`README.md:166–169,220`; `docs/agents.md:264–277`) |
| Conversations / scores / diagnostics | \(O(RCT)\), \(O(RCT)\) including strategy lists, and \(O(RC)\) failures, if separate run outputs survive. Reusing paths bounds storage by destroying prior evidence. (`schemas.py:64,81`; `loop/runner.py:112–123`) |
| Policy / metadata | Current policy designed \(O(P)\); archived versions \(O(RP)\). Metadata scales with active rules, archives with versions. Example cap is 800 tokens; loader does not enforce it. (`examples/gate_input.json:24`; `schemas.py:363–371`; `loop/policy.py:38–82`) |
| User notes / index | With \(a\) net additions/session: \(O(USa)\) notes and IDs; remote deletion backlog adds retired notes until deletion succeeds. No cap or expiry. (`schemas.py:431–451`; `docs/agents/i5-note-taker.md:148–152,207`) |
| Live sessions | Up to \(O(UST)\) retained transcripts; consent table restricts intended retention, but supplies no expiry schedule. (`docs/agents.md:337–344`) |
| Scenario/safety libraries | Fixed if reused; grow with newly admitted scenarios/cases. Neither has a collection-size limit. (`schemas.py:44–54,380–398`) |

Per-call prompt growth is:

| Agent | Input size |
|---|---|
| Simulator | \(O(T+\text{scenario})\); helper in training \(O(P+T)\); judge \(O(T)\). Full histories are passed. (`loop/prompts.py:37–58`; `schemas.py:244–247`) |
| Critic | \(O(P+R+kT+D)\), with complete earlier patterns and approximately ten worst conversations. “About ten” is not enforced. (`docs/agents/t4-critic.md:69–72`; `schemas.py:194–199`) |
| Researcher | \(O(qR+\text{tool text})\): complete earlier queries; no page-text budget. (`docs/agents/t5-researcher.md:53–54`; `schemas.py:202–204`) |
| Proposer | \(O(P+R+\text{sources}+D)\) by design; every earlier edit is included. (`docs/agents/t6-proposer.md:64–68,173`) |
| Live helper / note-taker | \(O(P+N(S)+T)\) / \(O(N(S)+T)\). Renderer includes every supplied note; no selection budget. (`loop/prompts.py:27–34`; `docs/agents/i5-note-taker.md:45–67`) |
| Crisis checker | Recent turns plus latest message; intended window is “last few,” without a numerical limit. Gate assertion prompt contains one reply/case/question. Listener, release, speaker have no agent prompt. (`docs/agents/i2-crisis-checker.md:60`; `docs/agents/t7-gatekeeper.md:45–66`; `docs/agents/README.md:28–33`) |

Consequently, complete-history training prompts accumulate \(O(R^2)\) history tokens across rounds; user-note inputs accumulate \(O(US^2)\) across sessions if additions continue linearly.

Illustrative sizing—not measured production usage—assumes 40 total conversations/round, 3 KB/transcript, 1 KB/score, four 1 KB canonical records and four 0.6 KB memory copies/round. Assume pattern/query/edit history contributes 20/24/40 tokens per round respectively.

| R | Conversations + scores | Canonical / remote records | Combined history tokens per round |
|---:|---:|---:|---:|
| 10 | 1.6 MB | 40 / 40 | 840 |
| 50 | 8 MB | 200 / 200 | 4,200 |
| 200 | 32 MB | 800 / 800 | 16,800 |

At three net notes/session, 30 prompt tokens and 0.5 KB stored/note:

| S, one user | Live notes | Note prompt | Note storage |
|---:|---:|---:|---:|
| 5 | 15 | 450 tokens | 7.5 KB |
| 50 | 150 | 4,500 tokens | 75 KB |
| 500 | 1,500 | 45,000 tokens | 750 KB |

These estimates follow the specified per-round records and all-notes rendering, not guaranteed upper bounds. (`README.md:164–170`; `loop/prompts.py:29`)

**3. Update semantics and failure modes**

Training correction means another historical entry; no typed supersession/merge relation exists. Hash validation establishes record authenticity, not whether its conclusion remains applicable. Pattern status comes from rule activity: “fixed” means a linked rule exists, rather than demonstrated continued effectiveness. (`README.md:220–221`; `schemas.py:85–148`; `docs/agents.md:317–329`)

Duplicate pattern names survive unless the critic recognizes equivalence and supplies the earlier ID. Research blocks repeated query strings, not paraphrased queries, repeated URLs, or equivalent techniques. Reverted-edit avoidance depends on summaries and permits another attempt whenever the summary explains a difference; semantic duplicate detection is unspecified. (`docs/agents/t4-critic.md:31,215–217`; `docs/agents/t5-researcher.md:150–152`; `docs/agents/t6-proposer.md:31,156`)

User-note correction depends on the LLM finding the old fact among existing notes. Equivalent additions, contradictory surviving notes, and silently stale facts have no deterministic reconciliation or expiry. The prompt permits deletion when a fact becomes false, but supplies no confirmation schedule. Invalid note-taker output changes nothing, including requested deletions. (`docs/agents/i5-note-taker.md:24–39,206`)

Forgetting removes indexed notes and retries remote deletion. It does not specify erasure from retained transcripts, training derivatives, or git history. Also, notes fetched once at session start may remain in the current prompt after index removal unless the session context is explicitly rebuilt. (`docs/agents.md:346–350`; `docs/agents/t2-i3-helper.md:104`; `README.md:371`)

**4. Concept coverage**

Current types cover failures, research, interventions, outcomes, scenarios, conversations, judgments, rules/provenance, safety cases/results, user facts, and visibility IDs. (`schemas.py:44–163,284–398,431–465`)

Missing or insufficient concepts are:

- **Positive patterns:** a judge note can mention success, but durable learning only formalizes failures. (`docs/agents/t3-judge.md:34`; `schemas.py:85–94`)
- **Conditions and subgroup outcomes:** scenarios contain emotion/problem/openness; patterns/rules have no structured applicability fields, and gate summaries contain overall means. (`schemas.py:49–53,85–94,131–148`)
- **Evidence strength:** counts exist for the critic’s selected sample, but not confidence, independent replications, contradictory evidence, or judge disagreement. (`schemas.py:72–94,315–318`)
- **Continuing rule utility:** provenance already links rule → edit → pattern → conversations; what is missing is evaluation-run linkage, exact motivating spans, and later per-rule utility/ablation history. (`schemas.py:93,120–121,355–360`; `docs/agents.md:317–321`)
- **Typed user knowledge:** creation time and source session exist, but category, last-confirmed date, sensitivity, follow-up state/due date, style preference, and intervention/outcome relationships do not. Free text can express these without enabling reliable lifecycle handling. (`schemas.py:27–34,431–451`)

For this deadline, learn candidate patterns from scored examples while retaining existing scenario labels for analysis; do not attempt learned ontology induction before those evidence links and slices exist. The necessary labels and conversation references already exist. (`schemas.py:44–54,75,93`)

**5. Inconsistencies and implementation status**

- Recurrence reuses the earlier **record ID**, conflicting with append-only occurrences and ID-based lookup. Copy-status code keys solely by ID and treats confirmed as permanent, so a later occurrence can inherit earlier confirmation. (`docs/agents/t4-critic.md:186`; `README.md:202,220`; `loop/decision_log.py:67–78`)
- Critic output/log show 2/3; saved pattern/example shows 14/40 and includes a third conversation exhibiting a different failure. (`examples/critic_output.json:5–9`; `examples/log_round_event.json:30–31`; `examples/training_H4_failure_pattern.json:13–18`; `examples/critic_input.json:55`)
- Research example still uses legacy `query`; edit example omits action/rule/text. Schema accepts these omissions, although replay requires the edit fields. (`examples/training_H5_research_note.json:10`; `examples/training_H6_policy_edit.json:9–15`; `schemas.py:108–109,126–128`)
- Researcher/proposer describe recalled detail but their input schemas lack a corresponding field. Filtering semantic recall through an index also does not guarantee fetching **all** live notes. (`docs/agents/t5-researcher.md:158–168`; `docs/agents/t6-proposer.md:173`; `schemas.py:202–220`; `docs/agents/i5-note-taker.md:164–174`)
- Session promises exceed schema: `Turn` stores only speaker/text, losing crisis risk and fixed-response provenance. Speaker saves sessions despite the consent table’s “nothing new” case. (`schemas.py:37–39,57–69`; `docs/agents/i2-crisis-checker.md:136`; `docs/agents/i3b-release.md:89`; `docs/agents/i4-speaker.md:39`; `docs/agents.md:341`)
- Code implements policy loading/hashing, supplied-note rendering, simulated conversations, batch persistence, log append/read/table, and copy-status replay. It lacks the designed memory client, canonical-record persistence/hydration, rule replay/meta checking, learning agents/gate, and live-note lifecycle. The batch command invokes only conversation generation; “replay” tests concern copy status. (`loop/run_batch.py:64–82`; `loop/policy.py:20–82`; `loop/prompts.py:27–34`; `loop/decision_log.py:28–86`; `tests/acceptance/test_acc_decision_log.py:123–142`)

**6. Five ranked weaknesses and smallest fixes**

1. **Trusted update guarantees are unimplemented.** Immediate consequence: no verifiable learning round; deployment consequence: unverified policy provenance. Add one round-finalization function that saves immutable records, hashes references, appends the event, and verifies the resulting policy before promotion. (`docs/agents.md:309–331`; `loop/run_batch.py:75–82`)
2. **Identity collisions destroy evidence.** Recurrences reuse IDs; conversations omit policy/run from IDs, and batch paths overwrite. Introduce unique occurrence/run IDs plus a separate canonical pattern ID; reject existing output paths. (`docs/agents/t4-critic.md:186`; `loop/runner.py:38,115`)
3. **Prompt growth defeats retrieval.** Complete histories and all notes eventually dominate inputs. Keep full archives, impose token budgets on prompt projections, and check exact duplicate queries/edits against the archive in code. (`schemas.py:197–220`; `loop/prompts.py:29`)
4. **User correction and forgetting depend on incomplete lifecycle machinery.** Maintain an authoritative local live-note map; apply changes atomically, queue remote deletions, refresh active context, and add last-confirmed/expiry fields for temporary facts. (`docs/agents/i5-note-taker.md:148–158`; `schemas.py:431–451`)
5. **Aggregate gains conceal conditional harm and decay.** Persist scenario-linked evaluation IDs and subgroup counts/means; retain a small regression set from motivating conversations and rerun it after each kept edit. This is buildable without adding another agent. (`schemas.py:49–53,93,138–148`; `docs/agents/t7-gatekeeper.md:22–32`)