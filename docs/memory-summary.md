# Memory in brief

A one-page summary of how memory works in puddles. Full detail, formats and open questions are in [memory.md](memory.md).

## The idea

The training loop improves one file, the **policy** (the helper's rulebook). Three training agents do the work each round: the **Critic** names one recurring mistake, the **Researcher** finds how real counsellors handle it, and the **Proposer** writes one rule change. The **Gatekeeper** keeps or reverts that change. Memory lets these agents learn from earlier rounds.

Two rules hold throughout:

- **Files are the truth, past.dev is the search index.** Code decides everything from the decision log and record files. A past.dev result is only a candidate record ID, which code checks before use.
- **Memory never changes the helper.** Only a policy edit the Gatekeeper keeps does.

## Two memories

| | Training memory | User memory |
|---|---|---|
| About | What the training agents found, tried and decided | Notes about one person (for the live robot) |
| Read by | Critic, Researcher, Proposer | Helper, through a live-note index |
| Stored in | Run folder files + past.dev audience `training-agents` | A separate past.dev audience per person |
| State | Built | Schemas only (`UserNote`, `LiveNoteIndex`) |

The simulated user, helper and judge never see training memory, so they stay unbiased. The Gatekeeper reads no memory; it only needs test results.

## Training memory: five layers

| Layer | Where | What | Why |
|---|---|---|---|
| Record files | `records/round-NN/<id>.json` | One file per agent output: `pattern-rNN`, `research-rNN`, `edit-rNN`, `gate-rNN` | Full content. Code assigns IDs; each file's SHA-256 is in the log |
| Decision log | `log/decisions.jsonl` (append-only) | One `round` event per round, plus `memory_copy` events | The single source of truth: pattern status, past edits, policy in force, copies to retry |
| Policy files | `policies/policy-vN.md` + `.meta.json` | Locked `S*` safety rules, editable `C*` rules, and where each rule came from | Replay check rebuilds the policy from the log and stops the loop on any mismatch |
| past.dev | audience `training-agents` | One plain-English sentence per record, with its ID and metadata | Lets agents search earlier rounds in plain language |
| Read log | `log/memory_reads.jsonl` | Who recalled which record IDs, and when | Shows who wrote each record and who read it later |

## The four records (schema version 3)

Every record has `id`, `schema_version`, `written_by`, `created_at` and `parent_ids`.

| Record | Written by | Key fields |
|---|---|---|
| `FailurePattern` | Critic | `name`, `description`, `seen_in` / `out_of`, `example_conversation_ids`, `same_as_pattern_id` |
| `ResearchNote` | Researcher | `pattern_id`, `queries`, `sources` (url, title, technique) |
| `PolicyEdit` | Proposer | `pattern_id`, `research_id`, `action` (add / reword / remove), `rule_id`, `rule_text`, base and new policy version |
| `GateOutcome` | Gatekeeper | `edit_id`, `decision` (kept / reverted), scores `before` / `after`, `safety_passed`, `policy_in_force`, `reason` |

**Patterns.** A pattern is one recurring helper mistake, such as `advice-before-understanding`. Each sighting gets its own record; a repeat points to the first sighting with `same_as_pattern_id`, and status (open, fixed, recurring) is tracked under that first ID.

## One round

1. **Start:** resend any past.dev copies still `pending`; three failed attempts stop the loop. Run the replay check.
2. **Converse and judge:** simulated user and helper talk under the policy in force; the judge scores each conversation. No memory access.
3. **Critic → Researcher → Proposer:** each gets the previous agent's record handed over directly (past.dev writes are slow to become searchable), plus:
   - a history list from the log: items about this pattern, the 10 most recent others, and a count of the rest;
   - up to 5 records recalled from past.dev, each verified against the log and its file hash.
4. **Edit:** code applies exactly one change to one `C*` rule; Safety is locked.
5. **Gate:** kept creates `policy-vN+1`; reverted leaves the old policy in force.
6. **End:** write the `round` event first, then copy each record to past.dev and log `confirmed` or `pending`. Rounds that end early (`no_pattern`, `no_evidence`) are logged and copied too.

## Safety against crashes

- A second `round` event for the same round is refused.
- On restart, policy files newer than the logged policy in force are removed, and the round reruns.
- A copy with no `confirmed` event is resent at the start of the next round.

## Known gaps

- The graph runner (`scripts/run_graph_round.py`) does not yet run the replay check or pick the policy from the log; `scripts/run_round.py` does.
- past.dev cannot filter recall by metadata, so a search can return the wrong record type.
- User memory store, expiry and deletion are not built.

See [memory.md](memory.md) sections 9 and 10 for the full list.
