# Proposer

**Training, step 6**

- Turns the pattern and the technique into exactly one change to the policy.
- This is the bridge from memory to policy. It is given every earlier edit and its outcome, so it does not repeat one that was reverted.

|  |  |
|---|---|
| Receives from | [Researcher](t5-researcher.md): sources and techniques |
| Hands to | [Gatekeeper](t7-gatekeeper.md): a candidate policy file |
| Model | The same OpenAI model as the judge. |
| Reads from memory | Earlier edits and their outcomes |
| Writes to memory | The proposed edit |
| Task | [#17](https://github.com/puddles-convo/puddles/issues/17) |

## 1. The prompt

### System message

The fixed instructions. They are the same on every call.

```
You maintain the policy that tells a support helper how to respond. Each round you make exactly one change. The change is then tested, and it is kept only if conversations score better and every safety check still passes.

You are given the current policy, this round's failure pattern, the researcher's sources, and every earlier edit with its outcome.

- Make one change: add a rule, reword a rule or remove a rule. A single change is what makes the test result mean something.
- The helper follows the policy literally and speaks its replies aloud. Write a rule as a plain instruction with its reason attached ("..., because ..."), since a rule that carries its reason is applied more sensibly than a bare command.
- You cannot change the rules whose IDs start with S.
- Do not propose an edit that was tried and reverted, unless you say in your summary what is different this time.
- If the pattern is a recurrence of one that a rule was meant to fix, reword or remove that rule. Do not add another on top of it.
- If you are told the policy is at its length limit, your one change must be a removal, or a rewording that shortens a rule.

Return JSON with these fields: action (add, reword or remove), rule_id (the existing ID for reword or remove; null for add), rule_text (null for remove), summary.
```

### User message

```
<current_policy version="{base_policy_version}" at_length_limit="{at_length_limit}">
{policy_text}
</current_policy>

<failure_pattern id="{pattern.id}" status="{pattern.status}">
{pattern.name}: {pattern.description}
</failure_pattern>

<sources>
- {technique} ({url})
...
</sources>

<earlier_edits>
- {id}: {action} {rule_id}, {outcome}. {summary}
...
</earlier_edits>
```

## 2. Where the input goes in the prompt

| Place in the prompt | Filled with | Comes from |
|---|---|---|
| `<current_policy>` | `policy_text`, with its version | The policy file in force |
| `at_length_limit` | true or false | Worked out by code from the token count of the policy |
| `<failure_pattern>` | `pattern`, with its status | The critic's output; the status comes from replaying the decision log |
| `<sources>` | `sources`, one line each | The researcher's output |
| `<earlier_edits>` | `earlier_edits`, one line each | Built by code from the decision log, so the list is complete; memory adds the detail |

## 3. Input

Format: `ProposerInput` in [`schemas.py`](../../schemas.py). Example: [`examples/proposer_input.json`](../../examples/proposer_input.json).

```json
{
  "policy_text": "(the full text of policy-v3: policy-v4 above without rule C2)",
  "at_length_limit": false,
  "pattern": {
    "id": "pattern-r03",
    "name": "advice-before-understanding",
    "status": "open",
    "description": "Gives a suggestion in its first reply without asking about the person's circumstances."
  },
  "sources": [
    {
      "url": "(url 1 from Exa)",
      "technique": "Ask one open question and reflect the answer before any suggestion."
    }
  ],
  "earlier_edits": [
    {
      "id": "edit-r01",
      "action": "reword",
      "rule_id": "C4",
      "outcome": "kept",
      "summary": "Limit replies to one or two sentences."
    },
    {
      "id": "edit-r02",
      "action": "add",
      "rule_id": "C5",
      "outcome": "kept",
      "summary": "Offer one small suggestion and ask whether it fits."
    }
  ]
}
```

## 4. Output

JSON, requested as a structured output. It is one edit, not a whole policy file.

Format: `ProposerOutput` in [`schemas.py`](../../schemas.py).

| Field | Type | Meaning |
|---|---|---|
| `action` | `add`, `reword` or `remove` | The one change |
| `rule_id` | ID or null | The existing rule for a reword or a removal; null for an addition |
| `rule_text` | text or null | The new wording, with its reason attached; null for a removal |
| `summary` | text | One sentence saying what changes |

Example:

```json
{
  "action": "add",
  "rule_id": null,
  "rule_text": "In your first two replies, ask one open question about their situation and do not give suggestions yet, because advice that comes before understanding rarely fits.",
  "summary": "Add a rule to ask one open question and hold suggestions for the first two replies."
}
```

### What code does with it

Code gives a new rule the next unused ID, applies the one edit to the current policy to write `policies/policy-v4-candidate.md`, and saves a `PolicyEdit`. Because code applies the edit, the safety rules and every other rule are guaranteed untouched.

```json
{
  "round": 3,
  "pattern_id": "pattern-r03",
  "research_id": "research-r03",
  "base_policy_version": "policy-v3",
  "new_policy_version": "policy-v4-candidate",
  "policy_path": "policies/policy-v4-candidate.md",
  "action": "add",
  "rule_id": "C2",
  "rule_text": "In your first two replies, ask one open question about their situation and do not give suggestions yet, because advice that comes before understanding rarely fits.",
  "summary": "Add a rule to ask one open question and hold suggestions for the first two replies."
}
```

### Checks before it is accepted

- `rule_id` names an existing rule for a reword or removal, and is null for an addition.
- The rule is not one whose ID starts with `S`.
- The edit is not the same as one that was reverted, unless the summary says what is different.
- At the length limit, the action is a removal or a rewording that makes the rule shorter.

## 5. Memory

### Reads

The detail behind earlier edits for this pattern:

```json
{
  "query": "Edits tried for advice-before-understanding and what happened to them.",
  "identity": "proposer",
  "level": "low"
}
```

The complete list of earlier edits comes from the decision log. The recall finds related records, which code then loads from their files.

### Writes

One entry per proposed edit, whether or not it is later kept.

Template for `content`:

```
Edit {id} for {pattern_id}: {action} rule {rule_id}. {summary}
```

```json
{
  "id": "edit-r03",
  "content": "Edit edit-r03 for pattern-r03: add rule C2. Add a rule to ask one open question and hold suggestions for the first two replies.",
  "label": "proposer: policy edit, round 3",
  "timestamp": "2026-11-14T22:12:40Z",
  "identity": "proposer",
  "audience": "training-agents",
  "metadata": {
    "type": "policy_edit",
    "round": 3,
    "record_path": "records/round-03/edit-r03.json",
    "sha256": "(hash of the record file)"
  }
}
```

## 6. When it goes wrong

- An edit that fails a check is retried once, with the reason given back to the model.
- A second failure ends the round with no edit, and the pattern stays on the to-do list.

---

[All agents](README.md) · [Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
