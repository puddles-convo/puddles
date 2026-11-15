# Same-edit check

**Training, step 6, inside the proposer**

- Asks whether the proposer's edit is the same idea as an edit that was already tried and rejected, in other words or under another rule number.
- Runs only with `PUDDLES_PARK_REVERTED=1`, after the proposer's coded checks pass, and only when there is a rejected edit to compare with ([spec 5.7](../specs/unattended-run.md)).

|  |  |
|---|---|
| Called by | [Proposer](t6-proposer.md), on its own model; its tokens count as the proposer's |
| Model | The same model as the proposer |
| Reads from memory | Nothing; the rejected edits come from the decision log |
| Writes to memory | Nothing |

## 1. The prompt

### System message

The fixed instructions. They are the same on every call.

```
You compare one proposed change to a support helper's policy with changes that were already tried and rejected. Say whether the proposed change would change the helper's behaviour in the same way as one of the rejected ones.

- Two changes are the same when they push the helper's behaviour in the same direction on the same situation, even if the wording or the rule number differs. For example, a new rule that restates a rejected rewording of another rule is the same change.
- They are not the same when the proposed change addresses a different situation, or pushes the behaviour in a different direction.
- If you are unsure, answer null.

Return JSON with these fields: same_as_edit_id (the ID of the rejected change it repeats, or null), reason (one sentence).
```

### User message

```
Proposed edit:
{action} {rule_id or "a new rule"}: {rule_text}

Rejected edits:
- {edit_id}: {action} {rule_id}: {rule_text} (round {round}, reverted, quality {before} -> {after})
...
```

## 2. What code does with the answer

- A `same_as_edit_id` that is in the list fails the proposer's check, naming that edit and its result and asking for a different idea. The proposer's usual single retry applies.
- Null, an ID not in the list, an invalid answer or a failed call lets the edit through. The gatekeeper remains the guard.

---

[All agents](README.md) · [Proposer](t6-proposer.md)
