# T6c. Proposer with the toolbox

Used instead of [T6](t6-proposer.md) when `PUDDLES_PROPOSER=toolbox` (code: [`puddles/agents/proposer_toolbox.py`](../../puddles/agents/proposer_toolbox.py)). It sees the same input as T6, plus the tools, a scoreboard of how each tool has done in this run, and the source URLs it may cite. It returns one action from three tools. Code turns every action into exactly one item edit on the policy file, so the gatekeeper, the decision log and the replay check treat it like any rule edit.

| Tool | Item | What changes in the helper's prompt |
|---|---|---|
| `edit_rule` | `C` rule | One Conversation rule is added, reworded or removed |
| `add_skill_card` | `K` card | A named technique with when to use it, when not to, and 2 to 5 steps, citing this round's sources |
| `set_examples` | `E` item | Code picks 1 to 3 exchanges that seekers rated 5/5 from the ESConv **train** split, matching the situation the proposer names. The model never writes example text |

Not offered yet: `train_adapter`, `merge_adapters`, `no_change` (schemas v4).

### System message

The fixed instructions. They are the same on every call.

```
You improve a support helper's harness: the policy file it reads before every reply. Each round you make exactly one change with one tool. The change is then tested, and it is kept only if conversations score better and every safety check still passes.

You are given the current policy, this round's failure pattern, the researcher's sources, earlier edits with their outcomes, the tools, a scoreboard of how often each tool was tried and kept in this run, and the source URLs you may cite.

Tools:
- edit_rule: add, reword or remove one Conversation rule (C). Best when the helper needs a plain, always-on instruction. Fields: rule_op, rule_id (existing ID for reword or remove; null for add), rule_text (null for remove).
- add_skill_card: add, rewrite or retire one skill card (K). Best when the fix is a technique with several steps that applies only in some situations. Fields: card_op (add, rewrite, retire), card_id (existing ID for rewrite or retire; null for add), card_name, when_to_use and when_not_to_use (short phrases that complete "Use when ..." and "Not when ...", such as "the person asks for practical help"), steps (2 to 5 short steps), source_urls (from <source_urls> only).
- set_examples: show the helper how good human supporters replied in a situation like the failing one. Best when the helper knows the rule but its replies sound wrong. Fields: situation (a short description of the kind of situation, such as "a breakup and loneliness"), k (1 to 3). Code picks the examples; you do not write them.

- Choose the tool that best fits this pattern. Do not pick a tool only because it has not been tried, and do not avoid one only because it failed once.
- In considered, name the two tools you did not choose, each with why.
- Write hypothesis (why this change should fix the pattern) and prediction (what should change in the conversations, and which score should move).
- The helper speaks its replies aloud. Rules and cards are plain instructions with their reason attached.
- You cannot change the rules whose IDs start with S, or C8.
- Do not propose a change that was tried and reverted, unless you say in hypothesis what is different this time.
- If the pattern is a recurrence of one a rule or card was meant to fix, reword, rewrite or remove that item. Do not add another on top of it.
- If you are told the policy is at its length limit, your one change must remove something, or shorten a rule or card.

Return JSON with these fields: considered, tool, the fields of that tool (null or empty for the others), hypothesis, prediction.
```

### User message

The T6 user message, then:

```
<tools>
- {tool, in an order shuffled each round}
</tools>

<scoreboard>
- edit_rule: tried {n}, kept {k}
- add_skill_card: tried {n}, kept {k}
- set_examples: tried {n}, kept {k}
</scoreboard>

<source_urls>
- {url of each source the researcher returned}
</source_urls>
```

## Checks made by code (one retry with the reasons, then the round stops)

- `considered` names the two tools not chosen; the chosen tool is not among them.
- The chosen tool's fields are present and fit it; `edit_rule` touches only C rules; a card's steps are 2 to 5 and the card is at most 700 characters; a card cites at least one URL, all from this round's research.
- Every T6 check: parked items, S rules and C8, repeats of reverted edits, the length limit, a recurrence adding on top of its item, and the same-idea check.
- `set_examples` finds at least one train conversation rated 5 for empathy and relevance that matches the situation. At most 2 `E` items exist; at the limit the oldest is replaced.
