# The agents

One file per agent. Every file has the same six parts:

1. **The prompt**: the fixed system message, and the message built from the input.
2. **Where the input goes in the prompt**: each slot, what fills it, and where that comes from.
3. **Input**: its format and an example.
4. **Output**: what the model returns, what code does with it, and the checks it must pass.
5. **Memory**: what it reads and what it writes, with the entry as it is stored.
6. **When it goes wrong.**

For anything about the memory as a whole (the levels, when each is updated, the formats, what is still undecided), go to [memory.md](../memory.md). Part 5 of each agent file is that agent's share of it.

## Training

| Step | Agent | What it does | Has a prompt |
|---|---|---|---|
| 1 | [Simulated user](t1-simulated-user.md) | Plays a person from a scenario | Yes |
| 2 | [Helper](t2-i3-helper.md) | Replies, following the policy file | Yes |
| 3 | [Judge](t3-judge.md) | Rates the finished conversation | Yes |
| 4 | [Critic](t4-critic.md) | Names the main failure of the round | Yes |
| 5 | [Researcher](t5-researcher.md) | Finds how counsellors handle it, through Exa | Yes |
| 6 | [Proposer](t6-proposer.md) | Writes one change to the policy (version 2, proposed: picks one action from a toolbox) | Yes |
| 7 | [Gatekeeper](t7-gatekeeper.md) | Tests the change; keeps it or reverts it | No: code. One small yes-or-no prompt for safety checks |

## Inference

| Step | Agent | What it does | Has a prompt |
|---|---|---|---|
| 1 | [Listener](i1-listener.md) | Turns speech into text | No: speech model |
| 2 | [Crisis checker](i2-crisis-checker.md) | Flags risk to life or safety | Yes |
| 3 | [Helper](t2-i3-helper.md) | Drafts the reply | Yes, the same as in training |
| | [Release step](i3b-release.md) | Holds audio until the crisis checker answers | No: code |
| 4 | [Speaker](i4-speaker.md) | Turns the reply into speech | No: speech model |
| 5 | [Note-taker](i5-note-taker.md) | Decides what to remember about the person | Yes |

## Evaluation

| Step | Agent | What it does | Has a prompt |
|---|---|---|---|
| Final | [Held-out jury](e1-held-out-jury.md) | Four jurors rate the final test conversations | Yes, one per juror |

Each file's `### System message` is the prompt the code uses today. Each file also has a **Proposed version 2 (not live yet)** section at the end; all version 2 prompts side by side, with the rules they share: [agent-prompts.md](../agent-prompts.md).

## An output format, or just the memory?

Both, in layers. Each agent has an output format of its own, and code writes the memory from it. No agent writes to memory directly.

```
+------------------------+        +------------------------+        +------------------------+
| 1. MODEL OUTPUT        |        | 2. RECORD              |        | 3. MEMORY ENTRY        |
| small and strict       | -----> | the output plus IDs,   | -----> | one readable sentence  |
| e.g. CriticOutput      |  code  | round, dates, versions |  code  | rendered from the      |
|                        | checks | e.g. FailurePattern    | renders| record by a template   |
| validated, or retried  | it and | saved as a file and    | it     | e.g. MemoryEntry       |
|                        | adds   | named in the log       |        | points back to the file|
+------------------------+        +------------------------+        +------------------------+
```

- **The output format is what makes an agent checkable.** A wrong field, a count that does not add up or an invented ID is caught before anything is stored. A sentence in memory cannot be checked that way.
- **The record is the truth.** It is a file in the repo, and the decision log points to it by path and hash.
- **The memory entry is for finding things.** past.dev searches sentences, so each entry is a sentence written from the record by a fixed template. Each agent file gives its template.
- **Reading works the same way in reverse.** A recall finds which records are relevant; code then loads those records from their files and puts them in the prompt.

Every format is in [`schemas.py`](../../schemas.py), with an example of each in [`examples/`](../../examples).

## How a prompt is put together

- **The system message is fixed.** It says who the agent is, what it is for and what to return.
- **Code builds the rest from the input.** Each input field goes into a named block, such as `<current_policy>`, so the model can tell the parts apart and an instruction can refer to one by name.
- **The helper and the simulated user are in a conversation,** so their turns are sent as chat messages instead.
- **Agents that return JSON use structured outputs,** so the reply always parses. Code still checks the values.
- **No model supplies IDs, dates, versions or who the user is.** Code adds those.

## Not yet tested

- The recall requests shown in each file follow past.dev's API description and have not been run.
- Which `identity` a read of one person's notes should use, and how that person's private audience is set up, is still to be worked out with past.dev.

---

[Design: memory, policy and the crisis path](../agents.md) · [Project README](../../README.md)
