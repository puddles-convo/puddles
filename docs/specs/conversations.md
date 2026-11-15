# Build spec: conversations and the decision log

Covers issues #9 (policy file and helper), #10 (simulated user), #11 (conversation runner) and #13 (decision log). It does not cover the judge, critic, researcher, proposer, gatekeeper, Modal or the robot.

The design is in the [README](../../README.md), [docs/agents.md](../agents.md) and the agent files [t1-simulated-user.md](../agents/t1-simulated-user.md) and [t2-i3-helper.md](../agents/t2-i3-helper.md). Record formats are in `schemas.py`, which this work imports and does not change.

## Ground rules

- Python 3.11 or later, with type hints. The only runtime dependency is `pydantic>=2`. HTTP uses the standard library.
- No network, and no API keys, in any test.
- No secrets in the repo. Endpoints and keys come from environment variables.
- No dataset rows in the repo. The scenario fixture is hand-written.
- Tests run from the repo root with:

```
uv run --no-project --python 3.12 --with pytest --with 'pydantic>=2' python -m pytest tests -q
```

## Layout

```
pyproject.toml
loop/__init__.py
loop/errors.py
loop/llm.py
loop/policy.py
loop/prompts.py
loop/prompts/helper_system.txt
loop/prompts/simulated_user_system.txt
loop/helper.py
loop/simulated_user.py
loop/runner.py
loop/run_batch.py
loop/decision_log.py
policies/policy-v1.md
policies/baseline-plain.md
fixtures/scenarios.jsonl
tests/unit/
tests/acceptance/
```

## 1. `loop/errors.py`

Exception classes, each a subclass of `Exception`: `LLMError`, `ConfigError`, `PolicyError`, `HelperError`, `SimulatedUserError`, and `ConversationFailed`.

`ConversationFailed(scenario_id: str, reason: str)` keeps both as attributes.

## 2. `loop/llm.py`

```python
class ChatClient(Protocol):
    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str: ...
```

- `messages` is a list of `{"role": "user" | "assistant", "content": str}`. The system message is passed separately.
- It returns the reply text. A transport or server failure raises `LLMError`.
- Any object with this method is a valid client. Tests supply their own.

`OpenAICompatClient(base_url, model, api_key=None, timeout=60.0, extra_body=None)`

- Sends `POST {base_url}/chat/completions`, where `base_url` already ends in `/v1`.
- The body has `model`, `messages` (the system message first, then the others), `max_tokens`, `temperature`, and `seed` only when it is not `None`. The keys of `extra_body` are merged in.
- It sends `Authorization: Bearer <api_key>` only when a key is given.
- It returns `choices[0].message.content`. Any HTTP error, timeout, dropped or cut-short connection, or unexpected body raises `LLMError`. No other exception type leaves `complete` for a failure of the request.

`FakeClient(replies)`

- `replies` is a list of strings, returned in order, or a callable taking `(system, messages)` and returning a string.
- It records every call in `.calls`, a list of dicts with `system`, `messages`, `max_tokens`, `temperature` and `seed`.
- When a list runs out, it raises `LLMError`.

`client_from_env(prefix) -> OpenAICompatClient`

- Reads `{prefix}_URL`, `{prefix}_MODEL` and the optional `{prefix}_API_KEY`. The prefixes in use are `HELPER` and `SIMUSER`.
- A missing URL or model raises `ConfigError` naming the variable.

## 3. `loop/policy.py`

A policy file looks like this:

```markdown
# policy-v1

## Safety (protected: no edit may change this section)
S1. First safety rule.
S2. Second safety rule.

## Conversation
C1. First conversation rule.
C3. Another rule. IDs are permanent, so they need not be consecutive.
```

```python
@dataclass(frozen=True)
class Policy:
    version: str            # the title line without "# ", stripped of surrounding whitespace
    text: str               # the whole file, exactly as read
    safety_text: str        # the Safety section
    rules: dict[str, str]   # rule ID -> rule text, in file order

    @property
    def sha256(self) -> str: ...          # of text
    @property
    def safety_sha256(self) -> str: ...   # of safety_text

def load_policy(path) -> Policy
def sha256_text(text: str) -> str         # hex digest of the UTF-8 bytes
```

- The Safety heading is a line that is exactly `## Safety`, or `## Safety` followed by a space and any text. `## SafetyNet` is not a Safety heading.
- `safety_text` runs from the Safety heading line to the last non-blank line before the next `## ` heading, without that line's line ending. It is an exact substring of `text`: line endings and spaces inside it are kept as they are in the file, so the hash changes if any byte of the section changes.
- A rule line is an ID (`S` or `C` followed by digits), a full stop, a space, then the text. The text is kept without the ID.
- A rule is one line. Lines are split the way Python's `str.splitlines` splits them, so rule text that contains any line separator, including U+2028, is cut there and the remainder is rejected as a line that is not a rule. This fails closed when a model writes a rule with a line break in it.
- The file is read as UTF-8.
- `load_policy` raises `PolicyError` when the first line is not a `# ` title, there is no `## Safety` section, the Safety section has no rules, a rule ID appears twice, or a non-blank line inside a section is not a rule.
- It also raises `PolicyError` when a rule sits in the wrong section: only `S` rules may appear in the Safety section, and an `S` rule may appear nowhere else. Otherwise a learned rule could hide inside the protected section, or a safety rule could sit outside the part the hash protects.
- A policy may have a Conversation section with no rules, or no Conversation section.

Two policy files:

- `policies/policy-v1.md`: the hand-written starting policy. Rules S1, S2, S3, C1, C3, C6, C7 and C8 are copied word for word from the example policy in `docs/agents.md` section 2. C4 is `Keep replies short.` There is no C2 and no C5.
- `policies/baseline-plain.md`: titled `# baseline-plain`, with the same Safety section, byte for byte, and a Conversation section with no rules.

## 4. `loop/prompts.py`

The two `.txt` files hold the system messages. Each is a word-for-word copy of the first fenced block under `### System message` in the matching agent file. A unit test compares them, so the docs and the code cannot drift.

```python
def render_helper_system(policy: Policy, user_notes: Sequence[str] = ()) -> str
def helper_messages(turns: Sequence[Turn]) -> list[dict[str, str]]
def render_simulated_user_system(scenario: Scenario) -> str
def simulated_user_messages(turns: Sequence[Turn]) -> list[dict[str, str]]
```

- `render_helper_system` replaces `{policy_version}` with `policy.version` and `{policy_text}` with `policy.text` stripped of surrounding whitespace. The line `{user_notes, or "Nothing yet. This is your first conversation."}` is replaced by the notes, one per line, each starting `- `; with no notes it becomes `Nothing yet. This is your first conversation.`
- `helper_messages` maps a `user` turn to role `user` and a `helper` turn to role `assistant`, in order.
- `render_simulated_user_system` replaces `{situation}`, `{emotion_type}`, `{initial_intensity}`, `{problem_type}` and `{openness}` from the scenario. Nothing else in the template changes.
- `simulated_user_messages` always starts with `{"role": "user", "content": "(The helper is listening. Begin.)"}`. Then the roles are swapped: a `helper` turn becomes role `user`, and a `user` turn becomes role `assistant`.
- No rendered prompt may contain an unfilled `{placeholder}` from the template.

## 5. `loop/helper.py`

```python
def helper_reply(client: ChatClient, policy: Policy, turns: Sequence[Turn], *,
                 session_id: str, turn: int, user_notes: Sequence[str] = (),
                 max_tokens: int = 80, seed: int | None = None) -> HelperReply
```

- Calls `client.complete` once with the helper system message and `helper_messages(turns)`.
- The reply is stripped. If it is empty, the call is made once more. A second empty reply raises `HelperError`.
- If the reply does not end in `.`, `!` or `?` (optionally followed by a closing quote or bracket), it is cut back to the end of its last complete sentence. If it has no complete sentence, it is kept as it is.
- Returns `HelperReply(session_id, turn, text, policy_version=policy.version, fixed_handoff_response=False)`.

## 6. `loop/simulated_user.py`

```python
END_MARKER = "[END]"

def next_user_line(client: ChatClient, scenario: Scenario, turns: Sequence[Turn], *,
                   seed: int | None = None, max_tokens: int = 120) -> str | None
```

- Calls `client.complete` with the simulated user system message, `simulated_user_messages(turns)`, `temperature=0.9` and the seed.
- The reply is stripped. Exactly `[END]` returns `None`.
- A reply is invalid when it is empty, has more than 80 words, or contains, in any letter case, `as an ai`, `language model` or `simulat`.
- An invalid reply is asked for once more. A second invalid reply raises `SimulatedUserError`.

## 7. `loop/runner.py`

```python
def scenario_seed(seed: int | None, scenario_id: str) -> int | None
def load_scenarios(path) -> list[Scenario]
def run_conversation(helper_client: ChatClient, user_client: ChatClient, policy: Policy,
                     scenario: Scenario, *, round: int, max_turns: int = 10,
                     seed: int | None = None, now: datetime | None = None) -> ConversationRecord
def run_batch(helper_client: ChatClient, user_client: ChatClient, policy: Policy,
              scenarios: Sequence[Scenario], *, round: int, max_turns: int = 10,
              seed: int | None = None, out_path=None, now: datetime | None = None) -> BatchResult
```

`scenario_seed`

- Returns `None` when `seed` is `None`. Otherwise a non-negative integer below 2**31, derived from `seed` and `scenario_id` with SHA-256, so it is the same in every process and on every machine. Different scenario IDs give different seeds, apart from chance collisions.

`load_scenarios`

- Reads a JSON Lines file, one `Scenario` per non-blank line, validated with `schemas.Scenario`.

`run_conversation`

- The simulated user speaks first. The two then alternate: user line, helper reply, user line, helper reply.
- It stops when the simulated user returns `None`, or straight after the helper's `max_turns`-th reply. It never ends on a user line that the helper did not answer, unless the helper's limit is zero.
- Both clients are given `scenario_seed(seed, scenario.scenario_id)`.
- `helper_reply` is called with `session_id` set to the record ID and `turn` counting the helper's replies from 1.
- It returns a `ConversationRecord` with `id` equal to `conv-r{round:02d}-{scenario_id}`, `written_by="helper"`, `created_at` equal to `now` or the current UTC time, `source="simulated"`, the scenario's `scenario_id` and `split`, `policy_version=policy.version`, and the turns.
- If the simulated user ends before saying anything, or any `LLMError`, `HelperError` or `SimulatedUserError` is raised, it raises `ConversationFailed` with the scenario ID and a reason.

`run_batch`

- Runs every scenario in order. A failed conversation does not stop the batch.
- `BatchResult` has `records` (a list of `ConversationRecord`), `failures` (a list of objects with `scenario_id` and `reason`), and `summary()`, which returns `{"round": ..., "conversations": <number of records>, "failed": <number of failures>}`.
- With `out_path`, it writes one record per line as JSON to that file, creating parent folders. Failures go to a second file next to it, with the suffix `.failures.jsonl` in place of the last suffix, one `{"scenario_id", "reason"}` object per line. That file is written only when there are failures.
- A failed conversation is never written among the records.
- The records file is overwritten on each run. When a run has no failures, a failures file left at that path by an earlier run is deleted, so a stale file cannot be read as belonging to this run.

## 8. `loop/run_batch.py`

```
python -m loop.run_batch --policy policies/policy-v1.md --scenarios fixtures/scenarios.jsonl \
    --round 1 --out runs/round-01/conversations.jsonl [--max-turns 10] [--seed 7] [--fake]
```

- Without `--fake`, the clients come from `client_from_env("HELPER")` and `client_from_env("SIMUSER")`.
- With `--fake`, it uses built-in scripted clients, so the command runs with no network and no keys.
- It prints the batch summary as one line of JSON on standard output.
- The exit code is 0 when at least one conversation succeeded, 1 when none did, and 2 for a configuration or file error, with a one-line message on standard error. A file error includes a missing file, a file that is not valid UTF-8, and a file whose contents do not validate. No traceback is printed for any of these.

`fixtures/scenarios.jsonl` has five hand-written scenarios with IDs `fixture-0001` to `fixture-0005` and split `validation`. They are not copied from any dataset. At least one is `guarded` and one `resistant`.

## 9. `loop/decision_log.py`

The log is a JSON Lines file of `RoundEvent` and `MemoryCopyEvent` objects from `schemas.py`. It is append-only. Examples: `examples/log_round_event.json` and `examples/log_memory_copy_event.json`.

```python
class DecisionLogError(Exception): ...

def append_event(path, event: RoundEvent | MemoryCopyEvent) -> None
def read_events(path) -> list[RoundEvent | MemoryCopyEvent]
def copy_status(events) -> dict[str, str]
def pending_copies(events) -> list[str]
def render_table(events) -> str
```

- `append_event` adds exactly one line and never rewrites or truncates the file. It creates the file and parent folders when missing. A second `round` event with a round number already in the file raises `DecisionLogError` and writes nothing.
- `read_events` returns the events in file order, each as the right model, chosen by its `event` field. A missing file gives an empty list. Blank lines are skipped. A line that is not valid JSON, or does not validate, raises `DecisionLogError` naming the line number.
- `copy_status` maps each `record_id` to its status after replaying the `memory_copy` events in order. Once a record is `confirmed` it stays `confirmed`, whatever comes later.
- `pending_copies` lists the record IDs whose status is `pending`, in the order they first appeared.
- `render_table` returns a Markdown table with one row per `round` event, in round order, and the columns Round, Outcome, Pattern, Edit, Quality, Safety, Policy in force, and Reason. Quality reads like `3.25 -> 3.75`. A part the round does not have shows as `-`. With no round events it returns the header and separator rows only.
- `python -m loop.decision_log <path>` prints that table.
