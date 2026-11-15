# Build spec: one round as a graph, with tracing

Covers stage 0 of the build plan ([docs/plan/stage-0-skeleton.md](../plan/stage-0-skeleton.md), issue #30). It does not cover token counts, the real judge, the real gate or the loop over rounds.

This file is the contract. The stage file says what and why; this file fixes the names, signatures and file formats. Where the two disagree, this file is right and the stage file needs fixing.

## Ground rules

- Python 3.11 or later, with type hints. New runtime dependencies: `langgraph>=1.2` and `mlflow>=3.18`. Add both to `pyproject.toml` and `requirements.txt`.
- No network and no API keys in any test.
- Do not change `schemas.py`, anything under `loop/`, or any existing test.
- Tests run from the repo root with:

```
env -u PYTHONPATH MLFLOW_DISABLE_AGENT_HINT=1 uv run --no-project --python 3.12 --with pytest --with 'pydantic>=2.6' --with 'openai>=1.40' --with 'httpx>=0.27' --with 'langgraph>=1.2' --with 'mlflow>=3.18' python -m pytest -q
```

Before this work the command, without the last two `--with`, gives `421 passed, 1 skipped`.

## Layout

```
puddles/round.py              changed: three new functions, run_agents kept
puddles/graph.py              new
puddles/tracing.py            new
scripts/run_graph_round.py    new
tests/test_graph.py           new: the implementer's tests
tests/acceptance/test_acc_graph.py, test_acc_tracing.py, test_acc_graph_script.py   new: written from this file alone
```

## 1. `puddles/round.py`

`run_agents` is split at its two early returns. `run_critic` makes the `RoundResult`. The other two take it, change it in place and return the same object.

```python
def run_critic(round_: int, conversations: list[ScoredConversation], policy_path: Path,
               log: DecisionLog, memory: Memory, llm: LLM, root: Path) -> RoundResult
def run_researcher(result: RoundResult, log: DecisionLog, memory: Memory, llm: LLM,
                   search: Callable[[str], list[dict]], root: Path) -> RoundResult
def run_proposer(result: RoundResult, log: DecisionLog, memory: Memory, llm: LLM,
                 root: Path) -> RoundResult
```

- `run_critic` returns a new `RoundResult` with `outcome="no_pattern"`. If the critic found a pattern, `pattern`, `status_before` and `refs["pattern"]` are set. If not, `reason` is the critic's reason.
- `run_researcher` requires `result.pattern`. It sets `research` and `refs["research"]` and saves the search log. With no sources it sets `outcome="no_evidence"` and the reason `run_agents` uses today.
- `run_proposer` requires `result.pattern` and `result.research`. It sets `edit`, `candidate_path`, `refs["edit"]` and `outcome="candidate"`.
- None of the three catches `puddles.llm.AgentFailed`.
- Calling `run_researcher` without a pattern, or `run_proposer` without a pattern and research, is a caller's mistake. What happens then is not specified.
- `run_agents` keeps its signature and behaviour. It calls the three in order and returns early exactly where it does today. `tests/test_memory_wiring.py` and `tests/test_agents.py` pass unchanged.

## 2. `puddles/tracing.py`

```python
def setup(tracking_uri: str | None = None, experiment: str = "puddles") -> None
class TracedClient:
    def __init__(self, inner: ChatClient, name: str) -> None
    inner: ChatClient
    name: str
    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str
```

- `setup` uses `tracking_uri`, else the environment variable `MLFLOW_TRACKING_URI`, else `sqlite:///runs/mlflow.db`. For a `sqlite:///` address it creates the file's parent folder. It sets the tracking address and the experiment, then calls `mlflow.langchain.autolog()` and `mlflow.openai.autolog()`. Calling it twice is harmless.
- `TracedClient.complete` calls `inner.complete` with the same arguments and returns its text unchanged. It opens one MLflow span per call, named `llm:<name>`, of span type `CHAT_MODEL`. The span's inputs hold `system`, `messages`, `max_tokens`, `temperature` and `seed`; its output is the reply text.
- An exception from `inner.complete` passes through unchanged.

## 3. `puddles/graph.py`

```python
@dataclass
class RoundDeps:
    helper_client: ChatClient
    user_client: ChatClient
    llm: LLM
    search: Callable[[str], list[dict]]
    memory: Memory
    log: DecisionLog
    root: Path
    judge: Callable[[list[ConversationRecord]], list[ScoredConversation]] = stub_judge
    gate: Callable[[RoundResult], GateOutcome] = stand_in_reverted

class RoundState(TypedDict, total=False):
    round: int
    policy_path: Path
    scenarios: list[Scenario]
    max_turns: int
    seed: int | None
    records: list[ConversationRecord]
    failures: list[Failure]            # loop.runner.Failure
    scored: list[ScoredConversation]
    result: RoundResult
    gate: GateOutcome | None
    event: RoundEvent

def stub_judge(records: list[ConversationRecord]) -> list[ScoredConversation]
def stand_in_reverted(result: RoundResult) -> GateOutcome
def check_policy_file(path: Path) -> loop.policy.Policy
def build_round_graph(deps: RoundDeps)            # returns the compiled LangGraph graph
def run_round(deps: RoundDeps, *, policy_path: Path, scenarios: list[Scenario],
              max_turns: int = 10, seed: int | None = None) -> RoundState
```

- `stub_judge` returns one `ScoredConversation` per record, in order: the record's `id` as `conversation_id`, its turns, `empathy=3`, `relevance=3`, and a note that says it is a stub. Stage 2 replaces it through `RoundDeps.judge`.
- `stand_in_reverted(result)` is `puddles.round.stand_in_gate(result, "reverted")`. Stage 2 replaces it through `RoundDeps.gate`.
- `check_policy_file` raises `loop.errors.PolicyError` if the file's bytes contain a carriage return, and otherwise returns `loop.policy.load_policy(path)`, which raises `PolicyError` on a malformed file, including a rule wrapped onto a second line.

### The graph

Seven nodes, with exactly these names: `converse`, `judge`, `critic`, `researcher`, `proposer`, `gate`, `finish`.

```
START -> converse -> judge --(scored)--> critic --(pattern)--> researcher --(sources)--> proposer --(edit)--> gate -> finish -> END
                       |                    |                      |                        |
                       +--(nothing scored)--+--(no pattern)--------+--(no sources)----------+--(no edit)-----------> finish
```

| Node | What it does |
|---|---|
| `converse` | Loads the policy with `check_policy_file(state["policy_path"])`. Wraps the two chat clients in `TracedClient`, named `helper` and `simulated_user`. Calls `loop.runner.run_batch(..., round=state["round"], max_turns=..., seed=..., out_path=root / "conversations" / f"round-{round:02d}-train.jsonl")`. Sets `records` and `failures` from the `BatchResult`. |
| `judge` | Sets `scored = deps.judge(records)`. If `scored` is empty it also sets `result` to a `RoundResult` with `outcome="no_pattern"`, and the round goes to `finish` without calling the critic. The reason is `No conversation finished.` when `records` was empty, and `The judge scored no conversation.` when there were records but no scores. |
| `critic` | `run_critic`. Sets `result`. |
| `researcher` | `run_researcher`. |
| `proposer` | `run_proposer`. |
| `gate` | Sets `gate = deps.gate(result)`. Runs only when `result.edit` is set. A proposer that succeeds always sets it, so the "no edit" path is the path of a proposer that failed twice. |
| `finish` | Sets `event = puddles.round.finish_round(result, state.get("gate"), deps.log, deps.memory, deps.root)`. |

### When an agent fails twice

A node catches `puddles.llm.AgentFailed` and the round still reaches `finish`. The failure is in `event.reason`, which then starts with the agent's name and contains the text of the exception.

| Node | `result` after the failure | `event.outcome` |
|---|---|---|
| `critic` | A new `RoundResult` with `outcome="no_pattern"`, no pattern | `no_pattern` |
| `researcher` | `outcome="no_evidence"`, `research` left as `None`, the pattern kept | `no_evidence` |
| `proposer` | `outcome` left as it was, `edit` left as `None`, the pattern and research kept | `no_pattern`, with `event.pattern` set |

The gate never runs after a failure. No other exception is caught.

### `run_round`

In this order:

1. `check_policy_file(policy_path)`. A `PolicyError` propagates. Nothing has touched the memory or the log by then, so the log is exactly as it was, even when it holds copies that are still pending.
2. `deps.memory.ensure_training_audience()`, then `puddles.round.retry_pending(deps.log, deps.memory, deps.root)`.
3. `round = deps.log.next_round()`.
4. Inside one MLflow run named `round-NN`, invoke the graph with `invoke`.
5. On that MLflow run: parameters `round` and `policy_version`; tag `outcome`, set to `event.outcome`; metrics `conversations` (finished) and `conversations_failed`.
6. Return the final state.

`run_round` does not call `tracing.setup`. The caller does, once.

**The round's outcome is `state["event"].outcome`.** `state["result"].outcome` is only `candidate`, `no_pattern` or `no_evidence`.

### What a round leaves behind

- One `round` event in `<root>/log/decisions.jsonl`, and the memory-copy events that `finish_round` writes.
- `<root>/conversations/round-NN-train.jsonl`, one `ConversationRecord` per line, and the failures file `run_batch` writes beside it when a conversation failed.
- The record files under `<root>/records/`.
- One MLflow trace, with a span named after each node that ran. Every `llm:helper` and `llm:simulated_user` span has the `converse` span as an ancestor.
- One MLflow run.

Traces are written in the background. Call `mlflow.flush_trace_async_logging()` before reading them.

## 4. `scripts/run_graph_round.py`

```
python -m scripts.run_graph_round --fake --root runs/dev
```

`main(argv: Sequence[str] | None = None) -> int`.

| Argument | Default | Meaning |
|---|---|---|
| `--root` | `runs/dev` | Where `log/`, `records/`, `conversations/` and `policies/` go |
| `--policy` | `policies/policy-v1.md` | The starting policy. The round runs on `puddles.round.policy_in_force(root, policy)`. |
| `--scenarios` | `fixtures/scenarios.jsonl` | Read with `loop.runner.load_scenarios` |
| `--max-turns` | `10` | |
| `--seed` | none | |
| `--gate` | `reverted` | `kept` or `reverted`: what the stand-in gate decides |
| `--fake` | off | Scripted chat clients, a scripted model, a fake search and `FakeMemory`. No keys, no network. |
| `--mlflow` | none | Tracking address, passed to `tracing.setup` |

- It calls `tracing.setup`, runs one round with `run_round`, and prints one line: `round N: <outcome>. <reason>`.
- With `--fake` on a root with no log, the round reaches the gate, so the outcome is `kept` or `reverted` as `--gate` says. The chat clients speak the same scripted lines as `python -m loop.run_batch --fake`. A second `--fake` run on the same root also finishes, as round 2; its outcome may be any of the four.
- Without `--fake` it builds `loop.llm.client_from_env("HELPER")`, `client_from_env("SIMUSER")`, `puddles.llm.OpenAILLM()`, `puddles.exa.search` and `puddles.memory.PastClient()`. It checks that `EXA_API_KEY` is set before the round starts, because the search would otherwise fail only in the middle of a round; a missing key is missing configuration.
- Exit code 0 when a round finished, whatever its outcome. Exit code 2, with one line on standard error and nothing in the log, for a policy or scenario file that cannot be read or is malformed, and for missing configuration.

## 5. Tests

The implementer writes `tests/test_graph.py` with the ten tests listed in the stage file. The acceptance tests are written from this file alone, by someone who has not seen the code.

Tests set MLflow up with `tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")`, so nothing is written outside the test's temporary folder.
