"""Public tracing behavior specified by docs/specs/graph.md."""
import os
from pathlib import Path
import socket

import mlflow
import pytest

from loop.llm import FakeClient
from puddles import tracing


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.endswith("API_KEY") or key in {"MLFLOW_TRACKING_URI", "MLFLOW_TRACING_DESTINATION"}:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("Acceptance tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()
    if mlflow.active_run() is not None:
        mlflow.end_run()


def spans():
    mlflow.flush_trace_async_logging()
    experiment = mlflow.get_experiment_by_name("puddles")
    found = mlflow.search_traces(experiment_ids=[experiment.experiment_id], return_type="list")
    assert len(found) == 1
    return found[0].data.spans


@pytest.mark.parametrize("source", ["argument", "environment", "default"])
def test_setup_address_precedence_parent_experiment_and_autolog(tmp_path, monkeypatch, source):
    explicit = f"sqlite:///{tmp_path}/explicit/nested/mlflow.db"
    environment = f"sqlite:///{tmp_path}/environment/nested/mlflow.db"
    if source != "default":
        monkeypatch.setenv("MLFLOW_TRACKING_URI", environment)
    else:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    expected = {"argument": explicit, "environment": environment,
                "default": "sqlite:///runs/mlflow.db"}[source]
    calls = []
    monkeypatch.setattr(mlflow.langchain, "autolog", lambda *a, **k: calls.append("langchain"))
    monkeypatch.setattr(mlflow.openai, "autolog", lambda *a, **k: calls.append("openai"))
    kwargs = {"tracking_uri": explicit} if source == "argument" else {}
    # Both invocations must work; no number of repeated autolog calls is prescribed.
    assert tracing.setup(experiment="acceptance-custom", **kwargs) is None
    assert {"langchain", "openai"} <= set(calls)
    assert tracing.setup(experiment="acceptance-custom", **kwargs) is None
    assert mlflow.get_tracking_uri() == expected
    database = {"argument": tmp_path / "explicit/nested/mlflow.db",
                "environment": tmp_path / "environment/nested/mlflow.db",
                "default": tmp_path / "runs/mlflow.db"}[source]
    assert database.parent.is_dir()
    assert {"langchain", "openai"} <= set(calls)
    with mlflow.start_run() as run:
        experiment = mlflow.get_experiment_by_name("acceptance-custom")
        assert run.info.experiment_id == experiment.experiment_id


@pytest.mark.parametrize("optional", [{}, {"temperature": 0.13, "seed": 812}])
def test_traced_client_forwards_arguments_text_and_chat_model_span(optional):
    reply = "  A reply with whitespace.\n"
    inner = FakeClient([reply])
    client = tracing.TracedClient(inner, "acceptance-model")
    messages = [{"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "What happened?"}]
    assert client.inner is inner and client.name == "acceptance-model"
    result = client.complete("System instruction.", messages, max_tokens=137, **optional)
    expected = dict(system="System instruction.", messages=messages, max_tokens=137,
                    temperature=optional.get("temperature", 0.7), seed=optional.get("seed"))
    assert result == reply
    assert inner.calls == [expected]
    recorded = spans()
    assert len(recorded) == 1
    span = recorded[0]
    assert span.name == "llm:acceptance-model"
    assert span.span_type == "CHAT_MODEL"
    assert all(span.inputs[key] == value for key, value in expected.items())
    assert span.outputs == reply


def test_traced_client_passes_the_identical_exception_through():
    sentinel = ValueError("inner failure survives unchanged")

    def fail(system, messages):
        raise sentinel

    inner = FakeClient(fail)
    client = tracing.TracedClient(inner, "broken-model")
    with pytest.raises(ValueError) as error:
        client.complete("System.", [{"role": "user", "content": "Hi."}], max_tokens=41)
    assert error.value is sentinel
    assert len(inner.calls) == 1
    recorded = spans()
    assert len(recorded) == 1
    assert recorded[0].name == "llm:broken-model"
    assert recorded[0].span_type == "CHAT_MODEL"


def test_setup_non_sqlite_address_does_not_create_folders(monkeypatch):
    uri = "http://127.0.0.1:1"
    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(mlflow, "set_tracking_uri", lambda value: calls.append(("uri", value)))
        patch.setattr(mlflow, "set_experiment", lambda value: calls.append(("experiment", value)))
        patch.setattr(mlflow.langchain, "autolog", lambda *a, **k: calls.append(("langchain", None)))
        patch.setattr(mlflow.openai, "autolog", lambda *a, **k: calls.append(("openai", None)))
        patch.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("non-sqlite setup created a directory"))
        tracing.setup(tracking_uri=uri, experiment="remote-acceptance")
    assert calls[:2] == [("uri", uri), ("experiment", "remote-acceptance")]
    assert {kind for kind, _ in calls[2:]} == {"langchain", "openai"}
