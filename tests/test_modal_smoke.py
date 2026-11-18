"""The endpoint smoke test, run against scripted clients: no network, no keys."""
import dataclasses
import json
import time

import pytest

from loop.llm import FakeClient, client_from_env
from loop.policy import load_policy
from puddles.inference import crisis
from scripts.modal_smoke import check_auth, percentile, reply_problems, run_smoke

GOOD_REPLY = "That sounds exhausting. What's been on your mind at night?"
NONE_LINE = "none | - | ordinary tiredness after work"
CRISIS_LINE = "crisis | self | says they do not want to be here"


@pytest.fixture
def policy():
    return load_policy("policies/policy-v1.md")


@pytest.fixture
def path():
    return crisis.load_path()


def _run(policy, path, helper, checker, **kw):
    return run_smoke(FakeClient(helper), FakeClient(checker), policy, path, **kw)


def test_passes_on_clean_replies(policy, path):
    report = _run(policy, path, [GOOD_REPLY], [NONE_LINE, CRISIS_LINE])
    assert report["passed"], report["failures"]
    assert [c["call"] for c in report["calls"]] == [
        "helper (first call)", "checker, harmless message", "checker, crisis message"]


def test_fails_on_an_empty_reply(policy, path):
    report = _run(policy, path, ["", ""], [NONE_LINE, CRISIS_LINE])
    assert not report["passed"]
    assert any("helper request failed" in f for f in report["failures"])


def test_fails_on_thinking_text(policy, path):
    report = _run(policy, path, ["<think>the user is stressed</think> That sounds hard."], [NONE_LINE, CRISIS_LINE])
    assert any("thinking text" in f for f in report["failures"])


def test_fails_on_an_unparseable_checker_line(policy, path):
    report = _run(policy, path, [GOOD_REPLY], ["looks fine to me", CRISIS_LINE])
    assert any("unreadable" in f for f in report["failures"])


def test_fails_when_the_crisis_message_is_not_rated_crisis(policy, path):
    report = _run(policy, path, [GOOD_REPLY], [NONE_LINE, NONE_LINE])
    assert any("not 'crisis'" in f for f in report["failures"])


def test_fails_when_the_checker_is_slower_than_its_limit(policy, path):
    fast_limit = dataclasses.replace(path, timeout_s=0.01)

    def slow(*args, **kwargs):
        time.sleep(0.05)
        return NONE_LINE if "okay" in json.dumps(args) else CRISIS_LINE

    report = run_smoke(FakeClient([GOOD_REPLY]), FakeClient(slow), policy, fast_limit)
    assert any("over its 0.01s limit" in f for f in report["failures"])


def test_warm_and_concurrent_calls_report_latency(policy, path):
    report = _run(policy, path, lambda *a, **k: GOOD_REPLY,
                  lambda *a, **k: CRISIS_LINE if "anymore" in json.dumps(a) else NONE_LINE,
                  warm=4, concurrent=3)
    assert report["passed"], report["failures"]
    assert report["latency"]["helper_warm"]["n"] == 4 and report["latency"]["checker_warm"]["n"] == 4
    assert report["latency"]["concurrent"]["n"] == 3


def test_reply_problems_and_percentile():
    assert reply_problems(GOOD_REPLY) == []
    assert reply_problems("  ") == ["the reply is empty"]
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0


def test_thinking_off_reaches_the_request_body(monkeypatch):
    """HELPER_EXTRA_BODY is what turns Qwen's thinking off on vLLM; check it is sent."""
    sent = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": GOOD_REPLY}}]}).encode()

    def fake_urlopen(request, timeout=None):
        sent["body"] = json.loads(request.data)
        sent["auth"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setenv("HELPER_URL", "https://example.modal.run/v1")
    monkeypatch.setenv("HELPER_MODEL", "qwen3.5-9b")
    monkeypatch.setenv("HELPER_API_KEY", "test-key")
    monkeypatch.setenv("HELPER_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')
    monkeypatch.setattr("loop.llm.urllib.request.urlopen", fake_urlopen)
    client_from_env("HELPER").complete("system", [{"role": "user", "content": "hi"}], max_tokens=10)
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["body"]["model"] == "qwen3.5-9b"
    assert sent["auth"] == "Bearer test-key"


def test_check_auth_reports_a_refusal_and_an_open_endpoint(monkeypatch):
    import urllib.error

    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("scripts.modal_smoke.urllib.request.urlopen", refuse)
    assert check_auth("https://x/v1", "m") == (True, "refused without a key (HTTP 401)")

    class Open:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("scripts.modal_smoke.urllib.request.urlopen", lambda request, timeout=None: Open())
    ok, message = check_auth("https://x/v1", "m")
    assert not ok and "accepted" in message
