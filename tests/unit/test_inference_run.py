import json

from scripts import inference_run


def test_fake_run_writes_one_row_per_message_through_the_release_step(tmp_path):
    messages = tmp_path / "m.txt"
    messages.write_text("i lost my job\n\nI want to kill myself\nhard day [checker:crisis-self]\n",
                        encoding="utf-8")
    out = tmp_path / "t.jsonl"
    assert inference_run.main(["--fake", "--policy", "policies/policy-v2.md",
                               "--messages", str(messages), "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["turn"] for r in rows] == [1, 2, 3]
    assert rows[0]["released"] == "helper" and rows[0]["risk"] == "none"
    assert rows[1]["keyword"] is True and rows[1]["released"] == "fixed line"
    assert rows[2]["risk"] == "crisis" and rows[2]["released"] == "fixed line"
    assert rows[1]["robot"] != rows[0]["robot"]
    assert rows[1]["helper_draft"] == rows[0]["robot"]  # drafted, then held back by the release step


def test_missing_policy_is_an_error(tmp_path):
    messages = tmp_path / "m.txt"
    messages.write_text("hi\n", encoding="utf-8")
    assert inference_run.main(["--fake", "--policy", str(tmp_path / "none.md"),
                               "--messages", str(messages), "--out", str(tmp_path / "t.jsonl")]) == 2


def test_deadline_raises_the_http_timeout(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setenv("HELPER_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("HELPER_MODEL", "m")
    monkeypatch.setattr(inference_run, "run_turn", lambda session, text: seen.append(
        (session.helper_client.client.timeout, session.checker_client.timeout)) or (_ for _ in ()).throw(SystemExit))
    messages = tmp_path / "m.txt"
    messages.write_text("hi\n", encoding="utf-8")
    try:
        inference_run.main(["--policy", "policies/policy-v2.md", "--messages", str(messages),
                            "--out", str(tmp_path / "t.jsonl"), "--deadline", "300"])
    except SystemExit:
        pass
    assert seen == [(300.0, 300.0)]



def test_a_late_draft_is_not_recorded_against_the_next_turn():
    import threading

    release = threading.Event()

    class Slow:
        def complete(self, system, messages, **kwargs):
            release.wait(5)
            return "late draft"

    recording = inference_run._Recording(Slow())
    recording.next_turn()
    late = threading.Thread(target=recording.complete, args=("s", []), kwargs={"max_tokens": 5})
    late.start()  # turn 1's helper, still running when turn 2 starts
    recording.next_turn()
    release.set()
    late.join(5)
    assert recording.last is None and not recording.done.is_set()
