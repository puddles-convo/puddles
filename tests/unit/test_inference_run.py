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


def test_missing_policy_is_an_error(tmp_path):
    messages = tmp_path / "m.txt"
    messages.write_text("hi\n", encoding="utf-8")
    assert inference_run.main(["--fake", "--policy", str(tmp_path / "none.md"),
                               "--messages", str(messages), "--out", str(tmp_path / "t.jsonl")]) == 2
