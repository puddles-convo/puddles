"""The demo page is built from the committed records and is committed up to date."""
import re

from demo import build


def test_committed_page_matches_the_records():
    assert build.main(["--check"]) == 0


def test_page_loads_nothing_from_the_network():
    page = build.render()
    assert not re.search(r"<(script|link|img|iframe)[^>]+(src|href)=[\"']https?:", page)
    assert "fetch(" not in page and "XMLHttpRequest" not in page


def test_every_recorded_round_and_session_is_in_the_page():
    data = build.data()
    stats = data["training"]["stats"]
    assert stats["rounds"] == len(data["training"]["rounds"])
    assert stats["rounds"] == stats["kept"] + stats["reverted"] + stats["no_candidate"]
    assert all(s["turns"] for c in data["chat"] if c["kind"] == "session" for s in c["sessions"])
    assert all(len(m["runs"]) == 2 for c in data["chat"] if c["kind"] == "pilot" for m in c["messages"])


def test_both_policies_are_compared_on_the_same_scenarios():
    compare = build.data()["compare"]
    assert [s["policy"] for s in compare["stats"]] == ["policy-v1", "policy-v4"]
    assert len(compare["scenarios"]) == 40
    for scenario in compare["scenarios"]:
        assert len(scenario["start"]) == len(scenario["final"]) == (2 if scenario["group"] == "validation" else 1)
