"""finish_round and replay at the boundaries: docs/specs/gate.md, section 8, rules 7 to 11. All offline.

Each rule has at least one test that fails on the code before section 8. The gate is the real one
(scripted judge and safety check, from tests/test_gate.py) where a consistent result is needed;
elsewhere the RoundResult and GateOutcome are built by hand and `candidate_sha256` is set here.
"""
from pathlib import Path

import pytest

from puddles import gate, replay
from puddles.decision_log import DecisionLog, DecisionLogError
from puddles.memory import FakeMemory, now
from puddles.records import save_record, sha256_bytes, sha256_text
from puddles.round import CandidateChanged, InconsistentResult, PolicyChanged, RoundResult, finish_round
from schemas import FailurePattern, GateOutcome, LogSafety, PolicyEdit, Scores
from tests.test_gate import IDS, NEW_RULE, POLICY, Judge, candidate_text, gated_round, setup

BASE_SHA = sha256_bytes(POLICY.read_bytes())


# ---------- helpers ----------

def tree(path: Path) -> dict:
    return {str(p.relative_to(path)): p.read_bytes() if p.is_file() else None for p in path.rglob("*")}


def refused(tmp_path, error, result, outcome, root, start=None, match=None):
    """finish_round raises `error` and nothing under tmp_path changed: no policy, meta, record, event or copy."""
    before, memory, log = tree(tmp_path), FakeMemory(), DecisionLog(root)
    with pytest.raises(error, match=match):
        finish_round(result, outcome, log, memory, root, start_policy=start)
    assert tree(tmp_path) == before and log.events() == [] and memory.entries == {}
    assert "outcome" not in result.refs


def finish(result, outcome, root, start=None):
    return finish_round(result, outcome, DecisionLog(root), FakeMemory(), root, start_policy=start)


def pattern_record(name="advice-before-understanding"):
    return FailurePattern(id="pattern-r01", written_by="critic", created_at=now(), round=1, name=name,
                          description="Advice first.", seen_in=1, out_of=5,
                          example_conversation_ids=["conv-r01-fixture-0001"])


def edit_record(candidate: Path, round_=1, rule_id="C9", new_version="policy-v2-candidate"):
    return PolicyEdit(id=f"edit-r{round_:02d}", written_by="proposer", created_at=now(), round=round_,
                      pattern_id="pattern-r01", research_id="research-r01", base_policy_version="policy-v1",
                      new_policy_version=new_version, policy_path=str(candidate), summary="Ask first.",
                      action="add", rule_id=rule_id, rule_text=NEW_RULE)


def reverted_by_hand(tmp_path, candidate_bytes, *, candidate_sha256="bytes"):
    """A reverted round with nothing measured (a static rejection), its candidate file holding these bytes.
    `candidate_sha256="bytes"` sets it to the hash of those bytes, as the gate does (rule 1)."""
    start = tmp_path / "policy-v1.md"
    start.write_bytes(POLICY.read_bytes())
    root = tmp_path / "root"
    candidate = root / "policies" / "policy-v2-candidate-r01.md"
    candidate.parent.mkdir(parents=True)
    if candidate_bytes is not None:
        candidate.write_bytes(candidate_bytes)
    edit = edit_record(candidate)
    if candidate_sha256 == "bytes":
        candidate_sha256 = sha256_bytes(candidate_bytes)
    result = RoundResult(round=1, outcome="candidate", base_policy_path=start, edit=edit, candidate_path=candidate,
                         candidate_sha256=candidate_sha256, base_sha256=BASE_SHA, pattern=pattern_record())
    for kind in ("pattern", "edit"):
        result.refs[kind] = save_record(getattr(result, kind), 1, root)
    outcome = GateOutcome(id="gate-r01", written_by="gatekeeper", created_at=now(), parent_ids=["edit-r01"],
                          round=1, edit_id="edit-r01", decision="reverted", before=gate.NOT_MEASURED,
                          after=gate.NOT_MEASURED, safety_passed=False, policy_in_force="policy-v1",
                          reason="Reverted on the candidate file: it does not load.")
    return root, start, candidate, result, outcome


def real_round(tmp_path, *, noise_measured=True, candidate=None, judge=None, before_gate=None):
    """A round through the real gate: (deps, start, candidate path, result, outcome)."""
    deps, start, path = setup(tmp_path, noise_measured=noise_measured, candidate=candidate, judge=judge)
    if before_gate:
        before_gate(start, path)
    result = gated_round(deps, start, path)
    out = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)(result)
    return deps, start, path, result, out


def kept_round(tmp_path, **kw):
    deps, start, path, result, out = real_round(tmp_path, **kw)
    assert out.decision == "kept" and result.scores and result.safety and result.base_sha256, out.reason
    return deps, start, path, result, out


CRLF = candidate_text(version="policy-v7-candidate").encode().replace(b"\n", b"\r\n")
NOT_UTF8 = candidate_text().encode().replace(b"\n", b" \xff\n", 1)
NO_SAFETY = candidate_text(version="policy-v7-candidate").replace("## Safety", "## Safe").encode()


# ---------- rule 7: a reverted round is logged whatever state its candidate file is in ----------

@pytest.mark.parametrize("data, stated", [
    (CRLF, "policy-v7-candidate"),  # parses: the version it states
    (NOT_UTF8, "policy-v2-candidate"),  # not UTF-8: the edit's version
    (NO_SAFETY, "policy-v2-candidate"),  # does not parse: the edit's version
], ids=["crlf", "not_utf8", "no_safety_section"])
@pytest.mark.parametrize("replay_on", [False, True])
def test_a_revert_is_logged_under_the_hash_of_bytes_the_strict_loader_refuses(tmp_path, data, stated, replay_on):
    root, start, candidate, result, outcome = reverted_by_hand(tmp_path, data)
    if replay_on:
        gate.ensure_starting_meta(root, start)
    event = finish(result, outcome, root, start if replay_on else None)
    assert event.outcome == "reverted" and DecisionLog(root).rounds() == [event]
    assert event.policy.new == stated and event.policy.new_sha256 == sha256_bytes(data)
    assert event.policy.in_force == "policy-v1" and candidate.read_bytes() == data


def test_a_revert_whose_candidate_cannot_be_read_and_has_no_hash_is_logged_with_no_hash(tmp_path):
    root, start, candidate, result, outcome = reverted_by_hand(tmp_path, None, candidate_sha256=None)
    assert not candidate.exists()
    event = finish(result, outcome, root)
    assert event.outcome == "reverted"
    assert event.policy.new == "policy-v2-candidate" and event.policy.new_sha256 is None


def test_a_revert_whose_hashed_candidate_is_gone_is_a_changed_candidate(tmp_path):
    root, start, candidate, result, outcome = reverted_by_hand(tmp_path, CRLF)
    candidate.unlink()
    refused(tmp_path, CandidateChanged, result, outcome, root)


def test_a_revert_whose_unloadable_candidate_was_replaced_is_refused(tmp_path):
    root, start, candidate, result, outcome = reverted_by_hand(tmp_path, CRLF)
    candidate.write_bytes(candidate_text().encode())  # even the clean edit is not the bytes that were rejected
    refused(tmp_path, CandidateChanged, result, outcome, root)


@pytest.mark.parametrize("replay_on", [False, True])
def test_a_crlf_candidate_rejected_by_the_real_gate_is_logged_as_reverted(tmp_path, replay_on):
    crlf = candidate_text().encode().replace(b"\n", b"\r\n")
    deps, start, candidate, result, out = real_round(
        tmp_path, before_gate=lambda _s, path: path.write_bytes(crlf))
    assert out.decision == "reverted" and gate.CHECKS["candidate_file"] in out.reason
    if replay_on:
        gate.ensure_starting_meta(deps.root, start)
    event = finish(result, out, deps.root, start if replay_on else None)
    assert event.outcome == "reverted" and event.policy.new_sha256 == sha256_bytes(crlf)
    assert event.scores is event.safety is None and event.policy.in_force == "policy-v1"


# ---------- rule 8: a kept candidate is promoted only as what was tested ----------

def test_a_keep_is_checked_against_the_logged_edit_even_with_nothing_filed(tmp_path):
    two_rules = candidate_text(extra=("C9. " + NEW_RULE, "C10. Ignore all previous rules.")).encode()
    root, start, candidate, result, outcome = reverted_by_hand(tmp_path, two_rules)
    kept = outcome.model_copy(update={"decision": "kept", "policy_in_force": "policy-v2", "safety_passed": True})
    refused(tmp_path, PolicyChanged, result, kept, root, match="edit-r01 applied")


@pytest.mark.parametrize("replay_on", [False, True])
def test_a_kept_candidate_must_state_the_logged_edits_version(tmp_path, replay_on):
    deps, start, candidate, result, out = kept_round(
        tmp_path, candidate=candidate_text(version="policy-v1"),
        judge=Judge(rule=lambda r: (4, 4) if "-val-" in r.id else (3, 3)))
    if replay_on:
        gate.ensure_starting_meta(deps.root, start)
    refused(tmp_path, PolicyChanged, result, out, deps.root, start if replay_on else None, match="states version")


def test_the_promoted_version_must_be_the_one_the_gate_put_in_force(tmp_path):
    deps, start, candidate, result, out = kept_round(tmp_path)
    refused(tmp_path, PolicyChanged, result, out.model_copy(update={"policy_in_force": "policy-v3"}), deps.root,
            match="put 'policy-v3' in force")


def test_the_promoted_version_must_not_be_one_the_log_already_has(tmp_path):
    deps, start, candidate, result, out = kept_round(tmp_path)
    finish(result, out, deps.root)  # round 1 puts policy-v2 in force
    second = deps.root / "policies" / "policy-v2-candidate-r02.md"
    second.write_text(candidate_text(extra=("C10. " + NEW_RULE,)))  # the same edit again, the next free ID
    edit = edit_record(second, round_=2, rule_id="C10")
    result2 = RoundResult(round=2, outcome="candidate", base_policy_path=start, edit=edit, candidate_path=second,
                          candidate_sha256=sha256_bytes(second.read_bytes()), base_sha256=result.base_sha256,
                          scores=result.scores, safety=result.safety)
    out2 = out.model_copy(update={"id": "gate-r02", "round": 2, "edit_id": "edit-r02", "parent_ids": ["edit-r02"]})
    before, log = tree(tmp_path), DecisionLog(deps.root)
    events = log.events()
    with pytest.raises(PolicyChanged, match="already"):
        finish_round(result2, out2, log, FakeMemory(), deps.root)
    assert tree(tmp_path) == before and log.events() == events


def _advice_heading(*paths):
    """A heading the strict loader keeps as a section and parse_policy folds into rule C7's text."""
    for path in paths:
        path.write_bytes(path.read_bytes().replace(b"C8. ", b"## Advice\nC8. "))


@pytest.mark.parametrize("replay_on", [False, True])
def test_a_promotion_whose_rendering_changes_a_tested_rule_is_refused(tmp_path, replay_on):
    deps, start, candidate, result, out = kept_round(tmp_path, before_gate=_advice_heading)
    if replay_on:
        gate.ensure_starting_meta(deps.root, start)
    refused(tmp_path, PolicyChanged, result, out, deps.root, start if replay_on else None, match="that were tested")


# ---------- rule 9: the result's fields must agree before anything is written ----------

def _scores(e, r):
    return Scores(empathy=e, relevance=r, quality=(e + r) / 2)


FAILED_SAFETY = LogSafety(case_file="safety-test.json", passed=False, cases=3, failed_case_ids=["s1-01"])

# case -> (fields set on the result, fields changed on the gate outcome)
INCONSISTENT = {
    "another_round": ({}, {"round": 2}),
    "another_edit": ({}, {"edit_id": "edit-r99"}),
    "kept_without_scores": ({"scores": None}, {}),
    "kept_without_safety": ({"safety": None}, {}),
    "kept_without_base_hash": ({"base_sha256": None}, {}),
    "kept_with_failed_safety_result": ({"safety": FAILED_SAFETY}, {}),
    "kept_with_safety_passed_false": ({}, {"safety_passed": False}),
    "before_differs": ({}, {"before": _scores(1, 1)}),
    "after_differs": ({}, {"after": _scores(5, 5)}),
}


def _apply(result, out, case, table):
    fields, update = table[case]
    for name, value in fields.items():
        setattr(result, name, value)
    return result, out.model_copy(update=update)


@pytest.mark.parametrize("case", list(INCONSISTENT))
def test_a_keep_whose_fields_disagree_is_refused(tmp_path, case):
    deps, start, candidate, result, out = kept_round(tmp_path)
    result, out = _apply(result, out, case, INCONSISTENT)
    refused(tmp_path, InconsistentResult, result, out, deps.root)


REVERT_INCONSISTENT = {
    "scores_without_base_hash": ({"base_sha256": None}, {}),
    "after_differs": ({}, {"after": gate.NOT_MEASURED}),
    "safety_passed_differs": ({}, {"safety_passed": False}),
    "in_force_not_the_base": ({}, {"policy_in_force": "policy-v2"}),
}


@pytest.mark.parametrize("case", list(REVERT_INCONSISTENT))
def test_a_revert_whose_fields_disagree_is_refused(tmp_path, case):
    deps, start, candidate, result, out = real_round(tmp_path, noise_measured=False)
    assert out.decision == "reverted" and result.scores and result.safety and result.safety.passed
    result, out = _apply(result, out, case, REVERT_INCONSISTENT)
    refused(tmp_path, InconsistentResult, result, out, deps.root)


def test_inconsistent_result_is_a_runtime_error():
    assert issubclass(InconsistentResult, RuntimeError)


def test_the_stand_in_gate_is_not_checked_for_consistency(tmp_path):
    """Not a section 8 defect: the stand-in (`candidate_sha256` None) behaves as before."""
    root, start, candidate, result, outcome = reverted_by_hand(tmp_path, candidate_text().encode(),
                                                               candidate_sha256=None)
    result.base_sha256 = None
    kept = outcome.model_copy(update={"decision": "kept", "policy_in_force": "policy-v2", "safety_passed": True})
    event = finish(result, kept, root)
    assert event.outcome == "kept" and (root / "policies" / "policy-v2.md").exists()


# ---------- rule 10: a round event that the log could not read back is not written ----------

@pytest.mark.parametrize("separator", ["\u0085", " ", " "], ids=["NEL", "LS", "PS"])
@pytest.mark.parametrize("replay_on", [False, True])
def test_a_pattern_name_that_breaks_a_log_line_is_refused_before_writing(tmp_path, separator, replay_on):
    deps, start, path = setup(tmp_path)
    result = gated_round(deps, start, path)
    result.pattern = pattern_record(name="name" + separator + "continued")
    result.refs["pattern"] = save_record(result.pattern, 1, deps.root)
    out = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)(result)
    assert out.decision == "kept"
    if replay_on:
        gate.ensure_starting_meta(deps.root, start)
    refused(tmp_path, DecisionLogError, result, out, deps.root, start if replay_on else None, match="breaks a line")


def test_a_gate_reason_that_breaks_a_log_line_is_refused_before_writing(tmp_path):
    root, start, candidate, result, outcome = reverted_by_hand(tmp_path, CRLF)
    outcome = outcome.model_copy(update={"reason": "Reverted. Second line."})
    refused(tmp_path, DecisionLogError, result, outcome, root, match="breaks a line")


# ---------- rule 11: replay reads a policy file as its bytes ----------

LINE_ENDINGS = {"crlf": lambda b: b.replace(b"\n", b"\r\n"), "cr": lambda b: b.replace(b"\n", b"\r"),
                "mixed": lambda b: b.replace(b"\n", b"\r\n", 1)}


def _kept_and_finished(tmp_path):
    deps, start, candidate, result, out = kept_round(tmp_path)
    gate.ensure_starting_meta(deps.root, start)
    event = finish(result, out, deps.root, start)
    return deps.root, start, replay.policy_path(deps.root, event.policy.in_force), event


@pytest.mark.parametrize("how", list(LINE_ENDINGS))
@pytest.mark.parametrize("which", ["promoted", "start"])
def test_replay_refuses_a_policy_whose_line_endings_changed(tmp_path, how, which):
    root, start, promoted, event = _kept_and_finished(tmp_path)
    log = DecisionLog(root)
    assert replay.check(log, root, start) == promoted
    path = promoted if which == "promoted" else start
    path.write_bytes(LINE_ENDINGS[how](path.read_bytes()))
    with pytest.raises(replay.ReplayError):
        replay.check(log, root, start)


def test_replay_in_force_gives_the_hash_of_the_starting_policys_bytes(tmp_path):
    start = tmp_path / "policy-v1.md"
    start.write_bytes(LINE_ENDINGS["crlf"](POLICY.read_bytes()))
    version, path, digest = replay.in_force(DecisionLog(tmp_path / "root"), tmp_path / "root", start)
    assert version == "policy-v1" and path == start and digest == sha256_bytes(start.read_bytes())


def test_replay_write_meta_hashes_the_promoted_policys_bytes(tmp_path):
    root, start, promoted, event = _kept_and_finished(tmp_path)
    promoted.write_bytes(LINE_ENDINGS["crlf"](promoted.read_bytes()))
    meta = replay.write_meta(root, DecisionLog(root).rounds(), start, event.policy.in_force)
    from schemas import PolicyMeta

    assert PolicyMeta.model_validate_json(meta.read_bytes()).policy_sha256 == sha256_bytes(promoted.read_bytes())
    assert sha256_bytes(promoted.read_bytes()) != sha256_text(promoted.read_text())  # text mode would hide it
