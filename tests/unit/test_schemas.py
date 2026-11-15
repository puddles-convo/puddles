"""The parts of schemas.py that carry behaviour: note limits and pattern identity."""
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

import schemas as S


def _live(kind: str, count: int) -> list[S.LiveNote]:
    return [S.LiveNote(note_id=f"note-{kind}-{i}", kind=kind, last_confirmed=date(2026, 11, 15))
            for i in range(count)]


def test_note_caps_cover_every_kind_and_total_23():
    assert set(S.NOTE_CAPS) == set(S.NoteKind.__args__)
    assert sum(S.NOTE_CAPS.values()) == 23


def test_index_accepts_every_kind_at_its_limit():
    notes = [note for kind, cap in S.NOTE_CAPS.items() for note in _live(kind, cap)]
    assert len(S.LiveNoteIndex(user_id="user-01", notes=notes).notes) == 23


@pytest.mark.parametrize("kind", sorted(S.NOTE_CAPS))
def test_index_rejects_one_note_over_a_limit(kind):
    with pytest.raises(ValidationError, match=kind):
        S.LiveNoteIndex(user_id="user-01", notes=_live(kind, S.NOTE_CAPS[kind] + 1))


def test_note_action_rejects_an_unknown_kind():
    with pytest.raises(ValidationError):
        S.NoteAction(action="add", kind="diagnosis", fact="x")


def _pattern(id: str, same_as: str | None = None) -> S.FailurePattern:
    return S.FailurePattern(
        id=id, written_by="critic", created_at=datetime(2026, 11, 14, tzinfo=timezone.utc),
        round=3, name="advice-before-understanding", description="d", seen_in=2, out_of=3,
        example_conversation_ids=["conv-1", "conv-2"], same_as_pattern_id=same_as)


def test_first_sighting_is_its_own_identity():
    assert _pattern("pattern-r03").pattern_id == "pattern-r03"


def test_recurrence_keeps_its_own_id_and_the_first_sightings_identity():
    again = _pattern("pattern-r07", same_as="pattern-r03")
    assert (again.id, again.pattern_id) == ("pattern-r07", "pattern-r03")
    assert "pattern_id" not in again.model_dump()


def test_log_pattern_identity_follows_the_first_sighting():
    logged = S.LogPattern(id="pattern-r07", same_as_pattern_id="pattern-r03", name="n",
                          seen_in=1, out_of=3, status_before="fixed")
    assert logged.pattern_id == "pattern-r03"
