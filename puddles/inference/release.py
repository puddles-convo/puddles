"""The release step: code only. It decides what the robot says from the keyword, the check and the draft.

No model, no clock, no file. The helper's draft is spoken only when there is no keyword match and a
`none` or `concern` check for this same session and turn; everything else gets a line from the path.
"""
from __future__ import annotations

from typing import Optional

from puddles.inference.crisis import CheckStatus, CrisisPath
from schemas import CheckedMessage, HelperReply


def _counts(session_id: str, turn: int, check_status: CheckStatus, checked: Optional[CheckedMessage]) -> bool:
    return (check_status == "answered" and checked is not None
            and checked.session_id == session_id and checked.turn == turn)


def _fixed(path: CrisisPath, category: Optional[str]) -> str:
    return path.fixed_responses[category or "unsure"]


def release(*, session_id: str, turn: int, policy_version: str, path: CrisisPath,
            keyword: Optional[str], check_status: CheckStatus, checked: Optional[CheckedMessage],
            draft: Optional[HelperReply], failed_before: int) -> tuple[HelperReply, int]:
    """Return the reply to speak and the new count of failed checks in a row."""
    counts = _counts(session_id, turn, check_status, checked)
    failed = 0 if counts else failed_before + 1
    crisis = counts and checked is not None and checked.risk == "crisis"
    category = checked.category if crisis and checked is not None else None

    if keyword is not None:                       # rule 1: a keyword forces a crisis
        text, fixed = _fixed(path, category), True
    elif crisis:                                  # rule 2
        text, fixed = _fixed(path, category), True
    elif counts and draft is not None:            # rule 3: the only rule that speaks the draft
        text, fixed = draft.text, False
    elif counts:                                  # rule 4: the helper failed
        text, fixed = path.holding_line, True
    elif failed >= 2:                             # rule 5
        text, fixed = path.offline_line, True
    else:                                         # rule 6
        text, fixed = path.holding_line, True

    reply = HelperReply(session_id=session_id, turn=turn, text=text, policy_version=policy_version,
                        fixed_handoff_response=fixed)
    return reply, failed
