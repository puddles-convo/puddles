"""Critic, researcher and proposer. Each turns one direct handoff plus memory into one record."""
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs" / "agents"


def system_prompt(doc: str) -> str:
    """The system message lives in the agent's doc, so the doc and the code cannot drift."""
    text = (DOCS / doc).read_text()
    m = re.search(r"### System message\n\n.*?\n\n```\n(.*?)\n```", text, re.S)
    if not m:
        raise RuntimeError(f"no system message found in docs/agents/{doc}")
    return m.group(1)


def bullet_lines(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "(none)"
