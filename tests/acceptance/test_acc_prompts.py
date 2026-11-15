"""Expected prompt text is derived from the permitted agent documentation."""

import importlib
import re

import pytest


@pytest.fixture
def api():
    return importlib.import_module("loop.prompts")


def system_block(repo_root, agent):
    doc = (repo_root / "docs" / "agents" / agent).read_text(encoding="utf-8")
    section = doc.split("### System message", 1)[1]
    return section.split("```", 2)[1].strip("\n")


@pytest.mark.parametrize("notes", [(), ("Works nights.", "Prefers quiet.")])
def test_helper_prompt_changes_only_documented_placeholders(api, repo_root, policy, notes):
    expected = system_block(repo_root, "t2-i3-helper.md")
    expected = expected.replace("{policy_version}", policy.version).replace("{policy_text}", policy.text.strip())
    expected = expected.replace(
        '{user_notes, or "Nothing yet. This is your first conversation."}',
        "\n".join("- " + note for note in notes) if notes else "Nothing yet. This is your first conversation.",
    )
    # The spec does not settle a file-ending newline on the .txt templates.
    assert api.render_helper_system(policy, notes).strip("\n") == expected
    assert not re.search(r"\{[^{}]+\}", api.render_helper_system(policy, notes))


def test_helper_does_not_expand_placeholders_inside_inserted_values(api, repo_root, tmp_path):
    from loop.policy import load_policy
    notes_placeholder = '{user_notes, or "Nothing yet. This is your first conversation."}'
    text = ('# {policy_text}\n\n## Safety\nS1. Keep {policy_version} literally.\n'
            '## Conversation\nC1. Keep ' + notes_placeholder + ' literally.\n')
    path = tmp_path / "braces.md"
    path.write_text(text, encoding="utf-8")
    policy = load_policy(path)
    notes = ("Keep {policy_text} and {policy_version}.",)
    values = {"{policy_version}": "{policy_text}", "{policy_text}": text.strip(),
              notes_placeholder: "- " + notes[0]}
    # Build expected text from template spans, never by cascading replacements.
    expected = re.sub(r"\{[^{}]+\}", lambda match: values[match.group()],
                      system_block(repo_root, "t2-i3-helper.md"))
    assert api.render_helper_system(policy, notes).strip("\n") == expected


def test_scenario_placeholder_like_text_is_preserved_in_every_inserted_field(api, repo_root, scenario_factory):
    scenario = scenario_factory(situation="I wrote {emotion_type} and {openness}.",
                                emotion_type="{problem_type}", problem_type="{initial_intensity}")
    expected = re.sub(r"\{([^{}]+)\}", lambda match: str(getattr(scenario, match[1])),
                      system_block(repo_root, "t1-simulated-user.md"))
    assert api.render_simulated_user_system(scenario).strip("\n") == expected


@pytest.mark.parametrize("openness", ["open", "guarded", "resistant"])
def test_simulated_user_prompt_changes_only_five_scenario_placeholders(api, repo_root, scenario_factory, openness):
    scenario = scenario_factory(openness=openness, situation="Café shifts were cut.")
    expected = system_block(repo_root, "t1-simulated-user.md")
    for name in ("situation", "emotion_type", "initial_intensity", "problem_type", "openness"):
        expected = expected.replace("{" + name + "}", str(getattr(scenario, name)))
    rendered = api.render_simulated_user_system(scenario)
    assert rendered.strip("\n") == expected
    assert not re.search(r"\{[^{}]+\}", rendered)


def test_helper_history_preserves_content_and_order(api, turns):
    assert api.helper_messages(turns) == [
        {"role": "user", "content": "I am worried."},
        {"role": "assistant", "content": "What happened?"},
        {"role": "user", "content": "My hours were cut."},
    ]


def test_empty_helper_history(api):
    assert api.helper_messages(()) == []


@pytest.mark.parametrize("nonempty", [False, True])
def test_simulated_user_history_always_has_opening_then_swapped_roles(api, turns, nonempty):
    history = turns if nonempty else ()
    expected = [{"role": "user", "content": "(The helper is listening. Begin.)"}]
    if nonempty:
        expected += [
            {"role": "assistant", "content": "I am worried."},
            {"role": "user", "content": "What happened?"},
            {"role": "assistant", "content": "My hours were cut."},
        ]
    assert api.simulated_user_messages(history) == expected
