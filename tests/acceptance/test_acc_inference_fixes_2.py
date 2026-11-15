"""Independent offline acceptance of fixes T, N and U (commit aa064ae).

The comparison test emits every old/new trimming difference with its judgement
in its tmp_path, so the review is reproducible without retaining personal text.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import re
import socket

import pytest

from loop.helper import helper_reply, trim_to_sentence
from loop.llm import FakeClient
from puddles.inference import crisis, live, notes, pilot, turn
from puddles.inference.note_store import FakeUserMemory, NoteStore
from schemas import NOTE_CAPS, Turn
from scripts import chat

TODAY = date(2026, 11, 15)
NOW = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)
ABBREVIATIONS = ('Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'St.', 'vs.', 'etc.', 'approx.',
                 'U.S.', 'u.s.', 'U.s.', 'e.g.', 'E.G.', 'i.e.', 'a.m.', 'P.M.', 'U.S.A.')
REAL_DRAFT = ('If you might act on those thoughts or feel in immediate danger, call emergency services now. '
              'In the U.S. or Canada, call or text 988; elsewhere, contact your')
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    import mlflow
    from mlflow.tracking import fluent

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MLFLOW_DISABLE_AGENT_HINT', '1')
    monkeypatch.setenv('DO_NOT_TRACK', '1')
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'sqlite:///{tmp_path}/tracking.db')
    old_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(fluent, '_active_experiment_id', fluent._active_experiment_id)
    mlflow.set_tracking_uri(f'sqlite:///{tmp_path}/tracking.db')

    def forbidden(*args, **kwargs):
        raise AssertionError('These acceptance tests are offline')

    for name in ('connect', 'connect_ex', 'bind', 'listen'):
        monkeypatch.setattr(socket.socket, name, forbidden)
    yield
    mlflow.flush_trace_async_logging()
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(old_uri)


# Original implementation, independently checked against git show aa064ae^:loop/helper.py.
def old_trim(text):
    closers = re.escape('\"\'”’)]}')
    if re.search(rf'[.!?][{closers}]?$', text):
        return text
    ends = list(re.finditer(rf'[.!?][{closers}]?(?=\s)', text))
    return text[:ends[-1].end()] if ends else text


def trimming_cases():
    # 360 ordinary cases: quotes, ellipses, lists, decimal numbers and whitespace.
    for opening in ('I hear you.', 'That sounds hard!', 'Would a pause help?', 'No.',
                    'You said "stop."', '(I am listening.)'):
        for middle in ('You can rest.', 'Try 7.5 hours.', 'Call 988.', '1. Sit. 2. Breathe.',
                       'Some days... are hard.'):
            for separator in (' ', '\n', '\t'):
                for tail in (' Can we', ' What would', ' It might', ''):
                    raw = opening + separator + middle + tail
                    yield raw, old_trim(raw), 'Right: ordinary reply unchanged'
    # Every named abbreviation, mixed-case initialisms and three opener/closer pairs.
    for abbr in ABBREVIATIONS:
        for left, right in (('', ''), ('"', '"'), ('(', ')')):
            fragment = f'Try {left}{abbr}{right} some support and'
            yield 'I hear you. ' + fragment, 'I hear you.', 'Right: discard abbreviation fragment'
            yield fragment, fragment, 'Right by rule: no boundary, keep text whole'
            terminal = f'I hear you. Try {left}{abbr}{right}'
            yield terminal, terminal, 'Right: terminal abbreviation retained'
    yield REAL_DRAFT, REAL_DRAFT.split(' In the')[0], 'Right: retain complete emergency-services sentence'
    for raw in ("I'm listening. You can call or text 988 in the U.S. If you",
                'I hear you. Bring water, snacks, etc. When you'):
        yield raw, old_trim(raw), 'Right: preserve complete sentence before uppercase continuation'


def test_generated_trimming_comparison(tmp_path):
    assert trim_to_sentence("I'm listening. You can call or text 988 in the U.S. If you") == (
        "I'm listening. You can call or text 988 in the U.S.")
    assert trim_to_sentence('I hear you. Bring water, snacks, etc. When you') == (
        'I hear you. Bring water, snacks, etc.')
    cases = list(trimming_cases())
    differences = []
    problems = []
    for raw, expected, judgement in cases:
        actual, previous = trim_to_sentence(raw), old_trim(raw)
        if actual != expected:
            problems.append(dict(input=raw, old=previous, actual=actual, expected=expected))
        if actual != previous:
            differences.append(dict(input=raw, old=previous, new=actual, expected=expected, judgement=judgement))
    assert len(cases) == 525 and len(differences) == 109
    (tmp_path / 'trimming-differences.json').write_text(json.dumps(
        dict(compared=len(cases), differences=differences), indent=2), encoding='utf-8')
    assert not problems, problems


def test_manager_abbreviation_ruling():
    titles = {'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'St.'}
    afters = [('Upper unfinished', True), ('"Upper unfinished', True),
              ("'Upper unfinished", True), ('“Upper unfinished', True), ('‘Upper unfinished', True),
              ('(Upper unfinished', True), ('[Upper unfinished', True), ('{Upper unfinished', True),
              ('lower unfinished', False), ('3 unfinished', False), (', unfinished', False),
              ('; unfinished', False), ('"lower unfinished', False), ('(lower unfinished', False)]
    for abbr in ABBREVIATIONS:
        prefix = f'I hear you. That abbreviation was {abbr}'
        for after, uppercase in afters:
            raw = prefix + ' \t ' + after
            expected = prefix if uppercase and abbr not in titles else 'I hear you.'
            assert trim_to_sentence(raw) == expected, raw
        assert trim_to_sentence(prefix + '   ') == prefix
    assert trim_to_sentence('I hear you. You could ask your family Dr. Would you') == 'I hear you.'


def test_saved_drafts_trimming_differences(tmp_path):
    rows = [json.loads(line) for line in (ROOT / 'results/crisis-pilot/results.jsonl').read_text().splitlines()
            if line.strip()]
    differences = []
    old_988 = new_988 = 0
    for row in rows:
        raw = row.get('helper_draft')
        if raw is None:
            continue
        raw = raw.strip()
        previous, actual = old_trim(raw), trim_to_sentence(raw)
        if row['expect'] == 'fixed':
            old_988 += '988' in previous
            new_988 += '988' in actual
        if previous != actual:
            differences.append(dict(id=row['id'], run=row['run'], spoken=row['spoken'], input=raw,
                old=previous, new=actual, judgement='Right by abbreviation rule: remove unfinished U.S. clause',
                lost_988=('988' in previous and '988' not in actual)))
    (tmp_path / 'saved-trimming-differences.json').write_text(json.dumps(differences, indent=2), encoding='utf-8')
    assert len(differences) == 13
    assert (old_988, new_988) == (24, 21)


@pytest.mark.parametrize('raw, expected', [
    ('No. Maybe we can', 'No.'),
    ('I hear you. Sleep 7.5 hours and', 'I hear you.'),
    ('Call 988. You can also', 'Call 988.'),
    ('You asked "why?" I wonder', 'You asked "why?"'),
    ('Really... I wonder', 'Really...'),
    ('1. Breathe.\n2. Sit.\n3. Try', '1. Breathe.\n2. Sit.\n3.'),
    ('No punctuation here', 'No punctuation here'),
])
def test_trimmer_boundaries_and_no_is_a_sentence(raw, expected):
    assert trim_to_sentence(raw) == expected


def test_helper_keeps_signature_arguments_and_uses_trimmer(policy):
    client = FakeClient([REAL_DRAFT])
    result = helper_reply(client, policy, [Turn(speaker='user', text='A hard day.')],
                          session_id='s1', turn=1, user_notes=['Likes walks.'], max_tokens=80, seed=17)
    assert result.text == REAL_DRAFT.split(' In the')[0]
    assert result.fixed_handoff_response is False
    assert client.calls[0]['max_tokens'] == 80 and client.calls[0]['seed'] == 17
    assert client.calls[0]['messages'][-1]['content'] == 'A hard day.'


class RecordingNotes(chat.FakeNoteTaker):
    def __init__(self):
        self.seen = []
        self.systems = []

    def structured(self, system, user, schema):
        self.systems.append(system)
        self.seen.append(user.rsplit('<conversation>\n', 1)[-1].rsplit('\n</conversation>', 1)[0])
        return super().structured(system, user, schema)


def make_session(policy, tmp_path, *, helper=None, training=False):
    path = crisis.load_path()
    return turn.Session(session_id='s1', user_id='person-a', policy=policy, path=path,
                        helper_client=helper or chat.fake_helper(), checker_client=chat.fake_checker(path),
                        remember_me=True, consent_for_training=training, root=tmp_path / 'sessions')


def take(store, actions, *, remember=True, forget=False):
    class Answer:
        def structured(self, system, user, schema):
            return schema.model_validate(dict(actions=actions, forget_everything=forget))
    return notes.take_notes(store, Answer(), user_id='person-a', session_id='seed', turns=[],
                            remember_me=remember, today=TODAY, now=NOW)


def add(fact, kind='goals'):
    return dict(action='add', kind=kind, fact=fact)


@pytest.mark.parametrize('category', ['self', 'danger', 'unsure', 'keyword'])
@pytest.mark.parametrize('training', [False, True])
def test_crisis_filtered_but_record_and_ordinary_forget_survive(category, training, policy, tmp_path):
    store = NoteStore(FakeUserMemory(), tmp_path / 'memory')
    [note_id] = take(store, [add('An old fact.')]).added
    session = make_session(policy, tmp_path, training=training)
    message = (session.path.keywords[0] if category == 'keyword' else f'[checker:crisis-{category}]')
    message += ' private-crisis-marker [note:goals:must not become a note]'
    ordinary = f'Please forget that [forget:{note_id}] [note:people:Has a friend.]'
    turn.run_turn(session, message)
    turn.run_turn(session, ordinary)
    assert len(session.turns) == 2 * len(session.lines) == 4
    model = RecordingNotes()
    record, result = live.finish_session(session, store, model, today=TODAY, now=NOW)
    assert len(model.seen) == 1 and ordinary in model.seen[0]
    assert 'private-crisis-marker' not in model.seen[0]
    assert not any(line in model.seen[0] for line in session.path.fixed_responses.values())
    assert result.deleted == [note_id] and len(result.added) == 1
    assert [n.fact for n in store.stored('person-a').values()] == ['Has a friend.']
    assert (record is not None) is training
    if training:
        assert [t.text for t in record.turns][0] == message and len(record.turns) == 4
    assert session.turns == session.lines == []


@pytest.mark.parametrize('mode', ['all-crisis', 'empty'])
def test_note_taker_runs_for_all_crisis_but_not_empty(mode, policy, tmp_path):
    session = make_session(policy, tmp_path)
    if mode == 'all-crisis':
        turn.run_turn(session, '[checker:crisis-self] [note:goals:private marker]')
    model = RecordingNotes()
    _, result = live.finish_session(session, NoteStore(FakeUserMemory(), tmp_path / 'memory'), model,
                                    today=TODAY, now=NOW)
    assert model.seen == ([''] if mode == 'all-crisis' else [])
    assert not result.failed and result.added == []


@pytest.mark.parametrize('reply_kind', ['identical-draft', 'holding', 'offline'])
def test_non_crisis_fixed_or_identical_draft_is_visible(reply_kind, policy, tmp_path):
    path = crisis.load_path()
    helper = FakeClient([path.fixed_responses['self']]) if reply_kind == 'identical-draft' else None
    session = make_session(policy, tmp_path, helper=helper)
    marker = '[note:goals:An ordinary fact.]'
    if reply_kind == 'offline':
        turn.run_turn(session, '[checker:fail] first ordinary failure')
    message = marker if reply_kind == 'identical-draft' else f'[checker:fail] {marker}'
    released = turn.run_turn(session, message).reply
    assert released.fixed_handoff_response is (reply_kind != 'identical-draft')
    if reply_kind != 'identical-draft':
        assert released.text == getattr(path, reply_kind + '_line')
    model = RecordingNotes()
    _, result = live.finish_session(session, NoteStore(FakeUserMemory(), tmp_path / 'memory'), model)
    assert message in model.seen[0] and released.text in model.seen[0]
    assert len(result.added) == 1


@pytest.mark.parametrize('ending', ['end', 'sweep'])
def test_service_end_and_sweep_filter_crisis_and_keep_forgetting(ending, policy_path, tmp_path):
    from fastapi.testclient import TestClient
    from puddles.inference.service import ServiceConfig, create_app

    clock = [0.0]
    model = RecordingNotes()
    store = NoteStore(FakeUserMemory(), tmp_path / 'memory')
    [note_id] = take(store, [add('Old ordinary note.')]).added
    path = crisis.load_path()
    clients = iter([chat.fake_helper(), chat.fake_checker(path)])
    app = create_app(ServiceConfig(token='offline-token', policy_path=policy_path, root=tmp_path / 'sessions',
        store=store, client_factory=lambda: next(clients), note_llm_for=lambda training: model,
        path=path, clock=lambda: clock[0], idle_timeout_s=10))
    auth = {'Authorization': 'Bearer offline-token'}
    with TestClient(app) as client:
        assert client.post('/sessions', headers=auth, json=dict(session_id='s1', user_id='person-a',
            remember_me=False, consent_for_training=False)).status_code == 201
        for text in ('[checker:crisis-self] [note:goals:service-crisis-secret]',
                     f'Please forget this [forget:{note_id}]'):
            assert client.post('/sessions/s1/turns', headers=auth, json={'text': text}).status_code == 200
        if ending == 'end':
            response = client.post('/sessions/s1/end', headers=auth)
            assert response.status_code == 200 and response.json()['notes']['deleted'] == 1
        else:
            clock[0] = 11
            assert app.state.service.sweep() == ['s1']
    assert len(model.seen) == 1 and 'service-crisis-secret' not in model.seen[0]
    assert f'[forget:{note_id}]' in model.seen[0] and not store.stored('person-a')


def test_cli_filters_crisis_offline(policy_path, tmp_path, monkeypatch):
    model = RecordingNotes()
    monkeypatch.setattr(chat, 'FakeNoteTaker', lambda: model)
    output = io.StringIO()
    assert chat.main(['--fake', '--policy', str(policy_path), '--root', str(tmp_path / 'cli'),
        '--user-id', 'person-a', '--session-id', 's1', '--remember-me', 'yes', '--training', 'no'],
        stdin=io.StringIO('[checker:crisis-danger] [note:goals:cli-secret]\n'
                          '[note:people:Has a friend.]\n'), stdout=output) == 0
    assert len(model.seen) == 1 and 'cli-secret' not in model.seen[0]
    assert 'Has a friend.' in model.seen[0]


def test_effective_note_prompt_prohibits_crisis_and_preserves_deletion(policy, tmp_path):
    session = make_session(policy, tmp_path)
    turn.run_turn(session, 'An ordinary message [checker:concern]')
    model = RecordingNotes()
    live.finish_session(session, NoteStore(FakeUserMemory(), tmp_path / 'memory'), model)
    [system] = model.systems
    for phrase in ('Never write a note about thoughts of suicide or self-harm',
                   'a plan or means to harm themselves or someone else', 'or being in danger',
                   'never add a follow_up to check on them',
                   'if the person asked the robot to forget something, still delete it'):
        assert phrase in system


@pytest.mark.parametrize('remember,forget,expected', [(False, False, 'consent'),
                                                    (True, True, 'forget_everything'),
                                                    (False, True, 'forget_everything')])
def test_reasons_count_adds_and_replaces_and_keep_deletes(remember, forget, expected, tmp_path):
    store = NoteStore(FakeUserMemory(), tmp_path / 'memory')
    one, two = take(store, [add('Old goal.'), add('Old person.', 'people')]).added
    result = take(store, [dict(action='replace', note_id=one, kind='goals', fact='New goal.'),
                          add('New person.', 'people'), dict(action='delete', note_id=two)],
                  remember=remember, forget=forget)
    assert result.not_written == 2 and result.not_written_reasons == {expected: 2}
    assert two in result.deleted and two not in store.stored('person-a')


@pytest.mark.parametrize('failure', ['error', 'timeout'])
def test_reasons_limit_and_write_failure_are_private_and_sum(failure, tmp_path, caplog, capsys):
    memory = FakeUserMemory()
    store = NoteStore(memory, tmp_path / 'memory')
    secret = 'personal-words-that-must-not-enter-a-reason'
    ingest = memory.ingest
    if failure == 'error':
        def broken(entry):
            if secret in entry.content:
                raise RuntimeError(secret)
            return ingest(entry)
        memory.ingest = broken
    else:
        memory.wait = lambda *args, **kwargs: False
    actions = [add(f'Goal {n}.') for n in range(NOTE_CAPS['goals'] + 2)]
    if failure == 'error':
        actions.append(add(secret, 'people'))
    result = take(store, actions)
    failed = 1 if failure == 'error' else NOTE_CAPS['goals']
    assert result.not_written_reasons == {'limit': 2, 'write_failed:PastError': failed}
    assert result.not_written == 2 + failed == sum(result.not_written_reasons.values())
    assert secret not in json.dumps(asdict(result))
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err + caplog.text


def test_pilot_crisis_filter_and_unstored_report_and_legacy_defaults(policy, tmp_path):
    person = pilot.load_personas(ROOT / 'fixtures/personas.json')[0].model_copy(deep=True)
    person.sessions[0].opener = '[checker:crisis-self] [note:goals:pilot-crisis-secret]'
    path = crisis.load_path()
    model = RecordingNotes()
    memory = FakeUserMemory()
    memory.wait = lambda *args, **kwargs: False
    store = NoteStore(memory, tmp_path / 'memory')
    clients = pilot.Clients(helper=chat.fake_helper(), checker=chat.fake_checker(path),
        user=FakeClient(['[note:goals:An ordinary goal.]']), path=path)
    result = pilot.run_session(person, 1, 'notes_on', clients=clients, policy=policy, store=store,
        note_llm=model, root=tmp_path / 'sessions', turns=2, start=TODAY, fake=True)
    assert not result.failed and len(result.record.turns) == 4
    assert 'pilot-crisis-secret' not in model.seen[0] and 'An ordinary goal.' in model.seen[0]
    assert result.changes.not_written == 1
    assert result.changes.not_written_reasons == {'write_failed:PastError': 1}
    entry = pilot.session_dict(result, pilot.SessionScore('unscored', None, 'offline'))
    assert entry['changes']['not_written'] == 1
    assert entry['changes']['not_written_reasons'] == {'write_failed:PastError': 1}
    off = deepcopy(entry)
    off.update(arm='notes_off')
    off['changes'].update(not_written=2, not_written_reasons={'consent': 2})
    data = dict(fake=True, policy_sentence=pilot.POLICY_SENTENCE, policy_version=policy.version,
                policy_path='offline-policy', turns=2, judge_version='offline',
                limits_sentence=pilot.LIMITS_SENTENCE, personas=[dict(persona_id=person.persona_id,
                description='offline', sessions=dict(notes_on=[entry], notes_off=[off]))])
    report = pilot.render_report(data)
    assert '| 1 | - | (1 proposed notes not stored: write_failed:PastError x1) | not stored |' in report
    assert '| Notes proposed but not stored | 1 | 2 | |' in report
    for item in (entry, off):
        del item['changes']['not_written'], item['changes']['not_written_reasons']
    legacy = pilot.render_report(data)
    assert '| Notes proposed but not stored | 0 | 0 | |' in legacy and '| not stored |' not in legacy
