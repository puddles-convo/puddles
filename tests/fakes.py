"""Scripted stand-ins for the model and Exa, so the agents run offline."""
from collections import defaultdict


class ScriptedLLM:
    """Returns queued outputs per schema; records every prompt it was given."""

    def __init__(self):
        self.queue = defaultdict(list)
        self.prompts = defaultdict(list)

    def add(self, schema, *outputs):
        self.queue[schema].extend(outputs)

    def _next(self, schema, user):
        self.prompts[schema].append(user)
        out = self.queue[schema].pop(0)
        return schema.model_validate(out) if isinstance(out, dict) else out

    def structured(self, system, user, schema):
        return self._next(schema, user)

    def structured_with_search(self, system, user, schema, search, max_searches):
        out = self._next(schema, user)
        for q in out.queries[:max_searches]:
            search(q)
        return out


class FakeSearch:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def __call__(self, query):
        self.calls.append(query)
        return self.results_by_query.get(query, [])
