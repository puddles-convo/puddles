"""Exa web search for the researcher (https://exa.ai/docs/reference/search).

When a meter is active (`puddles.usage.metering`), every request writes a `sent` and a `done`
line to the usage log, at the configured price per search. The second try of a failed request
is its own request, with its own two lines.
"""
import os

import httpx

from puddles import usage

EXA_URL = "https://api.exa.ai/search"


def search(query: str, num_results: int = 5, max_characters: int = 3000) -> list[dict]:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise RuntimeError("EXA_API_KEY is not set")
    body = {"query": query, "type": "auto", "numResults": num_results,
            "contents": {"text": {"maxCharacters": max_characters}}}
    meter, agent = usage.current()
    last_error: Exception | None = None
    for _ in range(2):  # retried once, as the researcher doc says
        call = meter.search_sent(agent or "researcher") if meter is not None else None
        try:
            resp = httpx.post(EXA_URL, json=body, headers={"x-api-key": key}, timeout=60)
            resp.raise_for_status()
            results = [{"url": r.get("url", ""), "title": r.get("title") or "", "text": r.get("text") or ""}
                       for r in resp.json().get("results", [])]
        except httpx.HTTPError as e:
            last_error = e
            if meter is not None and call is not None:
                meter.search_done(call, error=f"{type(e).__name__}: {e}")
            continue
        if meter is not None and call is not None:
            meter.search_done(call)
        return results
    raise RuntimeError(f"Exa search failed: {last_error}")
