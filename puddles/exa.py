"""Exa web search for the researcher (https://exa.ai/docs/reference/search)."""
import os

import httpx

EXA_URL = "https://api.exa.ai/search"


def search(query: str, num_results: int = 5, max_characters: int = 3000) -> list[dict]:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise RuntimeError("EXA_API_KEY is not set")
    body = {"query": query, "type": "auto", "numResults": num_results,
            "contents": {"text": {"maxCharacters": max_characters}}}
    last_error: Exception | None = None
    for _ in range(2):  # retried once, as the researcher doc says
        try:
            resp = httpx.post(EXA_URL, json=body, headers={"x-api-key": key}, timeout=60)
            resp.raise_for_status()
            return [{"url": r.get("url", ""), "title": r.get("title") or "", "text": r.get("text") or ""}
                    for r in resp.json().get("results", [])]
        except httpx.HTTPError as e:
            last_error = e
    raise RuntimeError(f"Exa search failed: {last_error}")
