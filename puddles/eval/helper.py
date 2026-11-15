# ruff: noqa: E501
"""A stand-in for the helper until A1 (#9) lands. Final evaluation must use A1's prompt-building code, so training and evaluation see the same prompt."""
import os
from typing import Optional

from schemas import Turn


class OpenAICompatibleHelper:
    def __init__(self, policy_text: str, model: str, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, temperature: Optional[float] = 0.0,
                 max_tokens: int = 120, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        self.policy_text = policy_text
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client

    @classmethod
    def from_env(cls, policy_text: str) -> "OpenAICompatibleHelper":
        model = os.environ.get("PUDDLES_HELPER_MODEL")
        if not model:
            raise ValueError("PUDDLES_HELPER_MODEL is not set")
        base_url = os.environ.get("PUDDLES_HELPER_BASE_URL")
        api_key = os.environ.get("PUDDLES_HELPER_API_KEY")
        if api_key is None and base_url is not None:
            api_key = "none"
        temperature = os.environ.get("PUDDLES_HELPER_TEMPERATURE", "0").strip()
        return cls(policy_text, model, base_url, api_key,
                   None if temperature.lower() == "none" else float(temperature))

    def __call__(self, turns: list[Turn]) -> str:
        messages = [{"role": "system", "content": self.policy_text}]
        messages += [{"role": "user" if turn.speaker == "user" else "assistant", "content": turn.text}
                     for turn in turns]
        extra = {"temperature": self.temperature} if self.temperature is not None else {}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **extra,
        )
        return (resp.choices[0].message.content or "").strip()
