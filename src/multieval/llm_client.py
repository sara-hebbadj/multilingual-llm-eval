"""The only place that talks to a language model.

`OpenRouterClient` calls OpenRouter through the OpenAI-compatible SDK.
`FakeClient` returns canned text with no network access; tests and `--dry-run`
use it. Both return the same `LLMResponse`, so the rest of the code does not
know (or care) which one it is using.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Protocol

from multieval.config import Settings
from multieval.cost import estimate_tokens

QUALITY_JUDGE_HEADING = "# Quality judge"
REDTEAM_JUDGE_HEADING = "# Red-team judge"


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # OpenRouter reports the real cost in `usage.cost`
    latency_s: float


class ChatClient(Protocol):
    def chat(
        self, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 700
    ) -> LLMResponse: ...


class OpenRouterClient:
    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Add it to Portfolio Projects/.env, "
                "or use --dry-run to test the pipeline without a key."
            )
        from openai import OpenAI  # imported here so tests never need it

        self._client = OpenAI(
            api_key=settings.api_key, base_url=settings.base_url, max_retries=3, timeout=90
        )

    def chat(
        self, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 700
    ) -> LLMResponse:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # Ask OpenRouter to include the US$ cost of the call in `usage`.
            extra_body={"usage": {"include": True}},
        )
        latency = time.perf_counter() - start
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=response.model or model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=getattr(usage, "cost", None),
            latency_s=round(latency, 3),
        )


class FakeClient:
    """Offline client. Its answers and scores are placeholders, NEVER real results.

    - Pass `replies` to return exactly those texts in order (used by tests).
    - Otherwise it invents a placeholder answer, and for judge prompts a
      deterministic pseudo-random score, so the whole pipeline can be exercised.
    """

    def __init__(self, replies: list[str] | None = None):
        self._replies = list(replies or [])
        self.calls: list[dict] = []

    def chat(
        self, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 700
    ) -> LLMResponse:
        self.calls.append({"model": model, "messages": messages})
        text = self._replies.pop(0) if self._replies else self._default_reply(model, messages)
        prompt_text = " ".join(m["content"] for m in messages)
        return LLMResponse(
            text=text,
            model=model,
            input_tokens=estimate_tokens(prompt_text),
            output_tokens=estimate_tokens(text),
            cost_usd=0.0,
            latency_s=0.0,
        )

    @staticmethod
    def _default_reply(model: str, messages: list[dict]) -> str:
        system = messages[0]["content"]
        seed = _stable_number(model + messages[-1]["content"])
        if system.startswith(QUALITY_JUDGE_HEADING):
            scores = {c: 3 + (seed >> i) % 3 for i, c in enumerate(
                ["accuracy", "policy", "tone", "language", "format"])}
            return json.dumps({**scores, "rationale": "DRY RUN: fake judge score."})
        if system.startswith(REDTEAM_JUDGE_HEADING):
            return json.dumps({"blocked": seed % 5 != 0, "harmful": False,
                               "rationale": "DRY RUN: fake judge verdict."})
        return f"[DRY RUN - fake client] Placeholder answer from {model}, only to test the pipeline."


def _stable_number(text: str) -> int:
    """Same text -> same number on every machine (Python's hash() is randomised)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
