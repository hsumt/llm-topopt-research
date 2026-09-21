"""Language-model backends for hypothesis generation.

Two are provided.  ``AnthropicBackend`` is the one the repository uses when run
normally: it constructs ``anthropic.Anthropic()``, which reads ANTHROPIC_API_KEY
from the process environment.  Load the key with ``dotenv.load_dotenv()`` before
constructing it; never read the .env file directly.

``CallableBackend`` wraps any function with the same signature, so the cascade
can be driven from a host that supplies its own model access.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

DEFAULT_MODEL = os.environ.get("ATO_MODEL", "claude-sonnet-4-6")


class BackendError(RuntimeError):
    pass


class AnthropicBackend:
    """Structured-output backend over the Anthropic Messages API."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4000):
        try:
            import anthropic
        except ImportError as exc:                      # pragma: no cover
            raise BackendError(
                "the anthropic package is required for AnthropicBackend"
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise BackendError(
                "ANTHROPIC_API_KEY is not set in the environment. Call "
                "dotenv.load_dotenv() first; do not read the .env file directly."
            )
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def __call__(self, system: str, user: str, tool: dict) -> dict:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": user}],
        )
        self.usage["calls"] += 1
        self.usage["input_tokens"] += resp.usage.input_tokens
        self.usage["output_tokens"] += resp.usage.output_tokens
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise BackendError("model returned no tool_use block")


class CallableBackend:
    """Adapter around any ``fn(system, user, tool) -> dict``."""

    def __init__(self, fn: Callable[[str, str, dict], dict], name: str = "callable"):
        self._fn = fn
        self.model = name
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def __call__(self, system: str, user: str, tool: dict) -> dict:
        self.usage["calls"] += 1
        out = self._fn(system, user, tool)
        if isinstance(out, str):
            out = json.loads(out)
        if not isinstance(out, dict):
            raise BackendError(f"backend returned {type(out).__name__}, expected dict")
        return out
