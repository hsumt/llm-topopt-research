import hashlib
import json
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from project.ai_errors import ModelCallError, classify_provider_error

from project.models import (
    ProblemSpec,
    Resolution,
    Review,
    Revision,
    Usage,
)
from project.prompts import (
    PARSE_PROMPT,
    RESOLVE_PROMPT,
    REVIEW_PROMPT,
)


T = TypeVar("T", bound=BaseModel)
MODEL = os.getenv("FORMULATION_MODEL", "claude-sonnet-4-6")
USE_CACHE = os.getenv("PRESOLVE_CACHE", "1") == "1"
CACHE_DIR = Path("artifacts/cache/simple_presolve")

# _ Underscores mean that these functions are internal helpers for others. Its a naming convention, please follow.

def _client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ModelCallError(
            "missing_api_key", "No Anthropic API key was supplied to the app runtime.",
            "If ANTHROPIC_API_KEY is already in the repository's .env, run "
            "./docker/compose up -d --force-recreate to load it, then retry. "
            "The wrapper also accepts an exported shell variable, which takes precedence. "
            "Do not enter a key in the design brief.",
            "not_sent",
        )
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ModelCallError(
            "sdk_unavailable", "The Anthropic client dependency is unavailable in this runtime.",
            "Rebuild the Docker image with the project's pinned requirements, then retry.",
            "not_sent",
        ) from None
    try:
        # Explicitly bounded waits; retry is a visible user action, not a hidden
        # sequence of paid calls that can hold the UI for minutes.
        return Anthropic(api_key=api_key, timeout=45.0, max_retries=0)
    except Exception:
        raise ModelCallError(
            "client_configuration_failed", "The Anthropic client could not be initialized.",
            "Check the container's API and network configuration, then retry.",
            "not_sent",
        ) from None

def _extract_json(text: str) -> dict:
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("AI response did not contain a JSON object")

    return json.loads(text[start : end + 1])

def _cache_key( # This is for caching. I've simplified cache.py into this.
    step: str,
    prompt: str,
    payload: dict,
) -> str:
    request = {
        "step": step,
        "model": MODEL,
        "prompt": prompt,
        "payload": payload,
    }

    text = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> dict | None:
    path = _cache_path(key)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("response_text"), str):
        return None
    return data


def _write_cache(
    key: str,
    response_text: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "response_text": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    _cache_path(key).write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def _validate_response(response_text: str, output_type: type[T]) -> T:
    try:
        data = _extract_json(response_text)
    except (ValueError, TypeError):
        raise ModelCallError(
            "invalid_json", "Claude responded, but its answer was not usable JSON.",
            "Retry the AI step. Your existing request and specification have been kept.",
            "response_received", True,
        ) from None
    try:
        return output_type.model_validate(data)
    except ValidationError:
        raise ModelCallError(
            "invalid_schema", "Claude responded, but its answer did not match the required specification format.",
            "Retry the AI step. If this persists, check model and schema compatibility; "
            "your existing request and specification have been kept.",
            "response_received", True,
        ) from None

def _call_model(
    *,
    step: str,
    prompt: str,
    payload: dict,
    output_type: type[T],
    max_tokens: int,
    bypass_cache: bool = False,
) -> tuple[T, Usage]:
    key = _cache_key(step, prompt, payload)
    cached = _read_cache(key) if USE_CACHE and not bypass_cache else None

    if cached is not None:
        try:
            result = _validate_response(cached["response_text"], output_type)
        except ModelCallError:
            # A damaged or obsolete cache is not evidence that the live API failed.
            pass
        else:
            return result, Usage(step=step, model=MODEL, cache_hit=True)

    client = _client()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=prompt,
            messages=[
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"))}
            ],
        )
    except Exception as error:
        raise classify_provider_error(error) from None

    if getattr(response, "stop_reason", None) == "max_tokens":
        raise ModelCallError(
            "truncated_response", "Claude's response reached its output limit before it finished.",
            "Retry the AI step. If it repeats, increase this stage's output allowance "
            "or shorten the request while retaining the engineering requirements.",
            "response_received", True,
        )

    response_text = "".join(
        block.text for block in (getattr(response, "content", None) or [])
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    )
    if not response_text.strip():
        raise ModelCallError(
            "empty_response", "Claude returned no text that could be read as a specification.",
            "Retry the AI step. Your existing request and specification have been kept.",
            "response_received", True,
        )

    result = _validate_response(response_text, output_type)

    input_tokens = int(response.usage.input_tokens)
    output_tokens = int(response.usage.output_tokens)

    if USE_CACHE:
        try:
            _write_cache(key, response_text, input_tokens, output_tokens)
        except OSError:
            # The optional cache must never discard a valid, paid-for response.
            pass

    return result, Usage(
        step=step,
        model=MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

def parse_problem(
    request: str,
    context: str | None = None,
) -> tuple[ProblemSpec, Usage]:
    payload = {
        "request": request,
        "context": context,
        "output_schema": ProblemSpec.model_json_schema(),
    }

    return _call_model(
        step="parse",
        prompt=PARSE_PROMPT,
        payload=payload,
        output_type=ProblemSpec,
        max_tokens=2500,
    )


def review_problem(
    spec: ProblemSpec,
    *,
    original_request: str | None = None,
    context: str | None = None,
    revisions: list[Revision] | None = None,
) -> tuple[Review, Usage]:
    payload = {
        "spec": spec.model_dump(exclude_none=True),
        "original_request": original_request,
        "context": context,
        "prior_rounds": [revision.model_dump() for revision in (revisions or [])],
        "output_schema": Review.model_json_schema(),
    }

    return _call_model(
        step="review",
        prompt=REVIEW_PROMPT,
        payload=payload,
        output_type=Review,
        max_tokens=1500,
    )

def resolve_answers(
    spec: ProblemSpec,
    review: Review,
    answers: dict[str, str],
    *,
    original_request: str | None = None,
    context: str | None = None,
    revisions: list[Revision] | None = None,
    bypass_cache: bool = False,
) -> tuple[Resolution, Usage]:
    """Interpret answers; bypass a rejected cached patch when explicitly retrying.

    A refreshed, schema-valid result replaces the prior cache entry. Applying
    its edits is separately checked by the workflow before accepting the spec.
    """
    payload = {
        "spec": spec.model_dump(exclude_none=True),
        "issues": [
            issue.model_dump()
            for issue in review.issues
            if issue.blocking
        ],
        "questions": [
            question.model_dump()
            for question in review.questions
        ],
        "answers": answers,
        "original_request": original_request,
        "context": context,
        "prior_rounds": [revision.model_dump() for revision in (revisions or [])],
        "output_schema": Resolution.model_json_schema(),
        "output_spec_schema": ProblemSpec.model_json_schema(),
    }

    return _call_model(
        step="resolve",
        prompt=RESOLVE_PROMPT,
        payload=payload,
        output_type=Resolution,
        max_tokens=3000,
        bypass_cache=bypass_cache,
    )
