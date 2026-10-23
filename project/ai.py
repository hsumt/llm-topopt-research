from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from project import trace
from project.ai_errors import ModelCallError, classify_provider_error

from project.models import (
    Clarification,
    ProblemSpec,
    Resolution,
    Review,
    Revision,
    Usage,
)
from project.prompts import (
    CLARIFY_PROMPT,
    PARSE_PROMPT,
    RESOLVE_PROMPT,
    REVIEW_PROMPT,
)


T = TypeVar("T", bound=BaseModel)
PROVIDERS = {  # FORMULATION_PROVIDER value -> (display name, API key variable, model variable)
    "anthropic": ("Anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"),
    "azure_openai": ("Azure OpenAI", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"),
}
PROVIDER = os.getenv("FORMULATION_PROVIDER") or "anthropic"
PROVIDER_NAME, KEY_VAR, MODEL_VAR = PROVIDERS.get(PROVIDER, (PROVIDER, "", ""))
# No default model: the provider's model variable must name it (on Azure OpenAI, the deployment name).
MODEL = (os.getenv(MODEL_VAR) if MODEL_VAR else None) or ""
# Azure OpenAI only: reasoning effort (low, medium, high). Unset sends none, so the deployment's default applies.
REASONING_EFFORT = os.getenv("AZURE_OPENAI_REASONING_EFFORT") or None
CACHE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "cache" / "formulation"


def _client():
    if PROVIDER not in PROVIDERS:
        raise ModelCallError(
            "unknown_provider", f"FORMULATION_PROVIDER is set to {PROVIDER!r}, which the app does not support.",
            "Set FORMULATION_PROVIDER to anthropic or azure_openai in the repository's .env, run "
            "./docker/compose up -d --force-recreate, then retry.",
            "not_sent",
        )
    if not MODEL:
        where = " to the deployment name shown in the Foundry portal" if PROVIDER == "azure_openai" else ""
        raise ModelCallError(
            "missing_model", f"No {PROVIDER_NAME} model was supplied to the app runtime.",
            f"Set {MODEL_VAR} in the repository's .env{where}, run "
            "./docker/compose up -d --force-recreate, then retry.",
            "not_sent",
        )
    return _anthropic_client() if PROVIDER == "anthropic" else _azure_openai_client()


def _anthropic_client():
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


def _azure_openai_client():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        raise ModelCallError(
            "missing_api_key", "No Azure OpenAI API key or endpoint was supplied to the app runtime.",
            "If AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are already in the repository's .env, run "
            "./docker/compose up -d --force-recreate to load them, then retry. "
            "The wrapper also accepts exported shell variables, which take precedence. "
            "Do not enter a key in the design brief.",
            "not_sent",
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise ModelCallError(
            "sdk_unavailable", "The OpenAI client dependency is unavailable in this runtime.",
            "Rebuild the Docker image with the project's pinned requirements, then retry.",
            "not_sent",
        ) from None
    # Azure's v1 API takes the standard OpenAI client at <endpoint>/openai/v1/.
    base_url = endpoint.rstrip("/")
    if not base_url.endswith("/openai/v1"):
        base_url += "/openai/v1"
    try:
        # The same bounded waits as the Anthropic client.
        return OpenAI(api_key=api_key, base_url=base_url + "/", timeout=45.0, max_retries=0)
    except Exception:
        raise ModelCallError(
            "client_configuration_failed", "The Azure OpenAI client could not be initialized.",
            "Check AZURE_OPENAI_ENDPOINT and the container's network configuration, then retry.",
            "not_sent",
        ) from None


def _send_anthropic(client, prompt: str, payload: dict, max_tokens: int):
    """One Messages API call -> (reply text, stop reason, input tokens, output tokens)."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=prompt,
        messages=[
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"))}
        ],
    )
    usage = getattr(response, "usage", None)
    text = "".join(
        block.text for block in (getattr(response, "content", None) or [])
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    )
    return (text, getattr(response, "stop_reason", None),
            getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))


def _send_azure_openai(client, prompt: str, payload: dict, max_tokens: int):
    """One Responses API call -> (reply text, stop reason, input tokens, output tokens).

    The stop reason is the incomplete reason (max_output_tokens, content_filter) when the
    response stopped early, otherwise its status. Reasoning tokens count against max_tokens.
    """
    options = {"reasoning": {"effort": REASONING_EFFORT}} if REASONING_EFFORT else {}
    response = client.responses.create(
        model=MODEL,
        instructions=prompt,
        input=json.dumps(payload, separators=(",", ":")),
        max_output_tokens=max_tokens,
        store=False,  # the formulation trace is the record; Azure keeps no retrievable copy
        **options,
    )
    usage = getattr(response, "usage", None)
    details = getattr(response, "incomplete_details", None)
    text = getattr(response, "output_text", None)
    return (text if isinstance(text, str) else "",
            getattr(details, "reason", None) or getattr(response, "status", None),
            getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))


def _extract_json(text: str) -> dict:
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("AI response did not contain a JSON object")

    return json.loads(text[start : end + 1])

def _cache_key(
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
            "invalid_json", "The model responded, but its answer was not usable JSON.",
            "Retry the AI step. Your existing request and specification have been kept.",
            "response_received", True,
        ) from None
    try:
        return output_type.model_validate(data)
    except ValidationError:
        raise ModelCallError(
            "invalid_schema", "The model responded, but its answer did not match the required specification format.",
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
    # The call record goes to the formulation trace whether the call succeeds or fails;
    # it holds the reply text but never exception text.
    key = _cache_key(step, prompt, payload)
    call = {"step": step, "provider": PROVIDER, "model": MODEL, "cache_key": key, "cache_hit": False,
            "payload": payload, "reply": None, "stop_reason": None, "tokens": None}
    started = time.monotonic()
    trace.calling(step)
    try:
        return _call_model_once(call, key=key, step=step, prompt=prompt, payload=payload,
                                output_type=output_type, max_tokens=max_tokens, bypass_cache=bypass_cache)
    finally:
        call["latency_ms"] = round((time.monotonic() - started) * 1000)
        trace.record_call(call)
        trace.called()


_scripted: ContextVar = ContextVar("ato_scripted_model", default=None)


@contextmanager
def scripted(responder):
    """Answer model calls on this thread with responder(step, payload) -> dict.

    For tests and offline runs: no API call is made and the response cache is not used.
    """
    token = _scripted.set(responder)
    try:
        yield
    finally:
        _scripted.reset(token)


def _call_model_once(call: dict, *, key: str, step: str, prompt: str, payload: dict,
                     output_type: type[T], max_tokens: int, bypass_cache: bool) -> tuple[T, Usage]:
    responder = _scripted.get()
    if responder is not None:
        text = json.dumps(responder(step, payload))
        call.update(provider="scripted", model="scripted", reply=text, stop_reason="scripted",
                    tokens={"in": 0, "out": 0})
        return _validate_response(text, output_type), Usage(step=step, model="scripted")

    cached = None if bypass_cache else _read_cache(key)

    if cached is not None:
        try:
            result = _validate_response(cached["response_text"], output_type)
        except ModelCallError:
            # A damaged or obsolete cache is not evidence that the live API failed.
            pass
        else:
            call.update(cache_hit=True, reply=cached["response_text"])
            return result, Usage(step=step, model=MODEL, cache_hit=True)

    client = _client()
    send = _send_anthropic if PROVIDER == "anthropic" else _send_azure_openai
    try:
        response_text, stop_reason, input_tokens, output_tokens = send(client, prompt, payload, max_tokens)
    except Exception as error:
        raise classify_provider_error(error, provider=PROVIDER_NAME, key_var=KEY_VAR, model_var=MODEL_VAR) from None

    call.update(stop_reason=stop_reason, tokens={"in": input_tokens, "out": output_tokens})
    # The trace keeps the reply even when it is partial or empty; the errors below are raised either way.
    call["reply"] = response_text

    if stop_reason in ("max_tokens", "max_output_tokens"):
        raise ModelCallError(
            "truncated_response", "The model's response reached its output limit before it finished.",
            "Retry the AI step. If it repeats, increase this stage's output allowance "
            "or shorten the request while retaining the engineering requirements.",
            "response_received", True,
        )
    if stop_reason == "content_filter":
        raise ModelCallError(
            "content_filtered", f"{PROVIDER_NAME}'s content filter stopped the response before it finished.",
            "Retry the AI step. If it repeats, check the brief for wording a content filter could misread; "
            "this is not a missing engineering decision.",
            "response_received", True,
        )
    if not response_text.strip():
        raise ModelCallError(
            "empty_response", "The model returned no text that could be read as a specification.",
            "Retry the AI step. Your existing request and specification have been kept.",
            "response_received", True,
        )

    result = _validate_response(response_text, output_type)

    input_tokens = int(input_tokens)
    output_tokens = int(output_tokens)

    try:
        _write_cache(key, response_text, input_tokens, output_tokens)
    except OSError:
        # The cache must never discard a valid, paid-for response.
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
    code_issues: list[dict],
    free_question_slots: int,
) -> tuple[Review, Usage]:
    payload = {
        "spec": spec.model_dump(exclude_none=True),
        "original_request": original_request,
        "context": context,
        "prior_rounds": [revision.model_dump() for revision in (revisions or [])],
        "code_issues": code_issues,
        "free_question_slots": free_question_slots,
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
    clarifications: list[dict] | None = None,
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
        "clarifications": clarifications or [],
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


def clarify_question(
    question: dict,
    issues: list[dict],
    spec: ProblemSpec,
    *,
    mode: str,
    query: str = "",
    original_request: str | None = None,
    context: str | None = None,
) -> tuple[Clarification, Usage]:
    """Explain or reword one shown question at the engineer's request. The reply is explanation only:
    code stores and shows it, and never applies anything from it to the spec."""
    payload = {
        "question": question,
        "issues": issues,
        "spec": spec.model_dump(exclude_none=True),
        "original_request": original_request,
        "context": context,
        "mode": mode,
        "query": query,
        "output_schema": Clarification.model_json_schema(),
    }
    return _call_model(
        step="clarify",
        prompt=CLARIFY_PROMPT,
        payload=payload,
        output_type=Clarification,
        max_tokens=1200,
        bypass_cache=True,   # each request the engineer makes gets a fresh reply; none is replayed into the spec
    )
