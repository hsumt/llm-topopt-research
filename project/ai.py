import hashlib
import json
import os
from pathlib import Path
from typing import TypeVar

from anthropic import Anthropic
from pydantic import BaseModel

from project.models import (
    ProblemSpec,
    Resolution,
    Review,
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

def _client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    return Anthropic(api_key=api_key)

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

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


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

def _call_model(
    *,
    step: str,
    prompt: str,
    payload: dict,
    output_type: type[T],
    max_tokens: int,
) -> tuple[T, Usage]:
    key = _cache_key(step, prompt, payload)
    cached = _read_cache(key) if USE_CACHE else None

    if cached is not None:
        response_text = cached["response_text"]

        data = _extract_json(response_text)
        result = output_type.model_validate(data)

        return result, Usage(
            step=step,
            model=MODEL,
            cache_hit=True,
        )

    response = _client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=prompt,
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    separators=(",", ":"),
                ),
            }
        ],
    )

    if not response.content:
        raise ValueError("AI returned an empty response")

    response_text = response.content[0].text

    try:
        data = _extract_json(response_text)
        result = output_type.model_validate(data)
    except Exception as error:
        debug_dir = Path("artifacts/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        debug_data = {
            "step": step,
            "model": MODEL,
            "response_text": response_text,
            "error": str(error),
        }

        (debug_dir / "simple_presolve_last_failure.json").write_text(
            json.dumps(debug_data, indent=2),
            encoding="utf-8",
        )

        raise

    input_tokens = int(response.usage.input_tokens)
    output_tokens = int(response.usage.output_tokens)

    _write_cache(
        key,
        response_text,
        input_tokens,
        output_tokens,
    )

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
) -> tuple[Review, Usage]:
    payload = {
        "spec": spec.model_dump(exclude_none=True),
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
) -> tuple[Resolution, Usage]:
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
        "output_schema": Resolution.model_json_schema(),
        "output_spec_schema": ProblemSpec.model_json_schema(),
    }

    return _call_model(
        step="resolve",
        prompt=RESOLVE_PROMPT,
        payload=payload,
        output_type=Resolution,
        max_tokens=1000,
    )