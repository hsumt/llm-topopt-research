"""LLM client for the generic engineering problem parser."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from anthropic import Anthropic
from dotenv import load_dotenv

from project.formulation.context import ContextAttachment
from project.formulation.models import ContextSnippet
from project.llm.content import build_user_content
from project.paths import ARTIFACT_ROOT

from .normalize import normalize_parser_payload
from .prompt import SYSTEM_PROMPT
from .provenance import normalize_field_provenance, validate_field_provenance
from .schema import ParserResult, ProblemRoute

load_dotenv()

ProgressCallback = Callable[[str], None]


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to the environment or .env file."
        )
    return Anthropic(api_key=api_key)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Parser response contains no complete JSON object:\n{text}")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Parser returned invalid JSON:\n{candidate}") from exc


def _write_debug_snapshot(
    *,
    raw_text: str | None,
    extracted_data: dict | None,
    normalized_data: dict | None,
    route: ProblemRoute,
    error: Exception,
    input_tokens: int,
    output_tokens: int,
) -> Path | None:
    """Persist the failed parser artifact locally so debugging costs no new call."""
    try:
        debug_dir = ARTIFACT_ROOT / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / "parser_last_failure.json"
        payload = {
            "error": str(error),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "route": route.model_dump(),
            "raw_text": raw_text,
            "extracted_data": extracted_data,
            "normalized_data": normalized_data,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
    except Exception:
        return None


def parse_problem(
    problem: str,
    *,
    route: ProblemRoute,
    context: str | None = None,
    retrieved_context: list[ContextSnippet] | None = None,
    attachments: list[ContextAttachment] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[ParserResult, dict]:
    """Parse one request into a solver-independent ProblemSpec.

    Cost/safety behavior:
    - one parser call by default;
    - max-token truncation never triggers an automatic second call;
    - schema synonyms are normalized deterministically before Pydantic;
    - compact provenance scopes are expanded/bound deterministically;
    - full parser regeneration is OFF by default (PARSER_AUTO_REPAIR=0);
    - every failed response is saved locally under artifacts/debug so the next
      code fix can inspect the exact model output without spending again.
    """

    if not problem or not problem.strip():
        raise ValueError("Problem description cannot be empty")

    max_tokens = int(os.getenv("PARSER_MAX_TOKENS", "5500"))
    if max_tokens < 3000:
        raise ValueError("PARSER_MAX_TOKENS must be at least 3000")

    user_payload = {
        "problem": problem,
        "supplied_context": context.strip() if context and context.strip() else None,
        "retrieved_context": [item.model_dump() for item in (retrieved_context or [])],
        "attached_visual_assets": [
            {
                "name": item.name,
                "media_type": item.media_type,
                "size_bytes": len(item.data),
            }
            for item in (attachments or [])
        ],
        "problem_route": route.model_dump(),
    }

    parser_visuals = [
        item for item in (attachments or [])
        if item.media_type.lower().startswith("image/")
    ]

    client = _client()
    total_input = 0
    total_output = 0
    last_error: Exception | None = None
    auto_repair = os.getenv("PARSER_AUTO_REPAIR", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    max_attempts = 2 if auto_repair else 1

    for attempt in range(max_attempts):
        if attempt == 0:
            message_payload = user_payload
            _report(
                progress,
                f"Generating structured engineering specification (output budget: {max_tokens:,} tokens)...",
            )
        else:
            message_payload = {
                **user_payload,
                "repair_request": (
                    "The previous JSON failed deterministic validation. Regenerate a "
                    "compact corrected ParserResult using the exact canonical keys in "
                    "the system prompt. Do not add new engineering facts."
                ),
                "validation_error": str(last_error),
            }
            _report(progress, "Repairing parser JSON after deterministic validation failure...")

        response = client.messages.create(
            model=os.getenv("FORMULATION_MODEL", "claude-sonnet-4-6"),
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_user_content(
                        message_payload,
                        parser_visuals,
                        max_visual_assets=2,
                    ),
                }
            ],
        )

        call_input = int(response.usage.input_tokens)
        call_output = int(response.usage.output_tokens)
        total_input += call_input
        total_output += call_output

        if response.stop_reason == "max_tokens":
            raise ValueError(
                "Parser reached its output budget before finishing JSON. "
                f"This call used {call_output:,} output tokens. No automatic retry was made. "
                "Do not keep raising the budget repeatedly; inspect artifacts/debug/parser_last_failure.json "
                "if present and reduce the parser output instead."
            )

        if not response.content:
            last_error = ValueError("Parser returned no content")
            continue

        raw_text = response.content[0].text
        extracted_data: dict | None = None
        normalized_data: dict | None = None
        _report(progress, "Normalizing schema aliases deterministically (no extra LLM call)...")

        try:
            extracted_data = _extract_json_object(raw_text)
            normalized_data, schema_normalization = normalize_parser_payload(
                extracted_data,
                route,
            )
            _report(
                progress,
                "Normalized parser structure deterministically"
                + (
                    f"; lifted {schema_normalization['lifted_inline_loads']} inline load(s)."
                    if schema_normalization.get("lifted_inline_loads")
                    else "."
                ),
            )
            sanitization = schema_normalization.get("semantic_sanitization", {})
            dropped_materials = sanitization.get("dropped_unproven_material_property_sets", [])
            relaxed_supports = sanitization.get("relaxed_unproven_fixed_supports", [])
            if dropped_materials:
                _report(
                    progress,
                    f"Removed {len(dropped_materials)} ungrounded material-property set(s); "
                    "they remain unresolved instead of being guessed.",
                )
            if relaxed_supports:
                _report(
                    progress,
                    f"Relaxed {len(relaxed_supports)} unsupported fully-fixed support assumption(s) "
                    "back to unresolved interface conditions.",
                )
            if sanitization.get("dropped_unproven_requested_outputs"):
                _report(
                    progress,
                    "Removed parser-invented requested outputs that were not explicitly requested.",
                )

            _report(progress, "Validating canonical schema and semantic provenance deterministically...")
            result = ParserResult.model_validate(normalized_data)

            normalized_provenance, provenance_normalization = normalize_field_provenance(
                result.spec,
                result.field_provenance,
            )
            if provenance_normalization["dropped_extra"]:
                _report(
                    progress,
                    "Removed non-semantic provenance emitted for administrative/empty fields.",
                )
            _report(
                progress,
                "Expanded compact provenance scopes deterministically: "
                f"{provenance_normalization['accepted_scope_records']} scope(s) -> "
                f"{provenance_normalization['expanded_leaf_records']} audited leaf values.",
            )
            result = result.model_copy(
                update={"field_provenance": normalized_provenance}
            )
            validate_field_provenance(result.spec, result.field_provenance)

            usage = {
                "component": "parser",
                "attempts": attempt + 1,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "max_tokens_per_call": max_tokens,
                "schema_normalization": schema_normalization,
                "provenance_normalization": provenance_normalization,
            }
            _report(progress, "Structured specification validated.")
            return result, usage

        except Exception as exc:
            last_error = exc
            debug_path = _write_debug_snapshot(
                raw_text=raw_text,
                extracted_data=extracted_data,
                normalized_data=normalized_data,
                route=route,
                error=exc,
                input_tokens=total_input,
                output_tokens=total_output,
            )
            debug_note = (
                f" Exact failed output was saved to {debug_path}."
                if debug_path is not None
                else ""
            )
            if not auto_repair:
                raise ValueError(
                    "Parser failed deterministic validation after the first call. "
                    "No automatic second full parser call was made (PARSER_AUTO_REPAIR=0). "
                    f"This call used {total_input:,} input and {total_output:,} output tokens. "
                    f"Validation error: {last_error}.{debug_note}"
                ) from last_error

    raise ValueError(
        "Parser failed deterministic validation after the explicitly enabled repair call: "
        f"{last_error}. Total parser usage: {total_input:,} input / "
        f"{total_output:,} output tokens."
    ) from last_error
