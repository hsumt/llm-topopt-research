"""Interpret user clarification(s) as a structured deterministic patch proposal."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from project.formulation.context import ContextAttachment
from project.formulation.models import FormulationSession, ResolutionProposal
from project.llm.content import build_user_content
from project.llm.cache import load_cached_response, save_cached_response
from project.paths import ARTIFACT_ROOT

load_dotenv()


SYSTEM_PROMPT = r"""
You are a formulation-resolution agent.

Translate the engineer's latest clarification answers into ONE structured patch
proposal against the solver-independent ProblemSpec. Deterministic Python will
apply and validate the patch. Do not solve physics, select a backend, or fill
unrelated missing information.

RULES
-----
- Apply every clear answer in the batch.
- Blank/uncertain answers remain unresolved.
- Only change fields supported by the latest engineer answer.
- If an answer merely points to existing CAD/drawing geometry, preserve that as
  semantic reference/description where the schema allows; do not invent numeric
  coordinates from a sketch.
- Never insert nominal material constants unless the user explicitly supplied
  or approved them.
- Do not introduce SIMP/level-set/MMA/PETSc/backend settings into ProblemSpec.

PATCH FORMAT
------------
Use JSON Pointer paths, e.g. /geometry/description,
/boundary_conditions/0/kind, /optimization/constraints/-.
Allowed operations: add, replace, remove. Use /- to append to a list.

If an answer resolves a parser unresolved item, include its id in
resolved_unresolved_ids. If it resolves a contradiction, include that id in
resolved_contradiction_ids. If the engineer explicitly authorizes a context
candidate to govern the formal problem, include its id in
incorporated_context_ids.

Return JSON only:
{
  "action":"apply|no_change|cancel",
  "operations":[{"op":"add|replace|remove","path":"/json/pointer","value":null}],
  "resolved_unresolved_ids":[],
  "resolved_contradiction_ids":[],
  "incorporated_context_ids":[],
  "assistant_note":"short note"
}
"""


def _client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return Anthropic(api_key=api_key)


def _candidate_json_text(text: str) -> str:
    """Return the resolver JSON candidate, preserving a possibly truncated tail.

    Do not use ``rfind("}")`` here.  If model output is truncated inside the
    optional ``assistant_note`` string, the last closing brace belongs to a
    nested operation, and slicing to it destroys the otherwise complete outer
    object.  Keeping the tail lets the bounded decoder distinguish truncation
    from an ordinary syntax error and salvage only the optional note.
    """
    text = text.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"Formulation resolver returned no JSON object:\n{text}")

    candidate = text[start:].strip()
    if candidate.endswith("```"):
        candidate = candidate[:-3].rstrip()
    return candidate


def _looks_like_value_start(ch: str) -> bool:
    return ch in {'"', "{", "[", "-"} or ch.isdigit() or ch in {"t", "f", "n"}


def _bounded_missing_comma_repair(candidate: str, *, max_repairs: int = 8) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse resolver JSON, repairing only a narrowly defined missing-comma error.

    The resolver occasionally emits syntactically valid JSON except for a missing
    comma between adjacent object members/array elements.  We repair ONLY
    ``JSONDecodeError: Expecting ',' delimiter`` and only by inserting a comma at
    the parser-reported token boundary.  Pydantic still validates the complete
    ResolutionProposal afterward, so this cannot create arbitrary patch fields.
    """
    working = candidate
    repairs: list[dict[str, Any]] = []

    for _ in range(max_repairs + 1):
        try:
            return json.loads(working), repairs
        except json.JSONDecodeError as exc:
            if exc.msg != "Expecting ',' delimiter" or len(repairs) >= max_repairs:
                raise

            pos = int(exc.pos)
            left = working[:pos].rstrip()
            right = working[pos:].lstrip()
            if not left or not right:
                raise

            previous = left[-1]
            upcoming = right[0]
            # Safe boundary check: the previous token must plausibly have ended a
            # JSON value and the next token must plausibly start another value/key.
            plausible_end = previous in {'"', "}", "]"} or previous.isdigit() or previous in {"e", "E", "l"}
            if not plausible_end or not _looks_like_value_start(upcoming):
                raise

            # Insert exactly at the JSON decoder's reported boundary.  We do not
            # rewrite strings, keys, paths, or values.
            working = working[:pos] + "," + working[pos:]
            repairs.append(
                {
                    "kind": "insert_missing_comma",
                    "line": int(exc.lineno),
                    "column": int(exc.colno),
                    "original_position": pos,
                }
            )

    raise ValueError("Resolver JSON could not be decoded after bounded comma repair")


def _salvage_truncated_assistant_note(candidate: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Salvage output truncated only inside the optional ``assistant_note``.

    The operations and resolved-id arrays must already be complete.  We discard
    only the unfinished human-facing note and close the outer object.  No patch
    operation, path, value, or resolution id is synthesized.
    """
    marker = '\n  "assistant_note":'
    idx = candidate.rfind(marker)
    if idx < 0:
        raise ValueError("Resolver JSON is truncated before a recoverable assistant_note tail")

    prefix = candidate[:idx].rstrip()
    if prefix.endswith(","):
        prefix = prefix[:-1].rstrip()
    repaired = prefix + "\n}"
    data = json.loads(repaired)

    required = {
        "action",
        "operations",
        "resolved_unresolved_ids",
        "resolved_contradiction_ids",
        "incorporated_context_ids",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(
            "Resolver output was truncated before required proposal fields were complete: "
            + ", ".join(missing)
        )

    data.setdefault("assistant_note", "")
    return data, {
        "kind": "drop_truncated_optional_assistant_note",
        "dropped_characters": len(candidate) - idx,
    }


def decode_resolution_response(raw_text: str) -> tuple[ResolutionProposal, dict[str, Any]]:
    """Decode/validate one resolver response without making any API call."""
    candidate = _candidate_json_text(raw_text)
    syntax_repairs: list[dict[str, Any]] = []
    tail_repair: dict[str, Any] | None = None

    try:
        data, syntax_repairs = _bounded_missing_comma_repair(candidate)
    except json.JSONDecodeError as exc:
        # The current known failure mode is model truncation inside the optional
        # assistant_note.  Recover only that tail; never guess missing operations.
        if exc.msg not in {
            "Unterminated string starting at",
            "Expecting ',' delimiter",
            "Expecting value",
        }:
            raise
        data, tail_repair = _salvage_truncated_assistant_note(candidate)

    proposal = ResolutionProposal.model_validate(data)

    if proposal.action == "apply" and not proposal.operations:
        raise ValueError("Resolver returned action='apply' with no patch operations")
    if proposal.action != "apply" and proposal.operations:
        raise ValueError(f"Resolver returned operations with action={proposal.action!r}")

    repairs = list(syntax_repairs)
    if tail_repair is not None:
        repairs.append(tail_repair)

    return proposal, {
        "syntax_repaired": bool(repairs),
        "syntax_repairs": repairs,
        "truncated_optional_tail_repaired": tail_repair is not None,
    }


def _save_resolver_failure(
    *,
    error: Exception,
    raw_text: str,
    payload: dict,
    cache_hit: bool,
    input_tokens: int,
    output_tokens: int,
) -> Path | None:
    try:
        path = ARTIFACT_ROOT / "debug" / "resolver_last_failure.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "error": str(error),
                    "cache_hit": bool(cache_hit),
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "payload": payload,
                    "raw_text": raw_text,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        return path
    except Exception:
        return None


def _trim(text: str, limit: int = 1400) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _compact_resolver_payload(session: FormulationSession, user_message: str, structured_answers: dict[str, str]) -> dict:
    answered_ids = {qid for qid, value in structured_answers.items() if str(value).strip()}
    packet = session.critic_result.clarification_packet.questions
    selected_questions = [q for q in packet if not answered_ids or q.id in answered_ids]
    relevant_issue_ids = {issue for q in selected_questions for issue in q.issue_ids}
    relevant_context_ids = {cid for q in selected_questions for cid in q.related_context_ids}

    unresolved = [
        item.model_dump()
        for item in session.parser_result.unresolved_items
        if not relevant_issue_ids or item.id in relevant_issue_ids
    ]
    contradictions = [
        item.model_dump()
        for item in session.parser_result.contradictions
        if not relevant_issue_ids or item.id in relevant_issue_ids
    ]
    context_candidates = [
        item.model_dump()
        for item in session.parser_result.context_candidates
        if item.id in relevant_context_ids or item.incorporated_into_spec
    ]

    retrieved = []
    for item in session.retrieved_context[:4]:
        dumped = item.model_dump()
        dumped["text"] = _trim(dumped.get("text", ""))
        retrieved.append(dumped)

    return {
        "original_problem": session.original_problem,
        "current_spec": session.parser_result.spec.model_dump(exclude_none=True),
        "questions_being_answered": [q.model_dump() for q in selected_questions],
        "relevant_unresolved_items": unresolved,
        "relevant_contradictions": contradictions,
        "relevant_context_candidates": context_candidates,
        "retrieved_context": retrieved,
        "user_reply": user_message,
        "structured_answers": structured_answers,
        "available_visual_assets": [item.model_dump() for item in session.attachments],
    }


def resolve_user_reply(
    session: FormulationSession,
    user_message: str,
    *,
    structured_answers: dict[str, str] | None = None,
    attachments: list[ContextAttachment] | None = None,
) -> tuple[ResolutionProposal, dict]:
    structured_answers = structured_answers or {}
    payload = _compact_resolver_payload(session, user_message, structured_answers)

    model = os.getenv("FORMULATION_MODEL", "claude-sonnet-4-6")
    cached = load_cached_response(
        component="formulation_resolver",
        model=model,
        system_prompt=SYSTEM_PROMPT,
        payload=payload,
    )
    if cached is not None:
        raw_text = str(cached.get("response_text", ""))
        input_tokens = 0
        output_tokens = 0
        cache_hit = True
    else:
        response = _client().messages.create(
            model=model,
            max_tokens=int(os.getenv("RESOLVER_MAX_TOKENS", "2400")),
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    # Do not repeatedly resend full PDFs/images. The parser already
                    # interpreted them and the current packet carries context ids.
                    "content": build_user_content(payload, None, max_visual_assets=0),
                }
            ],
        )
        if not response.content:
            raise ValueError("Formulation resolver returned no content")
        raw_text = response.content[0].text
        input_tokens = int(response.usage.input_tokens)
        output_tokens = int(response.usage.output_tokens)
        cache_hit = False
        # Save raw output BEFORE local validation.  If local decoding fails, the
        # exact paid response can be replayed after a code fix for zero new tokens.
        save_cached_response(
            component="formulation_resolver",
            model=model,
            system_prompt=SYSTEM_PROMPT,
            payload=payload,
            response_text=raw_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=getattr(response, "stop_reason", None),
        )

    try:
        proposal, decode_meta = decode_resolution_response(raw_text)
    except Exception as exc:
        _save_resolver_failure(
            error=exc,
            raw_text=raw_text,
            payload=payload,
            cache_hit=cache_hit,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        raise ValueError(
            "Formulation resolver output failed local JSON/schema validation. "
            "The raw response was saved to artifacts/debug/resolver_last_failure.json "
            f"and can be replayed without another model call. Details: {exc}"
        ) from exc

    # A successful local decode invalidates any stale resolver failure snapshot.
    try:
        stale = ARTIFACT_ROOT / "debug" / "resolver_last_failure.json"
        if stale.exists():
            stale.unlink()
    except OSError:
        pass

    usage = {
        "component": "formulation_resolver",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_hit": cache_hit,
        "syntax_repaired": bool(decode_meta.get("syntax_repaired")),
        "syntax_repairs": decode_meta.get("syntax_repairs", []),
    }
    return proposal, usage
