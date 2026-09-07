"""Interpreter for one formulation-clarification answer.

A small deterministic semantic layer handles unambiguous phrases before an LLM
is called. The LLM resolver still has no authority to apply changes: it only
converts a user's natural-language answer into a tiny structured proposal.
``project.formulation.patching`` performs the deterministic authorization and
Pydantic validation.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()


class FieldUpdate(BaseModel):
    field_path: str
    new_value: Any


class FormulationResolution(BaseModel):
    status: Literal["update", "confirm_current", "ask_again"]
    updates: list[FieldUpdate] = Field(default_factory=list)
    confirmed_fields: list[str] = Field(default_factory=list)
    interpretation: str
    follow_up_question: str | None = None

    @model_validator(mode="after")
    def validate_shape(self):
        if self.status == "update" and not self.updates:
            raise ValueError("update requires at least one field update")
        if self.status == "ask_again":
            if self.updates or self.confirmed_fields:
                raise ValueError("ask_again cannot carry updates/confirmations")
            if not (self.follow_up_question and self.follow_up_question.strip()):
                raise ValueError("ask_again requires a follow_up_question")
        if self.status != "ask_again" and self.follow_up_question is not None:
            raise ValueError("follow_up_question is only valid for ask_again")
        return self


SYSTEM_PROMPT = r"""
You interpret ONE user's answer to ONE topology-optimization formulation
clarification question.

You receive:
- current_spec,
- issue,
- allowed_fields,
- user_answer,
- recent dialogue history.

YOUR AUTHORITY IS STRICTLY LIMITED
1. You may propose changes ONLY to allowed_fields.
2. Never change mesh.nx, mesh.ny, simp.penal, simp.r_min, simp.max_iter, or
   simp.tol_change. Those are outside formulation-dialogue authority.
3. Do not add/remove loads, BCs, or list entries. This V1 resolver can only
   change existing canonical leaf fields.
4. If the user explicitly accepts the current interpretation, return
   status="confirm_current" and list the relevant allowed fields in
   confirmed_fields. Do not emit fake updates whose values are unchanged.
5. If the answer clearly specifies a different value for one or more allowed
   fields, return status="update" with only those changes.
6. If the answer is still ambiguous, especially about load semantics, return
   status="ask_again" with ONE short follow-up question. Do not guess.
7. "Total force", "total resultant", "net force", or equivalent wording means
   edge_resultant when the user also says the load is distributed over an edge.
   "Per unit length", "line load", or "traction" means edge_traction. Do not
   ask again when the user's wording already settles this distinction.
8. Preserve the user's intent; never improve or optimize the formulation on
   your own.

CURRENT SCHEMA VOCABULARY WHEN RELEVANT
Location values:
left_edge, right_edge, top_edge, bottom_edge,
top_left, top_right, bottom_left, bottom_right,
right_tip, right_center, top_center, bottom_center, left_center

DOF values: x, y
Load kind values: point_force, edge_resultant, edge_traction
Boundary-condition values are homogeneous only: 0.0

Return EXACTLY one JSON object:
{
  "status": "update | confirm_current | ask_again",
  "updates": [
    {"field_path": "allowed.path", "new_value": <value>}
  ],
  "confirmed_fields": ["allowed.path"],
  "interpretation": "short statement of what the user's answer means",
  "follow_up_question": null
}

For ask_again, updates and confirmed_fields must be empty and
follow_up_question must be a concrete single question.
"""


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
        raise ValueError("Formulation resolver returned no complete JSON object")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Formulation resolver returned invalid JSON: {candidate}"
        ) from exc


def _load_index_from_allowed_fields(allowed_fields: list[str]) -> int | None:
    indices = set()
    for path in allowed_fields:
        match = re.match(r"loads\[(\d+)\]\.", path)
        if match:
            indices.add(int(match.group(1)))
    if len(indices) == 1:
        return next(iter(indices))
    return None


def _deterministic_resolution_from_answer(
    *,
    spec,
    issue: dict,
    user_answer: str,
) -> dict | None:
    """Resolve a few unambiguous phrases without spending an LLM call.

    This is deliberately narrow. It only handles load-idealization wording
    whose semantics are explicit in the project's schema vocabulary. If the
    text does not cleanly map, return ``None`` and let the LLM resolver handle
    it under the same deterministic patch gate.
    """

    if issue.get("category") != "load_idealization":
        return None

    allowed = list(issue.get("affected_fields", []))
    index = _load_index_from_allowed_fields(allowed)
    if index is None or index >= len(spec.loads):
        return None

    text = re.sub(r"\s+", " ", user_answer.lower().replace("–", "-").replace("—", "-"))
    current = spec.loads[index]

    # Detect edge location only when the user names a complete edge. "near the
    # right side" alone is intentionally not enough.
    edge_location = None
    edge_phrases = {
        "right_edge": ("right edge", "entire right side", "whole right side", "full right side"),
        "left_edge": ("left edge", "entire left side", "whole left side", "full left side"),
        "top_edge": ("top edge", "entire top side", "whole top side", "full top side"),
        "bottom_edge": ("bottom edge", "entire bottom side", "whole bottom side", "full bottom side"),
    }
    for location, phrases in edge_phrases.items():
        if any(phrase in text for phrase in phrases):
            edge_location = location
            break

    # Accept ordinary modifiers between "total" and the quantity noun, e.g.
    # "total downward force". The earlier exact-substring check missed this
    # common phrasing and caused an unnecessary follow-up question.
    is_total = bool(
        re.search(
            r"\btotal(?:\s+[a-z]+){0,3}\s+(?:force|load|resultant)\b",
            text,
        )
    ) or any(
        phrase in text
        for phrase in (
            "resultant force",
            "net force",
            "overall force",
        )
    )
    is_per_length = any(
        phrase in text
        for phrase in (
            "per unit length",
            "force per length",
            "force per unit",
            "line load",
            "edge traction",
            "traction",
        )
    )
    is_point = any(
        phrase in text
        for phrase in ("point force", "concentrated force", "single point")
    )

    point_location = None
    point_phrases = {
        "right_center": ("right center", "right-centre", "right-center", "midpoint of the right edge", "middle of the right edge"),
        "top_right": ("top right", "top-right", "upper right", "upper-right"),
        "bottom_right": ("bottom right", "bottom-right", "lower right", "lower-right"),
        "top_left": ("top left", "top-left", "upper left", "upper-left"),
        "bottom_left": ("bottom left", "bottom-left", "lower left", "lower-left"),
        "top_center": ("top center", "top-center", "midpoint of the top edge"),
        "bottom_center": ("bottom center", "bottom-center", "midpoint of the bottom edge"),
        "left_center": ("left center", "left-center", "midpoint of the left edge"),
    }
    for location, phrases in point_phrases.items():
        if any(phrase in text for phrase in phrases):
            point_location = location
            break

    proposed: dict[str, object] = {}
    location_path = f"loads[{index}].location"
    kind_path = f"loads[{index}].kind"

    if edge_location is not None and is_total:
        if location_path in allowed:
            proposed[location_path] = edge_location
        if kind_path in allowed:
            proposed[kind_path] = "edge_resultant"
        interpretation = (
            f"The user specified a total resultant distributed over the "
            f"{edge_location.replace('_', ' ')}."
        )
    elif edge_location is not None and is_per_length:
        if location_path in allowed:
            proposed[location_path] = edge_location
        if kind_path in allowed:
            proposed[kind_path] = "edge_traction"
        interpretation = (
            f"The user specified a per-unit-length traction on the "
            f"{edge_location.replace('_', ' ')}."
        )
    elif point_location is not None and is_point:
        if location_path in allowed:
            proposed[location_path] = point_location
        if kind_path in allowed:
            proposed[kind_path] = "point_force"
        interpretation = f"The user specified a point force at {point_location}."
    else:
        return None

    if not proposed:
        return None

    changed = []
    confirmed = []
    for path, value in proposed.items():
        current_value = current.location if path.endswith(".location") else current.kind
        if current_value == value:
            confirmed.append(path)
        else:
            changed.append({"field_path": path, "new_value": value})

    if changed:
        # Fields already matching the answer are intentionally not included in
        # confirmed_fields here: apply_resolution forbids touching one field in
        # both update and confirmation sets, and the changed fields carry the
        # user clarification audit needed to close the issue.
        return {
            "status": "update",
            "updates": changed,
            "confirmed_fields": [],
            "interpretation": interpretation,
            "follow_up_question": None,
        }

    return {
        "status": "confirm_current",
        "updates": [],
        "confirmed_fields": confirmed or list(proposed),
        "interpretation": interpretation,
        "follow_up_question": None,
    }


def resolve_formulation_answer(
    *,
    spec,
    issue: dict,
    user_answer: str,
    dialogue_history: list[dict] | None = None,
) -> tuple[dict, dict]:
    """Interpret one human clarification without applying it."""

    deterministic = _deterministic_resolution_from_answer(
        spec=spec,
        issue=issue,
        user_answer=user_answer,
    )
    if deterministic is not None:
        return deterministic, {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "resolution_mode": "deterministic_phrase_map",
        }

    allowed_fields = list(issue.get("affected_fields", []))
    payload = {
        "current_spec": spec.model_dump(),
        "issue": issue,
        "allowed_fields": allowed_fields,
        "user_answer": user_answer,
        "recent_dialogue_history": list(dialogue_history or [])[-6:],
    }

    response = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=900,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
    )
    if response.stop_reason == "max_tokens":
        raise ValueError("Formulation resolver response was truncated at max_tokens")
    if not response.content:
        raise ValueError("Formulation resolver returned no content")

    data = _extract_json_object(response.content[0].text)
    resolution = FormulationResolution.model_validate(data)

    touched = {
        update.field_path for update in resolution.updates
    } | set(resolution.confirmed_fields)
    unauthorized = sorted(touched - set(allowed_fields))
    if unauthorized:
        raise ValueError(
            "Resolver proposed fields outside allowed_fields: "
            + ", ".join(unauthorized)
        )

    usage = {
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "total_tokens": int(
            response.usage.input_tokens + response.usage.output_tokens
        ),
        "resolution_mode": "llm",
    }
    return resolution.model_dump(), usage
