"""Anthropic parser client with deterministic physics-sensitive defaults."""

from __future__ import annotations

import json
import os
from typing import List, Tuple

from anthropic import Anthropic
from dotenv import load_dotenv

try:
    from project.parser.prompt import SYSTEM_PROMPT
except ModuleNotFoundError:
    from prompt import SYSTEM_PROMPT
from schema import DefaultedField, ParserResult, ProblemSpec

load_dotenv()

R_MIN_ELEMENT_MULTIPLIER = 2.5
R_MIN_CONVENTION = "cone_equivalent_radius"


def compute_default_r_min(Lx: float, Ly: float, nx: int, ny: int) -> float:
    """Return a cone-equivalent physical filter radius spanning 2.5 elements.

    No absolute-unit clamp is applied. A fixed clamp such as [0.01, 0.10]
    breaks mesh and unit scaling and contradicts the intended invariant.
    """
    values = (Lx, Ly, nx, ny)
    if not all(float(v) > 0.0 for v in values):
        raise ValueError(f"Invalid geometry/mesh for r_min default: {values}")
    element_size = min(float(Lx) / int(nx), float(Ly) / int(ny))
    return R_MIN_ELEMENT_MULTIPLIER * element_size


def _client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to the environment or .env file."
        )
    return Anthropic(api_key=api_key)


def _extract_json_object(text: str) -> dict:
    """Extract and decode one top-level JSON object from model output."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"LLM response contains no complete JSON object:\n{text}")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON:\n{candidate}") from exc


def parse_problem(prompt: str) -> Tuple[ProblemSpec, List[DefaultedField], dict]:
    """Parse natural language into a validated 2-D ``ProblemSpec``.

    The LLM may identify that ``simp.r_min`` was omitted, but Python computes
    the actual default deterministically from the resolved physical geometry
    and mesh. The LLM never has final authority over this arithmetic.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Problem description cannot be empty")

    response = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2200,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    parser_tokens = {
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "total_tokens": int(
            response.usage.input_tokens + response.usage.output_tokens
        ),
    }
    print("Parser usage:")
    for key, value in parser_tokens.items():
        print(f"  {key}: {value}")

    if response.stop_reason == "max_tokens":
        raise ValueError(
            "Parser response was truncated at max_tokens. Increase max_tokens "
            "or shorten the defaulted-field questions."
        )
    if not response.content:
        raise ValueError("Parser returned no content")

    data = _extract_json_object(response.content[0].text)
    try:
        result = ParserResult.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"JSON does not match the verified ParserResult schema: {data}"
        ) from exc

    spec = result.spec
    defaulted_fields = result.defaulted_fields

    r_min_default = next(
        (f for f in defaulted_fields if f.field_path == "simp.r_min"), None
    )
    if r_min_default is not None:
        Lx = spec.mesh.Lx if spec.mesh.Lx is not None else spec.mesh.nx / spec.mesh.ny
        Ly = spec.mesh.Ly if spec.mesh.Ly is not None else 1.0
        correct = compute_default_r_min(Lx, Ly, spec.mesh.nx, spec.mesh.ny)
        payload = spec.model_dump()
        payload["simp"]["r_min"] = correct
        spec = ProblemSpec.model_validate(payload)
        r_min_default.default_used = correct
        r_min_default.question = f"What filter radius should I use? Default: {correct:.6g}"

    return spec, defaulted_fields, parser_tokens
