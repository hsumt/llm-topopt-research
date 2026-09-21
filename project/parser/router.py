"""Problem router: classify the engineering formulation without choosing a solver."""

from __future__ import annotations

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from .schema import ProblemRoute
from project.llm.cache import load_cached_response, save_cached_response

load_dotenv()


SYSTEM_PROMPT = r"""
Classify the engineering problem described by the user.

You are routing the request to an engineering FORMULATION workflow. You are NOT
choosing a numerical solver or deciding that a particular backend can execute
it.

Return only JSON matching the supplied ProblemRoute schema.

Determine:
- task_type: simulation, optimization, inverse_problem, or unknown;
- physics_families involved;
- spatial_dimension only when stated or clearly implied;
- optimization_type when applicable;
- multiphysics=true when multiple physical fields are coupled;
- requested_methods only for methods explicitly requested by the user, such as
  SIMP, level set, FEM, finite volume, etc.;
- short evidence phrases supporting the route.

Do not invent missing formulation details. Do not infer a solver merely because
one is common for that problem class.

Examples:
"Optimize a 2-D cantilever using SIMP"
-> optimization, solid_mechanics, 2-D, topology, requested_methods=["SIMP"]

"Steady heat conduction through a plate"
-> simulation, thermal

"Airflow through a 3-D duct"
-> simulation, fluid, 3-D

"Minimize bracket mass while limiting thermally induced displacement"
-> optimization, solid_mechanics + thermal, multiphysics=true
"""


def _client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to the environment or .env file."
        )
    return Anthropic(api_key=api_key)


def _extract_json(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Router response contains no complete JSON object:\n{text}")
    return json.loads(text[start : end + 1])


def route_problem(problem: str, context: str | None = None) -> tuple[ProblemRoute, dict]:
    if not problem or not problem.strip():
        raise ValueError("Problem description cannot be empty")

    payload = {
        "problem": problem,
        "context_available": bool(context and context.strip()),
        "schema": ProblemRoute.model_json_schema(),
    }

    model = os.getenv("FORMULATION_MODEL", "claude-sonnet-4-6")
    cached = load_cached_response(
        component="router",
        model=model,
        system_prompt=SYSTEM_PROMPT,
        payload=payload,
    )
    if cached is not None:
        raw_text = cached["response_text"]
        route = ProblemRoute.model_validate(_extract_json(raw_text))
        usage = {
            "component": "router",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_hit": True,
            "cached_original_tokens": int(cached.get("input_tokens", 0)) + int(cached.get("output_tokens", 0)),
        }
        return route, usage

    response = _client().messages.create(
        model=model,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
    )

    if not response.content:
        raise ValueError("Problem router returned no content")

    raw_text = response.content[0].text
    save_cached_response(
        component="router",
        model=model,
        system_prompt=SYSTEM_PROMPT,
        payload=payload,
        response_text=raw_text,
        input_tokens=int(response.usage.input_tokens),
        output_tokens=int(response.usage.output_tokens),
        stop_reason=getattr(response, "stop_reason", None),
    )
    route = ProblemRoute.model_validate(_extract_json(raw_text))
    usage = {
        "component": "router",
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "total_tokens": int(response.usage.input_tokens + response.usage.output_tokens),
        "cache_hit": False,
    }
    return route, usage