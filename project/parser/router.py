"""Problem router: classify the engineering formulation without choosing a solver."""

from __future__ import annotations

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from .schema import ProblemRoute

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

    response = _client().messages.create(
        model=os.getenv("FORMULATION_MODEL", "claude-sonnet-4-6"),
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
    )

    if not response.content:
        raise ValueError("Problem router returned no content")

    route = ProblemRoute.model_validate(_extract_json(response.content[0].text))
    usage = {
        "component": "router",
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "total_tokens": int(response.usage.input_tokens + response.usage.output_tokens),
    }
    return route, usage