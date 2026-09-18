"""Interpret user clarification(s) as a structured deterministic patch proposal."""

from __future__ import annotations

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from project.formulation.context import ContextAttachment
from project.formulation.models import FormulationSession, ResolutionProposal
from project.llm.content import build_user_content

load_dotenv()


SYSTEM_PROMPT = r"""
You are a formulation-resolution agent.

You receive the current solver-independent engineering ProblemSpec, a BATCH of
clarification questions, unresolved items, contradictions, context candidates,
retrieved context, optional visual assets, and the engineer's new reply/answers.

Your job is ONLY to translate the engineer's answers into one structured patch
proposal. Deterministic Python will apply and validate the patch. You do not
run physics, choose a backend, or decide whether a numerical solve passes.

BATCH ANSWERS
-------------
The user may answer several clarification questions at once. Apply every answer
that is sufficiently clear in the same proposal. Blank/uncertain answers must
remain unresolved. Do not infer an answer to one question from an unrelated
answer to another.

PATCH FORMAT
------------
Use JSON Pointer paths into ProblemSpec, e.g.:
  /geometry/spatial_dimension
  /physics/0/model
  /boundary_conditions/0/kind
  /optimization/constraints/-

Allowed operations: add, replace, remove. Use "/-" to append to a list.

Do not patch fields that do not exist in the ProblemSpec schema. In particular,
do not introduce SIMP settings, level-set controls, solver tolerances, PETSc
settings, optimizer hyperparameters, or backend names into ProblemSpec.

USER AUTHORITY
--------------
Only apply changes supported by the user's latest message/structured answers.
Do not use the opportunity to fill unrelated missing information.

If the user explicitly authorizes a context candidate to govern the problem,
list its id in incorporated_context_ids and add/replace corresponding
ProblemSpec field(s).

If an answer resolves a parser unresolved item, list its id in
resolved_unresolved_ids. If it resolves a contradiction, list that id in
resolved_contradiction_ids.

VISUAL REFERENCES
-----------------
The user may identify a region by filename, color, label, or visual description.
Use attached drawings as evidence, but do not invent precise geometry absent
from the image/text. If an answer still does not identify the region well
enough to encode it semantically, leave it unresolved.

If no answer actually changes/resolves the formal problem, return
 action="no_change" and explain briefly in assistant_note.
If the user asks to cancel, return action="cancel".

Return only JSON matching the supplied ResolutionProposal schema.
"""


def _client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return Anthropic(api_key=api_key)


def _extract_json(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Formulation resolver returned no JSON object:\n{text}")
    return json.loads(text[start : end + 1])


def resolve_user_reply(
    session: FormulationSession,
    user_message: str,
    *,
    structured_answers: dict[str, str] | None = None,
    attachments: list[ContextAttachment] | None = None,
) -> tuple[ResolutionProposal, dict]:
    payload = {
        "original_problem": session.original_problem,
        "supplied_context": session.supplied_context,
        "route": session.route.model_dump(),
        "current_spec": session.parser_result.spec.model_dump(exclude_none=True),
        "unresolved_items": [
            item.model_dump() for item in session.parser_result.unresolved_items
        ],
        "contradictions": [
            item.model_dump() for item in session.parser_result.contradictions
        ],
        "context_candidates": [
            item.model_dump() for item in session.parser_result.context_candidates
        ],
        "retrieved_context": [item.model_dump() for item in session.retrieved_context],
        "critic": session.critic_result.model_dump(),
        "user_reply": user_message,
        "structured_answers": structured_answers or {},
        "output_schema": ResolutionProposal.model_json_schema(),
    }

    response = _client().messages.create(
        model=os.getenv("FORMULATION_MODEL", "claude-sonnet-4-6"),
        max_tokens=2600,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": build_user_content(payload, attachments, max_visual_assets=4),
            }
        ],
    )
    if not response.content:
        raise ValueError("Formulation resolver returned no content")

    proposal = ResolutionProposal.model_validate(_extract_json(response.content[0].text))

    if proposal.action == "apply" and not proposal.operations:
        raise ValueError("Resolver returned action='apply' with no patch operations")
    if proposal.action != "apply" and proposal.operations:
        raise ValueError(f"Resolver returned operations with action={proposal.action!r}")

    usage = {
        "component": "formulation_resolver",
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "total_tokens": int(response.usage.input_tokens + response.usage.output_tokens),
    }
    return proposal, usage
