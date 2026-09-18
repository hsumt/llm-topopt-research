"""LLM formulation critic for solver-independent engineering problem definitions."""

from __future__ import annotations

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from project.formulation.context import ContextAttachment
from project.formulation.models import (
    ClarificationPacket,
    ClarificationQuestion,
    CriticResult,
)
from project.llm.content import build_user_content
from project.parser.schema import ParserResult, ProblemRoute

load_dotenv()


SYSTEM_PROMPT = r"""
You are a pre-solve engineering formulation critic.

You review a solver-independent engineering ProblemSpec against the user's
original request and separately supplied context. Your role is to identify
missing, ambiguous, contradictory, or potentially misinterpreted ENGINEERING
FORMULATION choices before a numerical solver is selected.

You do not choose a solver and you do not tune numerics.

IN SCOPE
--------
You may review:
- intended physics and governing-model family;
- dimensionality and geometry meaning;
- material/constitutive assumptions when formulation-defining;
- boundary conditions;
- load/source idealization;
- initial conditions for transient problems;
- objective, constraints, and design variables;
- multiphysics coupling assumptions;
- manufacturing requirements when they constrain the engineering problem;
- units and dimensional consistency at the semantic level;
- whether relevant supplied context should become part of the formal problem.

OUT OF SCOPE
------------
Do not request or recommend numerical-control choices merely to make a backend
runnable. This includes SIMP penalization/filter/projection schedules,
level-set update controls, MMA/optimizer hyperparameters, PETSc/KSP choices,
convergence tolerances, iteration limits, mesh density for numerical behavior,
time-step size chosen only for numerical accuracy/stability, and backend
selection.

CONTEXT BEFORE QUESTIONS
------------------------
You receive retrieved context snippets and may receive attached drawings/images/
PDFs. Use them before asking the engineer. If supplied evidence already answers
an ambiguity, do not ask the same question again. Context-only information must
not silently become engineering intent when doing so would change the formal
problem; instead surface it and ask for confirmation in the clarification
packet.

VISUAL EVIDENCE
---------------
Use drawings and images to understand labels, regions, directions, interfaces,
obstacles, and qualitative geometry. Do not invent precise dimensions,
coordinates, or boundary conditions that are not visibly or textually stated.
If a visual is ambiguous, ask the user to identify/confirm the region or
meaning. Prefer a question that references the relevant filename/label.

CLARIFICATION PACKET POLICY
---------------------------
Do NOT ask one question per turn. Produce a bounded packet of independent
blocking questions that the engineer can answer together.

Rules:
1. Ask 2-5 questions when multiple independent blockers exist.
2. Include all high-value independent blockers that can be answered now, up to 5.
3. Do not include a question whose answer depends on another unanswered question
   in the same packet. Put only the prerequisite question in this round.
4. Do not ask about a fact already answered by the current ProblemSpec,
   provenance, retrieved context, or visual evidence.
5. Prefer concrete engineering choices over broad prompts.
6. Use single_choice when a small set of mutually exclusive interpretations is
   appropriate; otherwise use text. Use visual_reference when the user should
   identify or confirm a region/direction in an uploaded visual.
7. Each question must explain briefly why the answer changes the formulation.
8. Non-blocking curiosities should not interrupt approval.

READY STATE
-----------
Return ready_for_review only when there is no blocking contradiction, no
required unresolved formulation item, and no blocking concern that requires
human intent. In ready_for_review, clarification_packet.questions must be empty.

Do not claim the formulation is physically correct in an absolute sense. You
are checking specification adequacy relative to the provided request/context.

Return only JSON matching the supplied CriticResult schema.
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
        raise ValueError(f"Formulation critic returned no JSON object:\n{text}")
    return json.loads(text[start : end + 1])


def _category_from_path(path: str | None) -> str:
    value = (path or "").lower()
    if "geometry" in value:
        return "geometry"
    if "material" in value:
        return "material"
    if "boundary" in value:
        return "boundary_condition"
    if "source" in value or "load" in value:
        return "load_or_source"
    if "initial" in value:
        return "initial_condition"
    if "objective" in value:
        return "objective"
    if "constraint" in value:
        return "constraint"
    if "design_variable" in value:
        return "design_variable"
    if "coupling" in value:
        return "coupling"
    if "manufacturing" in value:
        return "manufacturing"
    if "unit" in value:
        return "units"
    if "physics" in value:
        return "physics_model"
    return "other"


def _known_blocker_questions(parser_result: ParserResult) -> list[ClarificationQuestion]:
    questions: list[ClarificationQuestion] = []
    for contradiction in parser_result.contradictions:
        evidence = "; ".join(contradiction.evidence) if contradiction.evidence else ""
        questions.append(
            ClarificationQuestion(
                id=f"Q_{contradiction.id}",
                prompt=(
                    f"The request contains a contradiction: {contradiction.description}. "
                    "Which interpretation should govern the formulation?"
                ),
                why_needed=(
                    f"Both interpretations cannot govern the same formal problem. {evidence}"
                ).strip(),
                issue_ids=[contradiction.id],
                category="contradiction",
            )
        )
    for item in parser_result.unresolved_items:
        if not item.required_for_execution:
            continue
        questions.append(
            ClarificationQuestion(
                id=f"Q_{item.id}",
                prompt=item.question,
                why_needed=item.issue,
                issue_ids=[item.id],
                category=_category_from_path(item.field_path),
            )
        )
    return questions


def _merge_guardrail_questions(
    result: CriticResult,
    parser_result: ParserResult,
    *,
    max_questions: int = 5,
) -> CriticResult:
    """Fail closed when parser-level blockers exist but the LLM omits them."""

    known = _known_blocker_questions(parser_result)
    blocking_concerns = [item for item in result.concerns if item.blocking]
    existing = list(result.clarification_packet.questions)
    covered_issue_ids = {
        issue_id for question in existing for issue_id in question.issue_ids
    }

    for question in known:
        if len(existing) >= max_questions:
            break
        if any(issue in covered_issue_ids for issue in question.issue_ids):
            continue
        existing.append(question)
        covered_issue_ids.update(question.issue_ids)

    if parser_result.contradictions:
        result.status = "blocked_by_contradiction"
    elif known or blocking_concerns:
        if result.status == "ready_for_review":
            result.status = "needs_clarification"

    if result.status in {"needs_clarification", "blocked_by_contradiction"} and not existing:
        if blocking_concerns:
            concern = blocking_concerns[0]
            existing.append(
                ClarificationQuestion(
                    id=f"Q_{concern.id}",
                    prompt=concern.description,
                    why_needed=concern.rationale,
                    issue_ids=[concern.id],
                    category=concern.category,
                    related_context_ids=concern.related_context_ids,
                )
            )
        else:
            existing.append(
                ClarificationQuestion(
                    id="Q_clarify",
                    prompt="What should be clarified before this formulation is approved?",
                    why_needed="The current formulation is not yet ready for approval.",
                )
            )

    if result.status == "ready_for_review":
        existing = []

    result.clarification_packet = ClarificationPacket(
        questions=existing[:max_questions]
    )
    return result


def review_formulation(
    *,
    problem: str,
    context: str | None,
    route: ProblemRoute,
    parser_result: ParserResult,
    retrieved_context: list[dict] | None = None,
    attachments: list[ContextAttachment] | None = None,
    recent_user_clarification: str | None = None,
) -> tuple[CriticResult, dict]:
    payload = {
        "original_problem": problem,
        "supplied_context": context,
        "route": route.model_dump(),
        "current_parser_result": parser_result.model_dump(),
        "retrieved_context": retrieved_context or [],
        "recent_user_clarification": recent_user_clarification,
        "attached_visual_assets": [
            {"name": a.name, "media_type": a.media_type, "size_bytes": len(a.data)}
            for a in (attachments or [])
        ],
        "output_schema": CriticResult.model_json_schema(),
    }

    response = _client().messages.create(
        model=os.getenv("FORMULATION_MODEL", "claude-sonnet-4-6"),
        max_tokens=3600,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": build_user_content(payload, attachments, max_visual_assets=4),
            }
        ],
    )
    if not response.content:
        raise ValueError("Formulation critic returned no content")

    result = CriticResult.model_validate(_extract_json(response.content[0].text))
    result = _merge_guardrail_questions(result, parser_result)

    usage = {
        "component": "formulation_critic",
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "total_tokens": int(response.usage.input_tokens + response.usage.output_tokens),
    }
    return result, usage
