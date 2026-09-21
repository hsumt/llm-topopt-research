"""LLM formulation critic for solver-independent engineering problem definitions."""

from __future__ import annotations

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from project.formulation.context import ContextAttachment
from project.formulation.critic_view import compact_critic_payload
from project.formulation.models import (
    ClarificationPacket,
    ClarificationQuestion,
    CriticResult,
)
from project.llm.content import build_user_content
from project.llm.cache import load_cached_response, save_cached_response
from project.parser.schema import ParserResult, ProblemRoute
from project.paths import ARTIFACT_ROOT

load_dotenv()


SYSTEM_PROMPT = r"""
You are a pre-solve engineering formulation critic.

Review a solver-independent engineering ProblemSpec against the user's original
request and retrieved user-supplied evidence. Identify only missing,
ambiguous, contradictory, or potentially misinterpreted ENGINEERING
FORMULATION choices. Do not choose a solver or tune numerics.

SCOPE
-----
Review physics/model family, geometry meaning, constitutive/material identity,
boundary conditions, loads/sources, objectives/constraints/design variables,
couplings, units, and manufacturing requirements that materially define the
problem. Do not request SIMP/level-set/MMA/PETSc settings, mesh tuning,
convergence tolerances, or backend choices.

EVIDENCE / AUTHORITY
--------------------
- Use the current spec, unresolved items, compact provenance summary, retrieved
  context, and context candidates before asking the engineer.
- Do not silently convert a context fact or model inference into user intent.
- Never offer or inject nominal engineering constants as a default. For example,
  do not propose a numeric elastic modulus merely because a material name is
  known. Ask for the governing grade/data source, or state that a downstream
  material-property lookup remains required.
- Do not treat an inferred assumption as resolved when the current unresolved
  items say the same topic is undecided.
- A physical part may be 3-D while its analysis idealization (3-D solid, shell,
  plane stress, etc.) is still unresolved. Do not conflate physical geometry
  dimension with numerical/continuum representation.

GEOMETRY REFERENCES
-------------------
Do not force engineers to type coordinates that already exist in authoritative
CAD/drawings. If geometry is referenced as "existing" but not numerically
present, prefer asking whether an uploaded/existing CAD or dimensioned drawing
should be the authoritative geometry source. Ask for coordinates only when a
numeric definition is actually needed and no authoritative geometry asset is
available.

CLARIFICATION PACKET
--------------------
Return a bounded packet of independent high-value decisions.
- Ask 2-5 questions when multiple independent blockers exist.
- One question = one engineering decision topic. Do NOT combine unrelated
  decisions (for example material properties + load-case simultaneity).
- Questions in one packet must not depend on answers to other questions in the
  same packet.
- Do not duplicate a parser unresolved item as a separate critic concern unless
  you are adding materially new information.
- Keep each prompt and rationale concise.
- Use single_choice for a small closed set, text otherwise, and visual_reference
  when the user should identify/confirm something in supplied visual evidence.
- Non-blocking curiosities should not interrupt approval.

CONCERNS
--------
Concerns should be NOVEL relative to parser unresolved_items/contradictions.
Return at most 6 concerns. If the parser already has an unresolved item for the
same field/topic, reference that issue in a clarification question rather than
creating a duplicate concern.

READY STATE
-----------
Return ready_for_review only when there is no blocking contradiction, no
required unresolved formulation item, and no novel blocking concern requiring
human intent. In ready_for_review, clarification_packet.questions must be empty.

OUTPUT
------
Return JSON only, with this compact shape:
{
  "status": "needs_clarification|ready_for_review|blocked_by_contradiction",
  "summary": "short engineer-facing summary",
  "concerns": [{
    "id":"c1", "category":"...", "severity":"low|medium|high",
    "blocking":true, "description":"...", "rationale":"...",
    "related_fields":["/path"], "related_context_ids":["context_001"]
  }],
  "clarification_packet": {"questions": [{
    "id":"q1", "prompt":"...", "why_needed":"...",
    "issue_ids":["u1"], "category":"...",
    "answer_type":"text|single_choice|multiple_choice|visual_reference",
    "options":[], "depends_on":[], "related_context_ids":[]
  }]},
  "relevant_context_ids": []
}

Do not claim the formulation is physically correct in an absolute sense. You
are checking specification adequacy relative to provided evidence.
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



_VALID_CATEGORIES = {
    "physics_model",
    "geometry",
    "material",
    "boundary_condition",
    "load_or_source",
    "initial_condition",
    "objective",
    "constraint",
    "design_variable",
    "coupling",
    "manufacturing",
    "units",
    "context",
    "contradiction",
    "other",
}


def _canonical_category(value: object, related_fields: list[str] | None = None) -> str:
    """Normalize harmless critic category aliases before strict Pydantic validation.

    Category is routing/display metadata, not engineering intent.  The model may
    emit semantically equivalent labels such as ``loads`` or
    ``boundary_conditions``.  Rejecting an otherwise useful critic response for
    that spelling difference wastes a model call and provides no safety benefit.
    """

    raw = str(value or "").strip().lower()
    if raw in _VALID_CATEGORIES:
        return raw

    # If the critic supplied a concrete JSON Pointer, prefer the category implied
    # by the canonical ProblemSpec path over free-form wording.
    path_categories = {
        _category_from_path(path)
        for path in (related_fields or [])
        if _category_from_path(path) != "other"
    }
    if len(path_categories) == 1:
        return next(iter(path_categories))

    compact = (
        raw.replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("&", "_")
    )
    while "__" in compact:
        compact = compact.replace("__", "_")

    aliases = {
        "physics": "physics_model",
        "physics_models": "physics_model",
        "material_model": "material",
        "materials": "material",
        "boundary_conditions": "boundary_condition",
        "boundary_condition(s)": "boundary_condition",
        "bc": "boundary_condition",
        "bcs": "boundary_condition",
        "load": "load_or_source",
        "loads": "load_or_source",
        "source": "load_or_source",
        "sources": "load_or_source",
        "load_source": "load_or_source",
        "loads_sources": "load_or_source",
        "initial_conditions": "initial_condition",
        "objectives": "objective",
        "constraints": "constraint",
        "optimization_constraint": "constraint",
        "optimization_constraints": "constraint",
        "design_variables": "design_variable",
        "couplings": "coupling",
        "manufacturing_constraint": "manufacturing",
        "manufacturing_constraints": "manufacturing",
        "unit": "units",
        "contexts": "context",
        "contradictions": "contradiction",
    }
    if compact in aliases:
        return aliases[compact]

    # Last-resort lexical normalization for composite labels such as
    # "manufacturing / optimization constraint".  This only affects metadata;
    # the concern text/paths themselves remain untouched and auditable.
    if "boundary" in compact or compact.startswith("bc_"):
        return "boundary_condition"
    if "load" in compact or "source" in compact or "force" in compact or "traction" in compact:
        return "load_or_source"
    if "manufactur" in compact or "fabricat" in compact or "print" in compact:
        return "manufacturing"
    if "constraint" in compact:
        return "constraint"
    if "objective" in compact:
        return "objective"
    if "design" in compact and "variable" in compact:
        return "design_variable"
    if "material" in compact:
        return "material"
    if "geometry" in compact or "geometric" in compact:
        return "geometry"
    if "initial" in compact:
        return "initial_condition"
    if "coupl" in compact:
        return "coupling"
    if "unit" in compact:
        return "units"
    if "physics" in compact or "constitutive" in compact:
        return "physics_model"
    if "context" in compact or "evidence" in compact:
        return "context"
    if "contrad" in compact or "conflict" in compact:
        return "contradiction"
    return "other"


def _normalize_critic_data(data: dict) -> tuple[dict, dict]:
    """Canonicalize critic metadata aliases without changing engineering content."""

    normalized = json.loads(json.dumps(data))
    changes: list[dict[str, str]] = []

    status_aliases = {
        "needs clarification": "needs_clarification",
        "needs-clarification": "needs_clarification",
        "ready for review": "ready_for_review",
        "ready-for-review": "ready_for_review",
        "blocked by contradiction": "blocked_by_contradiction",
        "blocked-by-contradiction": "blocked_by_contradiction",
    }
    if isinstance(normalized.get("status"), str):
        old = normalized["status"]
        new = status_aliases.get(old.strip().lower(), old)
        if new != old:
            normalized["status"] = new
            changes.append({"path": "/status", "from": old, "to": new})

    for index, concern in enumerate(normalized.get("concerns") or []):
        if not isinstance(concern, dict):
            continue
        old = concern.get("category")
        new = _canonical_category(old, concern.get("related_fields") or [])
        if new != old:
            concern["category"] = new
            changes.append({
                "path": f"/concerns/{index}/category",
                "from": str(old),
                "to": new,
            })
        severity = str(concern.get("severity") or "").strip().lower()
        severity_aliases = {
            "critical": "high",
            "blocking": "high",
            "warning": "medium",
            "info": "low",
            "informational": "low",
        }
        if severity in severity_aliases:
            old_severity = concern.get("severity")
            concern["severity"] = severity_aliases[severity]
            changes.append({
                "path": f"/concerns/{index}/severity",
                "from": str(old_severity),
                "to": concern["severity"],
            })

    packet = normalized.get("clarification_packet")
    questions = packet.get("questions") if isinstance(packet, dict) else []
    for index, question in enumerate(questions or []):
        if not isinstance(question, dict):
            continue
        old = question.get("category")
        new = _canonical_category(old)
        if new != old:
            question["category"] = new
            changes.append({
                "path": f"/clarification_packet/questions/{index}/category",
                "from": str(old),
                "to": new,
            })

        answer_type = str(question.get("answer_type") or "text").strip().lower()
        answer_aliases = {
            "choice": "single_choice",
            "select": "single_choice",
            "single-select": "single_choice",
            "single_select": "single_choice",
            "multi_choice": "multiple_choice",
            "multiselect": "multiple_choice",
            "multi-select": "multiple_choice",
            "visual": "visual_reference",
            "image": "visual_reference",
            "drawing": "visual_reference",
            "free_text": "text",
            "freeform": "text",
        }
        new_answer_type = answer_aliases.get(answer_type, answer_type)
        if new_answer_type not in {"text", "single_choice", "multiple_choice", "visual_reference"}:
            new_answer_type = "text"
        if new_answer_type != question.get("answer_type"):
            changes.append({
                "path": f"/clarification_packet/questions/{index}/answer_type",
                "from": str(question.get("answer_type")),
                "to": new_answer_type,
            })
            question["answer_type"] = new_answer_type

    return normalized, {"normalized": bool(changes), "changes": changes}


def _critic_debug_path():
    debug_dir = ARTIFACT_ROOT / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir / "critic_last_failure.json"


def _clear_critic_debug() -> None:
    path = _critic_debug_path()
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _save_critic_failure(
    *,
    error: Exception,
    raw_text: str,
    extracted_data: dict | None,
    normalized_data: dict | None,
    normalization: dict | None,
    input_tokens: int,
    output_tokens: int,
    cache_hit: bool,
) -> None:
    try:
        _critic_debug_path().write_text(
            json.dumps(
                {
                    "error": str(error),
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "cache_hit": bool(cache_hit),
                    "raw_text": raw_text,
                    "extracted_data": extracted_data,
                    "normalized_data": normalized_data,
                    "critic_normalization": normalization,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


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




def _paths_overlap(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    a = a.rstrip("/") or "/"
    b = b.rstrip("/") or "/"
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _drop_duplicate_concerns(result: CriticResult, parser_result: ParserResult) -> CriticResult:
    """Keep critic concerns novel relative to parser-level issues.

    The parser owns the canonical unresolved-item list.  The critic may add new
    formulation concerns, but it should not make the UI/readiness gate count the
    same issue twice under a second id.
    """
    parser_paths = [
        item.field_path for item in parser_result.unresolved_items if item.field_path
    ]
    contradiction_paths = [
        path for item in parser_result.contradictions for path in item.field_paths
    ]
    kept = []
    for concern in result.concerns:
        related = concern.related_fields
        if related and any(
            _paths_overlap(path, parser_path)
            for path in related
            for parser_path in (parser_paths + contradiction_paths)
        ):
            continue
        kept.append(concern)
    result.concerns = kept[:6]
    return result


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




def _sanitize_question_semantics(result: CriticResult, parser_result: ParserResult) -> CriticResult:
    """Repair a few high-impact wording errors without changing user intent.

    The critic is allowed to phrase questions, but the deterministic layer knows
    the canonical continuum representation already present in ProblemSpec.  In
    particular, a 3-D solid displacement formulation has translational
    displacement components, not independent rotational DOFs.  This pass also
    removes unnecessary solver-architecture claims from manufacturing questions
    and avoids asking whether already-inspected evidence was sufficient.
    """

    spec = parser_result.spec
    solid_displacement = any(
        physics.family == "solid_mechanics" and "displacement" in (physics.fields or [])
        for physics in spec.physics
    )
    dimension = spec.geometry.spatial_dimension if spec.geometry else None

    for question in result.clarification_packet.questions:
        prompt_low = question.prompt.lower()

        # A 3-D solid continuum has displacement DOFs ux, uy, uz.  Rotational
        # DOFs belong to beam/shell-type formulations, not the current spec.
        if (
            solid_displacement
            and dimension == 3
            and question.category == "boundary_condition"
            and any(token in prompt_low for token in ("6 dof", "rotational", "fully fixed", "pinned"))
        ):
            question.prompt = (
                "For the mounting-hole support, how should the plate be restrained at the "
                "mounting interfaces? For the current 3-D solid displacement model, specify "
                "which translational displacement components (u_x, u_y, u_z) are constrained, "
                "or describe the actual bolted/pinned interface so the later solver adapter can "
                "derive an appropriate idealization."
            )
            question.why_needed = (
                "The support idealization changes the stiffness and compliance. The current "
                "ProblemSpec identifies the mounting-hole region but does not yet define which "
                "displacement components are restrained."
            )
            question.answer_type = "text"
            question.options = []

        # If the parser still reports unresolved existing geometry after looking
        # at the supplied assets, the current evidence is already known to be
        # insufficient. Ask for the authoritative source directly rather than
        # asking the user whether the same PDF is dimensioned.
        if question.category == "geometry" and "authoritative geometry source" in prompt_low:
            question.prompt = (
                "Please identify or upload the authoritative CAD file or dimensioned drawing "
                "that defines: (a) mounting-hole positions and diameters, (b) pivot position "
                "and size, and (c) the bumper-clearance shape, size, and location. The currently "
                "supplied evidence was not sufficient to resolve those dimensions; if no such "
                "asset exists, provide the dimensions directly."
            )
            question.answer_type = "visual_reference"

        # Keep manufacturing clarification solver-independent.  The question is
        # about admissible geometry, not which numerical representation must be
        # used later.
        if question.category == "manufacturing" and "half-pocket" in prompt_low:
            question.why_needed = (
                "This determines the admissible pocket geometry and the manufacturing constraint "
                "that the approved problem must carry into solver handoff."
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
    # The deterministic/audit representation contains leaf-expanded provenance
    # and can be very large.  The critic receives a compact semantic view only.
    # Visual assets were already interpreted during parsing and are retained in
    # the UI; re-sending full PDFs here caused the pilot critic call to dominate
    # token cost without adding formulation state.
    # critic_last_failure.json belongs only to this critic attempt.
    _clear_critic_debug()

    payload = compact_critic_payload(
        problem=problem,
        route=route,
        parser_result=parser_result,
        retrieved_context=retrieved_context,
        recent_user_clarification=recent_user_clarification,
    )
    payload["available_visual_assets"] = [
        {"name": a.name, "media_type": a.media_type, "size_bytes": len(a.data)}
        for a in (attachments or [])
    ]

    model = os.getenv("FORMULATION_MODEL", "claude-sonnet-4-6")
    cached = load_cached_response(
        component="formulation_critic",
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
            max_tokens=int(os.getenv("CRITIC_MAX_TOKENS", "2400")),
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    # Do not re-send full image/PDF bytes on every critic pass. The
                    # parser/context layer has already converted them into auditable
                    # spec/context candidates. A later focused visual-review action
                    # can be added when a specific question genuinely needs pixels.
                    "content": build_user_content(payload, None, max_visual_assets=0),
                }
            ],
        )
        if not response.content:
            raise ValueError("Formulation critic returned no content")
        raw_text = response.content[0].text
        input_tokens = int(response.usage.input_tokens)
        output_tokens = int(response.usage.output_tokens)
        cache_hit = False
        save_cached_response(
            component="formulation_critic",
            model=model,
            system_prompt=SYSTEM_PROMPT,
            payload=payload,
            response_text=raw_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=getattr(response, "stop_reason", None),
        )

    extracted_data = None
    normalized_data = None
    normalization = None
    try:
        extracted_data = _extract_json(raw_text)
        normalized_data, normalization = _normalize_critic_data(extracted_data)
        result = CriticResult.model_validate(normalized_data)
    except Exception as exc:
        _save_critic_failure(
            error=exc,
            raw_text=raw_text,
            extracted_data=extracted_data,
            normalized_data=normalized_data,
            normalization=normalization,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=cache_hit,
        )
        raise ValueError(
            "Formulation critic failed deterministic validation. "
            "No second critic call was made. The raw/normalized critic response "
            "was saved to artifacts/debug/critic_last_failure.json. "
            f"Validation error: {exc}"
        ) from exc

    result = _drop_duplicate_concerns(result, parser_result)
    result = _merge_guardrail_questions(result, parser_result)
    result = _sanitize_question_semantics(result, parser_result)

    usage = {
        "component": "formulation_critic",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_hit": cache_hit,
        "critic_normalization": normalization,
    }
    return result, usage
