"""Advisory pre-solve critic for topology-optimization problem formulation.

This module does not modify ``ProblemSpec`` and does not own any deterministic
physics decision. It compares the user's stated intent with the final structured
specification after parser clarifications/overrides and returns a bounded,
structured list of formulation concerns for human review before the solve.
"""

from __future__ import annotations

import json
import os
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from project.parser.provenance import canonical_field_paths

load_dotenv()


IssueCategory = Literal[
    "objective_or_requirement",
    "load_idealization",
    "boundary_condition_idealization",
    "design_domain_or_non_design_region",
    "material_or_analysis_scope",
    "default_or_inference",
    "geometry_or_domain_definition",
    "other",
]
IssueSeverity = Literal["low", "medium", "high"]


class FormulationIssue(BaseModel):
    """One reviewable concern grounded in the supplied prompt/specification."""

    category: IssueCategory
    severity: IssueSeverity
    affected_fields: list[str] = Field(default_factory=list)
    observation: str
    why_it_matters: str
    question_for_user: str


class FormulationCritique(BaseModel):
    """Structured output produced by the advisory LLM critic."""

    summary: str
    issues: list[FormulationIssue] = Field(default_factory=list)


SYSTEM_PROMPT = r"""
You are a PRE-SOLVE topology-optimization problem-formulation critic.

You receive:
1. the original natural-language engineering request,
2. the final structured ProblemSpec that would be sent to the deterministic
   FEniCS/SIMP solver,
3. field-level provenance showing which values were explicit, inferred,
   defaulted, or later overridden.

Your task is narrow: identify cases where the structured, runnable problem may
fail to represent the engineering question the user appears to be asking.

BOUNDARIES OF YOUR ROLE
- Do NOT solve the finite-element problem.
- Do NOT write, modify, or suggest edits to physics code.
- Do NOT change the specification.
- Do NOT issue a pass/fail verdict.
- Do NOT claim that a future topology will contain an anomaly.
- Do NOT critique convergence, MMA behavior, SIMP sensitivities, beta
  continuation, PETSc, or post-solve verification. Those belong elsewhere.
- Do NOT duplicate parser/schema validation. Contradictory or unsupported
  fields should already have failed closed before you are called.
- Do NOT perform ATO's post-convergence adequacy diagnosis. This stage occurs
  before a solution field exists.

GROUNDING RULES
- Every issue must be grounded in the supplied original request, final spec, or
  field provenance. Do not import facts from outside the packet.
- Do NOT cite, name, or attribute a paper, author, benchmark implementation,
  codebase, or literature convention unless that exact attribution appears in
  the supplied request or provenance evidence.
- A generic problem-family phrase such as "cantilever beam" or "MBB beam" does
  NOT license naming a specific paper, implementation, or "classic" benchmark.
- Do not describe the spec as "faithfully capturing" a benchmark unless the
  user explicitly named that benchmark and the packet supports the statement.
- Distinguish what is observed from what is conditional. In why_it_matters,
  use bounded language such as "could", "may", or "would represent a
  different boundary-value problem". Do not predict the actual optimized
  topology or claim a visible result that has not been solved.

WHAT YOU MAY FLAG
Only flag a concern when it is grounded in the supplied request, specification,
or provenance. Typical categories are:
- objective_or_requirement: the request states or strongly implies an
  engineering requirement that the runnable formulation does not encode;
- load_idealization: a vague or physical load description has become a point
  force, edge resultant, or traction in a way that materially changes the
  question and was not explicit;
- boundary_condition_idealization: the support model is materially more
  specific than the request and was inferred/defaulted;
- design_domain_or_non_design_region: the request implies fixture regions,
  holes, preserved pads, or forbidden material regions not represented by the
  current ProblemSpec;
- material_or_analysis_scope: the request relies on a physical assumption the
  supported nondimensional 2-D plane-stress model does not encode;
- default_or_inference: a high-impact formulation choice was filled in without
  direct user evidence;
- geometry_or_domain_definition: domain shape, dimensions, or aspect ratio are
  materially defining the engineering problem but were inferred/defaulted;
- other: use only when no listed category fits.

GEOMETRY VERSUS NUMERICAL DISCRETIZATION
- Lx and Ly define the modeled domain geometry and may be formulation concerns.
- nx and ny are numerical discretization choices in this solver. Do NOT flag
  mesh density/resolution merely because nx or ny was defaulted or inferred.
- Do NOT recommend changing mesh density, SIMP penalty, filter radius, beta, or
  other numerical settings solely to improve numerical quality or optimizer
  behavior. Those are not formulation-critic decisions.
- If geometry is the concern, affected_fields should identify geometry fields
  such as mesh.Lx and mesh.Ly, not mesh.nx or mesh.ny unless the user's stated
  intent explicitly makes element count itself part of the requested problem.

OBJECTIVE REPRESENTATION IN THIS V1 SCHEMA
- The current ProblemSpec does not expose a standalone editable objective field;
  compliance minimization is fixed by the verified solver scope.
- Do NOT use simp.vol_frac (or any other existing field) as a proxy for the
  objective itself.
- A request such as "make it as stiff as possible" together with a material
  fraction is compatible with the current compliance-minimization scope and
  should not be re-opened merely to explain that terminology.
- If the user's request explicitly requires a different/additional objective or
  requirement that the schema cannot encode, report objective_or_requirement
  with affected_fields=[] so the human is told that the issue is cross-cutting
  rather than pretending an unrelated field can repair it.

IMPORTANT RESTRAINTS
- A defaulted or inferred field is NOT automatically a problem. Flag it only
  when it is plausibly consequential to the stated engineering intent.
- If a field was user_overridden or individually_confirmed_default, do not flag
  it merely because it was originally inferred/defaulted. You may still flag a
  separate mismatch with the original request if one remains.
- For load_idealization, boundary_condition_idealization, and
  geometry_or_domain_definition, do not raise an issue when the relevant
  fields were stated explicitly by the user or individually confirmed. In the
  current architecture those categories exist to surface unresolved semantic
  choices, not to second-guess an explicit benchmark definition.
- Never flag simp.r_min, simp.penal, simp.max_iter, simp.tol_change, mesh.nx,
  or mesh.ny as formulation concerns. Numerical-method adequacy is outside this
  agent's role even if you believe a different value would be preferable.
- A bulk "use defaults" / accepted_after_opt_out action is not equivalent to an
  individually reasoned engineering choice; a materially consequential default
  may still be surfaced once for review.
- Do not invent requirements merely because a real engineer could care about
  them. For example, do not demand stress, buckling, fatigue, manufacturability,
  or multiple load cases unless the request supplies evidence for them or the
  structured model clearly conflicts with the stated use.
- A concentrated point load may be a legitimate idealization. Flag it only when
  the original request describes the physical load differently or leaves load
  type/location materially ambiguous.
- Merge overlapping concerns. For example, if load kind and load location arise
  from the same ambiguous phrase, report one load_idealization issue rather
  than two near-duplicates.
- Return at most four issues, ordered from highest to lowest severity.

SEVERITY RUBRIC
- high: the runnable formulation contradicts a stated engineering goal or omits
  a requirement directly stated/strongly required by the user's wording;
- medium: a consequential load, support, geometry, domain, objective, or scope
  choice was inferred from ambiguous language and different plausible readings
  would define different optimization problems;
- low: a grounded ambiguity is real but less consequential or less strongly
  supported. Do not use severity to express confidence in outside knowledge.

Return EXACTLY one JSON object and no prose outside it:
{
  "summary": "short neutral summary tied only to the user's request and spec",
  "issues": [
    {
      "category": "one allowed category",
      "severity": "low | medium | high",
      "affected_fields": ["canonical.path"],
      "observation": "what in the supplied evidence creates the concern",
      "why_it_matters": "how this could change the engineering problem being solved",
      "question_for_user": "one concrete question that would resolve the concern"
    }
  ]
}

If no grounded concern is identified, return an empty issues list. Never use the
word "passed" or "valid" to describe that outcome; absence of an identified
concern is not proof that the formulation is adequate.
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
        raise ValueError("Formulation critic returned no complete JSON object")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Formulation critic returned invalid JSON: {candidate}"
        ) from exc



# Fields below are numerical/algorithmic controls in the current verified
# implementation. They may be important for V&V, but they are intentionally
# outside the semantic formulation critic's authority.
_NUMERICAL_CONTROL_FIELDS = {
    "mesh.nx",
    "mesh.ny",
    "simp.penal",
    "simp.r_min",
    "simp.max_iter",
    "simp.tol_change",
}

_USER_SETTLED_INTERACTIONS = {
    "user_overridden",
    "individually_confirmed_default",
    "formulation_clarification_update",
    "formulation_clarification_confirmed",
}

_USER_SETTLED_SOURCES = {
    "explicit",
    "user_confirmed",
    "user_overridden",
}

_INFERRED_OR_DEFAULTED_SOURCES = {
    "inferred_from_benchmark_name",
    "inferred_from_language",
    "defaulted",
}


def _provenance_index(final_field_provenance: list[dict]) -> dict[str, dict]:
    return {
        str(record.get("field_path")): record
        for record in final_field_provenance
        if record.get("field_path") is not None
    }


def _field_is_user_settled(record: dict | None) -> bool:
    """Whether a field is already a direct/individual human choice.

    Explicit prompt values are treated as settled. A user override or an
    individually confirmed default is also settled. Bulk opt-out is not.
    """
    if not record:
        return False
    if record.get("source") in _USER_SETTLED_SOURCES:
        return True
    return record.get("interaction_status") in _USER_SETTLED_INTERACTIONS


def _has_unsettled_field(
    affected_fields: list[str],
    provenance: dict[str, dict],
) -> bool:
    return any(
        not _field_is_user_settled(provenance.get(path))
        for path in affected_fields
    )


def _policy_accept_issue(
    issue: FormulationIssue,
    provenance: dict[str, dict],
) -> tuple[bool, str | None]:
    """Deterministically enforce the critic's semantic authority boundary.

    The LLM proposes concerns. This filter decides whether a proposed concern
    is admissible for user-facing pre-solve review. It does not judge physics
    correctness and does not alter the ProblemSpec.
    """
    fields = list(issue.affected_fields)
    field_set = set(fields)

    # Numerical tuning and mesh-resolution advice belongs to deterministic V&V
    # or a separate numerical-method review, never to formulation adequacy.
    if fields and field_set.issubset(_NUMERICAL_CONTROL_FIELDS):
        return False, "numerical_control_out_of_scope"

    if issue.category == "objective_or_requirement":
        # The current V1 ProblemSpec has no editable objective leaf. Never let
        # the model smuggle an objective clarification through vol_frac or any
        # other unrelated field. Truly unsupported requirements are surfaced as
        # cross-cutting issues with no affected_fields.
        if fields:
            return False, "objective_requirement_has_no_editable_schema_field"

    if issue.category == "default_or_inference":
        if not fields:
            return False, "default_issue_without_affected_field"
        if not any(
            provenance.get(path, {}).get("source") in _INFERRED_OR_DEFAULTED_SOURCES
            for path in fields
        ):
            return False, "claimed_default_or_inference_but_fields_are_explicit"
        if not _has_unsettled_field(fields, provenance):
            return False, "default_or_inference_already_individually_settled"

    if issue.category == "load_idealization":
        load_fields = [path for path in fields if path.startswith("loads[")]
        if not load_fields:
            return False, "load_issue_without_load_field"
        if not _has_unsettled_field(load_fields, provenance):
            return False, "load_fields_are_explicit_or_individually_settled"

    if issue.category == "boundary_condition_idealization":
        bc_fields = [path for path in fields if path.startswith("bcs[")]
        if not bc_fields:
            return False, "bc_issue_without_boundary_condition_field"
        if not _has_unsettled_field(bc_fields, provenance):
            return False, "boundary_condition_fields_are_explicit_or_individually_settled"

    if issue.category == "geometry_or_domain_definition":
        geometry_fields = [
            path for path in fields if path in {"mesh.Lx", "mesh.Ly"}
        ]
        if not geometry_fields:
            return False, "geometry_issue_without_Lx_or_Ly"
        if not _has_unsettled_field(geometry_fields, provenance):
            return False, "geometry_fields_are_explicit_or_individually_settled"

    return True, None


def _apply_output_policy(
    critique: FormulationCritique,
    final_field_provenance: list[dict],
) -> tuple[list[FormulationIssue], list[dict]]:
    provenance = _provenance_index(final_field_provenance)
    accepted: list[FormulationIssue] = []
    suppressed: list[dict] = []

    for issue in critique.issues:
        # A model can occasionally mix a legitimate formulation field with a
        # numerical-control field. Remove the numerical field before policy
        # evaluation instead of allowing it to piggyback on the valid concern.
        cleaned_fields = [
            path
            for path in issue.affected_fields
            if path not in _NUMERICAL_CONTROL_FIELDS
        ]
        removed_numerical = sorted(
            set(issue.affected_fields) & _NUMERICAL_CONTROL_FIELDS
        )
        cleaned_issue = issue.model_copy(
            update={"affected_fields": cleaned_fields}
        )

        keep, reason = _policy_accept_issue(cleaned_issue, provenance)
        if keep:
            accepted.append(cleaned_issue)
            if removed_numerical:
                suppressed.append(
                    {
                        "category": issue.category,
                        "severity": issue.severity,
                        "affected_fields": removed_numerical,
                        "observation": issue.observation,
                        "why_it_matters": issue.why_it_matters,
                        "question_for_user": issue.question_for_user,
                        "suppression_reason": "numerical_fields_removed_from_mixed_issue",
                    }
                )
        else:
            record = issue.model_dump()
            record["suppression_reason"] = reason
            suppressed.append(record)

    return accepted, suppressed


def _deterministic_summary(issues: list[FormulationIssue]) -> str:
    """Generate a bounded summary without importing model-written claims."""
    if not issues:
        return (
            "No grounded pre-solve formulation concern survived the critic's "
            "deterministic scope and provenance checks."
        )
    high = sum(issue.severity == "high" for issue in issues)
    medium = sum(issue.severity == "medium" for issue in issues)
    low = sum(issue.severity == "low" for issue in issues)
    counts = []
    if high:
        counts.append(f"{high} high")
    if medium:
        counts.append(f"{medium} medium")
    if low:
        counts.append(f"{low} low")
    return (
        f"The pre-solve critic identified {len(issues)} grounded formulation "
        f"concern(s) for human review ({', '.join(counts)} severity)."
    )

def critique_formulation(
    *,
    original_prompt: str,
    spec,
    final_field_provenance: list[dict],
    dialogue_history: list[dict] | None = None,
) -> tuple[dict, dict]:
    """Review one final parsed specification before deterministic execution.

    Returns a plain dictionary plus token accounting. Any model/API/schema
    failure is represented as ``status='unavailable'`` rather than being
    mistaken for a clean formulation review.
    """

    valid_paths = set(canonical_field_paths(spec))
    payload = {
        "original_prompt": original_prompt,
        "final_spec": spec.model_dump(),
        "final_field_provenance": final_field_provenance,
        "canonical_field_paths": sorted(valid_paths),
        "recent_formulation_dialogue": list(dialogue_history or [])[-8:],
    }

    try:
        response = _client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1800,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        )
        if response.stop_reason == "max_tokens":
            raise ValueError(
                "Formulation critic response was truncated at max_tokens"
            )
        if not response.content:
            raise ValueError("Formulation critic returned no content")

        data = _extract_json_object(response.content[0].text)
        critique = FormulationCritique.model_validate(data)

        for issue in critique.issues:
            unknown = sorted(set(issue.affected_fields) - valid_paths)
            if unknown:
                raise ValueError(
                    "Formulation critic referenced non-canonical field path(s): "
                    + ", ".join(unknown)
                )

        accepted_issues, suppressed_issues = _apply_output_policy(
            critique, final_field_provenance
        )

        result = {
            "status": (
                "review_recommended" if accepted_issues else "no_issue_identified"
            ),
            "review_recommended": bool(accepted_issues),
            "summary": _deterministic_summary(accepted_issues),
            "issues": [issue.model_dump() for issue in accepted_issues],
            # Preserve the model proposal and every deterministic suppression so
            # the research artifact remains auditable rather than silently hiding
            # critic overreach. These fields are not displayed as recommendations.
            "model_summary_raw": critique.summary,
            "policy_suppressed_issues": suppressed_issues,
            "policy_suppressed_count": len(suppressed_issues),
        }

        usage = {
            "input_tokens": int(response.usage.input_tokens),
            "output_tokens": int(response.usage.output_tokens),
            "total_tokens": int(
                response.usage.input_tokens + response.usage.output_tokens
            ),
        }
        return result, usage

    except Exception as exc:
        return {
            "status": "unavailable",
            "review_recommended": None,
            "summary": f"Formulation critic unavailable: {exc}",
            "issues": [],
        }, {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
