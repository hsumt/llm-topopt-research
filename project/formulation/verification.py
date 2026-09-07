"""Deterministic tests for formulation-agent authority and patch semantics.

No LLM/API call occurs here. These checks belong in the hash-bound verification
suite because they enforce the boundary between language interpretation and the
runnable physics specification.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from project.formulation.patching import apply_resolution
from project.formulation.dialogue import _merge_active_issues
from project.formulation.preview import format_intent_snapshot, generate_intent_preview
from project.llm.formulation_resolver import _deterministic_resolution_from_answer
from project.llm.formulation_critic import (
    FormulationCritique,
    FormulationIssue,
    _apply_output_policy,
    _policy_accept_issue,
)
from project.parser.provenance import canonical_field_paths, get_path
from project.parser.schema import ProblemSpec


def _spec() -> ProblemSpec:
    return ProblemSpec.model_validate(
        {
            "name": "formulation_test",
            "analysis": {
                "formulation": "plane_stress",
                "unit_system": "nondimensional",
                "thickness": 1.0,
                "edge_traction_definition": "line_load",
            },
            "mesh": {"nx": 60, "ny": 20, "Lx": 3.0, "Ly": 1.0},
            "material": {"E": 1.0, "nu": 0.3},
            "loads": [
                {
                    "location": "right_center",
                    "dof": "y",
                    "value": -1.0,
                    "kind": "point_force",
                }
            ],
            "bcs": [
                {"location": "left_edge", "dof": "x", "value": 0.0},
                {"location": "left_edge", "dof": "y", "value": 0.0},
            ],
            "simp": {
                "penal": 3.0,
                "vol_frac": 0.4,
                "r_min": 0.125,
                "max_iter": 250,
                "tol_change": 0.01,
            },
        }
    )


def _provenance(spec: ProblemSpec, *, load_source: str = "inferred_from_language"):
    payload = spec.model_dump()
    records = []
    for path in canonical_field_paths(spec):
        source = "explicit"
        evidence = "test fixture"
        status = "unchanged"
        if path.startswith("analysis."):
            source = "fixed_by_solver_scope"
            evidence = None
        if path in {"loads[0].location", "loads[0].kind"}:
            source = load_source
            evidence = "downward load near right side"
            status = "pending_formulation_review"
        records.append(
            {
                "field_path": path,
                "source": source,
                "value": get_path(payload, path),
                "final_value": get_path(payload, path),
                "evidence": evidence,
                "confidence": 1.0,
                "interaction_status": status,
            }
        )
    return records


def run_deterministic_formulation_tests(pass_callback) -> None:
    spec = _spec()

    inferred_prov = _provenance(spec, load_source="inferred_from_language")
    inferred_idx = {r["field_path"]: r for r in inferred_prov}
    load_issue = FormulationIssue(
        category="load_idealization",
        severity="medium",
        affected_fields=["loads[0].location", "loads[0].kind"],
        observation="load was underspecified",
        why_it_matters="different load idealizations define different problems",
        question_for_user="Where/how is the load applied?",
    )
    keep, _ = _policy_accept_issue(load_issue, inferred_idx)
    assert keep
    pass_callback("formulation policy admits unresolved load idealization")

    explicit_prov = _provenance(spec, load_source="explicit")
    explicit_idx = {r["field_path"]: r for r in explicit_prov}
    keep, reason = _policy_accept_issue(load_issue, explicit_idx)
    assert not keep
    assert reason == "load_fields_are_explicit_or_individually_settled"
    pass_callback("formulation policy suppresses second-guessing explicit load")

    numerical_issue = FormulationIssue(
        category="default_or_inference",
        severity="low",
        affected_fields=["simp.r_min"],
        observation="filter radius",
        why_it_matters="numerical control",
        question_for_user="change it?",
    )
    keep, reason = _policy_accept_issue(numerical_issue, inferred_idx)
    assert not keep and reason == "numerical_control_out_of_scope"
    pass_callback("formulation policy excludes numerical-control tuning")

    mixed = FormulationCritique(
        summary="mixed",
        issues=[
            FormulationIssue(
                category="geometry_or_domain_definition",
                severity="medium",
                affected_fields=["mesh.Lx", "mesh.nx"],
                observation="geometry inferred",
                why_it_matters="domain geometry changes problem",
                question_for_user="What length?",
            )
        ],
    )
    # Make Lx unsettled for this fixture while nx remains a numerical field.
    mixed_prov = _provenance(spec)
    for record in mixed_prov:
        if record["field_path"] == "mesh.Lx":
            record["source"] = "defaulted"
            record["evidence"] = None
            record["interaction_status"] = "accepted_after_opt_out"
    accepted, suppressed = _apply_output_policy(mixed, mixed_prov)
    assert len(accepted) == 1
    assert accepted[0].affected_fields == ["mesh.Lx"]
    assert any(
        item.get("suppression_reason") == "numerical_fields_removed_from_mixed_issue"
        for item in suppressed
    )
    pass_callback("formulation policy strips numerical fields from mixed concern")

    objective_proxy = FormulationIssue(
        category="objective_or_requirement",
        severity="low",
        affected_fields=["simp.vol_frac"],
        observation="objective wording",
        why_it_matters="objective semantics",
        question_for_user="Is this the objective?",
    )
    keep, reason = _policy_accept_issue(objective_proxy, inferred_idx)
    assert not keep
    assert reason == "objective_requirement_has_no_editable_schema_field"
    pass_callback("formulation policy blocks volume fraction as objective proxy")

    valid_resolution = {
        "status": "update",
        "updates": [
            {"field_path": "loads[0].location", "new_value": "right_edge"},
            {"field_path": "loads[0].kind", "new_value": "edge_resultant"},
        ],
        "confirmed_fields": [],
    }
    new_spec, new_prov, audit = apply_resolution(
        spec=spec,
        final_field_provenance=inferred_prov,
        issue=load_issue.model_dump(),
        resolution=valid_resolution,
        user_answer="Distribute a total downward force across the right edge.",
    )
    assert new_spec.loads[0].location == "right_edge"
    assert new_spec.loads[0].kind == "edge_resultant"
    by_path = {r["field_path"]: r for r in new_prov}
    assert by_path["loads[0].location"]["source"] == "user_overridden"
    assert by_path["loads[0].kind"]["source"] == "user_overridden"
    assert len(audit["updates"]) == 2
    pass_callback("authorized formulation patch updates spec and provenance")

    deterministic = _deterministic_resolution_from_answer(
        spec=spec,
        issue=load_issue.model_dump(),
        user_answer="Use a total downward force of -1 distributed across the entire right edge.",
    )
    assert deterministic is not None
    assert deterministic["status"] == "update"
    updates = {item["field_path"]: item["new_value"] for item in deterministic["updates"]}
    assert updates["loads[0].location"] == "right_edge"
    assert updates["loads[0].kind"] == "edge_resultant"
    pass_callback("deterministic resolver maps total full-edge load without follow-up")

    bad_resolution = {
        "status": "update",
        "updates": [{"field_path": "simp.r_min", "new_value": 0.2}],
        "confirmed_fields": [],
    }
    try:
        apply_resolution(
            spec=spec,
            final_field_provenance=inferred_prov,
            issue=load_issue.model_dump(),
            resolution=bad_resolution,
            user_answer="also increase the filter radius",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unauthorized formulation patch was not rejected")
    pass_callback("formulation patch gate rejects unauthorized field change")

    confirm_resolution = {
        "status": "confirm_current",
        "updates": [],
        "confirmed_fields": ["loads[0].location", "loads[0].kind"],
    }
    _, confirmed_prov, _ = apply_resolution(
        spec=spec,
        final_field_provenance=inferred_prov,
        issue=load_issue.model_dump(),
        resolution=confirm_resolution,
        user_answer="keep current",
    )
    confirmed_idx = {r["field_path"]: r for r in confirmed_prov}
    assert confirmed_idx["loads[0].location"]["source"] == "user_confirmed"
    assert confirmed_idx["loads[0].kind"]["source"] == "user_confirmed"
    pass_callback("formulation confirmation closes provenance without changing spec")

    keep, reason = _policy_accept_issue(load_issue, confirmed_idx)
    assert not keep and reason == "load_fields_are_explicit_or_individually_settled"
    pass_callback("re-critique does not reopen human-confirmed formulation field")

    bc_issue = {
        "category": "boundary_condition_idealization",
        "severity": "medium",
        "affected_fields": [
            "bcs[0].location", "bcs[0].dof",
            "bcs[1].location", "bcs[1].dof",
        ],
        "observation": "support was inferred",
        "why_it_matters": "support choice defines the boundary-value problem",
        "question_for_user": "Should the whole left edge be clamped?",
    }
    unsettled_bc_prov = _provenance(spec)
    for record in unsettled_bc_prov:
        if record["field_path"].startswith("bcs["):
            record["source"] = "inferred_from_language"
            record["evidence"] = "fix the left side"
            record["interaction_status"] = "pending_formulation_review"
    active1, pending, carried1 = _merge_active_issues(
        new_issues=[bc_issue],
        pending_issues={},
        final_field_provenance=unsettled_bc_prov,
    )
    assert len(active1) == 1 and carried1 == 0
    active2, _, carried2 = _merge_active_issues(
        new_issues=[],
        pending_issues=pending,
        final_field_provenance=unsettled_bc_prov,
    )
    assert len(active2) == 1 and carried2 == 1
    pass_callback("unresolved formulation concern survives later LLM omission")

    snapshot = format_intent_snapshot(spec)
    assert "right_center" in snapshot and "point force" in snapshot
    with TemporaryDirectory() as tmp:
        path = generate_intent_preview(spec, Path(tmp) / "preview.png")
        assert path.exists() and path.stat().st_size > 0
    pass_callback("deterministic intent preview renders from ProblemSpec")
