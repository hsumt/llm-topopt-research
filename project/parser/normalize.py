"""Deterministic normalization of compact parser JSON into the canonical IR.

This layer exists because an LLM can express the same engineering meaning with
reasonable schema synonyms (``type`` vs ``kind``, scalar vs EngineeringValue,
inline load-case loads, etc.). Those formatting differences must not trigger an
expensive second model call.

Normalization is deliberately structural. It does not invent missing physical
quantities or solver settings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .schema import ProblemRoute


_PROBLEM_KIND_ALIASES = {
    "topology_optimization": "optimization",
    "structural_optimization": "optimization",
    "design_optimization": "optimization",
    "shape_optimization": "optimization",
    "sizing_optimization": "optimization",
    "parameter_optimization": "optimization",
    "analysis": "simulation",
    "forward_simulation": "simulation",
    "forward_problem": "simulation",
    "inverse": "inverse_problem",
}

_PHYSICS_FAMILY_ALIASES = {
    "structural": "solid_mechanics",
    "structural_mechanics": "solid_mechanics",
    "mechanical": "solid_mechanics",
    "mechanics": "solid_mechanics",
    "heat_transfer": "thermal",
    "heat_conduction": "thermal",
    "thermal_conduction": "thermal",
    "cfd": "fluid",
    "fluid_mechanics": "fluid",
}

_REGIME_ALIASES = {
    "static": "quasi_static",
    "static_structural": "quasi_static",
    "quasistatic": "quasi_static",
    "quasi-static": "quasi_static",
    "steady_state": "steady",
    "steady-state": "steady",
    "dynamic": "transient",
}

_RELATION_ALIASES = {
    "upper": "<=",
    "upper_bound": "<=",
    "maximum": "<=",
    "max": "<=",
    "lower": ">=",
    "lower_bound": ">=",
    "minimum": ">=",
    "min": ">=",
    "equal": "=",
    "equality": "=",
    "equals": "=",
}

_UNIT_SUFFIXES = {
    "_in": "in",
    "_inch": "in",
    "_inches": "in",
    "_mm": "mm",
    "_cm": "cm",
    "_m": "m",
    "_lb": "lb",
    "_lbf": "lbf",
    "_n": "N",
    "_pa": "Pa",
    "_kpa": "kPa",
    "_mpa": "MPa",
    "_k": "K",
    "_c": "degC",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _eng(value: Any, *, unit: str | None = None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict) and "value" in value:
        result = deepcopy(value)
        if unit and not result.get("unit"):
            result["unit"] = unit
        return result
    return {"value": value, **({"unit": unit} if unit else {})}


def _unit_from_key(key: str) -> str | None:
    low = key.lower()
    for suffix, unit in sorted(_UNIT_SUFFIXES.items(), key=lambda kv: -len(kv[0])):
        if low.endswith(suffix):
            return unit
    return None


def _normalize_parameter_map(raw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in _as_dict(raw).items():
        if isinstance(value, dict) and "value" in value:
            out[key] = value
        else:
            out[key] = _eng(value, unit=_unit_from_key(key))
    return out


def _normalize_geometry(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    g = deepcopy(_as_dict(raw))
    dim = g.get("spatial_dimension")
    if isinstance(dim, str):
        cleaned = dim.strip().lower().replace("-", "").replace(" ", "")
        aliases = {"1d": 1, "2d": 2, "3d": 3, "one": 1, "two": 2, "three": 3}
        if cleaned in aliases:
            g["spatial_dimension"] = aliases[cleaned]
    g["parameters"] = _normalize_parameter_map(g.get("parameters"))
    g["regions"] = _as_list(g.get("regions"))
    return g


def _normalize_physics(raw: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _as_list(raw):
        p = deepcopy(_as_dict(item))
        family = p.get("family")
        if isinstance(family, str):
            p["family"] = _PHYSICS_FAMILY_ALIASES.get(family.strip().lower(), family)
        regime = p.get("regime")
        if isinstance(regime, str):
            p["regime"] = _REGIME_ALIASES.get(regime.strip().lower(), regime)
        p["fields"] = _as_list(p.get("fields"))
        p["assumptions"] = _as_list(p.get("assumptions"))
        p["parameters"] = _normalize_parameter_map(p.get("parameters"))
        result.append(p)
    return result


def _normalize_materials(raw: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for idx, item in enumerate(_as_list(raw)):
        m = deepcopy(_as_dict(item))
        m.setdefault("id", f"material_{idx + 1}")
        if not m.get("model") and m.get("name") is not None:
            m["model"] = m.pop("name")
        properties = _normalize_parameter_map(m.get("properties"))
        if "grade" in m:
            properties.setdefault("grade", _eng(m.pop("grade")))
        if "thickness_in" in m:
            properties.setdefault("thickness", _eng(m.pop("thickness_in"), unit="in"))
        if "thickness_mm" in m:
            properties.setdefault("thickness", _eng(m.pop("thickness_mm"), unit="mm"))
        if "notes" in m:
            properties.setdefault("notes", _eng(m.pop("notes")))
        # Preserve any remaining unfamiliar non-canonical material metadata as
        # named properties instead of dropping it.
        canonical = {"id", "region", "model", "properties"}
        for key in list(m):
            if key not in canonical:
                properties.setdefault(key, _eng(m.pop(key), unit=_unit_from_key(key)))
        m["properties"] = properties
        result.append(m)
    return result


def _normalize_bcs(raw: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for idx, item in enumerate(_as_list(raw)):
        bc = deepcopy(_as_dict(item))
        bc.setdefault("id", f"bc_{idx + 1}")
        if bc.get("kind") is None and "type" in bc:
            bc["kind"] = bc.pop("type")
        if bc.get("location") is None and "region" in bc:
            bc["location"] = bc.pop("region")
        params = _normalize_parameter_map(bc.get("parameters"))
        if "description" in bc:
            params.setdefault("description", _eng(bc.pop("description")))
        canonical = {
            "id", "physics_id", "location", "field", "kind", "components",
            "value", "parameters",
        }
        for key in list(bc):
            if key not in canonical:
                params.setdefault(key, _eng(bc.pop(key), unit=_unit_from_key(key)))
        bc["components"] = _as_list(bc.get("components"))
        if bc.get("value") is not None:
            bc["value"] = _eng(bc["value"])
        bc["parameters"] = params
        result.append(bc)
    return result


def _normalize_source(raw: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    src = deepcopy(raw)
    src.setdefault("id", fallback_id)
    if src.get("location") is None and "region" in src:
        src["location"] = src.pop("region")
    if src.get("kind") is None and "type" in src:
        src["kind"] = src.pop("type")

    # Promote one obvious magnitude field to canonical value. The key itself
    # provides the unit when present; no physical conversion is performed.
    if src.get("value") is None:
        magnitude_keys = [
            "magnitude", "force", "magnitude_lb", "magnitude_lbf",
            "magnitude_n", "load", "total_magnitude",
        ]
        for key in magnitude_keys:
            if key in src:
                src["value"] = _eng(src.pop(key), unit=_unit_from_key(key))
                break
    elif src.get("value") is not None:
        src["value"] = _eng(src["value"])

    params = _normalize_parameter_map(src.get("parameters"))
    canonical = {"id", "physics_id", "location", "field", "kind", "value", "parameters"}
    for key in list(src):
        if key not in canonical:
            params.setdefault(key, _eng(src.pop(key), unit=_unit_from_key(key)))
    src["parameters"] = params
    return src


def _normalize_sources(raw: Any) -> list[dict[str, Any]]:
    return [
        _normalize_source(_as_dict(item), fallback_id=f"source_{idx + 1}")
        for idx, item in enumerate(_as_list(raw))
    ]


def _normalize_load_cases(
    raw_cases: Any,
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[int, int]]]:
    """Normalize load cases and lift any inline ``loads`` into ``sources``.

    Returns ``(cases, sources, lifted)`` where lifted contains
    ``(load_case_index, source_index)`` pairs for provenance propagation.
    """

    cases: list[dict[str, Any]] = []
    lifted: list[tuple[int, int]] = []
    known_ids = {s.get("id") for s in sources}

    for ci, item in enumerate(_as_list(raw_cases)):
        case = deepcopy(_as_dict(item))
        case.setdefault("id", f"load_case_{ci + 1}")
        case.setdefault("name", case.get("id"))
        source_ids = list(_as_list(case.get("source_ids")))
        inline_loads = _as_list(case.pop("loads", []))
        for li, load in enumerate(inline_loads):
            candidate = _normalize_source(
                _as_dict(load),
                fallback_id=f"{case['id']}_source_{li + 1}",
            )
            source_id = candidate["id"]
            if source_id not in known_ids:
                sources.append(candidate)
                source_index = len(sources) - 1
                known_ids.add(source_id)
                lifted.append((ci, source_index))
            source_ids.append(source_id)
        case["source_ids"] = list(dict.fromkeys(source_ids))
        case["boundary_condition_ids"] = _as_list(case.get("boundary_condition_ids"))

        # Preserve unknown case metadata in the description rather than fail.
        canonical = {"id", "name", "source_ids", "boundary_condition_ids", "description"}
        extras = []
        for key in list(case):
            if key not in canonical:
                extras.append(f"{key}={case.pop(key)!r}")
        if extras:
            existing = case.get("description") or ""
            suffix = "; ".join(extras)
            case["description"] = f"{existing}; {suffix}".strip("; ")
        cases.append(case)

    return cases, sources, lifted


def _normalize_design_variables(raw: Any) -> list[dict[str, Any]]:
    result = []
    for idx, item in enumerate(_as_list(raw)):
        dv = deepcopy(_as_dict(item))
        dv.setdefault("id", f"design_variable_{idx + 1}")
        if dv.get("kind") is None and "type" in dv:
            dv["kind"] = dv.pop("type")
        if dv.get("target") is None and dv.get("region") is not None:
            dv["target"] = dv.get("region")
        params = _normalize_parameter_map(dv.get("parameters"))
        if "description" in dv:
            params.setdefault("description", _eng(dv.pop("description")))
        canonical = {
            "id", "kind", "target", "region", "lower_bound", "upper_bound", "parameters"
        }
        for key in list(dv):
            if key not in canonical:
                params.setdefault(key, _eng(dv.pop(key), unit=_unit_from_key(key)))
        if dv.get("lower_bound") is not None:
            dv["lower_bound"] = _eng(dv["lower_bound"])
        if dv.get("upper_bound") is not None:
            dv["upper_bound"] = _eng(dv["upper_bound"])
        dv["parameters"] = params
        result.append(dv)
    return result


def _normalize_objectives(raw: Any) -> list[dict[str, Any]]:
    result = []
    for idx, item in enumerate(_as_list(raw)):
        obj = deepcopy(_as_dict(item))
        obj.setdefault("id", f"objective_{idx + 1}")
        type_value = obj.pop("type", None)
        if isinstance(type_value, str):
            low = type_value.lower()
            for prefix in ("minimize_", "maximize_", "target_"):
                if low.startswith(prefix):
                    if obj.get("sense") is None:
                        obj["sense"] = prefix[:-1]
                    if obj.get("quantity") is None:
                        obj["quantity"] = type_value[len(prefix):]
                    break
            else:
                obj.setdefault("quantity", type_value)
        params = _normalize_parameter_map(obj.get("parameters"))
        if "load_cases" in obj:
            params.setdefault("load_cases", _eng(obj.pop("load_cases")))
        if "description" in obj:
            params.setdefault("description", _eng(obj.pop("description")))
        canonical = {"id", "sense", "quantity", "region", "target", "weight", "parameters"}
        for key in list(obj):
            if key not in canonical:
                params.setdefault(key, _eng(obj.pop(key), unit=_unit_from_key(key)))
        if obj.get("target") is not None:
            obj["target"] = _eng(obj["target"])
        obj["parameters"] = params
        result.append(obj)
    return result


def _normalize_constraints(raw: Any) -> list[dict[str, Any]]:
    result = []
    for idx, item in enumerate(_as_list(raw)):
        con = deepcopy(_as_dict(item))
        con.setdefault("id", f"constraint_{idx + 1}")
        if con.get("quantity") is None and "type" in con:
            con["quantity"] = con.pop("type")
        if con.get("limit") is None and "bound" in con:
            con["limit"] = _eng(con.pop("bound"))
        elif con.get("limit") is not None:
            con["limit"] = _eng(con["limit"])
        if con.get("relation") is None and "sense" in con:
            sense = con.pop("sense")
            if isinstance(sense, str):
                con["relation"] = _RELATION_ALIASES.get(sense.lower(), sense)
        if con.get("relation") is None and con.get("quantity") in {
            "passive_solid", "void_exclusion", "passive_void"
        }:
            con["relation"] = "="
            if con.get("limit") is None:
                con["limit"] = _eng(True)
        params = _normalize_parameter_map(con.get("parameters"))
        if "description" in con:
            params.setdefault("description", _eng(con.pop("description")))
        canonical = {"id", "quantity", "relation", "limit", "region", "parameters"}
        for key in list(con):
            if key not in canonical:
                params.setdefault(key, _eng(con.pop(key), unit=_unit_from_key(key)))
        con["parameters"] = params
        result.append(con)
    return result


def _normalize_optimization(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    opt = deepcopy(_as_dict(raw))
    return {
        "design_variables": _normalize_design_variables(opt.get("design_variables")),
        "objectives": _normalize_objectives(opt.get("objectives")),
        "constraints": _normalize_constraints(opt.get("constraints")),
    }


def _normalize_manufacturing(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    m = deepcopy(_as_dict(raw))
    requirements = [str(x) for x in _as_list(m.get("requirements"))]
    params = _normalize_parameter_map(m.get("parameters"))
    if "pocketing" in m:
        requirements.append(str(m.pop("pocketing")))
    if "thickness_tolerance" in m:
        params.setdefault("thickness_tolerance", _eng(m.pop("thickness_tolerance")))
    if "cross_section_tolerance_in" in m:
        params.setdefault(
            "cross_section_tolerance",
            _eng(m.pop("cross_section_tolerance_in"), unit="in"),
        )
    canonical = {
        "process", "material_form", "machine", "stock_material", "notes",
        "requirements", "parameters",
    }
    for key in list(m):
        if key not in canonical:
            value = m.pop(key)
            if isinstance(value, str) and not key.endswith(("_in", "_mm", "_n", "_lb")):
                requirements.append(f"{key}: {value}")
            else:
                params.setdefault(key, _eng(value, unit=_unit_from_key(key)))
    m["requirements"] = list(dict.fromkeys(x for x in requirements if x))
    m["parameters"] = params
    return m


def _normalize_problem_kind(value: Any, route: ProblemRoute) -> str:
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"simulation", "optimization", "inverse_problem", "unknown"}:
            return low
        if low in _PROBLEM_KIND_ALIASES:
            return _PROBLEM_KIND_ALIASES[low]
    # Router already made this classification in a separate call; using it here
    # is not a new inference.
    return route.task_type


def _rewrite_provenance_path(path: str) -> str:
    """Map common raw-schema aliases to stable canonical scopes.

    Specific fields that map to multiple canonical fields are intentionally
    broadened to their parent semantic object.
    """
    if not isinstance(path, str) or not path.startswith("/"):
        return path
    parts = path.split("/")
    if len(parts) >= 4 and parts[1] == "materials":
        if parts[3] in {"name", "grade", "thickness_in", "thickness_mm", "notes"}:
            return "/".join(parts[:3])
    if len(parts) >= 4 and parts[1] == "boundary_conditions":
        if parts[3] in {"type", "region", "description"}:
            return "/".join(parts[:3])
    if len(parts) >= 5 and parts[1] == "optimization":
        if parts[2] in {"design_variables", "objectives", "constraints"}:
            return "/".join(parts[:4])
    if len(parts) >= 3 and parts[1] == "manufacturing":
        return "/manufacturing"
    return path



def _scope_covers(scope: str, target: str) -> bool:
    if not isinstance(scope, str) or not scope.startswith("/"):
        return False
    scope = scope.rstrip("/") or "/"
    target = target.rstrip("/") or "/"
    return scope == target or target.startswith(scope + "/") or scope.startswith(target + "/")


def _specific_scope_present(provenance: list[dict[str, Any]], target: str) -> bool:
    """Return True only for provenance at ``target`` or below it.

    A broad parent record such as ``/materials/0`` is deliberately *not*
    sufficient to justify invented constitutive properties under
    ``/materials/0/properties``.
    """
    prefix = target.rstrip("/")
    for rec in provenance:
        path = rec.get("field_path")
        if not isinstance(path, str):
            continue
        path = path.rstrip("/")
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def _relevant_evidence(provenance: list[dict[str, Any]], target: str) -> str:
    chunks: list[str] = []
    for rec in provenance:
        path = rec.get("field_path")
        if isinstance(path, str) and _scope_covers(path, target):
            evidence = rec.get("evidence")
            if isinstance(evidence, str) and evidence.strip():
                chunks.append(evidence.strip().lower())
    return " ".join(chunks)


def _append_unresolved_once(
    unresolved: list[dict[str, Any]],
    *,
    field_path: str,
    issue: str,
    question: str,
    evidence: str | None,
    required_for_execution: bool,
) -> None:
    if any(item.get("field_path") == field_path for item in unresolved if isinstance(item, dict)):
        return
    unresolved.append(
        {
            "id": f"auto_{len(unresolved) + 1}",
            "field_path": field_path,
            "issue": issue,
            "evidence": evidence,
            "question": question,
            "required_for_execution": required_for_execution,
        }
    )


def _sanitize_unproven_semantics(
    spec: dict[str, Any],
    provenance: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> dict[str, Any]:
    """Remove high-risk values that are not specifically grounded.

    This is intentionally narrow. It does not try to judge all engineering
    semantics. It blocks two recurring failure modes seen in pilot runs:
    nominal material properties invented from general knowledge, and a
    fully-fixed support inferred merely from the phrase "attach using holes".
    """

    dropped_material_property_sets: list[str] = []
    relaxed_supports: list[str] = []
    dropped_requested_outputs = False

    # Requested outputs are a presentation preference, not solver-driving
    # formulation. Keep them only when explicitly scoped by provenance.
    if spec.get("requested_outputs") and not _specific_scope_present(
        provenance, "/requested_outputs"
    ):
        spec["requested_outputs"] = []
        dropped_requested_outputs = True

    # Constitutive/material numbers must be grounded specifically. A broad
    # /materials/0 record that only says "polycarbonate" cannot justify E, nu,
    # density, strength, conductivity, etc.
    for index, material in enumerate(_as_list(spec.get("materials"))):
        if not isinstance(material, dict):
            continue
        properties = _as_dict(material.get("properties"))
        target = f"/materials/{index}/properties"
        if properties and not _specific_scope_present(provenance, target):
            material["properties"] = {}
            dropped_material_property_sets.append(target)
            _append_unresolved_once(
                unresolved,
                field_path=target,
                issue="Material constitutive properties are not grounded in supplied problem/context.",
                evidence=None,
                question="Which material grade/properties should govern the analysis, or should they be supplied later by the solver/material database?",
                required_for_execution=False,
            )

    # "Attach/mount using holes" establishes the support interface but does not
    # establish that x/y/z are all rigidly fixed. Retain location/field, but
    # remove a fixed idealization unless the evidence explicitly says so or a
    # specific kind/components provenance scope exists.
    fixed_words = (
        "fixed", "clamped", "fully restrained", "rigidly restrained",
        "zero displacement", "ux=", "uy=", "uz=",
    )
    for index, bc in enumerate(_as_list(spec.get("boundary_conditions"))):
        if not isinstance(bc, dict):
            continue
        parent = f"/boundary_conditions/{index}"
        kind = str(bc.get("kind") or "").strip().lower()
        if kind != "fixed":
            continue
        evidence = _relevant_evidence(provenance, parent)
        has_specific = (
            _specific_scope_present(provenance, parent + "/kind")
            or _specific_scope_present(provenance, parent + "/components")
        )
        explicitly_fixed = any(word in evidence for word in fixed_words)
        if not has_specific and not explicitly_fixed:
            bc["kind"] = None
            bc["components"] = []
            bc["value"] = None
            relaxed_supports.append(parent)
            _append_unresolved_once(
                unresolved,
                field_path=parent + "/kind",
                issue="Support region is known, but the restraint idealization is not stated.",
                evidence=evidence or None,
                question="How should the mounting interface restrain motion (for example fully fixed, pinned/bolt-bearing, or another idealization)?",
                required_for_execution=True,
            )

    return {
        "dropped_unproven_requested_outputs": dropped_requested_outputs,
        "dropped_unproven_material_property_sets": dropped_material_property_sets,
        "relaxed_unproven_fixed_supports": relaxed_supports,
    }


def normalize_parser_payload(data: dict[str, Any], route: ProblemRoute) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return canonical-ish ParserResult JSON plus normalization diagnostics."""

    out = deepcopy(data)
    out.setdefault("field_provenance", [])
    out.setdefault("unresolved_items", [])
    out.setdefault("contradictions", [])
    out.setdefault("context_candidates", [])

    spec = deepcopy(_as_dict(out.get("spec")))
    spec.setdefault("name", "Unnamed engineering problem")
    spec["problem_kind"] = _normalize_problem_kind(spec.get("problem_kind"), route)
    spec["geometry"] = _normalize_geometry(spec.get("geometry"))
    spec["physics"] = _normalize_physics(spec.get("physics"))
    spec["materials"] = _normalize_materials(spec.get("materials"))
    spec["boundary_conditions"] = _normalize_bcs(spec.get("boundary_conditions"))
    spec["sources"] = _normalize_sources(spec.get("sources"))

    load_cases, sources, lifted = _normalize_load_cases(spec.get("load_cases"), spec["sources"])
    spec["load_cases"] = load_cases
    spec["sources"] = sources
    spec["optimization"] = _normalize_optimization(spec.get("optimization"))
    spec["manufacturing"] = _normalize_manufacturing(spec.get("manufacturing"))

    # Normalize initial conditions/couplings/discretization just enough to keep
    # parameter values canonical without inventing semantics.
    spec["initial_conditions"] = _as_list(spec.get("initial_conditions"))
    spec["couplings"] = _as_list(spec.get("couplings"))
    if isinstance(spec.get("discretization"), dict):
        disc = deepcopy(spec["discretization"])
        disc["parameters"] = _normalize_parameter_map(disc.get("parameters"))
        spec["discretization"] = disc
    spec["requested_outputs"] = _as_list(spec.get("requested_outputs"))
    spec["assumptions"] = [str(x) for x in _as_list(spec.get("assumptions"))]

    out["spec"] = spec

    # Rewrite provenance aliases and propagate load-case evidence to lifted
    # source objects. The provenance source/evidence still comes from the LLM;
    # Python only follows the structural move it just performed.
    provenance = []
    raw_provenance = _as_list(out.get("field_provenance"))
    for item in raw_provenance:
        rec = deepcopy(_as_dict(item))
        if "field_path" in rec:
            rec["field_path"] = _rewrite_provenance_path(rec["field_path"])
        provenance.append(rec)

    for case_index, source_index in lifted:
        case_prefix = f"/load_cases/{case_index}"
        candidates = [
            rec for rec in provenance
            if isinstance(rec.get("field_path"), str)
            and (
                rec["field_path"] == case_prefix
                or case_prefix.startswith(rec["field_path"] + "/")
                or rec["field_path"].startswith(case_prefix + "/")
            )
        ]
        if candidates:
            clone = deepcopy(candidates[0])
            clone["field_path"] = f"/sources/{source_index}"
            clone.pop("value", None)
            provenance.append(clone)

    semantic_sanitization = _sanitize_unproven_semantics(
        spec,
        provenance,
        out["unresolved_items"],
    )
    out["spec"] = spec
    out["field_provenance"] = provenance

    diagnostics = {
        "normalized": True,
        "lifted_inline_loads": len(lifted),
        "source_count": len(spec["sources"]),
        "load_case_count": len(spec["load_cases"]),
        "semantic_sanitization": semantic_sanitization,
    }
    return out, diagnostics
