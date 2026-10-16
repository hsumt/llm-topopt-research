"""Readable summary of a problem specification, plus the value formatting other components share."""
import re

import streamlit as st

from project.ui.api import UNPARSED_NAME

_MARKDOWN = re.compile(r"([\\`*_\[\]<>#$~|])")
_RELATIONS = {"<=": "≤", ">=": "≥", "=": "="}
_STRESS = {"volume_only": "volume constraint only", "pnorm_limit": "p-norm stress limit"}


def md(text) -> str:
    """Escape model-written text so st.markdown shows it literally."""
    return _MARKDOWN.sub(r"\\\1", str(text))


def words(text) -> str | None:
    """Plain words for an identifier; None when unset or unknown."""
    if text is None or str(text).strip() in ("", "unknown"):
        return None
    return str(text).replace("_", " ").strip()


def capital(text: str | None) -> str | None:
    return text[:1].upper() + text[1:] if text else text


def number(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(number(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {number(v)}" for k, v in value.items()) + "}"
    return str(value)


def _numbers(value) -> list[float] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [abs(value)]
    if isinstance(value, (list, tuple)):
        found: list[float] = []
        for item in value:
            inner = _numbers(item)
            if inner is None:
                return None
            found += inner
        return found
    return None


def _scale(value, factor: float):
    return [_scale(v, factor) for v in value] if isinstance(value, (list, tuple)) else value * factor


def readable(value, unit: str | None) -> tuple[str, str]:
    """(value, unit) as display text: metres as mm, pascals as MPa or GPa."""
    numbers = _numbers(value)
    if numbers and unit == "m":
        value, unit = _scale(value, 1e3), "mm"
    elif numbers and unit == "Pa":
        biggest = max(numbers)
        if biggest >= 1e9:
            value, unit = _scale(value, 1e-9), "GPa"
        elif biggest >= 1e6:
            value, unit = _scale(value, 1e-6), "MPa"
    return number(value), unit or ""


def quantity(item: dict | None) -> str | None:
    if not item or item.get("value") is None:
        return None
    return " ".join(part for part in readable(item["value"], item.get("unit")) if part)


def _dump(spec) -> dict:
    return spec.model_dump(mode="json") if hasattr(spec, "model_dump") else dict(spec or {})


def _heading(title: str) -> None:
    st.markdown(f"**{title}**")


def _not_stated() -> None:
    st.caption("Not stated")


def _fields(pairs: list[tuple[str, str | None]]) -> None:
    st.markdown("  \n".join(f"{name}: {md(value) if value else '*not stated*'}" for name, value in pairs))


def _bullets(lines: list[str]) -> None:
    st.markdown("\n".join(f"- {line}" for line in lines))


def _or(value, missing: str) -> str:
    return md(value) if value else f"*{missing}*"


def _problem(spec: dict) -> None:
    _heading("Problem")
    name = spec.get("name")
    _fields([("Name", None if name in (None, "", UNPARSED_NAME) else name),
             ("Type", capital(words(spec.get("problem_kind"))))])


def _physics(physics: dict | None) -> None:
    _heading("Physics")
    if not physics:
        return _not_stated()
    _fields([("Field", capital(words(physics.get("family")))), ("Regime", words(physics.get("regime"))),
             ("Representation", words(physics.get("representation")))])


def _geometry(geometry: dict | None) -> None:
    _heading("Geometry")
    if not geometry:
        return _not_stated()
    if geometry.get("description"):
        st.markdown(md(geometry["description"]))
    else:
        st.caption("Description not stated")
    dimension = geometry.get("physical_dimension")
    extra = [("Dimension", f"{dimension}D" if dimension else None)]
    extra += [(name, value) for name, value in (("Template", geometry.get("template")),
                                                ("Hole role", words(geometry.get("hole_role"))),
                                                ("Source", geometry.get("authoritative_source"))) if value]
    _fields(extra)
    parameters = geometry.get("parameters") or {}
    if parameters:
        rows = []
        for name, item in parameters.items():
            value, unit = readable((item or {}).get("value"), (item or {}).get("unit"))
            rows.append({"Parameter": capital(words(name)) or name, "Value": value, "Unit": unit})
        st.table(rows, hide_index=True)
    else:
        st.caption("No dimensions stated")
    regions = geometry.get("regions") or []
    if regions:
        _bullets([f"Region {md(r['name'])}: {md(r.get('description') or 'no description')}"
                  + (f" ({md(r['role'])})" if r.get("role") else "") for r in regions])


def _materials(materials: list[dict]) -> None:
    _heading("Materials")
    if not materials:
        return _not_stated()
    lines = []
    for index, material in enumerate(materials, 1):
        properties = ", ".join(f"{md(name)} = {_or(quantity(item), 'not stated')}"
                               for name, item in (material.get("properties") or {}).items())
        name = material.get("name") or f"Material {index}"
        line = (f"{md(name)} in {_or(material.get('region'), 'region not stated')}: "
                f"{properties or '*properties not stated*'}")
        if material.get("data_source"):
            line += f" (source: {md(material['data_source'])})"
        lines.append(line)
    _bullets(lines)


def _supports(conditions: list[dict]) -> None:
    _heading("Supports")
    if not conditions:
        return _not_stated()
    lines = []
    for condition in conditions:
        line = (f"{md(condition['name'])}: {_or(words(condition.get('kind')), 'type not stated')} at "
                f"{_or(condition.get('location'), 'location not stated')}")
        if condition.get("components"):
            line += f", components {md(', '.join(map(str, condition['components'])))}"
        if quantity(condition.get("value")):
            line += f", value {md(quantity(condition['value']))}"
        lines.append(line)
    _bullets(lines)


def _loads(loads: list[dict]) -> None:
    _heading("Loads")
    if not loads:
        return _not_stated()
    lines = []
    for load in loads:
        line = (f"{md(load['name'])}: {_or(words(load.get('kind')), 'type not stated')} of "
                f"{_or(quantity(load.get('magnitude')), 'magnitude not stated')} at "
                f"{_or(load.get('location'), 'location not stated')}")
        if load.get("direction"):
            line += f", direction {md(load['direction'])}"
        if load.get("load_case"):
            line += f", load case {md(load['load_case'])}"
        lines.append(line)
    _bullets(lines)


def _objective(optimization: dict | None) -> None:
    _heading("Objective")
    objectives = (optimization or {}).get("objectives") or []
    if not objectives:
        _not_stated()
    else:
        _bullets([f"{_or(capital(words(o.get('sense'))), 'sense not stated')} "
                  f"{_or(words(o.get('quantity')), 'quantity not stated')}"
                  + (f" ({md(words(o['aggregation']))})" if words(o.get("aggregation")) else "")
                  for o in objectives])
    if optimization and words(optimization.get("design_variable")):
        st.caption(f"Design variable: {md(words(optimization['design_variable']))}")


def _constraints(optimization: dict | None) -> None:
    _heading("Constraints")
    constraints = (optimization or {}).get("constraints") or []
    if not constraints:
        _not_stated()
    else:
        _bullets([f"{_or(capital(words(c.get('quantity'))), 'quantity not stated')} "
                  f"{_RELATIONS.get(c.get('relation'), '*relation not stated*')} "
                  f"{_or(quantity(c.get('limit')), 'limit not stated')}"
                  + (f" in {md(c['region'])}" if c.get("region") else "")
                  for c in constraints])
    requirement = (optimization or {}).get("stress_requirement")
    if words(requirement):
        st.caption(f"Stress requirement: {md(_STRESS.get(requirement, words(requirement)))}")


def _manufacturing(manufacturing: dict | None) -> None:
    if not manufacturing or not (manufacturing.get("process") or manufacturing.get("requirements")):
        return
    _heading("Manufacturing")
    if manufacturing.get("process"):
        st.markdown(f"Process: {md(manufacturing['process'])}")
    if manufacturing.get("requirements"):
        _bullets([md(r) for r in manufacturing["requirements"]])


def _assumptions(assumptions: list[str]) -> None:
    if assumptions:
        _heading("Assumptions")
        _bullets([md(a) for a in assumptions])


def render(spec) -> None:
    """Compact summary of a ProblemSpec (model or dict); missing core parts read "not stated"."""
    data = _dump(spec)
    _problem(data)
    _physics(data.get("physics"))
    _geometry(data.get("geometry"))
    _materials(data.get("materials") or [])
    _supports(data.get("boundary_conditions") or [])
    _loads(data.get("loads") or [])
    _objective(data.get("optimization"))
    _constraints(data.get("optimization"))
    _manufacturing(data.get("manufacturing"))
    _assumptions(data.get("assumptions") or [])
