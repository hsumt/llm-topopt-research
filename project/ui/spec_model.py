"""The specification view model: each part of a spec as a row tagged with its role for the connected solver.

Roles: input (the value goes into the solver config), checked (a choice the solver requires; no
number), default (fixed by the codebase, not in the spec), missing (the check needs a decision or
data), unsupported (no solver capability), context (the solver does not read it). Per row, a
code-check issue on one of the row's spec paths decides first, then a solver input, then a checked
choice. components/spec_card.py draws the sections; every text here is shown as text, never HTML.
"""
import re

from project.lbracket import GEOMETRY_ORDER, STRESS_QUANTITY, VOLUME_QUANTITY
from project.models import CODE_KEY_PREFIX
from project.ui.api import UNPARSED_NAME
from project.ui.components.spec_view import capital, number, quantity, readable, words

NOT_STATED = "not stated"
ROLES = ("input", "checked", "default", "missing", "unsupported")   # the roles a chip shows, in legend order
_RELATIONS = {"<=": "≤", ">=": "≥", "=": "="}
_STRESS = {"volume_only": "volume constraint only", "pnorm_limit": "p-norm stress limit"}
_QUANTITIES = {"stress_p_norm": "Stress p-norm"}   # constraint labels that plain words would misspell
_SI_UNITS = (("_m", "m"), ("_pa", "Pa"), ("_n", "N"))   # config field suffix -> its SI unit
_ISSUE_CHIP = {"decision": "decision", "data": "data needed", "unsupported": "unsupported"}
# Material property keys the check accepts (its aliases), with their label and canonical key.
_PROPERTIES = {"E": ("Young's modulus E", "E"), "youngs_modulus": ("Young's modulus E", "E"),
               "nu": ("Poisson's ratio ν", "nu"), "poisson_ratio": ("Poisson's ratio ν", "nu")}
# Where an issue that matches no row is listed: the first section whose path prefix it starts with.
_ISSUE_SECTIONS = (("problem_kind", "problem"), ("physics", "physics"), ("geometry", "geometry"),
                   ("materials", "materials"), ("boundary_conditions", "supports"), ("loads", "loads"),
                   ("optimization.objectives", "objective"), ("optimization.design_variable", "objective"),
                   ("optimization", "constraints"), ("manufacturing", "manufacturing"), ("assumptions", "assumptions"))


def _text(value) -> str | None:
    """Plain words for an identifier, sentence case, with 2d/3d written 2D/3D."""
    return re.sub(r"\b([123])d\b", r"\1D", capital(words(value)) or "") or None


def _shown(item: dict | None) -> tuple[str | None, str]:
    """(value, unit) of a spec Value for display; (None, "") when it has no value."""
    item = item or {}
    return readable(item["value"], item.get("unit")) if item.get("value") is not None else (None, "")


def _row(label, value=None, unit="", paths=(), wide=False, requires=(), inputs=None) -> dict:
    """A row before its role is known. paths: the spec paths it shows (issues and checks match
    any); inputs: the paths a solver input may come from (default: paths); requires: CHECKED
    entries that only add to an input's tooltip."""
    absent = value is None or value == ""
    return {"label": label, "value": NOT_STATED if absent else str(value), "unit": unit or "",
            "paths": list(dict.fromkeys(paths)), "inputs": list(inputs if inputs is not None else dict.fromkeys(paths)),
            "wide": wide, "absent": absent, "requires": list(requires)}


def _setting(source: dict) -> str:
    """Tooltip of a solver input: the config field and the SI value it receives."""
    field, value = source["field"], source["value"]
    if value is None:
        return (f"{field} = none: no stress constraint" if field == "stress_limit_pa" and source.get("built")
                else f"Goes to {field}; the configuration is built once nothing blocks.")
    unit = next((u for suffix, u in _SI_UNITS if field.endswith(suffix)), "")
    return f"{field} = {number(value)}{' ' + unit if unit else ''}"


def _problem(spec: dict) -> list[dict]:
    name = spec.get("name")
    return [_row("Name", None if name in (None, "", UNPARSED_NAME) else name),
            _row("Type", _text(spec.get("problem_kind")), paths=["problem_kind"])]


def _physics(physics: dict | None) -> list[dict]:
    if not physics:
        return [_row("Physics", paths=["physics"])]
    return [_row("Field", _text(physics.get("family")), paths=["physics.family", "physics"]),
            _row("Regime", _text(physics.get("regime")), paths=["physics.regime"]),
            _row("Representation", _text(physics.get("representation")), paths=["physics.representation"])]


def _geometry(geometry: dict | None, issues: dict) -> list[dict]:
    if not geometry:
        return [_row("Geometry", paths=["geometry", "geometry.template"])]
    dimension = geometry.get("physical_dimension")
    rows = [_row("Description", geometry["description"], wide=True)] if geometry.get("description") else []
    rows += [_row("Template", geometry.get("template"), paths=["geometry.template"]),
             _row("Dimension", f"{dimension}D" if dimension else None, paths=["geometry.physical_dimension"]),
             _row("Hole role", _text(geometry.get("hole_role")), paths=["geometry.hole_role"])]
    if geometry.get("authoritative_source"):
        rows.append(_row("Source", geometry["authoritative_source"], wide=True))
    parameters = geometry.get("parameters") or {}
    # The template's dimensions in its order, an absent one only when the check asks for it; then the rest.
    names = [n for n in GEOMETRY_ORDER if n in parameters or f"geometry.parameters.{n}" in issues]
    for name in names + [n for n in parameters if n not in GEOMETRY_ORDER]:
        value, unit = _shown(parameters.get(name))
        rows.append(_row(capital(words(name)) or name, value, unit, paths=[f"geometry.parameters.{name}"]))
    for region in geometry.get("regions") or []:
        rows.append(_row(f"Region {region['name']}", " · ".join(
            part for part in (region.get("description"), _text(region.get("role"))) if part)))
    return rows


def _materials(materials: list[dict], issues: dict) -> list[dict]:
    if not materials:
        return [_row("Material", paths=["materials"])]
    single, rows = len(materials) == 1, []
    for index, material in enumerate(materials):
        at = f"materials.{index}" if single else None   # the check reads a material only when it is the only one
        prefix = "" if single else f"{index + 1}. "
        rows += [_row(prefix + "Name", material.get("name"), paths=[f"{at}.name"] if at else []),
                 _row(prefix + "Region", _text(material.get("region")), paths=[f"{at}.region"] if at else [])]
        properties, constants = material.get("properties") or {}, []   # constants: (canonical key, row)
        for key, item in properties.items():
            label, canonical = _PROPERTIES.get(key, (capital(words(key)) or key, key))
            constants.append((canonical, _row(prefix + label, *_shown(item), paths=[
                f"{at}.properties#{key}", f"{at}.properties.{key}", f"{at}.properties.{canonical}"] if at else [],
                inputs=[f"{at}.properties.{key}"] if at else [])))
        if at:   # the two constants the solver needs, shown as absent when the check asks for them
            stated = {canonical for canonical, _ in constants}
            constants += [(canonical, _row(label, paths=[f"{at}.properties.{canonical}"])) for canonical, label in
                          (("E", "Young's modulus E"), ("nu", "Poisson's ratio ν"))
                          if canonical not in stated and f"{at}.properties.{canonical}" in issues]
        order = {"E": 0, "nu": 1}   # E, then ν, then anything else as written
        rows += [row for _, row in sorted(constants, key=lambda pair: order.get(pair[0], 2))]
        if material.get("data_source"):
            rows.append(_row(prefix + "Source", material["data_source"], wide=True))
    return rows


def _supports(conditions: list[dict]) -> list[dict]:
    if not conditions:
        return [_row("Support", paths=["boundary_conditions"])]
    single, rows = len(conditions) == 1, []
    for index, condition in enumerate(conditions):
        parts = [_text(condition.get("kind")), condition.get("location"), ", ".join(map(str, condition.get("components") or []))]
        if quantity(condition.get("value")):
            parts.append("value " + quantity(condition["value"]))
        paths = [f"boundary_conditions.{index}.value"] + (["boundary_conditions.0", "boundary_conditions"] if single else [])
        rows.append(_row(condition.get("name") or f"Support {index + 1}", " · ".join(p for p in parts if p) or None,
                         paths=paths))
    return rows


def _loads(loads: list[dict]) -> list[dict]:
    if not loads:
        return [_row("Load", paths=["loads"])]
    single, rows = len(loads) == 1, []
    for index, load in enumerate(loads):
        value, unit = _shown(load.get("magnitude"))
        if load.get("direction"):
            unit = " · ".join(part for part in (unit, str(load["direction"])) if part)
        at = [f"loads.{index}.magnitude", f"loads.{index}"] if single else []
        rows.append(_row(load.get("name") or f"Load {index + 1}", value, unit, paths=at))
        where = " · ".join(part for part in (load.get("location"), words(load.get("kind"))) if part)
        rows.append(_row("Applied at", where or None, paths=["loads.0.location", "loads.0.kind", "loads"] if single else []))
        if load.get("load_case"):
            rows.append(_row("Load case", load["load_case"]))
    return rows


def _objective(optimization: dict | None) -> list[dict]:
    if not optimization:
        return [_row("Objective", paths=["optimization"])]
    objectives = optimization.get("objectives") or []
    rows = [_row("Objective", paths=["optimization.objectives"])] if not objectives else []
    for index, objective in enumerate(objectives):
        sense, measure = _text(objective.get("sense")), words(objective.get("quantity"))
        text = (f"{sense or 'sense not stated'} {measure or 'quantity not stated'}"
                + (f" ({words(objective['aggregation'])})" if words(objective.get("aggregation")) else ""))
        rows.append(_row("Objective" if len(objectives) == 1 else f"Objective {index + 1}",
                         text if sense or measure else None,
                         paths=[f"optimization.objectives.{index}", "optimization.objectives"]))
    rows.append(_row("Design variable", _text(optimization.get("design_variable")), paths=["optimization.design_variable"]))
    return rows


def _constraints(optimization: dict) -> list[dict]:
    constraints = optimization.get("constraints") or []
    rows = [_row("Constraints", paths=["optimization.constraints"])] if not constraints else []
    for index, constraint in enumerate(constraints):
        value, unit = _shown(constraint.get("limit"))
        if constraint.get("region"):
            unit = " ".join(part for part in (unit, "over", constraint["region"]) if part)
        measure = constraint.get("quantity")
        relation = _RELATIONS.get(constraint.get("relation"), "?")
        if value is None:   # "not stated · ≤ limit over l_domain"
            unit = " ".join(part for part in ("·", relation, "limit", unit) if part)
        rows.append(_row(_QUANTITIES.get(measure) or _text(measure) or f"Constraint {index + 1}",
                         f"{relation} {value}" if value is not None else None, unit,
                         paths=[f"optimization.constraints.{index}.limit"],
                         requires=["optimization.constraints.*.relation", "optimization.constraints.*.region"]))
    requirement = optimization.get("stress_requirement")
    rows.append(_row("Stress requirement", _STRESS.get(requirement, words(requirement)),
                     paths=["optimization.stress_requirement"]))
    return rows


def _token(value) -> str:
    """The check's own normalisation of a vocabulary word (lbracket._token)."""
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _constraint_rows(spec: dict) -> dict[str, list[str]]:
    """Row paths of the constraints each code-check constraint issue concerns, by issue key. The check
    reads the first volume and the first stress constraint; every other one is unsupported."""
    constraints = ((spec.get("optimization") or {}).get("constraints")) or []
    volume = stress = None
    others = []
    for index, constraint in enumerate(constraints):
        measure, path = _token(constraint.get("quantity")), f"optimization.constraints.{index}.limit"
        if measure == VOLUME_QUANTITY and volume is None:
            volume = path
        elif measure == STRESS_QUANTITY and stress is None:
            stress = path
        else:
            others.append(path)
    rows = {CODE_KEY_PREFIX + key: [volume] for key in ("volume_reference", "volume_relation", "volume_limit") if volume}
    rows.update({CODE_KEY_PREFIX + key: [stress] for key in ("stress_contract", "stress_data") if stress})
    rows[CODE_KEY_PREFIX + "constraint"] = others
    return rows


def _expand(issues: list[dict], spec: dict) -> dict[str, list[dict]]:
    """Issues by spec path. Some issues are filed at a parent path; each is spread onto the rows it
    concerns: absent or extra dimensions (geometry.parameters), unknown or duplicate material
    properties (named at the end of the issue text), and constraint issues (by issue key)."""
    by_path: dict[str, list[dict]] = {}
    parameters = ((spec.get("geometry") or {}).get("parameters")) or {}
    constraint_rows, unsupported_constraints = _constraint_rows(spec), 0
    for issue in issues:
        field, key = issue.get("field") or "", issue.get("key")
        paths = [field]
        if field == "geometry.parameters":
            names = ([n for n in parameters if n not in GEOMETRY_ORDER] if issue["kind"] == "unsupported"
                     else [n for n in GEOMETRY_ORDER if n not in parameters])
            paths = [f"geometry.parameters.{n}" for n in names] or paths
        elif field == "materials.0.properties" and ": " in issue["description"]:
            # "#key": the row with exactly that key, never the canonical .E/.nu an accepted alias row also carries
            paths = ["materials.0.properties#" + issue["description"].rsplit(": ", 1)[1]]
        elif key == CODE_KEY_PREFIX + "constraint":   # one issue per extra constraint, in order
            others = constraint_rows[key]
            paths = others[unsupported_constraints:unsupported_constraints + 1] or paths
            unsupported_constraints += 1
        elif constraint_rows.get(key):
            paths = constraint_rows[key]
        for path in paths:
            by_path.setdefault(path, []).append(issue)
    return by_path


def _resolve(row: dict, roles: dict, issues: dict, used: set) -> dict:
    """The row's role, chip and tooltip, by precedence: issue > solver input > checked > context."""
    hits = [i for path in row["paths"] for i in issues.get(path, [])]
    sources, checked = roles.get("sources") or {}, roles.get("checked") or {}
    source = next((sources[p] for p in row["inputs"] if p in sources), None)
    phrases = [checked[p] for p in row["paths"] if p in checked]
    if hits:
        used.update(map(id, hits))
        unsupported = [i for i in hits if i["kind"] == "unsupported"]
        role, chip = ("unsupported", "unsupported") if unsupported else ("missing", _ISSUE_CHIP.get(hits[0]["kind"], "decision"))
        tip = " ".join(dict.fromkeys(i["description"] for i in (unsupported or hits)))
    elif source is not None:
        requires = [checked[p] for p in row["requires"] if p in checked]
        role, chip = "input", source["field"]
        tip = _setting({**source, "built": roles.get("config_built")})
        if requires:
            tip = tip.rstrip(".") + f". The check requires {' and '.join(requires)}."
    elif phrases and not row["absent"]:
        role, chip, tip = "checked", phrases[0], f"Checked: the connected solver requires {'; '.join(phrases)}."
    elif phrases and roles.get("connected"):
        # Left unset and accepted by the check: the solver applies its own choice.
        role, chip, tip = "default", phrases[0], f"Not in the specification; the solver uses {phrases[0]}."
    else:
        role, chip, tip = "context", "", "Not stated." if row["absent"] else "Not read by the solver."
    return {"label": row["label"], "value": row["value"], "unit": row["unit"], "role": role, "chip": chip,
            "tip": tip, "wide": row["wide"] and role == "context"}


def _issue_row(issue: dict) -> dict:
    """A row for an issue that matches no row: labelled by the last named part of its path."""
    names = [part for part in (issue.get("field") or "").split(".") if part and not part.isdigit()]
    unsupported = issue["kind"] == "unsupported"
    return {"label": capital(words(names[-1])) if names else "Specification", "value": issue["description"], "unit": "",
            "role": "unsupported" if unsupported else "missing",
            "chip": "unsupported" if unsupported else _ISSUE_CHIP.get(issue["kind"], "decision"),
            "tip": issue["description"], "wide": False}


def build(spec: dict, roles: dict) -> list[dict]:
    """Sections of a spec (its JSON dict) for the specification card, given api.spec_roles(spec)."""
    spec = spec or {}
    issues = _expand(roles.get("issues") or [], spec)
    # Source values are None until the config builds (stress_limit_pa stays None in a volume-only one).
    roles = {**roles, "config_built": any(s["value"] is not None for s in (roles.get("sources") or {}).values())}
    materials, conditions, loads = spec.get("materials") or [], spec.get("boundary_conditions") or [], spec.get("loads") or []
    optimization = spec.get("optimization")
    manufacturing = spec.get("manufacturing") or {}
    plural = lambda items, one, many: many if len(items) > 1 else one
    drafts = [  # (id, title, Material icon, rows)
        ("problem", "Problem", "assignment", _problem(spec)),
        ("physics", "Physics", "science", _physics(spec.get("physics"))),
        ("geometry", "Geometry", "category", _geometry(spec.get("geometry"), issues)),
        ("materials", plural(materials, "Material", "Materials"), "layers", _materials(materials, issues)),
        ("supports", plural(conditions, "Support", "Supports"), "anchor", _supports(conditions)),
        ("loads", plural(loads, "Load", "Loads"), "south", _loads(loads)),
        ("objective", "Objective", "flag", _objective(optimization)),
    ]
    if optimization:
        drafts.append(("constraints", "Constraints", "rule", _constraints(optimization)))
    if manufacturing.get("process") or manufacturing.get("requirements"):
        drafts.append(("manufacturing", "Manufacturing", "precision_manufacturing",
                       ([_row("Process", manufacturing["process"], paths=["manufacturing"])] if manufacturing.get("process") else [])
                       + [_row("Requirement", text, paths=["manufacturing"]) for text in manufacturing.get("requirements") or []]))
    if spec.get("assumptions"):
        drafts.append(("assumptions", "Assumptions", "lightbulb",
                       [_row("Assumption", text, paths=[f"assumptions.{n}", "assumptions"])
                        for n, text in enumerate(spec["assumptions"])]))
    used: set = set()
    sections = {key: {"title": title, "icon": icon, "rows": [_resolve(row, roles, issues, used) for row in rows]}
                for key, title, icon, rows in drafts}
    for issue in roles.get("issues") or []:
        if id(issue) not in used:
            field = issue.get("field") or ""
            key = next((key for prefix, key in _ISSUE_SECTIONS if field == prefix or field.startswith(prefix + ".")), "problem")
            sections.get(key, sections["problem"])["rows"].append(_issue_row(issue))
    if roles.get("connected"):
        sections["defaults"] = {"title": "Fixed by the solver (not in the specification)", "icon": "tune", "rows": [
            {"label": item["label"], "value": number(item["value"]), "unit": "", "role": "default", "chip": item["field"],
             "tip": f"{item['field']} = {number(item['value'])}: set by the solver, not by the specification.", "wide": False}
            for item in roles.get("defaults") or []]}
    for section in sections.values():
        found = [row["role"] for row in section["rows"]]
        section["counts"] = {role: found.count(role) for role in ROLES if role in found}
    return list(sections.values())
