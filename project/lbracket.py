"""Deterministic contract between the engineering specification and the 3D solver.

The template is intentionally narrow. An unrepresented requirement blocks a run;
neither a quiet critic nor an inferred benchmark default grants solver readiness.
"""

from dataclasses import dataclass
import math

from project.models import Issue, ProblemSpec, Question, Review, Value
from project.solver.lbracket3d.config import LBracket3DConfig, reference_config


TEMPLATE = "lbracket3d_five_holes"
GEOMETRY_FIELDS = {"lx", "ly", "thickness", "cut_length", "load_patch", "hole_radius", "hole_centers"}
_LENGTH = {"m": 1., "mm": .001, "cm": .01, "in": .0254, "inch": .0254, "inches": .0254}
_STRESS = {"pa": 1., "kpa": 1e3, "mpa": 1e6, "gpa": 1e9, "psi": 6894.757293168}
_FORCE = {"n": 1., "kn": 1e3, "lbf": 4.4482216152605}


@dataclass
class SpecAssessment:
    config: LBracket3DConfig | None
    issues: list[Issue]
    questions: list[Question]

    @property
    def ready(self) -> bool:
        return self.config is not None and not any(i.blocking for i in self.issues)


def _token(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("must be a finite number")
    return float(value)


def _factor(value: Value, factors: dict[str, float]) -> float:
    unit = (value.unit or "").strip().lower()
    if unit not in factors:
        raise ValueError("requires explicit supported units: " + ", ".join(factors))
    return factors[unit]


def _scalar(value: Value, factors: dict[str, float]) -> float:
    return _number(value.value) * _factor(value, factors)


def _fraction(value: Value) -> float:
    unit = (value.unit or "").strip().lower()
    if unit not in {"", "1", "dimensionless", "%", "percent"}:
        raise ValueError("requires a dimensionless fraction or percent")
    return _number(value.value) / (100 if unit in {"%", "percent"} else 1)


def assess_spec(spec: ProblemSpec) -> SpecAssessment:
    """Return a config only if every required decision is represented and valid."""
    issues: list[Issue] = []
    questions: list[Question] = []
    config_values: dict = {}

    def add(key, field, description, *, kind="decision", question=None):
        key = "solver_" + key
        issues.append(Issue(key=key, field=field, description=description, kind=kind))
        if question:
            questions.append(Question(key=key, issue_keys=[key], prompt=question,
                                      why=description))

    if spec.problem_kind != "optimization":
        add("problem", "problem_kind", "This solver performs topology optimization only.",
            kind="decision" if spec.problem_kind == "unknown" else "unsupported",
            question="What should be optimized, and what should the objective and constraints be?" if spec.problem_kind == "unknown" else None)

    geometry = spec.geometry
    if geometry is None or geometry.template is None:
        add("template", "geometry.template", "A solver-representable geometry has not been selected.",
            question="Is the geometry an equal-arm L with an upper-right square cut and five movable initial through-holes, or do any holes need to remain fixed?")
    elif geometry.template != TEMPLATE:
        add("template", "geometry.template", "Only the lbracket3d_five_holes geometry template is supported.", kind="unsupported")
    if geometry is not None:
        if geometry.physical_dimension is None:
            add("dimension", "geometry.physical_dimension", "The physical geometry dimension is missing.",
                question="Is this a physical 3D bracket with a through-thickness shape?")
        elif geometry.physical_dimension != 3:
            add("dimension", "geometry.physical_dimension", "This backend represents a 3D bracket.", kind="unsupported")
        if geometry.hole_role is None:
            add("hole_role", "geometry.hole_role", "Initial holes and preserved mounting holes are different requirements.",
                question="May the five initial holes move, merge, or close during optimization, or must they remain as mounting holes?" if geometry.template is not None else None)
        elif geometry.hole_role != "initial_void":
            add("hole_role", "geometry.hole_role", "This backend supports movable initial holes; preserved holes require a different geometry implementation.", kind="unsupported")
        extras = set(geometry.parameters) - GEOMETRY_FIELDS
        if extras:
            add("geometry_extra", "geometry.parameters", "Unrepresented geometry parameters: " + ", ".join(sorted(extras)), kind="unsupported")
        allowed_regions = {"top_arm": "support", "tip_band": "load", "initial_holes": "initial_void", "corner_cut": "passive_void", "domain": "design"}
        if any(allowed_regions.get(r.name) != r.role for r in geometry.regions):
            add("regions", "geometry.regions", "Additional protected, mounting, or other geometry regions are not represented by this template.", kind="unsupported")
        missing = sorted(GEOMETRY_FIELDS - geometry.parameters.keys())
        if missing:
            labels = {"lx": "outer x length", "ly": "outer y length", "thickness": "thickness",
                      "cut_length": "square cut size", "load_patch": "tip load-band height",
                      "hole_radius": "initial hole radius", "hole_centers": "five initial hole x/y centers"}
            missing_text = ", ".join(labels[name] for name in missing)
            source_note = (" The governing source is identified; CAD/drawing import is not implemented. Supply extracted, validated dimensional data from that source." if geometry.authoritative_source else "")
            add("dimensions", "geometry.parameters", "Missing physical geometry inputs: " + missing_text + "." + source_note,
                kind="data" if geometry.authoritative_source else "decision",
                question=None if geometry.authoritative_source else "What are the governing " + missing_text + ", with units? If a drawing governs them, identify that data source.")
        for name in GEOMETRY_FIELDS & geometry.parameters.keys():
            value = geometry.parameters[name]
            try:
                if name == "hole_centers":
                    factor = _factor(value, _LENGTH)
                    if not isinstance(value.value, (list, tuple)) or len(value.value) != 5:
                        raise ValueError("requires five x/y coordinate pairs")
                    centers = []
                    for center in value.value:
                        if not isinstance(center, (list, tuple)) or len(center) != 2:
                            raise ValueError("requires five x/y coordinate pairs")
                        centers.append(tuple(_number(v) * factor for v in center))
                    config_values["hole_centers_m"] = tuple(centers)
                else:
                    config_values[name + "_m"] = _scalar(value, _LENGTH)
            except ValueError as error:
                add("geometry_" + name, "geometry.parameters." + name, name + " " + str(error), kind="data")

    physics = spec.physics
    if physics is None or physics.family == "unknown":
        add("physics", "physics", "The governing physics and loading regime are unresolved.",
            question="Is the intended analysis static small-strain linear elasticity, or do inertia, large deformation, or other physics matter?")
    elif physics.family != "solid_mechanics":
        add("physics", "physics.family", "Only solid mechanics is implemented.", kind="unsupported")
    if physics is not None:
        if physics.regime is None:
            add("regime", "physics.regime", "A static linear-elastic regime has not been established.",
                question="Can the loading be treated as static and the material response as small-strain linear elasticity?")
        elif _token(physics.regime) not in {"linear_static", "static_linear_elasticity", "static_linear_elastic"}:
            add("regime", "physics.regime", "Only static small-strain linear elasticity is implemented.", kind="unsupported")
        # The adapter chooses the solid discretization, unless the engineer asks for another model.
        if physics.representation is not None and _token(physics.representation) not in {"3d_solid", "3d_solid_continuum", "three_dimensional_solid"}:
            add("representation", "physics.representation", "The requested analysis representation is not supported by the 3D solid solver.", kind="unsupported")

    if not spec.materials:
        add("material", "materials", "The material has not been identified.",
            question="Which material grade and authoritative material data should govern the bracket?")
    elif len(spec.materials) != 1:
        add("materials", "materials", "Exactly one uniform isotropic material is supported.", kind="unsupported")
    else:
        material = spec.materials[0]
        if not material.name and not material.properties:
            add("material", "materials.0.name", "The material identity and its governing data source are unresolved.",
                question="Which material grade and authoritative material data should govern the bracket?")
        if material.region not in {None, "domain", "entire_domain"}:
            add("material_region", "materials.0.region", "A material assigned to a subregion is not supported.", kind="unsupported")
        aliases = {"E": "E", "youngs_modulus": "E", "nu": "nu", "poisson_ratio": "nu"}
        properties = {}
        for key, val in material.properties.items():
            canonical = aliases.get(key)
            if canonical is None or canonical in properties:
                add("material_properties", "materials.0.properties", "Unknown or duplicate material property: " + key, kind="unsupported")
            else:
                properties[canonical] = val
        for name, target in (("E", "youngs_modulus_pa"), ("nu", "poisson_ratio")):
            if name not in properties:
                add("material_" + name, "materials.0.properties." + name,
                    "Authoritative material data required: " + ("Young's modulus with units." if name == "E" else "Poisson's ratio."), kind="data")
            else:
                try:
                    config_values[target] = _scalar(properties[name], _STRESS) if name == "E" else _fraction(properties[name])
                except ValueError as error:
                    add("material_" + name, "materials.0.properties." + name, name + " " + str(error), kind="data")

    if not spec.boundary_conditions:
        add("support", "boundary_conditions", "The mounting behavior has not been specified.",
            question="How is the top end of the vertical arm attached? This solver can restrain all three translations across that entire top face.")
    elif len(spec.boundary_conditions) != 1:
        add("support", "boundary_conditions", "Only one full top-arm clamp is supported.", kind="unsupported")
    else:
        bc = spec.boundary_conditions[0]
        if not bc.kind or not bc.location or not bc.components:
            add("support", "boundary_conditions", "The support location or translational restraint behavior is missing.",
                question="Should the entire top face of the vertical arm be clamped in x, y, and z, or does the real interface behave differently?")
        elif bc.location != "top_arm" or _token(bc.kind) not in {"clamped", "fixed"} or sorted(bc.components) != ["x", "y", "z"]:
            add("support", "boundary_conditions", "Only a top_arm clamp restraining x, y and z is supported.", kind="unsupported")
        if bc.value is not None:
            try:
                if _scalar(bc.value, _LENGTH) != 0:
                    raise ValueError("nonzero displacement")
            except ValueError:
                add("support_value", "boundary_conditions.0.value", "Only zero prescribed displacement is supported.", kind="unsupported")

    if not spec.loads:
        add("load", "loads", "The load is missing.",
            question="What total force (magnitude and direction or x/y/z components, with units) acts on the horizontal-arm tip band?")
    elif len(spec.loads) != 1:
        add("loads", "loads", "This backend supports exactly one load vector and one load case; multiple loads cannot be silently combined.", kind="unsupported")
    else:
        load = spec.loads[0]
        if load.location is None or load.kind is None:
            add("load_model", "loads", "The force distribution and location are unresolved.",
                question="Is this a total force uniformly distributed over the horizontal-arm tip band through the full thickness?")
        elif load.location != "tip_band" or _token(load.kind) != "total_force":
            add("load_model", "loads", "Only a total_force distributed over tip_band is supported; point loads, pressures and other locations are different specifications.", kind="unsupported")
        if load.magnitude is None:
            add("force", "loads.0.magnitude", "The applied force has no magnitude.",
                question="Provide the total force and its x/y/z direction with units.")
        else:
            try:
                factor = _factor(load.magnitude, _FORCE)
                raw = load.magnitude.value
                if isinstance(raw, (list, tuple)):
                    if len(raw) != 3 or load.direction not in {None, "vector"}:
                        raise ValueError("requires three components and no conflicting direction")
                    force = tuple(_number(v) * factor for v in raw)
                else:
                    magnitude = _number(raw) * factor
                    direction = {"+x": (1, 0, 0), "x": (1, 0, 0), "-x": (-1, 0, 0),
                                 "+y": (0, 1, 0), "y": (0, 1, 0), "-y": (0, -1, 0),
                                 "+z": (0, 0, 1), "z": (0, 0, 1), "-z": (0, 0, -1)}.get(load.direction)
                    if direction is None:
                        raise ValueError("requires an explicit coordinate direction or force vector")
                    if magnitude <= 0:
                        raise ValueError("scalar force magnitude must be positive; use direction for sign")
                    force = tuple(magnitude * v for v in direction)
                config_values["force_n"] = force
            except ValueError as error:
                add("force", "loads.0", "Force " + str(error),
                    question="Provide the applied force as [Fx, Fy, Fz] and its force units.")

    optimization = spec.optimization
    if optimization is None:
        add("optimization", "optimization", "Objective and constraints are missing.",
            question="Should compliance be minimized with what volume fraction of the L domain, and is a stress constraint required?")
    else:
        if optimization.design_variable not in {None, "material_distribution", "level_set"}:
            add("design_variable", "optimization.design_variable", "Only material distribution with a level-set boundary is supported.", kind="unsupported")
        objectives = optimization.objectives
        if not objectives:
            add("objective", "optimization.objectives", "The optimization objective is missing.", question="Is minimum compliance the design objective?")
        elif len(objectives) == 1 and (objectives[0].sense is None or objectives[0].quantity is None):
            add("objective", "optimization.objectives", "The objective quantity or optimization sense is unresolved.", question="What quantity should be minimized or maximized?")
        elif len(objectives) != 1 or objectives[0].sense != "minimize" or _token(objectives[0].quantity) != "compliance" or objectives[0].aggregation not in {None, "single_load_case"}:
            add("objective", "optimization.objectives", "Only minimum compliance for one load case is supported.", kind="unsupported")
        volume_seen = stress_seen = False
        for constraint in optimization.constraints:
            quantity = _token(constraint.quantity)
            if quantity == "volume_fraction" and not volume_seen:
                volume_seen = True
                if constraint.region != "l_domain":
                    add("volume_reference", "optimization.constraints", "The volume fraction must refer explicitly to the L domain, excluding the permanent cut.",
                        kind="decision" if constraint.region is None else "unsupported",
                        question="Is the material fraction measured relative to the full L-shaped domain (excluding the square cut), or a different reference volume?" if constraint.region is None else None)
                if constraint.relation != "<=":
                    add("volume_relation", "optimization.constraints", "Only an upper bound on volume fraction is supported.", kind="unsupported")
                if constraint.limit is None:
                    add("volume_limit", "optimization.constraints", "The material fraction is missing.", question="What maximum fraction of the full L domain may contain material?")
                else:
                    try:
                        config_values["volume_fraction"] = _fraction(constraint.limit)
                    except ValueError as error:
                        add("volume_limit", "optimization.constraints", "Volume fraction " + str(error), kind="data")
            elif quantity == "stress_p_norm" and not stress_seen:
                stress_seen = True
                if constraint.region != "l_domain" or constraint.relation != "<=":
                    add("stress_contract", "optimization.constraints", "Only the volume-averaged p=6 stress norm over the L domain with an upper bound is supported.", kind="unsupported")
                if constraint.limit is None:
                    add("stress_data", "optimization.constraints", "A calibrated stress p-norm limit with units is required; a material yield value is not interchangeable.", kind="data")
                else:
                    try:
                        config_values["stress_limit_pa"] = _scalar(constraint.limit, _STRESS)
                    except ValueError as error:
                        add("stress_data", "optimization.constraints", "Stress p-norm limit " + str(error), kind="data")
            else:
                add("constraint", "optimization.constraints", "Unsupported or duplicate constraint: " + str(constraint.quantity) + ". Peak/yield stress, displacement, buckling and manufacturing limits are not implemented.", kind="unsupported")
        if not volume_seen:
            add("volume", "optimization.constraints", "A volume fraction upper bound and its reference domain are required.",
                question="What maximum material fraction is permitted, and is it measured relative to the full L domain excluding the square cut?")
        if optimization.stress_requirement is None:
            add("stress_choice", "optimization.stress_requirement", "The engineer has not specified whether the problem is volume-only or includes a calibrated stress p-norm constraint.",
                question="Is this minimum compliance with only a volume limit, or do you also require a stress constraint? State any physical stress requirement as intended; this solver currently supports only a calibrated p=6 norm, not a peak/yield bound.")
        elif optimization.stress_requirement == "volume_only":
            if stress_seen:
                add("stress_conflict", "optimization", "The volume_only choice conflicts with an explicit stress constraint.", kind="unsupported")
            else:
                config_values["stress_limit_pa"] = None
        elif not stress_seen:
            add("stress_missing", "optimization.constraints", "The selected p-norm constraint needs a calibrated stress_p_norm upper bound with units.", kind="data")

    if spec.manufacturing is not None and (spec.manufacturing.process or spec.manufacturing.requirements):
        add("manufacturing", "manufacturing", "Manufacturing requirements are not enforced by this backend; they cannot be dropped from a requested solve.", kind="unsupported")
    if spec.assumptions:
        add("assumptions", "assumptions", "Additional assumptions must be expressed in the supported physics, geometry, material, load or constraint fields before this backend can enforce them.", kind="unsupported")
    config = None
    if not issues:
        try:
            config = LBracket3DConfig.from_dict(config_values)
        except (TypeError, ValueError) as error:
            add("config", "geometry", "Solver configuration is invalid: " + str(error), kind="data")
    return SpecAssessment(config=config, issues=issues, questions=questions[:4])


def combine_review(spec: ProblemSpec, review: Review) -> Review:
    """Deterministic issues survive any LLM verdict; questions remain bounded."""
    assessment = assess_spec(spec)
    model_issues = []
    known_material = len(spec.materials) == 1 and bool(spec.materials[0].name)
    for issue in review.issues:
        if issue.key.startswith("solver_"):
            continue
        text = " ".join((issue.key, issue.field or "", issue.description)).lower()
        # A model may disregard the prompt and ask for a guessed modulus. Keep
        # this as missing source data even when the model labels it a decision.
        if known_material and (".properties" in text or "young's modulus" in text or "youngs_modulus" in text or "poisson" in text):
            issue = issue.model_copy(update={"kind": "data"})
        model_issues.append(issue)
    issues = assessment.issues + model_issues
    questions = list(assessment.questions)
    decision_keys = {i.key for i in issues if i.blocking and i.kind == "decision"}
    seen = {q.key for q in questions}
    for question in review.questions:
        if len(questions) >= 4:
            break
        if question.key not in seen and set(question.issue_keys) & decision_keys:
            questions.append(question)
            seen.add(question.key)
    return Review(issues=issues, questions=questions)


def reference_spec() -> ProblemSpec:
    """Explicit benchmark selection; never merged into an incomplete user spec."""
    c = reference_config()
    parameters = {name: {"value": getattr(c, name + "_m"), "unit": "m"}
                  for name in GEOMETRY_FIELDS if name != "hole_centers"}
    parameters["hole_centers"] = {"value": c.hole_centers_m, "unit": "m"}
    return ProblemSpec.model_validate({
        "name": "3D L-bracket with five initial holes", "problem_kind": "optimization",
        "geometry": {"template": TEMPLATE, "physical_dimension": 3, "hole_role": "initial_void",
                     "description": "Equal-arm L with an upper-right square cut and five movable initial through-holes.",
                     "parameters": parameters},
        "physics": {"family": "solid_mechanics", "representation": "3d_solid", "regime": "linear_static"},
        "materials": [{"name": "Reference titanium model", "region": "domain", "data_source": "Explicit simplified_3D_holes benchmark constants",
                       "properties": {"E": {"value": c.youngs_modulus_pa, "unit": "Pa"},
                                      "nu": {"value": c.poisson_ratio}}}],
        "boundary_conditions": [{"name": "Top clamp", "location": "top_arm", "kind": "clamped", "components": ["x", "y", "z"]}],
        "loads": [{"name": "Tip-band force", "location": "tip_band", "kind": "total_force",
                   "magnitude": {"value": c.force_n, "unit": "N"}, "load_case": "reference"}],
        "optimization": {"design_variable": "material_distribution", "stress_requirement": "pnorm_limit",
                         "objectives": [{"sense": "minimize", "quantity": "compliance"}],
                         "constraints": [{"quantity": "volume_fraction", "relation": "<=", "region": "l_domain", "limit": {"value": c.volume_fraction}},
                                         {"quantity": "stress_p_norm", "relation": "<=", "region": "l_domain", "limit": {"value": c.stress_limit_pa, "unit": "Pa"}}]},
    })


def starter_spec() -> ProblemSpec:
    return ProblemSpec.model_validate({"name": "My 3D L-bracket", "problem_kind": "optimization",
                                      "geometry": {"physical_dimension": 3, "description": "L-bracket"},
                                      "physics": {"family": "solid_mechanics"}})
