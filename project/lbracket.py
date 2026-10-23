"""Deterministic contract between the engineering specification and the 2D plane-stress solver.

The template is intentionally narrow. An unrepresented requirement blocks a run;
neither a quiet critic nor an inferred benchmark default grants solver readiness.
"""

from dataclasses import dataclass, field, fields
import hashlib
import math
import sys

from project import trace
from project.models import CODE_KEY_PREFIX, Issue, ProblemSpec, Question, Review, Value
from project.solver.lbracket2d.config import LBracket2DConfig, reference_config


TEMPLATE = "lbracket2d_five_holes"
GEOMETRY_ORDER = ("lx", "ly", "thickness", "cut_length", "load_patch", "hole_radius", "hole_centers")
GEOMETRY_FIELDS = set(GEOMETRY_ORDER)
HOLES = 5
MAX_QUESTIONS = 4   # questions shown per round; code-check questions come first
# The names assess_spec accepts; prompts.py tells the model these exact names. The few
# aliases the check also accepts are written where it tests them.
HOLE_ROLE = "initial_void"
FAMILY, REGIME, REPRESENTATION = "solid_mechanics", "linear_static", "2d_plane_stress"
DOMAIN = "domain"   # the whole design domain, which the one material fills
SUPPORT_REGION, SUPPORT_KIND, SUPPORT_COMPONENTS = "top_arm", "clamped", ("x", "y")
LOAD_REGION, LOAD_KIND = "tip_band", "total_force"
FORCE_DIRECTIONS = {"+x": (1, 0), "x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "y": (0, 1), "-y": (0, -1)}
OBJECTIVE, DESIGN_VARIABLE = "compliance", "material_distribution"
VOLUME_QUANTITY, STRESS_QUANTITY, REFERENCE_REGION = "volume_fraction", "stress_p_norm", "l_domain"
VOLUME_ONLY, PNORM_LIMIT = "volume_only", "pnorm_limit"   # optimization.stress_requirement
# Named regions and the role each must carry.
REGIONS = {SUPPORT_REGION: "support", LOAD_REGION: "load", "initial_holes": "initial_void",
           "corner_cut": "passive_void", DOMAIN: "design"}
# Spec paths assess_spec compares with the vocabulary above, and what this solver requires there.
# The specification view shows them as checked choices; no number from them reaches the config.
# "*" stands for every constraint.
CHECKED = {
    "problem_kind": "topology optimization",
    "geometry.template": "five-hole L-bracket",
    "geometry.physical_dimension": "2D plate",
    "geometry.hole_role": "movable initial holes",
    "physics.family": "solid mechanics",
    "physics.regime": "linear static",
    "physics.representation": "2D plane stress",
    "materials.0.region": "whole design domain",
    "boundary_conditions.0": "top edge clamped in x, y",
    "loads.0.location": "horizontal-arm tip band",
    "loads.0.kind": "total force, uniform",
    "optimization.design_variable": "level-set material layout",
    "optimization.objectives.0": "minimum compliance",
    "optimization.stress_requirement": "volume only or p-norm limit",
    "optimization.constraints.*.region": "the L domain (l_domain)",
    "optimization.constraints.*.relation": "an upper bound (<=)",
}
# LBracket2DConfig fields no spec sets: the solver's own defaults, by label.
SOLVER_DEFAULTS = {"nelx": "Mesh elements along x", "p_norm": "Stress p-norm exponent",
                   "rho_min": "Minimum density", "move_limit": "Move limit (cells per iteration)",
                   "perturbation": "Initial hole perturbation (cells)"}
# The answers to an assumption question. The resolver removes the entry on the first; workflow.kept_assumptions
# turns the second (or an own answer that leaves the entry) into an unsupported item instead of asking again.
ASSUMPTION_OPTIONS = ("It only restates what the solver already models: remove it",
                      "It adds a requirement the solver must meet: keep it",
                      "Other (explain)")


def assumption_key(text: str) -> str:
    """The code key of the question about one assumption, stable across rounds (from its text)."""
    return CODE_KEY_PREFIX + "assumption_" + hashlib.sha1(text.strip().encode()).hexdigest()[:8]


# The stress norm's exponent. The spec carries no p, so the solver's default applies.
P_NORM = next(f.default for f in fields(LBracket2DConfig) if f.name == "p_norm")
# The stress the norm aggregates (solver/lbracket2d/main.py weights it by element density in the
# p-norm). The spec carries no measure, so this is the only one the solver offers.
STRESS_MEASURE = "plane-stress von Mises stress"
# What this solver models, in words: the code check's questions about free-text assumptions quote it.
SOLVER_MODEL = ("a 2D plane-stress plate of uniform thickness in static, small-strain linear elasticity, with one "
                "homogeneous isotropic material (E and ν) over the whole L; the top edge of the vertical arm clamped in "
                "x and y; one load case, a total force spread uniformly over a band at the tip of the horizontal arm, "
                "with a solid pad under it; minimum compliance under a volume limit measured against the full L "
                "(the outer square minus the corner cut, before the initial holes are subtracted) and an optional "
                f"calibrated p = {P_NORM:g} norm limit on the {STRESS_MEASURE}; and a level-set design that starts "
                "from five initial holes")
# Factors to SI for the units the check accepts (keys lower-case); drawing.py converts with them too.
LENGTH_TO_M = {"m": 1., "mm": .001, "cm": .01, "in": .0254, "inch": .0254, "inches": .0254}
STRESS_TO_PA = {"pa": 1., "kpa": 1e3, "mpa": 1e6, "gpa": 1e9, "psi": 6894.757293168}
FORCE_TO_N = {"n": 1., "kn": 1e3, "lbf": 4.4482216152605}
LENGTH_UNITS, STRESS_UNITS, FORCE_UNITS = (("m", "mm", "cm", "in"), ("Pa", "kPa", "MPa", "GPa", "psi"),
                                         ("N", "kN", "lbf"))

# These illustrate an answer's wording; they never enter config_values or the
# specification. The interface displays them separately from the answer input.
_SUPPORT_EXAMPLE = (
    "The entire top edge of the vertical arm is bonded to a rigid frame. "
    "Set displacement to zero in x and y over that whole edge."
)
_GEOMETRY_EXAMPLES = {
    "lx": "outer x length = <length> mm",
    "ly": "outer y length = <length> mm",
    "thickness": "thickness = <thickness> mm",
    "cut_length": "upper-right square cut = <size> by <size> mm, through thickness",
    "load_patch": "tip band = <height> mm high at the upper end of the horizontal-arm tip edge, through the full thickness",
    "hole_radius": "initial through-hole radius = <radius> mm",
    "hole_centers": "five initial hole centers = (<x1>,<y1>), …, (<x5>,<y5>) mm in the x-y plane, measured from the lower-left corner",
}


def _tip_patch_example(spec: ProblemSpec) -> str:
    """Use the stated thickness and band height when available; otherwise placeholders."""
    thickness = "full thickness"
    height = "<height> mm"
    if spec.geometry and "thickness" in spec.geometry.parameters:
        value = spec.geometry.parameters["thickness"]
        try:
            thickness = f"full {_scalar(value, LENGTH_TO_M) * 1000:g} mm thickness"
        except ValueError:
            pass
    if spec.geometry and "load_patch" in spec.geometry.parameters:
        try:
            height = f"{_scalar(spec.geometry.parameters['load_patch'], LENGTH_TO_M) * 1000:g} mm"
        except ValueError:
            pass
    return (
        f"Distribute the stated total force uniformly over a {height} high band at the "
        f"upper end of the horizontal-arm tip edge, through the {thickness}."
    )


def _volume_example(limit: Value | None) -> str:
    percentage = "<fraction>%"
    if limit is not None:
        try:
            percentage = f"{100 * _fraction(limit):g}%"
        except ValueError:
            pass
    return (
        f"The material limit is {percentage} of the full L-shaped domain, excluding "
        "the permanent square cut but before subtracting the five initial holes."
    )


@dataclass
class SpecAssessment:
    config: LBracket2DConfig | None
    issues: list[Issue]
    questions: list[Question]
    # Config field -> the dotted spec path its value was read from, for every value the check
    # parsed (also when another issue stops the config being built).
    sources: dict[str, str] = field(default_factory=dict)
    # For the trace only: what each rule created and what became of it. Never read by checks.
    trace_items: list[dict] = field(default_factory=list, compare=False, repr=False)
    trace_index: dict[int, int] = field(default_factory=dict, compare=False, repr=False)

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


def assess_spec(spec: ProblemSpec, kept: frozenset[str] = frozenset()) -> SpecAssessment:
    """Return a config only if every required decision is represented and valid.

    kept: assumption question keys the engineer answered with "keep" (workflow.kept_assumptions); those
    assumptions are reported as unsupported instead of asked about again."""
    issues: list[Issue] = []
    questions: list[Question] = []
    config_values: dict = {}
    sources: dict[str, str] = {}  # config field -> spec path, written only together with config_values by put()
    events: list[dict] = []      # trace record of each add(); see SpecAssessment.trace_items
    index: dict[int, int] = {}

    def put(target, value, path):
        config_values[target], sources[target] = value, path

    def add(name, field, description, *, key=None, kind="decision", question=None, example=None, why=None, options=None):
        # name: the rule (citations find the add("<name>" call); key: the issue key when it is not the name
        rule = f"assess.{name}@lbracket.py:{sys._getframe(1).f_lineno}"
        key = CODE_KEY_PREFIX + (key or name)
        issues.append(Issue(key=key, field=field, description=description, kind=kind))
        index[id(issues[-1])] = len(events)
        events.append(trace.item("issue", issues[-1], origin="code", fate="created", rule=rule))
        if question:
            questions.append(Question(key=key, issue_keys=[key], prompt=question, why=why or description,
                                      example_answer=example, answer_type="single_choice" if options else "text",
                                      options=list(options or ())))
            index[id(questions[-1])] = len(events)
            events.append(trace.item("question", questions[-1], origin="code", fate="created", rule=rule))

    if spec.problem_kind != "optimization":
        add("problem", "problem_kind", "This solver performs topology optimization only.",
            kind="decision" if spec.problem_kind == "unknown" else "unsupported",
            question="What should be optimized, and what should the objective and constraints be?" if spec.problem_kind == "unknown" else None,
            example="Minimize compliance, with material volume limited to <fraction>% of the full L-shaped domain excluding the corner cut. State any additional stress requirement separately.")

    geometry = spec.geometry
    if geometry is None or geometry.template is None:
        add("template", "geometry.template", "The bracket shape and the role of its holes need clarification.",
            question="Does the bracket have equal arms, an upper-right square cut through its thickness, and five initial through-holes? Are those holes free to change during optimization or must any be preserved?",
            example="Use an equal-arm L with an upper-right square cut and five through-holes. The holes are optimization seeds and may move, merge or close.")
    elif geometry.template != TEMPLATE:
        add("template", "geometry.template", "The connected solver supports an equal-arm L with an upper-right square cut and five initial through-holes. The requested shape needs a different geometry implementation.", kind="unsupported")
    if geometry is not None:
        if geometry.physical_dimension is None:
            add("dimension", "geometry.physical_dimension", "The physical geometry dimension is missing.",
                question="Is this a flat plate of uniform thickness, analysed in plane stress?")
        elif geometry.physical_dimension != 2:
            add("dimension", "geometry.physical_dimension", "This backend represents a 2D plane-stress plate.", kind="unsupported")
        if geometry.hole_role is None:
            add("hole_role", "geometry.hole_role", "Initial holes and preserved mounting holes are different requirements.",
                question="May the five initial holes move, merge, or close during optimization, or must they remain as mounting holes?" if geometry.template is not None else None)
        elif geometry.hole_role != HOLE_ROLE:
            add("hole_role", "geometry.hole_role", "This backend supports movable initial holes; preserved holes require a different geometry implementation.", kind="unsupported")
        extras = set(geometry.parameters) - GEOMETRY_FIELDS
        if extras:
            add("geometry_extra", "geometry.parameters", "Unrepresented geometry parameters: " + ", ".join(sorted(extras)), kind="unsupported")
        if any(REGIONS.get(r.name) != r.role for r in geometry.regions):
            add("regions", "geometry.regions", "Additional protected, mounting, or other geometry regions are not represented by this template.", kind="unsupported")
        missing = sorted(GEOMETRY_FIELDS - geometry.parameters.keys())
        if missing:
            labels = {"lx": "outer x length", "ly": "outer y length", "thickness": "thickness",
                      "cut_length": "square cut size", "load_patch": "tip load-band height",
                      "hole_radius": "initial hole radius", "hole_centers": "five initial hole x/y centers"}
            missing_text = ", ".join(labels[name] for name in missing)
            source_note = (" The governing source is identified; CAD/drawing import is not implemented. Supply extracted, validated dimensional data from that source." if geometry.authoritative_source else "")
            dimensions_question = "What are the governing " + missing_text + ", with units? If a drawing governs them, identify that data source."
            if missing == ["load_patch"]:
                dimensions_question = "What area receives the tip force? Specify the loaded edge, the band's height with units, and whether it spans the full thickness."
            add("dimensions", "geometry.parameters", "Missing physical geometry inputs: " + missing_text + "." + source_note,
                kind="data" if geometry.authoritative_source else "decision",
                question=None if geometry.authoritative_source else dimensions_question,
                example=_tip_patch_example(spec) if missing == ["load_patch"] else "; ".join(_GEOMETRY_EXAMPLES[name] for name in missing) + ".")
        for name in GEOMETRY_FIELDS & geometry.parameters.keys():
            value = geometry.parameters[name]
            try:
                if name == "hole_centers":
                    factor = _factor(value, LENGTH_TO_M)
                    if not isinstance(value.value, (list, tuple)) or len(value.value) != HOLES:
                        raise ValueError("requires five x/y coordinate pairs")
                    centers = []
                    for center in value.value:
                        if not isinstance(center, (list, tuple)) or len(center) != 2:
                            raise ValueError("requires five x/y coordinate pairs")
                        centers.append(tuple(_number(v) * factor for v in center))
                    put("hole_centers_m", tuple(centers), "geometry.parameters.hole_centers")
                else:
                    put(name + "_m", _scalar(value, LENGTH_TO_M), "geometry.parameters." + name)
            except ValueError as error:
                add("geometry_" + name, "geometry.parameters." + name, name + " " + str(error), kind="data")

    physics = spec.physics
    if physics is None or physics.family == "unknown":
        add("physics", "physics", "The governing physics and loading regime are unresolved.",
            question="Is the intended analysis static small-strain linear elasticity, or do inertia, large deformation, or other physics matter?")
    elif physics.family != FAMILY:
        add("physics", "physics.family", "Only solid mechanics is implemented.", kind="unsupported")
    if physics is not None:
        if physics.regime is None:
            add("regime", "physics.regime", "A static linear-elastic regime has not been established.",
                question="Can the loading be treated as static and the material response as small-strain linear elasticity?")
        elif _token(physics.regime) not in {REGIME, "static_linear_elasticity", "static_linear_elastic"}:
            add("regime", "physics.regime", "Only static small-strain linear elasticity is implemented.", kind="unsupported")
        # Plane stress is assumed unless the engineer requests another representation.
        if physics.representation is not None and _token(physics.representation) not in {REPRESENTATION, "plane_stress", "two_dimensional_plane_stress"}:
            add("representation", "physics.representation", "The requested analysis representation is not supported by the 2D plane-stress solver.", kind="unsupported")

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
        if material.region not in {None, DOMAIN, "entire_domain"}:
            add("material_region", "materials.0.region", "A material assigned to a subregion is not supported.", kind="unsupported")
        aliases = {"E": "E", "youngs_modulus": "E", "nu": "nu", "poisson_ratio": "nu"}
        properties, keys = {}, {}   # canonical name -> value, and -> the key the spec used
        for key, val in material.properties.items():
            canonical = aliases.get(key)
            if canonical is None or canonical in properties:
                add("material_properties", "materials.0.properties", "Unknown or duplicate material property: " + key, kind="unsupported")
            else:
                properties[canonical], keys[canonical] = val, key
        for name, target in (("E", "youngs_modulus_pa"), ("nu", "poisson_ratio")):
            if name not in properties:
                add("material_" + name, "materials.0.properties." + name,
                    "Authoritative material data required: " + ("Young's modulus with units." if name == "E" else "Poisson's ratio."), kind="data")
            else:
                try:
                    put(target, _scalar(properties[name], STRESS_TO_PA) if name == "E" else _fraction(properties[name]),
                        "materials.0.properties." + keys[name])
                except ValueError as error:
                    add("material_" + name, "materials.0.properties." + name, name + " " + str(error), kind="data")

    if not spec.boundary_conditions:
        add("support", "boundary_conditions", "The mounting behavior has not been specified.",
            question="How does the top of the vertical arm attach to the frame? Identify the restrained area and which translations it prevents (x and/or y); describe any motion it permits.",
            example=_SUPPORT_EXAMPLE)
    elif len(spec.boundary_conditions) != 1:
        add("support", "boundary_conditions", "Only one full top-arm clamp is supported.", kind="unsupported")
    else:
        bc = spec.boundary_conditions[0]
        if not bc.kind or not bc.location or not bc.components:
            add("support", "boundary_conditions", "The support location or translational restraint behavior is missing.",
                question="What motion does the mounting interface prevent? Specify whether the whole top edge is held at zero displacement in x and y, or describe the actual restraint and any permitted motion.",
                example=_SUPPORT_EXAMPLE)
        elif bc.location != SUPPORT_REGION or _token(bc.kind) not in {SUPPORT_KIND, "fixed"} or sorted(bc.components) != sorted(SUPPORT_COMPONENTS):
            add("support", "boundary_conditions", "The connected solver supports only a clamp holding the entire top edge of the vertical arm at zero displacement in x and y. The requested mounting behavior needs a different support implementation.", kind="unsupported")
        if bc.value is not None:
            try:
                if _scalar(bc.value, LENGTH_TO_M) != 0:
                    raise ValueError("nonzero displacement")
            except ValueError:
                add("support_value", "boundary_conditions.0.value", "Only zero prescribed displacement is supported.", kind="unsupported")

    if not spec.loads:
        add("load", "loads", "The load is missing.",
            question="What total force (magnitude and direction or x/y components, with units) acts on the horizontal-arm tip band?",
            example="The total force is [<Fx>, <Fy>] N in the stated x/y coordinates.")
    elif len(spec.loads) != 1:
        add("loads", "loads", "This backend supports exactly one load vector and one load case; multiple loads cannot be silently combined.", kind="unsupported")
    else:
        load = spec.loads[0]
        if load.location is None or load.kind is None:
            add("load_model", "loads", "The force distribution and location are unresolved.",
                question="Where and how is the stated force transferred to the bracket? Specify the loaded edge and whether the total force is uniformly distributed over it.",
                example=_tip_patch_example(spec))
        elif load.location != LOAD_REGION or _token(load.kind) != LOAD_KIND:
            add("load_model", "loads", "The connected solver supports a total force distributed uniformly over a band at the upper end of the horizontal-arm tip edge, through the full thickness. The requested load type or location needs a different implementation.", kind="unsupported")
        if load.magnitude is None:
            add("force", "loads.0.magnitude", "The applied force has no magnitude.",
                question="Provide the total force and its x/y direction with units.",
                example="The total force is [<Fx>, <Fy>] N in the stated x/y coordinates.")
        else:
            try:
                factor = _factor(load.magnitude, FORCE_TO_N)
                raw = load.magnitude.value
                if isinstance(raw, (list, tuple)):
                    if len(raw) != 2 or load.direction not in {None, "vector"}:
                        raise ValueError("requires two in-plane components and no conflicting direction")
                    force = tuple(_number(v) * factor for v in raw)
                else:
                    magnitude = _number(raw) * factor
                    direction = FORCE_DIRECTIONS.get(load.direction)
                    if direction is None:
                        raise ValueError("requires an explicit coordinate direction or force vector")
                    if magnitude <= 0:
                        raise ValueError("scalar force magnitude must be positive; use direction for sign")
                    force = tuple(magnitude * v for v in direction)
                put("force_n", force, "loads.0.magnitude")
            except ValueError as error:
                add("force", "loads.0", "Force " + str(error),
                    question="Provide the applied force as [Fx, Fy] and its force units.",
                    example="The total force is [<Fx>, <Fy>] N in the stated x/y coordinates.")

    # Patch extent and force distribution are one physical decision. A concise
    # answer should resolve both without asking for the same interface twice.
    if geometry and not geometry.authoritative_source and GEOMETRY_FIELDS - geometry.parameters.keys() == {"load_patch"}:
        patch_question = next((q for q in questions if q.key == CODE_KEY_PREFIX + "dimensions"), None)
        model_question = next((q for q in questions if q.key == CODE_KEY_PREFIX + "load_model"), None)
        if patch_question and model_question:
            patch_question.issue_keys.extend(model_question.issue_keys)
            patch_question.prompt = "Where and how is the tip force transferred? Specify the loaded edge, the band's height with units, whether it spans the full thickness, and whether the stated total force is uniformly distributed."
            patch_question.why = "The load's area and distribution are unresolved; both affect the local stress field."
            questions.remove(model_question)
            events[index[id(model_question)]].update(fate="merged", fate_rule="assess.merge_patch_load",
                                                    detail={"into": patch_question.key})
            events[index[id(patch_question)]]["detail"] = {"absorbed": model_question.key,
                                                          "rewritten_by": "assess.merge_patch_load"}

    optimization = spec.optimization
    if optimization is None:
        add("optimization", "optimization", "Objective and constraints are missing.",
            question="Should compliance be minimized with what volume fraction of the L domain, and is a stress constraint required?")
    else:
        if optimization.design_variable not in {None, DESIGN_VARIABLE, "level_set"}:
            add("design_variable", "optimization.design_variable", "Only material distribution with a level-set boundary is supported.", kind="unsupported")
        objectives = optimization.objectives
        if not objectives:
            add("objective", "optimization.objectives", "The optimization objective is missing.", question="Is minimum compliance the design objective?")
        elif len(objectives) == 1 and (objectives[0].sense is None or objectives[0].quantity is None):
            add("objective", "optimization.objectives", "The objective quantity or optimization sense is unresolved.", question="What quantity should be minimized or maximized?")
        elif len(objectives) != 1 or objectives[0].sense != "minimize" or _token(objectives[0].quantity) != OBJECTIVE or objectives[0].aggregation not in {None, "single_load_case"}:
            add("objective", "optimization.objectives", "Only minimum compliance for one load case is supported.", kind="unsupported")
        volume_seen = stress_seen = False
        for n, constraint in enumerate(optimization.constraints):
            quantity = _token(constraint.quantity)
            if quantity == VOLUME_QUANTITY and not volume_seen:
                volume_seen = True
                if constraint.region != REFERENCE_REGION:
                    add("volume_reference", "optimization.constraints", "The volume fraction must refer explicitly to the L domain, excluding the permanent cut.",
                        kind="decision" if constraint.region is None else "unsupported",
                        question="What volume does the stated material percentage refer to: the full L-shaped domain excluding the corner cut, the initial perforated shape, or another reference?" if constraint.region is None else None,
                        example=_volume_example(constraint.limit))
                if constraint.relation != "<=":
                    add("volume_relation", "optimization.constraints", "Only an upper bound on volume fraction is supported.", kind="unsupported")
                if constraint.limit is None:
                    add("volume_limit", "optimization.constraints", "The material fraction is missing.", question="What maximum fraction of the full L domain may contain material?",
                        example="At most <fraction>% of the full L-shaped domain may contain material.")
                else:
                    try:
                        put("volume_fraction", _fraction(constraint.limit), f"optimization.constraints.{n}.limit")
                    except ValueError as error:
                        add("volume_limit", "optimization.constraints", "Volume fraction " + str(error), kind="data")
            elif quantity == STRESS_QUANTITY and not stress_seen:
                stress_seen = True
                if constraint.region != REFERENCE_REGION or constraint.relation != "<=":
                    add("stress_contract", "optimization.constraints", f"Only the volume-averaged p={P_NORM:g} stress norm over the L domain with an upper bound is supported.", kind="unsupported")
                if constraint.limit is None:
                    add("stress_data", "optimization.constraints", "A calibrated stress p-norm limit with units is required; a material yield value is not interchangeable.", kind="data")
                else:
                    try:
                        put("stress_limit_pa", _scalar(constraint.limit, STRESS_TO_PA), f"optimization.constraints.{n}.limit")
                    except ValueError as error:
                        add("stress_data", "optimization.constraints", "Stress p-norm limit " + str(error), kind="data")
            else:
                add("constraint", "optimization.constraints", "Unsupported or duplicate constraint: " + str(constraint.quantity) + ". Peak/yield stress, displacement, buckling and manufacturing limits are not implemented.", kind="unsupported")
        if not volume_seen:
            add("volume", "optimization.constraints", "A volume fraction upper bound and its reference domain are required.",
                question="What maximum material fraction is permitted, and is it measured relative to the full L domain excluding the square cut?",
                example="Retain at most <fraction>% of the full L-shaped domain, excluding the permanent square cut but before subtracting the five initial holes.")
        if optimization.stress_requirement is None:
            add("stress_choice", "optimization.stress_requirement", "The engineer has not specified whether the problem is volume-only or includes a calibrated stress p-norm constraint.",
                question=f"Is this minimum compliance with only a volume limit, or do you also require a stress constraint? State any physical stress requirement as intended; this solver currently supports only a calibrated p={P_NORM:g} norm, not a peak/yield bound.")
        elif optimization.stress_requirement == VOLUME_ONLY:
            if stress_seen:
                add("stress_conflict", "optimization", f"The {VOLUME_ONLY} choice conflicts with an explicit stress constraint.", kind="unsupported")
            else:
                put("stress_limit_pa", None, "optimization.stress_requirement")
        elif not stress_seen:
            add("stress_missing", "optimization.constraints", f"The selected p-norm constraint needs a calibrated {STRESS_QUANTITY} upper bound with units.", kind="data")

    if spec.manufacturing is not None and (spec.manufacturing.process or spec.manufacturing.requirements):
        add("manufacturing", "manufacturing", "Manufacturing requirements are not enforced by this backend; they cannot be dropped from a requested solve.", kind="unsupported")
    # Free text cannot be enforced. Ask about each assumption, quoting it: one that only restates what the solver
    # models can go; one the engineer keeps is a requirement this solver cannot run.
    asked = set()
    for n, text in enumerate(spec.assumptions):
        key = assumption_key(text)
        if key in asked:
            continue
        asked.add(key)
        quoted = " ".join(text.split())
        if key in kept:
            add("assumption_kept", f"assumptions.{n}", key=key.removeprefix(CODE_KEY_PREFIX), description=
                f'You kept the assumption "{quoted}" as a requirement. The solver cannot enforce free text: it runs '
                f"{SOLVER_MODEL}, so it cannot run this specification as stated. Express the requirement through a "
                "supported field, or remove the assumption.", kind="unsupported")
        else:
            add("assumption", f"assumptions.{n}", key=key.removeprefix(CODE_KEY_PREFIX), description=
                f'The specification lists the free-text assumption "{quoted}", which the solver cannot enforce.',
                question=f'Does the assumption "{quoted}" only restate what the solver already models, or does it add '
                         "a requirement?",
                why=f"The solver runs this model: {SOLVER_MODEL}. Values the specification's own fields state "
                    "(dimensions, material, support, load, limits) are enforced through those fields, so an assumption "
                    "that repeats one of them is a restatement. If the assumption only restates part of this, removing "
                    "it changes nothing in the run; if it adds a requirement, this solver cannot run the specification "
                    "as stated.",
                options=ASSUMPTION_OPTIONS)
    config = None
    if not issues:
        try:
            config = LBracket2DConfig.from_dict(config_values)
        except (TypeError, ValueError) as error:
            add("config", "geometry", "Solver configuration is invalid: " + str(error), kind="data")
    for question in questions[MAX_QUESTIONS:]:
        events[index[id(question)]].update(fate="capped", fate_rule="assess.cap_questions_4")
    return SpecAssessment(config=config, issues=issues, questions=questions[:MAX_QUESTIONS], sources=sources,
                          trace_items=events, trace_index=index)


# Code-check issue (name) -> the config fields an answer to it feeds; names not listed feed none (checked choices).
_CODE_INPUTS = {"material": ["youngs_modulus_pa", "poisson_ratio"], "load": ["force_n"], "force": ["force_n"],
                "optimization": ["volume_fraction", "stress_limit_pa"], "volume": ["volume_fraction"],
                "volume_limit": ["volume_fraction"], "stress_choice": ["stress_limit_pa"],
                "stress_data": ["stress_limit_pa"], "stress_missing": ["stress_limit_pa"]}


def solver_inputs(field: str | None, key: str, spec: ProblemSpec) -> list[str]:
    """The LBracket2DConfig fields a value at this spec field would feed: what an answer reaches in the solver
    (shown with each question). Code-check issues are looked up by name; the paths follow assess_spec's put()."""
    field = (field or "").replace("[", ".").replace("]", "")
    if field == "geometry.parameters":   # the dimensions issue: the template's absent dimensions
        stated = spec.geometry.parameters if spec.geometry else {}
        return [f"{name}_m" for name in GEOMETRY_ORDER if name not in stated]
    if field.startswith("geometry.parameters."):
        name = field.split(".")[2]
        return [f"{name}_m"] if name in GEOMETRY_FIELDS else []
    if field.startswith("materials.0.properties."):
        last = field.rsplit(".", 1)[-1]
        return {"E": ["youngs_modulus_pa"], "youngs_modulus": ["youngs_modulus_pa"], "nu": ["poisson_ratio"],
                "poisson_ratio": ["poisson_ratio"]}.get(last, [])
    if key.startswith(CODE_KEY_PREFIX):
        return list(_CODE_INPUTS.get(key.removeprefix(CODE_KEY_PREFIX), []))
    if field in ("loads", "loads.0", "loads.0.magnitude", "loads.0.direction"):
        return ["force_n"]
    parts = field.split(".")
    if field.startswith("optimization.constraints.") and len(parts) >= 3 and parts[2].isdigit():
        if len(parts) > 3 and parts[3] != "limit":
            return []   # region, relation, quantity: checked choices
        constraints = spec.optimization.constraints if spec.optimization else []
        index = int(parts[2])
        quantity = (constraints[index].quantity or "") if index < len(constraints) else ""
        return {"volume_fraction": ["volume_fraction"], "stress_p_norm": ["stress_limit_pa"]}.get(
            quantity.strip().lower().replace(" ", "_").replace("-", "_"), [])
    if field == "optimization.stress_requirement":
        return ["stress_limit_pa"]
    return []


def start_issues(config: LBracket2DConfig | None) -> list[Issue]:
    """The solver-side dry run's findings for a built config, as code-check issues: ParaLeSTO could not
    build the starting design, or builds it with a different geometry than stated (solver/lbracket2d/dryrun.py).
    Kept apart from assess_spec, which reads only the spec: this runs ParaLeSTO, in a child process."""
    if config is None:
        return []
    from project.solver.lbracket2d.dryrun import dry_run

    run = dry_run(config)
    if run.transient:   # no verdict: the check itself could not run
        return [Issue(key=CODE_KEY_PREFIX + "start_check", field="geometry", kind="data", description=(
            f"The dry run of ParaLeSTO's starting design could not run ({run.failure}), so these solver inputs are "
            "unchecked. Reload the page to retry."))]
    issues = []
    if run.failure:
        issues.append(Issue(key=CODE_KEY_PREFIX + "start_design", field="geometry", kind="unsupported", description=(
            f"ParaLeSTO could not build the starting design from these solver inputs ({run.failure}); the solver runs "
            "the same step first, so a run would stop there. Initial holes that reach the loaded square at the tip, or "
            "leave a gap of about one element to an edge, the cut or another hole, cause this.")))
    issues += [Issue(key=CODE_KEY_PREFIX + "start_geometry", field=field, kind="unsupported", description=text)
               for field, text in run.problems]
    return issues


def combine_review(spec: ProblemSpec, review: Review, *, interim: bool = False,
                   kept: frozenset[str] = frozenset()) -> Review:
    """Deterministic issues survive any LLM verdict; questions remain bounded.

    interim marks the code-only merge made before the critic answers; it affects the trace only.
    kept: assumption questions answered "keep" (see assess_spec).
    """
    assessment = assess_spec(spec, kept)
    fates: dict[tuple[str, int], tuple[str, str | None]] = {}   # trace only
    model_issues = []
    known_material = len(spec.materials) == 1 and bool(spec.materials[0].name)
    for n, issue in enumerate(review.issues):
        if issue.key.startswith(CODE_KEY_PREFIX):
            fates["issue", n] = ("dropped", "merge.drop_solver_prefix")
            continue
        text = " ".join((issue.key, issue.field or "", issue.description)).lower()
        # A model may disregard the prompt and ask for a guessed modulus. Keep
        # this as missing source data even when the model labels it a decision.
        if known_material and (".properties" in text or "young's modulus" in text or "youngs_modulus" in text or "poisson" in text):
            issue = issue.model_copy(update={"kind": "data"})
            fates["issue", n] = ("relabeled", "merge.relabel_material_data")
        else:
            fates["issue", n] = ("kept", None)
        model_issues.append(issue)
    issues = assessment.issues + model_issues
    questions = list(assessment.questions)
    decision_keys = {i.key for i in issues if i.blocking and i.kind == "decision"}
    seen = {q.key for q in questions}
    for n, question in enumerate(review.questions):
        if question.key.startswith(CODE_KEY_PREFIX):
            fates["question", n] = ("dropped", "merge.drop_solver_prefix")
        elif len(questions) >= MAX_QUESTIONS:
            fates["question", n] = ("capped", "merge.cap_4")
        elif question.key in seen:
            fates["question", n] = ("dropped", "merge.dedupe_question_key")
        elif not set(question.issue_keys) & decision_keys:
            fates["question", n] = ("dropped", "merge.q_requires_decision_link")
        else:
            questions.append(question)
            seen.add(question.key)
            fates["question", n] = ("kept", None)
    result = Review(issues=issues, questions=questions)
    trace.merged(assessment, review, result, fates, interim=interim)
    return result


def reference_spec() -> ProblemSpec:
    """Explicit benchmark selection; never merged into an incomplete user spec."""
    c = reference_config()
    parameters = {name: {"value": getattr(c, name + "_m"), "unit": "m"}
                  for name in GEOMETRY_ORDER if name != "hole_centers"}
    parameters["hole_centers"] = {"value": c.hole_centers_m, "unit": "m"}
    return ProblemSpec.model_validate({
        "name": "2D L-bracket with five initial holes", "problem_kind": "optimization",
        "geometry": {"template": TEMPLATE, "physical_dimension": 2, "hole_role": HOLE_ROLE,
                     "description": "Equal-arm L with an upper-right square cut and five movable initial through-holes.",
                     "parameters": parameters},
        "physics": {"family": FAMILY, "representation": REPRESENTATION, "regime": REGIME},
        "materials": [{"name": "Reference titanium model", "region": DOMAIN, "data_source": "Explicit simplified_2D_holes benchmark constants (mfrto)",
                       "properties": {"E": {"value": c.youngs_modulus_pa, "unit": "Pa"},
                                      "nu": {"value": c.poisson_ratio}}}],
        "boundary_conditions": [{"name": "Top clamp", "location": SUPPORT_REGION, "kind": SUPPORT_KIND, "components": list(SUPPORT_COMPONENTS)}],
        "loads": [{"name": "Tip-band force", "location": LOAD_REGION, "kind": LOAD_KIND,
                   "magnitude": {"value": c.force_n, "unit": "N"}, "load_case": "reference"}],
        "optimization": {"design_variable": DESIGN_VARIABLE, "stress_requirement": PNORM_LIMIT,
                         "objectives": [{"sense": "minimize", "quantity": OBJECTIVE}],
                         "constraints": [{"quantity": VOLUME_QUANTITY, "relation": "<=", "region": REFERENCE_REGION, "limit": {"value": c.volume_fraction}},
                                         {"quantity": STRESS_QUANTITY, "relation": "<=", "region": REFERENCE_REGION, "limit": {"value": c.stress_limit_pa, "unit": "Pa"}}]},
    })
