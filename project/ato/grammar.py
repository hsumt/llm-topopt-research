"""The specification edit grammar.

Stage 3 of the cascade lets a language model propose candidate edits.  It does
not let the model decide what an edit *means*.  Every operation below declares,
independently of any model output:

* ``slot``        -- which slot of the canonical MDO problem statement it edits
* ``tier``        -- admissibility: T1 intent-preserving, T2 intent-revealing,
                     T3 intent-altering
* ``boundedness`` -- whether the specification plus the anomaly localisation
                     determine the edit's parameters (BOUND) or not (UNBOUND)
* ``parameter_source`` -- where a bound parameter comes from

A model may propose a tier; the proposal is recorded for audit and then
overridden by the value here.  Classifying the authority consequence of an edit
is not delegated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from project.topopt.stress.problem import StressSpec

TIER_ORDER = {"T1": 0, "T2": 1, "T3": 2}
TIER_MEANING = {
    "T1": "intent-preserving: a numerical or discretisation choice that does not change what was asked for",
    "T2": "intent-revealing: records a requirement or fixture the engineer already assumed but did not state",
    "T3": "intent-altering: asserts something the engineer never stated; reported, never applied silently",
}


@dataclass(frozen=True)
class EditOperation:
    name: str
    slot: str
    description: str
    tier: str
    boundedness: str
    parameter_source: str
    parameters: dict          # name -> (type, validator description)
    apply: Callable[[StressSpec, dict], None]
    validate: Callable[[dict], list]

    def to_prompt_dict(self) -> dict:
        return {
            "operation": self.name,
            "tuple_slot": self.slot,
            "description": self.description,
            "parameters": {k: v for k, v in self.parameters.items()},
        }


#: Slot -> admissibility tier, consulted for every operation including ones
#: synthesized mid-run. Tier is the authority-gate property, so it is decided here
#: from the slot and the declared intent, never by whichever agent proposed the
#: edit. A proposer's own suggestion is recorded for audit and then overridden.
SLOT_TIER_RULES = (
    ("discretisation", "T1",
     "a representation choice: changes what the model can be asked, not what was asked"),
    ("analysis model", "T1",
     "a representation choice: changes what the model can be asked, not what was asked"),
    ("regularisation", "T1", "a numerical choice with no requirement content"),
    ("load cases", "T2",
     "records a service condition the part is required to carry but the specification omitted"),
    ("design variables", "T2",
     "records a fixture or manufacturing reality the specification left implicit"),
    ("non-design region", "T2", "records a region the engineer already treats as fixed"),
    ("design domain", "T3",
     "asserts part geometry the engineer never stated"),
    ("objective", "T2",
     "restates what is being optimised for, which the engineer can confirm or deny"),
    ("inequality constraints", "T3",
     "moves a stated requirement; intent-altering unless the requirement is negotiable"),
)


def tier_for_slot(slot: str, *, intent=None, loosens: bool | None = None) -> tuple:
    """Return ``(tier, reason)`` for a slot, refined by declared intent.

    The intent document is what makes T2 and T3 separable at all: relaxing an
    allowable the engineer calls firm is intent-altering, while relaxing one they
    called negotiable is merely intent-revealing. With no intent supplied the rule
    stays conservative -- the higher tier -- because an unverifiable assumption
    about what the engineer meant must never lower the authority requirement.
    """
    s = slot.lower()
    tier, reason = "T3", "slot not recognised; defaulting to the most restrictive tier"
    for key, t, why in SLOT_TIER_RULES:
        if key in s:
            tier, reason = t, why
            break
    if intent is not None and tier == "T3" and loosens is False:
        tier = "T2"
        reason = f"{reason}; tightening rather than loosening, so no requirement is given up"
    if intent is not None and "inequality constraints" in s and loosens:
        firm = intent.stress_allowable_is_firm or intent.mass_budget_is_firm
        if not firm:
            tier = "T2"
            reason = "the engineer declared the stated limits negotiable"
    return tier, reason


def _pos_int(d, key, lo=1, hi=10**6):
    v = d.get(key)
    if not isinstance(v, (int, float)) or float(v) != int(v):
        return [f"{key} must be an integer"]
    if not (lo <= int(v) <= hi):
        return [f"{key}={v} outside [{lo}, {hi}]"]
    return []


def _pos_float(d, key, lo, hi):
    v = d.get(key)
    if not isinstance(v, (int, float)):
        return [f"{key} must be a number"]
    if not (lo <= float(v) <= hi):
        return [f"{key}={v} outside [{lo}, {hi}]"]
    return []


def _apply_formulation(spec, prm):
    spec.formulation = prm["formulation"]


def _apply_exclude(spec, prm):
    spec.geometry.exclude_load_elements = True
    spec.geometry.exclude_nx = int(prm["exclude_nx"])
    spec.geometry.exclude_ny = int(prm["exclude_ny"])


def _apply_distribute(spec, prm):
    spec.load.distribute_nodes = int(prm["distribute_nodes"])


def _apply_fillet(spec, prm):
    spec.geometry.corner_fillet_radius = float(prm["corner_fillet_radius"])


def _apply_refine(spec, prm):
    spec.geometry.n_cells_per_side = int(prm["n_cells_per_side"])


def _apply_stress_limit(spec, prm):
    spec.constraints.stress_limit = float(prm["stress_limit"])


def _apply_mass(spec, prm):
    spec.constraints.mass_fraction_limit = float(prm["mass_fraction_limit"])


def _apply_filter(spec, prm):
    spec.optimizer.r0_elements = float(prm["r0_elements"])


def _apply_clusters(spec, prm):
    spec.constraints.n_clusters = int(prm["n_clusters"])


OPERATIONS: dict[str, EditOperation] = {
    op.name: op
    for op in [
        EditOperation(
            name="add_stress_constraints",
            slot="objective and inequality constraints",
            description=(
                "Replace the stiffness-only statement with a stress-constrained one: "
                "P1 minimizes mass subject to clustered von Mises constraints, P2 "
                "minimizes compliance subject to those constraints and the stated mass limit."
            ),
            tier="T2",
            boundedness="BOUND",
            parameter_source=(
                "the stress limit is the material yield already present in the specification; "
                "the clustered P-norm settings are solver-side numerics"
            ),
            parameters={"formulation": "one of 'P1' or 'P2'"},
            apply=_apply_formulation,
            validate=lambda d: ([] if d.get("formulation") in ("P1", "P2")
                                else ["formulation must be 'P1' or 'P2'"]),
        ),
        EditOperation(
            name="exclude_load_elements",
            slot="design variables, plus a non-design region",
            description=(
                "Freeze a patch of elements at the load point as solid and remove them "
                "from the design variable vector. The material stays: load enters solid, "
                "not void. The frozen patch still acts through the design variable filter."
            ),
            tier="T2",
            boundedness="BOUND",
            parameter_source="element set from the load localisation; extent from the filter radius",
            parameters={"exclude_nx": "integer 1-8, elements along x",
                        "exclude_ny": "integer 1-8, elements along y"},
            apply=_apply_exclude,
            validate=lambda d: _pos_int(d, "exclude_nx", 1, 8) + _pos_int(d, "exclude_ny", 1, 8),
        ),
        EditOperation(
            name="distribute_load",
            slot="load cases",
            description=(
                "Spread the stated tip load over several adjacent nodes instead of one, "
                "enlarging the area over which it is introduced."
            ),
            tier="T2",
            boundedness="BOUND",
            parameter_source="node set from the stated load point; extent from the filter radius",
            parameters={"distribute_nodes": "integer 2-10"},
            apply=_apply_distribute,
            validate=lambda d: _pos_int(d, "distribute_nodes", 2, 10),
        ),
        EditOperation(
            name="declare_corner_fillet",
            slot="design domain, non-design region",
            description=(
                "Declare a fillet radius at the re-entrant corner, removing the sharp "
                "geometric feature from the design domain."
            ),
            tier="T2",
            boundedness="BOUND in place, UNBOUND in size",
            parameter_source=(
                "location from the predicate that fired; the radius is not determined by "
                "anything in the stated problem and is probed"
            ),
            parameters={"corner_fillet_radius": "float in mm, 2.0-60.0"},
            apply=_apply_fillet,
            validate=lambda d: _pos_float(d, "corner_fillet_radius", 2.0, 60.0),
        ),
        EditOperation(
            name="refine_mesh",
            slot="discretisation",
            description="Refine the discretisation of the whole domain.",
            tier="T1",
            boundedness="BOUND",
            parameter_source="element set where the stress predicate fired",
            parameters={"n_cells_per_side": "integer multiple of 5, 20-100"},
            apply=_apply_refine,
            validate=lambda d: (
                _pos_int(d, "n_cells_per_side", 20, 100)
                or ([] if int(d["n_cells_per_side"]) % 5 == 0
                    else ["n_cells_per_side must be a multiple of 5 so the arm lands on node lines"])
            ),
        ),
        EditOperation(
            name="relax_stress_limit",
            slot="inequality constraints",
            description="Raise the stated allowable stress.",
            tier="T3",
            boundedness="UNBOUND",
            parameter_source="none; no procedure in the stated problem supplies a new limit",
            parameters={"stress_limit": "float in MPa, 350.0-1200.0"},
            apply=_apply_stress_limit,
            validate=lambda d: _pos_float(d, "stress_limit", 350.0, 1200.0),
        ),
        EditOperation(
            name="increase_mass_budget",
            slot="inequality constraints",
            description="Raise the stated allowable mass fraction.",
            tier="T3",
            boundedness="UNBOUND",
            parameter_source="none; the stated problem fixes the mass budget",
            parameters={"mass_fraction_limit": "float, 0.05-0.95"},
            apply=_apply_mass,
            validate=lambda d: _pos_float(d, "mass_fraction_limit", 0.05, 0.95),
        ),
        EditOperation(
            name="increase_filter_radius",
            slot="discretisation and regularisation",
            description="Enlarge the design variable filter radius, thickening members.",
            tier="T1",
            boundedness="UNBOUND in size",
            parameter_source="none; the filter radius is a regularisation choice",
            parameters={"r0_elements": "float, filter radius in element sizes, 1.0-6.0"},
            apply=_apply_filter,
            validate=lambda d: _pos_float(d, "r0_elements", 1.0, 6.0),
        ),
        EditOperation(
            name="refine_stress_clusters",
            slot="inequality constraints",
            description=(
                "Change the number of stress clusters, i.e. how finely the local stress "
                "field is resolved by the constraint set."
            ),
            tier="T1",
            boundedness="UNBOUND in size",
            parameter_source="none; cluster count is an approximation parameter",
            parameters={"n_clusters": "integer 1-40"},
            apply=_apply_clusters,
            validate=lambda d: _pos_int(d, "n_clusters", 1, 40),
        ),
    ]
}


def validate_candidate(cand: dict) -> list:
    """Return a list of human-readable problems; empty means the candidate is well formed."""
    problems = []
    name = cand.get("operation")
    registry = all_operations()
    if name not in registry:
        return [f"unknown operation {name!r}; choose from {sorted(registry)}"]
    op = registry[name]
    prm = cand.get("parameters") or {}
    missing = [k for k in op.parameters if k not in prm]
    if missing:
        problems.append(f"{name}: missing parameters {missing}")
    else:
        problems.extend(f"{name}: {p}" for p in op.validate(prm))
    return problems


def apply_edits(base: StressSpec, candidates: list) -> StressSpec:
    """Apply a set of candidate edits to a copy of the base specification."""
    spec = base.copy()
    for cand in candidates:
        all_operations()[cand["operation"]].apply(spec, cand["parameters"])
    spec.name = base.name + "__" + "+".join(sorted(c["id"] for c in candidates))
    return spec


SYNTHESIZED: dict = {}
"""Operations authored during this run by the synthesis stage.

Kept separate from ``OPERATIONS`` so that a report can always say which edits came
from the committed grammar and which the system wrote for itself. Nothing here is
written back to source: a synthesized operation lives for the duration of the run
and its source is recorded in the run record for review.
"""


def all_operations() -> dict:
    """The committed grammar plus anything synthesized this run."""
    return {**OPERATIONS, **SYNTHESIZED}


def register_synthesized(op: EditOperation) -> None:
    if op.name in OPERATIONS:
        raise ValueError(f"{op.name!r} already exists in the committed grammar")
    SYNTHESIZED[op.name] = op


def grammar_for_prompt() -> list:
    return [op.to_prompt_dict() for op in all_operations().values()]
