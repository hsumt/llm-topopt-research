"""The specification tuple for the stress-constrained L-bracket.

This dataclass *is* the machine-readable problem statement that the ATO cascade
edits.  Its field groups are the slots of the canonical MDO problem statement:
objective, design variables, design domain / non-design region, load cases,
boundary conditions, inequality constraints and discretisation.  Keeping them
as explicit named slots is what makes a proposed edit checkable against a
grammar rather than being free-form text.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, replace


class ModelCannotRepresent(RuntimeError):
    """The analysis model has no degree of freedom for what the specification asks.

    Raised when a load case acts along an axis the model does not carry -- a
    transverse load on a plane-stress model, for instance.  This is deliberately a
    distinct exception rather than a large residual: the model does not answer the
    question weakly, it cannot be asked, and a diagnosis that treats the two the
    same is wrong.  ``project/ato/discharge.py`` turns this into its own verdict.
    """


@dataclass
class Geometry:
    L: float = 200.0
    arm_fraction: float = 0.4
    thickness: float = 1.0
    n_cells_per_side: int = 40            # discretisation slot
    corner_fillet_radius: float = 0.0     # design domain slot
    exclude_load_elements: bool = True    # non-design region slot
    exclude_nx: int = 3
    exclude_ny: int = 2
    model: str = "plane_stress"
    """Analysis model: ``plane_stress`` (Holmberg, 2 dof/node) or ``extruded_3d``
    (2.5D extrusion, 3 dof/node, density constant through the thickness)."""
    n_layers: int = 2
    """Element layers through the thickness; ignored by ``plane_stress``."""


MODELS = ("plane_stress", "extruded_3d")


@dataclass
class Material:
    E: float = 71000.0                    # MPa, aircraft aluminium
    nu: float = 0.33
    density: float = 2.8e-9               # ton/mm^3
    yield_stress: float = 350.0           # MPa


@dataclass
class LoadCase:
    magnitude: float = 1500.0             # N
    distribute_nodes: int = 1             # load cases slot
    direction: str = "y"                  # "x"/"y" in-plane, "z" out-of-plane
    name: str = "primary"

    def is_out_of_plane(self) -> bool:
        return self.direction == "z"


@dataclass
class Constraints:
    stress_limit: float = 350.0           # MPa, = yield
    n_clusters: int = 10
    p_norm: float = 8.0
    clustering: str = "stress_level"
    recluster_every: int = 1
    mass_fraction_limit: float = 0.30     # used by P2 and P3


@dataclass
class Optimizer:
    penal: float = 3.0                    # SIMP q
    r0_elements: float = 1.5              # filter radius in element sizes
    move: float = 0.1
    max_iter: int = 120
    tol_change: float = 5.0e-3
    x_min: float = 1.0e-3
    init_density: float = 0.5
    feasibility_tol: float = 1.0e-2
    """Tolerance on the normalized cluster constraint at termination.

    Reclustering every iteration changes which stress points belong to which
    constraint, so successive iterations solve slightly different problems and
    the constraint value oscillates about zero rather than settling on it;
    Holmberg Sec. 9 reports the same oscillation in the convergence plots.  A
    tolerance is therefore required to declare feasibility at all, and it is
    stated here rather than buried in the loop."""


@dataclass
class StressSpec:
    name: str = "holmberg_lbracket"
    formulation: str = "P1"               # objective slot: P1 | P2 | P3
    geometry: Geometry = field(default_factory=Geometry)
    material: Material = field(default_factory=Material)
    load_cases: list = field(default_factory=lambda: [LoadCase()])
    constraints: Constraints = field(default_factory=Constraints)
    optimizer: Optimizer = field(default_factory=Optimizer)

    @property
    def load(self) -> LoadCase:
        """The primary load case. Retained so single-case code reads unchanged."""
        return self.load_cases[0]

    def requires_out_of_plane(self) -> bool:
        return any(lc.is_out_of_plane() for lc in self.load_cases)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "StressSpec":
        d = copy.deepcopy(d)
        return cls(
            name=d.get("name", "spec"),
            formulation=d.get("formulation", "P1"),
            geometry=Geometry(**d.get("geometry", {})),
            material=Material(**d.get("material", {})),
            load_cases=_load_cases_from_dict(d),
            constraints=Constraints(**d.get("constraints", {})),
            optimizer=Optimizer(**d.get("optimizer", {})),
        )

    def copy(self) -> "StressSpec":
        return StressSpec.from_dict(self.to_dict())


def _load_cases_from_dict(d: dict) -> list:
    """Accept either the current ``load_cases`` list or a legacy single ``load``."""
    if "load_cases" in d and d["load_cases"]:
        return [LoadCase(**lc) for lc in d["load_cases"]]
    if "load" in d and d["load"]:
        return [LoadCase(**d["load"])]
    return [LoadCase()]


FORMULATIONS = {
    "P1": "minimize mass subject to clustered stress constraints",
    "P2": "minimize compliance subject to clustered stress constraints and a mass limit",
    "P3": "minimize compliance subject to a mass limit (no stress constraints)",
}


# ---------------------------------------------------------------------------
# Fidelity tiers
# ---------------------------------------------------------------------------
# arm_fraction * n_cells_per_side must be an integer so that the re-entrant
# corner and the load point land on node lines; for 0.4 that means a multiple
# of five.  ``reference`` is Holmberg's own discretisation (h = 2 mm, 6400
# elements) and exists to benchmark a small number of iterations against the
# paper -- it is not the tier the cascade runs on, because the cascade re-solves
# the problem once per candidate and once per subset.
FIDELITY = {
    "coarse":   {"n_cells_per_side": 20,  "max_iter": 60},
    "low":      {"n_cells_per_side": 40,  "max_iter": 120},
    "medium":   {"n_cells_per_side": 60,  "max_iter": 200},
    "reference": {"n_cells_per_side": 100, "max_iter": 400},
}

DEFAULT_FIDELITY = "low"


def elements_at_fidelity(tier: str, arm_fraction: float = 0.4) -> int:
    """Active element count for a tier, without building the mesh."""
    n = FIDELITY[tier]["n_cells_per_side"]
    void = round((1.0 - arm_fraction) * n)
    return n * n - void * void


def spec_at_fidelity(tier: str = DEFAULT_FIDELITY, *, max_iter: int | None = None,
                     **overrides) -> "StressSpec":
    """Build a specification at a named fidelity tier.

    ``max_iter`` follows the tier unless given here, in which case it wins -- so a
    reference-tier cost probe can be capped with ``max_iter=1`` in one call.
    ``overrides`` are ``StressSpec`` constructor fields (``formulation``,
    ``material``, ...); the tier owns ``geometry.n_cells_per_side``.
    """
    if tier not in FIDELITY:
        raise ValueError(f"unknown fidelity {tier!r}; choose from {sorted(FIDELITY)}")
    cfg = FIDELITY[tier]
    spec = StressSpec(**overrides) if overrides else StressSpec()
    spec.geometry.n_cells_per_side = cfg["n_cells_per_side"]
    spec.optimizer.max_iter = int(max_iter) if max_iter is not None else cfg["max_iter"]
    spec.name = f"{spec.name}_{tier}"
    return spec
