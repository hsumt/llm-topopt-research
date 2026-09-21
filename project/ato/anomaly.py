"""Deterministic anomaly predicates on a converged design.

Formal rules on the converged stress field, not learned classifiers.  Every
predicate reports the value it compared, the threshold it compared against and
where the evidence is located, so a verdict can be audited without re-running
anything.

Why the predicate is not "peak stress exceeds the allowable"
------------------------------------------------------------
It is tempting to fire on max(sigma_vM) > sigma_limit.  That predicate cannot
discriminate a specification defect from a formulation limitation, and must not
be used here.  Holmberg Sec. 5 proves that the clustered P-norm *underestimates*
the maximum local stress:

    (1/N sum (sigma_a)^p)^(1/p)  <=  max_a sigma_a  <=  (sum (sigma_a)^p)^(1/p)

so a design that satisfies every clustered constraint still carries local peaks
above the limit by construction, and the paper says so explicitly: "Stresses in
the optimized structure will locally become higher than the stress limit ... we
allow some stress peaks as long as the geometrical shape is such that stress
singularities are avoided."  A peak predicate therefore fires on *every*
attainable design, and the only edit that can entail it away is raising the
allowable -- which is intent-altering.  The cascade would be forced into an
authority refusal by an artefact of the stress measure, not by the physics.

The discriminating quantity is the *extent* of overstress, which is what the
paper's own stress plots show and what its conclusion states as the criterion:
a good design "only leaves a small number of points with stresses above the
stress limit".

OVERSTRESS_FRACTION_LIMIT is a declared calibration decision, not a tuned one.
Measured on this benchmark at the low-fidelity tier, the traditional stiffness
formulation P3 leaves 58% of solid material above the allowable and the
stress-constrained formulation P1 leaves 3.9%; any threshold between roughly
0.05 and 0.55 produces the same partition, so the verdict is insensitive to the
choice across an order of magnitude.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

SOLID_THRESHOLD = 0.5
"""Density above which an element counts as material.

Stress in a near-void element is driven to zero by the penalization eta_S, so
including void elements would make any stress statistic meaningless."""

OVERSTRESS_FRACTION_LIMIT = 0.10
"""Fraction of solid material allowed above the stated allowable."""

PROXIMITY_FILTER_RADII = 2.0
"""Reach, in filter radii, for attributing overstress to a named geometric feature.

Tied to the filter radius because no design feature smaller than the filter can
be resolved, so a tighter reach would test something the discretisation cannot
represent."""

ALL_PREDICATES = (
    "widespread_overstress",
    "reentrant_corner_overstress",
    "load_point_overstress",
    "unrepresented_load_axis",
)

PEAK_PREDICATE = "peak_exceeds_allowable"
"""Retained, and inactive by default, so the claim in this module's docstring is
reproducible as an ablation rather than asserted. See ``active=`` in ``detect``."""


@dataclass
class Anomaly:
    id: str
    fired: bool
    value: float
    threshold: float
    units: str
    location: list | None
    statement: str
    active: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnomalyReport:
    anomalies: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    @property
    def fired(self) -> list:
        return [a for a in self.anomalies if a.fired and a.active]

    @property
    def any_fired(self) -> bool:
        return bool(self.fired)

    def fired_ids(self) -> list:
        return [a.id for a in self.fired]

    def to_dict(self) -> dict:
        return {"evidence": self.evidence,
                "fired": self.fired_ids(),
                "anomalies": [a.to_dict() for a in self.anomalies]}


def _feature_predicate(pid, name, xy, centroids, over, solid, reach, limit):
    d = np.linalg.norm(centroids - xy, axis=1)
    near = d <= reach
    n_near_solid = int((near & solid).sum())
    n_near_over = int((near & over).sum())
    return Anomaly(
        id=pid,
        fired=bool(n_near_over > 0),
        value=float(n_near_over),
        threshold=0.0,
        units=f"solid elements above {limit:.0f} MPa within {reach:.1f} mm of the {name}",
        location=[float(xy[0]), float(xy[1])],
        statement=(
            f"{n_near_over} of {n_near_solid} solid elements within {reach:.1f} mm of the "
            f"{name} at ({xy[0]:.0f}, {xy[1]:.0f}) carry stress above the {limit:.0f} MPa "
            f"allowable"
        ),
    )


def detect(result, *, solid_threshold: float = SOLID_THRESHOLD,
           overstress_fraction_limit: float = OVERSTRESS_FRACTION_LIMIT,
           proximity_filter_radii: float = PROXIMITY_FILTER_RADII,
           active: tuple | None = None, intent=None) -> AnomalyReport:
    """Evaluate every predicate on a converged run.

    ``active`` selects which predicates the cascade will treat as targets; the
    rest are still computed and reported as evidence but cannot fire.

    ``intent`` is a :class:`project.ato.intent.DesignIntent`. It is required for
    the ``unrepresented_load_axis`` predicate and for nothing else: that predicate
    asks whether the *specification* omits something the part is required to do,
    which is a question no field predicate can answer. Passing ``None`` leaves it
    permanently unfired rather than silently assuming the specification is complete.

    All geometry is read from model-agnostic fields on the result, so the same
    predicates apply unchanged to the plane-stress and 2.5D extruded models. In the
    extruded case the stress field is per element (in-plane cell x layer) and the
    reported field is the envelope over load cases.
    """
    active = tuple(active) if active is not None else ALL_PREDICATES
    spec = result.spec
    limit = spec.constraints.stress_limit
    h = spec.geometry.L / spec.geometry.n_cells_per_side
    reach = proximity_filter_radii * spec.optimizer.r0_elements * h

    solid = result.rho_elem >= solid_threshold
    if not np.any(solid):
        raise RuntimeError("no element above the solid threshold; cannot assess stress")
    over = solid & (result.sigma_vm > limit)
    frac = float(over.sum()) / float(solid.sum())

    svm_solid = result.sigma_vm[solid]
    e_peak = int(np.argmax(np.where(solid, result.sigma_vm, -np.inf)))
    peak_xy = result.centroids2d[e_peak]
    missing = list(intent.missing_axes(spec)) if intent is not None else []

    anomalies = [
        Anomaly(
            id=PEAK_PREDICATE,
            fired=bool(svm_solid.max() > limit),
            value=float(svm_solid.max() / limit),
            threshold=1.0,
            units="ratio of peak penalized von Mises to the stated allowable",
            location=[float(peak_xy[0]), float(peak_xy[1])],
            statement=(
                f"peak penalized von Mises in material is {svm_solid.max():.1f} MPa against a "
                f"{limit:.0f} MPa allowable (ratio {svm_solid.max()/limit:.3f}); note the "
                f"clustered P-norm underestimates the maximum by construction"
            ),
        ),
        Anomaly(
            id="widespread_overstress",
            fired=bool(frac > overstress_fraction_limit),
            value=frac,
            threshold=overstress_fraction_limit,
            units="fraction of solid elements above the stated allowable",
            location=[float(peak_xy[0]), float(peak_xy[1])],
            statement=(
                f"{int(over.sum())} of {int(solid.sum())} solid elements ({100*frac:.1f}%) "
                f"exceed the {limit:.0f} MPa allowable; peak {svm_solid.max():.1f} MPa, "
                f"p95 {np.percentile(svm_solid, 95):.1f} MPa, median "
                f"{np.percentile(svm_solid, 50):.1f} MPa"
            ),
        ),
        _feature_predicate("reentrant_corner_overstress", "re-entrant corner",
                           result.corner_xy, result.centroids2d, over, solid, reach, limit),
        _feature_predicate("load_point_overstress", "load application point",
                           result.load_xy, result.centroids2d, over, solid, reach, limit),
        Anomaly(
            id="unrepresented_load_axis",
            fired=bool(missing),
            value=float(len(missing)),
            threshold=0.0,
            units="required load axes with no load case in the specification",
            location=None,
            statement=(
                (f"declared intent requires load along {', '.join(missing)} but the "
                 f"specification states no such load case; the analysis model is "
                 f"'{result.model}', which carries "
                 f"{'u, v only' if result.model == 'plane_stress' else 'u, v, w'}. "
                 f"An axis absent from the specification cannot appear in the stress "
                 f"field, so no field measurement can detect this.")
                if missing else
                (f"every required load axis "
                 f"({', '.join(intent.required_load_axes) if intent is not None else 'none declared'}) "
                 f"has a stated load case"
                 if intent is not None else
                 "no design intent supplied; specification completeness not assessed")
            ),
        ),
    ]
    for a in anomalies:
        a.active = a.id in active
        if not a.active:
            a.fired = False

    return AnomalyReport(
        anomalies=anomalies,
        evidence={
            "peak_stress_MPa": float(svm_solid.max()),
            "p95_stress_MPa": float(np.percentile(svm_solid, 95)),
            "median_stress_MPa": float(np.percentile(svm_solid, 50)),
            "peak_to_p95_ratio": float(svm_solid.max() / np.percentile(svm_solid, 95)),
            "overstressed_fraction": frac,
            "n_solid_elements": int(solid.sum()),
            "stress_limit_MPa": float(limit),
            "peak_location": [float(peak_xy[0]), float(peak_xy[1])],
            "attribution_reach_mm": float(reach),
            "active_predicates": list(active),
            "analysis_model": result.model,
            "n_layers": int(result.n_layers),
            "load_cases": [f"{lc.name}:{lc.direction}" for lc in spec.load_cases],
            "peak_stress_by_case_MPa": {k: float(v.max())
                                        for k, v in result.sigma_vm_by_case.items()},
            "required_load_axes": (list(intent.required_load_axes)
                                   if intent is not None else None),
            "missing_load_axes": missing,
        },
    )
