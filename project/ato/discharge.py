"""Stage 4: deductive discharge by re-solving.

Every candidate, and every subset of candidates up to a stated size, is applied
to the specification and the modified problem is re-solved from scratch.  A
subset SURVIVES if none of the anomalies that fired on the baseline fire again.
Nothing about the verdict comes from the model.

Subsets are enumerated rather than skipped because a candidate eliminated on its
own can re-enter in combination: that non-monotone case is the reason the layer
exists at all.

A re-solve that cannot be trusted does not produce a verdict.  A run whose
stress constraints end infeasible, or that fails outright, is recorded as
NOT_ASSESSED with a reason -- an absence of evidence, never an elimination.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field

from project.ato.anomaly import detect
from project.ato.grammar import apply_edits
from project.topopt.stress.problem import ModelCannotRepresent
from project.topopt.stress.solver import solve_stress_problem

SURVIVES = "SURVIVES"
ELIMINATED = "ELIMINATED"
NOT_ASSESSED = "NOT_ASSESSED"
INFEASIBLE = "INFEASIBLE"
NOT_REPRESENTABLE = "NOT_REPRESENTABLE"
"""The edited specification asks the analysis model for something it has no degree
of freedom to answer -- a transverse load on a plane-stress model, for instance.

This is deliberately NOT ``NOT_ASSESSED``. An unassessed discharge means the
re-solve gave no trustworthy verdict; this means the question could not be posed.
The distinction is actionable: the repair is an enabling model edit, and the
candidate must be re-tried in combination with one. It is also not ``ELIMINATED``:
nothing about the hypothesis was tested."""


@dataclass
class Discharge:
    ids: tuple
    operations: list
    verdict: str
    reason: str
    fired_after: list = field(default_factory=list)
    cleared: list = field(default_factory=list)
    remaining: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    anomaly: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ids": list(self.ids),
            "n_edits": len(self.ids),
            "operations": self.operations,
            "verdict": self.verdict,
            "reason": self.reason,
            "fired_after": self.fired_after,
            "cleared": self.cleared,
            "remaining": self.remaining,
            "run": self.summary,
            "anomaly": self.anomaly,
        }


def _assessable(result, *, plateau_window: float = 0.25,
                plateau_rel_improvement: float = 0.05) -> tuple:
    """Classify a re-solve as usable, genuinely infeasible, or budget-limited.

    An edit that makes the modified specification admit no feasible design is a
    *result*, not an absence of one: it says the edited requirement set is
    jointly unsatisfiable.  That is distinguished from simply running out of
    iterations by asking whether the worst constraint was still descending at
    the cap.  If it has plateaued, the specification is reported INFEASIBLE; if
    it was still improving, no verdict is claimed.
    """
    if result.stress_feasible:
        return "ok", ""
    tol = result.spec.optimizer.feasibility_tol
    hist = (result.history or {}).get("max_constraint") or []
    if len(hist) >= 8:
        k = max(4, int(len(hist) * plateau_window))
        start, end = hist[-k], hist[-1]
        improvement = (start - end) / max(abs(end), tol)
        if improvement > plateau_rel_improvement:
            return "budget", (
                f"the worst constraint was still descending at the iteration cap "
                f"({start:+.4g} -> {end:+.4g} over the last {k} iterations); no verdict claimed"
            )
    return "infeasible", (
        f"the modified specification admits no feasible design at this discretisation: the "
        f"worst constraint plateaued at {result.max_constraint:+.4g} against a tolerance of "
        f"{tol:g}. The edited requirement set is jointly unsatisfiable, which is itself "
        f"diagnostic."
    )


def discharge_subset(base_spec, candidates, subset, target_anomalies, *,
                     max_iter: int | None = None, active: tuple | None = None,
                     cache: dict | None = None, intent=None,
                     n_cells: int | None = None) -> Discharge:
    """Apply a subset of edits, re-solve, and report what the predicates say.

    ``n_cells`` overrides the in-plane discretisation for the re-solve only. It
    exists because discharge cost is the binding constraint on cascade depth: a
    2.5D re-solve at the baseline discretisation runs about 70 s, and the subset
    layer needs tens of them per epoch. Coarsening the re-solve is a declared
    approximation, not a free lunch -- the anomaly predicates are relative
    measures (the fraction of material over the allowable, proximity in mm) which
    makes them fairly mesh-robust, but a verdict established at a coarser mesh than
    the baseline is a weaker claim and the run record says so. Leave it ``None`` to
    discharge at the baseline discretisation.
    """
    chosen = [c for c in candidates if c["id"] in subset]
    # Coarsen BEFORE applying the edits, not after. Applied afterwards it silently
    # overwrites any edit in the subset that refines the mesh -- the candidate's own
    # change is erased and its verdict then answers a question nobody asked. Setting
    # the baseline first lets a mesh edit still win, which is the correct precedence:
    # the discharge discretisation is a cost approximation, the edit is the hypothesis.
    base = base_spec.copy()
    if n_cells is not None:
        base.geometry.n_cells_per_side = int(n_cells)
    spec = apply_edits(base, chosen)
    if max_iter is not None:
        spec.optimizer.max_iter = int(max_iter)

    ids = tuple(sorted(subset))
    ops = [f"{c['id']}:{c['operation']}({json.dumps(c['parameters'], sort_keys=True)})"
           for c in chosen]
    # A subset's re-solve depends only on the edits and the discretisation, not on
    # which predicates are being targeted, so an ablation over predicate sets reuses
    # the same physics. The discretisation is part of the key: a coarse result must
    # never be served to a caller that asked for the baseline mesh.
    key = (ids, spec.geometry.n_cells_per_side)
    if cache is not None and key in cache:
        result = cache[key]
        if isinstance(result, Exception):
            return Discharge(ids, ops, NOT_ASSESSED,
                             f"re-solve raised {type(result).__name__}: {result}")
    else:
        try:
            result = solve_stress_problem(spec, record_history=True)
        except ModelCannotRepresent as exc:
            return Discharge(ids, ops, NOT_REPRESENTABLE, str(exc))
        except Exception as exc:
            if cache is not None:
                cache[key] = exc
            return Discharge(ids, ops, NOT_ASSESSED, f"re-solve raised {type(exc).__name__}: {exc}")
        if cache is not None:
            cache[key] = result

    status, why = _assessable(result)
    # The declared intent must reach the predicates. Without it the
    # specification-completeness predicate cannot fire, and a target that never
    # fires is scored as CLEARED by every subset -- so a corner fillet appears to
    # have fixed a missing load axis.
    #
    # Note the direction of that error: a spuriously cleared target can only push a
    # subset toward SURVIVES, never toward ELIMINATED. Observed in one run, every
    # subset reported `cleared=['unrepresented_load_axis']` and was still eliminated
    # because the in-plane stress predicates persisted. The bug therefore corrupts
    # the attribution of which edit addressed which anomaly -- and would eventually
    # manufacture a false survivor -- rather than causing the eliminations.
    report = detect(result, active=active, intent=intent)
    fired = report.fired_ids()
    still = [a for a in target_anomalies if a in fired]

    cleared = [a for a in target_anomalies if a not in fired]
    if status != "ok":
        verdict = INFEASIBLE if status == "infeasible" else NOT_ASSESSED
        return Discharge(ids, ops, verdict, why, fired, cleared, still,
                         result.summary(), report.to_dict())

    if still:
        verdict, reason = ELIMINATED, (
            f"anomal{'y' if len(still) == 1 else 'ies'} {still} still fire after the edit"
        )
    else:
        verdict, reason = SURVIVES, (
            f"every targeted anomaly {list(target_anomalies)} is entailed away by the re-solve"
        )
    return Discharge(ids, ops, verdict, reason, fired, cleared, still,
                     result.summary(), report.to_dict())


def run_discharge(base_spec, candidates, target_anomalies, *,
                  max_subset_size: int = 2, max_iter: int | None = None,
                  active: tuple | None = None, cache: dict | None = None,
                  progress=None, intent=None, n_cells: int | None = None) -> list:
    """Discharge singletons and every subset up to ``max_subset_size``.

    The default of two enumerates the layer in which non-monotone re-entry was
    first demonstrated. Deeper layers are available but cost 2^k re-solves.
    """
    ids = [c["id"] for c in candidates]
    subsets = []
    for k in range(1, min(max_subset_size, len(ids)) + 1):
        subsets.extend(itertools.combinations(ids, k))

    out = []
    for i, sub in enumerate(subsets, 1):
        d = discharge_subset(base_spec, candidates, set(sub), target_anomalies,
                             max_iter=max_iter, active=active, cache=cache,
                             intent=intent, n_cells=n_cells)
        out.append(d)
        if progress:
            progress(i, len(subsets), d)
    return out
